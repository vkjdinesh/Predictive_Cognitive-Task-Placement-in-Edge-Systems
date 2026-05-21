#!/usr/bin/env python3
"""
originator_agent.py -- autonomous decentralized task scheduler

Core design principles:
  1. FAIR LOAD BALANCING  -- overloaded nodes are penalised; new/idle nodes
                             get a bonus so work spreads across all nodes
  2. TASK COMPLETION GUARANTEE -- deferred tasks are retried (not dropped)
                                  failed tasks are reassigned to another node
  3. FAULT TOLERANCE  -- heartbeat detects node death; failover reassigns
                         the IN-PROGRESS task automatically
  4. AUTONOMOUS DISCOVERY -- new nodes joining the network are found and
                             included in the next negotiation round
  5. NO GLOBAL LLM -- originator does pure arithmetic; all intelligence
                      lives inside each node's local LLM

Load-balanced score:
  adj = raw - 0.05 * max(0, my_assignments - avg_assignments)
      + 0.03 if my_assignments == 0  (new/idle node bonus)

Deferred task retry:
  Failed tasks go into _retry_queue and are attempted BEFORE new tasks.
  MAX_RETRIES per task before it is permanently dropped.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv, itertools, json, socket, subprocess, threading, time, uuid
from collections import deque
from pathlib import Path
from typing import Optional

import zmq

# ---- CSV latency log -----------------------------------------------------
_LATENCY_LOG = Path("scheduling_latency.csv")

def _log_latency(round_num, task_type, node_key, raw_score, adj_score,
                 lat_negotiation_ms, lat_assignment_ms, lat_total_ms,
                 retry_attempt=0):
    header = not _LATENCY_LOG.exists()
    with _LATENCY_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "round","task_type","node","raw_score","adj_score",
            "lat_negotiation_ms","lat_assignment_ms","lat_total_ms",
            "retry_attempt","timestamp"])
        if header: w.writeheader()
        w.writerow(dict(
            round=round_num, task_type=task_type, node=node_key,
            raw_score=round(raw_score,4), adj_score=round(adj_score,4),
            lat_negotiation_ms=round(lat_negotiation_ms,1),
            lat_assignment_ms=round(lat_assignment_ms,1),
            lat_total_ms=round(lat_total_ms,1),
            retry_attempt=retry_attempt,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))

# ---- config --------------------------------------------------------------
BEACON_BASE_PORT   = 6000
MAX_NODES_PER_HOST = 5
DISCOVER_TIMEOUT   = 1.0
BID_PORT           = 5580
BID_TIMEOUT        = 90.0        # Windows LLM takes 60-90s on CPU
TASK_INTERVAL      = 15          # seconds between new tasks
TASK_TYPES         = ["CLASSIFICATION", "CV_INFERENCE", "TIMESERIES"]

HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_TIMEOUT  = 4.0
MAX_MISSED_PINGS   = 2

ACCEPT_SCORE_MIN   = 0.50
LOAD_PENALTY       = 0.05        # score penalty per task above average
NEW_NODE_BONUS     = 0.03        # bonus for nodes with 0 assignments
FAIRNESS_CAP       = 3           # force idle nodes to get work when winner has this many tasks
MAX_RETRIES        = 3           # max retry attempts for a deferred task


# ---- ip helpers ----------------------------------------------------------
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except: return "127.0.0.1"

def get_all_local_ips():
    ips = {"127.0.0.1", "127.0.1.1", get_local_ip()}
    try:
        r = subprocess.run(["ip","-4","addr","show"],capture_output=True,text=True,timeout=3)
        for l in r.stdout.splitlines():
            l = l.strip()
            if l.startswith("inet "): ips.add(l.split()[1].split("/")[0])
    except: pass
    import os
    for ip in os.getenv("EXTRA_LOCAL_IPS","").split(","):
        if ip.strip(): ips.add(ip.strip())
    return ips

MY_IP     = get_local_ip()
LOCAL_IPS = get_all_local_ips()

def resolve_ip(node_ip):
    return "127.0.0.1" if node_ip in LOCAL_IPS else node_ip


# ---- shared mutable state ------------------------------------------------
_assign_counts: dict = {}
_assign_lock         = threading.Lock()

_live_nodes: dict    = {}        # current live topology -- updated on discovery + failure
_nodes_lock          = threading.Lock()

_dead_nodes: set     = set()
_dead_lock           = threading.Lock()

_retry_queue: deque  = deque()   # (task_type, attempts) -- deferred tasks
_retry_lock          = threading.Lock()


def record_assignment(key: str):
    with _assign_lock:
        _assign_counts[key] = _assign_counts.get(key, 0) + 1
    with _assign_lock:
        counts = dict(_assign_counts)
    print(f"[ORIG] Assignments: {counts}")


def mark_node_dead(key: str) -> bool:
    """Returns True only for the FIRST thread to declare this node dead."""
    with _dead_lock:
        if key in _dead_nodes: return False
        _dead_nodes.add(key)
    with _nodes_lock: _live_nodes.pop(key, None)
    print(f"[ORIG] {key} removed from live pool")
    return True


def update_live_nodes(nodes: dict):
    with _nodes_lock:
        _live_nodes.clear(); _live_nodes.update(nodes)
    with _dead_lock:
        for k in nodes: _dead_nodes.discard(k)


def enqueue_retry(task_type: str, attempts: int = 0):
    if attempts < MAX_RETRIES:
        with _retry_lock:
            _retry_queue.appendleft((task_type, attempts))
        print(f"[ORIG] Task {task_type} re-queued (attempt {attempts+1}/{MAX_RETRIES})")
    else:
        print(f"[ORIG] Task {task_type} permanently dropped after {MAX_RETRIES} retries")


def next_task(task_cycle) -> tuple:
    """Returns (task_type, retry_attempt). Retry queue takes priority."""
    with _retry_lock:
        if _retry_queue:
            task_type, attempts = _retry_queue.pop()
            print(f"[ORIG] Retrying deferred task: {task_type} (attempt {attempts+1})")
            return task_type, attempts + 1
    return next(task_cycle), 0


# ---- load-balanced score -------------------------------------------------
def load_balanced_score(node: dict, all_keys: list = None) -> float:
    """
    Adjusted score = raw - penalty + bonus

    The critical fix: average is computed over ALL live nodes (including
    those with zero assignments), not just those already in _assign_counts.
    Without this, a node that always wins has avg == my_count == penalty 0.

    Example with 2 nodes, Linux has 6 tasks, Windows has 0:
      counts = {linux: 6, windows: 0}  (windows explicitly included)
      avg    = (6+0)/2 = 3.0
      linux  penalty = 0.05*(6-3) = 0.15  -> adj = 0.71-0.15 = 0.56
      windows bonus  = 0.03              -> adj = 0.60+0.03 = 0.63
      Windows wins this round.
    """
    key = node.get("_key", node.get("ip", "?"))
    raw = node.get("score", 0.0)

    with _assign_lock: counts = dict(_assign_counts)

    # Build full count map including live nodes with 0 assignments
    with _nodes_lock: live_keys = list(_live_nodes.keys())
    if all_keys: live_keys = list(set(live_keys) | set(all_keys))

    full_counts = {k: counts.get(k, 0) for k in live_keys}
    if not full_counts:
        return round(raw + NEW_NODE_BONUS, 4)

    my  = full_counts.get(key, 0)
    avg = sum(full_counts.values()) / max(len(full_counts), 1)

    penalty = LOAD_PENALTY * max(0.0, my - avg)
    bonus   = NEW_NODE_BONUS if my == 0 else 0.0
    adj     = round(raw - penalty + bonus, 4)
    return max(0.0, min(1.0, adj))


# ---- node discovery ------------------------------------------------------
def discover_nodes() -> dict:
    nodes = {}
    sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0)); sock.settimeout(DISCOVER_TIMEOUT)
    for bport in [BEACON_BASE_PORT+i for i in range(MAX_NODES_PER_HOST)]:
        for host in ["127.0.0.1","255.255.255.255"]:
            try:
                sock.sendto(b"DISCOVER_MONITOR", (host, bport))
                dl = time.time() + DISCOVER_TIMEOUT
                while time.time() < dl:
                    try:
                        data, (src, _) = sock.recvfrom(1024)
                        info = json.loads(data)
                        nip  = info.get("ip", src); nid = info.get("node_id", 1)
                        sp   = info.get("score_port", 5557+(nid-1)*100)
                        ap   = info.get("a2a_port",   5560+(nid-1)*100)
                        key  = f"{nip}:{nid}"
                        if key not in nodes:
                            nodes[key] = dict(ip=nip, node_id=nid, score_port=sp, a2a_port=ap)
                            tag = "local" if nip in LOCAL_IPS else "remote"
                            print(f"[ORIG] Found {tag} node {key}  a2a={ap}")
                    except socket.timeout: break
                    except: pass
            except: pass
    sock.close()
    if nodes: print(f"[ORIG] Discovery: {len(nodes)} node(s)")
    return nodes


# ---- ping ----------------------------------------------------------------
def ping_node(ctx, info) -> bool:
    conn = resolve_ip(info["ip"])
    try:
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, int(HEARTBEAT_TIMEOUT*1000))
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{conn}:{info['a2a_port']}")
        s.send_string(json.dumps({"type":"PING"}))
        resp = json.loads(s.recv_string()); s.close()
        return resp.get("type") == "PONG"
    except: return False


# ---- A2A negotiation round -----------------------------------------------
def run_negotiation(ctx, pull_sock, nodes: dict, task_type: str) -> Optional[dict]:
    task_id = str(uuid.uuid4())[:8]
    print(f"\n[ORIG] {'='*52}")
    print(f"[ORIG] Task id={task_id}  type={task_type}")
    print(f"[ORIG] {'='*52}")

    # Drain stale bids
    pull_sock.setsockopt(zmq.RCVTIMEO, 0)
    while True:
        try: pull_sock.recv_string()
        except zmq.ZMQError: break
    pull_sock.setsockopt(zmq.RCVTIMEO, int(BID_TIMEOUT*1000))

    req = json.dumps(dict(type="TASK_REQUEST", task_id=task_id,
        task_type=task_type, originator_ip=MY_IP, bid_port=BID_PORT))

    broadcast_start = time.time()   # T1: first TASK_REQUEST sent
    sent = []
    for key, info in nodes.items():
        conn = resolve_ip(info["ip"])
        # Retry connection up to 3 times -- node may still be loading LLM
        acked = False
        for attempt in range(1, 4):
            try:
                s = ctx.socket(zmq.REQ)
                s.setsockopt(zmq.RCVTIMEO, 5000)
                s.setsockopt(zmq.LINGER, 0)
                s.connect(f"tcp://{conn}:{info['a2a_port']}")
                s.send_string(req)
                ack = json.loads(s.recv_string()); s.close()
                if ack.get("ack"):
                    sent.append(key)
                    print(f"[ORIG] TASK_REQUEST acked by {key}")
                    acked = True
                break
            except Exception as e:
                print(f"[ORIG] {key} attempt {attempt}/3: {e}")
                try: s.close()
                except: pass
                if attempt < 3:
                    time.sleep(2)
        if not acked:
            print(f"[ORIG] Could not reach {key} after 3 attempts -- skipping")

    if not sent:
        print("[ORIG] No nodes acknowledged -- skipping")
        return None

    print(f"[ORIG] Waiting {BID_TIMEOUT}s for bids ...")
    bids = []; dl = time.time() + BID_TIMEOUT
    while time.time() < dl:
        pull_sock.setsockopt(zmq.RCVTIMEO, max(100, int((dl-time.time())*1000)))
        try:
            raw = pull_sock.recv_string()
            bid = json.loads(raw)
            if bid.get("type")=="BID" and bid.get("task_id")==task_id:
                bid["_key"] = f"{bid['node_ip']}:{bid['node_id']}"
                bids.append(bid)
                print(f"[ORIG] Bid from {bid['_key']}  "
                      f"raw={bid['score']:.4f}  adj={load_balanced_score(bid):.4f}")
                if len(bids) >= len(sent):
                    print("[ORIG] All nodes responded -- closing early"); break
        except zmq.ZMQError: continue

    if not bids:
        print("[ORIG] No bids received")
        return None

    # Pass all bidding node keys so load_balanced_score sees the full picture
    all_keys = [b["_key"] for b in bids]
    ranked = sorted(bids,
        key=lambda b: load_balanced_score(b, all_keys), reverse=True)

    print(f"\n[ORIG] Final ranking:")
    with _assign_lock: counts = dict(_assign_counts)
    for i, b in enumerate(ranked):
        k = b["_key"]
        print(f"[ORIG]  {i+1}. {k:<26} raw={b['score']:.4f}  "
              f"adj={load_balanced_score(b, all_keys):.4f}  "
              f"tasks={counts.get(k,0)}  risk={b['risk']}")

    last_bid_time = time.time()      # T2: last bid received

    winner = ranked[0]
    winner["task_id"]        = task_id
    winner["all_bids"]       = ranked
    winner["broadcast_start"] = broadcast_start
    winner["last_bid_time"]   = last_bid_time
    print(f"[ORIG] Selected: {winner['_key']}  "
          f"adj={load_balanced_score(winner, all_keys):.4f}")
    print(f"[ORIG] LLM reason: {winner['reason']}")
    return winner


# ---- assign task ---------------------------------------------------------
def assign_task(ctx, nodes: dict, winner: dict, task_id: str) -> bool:
    key  = winner.get("_key", f"{winner['node_ip']}:{winner['node_id']}")
    info = nodes.get(key)
    if not info:
        print(f"[ORIG] {key} not in node map -- trying live pool")
        with _nodes_lock: info = _live_nodes.get(key)
    if not info: return False

    conn = resolve_ip(info["ip"])
    try:
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, 3000); s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{conn}:{info['a2a_port']}")
        s.send_string(json.dumps(dict(type="TASK_ASSIGN", task_id=task_id,
            winner_ip=winner["node_ip"], winner_node_id=winner["node_id"])))
        ack = json.loads(s.recv_string()); s.close()
        print(f"[ORIG] TASK_ASSIGN -> {key}  ack={ack}")
    except Exception as e:
        print(f"[ORIG] TASK_ASSIGN error: {e}"); return False

    try:
        s2 = ctx.socket(zmq.REQ)
        s2.setsockopt(zmq.RCVTIMEO, 3000); s2.setsockopt(zmq.LINGER, 0)
        s2.connect(f"tcp://{conn}:{info['score_port']}")
        s2.send_string("record_task:1"); s2.recv_string(); s2.close()
    except Exception as e: print(f"[ORIG] record_task error: {e}")
    return True


# ---- heartbeat + failover ------------------------------------------------
def monitor_and_failover(ctx, pull_sock, winner: dict,
                         task_id: str, task_type: str):
    """
    Monitors assigned node. On failure:
      - Marks node dead (only ONE thread handles each failure)
      - Re-negotiates with remaining LIVE nodes
      - Assigns same task to new winner (task completion guarantee)
    """
    wkey   = winner.get("_key", f"{winner['node_ip']}:{winner['node_id']}")
    missed = 0
    print(f"[ORIG] Heartbeat started for {wkey}")

    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        with _nodes_lock: info = _live_nodes.get(wkey)
        if not info:
            print(f"[ORIG] Heartbeat: {wkey} removed -- stopping"); break

        if ping_node(ctx, info):
            missed = 0
            print(f"[ORIG] Heartbeat: {wkey} OK")
        else:
            missed += 1
            print(f"[ORIG] Heartbeat: {wkey} no response ({missed}/{MAX_MISSED_PINGS})")
            if missed >= MAX_MISSED_PINGS:
                if not mark_node_dead(wkey):
                    print(f"[ORIG] {wkey} already handled -- stopping"); break

                print(f"\n[ORIG] NODE FAILURE: {wkey}")
                with _nodes_lock: remaining = dict(_live_nodes)

                if not remaining:
                    print("[ORIG] No remaining nodes -- re-queueing task")
                    enqueue_retry(task_type); break

                print(f"[ORIG] Failover with {len(remaining)} node(s)")
                new_winner = run_negotiation(ctx, pull_sock, remaining, task_type)

                if new_winner:
                    new_tid = str(uuid.uuid4())[:8]
                    ok = False
                    for candidate in new_winner.get("all_bids", [new_winner]):
                        candidate["_key"] = (f"{candidate['node_ip']}:"
                                             f"{candidate['node_id']}")
                        if assign_task(ctx, remaining, candidate, new_tid):
                            record_assignment(candidate["_key"])
                            print(f"\n[ORIG] FAILOVER -> {candidate['_key']}")
                            print(f"[ORIG] Score: {candidate['score']:.4f}")
                            print(f"[ORIG] Reason: {candidate['reason']}")
                            winner = candidate
                            wkey   = candidate["_key"]
                            task_id = new_tid; missed = 0; ok = True
                            break
                        else:
                            mark_node_dead(candidate["_key"])
                    if not ok:
                        print("[ORIG] All failover candidates failed -- re-queueing")
                        enqueue_retry(task_type); break
                else:
                    print("[ORIG] No bids during failover -- re-queueing")
                    enqueue_retry(task_type); break


# ---- main ----------------------------------------------------------------
if __name__ == "__main__":
    print(f"[ORIG] ip={MY_IP}  local_ips={LOCAL_IPS}")
    print(f"[ORIG] BID_PORT={BID_PORT}  BID_TIMEOUT={BID_TIMEOUT}s")
    print(f"[ORIG] TASK_INTERVAL={TASK_INTERVAL}s  MAX_RETRIES={MAX_RETRIES}")
    print(f"[ORIG] LOAD_PENALTY={LOAD_PENALTY}  NEW_NODE_BONUS={NEW_NODE_BONUS}")

    ctx        = zmq.Context()
    task_cycle = itertools.cycle(TASK_TYPES)
    round_num  = 0

    nodes = discover_nodes()
    while not nodes:
        print("[ORIG] No nodes -- retrying in 5s ..."); time.sleep(5)
        nodes = discover_nodes()
    update_live_nodes(nodes)

    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.setsockopt(zmq.RCVTIMEO, int(BID_TIMEOUT*1000))
    pull_sock.bind(f"tcp://0.0.0.0:{BID_PORT}")
    print(f"[ORIG] Bid PULL socket open :{BID_PORT}")

    while True:
        round_num += 1
        task_type, retry_attempt = next_task(task_cycle)

        # Periodic rediscovery (every 5 rounds)
        if round_num % 5 == 0:
            new = discover_nodes()
            if new: nodes = new; update_live_nodes(nodes)

        # Get current live nodes
        with _nodes_lock: current = dict(_live_nodes)
        if not current:
            print("[ORIG] No live nodes -- rediscovering ...")
            new = discover_nodes()
            if new: nodes = new; update_live_nodes(nodes)
            with _nodes_lock: current = dict(_live_nodes)

        result = run_negotiation(ctx, pull_sock, current, task_type)

        if result:
            all_bids = result.pop("all_bids", [result])
            task_id  = result["task_id"]
            assigned = False

            for candidate in all_bids:
                candidate["_key"] = (f"{candidate['node_ip']}:"
                                     f"{candidate['node_id']}")
                if assign_task(ctx, current, candidate, task_id):
                    assign_time = time.time()     # T3: TASK_ASSIGN sent
                    record_assignment(candidate["_key"])
                    assigned = True

                    # Latency breakdown (excludes task execution)
                    t1 = result.get("broadcast_start", assign_time)
                    t2 = result.get("last_bid_time",   assign_time)
                    t3 = assign_time

                    lat_negotiation = (t2 - t1) * 1000   # broadcast -> last bid
                    lat_assignment  = (t3 - t2) * 1000   # last bid  -> task assign
                    lat_total       = (t3 - t1) * 1000   # broadcast -> task assign

                    print(f"\n[ORIG] {'─'*52}")
                    print(f"[ORIG] RESULT  round={round_num}  task={task_type}"
                          + (f"  [retry {retry_attempt}]" if retry_attempt else ""))
                    print(f"[ORIG] Node        : {candidate['node_ip']} "
                          f"(N{candidate['node_id']})")
                    print(f"[ORIG] Raw score   : {candidate['score']:.4f}")
                    print(f"[ORIG] Adj score   : {load_balanced_score(candidate):.4f}")
                    print(f"[ORIG] LLM reason  : {candidate['reason']}")
                    print(f"[ORIG] ── Latency (excl. task execution) ──")
                    print(f"[ORIG] Broadcast→last bid : {lat_negotiation:>8.1f} ms")
                    print(f"[ORIG] Last bid→assign    : {lat_assignment:>8.1f} ms")
                    print(f"[ORIG] Total (sched)      : {lat_total:>8.1f} ms  "
                          f"({lat_total/1000:.2f}s)")
                    print(f"[ORIG] {'─'*52}")
                    _log_latency(round_num, task_type,
                        candidate["_key"], candidate["score"],
                        load_balanced_score(candidate),
                        lat_negotiation_ms=lat_negotiation,
                        lat_assignment_ms=lat_assignment,
                        lat_total_ms=lat_total,
                        retry_attempt=retry_attempt)
                    threading.Thread(target=monitor_and_failover,
                        args=(ctx, pull_sock, candidate, task_id, task_type),
                        daemon=True, name=f"hb-{task_id}").start()
                    break
                else:
                    print(f"[ORIG] {candidate['_key']} assignment failed -- next ...")
                    mark_node_dead(candidate["_key"])

            if not assigned:
                print(f"[ORIG] Round {round_num}: All candidates failed")
                enqueue_retry(task_type, retry_attempt)
        else:
            print(f"[ORIG] Round {round_num}: No bids")
            enqueue_retry(task_type, retry_attempt)

        time.sleep(TASK_INTERVAL)
