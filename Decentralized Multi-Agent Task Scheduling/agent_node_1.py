#!/usr/bin/env python3
"""
agent_node_1.py -- fully decentralized autonomous edge AI agent

Each node independently:
  1. Monitors own resources (psutil)
  2. Runs LSTM short-horizon prediction
  3. Computes composite suitability score
  4. LOCAL LLM decides ACCEPT/REJECT with task-aware reasoning
  5. Participates in A2A negotiation via ZMQ
  6. Responds to heartbeat PINGs for fault detection

Usage:  python agent_node_1.py 1   |   python agent_node_1.py 2
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, os, re, socket, sys, threading, time, traceback
from collections import deque

import numpy as np
import psutil
import tensorflow as tf
import torch
import zmq
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- identity ------------------------------------------------------------
NODE_ID      = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PORT_OFFSET  = (NODE_ID - 1) * 100

SCORE_PORT   = 5557 + PORT_OFFSET
A2A_PORT     = 5560 + PORT_OFFSET
BEACON_PORT  = 6000 + (NODE_ID - 1)
MONITOR_PORT = 5555 + PORT_OFFSET

DISK_PATH        = "/"
SAMPLE_INTERVAL  = 1.0
HORIZON_H        = 5
ACCEPT_SCORE_MIN = 0.50

LSTM_MODEL_PATH = "/home/node1/Desktop/lstm_llm/lstm_system_monitor1.keras"
LLM_MODEL_PATH  = "/home/node1/Desktop/lstm_llm/qwen2.5_local"

# ---- ip ------------------------------------------------------------------
def get_local_ip():
    ip = os.getenv("NODE_IP", "").strip()
    if ip: return ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except: return "127.0.0.1"

LOCAL_IP        = get_local_ip()
_shared_zmq_ctx = zmq.Context()
_a2a_ready      = threading.Event()   # set when A2A REP socket is bound
print(f"[N{NODE_ID}] ip={LOCAL_IP}  score={SCORE_PORT}  a2a={A2A_PORT}  beacon={BEACON_PORT}")

# ---- LSTM ----------------------------------------------------------------
print(f"[N{NODE_ID}] Loading LSTM ...")
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
WINDOW_LEN = lstm_model.input_shape[1]
print(f"[N{NODE_ID}] LSTM ready  window={WINDOW_LEN}")

# ---- LLM -----------------------------------------------------------------
print(f"[N{NODE_ID}] Loading LLM ...")
llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, local_files_only=True)
llm_mdl = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH, dtype=torch.float16, device_map="cpu", local_files_only=True)
llm_mdl.eval()
print(f"[N{NODE_ID}] LLM ready")

# ---- shared state --------------------------------------------------------
_lock       = threading.Lock()
_history    = deque(maxlen=WINDOW_LEN)
_task_cache = []
_is_busy    = False
_state      = dict(ip=LOCAL_IP, node_id=NODE_ID, score=0.5, risk="MEDIUM",
                   reputation=0.5, reliability=0.6, cpu=0.0, mem=0.0, disk=0.0,
                   cpu_pred=0.5, mem_pred=0.5, disk_pred=0.5, lstm_ready=False,
                   horizon=[[0.5,0.5,0.5]]*HORIZON_H, is_busy=False,
                   tasks_completed=0)

# ---- LSTM prediction -----------------------------------------------------
def predict_horizon(hist):
    if len(hist) < WINDOW_LEN:
        return np.full((HORIZON_H, 3), 0.5, np.float32)
    w = np.array(list(hist)[-WINDOW_LEN:], np.float32)[np.newaxis]
    preds = []
    for _ in range(HORIZON_H):
        p = np.clip(lstm_model.predict(w, verbose=0)[0], 0, 1)
        preds.append(p)
        w = np.roll(w, -1, axis=1); w[0,-1,:] = p
    return np.array(preds, np.float32)

# ---- scoring -------------------------------------------------------------
def compute_rep():
    c = list(_task_cache)
    if not c: return 0.5
    s = sum(1 for t in c if t.get("success")==1)
    return max(0.0, min(1.0, (s+3)/(len(c)+6)))

def compute_rel():
    c = list(_task_cache)
    if len(c) < 3: return 0.6
    on = sum(1 for t in c if t.get("success")==1)/len(c)
    return max(0.0, min(1.0, 0.7*on+0.3))

def compute_score(horizon, rep, rel):
    s = [0.35*(1-h[0])+0.20*(1-h[1])+0.15*(1-h[2])+0.20*rep+0.10*rel
         for h in horizon[:HORIZON_H]]
    return float(sum(s)/len(s)) if s else 0.5

def risk_level(score):
    if score >= 0.70: return "LOW"
    if score >= 0.50: return "MEDIUM"
    if score >= 0.30: return "HIGH"
    return "CRITICAL"

# ---- LLM decision --------------------------------------------------------
TASK_PROFILES = {
    "CLASSIFICATION": ("moderate CPU", "low memory",      "ML classification"),
    "CV_INFERENCE":   ("high CPU",     "moderate memory", "computer vision"),
    "TIMESERIES":     ("moderate CPU", "moderate memory", "time-series LSTM"),
    "GENERIC":        ("moderate CPU", "moderate memory", "general compute"),
}

def local_llm_decide(state: dict, task_type: str = "GENERIC") -> dict:
    cpu   = state["cpu"]      * 100
    mem   = state["mem"]      * 100
    disk  = state["disk"]     * 100
    cpu_p = state["cpu_pred"] * 100
    mem_p = state["mem_pred"] * 100
    score = state["score"]
    risk  = state["risk"]
    rep   = state["reputation"]
    rel   = state["reliability"]
    done  = state.get("tasks_completed", 0)

    def lvl(v, lo, hi): return "high" if v>hi else ("moderate" if v>lo else "low")

    cpu_need, mem_need, desc = TASK_PROFILES.get(task_type, TASK_PROFILES["GENERIC"])

    horizon = state.get("horizon", [])
    cpu_trend = mem_trend = "stable"
    if len(horizon) >= 2:
        cpu_trend = ("rising"  if horizon[-1][0]>horizon[0][0]+0.05 else
                     "falling" if horizon[-1][0]<horizon[0][0]-0.05 else "stable")
        mem_trend = ("rising"  if horizon[-1][1]>horizon[0][1]+0.03 else
                     "falling" if horizon[-1][1]<horizon[0][1]-0.03 else "stable")

    hard_reject = (score < ACCEPT_SCORE_MIN or risk=="CRITICAL"
                   or state.get("is_busy", False))

    # Build decision word SEPARATELY -- no nested f-string
    dw   = "REJECT" if hard_reject else "ACCEPT"
    rule = ("REJECT: score below threshold, CRITICAL risk, or node busy."
            if hard_reject else
            "ACCEPT: all thresholds met, node is available.")

    # JSON template as plain string concatenation
    json_tmpl = '{"decision": "' + dw + '", "reason": "<one sentence>"}'

    # Compute what makes this node specifically suitable or not
    cpu_gap   = cpu_p - cpu          # positive = CPU rising
    mem_gap   = mem_p - mem          # positive = memory rising
    score_gap = score - ACCEPT_SCORE_MIN

    # Task-specific fit assessment
    if task_type == "CV_INFERENCE":
        fit_note = (f"CV_INFERENCE needs high CPU; current CPU={cpu:.1f}% "
                    f"predicted {cpu_p:.1f}% ({cpu_trend})")
    elif task_type == "CLASSIFICATION":
        fit_note = (f"CLASSIFICATION needs moderate CPU; current CPU={cpu:.1f}% "
                    f"({lvl(cpu,30,65)}), memory={mem:.1f}% ({lvl(mem,40,70)})")
    elif task_type == "TIMESERIES":
        fit_note = (f"TIMESERIES needs moderate CPU and memory; "
                    f"CPU={cpu:.1f}%({cpu_trend}), mem={mem:.1f}%({mem_trend})")
    else:
        fit_note = f"CPU={cpu:.1f}%, mem={mem:.1f}%, score={score:.4f}"

    system_msg = (
        "You are a concise edge-AI node policy engine. "
        "Write exactly ONE sentence as the reason. "
        "The sentence MUST: "
        "(1) start with the task type name (e.g. 'CLASSIFICATION requires...'), "
        "(2) include at least two specific numbers from the node state, "
        "(3) explain the concrete fit or mismatch -- not just 'meets threshold'. "
        "BAD example: 'All metrics meet requirements.' "
        "GOOD example: 'CLASSIFICATION requires moderate CPU and this node shows "
        "only 3.1% CPU (stable trend) with score 0.7205, well above the 0.5 threshold.' "
        "Never start with I. Never be vague."
    )

    user_msg = (
        f"Node {LOCAL_IP} N{NODE_ID} — decision for '{task_type}' task.\n"
        f"Task profile: {desc} — needs {cpu_need}, {mem_need}.\n\n"
        f"Node metrics:\n"
        f"  CPU now={cpu:.1f}%  predicted={cpu_p:.1f}%  trend={cpu_trend}\n"
        f"  Mem now={mem:.1f}%  predicted={mem_p:.1f}%  trend={mem_trend}\n"
        f"  Disk={disk:.1f}%\n"
        f"  Composite score={score:.4f} (threshold={ACCEPT_SCORE_MIN}, "
        f"margin={score_gap:+.4f})\n"
        f"  Risk={risk}  Rep={rep:.3f}  Rel={rel:.3f}\n"
        f"  Tasks completed={done}\n\n"
        f"Fit assessment: {fit_note}\n"
        f"Decision: {rule}\n\n"
        f"Write the reason sentence — cite the specific numbers above.\n"
        f"Respond with ONLY this JSON:\n{json_tmpl}"
    )

    try:
        text   = llm_tok.apply_chat_template(
            [{"role":"system","content":system_msg},
             {"role":"user",  "content":user_msg}],
            tokenize=False, add_generation_prompt=True)
        inputs = llm_tok(text, return_tensors="pt")
        ilen   = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = llm_mdl.generate(**inputs, max_new_tokens=80,
                do_sample=False, pad_token_id=llm_tok.eos_token_id)
        raw = llm_tok.decode(out[0][ilen:], skip_special_tokens=True).strip()
        print(f"[N{NODE_ID}] LLM raw: {raw}")

        m = re.search(r'\{[^{}]*"decision"[^{}]*"reason"[^{}]*\}', raw, re.DOTALL)
        if m:
            p  = json.loads(m.group())
            d  = p.get("decision", dw).strip().upper()
            r  = p.get("reason", "").strip()
            if d not in ("ACCEPT","REJECT"): d = dw
            if r and len(r) > 15:
                return {"decision": d, "reason": r}
        print(f"[N{NODE_ID}] LLM JSON parse failed -- using fallback")
    except Exception as e:
        print(f"[N{NODE_ID}] LLM error: {e}\n{traceback.format_exc()}")

    # Meaningful fallback
    if dw == "ACCEPT":
        reason = (f"N{NODE_ID} accepts {task_type}: CPU={cpu:.1f}%({cpu_trend}), "
                  f"score={score:.4f}>{ACCEPT_SCORE_MIN}, risk={risk}, "
                  f"{done} prior tasks completed.")
    else:
        reason = (f"N{NODE_ID} rejects {task_type}: score={score:.4f} "
                  f"below {ACCEPT_SCORE_MIN} or risk={risk} or busy.")
    return {"decision": dw, "reason": reason}

# ---- metric loop ---------------------------------------------------------
def _metric_loop():
    print(f"[N{NODE_ID}] Metric loop started")
    while True:
        cpu  = psutil.cpu_percent(interval=0.1)
        mem  = psutil.virtual_memory().percent
        disk = psutil.disk_usage(DISK_PATH).percent
        sample = np.array([cpu/100, mem/100, disk/100], np.float32)
        with _lock:
            _history.append(sample)
            n = len(_history); ready = n >= WINDOW_LEN
            preds = predict_horizon(_history)
        avg = preds.mean(axis=0)
        rep = compute_rep(); rel = compute_rel()
        score = compute_score(preds.tolist(), rep, rel)
        risk  = risk_level(score)
        with _lock:
            _state.update(ip=LOCAL_IP, node_id=NODE_ID,
                score=round(score,4), risk=risk,
                reputation=round(rep,4), reliability=round(rel,4),
                cpu=round(cpu/100,4), mem=round(mem/100,4), disk=round(disk/100,4),
                cpu_pred=round(float(avg[0]),4), mem_pred=round(float(avg[1]),4),
                disk_pred=round(float(avg[2]),4), lstm_ready=ready,
                horizon=preds.tolist(), is_busy=_is_busy,
                tasks_completed=len(_task_cache))
        st = "READY" if ready else f"warming {n}/{WINDOW_LEN}"
        print(f"[N{NODE_ID}] cpu={cpu:.1f}% mem={mem:.1f}% "
              f"pred_cpu={avg[0]*100:.1f}% score={score:.4f} risk={risk} [{st}]")
        time.sleep(SAMPLE_INTERVAL)

# ---- UDP beacon ----------------------------------------------------------
def _beacon():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", BEACON_PORT))
    print(f"[N{NODE_ID}] UDP beacon :{BEACON_PORT}  (waiting for A2A ready ...)")
    _a2a_ready.wait()   # block until A2A REP socket is bound and listening
    print(f"[N{NODE_ID}] UDP beacon active -- A2A is ready")
    while True:
        try:
            data, (src_ip, src_port) = s.recvfrom(1024)
            if data == b"DISCOVER_MONITOR":
                s.sendto(json.dumps(dict(ip=LOCAL_IP, node_id=NODE_ID,
                    zmq_port=MONITOR_PORT, score_port=SCORE_PORT,
                    a2a_port=A2A_PORT)).encode(), (src_ip, src_port))
                print(f"[N{NODE_ID}] Announced to {src_ip}")
        except Exception as e: print(f"[N{NODE_ID}] Beacon: {e}")

# ---- score server --------------------------------------------------------
def _score_server():
    ctx = zmq.Context(); sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{SCORE_PORT}")
    print(f"[N{NODE_ID}] Score REP :{SCORE_PORT}")
    while True:
        try:
            msg = sock.recv_string()
            if msg == "get_score":
                with _lock: sock.send_string(json.dumps(dict(_state)))
            elif msg.startswith("record_task:"):
                with _lock: _task_cache.append({"success": int(msg.split(":")[1])})
                sock.send_string(json.dumps({"ok": True}))
            else: sock.send_string(json.dumps({"error": "unknown"}))
        except Exception as e: print(f"[N{NODE_ID}] Score server: {e}")

# ---- A2A server ----------------------------------------------------------
def _a2a_server():
    ctx  = zmq.Context(); sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{A2A_PORT}")
    print(f"[N{NODE_ID}] A2A REP :{A2A_PORT}")
    _a2a_ready.set()   # signal beacon thread that A2A is now listening

    while True:
        try:
            msg      = json.loads(sock.recv_string())
            msg_type = msg.get("type","")

            if msg_type == "TASK_REQUEST":
                tid = msg["task_id"]; ttype = msg.get("task_type","GENERIC")
                oip = msg["originator_ip"]; bport = msg["bid_port"]
                sock.send_string(json.dumps({"ack": True, "node_id": NODE_ID}))
                print(f"[N{NODE_ID}] TASK_REQUEST id={tid} type={ttype} from={oip}:{bport}")
                with _lock: snap = dict(_state)

                def _bid(snap, tid, ttype, oip, bport):
                    print(f"[N{NODE_ID}] Evaluating {tid}  score={snap['score']:.4f} "
                          f"risk={snap['risk']}  busy={snap.get('is_busy',False)}")
                    res = local_llm_decide(snap, ttype)
                    print(f"[N{NODE_ID}] Decision: {res['decision']}  reason: {res['reason']}")
                    if res["decision"] != "ACCEPT":
                        print(f"[N{NODE_ID}] Not bidding -- REJECT"); return

                    bid = json.dumps(dict(type="BID", task_id=tid,
                        node_ip=LOCAL_IP, node_id=NODE_ID,
                        score=snap["score"], risk=snap["risk"],
                        reason=res["reason"], decision="ACCEPT"))
                    try:
                        import subprocess as _sp
                        _local = {"127.0.0.1","127.0.1.1",LOCAL_IP}
                        try:
                            r = _sp.run(["ip","-4","addr","show"],
                                capture_output=True,text=True,timeout=2)
                            for l in r.stdout.splitlines():
                                l=l.strip()
                                if l.startswith("inet "):
                                    _local.add(l.split()[1].split("/")[0])
                        except: pass
                        for _ip in os.getenv("EXTRA_LOCAL_IPS","").split(","):
                            if _ip.strip(): _local.add(_ip.strip())
                        conn = "127.0.0.1" if oip in _local else oip
                        ps = _shared_zmq_ctx.socket(zmq.PUSH)
                        ps.setsockopt(zmq.LINGER,3000); ps.setsockopt(zmq.SNDTIMEO,8000)
                        ps.connect(f"tcp://{conn}:{bport}")
                        time.sleep(0.5); ps.send_string(bid); time.sleep(0.5); ps.close()
                        print(f"[N{NODE_ID}] BID sent via={conn} score={snap['score']:.4f}")
                    except Exception as e: print(f"[N{NODE_ID}] BID FAILED: {e}")

                threading.Thread(target=_bid,
                    args=(snap,tid,ttype,oip,bport), daemon=True).start()

            elif msg_type == "PING":
                sock.send_string(json.dumps(dict(type="PONG", node_id=NODE_ID,
                    ip=LOCAL_IP, score=_state.get("score",0.0),
                    is_busy=_state.get("is_busy",False))))

            elif msg_type == "TASK_ASSIGN":
                wid = msg.get("winner_node_id"); tid = msg.get("task_id")
                if str(wid) == str(NODE_ID):
                    print(f"[N{NODE_ID}] WON task {tid}")
                    with _lock:
                        global _is_busy
                        _is_busy = True; _state["is_busy"] = True
                    threading.Thread(target=_exec_task, args=(tid,), daemon=True).start()
                sock.send_string(json.dumps({"ack": True}))
            else:
                sock.send_string(json.dumps({"error": "unknown"}))

        except Exception as e:
            print(f"[N{NODE_ID}] A2A error: {e}")
            try: sock.send_string(json.dumps({"error": str(e)}))
            except: pass

def _exec_task(tid):
    global _is_busy
    print(f"[N{NODE_ID}] Executing {tid} ...")
    time.sleep(5)
    with _lock:
        _is_busy = False; _state["is_busy"] = False
        _task_cache.append({"success": 1})
        _state["tasks_completed"] = len(_task_cache)
    print(f"[N{NODE_ID}] Task {tid} complete")

if __name__ == "__main__":
    threading.Thread(target=_metric_loop, daemon=True).start()
    threading.Thread(target=_beacon,      daemon=True).start()
    threading.Thread(target=_score_server, daemon=True).start()
    _a2a_server()
