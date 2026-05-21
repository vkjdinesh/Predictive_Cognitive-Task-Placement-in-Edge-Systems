import socket
import requests
import time
import psutil
import csv
from datetime import datetime
from pathlib import Path
import random
from collections import defaultdict
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPLY_PORT = 12347
TIMEOUT = 5.0
trial_count = 0
cycle_count = 0
LOG_FILE = Path("source_trials_log.csv")
# =========================
# NODE IDENTIFICATION
# =========================
physical_id = os.getenv("PHYSICAL_ID", "machine1")
logical_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Task tracking per node
node_tasks = defaultdict(list)  # {node_id: [task_types]}
TASK_POOL = {
    "CLASSIFICATION": 150,
    "CV_INFERENCE": 50, 
    "TIMESERIES": 1000
}
TASK_TYPES = list(TASK_POOL.keys())

def discover_nodes():
    return [12345 + (i-1)*10 for i in range(1, 4)]

def get_source_cpu():
    return psutil.cpu_percent(interval=0.2)

def calculate_http_port(node_id):
    return 8080 + (node_id - 1) * 10

def get_node_task_info(node_id):
    """Return tasks count and last task for node"""
    tasks = node_tasks[node_id]
    count = len(tasks)
    last_task = tasks[-1] if tasks else "NONE"
    return count, last_task

def save_to_csv(data):
    create_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if create_header:
            writer.writeheader()
        writer.writerow(data)

def safe_recv(sock, timeout=3.0):
    sock.settimeout(timeout)
    try:
        reply, addr = sock.recvfrom(1024)
        return reply, addr
    except:
        return None, None

def main_loop():
    global trial_count, cycle_count, node_tasks
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", REPLY_PORT))
    sock.settimeout(TIMEOUT)

    while True:
        cycle_count += 1
        trial_count += 1
        
        print("\n" + "="*110)
        print(f"🔄 CYCLE #{cycle_count} | TRIAL {trial_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("="*110)

        print(f"💻 Source CPU: {get_source_cpu():.1f}%")

        # UDP Broadcast
        for udp_port in discover_nodes():
            sock.sendto(b"RESOURCES?", ("255.255.255.255", udp_port))

        # Collect replies
        replies = []
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            reply, addr = safe_recv(sock)
            if reply:
                try:
                    parts = reply.decode().split(',')
                    node_id, cpu, mem, disk = int(parts[0]), *map(float, parts[1:])
                    replies.append((1, node_id, cpu/100, mem/100, disk/100, addr[0]))
                except:
                    pass

        if not replies:
            print("⏳ No nodes responding. Retrying...")
            time.sleep(5)
            continue

        # === TABLE WITH TASK COUNTS ===
        print(f"\n📡 Received {len(replies)} unique node responses:")
        for phys_id, node_id, cpu, mem, disk, ip in replies:
            task_count, last_task = get_node_task_info(node_id)
            status_emoji = "🟢 FREE"
            print(f"  {physical_id }-N{node_id:1d} ({ip}) | CPU={cpu*100:.1f}% MEM={mem*100:.1f}% DISK={disk*100:.1f}% "
                  f"TASKS={task_count:2d} | LAST={last_task:12s} | {status_emoji}")

        # Select best node (lowest CPU→MEM→DISK→TASKS)
        best_node = min(replies, key=lambda x: (x[2], x[3], x[4], len(node_tasks[int(x[1])])))
        phys_id, node_id, cpu, mem, disk, best_ip = best_node
        http_port = calculate_http_port(node_id)

        task_type = random.choice(TASK_TYPES)
        task_size = TASK_POOL[task_type]
        task_payload = {"task": task_type, "duration": 1.5, "size": task_size}

        # Record task assignment
        node_tasks[node_id].append(task_type)

        # === YOUR EXACT FORMAT ===
        print(f"\n🏆 SELECTED: {physical_id}-N{node_id}")
        print(f"🎯 ASSIGNED: {task_payload}")
        print(f"🔗 Target: http://{best_ip}:{http_port}/execute")
        
        # Latency measurement
        latency_ms = 0
        status = "UNKNOWN"
        try:
            t_start = time.time()
            response = requests.post(f"http://{best_ip}:{http_port}/execute",
                                   json=task_payload, timeout=60)
            t_end = time.time()
            latency_ms = (t_end - t_start) * 1000
            resp_json = response.json()
            status = resp_json.get("status", "UNKNOWN")
            print(f"✅ Response: {resp_json}")
            print(f"⏱️  Latency: {latency_ms:.2f} ms")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            latency_ms = 0
            status = "FAILED"

        print("-"*110 + "\n")

        # Log
        save_to_csv({
            "trial": trial_count,
            "cycle": cycle_count,
            "node": f"machine1-N{node_id}",
            "cpu": cpu, "mem": mem, "disk": disk,
            "task_type": task_type, "size": task_size,
            "tasks_count": len(node_tasks[node_id]),
            "latency_ms": latency_ms,
            "status": status
        })

        time.sleep(10)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n🔚 Stopped.")
        print("Final task counts:")
        for node_id, tasks in sorted(node_tasks.items()):
            print(f"  {physical_id }-N{node_id}: {len(tasks)} tasks ({tasks[-5:] if tasks else []})")