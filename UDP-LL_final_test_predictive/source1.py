import socket
import requests
import time
import json
import psutil
import csv
from datetime import datetime
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPLY_PORT = 12347
TIMEOUT = 5.0
trial_count = 0
LOG_FILE = Path("source_trials_log.csv")

# =========================
# FIXED TASK POOL
# =========================
TASK_POOL = {
    "CLASSIFICATION": {"n_samples": 150},
    "CV_INFERENCE": {"n_images": 50},
    "TIMESERIES": {"n_steps": 1000}
}

# =========================
# NODE DISCOVERY
# =========================
def discover_nodes():
    nodes = []
    for phys_offset in [0, 30, 60, 90, 120]:
        for log_id in range(1, 4):
            udp_port = 12345 + phys_offset + (log_id-1)*10
            nodes.append(udp_port)
    return nodes

def get_source_cpu():
    return psutil.cpu_percent(interval=0.2)

def is_node_available(node_info):
    return node_info.get('pending_tasks',0)==0 and not node_info.get('is_busy',False)

# =========================
# COMPOSITE SELECTION
# =========================
def composite_selection(node_info):
    if node_info["cpu_pred"] > 0.95 or node_info["mem_pred"]>0.95:
        return 0.0
    tasks = node_info.get('tasks',0)
    fairness = max(0.1,1.0 - 0.1*tasks)
    return node_info["score"]*fairness

def calculate_http_port(phys_id,node_id):
    phys_offset = {
        'machine1':0,'machine2':30,'machine3':60,'machine4':90,'machine5':120
    }
    offset = phys_offset.get(phys_id,0)
    return 8080 + offset*10 + (node_id-1)*100

def save_to_csv(data):
    create_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if create_header:
            writer.writeheader()
        writer.writerow(data)

# =========================
# MAIN LOOP
# =========================
def main_loop():
    global trial_count
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
    sock.bind(("",REPLY_PORT))
    sock.settimeout(TIMEOUT)
    cycle=0

    while True:
        cycle+=1
        trial_count+=1
        print("\n"+"="*110)
        print(f"🔄 CYCLE #{cycle} | TRIAL {trial_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("="*110)

        source_cpu = get_source_cpu()
        print(f"💻 Source CPU: {source_cpu:.1f}%")

        # -----------------------------
        udp_ports = discover_nodes()
        for udp_port in udp_ports:
            sock.sendto(b"LL_RESOURCES?",("255.255.255.255",udp_port))

        replies = {}
        start_time = time.time()
        while time.time()-start_time<TIMEOUT:
            try:
                data, addr = sock.recvfrom(4096)
                info = json.loads(data.decode())
                key = (info["physical_id"],info["node_id"])
                replies[key] = (addr[0],info)
            except socket.timeout:
                break
            except:
                continue

        if not replies:
            print("⏳ Timeout: no node replies.")
            time.sleep(5)
            continue

        print(f"📡 Received {len(replies)} unique node responses:")

        available_nodes=[]
        for (phys_id,node_id),(ip,info) in replies.items():
            score = composite_selection(info)
            status = "🟢 FREE" if is_node_available(info) else "🔴 BUSY"
            print(f"  {phys_id}-N{node_id} ({ip}) | SCORE={score:.3f} CPU={info['cpu_pred']*100:.1f}% MEM={info['mem_pred']*100:.1f}% DISK={info['disk_pred']*100:.1f}% REPUT={info['reputation']:.2f} REL={info['reliability']:.2f} TASKS={info.get('tasks',0)} | {status}")
            if is_node_available(info):
                available_nodes.append((ip,info))

        if not available_nodes:
            print("🚫 No available nodes.")
            time.sleep(5)
            continue

        # -----------------------------
        best_ip,best_info = max(available_nodes,key=lambda x: composite_selection(x[1]))
        phys_id = best_info["physical_id"]
        node_id = best_info["node_id"]
        http_port = calculate_http_port(phys_id,node_id)

        task_type = "CV_INFERENCE"
        task_payload = {"task": task_type,"n_images":TASK_POOL["CV_INFERENCE"]["n_images"]}

        print(f"\n🏆 SELECTED: {phys_id}-N{node_id}")
        print(f"🎯 ASSIGNED: {task_payload}")
        print(f"🔗 Target: http://{best_ip}:{http_port}/execute")

        try:
            t_start = time.time()
            response = requests.post(f"http://{best_ip}:{http_port}/execute",json=task_payload,timeout=60)
            t_end = time.time()
            latency_ms = (t_end-t_start)*1000
            resp_json = response.json()
            status = resp_json.get("status","UNKNOWN")
            print(f"✅ Response: {resp_json}")
            print(f"⏱️  Latency: {latency_ms:.2f} ms")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            latency_ms=0
            status="FAILED"

        save_to_csv({
            "trial":trial_count,
            "node":f"{phys_id}-N{node_id}",
            "cpu":best_info["cpu_pred"]*100,
            "mem":best_info["mem_pred"]*100,
            "disk":best_info["disk_pred"]*100,
            "comp_score":best_info["score"],
            "reputation":best_info["reputation"],
            "reliability":best_info["reliability"],
            "task_type":task_type,
            "latency_ms":latency_ms,
            "status":status
        })

        print("-"*110)
        time.sleep(10)

if __name__=="__main__":
    try:
        print("⏳ Waiting 30 seconds for all nodes to start...")
        time.sleep(30)  
        main_loop()
    except KeyboardInterrupt:
        print("\n🔚 Stopped.")

