import psutil
import socket
import threading
import time
import sys
import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn
import tensorflow as tf
from collections import deque
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import torch
import torchvision.models as models
from datetime import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# =========================
# CONFIGURATION
# =========================
MODEL_PATH = r"/home/node1/Desktop/lstm_llm/lstm_system_monitor1.keras"
WINDOW_LEN = 30
HORIZON_H = 5
DISK_PATH = "."

# =========================
# THREAD-SAFE STATE
# =========================
local_task_cache = []  # {"type": task_type, "duration": sec, "success":0/1, "timestamp": epoch}
task_queue = []
is_busy = False
task_lock = threading.Lock()
history = deque(maxlen=WINDOW_LEN)

# =========================
# NODE IDENTIFICATION
# =========================
physical_id = os.getenv("PHYSICAL_ID", "machine1")
logical_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

phys_offset = {
    "machine1": 0,
    "machine2": 30,
    "machine3": 60,
    "machine4": 90,
    "machine5": 120,
}.get(physical_id, 0)

udp_port = 12345 + phys_offset + (logical_id - 1) * 10
http_port = 8080 + phys_offset * 10 + (logical_id - 1) * 100

# =========================
# GET IP
# =========================
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    SYSTEM_IP = s.getsockname()[0]
finally:
    s.close()

print(f"🖥️ {physical_id}-N{logical_id}")
print(f"🌐 IP={SYSTEM_IP} | UDP={udp_port} | HTTP={http_port}")

# =========================
# INIT
# =========================
app = FastAPI()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", udp_port))

lstm_model = tf.keras.models.load_model(MODEL_PATH)
ml_model = None
cv_model = None

# =========================
# LOAD ML MODELS
# =========================
def load_models():
    global ml_model, cv_model
    iris = load_iris()
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(iris.data, iris.target)

    cv_model = models.mobilenet_v2(pretrained=False)
    cv_model.eval()
    print(f"✅ Models loaded on Node{logical_id}")

# =========================
# TASK CLEANUP
# =========================
def finish_task(task_id):
    global is_busy
    with task_lock:
        task_queue.remove(task_id) if task_id in task_queue else None
        if len(task_queue) == 0:
            is_busy = False

# =========================
# TASK HANDLERS
# =========================
def process_classification(task_id, task_info):
    start = time.time()
    n_samples = task_info.get("n_samples", 150)
    test_data = np.random.randn(n_samples, 4)
    try:
        _ = ml_model.predict(test_data)
        success = 1
    except:
        success = 0
    duration = time.time() - start
    local_task_cache.append({"type": "CLASSIFICATION", "duration": duration, "success": success, "timestamp": time.time()})
    finish_task(task_id)

def process_cv(task_id, task_info):
    start = time.time()
    n_images = task_info.get("n_images", 50)
    try:
        for _ in range(n_images):
            dummy = torch.randn(3, 224, 224)
            with torch.no_grad():
                _ = cv_model(dummy.unsqueeze(0))
        success = 1
    except:
        success = 0
    duration = time.time() - start
    local_task_cache.append({"type": "CV_INFERENCE", "duration": duration, "success": success, "timestamp": time.time()})
    finish_task(task_id)

def process_timeseries(task_id, task_info):
    start = time.time()
    n_steps = task_info.get("n_steps", 1000)
    try:
        for _ in range(n_steps // 100):
            x = np.random.randn(1, 30, 3)
            _ = lstm_model.predict(x, verbose=0)
        success = 1
    except:
        success = 0
    duration = time.time() - start
    local_task_cache.append({"type": "TIMESERIES", "duration": duration, "success": success, "timestamp": time.time()})
    finish_task(task_id)

# =========================
# REPUTATION & RELIABILITY
# =========================
def compute_reputation():
    now = time.time()
    s_i = sum(t['success'] for t in local_task_cache)
    f_i = len(local_task_cache) - s_i
    t_i = s_i + f_i
    delta_t_days = 0  # For simplicity, could store last task timestamp for real decay
    rep = ((s_i + 3) / (t_i + 6)) * (0.95 ** delta_t_days)
    return rep

def compute_reliability():
    if not local_task_cache:
        return 0.6  # default cold-start
    total = len(local_task_cache)
    ontime = sum(1 for t in local_task_cache if t["duration"] < 1.0)/total  # example threshold 1s
    overload = sum(1 for t in local_task_cache if t["duration"] > 2.0)/total  # example threshold 2s
    durations = [t["duration"] for t in local_task_cache]
    jitter = np.std(durations)/np.mean(durations) if durations else 0
    rel = 0.4*ontime + 0.3*(1-overload) + 0.3*(1-jitter)
    return rel

# =========================
# COMPOSITE SCORE
# =========================
def compute_composite_score():
    cpu = psutil.cpu_percent() / 100.0
    mem = psutil.virtual_memory().percent / 100.0
    disk = psutil.disk_usage(DISK_PATH).percent / 100.0
    history.append([cpu, mem, disk])
    
    if len(history) < WINDOW_LEN:
        cpu_pred = [cpu]*HORIZON_H
        mem_pred = [mem]*HORIZON_H
        disk_pred = [disk]*HORIZON_H
    else:
        X = np.array(history)[-WINDOW_LEN:].reshape(1, WINDOW_LEN, 3)
        preds = lstm_model.predict(X, verbose=0)
        
        # ---- Fix start ----
        # Ensure preds is always 2D: (HORIZON_H, 3)
        preds = np.array(preds)
        if preds.ndim == 2 and preds.shape[1] == 3:
            cpu_pred = preds[:,0].tolist()
            mem_pred = preds[:,1].tolist()
            disk_pred = preds[:,2].tolist()
        else:
            # fallback in case shape is unexpected
            cpu_pred = [cpu]*HORIZON_H
            mem_pred = [mem]*HORIZON_H
            disk_pred = [disk]*HORIZON_H
        # ---- Fix end ----
    
    avg_cpu = np.mean(cpu_pred)
    avg_mem = np.mean(mem_pred)
    avg_disk = np.mean(disk_pred)
    rep = compute_reputation()
    rel = compute_reliability()
    score = 0.35*(1-avg_cpu)+0.2*(1-avg_mem)+0.15*(1-avg_disk)+0.2*rep+0.1*rel
    return float(score), avg_cpu, avg_mem, avg_disk, rep, rel
# =========================
# FASTAPI ENDPOINTS
# =========================
@app.post("/execute")
async def execute_task(request: dict):
    global is_busy
    with task_lock:
        if is_busy:
            raise HTTPException(status_code=503, detail="BUSY")
        task_type = request.get("task")
        if task_type not in {"CLASSIFICATION","CV_INFERENCE","TIMESERIES"}:
            raise HTTPException(status_code=400, detail="Invalid task type")
        task_id = f"task_{int(time.time()*1000)}_{logical_id}"
        task_queue.append(task_id)
        is_busy = True

    if task_type=="CLASSIFICATION":
        threading.Thread(target=process_classification,args=(task_id,request),daemon=True).start()
    elif task_type=="CV_INFERENCE":
        threading.Thread(target=process_cv,args=(task_id,request),daemon=True).start()
    elif task_type=="TIMESERIES":
        threading.Thread(target=process_timeseries,args=(task_id,request),daemon=True).start()

    return {"status":"ACCEPTED","task_id":task_id,"task_type":task_type}

@app.get("/metrics")
async def get_metrics():
    score, avg_cpu, avg_mem, avg_disk, rep, rel = compute_composite_score()
    return {
        "node_id": logical_id,
        "physical_id": physical_id,
        "ip": SYSTEM_IP,
        "score": score,
        "cpu_pred": avg_cpu,
        "mem_pred": avg_mem,
        "disk_pred": avg_disk,
        "reputation": rep,
        "reliability": rel,
        "completed_tasks": len(local_task_cache),
        "pending_tasks": len(task_queue),
        "is_busy": is_busy
    }

# =========================
# UDP LISTENER
# =========================
def udp_listener():
    while True:
        try:
            data, (source_ip, _) = sock.recvfrom(1024)
            if data == b"LL_RESOURCES?":
                score, cpu_pred, mem_pred, disk_pred, rep, rel = compute_composite_score()
                reply = json.dumps({
                    "node_id": logical_id,
                    "physical_id": physical_id,
                    "score": score,
                    "cpu_pred": cpu_pred,
                    "mem_pred": mem_pred,
                    "disk_pred": disk_pred,
                    "reputation": rep,
                    "reliability": rel,
                    "tasks": len(local_task_cache),
                    "pending_tasks": len(task_queue),
                    "is_busy": is_busy
                }).encode()
                sock.sendto(reply, (source_ip, 12347))
        except Exception as e:
            print("UDP listener error:", e)
            time.sleep(0.1)

# =========================
# START
# =========================
if __name__ == "__main__":
    load_models()
    threading.Thread(target=udp_listener, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=http_port)

