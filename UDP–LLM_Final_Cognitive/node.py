#!/usr/bin/env python3
"""
FINAL node.py - LSTM Composite Scoring (Your EXACT formulas)
NO asyncio - pure threading only
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import psutil, socket, threading, time, sys, os, json, numpy as np
from fastapi import FastAPI
import uvicorn
import tensorflow as tf
from datetime import datetime
from collections import deque
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import torch
import torchvision.models as models

# ========================= CONFIGURATION =========================
MODEL_PATH = r"/home/node1/Desktop/lstm_llm/lstm_system_monitor1.keras"
WINDOW_LEN = 30
HORIZON_H = 5
DISK_PATH = "."

# ========================= AVAILABILITY TRACKING =========================
local_task_cache = []
task_queue = []
is_busy = False
task_lock = threading.Lock()

# Node identification (EXACT from your code)
physical_id = os.getenv('PHYSICAL_ID', 'machine1')
logical_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
phys_offset = {'machine1': 0, 'machine2': 30, 'machine3': 60, 'machine4': 90, 'machine5': 120}.get(physical_id, 0)

udp_port = 12345 + phys_offset + (logical_id - 1) * 10
REPLY_PORT = 12347
http_port = 8080 + phys_offset * 10 + (logical_id - 1) * 100

# Get IP
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    SYSTEM_IP = s.getsockname()[0]
    s.close()
except:
    SYSTEM_IP = "127.0.0.1"

print(f"🖥️ {physical_id}-N{logical_id} | IP:{SYSTEM_IP} | UDP:{udp_port} | HTTP:{http_port}")

app = FastAPI()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', udp_port))

# Load LSTM + ML models
lstm_model = tf.keras.models.load_model(MODEL_PATH)
history = deque(maxlen=WINDOW_LEN)
ml_model = None
cv_model = None

def load_models():
    global ml_model, cv_model
    iris = load_iris()
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(iris.data, iris.target)
    cv_model = models.mobilenet_v2(pretrained=False)
    cv_model.eval()
    print(f"✅ Models loaded N{logical_id}")

# ========================= YOUR EXACT TASK HANDLERS (Threading only) =========================
def process_classification(task_id, task_info):
    global is_busy, task_queue
    print(f"🔍 N{logical_id} CLASSIFICATION: {task_id}")
    start_time = time.time()
    n_samples = task_info.get('n_samples', 100)
    test_data = np.random.randn(n_samples, 4)
    _ = ml_model.predict(test_data)
    duration = time.time() - start_time
    local_task_cache.append({"success": 1, "duration_actual": duration, "type": "CLASSIFICATION", "n_samples": n_samples})
    with task_lock:
        if task_id in task_queue: task_queue.remove(task_id)
        if not task_queue: is_busy = False
    print(f"✅ N{logical_id} CLASSIFICATION DONE: {duration:.1f}s")

def process_cv(task_id, task_info):
    global is_busy, task_queue
    print(f"🖼️ N{logical_id} CV: {task_id}")
    start_time = time.time()
    n_images = task_info.get('n_images', 50)
    for _ in range(n_images):
        dummy_img = torch.randn(3, 224, 224)
        with torch.no_grad(): _ = cv_model(dummy_img.unsqueeze(0))
    duration = time.time() - start_time
    local_task_cache.append({"success": 1, "duration_actual": duration, "type": "CV_INFERENCE", "n_images": n_images})
    with task_lock:
        if task_id in task_queue: task_queue.remove(task_id)
        if not task_queue: is_busy = False
    print(f"✅ N{logical_id} CV DONE: {duration:.1f}s")

def process_timeseries(task_id, task_info):
    global is_busy, task_queue
    print(f"📈 N{logical_id} TIMESERIES: {task_id}")
    start_time = time.time()
    n_steps = task_info.get('n_steps', 1000)
    for _ in range(n_steps // 100):
        x = np.random.randn(1, 30, 3)
        _ = lstm_model.predict(x, verbose=0)
    duration = time.time() - start_time
    local_task_cache.append({"success": 1, "duration_actual": duration, "type": "TIMESERIES", "n_steps": n_steps})
    with task_lock:
        if task_id in task_queue: task_queue.remove(task_id)
        if not task_queue: is_busy = False
    print(f"✅ N{logical_id} TIMESERIES DONE: {duration:.1f}s")

# ========================= YOUR EXACT SCORING FORMULAS =========================
def get_availability_status(): 
    return {'pending_tasks': len(task_queue), 'is_busy': is_busy, 'completed_tasks': len(local_task_cache)}

def compute_reputation_local():
    if not local_task_cache: return 0.5
    s_i = sum(1 for t in local_task_cache if t.get('success', 0) == 1)
    t_i = len(local_task_cache)
    return max(0.0, min(1.0, (s_i + 3) / (t_i + 6)))

def compute_reliability_local():
    if len(local_task_cache) < 3: return 0.6
    ontime_i = sum(1 for t in local_task_cache if t.get('success', 0) == 1) / len(local_task_cache)
    return max(0.0, min(1.0, 0.7 * ontime_i + 0.3))

def collect_stats():
    return np.array([psutil.cpu_percent(interval=0.1), psutil.virtual_memory().percent, psutil.disk_usage(DISK_PATH).percent], dtype=np.float32) / 100.0

def predict_horizon():
    try:
        if len(history) < WINDOW_LEN: return np.full((HORIZON_H, 3), 0.5, dtype=np.float32)
        window = np.array(list(history)[-WINDOW_LEN:], dtype=np.float32)[np.newaxis]
        preds = []
        for _ in range(HORIZON_H):
            pred = lstm_model.predict(window, verbose=0)[0]
            pred = np.clip(pred, 0.0, 1.0)
            preds.append(pred)
            window = np.roll(window, -1, axis=1)
            window[0, -1, :] = pred
        return np.array(preds, dtype=np.float32)
    except: 
        return np.full((HORIZON_H, 3), 0.5, dtype=np.float32)

def compute_composite_score():
    preds = predict_horizon()
    rep_i = compute_reputation_local()
    rel_i = compute_reliability_local()
    step_scores = []
    for h in range(HORIZON_H):
        cpu, mem, disk = preds[h]
        score_h = 0.35 * (1 - cpu) + 0.20 * (1 - mem) + 0.15 * (1 - disk) + 0.20 * rep_i + 0.10 * rel_i
        step_scores.append(score_h)
    final_score = float(np.mean(step_scores))
    avg_pred = preds.mean(axis=0)
    return final_score, rep_i, rel_i, avg_pred

# ========================= FASTAPI ENDPOINTS =========================
@app.post("/execute")
async def execute_task(request: dict):
    global is_busy, task_queue
    print(f"📥 N{logical_id} RECEIVED: '{request.get('task')}'")
    
    with task_lock:
        if is_busy or len(task_queue) > 0:
            print(f"🚫 N{logical_id} BUSY")
            return {"status": "BUSY", "pending_tasks": len(task_queue)}, 503
        
        task_type = request.get("task")
        if task_type not in {"CLASSIFICATION", "CV_INFERENCE", "TIMESERIES"}:
            return {"status": "ERROR", "message": f"Invalid task: {task_type}"}, 400
        
        task_id = f"task_{int(time.time()*1000)}_{logical_id}"
        task_queue.append(task_id)
        is_busy = True
    
    # Start background task (THREADING ONLY - NO ASYNCIO)
    if task_type == "CLASSIFICATION":
        threading.Thread(target=process_classification, args=(task_id, request), daemon=True).start()
    elif task_type == "CV_INFERENCE":
        threading.Thread(target=process_cv, args=(task_id, request), daemon=True).start()
    elif task_type == "TIMESERIES":
        threading.Thread(target=process_timeseries, args=(task_id, request), daemon=True).start()
    
    print(f"✅ N{logical_id} ACCEPTED: {task_type}")
    return {"status": "ACCEPTED", "task_id": task_id, "task_type": task_type, "node": f"{physical_id}-N{logical_id}"}

@app.get("/metrics")
async def get_metrics():
    score, rep, rel, avg_pred = compute_composite_score()
    availability = get_availability_status()
    return {
        "physical_id": physical_id, "node_id": logical_id, "ip": SYSTEM_IP,
        "score": float(score), "reputation": float(rep), "reliability": float(rel),
        "cpu_pred": float(avg_pred[0]), "mem_pred": float(avg_pred[1]), "disk_pred": float(avg_pred[2]),
        "cpu": float(psutil.cpu_percent()/100), "mem": float(psutil.virtual_memory().percent/100), "disk": float(psutil.disk_usage(DISK_PATH).percent/100),
        "pending_tasks": availability['pending_tasks'], "is_busy": availability['is_busy'], "tasks": availability['completed_tasks']
    }

# ========================= UDP + BACKGROUND THREADS =========================
def udp_listener():
    while True:
        try:
            data, (source_ip, _) = sock.recvfrom(1024)
            if data == b"LL_RESOURCES?":
                score, rep, rel, avg_pred = compute_composite_score()
                availability = get_availability_status()
                reply = {
                    "physical_id": physical_id, "node_id": logical_id, "ip": SYSTEM_IP,
                    "score": float(score), "reputation": float(rep), "reliability": float(rel),
                    "cpu_pred": float(avg_pred[0]), "mem_pred": float(avg_pred[1]), "disk_pred": float(avg_pred[2]),
                    "cpu": float(psutil.cpu_percent()/100), "mem": float(psutil.virtual_memory().percent/100), "disk": float(psutil.disk_usage(DISK_PATH).percent/100),
                    "pending_tasks": availability['pending_tasks'], "is_busy": availability['is_busy'], "tasks": availability['completed_tasks']
                }
                sock.sendto(json.dumps(reply).encode(), (source_ip, REPLY_PORT))
                print(f"📡 N{logical_id}→{source_ip} | SCORE={score:.3f} REP={rep:.3f}")
        except: time.sleep(0.1)

def update_history():
    print(f"[LSTM] N{logical_id} warming up...")
    for _ in range(WINDOW_LEN * 2):
        history.append(collect_stats())
        time.sleep(0.5)
    print(f"[READY] N{logical_id} | LSTM OK")
    while True:
        history.append(collect_stats())
        time.sleep(1.0)

if __name__ == "__main__":
    load_models()
    threading.Thread(target=update_history, daemon=True).start()
    time.sleep(5)
    threading.Thread(target=udp_listener, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=http_port)