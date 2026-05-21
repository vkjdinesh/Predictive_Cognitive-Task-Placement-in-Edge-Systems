import sys
import os
import psutil
import socket
import threading
import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# =========================
# NODE IDENTIFICATION
# =========================
physical_id = os.getenv("PHYSICAL_ID", "machine1")
logical_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# =========================
# Ports (single-machine)
# =========================
udp_port = 12345 + (logical_id - 1) * 10  # 12345,12355,12365
REPLY_PORT = 12347
http_port = 8080 + (logical_id - 1) * 10  # 8080,8090,8100

print(f"🌐 {physical_id}-N{logical_id} → UDP:{udp_port} HTTP:{http_port}")

# =========================
# FastAPI
# =========================
app = FastAPI()

# =========================
# UDP Socket
# =========================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', udp_port))

# =========================
# Task Model
# =========================
class Task(BaseModel):
    task: str
    duration: float = 1.5
    size: int = 0

# =========================
# Background Task Execution
# =========================
async def execute_background(task: Task):
    """Fire-and-forget execution - doesn't block HTTP response"""
    print(f"🔥 BG {physical_id}-N{logical_id} '{task.task}' d={task.duration}s size={task.size}")
    
    # Task-specific execution time
    exec_time = task.duration
    if "CV_INFERENCE" in task.task:
        exec_time *= max(1.5, task.size / 30.0)  # Heavy CV
    elif "TIMESERIES" in task.task:
        exec_time *= 0.7  # Light TS
    elif "CLASSIFICATION" in task.task:
        exec_time *= 1.0  # Medium ML
    
    await asyncio.sleep(exec_time)
    
    # Log completion
    cpu_after = psutil.cpu_percent()
    mem_after = psutil.virtual_memory().percent
    print(f"✅ BG COMPLETE {physical_id}-N{logical_id} '{task.task}' | CPU={cpu_after:.1f}% MEM={mem_after:.1f}%")

# =========================
# UDP Listener
# =========================
def udp_listener():
    while True:
        try:
            data, (source_ip, source_port) = sock.recvfrom(1024)
            if data == b"RESOURCES?":
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                disk = psutil.disk_usage('.').percent
                reply = f"{logical_id},{cpu},{mem},{disk}"
                sock.sendto(reply.encode(), (source_ip, REPLY_PORT))
                print(f"📡 {physical_id}-N{logical_id} → {source_ip}:{REPLY_PORT} | {reply}")
        except Exception as e:
            time.sleep(0.1)

# =========================
# IMMEDIATE ACK + Background Execution
# =========================
@app.post("/execute")
async def execute_task(task: Task):
    # IMMEDIATE ACCEPTANCE (pure network latency)
    cpu_now = psutil.cpu_percent()
    mem_now = psutil.virtual_memory().percent
    print(f"📥 {physical_id}-N{logical_id} ACCEPTED '{task.task}' size={task.size}")
    
    # Return ACK instantly
    ack_response = {
        "status": "ACCEPTED",
        "physical_id": physical_id,
        "node": logical_id,
        "cpu_at_accept": cpu_now,
        "mem_at_accept": mem_now,
        "task": task.task,
        "duration": task.duration,
        "size": task.size,
        "message": "Task queued for background execution"
    }
    
    # Fire-and-forget background execution (non-blocking)
    asyncio.create_task(execute_background(task))
    
    return ack_response

# =========================
# Metrics Endpoint (bonus)
# =========================
@app.get("/metrics")
async def get_metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage('.').percent
    return {
        "physical_id": physical_id,
        "node": logical_id,
        "cpu_percent": cpu,
        "mem_percent": mem,
        "disk_percent": disk,
        "timestamp": time.time()
    }

# =========================
# Main
# =========================
if __name__ == "__main__":
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"🌐 {physical_id}-N{logical_id} | Local: http://127.0.0.1:{http_port}/docs")
    print(f"🌐                          | Net:   http://{local_ip}:{http_port}/docs")
    print(f"🌐                          | Metrics: http://127.0.0.1:{http_port}/metrics")
    
    # Start UDP listener
    threading.Thread(target=udp_listener, daemon=True).start()
    
    # Start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=http_port)
