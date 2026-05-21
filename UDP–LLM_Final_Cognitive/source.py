#!/usr/bin/env python3
"""
MULTI-MACHINE UDP-LLM Cognitive Orchestrator - ALL WARNINGS FIXED
 LLM + Fairness + CPU/MEM/DISK = Production-ready edge AI!
"""
import socket
import requests
import time
import json
import psutil
import csv
from datetime import datetime
from pathlib import Path
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


# ========================= CONFIGURATION =========================
QWEN_PATH = r"/home/node1/Desktop/lstm_llm/qwen2.5_local"
REPLY_PORT = 12347
TIMEOUT = 5.0
trial_count = 0
LOG_FILE = Path("udp_llm_trials.csv")


# 🔥 GLOBAL TASK COUNTER for FAIRNESS PENALTY
global_task_counts = defaultdict(lambda: defaultdict(int))


TASK_POOL = {
    "CLASSIFICATION": {"n_samples": 150},
    "CV_INFERENCE": {"n_images": 50},
    "TIMESERIES": {"n_steps": 1000}
}


TASK_NAMES = list(TASK_POOL.keys())


# LLM Loading - ALL WARNINGS FIXED
model_llm = None
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True  # ✅ FIXES tokenizer warning
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_llm = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model_llm.eval()
    print("🧠 LLM Coordinator loaded PERFECTLY")
except Exception as e:
    print(f"⚠️ LLM unavailable: {e}")


def discover_nodes():
    return [12345 + phys_offset + (log_id-1)*10 for phys_offset in [0, 30, 60, 90, 120] for log_id in range(1, 4)]


def calculate_http_port(phys_id, node_id):
    phys_offset = {'machine1': 0, 'machine2': 30, 'machine3': 60, 'machine4': 90, 'machine5': 120}
    return 8080 + phys_offset.get(phys_id, 0)*10 + (node_id-1)*100


def get_task_description():
    global trial_count
    task_idx = trial_count % len(TASK_NAMES)
    task_type = TASK_NAMES[task_idx]
    task_info = TASK_POOL[task_type].copy()

    if task_type == "CLASSIFICATION":
        return f"{task_type} (150 samples) - moderate CPU", task_type, task_info
    elif task_type == "CV_INFERENCE":
        return f"{task_type} (50 images) - GPU + low CPU", task_type, task_info
    else:  # TIMESERIES
        return f"{task_type} (1000 steps) - memory + LSTM", task_type, task_info


def compute_fairness_penalty(tasks):
    """fairness = max(0.1, 1.0 - 0.1×tasks)"""
    fairness = max(0.1, 1.0 - 0.1 * tasks)
    return fairness


def print_decision_justification(selected_node_key, available_nodes, task_type):
    selected_info = available_nodes[selected_node_key]
    highest_score_key, highest_score_info = max(available_nodes.items(), key=lambda x: x[1].get("score", 0))

    print(f"\n🧠🤔 LLM DECISION ANALYSIS:")
    print(f"   📊 Highest score: {highest_score_key} (score={highest_score_info.get('score', 0):.3f})")
    print(f"   🎯 LLM picked:   {selected_node_key} (score={selected_info.get('score', 0):.3f})")

    score_diff = selected_info.get('score', 0) - highest_score_info.get('score', 0)
    phys_id = selected_info['physical_id']
    node_id = selected_info['node_id']
    tasks = global_task_counts[phys_id][node_id]
    fairness = compute_fairness_penalty(tasks)

    print(f"   ⚖️ Fairness: {fairness:.3f} (tasks={tasks})")

    if abs(score_diff) < 0.001:
        print(f"   ✅ LLM agrees with score-based (highest score)")
    else:
        print(f"   🔄 LLM OVERRIDE ({score_diff:+.3f} score diff):")
        if selected_info.get('reliability', 0) > highest_score_info.get('reliability', 0):
            print(f"      → REL={selected_info.get('reliability', 0):.3f} > {highest_score_info.get('reliability', 0):.3f}")
        if selected_info.get('cpu', 0)*100 < highest_score_info.get('cpu', 0)*100:
            print(f"      → CPU={selected_info.get('cpu', 0)*100:.1f}% < {highest_score_info.get('cpu', 0)*100:.1f}%")
        if tasks < global_task_counts[highest_score_info['physical_id']][highest_score_info['node_id']]:
            print(f"      → Fairness: fewer tasks ({tasks} vs {global_task_counts[highest_score_info['physical_id']][highest_score_info['node_id']]})")


def build_llm_prompt(task_desc, candidates):
    # Updated: force 2-line structured output including rationale
    prompt = (
        "<|im_start|>system\n"
        f"Select BEST node for task from these {len(candidates)} options.\n"
        f"Available: {list(candidates.keys())}\n"
        "FINAL_SCORE already includes fairness penalties.\n"
        "Respond in exactly two lines:\n"
        "SELECTED_NODE: machineX-NY@IP\n"
        "RATIONALE_EXPLANATION: <one concise sentence explaining why this node is preferred, "
        "even if it does not have the highest raw score>\n"
        "<|im_end|>\n"
    )
    prompt += f"<|im_start|>user\nTask: {task_desc}\n\nDetailed nodes:\n"

    for i, (node_key, info) in enumerate(candidates.items()):
        phys_id = info['physical_id']
        node_id = info['node_id']
        raw_score = info.get('score', 0)
        tasks_count = global_task_counts[phys_id][node_id]
        fairness = compute_fairness_penalty(tasks_count)
        final_score = raw_score * fairness
        cpu = info.get('cpu', 0)*100
        rel = info.get('reliability', 0)
        prompt += (
            f"{i+1}. {node_key} FINAL={final_score:.3f}"
            f"(RAW={raw_score:.3f},F={fairness:.3f}) CPU={cpu:.1f}% REL={rel:.3f}\n"
        )

    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def call_llm(prompt, candidates):
    if not model_llm:
        print("🤖 LLM unavailable - using fallback")
        return None, None

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model_llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=96,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        print(f"🤖 LLM RAW: '{response}'")

        candidate_keys = list(candidates.keys())
        selected_key = None

        # Direct match on explicit SELECTED_NODE line
        upper_resp = response.upper()
        for node_key in candidate_keys:
            if f"SELECTED_NODE: {node_key}".upper() in upper_resp or node_key.upper() in upper_resp:
                selected_key = node_key
                break

        # Regex-based node parsing if direct search fails
        if not selected_key:
            node_match = re.search(
                r'SELECTED_NODE[:\s]*([mM][aA][cC][hH][iI][nN][eE]\d+-N\d+@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
                response
            )
            if node_match and node_match.group(1) in candidate_keys:
                selected_key = node_match.group(1)

        # IP-based fallback
        if not selected_key:
            ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', response)
            if ip_match:
                ip_found = ip_match.group(1)
                for node_key in candidate_keys:
                    if candidates[node_key]['ip'] == ip_found:
                        selected_key = node_key
                        break

        # Rationale extraction (independent of node success)
        rationale = None
        rat_match = re.search(
            r'RATIONALE_EXPLANATION[:\s]*(.+)',
            response,
            re.IGNORECASE
        )
        if rat_match:
            rationale = rat_match.group(1).strip()

        if not selected_key:
            print("🤖 LLM parsing failed - using fallback")
            if rationale:
                print(f"🗣️ LLM RATIONALE (no node parsed): {rationale}")
            return None, rationale

        print(f"🧠 LLM MATCHED NODE: '{selected_key}'")
        if rationale:
            print(f"🗣️ LLM RATIONALE: {rationale}")

        return selected_key, rationale

    except Exception as e:
        print(f"🤖 LLM ERROR: {e}")
        return None, None


def fallback_selection(available_nodes):
    return max(available_nodes.items(), key=lambda x: x[1].get("score", 0))[0]


def safe_http_post(ip, port, payload):
    try:
        t_start = time.time()
        url = f"http://{ip}:{port}/execute"
        response = requests.post(url, json=payload, timeout=60)
        latency_ms = (time.time() - t_start) * 1000
        return True, response.json(), latency_ms
    except Exception as e:
        return False, {"status": f"ERROR: {str(e)[:30]}"}, 0


def save_to_csv(trial, node_label, node_key, info, task_type, status, success, latency_ms, llm_rationale=None):
    phys_id = info['physical_id']
    node_id = info['node_id']
    tasks_count = global_task_counts[phys_id][node_id]
    data = {
        "trial": trial,
        "node": node_label,
        "node_key": node_key,
        "ip": info['ip'],
        "physical_id": phys_id,
        "task": task_type,
        "cpu": info.get('cpu', info.get('cpu_pred', 0))*100,
        "mem": info.get('mem', 0)*100,
        "disk": info.get('disk', 0)*100,
        "score": info.get('score', 0),
        "reliability": info.get('reliability', 0),
        "tasks_count": tasks_count,
        "status": status,
        "success": 1 if success else 0,
        "latency_ms": latency_ms,
        "llm_rationale": llm_rationale or ""
    }
    create_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if create_header:
            writer.writeheader()
        writer.writerow(data)


def main_loop():
    global trial_count, global_task_counts
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", REPLY_PORT))
    sock.settimeout(TIMEOUT)

    cycle = 0
    while True:
        cycle += 1
        trial_count += 1
        print(f"\n{'='*110}")
        print(f"🔄 CYCLE #{cycle} | TRIAL {trial_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("="*110)

        source_cpu = psutil.cpu_percent(interval=0.2)
        print(f"💻 Source CPU: {source_cpu:.1f}%")

        # UDP Discovery
        udp_ports = discover_nodes()
        for udp_port in udp_ports:
            sock.sendto(b"LL_RESOURCES?", ("255.255.255.255", udp_port))

        replies = {}
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            try:
                data, addr = sock.recvfrom(4096)
                info = json.loads(data.decode())
                key = (info.get("physical_id", "unknown"), info.get("node_id", 0))
                replies[key] = (addr[0], info)
            except socket.timeout:
                break
            except:
                continue

        if not replies:
            print("⏳ No nodes discovered")
            time.sleep(5)
            continue

        # 🔥 PERFECT NODE DISPLAY (ALL resources shown)
        print(f"📡 {len(replies)} nodes discovered:")
        available_nodes = {}
        for (phys_id, node_id), (ip, info) in replies.items():
            pending = info.get('pending_tasks', 0)
            busy = info.get('is_busy', False)
            status = "🟢 FREE" if pending == 0 and not busy else f"🔴 BUSY (tasks={pending})"

            score = info.get('score', 0)
            rel = info.get('reliability', 0)
            cpu = info.get('cpu', info.get('cpu_pred', 0)) * 100
            mem = info.get('mem', 0) * 100
            disk = info.get('disk', 0) * 100

            print(
                f"  {phys_id}-N{node_id}@{ip:<15} | SCORE={score:.3f} | REL={rel:.3f} | "
                f"CPU={cpu:.1f}% | MEM={mem:.1f}% | DISK={disk:.1f}% | {status}"
            )

            if pending == 0:
                node_key = f"{phys_id}-N{node_id}@{ip}"
                available_nodes[node_key] = info.copy()
                available_nodes[node_key]["physical_id"] = phys_id
                available_nodes[node_key]["node_id"] = node_id
                available_nodes[node_key]["ip"] = ip

        if not available_nodes:
            print("🚫 No available nodes")
            time.sleep(5)
            continue

        print(f"\n🟢 {len(available_nodes)} available NODES for LLM:")
        for node_key in available_nodes:
            print(f"   → {node_key}")

        # Task + LLM decision
        task_desc, task_type, task_info = get_task_description()
        print(f"🎯 TASK: {task_type} {task_info}")
        print("🤖 LLM reasoning...")

        selected_node_key, rationale = call_llm(
            build_llm_prompt(task_desc, available_nodes),
            available_nodes
        )

        if selected_node_key:
            print_decision_justification(selected_node_key, available_nodes, task_type)
            if rationale:
                print(f"📝 One-line LLM rationale: {rationale}")

        # Execute selection
        if selected_node_key and selected_node_key in available_nodes:
            info = available_nodes[selected_node_key]
            phys_id = info["physical_id"]
            node_id = info["node_id"]
            ip = info["ip"]
            print(f"🧠 LLM SELECTED: {selected_node_key}")
            selection_method = "LLM"
        else:
            selected_node_key = fallback_selection(available_nodes)
            info = available_nodes[selected_node_key]
            phys_id = info["physical_id"]
            node_id = info["node_id"]
            ip = info["ip"]
            print(f"⚖️ FALLBACK (max score): {selected_node_key}")
            selection_method = "COMPOSITE"
            # On fallback, no reliable LLM rationale
            rationale = None

        # HTTP Dispatch
        http_port = calculate_http_port(phys_id, node_id)
        payload = {"task": task_type, **task_info}

        print(f"\n🏆 FINAL ASSIGNMENT: {phys_id}-N{node_id}@{ip}:{http_port}")
        print(f"🔗 Target: http://{ip}:{http_port}/execute")

        success, result, latency_ms = safe_http_post(ip, http_port, payload)
        status = result.get('status', 'UNKNOWN')
        print(f"{'✅ SUCCESS' if success else '❌ FAILED'}: {result}")
        print(f"⏱️  Latency: {latency_ms:.2f} ms")

        global_task_counts[phys_id][node_id] += 1

        w_comp, w_llm = 0.7, 0.3
        hybrid_score = w_comp * info.get('score', 0) + w_llm * (1.0 if selection_method == "LLM" else 0.8)
        print(f"📊 Hybrid Score: {hybrid_score:.3f}")

        save_to_csv(
            trial_count,
            f"{phys_id}-N{node_id}",
            selected_node_key,
            info,
            task_type,
            status,
            success,
            latency_ms,
            llm_rationale=rationale
        )
        print("-"*110)
        time.sleep(12)


if __name__ == "__main__":
    print("🚀 4-MACHINE LLM Orchestrator w/ FAIRNESS + CPU/MEM/DISK")
    print("✅ ALL WARNINGS FIXED - Production ready!")
    print("⏳ Waiting 30s for nodes + LSTM warmup...")
    time.sleep(30)
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n🔚 Coordinator stopped.")

