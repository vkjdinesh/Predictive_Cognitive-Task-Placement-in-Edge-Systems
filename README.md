#Deployment Guide

## How to Run the System

### Step 1 — Install Dependencies (All Machines)

Install all required Python packages on every machine participating in the EdgeMIND deployment.

```bash
pip install -r requirements.txt
```

---

### Step 2 — Update Model Paths

Update the local file paths for:

- LSTM model
- Large Language Models (LLMs)

Ensure the paths are correctly configured in the corresponding configuration or source files before execution.

---

### Step 3 — Connect All Machines to the Same Network

Ensure that all edge devices and nodes are connected to the same Wi-Fi or local network to enable inter-node communication.

---

### Step 4 — Configure Machine ID (Windows)

Set a unique physical ID for each machine before running the system.

```powershell
$env:PHYSICAL_ID="machine1"
```

Use different IDs for different machines, for example:

- `machine1`
- `machine2`
- `machine3`

---

### Step 5 — Start Edge Nodes

Run the following command on each edge node machine:

```bash
python node.py <node_id>
```

Example:

```bash
python node.py 1
```

---

### Step 6 — Start the Source / Originator

Run the source application on either:

- a single machine, or
- multiple machines depending on the deployment setup.

```bash
python source.py
```

or

```bash
python originator.py
```

---

## Example Multi-Node Setup

| Machine | Role        | Command |
|----------|-------------|----------|
| machine1 | Edge Node 1 | `python node.py 1` |
| machine2 | Edge Node 2 | `python node.py 2` |
| machine3 | Source Node | `python source.py` |

---

## Notes

- Ensure firewall settings allow local network communication.
- GPU acceleration requires compatible CUDA drivers and libraries.
- For distributed execution, verify that IP addresses and ports are correctly configured.
- It is recommended to test connectivity between nodes before deployment.

```
