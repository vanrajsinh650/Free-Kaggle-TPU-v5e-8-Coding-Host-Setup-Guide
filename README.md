# 🚀 Free Kaggle TPU v5e-8 Coding Host Setup Guide

> **Welcome from the video / Reel! 👋**  
> If you watched my video on running heavy 32B coding LLMs **100% free** with **zero credit card**, this is the exact, step-by-step setup guide and code shown in the demonstration.

[![100% Free](https://img.shields.io/badge/Cost-100%25_FREE-brightgreen.svg)]()
[![No Credit Card](https://img.shields.io/badge/Credit_Card-NOT_REQUIRED-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-TPU_v5e--8_(128GB_HBM)-orange.svg)]()
[![Context](https://img.shields.io/badge/Context_Window-16%2C384_Tokens-purple.svg)]()
[![Tunnel](https://img.shields.io/badge/Tunnel-Cloudflare_(Zero_Token)-green.svg)]()

---

## ⚡ What This Setup Does

As shown in the video, this pipeline runs state-of-the-art open-source coding models (**Qwen2.5-Coder-32B-Instruct** or **DeepSeek-R1-Distill-Qwen-32B**) on Kaggle's free cloud infrastructure (**TPU VM v5e-8** with 128GB HBM memory, with automatic fallback to **Dual Tesla T4 GPUs**).

It exposes an **OpenAI-compatible API endpoint** via a free Cloudflare Quick Tunnel (no ngrok tokens or sign-ups needed) so you can connect it directly into:
- **VS Code** (Roo Code, Cline, Continue.dev, OpenCode)
- **Cursor**
- **Claude Code CLI**
- **Terminal scripts & cURL**

---

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| **[`kaggle_tpu_v5e_coder.ipynb`](./kaggle_tpu_v5e_coder.ipynb)** | **Primary Notebook**: Uses `vllm` on Kaggle TPU v5e-8 (or GPU T4 x2) + Cloudflare Quick Tunnel (Zero setup). |
| **[`kaggle_qwen36_dual_t4.ipynb`](./kaggle_qwen36_dual_t4.ipynb)** | **Dual-T4 GGUF Alternative**: Uses `llama-server` on 2x Tesla T4 GPUs with tensor layer splitting. |

---

## 🛠️ Step 1: The 2-Minute Kaggle Setup (Exact Clicks)

1. **Verify Your Phone Number (One-Time)**:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings).
   - Under **Phone verification**, complete SMS verification.
   - *Why?* This unlocks your **free 30 hours per week** of TPU & GPU quota and allows you to toggle the internet connection inside notebooks.
2. **Create a New Notebook**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) → click **New Notebook**.
3. **Configure Settings (Right-Hand Sidebar)**:
   - **Accelerator**: `TPU VM v5e-8` *(or fallback `GPU T4 x2`)*
   - **Internet**: **`Internet on`** *(Mandatory: needed to download model weights and tunnel)*
   - **Environment**: `Always use latest environment`
   - **Persistence**: `Files only` (optional)

---

## 💻 Step 2: The Exact Notebook Cells (Run 1 → 3)

You can simply upload **[`kaggle_tpu_v5e_coder.ipynb`](./kaggle_tpu_v5e_coder.ipynb)** into Kaggle, or copy-paste these 3 clean cells:

### CELL 1 — Detect Hardware & Install Serving Engine + Cloudflare Tunnel (Bash)

```bash
%%bash
# CELL 1 — Detect Hardware & Install Serving Engine + Cloudflare Tunnel
set -e

echo "========================================="
echo "🔍 Detecting Hardware Environment..."
echo "========================================="

# 1. Download Cloudflare Quick Tunnel (100% Free, Zero Account Required)
if [ ! -f /usr/local/bin/cloudflared ]; then
    echo "⬇️ Downloading cloudflared binary..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi
echo "✅ cloudflared ready: $(cloudflared --version)"

# 2. Detect Accelerator & Install vLLM Engine
if [ -n "$TPU_NAME" ] || [ -d "/sys/class/tpu" ] || [ -f "/lib/libtpu.so" ]; then
    echo "⚡ TPU Environment Detected (v5e-8)"
    pip install -q --upgrade pip
    pip install -q vllm-tpu -f https://storage.googleapis.com/libtpu-releases/index.html || pip install -q "vllm[tpu]"
    echo "ENV_TYPE=TPU" > /kaggle/working/env_profile.txt
elif command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU Environment Detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install -q --upgrade pip
    pip install -q "vllm>=0.6.0"
    echo "ENV_TYPE=GPU" > /kaggle/working/env_profile.txt
else
    echo "⚠️ Neither TPU nor GPU detected! Set Settings -> Accelerator to TPU VM v5e-8 or GPU T4 x2."
    echo "ENV_TYPE=CPU" > /kaggle/working/env_profile.txt
fi

echo "========================================="
echo "✅ Dependencies & Serving Engine Installed!"
echo "========================================="
```

---

### CELL 2 — Launch Server + Cloudflare Public Tunnel (Python)

```python
# CELL 2 — Launch OpenAI-Compatible Server + Cloudflare Public Tunnel
import os, sys, time, subprocess, re, requests
from pathlib import Path

# Read environment type
env_type = "TPU"
if Path("/kaggle/working/env_profile.txt").exists():
    for line in Path("/kaggle/working/env_profile.txt").read_text().splitlines():
        if line.startswith("ENV_TYPE="):
            env_type = line.split("=")[1].strip()

print(f"[*] Active Hardware Mode: {env_type}")

# Model Selection:
# TPU v5e-8 (128GB HBM) -> runs full unquantized 32B model with 16,384 context!
# GPU T4 x2 (30GB total VRAM) -> uses high-speed AWQ quantized 32B model across both GPUs.
if env_type == "TPU":
    MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Alternative: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    MAX_LEN = "16384"
    EXTRA_FLAGS = [
        "--max-model-len", MAX_LEN,
    ]
else:  # Dual T4 GPU fallback
    MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    MAX_LEN = "8192"
    EXTRA_FLAGS = [
        "--tensor-parallel-size", "2",
        "--max-model-len", MAX_LEN,
        "--gpu-memory-utilization", "0.92",
    ]

SERVED_NAME = "free-coder"
PORT = 8000
LOG_FILE = "/kaggle/working/vllm-server.log"
TUNNEL_LOG = "/kaggle/working/tunnel.log"

# 1. Clean up any previous server/tunnel instances
subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
subprocess.run(["pkill", "-9", "-f", "cloudflared"], capture_output=True)
time.sleep(2)

# 2. Launch vLLM Server in Background
print(f"[*] Launching vLLM with {MODEL_ID} (Context: {MAX_LEN} tokens)...")
vllm_cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_ID,
    "--served-model-name", SERVED_NAME,
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--trust-remote-code",
    *EXTRA_FLAGS
]

vllm_log = open(LOG_FILE, "w")
vllm_proc = subprocess.Popen(
    vllm_cmd,
    stdout=vllm_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    close_fds=True,
)
print(f"[*] vLLM server process spawned (PID: {vllm_proc.pid})")

# 3. Launch Cloudflare Quick Tunnel in Background
print("[*] Starting Cloudflare Quick Tunnel (No auth token required)...")
tunnel_log = open(TUNNEL_LOG, "w")
tunnel_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=tunnel_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    close_fds=True,
)

# 4. Extract Public Cloudflare URL from Tunnel Log
public_url = None
for _ in range(30):
    if Path(TUNNEL_LOG).exists():
        content = Path(TUNNEL_LOG).read_text()
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
        if match:
            public_url = match.group(0)
            break
    time.sleep(1)

if not public_url:
    raise RuntimeError("Failed to extract Cloudflare public tunnel URL. Check /kaggle/working/tunnel.log")

api_base = f"{public_url}/v1"

# 5. Poll /v1/models until weights finish loading into memory
print("[*] Waiting for model weights to load into memory (~60-120s)...", flush=True)
ready = False
for i in range(70):
    try:
        resp = requests.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=3)
        if resp.status_code == 200:
            ready = True
            break
        print(f"    Loading model... ({i+1}/70) HTTP {resp.status_code}")
    except requests.RequestException:
        print(f"    Waiting for port {PORT}... ({i+1}/70)")
    time.sleep(5)

if not ready:
    print("❌ Timeout waiting for model to load. Inspecting recent log lines:")
    if Path(LOG_FILE).exists():
        print("\n".join(Path(LOG_FILE).read_text().splitlines()[-25:]))
    raise TimeoutError("Model failed to initialize in time.")

# 6. Print Formatted Connection Card for VS Code Extensions
print("\n" + "=" * 65)
print("🎉 FREE CLOUD TPU CODING ENDPOINT IS LIVE!")
print("=" * 65)
print(f"🌐 Base URL:   {api_base}")
print(f"🔑 API Key:    EMPTY  (or any dummy string)")
print(f"🧠 Model ID:   {SERVED_NAME}")
print(f"📏 Context:    {MAX_LEN} tokens")
print("=" * 65)
print("\n📋 Copy-Paste Config for VS Code (Roo Code / Cline / Continue):")
config_json = {
    "provider": "OpenAI Compatible",
    "baseUrl": api_base,
    "apiKey": "EMPTY",
    "modelId": SERVED_NAME,
    "contextWindow": int(MAX_LEN)
}
import json
print(json.dumps(config_json, indent=2))
print("=" * 65)
```

---

### CELL 3 — Keep-Alive Heartbeat Daemon (Python)

```python
# CELL 3 — Keep-Alive Heartbeat Daemon
from datetime import datetime
import time, requests

PORT = 8000
print('🚀 Serving active at:', api_base)
print('Started at:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Keep this cell running while coding in VS Code / Cursor.')
print('Press the Stop/Interrupt button when you are done to save quota.')

try:
    while True:
        time.sleep(300)
        try:
            check = requests.get(f'http://127.0.0.1:{PORT}/v1/models', timeout=5)
            status = 'healthy' if check.ok else f'HTTP {check.status_code}'
        except Exception as exc:
            status = f'not reachable: {exc}'
        print(datetime.now().strftime('%H:%M:%S'), 'heartbeat:', status)
except KeyboardInterrupt:
    print('Keep-alive stopped. You may now stop the Kaggle session.')
```

---

## 🔌 Step 3: Connect to VS Code / Cursor in 30 Seconds

### A. Roo Code & Cline (VS Code Extension)
1. Open the extension in VS Code and click the **Gear Icon (Settings)**.
2. Under **API Provider**, select **`OpenAI Compatible`**.
3. **Base URL**: `https://<your-subdomain>.trycloudflare.com/v1`
4. **API Key**: `EMPTY`
5. **Model ID**: `free-coder`

### B. Continue.dev (`~/.continue/config.json`)
```json
{
  "title": "Kaggle TPU Coder (32B)",
  "provider": "openai",
  "model": "free-coder",
  "apiBase": "https://<your-subdomain>.trycloudflare.com/v1",
  "apiKey": "EMPTY",
  "contextLength": 16384
}
```

### C. Claude Code CLI
```bash
export ANTHROPIC_BASE_URL="https://<your-subdomain>.trycloudflare.com/v1"
export ANTHROPIC_API_KEY="EMPTY"
claude --model free-coder
```

### D. Test Endpoint from Any Terminal
```bash
curl -X POST https://<your-subdomain>.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "free-coder",
    "messages": [{"role": "user", "content": "Write a python fibonacci generator function."}],
    "max_tokens": 150
  }'
```

---

## 💡 Pro Tips for Maximizing Free Kaggle Compute

1. **Conserve Your 30 Hours/Week**: When you finish coding, interrupt Cell 3 and select **Session → Stop Session** in Kaggle's top navigation bar.
2. **Switching to DeepSeek-R1**: In Cell 2, replace `MODEL_ID` with `"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"` to switch from pure code writing to high-depth chain-of-thought mathematical reasoning.
3. **Instant Restarts**: Kaggle sessions have a 9-hour limit. If the session expires, simply open your notebook and click **Run All**; model weights stream from Hugging Face at 1 Gbps+, ready to code in ~90 seconds.

---

⭐ **Found this helpful? Drop a star on the repo and share it with your fellow developers!**
