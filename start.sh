#!/bin/bash
set -e

echo "=== Wanly RunPod Worker Starting ==="

# ---------- 1. Download models (skips existing) ----------
/app/download_models.sh

# ---------- 2. Clone or update daemon ----------
DAEMON_DIR="/app/wanly-gpu-daemon"
DAEMON_REPO="https://github.com/DavidJBarnes/wanly-gpu-daemon.git"

# Use GITHUB_TOKEN for private repo auth if set
if [ -n "$GITHUB_TOKEN" ]; then
    DAEMON_REPO="https://${GITHUB_TOKEN}@github.com/DavidJBarnes/wanly-gpu-daemon.git"
fi

if [ -d "$DAEMON_DIR/.git" ]; then
    echo "Updating daemon..."
    cd "$DAEMON_DIR"
    git pull --ff-only origin main 2>/dev/null || echo "WARN: git pull failed, using existing code"
else
    echo "Cloning daemon..."
    git clone --depth 1 "$DAEMON_REPO" "$DAEMON_DIR"
fi

# Install/update daemon deps
pip install --no-cache-dir -q -r "$DAEMON_DIR/requirements.txt" 2>/dev/null || true

# ---------- 3. Write daemon .env ----------
cat > "$DAEMON_DIR/.env" << EOF
REGISTRY_URL=${REGISTRY_URL:-http://gpu-registry.wanly22.com:8000}
QUEUE_URL=${QUEUE_URL:-http://api.wanly22.com:8001}
FRIENDLY_NAME=${FRIENDLY_NAME:-runpod-${RUNPOD_POD_ID:-unknown}}
COMFYUI_URL=http://localhost:8188
COMFYUI_PATH=/app/ComfyUI
LORA_CACHE_DIR=/workspace/models/loras
EOF

echo "Daemon config:"
cat "$DAEMON_DIR/.env"

# ---------- 4. Start ComfyUI (background, no auth) ----------
mkdir -p /workspace/logs
cd /app/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 \
    --extra-model-paths-config extra_model_paths.yaml \
    --preview-method latent2rgb \
    --cache-none --lowvram \
    > /workspace/logs/comfyui.log 2>&1 &
COMFYUI_PID=$!
echo "ComfyUI started (PID $COMFYUI_PID)"

# ---------- 5. Wait for ComfyUI ready ----------
echo "Waiting for ComfyUI..."
for i in $(seq 1 180); do
    if curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI ready after ${i}s"
        break
    fi
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died. Check /workspace/logs/comfyui.log"
        tail -50 /workspace/logs/comfyui.log 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

if ! curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; then
    echo "ERROR: ComfyUI failed to start within 180s"
    tail -50 /workspace/logs/comfyui.log 2>/dev/null || true
    exit 1
fi

# ---------- 6. Start daemon (foreground) ----------
echo "Starting wanly-gpu-daemon..."
cd "$DAEMON_DIR"
exec python3 -m daemon.main
