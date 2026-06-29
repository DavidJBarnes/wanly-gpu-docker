#!/bin/bash
set -e

echo "=== Wanly RunPod Worker Starting ==="

# ---------- 0. Start sshd for direct TCP access (RunPod injects PUBLIC_KEY) ----------
# Lets us SSH straight into the container over exposed TCP instead of the interactive
# proxy, which can't run scripted commands.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    mkdir -p /run/sshd
    ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd 2>/dev/null && echo "sshd started (direct TCP on port 22)" || echo "WARN: sshd failed to start"
fi

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
# Generation config defaults = the 3090's VALIDATED pipeline (source of truth), so a fresh
# worker boots on parity. DaSiWa "Lightspeed" remix with the distillation baked in -> external
# lightx2v OFF (strength 0). All overridable via pod env vars if needed.
cat > "$DAEMON_DIR/.env" << EOF
QUEUE_URL=${QUEUE_URL:-http://api.wanly22.com:8001}
FRIENDLY_NAME=${FRIENDLY_NAME:-runpod-${RUNPOD_POD_ID:-unknown}}
COMFYUI_URL=http://localhost:8188
COMFYUI_PATH=/app/ComfyUI
LORA_CACHE_DIR=/workspace/models/loras
UNET_HIGH_MODEL=${UNET_HIGH_MODEL:-DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors}
UNET_LOW_MODEL=${UNET_LOW_MODEL:-DasiwaWAN22I2V14BLightspeed_snatchkissLowV11.safetensors}
LIGHTX2V_STRENGTH_HIGH=${LIGHTX2V_STRENGTH_HIGH:-0.0}
LIGHTX2V_STRENGTH_LOW=${LIGHTX2V_STRENGTH_LOW:-0.0}
PAINTER_MOTION_AMPLITUDE=${PAINTER_MOTION_AMPLITUDE:-1.4}
PAINTER_MOTION_FRAMES=${PAINTER_MOTION_FRAMES:-6}
UNET_WEIGHT_DTYPE=${UNET_WEIGHT_DTYPE:-fp8_e4m3fn}
HIGH_NOISE_REALISM=${HIGH_NOISE_REALISM:-false}
STEPS_TOTAL=${STEPS_TOTAL:-4}
HIGH_NOISE_STEPS=${HIGH_NOISE_STEPS:-2}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}
QUEUE_API_KEY=${QUEUE_API_KEY:-}
EOF

echo "Daemon config:"
cat "$DAEMON_DIR/.env"

# ---------- 3b. Disable ReActor NSFW filter ----------
# ReActor's hardcoded NSFW filter drops video frames it considers unsafe,
# which breaks RIFE (needs >= 2 frames). Inject early return into nsfw_image()
# so the NSFW model never loads or runs (same approach as local 3090).
#
# We also neutralize reactor_sfw.py's top-level `from transformers import pipeline`.
# ReActor's requirements.txt is unpinned and now pulls transformers >= 5.x, which
# references torch.float8_e8m0fnu (torch >= 2.7) at import time. This image pins
# torch 2.6.0, so that import raises AttributeError, the whole reactor node fails
# to register, and ComfyUI rejects any workflow using ReActorOptions with a 400.
# `pipeline` is only used inside nsfw_image(), which we disable above, so the import
# is dead weight — replacing it with a stub sidesteps the version skew entirely.
REACTOR_SFW="/app/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py"
if [ -f "$REACTOR_SFW" ]; then
    sed -i '/^def nsfw_image/a\    return False  # NSFW filter disabled by wanly start.sh' "$REACTOR_SFW"
    sed -i '1s|^from transformers import pipeline.*|pipeline = None  # transformers import disabled by wanly (torch<2.7 incompat)|' "$REACTOR_SFW"
    echo "Patched ReActor: nsfw_image() returns False (disabled), transformers import stubbed"
fi

# ---------- 4. Start ComfyUI (background, no auth) ----------
mkdir -p /workspace/logs
cd /app/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 \
    --extra-model-paths-config extra_model_paths.yaml \
    --preview-method latent2rgb \
    --cache-none \
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
