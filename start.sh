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

# ---------- Daemon deps ----------
# The image already installed these from daemon-requirements.txt at build time. This only has
# to catch deps ADDED to the daemon since the image was built.
#
# It used to be `pip install -r requirements.txt 2>/dev/null || true`, which had two problems:
# every error was swallowed, so a failed install looked identical to a clean one; and it
# reinstalled onnxruntime (CPU) over the onnxruntime-gpu that ReActor and FaceFusion need,
# plus opencv-python-headless over opencv-python -- both silently, on every boot, because the
# two packages in each pair provide the same module.
#
# So: skip the two that collide, install the rest, and SAY SO when something fails.
SKIP_DEPS="onnxruntime|opencv-python-headless"
if [ -f "$DAEMON_DIR/requirements.txt" ]; then
    MISSING=""
    while read -r dep; do
        [ -z "$dep" ] && continue
        case "$dep" in \#*) continue ;; esac
        echo "$dep" | grep -qE "^($SKIP_DEPS)([=<>]|$)" && continue
        mod=$(echo "$dep" | sed -E 's/[=<>!].*//')
        pip show "$mod" >/dev/null 2>&1 || MISSING="$MISSING $dep"
    done < "$DAEMON_DIR/requirements.txt"
    if [ -n "$MISSING" ]; then
        echo "Daemon deps added since image build:$MISSING"
        # shellcheck disable=SC2086
        pip install --no-cache-dir -q $MISSING || echo "!! DAEMON DEP INSTALL FAILED:$MISSING"
    fi
fi

# Loud, early warning if the image's pinned copy has drifted from the daemon's actual list.
if [ -f /app/daemon-requirements.txt ] && [ -f "$DAEMON_DIR/requirements.txt" ]; then
    DRIFT=$(comm -13 \
        <(grep -vE '^\s*(#|$)' /app/daemon-requirements.txt | sed -E 's/[=<>!].*//' | sort -u) \
        <(grep -vE '^\s*(#|$)' "$DAEMON_DIR/requirements.txt" | sed -E 's/[=<>!].*//' | sort -u) \
        | grep -vE "^($SKIP_DEPS)$" || true)
    [ -n "$DRIFT" ] && echo "!! daemon-requirements.txt is stale, missing:" $DRIFT
fi

# ---------- 3. Write daemon .env ----------
# Parity with the 3090 is checked by diffing this block against that box's .env, NOT by memory.
# It drifted once before (DaSiWa + lightx2v 0 vs base + lightx2v 2.0) and cost a full day of
# bogus LoRA results, because the failure is silent: jobs run and produce plausible output that
# is not comparable to anything.
#
# Last diffed 2026-08-06. Only PAINTER_MOTION_AMPLITUDE differed (was 1.4 here vs 1.7 there);
# everything else either matched or fell through to an identical daemon default.
#
# NOTE: the 3090's .env carries SHIFT_HIGH=5.0, which is a DEAD variable -- the daemon setting
# is `flow_shift`, there is no `shift_high`, so it has never had any effect. flow_shift defaults
# to 5.0 regardless, so behaviour is correct by accident. Do not copy it here and do not trust
# it there.
# Generation config defaults = the 3090's CURRENT base-model config (source of truth), so a
# fresh worker boots on parity: base Wan 2.2 i2v, NATIVE fp8-scaled (the fast path on the current
# ComfyUI build — no fp16->fp8 manual cast), 10 steps (5 high / 5 low), cfg 1. lightx2v/cfg are
# typically overridden per-job from the Wanly UI.
cat > "$DAEMON_DIR/.env" << EOF
QUEUE_URL=${QUEUE_URL:-http://api.wanly22.com:8001}
FRIENDLY_NAME=${FRIENDLY_NAME:-runpod-${RUNPOD_POD_ID:-unknown}}
COMFYUI_URL=http://localhost:8188
COMFYUI_PATH=/app/ComfyUI
LORA_CACHE_DIR=/workspace/models/loras
UNET_HIGH_MODEL=${UNET_HIGH_MODEL:-wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors}
UNET_LOW_MODEL=${UNET_LOW_MODEL:-wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors}
LIGHTX2V_STRENGTH_HIGH=${LIGHTX2V_STRENGTH_HIGH:-0.0}
LIGHTX2V_STRENGTH_LOW=${LIGHTX2V_STRENGTH_LOW:-0.0}
PAINTER_MOTION_AMPLITUDE=${PAINTER_MOTION_AMPLITUDE:-1.7}
PAINTER_MOTION_FRAMES=${PAINTER_MOTION_FRAMES:-6}
UNET_WEIGHT_DTYPE=${UNET_WEIGHT_DTYPE:-fp8_e4m3fn}
HIGH_NOISE_REALISM=${HIGH_NOISE_REALISM:-false}
CFG_HIGH=${CFG_HIGH:-1.0}
CFG_LOW=${CFG_LOW:-1.0}
STEPS_TOTAL=${STEPS_TOTAL:-10}
HIGH_NOISE_STEPS=${HIGH_NOISE_STEPS:-5}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}
QUEUE_API_KEY=${QUEUE_API_KEY:-}
EOF

# Dump the resolved config so a parity problem is visible in the first lines of the boot log
# rather than after a day of results that quietly are not comparable to anything.
# Secrets redacted: RunPod surfaces container logs in its console, and `cat` was printing
# QUEUE_API_KEY and RUNPOD_API_KEY in plaintext.
echo "Daemon config:"
sed -E 's/^(.*(KEY|TOKEN|SECRET|PASSWORD))=.+$/\1=<redacted>/' "$DAEMON_DIR/.env"

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

# ---------- 3c. Keep FaceFusion's install.py from replacing the pinned onnxruntime ----------
# The Dockerfile writes this marker at build time; see the long note there. Re-asserting it here
# costs nothing and covers the case where the node is updated to a commit that clears or renames
# it. Without the marker, ComfyUI's startup runs FaceFusion's install.py, which uninstalls the
# pinned onnxruntime-gpu and installs the newest — a CUDA-13 wheel this CUDA-12.4 image cannot
# load. The swap then runs on CPU, ~100x slower, with no error anywhere: the only symptom is the
# daemon's "no progress for 300s" watchdog, which reads as a hang.
FF_MARKER="/app/ComfyUI/custom_nodes/Facefusion_comfyui/.install_complete"
if [ -d "$(dirname "$FF_MARKER")" ] && [ ! -f "$FF_MARKER" ]; then
    touch "$FF_MARKER"
    echo "WARN: FaceFusion .install_complete was missing — recreated (would have clobbered onnxruntime)"
fi

# Record what onnxruntime we booted with, so a later change is visible rather than inferred.
ORT_BOOT=$(pip show onnxruntime-gpu 2>/dev/null | awk '/^Version:/{print $2}')
echo "onnxruntime-gpu at boot: ${ORT_BOOT:-<not installed>}"

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

# ---------- 5b. Did anything replace onnxruntime while ComfyUI was starting? ----------
# ComfyUI runs every custom node's install.py during startup, and at least one of them (see
# FaceFusion, above) will pip-install over a pinned dependency given the chance. Comparing the
# version now against the one recorded before ComfyUI started catches ANY such node, not just
# the one we know about — and catches it before the first job rather than 300s into one.
#
# Loud, not fatal: a mismatched onnxruntime still generates video correctly. It only makes the
# faceswap fall back to CPU, and a worker that produces slow-but-correct output beats a worker
# that refuses to boot.
ORT_NOW=$(pip show onnxruntime-gpu 2>/dev/null | awk '/^Version:/{print $2}')
if [ -n "$ORT_BOOT" ] && [ "$ORT_NOW" != "$ORT_BOOT" ]; then
    echo "!! onnxruntime-gpu CHANGED during ComfyUI startup: $ORT_BOOT -> $ORT_NOW"
    echo "!! A custom node's install.py replaced the pinned build. If the new wheel targets a"
    echo "!! different CUDA major than this image, the GPU faceswap will silently run on CPU"
    echo "!! (~100x slower) and surface as the daemon's 'no progress for 300s' watchdog."
fi

# ---------- 6. Start daemon (foreground) ----------
echo "Starting wanly-gpu-daemon..."
cd "$DAEMON_DIR"
exec python3 -m daemon.main
