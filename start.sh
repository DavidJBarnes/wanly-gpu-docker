#!/bin/bash
set -e

# ---------- Everything this script and the daemon print goes to a file as well as stdout ----
# The daemon is exec'd at the end of this script, so its stdout IS this script's stdout: the
# container log stream, and nowhere else. That is fine for a console and useless for anything
# automated — you cannot tail it over ssh, grep it, or diff two runs, because inside the
# container there is no file to read.
#
# That gap cost a whole test run. A pod was working through a job while every check for its
# progress came back empty, because they were reading a daemon.log that only exists when the
# daemon is started by hand. The pod looked idle and was reported as idle.
#
# Process substitution rather than a pipeline, so the final `exec python3 -m daemon.main` still
# replaces this shell and stays the container's main process — a pipeline would leave the shell
# alive as PID 1 and change how SIGTERM reaches the daemon on stop.
mkdir -p /workspace/logs
# Timestamp every line. Without it there is no way to tell a boot that is slow from one that is
# wedged: both look like a log that has stopped moving.
#
# A bash read loop rather than awk, because `awk` here is MAWK. mawk reads its input in blocks
# and only processes a block once it is full or the pipe closes -- and `fflush()` flushes its
# OUTPUT, which is not the side that is holding anything. During a 45-minute silent model
# download the boot produces a few hundred bytes, so nothing ever reached the log at all: on
# two real pods daemon.log sat at 0 bytes and the RunPod console showed nothing after the
# NVIDIA banner while the boot ran normally the whole time. Exactly the wedge/slow ambiguity
# this timestamping exists to remove, caused by the timestamping.
#
# It read as correct because a development box runs GAWK, which does not buffer this way, and
# because a test that lets the script EXIT flushes on close and passes. Reproduced in
# ubuntu:22.04 with the script still running: mawk 0 bytes, this loop 120 bytes.
#
# `read` on a pipe returns per line, so there is no buffering stage left to get this wrong.
exec > >(while IFS= read -r line; do
             printf '%s %s\n' "$(date +%H:%M:%S)" "$line"
         done | tee -a /workspace/logs/daemon.log) 2>&1

echo "=== Wanly GPU Worker (LTX 2.3) ==="
echo "image build: ${GIT_SHA:-unknown}"
echo "(this log is also at /workspace/logs/daemon.log)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

# ---------- 0. sshd for direct TCP access (RunPod injects PUBLIC_KEY) ----------
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    mkdir -p /run/sshd
    ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd 2>/dev/null && echo "sshd started (direct TCP on port 22)" || echo "WARN: sshd failed to start"
fi

# ---------- 1. Models ----------
echo "PHASE 1/6: models"
/app/download_models.sh

# ---------- 2. Daemon code ----------
# NOT baked into the image. The image is the environment; the daemon is the code, pulled fresh
# every boot so a daemon fix does not require an image rebuild.
echo "PHASE 2/6: daemon code"
DAEMON_DIR="/app/wanly-gpu-daemon"
DAEMON_REPO="https://github.com/DavidJBarnes/wanly-gpu-daemon.git"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    DAEMON_REPO="https://${GITHUB_TOKEN}@github.com/DavidJBarnes/wanly-gpu-daemon.git"
fi

# DAEMON_BRANCH exists so an unmerged branch can be booted on real hardware BEFORE it is
# merged. Without it the only way to test a daemon change in the image is to merge it first,
# which is exactly backwards. Defaults to main, and the resolved branch and commit are printed
# below — a worker running something other than main should never be a thing you have to infer.
DAEMON_BRANCH="${DAEMON_BRANCH:-main}"

if [ -d "$DAEMON_DIR/.git" ]; then
    echo "Updating daemon ($DAEMON_BRANCH)..."
    cd "$DAEMON_DIR"
    git fetch --depth 1 origin "$DAEMON_BRANCH" 2>/dev/null \
        && git checkout -B "$DAEMON_BRANCH" FETCH_HEAD 2>/dev/null \
        || echo "WARN: fetch/checkout failed, using existing code"
else
    echo "Cloning daemon ($DAEMON_BRANCH)..."
    git clone --depth 1 --branch "$DAEMON_BRANCH" "$DAEMON_REPO" "$DAEMON_DIR"
fi
cd "$DAEMON_DIR"
echo "daemon: $DAEMON_BRANCH @ $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "$DAEMON_BRANCH" != "main" ]; then
    echo "!! NOT running daemon main — this worker is on branch '$DAEMON_BRANCH'"
fi

# ---------- Daemon deps ----------
# The image already installed these from daemon-requirements.txt at build time. This only has
# to catch deps ADDED to the daemon since the image was built.
#
# Never `pip install -r ... || true`: that swallowed every error, so a failed install looked
# identical to a clean one.
if [ -f "$DAEMON_DIR/requirements.txt" ]; then
    MISSING=""
    while IFS= read -r line; do
        pkg=$(echo "$line" | sed -E 's/[=<>!;].*//' | tr -d '[:space:]')
        [ -z "$pkg" ] && continue
        case "$pkg" in \#*) continue ;; esac
        python3 -c "import importlib.metadata,sys; importlib.metadata.version('$pkg')" 2>/dev/null \
            || MISSING="$MISSING $pkg"
    done < "$DAEMON_DIR/requirements.txt"
    if [ -n "$MISSING" ]; then
        echo "Installing daemon deps added since the image was built:$MISSING"
        pip install --no-cache-dir -q $MISSING || echo "!! DAEMON DEP INSTALL FAILED:$MISSING"
    fi
fi

# Drift between what the image pre-installs and what the daemon actually needs is the quiet
# kind of breakage, so it is announced rather than inferred.
if [ -f /app/daemon-requirements.txt ] && [ -f "$DAEMON_DIR/requirements.txt" ]; then
    if ! diff -q \
        <(grep -vE '^\s*(#|$)' /app/daemon-requirements.txt | sed -E 's/[=<>!].*//' | sort -u) \
        <(grep -vE '^\s*(#|$)' "$DAEMON_DIR/requirements.txt" | sed -E 's/[=<>!].*//' | sort -u) \
        >/dev/null; then
        echo "WARN: daemon-requirements.txt has drifted from the daemon's own requirements.txt"
    fi
fi

# ---------- 3. Daemon config ----------
# ENGINE=ltx is the whole point of this image. The daemon defaults to wan22 so that merging LTX
# support could not retarget an existing worker; this is where it is turned on.
#
# Generation settings are NOT here any more. Under WAN this block carried sampler, steps, cfg
# and LoRA strengths, and keeping them at parity with the 3090's .env was a standing hazard —
# a different default produces plausible output that is not comparable to anything, and that
# cost a full day once. Under LTX those values live in the recipe, which wanly-api resolves and
# ships inside the claim. There is nothing here to drift.
cat > "$DAEMON_DIR/.env" << EOF
QUEUE_URL=${QUEUE_URL:-http://api.wanly22.com:8001}
FRIENDLY_NAME=${FRIENDLY_NAME:-ltx-${RUNPOD_POD_ID:-$(hostname)}}
ENGINE=ltx
LTX_ENGINE_URL=http://localhost:${API_PORT:-8190}
COMFYUI_URL=http://localhost:${COMFY_PORT:-8188}
# EMPTY on purpose. With a path set, the daemon takes ownership of ComfyUI: it checks for
# custom node packs and clones the ones it thinks are missing, and it runs a "resource sync"
# that downloads model files from S3. Both are WAN-era jobs and both break an LTX worker:
#
#   * the node check cloned Frame-Interpolation and ReActor into the LTX ComfyUI, and
#     ReActor's unpinned requirements pull transformers >=5, which breaks every workflow
#   * the resource sync tried to fetch rife49.pth, deleted with WAN, got a 404 from S3 and
#     exited -- restart-looping a pod that had just got past ComfyUI for the first time:
#
#         ERROR Resource rife49.pth: download failed ... HTTP 404
#         ERROR Resource sync failed. Exiting.
#
# ltx-engine owns this ComfyUI. The daemon drives the engine and syncs character LoRAs
# through LORA_CACHE_DIR; it has no business installing nodes or models.
COMFYUI_PATH=
LORA_CACHE_DIR=${LORA_DIR:-/workspace/models/loras}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}
QUEUE_API_KEY=${QUEUE_API_KEY:-}
EOF

# Dump the resolved config so a parity problem is visible in the first lines of the boot log
# rather than after a day of results. Secrets redacted: container logs are surfaced in consoles,
# and a plain `cat` printed QUEUE_API_KEY and RUNPOD_API_KEY in the clear.
echo "Daemon config:"
sed -E 's/^(.*(KEY|TOKEN|SECRET|PASSWORD))=.+$/\1=<redacted>/' "$DAEMON_DIR/.env" | sed 's/^/  /'

# ---------- 4. ComfyUI ----------
# Fail LOUDLY on a driver too old for this torch build, rather than letting ComfyUI die on
# import with its output going nowhere. A pod spent an hour restart-looping on exactly this:
# the container died 16s into phase 3 with an empty comfyui.log, and the reason was only
# visible by running ComfyUI by hand.
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "!! FATAL: torch cannot initialise CUDA on this host."
    echo "!! torch: $(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1)"
    echo "!! driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "!! A cu130 build needs driver >= 580. This image pins cu128, which runs on 550+."
    echo "!! If the driver is older than that, this host cannot run the image — pick another."
    exit 1
fi

echo "PHASE 3/6: starting ComfyUI on :${COMFY_PORT:-8188}"
cd /app/ComfyUI
python3 main.py --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" \
    --extra-model-paths-config /opt/extra_model_paths.yaml \
    --preview-method none \
    > /workspace/logs/comfyui.log 2>&1 &
COMFYUI_PID=$!

echo -n "PHASE 4/6: waiting for ComfyUI "
for i in $(seq 1 180); do
    curl -sf "http://localhost:${COMFY_PORT:-8188}/system_stats" >/dev/null 2>&1 && break
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
        echo; echo "ERROR: ComfyUI died on startup:"; tail -50 /workspace/logs/comfyui.log
        exit 1
    fi
    echo -n "."; sleep 1
done
echo
if ! curl -sf "http://localhost:${COMFY_PORT:-8188}/system_stats" >/dev/null 2>&1; then
    echo "ERROR: ComfyUI failed to answer within 180s"; tail -50 /workspace/logs/comfyui.log
    exit 1
fi
echo "ComfyUI up"
echo "custom nodes:"
ls -1 /app/ComfyUI/custom_nodes | grep -vE "__pycache__|example_node|websocket" | sed 's/^/  /'

# ---------- 5. ltx-engine ----------
# Between the daemon and ComfyUI, and the owner of the graph. Bound to loopback: the daemon
# reaches it over localhost and nothing outside the container should be submitting graphs.
echo "PHASE 5/6: starting ltx-engine on :${API_PORT:-8190}"
cd /opt/engine
python3 app.py --host 127.0.0.1 --port "${API_PORT:-8190}" \
    --public-base "http://localhost:${API_PORT:-8190}" \
    > /workspace/logs/ltx-engine.log 2>&1 &
ENGINE_PID=$!

echo -n "waiting for ltx-engine "
for i in $(seq 1 120); do
    curl -sf "http://localhost:${API_PORT:-8190}/health" >/dev/null 2>&1 && break
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo; echo "ERROR: ltx-engine died on startup:"; tail -50 /workspace/logs/ltx-engine.log
        exit 1
    fi
    echo -n "."; sleep 1
done
echo
if ! curl -sf "http://localhost:${API_PORT:-8190}/health" >/dev/null 2>&1; then
    echo "ERROR: ltx-engine failed to answer within 120s"; tail -50 /workspace/logs/ltx-engine.log
    exit 1
fi
# Its own health check reports whether it can see the models and the workflow. Printed rather
# than merely consulted: "models_present": false is the difference between a worker that fails
# every claim and one that works, and it should be readable in the boot log.
echo "ltx-engine up:"
curl -s "http://localhost:${API_PORT:-8190}/health" | sed 's/^/  /'

# ---------- 6. Daemon (foreground) ----------
echo "PHASE 6/6: starting wanly-gpu-daemon (claims once it registers)"
cd "$DAEMON_DIR"
exec python3 -m daemon.main
