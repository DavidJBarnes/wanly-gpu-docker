#!/bin/bash
# The render daemon's code and config, fetched at boot (wanly-gpu-docker#83).
#
# This is start.sh's old phase 2 and 3, moved here verbatim so the supervisor can run it as
# the render-daemon service's preflight. The image is the environment; the daemon is the
# code, pulled fresh every boot so a daemon fix does not require an image rebuild.
set -e
# ---------- 2. Daemon code ----------
# NOT baked into the image. The image is the environment; the daemon is the code, pulled fresh
# every boot so a daemon fix does not require an image rebuild.
echo "render-daemon: fetching daemon code"
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
# EMPTY on purpose. With a path set, the daemon takes ownership of ComfyUI and checks for
# custom node packs, cloning the ones it thinks are missing. That is a WAN-era job and it
# breaks an LTX worker: it cloned Frame-Interpolation and ReActor into the LTX ComfyUI, and
# ReActor unpinned requirements pull transformers >=5, which breaks every workflow.
#
# It also used to run a "resource sync" that fetched model files from S3 and exited the
# daemon when that failed, restart-looping the pod. That is gone (wanly-gpu-daemon#175);
# the node check is now the only reason this stays empty.
#
# ltx-engine owns this ComfyUI. The daemon drives the engine and syncs character LoRAs
# through LORA_CACHE_DIR; it has no business installing nodes or models.
COMFYUI_PATH=
LORA_CACHE_DIR=${LORA_DIR:-/workspace/models/loras}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}
QUEUE_API_KEY=${QUEUE_API_KEY:-}
EOF

# The resolved config is printed by the render-daemon service (render_env_dump), redacted.

