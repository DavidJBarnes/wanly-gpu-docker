#!/bin/bash
# The engine and supervisor code, fetched at boot (wanly-gpu-docker#116).
#
# Same pattern and same reasons as fetch_daemon.sh (#83) and fetch_services.sh
# (wanly-services#38): the image is the environment, the code is fetched fresh every boot, so
# a code fix does not wait on a 40+ minute image build and a 13 GiB pull.
#
# THE BRANCH GATE IS NOT COSMETICS HERE. engine/workflows/ carries the validated DR34 graph
# that every render is measured against, and CI gates main — ENGINE_BRANCH exists for pre-merge
# hardware tests and shouts, same as DAEMON_BRANCH.
#
# WHAT IS NOT FETCHED, ON PURPOSE: ComfyUI and the node packs (pinned commits in the image —
# a drifted node pack changes workflow behaviour silently), models, and the CUDA/torch stack.
# Those are the environment; they change by image rebuild, and update-worker.sh still
# converges them on :latest when the worker is idle.
set -e

REPO="${ENGINE_REPO:-https://github.com/DavidJBarnes/wanly-gpu-docker.git}"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    REPO="https://${GITHUB_TOKEN}@github.com/DavidJBarnes/wanly-gpu-docker.git"
fi
BRANCH="${ENGINE_BRANCH:-main}"
SRC_DIR="${ENGINE_SRC_DIR:-/opt/engine-src}"
# Dest paths are overridable so the boot path is testable off-box (and so a dev mount can
# point them elsewhere without editing the script). The defaults are the image layout.
ENGINE_DEST="${ENGINE_DEST:-/opt/engine}"
WORKER_DEST="${WORKER_DEST:-/app/wanly_worker}"
CODE_REF="${CODE_REF_FILE:-/run/wanly/code_ref}"

fetch_ok=1
if [ -d "$SRC_DIR/.git" ]; then
    echo "engine: updating code ($BRANCH)..."
    cd "$SRC_DIR"
    if ! git fetch --depth 1 origin "$BRANCH" 2>/dev/null \
        || ! git checkout -B "$BRANCH" FETCH_HEAD 2>/dev/null; then
        fetch_ok=0
    fi
else
    echo "engine: cloning code ($BRANCH)..."
    mkdir -p "$SRC_DIR"
    cd "$SRC_DIR"
    git init -q 2>/dev/null || true
    git remote add origin "$REPO" 2>/dev/null || git remote set-url origin "$REPO"
    if ! git fetch --depth 1 origin "$BRANCH" 2>/dev/null \
        || ! git checkout -B "$BRANCH" FETCH_HEAD 2>/dev/null; then
        fetch_ok=0
    fi
fi

if [ "$fetch_ok" != "1" ]; then
    echo "WARN: could not fetch $BRANCH — running BAKED image code instead (${WANLY_IMAGE_REF:-unknown})"
    exit 0
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "$BRANCH" != "main" ]; then
    echo "!! NOT running engine main — this worker is on branch '$BRANCH'"
fi

# Swap each fetched package over its baked copy: copy to a temp name, mv twice. A failed copy
# leaves the baked code exactly where it was — the fallback must survive a failed upgrade.
_swap_pkg() {
    local src="$1" dst="$2"
    [ -d "$src" ] || { echo "!! fetched tree has no $src — keeping baked $(basename "$dst")"; return 1; }
    rm -rf "$dst.new"
    cp -a "$src" "$dst.new"
    rm -rf "$dst.old"
    mv "$dst" "$dst.old"
    mv "$dst.new" "$dst"
    rm -rf "$dst.old"
}

_swap_pkg "$SRC_DIR/engine" "$ENGINE_DEST" || exit 1
_swap_pkg "$SRC_DIR/wanly_worker" "$WORKER_DEST" || exit 1

# The running code's identity, for /health and the boot banner (control.py reads this file).
# mkdir here, not in start.sh: this script runs before start.sh's own mkdir line.
mkdir -p "$(dirname "$CODE_REF")"
echo "$BRANCH @ $SHA" > "$CODE_REF"
echo "engine+supervisor: $BRANCH @ $SHA"

# Deps added since the image was built (engine + supervisor requirements), same loop shape as
# fetch_daemon.sh. A failed install ABORTS: the supervisor would otherwise come up serving
# workflows against a half-installed stack, which is the config-drift failure that has cost
# days here before.
for REQ in "$SRC_DIR/engine/requirements.txt" "$SRC_DIR/wanly_worker/requirements.txt"; do
    [ -f "$REQ" ] || continue
    MISSING=""
    while IFS= read -r line; do
        pkg=$(echo "$line" | sed -E 's/[=<>!;].*//' | tr -d '[:space:]')
        [ -z "$pkg" ] && continue
        case "$pkg" in \#*) continue ;; esac
        python3 -c "import importlib.metadata,sys; importlib.metadata.version('$pkg')" 2>/dev/null \
            || MISSING="$MISSING $pkg"
    done < "$REQ"
    if [ -n "$MISSING" ]; then
        echo "Installing deps added since the image was built:$MISSING"
        pip install --no-cache-dir -q $MISSING || { echo "!! DEP INSTALL FAILED:$MISSING"; exit 1; }
    fi
done
