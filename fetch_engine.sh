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

# Deps FIRST, swap second. The check only ever installs packages that are entirely absent
# (importlib.metadata, name-only), so if the swap later falls back to baked code the extra
# packages are additive and the baked code never sees a changed version of anything it
# imports. A failed install aborts BEFORE anything is swapped: the baked tree stays paired
# with the baked environment, and a half-installed stack must not boot with either tree.
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

# Swap each fetched package over its baked copy: stage to a temp name, then two guarded mv's.
#
# EVERY mv IS CHECKED, and this is not decoration (#121): on the 3090 the container had a
# bind mount INSIDE /opt/engine, and a directory containing a mount point cannot be renamed —
# mv fails EBUSY. Unguarded, the second mv then "succeeds" by nesting the new tree INSIDE the
# old one (mv dst.new dst -> dst/dst.new), exits 0, and the boot banner claims the fetched
# sha while the baked code runs. A silent no-op dressed as an upgrade is exactly the #72
# failure; the mount is gone now, but any future mount in these paths must degrade loudly.
_swap_pkg() {
    local src="$1" dst="$2" name
    name="$(basename "$dst")"
    [ -d "$src" ] || { echo "!! fetched tree has no $src — keeping current $name"; return 1; }
    rm -rf "$dst.new"
    if ! cp -a "$src" "$dst.new"; then
        echo "!! could not stage $name — keeping current code"
        rm -rf "$dst.new"
        return 1
    fi
    rm -rf "$dst.old"
    if ! mv "$dst" "$dst.old"; then
        echo "!! cannot move $dst aside — keeping current $name"
        echo "!!   (a mount point INSIDE $dst is the usual cause: docker inspect -f '{{json .Mounts}}')"
        rm -rf "$dst.new"
        return 1
    fi
    if ! mv "$dst.new" "$dst"; then
        echo "!! staged $name would not land — restoring previous code"
        mv "$dst.old" "$dst" || echo "!! !! ROLLBACK FAILED: $name is missing at $dst"
        rm -rf "$dst.new"
        return 1
    fi
    rm -rf "$dst.old"
}

# The running code's identity, for /health and the boot banner (control.py reads this file).
# mkdir here, not in start.sh: this script runs before start.sh's own mkdir line.
mkdir -p "$(dirname "$CODE_REF")"
swap_ok=1
_swap_pkg "$SRC_DIR/engine" "$ENGINE_DEST" || swap_ok=0
_swap_pkg "$SRC_DIR/wanly_worker" "$WORKER_DEST" || swap_ok=0
if [ "$swap_ok" = "1" ]; then
    echo "$BRANCH @ $SHA" > "$CODE_REF"
    echo "engine+supervisor: $BRANCH @ $SHA"
else
    # The honest part of #121: a failed swap keeps serving on baked code — that is what the
    # fallback is FOR — but the identity must say so. Never claim a sha the process is not
    # running.
    echo "baked-fallback (swap failed; image ${WANLY_IMAGE_REF:-unknown})" > "$CODE_REF"
    echo "!! SWAP FAILED — running BAKED code for at least one package. Not claiming $SHA."
fi
