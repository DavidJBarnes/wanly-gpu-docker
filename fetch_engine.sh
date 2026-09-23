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
# Default resolves NEXT TO THIS SCRIPT: /app/swap_sync.py in the image (start.sh runs
# /app/fetch_engine.sh by absolute path), the repo checkout off-box (the tests run this
# file from its real location). An explicit SWAP_SYNC overrides both.
SWAP_SYNC="${SWAP_SYNC:-$(dirname "$0")/swap_sync.py}"

# Swap a package tree over its current one: stage to a temp dir, then sync IN PLACE with
# swap_sync.py. Three incidents, one primitive:
#
#   #121 — a bind mount INSIDE /opt/engine makes the directory unrenamable (EBUSY).
#          Unguarded, the second move "succeeded" by nesting the new tree inside the old
#          one, exit 0, banner claiming the incoming sha while the old code ran.
#
#   #125 — guarding the mv was not enough: GNU mv treats EBUSY like EXDEV and falls back
#          to copy-then-unlink, reaching the mount only after emptying the source. The
#          "keeping current code" branch ran over a directory it had just hollowed, and
#          restart=always turned that into a crash loop saying the reassuring line.
#
#   #125b — os.rename refuses atomically (safe) but can NEVER succeed in production:
#           directories COPY'd by the image live on the overlay lower layer, and a
#           lower-layer directory cannot be renamed at all (measured: EXDEV on a plain,
#           unmounted /opt/worker). The mv copy-fallback is the only reason renaming ever
#           "worked" here — and it is the mechanism #125 destroys.
#
# In-place is the only mechanism that works against a lower layer: replace the files,
# never move the directory. swap_sync.py refuses a mount at or inside the destination
# BEFORE changing anything, so a failed swap leaves the current code exactly as it was.
_swap_pkg() {
    local src="$1" dst="$2" name
    name="$(basename "$dst")"
    [ -d "$src" ] || { echo "!! source tree has no $src — keeping current $name"; return 1; }
    rm -rf "$dst.new"
    if ! cp -a "$src" "$dst.new"; then
        echo "!! could not stage $name — keeping current code"
        rm -rf "$dst.new"
        return 1
    fi
    if [ ! -f "$SWAP_SYNC" ]; then
        # Same class of failure as #125: silently proceeding without the safe swap
        # primitive is how /opt/engine got emptied. Loud, per package, current code intact.
        echo "!! no swap primitive at $SWAP_SYNC — keeping current $name"
        rm -rf "$dst.new"
        return 1
    fi
    if ! python3 "$SWAP_SYNC" "$dst.new" "$dst"; then
        echo "!! swap refused for $name — keeping current code"
        echo "!!   (a mount point inside $dst is the usual cause: docker inspect -f '{{json .Mounts}}')"
        rm -rf "$dst.new"
        return 1
    fi
    rm -rf "$dst.new"
}

# THE DEV MOUNT (#117): iterate on engine/supervisor code in a checkout on the host box
# without a build, a pull, or even a push. run-worker.sh bind-mounts DEV_CODE_DIR read-only
# at /opt/dev-code and sets DEV_CODE=1; the mounted tree is swapped in with the same guarded
# mechanism as a fetched tree, so every consumer downstream (engine cwd, the supervisor's
# import path) is unchanged. Fetching main over the top of it would silently revert whatever
# is being developed, so the mount REPLACES the fetch.
#
# Deps are deliberately NOT installed here: a dev mount is for code iteration; a change that
# needs a new dependency is an environment change, and environment changes are the image's
# job (or a pip install in a shell, with the dev looking at it).
#
# Without the DEV_CODE=1 marker, DEV_CODE_DIR on the host changes nothing — a stale env line
# must not silently redirect a real worker.
if [ "${DEV_CODE:-0}" = "1" ]; then
    MOUNT="${DEV_MOUNT_PATH:-/opt/dev-code}"
    if [ ! -f "$MOUNT/engine/app.py" ] || [ ! -f "$MOUNT/wanly_worker/control.py" ]; then
        echo "!! DEV_CODE=1 but $MOUNT does not look like the repo (no engine/app.py or"
        echo "!! wanly_worker/control.py). Refusing to boot on it — this is the whole point:"
        echo "!! a half-mounted tree serving real claims is worse than waiting."
        exit 1
    fi
    mkdir -p "$(dirname "$CODE_REF")"
    SHA="$(git -C "$MOUNT" rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
    if _swap_pkg "$MOUNT/engine" "$ENGINE_DEST" \
        && _swap_pkg "$MOUNT/wanly_worker" "$WORKER_DEST"; then
        echo "DEV MOUNT $MOUNT @ $SHA — NOT DEPLOYED CODE"
        echo "dev-mount $MOUNT @ $SHA" > "$CODE_REF"
    else
        # Same rule as #121: if the swap failed, the mounted code is NOT running, and the
        # identity must not claim it is. A dev who sees this checks the mount before
        # believing anything their edit appears to do.
        echo "!! DEV MOUNT SWAP FAILED — the mounted code is NOT running. Booting on what"
        echo "!! is in the image instead; fix the mount before trusting any behaviour."
        echo "dev-mount-swap-FAILED $MOUNT @ $SHA — running image code" > "$CODE_REF"
    fi
    exit 0
fi

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
