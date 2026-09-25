#!/usr/bin/env bash
# Converge a long-lived worker on :latest, but never mid-render (wanly-gpu-docker#72).
#
# Run from a timer. Does nothing at all unless the published image has actually changed, so
# it is cheap to run often.
#
# WHY IDLE MATTERS MORE THAN FRESHNESS
#     A render is 10-13 minutes (measured: 613s and 759s on the 3090). Recreating the
#     container mid-render throws that work away, and the segment is then reclaimed by the
#     stale-heartbeat path -- so an eager updater costs more than the drift it fixes. This
#     checks the worker is not processing and, if it is, exits and leaves the next run to it.
#
# WHY IT COMPARES DIGESTS, NOT TAGS
#     :latest is a moving pointer. "Am I on latest?" is only answerable by comparing the
#     digest the container was created from against the digest the tag resolves to NOW.
#
# WHY IT LOCKS (wanly-gpu-docker#97)
#     Two copies of this script running at once — a manual run while a timer run is mid-pull,
#     which is 13 GiB and ~11 minutes on the 3090's link, so the window is wide — both see
#     "image changed" and both run run-worker.sh. One wins; another dies on a container-name
#     conflict, or worse removes the winner's container between its rm -f and its run.
#     flock -n makes the loser exit 0 with a message instead: stacked firings are made
#     harmless rather than impossible. The FD stays open for the script's whole life,
#     including the `exec run-worker.sh` paths, so it covers the spawn too.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# /run/user/$UID exists on a normal login ($XDG_RUNTIME_DIR points there); /run/lock as a
# fallback for cron-style contexts where it does not.
LOCK_DIR="${XDG_RUNTIME_DIR:-/run/lock}"
LOCK_FILE="$LOCK_DIR/wanly-worker-update.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another update is in progress — nothing to do"
    exit 0
fi

IMAGE="${IMAGE:-davidjbarnes/wanly-gpu-docker:latest}"
NAME="${NAME:-wanly-gpu-docker}"

# Needed for the idle check below: QUEUE_URL, QUEUE_API_KEY and FRIENDLY_NAME identify this
# worker to the API. Sourced rather than required, so the script still runs (and still
# refuses to act) on a box where the file is absent -- an unreadable status counts as busy.
ENV_FILE="${WORKER_ENV:-$HERE/worker.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; . "$ENV_FILE"; set +a
fi

# The idle gate, shared with wanly-mode.sh. Sourced AFTER worker.env: it reads QUEUE_URL,
# QUEUE_API_KEY, FRIENDLY_NAME and CONTROL_PORT from the environment.
# shellcheck source=deploy/worker-idle.sh
. "$HERE/worker-idle.sh"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# The container was called wanly-ltx before #77 renamed it. ADOPT it rather than walking past
# it into run-worker.sh, which would leave two containers with the same FRIENDLY_NAME claiming
# from the same queue -- the API identifies a worker by that name and cannot tell them apart,
# so they would fight over segments and each would look like the other going wrong.
#
# A rename keeps the running container exactly as it is, so this costs nothing and is a no-op
# once done. Harmless to leave in place.
LEGACY_NAME="wanly-ltx"
if ! docker inspect "$NAME" >/dev/null 2>&1 && docker inspect "$LEGACY_NAME" >/dev/null 2>&1; then
    log "adopting the pre-#77 container $LEGACY_NAME as $NAME"
    docker rename "$LEGACY_NAME" "$NAME"
fi

if ! docker inspect "$NAME" >/dev/null 2>&1; then
    log "no container named $NAME — creating it"
    exec "$HERE/run-worker.sh"
fi

# A CONTAINER THAT EXISTS BUT IS NOT RUNNING IS DEAD, NOT BUSY (wanly-gpu-docker#105).
#
# The failure that made this check necessary, twice: an nvidia driver/toolkit event
# regenerates /var/run/cdi/nvidia.yaml, the container's GPU dies with it, and its restart
# fails with `CDI device injection failed: unresolvable CDI devices nvidia.com/gpu=all` --
# docker never retries (RestartCount stayed 0), so the box sits with no worker for hours.
#
# A dead container is otherwise INVISIBLE to this script: `docker inspect` succeeds, the
# digest comparison below says "nothing to do", and every health check fails -- which every
# later check reads as BUSY and walks away. A dead container cannot be busy; its processes
# are gone. Recreate it here, BEFORE the pull and the idle checks, so recovery is one timer
# tick (~35 minutes with the randomized delay), not one human.
#
# run-worker.sh is the recreate path rather than `docker start`: it is the documented
# converge (safe to re-run, mounts/env from worker.env), and if the CDI spec is momentarily
# invalid -- the recreate lands inside the same driver-event window -- its `docker run`
# fails the same way `docker start` would, and the next timer tick retries.
if [ -z "$(docker ps -q -f "name=^/${NAME}$")" ]; then
    log "$NAME exists but is not running (state: $(docker inspect -f '{{.State.Status}}' "$NAME")) — recreating it"
    exec "$HERE/run-worker.sh"
fi

running_image=$(docker inspect -f '{{.Image}}' "$NAME")

log "pulling $IMAGE"
docker pull -q "$IMAGE" >/dev/null
latest_image=$(docker image inspect "$IMAGE" -f '{{.Id}}')

if [ "$running_image" = "$latest_image" ]; then
    log "already on the published image ($(echo "$latest_image" | cut -c8-19)) — nothing to do"
    exit 0
fi

log "image changed: $(echo "$running_image" | cut -c8-19) -> $(echo "$latest_image" | cut -c8-19)"

# IS IT SAFE TO RECREATE? Three signals, all of which must say idle, and all of which live
# in worker-idle.sh -- one definition, shared with wanly-mode.sh (#131), because a second
# hand-maintained copy of these rules is how one of the two quietly stops knowing about the
# third. The reasons each exist, and what each one cost, are in that file's header.
reason="$(worker_idle_reason "$NAME")"
if [ -n "$reason" ]; then
    log "$reason — leaving it alone, will retry next run"
    exit 0
fi

log "worker is idle — recreating on the new image"
"$HERE/run-worker.sh"
log "done"
