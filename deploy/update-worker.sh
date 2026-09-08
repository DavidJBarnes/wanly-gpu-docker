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
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

running_image=$(docker inspect -f '{{.Image}}' "$NAME")

log "pulling $IMAGE"
docker pull -q "$IMAGE" >/dev/null
latest_image=$(docker image inspect "$IMAGE" -f '{{.Id}}')

if [ "$running_image" = "$latest_image" ]; then
    log "already on the published image ($(echo "$latest_image" | cut -c8-19)) — nothing to do"
    exit 0
fi

log "image changed: $(echo "$running_image" | cut -c8-19) -> $(echo "$latest_image" | cut -c8-19)"

# TWO independent signals, and BOTH must say idle. Each covers the other's blind spot.
#
#   1. THE WORKER'S OWN STATUS, from the API. The daemon sets online-busy the instant it
#      receives a claim, BEFORE [1/6], and online-idle only once the segment finishes. That
#      is the only signal covering the WHOLE claim.
#
#      This is the check that was missing, and its absence cost a segment. The first version
#      asked only the engine -- which knows nothing until [3/6] Submitting. A container was
#      recreated 50% through a 673 MB LoRA download in [2/6]; the engine truthfully said
#      running=0, the segment was abandoned mid-claim, and because registration reuses the
#      worker row it was pinned to a live busy worker where no reclaim rule could reach it.
#      It sat in PROCESSING for seven hours.
#
#      The gap widened the same day: console#423 lets a worker fetch a 46 GB checkpoint on
#      demand, which is ~20 minutes inside [2/6] with the engine reporting idle throughout.
#
#   2. THE ENGINE. Kept, because the daemon's status push can fail -- the API's own reclaim
#      logic says so. If the daemon claims idle while the engine is rendering, believe the
#      engine.
#
# Asked with `docker exec`, NOT through the published port: the engine binds 127.0.0.1 INSIDE
# the container, so -p 8190:8190 resolves to nothing and a host curl returns empty -- which
# parsed naively reads as "idle".
worker_status=$(curl -sf --max-time 15 -H "X-API-Key: ${QUEUE_API_KEY:-}" \
                  "${QUEUE_URL:-}/workers" 2>/dev/null \
                | FRIENDLY_NAME="${FRIENDLY_NAME:-}" python3 -c '
import json, os, sys
name = os.environ.get("FRIENDLY_NAME", "")
try:
    rows = json.load(sys.stdin)
except Exception:
    print("unreadable"); raise SystemExit
me = [w for w in rows if w.get("friendly_name") == name]
# Not finding ourselves is NOT idleness. A worker mid-boot has not registered yet, and
# recreating then interrupts model staging.
print(me[0].get("status") or "unknown" if me else "not-registered")
' 2>/dev/null || echo unreadable)

if [ "$worker_status" != "online-idle" ]; then
    log "worker status is '$worker_status' (want online-idle) — leaving it alone, will retry next run"
    exit 0
fi

busy=$(docker exec "$NAME" curl -sf --max-time 10 http://127.0.0.1:8190/health 2>/dev/null \
       | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("running") or 0)+(d.get("queue_depth") or 0))' 2>/dev/null || echo unknown)

if [ "$busy" = "unknown" ]; then
    # Fail SAFE: an unreachable engine is not evidence of idleness. It may be mid-boot, and
    # recreating then would interrupt model staging.
    log "could not read the engine's health — assuming busy, will retry next run"
    exit 0
fi
if [ "$busy" != "0" ]; then
    # The daemon said idle and the engine disagrees. Believe the engine: a failed status push
    # is a known mode, and being wrong here costs a render.
    log "engine reports $busy job(s) in flight despite status '$worker_status' — leaving it alone"
    exit 0
fi

# THE TRAINER SHARES THIS CARD. A training run drains the render worker, and a drained
# worker is exactly what looks idle. Recreating it mid-run brought it back with no drain on
# 2026-09-08; it claimed a render beside the training and the box hard-reset. The trainer's
# control API says whether it is training; if it cannot be asked, assume it is.
if curl -sf --max-time 5 "http://127.0.0.1:${TRAINER_CONTROL_PORT:-8083}/health" >/dev/null 2>&1; then
    training=$(curl -sf --max-time 5 "http://127.0.0.1:${TRAINER_CONTROL_PORT:-8083}/health" \
               | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unknown"); raise SystemExit
# Only a trainer answers with a services list; anything else on that port is not one.
services = d.get("services") if isinstance(d, dict) else None
if not isinstance(services, list):
    print("no"); raise SystemExit
print("yes" if any(isinstance(s, dict) and s.get("training") for s in services) else "no")
' 2>/dev/null || echo unknown)
    if [ "$training" != "no" ]; then
        log "the trainer on this box reports training=$training — leaving the worker alone, will retry next run"
        exit 0
    fi
fi
log "worker is idle — recreating on the new image"
"$HERE/run-worker.sh"
log "done"
