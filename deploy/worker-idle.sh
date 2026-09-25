#!/usr/bin/env bash
# Is this worker safe to recreate right now? ONE definition, for every script that asks.
#
# WHY IT IS A FILE AND NOT A COPY
#     Two scripts recreate the container: update-worker.sh (a new image) and wanly-mode.sh
#     (a mode switch, #131). Both do it with `docker rm -f`, so both destroy an in-flight
#     segment, a model stage or a training run if they get this wrong. Every rule below was
#     bought with a real failure, and a second hand-maintained copy of them is how one of
#     the two quietly stops knowing about the third signal.
#
# THE SIGNALS, AND WHY THERE ARE THREE. Each covers the others' blind spot.
#
#   1. THE WORKER'S OWN STATUS, from the API. The daemon sets online-busy the instant it
#      receives a claim, BEFORE [1/6], and online-idle only once the segment finishes. That
#      is the only signal covering the WHOLE claim.
#
#      Its absence cost a segment on 2026-09-06. The first version asked only the engine --
#      which knows nothing until [3/6] Submitting. A container was recreated 50% through a
#      673 MB LoRA download in [2/6]; the engine truthfully said running=0, the segment was
#      abandoned mid-claim, and because registration reuses the worker row it was pinned to
#      a live busy worker where no reclaim rule could reach it. It sat in PROCESSING for
#      seven hours. The gap widened the same day: console#423 lets a worker fetch a 46 GB
#      checkpoint on demand, which is ~20 minutes inside [2/6] with the engine idle.
#
#   2. THE ENGINE. Kept, because the daemon's status push can fail -- the API's own reclaim
#      logic says so. If the daemon claims idle while the engine is rendering, believe the
#      engine.
#
#   3. THE TRAINER, which shares this card and since #83 this container. A training run
#      DRAINS the render worker, and a drained worker is exactly what looks idle. Recreating
#      mid-run brought one back with no drain on 2026-09-08; it claimed a render beside the
#      training and the box hard-reset.
#
# ANYTHING UNREADABLE COUNTS AS BUSY. A box mid-boot has not registered yet and must not be
# interrupted either; on a cold pod that is ~58 GB of staging thrown away.
#
# Sourced, not executed. Expects QUEUE_URL, QUEUE_API_KEY, FRIENDLY_NAME and CONTROL_PORT in
# the environment -- i.e. worker.env already sourced by the caller.

# Echoes NOTHING when the worker is safe to recreate, and a one-line reason when it is not.
# The reason is the message the caller prints, so it reads the same wherever it is refused.
worker_idle_reason() {
    local name="$1" worker_status busy training

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
        echo "worker status is '$worker_status' (want online-idle)"
        return 0
    fi

    # Through the supervisor's /health (#83): the ltx-engine-api entry carries the engine's
    # own running/queue_depth. `curl -s`, not `-sf`: a degraded container answers 503 and its
    # body is still the truth. An image from before the supervisor answers nothing on 8081;
    # fall back to asking the engine directly.
    #
    # Asked with `docker exec`, NOT through the published port: the engine binds 127.0.0.1
    # INSIDE the container, so -p 8190:8190 resolves to nothing and a host curl returns
    # empty -- which parsed naively reads as "idle".
    busy=$(docker exec "$name" curl -s --max-time 10 "http://127.0.0.1:${CONTROL_PORT:-8081}/health" 2>/dev/null \
           | python3 -c '
import json, sys
d = json.load(sys.stdin)
svc = {s.get("name"): s for s in d.get("services", []) if isinstance(s, dict)}
e = svc.get("ltx-engine-api")
if e is None: raise SystemExit(4)
print((e.get("running") or 0) + (e.get("queue_depth") or 0))
' 2>/dev/null \
           || docker exec "$name" curl -sf --max-time 10 http://127.0.0.1:8190/health 2>/dev/null \
           | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("running") or 0)+(d.get("queue_depth") or 0))' 2>/dev/null \
           || echo unknown)

    # An engine that is not enabled at all has no health to read. That is not evidence of
    # busyness -- it is a box in caption mode, where "is the engine rendering?" has no
    # meaning. The worker-status check above still covers a claim, and a container with no
    # ltx-engine holds none.
    if [ "$busy" = "unknown" ] && ! _services_include "$name" ltx-engine; then
        busy=0
    fi
    if [ "$busy" = "unknown" ]; then
        # Fail SAFE: an unreachable engine is not evidence of idleness. It may be mid-boot,
        # and recreating then would interrupt model staging.
        echo "could not read the engine's health -- assuming busy"
        return 0
    fi
    if [ "$busy" != "0" ]; then
        # The daemon said idle and the engine disagrees. Believe the engine: a failed status
        # push is a known mode, and being wrong here costs a render.
        echo "the engine reports $busy job(s) in flight despite status '$worker_status'"
        return 0
    fi

    # `training` is non-null on the lora-trainer entry while a run is on. Unparseable counts
    # as training. A container without the trainer has no such entry and nothing to wait for.
    training=$(docker exec "$name" curl -s --max-time 10 "http://127.0.0.1:${CONTROL_PORT:-8081}/health" 2>/dev/null \
               | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unknown"); raise SystemExit
services = d.get("services") if isinstance(d, dict) else None
if not isinstance(services, list):
    print("unknown"); raise SystemExit
trainer = [s for s in services if isinstance(s, dict) and s.get("name") == "lora-trainer"]
if not trainer:
    print("no"); raise SystemExit
print("yes" if any(s.get("training") for s in trainer) else "no")
' 2>/dev/null || echo unknown)

    # Same exemption as the engine: no trainer enabled, nothing to read, nothing to wait for.
    if [ "$training" = "unknown" ] && ! _services_include "$name" lora-trainer; then
        training=no
    fi
    if [ "$training" != "no" ]; then
        echo "the trainer in this container reports training=$training"
        return 0
    fi

    # Idle: no reason, and an EXPLICIT success. The last test above is false here, so
    # without this the function would return 1 and `set -e` would kill the caller on the
    # one path that is supposed to continue.
    return 0
}

# What the RUNNING container was actually given, read off the container rather than off
# worker.env -- the file may already have been rewritten by the time we ask.
_services_include() {
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1" 2>/dev/null \
        | sed -n 's/^SERVICES=//p' | head -1 | grep -q "\(^\|,\)$2\(,\|$\)"
}
