#!/usr/bin/env bash
# Switch a worker box between RENDERING and CAPTIONING, one command (#131).
#
# THE PROBLEM
#     The 3090 runs every service in one container -- SERVICES=ltx-engine,lora-trainer,
#     image-description,face-crop -- which is right: one box, one row, one GPU, no idle
#     copies of a 46 GB checkpoint. But with a job queued the box is continuously
#     online-busy, and the captioner's busy guard (wanly-api app/joycaption.py
#     busy_render_beside_the_captioner) then refuses captions rather than fighting the
#     render for VRAM. So after starting a job you cannot batch-caption the rest of the
#     repo until the queue drains. "Locked in captioning."
#
#     The one-image-many-modes design already supports the answer; it just had no lever.
#     This is the lever.
#
#         caption   SERVICES=image-description,face-crop
#                   No render daemon exists, so queued jobs simply WAIT -- nothing is lost,
#                   nothing is claimed, and the card is ollama's alone.
#         render    the full line back, from SERVICES_RENDER in worker.env.
#
# PAUSE/RESUME IS USUALLY THE BETTER ANSWER, and always is while a segment is in flight:
#
#     wanly-mode.sh pause     # finish the current segment, park, free the card
#     ...queue jobs, caption as much as you like...
#     wanly-mode.sh resume    # everything queued starts processing
#
# No recreate, no boot, no model re-stage, and the in-flight segment FINISHES instead of
# being destroyed. The daemon already does the parking itself -- on a live drain it finishes
# the segment, calls ComfyUI /free and stops claiming (wanly-gpu-daemon#182, main.py:283) --
# so there is no separate /free to remember.
#
# IT WAITS, AND THE WAIT IS THE POINT. `POST /drain` sets the worker to `draining`
# IMMEDIATELY, and wanly-api refuses an interactive caption only on `online-busy`
# (app/joycaption.py busy_render_beside_the_captioner). So the instant you drain, the
# captioner is un-gated -- while the render is still going, on a card that sits at ~23 of
# 24 GB mid-render. Captioning in that window is exactly the VRAM fight the guard exists to
# prevent. `pause` blocks until the engine reports nothing in flight, so the window is
# closed by the time it returns.
#
# `caption` MODE is for when you want the box captioning for a long stretch and would rather
# not have the render stack resident at all; `pause` is for everything else.
#
# WHY IT REFUSES WHILE BUSY
#     Switching means run-worker.sh, and run-worker.sh recreates with `docker rm -f`. That
#     kills an in-flight segment exactly the way #72 and 2026-09-06 did, and the abandoned
#     claim is pinned to a live worker row where no reclaim rule can reach it. So the same
#     three-signal idle gate the update timer uses applies here -- see worker-idle.sh, which
#     is the one place those rules live. --force says you mean it anyway.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${WORKER_ENV:-$HERE/worker.env}"
[ -f "$ENV_FILE" ] || { echo "!! no $ENV_FILE -- copy worker.env.example and fill it in"; exit 1; }

usage() {
    cat <<'USAGE'
usage: wanly-mode.sh pause|resume|caption|render|status [--force]

  pause     stop claiming, FINISH the segment in flight, park and free the card.
            No recreate. Queue jobs and caption freely; nothing is lost.
  resume    start claiming again -- everything queued processes.

  status    what the container is running now, and whether it could be switched
  caption   image-description + face-crop only, via a recreate; for a long
            captioning stretch where the render stack need not be resident
  render    the full service line back (SERVICES_RENDER in worker.env)
  --force   switch even though the worker is busy. This DESTROYS an in-flight
            segment or training run.
USAGE
}

FORCE=0
ACTION=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        caption|render|status|pause|resume) ACTION="$arg" ;;
        -h|--help) ACTION="help" ;;
        # A typo must not exit 0. A wrapper that reads `wanly-mode.sh caprion || echo failed`
        # would otherwise be told the switch happened.
        *) echo "!! unknown argument: $arg"; usage >&2; exit 1 ;;
    esac
done


[ -n "$ACTION" ] || { usage; exit 1; }
[ "$ACTION" != "help" ] || { usage; exit 0; }

# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
# shellcheck source=deploy/worker-idle.sh
. "$HERE/worker-idle.sh"

NAME="${NAME:-wanly-gpu-docker}"
SERVICES_CAPTION="${SERVICES_CAPTION:-image-description,face-crop}"

# The render line is whatever the box was running the first time it left render mode. Taken
# from SERVICES rather than hardcoded, and WRITTEN BACK, because the full set differs per box
# (a trainer box has lora-trainer; a lean render box has neither) and a restore that guessed
# would silently drop a service. Once recorded it is the durable answer to "what is this box
# for", surviving any number of switches.
render_line() {
    if [ -n "${SERVICES_RENDER:-}" ]; then
        echo "$SERVICES_RENDER"
    elif [ -n "${SERVICES:-}" ] && [ "$SERVICES" != "$SERVICES_CAPTION" ]; then
        echo "$SERVICES"
    else
        echo ""
    fi
}

# Mode is decided by WHETHER THE RENDER DAEMON EXISTS, not by matching a service list: that
# is the only thing the question turns on. Anything else is an honest "custom".
mode_of() {
    case ",${1}," in
        *,ltx-engine,*) echo render; return ;;
    esac
    if [ "$1" = "$SERVICES_CAPTION" ]; then echo caption; else echo custom; fi
}

# Rewrite one KEY=value line in place, or append it if it is not there. Anchored on `KEY=`,
# so `#SERVICES=` examples and `SERVICES_RENDER=` are left alone. The file carries the queue
# key, so its permissions are preserved rather than recreated.
set_env_line() {
    local key="$1" value="$2" tmp
    tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
    cp -p "$ENV_FILE" "$tmp"
    if grep -q "^${key}=" "$ENV_FILE"; then
        awk -v k="$key" -v v="$value" '
            !done && index($0, k "=") == 1 { print k "=" v; done=1; next }
            { print }
        ' "$ENV_FILE" > "$tmp"
    else
        { cat "$ENV_FILE"; echo "${key}=${value}"; } > "$tmp"
    fi
    mv "$tmp" "$ENV_FILE"
}

# What the RUNNING container has, which is the truth. worker.env is only what it would get on
# the next recreate -- and after a failed switch, or a hand-edit, the two disagree.
running_services() {
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null \
        | sed -n 's/^SERVICES=//p' | head -1
}

# ---- pause / resume: the drain, without a recreate ------------------------------------
#
# This box's worker row, found by FRIENDLY_NAME -- the same way update-worker.sh identifies
# it. The id is not knowable from the host otherwise, and hand-assembling a UUID into a curl
# is the reason this path went unused.
worker_field() {
    curl -sf --max-time 15 -H "X-API-Key: ${QUEUE_API_KEY:-}" "${QUEUE_URL:-}/workers" 2>/dev/null \
        | FRIENDLY_NAME="${FRIENDLY_NAME:-}" FIELD="$1" python3 -c '
import json, os, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
me = [w for w in rows if w.get("friendly_name") == os.environ.get("FRIENDLY_NAME")]
if not me:
    raise SystemExit(1)
v = me[0].get(os.environ["FIELD"])
print("" if v is None else v)
'
}

# How many segments the ENGINE still has. The worker's own status is useless here: `POST
# /drain` sets it to `draining` immediately, which is precisely the lie this has to see
# through.
engine_in_flight() {
    docker exec "$NAME" curl -s --max-time 10 "http://127.0.0.1:${CONTROL_PORT:-8081}/health" 2>/dev/null \
        | python3 -c '
import json, sys
d = json.load(sys.stdin)
svc = {s.get("name"): s for s in d.get("services", []) if isinstance(s, dict)}
e = svc.get("ltx-engine-api")
if e is None: raise SystemExit(4)
print((e.get("running") or 0) + (e.get("queue_depth") or 0))
' 2>/dev/null
}

if [ "$ACTION" = "pause" ] || [ "$ACTION" = "resume" ]; then
    WID="$(worker_field id)" || {
        echo "!! could not find a worker named '${FRIENDLY_NAME:-}' in $QUEUE_URL/workers."
        echo "!! Is the container running and registered? \`$0 status\` shows what it has."
        exit 1
    }

    if [ "$ACTION" = "resume" ]; then
        curl -sf -X DELETE --max-time 15 -H "X-API-Key: ${QUEUE_API_KEY:-}" \
             "$QUEUE_URL/workers/$WID/drain" >/dev/null || {
            echo "!! the API refused to cancel the drain"; exit 1; }
        echo "resumed -- claiming again; anything queued starts now"
        exit 0
    fi

    curl -sf -X POST --max-time 15 -H "X-API-Key: ${QUEUE_API_KEY:-}" \
         "$QUEUE_URL/workers/$WID/drain" >/dev/null || {
        echo "!! the API refused the drain"; exit 1; }
    echo "draining $FRIENDLY_NAME -- not claiming any more work."

    # THE WAIT. Without it this returns while the render is still going, and the caption
    # routes are already un-gated (see the header): a caption then lands on a card that is
    # ~23 of 24 GB into a render. An unreadable engine is NOT treated as finished.
    if ! _service_absent "$NAME" ltx-engine; then
        echo -n "waiting for the segment in flight to finish"
        while :; do
            n="$(engine_in_flight)" || n=""
            [ -n "$n" ] || { echo; echo "!! cannot read the engine -- NOT safe to caption yet."
                             echo "!! The drain stands; check \`$0 status\` before captioning."; exit 1; }
            [ "$n" = "0" ] && break
            echo -n "."
            sleep 15
        done
        echo
    fi
    echo "parked -- the daemon has freed the card. Caption away; queued jobs wait."
    echo "when you are done:  $0 resume"
    exit 0
fi

# What worker.env would give the container on the next recreate -- MODE wins over SERVICES,
# resolved the same way run-worker.sh resolves it.
env_services() {
    case "${MODE:-}" in
        ltx-engine|render) echo "${SERVICES_RENDER:-ltx-engine}" ;;
        caption|image-caption|image-description) echo "$SERVICES_CAPTION" ;;
        *) echo "${SERVICES:-}" ;;
    esac
}

RUNNING="$(running_services)"

if [ "$ACTION" = "status" ]; then
    if [ -z "$RUNNING" ]; then
        echo "container:  $NAME is not running"
    else
        echo "container:  $NAME -- mode $(mode_of "$RUNNING")"
        echo "  SERVICES: $RUNNING"
    fi
    echo "worker.env: MODE=${MODE:-<unset>} -> mode $(mode_of "$(env_services)")"
    echo "  SERVICES: $(env_services)"
    echo "  render:   $(render_line)"
    echo "  caption:  $SERVICES_CAPTION"
    wstatus="$(worker_field status 2>/dev/null || echo unknown)"
    echo "worker:     ${wstatus:-unknown}$([ "$wstatus" = draining ] && echo "  (paused -- $0 resume)")"
    reason="$(worker_idle_reason "$NAME")"
    echo "idle gate:  ${reason:-idle -- safe to switch}"
    exit 0
fi

if [ "$ACTION" = "render" ]; then
    WANT="$(render_line)"
    [ -n "$WANT" ] || {
        echo "!! nothing to restore: SERVICES_RENDER is unset and SERVICES is already the"
        echo "!! caption line. Put the full service line in $ENV_FILE as SERVICES_RENDER="
        exit 1
    }
else
    WANT="$SERVICES_CAPTION"
fi

if [ "$RUNNING" = "$WANT" ] && [ "$(env_services)" = "$WANT" ]; then
    echo "already in $ACTION mode (SERVICES=$WANT) -- nothing to do"
    exit 0
fi

# THE GATE. Before anything is written, so a refusal leaves the file exactly as it was.
if [ "$FORCE" != "1" ]; then
    reason="$(worker_idle_reason "$NAME")"
    if [ -n "$reason" ]; then
        echo "!! refusing to switch: $reason."
        echo "!! Switching recreates the container with \`docker rm -f\`, which destroys an"
        echo "!! in-flight segment or training run."
        echo "!!"
        echo "!! To caption WITHOUT interrupting it, drain instead -- the in-flight segment"
        echo "!! finishes and the queue pauses. See the header of this script."
        echo "!!"
        echo "!! If you mean it anyway: $0 $ACTION --force"
        exit 1
    fi
fi

# Record the render line BEFORE leaving render mode -- afterwards nothing else holds it.
if [ "$ACTION" = "caption" ] && [ -z "${SERVICES_RENDER:-}" ] && [ -n "$(render_line)" ]; then
    set_env_line SERVICES_RENDER "$(render_line)"
    echo "recorded SERVICES_RENDER=$(render_line) in $ENV_FILE"
fi

# MODE is the lever run-worker.sh reads; SERVICES stays as the box's own fine-grained line.
# Writing MODE rather than expanding it means worker.env reads as the choice that was made,
# and `MODE=caption ./run-worker.sh` by hand does exactly the same thing.
MODE_VALUE="$([ "$ACTION" = "caption" ] && echo caption || echo ltx-engine)"
set_env_line MODE "$MODE_VALUE"
echo "worker.env: MODE=$MODE_VALUE (SERVICES=$WANT) -- recreating $NAME"
"$HERE/run-worker.sh"
