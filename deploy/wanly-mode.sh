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
# THE LIGHTER MIDDLE PATH, and prefer it when a segment is ten minutes from done: drain
# instead of switching. It costs no recreate and no model reload.
#
#     curl -XPOST  -H "X-API-Key: $QUEUE_API_KEY" "$QUEUE_URL/workers/<id>/drain"
#     docker exec wanly-gpu-docker curl -s -XPOST http://127.0.0.1:8188/free \
#            -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
#     ...caption...
#     curl -XDELETE -H "X-API-Key: $QUEUE_API_KEY" "$QUEUE_URL/workers/<id>/drain"
#
# The queue pauses, the IN-FLIGHT SEGMENT FINISHES, ComfyUI releases its cache, and the
# captioner has the card. A switch is for when you want the box captioning for a while;
# a drain is for when you want it captioning now.
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
usage: wanly-mode.sh caption|render|status [--force]

  status    what the container is running now, and whether it could be switched
  caption   image-description + face-crop only; queued jobs wait, the GPU is the
            captioner's alone
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
        caption|render|status) ACTION="$arg" ;;
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

RUNNING="$(running_services)"

if [ "$ACTION" = "status" ]; then
    if [ -z "$RUNNING" ]; then
        echo "container:  $NAME is not running"
    else
        echo "container:  $NAME -- mode $(mode_of "$RUNNING")"
        echo "  SERVICES: $RUNNING"
    fi
    echo "worker.env: mode $(mode_of "${SERVICES:-}")"
    echo "  SERVICES: ${SERVICES:-<unset>}"
    echo "  render:   $(render_line)"
    echo "  caption:  $SERVICES_CAPTION"
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

if [ "$RUNNING" = "$WANT" ] && [ "${SERVICES:-}" = "$WANT" ]; then
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

# Record the render line BEFORE leaving render mode -- afterwards SERVICES no longer holds it.
if [ "$ACTION" = "caption" ] && [ -z "${SERVICES_RENDER:-}" ] && [ -n "$(render_line)" ]; then
    set_env_line SERVICES_RENDER "$(render_line)"
    echo "recorded SERVICES_RENDER=$(render_line) in $ENV_FILE"
fi

set_env_line SERVICES "$WANT"
echo "worker.env: SERVICES=$WANT -- recreating $NAME"
"$HERE/run-worker.sh"
