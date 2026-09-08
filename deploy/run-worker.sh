#!/usr/bin/env bash
# Create (or recreate) the long-lived worker container on a box we own — today the 3090.
#
# WHY THIS EXISTS AS A FILE
#     A RunPod pod is created from :latest every time, so it is always current on both update
#     channels: the image, and the daemon that start.sh clones from main at boot. A long-lived
#     container is only current on the second one -- `docker restart` reuses the image by
#     design. The 3090 therefore drifted to a 37-hour-old image and a 14-hour-old daemon while
#     a pod ran tonight's code, and the two produced different results from the same queue
#     (wanly-gpu-docker#72).
#
#     Nobody recreated it because the `docker run` existed only in one shell's history.
#     Reproducing it from memory is exactly the kind of thing that gets a mount or a port
#     wrong, so it lives here instead.
#
# SAFE TO RE-RUN. It stops and removes the existing container first, so this is also the
# rollback path: point IMAGE at an older tag and run it again.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${WORKER_ENV:-$HERE/worker.env}"
[ -f "$ENV_FILE" ] || { echo "!! no $ENV_FILE — copy worker.env.example and fill it in"; exit 1; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

IMAGE="${IMAGE:-davidjbarnes/wanly-gpu-docker:latest}"
NAME="${NAME:-wanly-gpu-docker}"

: "${QUEUE_API_KEY:?set QUEUE_API_KEY in $ENV_FILE}"
: "${FRIENDLY_NAME:?set FRIENDLY_NAME in $ENV_FILE}"

for d in "$JOBS_DIR" "$MODELS_DIR"; do
    [ -d "$d" ] || { echo "!! $d does not exist — refusing to create a worker with a broken mount"; exit 1; }
done

# The supervisor answers /health on CONTROL_PORT (wanly-gpu-docker#83); the update timer
# reads it. Refuse before removing anything if something else already listens there.
CONTROL_PORT="${CONTROL_PORT:-8081}"
if ss -tlnp 2>/dev/null | grep -q ":${CONTROL_PORT} " \
   && ! docker port "$NAME" 2>/dev/null | grep -q ":${CONTROL_PORT}$"; then
    echo "!! something other than $NAME already listens on :${CONTROL_PORT} — set CONTROL_PORT"
    exit 1
fi

echo "recreating $NAME from $IMAGE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# COMFYUI_PATH is EMPTY on purpose — see the note in start.sh. With a path set the daemon
# takes ownership of ComfyUI's custom nodes, which breaks an LTX worker.
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --gpus all \
    -p "${COMFY_HOST_PORT:-8191}:8188" \
    -p "${ENGINE_HOST_PORT:-8190}:8190" \
    -p "${CONTROL_PORT}:8081" \
    -v "$JOBS_DIR:/jobs" \
    -v "$RECIPES_DIR:/opt/engine/recipes:ro" \
    -v "$MODELS_DIR:/workspace/models:ro" \
    -v "$MODELS_DIR/loras:/workspace/models/loras" \
    -e "FRIENDLY_NAME=$FRIENDLY_NAME" \
    -e "SERVICES=${SERVICES:-ltx-engine}" \
    -e "ENGINE=ltx" \
    -e "QUEUE_URL=$QUEUE_URL" \
    -e "QUEUE_API_KEY=$QUEUE_API_KEY" \
    -e "LTX_ENGINE_URL=http://127.0.0.1:8190" \
    -e "COMFYUI_URL=http://127.0.0.1:8188" \
    -e "COMFYUI_PATH=" \
    -e "LORA_CACHE_DIR=/workspace/models/loras" \
    "$IMAGE"

echo "started: $(docker inspect -f '{{.Id}}' "$NAME" | cut -c1-12) on $(docker inspect -f '{{.Image}}' "$NAME" | cut -c8-19)"
echo "follow the boot with: docker logs -f $NAME"
