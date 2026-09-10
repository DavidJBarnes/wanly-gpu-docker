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

# WHICH SERVICES, AND WHAT EACH ONE NEEDS (wanly-gpu-docker#83). Mounts and ports are
# per-service: a pod-shaped render box has no ollama store and no run directories, and
# requiring them everywhere would mean inventing empty directories to satisfy a check.
# Publishing every port unconditionally is how a box that does not run image-description
# fails with `port is already allocated` on 11434.
SERVICES="${SERVICES:-ltx-engine}"
case ",$SERVICES," in *,lora-trainer,*)       WANT_TRAINER=1 ;; *) WANT_TRAINER=0 ;; esac
case ",$SERVICES," in *,image-description,*)  WANT_OLLAMA=1 ;;  *) WANT_OLLAMA=0 ;;  esac
case ",$SERVICES," in *,face-crop,*)          WANT_FACE_CROP=1 ;; *) WANT_FACE_CROP=0 ;; esac

MOUNTS=()
PORTS=()
if [ "$WANT_TRAINER" = "1" ] || [ "$WANT_OLLAMA" = "1" ]; then
    # Both live in the :full layer. The lean tag fails in the trainer's preflight after a
    # pull, which is a slow way to learn a one-line mistake -- refuse by name, and BEFORE
    # anything is removed, so a wrong IMAGE never costs the running container.
    case "$IMAGE" in
        *:latest|*:next|*:"${IMAGE##*:}" ) case "$IMAGE" in *full*) ;; *)
            echo "!! SERVICES includes lora-trainer/image-description but IMAGE=$IMAGE is built WITHOUT them."
            echo "!! Use IMAGE=davidjbarnes/wanly-gpu-docker:full (or :next-full) in $ENV_FILE"
            exit 1 ;; esac ;;
    esac
fi
if [ "$WANT_TRAINER" = "1" ]; then
    : "${LORA_RUNS_DIR:?lora-trainer is enabled — set LORA_RUNS_DIR in $ENV_FILE}"
    [ -d "$LORA_RUNS_DIR" ] || { echo "!! $LORA_RUNS_DIR does not exist — the trainer cannot work without it"; exit 1; }
    # Run directories writable, because that is where the checkpoints land. The models tree
    # is already mounted read-only below and the trainer reads the base checkpoint from it.
    MOUNTS+=(-v "$LORA_RUNS_DIR:/loras")
fi
if [ "$WANT_OLLAMA" = "1" ]; then
    : "${OLLAMA_HOST_STORE:?image-description is enabled — set OLLAMA_HOST_STORE in $ENV_FILE}"
    [ -d "$OLLAMA_HOST_STORE" ] || {
        echo "!! $OLLAMA_HOST_STORE does not exist — refusing to create a container whose model"
        echo "!! store would be container-local and lost on the next recreate."
        exit 1
    }
    MOUNTS+=(-v "$OLLAMA_HOST_STORE:/root/.ollama")
    PORTS+=(-p "${IMAGE_DESCRIPTION_PORT:-11434}:11434")
    # The host's ollama holds 11434 on a box that used to run the captioner natively. Two
    # things cannot bind one port, and the failure is a container stuck in `Created` with a
    # message that reads like a Docker problem.
    if ss -tlnp 2>/dev/null | grep -q ":${IMAGE_DESCRIPTION_PORT:-11434} " \
       && ! docker port "$NAME" 2>/dev/null | grep -q ":${IMAGE_DESCRIPTION_PORT:-11434}$"; then
        echo "!! something already listens on ${IMAGE_DESCRIPTION_PORT:-11434}, and it is not $NAME."
        echo "!! If it is the host ollama service: sudo systemctl disable --now ollama"
        exit 1
    fi
fi
if [ "$WANT_FACE_CROP" = "1" ]; then
    # buffalo_l is ~300 MB and insightface fetches it on first use. On a mount it survives a
    # recreate; in the container it is re-downloaded every time.
    INSIGHTFACE_HOST_DIR="${INSIGHTFACE_HOST_DIR:-$HOME/.insightface}"
    mkdir -p "$INSIGHTFACE_HOST_DIR"
    MOUNTS+=(-v "$INSIGHTFACE_HOST_DIR:/root/.insightface")
    PORTS+=(-p "${FACE_CROP_PORT:-8084}:8084")
fi
# SHM_SIZE, and it is not optional for the trainer. Docker gives a container 64 MB of
# /dev/shm, and PyTorch's DataLoader workers pass tensors through shared memory -- so a
# training run dies partway through stage 3 with
#
#   RuntimeError: unable to write to file </torch_311_...>: No space left on device (28)
#   DataLoader worker ... killed by signal: Bus error
#
# which reads like a full disk. The first real run lost forty minutes to it.
SHM_SIZE="${SHM_SIZE:-$([ "$WANT_TRAINER" = "1" ] && echo 8g || echo 64m)}"

# The supervisor answers /health on CONTROL_PORT (wanly-gpu-docker#83); the update timer
# reads it. Refuse before removing anything if something else already listens there.
CONTROL_PORT="${CONTROL_PORT:-8081}"
if ss -tlnp 2>/dev/null | grep -q ":${CONTROL_PORT} " \
   && ! docker port "$NAME" 2>/dev/null | grep -q ":${CONTROL_PORT}$"; then
    echo "!! something other than $NAME already listens on :${CONTROL_PORT} — set CONTROL_PORT"
    exit 1
fi

echo "recreating $NAME from $IMAGE (SERVICES=$SERVICES)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# COMFYUI_PATH is EMPTY on purpose — see the note in start.sh. With a path set the daemon
# takes ownership of ComfyUI's custom nodes, which breaks an LTX worker.
# CDI, NOT `--gpus all` (#95). With `--gpus all` the toolkit hook grants the GPU device nodes
# outside the container's OCI spec, so the next `systemctl daemon-reload` re-applies the
# spec's device rules WITHOUT them: every CUDA allocation then fails with "0 bytes allocated,
# 23 GiB free" until the container is recreated. Seen 2026-09-10, six minutes after a reboot.
# CDI (`/var/run/cdi/nvidia.yaml`, kept fresh by nvidia-cdi-refresh) puts the devices in the
# spec itself, where a reload preserves them. Needs Docker >= 28 and the toolkit's CDI spec.
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --device nvidia.com/gpu=all \
    --shm-size "$SHM_SIZE" \
    -p "${COMFY_HOST_PORT:-8191}:8188" \
    -p "${ENGINE_HOST_PORT:-8190}:8190" \
    -p "${CONTROL_PORT}:8081" \
    "${PORTS[@]}" \
    -v "$JOBS_DIR:/jobs" \
    -v "$RECIPES_DIR:/opt/engine/recipes:ro" \
    -v "$MODELS_DIR:/workspace/models:ro" \
    -v "$MODELS_DIR/loras:/workspace/models/loras" \
    "${MOUNTS[@]}" \
    -e "FRIENDLY_NAME=$FRIENDLY_NAME" \
    -e "SERVICES=$SERVICES" \
    -e "IMAGE_DESCRIPTION_MODEL=${IMAGE_DESCRIPTION_MODEL:-joycaption:beta-one}" \
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
