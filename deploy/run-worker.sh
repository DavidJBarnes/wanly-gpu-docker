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

# WHICH SERVICES, AND WHAT EACH ONE NEEDS (wanly-gpu-docker#83). Mounts and ports are
# per-service: a pod-shaped render box has no ollama store and no run directories, and
# requiring them everywhere would mean inventing empty directories to satisfy a check.
# Publishing every port unconditionally is how a box that does not run image-description
# fails with `port is already allocated` on 11434.
SERVICES="${SERVICES:-ltx-engine}"

# MODE is passed STRAIGHT THROUGH to the container, which is what decides (registry.py
# select_mode). SERVICES stays this box's capability line -- everything it is equipped to
# do -- and MODE says which of that runs right now.
#
#     docker run -e MODE=caption ...     is the whole switch
#     MODE=caption ./run-worker.sh       same thing through here
#
# Nothing is rewritten host-side: there is no full list to remember and put back, which is
# the failure mode of a switch that edits SERVICES in place.
#
# Validated HERE as well as in the container because this script's rule is to refuse before
# `docker rm -f`, not after: a typo must not cost the running worker.
case "${MODE:-}" in
    ""|render|ltx-engine|engine|train|training|trainer|lora-trainer|motion|motion-caption|caption|image-caption|image-description|edit|image-edit|full-edit) ;;
    *) echo "!! MODE=$MODE is not a mode. Known: render, train, motion, edit (old names ltx-engine, caption still work)"; exit 1 ;;
esac

case ",$SERVICES," in *,lora-trainer,*)       WANT_TRAINER=1 ;; *) WANT_TRAINER=0 ;; esac
case ",$SERVICES," in *,image-description,*)  WANT_OLLAMA=1 ;;  *) WANT_OLLAMA=0 ;;  esac
case ",$SERVICES," in *,face-crop,*)          WANT_FACE_CROP=1 ;; *) WANT_FACE_CROP=0 ;; esac
case ",$SERVICES," in *,face-edit,*)          WANT_FACE_EDIT=1 ;; *) WANT_FACE_EDIT=0 ;; esac
case ",$SERVICES," in *,ltx-engine,*)         WANT_ENGINE=1 ;;    *) WANT_ENGINE=0 ;;    esac
case ",$SERVICES," in *,image-edit,*)         WANT_IMAGE_EDIT=1 ;; *) WANT_IMAGE_EDIT=0 ;; esac
case ",$SERVICES," in *,scene-caption,*)      WANT_SCENE=1 ;;     *) WANT_SCENE=0 ;;     esac

# THE RENDER STACK'S MOUNTS AND PORTS ARE THE RENDER STACK'S (console#547). They used to be
# unconditional, which made a box that renders nothing -- the 2070, running face-edit beside
# Automatic1111 -- invent an empty jobs dir and models tree to get past this check, and publish
# ComfyUI and engine ports for processes that never start. Required exactly when something
# that reads them runs: ltx-engine (jobs, models, loras) or lora-trainer (the base checkpoint
# under the models tree). A box with either is unchanged, flag for flag.
ENGINE_ARGS=()
if [ "$WANT_ENGINE" = "1" ] || [ "$WANT_TRAINER" = "1" ]; then
    for d in "${JOBS_DIR:-}" "${MODELS_DIR:-}"; do
        [ -n "$d" ] && [ -d "$d" ] || { echo "!! '$d' does not exist — refusing to create a worker with a broken mount (set JOBS_DIR and MODELS_DIR in $ENV_FILE)"; exit 1; }
    done
    ENGINE_ARGS+=(-p "${COMFY_HOST_PORT:-8191}:8188")
    ENGINE_ARGS+=(-p "${ENGINE_HOST_PORT:-8190}:8190")
    ENGINE_ARGS+=(-v "$JOBS_DIR:/jobs")
    ENGINE_ARGS+=(-v "$MODELS_DIR:/workspace/models:ro")
    ENGINE_ARGS+=(-v "$MODELS_DIR/loras:/workspace/models/loras")
fi

MOUNTS=()
PORTS=()
TRAINER_ENV_ARGS=()
if [ "$WANT_TRAINER" = "1" ] || [ "$WANT_OLLAMA" = "1" ] || [ "$WANT_FACE_EDIT" = "1" ] \
   || [ "$WANT_SCENE" = "1" ]; then
    # All three live in the :full layer. The lean tag fails in the trainer's preflight after a
    # pull, which is a slow way to learn a one-line mistake -- refuse by name, and BEFORE
    # anything is removed, so a wrong IMAGE never costs the running container.
    case "$IMAGE" in
        *:latest|*:next|*:"${IMAGE##*:}" ) case "$IMAGE" in *full*) ;; *)
            echo "!! SERVICES includes lora-trainer/image-description/face-edit/scene-caption but IMAGE=$IMAGE is built WITHOUT them."
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
    # The trainer's FALLBACK base checkpoint (#145) -- a CONTAINER path, under
    # /workspace/models. A job's config.base_checkpoint wins over it; this only applies to a
    # job that names none, and replaces the image's default (10Eros). Forwarded only when set,
    # so an unset line leaves the default in the code rather than pinning an empty path.
    if [ -n "${LTX_BASE_CKPT:-}" ]; then
        TRAINER_ENV_ARGS+=(-e "LTX_BASE_CKPT=$LTX_BASE_CKPT")
    fi
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
SCENE_ENV_ARGS=()
if [ "$WANT_SCENE" = "1" ]; then
    # scene-caption (wanly-console#572): JoyCaption on its own ollama, always resident. Same
    # host store as image-description -- mounted once even when both run -- because the model
    # is already there on every box that ever captioned, and a rebuild is 5.8 GB of download.
    : "${OLLAMA_HOST_STORE:?scene-caption is enabled — set OLLAMA_HOST_STORE in $ENV_FILE}"
    [ -d "$OLLAMA_HOST_STORE" ] || {
        echo "!! $OLLAMA_HOST_STORE does not exist — refusing to create a container whose model"
        echo "!! store would be container-local and lost on the next recreate."
        exit 1
    }
    if [ "$WANT_OLLAMA" != "1" ]; then
        MOUNTS+=(-v "$OLLAMA_HOST_STORE:/root/.ollama")
    fi
    PORTS+=(-p "${SCENE_CAPTION_PORT:-11436}:11436")
    if ss -tlnp 2>/dev/null | grep -q ":${SCENE_CAPTION_PORT:-11436} " \
       && ! docker port "$NAME" 2>/dev/null | grep -q ":${SCENE_CAPTION_PORT:-11436}$"; then
        echo "!! something already listens on ${SCENE_CAPTION_PORT:-11436}, and it is not $NAME."
        exit 1
    fi
    # SCENE_CAPTION_SHARED=1 only where the card is shared with image-edit (3090b, interim):
    # each edit then asks JoyCaption to unload first, and hands the card back after.
    for v in SCENE_CAPTION_SHARED SCENE_CAPTION_YIELD_WAIT_S IMAGE_EDIT_SCENE_RESUME_IDLE_S MODEL_LIMIT_RATE; do
        if [ -n "${!v:-}" ]; then SCENE_ENV_ARGS+=(-e "$v=${!v}"); fi
    done
fi
if [ "$WANT_FACE_CROP" = "1" ] || [ "$WANT_IMAGE_EDIT" = "1" ]; then
    # buffalo_l is ~300 MB and insightface fetches it on first use. On a mount it survives a
    # recreate; in the container it is re-downloaded every time. image-edit scores every
    # result with AuraFace from the same store (console#548).
    INSIGHTFACE_HOST_DIR="${INSIGHTFACE_HOST_DIR:-$HOME/.insightface}"
    mkdir -p "$INSIGHTFACE_HOST_DIR"
    MOUNTS+=(-v "$INSIGHTFACE_HOST_DIR:/root/.insightface")
fi
if [ "$WANT_FACE_CROP" = "1" ]; then
    PORTS+=(-p "${FACE_CROP_PORT:-8084}:8084")
fi
IMAGE_EDIT_ENV_ARGS=()
if [ "$WANT_IMAGE_EDIT" = "1" ]; then
    # Qwen-Image-Edit "full mode" (console#548). It runs only in edit mode (POST /mode), but the
    # mount and the port are the box's, like every other capability on its SERVICES line: the
    # switch is in place, with no recreate to add them later. Read-only for the reason the LTX
    # tree is -- the host is the source of truth for 28 GB of checkpoint.
    : "${IMAGE_EDIT_MODELS_HOST_DIR:?image-edit is enabled — set IMAGE_EDIT_MODELS_HOST_DIR in $ENV_FILE (the 3090: /home/david/models/qwen)}"
    [ -d "$IMAGE_EDIT_MODELS_HOST_DIR" ] || {
        echo "!! $IMAGE_EDIT_MODELS_HOST_DIR does not exist — refusing to create a worker with a broken mount"
        exit 1
    }
    MOUNTS+=(-v "$IMAGE_EDIT_MODELS_HOST_DIR:/workspace/qwen:ro")
    PORTS+=(-p "${IMAGE_EDIT_PORT:-8086}:8086")
    # A STANDING image-edit box (console#570) shares its card with the host's Automatic1111:
    # the alias makes IMAGE_EDIT_A1111_URL=http://host.docker.internal:7860 resolve, so an edit
    # can wait out a generation and ask an idle A1111 to unload (services/image_edit/share.py).
    if [ -n "${IMAGE_EDIT_A1111_URL:-}" ]; then
        IMAGE_EDIT_ENV_ARGS+=(--add-host "host.docker.internal:host-gateway")
    fi
    for v in IMAGE_EDIT_STEPS IMAGE_EDIT_CFG IMAGE_EDIT_MAX_MP EDIT_IDLE_RETURN_S IMAGE_EDIT_A1111_URL \
             IMAGE_EDIT_A1111_WAIT_S IMAGE_EDIT_MIN_FREE_MIB IMAGE_EDIT_UNLOAD_IDLE_S; do
        if [ -n "${!v:-}" ]; then IMAGE_EDIT_ENV_ARGS+=(-e "$v=${!v}"); fi
    done
fi
FACE_EDIT_ENV_ARGS=()
if [ "$WANT_FACE_EDIT" = "1" ]; then
    # LivePortrait (console#547). Models are baked, so no mount. wanly-api calls it across the
    # network (face_edit_url), like face-crop.
    PORTS+=(-p "${FACE_EDIT_PORT:-8085}:8085")
    # A1111 on the same card is on the HOST, not in this container: the alias is what makes
    # FACE_EDIT_A1111_URL=http://host.docker.internal:7860 resolve, so face-edit can read its
    # progress (never borrow the GPU mid-generation) and ask it to unload an idle checkpoint.
    FACE_EDIT_ENV_ARGS+=(--add-host "host.docker.internal:host-gateway")
    # Forwarded only when set, so an unset line keeps the image's default instead of pinning
    # an empty value.
    for v in FACE_EDIT_A1111_URL FACE_EDIT_DEVICE FACE_EDIT_CPU_FALLBACK FACE_EDIT_MIN_FREE_MIB \
             FACE_EDIT_GPU_IDLE_S FACE_EDIT_FACE_PAD; do
        if [ -n "${!v:-}" ]; then FACE_EDIT_ENV_ARGS+=(-e "$v=${!v}"); fi
    done
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

# THE DEV MOUNT (wanly-gpu-docker#117): iterate on engine/supervisor code in a checkout on
# this box without a build or a pull. DEV_CODE_DIR is bind-mounted at /opt/dev-code and
# fetch_engine.sh swaps it in at boot INSTEAD of fetching main — edit on the host,
# `docker restart`, done, in seconds.
#
# TWO FLAGS ON PURPOSE. DEV_CODE_DIR alone does not arm anything: a stale env line must not
# silently redirect a real worker onto a half-edit. DEV_ALLOW_QUEUE=1 is the explicit
# statement that this box, running this code, may claim real segments. The container shouts
# DEV MOUNT in the boot log and /health reports `code: dev-mount ...` for as long as it
# lasts, because a mount that quietly outlives its session is #72 with extra steps.
DEV_MOUNT_ARGS=()
DEV_ENV_ARGS=()
if [ -n "${DEV_CODE_DIR:-}" ]; then
    if [ "${DEV_ALLOW_QUEUE:-0}" != "1" ]; then
        echo "!! DEV_CODE_DIR is set but DEV_ALLOW_QUEUE is not 1."
        echo "!! Half-edited code against the real queue is how #72 happened. If you mean it:"
        echo "!!   DEV_ALLOW_QUEUE=1   in $ENV_FILE"
        echo "!! If you don't: remove the DEV_CODE_DIR line."
        exit 1
    fi
    [ -d "$DEV_CODE_DIR" ] || { echo "!! DEV_CODE_DIR=$DEV_CODE_DIR does not exist"; exit 1; }
    DEV_CODE_DIR="$(cd "$DEV_CODE_DIR" && pwd)"   # docker wants an absolute path
    [ -f "$DEV_CODE_DIR/engine/app.py" ] && [ -f "$DEV_CODE_DIR/wanly_worker/control.py" ] || {
        echo "!! DEV_CODE_DIR=$DEV_CODE_DIR is not a wanly-gpu-docker checkout (no engine/app.py";
        echo "!! or wanly_worker/control.py) — refusing to start on it."; exit 1; }
    DEV_MOUNT_ARGS+=(-v "$DEV_CODE_DIR:/opt/dev-code:ro")
    DEV_ENV_ARGS+=(-e DEV_CODE=1)
    echo "!! DEV MOUNT: $DEV_CODE_DIR @ $(git -C "$DEV_CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo uncommitted) — NOT DEPLOYED CODE"
fi

# The supervisor answers /health on CONTROL_PORT (wanly-gpu-docker#83); the update timer
# reads it. Refuse before removing anything if something else already listens there.
CONTROL_PORT="${CONTROL_PORT:-8081}"
if ss -tlnp 2>/dev/null | grep -q ":${CONTROL_PORT} " \
   && ! docker port "$NAME" 2>/dev/null | grep -q ":${CONTROL_PORT}$"; then
    echo "!! something other than $NAME already listens on :${CONTROL_PORT} — set CONTROL_PORT"
    exit 1
fi

echo "recreating $NAME from $IMAGE (SERVICES=$SERVICES${MODE:+, MODE=$MODE})"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# COMFYUI_PATH is EMPTY on purpose — see the note in start.sh. With a path set the daemon
# takes ownership of ComfyUI's custom nodes, which breaks an LTX worker.
# CDI, NOT `--gpus all` (#95). With `--gpus all` the toolkit hook grants the GPU device nodes
# outside the container's OCI spec, so the next `systemctl daemon-reload` re-applies the
# spec's device rules WITHOUT them: every CUDA allocation then fails with "0 bytes allocated,
# 23 GiB free" until the container is recreated. Seen 2026-09-10, six minutes after a reboot.
# CDI (`/var/run/cdi/nvidia.yaml`, kept fresh by nvidia-cdi-refresh) puts the devices in the
# spec itself, where a reload preserves them. Needs Docker >= 28 and the toolkit's CDI spec.
#
# IMAGE_DESCRIPTION_MODEL DEFAULTS TO joycaption HERE even though the image's own default is a
# Qwen tag (wanly-gpu-docker#129): joycaption:beta-one runs on every GPU we own, whereas a
# Qwen-class captioner needs a 24 GB card — and a default that triggered a silent 6-21 GB
# boot-time pull on a box that never chose it is the class of drift this repo keeps ticketing.
# The 3090 gets joycaption from this default today (its worker.env does not set the var); a
# box that wants the Qwen captioner sets IMAGE_DESCRIPTION_MODEL explicitly.
# MEMORY CAP (wanly-gpu-docker#166). 3090a hung for ~5 h on 2026-10-02: from 05:05 the host logged
# "Under memory pressure" every minute, systemd-oomd only killed desktop processes (the worker
# lives outside the user slice), and 38 GB of swap let the box thrash until sshd, journald and
# the container's own start.sh were blocked for minutes -- reachable by ping, dead otherwise,
# until a power cycle. With a cap and NO extra swap, a runaway inside the container is OOM-killed
# INSIDE the container (one render fails, --restart brings the worker back) and the host stays up.
# Leave ~6-8 GB for the host. Unset = no cap (the old behaviour).
#   WORKER_MEMORY_LIMIT=54g      # 3090a (61 GB RAM); 3090b at 30 GB: ~24g
MEM_ARGS=()
if [ -n "${WORKER_MEMORY_LIMIT:-}" ]; then
    MEM_ARGS=(--memory "$WORKER_MEMORY_LIMIT" --memory-swap "$WORKER_MEMORY_LIMIT")
fi
docker run -d \
    --name "$NAME" \
    "${MEM_ARGS[@]}" \
    --restart unless-stopped \
    --device nvidia.com/gpu=all \
    --shm-size "$SHM_SIZE" \
    "${ENGINE_ARGS[@]}" \
    -p "${CONTROL_PORT}:8081" \
    "${PORTS[@]}" \
    "${MOUNTS[@]}" \
    "${FACE_EDIT_ENV_ARGS[@]}" \
    "${IMAGE_EDIT_ENV_ARGS[@]}" \
    "${SCENE_ENV_ARGS[@]}" \
    "${DEV_MOUNT_ARGS[@]}" \
    -e "FRIENDLY_NAME=$FRIENDLY_NAME" \
    -e "SERVICES=$SERVICES" \
    -e "MODE=${MODE:-}" \
    -e "MODE_SWITCH_VRAM_MAX_MIB=${MODE_SWITCH_VRAM_MAX_MIB:-}" \
    -e "IMAGE_DESCRIPTION_MODEL=${IMAGE_DESCRIPTION_MODEL:-joycaption:beta-one}" \
    -e "ENGINE=ltx" \
    -e "QUEUE_URL=$QUEUE_URL" \
    -e "QUEUE_API_KEY=$QUEUE_API_KEY" \
    -e "LTX_ENGINE_URL=http://127.0.0.1:8190" \
    -e "COMFYUI_URL=http://127.0.0.1:8188" \
    -e "COMFYUI_PATH=" \
    -e "LORA_CACHE_DIR=/workspace/models/loras" \
    "${DEV_ENV_ARGS[@]}" \
    "${TRAINER_ENV_ARGS[@]}" \
    "$IMAGE"

echo "started: $(docker inspect -f '{{.Id}}' "$NAME" | cut -c1-12) on $(docker inspect -f '{{.Image}}' "$NAME" | cut -c8-19)"
echo "follow the boot with: docker logs -f $NAME"

# PRUNE WHAT THE CONVERGE LEFT BEHIND (wanly-gpu-docker#111). Every image build adds a
# fresh ~25 GB image, and nothing ever removed the old one: the box accumulated 222 GB of
# images and filled its root disk to 99%, where the trainer's boot disk gate refused to
# start and took the WHOLE worker (render included) down for hours. The container just
# started pins its image; everything unreferenced is dead weight.
#
# Deliberately after the run, and non-fatal: a prune failure must never turn a successful
# recreate into a failed one. Rollback to an older image stays possible -- the tag is
# re-pulled when IMAGE names it.
#
# PRUNE_IMAGES=0 turns it off (console#547). `prune -af` removes EVERY image no container
# uses, not just this repo's -- right on the 3090, where Docker holds only the worker, and
# destructive on a box whose Docker also stores other projects' images with no container
# (the 2070: verbatim-worker, open-webui, ollama, postgres, ntfy -- ~40 GB of it).
if [ "${PRUNE_IMAGES:-1}" = "1" ]; then
    docker image prune -af >/dev/null 2>&1 || true
else
    echo "PRUNE_IMAGES=0 — leaving unreferenced images alone"
fi
