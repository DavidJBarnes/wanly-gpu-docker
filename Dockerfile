# The Wanly GPU worker image — LTX 2.3.
#
# This image WAS the WAN 2.2 worker: ComfyUI plus PainterLongVideo, ReActor,
# FaceFusion and Frame-Interpolation, driven directly by the daemon's
# workflow_builder. WAN 2.2 has been retired in favour of LTX 2.3 and that whole
# stack is gone — with it, three of the nastiest traps this repo carried (the
# FaceFusion install.py clobbering the pinned onnxruntime, ReActor's unpinned
# transformers pulling a torch-incompatible release, and the CFG-aware VRAM
# patch to comfy/model_base.py).
#
# Three processes now, not two:
#
#   ComfyUI      :8188  background   the sampler
#   ltx-engine   :8190  background   graph assembly + recipe resolution
#   daemon              foreground   claims from wanly-api, drives the engine
#
# ltx-engine sits between the daemon and ComfyUI on purpose. It owns the graph:
# it uploads keyframes, resolves the recipe, patches the workflow and submits.
# The daemon does NOT build LTX graphs — every structural bug on that project
# came from rewriting a downloaded graph's topology in a caller to cover a job
# shape it was not built for, and none came from the reference workflows.
#
# The daemon is still cloned at boot rather than baked (see start.sh). The image
# is the environment; the daemon is the code.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates \
        openssh-server \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# torch first and alone so a resolver backtrack cannot quietly pick another
# build. This is the pair proven on the 3090 for LTX 2.3.
#
# No torchaudio: there is no 2.12.x release (published stops at 2.11.0), and
# pinning it would either fail the build or drag torch backwards to match. If a
# node turns out to need it, add it explicitly against whatever torch resolves
# to rather than letting the resolver choose for both.
# CUDA 12.8 wheels, PINNED TO THE INDEX. Plain `pip install torch==2.12.1` resolves to
# 2.12.1+cu130, which requires an NVIDIA driver >= 580. The 3090 has 580.95 and worked; a
# RunPod 4090 had 570.195 and ComfyUI died on import with "The NVIDIA driver on your system
# is too old (found version 12080)", which failed the boot and put the pod in a restart loop.
#
# RunPod host drivers vary and cannot be chosen, so the image must run on the older one.
# cu128 runs on both.
# 2.11.0, not 2.12.1: the cu128 index tops out at 2.11.0+cu128 (2.12 ships cu130 only).
# Checked against the index rather than assumed --
# https://download.pytorch.org/whl/cu128/torch/ lists 2.10.0 and 2.11.0; torchvision 0.26.0
# is the matching pair. A newer torch is worth less than an image that boots on the drivers
# RunPod actually gives us.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" "torchvision==0.26.0"

ARG COMFYUI_COMMIT=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
 && cd /app/ComfyUI \
 && if [ "$COMFYUI_COMMIT" != "master" ]; then git checkout ${COMFYUI_COMMIT}; fi \
 && pip install --no-cache-dir -r requirements.txt

WORKDIR /app/ComfyUI/custom_nodes

# The five packs the LTX-2.3 workflows actually reference. Pinning to a commit
# would be better, but these move fast against ComfyUI core and a stale pin
# breaks node loading more often than a fresh clone does. Revisit if it bites.
#
# NOTE none of these is a faceswap or interpolation pack. If a workflow starts
# referencing a node class that is not here, add it AND add the class to the
# daemon's node check — a missing node fails at submit, which is ten minutes
# into a claimed segment.
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
 && git clone https://github.com/kijai/ComfyUI-KJNodes.git \
 && git clone https://github.com/rgthree/rgthree-comfy.git \
 && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
 && git clone https://github.com/city96/ComfyUI-GGUF.git

# Each pack's own deps, in one layer so a single failure is visible rather than
# silently leaving a half-installed node tree.
RUN for d in ComfyUI-LTXVideo ComfyUI-KJNodes rgthree-comfy \
             ComfyUI-VideoHelperSuite ComfyUI-GGUF; do \
        if [ -f "$d/requirements.txt" ]; then \
            echo "=== deps for $d ===" && pip install --no-cache-dir -r "$d/requirements.txt" || exit 1; \
        fi; \
    done

WORKDIR /app

# ltx-engine's own deps, then the daemon's.
COPY engine/requirements.txt /app/engine-requirements.txt
RUN pip install --no-cache-dir -r /app/engine-requirements.txt

# start.sh compares this against the daemon's own requirements.txt at boot and
# warns loudly when they drift. This only has to cover what the image installs
# ahead of time; anything added since is installed at boot.
#
# insightface ships no wheel and compiles C extensions (face3d/mesh/cython), so this needs a
# toolchain — dropping build-essential when this image was rewritten for LTX is what failed
# the first CI build with "x86_64-linux-gnu-gcc: No such file or directory".
#
# Installed and purged in ONE layer so the runtime image does not carry a compiler. Separate
# RUN steps would leave it in the layer below regardless of a later apt-get remove.
#
# insightface is not optional here: identity scoring runs on every finished render, LTX
# included, and it is what produces the identity chips the console shows.
COPY daemon-requirements.txt /app/daemon-requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev \
 && pip install --no-cache-dir -r /app/daemon-requirements.txt \
 && apt-get purge -y --auto-remove build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Models are bind-mounted, never baked: the LTX-2.3 set alone is ~126 GB.
ENV COMFY_PORT=8188 \
    API_PORT=8190 \
    COMFY_URL=http://127.0.0.1:8188 \
    JOBS_DIR=/jobs \
    LTX_WORKFLOW=/opt/engine/workflows/ltx23_base.api.json \
    LTX_TRANSFORMER=/workspace/models/ltx-2.3/diffusion_models/ltx-2.3-22b-dev.safetensors \
    LORA_DIR=/workspace/models/loras \
    MODELS_DIR=/workspace/models \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ENGINE=ltx

COPY extra_model_paths.yaml /opt/extra_model_paths.yaml
COPY engine/ /opt/engine/
COPY download_models.sh /app/download_models.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh /app/download_models.sh && mkdir -p /jobs /workspace/logs

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

EXPOSE 8188 8190 22
CMD ["/app/start.sh"]
