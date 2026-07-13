FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# System deps + Python 3.11 via deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git ffmpeg aria2 wget curl openssh-server \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && pip install --no-cache-dir setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.4 support (install before ComfyUI to avoid CPU-only torch)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ComfyUI pinned to a known-good commit
ARG COMFYUI_COMMIT=4a93a62371b6
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && git checkout ${COMFYUI_COMMIT}
RUN pip install --no-cache-dir -r ComfyUI/requirements.txt

# Apply the wanly CFG-aware VRAM-estimate patch to ComfyUI (dynamic /tmp/wanly_estimate;
# prevents over-reserve slow/OOM on 24GB cards and lets de-distilled/CFG>1 jobs fit).
COPY patch_comfyui_memory.py /app/patch_comfyui_memory.py
RUN python3 /app/patch_comfyui_memory.py /app/ComfyUI/comfy/model_base.py

# Custom node commits — PINNED so a rebuild can never pull a drifted version (the root
# cause of repeated boot breakage: unpinned clones pulled newer deps that broke import).
ARG FRAME_INTERP_COMMIT=26545cc2dd95bc3d27f056016300673bdeee78f5
ARG VHS_COMMIT=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG REACTOR_COMMIT=6ad6b35a4df250d14cb2abf0808c9ffedf59f747
ARG PAINTER_COMMIT=889b4ff67909561e52d6ae023f5b9e8c33fdba94
# WanVideoWrapper (KJ) — provides the VACE nodes (WanVideoVACEEncode/Sampler/...).
# Pinned to 1.3.9 (e926f7a0), the same 1.3.x minor the 3090 runs (its pyproject reports 1.3.3;
# there is no matching git tag, and VACE node inputs are stable across 1.3.x).
ARG WANVIDEO_COMMIT=e926f7a069e3efc25bdcc78e1bf65c39ac1a2cd4

# Custom nodes: Frame Interpolation (RIFE)
# Install cupy-cuda12x directly (pre-built wheel) — the requirements file's cupy-wheel
# is a source meta-package that fails on CI runners without a GPU
RUN pip install --no-cache-dir cupy-cuda12x
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    git -C ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation checkout ${FRAME_INTERP_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt

# Custom nodes: Video Helper Suite
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git -C ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout ${VHS_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# Custom nodes: ReActor (face swap)
# Directory name matches daemon's node_checker.py primary key.
# onnxruntime-gpu / insightface are required at runtime but NOT in ReActor's requirements,
# and unpinned onnxruntime-gpu now resolves to a CUDA-13 wheel (libcudart.so.13) that can't
# load on this CUDA 12.4 image. Pin both, and pin the node itself.
RUN git clone https://github.com/Gourieff/ComfyUI-ReActor.git \
    ComfyUI/custom_nodes/comfyui-reactor-node && \
    git -C ComfyUI/custom_nodes/comfyui-reactor-node checkout ${REACTOR_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt && \
    pip install --no-cache-dir "insightface==0.7.3" "onnxruntime-gpu==1.20.1"

# Custom nodes: PainterLongVideo (identity anchoring for chained segments)
RUN git clone https://github.com/princepainter/ComfyUI-PainterLongVideo.git \
    ComfyUI/custom_nodes/ComfyUI-PainterLongVideo && \
    git -C ComfyUI/custom_nodes/ComfyUI-PainterLongVideo checkout ${PAINTER_COMMIT}

# Custom nodes: WanVideoWrapper (KJ) — the VACE continuation path (nodes 5xx).
# Directory name matches the daemon's node_checker expectation.
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && \
    git -C ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper checkout ${WANVIDEO_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# Daemon Python dependencies (daemon code itself is cloned at boot for freshness)
RUN pip install --no-cache-dir httpx pydantic-settings python-dotenv websockets

# Config and scripts
COPY extra_model_paths.yaml /app/ComfyUI/extra_model_paths.yaml
COPY download_models.sh /app/download_models.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh /app/download_models.sh

EXPOSE 8188

ENTRYPOINT ["/app/start.sh"]
