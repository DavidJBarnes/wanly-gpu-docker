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
    git ffmpeg aria2 wget curl \
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

# Custom nodes: Frame Interpolation (RIFE)
# Install cupy-cuda12x directly (pre-built wheel) — the requirements file's cupy-wheel
# is a source meta-package that fails on CI runners without a GPU
RUN pip install --no-cache-dir cupy-cuda12x
RUN git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt

# Custom nodes: Video Helper Suite
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# Custom nodes: ReActor (face swap)
# Directory name matches daemon's node_checker.py primary key
# onnxruntime-gpu is required at runtime but not in ReActor's requirements.txt.
# PIN it: unpinned, newer onnxruntime-gpu wheels link CUDA 13 (libcudart.so.13) and fail
# to import on this CUDA 12.4 image ("libcudart.so.13: cannot open shared object file").
# 1.20.x is a CUDA-12 / cuDNN-9 build, satisfied by the libs torch 2.6 (cu124) bundles.
RUN git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git \
    ComfyUI/custom_nodes/comfyui-reactor-node && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt && \
    pip install --no-cache-dir "insightface==0.7.3" "onnxruntime-gpu==1.20.1"

# Custom nodes: PainterLongVideo (identity anchoring for chained segments)
RUN git clone --depth 1 https://github.com/princepainter/ComfyUI-PainterLongVideo.git \
    ComfyUI/custom_nodes/ComfyUI-PainterLongVideo

# Daemon Python dependencies (daemon code itself is cloned at boot for freshness)
RUN pip install --no-cache-dir httpx pydantic-settings python-dotenv websockets

# Config and scripts
COPY extra_model_paths.yaml /app/ComfyUI/extra_model_paths.yaml
COPY download_models.sh /app/download_models.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh /app/download_models.sh

EXPOSE 8188

ENTRYPOINT ["/app/start.sh"]
