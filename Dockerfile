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
ARG FACEFUSION_COMMIT=3e73829

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
#
# 1.20.2 not 1.20.1: 1.20.1 was REMOVED from PyPI (the index jumps 1.20.0 -> 1.20.2), so the
# old pin stopped resolving and broke the build. Staying inside 1.20.x keeps the CUDA-12
# wheel the pin exists to guarantee — do not bump to >=1.23 without checking libcudart.
RUN git clone https://github.com/Gourieff/ComfyUI-ReActor.git \
    ComfyUI/custom_nodes/comfyui-reactor-node && \
    git -C ComfyUI/custom_nodes/comfyui-reactor-node checkout ${REACTOR_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt && \
    pip install --no-cache-dir "insightface==0.7.3" "onnxruntime-gpu==1.20.2"

# Custom nodes: PainterLongVideo (identity anchoring for chained segments)
RUN git clone https://github.com/princepainter/ComfyUI-PainterLongVideo.git \
    ComfyUI/custom_nodes/ComfyUI-PainterLongVideo && \
    git -C ComfyUI/custom_nodes/ComfyUI-PainterLongVideo checkout ${PAINTER_COMMIT}

# Custom nodes: FaceFusion — the swapper the validated identity recipe uses.
# ReActor (above) selects the target face by POSITION only; FaceFusion is what the recipe and
# the console default (faceswap_method=facefusion) build node 183 `AdvancedSwapFaceImage` from.
# Without this every job using the recipe fails at ComfyUI prompt validation.
#
# A clean clone is NOT sufficient. See patches/README.md — the node needs the NSFW gate
# disabled, and an import shim without which it does not load at all on this ComfyUI build.
COPY patches/ /app/patches/
RUN git clone https://github.com/huygiatrng/Facefusion_comfyui.git \
    ComfyUI/custom_nodes/Facefusion_comfyui && \
    git -C ComfyUI/custom_nodes/Facefusion_comfyui checkout ${FACEFUSION_COMMIT} && \
    pip install --no-cache-dir -r ComfyUI/custom_nodes/Facefusion_comfyui/requirements.txt && \
    cp /app/patches/comfy_compat.py ComfyUI/custom_nodes/Facefusion_comfyui/facefusion_api/nodes/comfy_compat.py && \
    git -C ComfyUI/custom_nodes/Facefusion_comfyui apply /app/patches/facefusion_comfyui.patch && \
    python3 -c "import ast,sys; ast.parse(open('ComfyUI/custom_nodes/Facefusion_comfyui/content_filter/content_filter.py').read())" && \
    grep -q 'NSFW filter disabled' ComfyUI/custom_nodes/Facefusion_comfyui/content_filter/content_filter.py && \
    grep -q 'from .comfy_compat import' ComfyUI/custom_nodes/Facefusion_comfyui/facefusion_api/nodes/base.py && \
    touch ComfyUI/custom_nodes/Facefusion_comfyui/.install_complete

# The marker above is load-bearing, and its absence cost a whole RunPod job.
#
# ComfyUI runs every custom node's install.py at startup. FaceFusion's does:
#
#     if pip_show('onnxruntime-gpu').ok and exists('.install_complete'): return
#     pip uninstall onnx onnxruntime onnxruntime-gpu -y
#     pip install onnxruntime-gpu          # <- UNPINNED
#
# It needs BOTH conditions, and the marker is only ever written at RUNTIME. So a freshly built
# image always fails the guard on its first container start, uninstalls the pinned 1.20.2, and
# installs whatever is newest -- currently 1.28.0, a CUDA-13 wheel that cannot load against this
# image's CUDA 12.4 (libcublasLt.so.13: cannot open shared object file).
#
# Nothing crashes. onnxruntime just reports the CUDA provider unavailable and runs the swap on
# CPU: GPU at 0%, one process at 463%, and every RIFE-interpolated frame swapped in software.
# The daemon's 300s no-progress watchdog then reports "execution appears stuck" -- so the visible
# symptom is a hang, several layers away from the cause.
#
# It never reproduced on the 3090 because that install has long since written its marker. Fresh
# containers are exactly the case nobody was testing.
#
# Writing the marker at build time makes the guard short-circuit forever, keeping the pin the
# line above it already paid for.

# Fail the BUILD if the onnxruntime CUDA provider needs libraries this image does not ship.
# Checked via the provider .so's own NEEDED entries rather than by running it, because the
# builder has no GPU -- and checked at all because the runtime failure mode is silent.
RUN python3 -c "\
import glob, subprocess, sys, onnxruntime;\
so = glob.glob('/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so');\
sys.exit('onnxruntime-gpu missing its CUDA provider') if not so else None;\
need = [l.split()[1] for l in subprocess.run(['objdump','-p',so[0]],capture_output=True,text=True).stdout.splitlines() if 'NEEDED' in l];\
bad = [n for n in need if n.endswith('.so.13')];\
print('onnxruntime', onnxruntime.__version__, '| CUDA provider needs:', ' '.join(n for n in need if 'cublas' in n or 'cudart' in n));\
sys.exit('CUDA-13 libs required by onnxruntime but image is CUDA 12: %s' % bad) if bad else print('OK: onnxruntime CUDA provider matches this image CUDA major')"

# Bake the FaceFusion models the recipe actually uses (~900MB). The node self-downloads on
# first use into its OWN directory, which lives in the image rather than on the volume -- so
# without this every container start re-downloads them and the first job stalls. GitHub
# releases, not HuggingFace, so download_models.sh (hf_hub_download only) cannot stage them.
#
# This is exactly the set workflow_builder.py asks for:
#   inswapper_128         face_swapper_model (David's visual pick over hyperswap_1c_256)
#   retinaface_10g        face_detector_model
#   xseg_1                face_occluder_model  -- matters when hands cross the face
#   bisenet_resnet_34     face_parser_model
#   arcface_w600k_r50     recognizer, always loaded
#   scrfd_2.5g            the node's DEFAULT detector; 3MB, kept as a fallback
ARG FF_ASSETS=https://github.com/facefusion/facefusion-assets/releases/download
RUN FFM=ComfyUI/custom_nodes/Facefusion_comfyui/models && mkdir -p $FFM && \
    for u in models-3.0.0/inswapper_128.onnx \
             models-3.0.0/arcface_w600k_r50.onnx \
             models-3.0.0/retinaface_10g.onnx \
             models-3.0.0/scrfd_2.5g.onnx \
             models-3.0.0/bisenet_resnet_34.onnx \
             models-3.1.0/xseg_1.onnx ; do \
        f=$(basename $u); \
        curl -fsSL --retry 3 --retry-delay 5 -o "$FFM/$f" "${FF_ASSETS}/$u" || exit 1; \
        [ -s "$FFM/$f" ] || { echo "empty download: $f"; exit 1; }; \
    done && \
    ls -la $FFM

# NOTE: WanVideoWrapper (KJ) was installed here for the VACE continuation and Lynx paths.
# Both engines were retired from wanly-gpu-daemon, and nothing on the native i2v path uses
# its nodes, so it is no longer installed — its unpinned requirements were a recurring
# source of dependency skew against this image's torch 2.6.0 / CUDA 12.4.

# Daemon Python dependencies. The daemon CODE is cloned at boot for freshness, so its
# requirements.txt is not present at build time -- but hand-listing the deps here means the
# list silently rots as the daemon grows. It already had: numpy, onnxruntime,
# opencv-python-headless, Pillow, pyyaml and insightface were all missing, and identity
# scoring only worked because ComfyUI and the ReActor layer happened to pull them in.
#
# Pinned copy of daemon/requirements.txt. When that file changes, this must change with it --
# start.sh re-checks at boot and warns loudly if they have diverged.
COPY daemon-requirements.txt /app/daemon-requirements.txt
RUN pip install --no-cache-dir -r /app/daemon-requirements.txt

# Config and scripts
COPY extra_model_paths.yaml /app/ComfyUI/extra_model_paths.yaml
COPY download_models.sh /app/download_models.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh /app/download_models.sh

EXPOSE 8188

ENTRYPOINT ["/app/start.sh"]
