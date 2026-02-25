#!/bin/bash
# Download all required models to /workspace/models (RunPod persistent volume).
# Uses aria2c for parallel connections and resume support.
# Skips files that already exist with non-zero size.
set -e

MODELS_DIR="/workspace/models"
HF_BASE="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

mkdir -p "${MODELS_DIR}/clip"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/clip_vision"

download() {
    local url="$1"
    local dest="$2"

    # .aria2 control file means a previous download was interrupted
    if [ -f "${dest}.aria2" ]; then
        echo "RESUMING: $(basename "$dest") (incomplete from previous run)"
        rm -f "$dest"
    elif [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "SKIP: $(basename "$dest") (already exists)"
        return 0
    fi

    echo "DOWNLOADING: $(basename "$dest")"
    aria2c -x 8 -s 8 --console-log-level=error --summary-interval=30 \
        -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"

    if [ ! -f "$dest" ] || [ ! -s "$dest" ]; then
        echo "ERROR: Failed to download $(basename "$dest")"
        return 1
    fi
    echo "OK: $(basename "$dest")"
}

echo "=== Downloading models to ${MODELS_DIR} ==="

# Text encoder (~6.7 GB)
download "${HF_BASE}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "${MODELS_DIR}/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# VAE (~254 MB)
download "${HF_BASE}/vae/wan_2.1_vae.safetensors" \
    "${MODELS_DIR}/vae/wan_2.1_vae.safetensors"

# Diffusion models — fp16, ~28.6 GB each
download "${HF_BASE}/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "${MODELS_DIR}/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

download "${HF_BASE}/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "${MODELS_DIR}/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# LightX2V acceleration LoRAs
download "${HF_BASE}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
    "${MODELS_DIR}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"

download "${HF_BASE}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
    "${MODELS_DIR}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# CLIP Vision model for PainterLongVideo identity anchoring (~3.9 GB)
download "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors"

# ReActor face swap model — goes into ComfyUI/models/insightface (not persistent volume)
INSIGHTFACE_DIR="/app/ComfyUI/models/insightface"
mkdir -p "$INSIGHTFACE_DIR"
download "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx" \
    "${INSIGHTFACE_DIR}/inswapper_128.onnx"

echo "=== Model download complete ==="
