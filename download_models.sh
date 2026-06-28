#!/bin/bash
# Download all required models to /workspace/models (RunPod persistent volume).
#
# Uses huggingface_hub's native downloader (NOT aria2). HF has migrated many weights
# to "Xet" chunked storage, whose presigned URLs carry short-lived per-connection byte
# ranges that aria2's multi-connection mode cannot satisfy -> 403 mid-download (errorCode=22).
# Files in the SAME repo are mixed (some Xet, some classic LFS), so aria2 fails on some and
# not others, intermittently. The hf_hub_download path + hf_xet handles Xet correctly and
# refreshes URLs. Idempotent: skips files that already exist with a sane size, and clears
# any leftover aria2 partials from older image versions.
set -e

MODELS_DIR="/workspace/models"
INSIGHTFACE_DIR="/app/ComfyUI/models/insightface"

mkdir -p "${MODELS_DIR}/clip" "${MODELS_DIR}/vae" "${MODELS_DIR}/diffusion_models" \
         "${MODELS_DIR}/loras" "${MODELS_DIR}/clip_vision" "${INSIGHTFACE_DIR}"

# Xet support, pinned <1.0: huggingface_hub 1.x renames the CLI and trips transformers'
# `huggingface-hub<1.0` pin used by ComfyUI custom nodes (e.g. ReActor) on this image.
pip install --no-cache-dir -q "huggingface_hub>=0.34,<1.0" hf_xet || true

echo "=== Downloading models to ${MODELS_DIR} ==="

MODELS_DIR="$MODELS_DIR" INSIGHTFACE_DIR="$INSIGHTFACE_DIR" python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

M = os.environ["MODELS_DIR"]
IF = os.environ["INSIGHTFACE_DIR"]
WAN = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# (repo, path-in-repo, local destination)
JOBS = [
    (WAN, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          f"{M}/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    (WAN, "split_files/vae/wan_2.1_vae.safetensors",
          f"{M}/vae/wan_2.1_vae.safetensors"),
    (WAN, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
          f"{M}/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
    (WAN, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
          f"{M}/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"),
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
    # CLIP Vision (PainterLongVideo identity anchoring)
    ("h94/IP-Adapter", "models/image_encoder/model.safetensors",
          f"{M}/clip_vision/clip_vision_h.safetensors"),
    # ReActor face-swap model
    ("Gourieff/ReActor", "models/inswapper_128.onnx",
          f"{IF}/inswapper_128.onnx"),
]

for repo, path, dst in JOBS:
    name = os.path.basename(dst)
    ctrl = dst + ".aria2"
    if os.path.exists(ctrl):                      # leftover aria2 partial -> redownload
        os.remove(ctrl)
        if os.path.exists(dst):
            os.remove(dst)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"SKIP {name} (already exists)")
        continue
    print(f"DOWNLOADING {name}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = hf_hub_download(repo_id=repo, filename=path)
    shutil.copy(src, dst)                          # copy out of the hub cache
    print(f"OK {name} ({os.path.getsize(dst)//1_000_000} MB)")

print("=== Model download complete ===")
PY
