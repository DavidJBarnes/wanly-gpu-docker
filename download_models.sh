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
         "${MODELS_DIR}/loras" "${MODELS_DIR}/clip_vision" "${MODELS_DIR}/text_encoders" \
         "${INSIGHTFACE_DIR}"

# Xet support. Install hf_xet and ensure huggingface_hub is recent enough for Xet,
# but DON'T cap the version: this image ships transformers 5.x which requires
# huggingface-hub>=1.5.0, so an upper bound would downgrade it and break the resolver.
# No bound = pip keeps the image's existing (Xet-capable) hub and just adds hf_xet.
pip install --no-cache-dir -q "huggingface_hub>=0.34" hf_xet || true

echo "=== Downloading models to ${MODELS_DIR} ==="

MODELS_DIR="$MODELS_DIR" INSIGHTFACE_DIR="$INSIGHTFACE_DIR" python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

M = os.environ["MODELS_DIR"]
IF = os.environ["INSIGHTFACE_DIR"]
WAN = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# (repo, path-in-repo, local destination, repo_type, critical)
# critical=True  -> generation can't run without it; failure aborts the boot.
# critical=False -> auxiliary (faceswap / identity-anchor); failure only warns, so a
#                   gated/moved aux repo never bricks the whole pod boot.
JOBS = [
    (WAN, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          f"{M}/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "model", True),
    (WAN, "split_files/vae/wan_2.1_vae.safetensors",
          f"{M}/vae/wan_2.1_vae.safetensors", "model", True),
    # Base Wan 2.2 i2v diffusion models (high+low) — NATIVE fp8-scaled (~14GB each), matching
    # the 3090. Pre-quantized fp8 with scale tensors, so ComfyUI uses the native scaled-fp8 path
    # (no fp16->fp8 runtime cast, no "manual cast to fp16") — the path that runs fast on the
    # current ComfyUI build. Full base identity, no distillation.
    (WAN, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
          f"{M}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "model", True),
    (WAN, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
          f"{M}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "model", True),
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "model", True),
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "model", True),
    # === VACE continuation stack (identity-anchored multi-segment) ===
    # Local filenames MUST match the daemon config's vace_* settings (single source of truth), even
    # where the HF path differs. Critical: with the WanVideoWrapper nodes installed, the daemon
    # routes continuations to VACE, so a booted pod must have these or VACE segments fail at load.
    # T2V base (fp16, ~28GB each) — same weights the 3090 uses; block-swap keeps it on 24GB.
    (WAN, "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
          f"{M}/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", "model", True),
    (WAN, "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
          f"{M}/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", "model", True),
    # Fun-VACE modules (fp8, ~3GB each) — Kijai's VACE/ subfolder.
    ("Kijai/WanVideo_comfy_fp8_scaled", "VACE/Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
          f"{M}/diffusion_models/Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors", "model", True),
    ("Kijai/WanVideo_comfy_fp8_scaled", "VACE/Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors",
          f"{M}/diffusion_models/Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors", "model", True),
    # T2V 4-step Lightning distill LoRAs (lightx2v repo names them high/low_noise_model.safetensors;
    # saved under the daemon's expected names). Rank-64 Seko V1.1.
    ("lightx2v/Wan2.2-Lightning", "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
          f"{M}/loras/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors", "model", True),
    ("lightx2v/Wan2.2-Lightning", "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
          f"{M}/loras/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors", "model", True),
    # umt5-xxl text encoder, bf16 .pth (~11GB) — native Wan-AI name/path (Kijai ships a renamed .safetensors).
    ("Wan-AI/Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth",
          f"{M}/text_encoders/models_t5_umt5-xxl-enc-bf16.pth", "model", True),
    # CLIP Vision (PainterLongVideo identity anchoring) -- auxiliary
    ("h94/IP-Adapter", "models/image_encoder/model.safetensors",
          f"{M}/clip_vision/clip_vision_h.safetensors", "model", False),
    # ReActor face-swap model -- lives in a DATASET repo, and is auxiliary
    ("Gourieff/ReActor", "models/inswapper_128.onnx",
          f"{IF}/inswapper_128.onnx", "dataset", False),
]

for repo, path, dst, repo_type, critical in JOBS:
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
    # Download straight into the destination filesystem (local_dir), then atomically
    # move into place. Avoids the HF global cache, so we never transiently hold two
    # copies of a 27GB weight (matters on a fresh pod with a modest volume).
    stage = os.path.join(os.path.dirname(dst), ".hfstage")
    os.makedirs(stage, exist_ok=True)
    try:
        src = hf_hub_download(repo_id=repo, filename=path, repo_type=repo_type, local_dir=stage)
        os.replace(src, dst)                      # same filesystem -> instant, no copy
        print(f"OK {name} ({os.path.getsize(dst)//1_000_000} MB)")
    except Exception as e:
        if critical:
            raise
        print(f"WARN: optional {name} failed ({type(e).__name__}); continuing boot")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

print("=== HF model download complete ===")
PY

echo "=== Model download complete ==="
