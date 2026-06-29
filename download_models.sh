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
    # NOTE: base Wan 2.2 diffusion models are NOT downloaded — the validated pipeline uses
    # the DaSiWa "Lightspeed" remix (fetched from Civitai below), matching the 3090.
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "model", True),
    (WAN, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
          f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "model", True),
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

# --- DaSiWa "Lightspeed" remix (Civitai) — the VALIDATED inference model (matches 3090) ---
# These are gated downloads (HTTP 401 without auth), so a Civitai API token is required.
# Set CIVITAI_TOKEN as a pod env var at launch. The Lightspeed distillation is baked into
# this model, which is why the validated config runs LIGHTX2V_STRENGTH_* = 0 (see start.sh).
DASIWA_HIGH_URL="https://civitai.red/api/download/models/2953474?fileId=2837908"
DASIWA_LOW_URL="https://civitai.red/api/download/models/2953485?fileId=2837910"

dl_civitai() {  # url  dest
    local url="$1" dst="$2" name; name="$(basename "$dst")"
    if [ -f "$dst" ] && [ "$(stat -c%s "$dst" 2>/dev/null || echo 0)" -gt 1000000 ]; then
        echo "SKIP ${name} (already exists)"; return 0
    fi
    if [ -z "${CIVITAI_TOKEN:-}" ]; then
        echo "ERROR: CIVITAI_TOKEN not set — cannot download DaSiWa model ${name}" >&2
        return 1
    fi
    echo "DOWNLOADING ${name} (Civitai)"
    curl -fL --retry 3 --retry-delay 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
        -o "${dst}.part" "$url" && mv -f "${dst}.part" "$dst"
    echo "OK ${name} ($(( $(stat -c%s "$dst") / 1000000 )) MB)"
}

dl_civitai "$DASIWA_HIGH_URL" "${MODELS_DIR}/diffusion_models/DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors"
dl_civitai "$DASIWA_LOW_URL"  "${MODELS_DIR}/diffusion_models/DasiwaWAN22I2V14BLightspeed_snatchkissLowV11.safetensors"

echo "=== Model download complete ==="
