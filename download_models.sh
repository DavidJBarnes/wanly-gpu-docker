#!/bin/bash
# Model staging for the LTX 2.3 worker.
#
# On the 3090 the LTX-2.3 set is BIND-MOUNTED read-only from the host
# (/home/david/LTX-2/models -> /workspace/models), so there is nothing to fetch and this
# script only has to prove the mount is really there and really complete.
#
# On a fresh pod with no such mount there IS a fetch to do, and it is written below.
#
# WHAT a pod needs comes from the workflow, not from what happens to sit in the folder on the
# 3090. That box holds 217 GB because it accumulated alternates -- three 43 GB checkpoints the
# recipe never names. The workflow template names four files, and the recipe patches in two
# more, so a working pod needs ~108 GB and not the folder.
#
# Progress reporting follows what #39 established for the WAN downloader this replaced:
# announce each file BEFORE starting it, and report size, elapsed and MB/s on completion.
# hf downloads print nothing while a 43 GB weight comes down, so reporting only on success
# means many minutes of silence that are indistinguishable from a hang -- which is exactly how
# a boot gets misread as stuck.
#
# Either way, failing here is enormously cheaper than failing 10 minutes into a claimed
# segment, which is what a missing or half-downloaded model actually costs.
set -uo pipefail

MODELS="${MODELS_DIR:-/workspace/models}"
LTX="$MODELS/ltx-2.3"
FAIL=0

echo "model root: $MODELS"
# A read-only bind mount is the 3090; anything else is a pod that may have to fetch.
mkdir -p "$LTX/diffusion_models" "$LTX/text_encoders" "$LTX/latent_upscale_models" \
         "$LTX/loras" "$MODELS/loras" 2>/dev/null || true
if [ ! -d "$MODELS" ]; then
    echo "!! FATAL: $MODELS does not exist and could not be created."
    echo "!! On the 3090 this is a bind mount: -v /home/david/LTX-2/models:/workspace/models:ro"
    exit 1
fi

# ---------- Fetch anything missing ----------
#
# Sources are RECORDED, never guessed. Every public entry below is the repo the file was
# actually taken from, carried over from ~/LTX-2/download-ltx23.sh and download-gemma-comfy.sh
# on the 3090 -- which lived only on that box until now.
#
# Xet is disabled deliberately: measured on the 3090 it moved 5 MB/s against ~2.2 MB/s for a
# single raw curl off the CDN, so parallel plain connections win. It is the link, not the
# client.
export HF_HUB_DISABLE_XET=1

# dest_subdir|filename|repo|path_in_repo   (path empty = filename at repo root)
_PUBLIC=(
  "diffusion_models|ltx-2.3-22b-dev.safetensors|Lightricks/LTX-2.3|"
  "text_encoders|gemma_3_12B_it_fp8_scaled.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
  "loras|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|Lightricks/LTX-2.3|"
  "latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|"
)

# The recipe patches these over the workflow's defaults, so a pod cannot render the validated
# stack without them -- and neither has a public home. They are fetched from S3 through the
# queue API's presigned redirect, the same path the daemon already uses for character LoRAs,
# so a pod needs QUEUE_API_KEY and no AWS credentials at all.
#
# relpath_under_models|s3_uri_env_var
_PRIVATE=(
  "ltx-2.3/diffusion_models/sulphur_dev_bf16.safetensors|SULPHUR_CKPT_S3"
  "loras/sulphur_distill_lora_condsafe.safetensors|SULPHUR_DISTILL_S3"
)

_report() {   # path started_at
    local f="$1" t0="$2" bytes elapsed
    bytes=$(stat -c %s "$f" 2>/dev/null || echo 0)
    elapsed=$(( $(date +%s) - t0 )); [ "$elapsed" -le 0 ] && elapsed=1
    printf "     %.1f GB in %dm%02ds (%.1f MB/s)\n" \
        "$(echo "$bytes/1000000000" | bc -l)" $((elapsed/60)) $((elapsed%60)) \
        "$(echo "$bytes/1000000/$elapsed" | bc -l)"
}

_fetch_public() {
    local dest="$LTX/$1" name="$2" repo="$3" path="$4" t0
    [ -s "$dest/$name" ] && return 0
    echo "  fetching $name from $repo"
    t0=$(date +%s)
    if [ -n "$path" ]; then
        # A repo path lands nested; move the file to where the loader looks for it.
        hf download "$repo" "$path" --local-dir /tmp/hfdl >/dev/null || return 1
        mv -f "/tmp/hfdl/$path" "$dest/$name" || return 1
    else
        hf download "$repo" "$name" --local-dir "$dest" >/dev/null || return 1
    fi
    _report "$dest/$name" "$t0"
}

_fetch_private() {
    local rel="$1" var="$2" uri="${!2:-}" dest="$MODELS/$1" t0
    [ -s "$dest" ] && return 0
    if [ -z "$uri" ]; then
        echo "  !! $rel is missing and \$$var is not set."
        echo "     It has no public source. Put it in S3 and pass $var=s3://bucket/key,"
        echo "     or mount the model directory as the 3090 does."
        return 1
    fi
    echo "  fetching $rel from $uri"
    t0=$(date +%s)
    mkdir -p "$(dirname "$dest")"
    # Two steps, deliberately. GET /files answers with a 307 to a presigned S3 URL; following
    # it with -L would resend X-API-Key to AWS, because curl only strips Authorization across
    # hosts and passes custom headers straight through. So the redirect is resolved with the
    # key, and the transfer itself carries no headers at all.
    local signed
    signed=$(curl -fsS --retry 3 --retry-delay 5 -o /dev/null -w '%{redirect_url}' \
        -H "X-API-Key: ${QUEUE_API_KEY:-}" \
        --get --data-urlencode "path=$uri" \
        "${QUEUE_URL:-http://api.wanly22.com:8001}/files") || true
    if [ -z "$signed" ]; then
        echo "     !! the queue API did not return a presigned URL for $uri"
        echo "        (check QUEUE_URL and QUEUE_API_KEY, and that the object exists)"
        return 1
    fi
    # .part first: an interrupted download must not masquerade as a complete file, which is
    # the exact failure the truncation check below exists to catch.
    curl -fsSL --retry 3 --retry-delay 5 "$signed" -o "$dest.part" \
        || { echo "     !! transfer failed"; rm -f "$dest.part"; return 1; }
    mv -f "$dest.part" "$dest"
    _report "$dest" "$t0"
}

# Only fetch when something is actually absent, so the 3090's read-only mount is untouched.
NEED_FETCH=0
for row in "${_PUBLIC[@]}"; do
    IFS='|' read -r d n _r _p <<< "$row"
    [ -s "$LTX/$d/$n" ] || NEED_FETCH=1
done
for row in "${_PRIVATE[@]}"; do
    IFS='|' read -r rel _v <<< "$row"
    [ -s "$MODELS/$rel" ] || NEED_FETCH=1
done

if [ "$NEED_FETCH" -eq 1 ]; then
    if [ ! -w "$MODELS" ]; then
        echo "!! FATAL: models are missing and $MODELS is read-only."
        echo "!! That is the 3090's bind mount — fix the mount rather than downloading here."
        exit 1
    fi
    command -v hf >/dev/null || pip install --no-cache-dir -q "huggingface_hub[cli]" || true
    echo "staging models (~108 GB on a cold pod; already-present files are skipped)"
    for row in "${_PUBLIC[@]}"; do
        IFS='|' read -r d n r pth <<< "$row"
        _fetch_public "$d" "$n" "$r" "$pth" || FAIL=1
    done
    for row in "${_PRIVATE[@]}"; do
        IFS='|' read -r rel v <<< "$row"
        _fetch_private "$rel" "$v" || FAIL=1
    done
fi

for d in diffusion_models loras text_encoders latent_upscale_models; do
    p="$LTX/$d"
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
        echo "  $d: $(ls -1 "$p" | wc -l) file(s) — $(ls -1 "$p" | head -3 | tr '\n' ' ')"
    else
        # loras also resolves against $MODELS/loras (see extra_model_paths.yaml), so a
        # missing versioned dir is not automatically fatal for that one.
        if [ "$d" = "loras" ] && [ -d "$MODELS/loras" ] && [ -n "$(ls -A "$MODELS/loras" 2>/dev/null)" ]; then
            echo "  $d: (empty under ltx-2.3, using $MODELS/loras)"
        else
            echo "  !! $d: MISSING or empty at $p"
            FAIL=1
        fi
    fi
done

# ---------- Truncated safetensors ----------
# A partial .safetensors is a VALID HEADER OVER MISSING DATA. It looks fine on disk, passes
# every existence check, and fails only at load — deep inside a render, on a segment already
# claimed. One arrived at 14.40 of 27.16 GiB (47% short) and reported nothing at all.
#
# The header declares where the data ends, so the file's true size is knowable without
# reading it: 8 + header_len + the largest data_offsets end.
#
# In its own file, deliberately. Embedding this in a nested heredoc once hit a quoting
# SyntaxError and the check reported success by printing nothing.
cat > /tmp/check_safetensors.py <<'PYEOF'
import json, struct, sys
from pathlib import Path

bad = 0
for p in sys.argv[1:]:
    f = Path(p)
    try:
        size = f.stat().st_size
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            # Sanity-bound before allocating. A non-safetensors file yields a garbage length
            # here, and reading it raised MemoryError -- technically a failure, but the
            # message named the wrong problem and on a large file it tries to allocate first.
            if n <= 0 or n > size:
                print(f"  !! NOT A SAFETENSORS {f.name}: header length {n} vs file size {size}")
                bad += 1
                continue
            header = json.loads(fh.read(n))
        ends = [v["data_offsets"][1] for v in header.values()
                if isinstance(v, dict) and "data_offsets" in v]
        expected = 8 + n + (max(ends) if ends else 0)
        actual = size
        if actual < expected:
            pct = 100 * actual / expected
            print(f"  !! TRUNCATED {f.name}: {actual} of {expected} bytes ({pct:.1f}%)")
            bad += 1
    except Exception as e:
        print(f"  !! UNREADABLE {f.name}: {type(e).__name__}: {e}")
        bad += 1
sys.exit(1 if bad else 0)
PYEOF

echo "checking safetensors headers against actual byte counts..."
mapfile -t FILES < <(find "$LTX" "$MODELS/loras" -maxdepth 2 -name '*.safetensors' 2>/dev/null)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "  (none found)"
else
    if python3 /tmp/check_safetensors.py "${FILES[@]}"; then
        echo "  ${#FILES[@]} file(s) OK"
    else
        echo "  !! at least one model is incomplete — it will fail at load, mid-render"
        FAIL=1
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    echo "!! FATAL: model staging incomplete. Refusing to boot a worker that will fail its claims."
    exit 1
fi
echo "models OK"
