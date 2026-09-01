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
# Sources are RECORDED, never guessed. The Lightricks and Comfy-Org entries come from
# ~/LTX-2/download-ltx23.sh and download-gemma-comfy.sh on the 3090, which lived only on that
# box. The two sulphur files are Sulphur 2, a drop-in LTX 2.3 replacement published at
# SulphurAI/Sulphur-2-base; the distill LoRA is stored here under a shorter name.
#
# WHAT a pod needs is what a RENDER loads, which is not what the workflow template names and
# not what the folder holds. The template says ltx-2.3-22b-dev.safetensors in three loaders,
# but the recipe patches all three to sulphur_dev_bf16 before submission, so the dev
# checkpoint never loads -- 43 GB that a pod would download and never open. Same for
# ltx-2.3-22b-distilled-lora-384-1.1: the recipe substitutes the sulphur distill LoRA.
# Confirmed against the resolved graph.json of a real render rather than read off the graph
# template. That is the difference between fetching 58 GB and fetching 108.
#
# Transfer mechanism, decided by measurement rather than by what sounds fastest.
#
# Xet is the parallel path in current huggingface_hub, and HF_HUB_ENABLE_HF_TRANSFER is
# deprecated in favour of it ("hf_transfer is not used anymore"). Enabling Xet was the obvious
# move and it was measured to be the wrong one, twice:
#
#   * it does not stream into --local-dir. Chunks go to its own cache and the file is
#     assembled at the end, so a progress meter watching the staging tree reports "0 MB so
#     far" for the entire download -- the exact silence this is meant to end. Pointing
#     HF_HOME and HF_XET_CACHE at the volume did not change that.
#   * on the 3090's link it barely moved: 42 KB in 20 seconds, against a classic download
#     that shows bytes within one 5s sample. The original note in the 3090's own script said
#     the same thing (5 MB/s Xet against 2.2 MB/s single raw curl, with the caveat that the
#     chunk cache was empty).
#
# So the classic path stays. It is single-stream, and single-stream measured 29 MB/s on a good
# pod -- fast enough that the ceiling was never the mechanism. On the one bad pod, 8 parallel
# ranges reached 3 MB/s against 0.7, which is 4x of a hopeless number: that host wanted
# killing, not tuning.
#
# HF_HOME still moves onto $MODELS. Whatever the mechanism, its cache must not land on the
# container overlay, which is 30-40 GB against a 43 GB file.
export HF_HUB_DISABLE_XET=1
export HF_HOME="$MODELS/.hf"
mkdir -p "$HF_HOME" 2>/dev/null

# dest_dir_under_$MODELS|final_filename|repo|path_in_repo (empty = filename at repo root)
_WANTED=(
  "ltx-2.3/diffusion_models|sulphur_dev_bf16.safetensors|SulphurAI/Sulphur-2-base|"
  "ltx-2.3/text_encoders|gemma_3_12B_it_fp8_scaled.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
  "ltx-2.3/latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|"
  "loras|sulphur_distill_lora_condsafe.safetensors|SulphurAI/Sulphur-2-base|distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
)
# Character LoRAs are NOT here: the daemon syncs those per claim from S3, so a pod carries
# only the ones its jobs actually name.

_report() {   # path started_at
    # Integers and builtins only. `bc` is NOT in this image, and calling it printed
    # "bc: command not found" mid-boot and took the download with it.
    local f="$1" t0="$2" bytes elapsed gb_whole gb_tenth rate
    bytes=$(stat -c %s "$f" 2>/dev/null || echo 0)
    elapsed=$(( $(date +%s) - t0 )); [ "$elapsed" -le 0 ] && elapsed=1
    gb_whole=$(( bytes / 1073741824 ))
    gb_tenth=$(( (bytes % 1073741824) * 10 / 1073741824 ))
    rate=$(( bytes / 1048576 / elapsed ))
    echo "     ${gb_whole}.${gb_tenth} GB in $((elapsed/60))m$((elapsed%60))s (${rate} MB/s)"
}

_fetch() {
    local dest_dir="$MODELS/$1" name="$2" repo="$3" path="$4" t0
    [ -s "$dest_dir/$name" ] && return 0
    mkdir -p "$dest_dir"
    echo "  fetching $name from $repo"
    t0=$(date +%s)
    # Staged INSIDE $MODELS, never /tmp. On the 3090 /tmp is a large host disk; on a pod it is
    # the container overlay, which is 30-40 GB against a 43 GB checkpoint -- so staging there
    # cannot finish however big the volume is. The first real pod died exactly this way, at
    # 28 GB of 43 with the 60 GB volume sitting empty beside it.
    #
    # Same filesystem as the destination, so the move below is a rename rather than a second
    # 43 GB copy.
    #
    # Staged and moved rather than written in place because a repo path lands nested, the
    # stored name can differ from the repo's, and a partial download must never be visible
    # under the final name -- the silent failure the truncation check below exists to catch.
    local stage="$MODELS/.staging"
    rm -rf "$stage"
    mkdir -p "$stage"
    # Run the download in the background and report progress from HERE, rather than letting
    # huggingface_hub's own bar speak. That bar is drawn with \r and no newlines, so it produces
    # nothing a line-oriented log can show: a 43 GB file is 45 minutes of total silence, which
    # is exactly how an hour was spent waiting on a pod whose link had quietly collapsed from
    # 13 MB/s to 1 MB/s. A rate printed every 30s makes that visible in the first minute.
    python3 - "$repo" "${path:-$name}" "$stage" <<'PYEOF' &
import sys
from huggingface_hub import hf_hub_download
repo, filename, stage = sys.argv[1], sys.argv[2], sys.argv[3]
hf_hub_download(repo_id=repo, filename=filename, local_dir=stage)
PYEOF
    local dl_pid=$!

    # Progress reporting must never be able to break the download it is reporting on. The
    # first version did exactly that on a 37 MB/s pod: `bc` is not in this image, and awk
    # printed the byte total as 3.19816e+09, which bash arithmetic cannot parse. The errors
    # aborted _fetch, the models came up missing, the boot failed, and RunPod restarted it --
    # a dead loop caused entirely by the logging. So this subshell swallows its own failures
    # and uses nothing but integers and shell builtins.
    (
        local_last=0
        while kill -0 "$dl_pid" 2>/dev/null; do
            sleep "${PROGRESS_INTERVAL:-30}"
            kill -0 "$dl_pid" 2>/dev/null || break
            # No awk. This is the third mawk trap in one day: it buffers input by block
            # (which swallowed the whole boot log), prints large sums in scientific notation
            # (which bash cannot subtract), and its printf "%d" SATURATES AT INT32_MAX --
            # so a 28 GB download reported 2147483647 bytes, i.e. "1.9 GB so far, 0 MB/s",
            # for as long as it ran. Bash arithmetic is 64-bit; du and cut are enough.
            stage_bytes=$(du -sb "$stage" 2>/dev/null | cut -f1); stage_bytes=${stage_bytes:-0}
            hf_bytes=$(du -sb "$HF_HOME" 2>/dev/null | cut -f1); hf_bytes=${hf_bytes:-0}
            now_bytes=$(( stage_bytes + hf_bytes ))
            delta=$(( now_bytes - local_last ))
            [ "$delta" -lt 0 ] && delta=0
            rate=$(( delta / ${PROGRESS_INTERVAL:-30} / 1048576 ))
            # Integer maths only -- one decimal place done by hand, because `bc` is not
            # installed and reaching for it is what started this.
            gb_whole=$(( now_bytes / 1073741824 ))
            gb_tenth=$(( (now_bytes % 1073741824) * 10 / 1073741824 ))
            if [ "$gb_whole" -gt 0 ]; then
                echo "     $name: ${gb_whole}.${gb_tenth} GB so far, ${rate} MB/s"
            else
                echo "     $name: $(( now_bytes / 1048576 )) MB so far, ${rate} MB/s"
            fi
            local_last=$now_bytes
        done
    ) 2>/dev/null || true

    wait "$dl_pid" || return 1
    mv -f "$stage/${path:-$name}" "$dest_dir/$name" || return 1
    rm -rf "$stage"
    _report "$dest_dir/$name" "$t0"
}

NEED_FETCH=0
for row in "${_WANTED[@]}"; do
    IFS='|' read -r d n _r _p <<< "$row"
    [ -s "$MODELS/$d/$n" ] || NEED_FETCH=1
done

if [ "$NEED_FETCH" -eq 1 ]; then
    if [ ! -w "$MODELS" ]; then
        echo "!! FATAL: models are missing and $MODELS is read-only."
        echo "!! That is the 3090's bind mount — fix the mount rather than downloading here."
        exit 1
    fi
    python3 -c "import huggingface_hub" 2>/dev/null \
        || pip install --no-cache-dir -q huggingface_hub \
        || { echo "!! FATAL: huggingface_hub is not installed and could not be installed"; exit 1; }
    echo "staging models (~58 GB on a cold pod; anything already present is skipped)"
    for row in "${_WANTED[@]}"; do
        IFS='|' read -r d n r pth <<< "$row"
        _fetch "$d" "$n" "$r" "$pth" || FAIL=1
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
