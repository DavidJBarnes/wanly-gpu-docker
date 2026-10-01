#!/bin/bash
# Model staging for the LTX 2.3 worker.
#
# On the 3090 the LTX-2.3 set is BIND-MOUNTED read-only from the host
# (/home/david/LTX-2/models -> /workspace/models), so there is nothing to fetch and this
# script only has to prove the mount is really there and really complete.
#
# On a fresh pod with no such mount there IS a fetch to do, and it is written below.
#
# WHICH OF THE TWO IT IS, IS DECIDED BY THE MOUNT, NOT BY GUESSWORK. A bind-mounted $MODELS is
# host-provided and authoritative: a missing file there is a mount or host-tree fault and this
# script says so and stops. An unmounted $MODELS is a pod's own directory and gets filled.
#
# That distinction is the whole of #77. A container started by hand with no -v and no --gpus
# had an empty $MODELS, so every file legitimately looked missing, and it began downloading
# 10Eros at 3 MB/s over the 3090's own complete copy of it -- onto a 30-40 GB overlay that
# could never have held the 43 GB file. Nothing in the boot said any of that was wrong,
# because on a pod it is the correct behaviour. It is now three refusals, each in one second:
# no GPU (start.sh), an incomplete bind mount, and nowhere to put the bytes.
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

# `--image-edit` checks (and on a pod, fetches) the Qwen-Image-Edit set for the image-edit
# service instead of the LTX set (wanly-console#548). Same rules, same mount logic, same
# truncation check -- a second script would be a second copy of all three to drift.
TARGET=ltx
[ "${1:-}" = "--image-edit" ] && TARGET=image-edit

if [ "$TARGET" = image-edit ]; then
    MODELS="${IMAGE_EDIT_MODELS_DIR:-/workspace/qwen}"
    LTX="$MODELS"
else
    MODELS="${MODELS_DIR:-/workspace/models}"
    LTX="$MODELS/ltx-2.3"
fi
FAIL=0

echo "model root: $MODELS"
# A read-only bind mount is the 3090; anything else is a pod that may have to fetch.
if [ "$TARGET" = ltx ]; then
    mkdir -p "$LTX/diffusion_models" "$LTX/text_encoders" "$LTX/latent_upscale_models" \
             "$LTX/loras" "$MODELS/loras" 2>/dev/null || true
else
    mkdir -p "$MODELS/base" "$MODELS/text_encoders" "$MODELS/vae" "$MODELS/loras" 2>/dev/null || true
fi
if [ ! -d "$MODELS" ]; then
    echo "!! FATAL: $MODELS does not exist and could not be created."
    echo "!! On the 3090 this is a bind mount: -v /home/david/LTX-2/models:/workspace/models:ro"
    exit 1
fi

# ---------- Fetch anything missing ----------
#
# Sources are RECORDED, never guessed. The Lightricks and Comfy-Org entries come from
# ~/LTX-2/download-ltx23.sh and download-gemma-comfy.sh on the 3090, which lived only on that
# box. The distill LoRA is Sulphur 2's, from SulphurAI/Sulphur-2-base, stored here under a
# shorter name. The checkpoint is 10Eros v1.5 from TenStrip/LTX2.3-10Eros -- verified to be
# the same 46,139,886,366-byte file the 3090 has been rendering on.
#
# THE CHECKPOINT HERE IS THE DEFAULT, and that is not a coincidence to be maintained by hand.
# It must equal engine/recipe.py's DEFAULT_CHECKPOINT, which is asserted by
# tests/test_recipe_resolve.py. A pod that fetches a checkpoint other than the default
# reports it through the heartbeat, the API's model gate hides every default pose from it,
# and it sits there claiming nothing -- indistinguishable from an empty queue.
#
# Only the default is fetched, not every checkpoint a pose might name. Each is ~46 GB, so
# fetching both 10Eros and sulphur would take a cold boot from ~58 GB to ~104 and roughly
# double time-to-first-claim. A pod simply is not offered poses it cannot render; the model
# gate (wanly-api _model_gate, console#422) already handles that correctly.
#
# WHAT a pod needs is what a RENDER loads, which is not what the workflow template names and
# not what the folder holds. The template says ltx-2.3-22b-dev.safetensors in three loaders,
# but the recipe patches all three to the render's own checkpoint before submission, so the
# dev checkpoint never loads -- 43 GB that a pod would download and never open. Same for
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
# Is a path a mount point in its own right? Parsed from /proc/self/mountinfo rather than by
# calling `mountpoint`, which lives in util-linux and is not installed in this image. Field 5
# of each mountinfo line is the mount point, compared whole so a path that merely appears
# elsewhere on the line (the in-filesystem root, an option string) cannot match by accident.
_is_mountpoint() {
    local target="$1" mp
    while read -r _ _ _ _ mp _; do
        [ "$mp" = "$target" ] && return 0
    done < /proc/self/mountinfo
    return 1
}

export HF_HUB_DISABLE_XET=1
export HF_HOME="$MODELS/.hf"
mkdir -p "$HF_HOME" 2>/dev/null

# dest_dir_under_$MODELS|final_filename|repo|path_in_repo|approx_GiB
#
# The size column is for the free-space precheck below and nothing else. It is approximate on
# purpose: rounded up from the 3090's verified copies, so the check errs towards refusing a
# marginal disk rather than towards starting a download that cannot finish. Integrity is not
# its job -- the safetensors header check at the bottom of this file does that exactly.
_WANTED=(
  "ltx-2.3/diffusion_models|10Eros_v1.5_bf16.safetensors|TenStrip/LTX2.3-10Eros||43"
  "ltx-2.3/text_encoders|gemma_3_12B_it_fp8_scaled.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors|13"
  "ltx-2.3/latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3||1"
  "loras|sulphur_distill_lora_condsafe.safetensors|SulphurAI/Sulphur-2-base|distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors|1"
  # The identity-reference LoRAs (#156): face mode and character-sheet mode. Loaded only when a
  # render carries an identity reference, but staged on every worker -- a claim does not know in
  # advance whether its character has a sheet. Must match engine/recipe.py IDENTITY_LORAS
  # (test_identity_ref.py holds them together). 2.47 GB + 1.31 GB on HF.
  "loras|Best_FaceID_v1.0_LoRA.safetensors|Alissonerdx/LTX-Best-Face-ID||3"
  "loras|Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors|Alissonerdx/LTX-Best-Face-ID||2"
)
# Character LoRAs are NOT here: the daemon syncs those per claim from S3, so a pod carries
# only the ones its jobs actually name.

# THE IMAGE-EDIT SET, relative to IMAGE_EDIT_MODELS_DIR (the 3090's ~/models/qwen). Since
# wanly-console#574 / wanly-gpu-docker#157 it is the OFFICIAL Qwen-Image-Edit-2511 as Comfy-Org
# packages it for ComfyUI -- three files, the exact set the character-sheet recipe was proven
# with (loras/phase0-2026-10-01/qwen_setup.sh). It replaced Phr00t's Rapid-AIO v23 merge, which
# changed faces. Must match graph.MODEL_FILES (test_image_edit.py holds them together).
#   base/           the transformer, fp8mixed (sensitive layers kept in bf16), 20.5 GB on HF.
#                   NOT the qwen_image_edit_2511_fp8_e4m3fn already in ~/models/qwen/base: that
#                   is a third-party all-fp8 cast of unknown origin, and the recipe's results
#                   were made with this file.
#   text_encoders/  Qwen2.5-VL 7B, fp8 scaled, 9.4 GB
#   vae/            the Qwen-Image VAE, 0.25 GB
# Paths match service.paths_yaml. No LoRA: the head-angle recipe is prompt-only; the #548 spike
# rejected the one community angle LoRA for this base.
if [ "$TARGET" = image-edit ]; then
    _WANTED=(
      "base|qwen_image_edit_2511_fp8mixed.safetensors|Comfy-Org/Qwen-Image-Edit_ComfyUI|split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors|20"
      "text_encoders|qwen_2.5_vl_7b_fp8_scaled.safetensors|Comfy-Org/Qwen-Image_ComfyUI|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors|9"
      "vae|qwen_image_vae.safetensors|Comfy-Org/Qwen-Image_ComfyUI|split_files/vae/qwen_image_vae.safetensors|1"
    )
fi

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
MISSING_NAMES=""
NEED_GIB=0
for row in "${_WANTED[@]}"; do
    IFS='|' read -r d n _r _p gib <<< "$row"
    if [ ! -s "$MODELS/$d/$n" ]; then
        NEED_FETCH=1
        MISSING_NAMES="$MISSING_NAMES $d/$n"
        NEED_GIB=$(( NEED_GIB + ${gib:-0} ))
    fi
done

if [ "$NEED_FETCH" -eq 1 ]; then
    echo "missing:$MISSING_NAMES"

    # ---------- Is this model root the host's, or ours to fill? ----------
    #
    # A BIND-MOUNTED $MODELS is host-provided and AUTHORITATIVE. That is the 3090, where the
    # tree is 217 GB of weights that exist outside any container and are managed by hand. If a
    # file the recipe needs is not in it, the fault is in the mount or in the host tree, and
    # the answer is to fix that -- not to fetch a second copy of a 43 GB file the box already
    # has, and certainly not into a mount that is read-only anyway.
    #
    # An UNMOUNTED $MODELS is a pod's own directory, which starts empty and is ours to fill.
    #
    # Note this is only consulted when something is MISSING. A complete bind-mounted tree never
    # reaches here, which is the 3090's normal boot: "models OK" in about a second.
    if _is_mountpoint "$MODELS"; then
        echo "!! FATAL: $MODELS is a bind mount from the host, and it is incomplete."
        echo "!! The host provides these models; downloading a second copy here is wrong."
        echo "!! Fix the host tree (on the 3090: /home/david/LTX-2/models, or ~/models/qwen for --image-edit) or the mount."
        exit 1
    fi

    # Kept below the mount check, which now catches the 3090 with a better message. This
    # remains for a read-only model root that is NOT a mount -- a baked-in or squashed image
    # layer, say. Unreachable today; a one-line guard against a silent permission failure
    # 40 GB into a download is worth keeping anyway.
    if [ ! -w "$MODELS" ]; then
        echo "!! FATAL: models are missing and $MODELS is read-only, so they cannot be staged."
        exit 1
    fi

    # ---------- Somewhere to actually put it ----------
    #
    # The container overlay is 30-40 GB and 10Eros alone is 43. A fetch onto it cannot finish,
    # however long it runs -- the first real pod died exactly this way, at 28 GB of 43, with
    # its 60 GB volume sitting empty beside it, and a hand-started container on the 3090
    # repeated it (#77). Both spent hours proving something knowable in one `df`.
    #
    # A little headroom on top, because the staging copy and the final file briefly coexist
    # while the rename lands and hf's cache holds blocks of its own.
    #
    # No awk. `awk` here is mawk, which renders a large integer through CONVFMT "%.6g" -- so a
    # 150 GB free-space figure prints as 1.5e+11 and bash cannot subtract it. That is the same
    # trap that made a 28 GB download report itself as 1.9 GB. df's POSIX format is five fixed
    # fields and bash can read them directly.
    avail_bytes=0
    read -r _ _ _ avail_bytes _ <<< "$(df -PB1 "$MODELS" 2>/dev/null | tail -1)"
    avail_gib=$(( ${avail_bytes:-0} / 1073741824 ))
    want_gib=$(( NEED_GIB + 5 ))
    if [ "$avail_gib" -lt "$want_gib" ]; then
        echo "!! FATAL: need ~${want_gib} GiB free under $MODELS to stage the missing models,"
        echo "!! but only ${avail_gib} GiB is available."
        echo "!! $MODELS is not a mount here, so this is the container's own filesystem —"
        echo "!! it is almost certainly missing a volume (pod) or a -v bind mount (3090)."
        echo "!! Refusing to start a download that cannot complete."
        exit 1
    fi
    python3 -c "import huggingface_hub" 2>/dev/null \
        || pip install --no-cache-dir -q huggingface_hub \
        || { echo "!! FATAL: huggingface_hub is not installed and could not be installed"; exit 1; }
    echo "staging ~${NEED_GIB} GiB into $MODELS (${avail_gib} GiB free)"
    for row in "${_WANTED[@]}"; do
        IFS='|' read -r d n r pth _gib <<< "$row"
        _fetch "$d" "$n" "$r" "$pth" || FAIL=1
    done
fi

[ "$TARGET" = ltx ] && LTX_DIRS="diffusion_models loras text_encoders latent_upscale_models" \
                     || LTX_DIRS=""
for d in $LTX_DIRS; do
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
if [ "$TARGET" = image-edit ]; then
    # Only what the service loads: ~/models/qwen also holds v23, v19 and unrelated LoRAs, and a
    # broken file there is not this service's to refuse a boot over.
    FILES=()
    for row in "${_WANTED[@]}"; do
        IFS='|' read -r d n _r _p _gib <<< "$row"
        [ -f "$MODELS/$d/$n" ] && FILES+=("$MODELS/$d/$n")
    done

    # AuraFace (fal/AuraFace-v1 glintr100.onnx) scores each edit against its source. It lives in
    # insightface's store -- a writable host mount beside face-crop's buffalo_l -- and is
    # fetched there when absent. NOT fatal: an edit without a score is still an edit, and the
    # service says why the number is missing.
    AURA="${INSIGHTFACE_ROOT:-/root/.insightface}/models/auraface/glintr100.onnx"
    if [ ! -s "$AURA" ]; then
        echo "  fetching AuraFace glintr100.onnx into $(dirname "$AURA")"
        mkdir -p "$(dirname "$AURA")" 2>/dev/null
        python3 - "$(dirname "$AURA")" <<'PYEOF' || echo "  !! AuraFace fetch failed -- edits will come back unscored"
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="fal/AuraFace-v1", filename="glintr100.onnx", local_dir=sys.argv[1])
PYEOF
    else
        echo "  AuraFace present: $AURA"
    fi
else
    mapfile -t FILES < <(find "$LTX" "$MODELS/loras" -maxdepth 2 -name '*.safetensors' 2>/dev/null)
fi
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
