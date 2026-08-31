#!/bin/bash
# Model staging for the LTX 2.3 worker.
#
# On the 3090 the LTX-2.3 set is BIND-MOUNTED read-only from the host
# (/home/david/LTX-2/models -> /workspace/models), so there is nothing to fetch and this
# script only has to prove the mount is really there and really complete.
#
# On a fresh pod with no such mount there IS a fetch to do — roughly 126 GB — and it is not
# written yet, because inventing download URLs for checkpoints nobody has published to a
# fixed location would be worse than failing with a clear message. See the TODO below.
#
# Either way, failing here is enormously cheaper than failing 10 minutes into a claimed
# segment, which is what a missing or half-downloaded model actually costs.
set -uo pipefail

MODELS="${MODELS_DIR:-/workspace/models}"
LTX="$MODELS/ltx-2.3"
FAIL=0

echo "model root: $MODELS"
if [ ! -d "$MODELS" ]; then
    echo "!! FATAL: $MODELS does not exist."
    echo "!! On the 3090 this is a bind mount: -v /home/david/LTX-2/models:/workspace/models:ro"
    echo "!! TODO: a pod with no mount needs a real download step (~126 GB). Not implemented."
    exit 1
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
