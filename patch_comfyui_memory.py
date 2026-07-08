#!/usr/bin/env python3
"""Apply the wanly CFG-aware VRAM-estimate patch to ComfyUI's model_base.py.
Replaces the hardcoded 0.01 activation-memory coefficient with a per-run value read from
/tmp/wanly_estimate (the daemon writes 0.015 for CFG>1 / de-distilled, 0.003 for distilled).
This is what prevents over-reserve slow/OOM on 24GB cards and makes de-distilled jobs fit.
Idempotent; fails loud if the anchor line isn't found (ComfyUI version drift)."""
import sys, pathlib
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app/ComfyUI/comfy/model_base.py")
s = p.read_text()
if "_wanly_e" in s:
    print("model_base.py already patched — skipping"); sys.exit(0)
stock = "            return (area * comfy.model_management.dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (1024 * 1024)"
patched = (
    '            try:\n'
    '                _wanly_e = float(open("/tmp/wanly_estimate").read().strip())\n'
    '            except Exception:\n'
    '                _wanly_e = 0.003\n'
    "            return (area * comfy.model_management.dtype_size(dtype) * _wanly_e * self.memory_usage_factor) * (1024 * 1024)"
)
if stock not in s:
    print("ERROR: anchor line not found — ComfyUI model_base.py changed; patch needs review", file=sys.stderr)
    sys.exit(1)
p.write_text(s.replace(stock, patched, 1))
print("model_base.py patched: 0.01 -> /tmp/wanly_estimate (cfg-aware VRAM estimate)")
