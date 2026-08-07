#!/usr/bin/env python3
"""Fail the build if onnxruntime's CUDA provider cannot resolve its shared libraries.

This exists because the runtime failure is silent. onnxruntime does not raise when its CUDA
provider fails to load — it logs a warning, drops to CPUExecutionProvider, and carries on. The
faceswap then runs in software at roughly 1/100th the speed, so what actually reaches you is the
daemon's "no progress from ComfyUI for 300s" watchdog: a hang, four steps from the cause.

Two real failures, both shipped in images that looked correct on inspection:

  1. A custom node's install.py replaced the pinned onnxruntime-gpu with a CUDA-13 wheel on this
     CUDA-12.4 image -> libcublasLt.so.13 missing.
  2. cuDNN was present in site-packages but not on LD_LIBRARY_PATH -> libcudnn_adv.so.9 missing.

A version assertion catches only the first. Checking that every NEEDED entry resolves catches
both, and whatever the third one turns out to be.

ldd is used rather than loading the provider, because the CI builder has no GPU. It resolves
against the same LD_LIBRARY_PATH the container will use, so a pass here means a pass at runtime.
"""
import glob
import subprocess
import sys

PROVIDER_GLOB = "/usr/local/lib/python3.*/dist-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so"


def main() -> int:
    matches = glob.glob(PROVIDER_GLOB)
    if not matches:
        print("FAIL: onnxruntime-gpu is installed without its CUDA provider .so", file=sys.stderr)
        return 1
    provider = matches[0]

    import onnxruntime

    ldd = subprocess.run(["ldd", provider], capture_output=True, text=True).stdout
    missing = [line.split("=>")[0].strip() for line in ldd.splitlines() if "not found" in line]

    cuda_deps = sorted(
        {
            line.split("=>")[0].strip()
            for line in ldd.splitlines()
            if any(k in line for k in ("cublas", "cudart", "cudnn"))
        }
    )
    print(f"onnxruntime {onnxruntime.__version__}")
    print(f"  CUDA deps: {' '.join(cuda_deps)}")

    if missing:
        print(
            "FAIL: onnxruntime CUDA provider cannot resolve:\n  "
            + "\n  ".join(missing)
            + "\n\nThe container would fall back to CPUExecutionProvider silently and the "
            "faceswap would run ~100x slower.\nEither the wheel targets a different CUDA major "
            "than this image, or LD_LIBRARY_PATH is missing the\npip-installed nvidia/*/lib "
            "directories.",
            file=sys.stderr,
        )
        return 1

    print("OK: every CUDA provider dependency resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
