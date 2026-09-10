"""What to tell the caller when a render dies (wanly-gpu-docker#95).

Torch's allocator dump is not an explanation, so the engine has always rewritten an OOM into
"reduce resolution or frame count". That advice was WRONG on 2026-09-10: every render on the
3090 died on its first 384 KiB allocation with 23 GiB free, because the container had lost
its GPU device access after a systemd reload, and the rewrite blamed 1216x832 -- a size that
had rendered dozens of times. Pure, so the two signatures can be asserted without a GPU.
"""
import re

#: Torch's dump when the process could not allocate ANYTHING. A real out-of-memory reports
#: gigabytes already allocated; zero means the device refused the very first request.
_NOTHING_ALLOCATED = re.compile(r"Currently allocated\s*:\s*0 bytes")
#: What a fresh process says once the device cgroup rules are gone.
_NO_DEVICE = ("No CUDA GPUs are available", "Failed to initialize NVML",
              "CUDA-capable device(s) is/are busy or unavailable", "cudaErrorNoDevice")


def lost_gpu_access(msg: str) -> bool:
    """True when the failure is "cannot touch the GPU at all", not "the GPU is full"."""
    return bool(_NOTHING_ALLOCATED.search(msg)) or any(s in msg for s in _NO_DEVICE)


def explain_failure(msg: str, width: int, height: int, num_frames: int) -> str:
    """The error as the caller should read it.

    Two very different failures both print "out of memory". Distinguish them, because the
    remedies are opposites: one is "ask for less", the other is "restart the container".
    """
    if lost_gpu_access(msg):
        return (msg + " — the process could not allocate on the GPU at all (0 bytes held), "
                "which is lost GPU device access, not a full card: it happens when systemd "
                "reloads after the container started (wanly-gpu-docker#95). Check with "
                "`head -c1 /dev/nvidiactl` inside the container ('Operation not permitted' "
                "confirms it) and restart the container. Do NOT reduce resolution.")
    if "OOM" in msg or "out of memory" in msg.lower():
        # The card is 24 GB and shared. Say what to change rather than returning torch's
        # allocator dump.
        return (msg + f" — {width}x{height} at {num_frames} frames did not fit. Reduce "
                "resolution or frame count; 704x1280 at 241 frames is known to fit alongside "
                "the other resident containers.")
    return msg
