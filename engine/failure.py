"""What to tell the caller when a render dies (wanly-gpu-docker#95).

Torch's allocator dump is not an explanation, so the engine has always rewritten an OOM into
"reduce resolution or frame count". That advice was WRONG on 2026-09-10: every render on the
3090 died on its first 384 KiB allocation with 23 GiB free, because the container had lost
its GPU device access after a systemd reload, and the rewrite blamed 1216x832 -- a size that
had rendered dozens of times. Pure, so the two signatures can be asserted without a GPU.
"""
import os
import re
from pathlib import Path

#: Torch's dump when the process could not allocate ANYTHING. A real out-of-memory reports
#: gigabytes already allocated; zero means the device refused the very first request.
_NOTHING_ALLOCATED = re.compile(r"Currently allocated\s*:\s*0 bytes")
#: What a fresh process says once the device cgroup rules are gone.
_NO_DEVICE = ("No CUDA GPUs are available", "Failed to initialize NVML",
              "CUDA-capable device(s) is/are busy or unavailable", "cudaErrorNoDevice")


def lost_gpu_access(msg: str) -> bool:
    """True when the failure is "cannot touch the GPU at all", not "the GPU is full"."""
    return bool(_NOTHING_ALLOCATED.search(msg)) or any(s in msg for s in _NO_DEVICE)


#: The engine's own cgroup. With a private cgroup namespace (Docker's default on cgroup v2)
#: this is the CONTAINER's cgroup, i.e. the one WORKER_MEMORY_LIMIT caps.
CGROUP_DIR = Path(os.environ.get("ENGINE_CGROUP_DIR", "/sys/fs/cgroup"))

#: What requests says when ComfyUI went away under a poll: the port is closed (process dead),
#: or the socket was torn down mid-request. Not "neither queued nor in history": that is also
#: what a deliberate /interrupt looks like, and it must not read as a crash.
_COMFY_GONE = ("Connection refused", "RemoteDisconnected", "Connection aborted",
               "Connection reset")


def oom_kills(cgroup_dir: Path | None = None) -> int | None:
    """The container cgroup's lifetime OOM-kill count, or None if it cannot be read."""
    try:
        text = ((cgroup_dir or CGROUP_DIR) / "memory.events").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        key, _, val = line.partition(" ")
        if key == "oom_kill":
            return int(val)
    return None


def memory_cap(cgroup_dir: Path | None = None) -> int | None:
    """The container's memory.max in bytes, or None when uncapped or unreadable."""
    try:
        raw = ((cgroup_dir or CGROUP_DIR) / "memory.max").read_text().strip()
    except OSError:
        return None
    return None if raw == "max" else int(raw)


def comfy_gone(msg: str) -> bool:
    """True when the failure is "ComfyUI stopped answering", not something it reported."""
    return any(s in msg for s in _COMFY_GONE)


def explain_comfy_gone(msg: str, new_oom_kills: int | None, cap: int | None) -> str:
    """Why ComfyUI vanished mid-render (wanly-gpu-docker#166).

    On 2026-10-03 a render on 3090a failed as a bare "Connection refused" on /history: the
    kernel had OOM-killed ComfyUI at the container's 54 GiB cap when the in-container
    captioner loaded Qwen3-VL 32B beside it. The supervisor then restarted the container, so
    by the time anyone looked the evidence was in `journalctl -k` only. Say it here instead.
    """
    cap_s = f"{cap / 2**30:.0f} GiB" if cap else "no cap"
    if new_oom_kills:
        return (msg + f" — ComfyUI was OOM-killed: the container hit its memory cap "
                f"({cap_s}, memory.max) and the kernel killed {new_oom_kills} process(es) "
                "during this render. Something else in the container needed RAM at the same "
                "time — most often the image-description captioner loading its model mid-render. "
                "Host evidence: `journalctl -k | grep -i oom`. Raising WORKER_MEMORY_LIMIT "
                "trades this for the host-wide swap thrash in wanly-gpu-docker#166; keep the "
                "captioner off the box while it renders instead.")
    if new_oom_kills == 0:
        return (msg + " — ComfyUI stopped answering mid-render: the process exited or "
                f"restarted. It was NOT OOM-killed (no OOM kill in the container's cgroup, cap "
                f"{cap_s}), so look at /workspace/logs/comfyui.log for a crash and at "
                "`dmesg | grep -i xid` on the host for a GPU fault.")
    return (msg + " — ComfyUI stopped answering mid-render: the process exited or restarted. "
            "The container's memory.events was unreadable, so an OOM kill could not be ruled "
            "in or out: check `journalctl -k | grep -i oom` on the host and "
            "/workspace/logs/comfyui.log.")


def explain_failure(msg: str, width: int, height: int, num_frames: int,
                    oom_kills_before: int | None = None) -> str:
    """The error as the caller should read it.

    Two very different failures both print "out of memory". Distinguish them, because the
    remedies are opposites: one is "ask for less", the other is "restart the container".

    A third reads as a network error and is not one: ComfyUI dying under the poll. Pass
    `oom_kills_before` (oom_kills() at job start) so the explanation can say whether the
    container's memory cap killed it.
    """
    if comfy_gone(msg):
        now = oom_kills()
        new = None if now is None or oom_kills_before is None else now - oom_kills_before
        return explain_comfy_gone(msg, new, memory_cap())
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
