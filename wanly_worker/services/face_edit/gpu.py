"""Which device an edit runs on, decided per request -- and giving the card back.

LivePortrait is small (~0.5 GB of weights, ~1.5-2 GB peak with activations and the CUDA
context -- an estimate; /health reports the measured peak once an edit has run on GPU). It is
the smallest tenant of every card it could share, and the only one with a fallback: an edit
runs on CPU in seconds, a render or an A1111 generation that is starved of VRAM just dies. So
the rule is that face-edit NEVER takes VRAM a neighbour may need, and never asks a neighbour to
stop work in progress:

    1. FACE_EDIT_DEVICE=cpu, or no CUDA device        -> cpu
    2. the render engine on this card has work        -> cpu   (the 3090 in render mode)
    3. Automatic1111 on this card is generating       -> cpu   (the 2070 under generate-forever)
    4. already resident on the GPU                    -> cuda  (a slider session)
    5. enough free VRAM (FACE_EDIT_MIN_FREE_MIB)       -> cuda
    6. A1111 idle but holding its checkpoint          -> ask it to unload, re-measure
    7. otherwise                                      -> cpu

With FACE_EDIT_CPU_FALLBACK=0 every "cpu" above except (1) becomes a refusal, which the API
turns into a 503 saying why.

Step 6 is the captioner's `_yield_the_gpu` (wanly-api app/joycaption.py), moved to the box that
shares the card: A1111 holds its checkpoint (~1.7-1.9 GB idle on the 2070) whether or not it is
generating, reloads it from RAM in a few seconds on its next generation, and is asked ONLY when
the unload buys something -- never before every edit. It lives here rather than in wanly-api
because only this process knows which card it is on: wanly-api's `a1111_url` names the 2070,
and an API-side yield would unload the 2070's A1111 to make room for an edit running on the
3090.

Free VRAM is read through nvidia-smi (the supervisor's gpu_snapshot), not torch: asking torch
creates a CUDA context in this process, ~300 MB held for the life of the process, which is
exactly the VRAM this module exists not to take on a box that may never run an edit on GPU.
"""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("face-edit")

DEVICE = os.environ.get("FACE_EDIT_DEVICE", "auto").strip().lower()
#: Estimated peak ~1.5-2 GB; the bar leaves headroom so an edit cannot push a neighbour whose
#: allocation is momentarily low into an OOM on its next step.
MIN_FREE_MIB = int(os.environ.get("FACE_EDIT_MIN_FREE_MIB", "2560"))
CPU_FALLBACK = os.environ.get("FACE_EDIT_CPU_FALLBACK", "1") == "1"
#: Automatic1111 sharing this card. Empty = none (the 3090, where A1111 is gone). On the 2070,
#: from inside the container: http://host.docker.internal:7860 (run-worker.sh adds the host
#: alias whenever face-edit is enabled).
A1111_URL = os.environ.get("FACE_EDIT_A1111_URL", "").strip().rstrip("/")
A1111_TIMEOUT_S = float(os.environ.get("FACE_EDIT_A1111_TIMEOUT_S", "20"))
#: The render engine in this container (ltx-engine, loopback). Unreachable means no render
#: stack is running here -- a face-edit-only box, or caption mode.
ENGINE_HEALTH = os.environ.get(
    "FACE_EDIT_ENGINE_HEALTH", f"http://127.0.0.1:{os.environ.get('API_PORT', '8190')}/health")


class GpuUnavailable(RuntimeError):
    """The GPU cannot be used right now and CPU fallback is off. Becomes a 503."""


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def free_mib() -> int | None:
    from wanly_worker.supervisor import gpu_snapshot
    snap = gpu_snapshot()
    return None if snap is None else int(snap["vram_free_mib"])


def render_busy() -> bool:
    """True when the render engine in this container has a job running or queued."""
    try:
        r = httpx.get(ENGINE_HEALTH, timeout=3)
        if r.status_code != 200:
            return False
        d = r.json()
        return bool((d.get("running") or 0) + (d.get("queue_depth") or 0))
    except Exception:
        return False


def a1111_generating() -> bool:
    """A1111's own progress endpoint. Unreachable or unconfigured is "not generating"."""
    if not A1111_URL:
        return False
    try:
        r = httpx.get(f"{A1111_URL}/sdapi/v1/progress", timeout=5)
        return r.status_code == 200 and bool((r.json().get("state") or {}).get("job_count"))
    except Exception:
        return False


def yield_a1111() -> bool:
    """Ask Automatic1111 for the card back. True only if it actually let go.

    Never interrupts a generation: the check is repeated immediately before the unload, since
    generate-forever can start the next image between our first look and this call.
    """
    if not A1111_URL:
        return False
    try:
        with httpx.Client(timeout=A1111_TIMEOUT_S) as c:
            busy = c.get(f"{A1111_URL}/sdapi/v1/progress")
            if busy.status_code == 200 and (busy.json().get("state") or {}).get("job_count"):
                log.info("A1111 is generating; leaving its checkpoint alone")
                return False
            r = c.post(f"{A1111_URL}/sdapi/v1/unload-checkpoint")
            if r.status_code != 200:
                log.warning("A1111 refused to unload: %s", r.status_code)
                return False
    except httpx.HTTPError as e:
        log.info("A1111 not reachable at %s (%s) — nothing to free", A1111_URL, e)
        return False
    log.info("A1111 released its checkpoint for a face edit")
    return True


def neighbour_busy() -> str | None:
    """Why the GPU belongs to someone else right now, or None."""
    if render_busy():
        return "the render engine on this card has work in flight"
    if a1111_generating():
        return "Automatic1111 on this card is generating"
    return None


def choose(resident_on_gpu: bool) -> tuple[str, str]:
    """(device, reason). Raises GpuUnavailable when the answer is cpu and fallback is off."""
    if DEVICE == "cpu":
        return "cpu", "FACE_EDIT_DEVICE=cpu"
    if not cuda_available():
        return "cpu", "no CUDA device visible"

    reason = neighbour_busy()
    if reason is None:
        if resident_on_gpu:
            return "cuda", "already resident on the GPU"
        free = free_mib()
        if free is None or free >= MIN_FREE_MIB:
            return "cuda", (f"{free} MiB free" if free is not None else "free VRAM unreadable")
        if yield_a1111():
            free = free_mib()
            if free is None or free >= MIN_FREE_MIB:
                return "cuda", f"A1111 unloaded its checkpoint; {free} MiB free"
        reason = f"only {free} MiB of VRAM free (want {MIN_FREE_MIB})"

    if DEVICE == "cuda" or not CPU_FALLBACK:
        raise GpuUnavailable(f"GPU unavailable: {reason}")
    return "cpu", reason
