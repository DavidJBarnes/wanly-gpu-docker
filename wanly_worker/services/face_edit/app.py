"""The face-edit service's HTTP API (wanly-console#547).

Called, not claimed -- the same shape as face-crop and the captioner: wanly-api calls it inline
and waits. An edit is ~1 s of GPU (keyframe-server measured 1.0 s on the 3090) or a few seconds
of CPU, which is far below anything worth a queue row.

    POST /edit     {image: b64, expression?: {...}, prompt?: str, ...} -> {image: b64 png, ...}
    GET  /health   model_loaded, device, the measured VRAM peak

ONE EDIT AT A TIME. The pipeline is one set of weights on one device; two edits at once would
race the device move. A second request waits up to FACE_EDIT_QUEUE_WAIT_S for its turn and is
then refused as busy (503), which wanly-api passes on as "busy, try again" rather than a hang.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wanly_worker.services.face_edit import gpu
from wanly_worker.services.face_edit.engine import Engine, NoFace
from wanly_worker.services.face_edit.expression import (
    Expression, NothingToApply, resolve_expression, whole_face_motion,
)
from wanly_worker.services.face_edit.restore import DETAIL_SHARPEN, restore_detail

log = logging.getLogger("face-edit")

#: ExpressionEditor's crop_factor: how far past the detected face box the warp region reaches.
#: It needs hairline and jaw to anchor against. Texture scales inversely with it (1.6 -> 16%,
#: 2.0 -> 11%, 2.5 -> 7%, measured by keyframe-server), and the node's own range is 1.5-2.5.
FACE_PAD = float(os.environ.get("FACE_EDIT_FACE_PAD", "1.6"))
#: How long a request waits for the one in front of it.
QUEUE_WAIT_S = float(os.environ.get("FACE_EDIT_QUEUE_WAIT_S", "60"))
#: Seconds a GPU-resident pipeline may sit idle before it goes back to CPU and the VRAM goes
#: back to the neighbours. Long enough to cover a slider session in the editor, where each
#: drag is an edit; short enough that A1111 gets its card back before anyone notices.
GPU_IDLE_S = float(os.environ.get("FACE_EDIT_GPU_IDLE_S", "45"))
#: Largest input edge accepted. A phone photo is ~4000 px; above this is almost certainly a
#: mistake, and the full-frame composite is a few HxWx3 float arrays.
MAX_EDGE = int(os.environ.get("FACE_EDIT_MAX_EDGE", "6000"))

engine = Engine()
_turn = asyncio.Lock()
_last_reason: str | None = None

app = FastAPI(title="wanly face-edit")


@app.on_event("startup")
async def _startup() -> None:
    """Load the weights (into RAM, not VRAM) BEFORE anything can ask for an edit, and start
    the idle watcher.

    Loading at startup is what makes the first edit cost ~1 s rather than ~1 s plus a model
    load. In a thread, because it is blocking and the supervisor's readiness probe must keep
    getting answers meanwhile -- a probe that times out kills a service doing exactly what it
    should. Not fatal: /health says what went wrong and every edit retries the load.
    """
    async def load():
        try:
            await asyncio.to_thread(engine.load)
            print(f"[face-edit] LivePortrait loaded on cpu in {engine.loaded_in_s}s "
                  f"(device policy: {gpu.DEVICE}, cpu fallback: {gpu.CPU_FALLBACK}, "
                  f"min free {gpu.MIN_FREE_MIB} MiB, A1111: {gpu.A1111_URL or 'none'})",
                  flush=True)
        except Exception as e:
            print(f"[face-edit] could not load LivePortrait: {e}", flush=True)

    asyncio.create_task(load())
    asyncio.create_task(_release_when_idle())


async def _release_when_idle() -> None:
    """Move the pipeline off the GPU once it has been idle -- or at once, when a neighbour
    starts working. Holding VRAM across an A1111 generation is the collision this service is
    built to avoid, so "idle for GPU_IDLE_S" is not the only way out."""
    while True:
        await asyncio.sleep(5)
        if engine.device != "cuda" or _turn.locked():
            continue
        idle = time.time() - engine.last_used
        why = None
        if idle >= GPU_IDLE_S:
            why = f"idle {idle:.0f}s"
        else:
            why = await asyncio.to_thread(gpu.neighbour_busy)
        if why:
            async with _turn:
                if engine.device == "cuda":
                    await asyncio.to_thread(engine.to, "cpu")
                    print(f"[face-edit] released the GPU ({why})", flush=True)


class EditRequest(BaseModel):
    #: base64 image bytes (any format PIL reads). Base64 rather than multipart because the
    #: caller is wanly-api holding bytes it fetched from S3, not a browser with a file handle.
    image: str = Field(min_length=1)
    expression: Expression | None = None
    #: Sugar for a curl; wanly-api always sends numbers.
    prompt: str = ""
    face_pad: float | None = Field(default=None, ge=1.5, le=2.5)
    #: How much of the subject's resting expression to keep. Below 1 the face relaxes toward
    #: neutral before the requested change is applied.
    src_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    #: 1.0 restores source texture everywhere the warp did not move anything; 0 returns the
    #: node's output untouched. See restore.py.
    detail_restore: float = Field(default=1.0, ge=0.0, le=1.0)
    detail_sharpen: float | None = Field(default=None, ge=0.0, le=3.0)


def _decode(b64: str):
    from PIL import Image, ImageOps
    import numpy as np

    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64, validate=False)))
        # Phone photos carry their rotation in EXIF. The console shows them upright, so the
        # edit must happen on the upright pixels or the result comes back sideways.
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode the image: {e}") from e
    if max(im.size) > MAX_EDGE:
        raise HTTPException(413, f"image is {im.width}x{im.height}; the limit is {MAX_EDGE} px "
                                 f"on the long edge (FACE_EDIT_MAX_EDGE)")
    return np.asarray(im, dtype=np.uint8).copy()


def _encode_png(rgb) -> str:
    from PIL import Image

    buf = io.BytesIO()
    # PNG, never JPEG: the point of this engine is that the pixels it did not move are the
    # source's own, byte for byte. A lossy re-encode would throw that away on the way out.
    Image.fromarray(rgb).save(buf, "PNG", compress_level=6)
    return base64.b64encode(buf.getvalue()).decode()


def _is_oom(e: Exception) -> bool:
    """torch.cuda.OutOfMemoryError, recognised without importing torch here (the tests run
    without it, and so does a box whose torch is broken -- which preflight reports)."""
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _run(rgb, req: EditRequest, exp: Expression, pad: float) -> dict:
    """The blocking part: pick a device, edit, fall back to CPU on an OOM. In a thread."""
    global _last_reason

    t0 = time.time()
    device, reason = gpu.choose(resident_on_gpu=engine.device == "cuda")
    t_move = 0.0
    try:
        # Inside the try: moving ~0.5 GB of weights onto the card can itself be the OOM.
        engine.to(device)
        t_move = time.time() - t0
        out = engine.edit(rgb, exp, face_pad=pad, src_ratio=req.src_ratio)
    except Exception as e:
        if device != "cuda" or not _is_oom(e):
            raise
        # The free-VRAM check is a snapshot; a neighbour can grow between it and the warp.
        engine.to("cpu")
        if not gpu.CPU_FALLBACK or gpu.DEVICE == "cuda":
            raise gpu.GpuUnavailable("GPU ran out of memory mid-edit and CPU fallback is off")
        device, reason = "cpu", "the GPU ran out of memory mid-edit"
        out = engine.edit(rgb, exp, face_pad=pad, src_ratio=req.src_ratio)
    t_edit = time.time() - t0 - t_move
    _last_reason = reason

    t1 = time.time()
    if req.detail_restore > 0:
        sharpen = req.detail_sharpen if req.detail_sharpen is not None else DETAIL_SHARPEN
        out = restore_detail(rgb, out, req.detail_restore, sharpen,
                             whole_face_motion(exp, req.src_ratio))
    t_restore = time.time() - t1
    return {"out": out, "device": device, "reason": reason,
            "timings_ms": {"device": round(t_move * 1000), "edit": round(t_edit * 1000),
                           "restore": round(t_restore * 1000)}}


@app.post("/edit")
async def edit(req: EditRequest):
    t0 = time.time()
    try:
        exp, how = resolve_expression(req.expression, req.prompt)
    except NothingToApply as e:
        raise HTTPException(422, str(e)) from e
    if not engine.loaded:
        try:
            await asyncio.to_thread(engine.load)
        except Exception:
            raise HTTPException(503, f"face-edit could not load its models: {engine.load_error}")

    rgb = await asyncio.to_thread(_decode, req.image)
    pad = req.face_pad if req.face_pad is not None else FACE_PAD

    try:
        await asyncio.wait_for(_turn.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, f"face-edit is busy: another edit held the pipeline for "
                                 f"{QUEUE_WAIT_S:.0f}s")
    try:
        res = await asyncio.to_thread(_run, rgb, req, exp, pad)
    except NoFace:
        raise HTTPException(422, "no face detected in the image")
    except gpu.GpuUnavailable as e:
        raise HTTPException(503, str(e))
    finally:
        _turn.release()

    image = await asyncio.to_thread(_encode_png, res["out"])
    total = time.time() - t0
    print(f"[face-edit] {rgb.shape[1]}x{rgb.shape[0]} on {res['device']} ({res['reason']}) | "
          f"src={how} | {exp.nonzero() or 'neutral'} | {total:.1f}s "
          f"{res['timings_ms']}"
          + (f" | vram peak {engine.vram_peak_mib} MiB" if res["device"] == "cuda" else ""),
          flush=True)
    return {
        "image": image,
        "format": "png",
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "expression": exp.model_dump(),
        "source": how,
        "face_pad": pad,
        "device": res["device"],
        "device_reason": res["reason"],
        "timings_ms": {**res["timings_ms"], "total": round(total * 1000)},
        "vram_peak_mib": engine.vram_peak_mib if res["device"] == "cuda" else None,
    }


@app.get("/health")
async def health():
    # model_loaded separately from status: the service answers long before the weights are in
    # memory, and a bare 200 would say "ready" while the first edit still owed a model load --
    # the exact gap that made face-crop look unreachable (see face_crop/app.py).
    return {
        "status": "ok",
        "model_loaded": engine.loaded,
        "load_error": engine.load_error,
        "loaded_in_s": engine.loaded_in_s,
        "device": engine.device,
        "last_device_reason": _last_reason,
        "busy": _turn.locked(),
        "vram_peak_mib": engine.vram_peak_mib,
        "policy": {"device": gpu.DEVICE, "cpu_fallback": gpu.CPU_FALLBACK,
                   "min_free_mib": gpu.MIN_FREE_MIB, "a1111": gpu.A1111_URL or None,
                   "gpu_idle_s": GPU_IDLE_S, "face_pad": FACE_PAD},
    }
