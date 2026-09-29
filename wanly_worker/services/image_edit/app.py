"""The image-edit service's HTTP API: Qwen-Image-Edit "full mode" (wanly-console#548).

    POST /edit     {image: b64, instruction? | angle?: {yaw, pitch}, seed?, denoise?, ...}
                   -> {image: b64 png, width, height, prompt, identity: {aura, reason}, ...}
    GET  /health   comfy up, model loaded, busy, idle seconds, last edit's timings and VRAM

Called, not claimed, like face-edit -- but it only EXISTS in edit mode (registry.MODE_ONLY):
Qwen holds ~20 GB of the 3090, so the render stack is stopped (its segment finished first) before
this starts. wanly-api owns the queue and the mode switch; this answers one edit at a time.

WHAT IT IS ASKED TO DO is either free text (`instruction`) or a head angle (`angle`). The angle
recipe -- the words, the LoRA or not -- lives here, next to the model it was measured on, the way
face-edit's lexicon does; wanly-api sends numbers. See graph.py for the graph and the framing pins.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import subprocess
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wanly_worker.services.image_edit import graph, identity

COMFY_PORT = int(os.environ.get("IMAGE_EDIT_COMFY_PORT", "8286"))
COMFY = f"http://127.0.0.1:{COMFY_PORT}"
#: ComfyUI's input/output dirs for this instance, passed on its command line (service.py).
WORK_DIR = os.environ.get("IMAGE_EDIT_WORK_DIR", "/tmp/image-edit")
#: How long one edit may take, model load included. A cold first edit reads a 28 GB checkpoint.
EDIT_TIMEOUT_S = float(os.environ.get("IMAGE_EDIT_TIMEOUT_S", "900"))
QUEUE_WAIT_S = float(os.environ.get("IMAGE_EDIT_QUEUE_WAIT_S", "900"))
MAX_EDGE = int(os.environ.get("IMAGE_EDIT_MAX_EDGE", "8000"))

_turn = asyncio.Lock()
_state: dict = {"last_edit_at": None, "last": None, "model_loaded": False, "edits": 0}
_started_at = time.time()

app = FastAPI(title="wanly image-edit")


class Angle(BaseModel):
    #: Degrees. Negative = the face turns toward the LEFT EDGE OF THE IMAGE, as LivePortrait's
    #: rotate_yaw < 0 does -- see graph.py. 90 is a full profile.
    yaw: float = Field(0.0, ge=-graph.MAX_YAW, le=graph.MAX_YAW)
    #: Degrees. Positive = chin up.
    pitch: float = Field(0.0, ge=-graph.MAX_PITCH, le=graph.MAX_PITCH)


class EditRequest(BaseModel):
    image: str = Field(min_length=1)
    instruction: str | None = Field(None, max_length=2000)
    angle: Angle | None = None
    seed: int | None = Field(None, ge=0, le=2**48)
    #: < 1 starts from the source's latent; a head turn needs 1.0 (see graph.build_workflow).
    denoise: float = Field(1.0, gt=0.0, le=1.0)
    steps: int | None = Field(None, ge=1, le=30)
    #: Score the result against the source (AuraFace). On by default; the console shows it.
    score: bool = True


def _decode(b64: str):
    from PIL import Image, ImageOps

    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64, validate=False)))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode the image: {e}") from e
    if max(im.size) > MAX_EDGE:
        raise HTTPException(413, f"image is {im.width}x{im.height}; the limit is {MAX_EDGE} px")
    return im


def resolve(req: EditRequest) -> str:
    """The prompt for a request. ValueError -> 422."""
    if req.angle is not None and req.instruction:
        raise ValueError("send an instruction or a head angle, not both")
    if req.angle is not None:
        return graph.angle_prompt(req.angle.yaw, req.angle.pitch)
    if req.instruction:
        return graph.instruction_prompt(req.instruction)
    raise ValueError("nothing to apply: send an instruction or a head angle")


def _vram_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


async def _sample_vram(stop: asyncio.Event, peak: list[int]) -> None:
    """Card-wide peak while the edit runs. In edit mode this service is the card's only
    tenant, so the card's peak IS the edit's -- and nvidia-smi needs no torch in this process."""
    while not stop.is_set():
        v = await asyncio.to_thread(_vram_used_mib)
        if v is not None:
            peak[0] = max(peak[0], v)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass


async def _run_graph(client: httpx.AsyncClient, wf: dict) -> bytes:
    r = await client.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"ComfyUI rejected the graph: {r.text[:600]}")
    pid = r.json()["prompt_id"]
    deadline = time.time() + EDIT_TIMEOUT_S
    while time.time() < deadline:
        h = await client.get(f"{COMFY}/history/{pid}", timeout=30)
        entry = h.json().get(pid) if h.status_code == 200 else None
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise HTTPException(502, f"ComfyUI execution failed: {str(status)[:600]}")
            imgs = [i for i in entry.get("outputs", {}).get("6", {}).get("images", [])
                    if i.get("type") != "temp"]
            if imgs:
                v = await client.get(f"{COMFY}/view", timeout=120, params={
                    "filename": imgs[0]["filename"], "subfolder": imgs[0].get("subfolder", ""),
                    "type": imgs[0].get("type", "output")})
                v.raise_for_status()
                return v.content
            if status.get("completed"):
                raise HTTPException(502, "ComfyUI finished but the save node emitted no image")
        await asyncio.sleep(0.5)
    raise HTTPException(504, f"the edit did not finish within {EDIT_TIMEOUT_S:.0f}s")


@app.post("/edit")
async def edit(req: EditRequest):
    from PIL import Image
    import numpy as np

    try:
        prompt = resolve(req)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    src = await asyncio.to_thread(_decode, req.image)
    w, h = graph.latent_size(src.width, src.height)
    seed = req.seed if req.seed is not None else random.randrange(2**32)
    try:
        await asyncio.wait_for(_turn.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, f"image-edit is busy: another edit held it for {QUEUE_WAIT_S:.0f}s")
    t0 = time.time()
    stop, peak = asyncio.Event(), [0]
    sampler = asyncio.create_task(_sample_vram(stop, peak))
    try:
        # The source goes in at the latent's size: a denoise < 1 needs it (VAEEncode emits a
        # latent at the image's own size), and at 1.0 it costs nothing.
        cond = src if src.size == (w, h) else src.resize((w, h), Image.LANCZOS)
        os.makedirs(os.path.join(WORK_DIR, "in"), exist_ok=True)
        name = f"edit_{uuid.uuid4().hex}.png"
        path = os.path.join(WORK_DIR, "in", name)
        await asyncio.to_thread(cond.save, path, "PNG")
        wf = graph.build_workflow(name, w, h, prompt, seed, steps=req.steps or graph.STEPS,
                                  denoise=req.denoise)
        async with httpx.AsyncClient() as client:
            png = await _run_graph(client, wf)
        t_edit = time.time() - t0
        try:
            os.remove(path)
        except OSError:
            pass
    finally:
        stop.set()
        await sampler
        _turn.release()
    out = Image.open(io.BytesIO(png)).convert("RGB")
    ident = {"aura": None, "reason": "not requested"}
    t1 = time.time()
    if req.score:
        ident = await asyncio.to_thread(identity.score, np.asarray(cond), np.asarray(out))
    t_score = time.time() - t1
    _state.update(last_edit_at=time.time(), model_loaded=True, edits=_state["edits"] + 1,
                  last={"seconds": round(t_edit, 1), "vram_peak_mib": peak[0] or None,
                        "width": out.width, "height": out.height})
    print(f"[image-edit] {'angle ' + str(req.angle.model_dump()) if req.angle else 'instruction'}"
          f" {src.width}x{src.height} -> {w}x{h} seed {seed}"
          f" in {t_edit:.1f}s, vram peak {peak[0]} MiB,"
          f" aura {ident['aura']}", flush=True)
    buf = io.BytesIO()
    out.save(buf, "PNG", compress_level=6)
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "format": "png",
        "width": out.width,
        "height": out.height,
        "prompt": prompt,
        "seed": seed,
        "steps": req.steps or graph.STEPS,
        "denoise": req.denoise,
        "checkpoint": graph.CHECKPOINT,
        "identity": ident,
        "timings_ms": {"edit": round(t_edit * 1000), "score": round(t_score * 1000)},
        "vram_peak_mib": peak[0] or None,
    }


@app.get("/health")
async def health():
    comfy = False
    try:
        async with httpx.AsyncClient() as client:
            comfy = (await client.get(f"{COMFY}/system_stats", timeout=5)).status_code == 200
    except Exception:
        comfy = False
    last = _state["last_edit_at"]
    return {
        "status": "ok" if comfy else "degraded",
        "comfy_ready": comfy,
        "busy": _turn.locked(),
        # The control plane reads this to hand the card back to the render stack once edits
        # stop coming (control.py). Counted from the last edit, or from start-up if none yet.
        "idle_s": 0 if _turn.locked() else round(time.time() - (last or _started_at)),
        "edits": _state["edits"],
        "model_loaded": _state["model_loaded"],
        "last": _state["last"],
        "checkpoint": graph.CHECKPOINT
    }
