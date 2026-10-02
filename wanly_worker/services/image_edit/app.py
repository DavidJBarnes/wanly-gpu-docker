"""The image-edit service's HTTP API: Qwen-Image-Edit "full mode" (wanly-console#548, #569, #570).

    POST /edit     {image: b64, angle?: {yaw, pitch}, expression?, instruction?, face_box?,
                    seed?, denoise?, ...}
                   -> {image: b64 png, width, height, prompt, identity: {aura, reason}, ...}
    POST /faces    {image: b64} -> {width, height, faces: [{index, box, width}], default_index}
    POST /turnaround {image: b64 photo of the person (face + body), outfit, hair?, seed?,
                      crop_padding?, gender?, subject?}
                   -> {candidate: b64 png 1088x1024, sheet: b64 png 1536x1024, previews,
                       face_panel_preview, prompt, seed, model, settings, face_panel,
                       identity, ...}
    GET  /health   comfy up, model loaded, busy, idle seconds, what an edit is waiting for,
                   last edit's timings and VRAM

Called, not claimed, like face-edit. Two deployments of the same code:
  * THE MAIN 3090, EDIT MODE ONLY (registry.MODE_ONLY): Qwen holds ~20 GB, so the render stack
    is stopped (its segment finished first) before this starts. wanly-api owns the queue and
    the mode switch.
  * A STANDING BOX (console#570, the second 3090): SERVICES=image-edit and nothing that renders,
    so it runs in the box's default mode with no switch at all. It shares its card with
    Automatic1111 by the rule in share.py.

WHAT IT IS ASKED TO DO is any mix of a head angle (`angle`), an expression preset
(`expression`, console#569) and free text (`instruction`). The words -- angles and expressions
-- live here, next to the model they were written for, the way face-edit's lexicon does;
wanly-api sends numbers and names. See graph.py for the graph and the framing pins.

WHICH FACE (console#569): `face_box` from POST /faces edits that face alone -- crop, edit, paste
back feathered -- so nobody else in the picture is regenerated. See crop.py for why.

CHARACTER SHEETS (console#582, #585): /turnaround is the sheet recipe -- ONE photo of the person
(full body or most of it, in the outfit) plus outfit and hair words in, ONE candidate per call
(one seed): the generated front/side/back turnaround AND the 1536x1024 sheet already composed
from it, its face panel auto-cropped from the same photo (sheet.py). Composed here, at once,
because this is the only process with the face detector and an image library: wanly-api has
neither, and approving a candidate later must not need the card back. The caller asks for N
seeds as N calls, so each candidate is kept the moment it exists.

THE FACE IS FOUND BY buffalo_l (SCRFD), THE DETECTOR THIS SERVICE ALREADY RUNS, NOT BY THE
MEDIAPIPE NODES the tested UI workflow used. Both were weighed for #585; the deciding case is a
small face in a wide shot (a full-body photo by a tent). ComfyUI's MediaPipeFaceLandmarker
squeezes the whole frame into BlazeFace's 128 px (short range) or 192 px (full range) input, so
a face 2% of a 4000 px frame's width arrives as about 4 px. SCRFD sees 640 px, and this service
looks again at 1280 when 640 finds nothing (identity.DET_SIZE_SMALL_FACES): a 64 px face in a
4000x3000 frame is found that way. It also runs on the CPU BEFORE the card is touched, so a
photo with no findable face is a 422 in a second instead of minutes of GPU time followed by a
useless sheet, and it needs no new model file, no new ComfyUI node and no change to the 3090's
model tree. The crop itself is the workflow's (CropByBBoxes padding, white-padded panel).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import random
import subprocess
import time
import uuid

from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wanly_worker.services.image_edit import crop as cropping
from wanly_worker.services.image_edit import graph, identity, share
from wanly_worker.services.image_edit import sheet as sheets

COMFY_PORT = int(os.environ.get("IMAGE_EDIT_COMFY_PORT", "8286"))
COMFY = f"http://127.0.0.1:{COMFY_PORT}"
#: ComfyUI's input/output dirs for this instance, passed on its command line (service.py).
WORK_DIR = os.environ.get("IMAGE_EDIT_WORK_DIR", "/tmp/image-edit")
#: How long one edit may take, model load included. A cold first edit reads a 28 GB checkpoint.
EDIT_TIMEOUT_S = float(os.environ.get("IMAGE_EDIT_TIMEOUT_S", "900"))
QUEUE_WAIT_S = float(os.environ.get("IMAGE_EDIT_QUEUE_WAIT_S", "900"))
MAX_EDGE = int(os.environ.get("IMAGE_EDIT_MAX_EDGE", "8000"))

_turn = asyncio.Lock()
_state: dict = {"last_edit_at": None, "last": None, "model_loaded": False, "edits": 0,
                "waiting": None, "unloads": 0, "last_unload": None, "scene_yielded": False}
_started_at = time.time()
#: What this service can be asked for beyond #548's angle/instruction, so wanly-api can tell an
#: image that predates them (it would silently ignore the fields) from one that has them.
FEATURES = ["angle", "instruction", "expression", "face_box", "faces", "turnaround",
            "official_2511", "one_photo"]


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    task = asyncio.create_task(_watch()) if share.watching() else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(title="wanly image-edit", lifespan=_lifespan)


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
    #: An expression preset by name (graph.EXPRESSIONS, console#569). Combines with an angle.
    expression: str | None = Field(None, max_length=50)
    #: [x1, y1, x2, y2] in the source's pixels (from POST /faces): edit that face alone.
    face_box: list[float] | None = Field(None, min_length=4, max_length=4)
    seed: int | None = Field(None, ge=0, le=2**48)
    #: < 1 starts from the source's latent; a head turn needs 1.0 (see graph.build_workflow).
    denoise: float = Field(1.0, gt=0.0, le=1.0)
    #: Default graph.STEPS (40 since the switch to the official 2511, #574; v23 took 4).
    steps: int | None = Field(None, ge=1, le=80)
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


class FacesRequest(BaseModel):
    image: str = Field(min_length=1)


class TurnaroundRequest(BaseModel):
    #: ONE photo of the person, full body or most of it, in the outfit (console#585). It is
    #: image 1 of the turnaround -- her body and build come from it -- and the sheet's face
    #: panel is cropped from it. There is no body-words field: the model ignored it (graph.py).
    image: str = Field(min_length=1)
    #: What she wears IN THE PHOTO, described: the turnaround keeps it in all three views.
    outfit: str = Field(min_length=1, max_length=600)
    hair: str | None = Field(None, max_length=300)
    #: Source pixels around the detected face box for the face panel (CropByBBoxes' padding).
    crop_padding: int = Field(sheets.DEFAULT_PADDING, ge=0, le=sheets.MAX_PADDING)
    gender: Literal["female", "male"] = "female"
    #: Who image 1 shows, e.g. "young woman". Default "woman" / "man".
    subject: str | None = Field(None, max_length=60)
    seed: int | None = Field(None, ge=0, le=2**48)
    steps: int | None = Field(None, ge=1, le=80)
    cfg: float | None = Field(None, ge=1.0, le=10.0)
    #: AuraFace of the turnaround (its largest face: the front view) against the face panel.
    score: bool = True


def resolve(req: EditRequest) -> str:
    """The prompt for a request. ValueError -> 422."""
    if req.angle is None and not req.expression and not (req.instruction or "").strip():
        raise ValueError("nothing to apply: send a head angle, an expression or an instruction")
    a = req.angle or Angle()
    return graph.compose_prompt(a.yaw, a.pitch, req.expression, req.instruction)


def _check_box(box: list[float], size: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x1 < size[0] and y1 < size[1]):
        raise HTTPException(422, f"face_box {box} is not a box inside the "
                                 f"{size[0]}x{size[1]} image")
    return [x1, y1, min(x2, size[0]), min(y2, size[1])]


def _others(rgb, box: list[float]) -> list[list[float]]:
    """The OTHER faces, to keep out of the crop. Best effort: a detector that will not load
    means a crop sized by padding alone, not a failed edit."""
    try:
        found = identity.face_boxes(rgb)
    except Exception as e:                      # noqa: BLE001 -- the crop still works without
        print(f"[image-edit] face detection unavailable ({e}); cropping by padding alone",
              flush=True)
        return []
    return [b for b in found if _iou(b, box) < 0.5]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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


@contextlib.asynccontextmanager
async def _card_turn():
    """One job on the card at a time, after its other tenant (console#570), with the VRAM peak
    sampled while it runs. Yields {"waited": s, "peak": [MiB]}; on a clean exit Qwen is marked
    resident -- set before the turn is released, so the watcher never sees "not busy, not
    resident" with the weights still on the card."""
    try:
        await asyncio.wait_for(_turn.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, f"image-edit is busy: another edit held it for {QUEUE_WAIT_S:.0f}s")
    stop, info = asyncio.Event(), {"waited": 0.0, "peak": [0]}
    sampler = None
    try:
        # THE CARD'S OTHER TENANT FIRST (console#570): never start under an A1111 generation,
        # and make room only when Qwen is not already resident. A no-op on the main 3090.
        try:
            info["waited"] = await share.wait_for_a1111(_state)
        except share.A1111Busy as e:
            raise HTTPException(503, str(e)) from e
        # The scene captioner on this card, if any (wanly-console#572): JoyCaption off the card
        # before Qwen loads. Renews the yield on every edit; resumed by _watch once edits stop.
        scene = await share.yield_scene(_state)
        if scene:
            print(f"[image-edit] {scene}", flush=True)
        room = await share.make_room(_state["model_loaded"])
        if room:
            print(f"[image-edit] {room}", flush=True)
        sampler = asyncio.create_task(_sample_vram(stop, info["peak"]))
        yield info
        _state.update(last_edit_at=time.time(), model_loaded=True)
    finally:
        stop.set()
        if sampler is not None:
            await sampler
        _turn.release()


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


@app.post("/faces")
async def faces(req: FacesRequest):
    """The faces an edit can be pointed at, left to right, and the one an edit that names none
    is about: the largest (the one the identity score is taken on). CPU, a second or so."""
    src = await asyncio.to_thread(_decode, req.image)
    try:
        boxes = await asyncio.to_thread(identity.face_boxes, src)
    except Exception as e:                      # noqa: BLE001
        raise HTTPException(503, f"face detection is unavailable: {e}") from e
    default = (max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0])
                   * (boxes[i][3] - boxes[i][1])) if boxes else None)
    return {"width": src.width, "height": src.height, "default_index": default,
            "faces": [{"index": i, "box": b, "width": round(b[2] - b[0], 1)}
                      for i, b in enumerate(boxes)]}


@app.post("/edit")
async def edit(req: EditRequest):
    from PIL import Image
    import numpy as np

    try:
        prompt = resolve(req)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    src = await asyncio.to_thread(_decode, req.image)
    # WHICH FACE: the region to regenerate. The whole frame unless a face was chosen.
    region = (0, 0, src.width, src.height)
    near = 0
    if req.face_box is not None:
        box = _check_box(req.face_box, src.size)
        others = await asyncio.to_thread(_others, src, box)
        region = cropping.crop_box(box, src.size, others)
        near = cropping.neighbours_inside(region, others)
    part = src if region == (0, 0, src.width, src.height) else src.crop(region)
    # A small crop goes to the model scaled up; the result comes back to the crop's size.
    ww, wh = cropping.work_size(*part.size) if req.face_box is not None else part.size
    w, h = graph.latent_size(ww, wh)
    seed = req.seed if req.seed is not None else random.randrange(2**32)
    async with _card_turn() as turn:
        t0 = time.time()
        # The source goes in at the latent's size: a denoise < 1 needs it (VAEEncode emits a
        # latent at the image's own size), and at 1.0 it costs nothing.
        cond = part if part.size == (w, h) else part.resize((w, h), Image.LANCZOS)
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
    waited, peak = turn["waited"], turn["peak"]
    edited = Image.open(io.BytesIO(png)).convert("RGB")
    if part is src:
        out, scored_src, scored_out = edited, cond, edited
    else:
        piece = edited if edited.size == part.size else edited.resize(part.size, Image.LANCZOS)
        out = Image.fromarray(cropping.paste(np.asarray(src), np.asarray(piece), region))
        # Scored on the crop: the chosen face is its subject, whoever is larger elsewhere.
        scored_src, scored_out = part, piece
    ident = {"aura": None, "reason": "not requested"}
    t1 = time.time()
    if req.score:
        ident = await asyncio.to_thread(identity.score, np.asarray(scored_src),
                                        np.asarray(scored_out))
    t_score = time.time() - t1
    _state.update(edits=_state["edits"] + 1,
                  last={"seconds": round(t_edit, 1), "vram_peak_mib": peak[0] or None,
                        "width": out.width, "height": out.height,
                        "waited_for_a1111_s": round(waited, 1)})
    what = ", ".join(x for x in (
        f"angle {req.angle.model_dump()}" if req.angle else "",
        f"expression {req.expression}" if req.expression else "",
        "instruction" if (req.instruction or "").strip() else "") if x)
    print(f"[image-edit] {what} {src.width}x{src.height}"
          f"{f' face crop {region}' if part is not src else ''} -> {w}x{h} seed {seed}"
          f" in {t_edit:.1f}s{f' after {waited:.0f}s waiting for A1111' if waited >= 1 else ''},"
          f" vram peak {peak[0]} MiB, aura {ident['aura']}", flush=True)
    buf = io.BytesIO()
    out.save(buf, "PNG", compress_level=6)
    # A capped JPEG beside the PNG, for the console's "after" pane: wanly-api holds the job and
    # has no image library of its own, and the PNG is ~1-2 MB across the home uplink.
    small = out.copy()
    small.thumbnail((1024, 1024))
    pbuf = io.BytesIO()
    small.save(pbuf, "JPEG", quality=88)
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "preview": base64.b64encode(pbuf.getvalue()).decode(),
        "format": "png",
        "width": out.width,
        "height": out.height,
        "prompt": prompt,
        "seed": seed,
        "steps": req.steps or graph.STEPS,
        "denoise": req.denoise,
        "checkpoint": graph.CHECKPOINT,
        "model": graph.MODEL,
        "cfg": graph.CFG,
        "identity": ident,
        "timings_ms": {"edit": round(t_edit * 1000), "score": round(t_score * 1000),
                       "waited_for_a1111": round(waited * 1000)},
        "vram_peak_mib": peak[0] or None,
        # Echoed so wanly-api can tell a face-scoped edit from an image too old to know the
        # field (which would have edited the whole frame and said nothing).
        "face_box": req.face_box,
        "crop": list(region) if part is not src else None,
        "neighbours_in_crop": near if part is not src else None,
        "expression": req.expression,
    }


def _png_and_preview(im, edge: int = 1024) -> tuple[str, str]:
    """(b64 PNG, b64 capped JPEG) -- the JPEG for the console, which shows candidates side by
    side and should not pull several MB per tile across the home uplink."""
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=6)
    small = im.copy()
    small.thumbnail((edge, edge))
    pbuf = io.BytesIO()
    small.save(pbuf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), base64.b64encode(pbuf.getvalue()).decode()


def _sheet_face(photo) -> tuple[list[float] | None, int]:
    """(the subject's face box, the detector size that found it). Looked for at the usual size
    first, then again at DET_SIZE_SMALL_FACES -- a full-body photo's face is small. Raises
    whatever the detector raises (the caller decides that is not fatal)."""
    for size in (identity.DET_SIZE, identity.DET_SIZE_SMALL_FACES):
        box = sheets.largest(identity.face_boxes(photo, det_size=size))
        if box is not None:
            return box, size
    return None, identity.DET_SIZE_SMALL_FACES


@app.post("/turnaround")
async def turnaround(req: TurnaroundRequest):
    """One character-sheet candidate (console#582, #585): the turnaround for one seed from the
    one photo, and the sheet composed from it with the face panel cropped from that photo. See
    the module docstring and sheet.py."""
    from PIL import Image
    import numpy as np

    try:
        prompt = graph.turnaround_prompt(req.outfit, req.hair, req.gender, req.subject)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    photo = await asyncio.to_thread(_decode, req.image)
    # THE FACE FIRST, ON THE CPU: a photo with no face would make a sheet with no identity in
    # it, and finding that out costs a second here against minutes of card time.
    det_size = None
    try:
        box, det_size = await asyncio.to_thread(_sheet_face, photo)
        detect_note = None
        if box is None:
            raise HTTPException(422, "no face found in the photo: the sheet's face panel is "
                                     "cropped from it -- pick a photo where her face is clear "
                                     "and reasonably large")
    except HTTPException:
        raise
    except Exception as e:                      # noqa: BLE001 -- the sheet still works
        box, detect_note = None, f"face detection unavailable ({e}); face panel centred"
        print(f"[image-edit] {detect_note}", flush=True)
    # The panel does not depend on the seed: cut it before the card, so the preview is ready
    # whatever the turnaround does.
    panel, panel_info = await asyncio.to_thread(sheets.face_panel, photo, box, req.crop_padding)
    seed = req.seed if req.seed is not None else random.randrange(2**32)
    steps, cfg = req.steps or graph.STEPS, req.cfg or graph.CFG
    async with _card_turn() as turn:
        t0 = time.time()
        os.makedirs(os.path.join(WORK_DIR, "in"), exist_ok=True)
        name = f"photo_{uuid.uuid4().hex}.png"
        path = os.path.join(WORK_DIR, "in", name)
        await asyncio.to_thread(photo.save, path, "PNG")
        wf = graph.turnaround_workflow(name, prompt, seed, steps=steps, cfg=cfg)
        async with httpx.AsyncClient() as client:
            png = await _run_graph(client, wf)
        t_gen = time.time() - t0
        try:
            os.remove(path)
        except OSError:
            pass
    body = Image.open(io.BytesIO(png)).convert("RGB")
    composed, _, _ = await asyncio.to_thread(sheets.compose, photo, body, box, req.crop_padding)
    ident = {"aura": None, "reason": "not requested"}
    t1 = time.time()
    if req.score:
        # Against the PANEL, not the whole photo: in a wide shot the face is too small for the
        # scorer's 640 px detector, and the panel is exactly the face the sheet anchors on.
        ident = await asyncio.to_thread(identity.score, np.asarray(panel), np.asarray(body))
    t_score = time.time() - t1
    _state.update(edits=_state["edits"] + 1,
                  last={"seconds": round(t_gen, 1), "vram_peak_mib": turn["peak"][0] or None,
                        "width": body.width, "height": body.height, "kind": "turnaround",
                        "waited_for_a1111_s": round(turn["waited"], 1)})
    waited = turn["waited"]
    print(f"[image-edit] turnaround seed {seed} {steps} steps cfg {cfg:g} in {t_gen:.1f}s"
          f"{f' after {waited:.0f}s waiting for A1111' if waited >= 1 else ''},"
          f" vram peak {turn['peak'][0]} MiB, face panel {panel_info['mode']}"
          f" crop {panel_info['crop']} x{panel_info['scale']}, aura {ident['aura']}", flush=True)
    cand_png, cand_jpg = await asyncio.to_thread(_png_and_preview, body)
    sheet_png, sheet_jpg = await asyncio.to_thread(_png_and_preview, composed, 1536)
    _, panel_jpg = await asyncio.to_thread(_png_and_preview, panel)
    return {
        "candidate": cand_png,
        "candidate_preview": cand_jpg,
        "sheet": sheet_png,
        "sheet_preview": sheet_jpg,
        "face_panel_preview": panel_jpg,
        "format": "png",
        "width": body.width,
        "height": body.height,
        "sheet_width": composed.width,
        "sheet_height": composed.height,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "model": graph.MODEL,
        "files": dict(graph.MODEL_FILES),
        "settings": graph.settings_note(steps, cfg),
        # The panel's provenance: cut from the SAME photo the turnaround was drawn from.
        "face_panel": {"mode": panel_info["mode"], "source": "same_photo", "box": box,
                       "crop": panel_info["crop"], "padding": req.crop_padding,
                       "scale": panel_info["scale"], "detector": "buffalo_l",
                       "det_size": det_size if box is not None else None,
                       "photo_size": [photo.width, photo.height], "note": detect_note},
        "identity": ident,
        "timings_ms": {"generate": round(t_gen * 1000), "score": round(t_score * 1000),
                       "waited_for_a1111": round(turn["waited"] * 1000)},
        "vram_peak_mib": turn["peak"][0] or None,
    }


async def _watch() -> None:
    """Hand the card back (share.py rule 3): unload Qwen when A1111 starts generating or after
    the idle timeout, never while an edit runs. Only started where there is something to do."""
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(share.WATCH_S)
            try:
                if _turn.locked():
                    continue
                idle = time.time() - (_state["last_edit_at"] or _started_at)
                why_scene = share.scene_resume_reason(scene_yielded=_state.get("scene_yielded", False),
                                                      busy=_turn.locked(), idle_s=idle)
                if why_scene:
                    # Qwen off the card first: JoyCaption reloading beside a resident Qwen is
                    # the overlap the yield exists to prevent. Asked even when model_loaded
                    # says no -- an edit that failed part-way may have left weights behind --
                    # and the card is handed back only once ComfyUI confirms.
                    if await share.unload_qwen(COMFY, client):
                        if _state["model_loaded"]:
                            _state.update(unloads=_state.get("unloads", 0) + 1,
                                          last_unload={"at": round(time.time()),
                                                       "why": f"scene-caption's turn ({why_scene})"})
                        _state["model_loaded"] = False
                        if await share.resume_scene(_state, client):
                            print(f"[image-edit] Qwen unloaded; handed the card back to "
                                  f"scene-caption ({why_scene})", flush=True)
                    continue
                if not _state["model_loaded"]:
                    continue
                gen = await share.a1111_generating()
                why = share.unload_reason(resident=_state["model_loaded"], busy=_turn.locked(),
                                          idle_s=time.time() - (_state["last_edit_at"] or _started_at),
                                          a1111_generating=gen)
                if why and not _turn.locked() and await share.unload_qwen(COMFY, client):
                    _state.update(model_loaded=False, unloads=_state.get("unloads", 0) + 1,
                                  last_unload={"at": round(time.time()), "why": why})
                    print(f"[image-edit] unloaded Qwen: {why}", flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:              # noqa: BLE001 -- a watcher must not die
                print(f"[image-edit] watcher: {e}", flush=True)


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
        "checkpoint": graph.CHECKPOINT,
        "model": graph.MODEL,
        "settings": graph.settings_note(),
        "features": FEATURES,
        # Sharing the card (console#570). wanly-api reads `a1111_generating` to say why a job
        # waits ("second 3090 busy (A1111 generating)") and sends the edit only once it is not.
        "shared_with_a1111": share.shared(),
        "a1111_generating": await share.a1111_generating(),
        "waiting": _state.get("waiting"),
        "unload_idle_s": share.UNLOAD_IDLE_S,
        "last_unload": _state.get("last_unload"),
        # The scene captioner on this card (wanly-console#572): whether it is lent to us now.
        "shares_with_scene_caption": share.shares_with_scene(),
        "scene_yielded": _state.get("scene_yielded", False),
    }
