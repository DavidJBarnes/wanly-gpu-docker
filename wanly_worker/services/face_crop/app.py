"""The face-crop service's HTTP API.

Short, synchronous work, so it is called rather than claimed -- the same shape as joycaption,
which wanly-api calls inline and waits for. A couple of seconds per image on CPU.
"""
from __future__ import annotations

import asyncio
import base64

from fastapi import FastAPI, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from wanly_worker.services.face_crop import detect as fd
from wanly_worker.services.face_crop import upscale as up

#: What this build can do, on /health (#206). wanly-api checks it before starting a job that
#: needs one of them, rather than discovering half way through that an older service ignored a
#: field it did not know -- which is what an older service does with `upscale`.
FEATURES = ["crop", "embed", "head_shoulders", "measure"]


def features() -> list[str]:
    """FEATURES, plus "upscale" only when the weights are actually here. The code is fetched
    fresh at every boot (fetch_engine.sh) but the weights come with the IMAGE, so this code can
    run in an older image that has none -- and must not advertise what it cannot do there."""
    return FEATURES + (["upscale"] if up.available() else [])

app = FastAPI(title="wanly face-crop")


@app.on_event("startup")
async def _warm() -> None:
    """Load buffalo_l BEFORE anything can ask for a crop.

    insightface fetches the ~300 MB model on first use, and first use used to be inside a
    request. On a box that had never run this, the first crop therefore paid for a download
    inside wanly-api's HTTP call and blew its timeout -- which surfaces as `face-crop
    unreachable` in the console, pointing at the network rather than at a model that simply was
    not there yet. The second attempt would have worked, which is the worst kind of bug to be
    told about.

    In a thread: it is blocking and can take a minute, and the event loop must stay free or the
    supervisor's readiness probe times out and kills a service that is doing exactly what it
    should be.
    """
    import asyncio
    from wanly_worker.services.face_crop import detect as fd

    async def load():
        try:
            await asyncio.to_thread(fd._analyser)
            print("[face-crop] buffalo_l loaded", flush=True)
        except Exception as e:
            # Not fatal. The service still answers /health, and a crop will retry the load and
            # report the real error -- better than a container that will not start.
            print(f"[face-crop] could not preload buffalo_l: {e}", flush=True)

    asyncio.create_task(load())


class CropRequest(BaseModel):
    #: base64 image bytes. Base64 rather than multipart because the caller is another service
    #: holding bytes it already fetched, not a browser with a file handle.
    images: list[str] = Field(min_length=1, max_length=200)
    #: Reference embeddings to score against. Optional: with none, crops come back unscored and
    #: the caller does the culling by eye, which is exactly what the old flow did.
    reference: list[list[float]] = Field(default_factory=list)
    #: Only the largest face per image. The default, because a dataset of one person wants one
    #: face per photo -- but `false` is honest about two-person photos rather than silently
    #: picking the bigger woman.
    largest_only: bool = True
    #: "face" (the square crop, and the default) or "head_shoulders" (#187): a 4:5 portrait
    #: from just above the hairline to the upper chest.
    framing: Literal["face", "head_shoulders"] = "face"
    #: Real-ESRGAN each crop up to the trainer's ceiling (#206) when it is smaller than that.
    upscale: bool = False


class ImagesRequest(BaseModel):
    """/measure and /upscale: just the images, base64, as for /crop."""
    images: list[str] = Field(min_length=1, max_length=200)


class CropFace(BaseModel):
    source_index: int
    face_index: int
    #: The crop's bytes, base64. The field keeps its historical name so a wanly-api that has not
    #: been redeployed yet still reads it; `format` is what says how to decode and name it.
    png_b64: str
    #: "jpeg" now, "png" before. A caller that does not know this field should assume png, which
    #: is what the old contract was.
    format: str = "jpeg"
    width: int
    det_score: float
    yaw: float
    cos: float | None = None
    embedding: list[float] = Field(default_factory=list)
    #: Real-ESRGAN enlarged this crop (#206). False when it was already near the ceiling.
    upscaled: bool = False


class CropResponse(BaseModel):
    faces: list[CropFace]
    #: Images the detector found nothing in. Named rather than counted, because "10 of 38 had no
    #: face" is the number that tells you the source set is wrong.
    no_face: list[int]
    cos_floor: float
    #: The framing these crops were cut to, echoed (#187). A service that predates framing
    #: ignores the request field and sends face crops; without this the caller could not tell.
    framing: str = "face"
    #: Whether `upscale` was honoured, echoed for the same reason as `framing`: a service that
    #: predates it ignores the field, and the caller must not store plain crops as upscaled ones.
    upscale: bool = False


@app.get("/health")
async def health():
    # model_loaded, separately from status. The service answers on this port long before
    # buffalo_l is in memory, so a bare 200 said "ready" while the first crop was still going to
    # pay for a 300 MB download -- and that is exactly the gap that produced a `face-crop
    # unreachable` in the console with the service sitting there healthy.
    return {"status": "ok", "cos_floor": fd.COS_FLOOR, "model_loaded": fd.is_loaded(),
            "features": features(), "upscale_ready": up.available(),
            "train_edge": fd.TRAIN_EDGE}


@app.post("/embed")
async def embed(req: CropRequest):
    """Embeddings for a reference set, so the caller can build a mean once and reuse it."""
    out = [fd.embed(base64.b64decode(b)) for b in req.images]
    return {"embeddings": out, "mean": fd.reference_mean(out)}


@app.post("/crop", response_model=CropResponse)
async def crop(req: CropRequest):
    if req.upscale and not up.available():
        raise HTTPException(status_code=503, detail="upscale weights are missing from this image")
    # In a thread: an upscaled batch is seconds per image, and a blocked event loop stops
    # /health answering -- which reads as a dead service to everything that polls it.
    return await asyncio.to_thread(_crop, req)


def _crop(req: CropRequest) -> CropResponse:
    mean = fd.reference_mean(req.reference) if req.reference else []
    faces: list[CropFace] = []
    no_face: list[int] = []
    for i, b64 in enumerate(req.images):
        found = fd.detect(base64.b64decode(b64), req.framing, upscale=req.upscale)
        if not found:
            no_face.append(i)
            continue
        for f in (found[:1] if req.largest_only else found):
            faces.append(CropFace(
                source_index=i, face_index=f.index,
                png_b64=base64.b64encode(f.png).decode(),
                format=fd.IMAGE_FORMAT,
                width=f.width, det_score=f.det_score, yaw=f.yaw,
                cos=fd.cosine(f.embedding, mean) if mean else None,
                embedding=f.embedding,
                upscaled=f.upscaled,
            ))
    return CropResponse(faces=faces, no_face=no_face, cos_floor=fd.COS_FLOOR,
                        framing=req.framing, upscale=req.upscale)


@app.post("/measure")
async def measure(req: ImagesRequest):
    """Each image's size and faces, with the face height AT TRAINING SIZE (#206).

    `results` is parallel to `images`; null for bytes that are not an image. Every face is
    returned, largest first -- which one is the subject is the caller's call, as with /crop.
    """
    results = await asyncio.to_thread(
        lambda: [fd.measure(base64.b64decode(b)) for b in req.images])
    return {"results": results, "train_edge": fd.TRAIN_EDGE}


@app.post("/upscale")
async def upscale(req: ImagesRequest):
    """Whole images brought up to the trainer's ceiling with Real-ESRGAN (#206) -- the tiny
    close-ups, where there is nothing to crop and the whole frame is already the face.

    Parallel to `images`. An image already near the ceiling comes back `upscaled: false` with
    no bytes: the caller keeps its original rather than storing a pointless re-encode.
    """
    if not up.available():
        raise HTTPException(status_code=503, detail="upscale weights are missing from this image")
    return {"images": await asyncio.to_thread(lambda: [_upscale_one(b) for b in req.images]),
            "format": fd.IMAGE_FORMAT, "upscale": True}


def _upscale_one(b64: str) -> dict | None:
    import cv2
    img = fd._decode(base64.b64decode(b64))
    if img is None:
        return None
    h, w = img.shape[:2]
    out = up.upscale_bgr(img, min(up.TARGET_EDGE, fd.MAX_EDGE))
    if out is img:
        return {"upscaled": False, "b64": None, "source_width": w, "source_height": h,
                "width": w, "height": h}
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), fd.JPEG_QUALITY])
    if not ok:
        return None
    return {"upscaled": True, "b64": base64.b64encode(buf.tobytes()).decode(),
            "source_width": w, "source_height": h,
            "width": out.shape[1], "height": out.shape[0]}
