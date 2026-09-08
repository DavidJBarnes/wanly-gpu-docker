"""The face-crop service's HTTP API.

Short, synchronous work, so it is called rather than claimed -- the same shape as joycaption,
which wanly-api calls inline and waits for. A couple of seconds per image on CPU.
"""
from __future__ import annotations

import base64

from fastapi import FastAPI
from pydantic import BaseModel, Field

from wanly_worker.services.face_crop import detect as fd

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


class CropResponse(BaseModel):
    faces: list[CropFace]
    #: Images the detector found nothing in. Named rather than counted, because "10 of 38 had no
    #: face" is the number that tells you the source set is wrong.
    no_face: list[int]
    cos_floor: float


@app.get("/health")
async def health():
    # model_loaded, separately from status. The service answers on this port long before
    # buffalo_l is in memory, so a bare 200 said "ready" while the first crop was still going to
    # pay for a 300 MB download -- and that is exactly the gap that produced a `face-crop
    # unreachable` in the console with the service sitting there healthy.
    return {"status": "ok", "cos_floor": fd.COS_FLOOR, "model_loaded": fd.is_loaded()}


@app.post("/embed")
async def embed(req: CropRequest):
    """Embeddings for a reference set, so the caller can build a mean once and reuse it."""
    out = [fd.embed(base64.b64decode(b)) for b in req.images]
    return {"embeddings": out, "mean": fd.reference_mean(out)}


@app.post("/crop", response_model=CropResponse)
async def crop(req: CropRequest):
    mean = fd.reference_mean(req.reference) if req.reference else []
    faces: list[CropFace] = []
    no_face: list[int] = []
    for i, b64 in enumerate(req.images):
        found = fd.detect(base64.b64decode(b64))
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
            ))
    return CropResponse(faces=faces, no_face=no_face, cos_floor=fd.COS_FLOOR)
