"""Face detection and cropping, ported from ltx-char-loras/scripts/crop_faces.py.

WHY THIS IS A SERVICE. Cropping was a laptop script that rsync'd images to the 3090 because
insightface is not installed locally. That works for someone at a terminal and cannot be reached
from the console at all -- so a dataset uploaded in the UI had no way to become face crops, and
the whole documented pipeline (raw -> crop -> cull -> gate -> train) stopped at step one.

CPU, DELIBERATELY. buffalo_l detection on a handful of images is a second or two per image, and
the GPU on the box that has insightface is usually busy with a render or a training run. Taking
it for this would be the worst possible trade.

TWO THINGS THIS RETURNS THAT A CROPPER ALONE WOULD NOT:

  det_score and bbox     so the caller can tell a confident face from a guess
  the normed embedding   which is what makes the identity gate possible without a second pass

The gate matters more than the crop. Hand-culling let two different wrong people into p@y's
training set -- one of them into the set that had already been culled by eye -- and the only
thing that caught it was scoring against a reference. A cropper that does not also return
embeddings leaves that check to be bolted on later, which is how it got skipped the first time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

#: buffalo_l's same-person floor. On p@y the raw pool produced a selected minimum of -0.042 --
#: a different person entirely, in a set that looked fine by eye.
COS_FLOOR = float(os.environ.get("FACE_COS_FLOOR", "0.4"))
#: Padding around the detected box, as a fraction of its size. The crops that trained well were
#: not tight to the jaw; a face needs some hair and chin to be recognisable.
PAD = float(os.environ.get("FACE_CROP_PAD", "0.4"))
#: Below this the detector is guessing. Kept low, because the cull is a human's job and a
#: borderline face is worth showing rather than silently dropping.
MIN_DET_SCORE = float(os.environ.get("FACE_MIN_DET_SCORE", "0.5"))
#: Longest edge of a returned crop. MATCHES THE TRAINER'S `resolution`, which is a ceiling with
#: bucket_no_upscale -- so anything larger is downscaled during latent caching anyway, and
#: shipping it across the internet first buys nothing.
MAX_EDGE = int(os.environ.get("FACE_CROP_MAX_EDGE", "1024"))
#: JPEG quality. The sources are already JPEG out of a phone, so a q95 re-encode is invisible,
#: and PNG on photographic content is an order of magnitude larger for no gain a trainer can
#: use.
JPEG_QUALITY = int(os.environ.get("FACE_CROP_JPEG_QUALITY", "95"))

_app = None


def is_loaded() -> bool:
    """Whether buffalo_l is in memory. Reported by /health so "up" and "ready to crop" are not
    conflated -- they were, and the difference is a 300 MB download."""
    return _app is not None


def _analyser():
    """One FaceAnalysis for the process. Loading buffalo_l takes seconds and it is stateless."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        a = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        a.prepare(ctx_id=-1, det_size=(640, 640))
        _app = a
    return _app


#: What `Face.png` is encoded as. A field rather than a constant in the caller, so wanly-api
#: names the object it writes correctly without having to know this module's defaults.
IMAGE_FORMAT = "jpeg"


@dataclass
class Face:
    #: Square crop, encoded as IMAGE_FORMAT. The attribute keeps its old name so nothing that
    #: reads it breaks on the same deploy; the format is what changed, not the meaning.
    png: bytes
    width: int
    det_score: float
    yaw: float
    #: L2-normalised, so a dot product with another is the cosine similarity.
    embedding: list[float]
    #: 0 is the LARGEST face in the image, not necessarily the right one -- in a two-person
    #: photo the biggest face may be the wrong woman. That is the single most likely thing to go
    #: wrong here, and it did on p@y: an 887px crop of the wrong person survived a by-eye pass.
    index: int


def detect(image_bytes: bytes) -> list[Face]:
    """Every face in one image, largest first."""
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    faces = [f for f in _analyser().get(img) if float(f.det_score) >= MIN_DET_SCORE]
    faces.sort(key=lambda f: -( (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) ))

    out: list[Face] = []
    h, w = img.shape[:2]
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # SQUARE, because a training crop with a varying aspect ratio buckets unpredictably and
        # bucket_no_upscale means the bucket is whatever the image already is.
        side = max(x2 - x1, y2 - y1) * (1 + PAD)
        half = side / 2
        # Clamped to the image rather than padded with black: a black border is a feature the
        # model will happily learn.
        left, top = max(0, int(cx - half)), max(0, int(cy - half))
        right, bottom = min(w, int(cx + half)), min(h, int(cy + half))
        crop = img[top:bottom, left:right]
        if crop.size == 0:
            continue
        # DOWNSCALE, THEN JPEG. Both matter, and the reason is the wire, not the disk.
        #
        # A face box with PAD out of a 5 MB phone photo is ~1500 px square, and lossless PNG of
        # that is several MB. Fourteen group photos came to a ~150 MB response, which is minutes
        # on a home uplink -- so wanly-api's 300s read timed out while this service had already
        # logged 200 OK. The console said "cropping failed" and the logs on both sides looked
        # fine.
        #
        # Nothing is lost by capping at MAX_EDGE: the trainer's `resolution` is a ceiling with
        # bucket_no_upscale, so a larger crop is downscaled during latent caching regardless.
        h_c, w_c = crop.shape[:2]
        longest = max(h_c, w_c)
        if longest > MAX_EDGE:
            scale = MAX_EDGE / longest
            crop = cv2.resize(crop, (max(1, int(w_c * scale)), max(1, int(h_c * scale))),
                              interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        emb = getattr(f, "normed_embedding", None)
        out.append(Face(
            png=buf.tobytes(),
            width=right - left,
            det_score=float(f.det_score),
            yaw=float(getattr(f, "pose", [0, 0, 0])[1]) if getattr(f, "pose", None) is not None else 0.0,
            embedding=[float(x) for x in emb] if emb is not None else [],
            index=i,
        ))
    return out


def embed(image_bytes: bytes) -> list[float]:
    """The largest face's embedding, or []. Used to build a reference mean."""
    faces = detect(image_bytes)
    return faces[0].embedding if faces else []


def reference_mean(embeddings: list[list[float]]) -> list[float]:
    """The mean of a reference set, renormalised.

    Scored against the MEAN rather than any single reference image, because one reference is one
    lighting and one angle -- curate.py has always done it this way and it is what makes the
    0.4 floor meaningful.
    """
    usable = [np.asarray(e, dtype=np.float32) for e in embeddings if e]
    if not usable:
        return []
    mu = np.mean(usable, axis=0)
    n = float(np.linalg.norm(mu))
    return (mu / n).tolist() if n else []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -2.0
    return float(np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32))
