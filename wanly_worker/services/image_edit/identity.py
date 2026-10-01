"""AuraFace identity of an edit against its source (wanly-console#548).

WHY THE SERVICE SCORES ITS OWN OUTPUT. Qwen regenerates the whole frame; LivePortrait warps it.
keyframe-server measured Qwen de-ageing the subject and halving skin texture on every pass, and a
profile is a face the model has to INVENT. So every full-mode result comes back with a number
saying how far it drifted, and the console shows it before anything is saved.

WHY AURAFACE AND NOT ARCFACE. The identity harness (loras/scripts/identity_eval) scores with both
and trusts AuraFace, because inswapper builds its identity from an ArcFace embedding -- ArcFace
rates faceswapped datasets high by construction, and most of this project's datasets are swapped.
AuraFace is a network the swapper never saw. Detection is buffalo_l's (already on the box for
face-crop); the embedding is AuraFace's glintr100, from the same insightface store.

CPU, and never fatal: a missing model or no face in the result is a `reason`, not an error. The
edit already happened; failing it because it could not be graded would throw away the GPU time.
"""
from __future__ import annotations

import os
import threading

ROOT = os.environ.get("INSIGHTFACE_ROOT", "/root/.insightface")
AURA_PATH = os.environ.get("IMAGE_EDIT_AURAFACE",
                           os.path.join(ROOT, "models", "auraface", "glintr100.onnx"))

_lock = threading.Lock()
_models: tuple | None = None
_load_error: str | None = None
_dets: dict[int, object] = {}
#: The detector's input size. buffalo_l's SCRFD sees the photo shrunk to fit this square.
DET_SIZE = 640
#: The second look for a photo where DET_SIZE found nothing (console#585): a full-body shot's
#: face is small, and shrinking a 4000 px frame to 640 leaves it too few pixels -- a 64 px face
#: in a 4000x3000 frame is missed at 640 and found at 1280. Not the default: at 1280 a face
#: filling a close-up is larger than SCRFD's biggest anchors.
DET_SIZE_SMALL_FACES = 1280
#: Faces narrower than this are not offered for a face choice (console#569): too small to
#: edit on their own, and usually a background face or a detector guess.
MIN_FACE_PX = 30


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def detector(det_size: int = DET_SIZE):
    """buffalo_l detection alone, CPU, at `det_size`. Separate from AuraFace so the face list
    (console#569) works on a box whose AuraFace download failed -- a missing score is a reason,
    a missing face picker would be a silently wrong edit."""
    with _lock:
        if det_size not in _dets:
            from insightface.app import FaceAnalysis

            det = FaceAnalysis(name="buffalo_l", root=ROOT, providers=["CPUExecutionProvider"],
                               allowed_modules=["detection"])
            det.prepare(ctx_id=-1, det_size=(det_size, det_size))
            _dets[det_size] = det
        return _dets[det_size]


def face_boxes(rgb, det_size: int = DET_SIZE) -> list[list[float]]:
    """Every face's [x1, y1, x2, y2] in `rgb`'s pixels, LEFT TO RIGHT, narrow ones dropped."""
    import numpy as np

    faces = detector(det_size).get(np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]))
    boxes = [[round(float(v), 1) for v in f.bbox[:4]] for f in faces]
    boxes = [b for b in boxes if b[2] - b[0] >= MIN_FACE_PX]
    return sorted(boxes, key=lambda b: (b[0], b[2]))


def _load():
    global _models, _load_error
    if _models is not None:
        return _models
    if not os.path.isfile(AURA_PATH):
        _load_error = (f"AuraFace model not found at {AURA_PATH} (fal/AuraFace-v1 "
                       f"glintr100.onnx; download_models.sh --image-edit fetches it)")
        raise FileNotFoundError(_load_error)
    det = detector()
    with _lock:
        if _models is not None:
            return _models
        from insightface.model_zoo import get_model

        aura = get_model(AURA_PATH, providers=["CPUExecutionProvider"])
        aura.prepare(ctx_id=-1)
        _models = (det, aura)
        _load_error = None
        return _models


def _embed(det, aura, bgr):
    faces = det.get(bgr)
    if not faces:
        return None
    # The largest face: the subject of an edit fills the frame; a background face is not it.
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return aura.get(bgr, f)


def score(source_rgb, result_rgb) -> dict:
    """{"aura": cos | None, "reason": str | None}. Never raises."""
    try:
        det, aura = _load()
    except Exception as e:                      # noqa: BLE001 -- reported, not raised
        return {"aura": None, "reason": _load_error or f"could not load AuraFace: {e}"}
    try:
        import numpy as np

        src = _embed(det, aura, np.ascontiguousarray(source_rgb[:, :, ::-1]))
        if src is None:
            return {"aura": None, "reason": "no face found in the source"}
        out = _embed(det, aura, np.ascontiguousarray(result_rgb[:, :, ::-1]))
        if out is None:
            # Common for a full profile: the detector is trained mostly on faces it can see
            # both eyes of. Said plainly, so "no number" is not read as "identity lost".
            return {"aura": None, "reason": "no face detected in the result (a full profile "
                                            "often defeats the detector)"}
        return {"aura": round(cosine(src, out), 4), "reason": None}
    except Exception as e:                      # noqa: BLE001
        return {"aura": None, "reason": f"scoring failed: {e}"}
