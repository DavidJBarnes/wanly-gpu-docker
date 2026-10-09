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

import math
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
#: THE RETRY BORDER (#178), as a fraction of the longest edge, used ONLY when the plain image
#: yields no face. SCRFD at 640 misses some faces that fill the frame: on Joana v3 a 440x440
#: face crop scored det 0.00 as-is and 0.89 with this border, and its embedding then matched
#: the anchor at 0.74 (her set's median was 0.63). Only as a fallback, because padding MOVES
#: embeddings of faces that were already found -- raw vs padded cosine had a median of 0.975
#: but a minimum of 0.48 over the same set -- so padding everything would shift every score.
RETRY_PAD = float(os.environ.get("FACE_DETECT_RETRY_PAD", "0.5"))

#: HEAD-AND-SHOULDERS FRAMING (#187), in multiples of the detected face box's height. The box
#: runs roughly brow to chin, so 0.6 above clears the hairline with a little room, and 1.5 below
#: the chin reaches the collarbone and upper chest. A portrait, 4:5 (width / height) -- the
#: standard portrait shape, and wide enough at that height to keep both shoulders.
HS_ABOVE = float(os.environ.get("FACE_CROP_HS_ABOVE", "0.6"))
HS_BELOW = float(os.environ.get("FACE_CROP_HS_BELOW", "1.5"))
HS_ASPECT = float(os.environ.get("FACE_CROP_HS_ASPECT", "0.8"))

#: PAIR FRAMING (wanly-api#436): both people of a two-person photo, for a COMPOSITION set (the
#: pair's dataset, captioned with both triggers). A head-and-shoulders crop of ONE face there
#: would train under "d@vid, jo@na, 1girl, 1boy" while showing one person -- teaching the pair
#: LoRA that one face is both people, the face bleed of wanly-api#430. So the window is the
#: union of the two largest face boxes with room round each.
#:
#: TIGHT, HAIR TO CHIN (#209). The first cut gave each face head-and-shoulders room (0.6 above,
#: 1.5 below, 0.8 of a face height either side) and on DavidJoana the smaller face went from a
#: median 183 px to 210 px at training size -- 5 of 24 crops reached 250, some shrank. The
#: trainer takes about TRAIN_EDGE^2 of AREA, so a face trains at
#:     face_h * TRAIN_EDGE / sqrt(crop_w * crop_h)
#: and the only lever is the faces' share of the crop: every shoulder in it is face lost. So
#: each face gets its OWN room, in its own size -- PAIR_ABOVE of its height above the box (the
#: box starts about the brow; this clears the crown and the hair on it), PAIR_BELOW under the
#: chin, PAIR_SIDE of its width beyond the outer faces for ears and hair. Two 200 px faces
#: cheek to cheek in a phone photo go from 306 px at training size to 518; two faces three
#: face-heights apart from 190 to 338.
PAIR_ABOVE = float(os.environ.get("FACE_CROP_PAIR_ABOVE", "0.45"))
PAIR_BELOW = float(os.environ.get("FACE_CROP_PAIR_BELOW", "0.25"))
PAIR_SIDE = float(os.environ.get("FACE_CROP_PAIR_SIDE", "0.3"))
#: The aspect (width / height) the tight window is held between. Correcting it GROWS the window
#: and so costs face size (area), which is why the wide end is 2:1 and not the 3:2 of the first
#: cut: two heads a couple of face-heights apart are already 2:1 hair to chin, and holding them
#: to 3:2 adds a third of chest for nothing -- at four face-heights apart that is the difference
#: between 274 px and 237. Past 2:1 is a strip, not a picture of two people.
#: The tall end is the solo portrait's 4:5, for one head above the other.
PAIR_MIN_ASPECT = float(os.environ.get("FACE_CROP_PAIR_MIN_ASPECT", "0.8"))
PAIR_MAX_ASPECT = float(os.environ.get("FACE_CROP_PAIR_MAX_ASPECT", "2.0"))
#: TOO FAR APART (#209): the smaller face must reach this many px AT TRAINING SIZE in the pair
#: window, or no crop is made. It is wanly-api's "small face" line (#431), so a crop under it
#: would come back from the fix still small -- an image added to the set that the very next
#: measure flags again. Judged at the best the window can do (brought to the training area),
#: which distance mostly decides: two faces many face-widths apart leave each a sliver of a crop
#: that is mostly the room between them. Such a photo is reported, not cropped (app.py's
#: `too_far_apart`). 0 turns the rule off.
PAIR_MIN_PX = float(os.environ.get("FACE_CROP_PAIR_MIN_PX", "250"))

#: The framings a crop can ask for. "face" is the original square crop and the default.
FRAMINGS = ("face", "head_shoulders", "pair")

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
    #: Whether Real-ESRGAN enlarged this crop (#206). False for a crop already near the
    #: ceiling, even when an upscale was asked for.
    upscaled: bool = False


def head_shoulders_box(x1: float, y1: float, x2: float, y2: float,
                       w: int, h: int) -> tuple[int, int, int, int]:
    """The head-and-shoulders window (#187) for one face box in a w x h image, as
    (left, top, right, bottom).

    SLID, NOT CLAMPED. The square face crop clamps at an image edge and comes back a little
    narrower; here clamping would cut the shoulders off one side and change the aspect from
    crop to crop. So the window moves back inside the image and keeps its shape. Only when the
    photo is smaller than the window does it shrink -- from the bottom, keeping the head: a
    portrait missing some chest is still a portrait, one missing the hairline is not. Never
    padded with black, for the same reason as the face crop.
    """
    fh = y2 - y1
    top = y1 - HS_ABOVE * fh
    height = fh * (1 + HS_ABOVE + HS_BELOW)
    width = height * HS_ASPECT
    scale = min(1.0, w / width, h / height)
    width, height = width * scale, height * scale
    left = (x1 + x2) / 2 - width / 2
    left = min(max(0.0, left), w - width)
    top = min(max(0.0, top), h - height)
    return (int(left), int(top), min(w, int(left + width)), min(h, int(top + height)))


def pair_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float],
             w: int, h: int) -> tuple[int, int, int, int]:
    """The two-person window (wanly-api#436, tightened in #209) around face boxes `a` and `b`
    (x1, y1, x2, y2) in a w x h image, as (left, top, right, bottom).

    BOTH FACES ARE ALWAYS INSIDE IT. Everything else gives way first: the aspect is corrected by
    GROWING the window (wider for a tall pair, taller -- downward, toward the chests -- for a
    wide one), never by cutting into the union; and when the photo is smaller than the window
    it is clamped to the photo, which still holds the union because the union is in the photo.
    Then it slides back inside, like head_shoulders_box. Never padded with black.
    """
    # Room in each face's OWN size: a small face beside a big one needs less hair room, and
    # sizing both by the bigger one is area the smaller face pays for.
    def room(f):
        fw, fh = f[2] - f[0], f[3] - f[1]
        return (f[0] - PAIR_SIDE * fw, f[1] - PAIR_ABOVE * fh,
                f[2] + PAIR_SIDE * fw, f[3] + PAIR_BELOW * fh)
    ra, rb = room(a), room(b)
    # The union, clamped to the photo: a detector box can poke past the frame at an edge.
    x1, y1 = max(0.0, min(a[0], b[0])), max(0.0, min(a[1], b[1]))
    x2, y2 = min(float(w), max(a[2], b[2])), min(float(h), max(a[3], b[3]))
    left, right = min(ra[0], rb[0]), max(ra[2], rb[2])
    top, bottom = min(ra[1], rb[1]), max(ra[3], rb[3])
    width, height = right - left, bottom - top
    if width / height < PAIR_MIN_ASPECT:          # tall: widen, centred
        grow = PAIR_MIN_ASPECT * height - width
        left, width = left - grow / 2, width + grow
    elif width / height > PAIR_MAX_ASPECT:        # wide: deepen, downward (keep the heads)
        height = width / PAIR_MAX_ASPECT
    width, height = min(width, float(w)), min(height, float(h))
    # Clamping must not drop the union: a window narrower than the image is moved, not cut.
    left = min(max(0.0, left), w - width, x1)
    left = max(left, x2 - width, 0.0)
    top = min(max(0.0, top), h - height, y1)
    top = max(top, y2 - height, 0.0)
    return (int(left), int(top), min(w, int(math.ceil(left + width))),
            min(h, int(math.ceil(top + height))))


def area_edge(w: int, h: int) -> int:
    """The long edge a w x h crop has when brought to the trainer's TRAIN_EDGE^2 area -- what
    a PAIR crop is delivered at (#209). Capping its long edge at MAX_EDGE instead, as the solo
    framings do, would hand the trainer a 2:1 pair at 1024x512: half the area it would have
    taken, and every face in it 1/sqrt(2) the size."""
    if w <= 0 or h <= 0:
        return MAX_EDGE
    return int(TRAIN_EDGE * math.sqrt(max(w, h) / min(w, h)))


def pair_face_px(window: tuple[int, int, int, int], face_h: float) -> float:
    """How tall a face of `face_h` source px trains in `window` (#209), at the best that window
    can do: brought to the training area. The far-apart rule (PAIR_MIN_PX) is judged on this."""
    left, top, right, bottom = window
    return face_h * TRAIN_EDGE / math.sqrt(max(1, right - left) * max(1, bottom - top))


class PairTooFarApart(Exception):
    """framing="pair" on a photo whose two faces are too far apart for any crop of both to
    train the smaller at PAIR_MIN_PX (#209). Raised, not returned empty, so the service can
    say WHY there is no crop -- "fewer than two faces" would send someone looking for a
    detector miss that is not there."""

    def __init__(self, px: float):
        super().__init__(f"smaller face would train at {px:.0f} px (< {PAIR_MIN_PX:.0f})")
        self.px = px


def _decode(image_bytes: bytes):
    import cv2
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def _find(img) -> tuple[list, int]:
    """Every face the detector is confident in, largest first, and the border offset its boxes
    are in (0 unless the retry below ran). Shared by detect() and measure(), so a face that can
    be cropped is exactly a face that can be measured."""
    import cv2

    h, w = img.shape[:2]
    faces = [f for f in _analyser().get(img) if float(f.det_score) >= MIN_DET_SCORE]
    # A FACE THAT FILLS THE FRAME (#178): nothing found, so look again with a border round it.
    # Only here -- see RETRY_PAD for why an image that already works must not be padded. The
    # boxes come back in the padded frame and are shifted into this one below; the crop itself
    # is still cut from the original, so the border never reaches a training image.
    offset = 0
    if not faces and RETRY_PAD > 0:
        offset = int(max(h, w) * RETRY_PAD)
        padded = cv2.copyMakeBorder(img, offset, offset, offset, offset,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = [f for f in _analyser().get(padded) if float(f.det_score) >= MIN_DET_SCORE]
    faces.sort(key=lambda f: -( (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) ))
    return faces, offset


def detect(image_bytes: bytes, framing: str = "face", upscale: bool = False) -> list[Face]:
    """Every face in one image, largest first, cropped to `framing` (see FRAMINGS).

    `upscale` (#206): a crop smaller than the trainer's ceiling is brought up to it with
    Real-ESRGAN (upscale.py) before encoding. Without it a crop of a small face is exactly as
    small to the trainer as the face already was in the photograph.
    """
    if framing not in FRAMINGS:
        raise ValueError(f"framing {framing!r} is not one of {FRAMINGS}")
    import cv2

    img = _decode(image_bytes)
    if img is None:
        return []
    h, w = img.shape[:2]
    faces, offset = _find(img)

    def box(f):
        return tuple(float(v) - offset for v in f.bbox)

    # PAIR: one crop per image, of both people, or nothing. Fewer than two faces means it is
    # not a pair photo to the detector, and a one-face crop is exactly what must not happen
    # here (wanly-api#436); the caller sees it in `no_face`. Two faces too far apart for the
    # smaller to reach PAIR_MIN_PX raise PairTooFarApart (#209). Face 0's detection rides along
    # (its score, its embedding), as the largest face always has.
    if framing == "pair":
        if len(faces) < 2:
            return []
        a, b = box(faces[0]), box(faces[1])
        window = pair_box(a, b, w, h)
        px = pair_face_px(window, min(a[3] - a[1], b[3] - b[1]))
        if PAIR_MIN_PX > 0 and px < PAIR_MIN_PX:
            raise PairTooFarApart(px)
        windows = [(faces[0], window)]
    else:
        windows = [(f, None) for f in faces]

    out: list[Face] = []
    for i, (f, window) in enumerate(windows):
        x1, y1, x2, y2 = box(f)
        if window is not None:
            left, top, right, bottom = window
        elif framing == "head_shoulders":
            left, top, right, bottom = head_shoulders_box(x1, y1, x2, y2, w, h)
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # SQUARE, because a training crop with a varying aspect ratio buckets unpredictably
            # and bucket_no_upscale means the bucket is whatever the image already is.
            side = max(x2 - x1, y2 - y1) * (1 + PAD)
            half = side / 2
            # Clamped to the image rather than padded with black: a black border is a feature
            # the model will happily learn.
            left, top = max(0, int(cx - half)), max(0, int(cy - half))
            right, bottom = min(w, int(cx + half)), min(h, int(cy + half))
        crop = img[top:bottom, left:right]
        if crop.size == 0:
            continue
        # UP FIRST, when asked (#206): a small crop is brought to the ceiling before anything
        # else, and never past it -- the cap below would only throw the extra away.
        #
        # A PAIR crop is sized by AREA, not long edge (#209, area_edge): it is the one framing
        # that is not square-ish, and the trainer caps area.
        h_c, w_c = crop.shape[:2]
        cap = area_edge(w_c, h_c) if framing == "pair" else MAX_EDGE
        upscaled = False
        if upscale:
            from wanly_worker.services.face_crop import upscale as up
            before = crop.shape[:2]
            crop = up.upscale_bgr(crop, cap if framing == "pair" else min(up.TARGET_EDGE, cap))
            upscaled = crop.shape[:2] != before
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
        if longest > cap:
            scale = cap / longest
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
            upscaled=upscaled,
        ))
    return out


#: THE TRAINER'S AREA, as an edge: SDXL and LTX stills both bucket to about TRAIN_EDGE^2 pixels
#: with bucket_no_upscale (lora_trainer/recipe.py), so a larger image is scaled DOWN to that
#: area and a smaller one is left as it is. That is the size a face is learned at, and the only
#: size worth measuring it at -- a 400 px face in a 4000 px photo trains at about 100 px.
TRAIN_EDGE = int(os.environ.get("FACE_TRAIN_EDGE", "1024"))


def train_scale(w: int, h: int) -> float:
    """What the trainer scales a w x h image by: down to the TRAIN_EDGE^2 area, never up."""
    if w <= 0 or h <= 0:
        return 1.0
    return min(1.0, math.sqrt(TRAIN_EDGE * TRAIN_EDGE / (w * h)))


def measure(image_bytes: bytes) -> dict | None:
    """One image's size and every face in it, largest first, with the face height AT TRAINING
    SIZE (#206). None when the bytes are not an image.

    `face_h` is the detector box's height, which runs roughly brow to chin -- the same number
    the v3 audit was taken in (wanly-api#431: 29 of 48 under 250 px). Pose is insightface's
    (pitch, yaw, roll) in degrees, absent as None when the landmark model gave none.
    """
    img = _decode(image_bytes)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = train_scale(w, h)
    faces, offset = _find(img)
    out = []
    for f in faces:
        x1, y1, x2, y2 = [float(v) - offset for v in f.bbox]
        pose = getattr(f, "pose", None)
        pitch, yaw, roll = ([float(v) for v in pose[:3]] if pose is not None
                            else [None, None, None])
        face_h = y2 - y1
        out.append({
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "det_score": round(float(f.det_score), 4),
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "face_h": round(face_h, 1),
            "face_px_at_train": round(face_h * scale, 1),
        })
    return {"width": w, "height": h, "train_scale": round(scale, 4), "faces": out}


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
