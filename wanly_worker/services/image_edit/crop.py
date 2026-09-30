"""Editing ONE face of a multi-face image with Qwen: crop around it, edit, paste back (console#569).

WHY CROP/PASTE AND NOT "THE PERSON ON THE LEFT". Qwen at denoise 1 regenerates the WHOLE frame
it is given. Pointing it at one person with words still redraws everyone else -- and every
redraw drifts identity (keyframe-server measured Qwen de-ageing the subject and halving skin
texture on every pass). A crop confines the redraw: every pixel outside it is the source's own,
byte for byte, so the person nobody asked to edit is untouched by construction, not by
instruction-following. It also makes the instruction unambiguous ("the person" is the one in
the crop) and scores identity against the right face. The face-edit service (#553) chose the
same shape for the same reason.

THE COST is a seam where the regenerated crop meets the original. Qwen keeps tone and framing
well under the framing pin, but not to the pixel, so the paste is FEATHERED: full edit inside,
a linear ramp to the source over the outer FEATHER of the crop's short edge. An edge of the
crop that is an edge of the image needs no ramp and gets none.

THE CROP is generous -- a head turn to profile needs room, and hair, neck and shoulders anchor
the model -- and it is pulled in, where it can be, so a neighbouring face stays OUT of it: a
neighbour inside the crop is a neighbour Qwen may edit too. Small crops are scaled UP for the
model (a 300 px face crop is too few pixels for it) and the result scaled back to the crop's
own size, so the paste lands pixel for pixel.

Pure geometry and array blending: no model, no ComfyUI -- tested in CI.
"""
from __future__ import annotations

import math

Box = list[float]

#: Room around the chosen face, in face widths (sides) and face heights (above, below). Below
#: is the most: a turned head moves the chin and neck, and the shoulders anchor the pose.
PAD_SIDE = 1.1
PAD_UP = 0.8
PAD_DOWN = 1.5
#: However close a neighbour is, the chosen face keeps at least this much room each side.
MIN_PAD = 0.15
#: The feather, as a fraction of the crop's short edge.
FEATHER = 0.06
#: A crop is edited with its short edge at least this long (never more than MAX_MP total --
#: graph.latent_size caps it). Below ~700 px the model loses the face's detail.
MIN_EDGE = 768


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def crop_box(face: Box, size: tuple[int, int], others: list[Box] = ()) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the region to edit, in source pixels, clamped to the image.

    `others` are the other faces in the image. A neighbour whose box overlaps the region is
    pushed out by moving the nearer side edge to it -- but never closer to the chosen face than
    MIN_PAD face widths. Faces are side by side far more often than stacked, so only the side
    edges move; the top and bottom are the padding's.
    """
    w, h = size
    x1, y1, x2, y2 = face
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    left, right = x1 - PAD_SIDE * fw, x2 + PAD_SIDE * fw
    top, bottom = y1 - PAD_UP * fh, y2 + PAD_DOWN * fh
    cx = (x1 + x2) / 2
    for o in others:
        if o[2] <= left or o[0] >= right or o[3] <= top or o[1] >= bottom:
            continue                                        # no overlap
        ox, _ = _centre(o)
        if ox < cx:
            left = max(left, min(o[2], x1 - MIN_PAD * fw))
        else:
            right = min(right, max(o[0], x2 + MIN_PAD * fw))
    x0, t = max(0, math.floor(left)), max(0, math.floor(top))
    r, b = min(w, math.ceil(right)), min(h, math.ceil(bottom))
    return x0, t, max(r, x0 + 1), max(b, t + 1)


def neighbours_inside(crop: tuple[int, int, int, int], others: list[Box],
                      share: float = 0.25) -> int:
    """How many other faces still have at least `share` of their box inside the crop --
    reported, so a result where Qwen may have touched a second person says so."""
    x0, t, r, b = crop
    n = 0
    for o in others:
        ix = max(0.0, min(r, o[2]) - max(x0, o[0]))
        iy = max(0.0, min(b, o[3]) - max(t, o[1]))
        area = max(1e-6, (o[2] - o[0]) * (o[3] - o[1]))
        n += (ix * iy) / area >= share
    return n


def work_size(width: int, height: int, min_edge: int = MIN_EDGE) -> tuple[int, int]:
    """The size a crop is sent to the model at: scaled UP so its short edge is at least
    `min_edge` (aspect kept), never down -- graph.latent_size does the capping."""
    short = min(width, height)
    if short >= min_edge:
        return width, height
    s = min_edge / short
    return round(width * s), round(height * s)


def feather_mask(crop: tuple[int, int, int, int], size: tuple[int, int], feather: float = FEATHER):
    """HxW float32 alpha for the crop: 1 inside, a linear ramp to 0 over the outer band on each
    side that is NOT an image edge (there is no source beyond an image edge to blend into)."""
    import numpy as np

    x0, t, r, b = crop
    w, h = r - x0, b - t
    band = max(1, round(min(w, h) * feather))
    xs = np.arange(w, dtype="float32")
    ys = np.arange(h, dtype="float32")
    ax = np.ones(w, dtype="float32")
    ay = np.ones(h, dtype="float32")
    if x0 > 0:
        ax = np.minimum(ax, (xs + 1) / band)
    if r < size[0]:
        ax = np.minimum(ax, (w - xs) / band)
    if t > 0:
        ay = np.minimum(ay, (ys + 1) / band)
    if b < size[1]:
        ay = np.minimum(ay, (h - ys) / band)
    return np.clip(np.outer(ay, ax), 0.0, 1.0)


def paste(source, edited, crop: tuple[int, int, int, int]):
    """The source with `edited` (the crop's own size, RGB uint8) blended in through the feather.
    Everything outside the crop is the source's, unchanged. Returns a new HxWx3 uint8 array."""
    import numpy as np

    src = np.asarray(source)
    x0, t, r, b = crop
    if edited.shape[:2] != (b - t, r - x0):
        raise ValueError(f"edited crop is {edited.shape[1]}x{edited.shape[0]}, "
                         f"the crop is {r - x0}x{b - t}")
    a = feather_mask(crop, (src.shape[1], src.shape[0]))[:, :, None]
    out = src.copy()
    region = src[t:b, x0:r].astype("float32")
    out[t:b, x0:r] = np.clip(region * (1 - a) + edited.astype("float32") * a + 0.5, 0, 255) \
        .astype("uint8")
    return out
