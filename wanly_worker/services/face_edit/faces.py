"""Choosing WHICH face an edit applies to (wanly-console#553).

The node edits exactly one face and chooses it itself: nodes.py `detect_face` takes the box
whose horizontal centre is closest to the frame's (strict `<`, so a tie goes to whichever the
detector listed first) and skips boxes narrower than 30 px. There is no argument to point it at
another face, and the node's code stays unchanged (engine.py), so choosing a face means choosing
the IMAGE it sees: a crop in which the chosen face is the one the node's rule picks. The edit
runs on that crop and the result is pasted back at the same place, so every pixel outside the
crop is the source's own, byte for byte.

Pure geometry on plain lists -- no torch, no detector -- so every rule here is testable in CI.
The engine checks the result against the real detector on the crop (`isolates`) before it warps
anything, because a crop can reveal a face the full-frame detection missed, and a guess that is
wrong must be a 422, not an edit of the wrong person.
"""
from __future__ import annotations

import math

#: The node's detector skips boxes narrower than this (nodes.py detect_face).
MIN_FACE_PX = 30
#: How much closer to the crop's centre the chosen face must be than any competitor, as a
#: fraction of its width (never under MIN_MARGIN_PX). The detector re-runs on the crop and its
#: boxes move by a few pixels when the frame around them changes; a crop that wins by a hair on
#: the full-frame boxes can lose on the crop's own.
MARGIN = 0.1
MIN_MARGIN_PX = 4.0
#: A face_box from /faces names the detected face it overlaps at least this much.
MATCH_IOU = 0.5


class FacePickError(ValueError):
    """The requested face cannot be edited: out of range, matches nothing, or cannot be
    isolated from its neighbours. A 422 with this message."""


Box = list[float]


def valid(raw) -> list[Box]:
    """The detector's boxes the node would consider, in the detector's order."""
    return [[float(v) for v in b[:4]] for b in raw if (b[2] - b[0]) >= MIN_FACE_PX]


def node_pick(raw, width: int) -> int | None:
    """The index into `valid(raw)` the node picks unaided -- its loop, verbatim, including
    the strict `<` that gives a tie to the earlier box."""
    cx, best, best_diff = width / 2, None, width
    for i, (x1, _y1, x2, _y2) in enumerate(valid(raw)):
        diff = abs(cx - (x1 + (x2 - x1) / 2))
        if diff < best_diff:
            best, best_diff = i, diff
    return best


def analyse(raw, width: int) -> tuple[list[Box], int | None]:
    """(the valid faces left to right, the index of the one the node would pick)."""
    boxes = valid(raw)
    pick = node_pick(raw, width)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0], boxes[i][2]))
    return [boxes[i] for i in order], (order.index(pick) if pick is not None else None)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def choose(boxes: list[Box], face_index: int | None, face_box: Box | None) -> int:
    """Which of `boxes` (left to right) the caller means. The box wins over the index when
    both are sent: it names a face by where it is, which survives the list changing."""
    if face_box is not None:
        scored = [(iou(b, face_box), i) for i, b in enumerate(boxes)]
        best, i = max(scored, default=(0.0, None))
        if i is None or best < MATCH_IOU:
            raise FacePickError(f"face_box {[round(v) for v in face_box]} matches none of the "
                                f"{len(boxes)} faces detected in this image (best overlap "
                                f"{best:.2f}); ask /faces again for this image")
        return i
    if not 0 <= face_index < len(boxes):
        raise FacePickError(f"face_index {face_index} is out of range: {len(boxes)} "
                            f"face{'s' if len(boxes) != 1 else ''} detected "
                            f"(0-{len(boxes) - 1})")
    return face_index


def node_region(box: Box, w: int, h: int, pad: float) -> list[int]:
    """The square the node warps around `box` in a w x h image (detect_face, sort=True): side
    max(bw, bh) * crop_factor, centred on the face and shifted -- not shrunk -- to fit."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * pad
    kx, ky = int(x1 + bw / 2), int(y1 + bh / 2)
    nx1, nx2 = int(kx - side / 2), int(kx + side / 2)
    ny1, ny2 = int(ky - side / 2), int(ky + side / 2)
    if nx1 < 0:
        nx2, nx1 = nx2 - nx1, 0
    elif w < nx2:
        nx1, nx2 = nx1 - (nx2 - w), w
        if nx1 < 0:
            nx2, nx1 = nx2 - nx1, 0
    if ny1 < 0:
        ny2, ny1 = ny2 - ny1, 0
    elif h < ny2:
        ny1, ny2 = ny1 - (ny2 - h), h
        if ny1 < 0:
            ny2, ny1 = ny2 - ny1, 0
    return [nx1, ny1, nx2, ny2]


def _isolated(boxes, idx, x1, y1, x2, y2) -> float | None:
    """How much closer the chosen face is to the crop's horizontal centre than the nearest
    competitor (inf with none), or None if some face is as close or closer.

    A face the crop cuts through counts by the part inside it: that part is what the detector
    will see, and its centre sits nearer the crop's middle than the whole face's does -- the
    pessimistic reading, which is the one to plan on.
    """
    m = (x1 + x2) / 2
    f = boxes[idx]
    d0 = abs((f[0] + f[2]) / 2 - m)
    need = max(MIN_MARGIN_PX, MARGIN * (f[2] - f[0]))
    slack = math.inf
    for j, g in enumerate(boxes):
        if j == idx or g[2] <= x1 or g[0] >= x2 or g[3] <= y1 or g[1] >= y2:
            continue
        gx1, gx2 = max(g[0], x1), min(g[2], x2)
        s = abs((gx1 + gx2) / 2 - m) - d0
        if s <= need:
            return None
        slack = min(slack, s)
    return slack


def plan_crop(boxes: list[Box], idx: int, w: int, h: int, pad: float) -> list[int] | None:
    """A crop [x1, y1, x2, y2] of the w x h frame in which the node picks boxes[idx], or None.

    The crop always contains the whole square the node will warp for that face (node_region,
    plus a little slack for the detector's jitter on the crop), so the warp is the same one it
    would be on the full frame -- a tighter crop would make the node shift its square inward
    and warp a different patch. Full height is tried first (the node only compares horizontal
    centres, so height is free context); a crop to the square's own rows is the fallback for a
    face directly above or below another.

    Candidate edges are the frame's, the square's, the edges of each neighbour (to leave it
    wholly outside), and the mirror of each of those about the face's centre (to centre the
    face). Near a frame edge the mirror is clamped, which is what "the crop changes widths near
    the edges" costs, and why every candidate is re-checked rather than assumed. Among the
    crops that work, the widest wins: more context is closer to what the detector saw on the
    full frame.
    """
    f = boxes[idx]
    fcx = (f[0] + f[2]) / 2
    r = node_region(f, w, h, pad)
    slack = max(2, round(0.05 * (r[2] - r[0])))
    rx1, ry1 = max(0, r[0] - slack), max(0, r[1] - slack)
    rx2, ry2 = min(w, r[2] + slack), min(h, r[3] + slack)
    others = [g for j, g in enumerate(boxes) if j != idx]

    lefts = {0, rx1} | {math.ceil(g[2]) for g in others if g[2] <= rx1}
    rights = {w, rx2} | {math.floor(g[0]) for g in others if g[0] >= rx2}
    lefts |= {min(rx1, max(0, round(2 * fcx - x2))) for x2 in rights}
    rights |= {max(rx2, min(w, round(2 * fcx - x1))) for x1 in list(lefts)}

    for y1, y2 in ((0, h), (ry1, ry2)):
        best = None
        for x1 in lefts:
            for x2 in rights:
                if x2 - x1 < MIN_FACE_PX or _isolated(boxes, idx, x1, y1, x2, y2) is None:
                    continue
                key = (x2 - x1, -abs(fcx - (x1 + x2) / 2))
                if best is None or key > best[0]:
                    best = (key, [int(x1), int(y1), int(x2), int(y2)])
        if best:
            return best[1]
    return None


def isolates(raw, width: int, expect: Box) -> bool:
    """Whether the node, given these detector boxes for the crop, picks the face at `expect`
    (crop coordinates). The last word: plan_crop plans on the full frame's boxes, this checks
    the detector's own answer on the pixels the node will actually see."""
    pick = node_pick(raw, width)
    return pick is not None and iou(valid(raw)[pick], expect) >= MATCH_IOU
