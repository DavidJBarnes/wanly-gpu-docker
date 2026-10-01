"""Compose a 1536x1024 character sheet: the REAL face beside the generated turnaround (console#582).

THE LAYOUT is the one phase 0 proved: a 448x1024 panel of the person's real face on the left,
the 1088x1024 Qwen turnaround (front / side / back) on the right. That is the layout
Best-Face-ID's CharacterSheet LoRA was trained on, and the sheet is what the engine conditions
renders on (wanly-gpu-docker#156) -- so the face in it is the identity anchor, and it must be the
person's own pixels, never the model's.

ONE PHOTO, FACE AUTO-CROPPED FROM IT (console#585). The sheet is built from a single photo of the
person, full body or most of it (graph.py says why: Qwen only keeps a build it is SHOWN as image
1). The face panel is cut from that same photo, the way the tested UI workflow
(loras/phase0-2026-10-01/character_sheet_one_input.json) cuts it with MediaPipe -> CropByBBoxes
(padding 140, keep_aspect pad) -> ImageResizeKJv2 448x1024 pad on white:
  * the detected face box, grown by `padding` source pixels on every side and clamped to the
    photo (CropByBBoxes' rule: clamped, not shifted -- a face at the edge keeps its own side);
  * scaled, aspect kept, to fit the 448x1024 panel, centred on WHITE. A head-and-shoulders crop
    is about square, so it lands as a ~448 px square in the middle of the panel.
A small face in a wide shot is scaled UP to fill the panel's width, which is soft: `scale` in the
result says by how much, and the console tells the user to pick a photo with a reasonably large
face. The face is FOUND by the service's own detector, not by MediaPipe -- see app.py.

A photo with no face the detector can find is refused before any GPU time (app.py); the
centred strip below is only the fallback for a detector that would not load at all.

Pure PIL: no model, no ComfyUI -- tested in CI.
"""
from __future__ import annotations

FACE_W, BODY_W, H = 448, 1088, 1024
SHEET_W = FACE_W + BODY_W
#: Source pixels added on each side of the face box: the one-input workflow's CropByBBoxes value.
DEFAULT_PADDING = 140
MAX_PADDING = 1024


def _strip(pil, cx: float, cy: float):
    """A 448:1024 crop centred on (cx, cy), as large as the photo allows, clamped inside it."""
    from PIL import Image

    w0, h0 = pil.size
    ch = h0
    cw = FACE_W * h0 / H
    if cw > w0:                                 # narrower than a full-height strip
        cw = w0
        ch = cw * H / FACE_W
    x0 = min(max(cx - cw / 2, 0), w0 - cw)
    y0 = min(max(cy - ch / 2, 0), h0 - ch)
    return pil.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((FACE_W, H),
                                                                            Image.LANCZOS)


def crop_region(box: list[float], size: tuple[int, int],
                padding: int = DEFAULT_PADDING) -> tuple[int, int, int, int]:
    """(left, top, right, bottom): the face box grown by `padding` px each side, clamped to the
    photo -- CropByBBoxes' arithmetic."""
    w0, h0 = size
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1 - padding)), max(0, int(y1 - padding))
    right, bottom = min(w0, round(x2 + padding)), min(h0, round(y2 + padding))
    return left, top, max(right, left + 1), max(bottom, top + 1)


def face_panel(pil, box: list[float] | None, padding: int = DEFAULT_PADDING):
    """(the 448x1024 panel, info). info = {"mode": "auto_crop" | "centre", "crop": [l, t, r, b]
    | None, "scale": how much the crop was enlarged (> 1 is upscaled, i.e. soft) | None}.

    `box` is the face's [x1, y1, x2, y2] in `pil`'s pixels (the largest face), or None."""
    from PIL import Image

    w0, h0 = pil.size
    if box is None:
        return _strip(pil, w0 / 2, h0 / 2), {"mode": "centre", "crop": None, "scale": None}
    region = crop_region(box, pil.size, padding)
    crop = pil.crop(region)
    s = min(FACE_W / crop.width, H / crop.height)
    nw, nh = max(1, round(crop.width * s)), max(1, round(crop.height * s))
    panel = Image.new("RGB", (FACE_W, H), "white")
    panel.paste(crop.resize((nw, nh), Image.LANCZOS), ((FACE_W - nw) // 2, (H - nh) // 2))
    return panel, {"mode": "auto_crop", "crop": list(region), "scale": round(s, 2)}


def compose(photo, turnaround_pil, box: list[float] | None, padding: int = DEFAULT_PADDING):
    """(the 1536x1024 sheet, the face panel, the panel's info)."""
    from PIL import Image

    body = turnaround_pil.convert("RGB")
    if body.size != (BODY_W, H):
        body = body.resize((BODY_W, H), Image.LANCZOS)
    panel, info = face_panel(photo.convert("RGB"), box, padding)
    sheet = Image.new("RGB", (SHEET_W, H), "white")
    sheet.paste(panel, (0, 0))
    sheet.paste(body, (FACE_W, 0))
    return sheet, panel, info


def largest(boxes: list[list[float]]) -> list[float] | None:
    """The subject's face: the largest box."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
