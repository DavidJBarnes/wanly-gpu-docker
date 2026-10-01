"""Compose a 1536x1024 character sheet: the REAL face beside the generated turnaround (console#582).

THE LAYOUT IS THE ONE PHASE 0 PROVED AND THE USER APPROVED (loras/reftest-2026-09-30/sheets.py
`face_panel` + `compose`, reused by phase0-2026-10-01/compose_sheet.py): a 448x1024 panel of the
real face photo on the left, the 1088x1024 Qwen turnaround (front / side / back) on the right.
That is the layout Best-Face-ID's CharacterSheet LoRA was trained on, and the sheet is what the
engine conditions renders on (wanly-gpu-docker#156) -- so the real face in it is the identity
anchor, and it must be the person's own pixels, never the model's.

THE FACE PANEL IS FACE-DETECTED, NOT LETTERBOXED. The phase-0 ComfyUI workflow letterboxed the
whole photo into the panel (ImageResizeKJv2 pad); that is not the approved layout. sheets.py
crops around the detected face instead:
  * the face fits in a full-height 448:1024 strip -> that strip, centred on the face and clamped
    to the photo, full-bleed (the card's own examples);
  * the face is wider than such a strip (a tight square close-up) -> a head-and-shoulders crop
    with the face ~75% of the width, letterboxed on WHITE, so no part of the real face is cut
    or squashed.
Two cases sheets.py never met are handled here without changing either rule: a photo too narrow
for a full-height strip is cropped vertically around the face instead, and a photo with no face
the detector can find is centred (the service refuses those before spending GPU time; this is
the fallback for a detector that would not load).

Pure PIL: no model, no ComfyUI -- tested in CI.
"""
from __future__ import annotations

FACE_W, BODY_W, H = 448, 1088, 1024
SHEET_W = FACE_W + BODY_W


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


def face_panel(pil, box: list[float] | None):
    """(the 448x1024 panel, how it was made: "crop" | "letterbox" | "centre").

    `box` is the face's [x1, y1, x2, y2] in `pil`'s pixels (the largest face), or None."""
    from PIL import Image

    w0, h0 = pil.size
    if box is None:
        return _strip(pil, w0 / 2, h0 / 2), "centre"
    x1, y1, x2, y2 = box
    fw = x2 - x1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    full_w = FACE_W * h0 / H                    # widest crop that fills the panel at full height
    if fw <= 0.85 * full_w:                     # face fits -> full-bleed (sheets.py)
        return _strip(pil, cx, cy), "crop"
    # Face wider than the panel: head and shoulders, face ~75% of the width, on white.
    cw2 = min(fw / 0.75, w0)
    x0 = min(max(cx - cw2 / 2, 0), w0 - cw2)
    nh = int(round(h0 * FACE_W / cw2))
    if nh > H:                                  # would overflow the panel: crop instead
        return _strip(pil, cx, cy), "crop"
    crop = pil.crop((int(x0), 0, int(x0 + cw2), h0))
    panel = Image.new("RGB", (FACE_W, H), "white")
    panel.paste(crop.resize((FACE_W, nh), Image.LANCZOS), (0, (H - nh) // 2))
    return panel, "letterbox"


def compose(face_pil, turnaround_pil, box: list[float] | None):
    """(the 1536x1024 sheet, the face panel's mode)."""
    from PIL import Image

    body = turnaround_pil.convert("RGB")
    if body.size != (BODY_W, H):
        body = body.resize((BODY_W, H), Image.LANCZOS)
    panel, mode = face_panel(face_pil.convert("RGB"), box)
    sheet = Image.new("RGB", (SHEET_W, H), "white")
    sheet.paste(panel, (0, 0))
    sheet.paste(body, (FACE_W, 0))
    return sheet, mode


def largest(boxes: list[list[float]]) -> list[float] | None:
    """The subject's face: the largest box, as sheets.py's detector picked it."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
