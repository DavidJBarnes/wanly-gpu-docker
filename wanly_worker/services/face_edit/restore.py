"""Give the source its own pixels back wherever LivePortrait did not move anything.

Ported verbatim from keyframe-server's server.py (restore_detail), measurements included. The
constants keep their meaning and gain a FACE_EDIT_ prefix, because this container has other
services and an unprefixed DETAIL_LO would read as belonging to any of them.

WHY IT EXISTS. ExpressionEditor decodes the face crop through a fixed 256x256 bottleneck, so it
softens the ENTIRE crop -- including the parts it did not move. Measured on a 214x292 face:
texture falls to 16% of source, and crop_factor cannot fix it (the node clamps to 1.5-2.5, and
1.5 is already the sharp end). Blending on |edited - source| keeps the edit where it happened and
the original texture everywhere else: 16% -> 31%, with no visible seam.

Not a sharpening filter and not a restoration model: every output pixel comes from one of the
two real images, so it cannot invent detail or alter age -- which is the whole reason face mode
exists instead of a Qwen pass.
"""
from __future__ import annotations

import os

import numpy as np

#: Below LO the node changed nothing worth keeping, so the source pixel wins; above HI the
#: warp wins outright. The ramp between is the handover.
DETAIL_LO = float(os.environ.get("FACE_EDIT_DETAIL_LO", "10.0"))
DETAIL_HI = float(os.environ.get("FACE_EDIT_DETAIL_HI", "35.0"))
DETAIL_BLUR = float(os.environ.get("FACE_EDIT_DETAIL_BLUR", "2.0"))
#: Radius of the alpha max-pool, as a fraction of the width of the region the warp changed.
#: The blend is position-aligned, so it is only valid where LivePortrait left the pixel where
#: it found it. Skin sliding over skin is a large displacement with a small |edited - source|,
#: so it lands mid-ramp and composites two misaligned copies of the same feature at ~50/50 --
#: the doubled brows and lids that showed up on every rotate_* axis. Max-pooling first means a
#: neighbourhood containing any motion is taken wholly from the warp. Measured on rotate_pitch
#: 8: pixels in the 0.15-0.85 ghost band fell 47.5% -> 9.7%. Swept by eye at
#: 0.055/0.08/0.11/0.15 on three faces: 0.055 still doubles a lip, above 0.08 only costs texture.
DETAIL_DILATE = float(os.environ.get("FACE_EDIT_DETAIL_DILATE", "0.08"))
#: Unsharp amount inside the warp region. OFF by default: on real footage it reads as
#: processed and crunchy rather than sharp. Head rotations move every pixel, so there is
#: nothing unchanged to restore from; per request it is the one lever there (38% -> 68%).
DETAIL_SHARPEN = float(os.environ.get("FACE_EDIT_DETAIL_SHARPEN", "0.0"))


def restore_detail(source: np.ndarray, edited: np.ndarray, strength: float = 1.0,
                   sharpen: float = DETAIL_SHARPEN, motion: float = 1.0) -> np.ndarray:
    """HxWx3 uint8 in, HxWx3 uint8 out. Where alpha is 0 this returns the source byte-for-byte.

    `motion` (whole_face_motion) is what separates a smile from a head turn: a mouth edit moves
    a few percent of the face and needs almost no max-pool, a rotation moves all of it and needs
    the full radius.
    """
    import cv2

    if strength <= 0:
        return edited
    s = np.asarray(source, np.float32)
    e = np.asarray(edited, np.float32)
    if s.shape != e.shape:
        return edited
    # The change map MUST come from the unsharpened output. Sharpening first makes every
    # textured pixel in the frame differ from the source -- background included -- and the
    # "everything outside the face is bit-identical" guarantee is lost. Measured when this was
    # wrong: 0.63 drift in the frame corners and 36% of the frame touched, against 0.0000 and
    # 13% ordered correctly.
    d = np.abs(e - s).mean(2)
    a = np.clip((d - DETAIL_LO) / max(1e-6, DETAIL_HI - DETAIL_LO), 0, 1)

    # Face size from the node's own composite mask -- everything outside it is bit-identical,
    # so d is exactly 0 there. Contrast-free, which |edited - source| is not: sizing the radius
    # off the change map alone gave a low-contrast face a third of the radius of a
    # high-contrast one under the identical edit.
    mask = d > 0
    r = max(3, int(DETAIL_DILATE * np.sqrt(mask.sum()) * motion))

    # Only the confidently-moved core is grown, and it is opened first. The decoder's uniform
    # softening leaves single pixels scattered over the face just above the ramp; dilating those
    # directly merges the speckle into a sheet and softens the forehead on a smile.
    core = (a > 0.5).astype(np.float32)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    core = cv2.GaussianBlur(core, (0, 0), max(DETAIL_BLUR, r / 4.0))
    a = np.maximum(a, core)[..., None]
    a = 1.0 - (1.0 - a) * strength      # strength<1 keeps more of the node's output

    if sharpen > 0:
        # Confined to the node's composite mask: the max-pool pushes alpha past that boundary,
        # and an unconfined sharpen would drift pixels the node deliberately left untouched.
        blurred = cv2.GaussianBlur(e, (0, 0), 1.2)
        e = np.where((d > 0)[..., None], np.clip(e + (e - blurred) * sharpen, 0, 255), e)

    return np.clip(s * (1 - a) + e * a, 0, 255).astype(np.uint8)
