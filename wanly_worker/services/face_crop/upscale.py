"""Real-ESRGAN upscaling for face crops and small whole images (#206).

WHY. The SDXL and LTX still recipes both train with `bucket_no_upscale = true`: an image is
never enlarged to reach the 1024 ceiling, so a 254 px close-up trains at 254 px. On Joana v3,
29 of 48 faces were under 250 px at training size and none was ever seen large. v4 added
head-and-shoulders crops and upscaled close-ups and reached v3-e11 likeness in about half the
steps (wanly-api#431). Upscaling is what makes a small crop worth adding: a 300 px crop is no
bigger to the trainer than the face already was in the photograph.

WHICH MODEL, AND WHY NOT THE OBVIOUS ONE. `realesr-general-x4v3` (SRVGG, a plain conv stack)
blended with its `wdn` pair by Deep Network Interpolation at denoise 0.2 -- the setting David
validated by eye on the v4 close-ups as KEEPING freckles and skin texture. RealESRGAN_x4plus
(RRDB) is the "quality" model and the wrong one here: it smooths skin into plastic, and a LoRA
trained on that learns the plastic. It is deliberately not offered.

CPU, like detection. SRVGG is cheap -- 1-10 s an image -- and the GPU on this box belongs to a
render or a training run. Ported from ~/projects/loras/scripts/enhance.py (torch + numpy only,
no basicsr / realesrgan packages, which pin their own torch and would fight the image's cu128
build). torch is imported lazily so a service that never upscales never pays for importing it.

WEIGHTS ARE BAKED INTO THE IMAGE (Dockerfile, sha256-checked), not fetched at first use: the
face-edit precedent, and the same lesson as buffalo_l -- a model fetched inside a request makes
the first call depend on the network and blow wanly-api's timeout.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np

#: Where the Dockerfile puts the two .pth files.
MODELS_DIR = Path(os.environ.get("FACE_UPSCALE_MODELS_DIR", "/opt/face-crop/models"))
NORMAL = "realesr-general-x4v3.pth"
WDN = "realesr-general-wdn-x4v3.pth"
#: DNI weight on the NORMAL model; the wdn (denoise) model gets the rest. 0.2 is the validated
#: setting -- 0 keeps every grain of sensor noise, 1 is the full denoise that starts to smooth.
DENOISE = float(os.environ.get("FACE_UPSCALE_DENOISE", "0.2"))
#: The long edge an upscaled image is brought to. The trainer's 1024 ceiling: past it the
#: trainer downscales again, so more is wasted CPU and wire.
TARGET_EDGE = int(os.environ.get("FACE_UPSCALE_TARGET_EDGE", "1024"))
#: Below this factor the result is left alone. A 900 -> 1024 "upscale" buys the trainer
#: nothing and costs a re-encode and a model pass.
MIN_FACTOR = float(os.environ.get("FACE_UPSCALE_MIN_FACTOR", "1.2"))
#: One native 4x pass at most; anything past 4x is a Lanczos stretch. A second pass renders
#: 16x internally -- a 200 px crop becomes 3200 px of invented detail, then thrown away -- and
#: a face that small has no more identity to recover. It is the slow path for no gain.
MAX_PASSES = 1
#: CPU threads for the conv stack. Not every core: the box is also rendering or training, and
#: the trainer's dataloader and ComfyUI's own CPU work must not be starved by a dataset tool.
THREADS = int(os.environ.get("FACE_UPSCALE_THREADS", "4"))
#: Tile size in input px, with a feathered overlap. Bounds memory on CPU; the receptive field
#: is wider than the overlap, so tiles are blended rather than hard-cut (no seams).
TILE = 256
OVERLAP = 32
NATIVE = 4

_net = None
_lock = threading.Lock()


def available() -> bool:
    """Both weight files are where the image put them. Reported by /health, so wanly-api and
    the Workers page can tell an image that cannot upscale from one that can."""
    return (MODELS_DIR / NORMAL).is_file() and (MODELS_DIR / WDN).is_file()


def plan(w: int, h: int, target: int = TARGET_EDGE) -> float:
    """The factor an image of w x h would be scaled by, or 1.0 when it is left alone.

    Pure arithmetic, so it is tested without torch: the rule is the thing that must not drift.
    """
    longest = max(w, h)
    if longest <= 0:
        return 1.0
    factor = target / longest
    return factor if factor >= MIN_FACTOR else 1.0


def _build():
    """The SRVGG net with the DNI-blended weights. Built once per process, under a lock: two
    crop requests arriving together must not both pay for the load."""
    global _net
    with _lock:
        if _net is not None:
            return _net
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SRVGGNetCompact(nn.Module):
            """realesr-general-x4v3: a plain conv stack, a pixel shuffle, and a nearest-
            upsampled copy of the input added back as a global residual."""

            def __init__(self, num_feat=64, num_conv=32, scale=NATIVE):
                super().__init__()
                self.scale = scale
                body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
                for _ in range(num_conv):
                    body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
                body.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1))
                self.body = nn.Sequential(*body)
                self.upsampler = nn.PixelShuffle(scale)

            def forward(self, x):
                out = self.upsampler(self.body(x))
                return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")

        if not available():
            raise RuntimeError(f"Real-ESRGAN weights are missing from {MODELS_DIR} -- they are "
                               f"baked into the image, so this image is wrong")
        load = lambda n: (lambda sd: sd.get("params", sd))(
            torch.load(MODELS_DIR / n, map_location="cpu", weights_only=True))
        normal, wdn = load(NORMAL), load(WDN)
        sd = blend(normal, wdn, DENOISE)
        net = SRVGGNetCompact()
        net.load_state_dict(sd, strict=True)
        if THREADS > 0:
            torch.set_num_threads(THREADS)
        _net = net.eval()
        return _net


def blend(normal: dict, wdn: dict, denoise: float) -> dict:
    """Deep Network Interpolation: normal * denoise + wdn * (1 - denoise), key by key.

    The two checkpoints share an architecture, so interpolating their weights interpolates the
    behaviour between "keep the grain" and "denoise". Named and separate because the direction
    is easy to get backwards -- and backwards is the plastic skin this module exists to avoid.
    """
    return {k: normal[k] * denoise + wdn[k] * (1 - denoise) for k in normal}


def _ramp(n: int, left: int, right: int):
    import torch
    w = torch.ones(n)
    if left > 0:
        w[:left] = torch.linspace(0, 1, left + 2)[1:-1]
    if right > 0:
        w[n - right:] = torch.linspace(1, 0, right + 2)[1:-1]
    return w


def _run_tiled(net, x):
    """x: 1x3xHxW float in [0,1]. Returns the 4x result, tile by tile with feathered seams."""
    import torch
    _, _, h, w = x.shape
    if TILE >= w and TILE >= h:
        return net(x)
    s = NATIVE
    acc = torch.zeros(1, 3, h * s, w * s)
    wsum = torch.zeros(1, 1, h * s, w * s)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            py0, px0 = max(y0 - OVERLAP, 0), max(x0 - OVERLAP, 0)
            py1, px1 = min(y1 + OVERLAP, h), min(x1 + OVERLAP, w)
            patch = net(x[:, :, py0:py1, px0:px1])
            ph, pw = patch.shape[-2:]
            win = (_ramp(ph, (y0 - py0) * s, (py1 - y1) * s).view(-1, 1)
                   * _ramp(pw, (x0 - px0) * s, (px1 - x1) * s).view(1, -1))
            acc[:, :, py0 * s:py0 * s + ph, px0 * s:px0 * s + pw] += patch * win
            wsum[:, :, py0 * s:py0 * s + ph, px0 * s:px0 * s + pw] += win
    return acc / wsum.clamp(min=1e-8)


def upscale_bgr(img: np.ndarray, target: int = TARGET_EDGE) -> np.ndarray:
    """An OpenCV BGR image brought to `target` on its long edge, or returned unchanged when
    plan() says it is close enough already.

    One native 4x pass (MAX_PASSES), then Lanczos to the exact size: down when 4x overshoots,
    up for whatever the pass could not cover.
    """
    import cv2
    import torch

    h, w = img.shape[:2]
    factor = plan(w, h, target)
    if factor == 1.0:
        return img
    net = _build()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(min(MAX_PASSES, max(1, math.ceil(math.log(factor, NATIVE))))):
            x = _run_tiled(net, x).clamp(0, 1)
    out = (x.squeeze(0).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    want = (max(1, round(w * factor)), max(1, round(h * factor)))
    if (out.shape[1], out.shape[0]) != want:
        out = cv2.resize(out, want, interpolation=cv2.INTER_LANCZOS4)
    return out
