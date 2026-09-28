"""#148: a recipe's render size is derived from its start frame but capped, so an upscaled
start frame is downsampled into a normal-sized clip instead of producing a huge one."""
import pathlib
import sys

import pytest
from PIL import Image

ENGINE = pathlib.Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))
import app as engine_app  # noqa: E402

CAP = 1024 * 1024


def frame(tmp_path, w, h):
    p = tmp_path / f"f_{w}x{h}.png"
    Image.new("RGB", (w, h)).save(p)
    return p


def test_the_default_cap_covers_every_size_renders_have_used():
    assert engine_app.MAX_RENDER_PIXELS == CAP


@pytest.mark.parametrize("w,h,want", [
    (1824, 1248, (1216, 832)),      # the upscaled landscape frame that prompted #148
    (1248, 1824, (832, 1216)),      # and its portrait twin
    (2432, 1664, (1216, 832)),      # 2x upscale, same aspect
])
def test_an_upscaled_frame_is_brought_back_to_the_normal_size(tmp_path, w, h, want):
    assert engine_app.derive_size(frame(tmp_path, w, h)) == want


@pytest.mark.parametrize("w,h", [(1216, 832), (832, 1216), (1024, 1024), (960, 544), (512, 768)])
def test_a_frame_at_or_under_the_cap_renders_exactly_as_before(tmp_path, w, h):
    assert engine_app.derive_size(frame(tmp_path, w, h)) == ((w // 64) * 64, (h // 64) * 64)


@pytest.mark.parametrize("w,h", [(1824, 1248), (4000, 3000), (3000, 4000), (5312, 2988), (1300, 1300)])
def test_the_result_stays_inside_the_cap_on_the_64_grid_and_keeps_the_aspect(tmp_path, w, h):
    rw, rh = engine_app.derive_size(frame(tmp_path, w, h))
    assert rw * rh <= CAP
    assert rw % 64 == 0 and rh % 64 == 0
    assert abs(rw / rh - w / h) < 0.08


def test_it_never_scales_up(tmp_path):
    assert engine_app.derive_size(frame(tmp_path, 300, 200)) == (256, 192)


def test_the_cap_is_adjustable(tmp_path):
    p = frame(tmp_path, 1824, 1248)
    assert engine_app.derive_size(p, max_pixels=1824 * 1248) == (1792, 1216)
    assert engine_app.derive_size(p, max_pixels=0) == (1792, 1216)   # 0 = no cap
