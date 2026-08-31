"""The LTX-2.5 keyframe placement grid.

Everything here follows from two facts read out of the LTX-2 source rather than
guessed:

  * ``SpatioTemporalScaleFactors.default() == (time=8, height=32, width=32)``
    (``ltx-core/src/ltx_core/types.py``).
  * The video encoder is causal, so the first temporal latent frame covers a
    single pixel frame while every later one covers ``time`` of them
    (``ltx-core/src/ltx_core/tools.py``, ``_first_frame_keyframes_mask``).

So latent frame boundaries land at pixel indices 0, 1, 9, 17, 25 ... — that is,
``0`` or ``1 + 8k``. A keyframe placed anywhere else is NOT snapped for you:
``VideoConditionByKeyframeIndex.apply_to`` does ``positions[:, 0, ...] +=
frame_idx`` and then divides by fps, so an off-grid index is used literally and
the guide token sits between two latent slots, diffusing across both instead of
pinning either. That is what "the keyframe isn't working" looks like.

Index 0 is special in a second way: ``helpers.py`` routes it to
``VideoConditionByLatentIndex(latent_idx=0)`` — the true first-frame anchor —
while every other index becomes a ``VideoConditionByKeyframeIndex``.
"""
TEMPORAL_SCALE = 8


def is_on_grid(idx: int) -> bool:
    return idx == 0 or (idx >= 1 and (idx - 1) % TEMPORAL_SCALE == 0)


def snap(idx: int) -> int:
    """Nearest valid index. Ties go up, toward more frames of runway."""
    if idx <= 0:
        return 0
    return 1 + TEMPORAL_SCALE * round((idx - 1) / TEMPORAL_SCALE)


def grid_points(num_frames: int) -> list[int]:
    """Every legal keyframe index inside a clip of ``num_frames``."""
    pts = [0]
    i = 1
    while i < num_frames:
        pts.append(i)
        i += TEMPORAL_SCALE
    return pts


def valid_num_frames(n: int) -> int:
    """num_frames must itself close on a latent boundary — 121 = 1 + 8*15.

    Rounds DOWN, not to nearest, because that is what LTX does to it anyway:
    ``snap_frames_to_grid`` in ltx-pipelines/utils/helpers.py is
    ``((frames - 1) // time_scale) * time_scale + 1``. Rounding up here would
    make us report a frame count the pipeline then silently reduces, and every
    terminal keyframe index would be computed against a clip one slot longer
    than the one actually rendered."""
    return max(1 + TEMPORAL_SCALE,
               ((n - 1) // TEMPORAL_SCALE) * TEMPORAL_SCALE + 1)


def auto_place(count: int, num_frames: int) -> list[int]:
    """Spread ``count`` keyframes across the clip, on grid, first at 0 and last near the end.

    Even spacing is the neutral default, not the good one: the recipe weights the
    budget toward the beat that matters (0/41/73/121 concentrates time on the
    final action). Callers that know which beat is the payoff should send
    explicit indices; this only keeps an unspecified request runnable.
    """
    if count <= 0:
        return []
    if count == 1:
        return [0]
    # Terminal index sits INSIDE the clip: the last on-grid index strictly below
    # num_frames.
    #
    # It used to sit AT num_frames, which was right for the CLI -- run_richmond.sh
    # ran `--image kf4.png 121` with `--num-frames 121` and produced working
    # video, because LTX's VideoConditionByKeyframeIndex treats positions as
    # continuous time and tolerates a token just past the last frame.
    #
    # ComfyUI does not. LTXVAddGuideMulti asserts "Conditioning frames exceed the
    # length of the latent sequence", so a terminal guide at num_frames fails the
    # whole render. The assumption did not survive the move off the CLI, and
    # nothing caught it because a single-keyframe job places its only guide at 0
    # -- every 2+ keyframe storyboard failed.
    #
    # The cost is real and worth stating: the terminal waypoint now lands up to
    # 8 frames before the end, so the model improvises that tail. Send explicit
    # indices when the final beat has to be pinned exactly.
    last = max(0, num_frames - 1)
    last = last if is_on_grid(last) else snap(last - TEMPORAL_SCALE // 2)
    if last >= num_frames:
        last = snap(num_frames - 1 - TEMPORAL_SCALE)
    out = [0]
    for i in range(1, count - 1):
        out.append(snap(round(i * last / (count - 1))))
    out.append(last)
    # Two keyframes on the same latent slot means the later one silently wins.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + TEMPORAL_SCALE
    return out


def default_strengths(count: int) -> list[float]:
    """1.0 opener, 0.65 interior, 0.9 terminal — the recipe's ramp.

    Full strength on interior waypoints snaps the video onto them visibly, and
    also freezes the background into mannequins when keyframes share pixels.
    """
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [1.0] + [0.65] * (count - 2) + [0.9]
