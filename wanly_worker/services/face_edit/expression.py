"""What a face edit asks for, as numbers -- ported from keyframe-server's face mode.

keyframe-server (github.com/DavidJBarnes/keyframe-server, server.py) ran this exact contract in
front of ComfyUI's ExpressionEditor node, and every range and every lexicon entry here is its
code, not a re-derivation. Only the plumbing changed: that server took a `GenerateRequest` that
also carried Qwen's fields; here the face-edit request is its own model.

Ranges are the node's own (ComfyUI-AdvancedLivePortrait nodes.py, ExpressionEditor.INPUT_TYPES)
and are enforced here so an out-of-range value fails as a 422 at the edge rather than as a
tensor that quietly warps a face inside out.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Expression(BaseModel):
    rotate_pitch: float = Field(0, ge=-20, le=20)
    rotate_yaw: float = Field(0, ge=-20, le=20)
    rotate_roll: float = Field(0, ge=-20, le=20)
    blink: float = Field(0, ge=-20, le=5)       # eye openness: -20 closed .. 5 wide
    eyebrow: float = Field(0, ge=-10, le=15)
    wink: float = Field(0, ge=0, le=25)
    pupil_x: float = Field(0, ge=-15, le=15)
    pupil_y: float = Field(0, ge=-15, le=15)
    aaa: float = Field(0, ge=-30, le=120)       # jaw open
    eee: float = Field(0, ge=-20, le=15)        # wide mouth
    woo: float = Field(0, ge=-20, le=15)        # pursed mouth
    smile: float = Field(0, ge=-0.3, le=1.3)

    def nonzero(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v}


def bounds(axis: str) -> tuple[float, float]:
    f = Expression.model_fields[axis]
    lo = next(m.ge for m in f.metadata if hasattr(m, "ge"))
    hi = next(m.le for m in f.metadata if hasattr(m, "le"))
    return lo, hi


def clamp(params: dict[str, float]) -> dict[str, float]:
    """Clamp to the node's ranges. Used where a multiplier can overshoot, never on a caller's
    explicit numbers -- those are validated and refused, so a typo is not silently softened."""
    out = {}
    for k, v in params.items():
        lo, hi = bounds(k)
        out[k] = max(lo, min(hi, v))
    return out


# Prompt sugar over the numeric contract. Deliberately small and explicit: a caller that wants
# exact control passes `expression` and skips all of this. wanly-api sends numbers (its presets
# resolve server-side), so this exists for a curl and for keyframe-server's old callers.
_LEXICON: list[tuple[str, dict]] = [
    (r"\bwid(e|er) eyes?\b|\beyes? wide\b", {"blink": 4}),
    (r"\bsquint(s|ed|ing)?\b|\bnarrow(ed)? eyes?\b", {"blink": -8}),
    (r"\bclos(e|es|ed|ing) (her |his |their )?eyes?\b|\beyes? clos(ed|ing)\b", {"blink": -18}),
    (r"\bblink(s|ed|ing)?\b", {"blink": -12}),
    (r"\bwink(s|ed|ing)?\b", {"wink": 15}),
    (r"\brais(e|es|ed|ing) (her |his |their )?(eye)?brows?\b|\b(eye)?brows? rais(ed|e)\b", {"eyebrow": 8}),
    (r"\bfurrow(s|ed|ing)?\b|\bfrown(s|ed|ing)?\b", {"eyebrow": -6, "smile": -0.2}),
    (r"\bgrin(s|ning)?\b|\bbig smile\b|\bbroad smile\b", {"smile": 1.0}),
    (r"\bsmil(e|es|ing)\b", {"smile": 0.5}),
    (r"\blaugh(s|ed|ing)?\b", {"smile": 0.9, "aaa": 35}),
    (r"\bmouth open\b|\bopens? (her |his |their )?mouth\b|\bgasp(s|ed|ing)?\b", {"aaa": 45}),
    (r"\bpurs(e|es|ed|ing)\b|\bpout(s|ed|ing)?\b|\bwhistl(e|es|ing)\b", {"woo": 10}),
    (r"\blook(s|ing)? (to (her |his |their )?)?left\b|\bglanc(e|es|ing) left\b", {"pupil_x": -8}),
    (r"\blook(s|ing)? (to (her |his |their )?)?right\b|\bglanc(e|es|ing) right\b", {"pupil_x": 8}),
    (r"\blook(s|ing)? up\b|\bglanc(e|es|ing) up\b|\beyes? up\b", {"pupil_y": 8}),
    (r"\blook(s|ing)? down\b|\bglanc(e|es|ing) down\b|\beyes? down\b|\blower(s|ed|ing)? (her |his |their )?gaze\b", {"pupil_y": -8}),
    (r"\bturn(s|ed|ing)? (her |his |their )?head (to the )?left\b|\bhead left\b", {"rotate_yaw": -12}),
    (r"\bturn(s|ing)? (her |his |their )?head (to the )?right\b|\bhead right\b", {"rotate_yaw": 12}),
    (r"\btilt(s|ed|ing)? (her |his |their )?head\b|\bhead tilt(ed)?\b", {"rotate_roll": 8}),
    (r"\bchin up\b|\blift(s|ed|ing)? (her |his |their )?chin\b|\bhead up\b", {"rotate_pitch": -8}),
    (r"\bchin down\b|\bhead down\b|\bduck(s|ing)? (her |his |their )?head\b", {"rotate_pitch": 8}),
]

# Intensity adverbs scale whatever they precede. Applied globally rather than per-phrase:
# prompts at this length rarely mix intensities, and per-phrase scoping would need a parser.
_INTENSITY = [
    (r"\b(slight(ly)?|soft(ly)?|faint(ly)?|soften(ed|s)?|subtle|barely|a little|a bit|gentl[ey])\b", 0.5),
    (r"\b(very|much|strong(ly)?|wide(ly)?|big|broad(ly)?|deep(ly)?|hard)\b", 1.5),
]

RECOGNISED = ("smile, grin, laugh, frown, blink, wink, squint, wide eyes, closed eyes, raised "
              "brows, open mouth, purse/pout, look left/right/up/down, turn head left/right, "
              "tilt head, chin up/down")


class NothingToApply(ValueError):
    """Neither numbers nor a recognised phrase: refusing beats returning the input unchanged,
    which reads as "the edit worked and did nothing"."""


def resolve_expression(expression: Expression | None, prompt: str = "") -> tuple[Expression, str]:
    """Numeric expression wins; otherwise read the prompt. Returns (exp, source)."""
    if expression is not None:
        return expression, "explicit"

    text = (prompt or "").lower()
    params: dict[str, float] = {}
    hits: list[str] = []
    for pattern, delta in _LEXICON:
        if re.search(pattern, text):
            hits.append(pattern.split("\\b")[1] if "\\b" in pattern else pattern)
            for k, v in delta.items():
                # Largest magnitude wins when two phrases drive the same axis, so "smiling and
                # grinning" gives one grin, not a summed clamp.
                if abs(v) > abs(params.get(k, 0.0)):
                    params[k] = v

    scale = 1.0
    for pattern, factor in _INTENSITY:
        if re.search(pattern, text):
            scale = factor
            break
    if scale != 1.0:
        params = {k: v * scale for k, v in params.items()}

    if not params:
        raise NothingToApply(
            "nothing to apply: send an `expression` object, or a prompt using a known term. "
            f"Recognised terms: {RECOGNISED}.")
    return Expression(**clamp(params)), "prompt:" + ",".join(hits)


def whole_face_motion(exp: Expression, src_ratio: float) -> float:
    """0-1: how much of the face this edit displaces as a rigid body.

    Read off the request rather than estimated from |edited - source|. Difference magnitude is
    contrast-dependent and the estimate collapses on smooth skin: the same 12-degree turn put
    25% of the face above the ramp on a high-contrast subject and 7% on a soft-lit one, so a
    pixel-derived radius left the second face fully ghosted. The requested angles are exact.

    (keyframe-server also counted a driving image's rotation here. There is no driving image
    in this service -- phase 1 edits by numbers -- so that term is gone.)
    """
    m = max(abs(exp.rotate_pitch), abs(exp.rotate_yaw), abs(exp.rotate_roll)) / 8.0
    # Relaxing the resting expression moves the whole face without any rotate_* axis set.
    return min(1.0, max(m, 1.0 - src_ratio))
