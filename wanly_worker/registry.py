"""Which services exist, and which of them this container was asked to run.

`SERVICES` is a comma-separated list: SERVICES=ltx-engine,lora-trainer.

A NAME MAY EXPAND TO SEVERAL PROCESSES. `ltx-engine` is ComfyUI, the engine API and the
render daemon, started in that order; the flag names the capability and the registry knows
what it takes. /health lists every process; `provides` reports the names that were asked for.

AN UNKNOWN NAME IS FATAL, and that is the whole point of parsing it here rather than reading
the variable where it is used. A typo that is quietly ignored produces a container that boots
clean, reports healthy, and serves nothing -- and the caller learns about it as "captioner
unreachable" from a completely different machine. Failing at boot with the list of real names
turns a cross-host mystery into one line of log.

An EMPTY list is fatal for the same reason: a services container running no services is not a
degraded state to tolerate, it is a misconfiguration to report.
"""
from __future__ import annotations

import os

from typing import Callable

from wanly_worker.service import Service
from wanly_worker.services.face_crop import FaceCrop
from wanly_worker.services.face_edit import FaceEdit
from wanly_worker.services.image_description import ImageDescription
from wanly_worker.services.image_edit import image_edit_group
from wanly_worker.services.lora_trainer import LoraTrainer
from wanly_worker.services.ltx_engine import ltx_engine_group
from wanly_worker.services.scene_caption.service import scene_caption_group

#: Every service the image can run, by the name the SERVICES flag uses. A value is a factory
#: returning the ordered list of processes that name stands for (or one Service). Adding one
#: here is the only registration step.
KNOWN: dict[str, Callable[[], list[Service] | Service]] = {
    "ltx-engine": ltx_engine_group,
    "lora-trainer": LoraTrainer,
    "image-description": ImageDescription,
    "face-crop": FaceCrop,
    "face-edit": FaceEdit,
    "image-edit": image_edit_group,
    # JoyCaption for the <SCENE> half, always resident (wanly-console#572). Called, claims
    # nothing, adds no kind, and runs in every mode: on a card it shares with image-edit it
    # yields per edit (SCENE_CAPTION_SHARED) rather than being stopped by a mode -- unless
    # SCENE_CAPTION_MODES keeps it to some modes (#199, 3090b until the 2070).
    "scene-caption": scene_caption_group,
}

#: Services whose presence changes what KIND of worker this box is. The API's claim gates key
#: on kinds, not on `provides` -- a gate keyed on names needs an allowlist, and an engine
#: missing from that allowlist claims nothing, indistinguishable from an empty queue. So the
#: mapping lives here, in the one place that decides what to register as. image-description,
#: face-crop and face-edit add no kind: they are called, they claim nothing.
KIND_BY_SERVICE = {"ltx-engine": "render", "lora-trainer": "trainer"}


def kinds_for(names: list[str]) -> list[str]:
    """What this box registers as, render first (wanly-api's `kind` is kinds[0]). A box that
    runs only image-description is a `service`: it takes no work of any kind."""
    out = [KIND_BY_SERVICE[n] for n in names if n in KIND_BY_SERVICE]
    if "render" in out:
        out = ["render", *[k for k in out if k != "render"]]
    return out or ["service"]


class ConfigError(RuntimeError):
    """SERVICES named something this image cannot run, or MODE is not a mode."""


#: MODE selects WHICH OF `SERVICES` ACTUALLY RUNS, without changing what the box is for.
#:
#: SERVICES is a property of the BOX -- everything this machine is equipped to do, set once
#: and left alone. MODE is a property of RIGHT NOW. Keeping them separate is what makes the
#: switch a plain `docker run -e MODE=motion` with the box's own SERVICES line untouched,
#: instead of an edit that has to remember the full list to put back.
#:
#: FOUR MODES, ONE TENANT EACH (wanly-gpu-docker#164, wanly-console#572). A 3090 holds one
#: big model at a time -- an LTX render, the trainer, the 32B Qwen3-VL motion captioner, or
#: Qwen-Image-Edit, each ~14-21 GB of 24 -- so a mode IS the choice of tenant:
#:
#:     render   ltx-engine (and, until wanly-gpu-docker#165, the trainer riding along on its drain)
#:     train    lora-trainer, nothing else that claims work
#:     motion   image-description: every enabled service that claims no work
#:     edit     image-edit
#:
#: face-crop, face-edit and scene-caption run in every mode: CPU-first or on another card.
#:
#: `motion` is DERIVED, not a hardcoded pair of names: it is every enabled service that claims
#: no work. With nothing on the box that can claim, queued jobs simply wait and the GPU belongs
#: to the captioner. A service added to KIND_BY_SERVICE later is excluded automatically.
MODES = ("render", "train", "motion", "edit")
#: The names the modes had before #164, and other spellings people type. Accepted everywhere a
#: mode is read -- POST /mode, MODE in worker.env, run-worker.sh -- so nothing that already
#: says `ltx-engine` or `caption` breaks.
_MODE_ALIASES = {"ltx-engine": "render", "engine": "render",
                 "caption": "motion", "image-caption": "motion", "image-description": "motion",
                 "motion-caption": "motion",
                 "training": "train", "trainer": "train", "lora-trainer": "train",
                 "image-edit": "edit", "full-edit": "edit"}

#: THE SPELLING THE CONTRACT STILL USES. wanly-api and the console compare the reported mode
#: against `ltx-engine` and `caption` (the captioner refusal in app/joycaption.py, the Workers
#: page toggle), so /health keeps reporting those for render and motion until wanly-api#392 and
#: wanly-console#589 read `mode_name`. Reporting the new words today would make every caption
#: on a rendering box look allowed and leave the toggle with nothing selected.
LEGACY_NAME = {"render": "ltx-engine", "motion": "caption"}


def legacy_name(mode: str | None) -> str | None:
    """`mode` in the spelling wanly-api and the console compare against (see LEGACY_NAME)."""
    return LEGACY_NAME.get(mode, mode) if mode else mode


#: Services that run ONLY in one mode (console#548). image-edit is Qwen-Image-Edit: ~20 GB of
#: a 24 GB card, the same class of tenant as a render or a Qwen captioner, so it cannot sit
#: beside either and has no CPU fallback to retreat to. It is therefore not part of "everything
#: this box is equipped for" in render mode, nor a "claims no work" service in motion mode --
#: it is the card's tenant in edit mode and nowhere else.
MODE_ONLY = {"image-edit": "edit"}

#: What each mode leaves out on top of the claiming services. Edit and train drop the captioner
#: for the reason MODE_ONLY exists: a caption landing mid-edit or mid-run would load a 20 GB
#: vision model onto a card that already has its tenant. face-crop and face-edit stay --
#: CPU-first and ~2 GB at most.
_MODE_EXCLUDES = {"edit": {"image-description"}, "train": {"image-description"}}


def canonical_mode(raw: str | None) -> str:
    """The mode's one true spelling, so callers compare modes and not spellings.

    Unset is `render`: a box with no MODE runs everything, which IS render mode. Saying so
    explicitly keeps "am I already in this mode?" a string comparison rather than a special
    case for empty.
    """
    mode = (raw or "").strip().lower()
    if not mode:
        return "render"
    return _MODE_ALIASES.get(mode, mode)


#: The env var that scopes scene-caption to some modes (wanly-gpu-docker#199). Unset: every
#: mode, which is right for its own card (the 2070). Set, e.g. `edit`: only those modes.
SCENE_CAPTION_MODES_ENV = "SCENE_CAPTION_MODES"


def scene_caption_modes(raw: str | None) -> set[str] | None:
    """The modes scene-caption may run in, or None for all of them (the default).

    Exists for a box whose ONE card scene-caption shares with everything else -- 3090b until
    the 2070 is in. JoyCaption (~6 GB) is resident; it yields to image-edit per edit
    (SCENE_CAPTION_SHARED) but knows nothing about a render, so beside a 23 GB LTX render it
    would OOM the card. `SCENE_CAPTION_MODES=edit` keeps it to the mode it knows how to share.
    An unknown mode refuses: a typo here would otherwise drop scene captions from every mode
    and look exactly like a captioner that is down.
    """
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        return None
    out = {canonical_mode(p) for p in parts}
    bad = sorted(m for m in out if m not in MODES)
    if bad:
        raise ConfigError(
            f"{SCENE_CAPTION_MODES_ENV}={raw!r} names {', '.join(bad)}, which "
            f"{'is not a mode' if len(bad) == 1 else 'are not modes'}. "
            f"Known modes: {', '.join(MODES)}")
    return out


def select_mode(names: list[str], raw: str | None) -> list[str]:
    """Narrow `names` to the services MODE asks for, then drop scene-caption from any mode
    SCENE_CAPTION_MODES leaves it out of (#199). Unset means render mode."""
    picked = _select_mode(names, raw)
    allowed = scene_caption_modes(os.environ.get(SCENE_CAPTION_MODES_ENV))
    if allowed is None or canonical_mode(raw) in allowed:
        return picked
    return [n for n in picked if n != "scene-caption"]


def _select_mode(names: list[str], raw: str | None) -> list[str]:
    """Narrow `names` to the services MODE asks for. Unset means render mode.

    Order is preserved -- see parse_services; services start in the order given.
    """
    mode = canonical_mode(raw)
    if mode not in MODES:
        raise ConfigError(
            f"MODE={raw!r} is not a mode. Known modes: {', '.join(MODES)} "
            f"(aliases: {', '.join(sorted(_MODE_ALIASES))})"
        )
    # A service tied to another mode never runs here. Render mode is otherwise everything.
    mine = [n for n in names if MODE_ONLY.get(n, mode) == mode]
    if mode == "render":
        # A STANDING box -- nothing on it claims work (no render stack, no trainer) -- runs
        # everything it is equipped with in its default mode, mode-bound services included:
        # there is no render for image-edit to collide with, and nothing to switch to.
        # 3090b is SERVICES=image-edit,scene-caption (wanly-console#572); the old rule
        # ("only if nothing else is left") dropped image-edit the moment a second service
        # joined it.
        if not any(n in KIND_BY_SERVICE for n in names):
            return list(names)
        # NO CAPTIONER BESIDE THE RENDER STACK (wanly-gpu-docker#173). On 2026-10-03 a caption
        # landing mid-render loaded qwen3-vl 32B (~21.5 GB anon RAM, UseMmap:false) next to
        # ComfyUI (~33 GB); the container hit its 54 GiB memory cap and the kernel OOM-killed
        # ComfyUI, failing the render. Render and the 32B captioner cannot share one box's RAM
        # or its 24 GB card, so on a box that renders, the captioner runs in motion mode only.
        # The API-side guard (refuse captions while rendering) stays, but it matched boxes by
        # hostname and let this one through; the box itself is the authority now.
        rendering = [n for n in mine if n != "image-description"]
        return rendering or mine or list(names)
    excluded = _MODE_EXCLUDES.get(mode, set())
    if mode == "train":
        # The trainer and what claims nothing -- never the render stack, which is the other
        # claiming tenant and would take the card the run needs.
        if "lora-trainer" not in names:
            raise ConfigError(
                f"MODE=train needs lora-trainer in SERVICES (SERVICES={','.join(names)}). "
                f"Add it, with the trainer mounts -- see deploy/README.md.")
        return [n for n in mine
                if (n == "lora-trainer" or n not in KIND_BY_SERVICE) and n not in excluded]
    kept = [n for n in mine if n not in KIND_BY_SERVICE and n not in excluded]
    if mode == "edit" and "image-edit" not in kept:
        raise ConfigError(
            f"MODE=edit needs image-edit in SERVICES (SERVICES={','.join(names)}). Add it, "
            f"with the Qwen checkpoint mounted -- see deploy/README.md."
        )
    if not kept:
        # Refused rather than silently started empty, for the reason parse_services refuses
        # an empty list: a container running nothing boots clean, reports healthy and serves
        # nothing, and is diagnosed from another machine as "captioner unreachable".
        raise ConfigError(
            f"MODE={mode} leaves nothing to run: SERVICES={','.join(names)} contains only "
            f"services that claim work ({', '.join(sorted(KIND_BY_SERVICE))}). Add "
            f"image-description (and/or face-crop, face-edit) to SERVICES, or drop MODE."
        )
    return kept


def parse_services(raw: str | None, known: dict | None = None) -> list[str]:
    """Turn the SERVICES flag into an ordered, de-duplicated list of known names.

    Order is preserved because services start in the order given, and a later service may
    reasonably want an earlier one already up.
    """
    catalogue = KNOWN if known is None else known
    names: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in catalogue:
            raise ConfigError(
                f"SERVICES names {name!r}, which this image does not have. "
                f"Known services: {', '.join(sorted(catalogue)) or 'none'}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ConfigError(
            f"SERVICES is empty. A services container that runs no services is a "
            f"misconfiguration, not a valid state. Known services: "
            f"{', '.join(sorted(catalogue)) or 'none'}"
        )
    return names


def build(names: list[str], known: dict | None = None) -> list[Service]:
    """The processes to run, in order. A factory may return one Service or several."""
    catalogue = KNOWN if known is None else known
    out: list[Service] = []
    for n in names:
        made = catalogue[n]()
        made = list(made) if isinstance(made, (list, tuple)) else [made]
        for svc in made:
            svc.group = svc.group or n
        out.extend(made)
    return out
