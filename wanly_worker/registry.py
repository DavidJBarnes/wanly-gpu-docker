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
    # yields per edit (SCENE_CAPTION_SHARED) rather than being stopped by a mode.
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
#: switch a plain `docker run -e MODE=caption` with the box's own SERVICES line untouched,
#: instead of an edit that has to remember the full list to put back.
#:
#: `caption` is DERIVED, not a hardcoded pair of names: it is every enabled service that
#: claims no work. That is exactly the property that matters -- with nothing on the box that
#: can claim, queued jobs simply wait and the GPU belongs to the captioner. A service added
#: to KIND_BY_SERVICE later is excluded automatically, which is the right default: anything
#: that takes work does not belong in caption mode.
MODES = ("ltx-engine", "caption", "edit")
_MODE_ALIASES = {"render": "ltx-engine", "engine": "ltx-engine",
                 "image-caption": "caption", "image-description": "caption",
                 "image-edit": "edit", "full-edit": "edit"}

#: Services that run ONLY in one mode (console#548). image-edit is Qwen-Image-Edit: ~20 GB of
#: a 24 GB card, the same class of tenant as a render or a Qwen captioner, so it cannot sit
#: beside either and has no CPU fallback to retreat to. It is therefore not part of "everything
#: this box is equipped for" in render mode, nor a "claims no work" service in caption mode --
#: it is the card's tenant in edit mode and nowhere else.
MODE_ONLY = {"image-edit": "edit"}

#: What each mode leaves out on top of the claiming services. Edit mode drops the captioner for
#: the reason MODE_ONLY exists: a caption landing mid-edit would load a 20 GB vision model onto
#: a card Qwen is already holding. face-crop and face-edit stay -- CPU-first and ~2 GB at most.
_MODE_EXCLUDES = {"edit": {"image-description"}}


def canonical_mode(raw: str | None) -> str:
    """The mode's one true spelling, so callers compare modes and not spellings.

    Unset is `ltx-engine`: a box with no MODE runs everything, which IS render mode. Saying
    so explicitly keeps "am I already in this mode?" a string comparison rather than a
    special case for empty.
    """
    mode = (raw or "").strip().lower()
    if not mode:
        return "ltx-engine"
    return _MODE_ALIASES.get(mode, mode)


def select_mode(names: list[str], raw: str | None) -> list[str]:
    """Narrow `names` to the services MODE asks for. Unset means render mode.

    Order is preserved -- see parse_services; services start in the order given.
    """
    mode = (raw or "").strip().lower()
    mode = _MODE_ALIASES.get(mode, mode) if mode else "ltx-engine"
    if mode not in MODES:
        raise ConfigError(
            f"MODE={raw!r} is not a mode. Known modes: {', '.join(MODES)} "
            f"(aliases: {', '.join(sorted(_MODE_ALIASES))})"
        )
    # A service tied to another mode never runs here. Render mode is otherwise everything.
    mine = [n for n in names if MODE_ONLY.get(n, mode) == mode]
    if mode == "ltx-engine":
        # A STANDING box -- nothing on it claims work (no render stack, no trainer) -- runs
        # everything it is equipped with in its default mode, mode-bound services included:
        # there is no render for image-edit to collide with, and nothing to switch to.
        # 3090b is SERVICES=image-edit,scene-caption (wanly-console#572); the old rule
        # ("only if nothing else is left") dropped image-edit the moment a second service
        # joined it.
        if not any(n in KIND_BY_SERVICE for n in names):
            return list(names)
        return mine or list(names)
    excluded = _MODE_EXCLUDES.get(mode, set())
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
