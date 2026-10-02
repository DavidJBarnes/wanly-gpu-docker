"""Sharing one 24 GB card with Automatic1111: the STANDING image-edit box (console#570).

On the main 3090 image-edit is the card's only tenant -- edit mode stopped everything else
first -- and none of this applies (IMAGE_EDIT_A1111_URL unset, IMAGE_EDIT_UNLOAD_IDLE_S 0). On
the second 3090 (ex-2070) it runs full-time beside Automatic1111, and Qwen at ~23.5 GB peak
cannot sit beside an SDXL generation. So the card is taken in turns, by this rule:

    1. A1111 IS NEVER INTERRUPTED. An edit waits while A1111 is generating (its own progress
       endpoint), up to IMAGE_EDIT_A1111_WAIT_S, and says so on /health -- wanly-api shows it
       as "second 3090 busy (A1111 generating)". Past the wait the edit is refused (503), not
       forced: a Qwen load under a live generation OOMs one of the two.
    2. THEN IT MAKES ROOM, only when it has to: with Qwen not resident and less than
       IMAGE_EDIT_MIN_FREE_MIB free, an idle A1111 is asked to unload its checkpoint -- the
       face-edit service's yield_a1111, reused (it re-checks "generating" right before the
       unload, since generate-forever can start the next image in between). A1111 reloads it
       from RAM in seconds on its next generation.
    3. A1111 GETS THE CARD BACK AS SOON AS IT WANTS IT. With no edit running, Qwen is unloaded
       (ComfyUI POST /free) the moment A1111 starts generating, or after
       IMAGE_EDIT_UNLOAD_IDLE_S without an edit, whichever comes first. An edit in progress is
       never cut short: once started, it has the card until it finishes (~15 s warm).

The one overlap this cannot prevent is generate-forever starting its next image DURING an edit
(it re-clicks Generate on its own timer). That image then runs in what Qwen leaves free -- slow,
or an OOM for that one image, which generate-forever shrugs off -- and never the other way
round: ComfyUI has already loaded. For a long run of edits, pause generate-forever.

Free VRAM is read through nvidia-smi (the supervisor's gpu_snapshot), never torch, for the
reason face_edit/gpu.py gives: this process must not hold a CUDA context of its own.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from wanly_worker.services.face_edit import gpu as fe_gpu

log = logging.getLogger("image-edit")

#: Automatic1111 on the same card, as seen from inside the container. Empty = none (the main
#: 3090, where A1111 is gone). On the second 3090: http://host.docker.internal:7860.
A1111_URL = os.environ.get("IMAGE_EDIT_A1111_URL", "").strip().rstrip("/")
#: How long an edit waits for A1111 to finish generating before it is refused. wanly-api does
#: the long wait itself (it reads `a1111_generating` off /health and only then sends the edit),
#: so this is the backstop for the race between its look and the edit arriving.
A1111_WAIT_S = float(os.environ.get("IMAGE_EDIT_A1111_WAIT_S", "300"))
#: How often to look. Generate-forever leaves a gap of its configured delay between images
#: (a1111-tweaks' tweaks_generate_forever_delay, 2 s by default); polling faster than that is
#: what lets an edit start in the gap instead of waiting out the whole run.
A1111_POLL_S = float(os.environ.get("IMAGE_EDIT_A1111_POLL_S", "0.5"))
#: Below this much free VRAM (with Qwen not resident) an idle A1111 is asked to unload. Qwen's
#: measured peak is 23.1-23.7 GB of a 24 GB card, so in practice "anything else on the card".
MIN_FREE_MIB = int(os.environ.get("IMAGE_EDIT_MIN_FREE_MIB", "21504"))
#: Seconds without an edit before Qwen is unloaded. 0 = stay resident (the main 3090 in edit
#: mode, and a standing box with the card to itself: every edit then skips the ~9 s reload).
UNLOAD_IDLE_S = float(os.environ.get("IMAGE_EDIT_UNLOAD_IDLE_S", "0"))
#: How often the watcher looks at A1111 and at the idle time.
WATCH_S = float(os.environ.get("IMAGE_EDIT_WATCH_S", "2"))


# ------------------------------------------------------------------ the scene captioner
#
# THE OTHER TENANT ON 3090b (wanly-console#572 phase 1, interim). scene-caption keeps
# JoyCaption (~6 GB) resident on this card, and Qwen's ~23.5 GB peak cannot sit beside it. So,
# per edit: the scene captioner YIELDS FIRST (unloads through ollama keep_alive:0 and holds new
# captions back), and once edits stop -- SCENE_RESUME_IDLE_S without one -- Qwen leaves the card
# and the captioner is RESUMED. A run of edits pays for one yield, not one each; every edit
# renews the yield's lease, and the lease lets the captioner take the card back by itself if this
# process dies mid-run.
#
# Set by SCENE_CAPTION_SHARED=1 with scene-caption on this box's SERVICES (its loopback front),
# or by IMAGE_EDIT_SCENE_CAPTION_URL explicitly. Off everywhere else -- the future dedicated 2070
# shares its card with nobody.


def _scene_url_from_env() -> str:
    explicit = os.environ.get("IMAGE_EDIT_SCENE_CAPTION_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    services = {n.strip().lower() for n in os.environ.get("SERVICES", "").split(",")}
    if os.environ.get("SCENE_CAPTION_SHARED", "0").strip() == "1" and "scene-caption" in services:
        return f"http://127.0.0.1:{os.environ.get('SCENE_CAPTION_PORT', '11436')}"
    return ""


SCENE_CAPTION_URL = _scene_url_from_env()
#: Seconds without an edit before Qwen is unloaded and the scene captioner gets the card back.
SCENE_RESUME_IDLE_S = float(os.environ.get("IMAGE_EDIT_SCENE_RESUME_IDLE_S", "60"))
#: The yield's lease: longer than an edit's timeout, so only a dead image-edit ever hits it.
SCENE_YIELD_HOLD_S = float(os.environ.get("IMAGE_EDIT_SCENE_YIELD_HOLD_S", "1800"))


def shares_with_scene() -> bool:
    return bool(SCENE_CAPTION_URL)


async def yield_scene(state: dict, client: httpx.AsyncClient | None = None) -> str | None:
    """Ask the scene captioner for the card before an edit. What happened, for the log.

    Never fatal: a scene captioner that does not answer is not holding the card either (or is
    about to fall over on its own), and the edit is the thing a person is waiting for.
    """
    if not SCENE_CAPTION_URL:
        return None
    payload = {"reason": "an image edit", "hold_s": SCENE_YIELD_HOLD_S}
    try:
        if client is None:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{SCENE_CAPTION_URL}/yield", json=payload, timeout=120)
        else:
            r = await client.post(f"{SCENE_CAPTION_URL}/yield", json=payload, timeout=120)
        body = r.json() if r.status_code == 200 else {}
    except Exception as e:                      # noqa: BLE001
        state["scene_yielded"] = False
        return f"scene-caption did not answer /yield ({e}); editing anyway"
    first = not state.get("scene_yielded")
    state["scene_yielded"] = bool(body.get("yielded"))
    if not first:
        return None
    return (f"scene-caption yielded the card (unloaded={body.get('unloaded')}, "
            f"waited {body.get('waited_s')}s)" if body.get("yielded")
            else f"scene-caption did not yield: {body.get('note') or r.status_code}")


async def resume_scene(state: dict, client: httpx.AsyncClient | None = None) -> bool:
    """Give the scene captioner the card back. False on failure (retried next tick; the
    lease is the backstop)."""
    if not SCENE_CAPTION_URL:
        return False
    try:
        if client is None:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{SCENE_CAPTION_URL}/resume", timeout=30)
        else:
            r = await client.post(f"{SCENE_CAPTION_URL}/resume", timeout=30)
    except Exception as e:                      # noqa: BLE001
        log.warning("could not resume scene-caption: %s", e)
        return False
    if r.status_code == 200:
        state["scene_yielded"] = False
        return True
    return False


def scene_resume_reason(*, scene_yielded: bool, busy: bool, idle_s: float) -> str | None:
    """Why the scene captioner should get the card back now, or None."""
    if not scene_yielded or busy:
        return None
    if idle_s >= SCENE_RESUME_IDLE_S:
        return f"no edit for {idle_s:.0f}s"
    return None


class A1111Busy(RuntimeError):
    """A1111 kept generating for the whole wait. Becomes a 503 that says so."""


def shared() -> bool:
    return bool(A1111_URL)


async def a1111_generating() -> bool | None:
    """None when there is no A1111 on this card, else whether it is generating now."""
    if not A1111_URL:
        return None
    return await asyncio.to_thread(fe_gpu.a1111_generating, A1111_URL)


async def wait_for_a1111(state: dict) -> float:
    """Block until A1111 is not generating; the seconds waited. `state["waiting"]` carries the
    reason while it waits, for /health. A1111Busy past A1111_WAIT_S."""
    if not A1111_URL:
        return 0.0
    t0 = time.monotonic()
    try:
        while await a1111_generating():
            waited = time.monotonic() - t0
            if waited >= A1111_WAIT_S:
                raise A1111Busy(f"Automatic1111 on this card has been generating for "
                                f"{waited:.0f}s; the edit was not started (it never interrupts "
                                f"a generation -- pause generate-forever to let edits in)")
            if not state.get("waiting"):
                log.info("A1111 is generating; the edit waits for it")
            state["waiting"] = "Automatic1111 on this card is generating"
            await asyncio.sleep(A1111_POLL_S)
    finally:
        state["waiting"] = None
    return time.monotonic() - t0


async def make_room(resident: bool) -> str | None:
    """Ask an idle A1111 to unload when Qwen needs the space. What was done, for the log."""
    if not A1111_URL or resident:
        return None
    free = await asyncio.to_thread(fe_gpu.free_mib)
    if free is None or free >= MIN_FREE_MIB:
        return None
    if await asyncio.to_thread(fe_gpu.yield_a1111, A1111_URL, "a Qwen image edit"):
        return f"A1111 unloaded its checkpoint ({free} MiB free before)"
    # Not fatal: ComfyUI loads what fits and offloads the rest -- slower, not broken.
    return f"only {free} MiB free and A1111 did not unload; ComfyUI will offload what does not fit"


async def unload_qwen(comfy_url: str, client: httpx.AsyncClient | None = None) -> bool:
    """ComfyUI's own unload: models out of VRAM, cache freed. The process stays up, so the next
    edit pays a reload from page cache (~9 s measured), not a boot."""
    try:
        if client is None:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{comfy_url}/free", json={"unload_models": True,
                                                            "free_memory": True}, timeout=30)
        else:
            r = await client.post(f"{comfy_url}/free", json={"unload_models": True,
                                                             "free_memory": True}, timeout=30)
        return r.status_code == 200
    except Exception as e:                          # noqa: BLE001 -- logged, retried next tick
        log.warning("could not unload Qwen from ComfyUI: %s", e)
        return False


def unload_reason(*, resident: bool, busy: bool, idle_s: float,
                  a1111_generating: bool | None) -> str | None:
    """Why Qwen should leave the card now, or None. Rule 3 above, as a pure function."""
    if not resident or busy:
        return None
    if a1111_generating:
        return "Automatic1111 started generating"
    if UNLOAD_IDLE_S > 0 and idle_s >= UNLOAD_IDLE_S:
        return f"no edit for {idle_s:.0f}s"
    return None


def watching() -> bool:
    """Whether the watcher has anything to do on this box."""
    return bool(A1111_URL) or UNLOAD_IDLE_S > 0 or shares_with_scene()
