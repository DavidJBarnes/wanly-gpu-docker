"""The scene-caption front: ollama's wire, one model, always resident, and a yield for a
shared card (wanly-console#572).

    POST /api/generate   ollama's own request, passed to the loopback ollama -- the wire
                         wanly-api already speaks (app/joycaption.py), so its scene_caption_url
                         points here and nothing else changes. Only joycaption:beta-one is
                         served, and keep_alive is always -1: this service exists to keep one
                         small model on the card.
    GET  /api/tags, /api/version, /api/ps   passed through, for anyone poking at it.
    POST /yield  {reason, hold_s}   unload JoyCaption so an image edit can have the card.
    POST /resume                    load it again.
    GET  /health                    up, resident, yielded and why, caption counts.

WHY A FRONT AT ALL, rather than ollama on the public port

    Because of the yield. On the INTERIM box (3090b) JoyCaption shares a 24 GB card with
    Qwen-Image-Edit, whose ~23.5 GB peak cannot sit beside it. Unloading through ollama's
    keep_alive:0 is easy; KEEPING it unloaded is not -- the next caption would load it straight
    back onto a card Qwen is using. So captions come through here, and while the card is lent
    out they WAIT (up to SCENE_CAPTION_YIELD_WAIT_S) instead of loading. Past that they get a
    503, and wanly-api falls back to the old single-captioner path.

THE SHARING RULE, set by SCENE_CAPTION_SHARED (mirrors image_edit/share.py, from the other side)

    SHARED=1 (3090b, interim): image-edit calls /yield before every edit (wanly_worker/
        services/image_edit/share.py) and /resume once edits stop and Qwen has left the card.
        A caption in flight finishes first -- it is seconds -- and then the model is dropped.
        The yield is a LEASE (hold_s): an image-edit that dies mid-edit cannot strand the
        captioner, because the lease runs out and the model comes back on its own. Every edit
        renews it.
    SHARED=0 (the default; the future dedicated 2070): /yield answers "never yields" and does
        nothing. Nothing else shares that card.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL = "joycaption:beta-one"
OLLAMA_PORT = int(os.environ.get("SCENE_CAPTION_OLLAMA_PORT", "11437"))
OLLAMA = f"http://127.0.0.1:{OLLAMA_PORT}"
#: The flag. Off by default: a dedicated card has nobody to yield to.
SHARED = os.environ.get("SCENE_CAPTION_SHARED", "0").strip() == "1"
#: How long a caption waits for a lent-out card before it is refused (503 -> wanly-api falls
#: back). Under wanly-api's scene_caption_timeout_s (900) on purpose, so the refusal arrives as
#: a refusal and not as a timeout.
YIELD_WAIT_S = float(os.environ.get("SCENE_CAPTION_YIELD_WAIT_S", "600"))
#: How long /yield lets a caption in flight finish before unloading anyway.
DRAIN_S = float(os.environ.get("SCENE_CAPTION_DRAIN_S", "60"))
#: The default lease when /yield names none: an edit plus its timeout, with room.
DEFAULT_HOLD_S = float(os.environ.get("SCENE_CAPTION_YIELD_HOLD_S", "1800"))
#: One caption can take this long, load included (a cold JoyCaption load is seconds).
CAPTION_TIMEOUT_S = float(os.environ.get("SCENE_CAPTION_TIMEOUT_S", "300"))
LEASE_CHECK_S = 5.0

#: One caption at a time: ollama runs one slot anyway, and holding the turn here is what lets
#: /yield wait for the caption in flight before it unloads.
_turn = asyncio.Lock()
#: Set while the card is ours; cleared while it is lent to an image edit.
_have_card = asyncio.Event()
_have_card.set()
_state: dict = {"yielded": False, "reason": None, "since": None, "until": 0.0,
                "yields": 0, "captions": 0, "failures": 0, "last_ms": None,
                "last_at": None, "waiting": 0, "last_resume": None}


def reset() -> None:
    """Fresh locks and state. Tests only: each TestClient runs its own event loop."""
    global _turn, _have_card
    _turn = asyncio.Lock()
    _have_card = asyncio.Event()
    _have_card.set()
    _state.update(yielded=False, reason=None, since=None, until=0.0, yields=0, captions=0,
                  failures=0, last_ms=None, last_at=None, waiting=0, last_resume=None)


async def _warm(keep_alive: int) -> bool:
    """Load (-1) or drop (0) the model with an empty prompt -- ollama's documented way."""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{OLLAMA}/api/generate",
                             json={"model": MODEL, "prompt": "", "keep_alive": keep_alive},
                             timeout=300)
        return r.status_code == 200
    except Exception as e:                      # noqa: BLE001 -- reported, never fatal
        print(f"[scene-caption] could not {'load' if keep_alive else 'unload'} {MODEL}: {e}",
              flush=True)
        return False


async def _resident() -> bool | None:
    """Is the model on the card right now? None when ollama will not say."""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{OLLAMA}/api/ps", timeout=5)
        return any((m.get("name") or m.get("model")) == MODEL
                   for m in (r.json().get("models") or []))
    except Exception:                           # noqa: BLE001
        return None


def _resume(why: str) -> None:
    """Take the card back: captions flow again, and the model is reloaded in the background
    so the first caption after an edit does not pay the load."""
    if not _state["yielded"]:
        return
    held = time.time() - (_state["since"] or time.time())
    _state.update(yielded=False, reason=None, since=None, until=0.0,
                  last_resume={"at": round(time.time()), "why": why, "held_s": round(held)})
    _have_card.set()
    print(f"[scene-caption] resumed after {held:.0f}s yielded ({why}); reloading {MODEL}",
          flush=True)
    asyncio.get_running_loop().create_task(_warm(-1))


async def _lease_watch() -> None:
    while True:
        await asyncio.sleep(LEASE_CHECK_S)
        try:
            if _state["yielded"] and time.time() >= _state["until"]:
                _resume("the yield lease ran out -- image-edit never resumed it")
        except asyncio.CancelledError:
            raise
        except Exception as e:                  # noqa: BLE001 -- a watcher must not die
            print(f"[scene-caption] lease watcher: {e}", flush=True)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    task = asyncio.create_task(_lease_watch()) if SHARED else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(title="wanly scene-caption", lifespan=_lifespan)


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model = body.get("model") or MODEL
    if model != MODEL:
        # Not a pass-through for anything in the store: asking for another model would load
        # it beside (or instead of) JoyCaption on a card sized for one small model.
        raise HTTPException(400, f"scene-caption serves only {MODEL}, not {model!r}")
    body["model"] = MODEL
    body["keep_alive"] = -1
    deadline = time.monotonic() + YIELD_WAIT_S
    _state["waiting"] += 1
    try:
        while True:
            if not _have_card.is_set():
                left = deadline - time.monotonic()
                try:
                    await asyncio.wait_for(_have_card.wait(), timeout=max(left, 0.0))
                except asyncio.TimeoutError:
                    raise HTTPException(
                        503, f"scene-caption has lent its card to {_state['reason']} for "
                             f"{time.time() - (_state['since'] or time.time()):.0f}s; "
                             f"not captioning until it is back") from None
            await _turn.acquire()
            if _have_card.is_set():
                break
            _turn.release()             # lent out while we queued for the turn: wait again
    finally:
        _state["waiting"] -= 1
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{OLLAMA}/api/generate", json=body, timeout=CAPTION_TIMEOUT_S)
    except httpx.HTTPError as e:
        _state["failures"] += 1
        raise HTTPException(502, f"ollama did not answer: {e!r}") from e
    finally:
        _turn.release()
    if r.status_code == 200:
        _state.update(captions=_state["captions"] + 1, last_at=round(time.time()),
                      last_ms=round((time.monotonic() - t0) * 1000))
    else:
        _state["failures"] += 1
    try:
        payload = r.json()
    except ValueError:
        payload = {"error": r.text[:500]}
    return JSONResponse(payload, status_code=r.status_code)


async def _passthrough(path: str):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{OLLAMA}{path}", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:                      # noqa: BLE001
        raise HTTPException(502, f"ollama did not answer: {e!r}") from e


@app.get("/api/tags")
async def tags():
    return await _passthrough("/api/tags")


@app.get("/api/version")
async def version():
    return await _passthrough("/api/version")


@app.get("/api/ps")
async def ps():
    return await _passthrough("/api/ps")


@app.post("/yield")
async def yield_card(request: Request):
    """Lend the card: no new captions start, the one in flight finishes, the model drops."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    reason = str((body or {}).get("reason") or "an image edit")[:200]
    if not SHARED:
        return {"yielded": False, "shared": False,
                "note": "SCENE_CAPTION_SHARED is off: this card is the captioner's alone"}
    hold_s = float((body or {}).get("hold_s") or DEFAULT_HOLD_S)
    renewing = _state["yielded"]
    _state.update(yielded=True, reason=reason, until=time.time() + hold_s,
                  since=_state["since"] or time.time())
    _have_card.clear()
    if renewing:
        return {"yielded": True, "renewed": True, "unloaded": await _resident() is False}
    _state["yields"] += 1
    t0 = time.monotonic()
    drained = True
    try:
        await asyncio.wait_for(_turn.acquire(), timeout=DRAIN_S)
    except asyncio.TimeoutError:
        drained = False                         # unload anyway: the edit will not wait longer
    else:
        _turn.release()
    unloaded = await _warm(0)
    for _ in range(20):                         # ollama drops the runner asynchronously
        if await _resident() is not True:
            break
        await asyncio.sleep(0.5)
    resident = await _resident()
    print(f"[scene-caption] yielded the card to {reason} (unloaded={unloaded}, "
          f"resident={resident}, drained={drained}, waited "
          f"{time.monotonic() - t0:.1f}s)", flush=True)
    return {"yielded": True, "renewed": False, "unloaded": unloaded and resident is not True,
            "drained": drained, "waited_s": round(time.monotonic() - t0, 1)}


@app.post("/resume")
async def resume():
    was = _state["yielded"]
    _resume("image-edit handed the card back")
    return {"resumed": was, "shared": SHARED}


@app.get("/health")
async def health():
    up = False
    try:
        async with httpx.AsyncClient() as c:
            up = (await c.get(f"{OLLAMA}/api/version", timeout=5)).status_code == 200
    except Exception:                           # noqa: BLE001
        up = False
    since = _state["since"]
    return JSONResponse({
        "status": "ok" if up else "degraded",
        "ollama_up": up,
        "model": MODEL,
        "resident": await _resident() if up else None,
        "shared": SHARED,
        "yielded": _state["yielded"],
        "yielded_to": _state["reason"],
        "yielded_for_s": round(time.time() - since) if since else None,
        "lease_left_s": (round(_state["until"] - time.time()) if _state["yielded"] else None),
        "waiting": _state["waiting"],
        "busy": _turn.locked(),
        "captions": _state["captions"],
        "failures": _state["failures"],
        "yields": _state["yields"],
        "last_ms": _state["last_ms"],
        "last_at": _state["last_at"],
        "last_resume": _state["last_resume"],
    }, status_code=200 if up else 503)
