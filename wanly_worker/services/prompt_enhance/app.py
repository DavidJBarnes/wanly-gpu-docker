"""The prompt-enhance front: one request at a time, the model loaded only while it is wanted
(wanly-gpu-docker#225).

    POST /enhance  {text, image_b64?, max_tokens?}  -> {prompt, ms, cold}
                   `text` is what we know about the frame (wanly-api sends the image's scene
                   description); `image_b64` is the frame itself. No system prompt: the model
                   card says to send the text and the image and nothing else.
    POST /unload   stop llama-server now and hand the card back (else it waits IDLE_S).
    GET  /health   up, loaded, busy, counts, the last error.

WHY ON DEMAND, AND WHY THE YIELD

    The 2070 has 8 GB. JoyCaption (scene-caption, always resident) holds ~6.2 GB of it and the
    enhancer needs ~6.5 GB, so the two take turns. An enhance asks scene-caption to /yield --
    the lease mechanism image-edit already uses on 3090b -- starts llama-server, answers, and
    after PROMPT_ENHANCE_IDLE_S without another request stops llama-server and /resumes
    scene-caption. A run of enhances pays one load (~10 s from the page cache), not one each.
    Captions that arrive meanwhile wait in scene-caption (up to its YIELD_WAIT_S) rather than
    loading JoyCaption onto a card that is lent out. If this process dies, the lease runs out
    and JoyCaption comes back on its own.

    llama-server is this process's child, not a supervisor service, because it must NOT run
    all the time -- the supervisor's job is to keep things running.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import time
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request

from wanly_worker.services.prompt_enhance import model

PORT = int(os.environ.get("PROMPT_ENHANCE_PORT", "11438"))
LLAMA_PORT = int(os.environ.get("PROMPT_ENHANCE_LLAMA_PORT", "11439"))
LLAMA = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_SERVER = os.environ.get("PROMPT_ENHANCE_LLAMA_SERVER", "/opt/llama.cpp/llama-server")
#: Seconds without a request before the model leaves the card and JoyCaption gets it back.
IDLE_S = float(os.environ.get("PROMPT_ENHANCE_IDLE_S", "120"))
#: A cold load from the page cache is ~10 s; from disk on a busy box, longer.
LOAD_TIMEOUT_S = float(os.environ.get("PROMPT_ENHANCE_LOAD_TIMEOUT_S", "240"))
REQUEST_TIMEOUT_S = float(os.environ.get("PROMPT_ENHANCE_TIMEOUT_S", "300"))
#: Context. A prompt is ~1k tokens out and the image ~1-2k in; the model's 262k default would
#: spend the card's last gigabyte on KV cache nobody uses.
CTX = int(os.environ.get("PROMPT_ENHANCE_CTX", "8192"))
#: Thinking included: the model reasons first (llama-server returns that apart, as
#: reasoning_content) and spent ~600 tokens of it on a test image before an ~230-token prompt.
#: At 1024 the reasoning alone ran out once, leaving no prompt. Thinking is left on because
#: with it off the prompt invented details the image did not have. ~46 tok/s on the 2070.
MAX_TOKENS = int(os.environ.get("PROMPT_ENHANCE_MAX_TOKENS", "3072"))
#: Who to ask for the card. Default: scene-caption on this box, when it is on SERVICES.
WATCH_S = 5.0
LOG_PATH = os.environ.get("PROMPT_ENHANCE_LLAMA_LOG", "/tmp/prompt-enhance-llama.log")


def _scene_url() -> str:
    explicit = os.environ.get("PROMPT_ENHANCE_SCENE_CAPTION_URL", "").strip().rstrip("/")
    if explicit:
        return "" if explicit == "none" else explicit
    services = {n.strip().lower() for n in os.environ.get("SERVICES", "").split(",")}
    if "scene-caption" in services:
        return f"http://127.0.0.1:{os.environ.get('SCENE_CAPTION_PORT', '11436')}"
    return ""


SCENE_URL = _scene_url()

_turn = asyncio.Lock()
_proc: asyncio.subprocess.Process | None = None
_state: dict = {"loaded": False, "loads": 0, "enhances": 0, "failures": 0, "last_ms": None,
                "last_at": None, "last_used": 0.0, "last_error": None, "scene_yielded": False,
                "unloads": 0}


def reset() -> None:
    """Fresh lock and state. Tests only: each TestClient runs its own event loop."""
    global _turn, _proc
    _turn = asyncio.Lock()
    _proc = None
    _state.update(loaded=False, loads=0, enhances=0, failures=0, last_ms=None, last_at=None,
                  last_used=0.0, last_error=None, scene_yielded=False, unloads=0)


def command() -> list[str]:
    return [LLAMA_SERVER,
            "-m", str(model.path(model.WEIGHTS)), "--mmproj", str(model.path(model.MMPROJ)),
            "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
            "-ngl", "99", "-c", str(CTX), "-np", "1", "--jinja", "--no-webui"]


def strip_think(text: str) -> str:
    """The template closes thinking before the answer, but a model can still open it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def user_content(text: str, image_b64: str | None) -> list[dict] | str:
    if not image_b64:
        return text
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": text}]


async def _yield_scene(client: httpx.AsyncClient) -> None:
    if not SCENE_URL:
        return
    hold = IDLE_S + REQUEST_TIMEOUT_S + LOAD_TIMEOUT_S
    try:
        r = await client.post(f"{SCENE_URL}/yield", json={"reason": "prompt-enhance",
                                                          "hold_s": hold}, timeout=120)
        body = r.json() if r.status_code == 200 else {}
    except Exception as e:                      # noqa: BLE001 -- loading anyway is the answer
        print(f"[prompt-enhance] scene-caption did not answer /yield ({e})", flush=True)
        return
    if not _state["scene_yielded"]:
        print(f"[prompt-enhance] scene-caption: {body}", flush=True)
    _state["scene_yielded"] = bool(body.get("yielded"))


async def _resume_scene() -> None:
    if not SCENE_URL or not _state["scene_yielded"]:
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{SCENE_URL}/resume", timeout=30)
        if r.status_code == 200:
            _state["scene_yielded"] = False
    except Exception as e:                      # noqa: BLE001 -- the lease is the backstop
        print(f"[prompt-enhance] could not resume scene-caption: {e}", flush=True)


def _log_tail(n: int = 15) -> str:
    try:
        with open(LOG_PATH, errors="replace") as fh:
            return "".join(deque(fh, maxlen=n)).strip()
    except OSError:
        return ""


async def _up(client: httpx.AsyncClient) -> bool:
    try:
        return (await client.get(f"{LLAMA}/health", timeout=3)).status_code == 200
    except Exception:                           # noqa: BLE001
        return False


async def _ensure_loaded(client: httpx.AsyncClient) -> bool:
    """llama-server up and answering; True when this call had to start it."""
    global _proc
    if _proc is not None and _proc.returncode is None and await _up(client):
        await _yield_scene(client)              # renew the lease
        return False
    if model.missing():
        raise HTTPException(503, f"the enhancer's files are not in {model.MODEL_DIR} yet "
                                 f"(missing: {', '.join(f.local for f in model.missing())})")
    await _yield_scene(client)
    log = open(LOG_PATH, "ab")                  # noqa: SIM115 -- handed to the child
    lib = os.path.dirname(LLAMA_SERVER)
    env = {**os.environ,
           "LD_LIBRARY_PATH": f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")}
    _proc = await asyncio.create_subprocess_exec(*command(), stdout=log, stderr=log, env=env)
    log.close()
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        if _proc.returncode is not None:
            break
        if await _up(client):
            _state.update(loaded=True, loads=_state["loads"] + 1)
            return True
        await asyncio.sleep(1)
    await _stop()
    tail = _log_tail()
    raise HTTPException(503, f"llama-server did not come up within {LOAD_TIMEOUT_S:.0f}s"
                             + (f":\n{tail}" if tail else ""))


async def _stop() -> None:
    global _proc
    proc, _proc = _proc, None
    if proc is not None and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=20)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    if _state["loaded"]:
        _state.update(loaded=False, unloads=_state["unloads"] + 1)
    await _resume_scene()


async def _idle_watch() -> None:
    while True:
        await asyncio.sleep(WATCH_S)
        try:
            idle = time.time() - _state["last_used"]
            if _proc is not None and not _turn.locked() and idle >= IDLE_S:
                async with _turn:
                    print(f"[prompt-enhance] idle {idle:.0f}s: unloading, card back to "
                          f"scene-caption", flush=True)
                    await _stop()
        except asyncio.CancelledError:
            raise
        except Exception as e:                  # noqa: BLE001 -- a watcher must not die
            print(f"[prompt-enhance] idle watcher: {e}", flush=True)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    task = asyncio.create_task(_idle_watch())
    try:
        yield
    finally:
        task.cancel()
        await _stop()


app = FastAPI(title="wanly prompt-enhance", lifespan=_lifespan)


@app.post("/enhance")
async def enhance(request: Request):
    body = await request.json()
    text = str(body.get("text") or "").strip()
    image_b64 = body.get("image_b64") or None
    if not text and not image_b64:
        raise HTTPException(400, "send text, an image, or both")
    if image_b64:
        try:
            base64.b64decode(image_b64, validate=True)
        except Exception:                       # noqa: BLE001
            raise HTTPException(400, "image_b64 is not base64") from None
    max_tokens = int(body.get("max_tokens") or MAX_TOKENS)
    payload = {"messages": [{"role": "user", "content": user_content(text, image_b64)}],
               "max_tokens": max_tokens, "stream": False}
    async with _turn:
        _state["last_used"] = time.time()
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as c:
                cold = await _ensure_loaded(c)
                r = await c.post(f"{LLAMA}/v1/chat/completions", json=payload,
                                 timeout=REQUEST_TIMEOUT_S)
        except HTTPException as e:
            _state.update(failures=_state["failures"] + 1, last_error=str(e.detail)[:500])
            raise
        except httpx.HTTPError as e:
            _state.update(failures=_state["failures"] + 1, last_error=repr(e)[:500])
            raise HTTPException(502, f"llama-server did not answer: {e!r}") from e
        finally:
            _state["last_used"] = time.time()
        if r.status_code != 200:
            _state.update(failures=_state["failures"] + 1, last_error=r.text[:500])
            raise HTTPException(502, f"llama-server {r.status_code}: {r.text[:500]}")
        out = r.json()
        prompt = strip_think(out["choices"][0]["message"].get("content") or "")
        if not prompt:
            why = ("its reasoning used the whole max_tokens budget"
                   if out["choices"][0].get("finish_reason") == "length" else "no reason given")
            _state.update(failures=_state["failures"] + 1, last_error=f"empty answer: {why}")
            raise HTTPException(502, f"the enhancer answered with nothing ({why})")
        ms = round((time.monotonic() - t0) * 1000)
        _state.update(enhances=_state["enhances"] + 1, last_ms=ms, last_at=round(time.time()),
                      last_error=None)
        return {"prompt": prompt, "ms": ms, "cold": cold,
                "usage": out.get("usage"), "finish_reason": out["choices"][0].get("finish_reason")}


@app.post("/unload")
async def unload():
    async with _turn:
        was = _proc is not None
        await _stop()
    return {"unloaded": was}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "files_present": not model.missing(),
        "loaded": _state["loaded"],
        "busy": _turn.locked(),
        "idle_unload_s": IDLE_S,
        "shares_card_with": SCENE_URL or None,
        "scene_yielded": _state["scene_yielded"],
        "loads": _state["loads"],
        "unloads": _state["unloads"],
        "enhances": _state["enhances"],
        "failures": _state["failures"],
        "last_ms": _state["last_ms"],
        "last_at": _state["last_at"],
        "last_error": _state["last_error"],
    }
