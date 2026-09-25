"""The control plane: one endpoint that says what this container is actually doing.

This is the part wanly-console#426 is really asking for. The services it replaces cannot be
asked anything -- "is JoyCaption up?" is answered today by ssh'ing to 2070.zero and reading
`systemctl status`, and "is it the right model?" is not answerable at all without listing
ollama's tags by hand. A `/health` that names the enabled services, whether each is answering,
and what the card looks like turns both into one curl.

It is deliberately NOT a proxy for the services themselves. JoyCaption answers ollama's API on
its own port, unchanged, so wanly-api keeps working with no change at all -- see
services/joycaption.py for why that is the right trade today.
"""
from __future__ import annotations

import asyncio
import contextlib
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wanly_worker import registry
from wanly_worker.queue_client import QueueClient, export_identity
from wanly_worker.supervisor import Supervisor, gpu_snapshot

BUILD = os.environ.get("WANLY_IMAGE_REF", "unknown")
# The engine/supervisor code actually running, which since #116 is fetched from main at boot
# and is therefore NOT necessarily what the image was built with. "build" is the environment
# (the image), "code" is the software answering the queue — the #72 lesson is that one ref
# for both is how two boxes ran different code for fourteen hours.
CODE_REF_FILE = os.environ.get("CODE_REF_FILE", "/run/wanly/code_ref")


def _code_ref() -> str:
    try:
        with open(CODE_REF_FILE) as f:
            return f.read().strip()
    except OSError:
        return f"baked ({BUILD[:12]})"

_sup: Supervisor | None = None
_queue: QueueClient | None = None
_poller = None
#: Everything SERVICES said this box is equipped to run, and which subset is live. The
#: capability list never changes for the life of the container; the mode does.
_equipped: list[str] = []
_mode: str = ""
#: The lifespan's client, kept so POST /mode can use it for readiness probes. Starting a
#: service means waiting for it to answer, which needs one.
_client: httpx.AsyncClient | None = None
#: One mode change at a time. Two overlapping flips interleave stops and starts and can
#: leave the box with the render stack half up beside the captioner -- the exact collision
#: the stop-before-start ordering exists to prevent.
_mode_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the enabled services before the control API accepts a request.

    Starting inside the lifespan rather than in a shell entrypoint means a failure to start
    is a failure to boot: uvicorn never begins serving, the container exits non-zero, and
    `docker ps` shows it. An entrypoint that starts things and then execs the API would
    happily serve a healthy-looking /health beside a service that never came up.
    """
    global _sup
    print(f"=== wanly-gpu-docker === image build: {BUILD} | code: {_code_ref()}", flush=True)
    try:
        global _equipped, _mode
        equipped = registry.parse_services(os.environ.get("SERVICES"))
        _equipped = equipped
        # MODE narrows what actually runs; SERVICES stays the box's own capability line, so
        # the switch is `docker run -e MODE=caption` and nothing has to remember the full
        # list to put back afterwards.
        names = registry.select_mode(equipped, os.environ.get("MODE"))
        mode = (os.environ.get("MODE") or "").strip()
        _mode = registry.canonical_mode(os.environ.get("MODE"))
        print(f"SERVICES={','.join(names)}"
              + (f"  (MODE={mode} of {','.join(equipped)})" if mode else ""), flush=True)
        # Before any child starts: the render daemon reads these from its env and registers
        # the box with them (one row per box, wanly-gpu-docker#83). WORKER_ID_FILE too, and
        # for THIS process as much as for the daemon: the trainer's poller and its drain read
        # the box's own row id from that file, and without the variable in the control
        # process's env they saw no id and never claimed (Payton v1, first night).
        os.environ.update(export_identity(names))
        if "ltx-engine" in names:
            from wanly_worker.services.ltx_engine import WORKER_ID_FILE
            os.environ["WORKER_ID_FILE"] = WORKER_ID_FILE
        _sup = Supervisor(registry.build(names))
    except Exception as e:
        _fatal(e)
        raise
    async with httpx.AsyncClient() as client:
        global _client
        _client = client
        try:
            await _sup.start(client)
        except Exception as e:
            _fatal(e)
            raise
        # AFTER the services are up, so the first thing the API hears is the truth. Registering
        # before would advertise a box that is still staging, and a Workers page that says
        # "online" thirty seconds early is the same class of lie as `online-idle` on a dead
        # ComfyUI.
        global _queue, _poller
        _queue = QueueClient(_sup, client)
        _queue.start()
        # The trainer claims its own work, but only when it is one of the enabled services --
        # a joycaption-only box must not poll for training jobs it could never run.
        if "lora-trainer" in names:
            from wanly_worker.services.lora_trainer.poller import Poller
            _poller = Poller(client, _worker_id)
            _poller.start()
        try:
            yield
        finally:
            if _poller:
                await _poller.stop()
            await _queue.stop()
            await _sup.stop()


def _worker_id() -> str | None:
    """This box's worker row, whoever registered it.

    With the render daemon in the container it is the registrar and writes the id to
    WORKER_ID_FILE after registering (wanly-gpu-daemon#185); the supervisor's own queue
    client stays out of the way. Without one, the queue client registered and knows the id.
    A callable rather than a value because the id does not exist until registration and a
    404 makes either writer re-register with a new one.
    """
    from wanly_worker.services.lora_trainer import gpu
    return gpu.own_worker_id() or (_queue.worker_id if _queue else None)


def _fatal(e: Exception) -> None:
    """Say what went wrong in a line, above the traceback rather than inside it.

    uvicorn reports a failed lifespan as a stack trace ending in "Application startup failed",
    which buries the one sentence that matters -- and the one sentence is always something a
    person can act on: a service name that does not exist, a mount that is not there. The
    traceback still prints; this makes sure the reason is the first thing visible.
    """
    print(f"\n!! FATAL: {e}\n!! this worker is not starting. Nothing was left running.",
          flush=True)


app = FastAPI(title="wanly-gpu-docker", lifespan=lifespan)


class ModeRequest(BaseModel):
    mode: str


@app.post("/mode")
async def set_mode(body: ModeRequest):
    """Change what this box is doing, WITHOUT recreating the container.

    `MODE` as an env var can only be set when a container is created, so every flip through
    it costs a `docker run`: a boot and a model re-stage, minutes each way. That is long
    enough that nobody flips, which is how a box ends up locked into whichever job it
    happened to start on. This does it in place, in seconds.

    The mode is not persisted. A container that restarts comes back on its env MODE, which
    is the right default: the flip is a thing you are doing right now, and a box that came
    back from a crash still captioning -- because of a curl someone made on Tuesday -- is a
    silently idle queue. `docker restart` is the way back to the declared state.
    """
    if _sup is None or _client is None:
        raise HTTPException(status_code=503, detail="not started")
    try:
        names = registry.select_mode(_equipped, body.mode)
    except registry.ConfigError as e:
        # The caller's fault, and every one of them is actionable text: an unknown mode, or
        # a mode that would leave this box running nothing.
        raise HTTPException(status_code=400, detail=str(e)) from e

    global _mode
    # Serialised: see _mode_lock. Held across the whole apply, so a second caller waits for
    # a coherent box rather than racing into the middle of one.
    async with _mode_lock:
        if registry.canonical_mode(body.mode) == _mode:
            return {"mode": _mode, "services": names, "changed": False}
        print(f"mode: {_mode or 'ltx-engine'} -> {registry.canonical_mode(body.mode)} "
              f"({','.join(names)})", flush=True)
        await _sup.apply(names, _client)
        _mode = registry.canonical_mode(body.mode)
    return {"mode": _mode, "services": names, "changed": True}


@app.get("/health")
async def health():
    """503 when something it was asked to run is not answering.

    A health endpoint that returns 200 whatever the state is only useful to a human reading
    the body. Making the STATUS CODE mean it is what lets a timer, a probe or a one-line
    `curl -sf` act on it -- the same reason wanly-gpu-docker's update timer treats an
    unreadable status as busy rather than idle.
    """
    services = _sup.snapshot() if _sup else []
    # A service stopped ON PURPOSE is not a fault. Without this, every box in caption mode
    # answers 503 and every probe that keys on the status code calls it dead.
    live = [s for s in services if not s.get("stopped")]
    ok = bool(live) and all(s["ready"] for s in live)
    body = {
        "status": "ok" if ok else "degraded",
        # What this box CAN do and what it is doing, so a caller can offer the other mode
        # without knowing anything about this container.
        "equipped": _equipped,
        "mode": _mode,
        "build": BUILD,
        "code": _code_ref(),
        "services": services,
        "gpu": gpu_snapshot(),
    }
    return JSONResponse(body, status_code=200 if ok else 503)
