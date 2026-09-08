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

import contextlib
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from wanly_worker import registry
from wanly_worker.queue_client import QueueClient
from wanly_worker.supervisor import Supervisor, gpu_snapshot

BUILD = os.environ.get("WANLY_IMAGE_REF", "unknown")

_sup: Supervisor | None = None
_queue: QueueClient | None = None
_poller = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the enabled services before the control API accepts a request.

    Starting inside the lifespan rather than in a shell entrypoint means a failure to start
    is a failure to boot: uvicorn never begins serving, the container exits non-zero, and
    `docker ps` shows it. An entrypoint that starts things and then execs the API would
    happily serve a healthy-looking /health beside a service that never came up.
    """
    global _sup
    print(f"=== wanly-gpu-docker === build: {BUILD}", flush=True)
    try:
        names = registry.parse_services(os.environ.get("SERVICES"))
        print(f"SERVICES={','.join(names)}", flush=True)
        _sup = Supervisor(registry.build(names))
    except Exception as e:
        _fatal(e)
        raise
    async with httpx.AsyncClient() as client:
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
            _poller = Poller(client, lambda: _queue.worker_id if _queue else None)
            _poller.start()
        try:
            yield
        finally:
            if _poller:
                await _poller.stop()
            await _queue.stop()
            await _sup.stop()


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


@app.get("/health")
async def health():
    """503 when something it was asked to run is not answering.

    A health endpoint that returns 200 whatever the state is only useful to a human reading
    the body. Making the STATUS CODE mean it is what lets a timer, a probe or a one-line
    `curl -sf` act on it -- the same reason wanly-gpu-docker's update timer treats an
    unreadable status as busy rather than idle.
    """
    services = _sup.snapshot() if _sup else []
    ok = bool(services) and all(s["ready"] for s in services)
    body = {
        "status": "ok" if ok else "degraded",
        "build": BUILD,
        "services": services,
        "gpu": gpu_snapshot(),
    }
    return JSONResponse(body, status_code=200 if ok else 503)
