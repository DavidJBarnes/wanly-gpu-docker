"""Tell wanly-api this box exists (wanly-api#269).

WHY A SERVICE REGISTERS AT ALL

    It is a long-lived, GPU-holding process, and until now the API had no idea it was there.
    wanly-console#426's actual goal is one VRAM contract instead of two bespoke yield
    protocols, and a contract needs both tenants visible in one place. Today half the fleet
    is invisible: "is JoyCaption up?" is answered by ssh'ing to the box.

REGISTRATION MUST NEVER BE ABLE TO STOP A CAPTION

    Everything here is best-effort and non-fatal, deliberately. This container's job is to
    serve JoyCaption; telling the API about it is observability. An unreachable API, a wrong
    key, no key at all -- none of that is a reason to refuse to caption, and a container that
    died because a *reporting* channel was down would be a worse outage than the one it was
    reporting. So failures are logged and the loop keeps going.

    That is the opposite of the boot-time guards in service.py, and the difference is real:
    those check whether the work can be done at all, this one only tells someone about it.

WHY IT SAYS kind=service

    The API's claim gate keys on it. A service registering as a render worker would be offered
    segments, and `_model_gate` explicitly does not filter a worker that has never reported
    checkpoints -- so it would be offered ALL of them. Saying so is not a label; it is what
    keeps this container out of the queue.
"""
from __future__ import annotations

import asyncio
import os
import socket


def _log(msg: str) -> None:
    """print(), not logging.

    Everything else this container says on the boot path uses print, and for a reason that is
    not style: uvicorn configures handlers for its own loggers and leaves the root logger
    alone, so a module logger's INFO records go nowhere. The registration line was written
    with logging.info and was invisible in `docker logs` on the very first real deployment --
    the box HAD registered, and the only way to find that out was to ask the API.

    Warnings would have escaped through logging's lastResort handler, which makes it worse
    rather than better: success silent, failure visible, so a healthy boot and a
    never-attempted one look identical.
    """
    print(f"[queue] {msg}", flush=True)

QUEUE_URL = os.environ.get("QUEUE_URL", "").rstrip("/")
QUEUE_API_KEY = os.environ.get("QUEUE_API_KEY", "")

#: 30s, matching the daemon. `heartbeat_offline_seconds` on the API is 120, so anything much
#: slower reads as a worker that keeps flapping offline between beats.
INTERVAL_S = int(os.environ.get("HEARTBEAT_INTERVAL_S", "30"))


def friendly_name() -> str:
    """`<host>/services`, and the suffix is not decoration.

    `register_worker` upserts on friendly_name and REUSES the row it finds. A services
    container registering as plain `2070.zero` would silently share one row with anything else
    on that host that used the same name, each overwriting the other's fields -- the same row
    reuse that pinned a segment to a live worker in wanly-gpu-docker#74.
    """
    return os.environ.get("FRIENDLY_NAME") or f"{socket.gethostname()}/services"


def _ip() -> str:
    """Best guess at the address the API would reach us on.

    A UDP connect to a public address never sends a packet; it just makes the kernel pick the
    interface it would route through, which is the one that matters here. gethostbyname often
    answers 127.0.0.1 in a container, which is useless to a reader trying to find the box.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "0.0.0.0"


def enabled() -> bool:
    return bool(QUEUE_URL and QUEUE_API_KEY)


#: Services whose presence changes what KIND of worker this box is. The API's claim gates key
#: on kind, not on `provides` -- a gate keyed on names needs an allowlist, and an engine missing
#: from that allowlist claims nothing, indistinguishable from an empty queue. So the mapping
#: lives here, in the one place that decides what to register as.
KIND_BY_SERVICE = {"lora-trainer": "trainer"}


def worker_kind(provides: list[str]) -> str:
    """What to register as, given what is running.

    Defaults to `service`, which takes no work of any kind. A container that runs a trainer must
    say `trainer` or it will never be offered a training job -- and that failure is silent: it
    registers, heartbeats, shows green on the Workers page, and quietly never claims.
    """
    for name in provides:
        if name in KIND_BY_SERVICE:
            return KIND_BY_SERVICE[name]
    return "service"


class QueueClient:
    """Registers, heartbeats, and deregisters. Never raises at the caller."""

    def __init__(self, supervisor, client):
        self._sup = supervisor
        self._client = client
        self.worker_id: str | None = None
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ payloads

    def provides(self) -> list[str]:
        """The SERVICES names, not the processes. `ltx-engine` is three processes and one
        capability; the Workers page should read the capability."""
        out: list[str] = []
        for row in self._sup.snapshot():
            name = row.get("group") or row["name"]
            if name not in out:
                out.append(name)
        return out

    def render_daemon_registers(self) -> bool:
        """When the render daemon is one of the processes it is the single registrar for
        this box (wanly-gpu-docker#83): it heartbeats the rich payload and owns the status
        vocabulary, and a second writer on the same row would fight it."""
        return any(row["name"] == "render-daemon" for row in self._sup.snapshot())

    def status(self) -> str:
        """`online` when everything asked for answers, `degraded` when some of it does not.

        Not `online-idle`. That word already means both "waiting for work" and "cannot do
        work" on render workers, and the ambiguity hid a dead ComfyUI for 33 minutes
        (wanly-gpu-docker#80). A service is never waiting for work, so borrowing it would
        extend exactly the confusion that is already costing us.
        """
        rows = self._sup.snapshot()
        return "online" if rows and all(r["ready"] for r in rows) else "degraded"

    def _gpu_stats(self) -> dict | None:
        """The daemon's shape exactly -- gpu_name / vram_used_mb / vram_total_mb -- so the
        console's existing renderer works with no console change. A second vocabulary for the
        same three numbers would be a gratuitous fork."""
        from wanly_worker.supervisor import gpu_snapshot
        g = gpu_snapshot()
        if not g:
            return None
        return {"gpu_name": g["name"], "vram_used_mb": g["vram_used_mib"],
                "vram_total_mb": g["vram_total_mib"]}

    # ------------------------------------------------------------------ calls

    async def register(self) -> None:
        if not enabled():
            _log("QUEUE_URL/QUEUE_API_KEY not set — not registering with wanly-api. "
                 "Captioning is unaffected; this box just will not appear on the Workers page.")
            return
        body = {
            "friendly_name": friendly_name(),
            "hostname": socket.gethostname(),
            "ip_address": _ip(),
            "comfyui_running": False,
            "kind": worker_kind(self.provides()),
            "provides": self.provides(),
        }
        try:
            r = await self._client.post(f"{QUEUE_URL}/workers", json=body,
                                        headers={"X-API-Key": QUEUE_API_KEY}, timeout=15)
            r.raise_for_status()
            self.worker_id = r.json()["id"]
            _log(f"registered with wanly-api as {body['friendly_name']} ({self.worker_id})")
        except Exception as e:
            # Non-fatal, on purpose. See the module docstring.
            _log(f"could not register with wanly-api ({e}) — continuing anyway")

    async def beat(self) -> None:
        if not self.worker_id:
            # Re-register rather than give up. The API may have been restarted, or this row
            # deleted; a service that stopped reporting after one bad minute would be a
            # silently invisible box, which is the thing this exists to prevent.
            await self.register()
            return
        body = {
            "comfyui_running": False,
            "status": self.status(),
            "provides": self.provides(),
            "gpu_stats": self._gpu_stats(),
        }
        try:
            r = await self._client.post(f"{QUEUE_URL}/workers/{self.worker_id}/heartbeat",
                                        json=body, headers={"X-API-Key": QUEUE_API_KEY},
                                        timeout=15)
            if r.status_code == 404:
                # The row is gone. Registering again is the correct response, and upsert on
                # friendly_name means it lands on the same identity.
                _log("worker row is gone — re-registering")
                self.worker_id = None
                return
            r.raise_for_status()
        except Exception as e:
            _log(f"heartbeat failed ({e}) — will retry")

    async def deregister(self) -> None:
        """Clean shutdown should not look like a crash.

        Without this a deliberate `docker stop` leaves a row that goes stale and is swept to
        `offline` two minutes later -- indistinguishable from a box that fell over, which is
        exactly the ambiguity worth spending one HTTP call to remove.
        """
        if not (enabled() and self.worker_id):
            return
        try:
            await self._client.delete(f"{QUEUE_URL}/workers/{self.worker_id}",
                                      headers={"X-API-Key": QUEUE_API_KEY}, timeout=10)
            _log("deregistered from wanly-api")
        except Exception as e:
            _log(f"could not deregister ({e}) — the row will go stale instead")

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        await self.register()
        while True:
            await asyncio.sleep(INTERVAL_S)
            await self.beat()

    def start(self) -> None:
        if self.render_daemon_registers():
            _log("the render daemon registers this box; the supervisor will not")
            return
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self.deregister()
