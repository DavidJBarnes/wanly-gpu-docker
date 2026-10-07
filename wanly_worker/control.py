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

from wanly_worker import gpu_pin, registry
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
#: The mode, in its canonical #164 spelling (render / train / motion / edit). /health reports
#: it twice: `mode_name` as is, `mode` in the spelling wanly-api still compares against.
_mode: str = ""
#: What is running right now: select_mode(_equipped, _mode) once a switch lands. Kept rather
#: than recomputed so the switch knows exactly what it is about to stop.
_active: list[str] = []
#: The lifespan's client, kept so POST /mode can use it for readiness probes. Starting a
#: service means waiting for it to answer, which needs one.
_client: httpx.AsyncClient | None = None
#: One mode change at a time. Two overlapping flips interleave stops and starts and can
#: leave the box with the render stack half up beside the captioner -- the exact collision
#: the stop-before-start ordering exists to prevent.
_mode_lock = asyncio.Lock()
#: The mode being switched TO, while the switch is still running, and why the last one
#: failed. A switch is not instant: stopping the render daemon lets the segment in flight
#: FINISH first, which is up to ~27 minutes. Blocking the request for that long is what
#: makes a working switch look like a failed one -- the caller times out and reports an
#: error while the box is quietly doing exactly what was asked.
_pending: str | None = None
_mode_error: str | None = None
#: Held so the task is not garbage collected mid-switch. asyncio keeps only a weak
#: reference, and a collected task stops the box half-flipped.
_mode_task: asyncio.Task | None = None


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
        global _equipped, _mode, _active
        # FIRST, before any service picks a card (#163): with GPU_UUID set, this container must
        # see exactly that card. A worker that boots on the wrong one fails every render later.
        gpu_pin.enforce()
        equipped = registry.parse_services(os.environ.get("SERVICES"))
        _equipped = equipped
        # MODE narrows what actually runs; SERVICES stays the box's own capability line, so
        # the switch is `docker run -e MODE=caption` and nothing has to remember the full
        # list to put back afterwards.
        names = registry.select_mode(equipped, os.environ.get("MODE"))
        mode = (os.environ.get("MODE") or "").strip()
        _mode = registry.canonical_mode(os.environ.get("MODE"))
        _active = list(names)
        print(f"SERVICES={','.join(names)}"
              + (f"  (MODE={mode} of {','.join(equipped)})" if mode else ""), flush=True)
        # Before any child starts: the render daemon reads these from its env and registers
        # the box with them (one row per box, wanly-gpu-docker#83). WORKER_ID_FILE too, and
        # for THIS process as much as for the daemon: the trainer's poller and its drain read
        # the box's own row id from that file, and without the variable in the control
        # process's env they saw no id and never claimed (Payton v1, first night).
        os.environ.update(export_identity(names, equipped))
        if "ltx-engine" in names:
            from wanly_worker.services.ltx_engine import WORKER_ID_FILE
            os.environ["WORKER_ID_FILE"] = WORKER_ID_FILE
        # Built from what the box is EQUIPPED for, started for what the mode asks. A service
        # that only runs in another mode (image-edit, console#548) has to exist here for
        # POST /mode to start it later; it is held stopped until then.
        _sup = Supervisor(registry.build(equipped))
    except Exception as e:
        _fatal(e)
        raise
    async with httpx.AsyncClient() as client:
        global _client
        _client = client
        try:
            await _sup.start(client, active=names)
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
        # The trainer claims its own work, but only on a box EQUIPPED with it -- a captioner-only
        # box must not poll for training jobs it could never run. Started whatever the boot
        # mode (#164): finished runs' publish and delete requests are answered in every mode,
        # and the gate below keeps it from CLAIMING outside a mode that trains.
        if "lora-trainer" in equipped:
            from wanly_worker.services.lora_trainer.poller import Poller
            _poller = Poller(client, _worker_id, may_claim=_trainer_may_claim)
            _poller.start()
        _sync_trainer_tenancy()
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
    # In train mode the daemon is stopped and deregistered its row on the way out, but its id
    # file still names that row. The supervisor's queue client is the registrar then (#131),
    # and its id is the live one.
    if _queue is not None and not _queue.render_daemon_registers():
        return _queue.worker_id or None
    return gpu.own_worker_id() or (_queue.worker_id if _queue else None)


def _trainer_may_claim() -> bool:
    """The poller's claim gate (#164): the trainer is in the running set and no switch is
    under way. A switch accepted mid-claim would otherwise land a run on a card the next
    mode's model is about to load onto."""
    return "lora-trainer" in _active and _pending is None


def _sync_trainer_tenancy() -> None:
    """Tell the trainer whether it has the card to itself (train mode) or shares it with a
    render daemon it must drain first (render mode). See gpu.SOLE_TENANT."""
    if "lora-trainer" not in _equipped:
        return
    from wanly_worker.services.lora_trainer import gpu
    gpu.SOLE_TENANT = _mode == "train"


def _training_now() -> str | None:
    """`<character> v<version> (step n/m)` while a training run is live in this process, else
    None. The poller runs training here, in the control process -- the trainer service on
    :8082 is POST /train's and never sees a claimed run."""
    if "lora-trainer" not in _equipped:
        return None
    try:
        from wanly_worker.services.lora_trainer.app import STORE
        job = STORE.active()
    except Exception:                           # noqa: BLE001 -- no trainer importable
        return None
    if job is None:
        return None
    snap = job.snapshot()
    return f"{snap['character']} v{snap['version']} (step {snap['step']}/{snap['steps']})"


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


#: How long to let captions finish before starting a render anyway. A caption is ~25s and
#: two make an image, so a handful is a couple of minutes; past this something is wedged and
#: holding the box hostage is worse than the overlap.
CAPTION_DRAIN_TIMEOUT_S = float(os.environ.get("CAPTION_DRAIN_TIMEOUT_S") or "600")

#: THE UNLOAD CHECK (#164). After the old mode's services stop and before the new mode's start,
#: the card must come down under this. Every mode's tenant is 14-21 GB, so a tenant left behind
#: always shows; what may legitimately stay is the small stuff every mode keeps -- face-crop
#: (~1.5 GB) and, on a card it shares, the scene captioner (JoyCaption, ~6.4 GB on 3090b).
#: A box with another permanent tenant on the same card sets it higher in worker.env.
MODE_SWITCH_VRAM_MAX_MIB = int(os.environ.get("MODE_SWITCH_VRAM_MAX_MIB") or "8192")
#: How long the card gets to empty. Processes free their memory as they exit; ollama's unload
#: takes a few seconds. Past this something is holding on, and starting anyway is the OOM.
MODE_SWITCH_UNLOAD_TIMEOUT_S = float(os.environ.get("MODE_SWITCH_UNLOAD_TIMEOUT_S") or "90")
#: The last switch's unload numbers, for /health and the ticket: what the card held when the
#: switch found it, and after the unload.
_last_unload: dict | None = None


def _vram_used() -> int | None:
    snap = gpu_snapshot()
    return snap["vram_used_mib"] if snap else None


class CardNotEmpty(RuntimeError):
    """The old mode's services stopped and the card still holds a tenant's worth of memory."""


async def _free_comfyui() -> None:
    """Ask ComfyUI to drop its models before it is stopped. Best effort: the process exiting
    frees the card anyway; this makes the release start sooner and is harmless between
    prompts (ComfyUI acts on the flag only when nothing is executing)."""
    from wanly_worker.services.ltx_engine import COMFY_PORT
    try:
        await _client.post(f"http://127.0.0.1:{COMFY_PORT}/free",
                           json={"unload_models": True, "free_memory": True}, timeout=10)
    except Exception as e:                      # noqa: BLE001
        print(f"mode: ComfyUI /free did not answer ({e}) — stopping it frees the card anyway",
              flush=True)


async def _wait_for_empty_card(before: str, target: str, found: int | None) -> None:
    """Wait until the card is under MODE_SWITCH_VRAM_MAX_MIB, or raise CardNotEmpty.

    Logs what it found and what is left, always: the numbers are the record of whether a switch
    really cleared the card, and the one time it did not is the time they are needed.
    """
    global _last_unload
    if found is None:
        print("!! mode: cannot read the card (no nvidia-smi) — the unload is NOT verified",
              flush=True)
        _last_unload = {"from": before, "to": target, "found_mib": None, "after_mib": None,
                        "limit_mib": MODE_SWITCH_VRAM_MAX_MIB, "ok": None}
        return
    loop = asyncio.get_running_loop()
    started = loop.time()
    used = _vram_used()
    while used is not None and used > MODE_SWITCH_VRAM_MAX_MIB:
        if loop.time() - started >= MODE_SWITCH_UNLOAD_TIMEOUT_S:
            break
        await asyncio.sleep(2)
        used = _vram_used()
    took = loop.time() - started
    ok = used is not None and used <= MODE_SWITCH_VRAM_MAX_MIB
    _last_unload = {"from": before, "to": target, "found_mib": found, "after_mib": used,
                    "limit_mib": MODE_SWITCH_VRAM_MAX_MIB, "seconds": round(took, 1), "ok": ok}
    print(f"mode: unload {before} -> {target}: card held {found} MiB, {used} MiB after "
          f"{took:.0f}s (limit {MODE_SWITCH_VRAM_MAX_MIB} MiB)", flush=True)
    if not ok:
        raise CardNotEmpty(
            f"the card still holds {used} MiB {took:.0f}s after stopping {before} (limit "
            f"{MODE_SWITCH_VRAM_MAX_MIB} MiB) — not starting {target} on it. Something the "
            f"switch does not own is on this GPU (nvidia-smi on the host says what); if it "
            f"belongs there, raise MODE_SWITCH_VRAM_MAX_MIB in worker.env.")


async def _caption_queue_depth() -> int | None:
    """How much the API still has queued for this captioner. None when it will not say.

    Asked of wanly-api, not of ollama: the queue lives in the API (its caption_queue), and
    ollama only ever knows about the one request it is serving.
    """
    from wanly_worker.queue_client import QUEUE_API_KEY, QUEUE_URL

    if not (QUEUE_URL and QUEUE_API_KEY and _client):
        return None
    try:
        r = await _client.get(f"{QUEUE_URL}/images/caption-queue",
                              headers={"X-API-Key": QUEUE_API_KEY}, timeout=10)
        if r.status_code != 200:
            return None
        return int((r.json() or {}).get("depth") or 0)
    except Exception:
        return None


async def _drain_captions() -> None:
    """Wait for outstanding captions before the render stack takes the card.

    THE SYMMETRIC HALF. Switching TO captions already waits for the segment in flight --
    the render daemon is allowed to finish what it started. Switching to render did not wait
    for anything, so a caption mid-flight raced ComfyUI loading its models on the same card.
    One GPU doing one job at a time has to mean both directions or it means neither.

    A queue that cannot be read is not a reason to wait: an older API has no such endpoint,
    and blocking every switch on a question nothing answers would make the mode unusable.
    """
    depth = await _caption_queue_depth()
    if depth is None:
        return
    if depth:
        print(f"mode: waiting for {depth} caption(s) to finish before rendering", flush=True)
    waited = 0.0
    while depth:
        await asyncio.sleep(5)
        waited += 5
        if waited >= CAPTION_DRAIN_TIMEOUT_S:
            # Loud, and then proceed: a wedged captioner must not hold the box forever.
            print(f"!! mode: captions still queued after {waited:.0f}s — starting the "
                  f"render stack anyway", flush=True)
            return
        depth = await _caption_queue_depth()
        if depth is None:
            return
    print("mode: captions drained", flush=True)


#: Edit mode hands the card back by itself once edits stop coming (console#548). wanly-api asks
#: for render mode when its edit queue empties, but a box left in edit mode because that call
#: never came -- an API restart mid-queue -- would sit with every queued render waiting behind
#: an idle 20 GB model. This is the backstop, not the mechanism.
EDIT_IDLE_RETURN_S = float(os.environ.get("EDIT_IDLE_RETURN_S") or "600")
#: The mode edit mode goes back to. Recorded on the way in; render when unknown.
_return_from_edit: str = "render"
_edit_watch: asyncio.Task | None = None


async def _edit_idle_s() -> float | None:
    """Seconds since image-edit last worked, from its own /health. None when it will not say."""
    from wanly_worker.services.image_edit.service import PORT
    try:
        r = await _client.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
        return float(r.json().get("idle_s"))
    except Exception:
        return None


async def _watch_edit_idle() -> None:
    """While in edit mode: switch back once image-edit has been idle EDIT_IDLE_RETURN_S."""
    while _mode == "edit":
        await asyncio.sleep(min(30.0, max(1.0, EDIT_IDLE_RETURN_S / 4)))
        if _mode != "edit" or _pending is not None:
            continue
        idle = await _edit_idle_s()
        if idle is not None and idle >= EDIT_IDLE_RETURN_S:
            print(f"mode: no edit for {idle:.0f}s — handing the card back "
                  f"({_return_from_edit})", flush=True)
            _begin(_return_from_edit, registry.select_mode(_equipped, _return_from_edit))
            return


def _begin(target: str, names: list[str]) -> None:
    """Start a switch in the background. The caller has checked nothing is pending."""
    global _pending, _mode_error, _mode_task
    _mode_error = None
    _pending = target
    print(f"mode: {_mode} -> {target} ({','.join(names)}) — "
          f"a segment in flight will finish first", flush=True)
    _mode_task = asyncio.create_task(_switch(target, names))


async def _switch(target: str, names: list[str]) -> None:
    """Do the switch, off the request. Never raises: it has no caller left to raise to.

    THE ORDER (#164): finish what is in flight, drop the old tenant's model, stop its services,
    VERIFY THE CARD EMPTIED, then start the new mode. A 20 GB model left resident by the last
    mode is how a switch turns into an OOM; checking the card rather than trusting the stops is
    what turns that into a refusal with a number in it.
    """
    global _mode, _active, _pending, _mode_error, _return_from_edit, _edit_watch
    from wanly_worker.services import image_description as imgdesc
    before, before_names = _mode, list(_active)
    touched = False          # whether anything was stopped or started, i.e. needs putting back
    try:
        async with _mode_lock:
            # Re-checked under the lock: the poller may have claimed between the accept and
            # now. Training is never interrupted mid-step.
            training = _training_now()
            if training:
                raise RuntimeError(f"training {training} started before the switch ran; "
                                   f"switch after it finishes")
            leaving = [n for n in before_names if n not in names]
            # LEAVING motion mode: let the captions FINISH, then drop the model. Dropping
            # first would pull the card out from under a caption that is still running.
            if "image-description" in leaving and before == "motion":
                await _drain_captions()
            if "image-description" in leaving:
                await imgdesc.service.release(_client)
            if "ltx-engine" in leaving:
                await _free_comfyui()
            if leaving:
                found = _vram_used()
                touched = True
                # Stop only: everything that stays is already up, so this starts nothing.
                await _sup.apply([n for n in before_names if n in names], _client)
                _active = [n for n in before_names if n in names]
                await _wait_for_empty_card(before, target, found)

            touched = True
            await _sup.apply(names, _client)
            _mode, _active = target, list(names)
            _sync_trainer_tenancy()
            # WHO REGISTERS THIS BOX depends on what is running, and the switch just changed
            # that. Without this the first flip to captions left no registrar at all: the
            # daemon deregisters as it exits, so the row was deleted and the box disappeared
            # from the Workers page -- taking the control that would switch it back with it.
            if _queue is not None:
                _queue.rebalance()

            # ENTERING motion mode: pay the cold load here, where it is expected, instead
            # of inside the first caption, where 88 seconds reads as a hung request. Pinned
            # rather than left on wanly-api's 15m keep_alive, so it stays resident until the
            # box is flipped back.
            if target == "motion":
                await imgdesc.service.warm(_client)
            if target == "edit":
                if before != "edit":
                    _return_from_edit = before
                _edit_watch = asyncio.create_task(_watch_edit_idle())
        print(f"mode: now {target} ({','.join(names)})", flush=True)
    except Exception as e:                      # noqa: BLE001 -- reported, not swallowed
        _mode_error = str(e)
        print(f"!! mode switch to {target} failed: {e}", flush=True)
        if touched and _mode != target:
            # PUT THE BOX BACK. The old mode's services were stopped; leaving them stopped
            # would park every queued job behind a mode that never arrived.
            try:
                async with _mode_lock:
                    await _sup.apply(before_names, _client)
                    _mode, _active = before, before_names
                    _sync_trainer_tenancy()
                    if _queue is not None:
                        _queue.rebalance()
                print(f"mode: restored {before} after the failed switch", flush=True)
            except Exception as e2:             # noqa: BLE001
                _mode_error = f"{e}; and restoring {before} failed too: {e2}"
                print(f"!! could not restore {before}: {e2}", flush=True)
    finally:
        _pending = None


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

    target = registry.canonical_mode(body.mode)

    # NEVER MID-RUN (#164). A training run is an hour or more -- not a segment to wait out
    # behind a pending switch that blocks every other switch meanwhile -- so it is refused
    # with the run named, and the caller switches when it ends.
    training = _training_now() if target != _mode else None
    if training and _pending is None:
        raise HTTPException(
            status_code=409,
            detail=f"training {training} on this box; switch to {target} after it finishes")

    # ACCEPTED AND RETURNED IMMEDIATELY. Stopping the render daemon waits for the segment in
    # flight to finish -- by design, so nothing is destroyed -- and that is up to ~27
    # minutes. Waiting for it here means the client times out and reports a failure while
    # the switch is going perfectly well. The state is readable from /health instead.
    if _pending is not None:
        if _pending == target:
            return _mode_body(names, changed=False)
        raise HTTPException(
            status_code=409,
            detail=f"already switching to {_pending}; wait for it to land")
    if target == _mode:
        return _mode_body(names, changed=False)

    _begin(target, names)
    return _mode_body(names, changed=True)


def _mode_body(names: list[str], changed: bool) -> dict:
    """POST /mode's answer. `mode`/`pending` in the spelling wanly-api compares against
    (registry.LEGACY_NAME), `mode_name`/`pending_mode_name` in the four-mode one."""
    return {"mode": registry.legacy_name(_mode), "pending": registry.legacy_name(_pending),
            "mode_name": _mode, "pending_mode_name": _pending,
            "services": names, "changed": changed}


def _available_modes() -> list[str]:
    """The modes this box can enter -- the ones select_mode does not refuse for its SERVICES."""
    out = []
    for m in registry.MODES:
        try:
            registry.select_mode(_equipped, m)
            out.append(m)
        except registry.ConfigError:
            pass
    return out


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
        # `mode` and `pending_mode` keep the spelling wanly-api and the console compare against
        # (ltx-engine / caption / edit, and train); `mode_name` and `pending_mode_name` are
        # the four-mode names (#164). See registry.LEGACY_NAME.
        "mode": registry.legacy_name(_mode),
        "mode_name": _mode,
        "modes": _available_modes() if _equipped else [],
        # Set while a switch is running. The caller shows it as in-progress rather than as
        # the mode it is not in yet -- and `mode_error` is how a switch that failed says so,
        # since by then there is no request left to answer.
        "pending_mode": registry.legacy_name(_pending),
        "pending_mode_name": _pending,
        "mode_error": _mode_error,
        # What the last switch found on the card and what was left after the unload.
        "last_unload": _last_unload,
        "build": BUILD,
        "code": _code_ref(),
        "services": services,
        "gpu": gpu_snapshot(),
    }
    return JSONResponse(body, status_code=200 if ok else 503)
