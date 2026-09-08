"""Getting the GPU, and giving it back (wanly-services#7).

A training run is ~50 minutes at ~23 GB of 24. The render worker sits at 17-23 GB while working.
They cannot overlap, so something has to arbitrate — and the something is deliberately NOT a
`docker stop`.

WHY DRAIN AND NOT docker stop
    `run_all.sh` used to stop a render container from an EXIT trap. The name it stopped had been
    dead for days while a different container held the VRAM, so it freed nothing, and the trap
    could have started a stale container onto a port already in use. Draining goes through the
    API the worker already talks to: it finishes what it holds, claims nothing new, and comes
    back when released. No docker socket, no container names, nothing to go stale.

THE RECONCILER IS NOT OPTIONAL
    A drain SURVIVES worker re-registration by design — wanly-api's `reregistered_drain_state`
    says so in as many words, because a re-register silently cancelling an operator's drain was
    its own incident. The consequence here is that a trainer which dies holding a drain leaves
    the render queue stopped, permanently, with nothing pointing at the cause. So the drain is
    persisted with the job and released at startup if its job is not running.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time

QUEUE_URL = os.environ.get("QUEUE_URL", "").rstrip("/")
QUEUE_API_KEY = os.environ.get("QUEUE_API_KEY", "")
#: How long to wait for a drained worker to actually finish what it holds. A 480p render is
#: 10-13 minutes; a 720x1056 one is ~1780 s measured (the daemon's own drain_wait_seconds is
#: 3600 for that reason). 1800 was under a 720p render, so a training job that arrived just
#: after one started failed with "did not free the GPU" a minute before it would have. It
#: fails rather than training beside a render, which is the OOM this exists to prevent.
DRAIN_TIMEOUT_S = int(os.environ.get("DRAIN_TIMEOUT_S", "3600"))
#: Below this the card is ours in practice. Our own trainer has not started yet at this point,
#: so anything above it is somebody else.
FREE_VRAM_FLOOR_MIB = int(os.environ.get("FREE_VRAM_FLOOR_MIB", "4000"))
#: Which render worker shares this GPU. See find_render_worker for why this cannot be detected.
RENDER_WORKER_NAME = os.environ.get("RENDER_WORKER_NAME", "")
#: ONE CONTAINER PER GPU (wanly-gpu-docker#83): when the render daemon runs in this same
#: container it registers the box once, as ["render", "trainer"], and writes the row's id
#: here. That row IS the render worker sharing the card -- there is nothing to look up by name,
#: and every incident of 2026-09-08 came from looking one up by name.
WORKER_ID_FILE = os.environ.get("WORKER_ID_FILE", "")


def own_worker_id() -> str:
    """The id of this box's own worker row, when the render daemon in this container wrote
    it. Empty on a trainer-only box, which then falls back to finding the render worker."""
    if not WORKER_ID_FILE:
        return ""
    try:
        return open(WORKER_ID_FILE).read().strip()
    except OSError:
        return ""


def _log(msg: str) -> None:
    print(f"[trainer/gpu] {msg}", flush=True)


def enabled() -> bool:
    return bool(QUEUE_URL and QUEUE_API_KEY)


def vram_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


async def _workers(client) -> list[dict]:
    r = await client.get(f"{QUEUE_URL}/workers",
                         headers={"X-API-Key": QUEUE_API_KEY}, timeout=20)
    r.raise_for_status()
    return r.json()


async def find_render_worker(client, _hostname: str = "") -> dict | None:
    """The render worker whose GPU we are about to take, or None.

    NAMED EXPLICITLY, because "the worker on my host" is not inferable from inside a container.
    Every worker registers the hostname it sees, which is its own container id -- the render
    worker on 3090.zero reports `4b6d0f9c3b3b` and the trainer beside it reports `54a6e63cdd96`.
    Matching those was the first attempt and it silently found nothing, so the trainer took the
    card without draining anything. It only worked because the worker happened to be idle.

    So: RENDER_WORKER_NAME wins. With it unset, a fleet containing exactly one online render
    worker is unambiguous and that one is used, loudly. More than one is a guess worth refusing
    -- draining the wrong box stops a queue for no reason.
    """
    rows = [w for w in await _workers(client)
            if w.get("kind") == "render" and w.get("status") != "offline"]
    if RENDER_WORKER_NAME:
        for w in rows:
            if w.get("friendly_name") == RENDER_WORKER_NAME:
                return w
        if len(rows) == 1:
            # The name in services.env went stale -- the render worker was renamed in the
            # console -- but there is exactly one, so it is not a guess. Say so loudly:
            # the env still needs fixing, and this must not become the silent normal.
            _log(f"!! RENDER_WORKER_NAME={RENDER_WORKER_NAME} matches no online render worker; "
                 f"the only one online is {rows[0]['friendly_name']!r} — using it. "
                 f"Fix RENDER_WORKER_NAME in worker.env.")
            return rows[0]
        _log(f"RENDER_WORKER_NAME={RENDER_WORKER_NAME} matches no online render worker")
        return None
    if len(rows) == 1:
        _log(f"RENDER_WORKER_NAME is unset; one online render worker "
             f"({rows[0]['friendly_name']}) so using it")
        return rows[0]
    if rows:
        raise RuntimeError(
            f"{len(rows)} online render workers and RENDER_WORKER_NAME is unset. Refusing to "
            f"guess which one shares this GPU: {[w['friendly_name'] for w in rows]}")
    return None


async def acquire(client, hostname: str, on_wait=None) -> str:
    """Drain the render worker and wait for the card. Returns its id, or "" if there is none.

    The returned id is what the caller must persist BEFORE training, so a crash can release it.

    `on_wait(message)` is called on every poll of the wait. This can last as long as a 720p
    render, and a job that says nothing for that long looks abandoned -- to the person
    watching the console, and to wanly-api's orphan reclaim, which puts a claim with no
    progress back in the queue after twenty minutes.
    """
    if not enabled():
        _log("no QUEUE_URL/QUEUE_API_KEY — cannot coordinate, proceeding on trust")
        return ""

    own = own_worker_id()
    if own:
        # The render daemon is in this container and registered this box. Drain that row: the
        # daemon parks (wanly-gpu-daemon#182), the card frees, and release() below brings it back.
        worker = next((w for w in await _workers(client) if w.get("id") == own), None)
        if worker is None:
            raise RuntimeError(
                f"this box's own worker row {own} (from {WORKER_ID_FILE}) is not on the API. "
                f"Refusing to train beside a render daemon nothing can drain.")
    else:
        worker = await find_render_worker(client, hostname)
    if worker is None:
        used = vram_used_mib()
        if used > FREE_VRAM_FLOOR_MIB:
            # Something holds the card and nothing we know how to drain owns it. Training
            # anyway is the OOM this whole module exists to prevent -- and it happened: the
            # render worker had been renamed, the trainer found nothing to drain, and the run
            # died twenty seconds in loading the text encoder beside 13.5 GB of idle engine.
            raise RuntimeError(
                f"{used} MiB of the card is in use and no render worker matches "
                f"RENDER_WORKER_NAME={RENDER_WORKER_NAME!r} (online render workers: "
                f"{[w['friendly_name'] for w in await _workers(client) if w.get('kind') == 'render']}). "
                f"Refusing to train beside it — fix RENDER_WORKER_NAME in worker.env.")
        _log("no render worker on this host and the card is free — nothing to drain")
        return ""

    wid = worker["id"]
    _log(f"draining {worker['friendly_name']} ({wid})")
    r = await client.post(f"{QUEUE_URL}/workers/{wid}/drain", json={},
                          headers={"X-API-Key": QUEUE_API_KEY}, timeout=20)
    r.raise_for_status()

    deadline = time.time() + DRAIN_TIMEOUT_S
    while time.time() < deadline:
        await asyncio.sleep(15)
        rows = {w["id"]: w for w in await _workers(client)}
        w = rows.get(wid)
        busy = w and w.get("status") == "online-busy"
        used = vram_used_mib()
        if not busy and used < FREE_VRAM_FLOOR_MIB:
            _log(f"card is free ({used} MiB in use)")
            return wid
        msg = (f"waiting for {worker['friendly_name']} to finish rendering "
               f"({w.get('status') if w else 'gone'}, {used} MiB in use)")
        _log(msg)
        if on_wait:
            await on_wait(msg)

    # Do NOT train anyway. Sharing the card is the OOM this exists to prevent, and the drain we
    # are holding has to come off before we give up.
    await release(client, wid)
    raise RuntimeError(
        f"the render worker did not free the GPU within {DRAIN_TIMEOUT_S}s — not training "
        f"beside it")


async def release(client, worker_id: str) -> bool:
    if not (enabled() and worker_id):
        return True
    try:
        r = await client.delete(f"{QUEUE_URL}/workers/{worker_id}/drain",
                                headers={"X-API-Key": QUEUE_API_KEY}, timeout=20)
        if r.status_code == 404:
            # THE ROW IS GONE, AND THAT IS THE DRAIN WORKING. On a box with a restart policy
            # a drained daemon exits, the container comes back, and it registers a NEW row
            # with no drain on it -- the old id has nothing left to release. Seen on
            # 3090.zero on every run. Not a failure, and not the "queue stopped forever"
            # case below, which is a row that still exists and still says draining.
            _log(f"worker {worker_id} re-registered under a new id while drained — "
                 f"nothing left to release")
            return True
        ok = r.status_code < 400
        _log(f"released the drain on {worker_id}" if ok
             else f"could not release the drain on {worker_id}: HTTP {r.status_code}")
        return ok
    except Exception as e:
        # Loud, because the consequence is a render queue that has silently stopped.
        _log(f"!! COULD NOT RELEASE THE DRAIN on {worker_id} ({e}). "
             f"The render worker will not claim until this is undone by hand: "
             f"DELETE {QUEUE_URL}/workers/{worker_id}/drain")
        return False


async def reconcile(client, store) -> None:
    """Release drains left behind by a previous life of this container."""
    for job_id, worker_id in store.orphaned_drains():
        _log(f"job {job_id} is finished but still holds a drain on {worker_id} — releasing")
        if await release(client, worker_id):
            job = store.get(job_id)
            if job:
                store.clear_drain(job)
