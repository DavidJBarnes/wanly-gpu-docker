"""The trainer's own HTTP API — the endpoints wanly-console#453 asks for.

Two ways in, one pipeline:

    POST /train              direct. What make_lora.py and a curl use.
    the poller               claims from wanly-api. What the console's path uses.

Both land in `_run`, so there is one implementation of "train a LoRA" and one place where the
GPU is acquired and released.

ONE AT A TIME, enforced rather than assumed. Training takes essentially the whole card; a second
concurrent run would not be twice as fast, it would be an OOM.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from wanly_worker.services.lora_trainer import gpu, pipeline, recipe
from wanly_worker.services.lora_trainer.jobs import Job, Store

STORE = Store()
HOSTNAME = os.environ.get("TRAINER_HOSTNAME") or socket.gethostname()


class TrainRequest(BaseModel):
    character: str = Field(min_length=1, max_length=64)
    trigger: str = Field(min_length=1, max_length=64)
    version: int = Field(default=1, ge=1, le=99)
    #: Where the images are. Either URLs to fetch (what a claim gives us: presigned, so no AWS
    #: credentials live here) or a directory already on this box.
    image_urls: list[str] = Field(default_factory=list)
    image_dir: str = ""
    caption: str | None = None
    #: THE EXTRA GROUPS (#102, #106). ABSENT means single-identity, which every run before
    #: this is. Each entry: {character, trigger, gender, caption, image_urls, num_repeats}
    #: -- presigned by the API alongside group 0's. An IDENTITY group has a trigger and its
    #: caption is "<trigger>, <gender>"; a COMPOSITION group (#106) has no trigger and its
    #: caption names the people in its frames, which is what teaches the model they appear
    #: together. The claim builds the list; a POST body may also give it directly.
    identities: list[dict] = Field(default_factory=list)
    steps: int = Field(default=1200, ge=100, le=6000)
    config: dict = Field(default_factory=dict)
    remote_id: str = ""


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Release drains left behind by a previous life of this container.

    Not optional. A drain survives worker re-registration by design, so a trainer that died
    holding one leaves the render queue stopped with nothing pointing at the cause.

    Suppressed rather than fatal: a reconcile that cannot reach the API must not stop the
    trainer booting, and it will be retried on the next start.
    """
    with contextlib.suppress(Exception):
        async with httpx.AsyncClient() as client:
            await gpu.reconcile(client, STORE)
            # A job that was mid-run when the container died was failed by the store on
            # load; the API row is still CLAIMED or RUNNING and has to be told, or it sits
            # there until someone notices the queue is not moving.
            for job in STORE.failed_by_restart():
                if not (gpu.enabled() and job.remote_id):
                    continue
                r = await client.patch(
                    f"{gpu.QUEUE_URL}/training/{job.remote_id}",
                    json={"status": "failed", "error_message": job.error,
                          "progress_log": job.error},
                    headers={"X-API-Key": gpu.QUEUE_API_KEY}, timeout=20)
                print(f"[trainer] reported {job.character} v{job.version} failed by "
                      f"restart -> HTTP {r.status_code}", flush=True)
    yield


app = FastAPI(title="wanly lora-trainer", lifespan=lifespan)


@app.get("/health")
async def health():
    """200 always — the service is up even with nothing to do.

    Deliberately not 503 when idle: the supervisor's readiness probe polls this, and an idle
    trainer is ready, not degraded.
    """
    active = STORE.active()
    return {"status": "ok", "active": active.snapshot() if active else None,
            "jobs": len(STORE.all())}


@app.post("/train", status_code=202)
async def start_training(req: TrainRequest):
    if not STORE.claim_slot():
        running = STORE.active()
        raise HTTPException(
            status_code=409,
            detail=f"already training {running.character} v{running.version} "
                   f"(step {running.step}/{running.steps}). One at a time — the card cannot "
                   f"hold two.")
    if not req.image_urls and not req.image_dir:
        raise HTTPException(status_code=400, detail="give image_urls or image_dir")

    job = Job(id=uuid.uuid4().hex[:12], character=req.character, trigger=req.trigger,
              version=req.version, steps=req.steps, remote_id=req.remote_id)
    STORE.add(job)
    asyncio.create_task(_run(job, req))
    return {"id": job.id, "phase": job.phase}


@app.get("/train")
async def list_jobs():
    return [j.snapshot() for j in STORE.all()]


@app.get("/train/{job_id}")
async def get_job(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    return job.snapshot()


@app.post("/train/{job_id}/cancel")
async def cancel(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    if job.done:
        raise HTTPException(status_code=400, detail=f"already {job.phase}")
    # Cooperative: the run notices at its next progress tick. Killing the trainer outright would
    # leave the drain held and a half-written checkpoint.
    #
    # THE FLAG, NOT THE PHASE. _run writes phase as it moves through staging/training/collecting,
    # so a phase written from here was overwritten within seconds and the run carried on to
    # completion. Nothing read it either. The phase is set as well, but only for a job that has
    # not started -- there is no loop to notice one of those.
    job.cancel_requested = True
    if job.phase == "pending":
        job.phase = "cancelled"
    STORE.persist(job)
    return job.snapshot()


@app.get("/train/{job_id}/log", response_class=PlainTextResponse)
async def job_log(job_id: str, tail: int = 4000):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    log = recipe.run_dir(job.character, job.version) / "logs" / "03_train.log"
    if not log.exists():
        return "training has not started yet"
    with log.open("rb") as fh:
        fh.seek(max(0, log.stat().st_size - tail))
        return fh.read().decode("utf8", "replace")


async def _fetch(client, urls: list[str]) -> list[tuple[str, bytes]]:
    """Download the dataset. Presigned URLs, so no AWS credentials live in this container."""
    out = []
    for i, u in enumerate(urls):
        r = await client.get(u, timeout=120, follow_redirects=True)
        r.raise_for_status()
        name = Path(u.split("?", 1)[0]).name or f"img{i}.jpg"
        out.append((name, r.content))
    return out


async def _run(job: Job, req: TrainRequest, on_progress=None) -> None:
    """One implementation of training, whichever way the job arrived."""
    async with httpx.AsyncClient() as client:
        try:
            job.phase = "staging"
            STORE.persist(job)

            if req.image_urls:
                images = await _fetch(client, req.image_urls)
            else:
                d = Path(req.image_dir)
                images = [(f.name, f.read_bytes()) for f in sorted(d.iterdir())
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            if not images:
                raise pipeline.PipelineError("no images to train on")
            job.images = len(images)

            # ONE ENTRY PER GROUP. Group 0 is the job's own character; the extra groups
            # (#102, #106) ride req.identities, each with its own URLs, caption and repeats.
            # The stage call takes the whole list so per-group captions and dirs stay paired
            # -- separate stage calls would mean separate dataset.toml writes, and the last
            # one would win.
            groups = [{"images": images, "caption": req.caption,
                       "num_repeats": (req.config or {}).get("num_repeats")
                       or recipe.DEFAULTS["num_repeats"]}]
            for gi, g in enumerate(req.identities):
                g_urls = g.get("image_urls") or []
                if not g_urls and g.get("image_dir"):
                    d = Path(g["image_dir"])
                    g_images = [(f.name, f.read_bytes()) for f in sorted(d.iterdir())
                                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                else:
                    g_images = await _fetch(client, g_urls)
                if not g_images:
                    raise pipeline.PipelineError(
                        f"group {gi + 2} has no images to train on")
                groups.append({
                    "images": g_images,
                    "caption": g.get("caption"),
                    "num_repeats": g.get("num_repeats")
                    or recipe.DEFAULTS["num_repeats"],
                })
                job.images += len(g_images)
            if req.identities:
                # The disk gate reads repeats x total images; every group counts, and mixed
                # repeats across groups would make the estimate a guess.
                job.effective_repeats = max(g["num_repeats"] for g in groups)
            pipeline.preflight(job)
            run = await pipeline.stage(job, groups)
            # SAY SO. Nothing reported between the claim and the first training step left the
            # API row at "claimed" with no progress for the whole of staging and the drain
            # wait -- which reads as queued in the console, and which the orphan reclaim
            # hands back to the queue after twenty minutes.
            if on_progress:
                await on_progress(job)

            async def waiting(msg: str) -> None:
                job.message = msg
                if on_progress:
                    await on_progress(job)

            # ACQUIRE, THEN PERSIST, THEN TRAIN. The id is written down before the long part so
            # a crash during training can still find the drain to release.
            worker_id = await gpu.acquire(client, HOSTNAME, on_wait=waiting)
            job.drained_worker_id = worker_id
            STORE.persist(job)

            try:
                job.phase = "training"
                STORE.persist(job)
                await pipeline.train(job, run, {**req.config, "steps": req.steps}, on_progress)
                job.phase = "collecting"
                pipeline.collect(job, run)
                job.phase = "completed"
            finally:
                await gpu.release(client, worker_id)
                STORE.clear_drain(job)
        except pipeline.Cancelled:
            # Not a failure. Filing it as one puts a red row on the board for something that
            # went exactly as asked, and leaves an error_message nobody can act on.
            job.phase = "cancelled"
            print(f"[trainer] {job.character} v{job.version} cancelled", flush=True)
        except Exception as e:
            job.phase = "failed"
            job.error = str(e)
            print(f"[trainer] {job.character} v{job.version} FAILED: {e}", flush=True)
        finally:
            import time
            job.finished_at = time.time()
            STORE.persist(job)
            # REPORT THE TERMINAL STATE, whatever it is. on_progress is otherwise only called
            # from inside the training loop, so a failure while staging -- or anywhere before
            # the first step -- left the API row CLAIMED forever. The orphan reclaim would then
            # put it back to pending, the trainer would claim it again, and fail again: a loop
            # that looks like a queue quietly not moving.
            if on_progress:
                try:
                    await on_progress(job)
                except Exception as report_error:
                    print(f"[trainer] could not report the final state of {job.id}: "
                          f"{report_error}", flush=True)
