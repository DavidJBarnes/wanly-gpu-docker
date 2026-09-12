"""Claim training jobs from wanly-api (wanly-services#7).

PULL, because everything in this system is. The API never calls a worker; work is claimed and
reported on. That is what gives a training job orphan reclaim, the heartbeat/offline sweep and
queue-health without any of them being written twice.

The poller is the console's path. `POST /train` on this container is the direct path. Both end
up in the same `_run`, so there is one implementation of training and one place the GPU is
acquired and released.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path

import httpx

from wanly_worker.services.lora_trainer import app as trainer_app
from wanly_worker.services.lora_trainer import recipe
from wanly_worker.services.lora_trainer.jobs import Job

QUEUE_URL = os.environ.get("QUEUE_URL", "").rstrip("/")
QUEUE_API_KEY = os.environ.get("QUEUE_API_KEY", "")
#: Slow on purpose. A training job is a rare event and a 50-minute unit of work; polling hard
#: buys nothing and puts a request per second on the API for no reason.
POLL_INTERVAL_S = int(os.environ.get("TRAIN_POLL_INTERVAL_S", "60"))
#: A checkpoint is uploaded once its size has stopped changing for this long. The trainer
#: writes safetensors in place, not via a rename, so a file that is still growing is a file
#: that is still being written.
CHECKPOINT_SETTLE_S = int(os.environ.get("CHECKPOINT_SETTLE_S", "45"))
#: Per file. A 650 MB checkpoint takes about ten minutes to cross a home uplink; a transient
#: failure should cost one retry, not the epoch.
UPLOAD_ATTEMPTS = 3
UPLOAD_CHUNK = 8 * 1024 * 1024
EPOCH_RE = re.compile(r"-(\d+)\.comfy\.safetensors$")
#: How far back a finished run is still watched for publish requests. The run directory is
#: what makes a request satisfiable, and those are not cleaned up, but polling every job
#: this box ever trained on every tick is not free either.
PUBLISH_WATCH_DAYS = int(os.environ.get("PUBLISH_WATCH_DAYS", "14"))
#: How often the no-worker-id warning repeats. A tick with no id must SAY SO -- a silent
#: return is invisible behind an entirely healthy-looking box, which is how Payton v1 sat in
#: the queue for an hour on its first night. Repeating because docker logs are not
#: necessarily watched as they happen.
NO_WORKER_ID_LOG_S = int(os.environ.get("NO_WORKER_ID_LOG_S", "300"))


def _log(msg: str) -> None:
    print(f"[trainer/poll] {msg}", flush=True)


def enabled() -> bool:
    return bool(QUEUE_URL and QUEUE_API_KEY)


class Poller:
    def __init__(self, client, worker_id_getter):
        self._client = client
        #: A callable rather than a value: the worker id does not exist until the queue client
        #: has registered, and a 404 makes it re-register with a new one.
        self._worker_id = worker_id_getter
        self._task: asyncio.Task | None = None
        #: Checkpoints already in the bucket, per job -- local path -> s3 uri. Uploads run
        #: WHILE training does, so this is what stops the same epoch going up twice.
        self._published: dict[str, dict[str, str]] = {}
        #: Checkpoints that failed every attempt, per job. Reported at the end, loudly.
        self._failed: dict[str, list[str]] = {}
        #: Sizes seen last tick, so "settled" can be judged. path -> (size, first_seen_at).
        self._seen: dict[str, tuple[int, float]] = {}
        #: Waiting to go up, per job, and the one task per job that sends them ONE AT A TIME.
        #: They share the uplink -- 0.6 MB/s measured, eighteen minutes per checkpoint -- and
        #: two in flight just halve each other.
        self._queue: dict[str, list[Path]] = {}
        self._uploader: dict[str, asyncio.Task] = {}
        #: The label going up right now, per job, for the progress line, and how far it is:
        #: (bytes sent, bytes total). The trainer streams the file itself, so it knows.
        self._current: dict[str, str] = {}
        self._sent: dict[str, tuple[int, int]] = {}
        #: The file going up right now. Popped from the queue when the upload starts, it
        #: was neither queued nor published for twelve minutes, and the sweep that runs
        #: every ten seconds while the queue drains put it straight back -- so the final
        #: checkpoint of the first end-to-end run went up twice.
        self._inflight: set[str] = set()
        #: Per job, the recipe it ran with and the labels the console asked for after the
        #: fact -- the two things that decide which of the files on disk are wanted.
        self._policy: dict[str, str] = {}
        self._requested: dict[str, set[str]] = {}
        #: When the no-worker-id warning was last said, so the loud tick is not a flood.
        self._no_id_said_at: float = 0.0

    def start(self) -> None:
        if not enabled():
            _log("no QUEUE_URL/QUEUE_API_KEY — not claiming work. "
                 "POST /train still works.")
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            try:
                await self.tick()
            except Exception as e:
                # Never let a bad poll kill the loop. The container's job is to train; failing
                # to ask for work is a reason to ask again, not to stop.
                _log(f"poll failed ({e}) — retrying")

    async def tick(self) -> None:
        wid = self._worker_id()
        if not wid:
            # SAY WHY, then say it again every few minutes. Every other part of a broken
            # claim path can be seen by asking elsewhere -- the row is green, the service
            # is ready, the store is idle -- so this line is the ONLY thing that tells
            # "the queue is empty" apart from "this box can never ask".
            if time.time() - self._no_id_said_at >= NO_WORKER_ID_LOG_S:
                _log("no worker id — NOT claiming any work. In one-container mode the render "
                     "daemon owns the row and writes its id to WORKER_ID_FILE; if this keeps "
                     "appearing while the worker row is registered, the control plane never "
                     "saw that id.")
                self._no_id_said_at = time.time()
            return
        await self.check_publish_requests()
        # Do not even ask while busy. Claiming marks the row on the API side, so asking for work
        # we cannot start would take a job out of the queue and sit on it.
        if not trainer_app.STORE.claim_slot():
            return

        r = await self._client.get(
            f"{QUEUE_URL}/training/next", params={"worker_id": wid},
            headers={"X-API-Key": QUEUE_API_KEY}, timeout=30)
        if r.status_code != 200:
            _log(f"claim returned HTTP {r.status_code}")
            return
        row = r.json()
        if not row:
            return

        _log(f"claimed {row['character']} v{row['version']} ({len(row['download_urls'])} images)")
        # THE EXTRA GROUPS (#102, #106). `identities` rides the claim response only for a
        # joint run; ABSENT for every single-identity job, so this maps to [] and the
        # trainer's stage() writes the one-group shape it always has. Each entry keeps its
        # own caption -- an identity group's is "<trigger>, <gender>", a composition group's
        # names the people in its frames.
        identities = [
            {
                "character": g.get("character"),
                "trigger": g.get("trigger"),
                "image_urls": g.get("download_urls") or [],
                "caption": g.get("caption"),
                "num_repeats": g.get("num_repeats"),
            }
            for g in (row.get("identities") or [])
        ]
        req = trainer_app.TrainRequest(
            character=row["character"], trigger=row["trigger"], version=row["version"],
            image_urls=row["download_urls"],
            caption=(row.get("config") or {}).get("caption"),
            identities=identities,
            steps=(row.get("config") or {}).get("steps") or 1200,
            config=row.get("config") or {},
            remote_id=row["id"],
        )
        job = Job(id=row["id"][:12], character=req.character, trigger=req.trigger,
                  version=req.version, steps=req.steps, remote_id=row["id"])
        self._policy[job.id] = (row.get("config") or {}).get("publish") or "final"
        trainer_app.STORE.add(job)
        asyncio.create_task(trainer_app._run(job, req, on_progress=self._report))
        await self._report(job)

    async def check_publish_requests(self) -> None:
        """Upload, after the fact, the epochs the console asked for.

        Only the final checkpoint goes up by default; the others sit in the run directory.
        The console records a request on the API row, and this -- run from the poll loop --
        is what notices. Pull, like everything else: the API never calls this box.
        """
        if not enabled():
            return
        cutoff = time.time() - PUBLISH_WATCH_DAYS * 86400
        for job in trainer_app.STORE.all():
            if not (job.done and job.remote_id and job.finished_at > cutoff):
                continue
            if not self._local_checkpoints(job):
                continue
            try:
                r = await self._client.get(
                    f"{QUEUE_URL}/training/{job.remote_id}",
                    headers={"X-API-Key": QUEUE_API_KEY}, timeout=20)
                if r.status_code != 200:
                    continue
                row = r.json()
            except Exception:
                continue
            self._policy.setdefault(job.id, (row.get("config") or {}).get("publish") or "final")
            # What the bucket already has, by label, so a restart does not re-upload.
            have = {_label_of_uri(u) for u in (row.get("checkpoints") or [])}
            for path in self._local_checkpoints(job):
                if _label(path) in have:
                    self._published.setdefault(job.id, {})[str(path)] = "(in bucket)"
            asked = set(row.get("publish_requests") or []) - have
            if asked - self._requested.get(job.id, set()):
                _log(f"{job.character} v{job.version}: asked to publish {sorted(asked)}")
            self._requested[job.id] = asked
            self._sweep_checkpoints(job)
            # A RUN THE RESTART INTERRUPTED MID-UPLOAD. Training had finished (the store says
            # completed) but the container died before "completed" reached the API, so the
            # row still says running with "0 of 1 in the bucket" and nothing will ever
            # finish it. Once everything wanted is up -- the sweep above queues what is
            # missing -- say so. Me v2 sat like that after the 15:25 reboot on 2026-09-08.
            if (job.phase == "completed" and row.get("status") in ("running", "claimed")
                    and not self._unpublished(job)
                    and not (self._uploader.get(job.id) and not self._uploader[job.id].done())):
                published = len(self._published.get(job.id, {}))
                _log(f"{job.character} v{job.version}: every wanted checkpoint is in the "
                     f"bucket; telling the API it is completed")
                await self._patch(job, {"status": "completed",
                                        "progress_log": f"{published} checkpoint(s) published"})

    # ------------------------------------------------------------- checkpoints

    def _local_checkpoints(self, job: Job) -> list[Path]:
        """The .comfy checkpoints on disk right now, in epoch order, final last.

        The .comfy variant is the one the engine loads. Every epoch is kept because choosing
        between them is a judgement made by eye at a fixed seed -- loss does not rank them.
        """
        out = recipe.run_dir(job.character, job.version) / "output"
        if not out.is_dir():
            return []
        files = sorted(out.glob("*.comfy.safetensors"))
        # The unnumbered file is the one written when the steps ran out. It sorts before the
        # numbered ones alphabetically and must not: it has the most training in it.
        return [f for f in files if EPOCH_RE.search(f.name)] + \
               [f for f in files if not EPOCH_RE.search(f.name)]

    def _wanted(self, job: Job, path: Path) -> bool:
        """Should this file go to the bucket? The final always; every epoch only when the
        run asked for "all"; anything else only when the console asked for it by name."""
        label = _label(path)
        if label == "final" or self._policy.get(job.id, "final") == "all":
            return True
        return label in self._requested.get(job.id, set())

    @staticmethod
    def _readable(path: Path) -> bool:
        """Is this a safetensors file at all? The header is a little-endian u64 length and a
        JSON blob; a file whose first eight bytes are zero has NO header.

        2026-09-08: the host hard-reset seconds after the final checkpoint was written. ext4
        kept the file's size and replaced its contents with zeros, the poller's "interrupted
        run" path uploaded it as Me_v2_final, and every render with that character failed at
        LoraLoaderModelOnly with "Expecting value: line 1 column 1" -- a JSON error that names
        nothing. The musubi-format twin beside it was intact. Checked here so a file like that
        is reported and never published.
        """
        try:
            with path.open("rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                if not 0 < n < 64 * 1024 * 1024:
                    return False
                import json
                json.loads(fh.read(n))
            return True
        except Exception:
            return False

    def _settled(self, path: Path) -> bool:
        """Has this file stopped growing? The trainer writes it in place, so a checkpoint that
        is still being written looks like a checkpoint with a smaller size."""
        try:
            size = path.stat().st_size
        except OSError:
            return False
        now = time.time()
        prev = self._seen.get(str(path))
        if prev is None or prev[0] != size:
            self._seen[str(path)] = (size, now)
            return False
        return now - prev[1] >= CHECKPOINT_SETTLE_S

    def _sweep_checkpoints(self, job: Job) -> None:
        """Queue every settled checkpoint not yet in the bucket, and make sure the uploader
        for this job is running.

        CALLED FROM EVERY PROGRESS TICK, so an epoch goes up while the next one trains. The
        uplink is slower than the trainer -- an epoch every ten minutes, eighteen minutes to
        upload one -- so the queue falls behind and the last of it goes after training ends;
        overlapping still hides most of it.
        """
        if not (enabled() and job.remote_id):
            return
        done = self._published.setdefault(job.id, {})
        failed = self._failed.setdefault(job.id, [])
        queue = self._queue.setdefault(job.id, [])
        for path in self._local_checkpoints(job):
            if (str(path) in done or str(path) in failed or path in queue
                    or str(path) in self._inflight):
                continue
            if not self._wanted(job, path):
                continue
            if self._settled(path):
                if not self._readable(path):
                    _log(f"!! {path.name} is not a readable safetensors file (no header) — "
                         f"NOT publishing it. If the host reset right after it was written, "
                         f"the .safetensors twin beside it is usually intact: regenerate with "
                         f"musubi_tuner.ltx_2.convert_lora_to_comfy.")
                    failed.append(str(path))
                    continue
                queue.append(path)
        task = self._uploader.get(job.id)
        if queue and (task is None or task.done()):
            self._uploader[job.id] = asyncio.create_task(self._upload_worker(job))

    async def _upload_worker(self, job: Job) -> None:
        """Send this job's queue, one file at a time, THE FINAL CHECKPOINT FIRST when it is
        there. It is the one the character row will point at and the one with the most
        training in it; with the queue behind the trainer it would otherwise wait behind
        every epoch that finished before it."""
        queue = self._queue[job.id]
        while queue:
            final = [p for p in queue if not EPOCH_RE.search(p.name)]
            path = final[0] if final else queue[0]
            queue.remove(path)
            self._inflight.add(str(path))
            try:
                await self._publish(job, path)
            finally:
                self._inflight.discard(str(path))

    def _unpublished(self, job: Job) -> list[Path]:
        """Wanted, and not in the bucket. An epoch nobody asked for is not missing."""
        return [p for p in self._local_checkpoints(job)
                if self._wanted(job, p) and str(p) not in self._published.get(job.id, {})]

    def _epochs(self, job: Job) -> list[dict]:
        """Every checkpoint on disk, with the step it was written at and the loss then.

        The step is arithmetic -- an epoch is images x num_repeats samples -- and the loss is
        the nearest point of the curve at or before it. Reported whether or not the file is
        uploaded, so the console can list the ones that stayed behind and offer them.
        """
        per_epoch = max(1, job.images * recipe.DEFAULTS["num_repeats"])
        out = []
        for path in self._local_checkpoints(job):
            label = _label(path)
            step = job.steps if label == "final" else min(job.steps, int(label[1:]) * per_epoch)
            before = [l for s_, l in job.loss_log if s_ <= step]
            out.append({"label": label, "step": step,
                        "loss": round(before[-1], 4) if before else None})
        return out

    def _upload_status(self, job: Job) -> str:
        total = len([p for p in self._local_checkpoints(job) if self._wanted(job, p)])
        done = len(self._published.get(job.id, {}))
        current = self._current.get(job.id)
        if not current:
            return f"{done} of {total} checkpoints in the bucket"
        sent, size = self._sent.get(job.id, (0, 0))
        pct = f" — {100 * sent // size}% of {size / 1024 ** 2:.0f} MB" if size else ""
        return f"uploading {current}{pct} — {done} of {total} in the bucket"

    def _upload_suffix(self, job: Job) -> str:
        """For the training-time progress line, when an upload overlaps the run."""
        current = self._current.get(job.id)
        if not current:
            return ""
        sent, size = self._sent.get(job.id, (0, 0))
        return f" · uploading {current} {100 * sent // size}%" if size else f" · uploading {current}"

    async def _drain_uploads(self, job: Job) -> None:
        """Get every checkpoint of this run into the bucket, or give up on it.

        Training has finished, so what is usually left is the final checkpoint -- which needs
        its settle window before it is taken -- and whatever the queue is behind on. Waits
        for those, reporting progress as it goes, and no longer: a file that never settles is
        reported missing, not waited on forever.
        """
        settle_deadline = time.time() + CHECKPOINT_SETTLE_S + 30
        last = ""
        while True:
            self._sweep_checkpoints(job)
            task = self._uploader.get(job.id)
            failed = self._failed.get(job.id, [])
            waiting = [p for p in self._unpublished(job) if str(p) not in failed]
            busy = task is not None and not task.done()
            if not waiting and not busy:
                return
            if not busy and time.time() > settle_deadline:
                return
            status = self._upload_status(job)
            if status != last:
                await self._patch(job, {"progress_log": status})
                last = status
            await asyncio.sleep(10)

    async def _publish(self, job: Job, path: Path) -> None:
        """One checkpoint, straight to S3.

        THREE CALLS AND NO AWS CREDENTIALS. The API signs a PUT for exactly this key, the
        file goes from this box to the bucket, and the API is told it landed -- it checks
        before believing. Through the API instead, as it was, each file was read whole into a
        t3.small's memory and re-sent; two of five did not survive and the job said
        "completed" anyway.

        A failure here is recorded, not raised: it must not stop the training run it is
        overlapping with, and the terminal report says which epochs are missing.
        """
        m = EPOCH_RE.search(path.name)
        params = {"epoch": int(m.group(1))} if m else {"final": "true"}
        label = f"e{int(m.group(1)):02d}" if m else "final"
        hdrs = {"X-API-Key": QUEUE_API_KEY}
        self._current[job.id] = label
        self._sent[job.id] = (0, 0)
        try:
            r = await self._client.post(
                f"{QUEUE_URL}/training/{job.remote_id}/artifact-url",
                params=params, headers=hdrs, timeout=30)
            r.raise_for_status()
            target = r.json()
            size = path.stat().st_size
            _log(f"uploading {label} ({size / 1024 ** 2:.0f} MB) -> {target['uri']}")
            t0 = time.time()
            for attempt in range(1, UPLOAD_ATTEMPTS + 1):
                t0 = time.time()
                try:
                    self._sent[job.id] = (0, size)
                    put = await self._client.put(
                        target["put_url"], content=self._counted(job, path),
                        headers={"Content-Length": str(size)},
                        # Per-operation, not total: a chunk that gets no ack in 120 s is a
                        # dead connection, but the whole file is allowed to take as long as
                        # the uplink needs.
                        timeout=httpx.Timeout(120.0))
                    if put.status_code < 300:
                        break
                    _log(f"{label} attempt {attempt}: S3 said HTTP {put.status_code}: "
                         f"{put.text[:200]}")
                except Exception as e:
                    _log(f"{label} attempt {attempt} failed after {time.time() - t0:.0f}s: {e}")
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"{label}: every attempt failed")
            r = await self._client.post(
                f"{QUEUE_URL}/training/{job.remote_id}/artifact-commit",
                params={"uri": target["uri"]}, headers=hdrs, timeout=30)
            if r.status_code >= 400:
                raise RuntimeError(f"{label}: the API refused the commit: {r.text[:300]}")
            self._published.setdefault(job.id, {})[str(path)] = target["uri"]
            _log(f"published {label} in {(time.time() - t0) / 60:.1f} min")
        except Exception as e:
            self._failed.setdefault(job.id, []).append(str(path))
            _log(f"could not publish {path.name}: {e}")
        finally:
            self._current.pop(job.id, None)
            self._sent.pop(job.id, None)

    async def _counted(self, job: Job, path: Path):
        """The file, a chunk at a time, keeping score for the status line."""
        sent = 0
        async for chunk in _file_chunks(path):
            yield chunk
            sent += len(chunk)
            self._sent[job.id] = (sent, self._sent.get(job.id, (0, 0))[1])

    async def _report(self, job: Job) -> None:
        """Mirror local state up to the API row.

        Every field is sent only when it has a value, matching the API's conditional writes: a
        report that omits a field must not blank what an earlier one set.
        """
        if not (enabled() and job.remote_id):
            return
        phase_to_status = {
            "staging": "running", "training": "running", "collecting": "running",
            "completed": "completed", "failed": "failed", "cancelled": "cancelled",
        }
        body: dict = {}
        if (st := phase_to_status.get(job.phase)):
            body["status"] = st
        if job.message:
            body["progress_log"] = job.message + (
                self._upload_suffix(job) if job.phase in ("training", "collecting") else "")
        if job.step:
            body["step"] = job.step
        if job.steps:
            body["total_steps"] = job.steps
        if job.error:
            body["error_message"] = job.error
        if job.loss_log:
            body["loss_log"] = job.loss_log
        if job.phase in ("training", "collecting", "completed"):
            epochs = self._epochs(job)
            if epochs:
                body["epochs"] = epochs
        # Local checkpoint paths are deliberately NOT sent. The checkpoints the API records
        # are the S3 URIs committed by _publish, and local ones would be buttons that 404.
        if job.phase in ("training", "collecting"):
            self._sweep_checkpoints(job)
        if job.phase == "completed":
            # EVERYTHING IS IN THE BUCKET BEFORE "completed" IS SAID. A console that reacts to
            # completed must find the checkpoints there, and the character row is pointed at
            # the last one recorded -- which has to be the final checkpoint, not whichever
            # epoch happened to finish uploading last.
            body["progress_log"] = "uploading checkpoints"
            await self._patch(job, dict(body, status="running"))
            await self._drain_uploads(job)
            missing = self._unpublished(job)
            published = self._published.get(job.id, {})
            if missing or not published:
                # The run trained, and that is not the same as the run having produced a LoRA
                # anyone can use. A green row with the final checkpoint absent is what this
                # replaces; better a red one that says where the files are.
                body["status"] = "failed"
                body["error_message"] = (
                    f"trained, but {len(missing) or 'all'} checkpoint(s) could not be uploaded: "
                    + ", ".join(Path(p).name for p in missing)
                    + f". They are on the trainer under {recipe.run_dir(job.character, job.version)}/output")
                body["progress_log"] = body["error_message"]
            else:
                body["progress_log"] = f"{len(published)} checkpoint(s) published"
        if not body:
            return
        await self._patch(job, body)

    async def _patch(self, job: Job, body: dict) -> None:
        try:
            r = await self._client.patch(
                f"{QUEUE_URL}/training/{job.remote_id}", json=body,
                headers={"X-API-Key": QUEUE_API_KEY}, timeout=20)
            # THE REPLY IS HOW A CANCEL ARRIVES. The console can only write the row -- nothing
            # in the API reaches into this box -- so the row we get back from the report we just
            # sent is the message. No polling and no second call, and it works for a job claimed
            # from the queue, which POST /train/{id}/cancel on this container cannot reach.
            if r.status_code < 400 and r.json().get("status") == "cancelled":
                if not job.cancel_requested:
                    _log(f"{job.remote_id} was cancelled in the console — stopping")
                job.cancel_requested = True
        except Exception as e:
            # Non-fatal, like every other report in this container. Losing a progress update
            # must not lose the run.
            _log(f"could not report progress for {job.remote_id}: {e}")


def _label(path: Path) -> str:
    m = EPOCH_RE.search(path.name)
    return f"e{int(m.group(1)):02d}" if m else "final"


def _label_of_uri(uri: str) -> str:
    m = re.search(r"_(e\d{2}|final)\.safetensors$", uri)
    return m.group(1) if m else ""


async def _file_chunks(path: Path):
    """The file, a piece at a time, so a 650 MB upload never holds 650 MB."""
    with path.open("rb") as fh:
        while chunk := fh.read(UPLOAD_CHUNK):
            yield chunk
