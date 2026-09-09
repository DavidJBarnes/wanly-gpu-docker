"""Stage a dataset, train, collect the checkpoints (wanly-services#7).

The phases `make_lora.py` established, but LOCAL — this runs on the GPU box rather than ssh'ing
to it, which removes most of that script's machinery. What it keeps is the ordering and the
checks, because those were each paid for.

PROGRESS IS READ WITH tail -c SEMANTICS, NOT LINES. Stage 3 writes a carriage-return progress
bar with no newlines, so a line-oriented reader never moves and a perfectly healthy 50-minute
run looks hung. That has cost real evenings.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path

from wanly_worker.services.lora_trainer import recipe
from wanly_worker.services.lora_trainer.jobs import Job

#: The trainer emits "  45%|####  | 540/1200 [..., 2.35s/it]".
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
RATE_RE = re.compile(r"([\d.]+)s/it")
#: The bar's postfix: "..., avr_loss=0.734, loss_a=n/a, loss_v=0.562]". avr_loss is the running
#: average the trainer itself reports; it is the number a person watching a run reads.
LOSS_RE = re.compile(r"avr_loss=([\d.]+)")
#: Measured on p@y and l@ura: 0.65 GB of weights per epoch, written twice plus a resume state.
GB_PER_EPOCH = 2.25


def _log(job: Job, msg: str) -> None:
    job.message = msg
    print(f"[trainer] {job.character} v{job.version}: {msg}", flush=True)


class PipelineError(RuntimeError):
    pass


class Cancelled(PipelineError):
    """Asked to stop, and stopped. Distinct from PipelineError so the run reports `cancelled`
    rather than `failed` -- a cancelled run is not a broken one, and filing it as a failure puts
    a red row on the board for something that went exactly as asked."""


def preflight(job: Job) -> None:
    """Everything knowable before an hour of GPU is spent. All of it has bitten someone."""
    for label, path in (("base checkpoint", recipe.CKPT), ("Gemma", recipe.GEMMA)):
        if not Path(path).exists():
            raise PipelineError(
                f"{label} is not at {path}. It is bind-mounted from the host; without the mount "
                f"there is nothing to train against.")
    if not Path(recipe.TRAINER_PYTHON).exists():
        raise PipelineError(
            f"no trainer at {recipe.TRAINER_PYTHON} — this image was built without "
            f"WITH_TRAINER=1")

    epochs = recipe.estimated_epochs(job.images, job.steps)
    need = epochs * GB_PER_EPOCH
    free = shutil.disk_usage(recipe.RUNS_DIR).free / 1024 ** 3
    if free < need + 5:
        raise PipelineError(
            f"~{need:.0f} GB of checkpoints expected ({epochs} epochs) and only {free:.0f} GB "
            f"free under {recipe.RUNS_DIR}")


async def stage(job: Job, images: list[tuple[str, bytes]], caption: str | None) -> Path:
    """Write the dataset and the config. Returns the run directory.

    The run directory is REBUILT, not added to. A version that reuses a directory trains on the
    previous version's leftover images while reporting the new count -- the failure
    new_character.sh had until it grew a version argument.
    """
    run = recipe.run_dir(job.character, job.version)
    if run.exists():
        shutil.rmtree(run)
    for sub in ("data", "cache", "output", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    _log(job, f"staging {len(images)} images")
    for i, (name, blob) in enumerate(images):
        ext = Path(name).suffix.lower() or ".jpg"
        (run / "data" / f"sel_{i:03d}{ext}").write_bytes(blob)
        # Captions bind whatever they do not name. All 13 of p@y's read "p@y, woman" over
        # close-ups, so the trigger carried close-up framing as part of its identity.
        (run / "data" / f"sel_{i:03d}.txt").write_text(
            (caption or f"{job.trigger}, woman") + "\n")

    (run / "dataset.toml").write_text(recipe.dataset_toml(run))
    return run


async def _stage_cmd(job: Job, run: Path, label: str, argv: list[str], logfile: str) -> None:
    _log(job, label)
    t0 = time.time()
    with (run / "logs" / logfile).open("wb") as out:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=recipe.TRAINER_DIR, stdout=out, stderr=asyncio.subprocess.STDOUT)
        rc = await proc.wait()
    if rc != 0:
        tail = (run / "logs" / logfile).read_bytes()[-1500:].decode("utf8", "replace")
        raise PipelineError(f"{label} failed (rc={rc}). Last output:\n{tail}")
    _log(job, f"{label} done in {(time.time() - t0) / 60:.1f} min")


def read_progress(run: Path, expect_total: int = 0) -> tuple[int, int, float, float | None]:
    """Steps done, total, seconds per iteration, and the running average loss (or None).

    THE LAST 2 KB, not the last lines. The progress bar is carriage-return separated, so a
    healthy run produces one enormous line and `tail -n` shows nothing new for 50 minutes.

    ANCHORED ON THE EXPECTED TOTAL, because a training log carries several progress bars and
    "any pair with a big enough denominator" picks the wrong one. The first real run reported
    `step 1747/5947` for a 1200-step job -- a model-loading bar -- which then overwrote the
    job's own idea of how many steps it had. A bar whose total is the configured step count is
    the training bar; nothing else is.

    With no expected total (a direct POST that did not say), it falls back to the old
    heuristic, which is better than nothing and no worse than it was.
    """
    log = run / "logs" / "03_train.log"
    if not log.exists():
        return 0, 0, 0.0, None
    with log.open("rb") as fh:
        fh.seek(max(0, log.stat().st_size - 2048))
        tail = fh.read().decode("utf8", "replace")
    done = total = 0
    for m in STEP_RE.finditer(tail):
        a, b = int(m.group(1)), int(m.group(2))
        if expect_total:
            if b == expect_total:
                done, total = a, b
        elif b >= 50:
            done, total = a, b
    rate = float(m.group(1)) if (m := RATE_RE.search(tail)) else 0.0
    losses = LOSS_RE.findall(tail)
    loss = float(losses[-1]) if losses else None
    return done, total, rate, loss


async def train(job: Job, run: Path, config: dict, on_progress=None) -> None:
    """Cache latents, cache text, train. Reports progress while the last one runs."""
    await _stage_cmd(job, run, "caching latents",
                     recipe.cache_latents_cmd(run), "01_cache_latents.log")
    await _stage_cmd(job, run, "caching text encoder",
                     recipe.cache_text_cmd(run), "02_cache_text.log")

    _log(job, "training")
    t0 = time.time()
    logfile = run / "logs" / "03_train.log"
    with logfile.open("wb") as out:
        proc = await asyncio.create_subprocess_exec(
            *recipe.train_cmd(run, job.character, job.version, config),
            cwd=recipe.TRAINER_DIR, stdout=out, stderr=asyncio.subprocess.STDOUT)
        while proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=20)
            except asyncio.TimeoutError:
                pass
            done, total, rate, loss = read_progress(run, expect_total=job.steps)
            if total:
                # job.steps is NOT reassigned from the log. It is what the job asked for, and
                # letting a parsed number overwrite it is how a mis-read bar rewrote the job.
                job.step, job.rate_s_per_it = done, rate
                if loss is not None and (not job.loss_log or job.loss_log[-1][0] != done):
                    # One point per distinct step seen: the curve the console draws.
                    job.loss_log.append([done, loss])
                left = (total - done) * rate / 60 if rate else 0
                _log(job, f"step {done}/{total} ({100*done/total:.0f}%), "
                          f"{rate:.2f}s/it, ~{left:.0f} min left"
                          + (f", loss {loss:.3f}" if loss is not None else ""))
            if on_progress:
                # The report is also how a cancel arrives: the API returns the row, and a row
                # that says cancelled sets the flag below. So this call has to come BEFORE the
                # check, or every cancel waits an extra tick.
                await on_progress(job)
            if job.cancel_requested:
                # TERMINATE, NOT KILL. accelerate cleans up its child processes and the CUDA
                # context on SIGTERM; SIGKILL leaves them holding the card, and the next run
                # then OOMs against a GPU that looks free. Escalate only if it will not go.
                _log(job, "cancelled — stopping the trainer")
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=60)
                except asyncio.TimeoutError:
                    _log(job, "trainer did not stop in 60s — killing it")
                    proc.kill()
                    await proc.wait()
                raise Cancelled("cancelled")
    if proc.returncode != 0:
        tail = logfile.read_bytes()[-1500:].decode("utf8", "replace")
        raise PipelineError(f"training failed (rc={proc.returncode}). Last output:\n{tail}")
    _log(job, f"training finished in {(time.time() - t0) / 60:.0f} min")


def collect(job: Job, run: Path) -> list[str]:
    """The .comfy checkpoints, in epoch order.

    The .comfy variant is the one the engine loads. Every epoch is kept because choosing between
    them is a judgement made by eye at a fixed seed -- loss does not rank them, and a confident
    "later epochs overfit" call read off a loss curve was refuted outright on d0ggyff.
    """
    out = sorted((run / "output").glob("*.comfy.safetensors"))
    job.checkpoints = [str(p) for p in out]
    _log(job, f"{len(out)} checkpoint(s)")
    return job.checkpoints
