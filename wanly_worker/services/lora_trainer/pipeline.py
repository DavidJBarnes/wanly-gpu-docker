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
    # THE JOB'S base, not the boot-time default (#145). The service preflight proved the
    # default is mounted; a job naming a checkpoint this box does not hold must fail HERE,
    # before the drain and the hour, and must say which name it was and where it looked --
    # "not at <path>" alone reads as a broken mount when the real answer is "wrong box".
    ckpt = job.base_checkpoint or recipe.CKPT
    if not Path(ckpt).exists():
        raise PipelineError(
            f"base checkpoint {Path(ckpt).stem!r} is not at {ckpt}. The job's "
            f"config.base_checkpoint names a file under {recipe.MODELS_DIR}/ltx-2.3/"
            f"diffusion_models, which is bind-mounted from the host: either this box does not "
            f"hold that checkpoint or the mount is missing.")
    if not Path(recipe.GEMMA).exists():
        raise PipelineError(
            f"Gemma is not at {recipe.GEMMA}. It is bind-mounted from the host; without the "
            f"mount there is nothing to train against.")
    if not Path(recipe.TRAINER_PYTHON).exists():
        raise PipelineError(
            f"no trainer at {recipe.TRAINER_PYTHON} — this image was built without "
            f"WITH_TRAINER=1")

    epochs = recipe.estimated_epochs(job.images, job.steps, per_epoch=job.per_epoch)
    need = epochs * GB_PER_EPOCH
    free = shutil.disk_usage(recipe.RUNS_DIR).free / 1024 ** 3
    if free < need + 5:
        raise PipelineError(
            f"~{need:.0f} GB of checkpoints expected ({epochs} epochs) and only {free:.0f} GB "
            f"free under {recipe.RUNS_DIR}")


def check_captions(groups: list[dict]) -> None:
    """Refuse a caption list that does not line up with its images, before anything is staged.

    PER-IMAGE CAPTIONS (#145) are a list parallel to the group's images, in download order.
    A list one short does not fail anywhere downstream: every caption after the gap binds to
    the NEXT image, the run trains for an hour on text describing the wrong pictures, and the
    LoRA comes out subtly wrong with nothing in any log to say why. So a length mismatch is a
    refusal, and so is an empty entry -- an uncaptioned image trains the trigger onto nothing.

    NO FALLBACK ON THE NEW SHAPE. A job that carries per-image captions for ANY group must
    carry them for EVERY group: the old `"<trigger>, woman"` default is wrong for a
    regularization or composition group (no trigger; not necessarily a woman), and inventing
    one silently is exactly what this contract replaces. A job with no lists anywhere is a
    legacy row -- a retry from before #145 -- and keeps the single-caption path it was built
    for.
    """
    new_shape = any(g.get("captions") is not None for g in groups)
    for gi, g in enumerate(groups):
        caps = g.get("captions")
        if caps is None:
            if new_shape:
                raise PipelineError(
                    f"group {gi} ({g.get('kind') or 'identity'}) has no per-image captions "
                    f"while other groups do. Every group of a per-image-caption job needs its "
                    f"own list; there is no default caption to fall back on.")
            continue
        n = len(g["images"])
        if len(caps) != n:
            raise PipelineError(
                f"group {gi} ({g.get('kind') or 'identity'}) has {n} images but {len(caps)} "
                f"captions. They are paired by position, so a mismatch would caption every "
                f"image after the gap with another image's text.")
        blank = [i for i, c in enumerate(caps) if not str(c or "").strip()]
        if blank:
            raise PipelineError(
                f"group {gi} ({g.get('kind') or 'identity'}) has empty captions at "
                f"{blank[:10]}{'...' if len(blank) > 10 else ''}")


async def stage(job: Job, groups: list[dict]) -> Path:
    """Write the dataset(s) and the config. Returns the run directory.

    `groups` is one entry per group: {images: [(name, bytes)], captions | caption,
    num_repeats, kind}. `captions` is the per-image list (#145) and wins; `caption` is the
    single string every image of a legacy group shares. ONE entry is the single-identity
    shape every run before #102 wrote — same directories (`data/`, `cache/`), same toml
    bytes. A joint run writes `data0/`+`cache0/`, `data1/`+`cache1/`, per-group captions and
    a toml entry per group, each with its OWN num_repeats.

    The run directory is REBUILT, not added to. A version that reuses a directory trains on the
    previous version's leftover images while reporting the new count -- the failure
    new_character.sh had until it grew a version argument.
    """
    # Before the rmtree below: a refused job must not also destroy the run directory a retry
    # of the previous attempt could still be read from.
    check_captions(groups)
    for gi, g in enumerate(groups):
        if not isinstance(g["num_repeats"], int) or g["num_repeats"] < 1:
            raise PipelineError(f"group {gi} num_repeats={g['num_repeats']!r}; it must be >= 1")

    run = recipe.run_dir(job.character, job.version)
    if run.exists():
        shutil.rmtree(run)
    for sub in ("data", "cache", "output", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    toml_groups = []
    for gi, g in enumerate(groups):
        # Group 0 keeps the bare dirs so a single-identity run hashes and stages exactly
        # as it always has; joint groups get numbered ones, which cannot collide with a
        # pre-existing bare dir after the rebuild above.
        data_dir = run / "data" if gi == 0 else run / f"data{gi}"
        cache_dir = run / "cache" if gi == 0 else run / f"cache{gi}"
        if gi > 0:
            data_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

        caps = g.get("captions")
        _log(job, f"staging group {gi}"
                  + (f" ({g['kind']})" if g.get("kind") else "")
                  + f": {len(g['images'])} images ({g['num_repeats']} repeats, "
                  + ("per-image captions" if caps is not None else "one shared caption")
                  + ")")
        for i, (name, blob) in enumerate(g["images"]):
            ext = Path(name).suffix.lower() or ".jpg"
            (data_dir / f"sel_{i:03d}{ext}").write_bytes(blob)
            # Captions bind whatever they do not name. All 13 of p@y's read "p@y, woman" over
            # close-ups, so the trigger carried close-up framing as part of its identity --
            # which is why a caption now describes ITS image (#145) rather than the set.
            # PER GROUP: a joint run's group 1 must say its OWN trigger, or its face binds
            # to whatever the text happens to be.
            if caps is not None:
                text = str(caps[i]).strip()
            else:
                # LEGACY ONLY: a row from before per-image captions, retried. The fallback is
                # what those jobs always got; check_captions keeps it off the new shape.
                text = g.get("caption") or f"{job.trigger}, woman"
            (data_dir / f"sel_{i:03d}.txt").write_text(text + "\n")
        toml_groups.append({"data": str(data_dir), "cache": str(cache_dir),
                            "num_repeats": g["num_repeats"]})

    (run / "dataset.toml").write_text(recipe.dataset_toml(run, toml_groups))
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
    """Cache latents, cache text, train. Reports progress while the last one runs.

    ONE base for all three (#145): resolved once, from the job, and handed to each command.
    Latents cached against one checkpoint and trained against another is a run that finishes
    cleanly and learns the wrong thing."""
    ckpt = job.base_checkpoint or recipe.checkpoint_path((config or {}).get("base_checkpoint"))
    await _stage_cmd(job, run, "caching latents",
                     recipe.cache_latents_cmd(run, ckpt), "01_cache_latents.log")
    await _stage_cmd(job, run, "caching text encoder",
                     recipe.cache_text_cmd(run, ckpt), "02_cache_text.log")

    _log(job, "training")
    t0 = time.time()
    logfile = run / "logs" / "03_train.log"
    with logfile.open("wb") as out:
        proc = await asyncio.create_subprocess_exec(
            *recipe.train_cmd(run, job.character, job.version, config, ckpt),
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
