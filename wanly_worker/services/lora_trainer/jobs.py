"""Job state, and why it is on disk rather than only in memory.

A training run is ~50 minutes. A container restart inside that window must not leave the fleet
guessing: the API still has the row CLAIMED, the render worker may still be drained, and the
checkpoints on disk are real work. So the state that matters survives, in the run directory
beside the artifacts it describes.

The persisted file is also what the drain reconciler reads at startup. That is the part that is
not optional -- see gpu.py.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from wanly_worker.services.lora_trainer import recipe

#: Where a restart looks for unfinished business.
STATE_DIR = Path(recipe.RUNS_DIR) / ".trainer"


@dataclass
class Job:
    id: str
    character: str
    trigger: str
    version: int
    steps: int
    images: int = 0
    #: pending | staging | training | collecting | completed | failed | cancelled
    phase: str = "pending"
    step: int = 0
    rate_s_per_it: float = 0.0
    message: str = ""
    error: str = ""
    checkpoints: list[str] = field(default_factory=list)
    #: [[step, avr_loss], ...] as read off the progress bar, one point per step seen.
    loss_log: list = field(default_factory=list)
    output: str = ""
    #: The wanly-api row this came from, when it was claimed rather than posted directly.
    remote_id: str = ""
    #: The worker whose drain we are holding. Persisted so a crash can release it.
    drained_worker_id: str = ""
    #: SET BY A CANCEL, READ BY THE RUN. Not `phase`, because _run writes phase as it moves
    #: through staging/training/collecting and would overwrite a cancel written from outside
    #: within seconds -- which is exactly how cancelling came to do nothing at all.
    cancel_requested: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def done(self) -> bool:
        return self.phase in ("completed", "failed", "cancelled")

    def snapshot(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished_at or time.time()) - self.started_at)
        d["pct"] = round(100 * self.step / self.steps, 1) if self.steps else 0.0
        return d


class Store:
    """The jobs this container knows about. One at a time actually runs.

    Serialised on purpose: training takes essentially the whole card, so a second concurrent run
    would not be twice as fast, it would be an OOM. The lock is what makes `POST /train` while
    one is running a clean 409 rather than a race.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        #: Jobs found mid-run at startup and failed here. The API has to be told, by whoever
        #: starts this store and can reach it.
        self._unreported: list[Job] = []
        # Tolerated, not required. This module is imported by tests and by anything that just
        # wants the app object, neither of which has the /loras mount -- and a service that
        # cannot be imported without its bind mount cannot be tested at all. The directory is
        # created again on the first write, which is the point where it genuinely must exist.
        self._ready = self._ensure_dir()
        if self._ready:
            self._load()

    @staticmethod
    def _ensure_dir() -> bool:
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    # ---------------------------------------------------------------- disk

    def _path(self, job_id: str) -> Path:
        return STATE_DIR / f"{job_id}.json"

    def _load(self) -> None:
        for f in STATE_DIR.glob("*.json"):
            try:
                job = Job(**json.loads(f.read_text()))
            except Exception as e:
                # A corrupt state file must not stop the container booting. Losing one job's
                # metadata is recoverable; refusing to start is not.
                print(f"[trainer] ignoring unreadable state {f.name}: {e}", flush=True)
                continue
            if not job.done:
                # NOTHING SURVIVES A RESTART BUT THE FILE. The process that was staging or
                # training is gone, so a job persisted mid-run is finished, and finished
                # badly. Left as it was, it counted as "active": the one-at-a-time slot was
                # taken forever, the drain it held was not an orphan to the reconciler
                # (which only looks at done jobs), and the API row stayed RUNNING with a
                # progress log, which is the one shape the orphan reclaim leaves alone.
                was = job.phase
                job.phase = "failed"
                job.error = f"the trainer restarted while this run was {was}"
                self._unreported.append(job)
                self.persist(job)
            self._jobs[f.stem] = job

    def persist(self, job: Job) -> None:
        """Write the job's state. Failing to persist must not fail the training run.

        The state file is how a restart reconciles; losing it costs recovery, not the run. A
        crash here would throw away 50 minutes of GPU to protect a bookkeeping file.
        """
        if not (self._ready or self._ensure_dir()):
            return
        self._ready = True
        try:
            self._path(job.id).write_text(json.dumps(asdict(job), indent=1))
        except OSError as e:
            print(f"[trainer] could not persist {job.id}: {e}", flush=True)

    # --------------------------------------------------------------- access

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
        self.persist(job)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def active(self) -> Job | None:
        """The one that is running, if any."""
        return next((j for j in self._jobs.values() if not j.done), None)

    def claim_slot(self) -> bool:
        """True if nothing else is running. Held only long enough to decide."""
        with self._lock:
            return self.active() is None

    def failed_by_restart(self) -> list[Job]:
        """Jobs that were mid-run when the container last died. Cleared on read."""
        out, self._unreported = self._unreported, []
        return out

    def orphaned_drains(self) -> list[tuple[str, str]]:
        """(job_id, worker_id) for drains recorded against jobs that are not running.

        This is the reconciler's input. A drain SURVIVES worker re-registration by design --
        `reregistered_drain_state` in wanly-api says cancelling one is an explicit action -- so a
        trainer that died holding one leaves the render queue stopped with nothing pointing at
        the cause. Every start has to check.
        """
        return [(j.id, j.drained_worker_id) for j in self._jobs.values()
                if j.drained_worker_id and j.done]

    def clear_drain(self, job: Job) -> None:
        job.drained_worker_id = ""
        self.persist(job)
