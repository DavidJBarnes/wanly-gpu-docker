"""lora-trainer as a wanly_worker service (wanly-services#7, wanly-console#453; moved into the
one-container-per-GPU image in wanly-gpu-docker#83).

FITTING THE CONTRACT. `Service` expects a long-running child process that answers on a port, and
a trainer is a job runner rather than a server — so the child IS a small FastAPI app, and
`ready()` probes it. That keeps the supervisor's lifecycle honest: preflight before anything
starts, readiness meaning "answering" rather than "spawned", and a dead child taking the
container down for Docker to restart.

WHY IT IS ITS OWN WorkerKind. A box running the trainer registers with `trainer` among its
kinds. The API's claim gates key on it: a TRAINER takes training jobs, a SERVICE takes none.
On the 3090 the same row is also `render` -- the render daemon registers the box once as
["render", "trainer"] and the trainer drains that very row before it takes the card.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.lora_trainer import recipe

PORT = int(os.environ.get("TRAINER_PORT", "8082"))


class LoraTrainer(Service):
    name = "lora-trainer"
    port = PORT
    summary = f"character-LoRA training on :{PORT}"

    def preflight(self) -> None:
        """Refuse now rather than fifty minutes in. Each of these has actually happened."""
        if not Path(recipe.TRAINER_PYTHON).exists():
            raise PreflightError(
                f"no trainer at {recipe.TRAINER_PYTHON}. This image was built without "
                f"WITH_TRAINER=1 — the lean :latest tag carries the render stack only, and "
                f"the trainer needs the :full tag (IMAGE=davidjbarnes/wanly-gpu-docker:full).")
        for label, path in (("base checkpoint", recipe.CKPT), ("Gemma", recipe.GEMMA)):
            if not Path(path).exists():
                raise PreflightError(
                    f"{label} is not at {path}. It is bind-mounted from the host "
                    f"(-v <host models>:{recipe.MODELS_DIR}:ro); the image does not carry it.")
        # THE CARD MUST BE VISIBLE TO TORCH, not just to nvidia-smi. After the host rebooted
        # on 2026-09-08 the container came back under its restart policy with the device
        # gone -- nvidia-smi listed it, torch said "No CUDA GPUs are available" -- and the
        # first job died in latent caching twenty seconds after the claim. Checked here so
        # the container refuses to come up ready and the fix (recreate it) is obvious.
        probe = subprocess.run(
            [recipe.TRAINER_PYTHON, "-c",
             "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
            capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise PreflightError(
                "torch in the trainer venv sees no CUDA device. If the host rebooted, the "
                "container lost its GPU: recreate it with deploy/run-worker.sh. "
                f"({probe.stderr.strip()[-200:] or 'no CUDA device'})")
        runs = Path(recipe.RUNS_DIR)
        if not runs.is_dir():
            raise PreflightError(
                f"{recipe.RUNS_DIR} does not exist. Run directories and their checkpoints live "
                f"there and it must be a mount — the container overlay cannot hold them.")
        if not os.access(runs, os.W_OK):
            raise PreflightError(f"{recipe.RUNS_DIR} is not writable.")
        # ~2.25 GB per epoch, measured. A run that cannot store its checkpoints is a run that
        # fails at the end, having spent the whole hour.
        free = shutil.disk_usage(runs).free / 1024 ** 3
        if free < 25:
            raise PreflightError(
                f"only {free:.0f} GB free under {recipe.RUNS_DIR}; a 1200-step run on a small "
                f"dataset writes ~20 GB of checkpoints.")

    def command(self) -> list[str]:
        return ["python3", "-m", "uvicorn",
                "wanly_worker.services.lora_trainer.app:app",
                "--host", "127.0.0.1", "--port", str(self.port)]

    def env(self) -> dict[str, str]:
        # 127.0.0.1 above, not 0.0.0.0: unlike image-description, nothing outside this container should
        # reach the trainer directly. The console goes through wanly-api, and the CLI goes
        # through the published control port.
        return {"PYTHONUNBUFFERED": "1"}

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def details(self) -> dict:
        """What /health says about this service. Must never raise."""
        try:
            from wanly_worker.services.lora_trainer.app import STORE
            active = STORE.active()
            if not active:
                return {"training": None, "jobs": len(STORE.all())}
            s = active.snapshot()
            return {"training": f"{s['character']} v{s['version']}",
                    "step": s["step"], "of": s["steps"], "pct": s["pct"],
                    "phase": s["phase"]}
        except Exception:
            return {}
