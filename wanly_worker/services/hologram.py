"""`SERVICES=hologram`: AR holograms and smashcut concats, off the render boxes (#228).

Holograms used to run only inside the render daemon, behind its work lock (wanly-gpu-daemon
#192): one waited for the render in flight -- up to ~30 minutes at 720p -- and a 3090 that
picked one up held its next render for the 4-6.5 minutes it took. Neither needs a 3090, so this
runs them in the 2070's scene container instead, beside the captioner.

ONE PROCESS: the daemon's hologram-only entrypoint (`python -m daemon.hologram_worker`), from
the same daemon code the render daemon fetches. No ComfyUI, no render loop, no LoRA sync.

IT REGISTERS NOTHING. The supervisor's queue client is this box's registrar, and kinds_for()
makes the row kind `hologram`, which is what the API's claim gate lets claim CPU carriers. The
worker claims against that row's id, which the queue client writes to WORKER_ID_FILE.

SHARING THE BOX. The image's onnxruntime is CPU-only, so a hologram never touches the card that
scene-caption and prompt-enhance take turns on. It shares RAM with them, under the container's
cap: the daemon streams frames from disk (376 MB peak for a real 241-frame 832x1216 clip, where
the old whole-clip pass held ~2.4 GB) and claims only with HOLOGRAM_MIN_HEADROOM_GB free. It
shares the host's CPUs with the 3090's render, so ONNX_THREADS defaults to half the cores.
"""
from __future__ import annotations

import os
import subprocess

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.ltx_engine import DAEMON_DIR, FETCH_DAEMON, RUN_DIR, _run_script

READY_FILE = f"{RUN_DIR}/hologram.ready"
#: The registrar's row id. The same path the render daemon writes on a render box.
WORKER_ID_FILE = f"{RUN_DIR}/worker-id"


class HologramWorker(Service):
    name = "hologram"
    port = 0
    summary = "AR holograms + smashcut concats (CPU), claimed against this box's row"
    ready_timeout_s = 120.0
    cwd = DAEMON_DIR
    #: A hologram in flight is allowed to finish: 723 frames took 6.5 minutes on 3090a.
    stop_grace_s = 900.0

    def preflight(self) -> None:
        # The same fetch the render daemon uses: its code, deps added since the build, .env.
        _run_script(FETCH_DAEMON, "hologram daemon code")
        if not os.path.exists(os.path.join(DAEMON_DIR, "daemon", "hologram_worker.py")):
            raise PreflightError(
                f"{DAEMON_DIR} has no daemon/hologram_worker.py -- the daemon fetched is older "
                f"than the hologram worker (wanly-gpu-docker#228). Check DAEMON_BRANCH.")

    def prestart(self) -> None:
        try:
            os.unlink(READY_FILE)
        except FileNotFoundError:
            pass

    def command(self) -> list[str]:
        return ["python3", "-m", "daemon.hologram_worker"]

    def env(self) -> dict[str, str]:
        return {
            "WANLY_READY_FILE": READY_FILE,
            "WORKER_ID_FILE": WORKER_ID_FILE,
            "ONNX_THREADS": os.environ.get("ONNX_THREADS")
                            or str(max(1, (os.cpu_count() or 2) // 2)),
        }

    async def ready(self, client) -> bool:
        return os.path.exists(READY_FILE)

    def details(self) -> dict:
        try:
            out = subprocess.run(["git", "-C", DAEMON_DIR, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            return {"daemon_commit": out.stdout.strip() or None}
        except Exception:
            return {}
