"""The render stack as services: ComfyUI, the engine API, the render daemon (wanly-gpu-docker#83).

`SERVICES=ltx-engine` is three processes, started in this order, and this module is what
start.sh's phases 1-6 became. Each one says how to start, whether it is answering, and what
must be true first -- and the supervisor's watchdog stops the container when any of them
dies, which for the daemon is exactly what its exit did as PID 1 and for ComfyUI and the
engine is new: they used to die unnoticed behind a container that stayed Up.

WHAT MOVED WHERE

    phase 1  models              -> ComfyUI.preflight (download_models.sh, unchanged)
    phase 2  daemon code + .env  -> RenderDaemon.preflight (fetch_daemon.sh, unchanged text)
    phase 3  ComfyUI             -> ComfyUI.command / ready
    phase 4  CUDA gate + wait    -> ComfyUI.preflight / ready
    phase 5  ltx-engine          -> LtxEngineApi.command / ready
    phase 6  exec daemon         -> RenderDaemon.command / ready (a file the daemon writes)
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from wanly_worker.service import PreflightError, Service

COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
API_PORT = int(os.environ.get("API_PORT", "8190"))
DAEMON_DIR = os.environ.get("DAEMON_DIR", "/app/wanly-gpu-daemon")
LOGS = os.environ.get("WANLY_LOG_DIR", "/workspace/logs")
RUN_DIR = os.environ.get("WANLY_RUN_DIR", "/run/wanly")
READY_FILE = f"{RUN_DIR}/render-daemon.ready"
WORKER_ID_FILE = f"{RUN_DIR}/worker-id"
#: Scripts, overridable so the tests can point at copies.
DOWNLOAD_MODELS = os.environ.get("DOWNLOAD_MODELS_SH", "/app/download_models.sh")
FETCH_DAEMON = os.environ.get("FETCH_DAEMON_SH", "/app/fetch_daemon.sh")


def _run_script(path: str, label: str) -> None:
    """Run a boot script with the container's stdout, so the timestamped log keeps moving.

    Not captured: download_models.sh prints progress for up to an hour on a cold pod, and a
    boot log that goes silent for that long is indistinguishable from a hang.
    """
    rc = subprocess.call(["bash", path])
    if rc != 0:
        raise PreflightError(f"{label} failed (exit {rc}) — see the lines above")


def render_env_dump(text: str) -> str:
    """The daemon's resolved .env, comments stripped and secrets redacted, indented.

    Shown in the boot log so a parity problem is visible in its first lines rather than after
    a day of results. Comments are stripped because the reasoning belongs in the source, not
    in every boot log -- and because a comment quoting an old ERROR line was once read as a
    live failure (wanly-gpu-daemon#175).
    """
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        line = re.sub(r"^(.*(KEY|TOKEN|SECRET|PASSWORD))=.+$", r"\1=<redacted>", line)
        out.append("  " + line)
    return "\n".join(out)


class ComfyUI(Service):
    name = "comfyui"
    port = COMFY_PORT
    summary = f"ComfyUI on :{COMFY_PORT} (loopback)"
    #: Cold import of the node packs has taken more than the old 120 s default; start.sh
    #: gave it 180 and so does this.
    ready_timeout_s = 180.0
    cwd = "/app/ComfyUI"
    log_path = f"{LOGS}/comfyui.log"

    def preflight(self) -> None:
        # Phase 1: the models. Refuses on a bind mount that is not a mount, a full disk, or a
        # truncated safetensors -- see download_models.sh, which is unchanged.
        _run_script(DOWNLOAD_MODELS, "models")
        # Phase 4's CUDA gate, before ComfyUI can die on import with its output going nowhere.
        probe = subprocess.run(
            ["python3", "-c", "import torch; assert torch.cuda.is_available()"],
            capture_output=True, text=True)
        if probe.returncode != 0:
            raise PreflightError(
                "torch cannot initialise CUDA on this host. A cu130 build needs driver >= 580; "
                "this image pins cu128, which runs on 550+. If the driver is older than that, "
                "this host cannot run the image — pick another. If the host just rebooted, the "
                "container may have lost its GPU: recreate it with deploy/run-worker.sh.")

    def command(self) -> list[str]:
        return ["python3", "main.py", "--listen", "127.0.0.1", "--port", str(self.port),
                "--extra-model-paths-config", "/opt/extra_model_paths.yaml",
                "--preview-method", "none"]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def after_ready(self, client) -> None:
        nodes = Path(self.cwd) / "custom_nodes"
        if nodes.is_dir():
            print("custom nodes:", flush=True)
            for n in sorted(p.name for p in nodes.iterdir()):
                if not re.search(r"__pycache__|example_node|websocket", n):
                    print(f"  {n}", flush=True)


class LtxEngineApi(Service):
    name = "ltx-engine-api"
    port = API_PORT
    summary = f"ltx-engine on :{API_PORT} (loopback)"
    ready_timeout_s = 120.0
    cwd = "/opt/engine"
    log_path = f"{LOGS}/ltx-engine.log"

    def command(self) -> list[str]:
        return ["python3", "app.py", "--host", "127.0.0.1", "--port", str(self.port),
                "--public-base", f"http://localhost:{self.port}"]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def after_ready(self, client) -> None:
        # Its own health check reports whether it can see the models and the workflow.
        # Printed rather than merely consulted: "models_present": false is the difference
        # between a worker that fails every claim and one that works.
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            print("ltx-engine up:", flush=True)
            for line in r.text.splitlines():
                print(f"  {line}", flush=True)
        except Exception:
            pass

    def details(self) -> dict:
        """`running` and `queue_depth`, so the update timer can ask this container whether
        a render is in flight the way it used to ask the engine directly."""
        try:
            import json
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                d = json.load(r)
            return {"running": d.get("running"), "queue_depth": d.get("queue_depth")}
        except Exception:
            return {}


class RenderDaemon(Service):
    name = "render-daemon"
    port = 0
    summary = "wanly-gpu-daemon: registers, claims segments, drives the engine"
    #: The daemon syncs every character LoRA it lacks before it registers; a fresh pod with
    #: several 650 MB files takes minutes. It also cannot answer a port -- readiness is the
    #: file it writes after registering.
    ready_timeout_s = 3600.0
    cwd = DAEMON_DIR
    #: A segment in flight is allowed to finish: a render is up to ~27 minutes. Docker's own
    #: stop timeout still bounds this; use `docker stop -t` to give it the room.
    stop_grace_s = 1800.0

    def preflight(self) -> None:
        # Phase 2 and 3, verbatim: clone/pull the daemon, install deps added since the image
        # was built, write its .env, print the redacted config.
        _run_script(FETCH_DAEMON, "render-daemon code")
        env = Path(DAEMON_DIR) / ".env"
        if not env.exists():
            raise PreflightError(f"{env} was not written by {FETCH_DAEMON}")
        print("Daemon config:", flush=True)
        print(render_env_dump(env.read_text()), flush=True)
        for stale in (READY_FILE,):
            try:
                os.unlink(stale)
            except FileNotFoundError:
                pass

    def command(self) -> list[str]:
        return ["python3", "-m", "daemon.main"]

    def env(self) -> dict[str, str]:
        # The handshake (wanly-gpu-daemon#185): the daemon writes these after registering.
        return {"WANLY_READY_FILE": READY_FILE, "WORKER_ID_FILE": WORKER_ID_FILE}

    async def ready(self, client) -> bool:
        return os.path.exists(READY_FILE)

    def details(self) -> dict:
        try:
            out = subprocess.run(["git", "-C", DAEMON_DIR, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            return {"daemon_commit": out.stdout.strip() or None}
        except Exception:
            return {}


def ltx_engine_group() -> list[Service]:
    """`SERVICES=ltx-engine`: the three processes, in the order they must start."""
    return [ComfyUI(), LtxEngineApi(), RenderDaemon()]
