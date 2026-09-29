"""image-edit as wanly-gpu-docker services (wanly-console#548).

TWO PROCESSES, ONE NAME, like ltx-engine: a ComfyUI of its own (loopback) with the Qwen tree as
its model path, and the small API wanly-api calls (app.py). Its own ComfyUI rather than the
render stack's because the two never run together -- edit mode stops the render stack -- and a
separate instance with its own extra_model_paths keeps the LTX tree's config untouched. Same
install (/app/ComfyUI), no custom nodes: every node in the graph is core ComfyUI.

MODE-ONLY (registry.MODE_ONLY). Qwen-Rapid-AIO is ~20 GB on the card with no CPU fallback worth
the name, so it runs in edit mode and nowhere else. The supervisor builds it at boot and holds it
stopped; POST /mode {"mode": "edit"} starts it once the render daemon has finished its segment.

THE MODELS ARE MOUNTED, read-only, from the 3090's ~/models/qwen (run-worker.sh). The preflight
runs `download_models.sh --image-edit`, which checks the checkpoint (and AuraFace) against
their safetensors headers (a truncated file is a valid header over missing data) and fetches
them only where the tree is not a host mount.
"""
from __future__ import annotations

import os
import subprocess

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.image_edit import app as edit_app

PORT = int(os.environ.get("IMAGE_EDIT_PORT", "8086"))
MODELS_DIR = os.environ.get("IMAGE_EDIT_MODELS_DIR", "/workspace/qwen")
COMFY_DIR = os.environ.get("IMAGE_EDIT_COMFY_DIR", "/app/ComfyUI")
LOGS = os.environ.get("WANLY_LOG_DIR", "/workspace/logs")
DOWNLOAD_MODELS = os.environ.get("DOWNLOAD_MODELS_SH", "/app/download_models.sh")
PATHS_YAML = os.path.join(edit_app.WORK_DIR, "extra_model_paths.yaml")


def paths_yaml(models_dir: str = MODELS_DIR) -> str:
    """ComfyUI's model paths for this instance: the checkpoint's version dir and the LoRAs.

    Written at start rather than baked, so IMAGE_EDIT_MODELS_DIR moves it without a rebuild.
    """
    return (
        "qwen:\n"
        f"    base_path: {models_dir}\n"
        "    checkpoints: |\n"
        "        v23/\n"
        "        v19/\n"
        "    loras: loras/\n"
    )


class ImageEditComfy(Service):
    name = "image-edit-comfyui"
    port = edit_app.COMFY_PORT
    summary = f"ComfyUI for Qwen-Image-Edit on :{edit_app.COMFY_PORT} (loopback)"
    ready_timeout_s = 180.0
    cwd = COMFY_DIR
    log_path = f"{LOGS}/image-edit-comfyui.log"
    #: A Qwen edit is under a minute; nothing is lost by not waiting longer for it.
    stop_grace_s = 60.0

    def preflight(self) -> None:
        if not os.path.isfile(os.path.join(COMFY_DIR, "main.py")):
            raise PreflightError(f"ComfyUI is not at {COMFY_DIR}")
        if not os.path.isdir(MODELS_DIR):
            raise PreflightError(
                f"the Qwen model tree {MODELS_DIR} is not there. On the 3090 it is a read-only "
                f"bind mount of ~/models/qwen (IMAGE_EDIT_MODELS_DIR in worker.env, see "
                f"deploy/README.md).")
        rc = subprocess.call(["bash", DOWNLOAD_MODELS, "--image-edit"])
        if rc != 0:
            raise PreflightError(f"image-edit models failed their check (exit {rc}) -- see above")
        os.makedirs(LOGS, exist_ok=True)
        os.makedirs(os.path.join(edit_app.WORK_DIR, "in"), exist_ok=True)
        os.makedirs(os.path.join(edit_app.WORK_DIR, "out"), exist_ok=True)
        with open(PATHS_YAML, "w") as f:
            f.write(paths_yaml())

    def command(self) -> list[str]:
        return ["python3", "main.py", "--listen", "127.0.0.1", "--port", str(self.port),
                "--extra-model-paths-config", PATHS_YAML,
                "--input-directory", os.path.join(edit_app.WORK_DIR, "in"),
                "--output-directory", os.path.join(edit_app.WORK_DIR, "out"),
                # Every node in the graph is core; the LTX packs only slow the boot.
                "--disable-all-custom-nodes", "--preview-method", "none"]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


class ImageEditApi(Service):
    name = "image-edit"
    port = PORT
    summary = f"Qwen-Image-Edit full-mode edits (instruction or head angle) on :{PORT}"

    def command(self) -> list[str]:
        # 0.0.0.0: wanly-api calls this across the network, like face-edit.
        return ["python3", "-m", "uvicorn", "wanly_worker.services.image_edit.app:app",
                "--host", "0.0.0.0", "--port", str(self.port)]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def details(self) -> dict:
        return {"models_dir": MODELS_DIR}


def image_edit_group() -> list[Service]:
    """ComfyUI first: the API is useless until it answers."""
    return [ImageEditComfy(), ImageEditApi()]
