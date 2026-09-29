"""face-edit as a wanly-gpu-docker service (wanly-console#547).

WHY IT IS NOT A MODE. Modes exist to stop one tenant so another can have the card: a render
holds ~23 of 24 GB, a Qwen-class captioner ~20. LivePortrait needs ~1.5-2 GB for about a second
per edit, and it has a CPU fallback the others do not. A mode switch for it would stop a render
(minutes, a segment finished first) to do one second of work. Instead it runs in EVERY mode --
it claims no work, so registry.select_mode keeps it in caption mode automatically -- and decides
per edit whether the GPU is free to borrow (gpu.py), falling back to CPU when it is not.

NON-ESSENTIAL. Its preflight failing (the lean tag, a missing model) leaves the service down and
reported on /health, and boots everything else. The trainer's #105/#111 is the precedent: a
side service that takes the render box down with it hides an entire queue.
"""
from __future__ import annotations

import os

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.face_edit import engine as eng

PORT = int(os.environ.get("FACE_EDIT_PORT", "8085"))


class FaceEdit(Service):
    name = "face-edit"
    port = PORT
    summary = f"LivePortrait face edits (expression, gaze, small head turns) on :{PORT}"
    essential = False

    def preflight(self) -> None:
        if not os.path.isfile(os.path.join(eng.NODE_DIR, "nodes.py")):
            raise PreflightError(
                f"ComfyUI-AdvancedLivePortrait is not at {eng.NODE_DIR}. face-edit ships in the "
                f":full tag only (WITH_TRAINER=1 builds); this looks like the lean :latest.")
        missing = eng.missing_models()
        if missing:
            raise PreflightError(
                f"LivePortrait models missing: {', '.join(missing)}. They are baked into the "
                f":full image, so a gap here means the image is wrong, not the deployment.")
        for mod in ("torch", "cv2", "ultralytics", "dill", "safetensors"):
            try:
                __import__(mod)
            except Exception as e:
                raise PreflightError(f"{mod} is not importable ({e}); the image is wrong.")

    def command(self) -> list[str]:
        # 0.0.0.0: wanly-api calls this across the network, like face-crop and the captioner.
        return ["python3", "-m", "uvicorn", "wanly_worker.services.face_edit.app:app",
                "--host", "0.0.0.0", "--port", str(self.port)]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def details(self) -> dict:
        return {"node_dir": eng.NODE_DIR, "models_dir": eng.MODELS_DIR}
