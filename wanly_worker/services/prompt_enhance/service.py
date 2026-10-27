"""prompt-enhance as a wanly-gpu-docker service (#225). See app.py for how it shares the 2070
with scene-caption, and model.py for what the model is and why it is llama.cpp, not ollama."""
from __future__ import annotations

import os
import shutil

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.prompt_enhance import app as pe_app
from wanly_worker.services.prompt_enhance import model


def _auto_fetch() -> bool:
    return os.environ.get("PROMPT_ENHANCE_AUTO_FETCH", "1") == "1"


class PromptEnhance(Service):
    name = "prompt-enhance"
    port = pe_app.PORT
    summary = (f"prompt-enhance: Sulphur-2 prompt enhancer (Qwen3.5-9B VL, Q4_K_M) on :{pe_app.PORT}, "
               f"loaded on demand" + (" (takes turns with scene-caption)" if pe_app.SCENE_URL else ""))
    # A box that has never had the files fetches 6.2 GB before it answers.
    ready_timeout_s = 7200.0

    def preflight(self) -> None:
        if not os.access(pe_app.LLAMA_SERVER, os.X_OK):
            raise PreflightError(
                f"{pe_app.LLAMA_SERVER} is missing or not executable. llama.cpp ships in the "
                f"image (Dockerfile, LLAMA_CPP_BUILD); a lean tag does not carry it.")
        need = model.missing()
        if not need:
            return
        if not _auto_fetch():
            raise PreflightError(
                f"the enhancer is not in {model.MODEL_DIR} and PROMPT_ENHANCE_AUTO_FETCH=0. "
                f"Fetch it first: download_models.sh --prompt-enhance")
        root = model.MODEL_DIR
        probe = root if os.path.isdir(root) else os.path.dirname(root.rstrip("/"))
        if not os.path.isdir(probe):
            raise PreflightError(
                f"neither {root} nor its parent exists. It belongs on a host mount (the scene "
                f"box's ollama store), or the 6 GB is re-fetched on every recreate.")
        free_gb = shutil.disk_usage(probe).free / 1024 ** 3
        want = sum(f.size for f in need) / 1024 ** 3 + model.HEADROOM_GB
        if free_gb < want:
            raise PreflightError(f"only {free_gb:.1f} GB free under {probe}, want {want:.0f} GB")

    def command(self) -> list[str]:
        # 0.0.0.0: wanly-api calls this across the network.
        return ["python3", "-m", "uvicorn", "wanly_worker.services.prompt_enhance.app:app",
                "--host", "0.0.0.0", "--port", str(self.port)]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def after_ready(self, client) -> None:
        """The files present (fetched if they must be). The model is NOT loaded here: it loads
        per request, so JoyCaption keeps the card until somebody asks."""
        if model.missing():
            await model.provision(client, log=lambda m: print(m, flush=True))
        print(f"[prompt-enhance] files present in {model.MODEL_DIR}", flush=True)

    def details(self) -> dict:
        return {"model": f"{model.HF_REPO}@{model.HF_REVISION[:8]}",
                "files_present": not model.missing(), "shares_card_with": pe_app.SCENE_URL or None}
