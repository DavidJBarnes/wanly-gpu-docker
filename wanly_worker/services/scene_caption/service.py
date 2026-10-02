"""scene-caption as wanly-gpu-docker services (wanly-console#572 phase 1).

WHY IT EXISTS

    One captioner used to do both halves of a video prompt -- the static SCENE and the MOTION
    paragraph -- from one URL with one model setting. On a 3090 that meant two vision models
    loaded and unloaded per image (30 images, 60 swaps, across the ZeroTier link), which is
    the likeliest cause of 3090a hanging on 2026-10-02. The halves are now separate jobs:

        scene   JoyCaption Beta One, ~6 GB, ALWAYS RESIDENT -- this service.
        motion  Qwen3-VL 32B on a 3090 in motion mode (image-description).

    wanly-api's scene_caption_url points here.

TWO PROCESSES, ONE NAME, like image-edit:

    scene-caption-ollama   ollama serve on LOOPBACK (SCENE_CAPTION_OLLAMA_PORT, 11437), its
                           own instance -- not image-description's -- so the two captioners'
                           keep_alive, model and GPU scheduling never interact. Same pinned
                           ollama binary, same mounted store (OLLAMA_STORE).
    scene-caption          the front (app.py) on SCENE_CAPTION_PORT (11436): ollama's own
                           /api/generate wire, one model, keep_alive -1, plus /yield and
                           /resume for a shared card and /health.

THE MODEL is the same joycaption:beta-one the 3090's ollama uses. A store that has it (3090b,
the 3090) uses it as it is -- checked against the store's own manifest, never rewritten
(store.py). A store that does not gets it rebuilt from the two public GGUFs, pinned by
revision and digest, plus the small layers shipped in this image (image_description/model.py):
~5.8 GB, so a fresh box can do it. `download_models.sh --scene-caption` does the same from the
command line, rate-limited, for a box on a shared link.

NOTHING IN THE STORE IS PRUNED. ollama deletes blobs no manifest references when it starts,
unless told not to. These stores hold other people's models and hand imports (3090b's carries
dolphin-mistral, gemma4 and a qwen2.5vl), so OLLAMA_NOPRUNE is set.
"""
from __future__ import annotations

import os
import shutil

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.image_description import model as jc_model
from wanly_worker.services.scene_caption import app as scene_app
from wanly_worker.services.scene_caption import store as jc_store

PORT = int(os.environ.get("SCENE_CAPTION_PORT", "11436"))
STORE = os.environ.get("OLLAMA_STORE", "/root/.ollama")
#: Room for ollama to work in, on top of the model; and, when the model is absent, the build.
MIN_FREE_GB = 2.0


def _auto_build() -> bool:
    return os.environ.get("SCENE_CAPTION_AUTO_BUILD", "1") == "1"


class SceneOllama(Service):
    name = "scene-caption-ollama"
    port = scene_app.OLLAMA_PORT
    summary = f"ollama for scene-caption ({scene_app.MODEL}) on :{scene_app.OLLAMA_PORT} (loopback)"
    ready_timeout_s = 120.0

    def preflight(self) -> None:
        if not os.path.isdir(STORE):
            raise PreflightError(
                f"the ollama store {STORE} does not exist. It is a bind mount from the host "
                f"(deploy/run-worker.sh: -v $OLLAMA_HOST_STORE:{STORE}); without it the model "
                f"would be rebuilt into the container and lost on recreate.")
        ok, how = jc_store.present(STORE)
        free_gb = shutil.disk_usage(STORE).free / (1024 ** 3)
        want = MIN_FREE_GB
        if not ok:
            if not _auto_build():
                raise PreflightError(
                    f"{scene_app.MODEL} is not in {STORE} ({how}) and SCENE_CAPTION_AUTO_BUILD=0. "
                    f"Fetch it first: download_models.sh --scene-caption")
            want = (sum(b.size for b in jc_store.to_fetch(STORE)) / (1024 ** 3)
                    + jc_model.BUILD_HEADROOM_GB)
            if not os.access(STORE, os.W_OK):
                raise PreflightError(
                    f"{scene_app.MODEL} must be built into {STORE}, which is not writable "
                    f"(on SELinux hosts a bind mount needs :z).")
        if free_gb < want:
            raise PreflightError(f"only {free_gb:.1f} GB free on {STORE}, want {want:.0f} GB")

    def command(self) -> list[str]:
        return ["ollama", "serve"]

    def env(self) -> dict[str, str]:
        return {
            # Loopback: only the front (app.py) talks to this ollama.
            "OLLAMA_HOST": f"127.0.0.1:{self.port}",
            "OLLAMA_MODELS": os.path.join(STORE, "models"),
            # Always resident -- the reason this service exists.
            "OLLAMA_KEEP_ALIVE": "-1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            # See the module docstring: the store holds other models; never delete blobs.
            "OLLAMA_NOPRUNE": "1",
        }

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def after_ready(self, client) -> None:
        """The model present (built if it must be), then loaded, so the first caption does
        not pay the load."""
        ok, how = jc_store.present(STORE)
        if ok:
            print(f"[scene-caption] {scene_app.MODEL} present in {STORE}: {how}", flush=True)
        else:
            print(f"[scene-caption] {scene_app.MODEL} not in {STORE} ({how}) -- building it "
                  f"({jc_model.total_bytes() / 1024 ** 3:.1f} GB)", flush=True)
            await jc_model.provision(STORE, client, log=lambda m: print(m, flush=True))
        r = await client.get(f"http://127.0.0.1:{self.port}/api/tags", timeout=30)
        names = {m.get("name") for m in (r.json().get("models") or [])}
        if scene_app.MODEL not in names:
            raise RuntimeError(
                f"{scene_app.MODEL} is in {STORE} but ollama does not list it -- check the "
                f"ollama version against the pin in the Dockerfile.")
        r = await client.post(f"http://127.0.0.1:{self.port}/api/generate",
                              json={"model": scene_app.MODEL, "prompt": "", "keep_alive": -1},
                              timeout=300)
        print(f"[scene-caption] {scene_app.MODEL} "
              f"{'loaded and resident' if r.status_code == 200 else f'not preloaded ({r.status_code})'}",
              flush=True)

    def details(self) -> dict:
        return {"model": scene_app.MODEL, "store": STORE}


class SceneCaptionApi(Service):
    name = "scene-caption"
    port = PORT
    summary = (f"scene-caption: {scene_app.MODEL}, always resident, on :{PORT}"
               + (" (shares its card: yields to image-edit)" if scene_app.SHARED else ""))

    def command(self) -> list[str]:
        # 0.0.0.0: wanly-api calls this across the network.
        return ["python3", "-m", "uvicorn", "wanly_worker.services.scene_caption.app:app",
                "--host", "0.0.0.0", "--port", str(self.port)]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def details(self) -> dict:
        return {"model": scene_app.MODEL, "shared": scene_app.SHARED}


def scene_caption_group() -> list[Service]:
    """ollama first: the front is useless until it answers."""
    return [SceneOllama(), SceneCaptionApi()]
