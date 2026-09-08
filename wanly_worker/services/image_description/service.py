"""image-description: the captioner behind <SCENE> (console#405) and dataset tagging.

NAMED FOR THE CAPABILITY, NOT THE TOOL (wanly-gpu-docker#83). Today it is JoyCaption beta-one
served by ollama; that is an internal detail of this package. The service name on the Workers
page, the SERVICES flag, the env vars and wanly-api's config all say image-description, so the
model can change without renaming anything outside this directory.

WHY OLLAMA'S OWN API IS THE WIRE CONTRACT

    wanly-api already speaks it, in app/joycaption.py:

        POST {joycaption_url}/api/generate
        {"model": ..., "prompt": ..., "images": [b64], "stream": false, "keep_alive": ...}

    The console never touches it at all -- it calls /captions/describe and /images/scene on
    wanly-api, which is the only caller. So the entire integration surface is one config
    value, `joycaption_url`, and if this container answers ollama on the same port there is
    nothing to change in wanly-api or wanly-console to adopt it.

    That is worth having. Captioning works today; moving the deployment and re-cutting the
    wire contract in the same change doubles what can break, for a decoupling nothing is
    asking for yet. A stable /caption facade that hides ollama is additive, and is better
    designed once a second service exists to share the contract with.

WHY THE VERSION IS PINNED, AND PINNED OLD

    0.20.2 is what 2070.zero has been serving joycaption:beta-one with. The current release
    is 0.33.3. Reproducing what works is the job here; a runtime upgrade is a separate change
    with its own risk (model format and GPU-scheduling behaviour both move between releases)
    and it should not ride along inside a containerisation whose whole appeal is that the
    captions come out the same.

WHY THE MODEL STORE IS A MOUNT

    joycaption:beta-one is 5.8 GB and 2070.zero already holds it in a 16 GB store. Pulling a
    second copy into the container is 5.8 GB of nothing, and pulling it onto the container
    overlay is wanly-gpu-docker#77 exactly. So the host store is mounted, and preflight
    checks it is writable with room to spare BEFORE anything starts.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.image_description import model as joycaption_model

#: Where the host's ollama store is mounted. The host path on 2070.zero is
#: /usr/share/ollama/.ollama -- 16 GB, owned by the `ollama` system user.
STORE = os.environ.get("OLLAMA_STORE", "/root/.ollama")

#: Must match wanly-api's `joycaption_model`. A mismatch is not an error anywhere -- ollama
#: would simply pull whatever wanly-api asks for on first use, silently, mid-caption.
MODEL = (os.environ.get("IMAGE_DESCRIPTION_MODEL") or os.environ.get("JOYCAPTION_MODEL")
         or "joycaption:beta-one")
PORT = int(os.environ.get("IMAGE_DESCRIPTION_PORT") or os.environ.get("JOYCAPTION_PORT") or "11434")

#: Enough for the model plus working room. The model is 5.8 GB and ollama writes blobs
#: straight into the store, so this is not double the model -- it is the model plus enough
#: that a second one, or a re-pull, does not wedge the box.
MIN_FREE_GB = 10.0


class ImageDescription(Service):
    name = "image-description"
    port = PORT
    summary = f"image-description: ollama serving {MODEL} on :{PORT}"

    def preflight(self) -> None:
        models = os.path.join(STORE, "models")
        if not os.path.isdir(STORE):
            raise PreflightError(
                f"the ollama store {STORE} does not exist. It is a bind mount from the host "
                f"(deploy/run-worker.sh: -v $OLLAMA_HOST_STORE:{STORE}); without it every "
                f"model would be re-pulled into the container and lost on recreate."
            )
        if not os.access(STORE, os.W_OK):
            raise PreflightError(
                f"the ollama store {STORE} is not writable. ollama writes its manifests and "
                f"blobs here, so a read-only mount fails at the first pull rather than now. "
                f"If the host directory looks writable, suspect SELinux: on Fedora or RHEL a "
                f"bind mount needs :z (e.g. -v /host/.ollama:{STORE}:z), and without it the "
                f"mode bits say yes while the kernel says no."
            )
        # Not fatal on its own -- ollama creates it -- but its absence usually means the
        # mount points somewhere unintended, which is worth saying while it is still cheap.
        if not os.path.isdir(models):
            print(f"[image-description] note: {models} does not exist yet; ollama will create it",
                  flush=True)
        # The bar depends on whether this machine has the model already. Demanding build-sized
        # space on 2070.zero, which has had the model for weeks, would refuse a box that works.
        need_build = bool(joycaption_model.missing(STORE))
        want = (joycaption_model.total_bytes() / (1024 ** 3)
                + joycaption_model.BUILD_HEADROOM_GB) if need_build else MIN_FREE_GB
        free_gb = shutil.disk_usage(STORE).free / (1024 ** 3)
        if free_gb < want:
            raise PreflightError(
                f"only {free_gb:.1f} GB free on {STORE}, want {want:.0f} GB "
                f"({'to build the model from scratch' if need_build else 'for working room'}). "
                f"{MODEL} is ~5.8 GB."
            )

    def command(self) -> list[str]:
        return ["ollama", "serve"]

    def env(self) -> dict[str, str]:
        return {
            # 0.0.0.0, not localhost: wanly-api reaches this across the network. The host
            # service it replaces sets exactly this.
            "OLLAMA_HOST": f"0.0.0.0:{self.port}",
            "OLLAMA_MODELS": os.path.join(STORE, "models"),
        }

    async def ready(self, client) -> bool:
        """Answering /api/tags, not merely spawned.

        ollama's process is up well before the server binds, so watching the process reports
        ready while every request connection-refuses.
        """
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def after_ready(self, client) -> None:
        """The model has to be present, and finding that out here is much cheaper than
        finding it out during somebody's caption.

        Without this the first /api/generate goes looking for the model inside a request that
        wanly-api gave 60 seconds (`joycaption_timeout_s`). It times out and the console says
        "captioner unreachable", which is true of the symptom and useless about the cause.

        A MISSING MODEL IS NOW BUILT, NOT REFUSED (#3). #1 refused it, on the evidence that
        `ollama pull joycaption:beta-one` fails -- which it does, because the tag lives in the
        library namespace and the library has no such model. The bytes underneath are a public
        Hugging Face repo, and ollama names a GGUF layer by the sha256 of the file, which is
        Hugging Face's LFS oid, so the two are directly comparable and identical. See
        joycaption_model.py.

        The mount stays the fast path. A machine that already has the model does nothing at
        all here -- 2070.zero behaves exactly as it did before this existed.
        """
        if MODEL in await self._tags(client):
            print(f"[image-description] {MODEL} present", flush=True)
            return

        if (os.environ.get("IMAGE_DESCRIPTION_AUTO_BUILD") or os.environ.get("JOYCAPTION_AUTO_BUILD") or "1") != "1":
            raise RuntimeError(
                f"{MODEL} is not in the store {STORE} and IMAGE_DESCRIPTION_AUTO_BUILD=0.")

        print(f"[image-description] {MODEL} is not in {STORE} — building it "
              f"({joycaption_model.total_bytes() / 1024 ** 3:.1f} GB)", flush=True)
        await joycaption_model.provision(STORE, client, log=lambda m: print(m, flush=True))

        # Ask ollama, rather than trusting that the files were written. Writing a
        # content-addressed store by hand is exact only while the layout is what we think it
        # is, and being served is the only evidence that it was.
        if MODEL not in await self._tags(client, refresh=True):
            raise RuntimeError(
                f"built {MODEL} into {STORE}, but ollama still does not list it. The blobs and "
                f"manifest are written and verified, so this is a store-layout mismatch — "
                f"check the ollama version against the pin in the Dockerfile.")
        print(f"[image-description] {MODEL} built and served", flush=True)

    async def _tags(self, client, refresh: bool = False) -> set[str]:
        """What ollama says it has. `refresh` retries briefly after a build.

        ollama rescans the manifest directory rather than watching it, so a model written
        underneath a running server can take a moment to appear. Without the retry the
        verification above would fail on a build that in fact worked -- the worst kind of
        check, because the response is to distrust a correct thing.
        """
        deadline = time.time() + (30 if refresh else 0)
        while True:
            r = await client.get(f"http://127.0.0.1:{self.port}/api/tags", timeout=30)
            names = {m.get("name") for m in (r.json().get("models") or [])}
            if names or time.time() >= deadline:
                if MODEL in names or time.time() >= deadline:
                    return names
            await asyncio.sleep(2)

    def details(self) -> dict:
        return {"model": MODEL, "store": STORE}
