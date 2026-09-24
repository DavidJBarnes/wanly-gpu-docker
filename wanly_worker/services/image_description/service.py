"""image-description: the captioner behind <SCENE> (console#405) and dataset tagging.

NAMED FOR THE CAPABILITY, NOT THE TOOL (wanly-gpu-docker#83). Today it is an ollama serving a
vision model; that is an internal detail of this package. The service name on the Workers page,
the SERVICES flag, the env vars and wanly-api's config all say image-description, so the
model can change without renaming anything outside this directory.

TWO MODELS, ONE ACTIVE (wanly-gpu-docker#129, from wanly-services #326/#36):

    IMAGE_DESCRIPTION_MODEL      the model wanly-api asks for. Since wanly-api#326 that is a
                                 Qwen2.5/3-VL class model, because one model has to do BOTH
                                 halves of a video prompt: the static scene AND the motion
                                 paragraph. Any ordinary library tag — `ollama pull` reaches it.
    joycaption:beta-one          the uncensored static captioner. Kept installed per David's
                                 "nothing deleted" rule; the only model this package can BUILD
                                 (model.py), and the one an env override can point wanly-api
                                 back at via the JOYCAPTION_* aliases.

    after_ready guarantees both are present: the active model by PULL when it is not the kept
    one, the kept one by BUILD (its tag lives in the library namespace and `ollama pull`
    cannot see it — see model.py).

WHY OLLAMA'S OWN API IS THE WIRE CONTRACT

    wanly-api already speaks it, in app/joycaption.py:

        POST {image_description_url}/api/generate
        {"model": ..., "prompt": ..., "images": [b64], "stream": false, "keep_alive": ...}

    The console never touches it at all -- it calls /captions/describe and /images/scene on
    wanly-api, which is the only caller. So the entire integration surface is one config
    value, `image_description_url`, and if this container answers ollama on the same port there
    is nothing to change in wanly-api or wanly-console to adopt it.

    That is worth having. Captioning works today; moving the deployment and re-cutting the
    wire contract in the same change doubles what can break, for a decoupling nothing is
    asking for yet. A stable /caption facade that hides ollama is additive, and is better
    designed once a second service exists to share the contract with.

WHY THE VERSION IS PINNED, AND PINNED OLD

    0.20.2 is what 2070.zero has been serving joycaption:beta-one with. The current release
    is 0.33.3. Reproducing what works is the job here; a runtime upgrade is a separate change
    with its own risk (model format and GPU-scheduling behaviour both move between releases)
    and it should not ride along inside a containerisation whose whole appeal is that the
    captions come out the same. NOTE: qwen3-vl (wanly-gpu-docker#128) is a newer architecture
    family than this pin; if 0.20.2 cannot load it, the pin bump is its own change, judged on
    its own, exactly as this paragraph demands.

WHY THE MODEL STORE IS A MOUNT

    joycaption:beta-one is 5.8 GB and a Qwen-class captioner is 6-21 GB, and the hosts already
    hold what they hold in stores mounted here. Pulling copies into the container is many GB of
    nothing, and pulling them onto the container overlay is wanly-gpu-docker#77 exactly. So the
    host store is mounted, and preflight checks it is writable with room to spare BEFORE
    anything starts.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time

from wanly_worker.service import PreflightError, Service
from wanly_worker.services.image_description import model as joycaption_model

#: Where the host's ollama store is mounted. The host path on 2070.zero is
#: /usr/share/ollama/.ollama -- 16 GB, owned by the `ollama` system user; on the 3090 it is
#: /home/david/.ollama (wanly-gpu-docker's run-worker mount).
STORE = os.environ.get("OLLAMA_STORE", "/root/.ollama")

#: The kept captioner. The only model this package can build (its tag is unpullable — see
#: model.py), and the one wanly-api's old JOYCAPTION_* aliases still reach.
KEPT_MODEL = "joycaption:beta-one"

#: The model wanly-api asks for. Must match wanly-api's `image_description_model` — a mismatch
#: is not an error anywhere: ollama would simply pull whatever wanly-api asks for on first use,
#: silently, mid-caption. Deployments override by env: run-worker.sh passes joycaption:beta-one
#: as the default (it fits every GPU we own; a Qwen-class tag needs a 24 GB card), so a box
#: opts into the Qwen captioner explicitly — wanly-api#345 will point the api at whatever
#: wanly-gpu-docker#128 proves out.
MODEL = (os.environ.get("IMAGE_DESCRIPTION_MODEL") or os.environ.get("JOYCAPTION_MODEL")
         or "qwen2.5vl:7b-q4_K_M")
PORT = int(os.environ.get("IMAGE_DESCRIPTION_PORT") or os.environ.get("JOYCAPTION_PORT") or "11434")

#: Enough working room on top of whatever is already in the store. It was 10 GB with one
#: 5.8 GB model and stays 10 here: this bar is about ollama having room to write blobs during
#: a re-pull, not about fitting a model — fitting the *active* model is the pull bar below.
MIN_FREE_GB = 10.0

#: The bar when the active model is a pull tag and is NOT yet in the store. qwen3-vl:32b-class
#: is ~21 GB (wanly-gpu-docker#128) and ollama writes blobs straight into the mounted store
#: while serving beside a renderer on the same filesystem — a full disk there is not a failed
#: pull, it is a dead box. Overridable for a box pulling something smaller or bigger.
PULL_MIN_FREE_GB = float(os.environ.get("IMAGE_DESCRIPTION_PULL_MIN_FREE_GB", "22"))

#: Bound on `ollama pull`, for a wedged connection — not a budget. Measured: qwen2.5vl 7b-q4
#: (6.0 GB) took 12m41s on 2070.zero (wanly-services#326); qwen3-vl 32b-q4 (20.9 GB) took
#: 37m at ~9.4 MB/s on 3090.zero (2026-09-24, wanly-gpu-docker#128) — ollama resumes a killed
#: pull, so the bound only needs to outlast a bad hour of a good connection, not a slow one.
#: Default: ~2x the measured 21 GB pull. A mirror or slower link raises it by env.
PULL_TIMEOUT_S = float(os.environ.get("IMAGE_DESCRIPTION_PULL_TIMEOUT_S", "4800"))


class ImageDescription(Service):
    name = "image-description"
    port = PORT
    summary = (f"image-description: ollama serving {MODEL}"
               + (f" (+{KEPT_MODEL})" if MODEL != KEPT_MODEL else "")
               + f" on :{PORT}")

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
        # The bar depends on what this machine is about to write, not on a fixed number:
        # working room always; the pull's size when the ACTIVE model is a pull tag and is
        # not in the store; the build's size when the KEPT model is absent. A box that has
        # everything (2070.zero, the 3090) is asked only for working room — demanding
        # build- or pull-sized space there would refuse a box that works.
        free_gb = shutil.disk_usage(STORE).free / (1024 ** 3)
        bars = [(MIN_FREE_GB, "for working room")]
        if MODEL != KEPT_MODEL and not self._in_store(MODEL):
            bars.append((PULL_MIN_FREE_GB,
                         f"to pull {MODEL} (Qwen-class captioners are 6-21 GB)"))
        if joycaption_model.missing(STORE):
            bars.append((joycaption_model.total_bytes() / (1024 ** 3)
                         + joycaption_model.BUILD_HEADROOM_GB,
                         "to build the model from scratch"))
        want, why = max(bars)
        if free_gb < want:
            raise PreflightError(
                f"only {free_gb:.1f} GB free on {STORE}, want {want:.0f} GB ({why}).")

    def command(self) -> list[str]:
        return ["ollama", "serve"]

    def env(self) -> dict[str, str]:
        return {
            # 0.0.0.0, not localhost: wanly-api reaches this across the network. The host
            # service it replaces sets exactly this.
            "OLLAMA_HOST": f"0.0.0.0:{self.port}",
            "OLLAMA_MODELS": os.path.join(STORE, "models"),
            # NO OLLAMA_LLM_LIBRARY HERE, ON PURPOSE (wanly-services#36, reverting #34).
            #
            # #34 pinned this serve to CPU because the caption model of the day, qwen2.5vl,
            # does not fit the 2070's 8 GB card — and does not fit at ANY size: 7b spills to
            # 16/33 layers at the VRAM's pleasure, and 3b fails harder, one 7.35 GiB
            # cudaMalloc for the vision tower that OOMs an empty card before a single LLM
            # layer loads ("offloaded 0/37", 2026-09-21). Do not re-litigate whether a smaller
            # qwen fits; it is the vision tower, not the LLM. CPU-qwen measured 15-95 s a
            # caption.
            #
            # Production then went back to JoyCaption, which DOES fit: peak 3960 MiB on GPU.
            # The #34 pin outlived #34's premise and made JoyCaption itself run on CPU — a
            # captioner pinned CPU by a decision about a model it no longer uses. The lesson
            # is structural: a CPU pin belongs with the model decision, never here permanently.
            # On a 24 GB card (the 3090) a Qwen-class captioner runs on GPU outright.
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
        """Every model wanly-api may ask for has to be present, and finding that out here is
        much cheaper than finding it out during somebody's caption.

        Without this the first /api/generate goes looking for the model inside a request that
        wanly-api gave its own timeout for. It times out and the console says "captioner
        unreachable", which is true of the symptom and useless about the cause.

        THE ACTIVE MODEL IS PULLED IF IT IS NOT THE KEPT ONE. A Qwen-class tag is an ordinary
        library tag — `ollama pull` just works (wanly-services#326 measured 6.0 GB in 12m41s
        on 2070.zero from an empty store; #128 measured 20.9 GB in ~37m on 3090.zero). The
        pull runs the container's own ollama binary against the running server, which is the
        only writer that cannot corrupt the store's layout.

        A MISSING KEPT MODEL IS BUILT, NOT REFUSED (wanly-services#3). #1 refused it, on the
        evidence that `ollama pull joycaption:beta-one` fails -- which it does, because the tag
        lives in the library namespace and the library has no such model. The bytes underneath
        are a public Hugging Face repo, and ollama names a GGUF layer by the sha256 of the
        file, which is Hugging Face's LFS oid, so the two are directly comparable and
        identical. See model.py.

        The mount stays the fast path. A machine that already has the models does nothing at
        all here -- 2070.zero and the 3090 behave exactly as they did before this existed.
        """
        if MODEL != KEPT_MODEL:
            if MODEL in await self._tags(client, wanted=(MODEL,)):
                print(f"[image-description] {MODEL} present", flush=True)
            else:
                await self._pull_model(client, MODEL)

        if KEPT_MODEL in await self._tags(client, wanted=(KEPT_MODEL,)):
            print(f"[image-description] {KEPT_MODEL} present", flush=True)
            return

        if not self._auto_build():
            raise RuntimeError(
                f"{KEPT_MODEL} is not in the store {STORE} and IMAGE_DESCRIPTION_AUTO_BUILD=0.")

        print(f"[image-description] {KEPT_MODEL} is not in {STORE} — building it "
              f"({joycaption_model.total_bytes() / 1024 ** 3:.1f} GB)", flush=True)
        await joycaption_model.provision(STORE, client, log=lambda m: print(m, flush=True))

        # Ask ollama, rather than trusting that the files were written. Writing a
        # content-addressed store by hand is exact only while the layout is what we think it
        # is, and being served is the only evidence that it was.
        if KEPT_MODEL not in await self._tags(client, wanted=(KEPT_MODEL,), refresh=True):
            raise RuntimeError(
                f"built {KEPT_MODEL} into {STORE}, but ollama still does not list it. The blobs "
                f"and manifest are written and verified, so this is a store-layout mismatch — "
                f"check the ollama version against the pin in the Dockerfile.")
        print(f"[image-description] {KEPT_MODEL} built and served", flush=True)

    def _auto_build(self) -> bool:
        return (os.environ.get("IMAGE_DESCRIPTION_AUTO_BUILD")
                or os.environ.get("JOYCAPTION_AUTO_BUILD") or "1") == "1"

    def _in_store(self, model: str) -> bool:
        """Whether a name:tag manifest already exists on disk, for the preflight space bar.
        The authoritative check is ollama itself in after_ready; this only decides how much
        disk to demand before starting anything."""
        name, _, tag = model.partition(":")
        manifest = os.path.join(STORE, "models", "manifests",
                                "registry.ollama.ai", "library", name, tag or "latest")
        return os.path.isfile(manifest)

    async def _pull_model(self, client, model: str) -> None:
        """Provision the active captioner by pulling it, then prove ollama serves it.

        Failure is a boot failure, not a shrug: a box without the model wanly-api asks for
        is a box that 503s on every caption, and that is loud here rather than slow later.
        """
        if not self._auto_build():
            raise RuntimeError(
                f"{model} is not in the store {STORE} and IMAGE_DESCRIPTION_AUTO_BUILD=0.")

        print(f"[image-description] {model} is not in {STORE} — pulling it "
              f"(Qwen-class captioners are 6-21 GB)", flush=True)
        # OLLAMA_HOST is passed explicitly, not inherited: env() reaches the `serve` child
        # only, so the container's own environ has no OLLAMA_HOST and a bare `ollama pull`
        # would talk to 127.0.0.1:11434 — wrong port whenever IMAGE_DESCRIPTION_PORT moves.
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", model,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{self.port}"})
        # The pull runs against OLLAMA_HOST (this server, the mount), so the blobs land in
        # the mounted store and survive the next recreate — never on the overlay.
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=PULL_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"pulling {model} took over {PULL_TIMEOUT_S:.0f}s; killed "
                               f"(IMAGE_DESCRIPTION_PULL_TIMEOUT_S)")
        if proc.returncode != 0:
            tail = (out or b"").decode(errors="replace").strip().splitlines()[-3:]
            raise RuntimeError(f"ollama pull {model} failed ({proc.returncode}): "
                               f"{' | '.join(tail)}")

        # Same proof-over-files rule as the build path: being listed is the evidence.
        if model not in await self._tags(client, wanted=(model,), refresh=True):
            raise RuntimeError(
                f"pulled {model} but ollama does not list it — check the ollama version "
                f"against the pin in the Dockerfile.")
        print(f"[image-description] {model} pulled and served", flush=True)

    async def _tags(self, client, refresh: bool = False, wanted: tuple = ()) -> set[str]:
        """What ollama says it has. `refresh` retries briefly after a build or pull.

        `wanted` names what the caller is waiting for; without it the retry loop would exit
        on the first model it happens to find, which in a two-model world can mean never
        seeing the one being waited on.

        ollama rescans the manifest directory rather than watching it, so a model written
        underneath a running server can take a moment to appear. Without the retry the
        verification would fail on a build that in fact worked -- the worst kind of check,
        because the response is to distrust a correct thing.
        """
        want = set(wanted) or {MODEL}
        deadline = time.time() + (30 if refresh else 0)
        while True:
            r = await client.get(f"http://127.0.0.1:{self.port}/api/tags", timeout=30)
            names = {m.get("name") for m in (r.json().get("models") or [])}
            if want <= names or time.time() >= deadline:
                return names
            await asyncio.sleep(2)

    def details(self) -> dict:
        return {"model": MODEL, "kept_model": KEPT_MODEL, "store": STORE}
