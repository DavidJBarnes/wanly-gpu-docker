"""Rebuild `joycaption:beta-one` on a machine that has never had it (#3).

WHY THIS IS A REBUILD AND NOT A PULL

    `ollama pull joycaption:beta-one` fails, and #1 concluded from that the model was
    unfetchable. That was right about the symptom and wrong about the cause. The tag lives in
    ollama's *library* namespace, where it does not exist -- it was imported by hand on
    2070.zero. The bytes underneath are a public Hugging Face repo.

    ollama stores a GGUF layer under the sha256 OF THE FILE, which is exactly what Hugging Face
    uses as its LFS oid. So the two can be compared directly, and they are equal:

        4,920,735,936  e8ae55dd07e6…  Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf
          877,771,808  94002cb5c354…  llama-joycaption-beta-one-llava-mmproj-model-f16.gguf

    Not "the same model" -- the same bytes. Which means a fresh machine does not need us to host
    5.8 GB anywhere; it needs the two public files and the 1,070 bytes that are genuinely ours.

WHY THE STORE IS WRITTEN DIRECTLY RATHER THAN THROUGH `ollama create`

    The store is content-addressed, so writing it is not a trick -- a blob's name IS its digest,
    and a manifest is a list of digests. Writing it directly:

      * costs 5.8 GB instead of 11.6. `ollama create` copies the GGUFs into the store beside the
        downloads, and on a 20 GB pod disk that difference decides whether it works at all.
      * cannot renormalise the small layers. `ollama create` regenerates template/system/params
        from a Modelfile, and any difference in whitespace or key order changes their digests --
        which changes the model's identity, silently, for a captioner whose temperature and
        system prompt are why console#405's captions read the way they do.

    The cost is a coupling to ollama's on-disk layout, which is why the version is pinned and
    why step 4 asks ollama itself whether the model is there. Writing files is not evidence;
    being served is.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

#: The public repo the two GGUFs come from, pinned by commit rather than by branch: a moving
#: `main` would hand us different bytes with no warning. The digest check below is the real
#: guarantee, but pinning means a changed upstream fails before 5.8 GB is transferred.
HF_REPO = "concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf"
HF_REVISION = "acfe6bf78ae4e411cd5c7c8f4a71ba01f26a5b97"

#: Overridable so a mirror can be used without a code change, if that community repo ever
#: disappears. Recording the digests is what makes a mirror safe to trust: content is checked,
#: not provenance.
BASE_URL = os.environ.get(
    "JOYCAPTION_GGUF_BASE_URL", f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}")

#: Where the model is published in the store. `registry.ollama.ai/library` is the namespace a
#: bare `joycaption:beta-one` resolves to, which is why the tag has to live there and why
#: pulling it fails -- the real library has no such model.
MANIFEST_PATH = "manifests/registry.ollama.ai/library/joycaption/beta-one"

LAYERS_DIR = Path(__file__).parent / "layers"


@dataclass(frozen=True)
class Blob:
    kind: str
    media_type: str
    digest: str          # sha256, no prefix
    size: int
    filename: str = ""   # set for the two fetched from Hugging Face

    @property
    def remote(self) -> bool:
        return bool(self.filename)


#: The whole model, in order. `size` is asserted rather than trusted: a mirror that serves a
#: different length is caught before a single byte is hashed.
BLOBS: list[Blob] = [
    Blob("model", "application/vnd.ollama.image.model",
         "e8ae55dd07e61d541ab741d6ed63e7810192cea65d7ef8cda69b2a99fb06dc15", 4920735936,
         "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf"),
    Blob("projector", "application/vnd.ollama.image.projector",
         "94002cb5c354c7c9e538e64f37d593db9eceeca2e94573bae6cd3b2bd8bb1952", 877771808,
         "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"),
    # Ours. Tiny, and the part with no upstream: the llama-3 chat template, the uncensored
    # system prompt, and temperature 0.4 / top_p 0.9 / num_ctx 4096. Losing these loses the
    # captioner's behaviour even with both GGUFs in hand.
    Blob("template", "application/vnd.ollama.image.template",
         "8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f", 254),
    Blob("system", "application/vnd.ollama.image.system",
         "8c5425a7658486f7fa6c42cfe9845d9e63a0cdfe8a4fcef91d7ab7b12f307cc1", 109),
    Blob("params", "application/vnd.ollama.image.params",
         "88e0a2d50d7813ca587bd3d206fc6020d0fc48a43886d34610bbba5bb488be1c", 139),
]

CONFIG = Blob("config", "application/vnd.docker.container.image.v1+json",
              "1abb28e5e5a9fe314cf5da8ae39a9d310e67c6b0e4287736200f9c6e9c8b93fa", 568)

#: What the download needs on top of the finished model: the largest single file, while it is
#: still a .part beside everything already written.
BUILD_HEADROOM_GB = 6.0


def total_bytes() -> int:
    return sum(b.size for b in BLOBS) + CONFIG.size


def _blob_path(store: str, digest: str) -> Path:
    return Path(store) / "models" / "blobs" / f"sha256-{digest}"


def have(store: str, blob: Blob) -> bool:
    """A blob counts as present only at the right length.

    Size, not existence. A truncated blob satisfies `exists()`, passes every check that is not
    this one, and fails at load -- inside a caption, on another machine.
    """
    p = _blob_path(store, blob.digest)
    return p.is_file() and p.stat().st_size == blob.size


def missing(store: str) -> list[Blob]:
    return [b for b in [*BLOBS, CONFIG] if not have(store, b)]


def local_source(blob: Blob) -> Path:
    return LAYERS_DIR / blob.kind


def manifest() -> dict:
    """The manifest ollama reads, built from the table above.

    `from` is deliberately absent. The original carries absolute paths into 2070.zero's store,
    which mean nothing anywhere else; ollama resolves layers by digest.
    """
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": CONFIG.media_type,
                   "digest": f"sha256:{CONFIG.digest}", "size": CONFIG.size},
        "layers": [{"mediaType": b.media_type, "digest": f"sha256:{b.digest}", "size": b.size}
                   for b in BLOBS],
    }


class ProvisionError(RuntimeError):
    pass


async def provision(store: str, client, log=print) -> None:
    """Put the model in `store` so ollama can serve it. Assumes it is not already there."""
    blobs_dir = _blob_path(store, "x").parent
    blobs_dir.mkdir(parents=True, exist_ok=True)

    need = missing(store)
    need_bytes = sum(b.size for b in need)
    free_gb = shutil.disk_usage(store).free / (1024 ** 3)
    want_gb = need_bytes / (1024 ** 3) + BUILD_HEADROOM_GB
    if free_gb < want_gb:
        raise ProvisionError(
            f"need ~{want_gb:.0f} GB free under {store} to build {len(need)} missing layer(s), "
            f"but only {free_gb:.1f} GB is available")

    log(f"[image-description] building the model: {len(need)} layer(s), "
        f"{need_bytes / 1024 ** 3:.1f} GB")

    for blob in need:
        if blob.remote:
            await _fetch(blob, store, client, log)
        else:
            _install_local(blob, store, log)

    # The manifest last, so an interrupted build leaves no half-model that ollama would list
    # and then fail to load. Blobs on their own are invisible to it.
    mpath = Path(store) / "models" / MANIFEST_PATH
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest()))
    log(f"[image-description] wrote manifest {MANIFEST_PATH}")


def _install_local(blob: Blob, store: str, log) -> None:
    src = local_source(blob)
    if not src.is_file():
        raise ProvisionError(f"{blob.kind} layer is missing from the image at {src}")
    data = src.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != blob.digest:
        raise ProvisionError(
            f"the {blob.kind} layer shipped in this image hashes to {got[:16]}…, "
            f"expected {blob.digest[:16]}… — it has been modified")
    _blob_path(store, blob.digest).write_bytes(data)
    log(f"[image-description]   {blob.kind}: {len(data)} bytes (from the image)")


async def _fetch(blob: Blob, store: str, client, log) -> None:
    """Stream one GGUF, hashing as it goes, into a .part that is renamed only once it verifies.

    Hashed DURING the download rather than after: re-reading 4.6 GB to check it doubles the
    slowest part of a cold start for no additional certainty.
    """
    url = f"{BASE_URL}/{blob.filename}"
    final = _blob_path(store, blob.digest)
    # Explicit rather than with_suffix(): the name is "sha256-<hex>", and with_suffix would
    # be replacing a suffix that only happens not to exist.
    part = final.parent / (final.name + ".part")
    log(f"[image-description]   {blob.kind}: fetching {blob.filename} "
        f"({blob.size / 1024 ** 3:.1f} GB)")

    h = hashlib.sha256()
    done = 0
    t0 = last = time.time()
    async with client.stream("GET", url, follow_redirects=True, timeout=None) as resp:
        if resp.status_code != 200:
            raise ProvisionError(f"{url} -> HTTP {resp.status_code}")
        declared = int(resp.headers.get("content-length") or 0)
        if declared and declared != blob.size:
            # Caught before hashing 4.6 GB of the wrong file.
            raise ProvisionError(
                f"{blob.filename} is {declared} bytes upstream, expected {blob.size}. "
                f"The source has changed; do not trust it.")
        with part.open("wb") as fh:
            async for chunk in resp.aiter_bytes(8 * 1024 * 1024):
                fh.write(chunk)
                h.update(chunk)
                done += len(chunk)
                # Progress on a clock, not per chunk. 5.8 GB of silence is indistinguishable
                # from a hang, and that ambiguity has cost this project whole evenings.
                if time.time() - last >= 15:
                    pct = 100 * done / blob.size
                    rate = done / (time.time() - t0) / 1024 ** 2
                    log(f"[image-description]     {blob.kind}: {pct:.0f}% "
                        f"({done / 1024 ** 3:.1f} GB, {rate:.0f} MB/s)")
                    last = time.time()

    got = h.hexdigest()
    if got != blob.digest:
        part.unlink(missing_ok=True)
        raise ProvisionError(
            f"{blob.filename} hashed to {got[:16]}…, expected {blob.digest[:16]}…. "
            f"Refusing it — a wrong or truncated model fails inside a caption, not here.")
    part.rename(final)
    log(f"[image-description]   {blob.kind}: verified {got[:16]}… "
        f"in {time.time() - t0:.0f}s")
