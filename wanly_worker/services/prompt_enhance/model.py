"""The Sulphur-2 prompt enhancer's two files: pinned, digest-checked, fetched if absent (#225).

WHAT IT IS

    SulphurAI ship a prompt enhancer beside Sulphur-2-base, the LTX-2.3 fine-tune 10Eros is
    merged from: Qwen3.5-9B VL, trained to turn an image plus a short description into the long
    scene script LTX needs (LTX does almost nothing a prompt does not spell out). We run the
    UNCENSORED variant at Q4_K_M -- 5.2 GiB of weights plus a 0.9 GiB vision projector -- because
    the official Q8 (9.5 GB) does not fit the 2070 and the censored one tones explicit scenes
    down. The Q4 is a community quant of SulphurAI's own uncensored safetensors (Apache-2.0).

WHY NOT OLLAMA, which serves every other captioner here

    The vision projector is a separate mmproj file, and ollama's GGUF import does not take one
    for this architecture (qwen35). llama.cpp's llama-server does, as `--mmproj`, which is how
    the model card runs it. So the files live as plain GGUFs in their own directory, not in the
    ollama store's content-addressed layout.

Pinned by commit and digest for the reason image_description/model.py gives: a moving `main`
hands over different bytes with no warning, and a wrong model fails inside a request, not here.

CLI, for download_models.sh --prompt-enhance (stdlib only, so it also runs on a host):

    python3 -m wanly_worker.services.prompt_enhance.model check DIR
        exit 0 if both files are present at their sizes; else exit 1 and print one line per
        file to fetch: "<sha256> <size> <url> <path>"
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HF_REPO = "BOBBYBONKERSBFD/Sulphur-2-Prompt-Enhancer-Uncensored-GGUF"
HF_REVISION = "25674afcb7a5e7eb93f1f53503b58b25558c7f54"
BASE_URL = os.environ.get(
    "PROMPT_ENHANCE_GGUF_BASE_URL", f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}")

#: Where the two files live. Under the scene box's ollama store mount by default, because that
#: is the one persistent host directory the 2070's container already has -- a sibling of
#: `models/`, which ollama owns; ollama never looks outside it.
MODEL_DIR = os.environ.get("PROMPT_ENHANCE_MODEL_DIR", "/root/.ollama/prompt-enhance")
#: Room to work in on top of the files.
HEADROOM_GB = 1.0


@dataclass(frozen=True)
class File:
    kind: str
    filename: str      # upstream
    local: str         # in MODEL_DIR
    size: int
    digest: str


WEIGHTS = File("weights", "prompt_enhancer_uncensored-Q4_K_M.gguf", "model-Q4_K_M.gguf",
               5_629_108_608, "bf89cbbd1380fe356d89ec1225ac48ed80df4ef5d8e14dc103acf7baf074d3b2")
MMPROJ = File("mmproj", "mmproj-prompt_enhancer_uncensored.gguf", "mmproj.gguf",
              921_704_384, "6af8cbc85b7e25459e687da3674c3e5a46c4eaaf6b8fabb3f4447ebf6373a888")
FILES = (WEIGHTS, MMPROJ)


class ProvisionError(RuntimeError):
    pass


def path(f: File, model_dir: str | None = None) -> Path:
    return Path(model_dir or MODEL_DIR) / f.local


def missing(model_dir: str | None = None) -> list[File]:
    """Files absent or the wrong size. Size, not digest: re-hashing 6 GB on every boot costs a
    minute for nothing -- the digest is checked once, as the file arrives."""
    out = []
    for f in FILES:
        p = path(f, model_dir)
        if not p.is_file() or p.stat().st_size != f.size:
            out.append(f)
    return out


def total_bytes() -> int:
    return sum(f.size for f in FILES)


async def provision(client, model_dir: str | None = None, log=print) -> None:
    """Fetch whatever is missing. Each file lands as a .part renamed only once it verifies."""
    from wanly_worker.services.image_description.model import limit_rate

    root = Path(model_dir or MODEL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    need = missing(str(root))
    want_gb = sum(f.size for f in need) / 1024 ** 3 + HEADROOM_GB
    free_gb = shutil.disk_usage(root).free / 1024 ** 3
    if free_gb < want_gb:
        raise ProvisionError(f"need ~{want_gb:.0f} GB free under {root}, only {free_gb:.1f} GB")
    for f in need:
        url = f"{BASE_URL}/{f.filename}"
        final = root / f.local
        part = root / (f.local + ".part")
        log(f"[prompt-enhance] fetching {f.filename} ({f.size / 1024 ** 3:.1f} GB)")
        h = hashlib.sha256()
        done = 0
        t0 = last = time.time()
        rate = limit_rate()
        async with client.stream("GET", url, follow_redirects=True, timeout=None) as resp:
            if resp.status_code != 200:
                raise ProvisionError(f"{url} -> HTTP {resp.status_code}")
            declared = int(resp.headers.get("content-length") or 0)
            if declared and declared != f.size:
                raise ProvisionError(f"{f.filename} is {declared} bytes upstream, expected "
                                     f"{f.size}. The source has changed; do not trust it.")
            with part.open("wb") as fh:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    fh.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if rate:
                        ahead = done / rate - (time.time() - t0)
                        if ahead > 0:
                            await asyncio.sleep(ahead)
                    if time.time() - last >= 15:
                        mbs = done / (time.time() - t0) / 1024 ** 2
                        log(f"[prompt-enhance]   {f.kind}: {100 * done / f.size:.0f}% "
                            f"({mbs:.1f} MB/s)")
                        last = time.time()
        got = h.hexdigest()
        if got != f.digest:
            part.unlink(missing_ok=True)
            raise ProvisionError(f"{f.filename} hashed to {got[:16]}…, expected "
                                 f"{f.digest[:16]}… -- refused")
        part.rename(final)
        log(f"[prompt-enhance]   {f.kind}: verified in {time.time() - t0:.0f}s")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "check":
        print(__doc__, file=sys.stderr)
        return 2
    need = missing(argv[1])
    if not need:
        print("prompt-enhance: both files present", file=sys.stderr)
        return 0
    for f in need:
        print(f"{f.digest} {f.size} {BASE_URL}/{f.filename} {path(f, argv[1])}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
