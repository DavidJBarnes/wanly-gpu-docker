"""One card per container, chosen by UUID (wanly-gpu-docker#163).

After the hardware move each box holds two cards: 3090a the 3090 plus the 2070 for scene
captions, 3090b the 3090 plus the 3080 for A1111. With `--device nvidia.com/gpu=all` every
container sees both, and ollama, ComfyUI and torch pick one BY INDEX -- an order that can change
between boots. A 3090 worker that lands on the 8 GB card fails every render with an OOM that
reads like a model problem.

run-worker.sh passes `--device nvidia.com/gpu=<GPU_UUID>` through CDI, so the container sees one
card. This is the check that it actually did: when GPU_UUID is set, the boot refuses unless the
container sees EXACTLY that one card. Unset keeps the old behaviour (every card), with a loud
warning when that is more than one.

Read through nvidia-smi, like supervisor.gpu_snapshot, so nothing here depends on a python CUDA
binding.
"""
from __future__ import annotations

import os
import subprocess


class GpuPinError(RuntimeError):
    """GPU_UUID is set and the container does not see exactly that card."""


def expected_uuid() -> str | None:
    """GPU_UUID, or GPU_DEVICE (the epic's spelling). Empty is unset."""
    for var in ("GPU_UUID", "GPU_DEVICE"):
        v = (os.environ.get(var) or "").strip()
        if v:
            return v
    return None


def visible_cards() -> list[dict] | None:
    """[{index, name, uuid}] for every card this container can see; None when nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    cards = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            cards.append({"index": parts[0], "name": parts[1], "uuid": parts[2]})
    return cards


def check(expected: str | None, cards: list[dict] | None) -> str:
    """Raise GpuPinError when a pin is set and not met; return a one-line summary otherwise."""
    listing = ", ".join(f"{c['name']} ({c['uuid']})" for c in cards or []) or "none"
    if expected:
        if cards is None:
            raise GpuPinError(f"GPU_UUID={expected} is set but nvidia-smi could not list the cards")
        if len(cards) != 1:
            raise GpuPinError(
                f"GPU_UUID={expected} is set but this container sees {len(cards)} cards: {listing}. "
                "The pin did not reach docker -- recreate with deploy/run-worker.sh, which passes "
                "--device nvidia.com/gpu=$GPU_UUID")
        if cards[0]["uuid"].lower() != expected.lower():
            raise GpuPinError(
                f"GPU_UUID={expected} is set but this container sees {listing}. "
                "Wrong card -- check `nvidia-smi -L` on the host and fix GPU_UUID in worker.env")
        return f"GPU pinned: {cards[0]['name']} ({cards[0]['uuid']})"
    if cards and len(cards) > 1:
        return (f"!! WARNING: {len(cards)} cards visible and no GPU_UUID: {listing}. "
                "Services pick a card by index, which can change between boots. Set GPU_UUID "
                "in worker.env (wanly-gpu-docker#163).")
    return f"GPU: {listing} (not pinned)"


def enforce() -> str:
    """The boot check. Prints and returns the summary; raises GpuPinError on a broken pin."""
    line = check(expected_uuid(), visible_cards())
    print(line, flush=True)
    return line
