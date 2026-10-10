#!/usr/bin/env python3
"""Report which LoRAs A1111 has actually used, to wanly-api (#211, wanly-api#458).

WHY THIS EXISTS
    SDXL character LoRAs are used by hand in A1111 on 3090b, never by a Wanly render, so the
    Characters page could not say whether a new SDXL LoRA had been tried. A1111 writes its
    generation settings -- `<lora:Name:weight>` included -- into a `parameters` text chunk of
    every PNG it saves. The faceswapped copies uploaded to Wanly carry no text chunks, so this
    folder is the only record there is.

WHAT IT DOES
    Walks A1111's output folders (A1111_OUTPUT_DIRS, `:`-separated), reads each PNG's text
    chunks up to the first IDAT (never the pixel data), counts one image per LoRA name per
    file, and POSTs per-name TOTALS -- images, first and last used (file mtime) -- to
    `{QUEUE_URL}/lora-usage`. Totals, not increments, so a repeated POST is harmless.

    A state file remembers each scanned file's mtime and what it contributed, so a run only
    reads new or changed files. The first run reads everything once.

    Stdlib only, safe under `python3 -I`. `--dry-run` reads the folders and prints the totals
    without POSTing and without touching the state file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIRS = (
    "/home/david/StabilityMatrix-linux-x64/Data/Images/Text2Img:"
    "/home/david/StabilityMatrix-linux-x64/Data/Images/Img2Img"
)
DEFAULT_STATE = Path.home() / ".local/state/wanly/a1111-lora-usage.json"
DEFAULT_ENV = Path.home() / "wanly-gpu-docker/deploy/worker.env"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
#: `<lora:NAME>` or `<lora:NAME:0.8>` (also `<lora:NAME:0.8:0.6>`); NAME is everything to the
#: first `:` or `>`.
LORA_RE = re.compile(r"<lora:([^:>]+)(?::[^>]*)?>")
SOURCE = "a1111"


def png_parameters(path: Path) -> str | None:
    """The `parameters` text of a PNG, or None. Reads chunk headers and text chunks only and
    stops at IDAT, so a 2 MB image costs a few hundred bytes of reading."""
    try:
        with open(path, "rb") as f:
            if f.read(8) != PNG_SIG:
                return None
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return None
                n, kind = struct.unpack(">I4s", head)
                if kind in (b"IDAT", b"IEND"):
                    return None
                if kind in (b"tEXt", b"iTXt", b"zTXt"):
                    data = f.read(n)
                    f.seek(4, os.SEEK_CUR)  # CRC
                    text = _text_chunk(kind, data)
                    if text is not None:
                        return text
                else:
                    f.seek(n + 4, os.SEEK_CUR)
    except OSError:
        return None


def _text_chunk(kind: bytes, data: bytes) -> str | None:
    key, _, rest = data.partition(b"\x00")
    if key != b"parameters":
        return None
    try:
        if kind == b"tEXt":
            return rest.decode("latin-1")
        if kind == b"zTXt":
            return zlib.decompress(rest[1:]).decode("latin-1")
        # iTXt: compression flag, method, language\0, translated keyword\0, text
        flag = rest[0]
        rest = rest[2:]
        _, _, rest = rest.partition(b"\x00")
        _, _, rest = rest.partition(b"\x00")
        return (zlib.decompress(rest) if flag else rest).decode("utf-8", "replace")
    except (zlib.error, IndexError):
        return None


def loras_in(text: str | None) -> set[str]:
    """Distinct LoRA names in one image's parameters -- one image counts once per name."""
    return {m.strip() for m in LORA_RE.findall(text or "") if m.strip()}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def load_state(path: Path) -> dict:
    try:
        st = json.loads(path.read_text())
        if st.get("version") == 1:
            return st
    except (OSError, ValueError):
        pass
    return {"version": 1, "files": {}}


def scan(dirs: list[Path], state: dict) -> tuple[dict, int]:
    """Update `state` from the folders; return (state, files read this run).

    state["files"][path] = [mtime_ns, [names]] -- what each file contributed. A file whose
    mtime is unchanged is not read again; a vanished file keeps its contribution (the image was
    made, which is the fact being reported)."""
    files = state.setdefault("files", {})
    read = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for root, _, names in os.walk(d, followlinks=True):
            for nm in names:
                if not nm.lower().endswith(".png"):
                    continue
                p = os.path.join(root, nm)
                try:
                    mt = os.stat(p).st_mtime_ns
                except OSError:
                    continue
                prev = files.get(p)
                if prev and prev[0] == mt:
                    continue
                files[p] = [mt, sorted(loras_in(png_parameters(Path(p))))]
                read += 1
    return state, read


def totals(state: dict) -> dict[str, dict]:
    """{name: {images, first_used_at, last_used_at}} from every file's contribution."""
    out: dict[str, dict] = {}
    for mt, names in state.get("files", {}).values():
        ts = mt / 1e9
        for n in names:
            t = out.setdefault(n, {"images": 0, "first": ts, "last": ts})
            t["images"] += 1
            t["first"] = min(t["first"], ts)
            t["last"] = max(t["last"], ts)
    return {n: {"images": t["images"], "first_used_at": _iso(t["first"]),
                "last_used_at": _iso(t["last"])} for n, t in out.items()}


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


def post(url: str, key: str, body: list[dict]) -> int:
    req = urllib.request.Request(
        url.rstrip("/") + "/lora-usage", data=json.dumps({"items": body}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print totals; no POST, state file untouched")
    ap.add_argument("--state", type=Path, default=Path(os.environ.get("A1111_USAGE_STATE", DEFAULT_STATE)))
    ap.add_argument("--env", type=Path, default=Path(os.environ.get("WORKER_ENV", DEFAULT_ENV)))
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    dirs = [Path(p) for p in os.environ.get("A1111_OUTPUT_DIRS", DEFAULT_DIRS).split(":") if p]
    state, read = scan(dirs, load_state(a.state))
    tot = totals(state)
    print(f"{len(state['files'])} images known ({read} read this run), {len(tot)} LoRA names")
    for n, t in sorted(tot.items(), key=lambda kv: -kv[1]["images"])[:a.top]:
        print(f"  {t['images']:6d}  {n}  (last {t['last_used_at'][:16]})")
    if a.dry_run:
        return 0
    env = read_env(a.env)
    url = os.environ.get("QUEUE_URL") or env.get("QUEUE_URL")
    key = os.environ.get("QUEUE_API_KEY") or env.get("QUEUE_API_KEY")
    if not url or not key:
        print(f"!! QUEUE_URL / QUEUE_API_KEY not found (env or {a.env})", file=sys.stderr)
        return 2
    body = [{"name": n, "source": SOURCE, **t} for n, t in tot.items()]
    status = post(url, key, body)
    a.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.state.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(a.state)
    print(f"posted {len(body)} names -> {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
