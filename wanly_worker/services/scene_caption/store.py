"""Is joycaption:beta-one in an ollama store, and what would it take to put it there?

Checked against the manifest ACTUALLY IN THE STORE, not the one image_description/model.py
would write. A store whose joycaption was imported by hand (3090b, ex-2070.zero) carries a
different 568-byte config blob than the rebuilt one, while every weight is byte-identical, and
"rebuild it" would overwrite a working model's manifest to fix nothing. So: if the store's own
manifest lists layers that are all present at their declared sizes, the model is there.

Otherwise the rebuild in image_description/model.py is the way in -- the same two public
GGUFs (pinned by revision and digest) plus the 1,070 bytes of template/system/params shipped
in this image. ~5.8 GB, which is what makes it fit a fresh box.

CLI, for download_models.sh --scene-caption (stdlib only, so it also runs on a host):

    python3 -m wanly_worker.services.scene_caption.store check  STORE
        exit 0 if present; else exit 1 and print one line per file to fetch:
        "<sha256> <size> <url>"
    python3 -m wanly_worker.services.scene_caption.store finish STORE
        install the small layers shipped in the image and write the manifest, once the two
        GGUFs are in place (fetched by curl, rate-limited, sha-verified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wanly_worker.services.image_description import model as jc

MODEL = "joycaption:beta-one"


def manifest_path(store: str) -> Path:
    return Path(store) / "models" / jc.MANIFEST_PATH


def present(store: str) -> tuple[bool, str]:
    """(True, how) when the store's own manifest is complete; else (False, why)."""
    mpath = manifest_path(store)
    if not mpath.is_file():
        return False, f"no manifest at {mpath}"
    try:
        m = json.loads(mpath.read_text())
        layers = [m["config"], *m["layers"]]
    except Exception as e:                      # noqa: BLE001
        return False, f"unreadable manifest {mpath}: {e}"
    for layer in layers:
        digest = layer["digest"].split(":", 1)[-1]
        p = Path(store) / "models" / "blobs" / f"sha256-{digest}"
        if not p.is_file() or p.stat().st_size != int(layer["size"]):
            return False, f"blob {digest[:16]}... missing or the wrong size"
    return True, f"manifest and {len(layers)} blobs present"


def to_fetch(store: str) -> list[jc.Blob]:
    """The GGUFs a rebuild would have to download (the small layers ship in the image)."""
    return [b for b in jc.missing(store) if b.remote]


def finish(store: str) -> None:
    """Small layers + manifest, after the GGUFs are in. Refuses if a GGUF is still missing."""
    still = to_fetch(store)
    if still:
        raise SystemExit(f"still missing: {', '.join(b.filename for b in still)}")
    blobs = Path(store) / "models" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for b in jc.missing(store):
        jc._install_local(b, store, log=print)
    mpath = manifest_path(store)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(jc.manifest()))
    print(f"wrote {mpath}")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("check", "finish"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd, store = argv
    if cmd == "check":
        ok, how = present(store)
        if ok:
            print(f"{MODEL}: {how}", file=sys.stderr)
            return 0
        print(f"{MODEL}: not present ({how})", file=sys.stderr)
        for b in to_fetch(store):
            print(f"{b.digest} {b.size} {jc.BASE_URL}/{b.filename}")
        return 1
    finish(store)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
