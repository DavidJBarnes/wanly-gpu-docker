"""Move a staged package tree onto its destination IN PLACE (wanly-gpu-docker#125).

    swap_sync.py <staged-dir> <destination-dir>

`mv /opt/engine /opt/engine.old` — the obvious way to swap a package directory — cannot
work inside a container, and the failure modes are evil rather than loud:

  * A directory baked into the image lives on an overlayfs LOWER layer. Renaming it is
    cross-device (EXDEV); GNU mv responds by COPYING every file to the upper layer and
    UNLINKING the source, which fails only when it reaches a mount point — by which time
    the directory has been emptied. That is the 2026-09-22 3090 crash loop: "keeping
    current engine" printed over an /opt/engine holding nothing but the mount point.
  * os.rename refuses atomically — which is SAFE but can never succeed in production
    either, because every boot starts from lower-layer directories (measured: an
    unmounted /opt/worker also raises EXDEV).

In-place is the only mechanism that works against a lower layer: files get replaced, the
directory itself is never moved. Content is staged first (by the caller, at
`<staged-dir>`), so every read of the fetched tree has already succeeded before this
script touches the running code; a corrupt or partial FETCH therefore cannot reach the
live tree. The flip itself is per-file os.replace (atomic on overlayfs — verified, not
assumed, against an image-baked file).

Refusals happen BEFORE anything changes:

  exit 3  a mount point at or inside the destination — syncing into a mounted directory
          would mutate the host's bind mount, not the image; the caller keeps the current
          code and says so. A mounted directory inside a package is a refusal, full stop.
  exit 4  a file in the destination collides with a DIRECTORY the incoming tree brings —
          replacing mid-flip and discovering this halfway would leave a mix.

A crash mid-flip leaves a mix of old and new — no in-place mechanism can promise
otherwise. What #125 requires instead is narrower and absolute: a FAILED swap never
DESTROYS the current code, and the boot identity never claims a tree that is not running.
Those are the properties the tests pin.
"""
import os
import sys


def _files(root):
    out = set()
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for name in filenames:
            out.add(os.path.normpath(os.path.join(rel, name)))
    return out


def _rel_dirs(root):
    out = set()
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            out.add(os.path.normpath(os.path.join(os.path.relpath(dirpath, root), d)))
    return out


def _mounts_under(root):
    found = [root] if os.path.ismount(root) else []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if os.path.ismount(p):
                found.append(p)
    return found


def main(staged, dst):
    if os.path.ismount(dst):
        print(f"swap_sync: {dst} is itself a mount point — refusing", file=sys.stderr)
        return 3
    mounts = _mounts_under(dst)
    if mounts:
        print(f"swap_sync: mount point(s) inside {dst}: {', '.join(mounts)} — refusing "
              "to sync (changes would land on the mounted filesystem, not the image)",
              file=sys.stderr)
        return 3

    os.makedirs(dst, exist_ok=True)
    incoming = _files(staged)
    current = _files(dst)

    # Type collisions fail mid-flip and leave a mix; refuse before touching anything.
    for rel in sorted(incoming):
        target = os.path.join(dst, rel)
        if os.path.isdir(target) and not os.path.islink(target):
            print(f"swap_sync: {target} is a directory but the incoming tree has a file "
                  "there — refusing to flip a half-tree", file=sys.stderr)
            return 4
    for rel in sorted(_rel_dirs(staged)):
        target = os.path.join(dst, rel)
        if os.path.exists(target) and not os.path.isdir(target):
            print(f"swap_sync: {target} is a file but the incoming tree has it as a "
                  "directory — refusing to flip a half-tree", file=sys.stderr)
            return 4

    for rel in sorted(incoming):
        target = os.path.join(dst, rel)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.replace(os.path.join(staged, rel), target)

    for rel in sorted(current - incoming):
        os.remove(os.path.join(dst, rel))

    # Directories the incoming tree no longer has. topdown=False yields children before
    # parents, so rmdir's naturally; mount points are never removed and leftovers from a
    # race are benign (an empty dir changes no imports).
    for dirpath, _dirnames, _ in os.walk(dst, topdown=False):
        rel = os.path.relpath(dirpath, dst)
        if rel == ".":
            continue
        try:
            if not os.path.ismount(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
