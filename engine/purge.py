"""Reclaiming a finished render's local media (console#380).

Pure filesystem logic, deliberately in its own module rather than app.py: app.py imports
comfy, which drags in the whole ComfyUI client, and a rule about which files to delete
should be testable without any of that. Same reason lora_stack_note lives in recipe.py.

WHAT GOES AND WHAT STAYS

    out.mp4 and the keyframe pngs are ~2.7 GB across 500 jobs and grow by ~5 MB per render,
    forever. By the time this runs they are duplicates: the daemon uploads to S3 first and
    only then asks for the purge.

    graph.json and prompt.txt are NOT duplicates. Together they are ~7 MB across those same
    500 jobs — 0.3% of the space — and they are the only local record of what a render
    actually did: which LoRAs at which strengths, against which checkpoint, from which
    prompt. That record answered "which base model did job 905e9265 use" without re-running
    anything. Deleting it to reclaim 7 MB would be a bad trade.
"""
from pathlib import Path

PURGE_SUFFIXES = (".mp4", ".png", ".jpg", ".jpeg", ".webp", ".webm")
KEEP_NAMES = ("graph.json", "prompt.txt")


def purge_job_dir(workdir: Path) -> dict:
    """Delete a finished job's media, keep its record. Returns what went."""
    removed, freed = [], 0
    if not workdir.is_dir():
        return {"removed": [], "freed_bytes": 0, "missing": True}
    for f in sorted(workdir.iterdir()):
        if not f.is_file() or f.name in KEEP_NAMES:
            continue
        if f.suffix.lower() not in PURGE_SUFFIXES:
            # Only known media is removed. Anything else in a job dir is there for a reason
            # nobody has written down, and a purge should not be what discovers what.
            continue
        try:
            size = f.stat().st_size
            f.unlink()
            removed.append(f.name)
            freed += size
        except OSError as e:
            # One unlink failing must not abort the rest — a file held open elsewhere is a
            # reason to skip it, not to leave the other 5 MB behind.
            print(f"[purge] {f}: {type(e).__name__}: {e}", flush=True)
    return {"removed": removed, "freed_bytes": freed, "missing": False}


def purge_all(jobs_dir: Path, keep_recent: int = 5) -> dict:
    """Sweep everything already on disk (the one-time half of console#380).

    `keep_recent` leaves the newest few dirs alone. A render that finished seconds ago may
    still be being fetched by the daemon, and there is no coordination between this sweep
    and an in-flight job — so the newest are skipped rather than raced.
    """
    if not jobs_dir.is_dir():
        return {"dirs_purged": 0, "files_removed": 0, "freed_bytes": 0, "skipped_recent": []}
    dirs = sorted((d for d in jobs_dir.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    skipped = [d.name for d in dirs[:keep_recent]]
    freed = files = touched = 0
    for d in dirs[keep_recent:]:
        r = purge_job_dir(d)
        if r["removed"]:
            touched += 1
            files += len(r["removed"])
            freed += r["freed_bytes"]
    return {"dirs_purged": touched, "files_removed": files,
            "freed_bytes": freed, "skipped_recent": skipped}
