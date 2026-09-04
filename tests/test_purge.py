"""Purging a finished render's local media (console#380).

out.mp4 and the keyframe pngs are ~2.7 GB across 500 jobs and grow by ~5 MB per render,
forever. They are duplicates by the time this runs — the daemon uploads to S3 first.

graph.json and prompt.txt are NOT duplicates. Together they are ~7 MB across those same 500
jobs and are the only local record of what a render actually did: which LoRAs at which
strengths, against which checkpoint, from which prompt. Deleting them to reclaim 0.3% of the
space would be a bad trade, and these tests are what stops a future "purge everything"
simplification.
"""
import json

import pytest

from engine.purge import purge_all, purge_job_dir


@pytest.fixture
def jobdir(tmp_path):
    d = tmp_path / "abc123"
    d.mkdir()
    (d / "out.mp4").write_bytes(b"x" * 5_000_000)
    (d / "kf1.png").write_bytes(b"y" * 1_300_000)
    (d / "graph.json").write_text(json.dumps({"9601": {}}))
    (d / "prompt.txt").write_text("k3lly2026, a woman...")
    return d


def test_the_media_goes(jobdir):
    r = purge_job_dir(jobdir)
    assert not (jobdir / "out.mp4").exists()
    assert not (jobdir / "kf1.png").exists()
    assert set(r["removed"]) == {"out.mp4", "kf1.png"}
    assert r["freed_bytes"] == 6_300_000


def test_the_record_stays(jobdir):
    """This is the assertion that matters. graph.json answered "which base model did job
    905e9265 use" without re-running anything; it costs 13 KB."""
    purge_job_dir(jobdir)
    assert (jobdir / "graph.json").exists()
    assert (jobdir / "prompt.txt").exists()
    assert json.loads((jobdir / "graph.json").read_text()) == {"9601": {}}


def test_purging_twice_is_harmless(jobdir):
    """It runs after every upload, and an upload can be retried."""
    purge_job_dir(jobdir)
    r = purge_job_dir(jobdir)
    assert r["removed"] == []
    assert r["freed_bytes"] == 0


def test_a_missing_directory_is_reported_not_raised(tmp_path):
    """A job dir cleaned by hand, or a job that never wrote one. Purging must not fail the
    segment that just succeeded."""
    r = purge_job_dir(tmp_path / "does-not-exist")
    assert r["missing"] is True
    assert r["removed"] == []


def test_unknown_file_types_are_left_alone(jobdir):
    """Only known media is removed. Anything else in a job dir is there for a reason nobody
    has written down, and a purge should not be the thing that discovers what."""
    (jobdir / "notes.md").write_text("something someone left")
    purge_job_dir(jobdir)
    assert (jobdir / "notes.md").exists()


def test_the_sweep_leaves_the_newest_dirs_alone(tmp_path):
    """A render that finished seconds ago may still be being fetched by the daemon, and the
    sweep has no coordination with in-flight jobs. Skipping the newest is cheaper than a
    race whose loser is a lost render."""
    import os, time
    for i in range(8):
        d = tmp_path / f"job{i}"
        d.mkdir()
        (d / "out.mp4").write_bytes(b"x" * 1000)
        os.utime(d, (time.time() - (8 - i) * 60,) * 2)   # job7 newest
    r = purge_all(tmp_path, keep_recent=3)
    assert set(r["skipped_recent"]) == {"job7", "job6", "job5"}
    assert (tmp_path / "job7" / "out.mp4").exists()
    assert not (tmp_path / "job0" / "out.mp4").exists()
    assert r["dirs_purged"] == 5
