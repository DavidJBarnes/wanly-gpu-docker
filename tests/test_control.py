"""/health has to be actionable, not decorative.

The point of this endpoint is that a timer, a probe or a one-line `curl -sf` can act on it
without parsing prose. That means the STATUS CODE carries the verdict -- the same reason
wanly-gpu-docker's update timer treats an unreadable worker status as busy rather than idle.
A /health that is always 200 can only be read by a human who already suspected something.
"""
import json

import pytest

from wanly_worker import control


class _FakeSup:
    def __init__(self, services):
        self._services = services

    def snapshot(self):
        return self._services


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    # The runner has no GPU and the 2070 does; neither should decide whether this passes.
    monkeypatch.setattr(control, "gpu_snapshot", lambda: None)


def _get():
    import asyncio
    resp = asyncio.run(control.health())
    return resp.status_code, json.loads(bytes(resp.body))


def test_all_ready_is_200_ok(monkeypatch):
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True},
    ]))
    code, body = _get()
    assert code == 200
    assert body["status"] == "ok"
    assert body["services"][0]["name"] == "joycaption"


def test_one_service_down_is_503(monkeypatch):
    """Partially up is the state that must not read as healthy: wanly-api would keep sending
    captions to a port that answers nothing."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True},
        {"name": "other", "ready": False, "running": False},
    ]))
    code, body = _get()
    assert code == 503
    assert body["status"] == "degraded"


def test_no_services_at_all_is_503(monkeypatch):
    """Before the lifespan has started anything, or after it gave up. Not 'ok, zero services'."""
    monkeypatch.setattr(control, "_sup", _FakeSup([]))
    code, _ = _get()
    assert code == 503


def test_it_reports_which_build_it_is(monkeypatch):
    """wanly-gpu-docker#72 was fourteen hours of two boxes running different code with no way
    to tell them apart. The fix was making the build say so; same here."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True}]))
    _, body = _get()
    assert "build" in body


def test_health_reports_running_code_separately_from_the_image(tmp_path, monkeypatch):
    """#116: engine/ and wanly_worker/ are fetched at boot, so the image's sha no longer
    answers 'which code'. Both refs, or #72 is back with a new mechanism."""
    ref = tmp_path / "code_ref"
    ref.write_text("main @ abc1234\n")
    monkeypatch.setattr(control, "CODE_REF_FILE", str(ref))
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "ltx-engine-api", "ready": True, "running": True}]))
    code, body = _get()
    assert code == 200
    assert body["code"] == "main @ abc1234"


def test_no_code_ref_file_means_baked_code(tmp_path, monkeypatch):
    """The fetch failed before writing the ref (or this image predates #116) — say baked,
    don't omit the field."""
    monkeypatch.setattr(control, "CODE_REF_FILE", str(tmp_path / "absent"))
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "ltx-engine-api", "ready": True, "running": True}]))
    _, body = _get()
    assert body["code"].startswith("baked")


def test_gpu_snapshot_never_raises(monkeypatch):
    """It is diagnostic. A missing or broken nvidia-smi must not take /health down with it."""
    import wanly_worker.supervisor as sup_mod
    monkeypatch.setattr(sup_mod.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert sup_mod.gpu_snapshot() is None
