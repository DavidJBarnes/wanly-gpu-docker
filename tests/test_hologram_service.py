"""`SERVICES=hologram`: AR holograms on the 2070's scene box, off the render boxes (#228).

What has to hold for it to claim at all, and to claim without disturbing its neighbours: the
box registers as kind `hologram` (the API's gate), the hologram worker learns the row id from
the registrar, and adding it changes nothing about which other services each mode runs.
"""
import asyncio

import pytest

from wanly_worker import queue_client as qc
from wanly_worker import registry
from wanly_worker.services import hologram as holo

SCENE = ["scene-caption", "prompt-enhance", "hologram"]


class TestWhatTheBoxIs:
    def test_the_scene_box_registers_as_hologram(self):
        assert registry.kinds_for(SCENE) == ["hologram"]

    def test_without_it_the_scene_box_is_still_a_service(self):
        assert registry.kinds_for(["scene-caption", "prompt-enhance"]) == ["service"]

    def test_a_render_box_stays_render_first(self):
        assert registry.kinds_for(["ltx-engine", "hologram"]) == ["render", "hologram"]


class TestModesAreUnchanged:
    """hologram is CPU work and no card's tenant: it must not turn the scene box into a
    'rendering' box, which would drop prompt-enhance from render mode (#173's rule)."""

    def test_render_mode_runs_everything(self):
        assert registry.select_mode(SCENE, None) == SCENE

    def test_it_runs_in_motion_mode_too(self):
        assert registry.select_mode(SCENE, "motion") == SCENE


class _Sup:
    def __init__(self, names):
        self._rows = [{"name": n, "ready": True, "running": True, "stopped": False} for n in names]

    def snapshot(self):
        return self._rows


class _Resp:
    status_code = 200

    def json(self):
        return {"id": "row-42"}

    def raise_for_status(self):
        pass


class _Client:
    async def post(self, url, json=None, headers=None, timeout=None):
        return _Resp()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(qc, "QUEUE_URL", "http://api.example:8001")
    monkeypatch.setattr(qc, "QUEUE_API_KEY", "k")


class TestTheRowIdHandoff:
    def test_the_registrar_writes_the_id_for_the_hologram_worker(self, api, tmp_path, monkeypatch):
        path = tmp_path / "run" / "worker-id"
        monkeypatch.setenv("WORKER_ID_FILE", str(path))
        c = qc.QueueClient(_Sup(SCENE), _Client())
        asyncio.run(c.register())
        assert path.read_text() == "row-42\n"

    def test_not_on_a_box_without_hologram(self, api, tmp_path, monkeypatch):
        """There the render daemon owns the file; a second writer races it."""
        path = tmp_path / "worker-id"
        monkeypatch.setenv("WORKER_ID_FILE", str(path))
        asyncio.run(qc.QueueClient(_Sup(["scene-caption"]), _Client()).register())
        assert not path.exists()


class TestTheService:
    def test_it_is_the_daemons_hologram_entrypoint(self):
        svc = holo.HologramWorker()
        assert svc.command() == ["python3", "-m", "daemon.hologram_worker"]
        assert svc.cwd == holo.DAEMON_DIR

    def test_it_is_handed_the_handshake_files(self):
        env = holo.HologramWorker().env()
        assert env["WANLY_READY_FILE"] == holo.READY_FILE
        assert env["WORKER_ID_FILE"] == holo.WORKER_ID_FILE

    def test_onnx_threads_default_to_half_the_cores(self, monkeypatch):
        monkeypatch.delenv("ONNX_THREADS", raising=False)
        monkeypatch.setattr(holo.os, "cpu_count", lambda: 12)
        assert holo.HologramWorker().env()["ONNX_THREADS"] == "6"
        monkeypatch.setenv("ONNX_THREADS", "3")
        assert holo.HologramWorker().env()["ONNX_THREADS"] == "3"

    def test_ready_is_the_file_the_worker_writes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(holo, "READY_FILE", str(tmp_path / "hologram.ready"))
        svc = holo.HologramWorker()
        assert asyncio.run(svc.ready(None)) is False
        (tmp_path / "hologram.ready").write_text("ready\n")
        assert asyncio.run(svc.ready(None)) is True
        svc.prestart()
        assert asyncio.run(svc.ready(None)) is False

    def test_a_stale_daemon_is_refused_at_boot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(holo, "_run_script", lambda *a: None)
        monkeypatch.setattr(holo, "DAEMON_DIR", str(tmp_path))
        with pytest.raises(holo.PreflightError, match="hologram_worker"):
            holo.HologramWorker().preflight()
