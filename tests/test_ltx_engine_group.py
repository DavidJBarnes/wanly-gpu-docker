"""The render stack as three supervised processes (wanly-gpu-docker#83).

What start.sh's phases guaranteed and the group must keep: models and the CUDA gate before
anything starts, ComfyUI before the engine before the daemon, each with the room it needs to
come up, and the daemon's readiness read from the file it writes after registering.
"""
import asyncio
import inspect
import pathlib

import pytest

from wanly_worker import supervisor as sup_mod
from wanly_worker.service import PreflightError
from wanly_worker.services import ltx_engine as mod
from wanly_worker.supervisor import Supervisor

HERE = pathlib.Path(__file__).parent


def test_the_order_is_comfyui_engine_daemon():
    assert [s.name for s in mod.ltx_engine_group()] == ["comfyui", "ltx-engine-api", "render-daemon"]


def test_each_process_has_its_own_room_to_boot():
    """One 120 s default would fail every cold pod: ComfyUI's import is minutes and the
    daemon syncs every character LoRA before it registers."""
    c, e, d = mod.ltx_engine_group()
    assert c.ready_timeout_s >= 180
    assert e.ready_timeout_s >= 120
    assert d.ready_timeout_s >= 3600


def test_children_run_where_their_code_expects():
    """The engine imports sibling modules from cwd; the daemon reads .env from cwd."""
    c, e, d = mod.ltx_engine_group()
    assert c.cwd == "/app/ComfyUI"
    assert e.cwd == "/opt/engine"
    assert d.cwd == mod.DAEMON_DIR


def test_the_chatty_children_log_to_files_and_the_daemon_to_stdout():
    c, e, d = mod.ltx_engine_group()
    assert c.log_path and c.log_path.endswith("comfyui.log")
    assert e.log_path and e.log_path.endswith("ltx-engine.log")
    assert d.log_path is None


def test_the_daemon_gets_time_to_finish_a_segment_on_stop():
    assert mod.RenderDaemon().stop_grace_s >= 1500


def test_comfyui_preflight_runs_the_models_script_then_the_cuda_gate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "DOWNLOAD_MODELS", str(tmp_path / "dl.sh"))
    (tmp_path / "dl.sh").write_text("#!/bin/sh\necho models\n")
    real = mod.subprocess.run

    def fake_run(argv, **kw):
        calls.append(argv[:2])
        class R:
            returncode = 0
        return R()
    monkeypatch.setattr(mod.subprocess, "call", lambda argv: (calls.append(argv), 0)[1])
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.ComfyUI().preflight()
    assert calls[0][0] == "bash" and calls[0][1].endswith("dl.sh")
    assert calls[1] == ["python3", "-c"]


def test_a_failed_models_script_is_a_preflight_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DOWNLOAD_MODELS", str(tmp_path / "dl.sh"))
    (tmp_path / "dl.sh").write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(PreflightError, match="models failed"):
        mod.ComfyUI().preflight()


def test_a_missing_cuda_device_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DOWNLOAD_MODELS", str(tmp_path / "dl.sh"))
    (tmp_path / "dl.sh").write_text("#!/bin/sh\nexit 0\n")

    class R:
        returncode = 1
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(PreflightError, match="run-worker.sh"):
        mod.ComfyUI().preflight()


def test_the_daemon_is_ready_when_it_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "READY_FILE", str(tmp_path / "ready"))
    d = mod.RenderDaemon()
    assert asyncio.run(d.ready(None)) is False
    (tmp_path / "ready").write_text("ready\n")
    assert asyncio.run(d.ready(None)) is True


def test_the_daemon_is_told_where_to_say_so():
    env = mod.RenderDaemon().env()
    assert env["WANLY_READY_FILE"] == mod.READY_FILE
    assert env["WORKER_ID_FILE"] == mod.WORKER_ID_FILE


def test_a_stale_ready_file_is_removed_before_the_daemon_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "READY_FILE", str(tmp_path / "ready"))
    monkeypatch.setattr(mod, "DAEMON_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "FETCH_DAEMON", str(tmp_path / "fetch.sh"))
    (tmp_path / "fetch.sh").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / ".env").write_text("ENGINE=ltx\n")
    (tmp_path / "ready").write_text("ready\n")
    mod.RenderDaemon().preflight()
    assert not (tmp_path / "ready").exists()


def test_the_supervisor_honours_per_service_timeouts(monkeypatch):
    """The module default is 120 s; a service that asks for more must get it."""
    src = inspect.getsource(sup_mod.Supervisor._start_one)
    assert "svc.ready_timeout_s or READY_TIMEOUT_S" in src
    assert "cwd=svc.cwd" in src


def test_children_stop_in_reverse_order():
    """The daemon stops claiming before the engine it drives goes."""
    src = inspect.getsource(sup_mod.Supervisor.stop)
    assert "reversed(self.states)" in src


def test_start_sh_execs_the_supervisor_and_keeps_its_gates():
    start = (HERE.parent / "start.sh").read_text()
    assert "exec python3 -m uvicorn wanly_worker.control:app" in start
    assert "if ! GPU_LINE=" in start          # the pinned GPU gate
    assert 'image build:' in start
    assert "PHASE 3/6" not in start          # the phases live in the group now
