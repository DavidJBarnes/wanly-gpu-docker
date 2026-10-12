"""A busy box must never look offline (wanly-gpu-daemon#189).

While the supervisor holds the worker row -- booting, and switching modes -- it reports what the
box is doing as a `phase` on registration and every heartbeat, and beats at once when the
phase changes. On 2026-10-12 3090a read "offline" for 10+ minutes during a switch into render;
these pin the supervisor's half of the fix (the daemon reports its own LoRA-sync phase).
"""
import asyncio

import pytest

from wanly_worker import control, phase
from wanly_worker import queue_client as qc
from wanly_worker.registry import select_mode

from tests.test_four_modes import THE_3090, _settle, _vram, box  # noqa: F401 -- fixture


class _Resp:
    status_code = 200

    def json(self):
        return {"id": "w-1"}

    def raise_for_status(self):
        pass


class _Client:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        return _Resp()


class _Sup:
    def snapshot(self):
        return [{"name": "image-edit", "ready": True, "running": True, "stopped": False}]


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(qc, "QUEUE_URL", "http://api.example:8001")
    monkeypatch.setattr(qc, "QUEUE_API_KEY", "k")
    monkeypatch.setattr(qc.QueueClient, "_gpu_stats", lambda self: None)


def test_registration_and_heartbeat_carry_the_phase():
    c = _Client()
    q = qc.QueueClient(_Sup(), c)
    phase.switching("edit", "render", "starting comfyui")
    asyncio.run(q.register())
    asyncio.run(q.beat())
    for _url, body in c.posts:
        assert body["phase"]["phase"] == "switching_mode"
        assert body["phase"]["from_mode"] == "edit" and body["phase"]["to_mode"] == "render"
        assert body["phase"]["detail"] == "Switching edit → render: starting comfyui"
    phase.ready()


def test_a_phase_change_beats_at_once_not_thirty_seconds_later(monkeypatch):
    monkeypatch.setattr(qc, "INTERVAL_S", 3600)
    c = _Client()
    q = qc.QueueClient(_Sup(), c)
    phase.on_change(q.poke)

    async def go():
        task = asyncio.create_task(q.run())
        await asyncio.sleep(0.05)                 # registered
        n = len(c.posts)
        phase.booting("starting comfyui")
        await asyncio.sleep(0.05)
        task.cancel()
        return n, len(c.posts)

    before, after = asyncio.run(go())
    assert after == before + 1
    phase.ready()


def test_a_step_keeps_the_switch_and_names_the_service():
    phase.switching("motion", "render")
    phase.step("starting render-daemon", service="render-daemon")
    s = phase.snapshot()
    assert s["phase"] == "switching_mode" and s["service"] == "render-daemon"
    assert s["detail"] == "Switching motion → render: starting render-daemon"
    phase.ready()
    phase.step("starting x")                      # outside a phase: a no-op, not a lie
    assert phase.snapshot()["detail"] == "Ready"


class TestTheSwitchReportsItself:
    def test_the_switch_ends_ready(self, box, monkeypatch):
        _vram(monkeypatch, 21000, 900)
        seen = []
        orig = phase.set_phase

        def spy(p, d, **f):
            seen.append(p)
            orig(p, d, **f)
        monkeypatch.setattr(phase, "set_phase", spy)
        _settle("train")
        assert seen[0] == "switching_mode"
        assert phase.current() == "ready"

    def test_a_failed_switch_still_ends_ready(self, box, monkeypatch):
        monkeypatch.setattr(control, "_training_now",
                            iter([None, "Joana v3 (step 1/9)"]).__next__)
        _vram(monkeypatch, 900)
        _settle("motion")
        assert phase.current() == "ready"

    def test_the_supervisor_takes_the_row_as_soon_as_the_daemon_stops(self, box, monkeypatch):
        """Not after the new mode is up: the daemon deletes the row as it exits, and the
        unload wait plus the starts would otherwise be a box missing from the page."""
        calls = []

        class Q:
            def rebalance(self):
                calls.append(list(box.applied))

            def poke(self):
                pass
        monkeypatch.setattr(control, "_queue", Q())
        _vram(monkeypatch, 21000, 900)
        _settle("motion")
        # The first rebalance comes after the stop (one apply), before the start (two).
        assert calls[0] == [[n for n in select_mode(THE_3090, "render")
                             if n in select_mode(THE_3090, "motion")]]


def test_the_queue_client_starts_before_the_services():
    """Boot reports `booting` instead of being absent until everything is up."""
    import inspect
    src = inspect.getsource(control.lifespan)
    assert src.index("_queue.start()") < src.index("_sup.start(")
    assert src.index("_sup.start(") < src.index("_queue.rebalance()")
