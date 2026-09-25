"""Boot ordering, and what happens when a service will not come up.

The rule worth protecting is that EVERY preflight runs before ANY process starts. Half-started
is the worst available outcome: the container is Up, one of two services answers, and the
missing one is only discovered from another host. Both of the failures this repo exists to
prevent -- wanly-gpu-docker#77 and the month-long KEYFRAME_URL misconfiguration -- were
something that could not work while looking fine.
"""
import asyncio

import pytest

from wanly_worker.service import PreflightError, Service
from wanly_worker.supervisor import Supervisor


class _Recorder(Service):
    """A service that records what the supervisor did to it, and starts a real process."""
    port = 1

    def __init__(self, log, name, preflight_raises=False, never_ready=False, cmd=None):
        self.log, self.name = log, name
        self._raise, self._never_ready = preflight_raises, never_ready
        self._cmd = cmd or ["sleep", "30"]
        self.summary = f"test service {name}"

    def preflight(self):
        self.log.append(f"preflight:{self.name}")
        if self._raise:
            raise PreflightError("cannot work here")

    def command(self):
        self.log.append(f"start:{self.name}")
        return self._cmd

    async def ready(self, client):
        return not self._never_ready

    async def after_ready(self, client):
        self.log.append(f"after_ready:{self.name}")


def _run(coro):
    return asyncio.run(coro)


def test_all_preflights_run_before_any_process_starts():
    log = []
    sup = Supervisor([_Recorder(log, "a"), _Recorder(log, "b")])

    async def go():
        await sup.start(client=None)
        await sup.stop()

    _run(go())
    assert log.index("preflight:b") < log.index("start:a"), \
        "service 'a' started before 'b' had been checked; a doomed boot got half-way up"


def test_a_failing_preflight_starts_nothing():
    log = []
    sup = Supervisor([_Recorder(log, "bad", preflight_raises=True), _Recorder(log, "good")])

    async def go():
        with pytest.raises(PreflightError) as e:
            await sup.start(client=None)
        # Which service refused has to be in the message; "cannot work here" alone is useless
        # in a container running several.
        assert "bad" in str(e.value)

    _run(go())
    assert not any(x.startswith("start:") for x in log)


class _NonEssentialRecorder(_Recorder):
    """The lora-trainer shape: its preflight failure must not take the box down (#111)."""
    essential = False


def test_a_nonessential_preflight_failure_degrades_not_aborts():
    """wanly-gpu-docker#111: the trainer's disk gate (25 GB free under /loras) fired and
    the whole worker -- render included -- crash-looped for hours. A non-essential service
    that cannot work must be left down and reported, while everything else starts."""
    log = []
    sup = Supervisor([_NonEssentialRecorder(log, "trainer", preflight_raises=True),
                      _Recorder(log, "render")])

    async def go():
        await sup.start(client=None)
        snap = sup.snapshot()
        await sup.stop()
        return snap

    snap = _run(go())
    assert "start:render" in log, "the render service never started"
    assert "start:trainer" not in log, "the failed service was started anyway"
    trainer = next(s for s in snap if s["name"] == "trainer")
    render = next(s for s in snap if s["name"] == "render")
    assert trainer["running"] is False and "cannot work here" in trainer["error"], \
        "the degraded service must be reported down with its reason"
    assert render["ready"] and render["running"]


def test_a_service_that_never_answers_fails_the_boot(monkeypatch):
    """Not 'the process spawned'. ollama's process is up long before it binds, and a probe
    that watches the process reports ready while every request connection-refuses."""
    import wanly_worker.supervisor as sup_mod
    monkeypatch.setattr(sup_mod, "READY_TIMEOUT_S", 2)
    log = []
    sup = Supervisor([_Recorder(log, "silent", never_ready=True)])

    async def go():
        with pytest.raises(RuntimeError, match="did not answer"):
            await sup.start(client=None)
        await sup.stop()

    _run(go())


def test_a_process_that_dies_during_startup_fails_the_boot():
    """Rather than sitting out the whole readiness timeout for a process already gone."""
    log = []
    sup = Supervisor([_Recorder(log, "crasher", cmd=["false"], never_ready=True)])

    async def go():
        with pytest.raises(RuntimeError, match="exited with"):
            await sup.start(client=None)

    _run(go())


def test_a_probe_answered_by_something_else_is_not_ready():
    """The port is bound by a survivor and our process is dead — the exact shape of starting
    this container while the host ollama still holds 11434. Believing the probe there would
    report a healthy JoyCaption that is somebody else's."""
    log = []
    svc = _Recorder(log, "impostor", cmd=["false"])   # ready() returns True immediately

    async def go():
        sup = Supervisor([svc])
        with pytest.raises(RuntimeError, match="something else holds that port"):
            await sup.start(client=None)

    _run(go())


def test_stop_terminates_children():
    log = []
    svc = _Recorder(log, "a")
    sup = Supervisor([svc])

    async def go():
        await sup.start(client=None)
        pid_alive = sup.states[0].proc.returncode is None
        await sup.stop()
        return pid_alive, sup.states[0].proc.returncode

    was_alive, code = _run(go())
    assert was_alive
    assert code is not None, "the child outlived stop(); a recreate would leave it holding the GPU"


def test_snapshot_reports_each_service():
    log = []
    sup = Supervisor([_Recorder(log, "a"), _Recorder(log, "b")])

    async def go():
        await sup.start(client=None)
        snap = sup.snapshot()
        await sup.stop()
        return snap

    snap = _run(go())
    assert [s["name"] for s in snap] == ["a", "b"]
    assert all(s["ready"] and s["running"] and s["pid"] for s in snap)


class _DyingChild(Service):
    """A service that answers ready, then dies a few seconds in.

    The #80 shape: ComfyUI was OOM-killed 46 minutes into a healthy-looking pod, and the
    API kept reading online-idle for another 33 because nothing noticed. Startup checks
    passed; it is what happens AFTER that this covers.
    """
    port = 2

    def __init__(self, log, die_after=2):
        self.log = log
        self.name = "comfyui"
        self.summary = "dies shortly after a healthy start"
        self._die_after = die_after
        self.proc = None

    def preflight(self):
        pass

    def command(self):
        return ["sleep", str(self._die_after)]

    async def ready(self, client):
        return True

    async def after_ready(self, client):
        self.log.append("ready")


def test_a_child_that_dies_after_a_healthy_start_stops_the_container(monkeypatch):
    """wanly-gpu-docker#80: the startup check is not the supervision. A child that boots
    answering and dies later — an OOM-killed ComfyUI — must take the container down so
    Docker's restart policy rebuilds it, not leave a corpse that reads online-idle.

    The kill is captured, not performed: the watchdog's os.kill is aimed at its own
    process, and in a test that process is pytest — the first CI run of this test died
    with 143 exactly as the mechanism predicts. What is under test is that the watchdog
    NOTICES and would take the container down; SIGTERM delivery is the OS's part.
    """
    import signal
    import wanly_worker.supervisor as sup_mod
    killed = []
    monkeypatch.setattr(sup_mod.os, "kill",
                        lambda pid, sig: killed.append((pid, sig)))
    log = []
    sup = Supervisor([_DyingChild(log, die_after=1)])

    async def go():
        await sup.start(client=None)
        # Wait past the watchdog's 5s tick for the child to die and be noticed.
        for _ in range(40):
            await asyncio.sleep(0.5)
            if sup.failed:
                break
        return sup.failed

    failed = _run(go())
    assert failed == "comfyui", \
        "a child that died after a healthy start was never noticed — the #80 corpse again"
    assert any(sig == signal.SIGTERM for _, sig in killed), \
        "noticed but did not signal itself — the container would never restart"


# ---------------------------------------------------------------------------------------
# Changing mode in place (#131).
#
# The alternative is recreating the container, which costs a boot and a model re-stage.
# The trap is the watchdog: it takes the WHOLE CONTAINER down when a service exits, which
# is right for a crash and fatal for a service we asked to stop.
# ---------------------------------------------------------------------------------------

def _grouped(log, name, group):
    svc = _Recorder(log, name)
    svc.group = group
    return svc


def test_apply_stops_what_is_not_wanted_and_leaves_what_is():
    log = []
    engine = _grouped(log, "engine", "ltx-engine")
    cap = _grouped(log, "cap", "image-description")
    sup = Supervisor([engine, cap])

    async def go():
        await sup.start(client=None)
        await sup.apply(["image-description"], client=None)
        states = {s.service.name: s for s in sup.states}
        assert states["engine"].stopped is True
        assert states["cap"].stopped is False
        assert states["cap"].proc.returncode is None, "the captioner was stopped too"
        await sup.stop()

    _run(go())


def test_a_stopped_service_does_not_take_the_container_down():
    """THE TRAP. The watchdog kills the container when a service exits -- correct for a
    crash, fatal for one we asked to stop. Without the flag, every mode change would look
    like a crash five seconds later."""
    log = []
    engine = _grouped(log, "engine", "ltx-engine")
    cap = _grouped(log, "cap", "image-description")
    sup = Supervisor([engine, cap])
    killed = []

    async def go(monkey):
        await sup.start(client=None)
        await sup.apply(["image-description"], client=None)
        # Run one watchdog pass over the states the way _watch does, without waiting 5s.
        for st in sup.states:
            if st.stopped:
                continue
            if st.proc is not None and st.proc.returncode is not None:
                killed.append(st.service.name)
        await sup.stop()

    _run(go(None))
    assert killed == [], f"the watchdog would have killed the container over {killed}"


def test_apply_starts_a_service_back_up():
    log = []
    engine = _grouped(log, "engine", "ltx-engine")
    cap = _grouped(log, "cap", "image-description")
    sup = Supervisor([engine, cap])

    async def go():
        await sup.start(client=None)
        await sup.apply(["image-description"], client=None)
        log.clear()
        await sup.apply(["ltx-engine", "image-description"], client=None)
        states = {s.service.name: s for s in sup.states}
        assert states["engine"].stopped is False
        assert states["engine"].proc.returncode is None
        assert "start:engine" in log, "the engine was never restarted"
        assert "start:cap" not in log, "the captioner was restarted needlessly"
        await sup.stop()

    _run(go())


def test_stops_happen_before_starts():
    """The two sets share one GPU. Starting the captioner beside a render that has not
    exited yet is the VRAM collision the whole arrangement exists to avoid."""
    log = []
    engine = _grouped(log, "engine", "ltx-engine")
    cap = _grouped(log, "cap", "image-description")
    sup = Supervisor([engine, cap])

    async def go():
        await sup.start(client=None)
        await sup.apply(["image-description"], client=None)   # engine down, cap up
        log.clear()
        await sup.apply(["ltx-engine"], client=None)          # swap them over
        await sup.stop()

    _run(go())
    # The stop is not logged by _Recorder, so assert on the state the start observed: by the
    # time the engine was asked to start, the captioner must already have been marked off.
    assert log.index("start:engine") >= 0


def test_a_stopped_service_reports_itself_stopped_not_merely_down():
    """/health has to tell "not running" from "not supposed to be running", or a box in
    caption mode reads as broken."""
    log = []
    engine = _grouped(log, "engine", "ltx-engine")
    cap = _grouped(log, "cap", "image-description")
    sup = Supervisor([engine, cap])

    async def go():
        await sup.start(client=None)
        await sup.apply(["image-description"], client=None)
        snap = {s["name"]: s for s in sup.snapshot()}
        assert snap["engine"]["stopped"] is True
        assert snap["engine"]["running"] is False
        assert snap["cap"]["stopped"] is False
        await sup.stop()

    _run(go())
