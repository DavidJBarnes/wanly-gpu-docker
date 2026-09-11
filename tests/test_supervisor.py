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
