"""Registering with wanly-api (wanly-api#269).

Two properties matter more than the payload shape.

FIRST, this must never be able to stop a caption. The container's job is to serve JoyCaption;
telling the API about it is observability. An unreachable API, a wrong key, no key at all --
none of those is a reason to refuse to caption, and a container that died because a reporting
channel was down would be a worse outage than the one it was reporting.

SECOND, it must say kind=service. The API's claim gate keys on that, and `_model_gate`
explicitly does not filter a worker that has never reported checkpoints -- so a service that
registered as `render` would be offered every pending segment.
"""
import asyncio

import pytest

from wanly_worker import queue_client as qc


class _Sup:
    def __init__(self, rows): self._rows = rows
    def snapshot(self): return self._rows


def _ready(*names): return [{"name": n, "ready": True} for n in names]


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}
    def json(self): return self._body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, resp=None, boom=None):
        self.posts, self.deletes = [], []
        self._resp = resp or _Resp(200, {"id": "w-1"})
        self._boom = boom
    async def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        if self._boom: raise self._boom
        return self._resp
    async def delete(self, url, headers=None, timeout=None):
        self.deletes.append(url)
        if self._boom: raise self._boom
        return _Resp(204)


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(qc, "QUEUE_URL", "http://api.example:8001")
    monkeypatch.setattr(qc, "QUEUE_API_KEY", "k")


def run(c): return asyncio.run(c)


class TestItNeverBreaksTheContainer:
    def test_an_unreachable_api_does_not_raise(self):
        c = qc.QueueClient(_Sup(_ready("joycaption")), _Client(boom=OSError("no route")))
        run(c.register())               # must not raise
        assert c.worker_id is None

    def test_a_failed_heartbeat_does_not_raise(self):
        c = qc.QueueClient(_Sup(_ready("joycaption")), _Client(boom=OSError("timeout")))
        c.worker_id = "w-1"
        run(c.beat())                   # must not raise

    def test_no_credentials_means_no_calls_at_all(self, monkeypatch):
        """A dev box, or the 2070 before the key is added. It should caption, quietly absent
        from the Workers page, rather than log a failure every 30 seconds."""
        monkeypatch.setattr(qc, "QUEUE_URL", "")
        monkeypatch.setattr(qc, "QUEUE_API_KEY", "")
        cl = _Client()
        run(qc.QueueClient(_Sup(_ready("joycaption")), cl).register())
        assert cl.posts == []

    def test_deregister_is_silent_when_it_fails(self):
        c = qc.QueueClient(_Sup(_ready("joycaption")), _Client(boom=OSError("gone")))
        c.worker_id = "w-1"
        run(c.deregister())             # must not raise


class TestWhatItClaimsToBe:
    def test_it_registers_as_a_service(self):
        """The single most important assertion here: registering as `render` would put this
        container in the claim pool, and the never-reported-checkpoints exemption would then
        offer it the entire queue."""
        cl = _Client()
        run(qc.QueueClient(_Sup(_ready("joycaption")), cl).register())
        _, body = cl.posts[0]
        assert body["kind"] == "service"

    def test_it_reports_what_it_runs(self):
        cl = _Client()
        run(qc.QueueClient(_Sup(_ready("joycaption", "qwen-edit")), cl).register())
        assert cl.posts[0][1]["provides"] == ["joycaption", "qwen-edit"]

    def test_the_name_is_host_scoped(self, monkeypatch):
        """register_worker upserts on friendly_name and REUSES the row. Plain `2070.zero`
        would silently share one row with anything else on that host."""
        monkeypatch.delenv("FRIENDLY_NAME", raising=False)
        assert qc.friendly_name().endswith("/services")

    def test_an_explicit_name_wins(self, monkeypatch):
        monkeypatch.setenv("FRIENDLY_NAME", "2070.zero/services")
        assert qc.friendly_name() == "2070.zero/services"


class TestStatus:
    def test_everything_ready_is_online(self):
        c = qc.QueueClient(_Sup(_ready("joycaption")), _Client())
        assert c.status() == "online"

    def test_one_service_down_is_degraded(self):
        """A state the render vocabulary cannot express, which is why the API grew a word for
        it rather than reusing online-idle."""
        rows = _ready("joycaption") + [{"name": "qwen-edit", "ready": False}]
        assert qc.QueueClient(_Sup(rows), _Client()).status() == "degraded"

    def test_nothing_running_is_degraded_not_online(self):
        assert qc.QueueClient(_Sup([]), _Client()).status() == "degraded"

    def test_it_never_says_online_idle(self):
        """Borrowing that word would extend the exact ambiguity wanly-gpu-docker#80 is open
        about: `online-idle` already means both waiting-for-work and cannot-do-work."""
        for rows in ([], _ready("joycaption"),
                     _ready("a") + [{"name": "b", "ready": False}]):
            assert qc.QueueClient(_Sup(rows), _Client()).status() in ("online", "degraded")


class TestTheHeartbeat:
    def test_it_carries_status_provides_and_gpu(self, monkeypatch):
        import wanly_worker.supervisor as sup_mod
        monkeypatch.setattr(sup_mod, "gpu_snapshot",
                            lambda: {"name": "RTX 2070", "vram_total_mib": 8192,
                                     "vram_used_mib": 900, "vram_free_mib": 7292})
        cl = _Client()
        c = qc.QueueClient(_Sup(_ready("joycaption")), cl)
        c.worker_id = "w-1"
        run(c.beat())
        _, body = cl.posts[0]
        assert body["status"] == "online"
        assert body["provides"] == ["joycaption"]
        # The daemon's exact shape, so the console renders it with no console change.
        assert body["gpu_stats"] == {"gpu_name": "RTX 2070", "vram_used_mb": 900,
                                     "vram_total_mb": 8192}

    def test_no_gpu_is_reported_as_none_not_a_crash(self, monkeypatch):
        import wanly_worker.supervisor as sup_mod
        monkeypatch.setattr(sup_mod, "gpu_snapshot", lambda: None)
        cl = _Client()
        c = qc.QueueClient(_Sup(_ready("joycaption")), cl)
        c.worker_id = "w-1"
        run(c.beat())
        assert cl.posts[0][1]["gpu_stats"] is None

    def test_a_404_triggers_re_registration_rather_than_silence(self):
        """The row was deleted. A service that stopped reporting after one bad response would
        be an invisible box, which is what this whole client exists to prevent."""
        cl = _Client(resp=_Resp(404))
        c = qc.QueueClient(_Sup(_ready("joycaption")), cl)
        c.worker_id = "w-1"
        run(c.beat())
        assert c.worker_id is None

    def test_beating_without_an_id_registers_first(self):
        cl = _Client()
        c = qc.QueueClient(_Sup(_ready("joycaption")), cl)
        run(c.beat())
        assert cl.posts[0][0].endswith("/workers")
        assert c.worker_id == "w-1"


class TestShutdown:
    def test_a_clean_stop_deregisters(self):
        """Otherwise a deliberate `docker stop` leaves a row that goes stale and is swept to
        offline two minutes later — indistinguishable from a box that fell over."""
        cl = _Client()
        c = qc.QueueClient(_Sup(_ready("joycaption")), cl)
        c.worker_id = "w-1"
        run(c.deregister())
        assert cl.deletes == ["http://api.example:8001/workers/w-1"]

    def test_nothing_to_deregister_is_not_an_error(self):
        cl = _Client()
        run(qc.QueueClient(_Sup(_ready("joycaption")), cl).deregister())
        assert cl.deletes == []


class TestItSaysWhatItDid:
    """The registration line was written with logging.info and was invisible in `docker logs`
    on the first real deployment. The box HAD registered; the only way to learn that was to
    ask the API.

    Worse than merely quiet: warnings escape through logging's lastResort handler while INFO
    does not, so success was silent and failure was visible — a healthy boot and one that
    never attempted registration looked identical.
    """

    def test_registering_is_announced(self, capsys):
        run(qc.QueueClient(_Sup(_ready("joycaption")), _Client()).register())
        assert "registered with wanly-api" in capsys.readouterr().out

    def test_failing_to_register_is_announced(self, capsys):
        run(qc.QueueClient(_Sup(_ready("joycaption")), _Client(boom=OSError("no route"))).register())
        out = capsys.readouterr().out
        assert "could not register" in out and "continuing" in out

    def test_skipping_registration_is_announced(self, capsys, monkeypatch):
        monkeypatch.setattr(qc, "QUEUE_URL", "")
        monkeypatch.setattr(qc, "QUEUE_API_KEY", "")
        run(qc.QueueClient(_Sup(_ready("joycaption")), _Client()).register())
        assert "not registering" in capsys.readouterr().out

    def test_nothing_here_uses_the_logging_module(self):
        """A module logger's INFO records go nowhere under uvicorn's config, which leaves the
        root logger alone. Everything on this container's boot path prints."""
        import pathlib
        src = (pathlib.Path(qc.__file__)).read_text()
        assert "logger." not in src
        assert "import logging" not in src


class TestWhatKindOfWorkerThisIs:
    """The API's claim gates key on `kind`, so registering as the wrong one fails SILENTLY: the
    box registers, heartbeats, shows green on the Workers page, and quietly never claims.

    Caught in production — the trainer booted, registered, and returned null from
    /training/next because it had said `service`."""

    def test_a_trainer_says_trainer(self):
        assert qc.worker_kind(["lora-trainer"]) == "trainer"

    def test_a_captioner_says_service(self):
        assert qc.worker_kind(["joycaption"]) == "service"

    def test_a_box_running_both_says_trainer(self):
        """The stronger claim wins: a trainer that also captions still needs training jobs."""
        assert qc.worker_kind(["joycaption", "lora-trainer"]) == "trainer"

    def test_nothing_running_is_still_a_service(self):
        assert qc.worker_kind([]) == "service"

    def test_registration_sends_the_derived_kind(self):
        cl = _Client()
        run(qc.QueueClient(_Sup(_ready("lora-trainer")), cl).register())
        assert cl.posts[0][1]["kind"] == "trainer"


class TestTheRenderDaemonIsTheRegistrar:
    """One row per box, one writer (wanly-gpu-docker#83). When the render daemon is one of
    the processes it registers, heartbeats the rich payload and owns the status vocabulary;
    a second writer on the same row would undo its drain and fight its status."""

    def test_the_supervisor_stays_quiet_beside_a_render_daemon(self):
        rows = _ready("comfyui", "ltx-engine-api", "render-daemon")
        c = qc.QueueClient(_Sup(rows), _Client())
        c.start()
        assert c._task is None

    def test_a_box_without_one_registers_itself(self):
        c = qc.QueueClient(_Sup(_ready("joycaption")), _Client())
        assert not c.render_daemon_registers()

    def test_provides_reports_the_capability_not_the_processes(self):
        rows = [{"name": n, "group": "ltx-engine", "ready": True}
                for n in ("comfyui", "ltx-engine-api", "render-daemon")]
        c = qc.QueueClient(_Sup(rows), _Client())
        assert c.provides() == ["ltx-engine"]
