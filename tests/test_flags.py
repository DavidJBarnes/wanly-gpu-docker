"""The SERVICES flag: what is asked for is what runs, and nothing else boots clean.

An unknown name is fatal at boot with the list of real names; an empty list is fatal; a name
may stand for several processes (ltx-engine is ComfyUI, the engine API and the render daemon,
in that order) and `provides` reports the name, not the processes.
"""
import pathlib

import pytest

from wanly_worker import registry
from wanly_worker.registry import ConfigError, build, parse_services
from wanly_worker.service import Service


class _One(Service):
    name = "one"
    port = 1
    summary = "test"

    def command(self):
        return ["true"]

    async def ready(self, client):
        return True


KNOWN = {"ltx-engine": registry.KNOWN["ltx-engine"], "one": lambda: _One()}


def test_a_single_name_parses():
    assert parse_services("ltx-engine", KNOWN) == ["ltx-engine"]


def test_whitespace_and_case_are_forgiven():
    assert parse_services("  LTX-Engine , one ", KNOWN) == ["ltx-engine", "one"]


def test_order_is_preserved():
    assert parse_services("one,ltx-engine", KNOWN) == ["one", "ltx-engine"]


def test_duplicates_collapse():
    assert parse_services("one,one,ltx-engine,one", KNOWN) == ["one", "ltx-engine"]


def test_an_unknown_name_is_fatal_and_says_what_exists():
    with pytest.raises(ConfigError, match="joycaption.*Known services: ltx-engine, one"):
        parse_services("joycaption", KNOWN)


def test_one_bad_name_rejects_the_whole_list():
    with pytest.raises(ConfigError):
        parse_services("ltx-engine,nope", KNOWN)


def test_empty_is_fatal():
    with pytest.raises(ConfigError, match="empty"):
        parse_services("", KNOWN)
    with pytest.raises(ConfigError, match="empty"):
        parse_services(None, KNOWN)


def test_ltx_engine_is_three_processes_in_order():
    """ComfyUI must answer before the engine that drives it, and the engine before the
    daemon that submits to it -- start.sh's phases 3, 5, 6."""
    names = [s.name for s in build(["ltx-engine"])]
    assert names == ["comfyui", "ltx-engine-api", "render-daemon"]


def test_every_process_of_a_group_knows_its_group():
    assert {s.group for s in build(["ltx-engine"])} == {"ltx-engine"}


def test_build_instantiates_in_order():
    names = [s.name for s in build(["one", "ltx-engine"], KNOWN)]
    assert names == ["one", "comfyui", "ltx-engine-api", "render-daemon"]


def test_every_registered_name_has_a_summary_and_every_port_a_probe():
    for name in registry.KNOWN:
        for svc in build([name]):
            assert svc.summary, f"{svc.name} has no summary"
            if svc.port:
                assert callable(svc.ready)


def test_the_image_defaults_to_the_render_stack():
    """A pod's environment never sets SERVICES and an empty list is fatal, so the default
    lives in the image. Without it every pod launched after #83 would fail to boot."""
    dockerfile = (pathlib.Path(__file__).parent.parent / "Dockerfile").read_text()
    assert "SERVICES=ltx-engine" in dockerfile


class TestWhatABoxRegistersAs:
    """kinds, render first (wanly-api's `kind` is kinds[0]). The API's claim gates key on it:
    a box that runs the trainer and does not say `trainer` registers, heartbeats, shows green
    and quietly never claims a training job."""

    def test_the_render_stack_alone_is_render(self):
        assert registry.kinds_for(["ltx-engine"]) == ["render"]

    def test_the_3090_is_render_and_trainer_render_first(self):
        assert registry.kinds_for(["face-crop", "lora-trainer", "image-description", "ltx-engine"]) \
            == ["render", "trainer"]

    def test_a_captioner_alone_is_a_service(self):
        assert registry.kinds_for(["image-description", "face-crop"]) == ["service"]

    def test_every_service_the_flag_names_is_known(self):
        for name in ("ltx-engine", "lora-trainer", "image-description", "face-crop"):
            assert name in registry.KNOWN
        assert "joycaption" not in registry.KNOWN, "named for the capability, not the tool"

    def test_the_daemon_is_handed_the_identity(self):
        """The render daemon registers the box; it reads WORKER_KINDS/WORKER_PROVIDES
        (wanly-gpu-daemon#185). Exported before any child starts."""
        from wanly_worker.queue_client import export_identity
        env = export_identity(["ltx-engine", "lora-trainer", "image-description"])
        assert env == {"WORKER_KINDS": "render,trainer",
                       "WORKER_PROVIDES": "ltx-engine,lora-trainer,image-description"}
        import inspect
        from wanly_worker import control
        src = inspect.getsource(control.lifespan)
        assert src.index("os.environ.update(export_identity(names))") < src.index("Supervisor(")


class TestTheTrainerDrainsItsOwnRow:
    """One container per GPU: the render daemon registers the box once and writes the row id
    to WORKER_ID_FILE. The trainer drains THAT row -- nothing is looked up by name, which is
    what broke on 2026-09-08 when the render worker was renamed."""

    def test_the_own_id_comes_from_the_daemons_file(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu
        f = tmp_path / "worker-id"
        monkeypatch.setattr(gpu, "WORKER_ID_FILE", str(f))
        assert gpu.own_worker_id() == ""
        f.write_text("abc-123\n")
        assert gpu.own_worker_id() == "abc-123"

    def test_acquire_drains_the_own_row_and_never_looks_up_by_name(self, tmp_path, monkeypatch):
        import asyncio
        from wanly_worker.services.lora_trainer import gpu
        f = tmp_path / "worker-id"; f.write_text("own-id")
        monkeypatch.setattr(gpu, "WORKER_ID_FILE", str(f))
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api"); monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "vram_used_mib", lambda: 100)
        looked_up = []
        async def find(*a, **k):
            looked_up.append(1)
        monkeypatch.setattr(gpu, "find_render_worker", find)
        calls = []

        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return [{"id": "own-id", "friendly_name": "3090.zero", "kind": "render",
                         "kinds": ["render", "trainer"], "status": "draining"}]

        class C:
            async def get(self, url, **k): return R()
            async def post(self, url, **k): calls.append(url); return R()
        async def fast(_s): pass
        monkeypatch.setattr(gpu.asyncio, "sleep", fast)
        wid = asyncio.run(gpu.acquire(C(), "3090.zero"))
        assert wid == "own-id"
        assert calls == ["http://api/workers/own-id/drain"]
        assert not looked_up

    def test_the_poller_reads_the_daemons_id(self):
        import inspect
        from wanly_worker import control
        src = inspect.getsource(control._worker_id)
        assert "gpu.own_worker_id()" in src
