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
