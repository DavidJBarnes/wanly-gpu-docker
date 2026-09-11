"""image-description's preflight, and the wire contract wanly-api depends on.

Named for the capability (wanly-gpu-docker#83): the service is image-description, and JoyCaption
on ollama is what it happens to run today.

The contract test is the important one. This whole migration is worth doing precisely because
wanly-api needs no change -- and that is only true while the container answers ollama's API on
ollama's port. If either moves, `joycaption_url` silently points at nothing and captioning
fails from a machine that has not changed.
"""
import os
import pathlib

import pytest

from wanly_worker.service import PreflightError
from wanly_worker.services.image_description import service as jc


def test_it_serves_ollamas_port():
    """wanly-api's image_description_url points at :11434 and is not changing as part of this."""
    assert jc.ImageDescription.port == 11434


def test_it_is_named_for_the_capability_not_the_tool():
    """The Workers page, SERVICES and the env vars say image-description; JoyCaption is the
    model inside. Renaming the model must not rename anything outside the package."""
    from wanly_worker import registry
    assert jc.ImageDescription.name == "image-description"
    assert "image-description" in registry.KNOWN and "joycaption" not in registry.KNOWN


def test_it_binds_all_interfaces():
    """wanly-api reaches this across the network; 127.0.0.1 would work only from inside."""
    env = jc.ImageDescription().env()
    assert env["OLLAMA_HOST"] == "0.0.0.0:11434"


def test_the_model_store_is_pointed_at_the_mount():
    """Not ollama's default. The default is inside the container and is lost on recreate,
    taking 5.8 GB with it and re-pulling on next boot."""
    env = jc.ImageDescription().env()
    assert env["OLLAMA_MODELS"].startswith(jc.STORE)


def test_it_starts_ollama():
    assert jc.ImageDescription().command() == ["ollama", "serve"]


class TestPreflight:
    def test_a_missing_store_is_refused_before_anything_starts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jc, "STORE", str(tmp_path / "nope"))
        with pytest.raises(PreflightError) as e:
            jc.ImageDescription().preflight()
        # The message has to name the mount, or the reader goes looking at ollama.
        assert "bind mount" in str(e.value)

    def test_a_read_only_store_is_refused(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        store.mkdir()
        store.chmod(0o555)
        monkeypatch.setattr(jc, "STORE", str(store))
        try:
            if os.access(str(store), os.W_OK):
                pytest.skip("running as root; the mode bits do not bind")
            with pytest.raises(PreflightError) as e:
                jc.ImageDescription().preflight()
            assert "not writable" in str(e.value)
        finally:
            store.chmod(0o755)

    def test_too_little_space_is_refused(self, tmp_path, monkeypatch):
        """5.8 GB into a build is a bad time to discover the disk is full."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 10 ** 9)
        with pytest.raises(PreflightError) as e:
            jc.ImageDescription().preflight()
        assert "free" in str(e.value)
        assert "build the model from scratch" in str(e.value)

    def test_a_machine_that_already_has_the_model_is_not_asked_for_build_space(
            self, tmp_path, monkeypatch):
        """2070.zero has had the model for weeks. Demanding build-sized headroom there would
        refuse a box that works perfectly well."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "MIN_FREE_GB", 0.001)
        monkeypatch.setattr(jc.joycaption_model, "missing", lambda store: [])
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 10 ** 9)
        jc.ImageDescription().preflight()      # must not raise

    def test_a_good_store_passes(self, tmp_path, monkeypatch):
        # EVERY size input is pinned, not just MIN_FREE_GB. An empty tmp_path means preflight
        # takes the build branch, whose bar is total_bytes + BUILD_HEADROOM_GB -- so leaving
        # those real made this test pass or fail on the runner's spare capacity, which is
        # testing the runner. It failed exactly that way once.
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "MIN_FREE_GB", 0.001)
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 0.001)
        monkeypatch.setattr(jc.joycaption_model, "total_bytes", lambda: 1)
        (tmp_path / "models").mkdir()
        jc.ImageDescription().preflight()


def test_the_default_model_matches_wanly_apis_default():
    """A mismatch is silent and expensive: ollama pulls whatever wanly-api asks for, inside a
    request wanly-api gave 60 seconds (joycaption_timeout_s), so it times out AND leaves a
    multi-GB pull running invisibly.

    Read out of wanly-api's config rather than restated, so the two cannot drift apart.
    """
    cfg = pathlib.Path(__file__).resolve().parents[2] / "wanly-api" / "app" / "config.py"
    if not cfg.exists():
        pytest.skip("wanly-api is not checked out beside this repo")
    import re
    # `image_description_model: str = Field("joycaption:beta-one", validation_alias=...)` --
    # the first quoted string after the field name, whichever line it lands on.
    m = re.search(r"(?:image_description_model|joycaption_model)\s*:\s*str\s*=\s*(?:Field\(\s*)?[\"']([^\"']+)[\"']",
                  cfg.read_text())
    if not m:
        pytest.fail("image_description_model/joycaption_model not found in wanly-api/app/config.py")
    want = m.group(1)
    assert jc.MODEL == want, f"this image defaults to {jc.MODEL}, wanly-api asks for {want}"


class TestAMissingModelIsBuilt:
    """#1 refused a missing model; #3 builds it. The correction matters because the reason #1
    gave was wrong: `ollama pull joycaption:beta-one` fails only because the tag lives in the
    library namespace, and the bytes underneath are a public HF repo with identical digests.
    """

    class _Client:
        def __init__(self, tags_over_time):
            self._tags = list(tags_over_time)
            self.gets = 0

        async def get(self, url, timeout=None):
            self.gets += 1
            tags = self._tags[min(self.gets - 1, len(self._tags) - 1)]

            class R:
                def json(self_inner): return {"models": [{"name": t} for t in tags]}
            return R()

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_a_present_model_builds_nothing(self, monkeypatch):
        called = []
        monkeypatch.setattr(jc.joycaption_model, "provision",
                            lambda *a, **k: called.append(1))
        self._run(jc.ImageDescription().after_ready(self._Client([[jc.MODEL]])))
        assert not called, "rebuilt a model that was already there"

    def test_a_missing_model_is_provisioned_then_verified(self, monkeypatch):
        calls = []

        async def fake_provision(store, client, log=None):
            calls.append(store)

        monkeypatch.setattr(jc.joycaption_model, "provision", fake_provision)
        # absent on the first look, present after the build
        self._run(jc.ImageDescription().after_ready(self._Client([[], [jc.MODEL]])))
        assert calls == [jc.STORE]

    def test_a_build_ollama_does_not_pick_up_is_a_failure(self, monkeypatch):
        """Writing files is not evidence. Being served is."""
        async def fake_provision(store, client, log=None):
            pass

        monkeypatch.setattr(jc.joycaption_model, "provision", fake_provision)
        monkeypatch.setattr(jc.asyncio, "sleep", _instant)
        monkeypatch.setattr(jc.time, "time", _clock())
        with pytest.raises(RuntimeError, match="still does not list it"):
            self._run(jc.ImageDescription().after_ready(self._Client([[], ["other:tag"]])))

    def test_auto_build_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("IMAGE_DESCRIPTION_AUTO_BUILD", "0")
        called = []
        monkeypatch.setattr(jc.joycaption_model, "provision",
                            lambda *a, **k: called.append(1))
        with pytest.raises(RuntimeError, match="AUTO_BUILD"):
            self._run(jc.ImageDescription().after_ready(self._Client([[]])))
        assert not called


async def _instant(_):
    return None


def _clock():
    """A clock that jumps past the refresh deadline, so the retry loop is exercised without
    the test taking 30 seconds to prove it."""
    state = {"t": 0.0}

    def now():
        state["t"] += 20.0
        return state["t"]
    return now
