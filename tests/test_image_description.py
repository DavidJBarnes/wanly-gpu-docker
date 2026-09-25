"""image-description's preflight, and the wire contract wanly-api depends on.

Named for the capability (wanly-gpu-docker#83): the service is image-description, and the
models inside are its business. The dual-model guarantee (wanly-gpu-docker#129, ported from
wanly-services #326/#36): the ACTIVE model wanly-api asks for (a Qwen-class pull tag) plus the
KEPT joycaption:beta-one (the build-only tag) are both present before the box calls itself up.

The contract test is the important one. This whole migration is worth doing precisely because
wanly-api needs no change -- and that is only true while the container answers ollama's API on
ollama's port. If either moves, `image_description_url` silently points at nothing and
captioning fails from a machine that has not changed.
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
    """The Workers page, SERVICES and the env vars say image-description; the models inside
    are not named outside the package. Renaming a model must not rename anything else."""
    from wanly_worker import registry
    assert jc.ImageDescription.name == "image-description"
    assert "image-description" in registry.KNOWN and "joycaption" not in registry.KNOWN


def test_it_binds_all_interfaces():
    """wanly-api reaches this across the network; 127.0.0.1 would work only from inside."""
    env = jc.ImageDescription().env()
    assert env["OLLAMA_HOST"] == "0.0.0.0:11434"


def test_the_model_store_is_pointed_at_the_mount():
    """Not ollama's default. The default is inside the container and is lost on recreate,
    taking multi-GB models with it and re-pulling on next boot."""
    env = jc.ImageDescription().env()
    assert env["OLLAMA_MODELS"].startswith(jc.STORE)


def test_it_does_not_pin_the_llm_library():
    """wanly-services#34 pinned this serve to CPU for the benefit of qwen2.5vl, which does
    not fit the 2070's 8 GB card at ANY size; #36 reverted it, and the lesson is structural:
    a CPU pin belongs with a model decision, never permanently in the service. A pin here
    would silently run whatever GPU model this box serves (joycaption, a 3090's qwen3-vl) on
    the CPU."""
    env = jc.ImageDescription().env()
    assert "OLLAMA_LLM_LIBRARY" not in env


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

    def test_too_little_space_for_a_pull_is_refused_naming_the_pull(self, tmp_path, monkeypatch):
        """The active model is a 21 GB pull on a box that does not have it (a fresh 3090).
        Discovering a full disk 20 GB into that pull is worse than refusing at boot: the
        disk is shared with a renderer."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "PULL_MIN_FREE_GB", 10 ** 6)
        with pytest.raises(PreflightError) as e:
            jc.ImageDescription().preflight()
        assert "free" in str(e.value)
        assert "to pull" in str(e.value)

    def test_too_little_space_for_a_build_is_refused(self, tmp_path, monkeypatch):
        """5.8 GB into a build is a bad time to discover the disk is full."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "MODEL", jc.KEPT_MODEL)   # build-only box: 2070 shape
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 10 ** 9)
        with pytest.raises(PreflightError) as e:
            jc.ImageDescription().preflight()
        assert "free" in str(e.value)
        assert "build the model from scratch" in str(e.value)

    def test_a_machine_that_already_has_everything_is_not_asked_for_model_space(
            self, tmp_path, monkeypatch):
        """The 3090 has the qwen already; 2070.zero has had joycaption for weeks. Demanding
        pull- or build-sized headroom there would refuse a box that works perfectly well."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "MIN_FREE_GB", 0.001)
        monkeypatch.setattr(jc.ImageDescription, "_in_store", lambda self, model: True)
        monkeypatch.setattr(jc.joycaption_model, "missing", lambda store: [])
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 10 ** 9)
        jc.ImageDescription().preflight()      # must not raise

    def test_a_good_store_passes(self, tmp_path, monkeypatch):
        # EVERY size input is pinned, not just MIN_FREE_GB — an empty tmp_path means
        # preflight takes the pull/build branches, whose bars are model-sized, so leaving
        # those real made this test pass or fail on the runner's spare capacity, which is
        # testing the runner. It failed exactly that way once upstream.
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        monkeypatch.setattr(jc, "MIN_FREE_GB", 0.001)
        monkeypatch.setattr(jc, "PULL_MIN_FREE_GB", 0.001)
        monkeypatch.setattr(jc.joycaption_model, "BUILD_HEADROOM_GB", 0.001)
        monkeypatch.setattr(jc.joycaption_model, "total_bytes", lambda: 1)
        (tmp_path / "models").mkdir()
        jc.ImageDescription().preflight()

    def test_in_store_matches_ollamas_manifest_layout(self, tmp_path, monkeypatch):
        """_in_store is the preflight's cheap disk peek; ollama's manifest layout is the
        contract. If the layout moves, this test moves — silently-False would demand
        pull-sized space from a box that already has the model."""
        monkeypatch.setattr(jc, "STORE", str(tmp_path))
        m = (tmp_path / "models" / "manifests" / "registry.ollama.ai" / "library"
             / "qwen3-vl" / "32b-instruct-q4_K_M")
        assert not jc.ImageDescription()._in_store("qwen3-vl:32b-instruct-q4_K_M")
        m.parent.mkdir(parents=True)
        m.write_text("{}")
        assert jc.ImageDescription()._in_store("qwen3-vl:32b-instruct-q4_K_M")


def test_the_default_model_matches_wanly_apis_default():
    """A mismatch is silent and expensive: ollama pulls whatever wanly-api asks for, inside
    a request wanly-api gave its own timeout for, so it times out AND leaves a multi-GB pull
    running invisibly.

    Read out of wanly-api's config rather than restated, so the two cannot drift apart.
    """
    cfg = pathlib.Path(__file__).resolve().parents[2] / "wanly-api" / "app" / "config.py"
    if not cfg.exists():
        pytest.skip("wanly-api is not checked out beside this repo")
    import re
    # `image_description_model: str = Field("qwen2.5vl:7b-q4_K_M", validation_alias=...)` --
    # the first quoted string after the field name, whichever line it lands on.
    m = re.search(r"(?:image_description_model|joycaption_model)\s*:\s*str\s*=\s*(?:Field\(\s*)?[\"']([^\"']+)[\"']",
                  cfg.read_text())
    if not m:
        pytest.fail("image_description_model/joycaption_model not found in wanly-api/app/config.py")
    want = m.group(1)
    assert jc.MODEL == want, f"this image defaults to {jc.MODEL}, wanly-api asks for {want}"


def test_the_kept_model_is_the_buildable_one():
    """joycaption:beta-one stays installed per David's "nothing deleted" rule; it is the only
    model this package can BUILD (model.py), and an env override pointing wanly-api back at
    it via JOYCAPTION_* must find it present. A deployment (the 3090 today) sets
    IMAGE_DESCRIPTION_MODEL to it, which collapses both bars."""
    assert jc.KEPT_MODEL == "joycaption:beta-one"


class TestModelsArePresent:
    """after_ready guarantees BOTH models #326 made load-bearing: the active one (pulled)
    and joycaption (built, wanly-services#3). #1's refuse-a-missing-model stance was
    corrected in #3 for joycaption — the tag is unpullable but the bytes are public — and
    Qwen-class tags have never had that problem: ordinary library tags, so they are pulled."""

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

    def test_both_models_present_provisions_and_pulls_nothing(self, monkeypatch):
        built, pulled = [], []
        monkeypatch.setattr(jc.joycaption_model, "provision",
                            lambda *a, **k: built.append(1))

        async def no_pull(self, client, model):
            pulled.append(model)

        monkeypatch.setattr(jc.ImageDescription, "_pull_model", no_pull)
        self._run(jc.ImageDescription().after_ready(
            self._Client([[jc.MODEL, jc.KEPT_MODEL]])))
        assert not built and not pulled, "re-provisioned a store that already had everything"

    def test_a_missing_active_model_is_pulled(self, monkeypatch):
        pulled = []

        async def fake_pull(self, client, model):
            pulled.append(model)

        monkeypatch.setattr(jc.ImageDescription, "_pull_model", fake_pull)
        # active absent on the look, kept present
        self._run(jc.ImageDescription().after_ready(self._Client([[], [jc.KEPT_MODEL]])))
        assert pulled == [jc.MODEL]

    def test_a_deployment_pointed_at_the_kept_model_pulls_nothing(self, monkeypatch):
        """The 3090's worker.env sets IMAGE_DESCRIPTION_MODEL=joycaption:beta-one. When the
        active model IS the kept one there is nothing to pull — pulling its own tag would
        fail (library namespace) and must not even be attempted."""
        pulled = []

        async def fake_pull(self, client, model):
            pulled.append(model)

        monkeypatch.setattr(jc, "MODEL", jc.KEPT_MODEL)
        monkeypatch.setattr(jc.ImageDescription, "_pull_model", fake_pull)
        self._run(jc.ImageDescription().after_ready(self._Client([[jc.KEPT_MODEL]])))
        assert pulled == []

    def test_a_present_kept_model_builds_nothing(self, monkeypatch):
        called = []
        monkeypatch.setattr(jc.joycaption_model, "provision",
                            lambda *a, **k: called.append(1))
        monkeypatch.setattr(jc.ImageDescription, "_pull_model",
                            lambda self, client, model: _noop())
        self._run(jc.ImageDescription().after_ready(
            self._Client([[jc.MODEL], [jc.KEPT_MODEL]])))
        assert not called, "rebuilt a model that was already there"

    def test_a_missing_kept_model_is_provisioned_then_verified(self, monkeypatch):
        calls = []

        async def fake_provision(store, client, log=None):
            calls.append(store)

        monkeypatch.setattr(jc.joycaption_model, "provision", fake_provision)
        monkeypatch.setattr(jc.ImageDescription, "_pull_model",
                            lambda self, client, model: _noop())
        # active present first look; kept absent, present after the build
        self._run(jc.ImageDescription().after_ready(
            self._Client([[jc.MODEL], [], [jc.MODEL, jc.KEPT_MODEL]])))
        assert calls == [jc.STORE]

    def test_a_build_ollama_does_not_pick_up_is_a_failure(self, monkeypatch):
        """Writing files is not evidence. Being served is."""
        async def fake_provision(store, client, log=None):
            pass

        monkeypatch.setattr(jc.joycaption_model, "provision", fake_provision)
        monkeypatch.setattr(jc.ImageDescription, "_pull_model",
                            lambda self, client, model: _noop())
        monkeypatch.setattr(jc.asyncio, "sleep", _instant)
        monkeypatch.setattr(jc.time, "time", _clock())
        with pytest.raises(RuntimeError, match="still does not list it"):
            self._run(jc.ImageDescription().after_ready(
                self._Client([[jc.MODEL], [], ["other:tag"]])))

    def test_auto_build_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("IMAGE_DESCRIPTION_AUTO_BUILD", "0")
        called = []
        monkeypatch.setattr(jc.joycaption_model, "provision",
                            lambda *a, **k: called.append(1))
        with pytest.raises(RuntimeError, match="AUTO_BUILD"):
            # the active model is waited on first; with AUTO_BUILD=0 the pull refuses first
            self._run(jc.ImageDescription().after_ready(self._Client([[]])))
        assert not called

    def test_a_pull_ollama_does_not_pick_up_is_a_failure(self, monkeypatch):
        """The pull path gets the same proof-over-files rule as the build path."""
        async def fake_exec(*args, **k):
            class P:
                returncode = 0
                async def communicate(self): return (b"exporting layers", None)
                def kill(self): pass
            return P()

        monkeypatch.setattr(jc.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(jc.asyncio, "sleep", _instant)
        monkeypatch.setattr(jc.time, "time", _clock())
        with pytest.raises(RuntimeError, match="does not list it"):
            self._run(jc.ImageDescription().after_ready(self._Client([[], ["other:tag"]])))

    def test_a_failing_pull_fails_the_boot_naming_the_reason(self, monkeypatch):
        async def fake_exec(*args, **k):
            class P:
                returncode = 1
                async def communicate(self): return (b"dial tcp: no route to host", None)
                def kill(self): pass
            return P()

        monkeypatch.setattr(jc.asyncio, "create_subprocess_exec", fake_exec)
        with pytest.raises(RuntimeError, match="no route"):
            self._run(jc.ImageDescription().after_ready(self._Client([[]])))

    def test_the_pull_names_the_server_it_talks_to(self, monkeypatch):
        """A bare `ollama pull` inside the container targets 127.0.0.1:11434. The child needs
        OLLAMA_HOST passed explicitly — env() reaches only the `serve` child — or the pull
        hits the wrong port whenever IMAGE_DESCRIPTION_PORT moves."""
        seen = {}

        async def fake_exec(*args, **k):
            seen["args"] = args
            seen["env"] = k.get("env", {})

            class P:
                returncode = 0
                async def communicate(self): return (b"", None)
                def kill(self): pass
            return P()

        monkeypatch.setattr(jc.asyncio, "create_subprocess_exec", fake_exec)
        # absent on the look, present after the pull (kept model there all along)
        self._run(jc.ImageDescription().after_ready(
            self._Client([[], [jc.MODEL, jc.KEPT_MODEL]])))
        assert seen["args"][:3] == ("ollama", "pull", jc.MODEL)
        assert seen["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"


async def _noop():
    return None


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


# ---------------------------------------------------------------------------------------
# Warming and dropping the model around a mode switch (#131).
#
# The cold load is 88s (measured, qwen3-vl:32b on the 3090). Paid during the switch it is
# expected; paid inside the first caption it reads as a hung request.
# ---------------------------------------------------------------------------------------

import asyncio

from wanly_worker.services.image_description import service as imgsvc


class _Ollama:
    def __init__(self, status=200):
        self.calls = []
        self._status = status

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))

        class _R:
            status_code = self._status
        return _R()


def test_warm_pins_the_model_rather_than_leaving_it_on_a_timer():
    """-1 is "hold it until told otherwise". Anything else lapses -- which is the one thing
    "keep it loaded until I flip back" rules out."""
    o = _Ollama()
    assert asyncio.run(imgsvc.warm(o)) is True
    _, body = o.calls[0]
    assert body["keep_alive"] == -1


def test_warm_loads_without_generating():
    """An empty prompt is the documented way to ask ollama to load a model and return as
    soon as it is resident."""
    o = _Ollama()
    asyncio.run(imgsvc.warm(o))
    _, body = o.calls[0]
    assert body["prompt"] == ""
    assert body["model"] == imgsvc.MODEL


def test_release_drops_it_now():
    o = _Ollama()
    asyncio.run(imgsvc.release(o))
    _, body = o.calls[0]
    assert body["keep_alive"] == 0


def test_a_captioner_that_will_not_warm_is_not_fatal():
    """It still answers requests; the first one just pays the load, which is the old
    behaviour. Failing the mode switch over a warm-up would be worse than the warm-up."""
    class _Dead:
        async def post(self, *a, **k):
            raise OSError("connection refused")

    assert asyncio.run(imgsvc.warm(_Dead())) is False


def test_there_is_no_periodic_pin():
    """A 5s re-assert made captions WORSE by a wide margin. ollama runs NUM_PARALLEL=1, so
    every pin took a turn in the single slot and disturbed the model between captions:
    25-35s climbed to 42s, 54s, 64s, 86s and then a 500 -- against the 6-10s page-cache
    reload it was there to avoid. Warming on the flip is kept; re-asserting forever is not."""
    assert not hasattr(imgsvc, "PIN_INTERVAL_S")