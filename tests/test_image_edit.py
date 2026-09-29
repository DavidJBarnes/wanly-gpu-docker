"""image-edit: Qwen-Image-Edit full mode and edit mode (wanly-console#548).

What is pinned here is everything decided BEFORE the card: the graph (topology fixed, values
only), the framing pins, the head-angle words and their direction, which mode runs the service,
that the box is put back when the service cannot start, and that it hands the card back when
edits stop. ComfyUI, torch and insightface are not installed in CI; the API is driven against a
fake graph runner.
"""
import asyncio
import base64
import io
import os
import struct
import subprocess
import json

import pytest

from wanly_worker import registry
from wanly_worker.registry import ConfigError, select_mode
from wanly_worker.services.image_edit import app as edit_app
from wanly_worker.services.image_edit import graph

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THE_3090 = ["ltx-engine", "lora-trainer", "image-description", "face-crop", "image-edit"]


# ------------------------------------------------------------------------------ modes


class TestEditMode:
    def test_render_mode_never_runs_image_edit(self):
        """20 GB of Qwen beside a render that holds ~23 of 24 GB is the collision modes exist
        to prevent. It is on the SERVICES line as a capability, not as something to run."""
        assert "image-edit" not in select_mode(THE_3090, None)
        assert "image-edit" not in select_mode(THE_3090, "ltx-engine")

    def test_caption_mode_never_runs_image_edit(self):
        assert select_mode(THE_3090, "caption") == ["image-description", "face-crop"]

    def test_edit_mode_is_image_edit_plus_the_small_services(self):
        """No render stack, no trainer (both claim work), and no captioner: a caption landing
        mid-edit would load a 20 GB vision model onto a card Qwen already holds."""
        assert select_mode(THE_3090, "edit") == ["face-crop", "image-edit"]
        assert select_mode(THE_3090 + ["face-edit"], "edit") == \
            ["face-crop", "image-edit", "face-edit"]

    @pytest.mark.parametrize("alias", ["image-edit", "full-edit", " EDIT "])
    def test_aliases(self, alias):
        assert registry.canonical_mode(alias) == "edit"
        assert select_mode(THE_3090, alias) == ["face-crop", "image-edit"]

    def test_edit_mode_on_a_box_without_image_edit_says_so(self):
        with pytest.raises(ConfigError) as e:
            select_mode(["ltx-engine", "image-description", "face-crop"], "edit")
        assert "image-edit" in str(e.value)

    def test_an_edit_only_box_runs_it_without_a_mode(self):
        """A box equipped only with mode-bound services has nothing else to run."""
        assert select_mode(["image-edit"], None) == ["image-edit"]

    def test_it_is_a_known_service_of_two_processes(self):
        procs = registry.build(["image-edit"])
        assert [p.name for p in procs] == ["image-edit-comfyui", "image-edit"]
        assert {p.group for p in procs} == {"image-edit"}

    def test_it_claims_nothing(self):
        assert "image-edit" not in registry.KIND_BY_SERVICE
        assert registry.kinds_for(["image-edit"]) == ["service"]


# ------------------------------------------------------------------------------ graph


class TestFraming:
    def test_the_latent_is_the_sources_own_size(self):
        """keyframe-server's pin 1: a latent the model chose re-frames by construction."""
        assert graph.latent_size(1024, 1024) == (1024, 1024)
        assert graph.latent_size(1072, 720) == (1072, 720)

    def test_multiples_of_16(self):
        assert graph.latent_size(693, 732) == (688, 720)

    def test_a_large_source_is_capped_keeping_its_aspect(self):
        w, h = graph.latent_size(3000, 4000, max_mp=1.2)
        assert w * h <= 1.2e6 and w % 16 == 0 and h % 16 == 0
        assert abs(w / h - 0.75) < 0.02

    def test_a_small_source_is_never_scaled_up(self):
        assert graph.latent_size(512, 768) == (512, 768)

    def test_an_instruction_gets_the_pin_once(self):
        p = graph.instruction_prompt("Replace her sweater with a purple tank top")
        assert p.endswith(graph.FRAMING_PIN)
        assert graph.instruction_prompt(p).count(graph.FRAMING_PIN) == 1

    def test_an_empty_instruction_is_nothing_to_apply(self):
        with pytest.raises(ValueError, match="nothing to apply"):
            graph.instruction_prompt("   ")


class TestAngleWords:
    def test_negative_yaw_is_the_left_edge_of_the_image(self):
        """Matches phase 1: LivePortrait's rotate_yaw < 0 points the nose image-left
        (checked on sel_008), so "left" means the same on both engines."""
        assert "left edge of the image" in graph.angle_prompt(-45, 0)
        assert "right edge of the image" in graph.angle_prompt(45, 0)

    def test_the_bands(self):
        assert "three-quarter" in graph.angle_prompt(-45, 0)
        assert "full side profile" in graph.angle_prompt(90, 0)
        assert "full side profile" in graph.angle_prompt(-75, 0)
        assert "three-quarter" not in graph.angle_prompt(-25, 0)

    def test_pitch(self):
        assert "looks up" in graph.angle_prompt(0, 25)
        assert "looks down" in graph.angle_prompt(0, -25)

    def test_combined_and_pinned(self):
        p = graph.angle_prompt(-45, 20)
        assert "three-quarter" in p and "looks up" in p
        assert p.endswith(graph.FRAMING_PIN)

    def test_under_five_degrees_is_nothing(self):
        with pytest.raises(ValueError, match="nothing to apply"):
            graph.angle_prompt(3, -2)

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            graph.angle_prompt(120, 0)
        with pytest.raises(ValueError, match="out of range"):
            graph.angle_prompt(0, 60)


class TestTheGraph:
    def test_keyframe_servers_topology(self):
        wf = graph.build_workflow("src.png", 1024, 768, "turn", seed=7)
        assert {k: v["class_type"] for k, v in wf.items()} == {
            "1": "CheckpointLoaderSimple", "9": "EmptyLatentImage", "101": "LoadImage",
            "4": "TextEncodeQwenImageEditPlus", "3": "TextEncodeQwenImageEditPlus",
            "2": "KSampler", "5": "VAEDecode", "6": "SaveImage"}
        k = wf["2"]["inputs"]
        assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["seed"]) == \
            (4, 1.0, "euler_ancestral", "beta", 7)
        assert wf["3"]["inputs"]["image1"] == ["101", 0]
        assert wf["9"]["inputs"] == {"width": 1024, "height": 768, "batch_size": 1}
        assert wf["1"]["inputs"]["ckpt_name"] == "Qwen-Rapid-AIO-NSFW-v23.safetensors"

    def test_the_lora_is_spliced_in_only_with_a_strength(self):
        assert "20" not in graph.build_workflow("s.png", 64, 64, "p", 1, lora="x", lora_strength=0)
        wf = graph.build_workflow("s.png", 64, 64, "p", 1, lora="angles.safetensors",
                                  lora_strength=0.9)
        assert wf["20"]["class_type"] == "LoraLoaderModelOnly"
        assert wf["20"]["inputs"]["model"] == ["1", 0]
        assert wf["2"]["inputs"]["model"] == ["20", 0]

    def test_a_partial_denoise_starts_from_the_source(self):
        wf = graph.build_workflow("s.png", 64, 64, "p", 1, denoise=0.4)
        assert "9" not in wf
        assert wf["110"]["inputs"]["pixels"] == ["101", 0]
        assert wf["2"]["inputs"]["latent_image"] == ["110", 0]
        assert wf["2"]["inputs"]["denoise"] == 0.4


# -------------------------------------------------------------------------------- API


def _png(w=96, h=64, color=(200, 150, 120)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    ran = []

    async def fake_graph(client, wf):
        ran.append(wf)
        w, h = wf["9"]["inputs"]["width"], wf["9"]["inputs"]["height"]
        return _png(w, h, (10, 20, 30))

    monkeypatch.setattr(edit_app, "_run_graph", fake_graph)
    monkeypatch.setattr(edit_app, "WORK_DIR", str(tmp_path))
    monkeypatch.setattr(edit_app, "_vram_used_mib", lambda: 20480)
    monkeypatch.setattr(edit_app.identity, "score", lambda a, b: {"aura": 0.71, "reason": None})
    return TestClient(edit_app.app), ran


class TestTheAPI:
    def test_an_angle_edit(self, api):
        client, ran = api
        r = client.post("/edit", json={"image": base64.b64encode(_png(100, 70)).decode(),
                                       "angle": {"yaw": -90, "pitch": 0}, "seed": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["width"], body["height"]) == (96, 64), "latent is the source's, /16"
        assert "full side profile" in body["prompt"] and "left edge" in body["prompt"]
        assert body["identity"] == {"aura": 0.71, "reason": None}
        assert body["seed"] == 5 and body["vram_peak_mib"] == 20480
        assert base64.b64decode(body["preview"])[:2] == b"\xff\xd8", "a JPEG preview"
        assert ran[0]["3"]["inputs"]["prompt"] == body["prompt"]
        assert "20" not in ran[0], "the recipe is prompt-only: no LoRA node"

    def test_an_instruction_edit(self, api):
        client, ran = api
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "instruction": "make the sweater red"})
        assert r.status_code == 200
        assert r.json()["prompt"].endswith(graph.FRAMING_PIN)

    @pytest.mark.parametrize("extra", [{}, {"instruction": "x", "angle": {"yaw": 45}},
                                       {"angle": {"yaw": 2}}])
    def test_nothing_or_both_is_a_422(self, api, extra):
        client, _ = api
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(), **extra})
        assert r.status_code == 422

    def test_out_of_range_is_a_422(self, api):
        client, _ = api
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "angle": {"yaw": 135}})
        assert r.status_code == 422

    def test_garbage_is_a_400(self, api):
        client, _ = api
        r = client.post("/edit", json={"image": "bm90IGFuIGltYWdl", "angle": {"yaw": 45}})
        assert r.status_code == 400

    def test_health_reports_idle_time_for_the_hand_back(self, api, monkeypatch):
        client, _ = api
        monkeypatch.setattr(edit_app, "_state", {"last_edit_at": None, "last": None,
                                                 "model_loaded": False, "edits": 0})
        body = client.get("/health").json()
        assert body["busy"] is False and body["idle_s"] >= 0 and body["edits"] == 0


class TestIdentity:
    def test_a_missing_model_is_a_reason_not_an_error(self, monkeypatch, tmp_path):
        from wanly_worker.services.image_edit import identity
        import numpy as np
        monkeypatch.setattr(identity, "AURA_PATH", str(tmp_path / "nope.onnx"))
        monkeypatch.setattr(identity, "_models", None)
        z = np.zeros((8, 8, 3), dtype=np.uint8)
        out = identity.score(z, z)
        assert out["aura"] is None and "AuraFace model not found" in out["reason"]

    def test_cosine(self):
        from wanly_worker.services.image_edit.identity import cosine
        assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine([1, 0], [0, 2]) == pytest.approx(0.0)


# --------------------------------------------------------------------- the supervisor


def _svc(log, name, group, raises=False):
    from wanly_worker.service import PreflightError, Service

    class S(Service):
        port = 1

        def preflight(self):
            log.append(f"preflight:{name}")
            if raises:
                raise PreflightError("no mount")

        def command(self):
            log.append(f"start:{name}")
            return ["sleep", "30"]

        async def ready(self, client):
            return True

    s = S()
    s.name, s.group, s.summary = name, group, name
    return s


class TestModeOnlyServicesInTheSupervisor:
    def test_boot_leaves_another_modes_service_stopped_and_unpreflighted(self):
        """Its preflight checks a 28 GB mount; a render boot must not be refused over it."""
        from wanly_worker.supervisor import Supervisor
        log = []
        sup = Supervisor([_svc(log, "engine", "ltx-engine"), _svc(log, "qwen", "image-edit")])

        async def go():
            await sup.start(client=None, active=["ltx-engine"])
            st = {s.service.name: s for s in sup.states}
            assert st["qwen"].stopped and st["qwen"].proc is None
            assert "preflight:qwen" not in log
            snap = {r["name"]: r for r in sup.snapshot()}
            assert snap["qwen"]["stopped"] is True
            await sup.stop()

        asyncio.run(go())

    def test_the_switch_preflights_it_then_starts_it(self):
        from wanly_worker.supervisor import Supervisor
        log = []
        sup = Supervisor([_svc(log, "engine", "ltx-engine"), _svc(log, "qwen", "image-edit")])

        async def go():
            await sup.start(client=None, active=["ltx-engine"])
            log.clear()
            await sup.apply(["image-edit"], client=None)
            assert log == ["preflight:qwen", "start:qwen"]
            await sup.stop()

        asyncio.run(go())

    def test_a_failed_preflight_fails_the_switch(self):
        from wanly_worker.service import PreflightError
        from wanly_worker.supervisor import Supervisor
        log = []
        sup = Supervisor([_svc(log, "engine", "ltx-engine"),
                          _svc(log, "qwen", "image-edit", raises=True)])

        async def go():
            await sup.start(client=None, active=["ltx-engine"])
            with pytest.raises(PreflightError, match="qwen"):
                await sup.apply(["image-edit"], client=None)
            await sup.stop()

        asyncio.run(go())


# ----------------------------------------------------------------------- control plane


from wanly_worker import control  # noqa: E402


class _Sup:
    def __init__(self, fail_on=None):
        self.applied, self.fail_on = [], fail_on

    def snapshot(self):
        return []

    async def apply(self, groups, client):
        self.applied.append(list(groups))
        if self.fail_on and self.fail_on in groups:
            raise RuntimeError(f"{self.fail_on}: the Qwen tree is not mounted")


@pytest.fixture
def box(monkeypatch):
    def make(fail_on=None):
        s = _Sup(fail_on)
        monkeypatch.setattr(control, "_sup", s)
        monkeypatch.setattr(control, "_client", object())
        monkeypatch.setattr(control, "_equipped", list(THE_3090))
        monkeypatch.setattr(control, "_mode", "ltx-engine")
        monkeypatch.setattr(control, "_mode_lock", asyncio.Lock())
        monkeypatch.setattr(control, "_pending", None)
        monkeypatch.setattr(control, "_mode_error", None)
        monkeypatch.setattr(control, "_queue", None)
        return s
    return make


def _settle(mode):
    async def go():
        await control.set_mode(control.ModeRequest(mode=mode))
        if control._mode_task is not None:
            await control._mode_task
        if control._edit_watch is not None:
            control._edit_watch.cancel()
        return control._mode, control._mode_error
    return asyncio.run(go())


class TestEditModeSwitch:
    def test_entering_edit_mode(self, box):
        s = box()
        mode, err = _settle("edit")
        assert (mode, err) == ("edit", None)
        assert s.applied == [["face-crop", "image-edit"]]
        assert control._return_from_edit == "ltx-engine"

    def test_a_failed_start_puts_the_render_stack_back(self, box):
        """Entering edit mode stops the render stack first. If image-edit then cannot start,
        leaving the box there would park every queued render behind nothing."""
        s = box(fail_on="image-edit")
        mode, err = _settle("edit")
        assert mode == "ltx-engine"
        assert "not mounted" in err
        assert s.applied[-1] == select_mode(THE_3090, "ltx-engine")

    def test_idle_hands_the_card_back(self, box, monkeypatch):
        s = box()
        monkeypatch.setattr(control, "EDIT_IDLE_RETURN_S", 4.0)
        monkeypatch.setattr(control, "_return_from_edit", "ltx-engine")

        async def idle():
            return 999.0

        async def fast_sleep(_):
            return None

        monkeypatch.setattr(control, "_edit_idle_s", idle)

        async def go():
            control._mode = "edit"
            real_sleep = asyncio.sleep
            monkeypatch.setattr(control.asyncio, "sleep", fast_sleep)
            await control._watch_edit_idle()
            monkeypatch.setattr(control.asyncio, "sleep", real_sleep)
            await control._mode_task
            return control._mode

        assert asyncio.run(go()) == "ltx-engine"
        assert s.applied[-1] == select_mode(THE_3090, "ltx-engine")

    def test_not_idle_long_enough_stays(self, box, monkeypatch):
        box()
        monkeypatch.setattr(control, "EDIT_IDLE_RETURN_S", 600.0)
        calls = []

        async def idle():
            calls.append(1)
            if len(calls) >= 3:
                control._mode = "ltx-engine"      # someone else switched; the watcher ends
            return 10.0

        async def fast_sleep(_):
            return None

        monkeypatch.setattr(control, "_edit_idle_s", idle)

        async def go():
            control._mode = "edit"
            monkeypatch.setattr(control.asyncio, "sleep", fast_sleep)
            await control._watch_edit_idle()

        asyncio.run(go())
        assert control._pending is None


# ------------------------------------------------------------------ download_models.sh


def _safetensors(path, truncate=False):
    header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
    data = struct.pack("<Q", len(header)) + header + (b"\0" * (8 if truncate else 16))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _check(tmp_path, truncate=False):
    models = tmp_path / "qwen"
    _safetensors(models / "v23" / "Qwen-Rapid-AIO-NSFW-v23.safetensors", truncate)
    aura = tmp_path / "if" / "models" / "auraface" / "glintr100.onnx"
    aura.parent.mkdir(parents=True)
    aura.write_bytes(b"onnx")
    env = {**os.environ, "IMAGE_EDIT_MODELS_DIR": str(models), "INSIGHTFACE_ROOT": str(tmp_path / "if")}
    return subprocess.run(["bash", os.path.join(REPO, "download_models.sh"), "--image-edit"],
                          capture_output=True, text=True, env=env)


class TestModelCheck:
    def test_a_complete_set_passes(self, tmp_path):
        r = _check(tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "1 file(s) OK" in r.stdout and "models OK" in r.stdout
        assert "diffusion_models" not in r.stdout, "the LTX tree is not this target's"

    def test_a_truncated_checkpoint_fails_the_check(self, tmp_path):
        r = _check(tmp_path, truncate=True)
        assert r.returncode != 0
        assert "TRUNCATED Qwen-Rapid-AIO-NSFW-v23.safetensors" in r.stdout
