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
        assert p.endswith(graph.EDIT_PIN)
        assert graph.instruction_prompt(p).count(graph.EDIT_PIN) == 1
        # An instruction that already carries the angle pin is left alone too.
        pinned = f"Make the sky pink. {graph.FRAMING_PIN}"
        assert graph.instruction_prompt(pinned) == pinned

    def test_the_edit_pin_does_not_forbid_the_edit(self):
        """console#569: free text is how expressions are asked for now. FRAMING_PIN's "the same
        facial expression" appended to "make her smile" asked for the opposite."""
        assert "facial expression" not in graph.EDIT_PIN
        assert "facial expression" in graph.FRAMING_PIN
        assert graph.EDIT_PIN.endswith("Do not zoom out.")

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
    def test_the_official_2511_topology(self):
        """#574: three Comfy-Org files, the template's settings -- not v23's 4 steps at cfg 1."""
        wf = graph.build_workflow("src.png", 1024, 768, "turn", seed=7)
        assert {k: v["class_type"] for k, v in wf.items()} == {
            "1": "UNETLoader", "7": "CLIPLoader", "8": "VAELoader",
            "9": "EmptySD3LatentImage", "101": "LoadImage",
            "4": "TextEncodeQwenImageEditPlus", "3": "TextEncodeQwenImageEditPlus",
            "31": "FluxKontextMultiReferenceLatentMethod",
            "41": "FluxKontextMultiReferenceLatentMethod",
            "10": "ModelSamplingAuraFlow", "11": "CFGNorm",
            "2": "KSampler", "5": "VAEDecode", "6": "SaveImage"}
        k = wf["2"]["inputs"]
        assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["seed"]) == \
            (40, 4.0, "euler", "simple", 7)
        assert k["model"] == ["11", 0]
        assert (k["positive"], k["negative"]) == (["31", 0], ["41", 0])
        assert wf["10"]["inputs"]["shift"] == 3.1 and wf["11"]["inputs"]["strength"] == 1.0
        assert wf["31"]["inputs"]["reference_latents_method"] == "index_timestep_zero"
        assert wf["3"]["inputs"]["image1"] == ["101", 0]
        assert wf["4"]["inputs"]["image1"] == ["101", 0], "the negative sees the source too"
        assert wf["4"]["inputs"]["prompt"] == ""
        assert wf["9"]["inputs"] == {"width": 1024, "height": 768, "batch_size": 1}
        assert wf["1"]["inputs"]["unet_name"] == "qwen_image_edit_2511_fp8mixed.safetensors"
        assert wf["7"]["inputs"] == {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                     "type": "qwen_image", "device": "default"}
        assert wf["8"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
        assert not any("Checkpoint" in v["class_type"] for v in wf.values()), "no v23 AIO"

    def test_the_lora_is_spliced_in_only_with_a_strength(self):
        assert "20" not in graph.build_workflow("s.png", 64, 64, "p", 1, lora="x", lora_strength=0)
        wf = graph.build_workflow("s.png", 64, 64, "p", 1, lora="angles.safetensors",
                                  lora_strength=0.9)
        assert wf["20"]["class_type"] == "LoraLoaderModelOnly"
        assert wf["20"]["inputs"]["model"] == ["1", 0]
        assert wf["10"]["inputs"]["model"] == ["20", 0]

    def test_a_partial_denoise_starts_from_the_source(self):
        wf = graph.build_workflow("s.png", 64, 64, "p", 1, denoise=0.4)
        assert "9" not in wf
        assert wf["110"]["inputs"]["pixels"] == ["101", 0]
        assert wf["110"]["inputs"]["vae"] == ["8", 0]
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
        assert r.json()["prompt"].endswith(graph.EDIT_PIN)

    def test_an_expression_edit(self, api):
        client, ran = api
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "expression": "big_laugh"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "laugh" in body["prompt"] and body["prompt"].endswith(graph.EDIT_PIN)
        assert body["expression"] == "big_laugh"
        assert body["face_box"] is None and body["crop"] is None

    def test_an_angle_and_an_expression_together(self, api):
        client, _ = api
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "angle": {"yaw": -45}, "expression": "smile"})
        assert r.status_code == 200, r.text
        p = r.json()["prompt"]
        assert "three-quarter" in p and "smile" in p and p.endswith(graph.EDIT_PIN)

    @pytest.mark.parametrize("extra", [{}, {"angle": {"yaw": 2}}, {"expression": "wink"},
                                       {"instruction": "   "}])
    def test_nothing_or_nonsense_is_a_422(self, api, extra):
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
                                                 "model_loaded": False, "edits": 0,
                                                 "waiting": None})
        body = client.get("/health").json()
        assert body["busy"] is False and body["idle_s"] >= 0 and body["edits"] == 0


class TestExpressions:
    """console#569: the Edit dialog's expression presets are Qwen instructions now."""

    def test_every_preset_the_api_offers_has_words(self):
        for name in ("smile", "big_laugh", "surprised", "eyes_closed", "sad", "angry",
                     "serious", "speaking", "look_left", "look_right", "look_up", "look_down"):
            assert graph.EXPRESSIONS[name][1].endswith(".")

    def test_gaze_is_image_space_like_the_angles(self):
        assert "left edge of the image" in graph.expression_words("look_left")
        assert "right edge of the image" in graph.expression_words("look_right")
        assert "move only the eyes" in graph.expression_words("look_up")

    def test_an_unknown_expression(self):
        with pytest.raises(ValueError, match="unknown expression"):
            graph.compose_prompt(expression="wink")

    def test_an_angle_alone_is_the_measured_recipe_verbatim(self):
        """#548's 36 renders were of angle_prompt; composing must not change a word of it."""
        for yaw, pitch in ((-90, 0), (45, 0), (0, 30), (-45, -20)):
            assert graph.compose_prompt(yaw, pitch) == graph.angle_prompt(yaw, pitch)

    def test_an_expression_changes_the_pin(self):
        p = graph.compose_prompt(expression="smile")
        assert p.endswith(graph.EDIT_PIN) and graph.FRAMING_PIN not in p

    def test_all_three_compose_in_order(self):
        p = graph.compose_prompt(-90, 0, "smile", "make the sweater red")
        assert p.index("profile") < p.index("smile") < p.index("sweater")
        assert p.count(graph.EDIT_PIN) == 1

    def test_the_angle_range_still_holds_with_an_expression(self):
        with pytest.raises(ValueError, match="out of range"):
            graph.compose_prompt(120, 0, "smile")


class TestFaceCrop:
    """console#569: one face of several is edited by crop, edit, feathered paste."""

    def test_the_crop_is_generous_and_clamped(self):
        from wanly_worker.services.image_edit import crop
        l, t, r, b = crop.crop_box([400, 300, 500, 420], (2000, 1500))
        assert l < 400 - 100 and r > 500 + 100, "room to turn the head"
        assert b - 420 > 300 - t, "more below (neck, shoulders) than above"
        assert crop.crop_box([10, 10, 110, 130], (400, 300)) [:2] == (0, 0)
        assert crop.crop_box([300, 150, 390, 280], (400, 300))[2:] == (400, 300)

    def test_a_neighbour_is_kept_out(self):
        from wanly_worker.services.image_edit import crop
        me, left_n, right_n = [800, 300, 900, 420], [560, 300, 660, 420], [1060, 300, 1160, 420]
        l, t, r, b = crop.crop_box(me, (2000, 1500), [left_n, right_n])
        assert l >= 660 and r <= 1060
        assert crop.neighbours_inside((l, t, r, b), [left_n, right_n]) == 0

    def test_a_neighbour_too_close_still_leaves_the_face_its_margin(self):
        from wanly_worker.services.image_edit import crop
        me, n = [800, 300, 900, 420], [880, 300, 980, 420]
        l, t, r, b = crop.crop_box(me, (2000, 1500), [n])
        assert r >= 900 + crop.MIN_PAD * 100 - 1
        assert crop.neighbours_inside((l, t, r, b), [n]) == 1, "reported, not hidden"

    def test_small_crops_are_scaled_up_for_the_model(self):
        from wanly_worker.services.image_edit import crop
        assert crop.work_size(300, 450) == (768, 1152)
        assert crop.work_size(900, 1200) == (900, 1200)

    def test_the_paste_leaves_everything_outside_the_crop_alone(self):
        import numpy as np
        from wanly_worker.services.image_edit import crop
        rng = np.random.default_rng(0)
        src = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        region = (100, 50, 300, 250)
        edited = np.full((200, 200, 3), 7, dtype=np.uint8)
        out = crop.paste(src, edited, region)
        mask = np.ones(src.shape[:2], bool)
        mask[50:250, 100:300] = False
        assert (out[mask] == src[mask]).all(), "byte for byte outside the crop"
        assert (out[150, 200] == 7).all(), "the edit inside"
        # Feathered: the crop's outermost column is mostly source.
        assert abs(int(out[150, 100, 0]) - int(src[150, 100, 0])) <= \
            abs(int(src[150, 100, 0]) - 7) * 0.2 + 1

    def test_an_image_edge_gets_no_feather(self):
        import numpy as np
        from wanly_worker.services.image_edit import crop
        m = crop.feather_mask((0, 0, 100, 100), (400, 300))
        assert m[0, 0] == 1.0 and m[50, 50] == 1.0 and m[99, 99] < 0.2


@pytest.fixture
def faces_api(api, monkeypatch):
    client, ran = api
    boxes = [[10.0, 10.0, 40.0, 45.0], [60.0, 8.0, 90.0, 44.0]]
    monkeypatch.setattr(edit_app.identity, "face_boxes", lambda rgb: [list(b) for b in boxes])

    async def fake_graph(client_, wf):
        ran.append(wf)
        w, h = wf["9"]["inputs"]["width"], wf["9"]["inputs"]["height"]
        return _png(w, h, (0, 0, 255))

    monkeypatch.setattr(edit_app, "_run_graph", fake_graph)
    return client, ran, boxes


class TestFaceChoiceInTheAPI:
    def test_faces_left_to_right_with_the_largest_as_default(self, faces_api):
        client, _, boxes = faces_api
        r = client.post("/faces", json={"image": base64.b64encode(_png(100, 60)).decode()})
        body = r.json()
        assert r.status_code == 200 and (body["width"], body["height"]) == (100, 60)
        assert [f["box"] for f in body["faces"]] == boxes
        assert body["default_index"] == 1, "the second is larger (30x36 vs 30x35)"

    def test_a_face_box_edits_that_face_alone(self, faces_api):
        import numpy as np
        from PIL import Image
        client, ran, boxes = faces_api
        src = _png(100, 60, (200, 150, 120))
        r = client.post("/edit", json={"image": base64.b64encode(src).decode(),
                                       "expression": "smile", "face_box": boxes[0]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["face_box"] == boxes[0]
        l, t, rr, b = body["crop"]
        assert rr <= 60, "kept out of the neighbour's box"
        assert body["neighbours_in_crop"] == 0
        assert (body["width"], body["height"]) == (100, 60), "pasted back at full size"
        out = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB"))
        assert (out[:, 70:] == (200, 150, 120)).all(), "the other person is not regenerated"
        assert tuple(out[25, 25]) != (200, 150, 120), "the chosen one is"
        # The crop went to the model scaled up to MIN_EDGE on its short side.
        assert min(ran[0]["9"]["inputs"]["width"], ran[0]["9"]["inputs"]["height"]) >= 752

    def test_a_box_outside_the_image_is_a_422(self, faces_api):
        client, _, _ = faces_api
        r = client.post("/edit", json={"image": base64.b64encode(_png(100, 60)).decode(),
                                       "expression": "smile", "face_box": [500, 5, 600, 50]})
        assert r.status_code == 422

    def test_health_says_what_it_can_do(self, api):
        client, _ = api
        body = client.get("/health").json()
        assert {"expression", "face_box", "faces"} <= set(body["features"])
        assert body["shared_with_a1111"] is False and body["a1111_generating"] is None


class TestSharingTheCardWithA1111:
    """console#570: the standing box shares its 3090 with Automatic1111."""

    def test_an_edit_waits_for_a1111_and_says_why(self, api, monkeypatch):
        from wanly_worker.services.image_edit import share
        client, ran = api
        looks = iter([True, True, False])
        seen = []
        monkeypatch.setattr(share, "A1111_URL", "http://a1111:7860")
        monkeypatch.setattr(share, "A1111_POLL_S", 0.01)
        monkeypatch.setattr(share.fe_gpu, "a1111_generating",
                            lambda url=None: (seen.append(edit_app._state.get("waiting")),
                                              next(looks))[1])
        monkeypatch.setattr(share.fe_gpu, "free_mib", lambda: 23000)
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "expression": "smile"})
        assert r.status_code == 200, r.text
        assert "Automatic1111 on this card is generating" in seen
        assert edit_app._state["waiting"] is None, "cleared once it stops"
        assert len(ran) == 1

    def test_an_endless_generation_is_a_503_not_a_forced_load(self, api, monkeypatch):
        from wanly_worker.services.image_edit import share
        client, ran = api
        monkeypatch.setattr(share, "A1111_URL", "http://a1111:7860")
        monkeypatch.setattr(share, "A1111_POLL_S", 0.01)
        monkeypatch.setattr(share, "A1111_WAIT_S", 0.05)
        monkeypatch.setattr(share.fe_gpu, "a1111_generating", lambda url=None: True)
        r = client.post("/edit", json={"image": base64.b64encode(_png()).decode(),
                                       "expression": "smile"})
        assert r.status_code == 503 and "generating" in r.json()["detail"]
        assert ran == [], "Qwen never started under a live generation"
        assert not edit_app._turn.locked()

    def test_room_is_made_only_when_short_and_not_resident(self, monkeypatch):
        from wanly_worker.services.image_edit import share
        yields = []
        monkeypatch.setattr(share, "A1111_URL", "http://a1111:7860")
        monkeypatch.setattr(share.fe_gpu, "yield_a1111",
                            lambda url, purpose: yields.append((url, purpose)) or True)
        monkeypatch.setattr(share.fe_gpu, "free_mib", lambda: 16000)
        assert asyncio.run(share.make_room(resident=True)) is None
        assert yields == []
        assert "unloaded" in asyncio.run(share.make_room(resident=False))
        assert yields == [("http://a1111:7860", "a Qwen image edit")]
        monkeypatch.setattr(share.fe_gpu, "free_mib", lambda: 23000)
        assert asyncio.run(share.make_room(resident=False)) is None
        assert len(yields) == 1

    def test_no_a1111_is_no_sharing(self, monkeypatch):
        from wanly_worker.services.image_edit import share
        monkeypatch.setattr(share, "A1111_URL", "")
        monkeypatch.setattr(share, "UNLOAD_IDLE_S", 0.0)
        assert asyncio.run(share.wait_for_a1111({})) == 0.0
        assert asyncio.run(share.make_room(resident=False)) is None
        assert not share.watching()

    @pytest.mark.parametrize("kw,idle,want", [
        (dict(resident=False, busy=False, a1111_generating=True), 0, None),
        (dict(resident=True, busy=True, a1111_generating=True), 0, None),
        (dict(resident=True, busy=False, a1111_generating=True), 0, "Automatic1111"),
        (dict(resident=True, busy=False, a1111_generating=False), 50, None),
        (dict(resident=True, busy=False, a1111_generating=False), 61, "no edit for"),
        (dict(resident=True, busy=False, a1111_generating=None), 61, "no edit for"),
    ])
    def test_when_qwen_leaves_the_card(self, monkeypatch, kw, idle, want):
        from wanly_worker.services.image_edit import share
        monkeypatch.setattr(share, "UNLOAD_IDLE_S", 60.0)
        why = share.unload_reason(idle_s=idle, **kw)
        assert (why is None) if want is None else (want in why)

    def test_the_watcher_unloads_when_a1111_starts(self, monkeypatch):
        from wanly_worker.services.image_edit import share
        unloads = []

        async def gen():
            return True

        async def unload(url, client=None):
            unloads.append(url)
            return True

        monkeypatch.setattr(share, "WATCH_S", 0.01)
        monkeypatch.setattr(share, "a1111_generating", gen)
        monkeypatch.setattr(share, "unload_qwen", unload)
        monkeypatch.setattr(edit_app, "_state", {"last_edit_at": 0, "last": None,
                                                 "model_loaded": True, "edits": 1,
                                                 "waiting": None, "unloads": 0})

        async def run():
            t = asyncio.create_task(edit_app._watch())
            for _ in range(100):
                await asyncio.sleep(0.01)
                if unloads:
                    break
            t.cancel()

        asyncio.run(run())
        assert unloads == [edit_app.COMFY]
        assert edit_app._state["model_loaded"] is False
        assert "Automatic1111" in edit_app._state["last_unload"]["why"]

    def test_resident_forever_when_asked(self, monkeypatch):
        from wanly_worker.services.image_edit import share
        monkeypatch.setattr(share, "UNLOAD_IDLE_S", 0.0)
        assert share.unload_reason(resident=True, busy=False, idle_s=10**6,
                                   a1111_generating=None) is None

    def test_a1111_url_is_passed_through_to_face_edits_helpers(self, monkeypatch):
        """The helpers are face-edit's, reused; the URL must be image-edit's own."""
        from wanly_worker.services.face_edit import gpu
        got = []

        class R:
            status_code = 200

            def json(self):
                return {"state": {"job_count": 1}}

        monkeypatch.setattr(gpu, "A1111_URL", "http://not-this:7860")
        monkeypatch.setattr(gpu.httpx, "get", lambda url, timeout: got.append(url) or R())
        assert gpu.a1111_generating("http://a1111:7860/") is True
        assert got == ["http://a1111:7860/sdapi/v1/progress"]


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
    for i, (folder, name) in enumerate(graph.MODEL_FILES.items()):
        _safetensors(models / folder / name, truncate and i == 0)
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
        assert "3 file(s) OK" in r.stdout and "models OK" in r.stdout
        assert "diffusion_models" not in r.stdout, "the LTX tree is not this target's"

    def test_a_truncated_checkpoint_fails_the_check(self, tmp_path):
        r = _check(tmp_path, truncate=True)
        assert r.returncode != 0
        assert "TRUNCATED qwen_image_edit_2511_fp8mixed.safetensors" in r.stdout

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                        reason="root ignores the read-only bit, and would start a real fetch")
    def test_a_missing_text_encoder_on_a_host_mount_is_named(self, tmp_path):
        """The 3090's tree had the transformer and no text encoder or VAE: the boot must say
        which file is missing, not just fail."""
        models = tmp_path / "qwen"
        _safetensors(models / "base" / graph.UNET)
        _safetensors(models / "vae" / graph.VAE)
        (models / ".hf").mkdir()
        # Read-only, like the 3090's mount: refused before any fetch is attempted.
        models.chmod(0o555)
        env = {**os.environ, "IMAGE_EDIT_MODELS_DIR": str(models),
               "INSIGHTFACE_ROOT": str(tmp_path / "if")}
        try:
            r = subprocess.run(["bash", os.path.join(REPO, "download_models.sh"),
                                "--image-edit"], capture_output=True, text=True, env=env,
                               timeout=60)
        finally:
            models.chmod(0o755)
        assert r.returncode != 0
        assert "missing: text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" in r.stdout


class TestTheModelSetIsOneList:
    def test_the_graph_loads_exactly_what_the_preflight_checks(self):
        """graph.MODEL_FILES and download_models.sh's --image-edit _WANTED name the same files
        in the same folders -- the daemon-side rule: a model in one and not the other either
        never loads or never gets checked."""
        script = open(os.path.join(REPO, "download_models.sh")).read()
        block = script[script.index('if [ "$TARGET" = image-edit ]; then\n    _WANTED=('):]
        block = block[:block.index("\n    )")]
        rows = [ln.strip().strip('"') for ln in block.splitlines() if "|" in ln]
        wanted = {r.split("|")[0]: r.split("|")[1] for r in rows}
        assert wanted == graph.MODEL_FILES
        for r in rows:
            folder, name, repo, path, _gib = r.split("|")
            assert repo.startswith("Comfy-Org/"), "the official files, as Comfy-Org ships them"
            assert path.endswith("/" + name)

    def test_comfy_is_pointed_at_those_folders(self):
        from wanly_worker.services.image_edit import service
        y = service.paths_yaml("/m")
        assert "base_path: /m" in y
        for key, folder in (("diffusion_models", "base"), ("text_encoders", "text_encoders"),
                            ("vae", "vae")):
            assert f"{key}: {folder}/" in y
        assert set(graph.MODEL_FILES) == {"base", "text_encoders", "vae"}
        assert "v23" not in y


# ------------------------------------------------------------------ character sheets (#582, #585)

#: loras/phase0-2026-10-01/character_sheet_one_input.json, the tested one-photo workflow: its
#: four text nodes joined by its three StringConcatenate nodes, verbatim.
RECIPE_K2026 = (
    "Create a photorealistic full-body character turnaround of the woman in image 1 on a plain "
    "pure white studio background. Three full-body views of the same person side by side, left "
    "to right: a front view facing the camera, a side view facing 90 degrees to the right, and a "
    "back view facing completely away from the camera. Each view shows her whole body from head "
    "to toe with no cropping, standing upright with arms relaxed at her sides, feet visible. "
    "Keep her exact face, facial features, skin tone, body shape, build and proportions from "
    "image 1, and her ash-blonde hair pulled back into a low bun. She wears the same light grey "
    "zip-neck fleece pullover over a turquoise t-shirt that she wears in image 1, dark blue "
    "straight-leg jeans and white sneakers, identical in all three views. Soft even studio "
    "lighting, equal white spacing between the views, no text, no labels, no borders.")
K2026_OUTFIT = ("the same light grey zip-neck fleece pullover over a turquoise t-shirt that she "
                "wears in image 1, dark blue straight-leg jeans and white sneakers")
K2026_HAIR = "her ash-blonde hair pulled back into a low bun"


class TestTheTurnaroundPrompt:
    def test_the_one_photo_workflow_word_for_word(self):
        assert graph.turnaround_prompt(K2026_OUTFIT, K2026_HAIR) == RECIPE_K2026

    def test_build_comes_from_the_photo_not_from_words(self):
        """#585: the model ignored body words, so there is no way to send any."""
        import inspect
        assert "body" not in inspect.signature(graph.turnaround_prompt).parameters
        assert "body shape, build and proportions from image 1" in RECIPE_K2026
        assert "She has" not in RECIPE_K2026

    def test_a_man(self):
        p = graph.turnaround_prompt("a navy suit", "short dark hair", gender="male")
        assert "of the man in image 1" in p and "his whole body" in p and "his sides" in p
        assert "Keep his exact face" in p and "and short dark hair. He wears a navy suit" in p
        assert " her " not in p and "She " not in p

    def test_subject_and_default_hair(self):
        p = graph.turnaround_prompt("jeans", subject="young woman")
        assert "of the young woman in image 1" in p
        assert "proportions from image 1, and her hair exactly as in image 1." in p

    def test_an_outfit_is_required(self):
        with pytest.raises(ValueError, match="outfit"):
            graph.turnaround_prompt("  . ")

    def test_the_recipe_graph(self):
        """The one-input workflow: FluxKontextImageScale on the photo, 1088x1024, 40/4/euler."""
        wf = graph.turnaround_workflow("photo.png", "p", seed=11)
        assert wf["101"]["inputs"]["image"] == "photo.png"
        assert wf["102"] == {"class_type": "FluxKontextImageScale",
                             "inputs": {"image": ["101", 0]}}
        assert wf["3"]["inputs"]["image1"] == ["102", 0] == wf["4"]["inputs"]["image1"]
        assert "image2" not in wf["3"]["inputs"], "one photo, one reference"
        assert wf["9"]["inputs"] == {"width": 1088, "height": 1024, "batch_size": 1}
        k = wf["2"]["inputs"]
        assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["seed"],
                k["denoise"]) == (40, 4.0, "euler", "simple", 11, 1.0)
        assert wf["6"]["class_type"] == "SaveImage", "the node _run_graph reads"
        assert "40 steps, cfg 4, euler/simple, AuraFlow shift 3.1, CFGNorm 1" in \
            graph.settings_note()


def _full_body(w=3000, h=4000, face=(1400, 600, 1600, 860)):
    """A full-body photo stand-in: grey all over (no white anywhere), the face a red box."""
    from PIL import Image
    im = Image.new("RGB", (w, h), (90, 90, 90))
    im.paste((255, 0, 0), face)
    return im


class TestTheFacePanel:
    def test_the_face_box_plus_padding_is_cropped_from_the_photo(self):
        from wanly_worker.services.image_edit import sheet
        assert sheet.crop_region([1400, 600, 1600, 860], (3000, 4000), 140) == \
            (1260, 460, 1740, 1000)

    def test_the_crop_is_clamped_not_shifted_at_the_edge(self):
        """CropByBBoxes' rule: a face near the edge keeps its own side, nothing invented."""
        from wanly_worker.services.image_edit import sheet
        assert sheet.crop_region([20, 30, 220, 280], (1000, 1500), 140) == (0, 0, 360, 420)

    def test_the_panel_is_the_face_centred_on_white(self):
        from wanly_worker.services.image_edit import sheet
        im = _full_body()
        panel, info = sheet.face_panel(im, [1400, 600, 1600, 860], 140)
        assert panel.size == (448, 1024)
        assert info["mode"] == "auto_crop" and info["crop"] == [1260, 460, 1740, 1000]
        # 480x540 crop -> fits 448 wide: x0.93, 448x504, centred vertically on white.
        assert info["scale"] == 0.93
        assert panel.getpixel((224, 5)) == (255, 255, 255), "white above"
        assert panel.getpixel((224, 1018)) == (255, 255, 255), "white below"
        assert panel.getpixel((224, 512)) == (255, 0, 0), "the face in the middle"
        assert panel.getpixel((10, 512)) == (90, 90, 90), "padding: the photo around the face"

    def test_more_padding_shows_more_of_her(self):
        from wanly_worker.services.image_edit import sheet
        im = _full_body()
        _, tight = sheet.face_panel(im, [1400, 600, 1600, 860], 0)
        _, wide = sheet.face_panel(im, [1400, 600, 1600, 860], 400)
        assert tight["crop"] == [1400, 600, 1600, 860] and wide["crop"] == [1000, 200, 2000, 1260]
        assert tight["scale"] > wide["scale"]

    def test_a_small_face_is_upscaled_and_says_so(self):
        """A full-body shot by a tent: the face is small, the panel is soft, and `scale` > 1
        is how the record shows it."""
        from wanly_worker.services.image_edit import sheet
        _, info = sheet.face_panel(_full_body(4000, 3000, (2000, 900, 2064, 980)),
                                   [2000, 900, 2064, 980], 140)
        assert info["scale"] > 1

    def test_no_detector_is_centred(self):
        from wanly_worker.services.image_edit import sheet
        panel, info = sheet.face_panel(_full_body(2000, 1500), None)
        assert info == {"mode": "centre", "crop": None, "scale": None}
        assert panel.size == (448, 1024)

    def test_the_sheet_is_1536x1024_real_face_left(self):
        from wanly_worker.services.image_edit import sheet
        from PIL import Image
        body = Image.new("RGB", (1088, 1024), (0, 255, 0))
        out, panel, info = sheet.compose(_full_body(), body, [1400, 600, 1600, 860])
        assert out.size == (1536, 1024) and info["mode"] == "auto_crop"
        assert out.getpixel((224, 512)) == (255, 0, 0), "the photo's face, left"
        assert out.crop((0, 0, 448, 1024)).tobytes() == panel.tobytes()
        assert out.getpixel((448, 500)) == (0, 255, 0) and out.getpixel((1535, 0)) == (0, 255, 0)

    def test_the_largest_face_is_the_subject(self):
        from wanly_worker.services.image_edit import sheet
        assert sheet.largest([[0, 0, 10, 10], [5, 5, 50, 60], [0, 0, 20, 20]]) == [5, 5, 50, 60]
        assert sheet.largest([]) is None


class TestTheTurnaroundAPI:
    def _req(self, **kw):
        return {"image": base64.b64encode(_png(600, 800)).decode(), "outfit": "jeans",
                "hair": "her red hair", "seed": 22, **kw}

    def _faces(self, monkeypatch, found_at):
        """face_boxes that finds the face only at the detector sizes in `found_at`."""
        asked = []

        def boxes(rgb, det_size=640):
            asked.append(det_size)
            return [[250.0, 200.0, 350.0, 330.0]] if det_size in found_at else []
        monkeypatch.setattr(edit_app.identity, "face_boxes", boxes)
        return asked

    def test_one_candidate_and_its_sheet(self, api, monkeypatch):
        from PIL import Image
        client, ran = api
        asked = self._faces(monkeypatch, {640})
        r = client.post("/turnaround", json=self._req())
        assert r.status_code == 200, r.text
        body = r.json()
        cand = Image.open(io.BytesIO(base64.b64decode(body["candidate"])))
        sheet_im = Image.open(io.BytesIO(base64.b64decode(body["sheet"])))
        assert cand.size == (1088, 1024) and sheet_im.size == (1536, 1024)
        assert sheet_im.getpixel((1000, 500)) == (10, 20, 30), "the turnaround, right"
        assert sheet_im.getpixel((224, 512)) == (200, 150, 120), "the photo's face, left"
        assert sheet_im.getpixel((224, 3)) == (255, 255, 255), "on white"
        assert body["seed"] == 22 and body["steps"] == 40 and body["cfg"] == 4.0
        assert "build and proportions from image 1, and her red hair. She wears jeans" in \
            body["prompt"]
        fp = body["face_panel"]
        assert fp["mode"] == "auto_crop" and fp["source"] == "same_photo"
        assert fp["box"] == [250.0, 200.0, 350.0, 330.0] and fp["padding"] == 140
        assert fp["crop"] == [110, 60, 490, 470] and fp["det_size"] == 640
        assert fp["photo_size"] == [600, 800] and asked == [640]
        assert body["model"] == graph.MODEL and body["files"] == graph.MODEL_FILES
        assert body["identity"] == {"aura": 0.71, "reason": None}
        assert base64.b64decode(body["sheet_preview"])[:2] == b"\xff\xd8"
        panel = Image.open(io.BytesIO(base64.b64decode(body["face_panel_preview"])))
        assert panel.size == (448, 1024), "the panel, for the console's preview"
        assert ran[0]["9"]["inputs"]["width"] == 1088 and ran[0]["3"]["inputs"]["prompt"] == \
            body["prompt"]

    def test_the_padding_is_the_callers(self, api, monkeypatch):
        client, _ = api
        self._faces(monkeypatch, {640})
        fp = client.post("/turnaround", json=self._req(crop_padding=20)).json()["face_panel"]
        assert fp["padding"] == 20 and fp["crop"] == [230, 180, 370, 350]

    def test_a_small_face_is_looked_for_again_at_1280(self, api, monkeypatch):
        client, ran = api
        asked = self._faces(monkeypatch, {1280})
        r = client.post("/turnaround", json=self._req())
        assert r.status_code == 200, r.text
        assert asked == [640, 1280] and r.json()["face_panel"]["det_size"] == 1280

    def test_a_body_field_is_not_a_prompt(self, api, monkeypatch):
        """#585 dropped the field; a stale caller's `body` must not reach the words."""
        client, _ = api
        self._faces(monkeypatch, {640})
        r = client.post("/turnaround", json=self._req(body="a petite frame"))
        assert r.status_code == 200 and "petite" not in r.json()["prompt"]

    def test_a_photo_with_no_face_is_refused_before_the_card(self, api, monkeypatch):
        client, ran = api
        asked = self._faces(monkeypatch, set())
        r = client.post("/turnaround", json=self._req())
        assert r.status_code == 422 and "no face" in r.json()["detail"]
        assert ran == [] and asked == [640, 1280]

    def test_a_detector_that_will_not_load_still_makes_a_sheet(self, api, monkeypatch):
        client, ran = api

        def broken(rgb, det_size=640):
            raise RuntimeError("no buffalo_l")
        monkeypatch.setattr(edit_app.identity, "face_boxes", broken)
        r = client.post("/turnaround", json=self._req())
        assert r.status_code == 200, r.text
        fp = r.json()["face_panel"]
        assert fp["mode"] == "centre" and "unavailable" in fp["note"]

    def test_no_outfit_is_a_422(self, api):
        client, _ = api
        assert client.post("/turnaround", json=self._req(outfit=" ")).status_code == 422

    def test_padding_out_of_range_is_a_422(self, api):
        client, _ = api
        assert client.post("/turnaround", json=self._req(crop_padding=-1)).status_code == 422
        assert client.post("/turnaround", json=self._req(crop_padding=5000)).status_code == 422

    def test_health_advertises_it(self, api):
        client, _ = api
        body = client.get("/health").json()
        assert {"turnaround", "official_2511", "one_photo"} <= set(body["features"])
        assert body["model"] == graph.MODEL and body["checkpoint"] == graph.UNET
