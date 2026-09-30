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
