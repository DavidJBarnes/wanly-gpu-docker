"""face-edit: LivePortrait expression edits (wanly-console#547).

LivePortrait itself is not run here -- CI has no torch, and the node's maths is keyframe-server's
measured code, unchanged. What is tested is everything around it, each piece of which encodes a
way this has gone wrong before or would go wrong silently:

  * the numeric contract and the prompt lexicon (ported, so a port error shows up here)
  * detail restoration's one hard guarantee: pixels the warp did not move come back
    byte-for-byte from the source
  * the device policy -- never borrow the GPU from a render or an A1111 generation, and never
    ask A1111 to unload mid-generation
  * the HTTP surface, against a fake engine: validation, no-face, busy, GPU-unavailable
  * the service contract: registered, runs in every mode, non-essential, :full-only
"""
import base64
import io
import pathlib

import numpy as np
import pytest

from wanly_worker import registry
from wanly_worker.services.face_edit import gpu
from wanly_worker.services.face_edit.expression import (
    Expression, NothingToApply, bounds, resolve_expression, whole_face_motion,
)

ROOT = pathlib.Path(__file__).parent.parent


# --------------------------------------------------------------------- the contract

class TestTheNumbers:
    def test_the_ranges_are_the_nodes(self):
        """ExpressionEditor.INPUT_TYPES. Wider and the node warps a face inside out; narrower
        and a slider cannot reach what the node can do."""
        assert bounds("rotate_yaw") == (-20, 20)
        assert bounds("blink") == (-20, 5)
        assert bounds("aaa") == (-30, 120)
        assert bounds("smile") == (-0.3, 1.3)
        assert bounds("wink") == (0, 25)

    def test_out_of_range_is_refused_not_clamped(self):
        """A caller's explicit number is validated, never softened: a typo that silently
        became the maximum would look like the model misbehaving."""
        with pytest.raises(Exception):
            Expression(rotate_yaw=45)

    def test_explicit_numbers_win_over_a_prompt(self):
        exp, how = resolve_expression(Expression(smile=0.3), "big laugh")
        assert how == "explicit" and exp.smile == 0.3 and exp.aaa == 0

    def test_the_lexicon(self):
        exp, how = resolve_expression(None, "she closes her eyes")
        assert exp.blink == -18 and how.startswith("prompt:")

    def test_largest_magnitude_wins_per_axis(self):
        """"smiling and grinning" is one grin, not a summed clamp."""
        exp, _ = resolve_expression(None, "smiling and grinning")
        assert exp.smile == 1.0

    def test_an_intensity_adverb_scales_and_is_clamped(self):
        exp, _ = resolve_expression(None, "slightly look left")
        assert exp.pupil_x == -4
        exp, _ = resolve_expression(None, "a very big grin")
        assert exp.smile == 1.3            # 1.0 * 1.5, clamped to the node's ceiling

    def test_nothing_to_apply_is_an_error_not_a_no_op(self):
        """Returning the input unchanged reads as "the edit worked and did nothing"."""
        with pytest.raises(NothingToApply, match="Recognised terms"):
            resolve_expression(None, "make it nicer")

    def test_a_head_turn_is_whole_face_motion_and_a_smile_is_not(self):
        """What sizes the restore's max-pool: a rotation moves every pixel, a smile a few."""
        assert whole_face_motion(Expression(rotate_yaw=12), 1.0) == 1.0
        assert whole_face_motion(Expression(smile=1.0), 1.0) == 0.0
        assert whole_face_motion(Expression(rotate_pitch=4), 1.0) == 0.5
        assert whole_face_motion(Expression(), 0.4) == pytest.approx(0.6)


# ---------------------------------------------------------------- detail restoration

class TestDetailRestoration:
    @pytest.fixture(autouse=True)
    def _cv2(self):
        pytest.importorskip("cv2")

    def _pair(self):
        rng = np.random.default_rng(0)
        src = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        edited = src.copy()
        edited[40:70, 30:60] = np.clip(edited[40:70, 30:60].astype(int) + 90, 0, 255)
        return src, edited

    def test_pixels_far_from_the_edit_are_the_sources_byte_for_byte(self):
        """The guarantee the whole engine choice rests on: outside what moved, the output IS
        the input. keyframe-server measured 0.0000 drift; a sharpen applied before the change
        map once made it 0.63 with 36% of the frame touched."""
        from wanly_worker.services.face_edit.restore import restore_detail
        src, edited = self._pair()
        out = restore_detail(src, edited, strength=1.0, sharpen=0.0, motion=0.0)
        assert np.array_equal(out[:15], src[:15])
        assert np.array_equal(out[-15:], src[-15:])

    def test_the_edit_itself_survives(self):
        from wanly_worker.services.face_edit.restore import restore_detail
        src, edited = self._pair()
        out = restore_detail(src, edited, strength=1.0, sharpen=0.0, motion=0.0)
        assert np.array_equal(out[50:60, 40:50], edited[50:60, 40:50])

    def test_a_sharpen_does_not_leak_outside_the_changed_region(self):
        from wanly_worker.services.face_edit.restore import restore_detail
        src, edited = self._pair()
        out = restore_detail(src, edited, strength=1.0, sharpen=2.0, motion=0.0)
        assert np.array_equal(out[:15], src[:15])

    def test_zero_strength_returns_the_nodes_output(self):
        from wanly_worker.services.face_edit.restore import restore_detail
        src, edited = self._pair()
        assert restore_detail(src, edited, strength=0.0) is edited

    def test_a_shape_mismatch_returns_the_nodes_output_rather_than_crashing(self):
        from wanly_worker.services.face_edit.restore import restore_detail
        src, edited = self._pair()
        assert restore_detail(src[:50], edited) is edited


# ------------------------------------------------------------------ device policy

@pytest.fixture
def policy(monkeypatch):
    """A card with CUDA, plenty free, no neighbours -- each test changes one thing."""
    state = {"cuda": True, "free": 6000, "render": False, "a1111": False, "yield": False,
             "yields": 0}
    monkeypatch.setattr(gpu, "DEVICE", "auto")
    monkeypatch.setattr(gpu, "CPU_FALLBACK", True)
    monkeypatch.setattr(gpu, "MIN_FREE_MIB", 2560)
    monkeypatch.setattr(gpu, "cuda_available", lambda: state["cuda"])
    monkeypatch.setattr(gpu, "free_mib", lambda: state["free"])
    monkeypatch.setattr(gpu, "render_busy", lambda: state["render"])
    monkeypatch.setattr(gpu, "a1111_generating", lambda: state["a1111"])

    def fake_yield():
        state["yields"] += 1
        if state["yield"]:
            state["free"] = 6000
        return state["yield"]

    monkeypatch.setattr(gpu, "yield_a1111", fake_yield)
    return state


class TestTheDevicePolicy:
    def test_a_free_card_is_used(self, policy):
        assert gpu.choose(False)[0] == "cuda"

    def test_forced_cpu(self, policy, monkeypatch):
        monkeypatch.setattr(gpu, "DEVICE", "cpu")
        assert gpu.choose(False)[0] == "cpu"

    def test_no_cuda_is_cpu(self, policy):
        policy["cuda"] = False
        assert gpu.choose(False) == ("cpu", "no CUDA device visible")

    def test_a_render_in_flight_keeps_it_off_the_card(self, policy):
        """The 3090 at ~23 of 24 GB mid-render: an edit's 1.5-2 GB there is an OOM in a
        segment that has no fallback. The edit has one."""
        policy["render"] = True
        device, why = gpu.choose(False)
        assert device == "cpu" and "render" in why

    def test_a_render_in_flight_evicts_even_a_resident_pipeline(self, policy):
        policy["render"] = True
        assert gpu.choose(True)[0] == "cpu"

    def test_an_a1111_generation_keeps_it_off_the_card(self, policy):
        """The 2070 under generate-forever. A starved generation dies; an edit on CPU is just
        slower."""
        policy["a1111"] = True
        device, why = gpu.choose(False)
        assert device == "cpu" and "Automatic1111" in why
        assert policy["yields"] == 0, "never ask a generating A1111 to unload"

    def test_a_resident_pipeline_needs_no_free_check(self, policy):
        """Its own weights are what makes free VRAM look low."""
        policy["free"] = 100
        assert gpu.choose(True)[0] == "cuda"

    def test_a_tight_card_asks_an_idle_a1111_to_let_go(self, policy):
        policy["free"], policy["yield"] = 1500, True
        device, why = gpu.choose(False)
        assert device == "cuda" and "unloaded" in why
        assert policy["yields"] == 1

    def test_a1111_is_only_asked_when_it_would_buy_something(self, policy):
        """The captioner's rule: an unload costs A1111 a reload, so it is paid only when the
        edit could not otherwise use the card."""
        gpu.choose(False)
        assert policy["yields"] == 0

    def test_a_tight_card_with_nothing_to_free_falls_back_to_cpu(self, policy):
        policy["free"] = 1500
        device, why = gpu.choose(False)
        assert device == "cpu" and "1500 MiB" in why

    def test_without_cpu_fallback_it_refuses_and_says_why(self, policy, monkeypatch):
        monkeypatch.setattr(gpu, "CPU_FALLBACK", False)
        policy["a1111"] = True
        with pytest.raises(gpu.GpuUnavailable, match="Automatic1111"):
            gpu.choose(False)


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


class TestTheA1111Yield:
    """Ported from wanly-api's _yield_the_gpu -- same rules, on the box that shares the card."""

    def _client(self, monkeypatch, job_count, unload_status=200):
        calls = []

        class C:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url):
                calls.append(("GET", url))
                return _Resp(200, {"state": {"job_count": job_count}})

            def post(self, url):
                calls.append(("POST", url))
                return _Resp(unload_status)

        monkeypatch.setattr(gpu.httpx, "Client", C)
        monkeypatch.setattr(gpu, "A1111_URL", "http://a1111:7860")
        return calls

    def test_an_idle_a1111_is_asked_to_unload(self, monkeypatch):
        calls = self._client(monkeypatch, job_count=0)
        assert gpu.yield_a1111() is True
        assert ("POST", "http://a1111:7860/sdapi/v1/unload-checkpoint") in calls

    def test_a_generating_a1111_is_never_unloaded(self, monkeypatch):
        calls = self._client(monkeypatch, job_count=1)
        assert gpu.yield_a1111() is False
        assert not [c for c in calls if c[0] == "POST"]

    def test_a_refusal_is_not_a_success(self, monkeypatch):
        self._client(monkeypatch, job_count=0, unload_status=500)
        assert gpu.yield_a1111() is False

    def test_no_a1111_configured_is_nothing_to_do(self, monkeypatch):
        monkeypatch.setattr(gpu, "A1111_URL", "")
        assert gpu.yield_a1111() is False
        assert gpu.a1111_generating() is False


# ------------------------------------------------------------------- the HTTP API

def _png(w=64, h=48) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 90, 60)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEngine:
    """Stands in for LivePortrait: brightens a patch, the way an edit changes a region."""

    def __init__(self, face=True, fail=None):
        import threading
        self.face, self.fail = face, fail
        self.device = "cpu"
        self.loaded = True
        self.load_error = None
        self.loaded_in_s = 0.0
        self.last_used = 0.0
        self.vram_peak_mib = None
        self.lock = threading.Lock()
        self.calls = []

    def load(self):
        pass

    def to(self, device):
        self.device = device

    def edit(self, rgb, exp, *, face_pad, src_ratio):
        from wanly_worker.services.face_edit.engine import NoFace
        self.calls.append((exp, face_pad, src_ratio, self.device))
        if self.fail:
            raise self.fail
        if not self.face:
            raise NoFace("no face detected")
        out = rgb.copy()
        out[10:20, 10:20] = 255
        return out


@pytest.fixture
def api(monkeypatch):
    pytest.importorskip("PIL")
    from fastapi.testclient import TestClient
    from wanly_worker.services.face_edit import app as mod

    fake = FakeEngine()
    monkeypatch.setattr(mod, "engine", fake)
    monkeypatch.setattr(mod.gpu, "choose", lambda resident_on_gpu: ("cpu", "test"))
    with TestClient(mod.app) as c:
        yield c, fake, mod


class TestTheAPI:
    def test_an_edit_returns_a_png_of_the_same_size(self, api):
        from PIL import Image
        c, fake, _ = api
        r = c.post("/edit", json={"image": _png(), "expression": {"smile": 0.5},
                                  "detail_restore": 0})
        assert r.status_code == 200, r.text
        d = r.json()
        im = Image.open(io.BytesIO(base64.b64decode(d["image"])))
        assert im.format == "PNG" and im.size == (64, 48)
        assert d["expression"]["smile"] == 0.5 and d["source"] == "explicit"
        assert d["device"] == "cpu" and d["face_pad"] == 1.6

    def test_png_not_jpeg(self, api):
        """The untouched pixels are the source's own; a lossy encode would undo that."""
        c, _, _ = api
        d = c.post("/edit", json={"image": _png(), "expression": {"blink": -10},
                                  "detail_restore": 0}).json()
        assert d["format"] == "png"

    def test_a_preview_is_a_small_jpeg_of_the_same_edit(self, api):
        """The slider loop's return trip: ~120 KB instead of ~3 MB behind a home uplink."""
        from PIL import Image
        c, _, _ = api
        d = c.post("/edit", json={"image": _png(400, 300), "expression": {"smile": 0.5},
                                  "detail_restore": 0, "format": "jpeg", "max_edge": 100}).json()
        im = Image.open(io.BytesIO(base64.b64decode(d["image"])))
        assert im.format == "JPEG" and max(im.size) == 100
        assert (d["width"], d["height"]) == (400, 300)

    def test_a_prompt_is_accepted_as_sugar(self, api):
        c, fake, _ = api
        r = c.post("/edit", json={"image": _png(), "prompt": "look up", "detail_restore": 0})
        assert r.status_code == 200 and fake.calls[-1][0].pupil_y == 8

    def test_nothing_to_apply_is_422(self, api):
        c, _, _ = api
        r = c.post("/edit", json={"image": _png()})
        assert r.status_code == 422 and "nothing to apply" in r.json()["detail"]

    def test_out_of_range_is_422(self, api):
        c, _, _ = api
        r = c.post("/edit", json={"image": _png(), "expression": {"rotate_yaw": 40}})
        assert r.status_code == 422

    def test_face_pad_is_held_to_the_nodes_range(self, api):
        c, _, _ = api
        r = c.post("/edit", json={"image": _png(), "expression": {"smile": 1}, "face_pad": 3.0})
        assert r.status_code == 422

    def test_no_face_is_422(self, api):
        c, fake, _ = api
        fake.face = False
        r = c.post("/edit", json={"image": _png(), "expression": {"smile": 0.5}})
        assert r.status_code == 422 and "no face" in r.json()["detail"]

    def test_garbage_is_400(self, api):
        c, _, _ = api
        r = c.post("/edit", json={"image": base64.b64encode(b"not an image").decode(),
                                  "expression": {"smile": 0.5}})
        assert r.status_code == 400

    def test_gpu_unavailable_is_503_with_the_reason(self, api, monkeypatch):
        c, _, mod = api

        def refuse(resident_on_gpu):
            raise gpu.GpuUnavailable("GPU unavailable: Automatic1111 on this card is generating")

        monkeypatch.setattr(mod.gpu, "choose", refuse)
        r = c.post("/edit", json={"image": _png(), "expression": {"smile": 0.5}})
        assert r.status_code == 503 and "Automatic1111" in r.json()["detail"]

    def test_a_gpu_oom_falls_back_to_cpu(self, api, monkeypatch):
        """The free-VRAM check is a snapshot; a neighbour can grow between it and the warp."""
        c, fake, mod = api

        class OutOfMemoryError(RuntimeError):
            pass

        monkeypatch.setattr(mod.gpu, "choose", lambda resident_on_gpu: ("cuda", "free"))
        real_edit = fake.edit

        def edit(rgb, exp, **kw):
            if fake.device == "cuda":
                raise OutOfMemoryError("CUDA out of memory")
            return real_edit(rgb, exp, **kw)

        fake.edit = edit
        r = c.post("/edit", json={"image": _png(), "expression": {"smile": 0.5},
                                  "detail_restore": 0})
        assert r.status_code == 200, r.text
        assert r.json()["device"] == "cpu" and fake.device == "cpu"

    def test_a_second_edit_waits_then_is_refused_as_busy(self, api, monkeypatch):
        c, _, mod = api
        monkeypatch.setattr(mod, "QUEUE_WAIT_S", 0.2)

        async def hold():
            await mod._turn.acquire()

        async def release():
            mod._turn.release()

        c.portal.call(hold)
        try:
            r = c.post("/edit", json={"image": _png(), "expression": {"smile": 0.5}})
        finally:
            c.portal.call(release)
        assert r.status_code == 503 and "busy" in r.json()["detail"]

    def test_health_separates_up_from_loaded(self, api):
        c, _, _ = api
        d = c.get("/health").json()
        assert d["status"] == "ok" and d["model_loaded"] is True
        assert {"device", "vram_peak_mib", "policy", "load_error"} <= set(d)


# ---------------------------------------------------------------- the service contract

class TestTheService:
    def test_it_is_registered_under_the_flag_name(self):
        assert "face-edit" in registry.KNOWN

    def test_it_claims_no_work(self):
        assert registry.kinds_for(["face-edit"]) == ["service"]
        assert "face-edit" not in registry.KIND_BY_SERVICE

    def test_it_runs_in_caption_mode_too(self):
        """Not its own mode: a mode stops a tenant, and stopping a render for one second of
        work is absurd. It decides per edit instead."""
        names = ["ltx-engine", "image-description", "face-edit"]
        assert "face-edit" in registry.select_mode(names, "caption")
        assert "face-edit" in registry.select_mode(names, None)

    def test_it_alone_is_a_valid_box(self):
        """The 2070: face-edit and nothing else."""
        assert registry.select_mode(["face-edit"], "caption") == ["face-edit"]

    def test_it_binds_all_interfaces(self):
        from wanly_worker.services.face_edit import FaceEdit
        assert "0.0.0.0" in FaceEdit().command()

    def test_it_is_not_essential(self):
        """The trainer's lesson (#105/#111): a side service's preflight must not take the
        render box down with it."""
        from wanly_worker.services.face_edit import FaceEdit
        assert FaceEdit.essential is False

    def test_the_lean_tag_fails_preflight_by_name(self, monkeypatch, tmp_path):
        from wanly_worker.service import PreflightError
        from wanly_worker.services.face_edit import FaceEdit, engine
        monkeypatch.setattr(engine, "NODE_DIR", str(tmp_path / "absent"))
        with pytest.raises(PreflightError, match=":full"):
            FaceEdit().preflight()

    def test_missing_models_fail_preflight(self, monkeypatch, tmp_path):
        from wanly_worker.service import PreflightError
        from wanly_worker.services.face_edit import FaceEdit, engine
        (tmp_path / "node").mkdir()
        (tmp_path / "node" / "nodes.py").write_text("")
        monkeypatch.setattr(engine, "NODE_DIR", str(tmp_path / "node"))
        monkeypatch.setattr(engine, "MODELS_DIR", str(tmp_path / "models"))
        with pytest.raises(PreflightError, match="models missing"):
            FaceEdit().preflight()

    def test_details_never_raises(self):
        from wanly_worker.services.face_edit import FaceEdit
        assert isinstance(FaceEdit().details(), dict)


class TestTheImage:
    SRC = (ROOT / "Dockerfile").read_text()

    def test_the_node_is_pinned_to_a_commit(self):
        import re
        m = re.search(r"ARG ALP_COMMIT=([0-9a-f]{40})", self.SRC)
        assert m, "ComfyUI-AdvancedLivePortrait must be pinned to a full commit"

    def test_every_model_is_checked_by_digest(self):
        from wanly_worker.services.face_edit.engine import DETECTOR_MODEL, LIVEPORTRAIT_MODELS
        for m in LIVEPORTRAIT_MODELS:
            assert f"liveportrait/{m}.safetensors" in self.SRC
        assert f"ultralytics/{DETECTOR_MODEL}" in self.SRC
        assert "sha256sum -c" in self.SRC

    def test_its_deps_install_under_the_images_own_pins(self):
        """ultralytics pulls torch/opencv/numpy requirements; this image has lost a day to a
        transitive install replacing a pinned build before."""
        assert "-c /tmp/face-edit-constraints.txt" in self.SRC

    def test_it_is_in_the_full_layer_only(self):
        block = self.SRC[self.SRC.index("ARG ALP_COMMIT"):]
        assert block.index('if [ "$WITH_TRAINER" = "1" ]') < block.index("ultralytics==")

    def test_the_port_is_exposed(self):
        assert "8085" in self.SRC.split("EXPOSE", 1)[1].splitlines()[0]


class TestDeployingIt:
    """run-worker.sh for a face-edit box. The 2070 renders nothing, so it must not need the
    render stack's mounts or publish its ports -- and a render box must not change at all."""

    def _run(self, tmp_path, env_lines):
        import os
        import shutil
        import subprocess
        stage = tmp_path / "deploy"
        stage.mkdir(exist_ok=True)
        shutil.copy(ROOT / "deploy" / "run-worker.sh", stage / "run-worker.sh")
        (stage / "worker.env").write_text(env_lines)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "docker").write_text('#!/usr/bin/env bash\necho "docker $*"\n')
        (bin_dir / "docker").chmod(0o755)
        for stub in ("ss", "nvidia-smi"):
            (bin_dir / stub).write_text("#!/usr/bin/env bash\nexit 0\n")
            (bin_dir / stub).chmod(0o755)
        e = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]),
                 WORKER_ENV=str(stage / "worker.env"))
        return subprocess.run(["bash", str(stage / "run-worker.sh")],
                              capture_output=True, text=True, env=e)

    BASE = "QUEUE_API_KEY=k\nQUEUE_URL=http://api.test:8001\nFRIENDLY_NAME=2070.zero\n"

    def test_a_face_edit_only_box_needs_no_render_mounts(self, tmp_path):
        r = self._run(tmp_path, self.BASE + "IMAGE=x:full\nSERVICES=face-edit\n"
                      "FACE_EDIT_A1111_URL=http://host.docker.internal:7860\n")
        assert r.returncode == 0, r.stdout + r.stderr
        run = [l for l in r.stdout.splitlines() if l.startswith("docker run")][0]
        assert "-p 8085:8085" in run
        assert "--add-host host.docker.internal:host-gateway" in run
        assert "-e FACE_EDIT_A1111_URL=http://host.docker.internal:7860" in run
        assert "-p 8191:8188" not in run and "-p 8190:" not in run and ":/jobs " not in run
        assert "/workspace/models:ro" not in run

    def test_the_lean_tag_is_refused_before_anything_is_removed(self, tmp_path):
        r = self._run(tmp_path, self.BASE + "IMAGE=davidjbarnes/wanly-gpu-docker:latest\n"
                      "SERVICES=face-edit\n")
        assert r.returncode != 0
        assert "face-edit" in r.stdout and "rm -f" not in r.stdout

    def test_unset_tuning_is_not_forwarded_as_empty(self, tmp_path):
        r = self._run(tmp_path, self.BASE + "IMAGE=x:full\nSERVICES=face-edit\n")
        assert "FACE_EDIT_MIN_FREE_MIB" not in r.stdout

    def test_a_render_box_keeps_every_render_flag(self, tmp_path):
        (tmp_path / "jobs").mkdir()
        (tmp_path / "models" / "loras").mkdir(parents=True)
        r = self._run(tmp_path, self.BASE + "IMAGE=x:full\nSERVICES=ltx-engine,face-edit\n"
                      f"JOBS_DIR={tmp_path}/jobs\nMODELS_DIR={tmp_path}/models\n")
        assert r.returncode == 0, r.stdout + r.stderr
        run = [l for l in r.stdout.splitlines() if l.startswith("docker run")][0]
        for flag in ("-p 8191:8188", "-p 8190:8190", f"-v {tmp_path}/jobs:/jobs",
                     f"-v {tmp_path}/models:/workspace/models:ro",
                     f"-v {tmp_path}/models/loras:/workspace/models/loras", "-p 8085:8085"):
            assert flag in run, flag

    def test_a_render_box_without_its_mounts_is_still_refused(self, tmp_path):
        r = self._run(tmp_path, self.BASE + "SERVICES=ltx-engine\nJOBS_DIR=/nope\n"
                      "MODELS_DIR=/nope\n")
        assert r.returncode != 0 and "broken mount" in r.stdout

    def test_image_pruning_can_be_turned_off(self, tmp_path):
        """The 2070's Docker holds other projects' images with no container; `prune -af`
        would delete them."""
        on = self._run(tmp_path, self.BASE + "IMAGE=x:full\nSERVICES=face-edit\n")
        assert on.returncode == 0 and "leaving unreferenced images alone" not in on.stdout
        off = self._run(tmp_path, self.BASE + "IMAGE=x:full\nSERVICES=face-edit\nPRUNE_IMAGES=0\n")
        assert off.returncode == 0 and "leaving unreferenced images alone" in off.stdout
        src = (ROOT / "deploy" / "run-worker.sh").read_text()
        assert 'if [ "${PRUNE_IMAGES:-1}" = "1" ]; then\n    docker image prune -af' in src
