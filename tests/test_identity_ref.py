"""The optional identity reference (#156): a character sheet or a face, on the recipe path.

Phase 0 (#155, 2026-10-01) rendered 36 clips through `add_identity_ref` patched onto wanly's
own resolved recipe graph, and LoRA + character sheet held identity best -- see
https://github.com/DavidJBarnes/wanly-gpu-docker/issues/155. What these tests hold:

  * WITHOUT a reference nothing moves: the submitted graph is byte-identical to what the engine
    built before this existed, pinned by hash. Every validated pose renders as it did.
  * WITH one, the patch reproduces the exact graphs phase 0 rendered (fixtures/phase0, copied
    verbatim from ~/projects/loras/phase0-2026-10-01/graphs), in both modes.
  * it works with no character LoRA at all -- a sheet-only character;
  * the request is refused early when half-specified, and the image stays in step with the
    engine: both LoRAs are in download_models.sh, BFSNodes is pinned at phase 0's commit.
"""
import base64
import hashlib
import io
import json
import pathlib
import queue
import re
import sys

import pytest
from fastapi import HTTPException
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
ENGINE = ROOT / "engine"
WORKFLOW = ENGINE / "workflows/ltx23_recipe.api.json"
PHASE0 = pathlib.Path(__file__).parent / "fixtures/phase0"

sys.path.insert(0, str(ENGINE))
import app as engine_app  # noqa: E402
import recipe as engine_recipe  # noqa: E402


@pytest.fixture
def graph():
    return json.loads(WORKFLOW.read_text())


def _phase0(arm):
    return json.loads((PHASE0 / f"k2026_{arm}_P1turn_s1001.api.json").read_text())


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 180, 170)).save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------- phase 0, reproduced exactly

@pytest.mark.parametrize("arm,mode,ref", [("SC", "sheet", "p0_k2026_sheet.png"),
                                          ("D", "face", "p0_k2026_face.png")])
def test_the_patch_reproduces_the_graphs_phase_0_rendered(arm, mode, ref):
    """The C (LoRA only) graph plus add_identity_ref must BE the D / SC graph that was
    rendered and scored -- the only difference allowed is the output name phase 0 gave it."""
    got = engine_recipe.add_identity_ref(_phase0("C"), ref, mode)
    want = _phase0(arm)
    for g in (got, want):
        g["140"]["inputs"]["filename_prefix"] = "<out>"
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


def test_the_input_graph_is_not_mutated():
    c = _phase0("C")
    before = json.dumps(c, sort_keys=True)
    engine_recipe.add_identity_ref(c, "ref.png", "sheet")
    assert json.dumps(c, sort_keys=True) == before


@pytest.mark.parametrize("mode", ["sheet", "face"])
def test_both_stages_are_patched(graph, mode):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="k3lly2026, woman, turns",
                              char_lora="k3lly2026_v2")
    g = engine_recipe.add_identity_ref(g, "idref.png", mode)
    lora = engine_recipe.IDENTITY_LORAS[mode]
    # LoRA after the character, before the preview override (and so before the distill).
    assert g["9641"]["inputs"] == {"lora_name": lora, "strength_model": 1.0, "model": ["9621", 0]}
    assert g["9642"]["inputs"] == {"lora_name": lora, "strength_model": 1.0, "model": ["9622", 0]}
    assert g["337"]["inputs"]["model"] == ["9641", 0]
    assert g["372"]["inputs"]["model"] == ["9642", 0]
    # Overlap conditioning reads the post-distill model and post-i2v latent ...
    assert g["9643"]["inputs"]["model"] == ["361", 0]
    assert g["9644"]["inputs"]["model"] == ["362", 0]
    assert g["9643"]["inputs"]["latent"] == ["109", 0]
    assert g["9644"]["inputs"]["latent"] == ["117", 0]
    # ... and feeds every guider and both samplers.
    assert g["383"]["inputs"]["model"] == ["9643", 0]
    assert g["129"]["inputs"]["model"] == ["9643", 0]
    assert g["103"]["inputs"]["model"] == ["9644", 0]
    assert g["113"]["inputs"]["latent_image"] == ["9643", 3]
    assert g["119"]["inputs"]["latent_image"] == ["9644", 3]
    want = "native_resolution" if mode == "sheet" else "match_target"
    for cid in ("9643", "9644"):
        assert g[cid]["class_type"] == "LTXIdentityOverlapConditioning"
        assert g[cid]["inputs"]["ref_resize_mode"] == want
        assert g[cid]["inputs"]["layout"] == "overlap"
        assert g[cid]["inputs"]["source_id"] == 2.0
    assert g["121"]["inputs"]["text"] == "ref_t2v: k3lly2026, woman, turns"


def test_a_sheet_stays_native_and_a_face_is_resized_to_512(graph):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="p")
    sheet = engine_recipe.add_identity_ref(g, "s.png", "sheet")
    face = engine_recipe.add_identity_ref(g, "f.png", "face")
    assert "9646" not in sheet and sheet["9643"]["inputs"]["reference_image"] == ["9645", 0]
    assert face["9646"]["inputs"]["width"] == face["9646"]["inputs"]["height"] == 512
    assert face["9643"]["inputs"]["reference_image"] == ["9646", 0]


def test_a_sheet_only_character_has_no_character_lora_in_the_chain(graph):
    """A character can be a sheet and nothing else. The identity LoRA then follows the
    checkpoint directly -- no character LoRA is loaded, the reference carries the identity."""
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="a woman turns",
                              char_loras=[])
    g = engine_recipe.add_identity_ref(g, "s.png", "sheet")
    assert "9621" not in g and "9622" not in g
    assert g["9641"]["inputs"]["model"] == ["301", 0]
    assert g["9642"]["inputs"]["model"] == ["301", 0]
    assert engine_recipe.lora_stack_note(g).startswith("char none")


def test_content_loras_still_chain_ahead_of_the_reference(graph):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="p", char_loras=[],
                              content_loras=[{"name": "motion"}])
    g = engine_recipe.add_identity_ref(g, "s.png", "sheet")
    assert g["9641"]["inputs"]["model"] == ["9601", 0]


def test_the_prefix_is_not_doubled(graph):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="ref_t2v: already")
    g = engine_recipe.add_identity_ref(g, "s.png", "sheet")
    assert g["121"]["inputs"]["text"] == "ref_t2v: already"


def test_an_unknown_mode_is_refused(graph):
    with pytest.raises(ValueError, match="identity mode"):
        engine_recipe.add_identity_ref(graph, "s.png", "both")


def test_a_template_whose_guiders_moved_is_refused_not_silently_unconditioned(graph):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="p")
    g["103"]["inputs"]["model"] = ["372", 0]
    with pytest.raises(ValueError, match="changed shape"):
        engine_recipe.add_identity_ref(g, "s.png", "sheet")


def test_identity_note_states_presence_and_absence(graph):
    g = engine_recipe.resolve(graph, "kf1.png", 1152, 832, prompt="p")
    assert engine_recipe.identity_note(g) == "identity none"
    note = engine_recipe.identity_note(engine_recipe.add_identity_ref(g, "s.png", "sheet"))
    assert note == ("identity Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors @1.0 "
                    "ref=s.png (native_resolution)")


# ---------------------------------------------------------------- the request

def _req(**kw):
    body = {"prompt": "a woman walks", "keyframes": [], "recipe": "Reg Walk",
            "width": 640, "height": 384, "num_frames": 121, "seed": 7,
            "loras": [{"name": "k3lly2026_v2.safetensors", "strength_stage_1": 0.8,
                       "strength_stage_2": 1.5}]}
    body.update(kw)
    return engine_app.JobRequest(**body)


@pytest.fixture
def no_queue(monkeypatch):
    monkeypatch.setattr(engine_app, "QUEUE", queue.Queue())
    monkeypatch.setattr(engine_app, "JOBS", {})


@pytest.mark.parametrize("kw", [{"identity_ref": "data:image/png;base64,AA=="},
                                {"identity_mode": "sheet"}])
def test_half_a_reference_is_refused(no_queue, kw):
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(**kw))
    assert e.value.status_code == 422 and "go together" in e.value.detail


def test_a_reference_off_the_recipe_path_is_refused(no_queue):
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(recipe=None, identity_ref="data:,", identity_mode="face",
                               keyframes=[{"image": "data:,"}]))
    assert e.value.status_code == 422 and "recipe path" in e.value.detail


def test_an_unknown_mode_is_refused_by_the_model():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _req(identity_ref="data:,", identity_mode="both")


def test_health_advertises_the_feature():
    """The daemon reads this before sending a reference: an engine without it would ignore
    the field (no extra="forbid") and render somebody else."""
    assert "identity_ref" in engine_app.health()["features"]


# ---------------------------------------------------------------- the run

class FakeComfy:
    def __init__(self, *_a, **_k):
        FakeComfy.last = self
        self.uploads, self.graph = {}, None

    def upload_image(self, data, name):
        self.uploads[name] = data
        return name

    def submit(self, graph):
        self.graph = graph
        return "pid"

    def wait(self, *_a, **_k):
        return {}

    @staticmethod
    def output_video(_entry):
        return "out.mp4"

    def view(self, _fn):
        return b"mp4"


@pytest.fixture
def engine_run(monkeypatch, tmp_path):
    loras = tmp_path / "loras"
    loras.mkdir()
    for name in engine_recipe.IDENTITY_LORAS.values():
        (loras / name).write_bytes(b"x")
    monkeypatch.setattr(engine_app, "LORA_DIR", loras)
    monkeypatch.setattr(engine_app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(engine_app, "free_the_gpu", lambda: 99.0)
    monkeypatch.setattr(engine_app.comfy, "Comfy", FakeComfy)
    # The character LoRA's fusion coverage reads the checkpoint; not this file's business.
    monkeypatch.setattr(engine_app, "lora_coverage", lambda *_a: (1, 1))

    def go(req):
        job = engine_app.Job(id="idjob", req=req, placement=engine_app.plan(req))
        engine_app.run_job(job)
        return job
    go.loras = loras
    return go


#: sha256 of the graph origin/main (fcc2c0d, before #156) submitted for `_req()` -- computed
#: by running this exact request through that commit's run_job against FakeComfy. If this
#: moves, a render WITHOUT a reference changed, which this ticket promised it would not.
NO_REF_GRAPH_SHA256 = "935291e08a1e813b78c1dd94bdbfe3aa7e0e8e2a1ed0058d52594cb0140f409c"


def test_without_a_reference_the_graph_is_byte_identical(engine_run, monkeypatch):
    def never(*_a, **_k):
        raise AssertionError("add_identity_ref ran for a render with no reference")
    monkeypatch.setattr(engine_app.recipe_mod, "add_identity_ref", never)
    job = engine_run(_req())
    assert job.status == "Done", job.error
    g = FakeComfy.last.graph
    assert not set(engine_recipe.IDENTITY_NODE_IDS) & set(g)
    assert g["121"]["inputs"]["text"] == "a woman walks"
    digest = hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
    assert digest == NO_REF_GRAPH_SHA256
    assert not any(n.startswith("idref_") for n in FakeComfy.last.uploads)
    assert any("identity none" in n for n in job.notes)


def test_a_sheet_render_is_patched_and_said_so(engine_run):
    sheet = _png(1536, 1024)
    job = engine_run(_req(identity_ref=_data_uri(sheet), identity_mode="sheet"))
    assert job.status == "Done", job.error
    g = FakeComfy.last.graph
    # Uploaded under its content hash, and that is what the LoadImage names.
    name = f"idref_{hashlib.sha256(sheet).hexdigest()[:16]}.png"
    assert FakeComfy.last.uploads[name] == sheet
    assert g["9645"]["inputs"]["image"] == name
    assert g["9643"]["inputs"]["ref_resize_mode"] == "native_resolution"
    assert g["121"]["inputs"]["text"].startswith("ref_t2v: ")
    assert any("Best_FaceID_CharacterSheet_v1.0_LoRA" in n for n in job.notes)
    assert not any("not the 1536x1024" in n for n in job.notes)


def test_an_off_size_sheet_renders_but_is_noted(engine_run):
    job = engine_run(_req(identity_ref=_data_uri(_png(1024, 1024)), identity_mode="sheet"))
    assert job.status == "Done", job.error
    assert any("1024x1024, not the 1536x1024" in n for n in job.notes)


def test_a_face_render_with_no_character_lora(engine_run):
    job = engine_run(_req(identity_ref=_data_uri(_png(800, 900)), identity_mode="face",
                          loras=[]))
    assert job.status == "Done", job.error
    g = FakeComfy.last.graph
    assert "9621" not in g
    assert g["9641"]["inputs"]["lora_name"] == "Best_FaceID_v1.0_LoRA.safetensors"
    assert g["9643"]["inputs"]["ref_resize_mode"] == "match_target"


def test_a_missing_identity_lora_fails_before_submitting(engine_run):
    (engine_run.loras / engine_recipe.IDENTITY_LORAS["sheet"]).unlink()
    FakeComfy.last = None
    job = engine_run(_req(identity_ref=_data_uri(_png(1536, 1024)), identity_mode="sheet"))
    assert job.status == "Failed"
    assert "Best_FaceID_CharacterSheet_v1.0_LoRA" in job.error
    assert FakeComfy.last is None or FakeComfy.last.graph is None


def test_an_unreadable_reference_fails_the_job(engine_run):
    job = engine_run(_req(identity_ref=_data_uri(b"not an image"), identity_mode="sheet"))
    assert job.status == "Failed" and "not a readable image" in job.error


# ---------------------------------------------------------------- the image

def test_both_identity_loras_are_staged_by_download_models():
    """The engine names them; download_models.sh is what puts them on a cold worker. A pod
    without them fails every sheet render at the LoRA check -- held together here because the
    two live in different languages and nothing else connects them."""
    script = (ROOT / "download_models.sh").read_text()
    block = script[script.index("_WANTED=("):]
    block = block[: block.index("\n)\n")]
    rows = re.findall(r'^\s*"([^"]+)"\s*$', block, re.MULTILINE)
    staged = {r.split("|")[1] for r in rows if r.split("|")[0] == "loras"}
    assert set(engine_recipe.IDENTITY_LORAS.values()) <= staged
    for r in rows:
        if r.split("|")[1] in engine_recipe.IDENTITY_LORAS.values():
            assert r.split("|")[2] == "Alissonerdx/LTX-Best-Face-ID"


def test_bfsnodes_is_pinned_at_the_commit_phase_0_proved():
    docker = (ROOT / "Dockerfile").read_text()
    assert "alisson-anjos/ComfyUI-BFSNodes" in docker
    m = re.search(r"^ARG BFSNODES_COMMIT=(\w+)$", docker, re.MULTILINE)
    assert m and m.group(1).startswith("bd23236")
    assert 'checkout "$BFSNODES_COMMIT"' in docker
