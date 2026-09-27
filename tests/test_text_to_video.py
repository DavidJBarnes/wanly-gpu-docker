"""Text-to-video on the recipe path (#145): a render with NO start frame.

The regularization pool for character training is rendered from prompts alone, and the daemon
already sends `keyframes: []` with width/height for a segment that has no start image. Before
this the engine refused that shape outright (keyframes min_length=1), and even past the model
the recipe path would have died on `guides[0]` and on deriving a size from a frame that does
not exist.

The recipe graph already carries the switch: node 290, "Text To Video (no image ref)", which
bypasses both LTXVImgToVideoInplace nodes. What these tests hold:

  * the switch is set from the request, and an image render's graph is byte-unchanged;
  * the size is the REQUEST's -- explicit, /64, never the model's 512x768 default -- and it
    reaches the graph through the only door it has, the LoadImage the latent size is derived
    from, via a blank frame of exactly that size;
  * the free-form path, which has no such switch, refuses empty keyframes by name.
"""
import io
import json
import pathlib
import sys

import pytest
from fastapi import HTTPException
from PIL import Image

ENGINE = pathlib.Path(__file__).parent.parent / "engine"
WORKFLOW = ENGINE / "workflows/ltx23_recipe.api.json"

# The engine is a flat module tree (it runs as `python app.py` from engine/), so its modules
# import each other by bare name. Importing app has no side effects: the worker thread and
# the jobs directory are created under __main__ only.
sys.path.insert(0, str(ENGINE))
import app as engine_app  # noqa: E402
import recipe as engine_recipe  # noqa: E402


@pytest.fixture
def graph():
    return json.loads(WORKFLOW.read_text())


BASE = dict(image_name="kf1.png", width=832, height=1216, prompt="a prompt")


# ---------------------------------------------------------------- the resolver

def test_the_switch_is_the_node_the_graph_titles_text_to_video(graph):
    """Pin the node id to its meaning: if the template is re-exported and 290 becomes
    something else, this must fail rather than resolve() flipping an unrelated boolean."""
    assert graph["290"]["class_type"] == "PrimitiveBoolean"
    assert graph["290"]["_meta"]["title"] == "Text To Video (no image ref)"
    for nid in ("160", "161"):
        assert graph[nid]["class_type"] == "LTXVImgToVideoInplace"
        assert graph[nid]["inputs"]["bypass"] == ["290", 0]


def test_text_to_video_sets_the_switch(graph):
    g = engine_recipe.resolve(graph, **BASE, text_to_video=True)
    assert g["290"]["inputs"]["value"] is True


def test_an_image_render_leaves_it_off_and_hashes_as_before(graph):
    """The template ships the switch off, so writing False is not a change: an image
    render's graph -- and so its regression hash -- is exactly what it was."""
    before = engine_recipe.graph_hash(engine_recipe.resolve(graph, **BASE))
    g = engine_recipe.resolve(graph, **BASE, text_to_video=False)
    assert g["290"]["inputs"]["value"] is False
    assert engine_recipe.graph_hash(g) == before


def test_a_template_left_switched_on_is_switched_off_for_an_image_render(graph):
    graph["290"]["inputs"]["value"] = True
    assert engine_recipe.resolve(graph, **BASE)["290"]["inputs"]["value"] is False


def test_the_size_goes_where_the_latent_size_comes_from(graph):
    """292/293 feed the resize of the LoadImage, and that resized image is what the latent's
    size is read off. A blank frame at exactly this size keeps the resize an identity."""
    g = engine_recipe.resolve(graph, "t2v.png", 640, 384, prompt="p", text_to_video=True)
    assert (g["292"]["inputs"]["value"], g["293"]["inputs"]["value"]) == (640, 384)
    assert g["165"]["inputs"]["width"] == ["292", 0]
    assert g["165"]["inputs"]["image"] == ["167", 0]
    assert g["167"]["inputs"]["image"] == "t2v.png"


# ---------------------------------------------------------------- the request

def _req(**kw):
    body = {"prompt": "a woman walks", "keyframes": [], "recipe": "Reg Walk"}
    body.update(kw)
    return engine_app.JobRequest(**body)


@pytest.fixture
def no_queue(monkeypatch):
    """submit() queues the job; nothing here should leave one behind for another test."""
    import queue
    monkeypatch.setattr(engine_app, "QUEUE", queue.Queue())
    monkeypatch.setattr(engine_app, "JOBS", {})


def test_the_model_accepts_no_keyframes():
    assert _req(width=512, height=768).keyframes == []


def test_keyframes_is_still_a_required_key():
    """Forgetting the field is a bug, not a request for text-to-video."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        engine_app.JobRequest(prompt="p", recipe="r")


def test_a_text_to_video_request_is_queued(no_queue):
    out = engine_app.submit(_req(width=640, height=384))
    assert out["placement"] == []
    assert engine_app.QUEUE.qsize() == 1


def test_the_free_form_path_refuses_no_keyframes_by_name(no_queue):
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(recipe=None, width=512, height=768))
    assert e.value.status_code == 422 and "recipe path" in e.value.detail
    assert engine_app.QUEUE.qsize() == 0


@pytest.mark.parametrize("given", [{}, {"width": 512}, {"height": 768}])
def test_the_size_must_be_sent_not_defaulted(no_queue, given):
    """512x768 is the model default. With no frame to derive from, a caller who forgot the
    size would get a portrait thumbnail and nothing would look wrong."""
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(**given))
    assert e.value.status_code == 422 and "width and height must be sent" in e.value.detail


def test_the_size_is_checked_against_the_64_grid(no_queue):
    """The same rule as every render: the two-stage graph halves the size for stage 1."""
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(width=544, height=768))
    assert "divisible by 64" in e.value.detail


def test_a_zero_size_is_refused(no_queue):
    """0 is divisible by 64; on this path the request's number IS the size."""
    with pytest.raises(HTTPException) as e:
        engine_app.submit(_req(width=0, height=768))
    assert "at least 64" in e.value.detail


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
    monkeypatch.setattr(engine_app, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(engine_app, "free_the_gpu", lambda: 99.0)
    monkeypatch.setattr(engine_app.comfy, "Comfy", FakeComfy)

    def no_derive(_path):
        raise AssertionError("text-to-video has no start frame to derive a size from")
    monkeypatch.setattr(engine_app, "derive_size", no_derive)

    def go(req):
        job = engine_app.Job(id="t2vjob", req=req, placement=engine_app.plan(req))
        engine_app.run_job(job)
        return job
    return go


def test_a_text_to_video_render_runs_at_the_requested_size(engine_run):
    job = engine_run(_req(width=640, height=384, num_frames=121, seed=7))
    assert job.status == "Done", job.error
    g = FakeComfy.last.graph
    assert g["290"]["inputs"]["value"] is True
    assert (g["292"]["inputs"]["value"], g["293"]["inputs"]["value"]) == (640, 384)
    # The LoadImage names the blank frame that was uploaded, at exactly the clip size.
    name = g["167"]["inputs"]["image"]
    with Image.open(io.BytesIO(FakeComfy.last.uploads[name])) as im:
        assert im.size == (640, 384)
    assert (job.req.width, job.req.height) == (640, 384)
    assert any("text-to-video" in n for n in job.notes)


def test_an_image_render_still_derives_its_size_and_leaves_the_switch_off(
        engine_run, monkeypatch):
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (900, 1300), "white").save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    monkeypatch.setattr(engine_app, "derive_size", lambda p: (832, 1216))
    job = engine_run(_req(keyframes=[{"image": uri}]))
    assert job.status == "Done", job.error
    g = FakeComfy.last.graph
    assert g["290"]["inputs"]["value"] is False
    assert (g["292"]["inputs"]["value"], g["293"]["inputs"]["value"]) == (832, 1216)
    assert any("image-to-video" in n for n in job.notes)
