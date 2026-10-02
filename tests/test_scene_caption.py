"""scene-caption: JoyCaption, always resident, for the <SCENE> half (wanly-console#572).

Pinned here, without ollama or a GPU:

  * it is a registered service that claims nothing, runs in every mode, and -- with image-edit
    on a standing box (3090b) -- does not push image-edit out of the default mode;
  * the model is checked against the STORE'S OWN manifest, so a hand-imported joycaption is
    used as it is and never rewritten, and ollama is told not to prune the store;
  * the front serves one model, always with keep_alive -1;
  * on a shared card it yields to image-edit: unloads, holds captions back (they wait, then
    503), resumes on request or when the lease runs out; on a dedicated card it never yields;
  * image-edit asks for the card before an edit and hands it back once edits stop.
"""
import asyncio
import json
import os
import subprocess
import time

import pytest

from wanly_worker import registry
from wanly_worker.registry import select_mode
from wanly_worker.services.image_description import model as jc
from wanly_worker.services.image_edit import share
from wanly_worker.services.scene_caption import app as scene_app
from wanly_worker.services.scene_caption import service as scene_svc
from wanly_worker.services.scene_caption import store

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THE_3090 = ["ltx-engine", "lora-trainer", "image-description", "face-crop", "image-edit"]


# ------------------------------------------------------------------------------ registry


class TestItIsAService:
    def test_two_processes_ollama_first(self):
        assert [p.name for p in registry.build(["scene-caption"])] == \
            ["scene-caption-ollama", "scene-caption"]

    def test_it_claims_nothing(self):
        assert registry.kinds_for(["scene-caption"]) == ["service"]
        assert registry.kinds_for(["image-edit", "scene-caption"]) == ["service"]

    def test_3090b_runs_image_edit_and_scene_caption_in_its_default_mode(self):
        """The old rule ran mode-bound services only when nothing else was left, so adding
        scene-caption to SERVICES=image-edit would have silently dropped image-edit."""
        assert select_mode(["image-edit", "scene-caption"], None) == \
            ["image-edit", "scene-caption"]

    def test_a_render_box_still_keeps_image_edit_out_of_render_mode(self):
        names = THE_3090 + ["scene-caption"]
        assert "image-edit" not in select_mode(names, None)
        assert "scene-caption" in select_mode(names, None)

    @pytest.mark.parametrize("mode", ["caption", "edit"])
    def test_it_runs_in_every_mode(self, mode):
        assert "scene-caption" in select_mode(THE_3090 + ["scene-caption"], mode)


# ------------------------------------------------------------------------------ the store


def _store(tmp_path, *, config_digest=None, drop=None, short=None):
    """A fake ollama store holding joycaption's manifest and (tiny stand-ins for) its blobs.
    Sizes in the manifest are the stand-ins' sizes, as for any real manifest."""
    blobs = tmp_path / "models" / "blobs"
    blobs.mkdir(parents=True)
    layers = []
    for i, kind in enumerate(["config", "model", "projector", "template", "system", "params"]):
        digest = (config_digest if kind == "config" and config_digest else f"{i:064x}")
        data = b"x" * (10 + i)
        if kind != drop:
            (blobs / f"sha256-{digest}").write_bytes(data[:-1] if kind == short else data)
        layers.append({"mediaType": kind, "digest": f"sha256:{digest}", "size": len(data)})
    m = tmp_path / "models" / jc.MANIFEST_PATH
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"config": layers[0], "layers": layers[1:]}))
    return str(tmp_path)


class TestTheModelInTheStore:
    def test_a_hand_imported_model_is_present_as_it_is(self, tmp_path):
        """3090b's joycaption has a different config blob from the rebuilt one; its own
        manifest is complete, and that is what counts."""
        s = _store(tmp_path, config_digest="b3f22097ba5bf6c2" + "0" * 48)
        ok, how = store.present(s)
        assert ok, how

    @pytest.mark.parametrize("drop,short", [("model", None), (None, "projector")])
    def test_a_missing_or_short_blob_is_not_present(self, tmp_path, drop, short):
        ok, why = store.present(_store(tmp_path, drop=drop, short=short))
        assert not ok and "missing or the wrong size" in why

    def test_an_empty_store_needs_the_two_ggufs(self, tmp_path):
        ok, _ = store.present(str(tmp_path))
        assert not ok
        assert [b.filename for b in store.to_fetch(str(tmp_path))] == [
            "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf",
            "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"]

    def test_the_cli_prints_what_to_fetch(self, tmp_path):
        r = subprocess.run(["python3", "-m", "wanly_worker.services.scene_caption.store",
                            "check", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 1
        lines = [ln.split() for ln in r.stdout.splitlines()]
        assert [(d, int(n)) for d, n, _ in lines] == [
            (b.digest, b.size) for b in jc.BLOBS if b.remote]
        assert all(u.startswith("https://huggingface.co/") for _, _, u in lines)

    def test_download_models_accepts_a_complete_store_without_fetching(self, tmp_path):
        s = _store(tmp_path)
        r = subprocess.run(["bash", os.path.join(REPO, "download_models.sh"), "--scene-caption"],
                           capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "OLLAMA_STORE": s, "PATH": os.environ["PATH"]})
        assert r.returncode == 0, r.stdout + r.stderr
        assert "scene-caption model OK" in r.stdout

    def test_ollama_never_prunes_the_store_and_keeps_the_model_resident(self):
        env = scene_svc.SceneOllama().env()
        assert env["OLLAMA_NOPRUNE"] == "1"
        assert env["OLLAMA_KEEP_ALIVE"] == "-1"
        assert env["OLLAMA_HOST"].startswith("127.0.0.1:")

    def test_preflight_refuses_a_store_that_is_not_there(self, monkeypatch, tmp_path):
        monkeypatch.setattr(scene_svc, "STORE", str(tmp_path / "nope"))
        with pytest.raises(Exception, match="does not exist"):
            scene_svc.SceneOllama().preflight()

    @pytest.mark.parametrize("raw,want", [("3M", 3 * 1024 ** 2), ("500K", 500 * 1024),
                                          ("", 0.0), ("0", 0.0), ("junk", 0.0)])
    def test_the_rebuild_reads_curls_limit_rate_spelling(self, raw, want):
        assert jc.limit_rate(raw) == want


# ------------------------------------------------------------------------------ the front


class _FakeOllama:
    """httpx.AsyncClient stand-in for the loopback ollama."""
    def __init__(self):
        self.posts = []
        self.resident = True
        self.gate = None

    def __call__(self, *a, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, timeout=None):
        self.posts.append((url, dict(json or {})))
        if json and json.get("keep_alive") == 0:
            self.resident = False
        elif json and json.get("prompt"):
            if self.gate is not None:
                await self.gate.wait()
            self.resident = True
        return _R(200, {"response": "a woman on a pier", "done": True})

    async def get(self, url, timeout=None):
        if url.endswith("/api/ps"):
            return _R(200, {"models": [{"name": scene_app.MODEL}] if self.resident else []})
        return _R(200, {"version": "0.20.2", "models": [{"name": scene_app.MODEL}]})


class _R:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def ollama(monkeypatch):
    import httpx as real_httpx
    fake = _FakeOllama()

    class _Httpx:                   # the app's httpx only; the test's own clients stay real
        AsyncClient = fake
        HTTPError = real_httpx.HTTPError
    monkeypatch.setattr(scene_app, "httpx", _Httpx)
    scene_app.reset()
    yield fake
    scene_app.reset()


def _client():
    from fastapi.testclient import TestClient
    return TestClient(scene_app.app)


def _caption(c, model=scene_app.MODEL, keep_alive="5s"):
    return c.post("/api/generate", json={"model": model, "prompt": "describe", "images": ["x"],
                                         "stream": False, "keep_alive": keep_alive})


class TestTheFront:
    def test_it_forces_keep_alive_minus_one(self, ollama):
        with _client() as c:
            r = _caption(c)
        assert r.status_code == 200 and r.json()["response"] == "a woman on a pier"
        assert ollama.posts[-1][1]["keep_alive"] == -1

    def test_it_serves_one_model_only(self, ollama):
        with _client() as c:
            r = _caption(c, model="gemma4:latest")
        assert r.status_code == 400 and not ollama.posts

    def test_health_says_resident_and_counts(self, ollama):
        with _client() as c:
            _caption(c)
            h = c.get("/health").json()
        assert h["status"] == "ok" and h["resident"] is True and h["captions"] == 1

    def test_a_dedicated_card_never_yields(self, ollama, monkeypatch):
        monkeypatch.setattr(scene_app, "SHARED", False)
        with _client() as c:
            r = c.post("/yield", json={"reason": "an image edit"}).json()
            assert r["yielded"] is False and r["shared"] is False
            assert _caption(c).status_code == 200
        assert not any(p[1].get("keep_alive") == 0 for p in ollama.posts)


class TestASharedCard:
    @pytest.fixture(autouse=True)
    def _shared(self, monkeypatch):
        monkeypatch.setattr(scene_app, "SHARED", True)
        monkeypatch.setattr(scene_app, "YIELD_WAIT_S", 0.3)

    def test_yield_unloads_and_holds_captions_back_until_resume(self, ollama):
        with _client() as c:
            r = c.post("/yield", json={"reason": "an image edit", "hold_s": 60}).json()
            assert r["yielded"] is True and r["unloaded"] is True
            assert ollama.posts[-1][1] == {"model": scene_app.MODEL, "prompt": "",
                                           "keep_alive": 0}
            h = c.get("/health").json()
            assert h["yielded"] is True and h["yielded_to"] == "an image edit"
            # A caption while the card is lent out waits, then is refused -- never loaded.
            r = _caption(c)
            assert r.status_code == 503 and "lent its card" in r.json()["detail"]
            assert not ollama.resident
            assert c.post("/resume").json()["resumed"] is True
            assert _caption(c).status_code == 200
        # Resume reloads it (keep_alive -1, empty prompt) without waiting for a caption.
        assert {"model": scene_app.MODEL, "prompt": "", "keep_alive": -1} in \
            [p[1] for p in ollama.posts]

    def test_a_waiting_caption_goes_as_soon_as_the_card_is_back(self, ollama, monkeypatch):
        monkeypatch.setattr(scene_app, "YIELD_WAIT_S", 5)

        async def go():
            import httpx
            transport = httpx.ASGITransport(app=scene_app.app)
            scene_app.reset()
            async with httpx.AsyncClient(transport=transport, base_url="http://s") as c:
                await c.post("/yield", json={"hold_s": 60})
                pending = asyncio.ensure_future(c.post("/api/generate", json={
                    "model": scene_app.MODEL, "prompt": "p", "images": []}))
                await asyncio.sleep(0.2)
                assert not pending.done()
                await c.post("/resume")
                r = await asyncio.wait_for(pending, 2)
                assert r.status_code == 200
        asyncio.run(go())

    def test_a_second_yield_renews_the_lease_without_unloading_again(self, ollama):
        with _client() as c:
            c.post("/yield", json={"hold_s": 60})
            n = len(ollama.posts)
            r = c.post("/yield", json={"hold_s": 60}).json()
            assert r["renewed"] is True and len(ollama.posts) == n
            assert c.get("/health").json()["yields"] == 1

    def test_the_lease_running_out_gives_the_card_back(self, ollama, monkeypatch):
        monkeypatch.setattr(scene_app, "LEASE_CHECK_S", 0.05)
        with _client() as c:
            c.post("/yield", json={"hold_s": 0.1})
            deadline = time.time() + 3
            while c.get("/health").json()["yielded"] and time.time() < deadline:
                time.sleep(0.05)
            h = c.get("/health").json()
        assert h["yielded"] is False and "lease ran out" in h["last_resume"]["why"]


# ------------------------------------------------------------------------------ image-edit


class TestImageEditAsksForTheCard:
    def test_the_url_comes_from_the_flag_and_the_services_line(self, monkeypatch):
        monkeypatch.delenv("IMAGE_EDIT_SCENE_CAPTION_URL", raising=False)
        monkeypatch.setenv("SERVICES", "image-edit,scene-caption")
        monkeypatch.setenv("SCENE_CAPTION_SHARED", "1")
        assert share._scene_url_from_env() == "http://127.0.0.1:11436"
        monkeypatch.setenv("SCENE_CAPTION_SHARED", "0")
        assert share._scene_url_from_env() == ""
        monkeypatch.setenv("SCENE_CAPTION_SHARED", "1")
        monkeypatch.setenv("SERVICES", "image-edit")
        assert share._scene_url_from_env() == ""
        monkeypatch.setenv("IMAGE_EDIT_SCENE_CAPTION_URL", "http://other:11436/")
        assert share._scene_url_from_env() == "http://other:11436"

    def test_the_card_goes_back_only_after_edits_stop(self, monkeypatch):
        monkeypatch.setattr(share, "SCENE_RESUME_IDLE_S", 60)
        assert share.scene_resume_reason(scene_yielded=True, busy=False, idle_s=61)
        assert share.scene_resume_reason(scene_yielded=True, busy=False, idle_s=5) is None
        assert share.scene_resume_reason(scene_yielded=True, busy=True, idle_s=600) is None
        assert share.scene_resume_reason(scene_yielded=False, busy=False, idle_s=600) is None

    def test_a_scene_captioner_that_does_not_answer_never_blocks_an_edit(self, monkeypatch):
        monkeypatch.setattr(share, "SCENE_CAPTION_URL", "http://127.0.0.1:1")
        state = {}
        note = asyncio.run(share.yield_scene(state))
        assert "editing anyway" in note and state["scene_yielded"] is False

    def test_nothing_is_asked_on_a_box_without_it(self, monkeypatch):
        monkeypatch.setattr(share, "SCENE_CAPTION_URL", "")
        assert asyncio.run(share.yield_scene({})) is None
        assert asyncio.run(share.resume_scene({})) is False
        assert share.shares_with_scene() is False

    def test_the_watcher_unloads_qwen_before_handing_the_card_back(self, monkeypatch):
        from wanly_worker.services.image_edit import app as edit_app
        calls = []

        async def unload(url, client=None):
            calls.append("unload")
            return True

        async def resume(state, client=None):
            calls.append("resume")
            state["scene_yielded"] = False
            return True
        monkeypatch.setattr(share, "SCENE_CAPTION_URL", "http://127.0.0.1:11436")
        monkeypatch.setattr(share, "SCENE_RESUME_IDLE_S", 0)
        monkeypatch.setattr(share, "WATCH_S", 0.01)
        monkeypatch.setattr(share, "unload_qwen", unload)
        monkeypatch.setattr(share, "resume_scene", resume)
        monkeypatch.setattr(edit_app, "_state", {"last_edit_at": 0, "last": None,
                                                 "model_loaded": True, "edits": 1,
                                                 "waiting": None, "unloads": 0,
                                                 "last_unload": None, "scene_yielded": True})

        async def run():
            edit_app._turn = asyncio.Lock()
            task = asyncio.create_task(edit_app._watch())
            for _ in range(100):
                await asyncio.sleep(0.01)
                if "resume" in calls:
                    break
            task.cancel()
        asyncio.run(run())
        assert calls[:2] == ["unload", "resume"]
        assert edit_app._state["model_loaded"] is False


class TestTheStoreCheckNeedsOnlyTheStdlib:
    def test_it_imports_without_httpx_or_fastapi(self, tmp_path):
        """It runs on the host (download_models.sh --scene-caption), where neither is
        installed. 3090b's first run of it died on `import httpx` via the package __init__."""
        code = ("import sys\n"
                "class _Block:\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in ('httpx', 'fastapi', 'uvicorn'):\n"
                "            raise ImportError('blocked: ' + name)\n"
                "sys.meta_path.insert(0, _Block())\n"
                "from wanly_worker.services.scene_caption import store\n"
                f"print(store.present({str(tmp_path)!r})[0])\n")
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "False"
