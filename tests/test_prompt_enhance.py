"""prompt-enhance: the Sulphur-2 prompt enhancer, loaded per request (#225).

Pinned here, without llama.cpp or a GPU (a stand-in llama-server answers instead):

  * it is a registered service that claims nothing, never runs beside a render, and is left
    out of edit and train like the captioner;
  * scene-caption on the same SERVICES line becomes SHARED, so it yields the card;
  * a request starts llama-server (yielding scene-caption first), a second reuses it, and an
    idle spell stops it and hands the card back;
  * the request is the model card's: text + image, no system prompt; <think> is stripped;
  * missing files refuse with 503 rather than starting a server with nothing to load.
"""
import base64
import importlib
import os
import stat
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from wanly_worker import registry
from wanly_worker.registry import select_mode
from wanly_worker.services.prompt_enhance import app as pe_app
from wanly_worker.services.prompt_enhance import model

THE_3090 = ["ltx-engine", "lora-trainer", "image-description", "face-crop", "image-edit"]


class TestRegistry:
    def test_registered_and_claims_nothing(self):
        assert [p.name for p in registry.build(["prompt-enhance"])] == ["prompt-enhance"]
        assert registry.kinds_for(["scene-caption", "prompt-enhance"]) == ["service"]

    def test_the_scene_box_runs_both(self):
        assert select_mode(["scene-caption", "prompt-enhance"], None) == \
            ["scene-caption", "prompt-enhance"]

    def test_never_beside_a_render(self):
        assert "prompt-enhance" not in select_mode(THE_3090 + ["prompt-enhance"], None)

    @pytest.mark.parametrize("mode", ["edit", "train"])
    def test_not_in_edit_or_train(self, mode):
        assert "prompt-enhance" not in select_mode(THE_3090 + ["prompt-enhance"], mode)

    def test_scene_caption_shares_its_card_when_enhance_is_beside_it(self, monkeypatch):
        from wanly_worker.services.scene_caption import app as scene_app
        monkeypatch.setenv("SERVICES", "scene-caption,prompt-enhance")
        monkeypatch.delenv("SCENE_CAPTION_SHARED", raising=False)
        try:
            assert importlib.reload(scene_app).SHARED is True
            monkeypatch.setenv("SERVICES", "scene-caption")
            assert importlib.reload(scene_app).SHARED is False
        finally:
            monkeypatch.delenv("SERVICES", raising=False)
            importlib.reload(scene_app)


class TestModelPins:
    def test_two_files_pinned_by_digest(self):
        assert {f.kind for f in model.FILES} == {"weights", "mmproj"}
        for f in model.FILES:
            assert len(f.digest) == 64 and f.size > 100_000_000
        assert len(model.HF_REVISION) == 40

    def test_missing_by_size(self, tmp_path):
        assert len(model.missing(str(tmp_path))) == 2
        for f in model.FILES:
            with open(tmp_path / f.local, "wb") as fh:
                fh.truncate(f.size)
        assert model.missing(str(tmp_path)) == []
        with open(tmp_path / model.MMPROJ.local, "wb") as fh:
            fh.truncate(10)
        assert model.missing(str(tmp_path)) == [model.MMPROJ]


FAKE_SERVER = textwrap.dedent('''\
    #!{python}
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(sys.argv[sys.argv.index("--port") + 1])
    log = {log!r}
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self._send(200, {{"status": "ok"}})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(log, "a") as fh:
                fh.write(json.dumps({{"argv": sys.argv[1:], "body": body}}) + "\\n")
            if body.get("max_tokens") == 1:   # the reasoning ran out the budget
                self._send(200, {{"choices": [{{"message": {{"content": "",
                    "reasoning_content": "The user wants"}}, "finish_reason": "length"}}]}})
                return
            self._send(200, {{"choices": [{{"message": {{"content":
                "<think>\\nhmm\\n</think>\\n\\nA woman turns toward the camera."}},
                "finish_reason": "stop"}}], "usage": {{"completion_tokens": 7}}}})
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def served(tmp_path, monkeypatch):
    import json
    import socket
    log = tmp_path / "requests.jsonl"
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE_SERVER.format(python=sys.executable, log=str(log)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    for f in model.FILES:
        with open(tmp_path / f.local, "wb") as fh:
            fh.truncate(f.size)
    monkeypatch.setattr(model, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(pe_app, "LLAMA_SERVER", str(exe))
    monkeypatch.setattr(pe_app, "LLAMA_PORT", port)
    monkeypatch.setattr(pe_app, "LLAMA", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(pe_app, "LOG_PATH", str(tmp_path / "llama.log"))
    calls = []

    async def fake_yield(client):
        calls.append("yield")
        pe_app._state["scene_yielded"] = True

    async def fake_resume():
        if pe_app._state["scene_yielded"]:
            calls.append("resume")
        pe_app._state["scene_yielded"] = False

    monkeypatch.setattr(pe_app, "_yield_scene", fake_yield)
    monkeypatch.setattr(pe_app, "_resume_scene", fake_resume)
    pe_app.reset()
    with TestClient(pe_app.app) as client:
        yield client, calls, (lambda: [json.loads(l) for l in log.read_text().splitlines()])
    pe_app.reset()


class TestEnhance:
    def test_cold_then_warm_then_idle_unload(self, served):
        client, calls, requests = served
        img = base64.b64encode(b"\xff\xd8 not really a jpeg").decode()
        r = client.post("/enhance", json={"text": "a woman on a sofa", "image_b64": img})
        assert r.status_code == 200, r.text
        assert r.json()["prompt"] == "A woman turns toward the camera."
        assert r.json()["cold"] is True
        assert calls == ["yield"]
        r2 = client.post("/enhance", json={"text": "again"})
        assert r2.json()["cold"] is False
        assert client.get("/health").json()["loads"] == 1

        sent = requests()[0]
        assert "--mmproj" in sent["argv"] and "--jinja" in sent["argv"]
        msgs = sent["body"]["messages"]
        assert [m["role"] for m in msgs] == ["user"], "no system prompt, per the model card"
        content = msgs[0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[1] == {"type": "text", "text": "a woman on a sofa"}
        assert requests()[1]["body"]["messages"][0]["content"] == "again"

        assert client.post("/unload").json() == {"unloaded": True}
        assert calls[-1] == "resume"
        assert client.get("/health").json()["loaded"] is False

    def test_missing_files_refuse(self, served, tmp_path):
        client, calls, _ = served
        os.remove(tmp_path / model.WEIGHTS.local)
        r = client.post("/enhance", json={"text": "x"})
        assert r.status_code == 503 and "not in" in r.json()["detail"]

    def test_reasoning_that_runs_out_is_an_error_not_an_empty_prompt(self, served):
        client, _, _ = served
        r = client.post("/enhance", json={"text": "x", "max_tokens": 1})
        assert r.status_code == 502 and "max_tokens" in r.json()["detail"]
        assert client.get("/health").json()["failures"] == 1

    def test_needs_something_to_enhance(self, served):
        client, _, _ = served
        assert client.post("/enhance", json={}).status_code == 400
        assert client.post("/enhance", json={"image_b64": "@@@"}).status_code == 400


def test_strip_think():
    assert pe_app.strip_think("<think>a</think>\n\nB") == "B"
    assert pe_app.strip_think("plain") == "plain"
    assert pe_app.strip_think("dangling</think> C") == "C"
