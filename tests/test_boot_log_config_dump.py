"""The daemon's resolved config is printed at boot, and it must not lie or leak.

It used to be a sed pipeline in start.sh; it is `render_env_dump()` in the render-daemon
service now (wanly-gpu-docker#83), printed right after fetch_daemon.sh writes the .env. The
four properties are the same ones the pipeline was tested for.
"""
import pathlib

from wanly_worker.services.ltx_engine import render_env_dump

HERE = pathlib.Path(__file__).parent

SAMPLE = """QUEUE_URL=http://api.wanly22.com:8001
FRIENDLY_NAME=3090.zero
ENGINE=ltx
LTX_ENGINE_URL=http://localhost:8190
COMFYUI_URL=http://localhost:8188
# EMPTY on purpose. ... it used to print
#     ERROR Resource rife49.pth: download failed ... HTTP 404
# -- so every healthy boot printed lines that grep as ERROR.
COMFYUI_PATH=
LORA_CACHE_DIR=/workspace/models/loras
RUNPOD_API_KEY=supersecret-runpod
QUEUE_API_KEY=supersecret-queue
"""


def test_no_line_of_the_dump_reads_as_an_error():
    """A comment quoting an old ERROR line was read as a live failure once (daemon#175)."""
    for line in render_env_dump(SAMPLE).splitlines():
        assert "ERROR" not in line


def test_comments_are_stripped():
    assert "#" not in render_env_dump(SAMPLE)


def test_every_key_survives_including_the_empty_one():
    out = render_env_dump(SAMPLE)
    for key in ("QUEUE_URL", "FRIENDLY_NAME", "ENGINE", "COMFYUI_PATH", "LORA_CACHE_DIR",
                "RUNPOD_API_KEY", "QUEUE_API_KEY"):
        assert f"  {key}=" in out, key
    assert "  COMFYUI_PATH=" in out.splitlines()


def test_secrets_are_redacted_and_nothing_else_is():
    out = render_env_dump(SAMPLE)
    assert out.count("<redacted>") == 2
    assert "supersecret" not in out


def test_fetch_daemon_writes_the_env_and_the_service_prints_it():
    fetch = (HERE.parent / "fetch_daemon.sh").read_text()
    assert 'cat > "$DAEMON_DIR/.env" << EOF' in fetch
    assert "COMFYUI_PATH=" in fetch and "ENGINE=ltx" in fetch
    import inspect
    from wanly_worker.services.ltx_engine import RenderDaemon
    src = inspect.getsource(RenderDaemon.preflight)
    assert "render_env_dump(env.read_text())" in src
