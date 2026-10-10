"""wanly-gpu-docker#211: the A1111 LoRA-usage reporter's parser and incremental state."""
import importlib.util
import os
import struct
import zlib
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "a1111_lora_usage", Path(__file__).resolve().parents[1] / "deploy" / "a1111-lora-usage.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png(path: Path, params: str | None = None, kind: bytes = b"tEXt") -> Path:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    body = mod.PNG_SIG + _chunk(b"IHDR", ihdr)
    if params is not None:
        if kind == b"tEXt":
            body += _chunk(b"tEXt", b"parameters\x00" + params.encode("latin-1"))
        elif kind == b"iTXt":
            body += _chunk(b"iTXt", b"parameters\x00\x00\x00\x00\x00" + params.encode())
        elif kind == b"zTXt":
            body += _chunk(b"zTXt", b"parameters\x00\x00" + zlib.compress(params.encode()))
    body += _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + _chunk(b"IEND", b"")
    path.write_bytes(body)
    return path


@pytest.mark.parametrize("kind", [b"tEXt", b"iTXt", b"zTXt"])
def test_it_reads_the_parameters_chunk(tmp_path, kind):
    p = _png(tmp_path / "a.png", "jo@na, 1girl <lora:Joana_sdxl_v4_e11:0.8>", kind)
    assert "Joana_sdxl_v4_e11" in mod.png_parameters(p)


def test_text_after_idat_is_never_read(tmp_path):
    p = tmp_path / "late.png"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    p.write_bytes(mod.PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"x")
                  + _chunk(b"tEXt", b"parameters\x00<lora:Late:1>") + _chunk(b"IEND", b""))
    assert mod.png_parameters(p) is None


def test_not_a_png_or_no_parameters(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not a png")
    assert mod.png_parameters(tmp_path / "x.png") is None
    assert mod.png_parameters(_png(tmp_path / "y.png")) is None
    assert mod.png_parameters(tmp_path / "missing.png") is None


def test_one_image_counts_once_per_name():
    text = ("<lora:Joana_sdxl_v4_e11:0.8>, <lora:Joana_sdxl_v4_e11:0.5> <lora:PAWG:1:0.6> "
            "<lora:dmd2_sdxl_4step_lora_fp16>")
    assert mod.loras_in(text) == {"Joana_sdxl_v4_e11", "PAWG", "dmd2_sdxl_4step_lora_fp16"}
    assert mod.loras_in(None) == set()


def test_totals_and_incremental_state(tmp_path):
    d = tmp_path / "out"
    (d / "2026-10-08").mkdir(parents=True)
    a = _png(d / "2026-10-08" / "a.png", "<lora:Joana_sdxl_v4_e11:1>")
    b = _png(d / "2026-10-08" / "b.png", "<lora:Joana_sdxl_v4_e11:1> <lora:PAWG:0.5>")
    os.utime(a, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(b, ns=(1_700_000_100_000_000_000, 1_700_000_100_000_000_000))
    state, read = mod.scan([d], mod.load_state(tmp_path / "none.json"))
    assert read == 2
    t = mod.totals(state)
    assert t["Joana_sdxl_v4_e11"]["images"] == 2 and t["PAWG"]["images"] == 1
    assert t["Joana_sdxl_v4_e11"]["first_used_at"] < t["Joana_sdxl_v4_e11"]["last_used_at"]
    # Unchanged files are not read again; a new one is.
    state, read = mod.scan([d], state)
    assert read == 0
    _png(d / "2026-10-08" / "c.png", "<lora:KimJule_sdxl_v1_final:1>")
    state, read = mod.scan([d], state)
    assert read == 1 and mod.totals(state)["KimJule_sdxl_v1_final"]["images"] == 1
    # A deleted image keeps its contribution: it was made, which is the fact reported.
    a.unlink()
    state, read = mod.scan([d], state)
    assert mod.totals(state)["Joana_sdxl_v4_e11"]["images"] == 2


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    d = tmp_path / "out"
    d.mkdir()
    _png(d / "a.png", "<lora:Brandy_sdxl_v1_e06:1>")
    monkeypatch.setenv("A1111_OUTPUT_DIRS", str(d))
    state = tmp_path / "state.json"
    called = []
    monkeypatch.setattr(mod, "post", lambda *a: called.append(a) or 200)
    assert mod.main(["--dry-run", "--state", str(state)]) == 0
    assert not state.exists() and not called
    assert "Brandy_sdxl_v1_e06" in capsys.readouterr().out


def test_a_real_run_posts_totals_then_saves_state(tmp_path, monkeypatch):
    d = tmp_path / "out"
    d.mkdir()
    _png(d / "a.png", "<lora:Brandy_sdxl_v1_e06:1>")
    env = tmp_path / "worker.env"
    env.write_text("QUEUE_URL=http://api.example:8001\nQUEUE_API_KEY='k'\n")
    monkeypatch.setenv("A1111_OUTPUT_DIRS", str(d))
    monkeypatch.delenv("QUEUE_URL", raising=False)
    monkeypatch.delenv("QUEUE_API_KEY", raising=False)
    sent = []
    monkeypatch.setattr(mod, "post", lambda url, key, body: sent.append((url, key, body)) or 200)
    state = tmp_path / "s" / "state.json"
    assert mod.main(["--state", str(state), "--env", str(env)]) == 0
    url, key, body = sent[0]
    assert url == "http://api.example:8001" and key == "k"
    assert body == [{"name": "Brandy_sdxl_v1_e06", "source": "a1111", "images": 1,
                     "first_used_at": body[0]["first_used_at"],
                     "last_used_at": body[0]["last_used_at"]}]
    assert state.exists()
