"""A zero-filled checkpoint is never published (2026-09-08).

The host hard-reset seconds after the trainer wrote the final ComfyUI-format checkpoint;
ext4 kept the size and zeroed the contents. The "interrupted run" path uploaded it, and every
render with that character then failed inside ComfyUI with a JSON error that named nothing.
"""
import json
import struct
from pathlib import Path

from wanly_worker.services.lora_trainer.poller import Poller


def _good(p: Path) -> None:
    header = json.dumps({"__metadata__": {}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def test_a_real_safetensors_header_passes(tmp_path):
    f = tmp_path / "x.comfy.safetensors"; _good(f)
    assert Poller._readable(f)


def test_a_zero_filled_file_is_refused(tmp_path):
    f = tmp_path / "Me_v2.comfy.safetensors"; f.write_bytes(b"\0" * 4096)
    assert not Poller._readable(f)


def test_a_truncated_header_is_refused(tmp_path):
    f = tmp_path / "x.comfy.safetensors"; f.write_bytes(struct.pack("<Q", 500) + b"{\"a\":")
    assert not Poller._readable(f)


def test_the_sweep_refuses_it_and_records_the_failure():
    import inspect
    src = inspect.getsource(Poller._sweep_checkpoints)
    assert "if not self._readable(path):" in src
    assert "failed.append(str(path))" in src
    # Ordered: only a SETTLED file is judged, so a checkpoint still being written is not
    # mistaken for a corrupt one.
    assert src.index("if self._settled(path):") < src.index("self._readable(path)")
