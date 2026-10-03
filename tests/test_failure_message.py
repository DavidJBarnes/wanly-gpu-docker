"""The engine's failure rewrite tells lost GPU access apart from a full card (#95)."""
from engine.failure import explain_failure, lost_gpu_access

REAL_OOM = ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total "
            "capacity of 23.56 GiB of which 1.10 GiB is free. Including non-PyTorch memory, this "
            "process has 21.9 GiB memory in use.")
LOST_ACCESS = ("ComfyError: execution failed: Allocation on device 0 would exceed allowed memory. "
               "(out of memory)\nCurrently allocated     : 0 bytes\nRequested               : "
               "384.00 KiB\nDevice limit            : 23.56 GiB\nFree (according to CUDA): 23.28 GiB")


def test_a_real_oom_still_says_reduce_the_render():
    out = explain_failure(REAL_OOM, 1216, 832, 241)
    assert "1216x832 at 241 frames did not fit" in out
    assert "device access" not in out


def test_zero_bytes_allocated_is_lost_access_not_a_full_card():
    """The 2026-09-10 signature: first 384 KiB refused with 23 GiB free."""
    out = explain_failure(LOST_ACCESS, 1216, 832, 241)
    assert "lost GPU device access" in out
    assert "/dev/nvidiactl" in out
    assert "did not fit" not in out, "blamed the resolution for a card it could not reach"


def test_a_fresh_process_with_no_device_is_lost_access():
    assert lost_gpu_access("RuntimeError: No CUDA GPUs are available")
    assert lost_gpu_access("Failed to initialize NVML: Unknown Error")
    assert not lost_gpu_access(REAL_OOM)


def test_anything_else_is_passed_through():
    assert explain_failure("RuntimeError: ComfyUI finished but produced no video", 704, 1280, 241) \
        == "RuntimeError: ComfyUI finished but produced no video"


# --- ComfyUI vanishing mid-render (#166) ----------------------------------------------------

# Verbatim from 3090a, 2026-10-03 16:22:57Z: ComfyUI OOM-killed at the 54 GiB container cap.
COMFY_REFUSED = ("ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8188): Max retries "
                 "exceeded with url: /history/019f50ff-d01c-4d82-89f9-6ebbe2b206f0 (Caused by "
                 "NewConnectionError(\"HTTPConnection(host='127.0.0.1', port=8188): Failed to "
                 "establish a new connection: [Errno 111] Connection refused\"))")


def _cgroup(tmp_path, oom_kill: int, cap: str = "57982058496"):
    (tmp_path / "memory.events").write_text(
        f"low 0\nhigh 0\nmax 2102\noom 0\noom_kill {oom_kill}\noom_group_kill 0\n")
    (tmp_path / "memory.max").write_text(cap + "\n")
    return tmp_path


def test_refused_connection_with_a_new_oom_kill_blames_the_memory_cap(tmp_path, monkeypatch):
    import engine.failure as f
    monkeypatch.setattr(f, "CGROUP_DIR", _cgroup(tmp_path, oom_kill=2))
    out = f.explain_failure(COMFY_REFUSED, 1216, 832, 241, oom_kills_before=0)
    assert out.startswith(COMFY_REFUSED), "the original error must survive"
    assert "ComfyUI was OOM-killed" in out
    assert "54 GiB" in out and "2 process(es)" in out
    assert "did not fit" not in out, "a host-RAM kill is not a VRAM problem"


def test_refused_connection_without_an_oom_kill_says_it_was_not_the_cap(tmp_path, monkeypatch):
    import engine.failure as f
    monkeypatch.setattr(f, "CGROUP_DIR", _cgroup(tmp_path, oom_kill=3))
    out = f.explain_failure(COMFY_REFUSED, 1216, 832, 241, oom_kills_before=3)
    assert "NOT OOM-killed" in out and "comfyui.log" in out


def test_unreadable_cgroup_says_so_rather_than_guessing(tmp_path, monkeypatch):
    import engine.failure as f
    monkeypatch.setattr(f, "CGROUP_DIR", tmp_path / "absent")
    out = f.explain_failure(COMFY_REFUSED, 1216, 832, 241, oom_kills_before=None)
    assert "could not be ruled in or out" in out


def test_uncapped_container_reads_as_no_cap(tmp_path):
    from engine.failure import memory_cap, oom_kills
    d = _cgroup(tmp_path, oom_kill=1, cap="max")
    assert memory_cap(d) is None and oom_kills(d) == 1


def test_an_interrupt_is_not_reported_as_comfyui_dying():
    msg = "ComfyError: prompt is neither queued nor in history — interrupted or dropped by ComfyUI"
    assert explain_failure(msg, 704, 1280, 241) == msg
