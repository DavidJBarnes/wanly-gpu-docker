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


def _proc(tmp_path, procs):
    """A fake /proc: {pid: (rss_kb, argv list)}. Kernel threads have an empty cmdline."""
    root = tmp_path / "proc"
    root.mkdir()
    (root / "self").mkdir()           # non-numeric entries are skipped
    for pid, (rss_kb, argv) in procs.items():
        d = root / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\tx\nVmRSS:\t{rss_kb} kB\n" if rss_kb is not None
                                  else "Name:\tkthread\n")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    return root


def test_an_oom_kill_names_what_is_actually_resident_not_the_captioner(tmp_path, monkeypatch):
    """2026-10-26, 3090b: the message blamed the captioner, which was not running. The other
    big process was the render daemon matting an AR hologram (#223)."""
    import engine.failure as f
    monkeypatch.setattr(f, "CGROUP_DIR", _cgroup(tmp_path, oom_kill=1, cap="27917287424"))
    monkeypatch.setattr(f, "PROC_DIR", _proc(tmp_path, {
        3167: (4_600_000, ["/usr/bin/python3", "-m", "daemon.main"]),
        2476: (360_000, ["python3", "-m", "uvicorn", "wanly_worker.control:app"]),
        3151: (340_000, ["python3", "app.py", "--port", "8190"]),
        2735: (3_000, ["tee", "-a", "/workspace/logs/daemon.log"]),
        2733: (2_000, ["/bin/bash", "/app/start.sh"]),
        7: (None, []),
    }))
    out = f.explain_failure(COMFY_REFUSED, 1216, 832, 241, oom_kills_before=0)
    assert "26 GiB" in out and "ComfyUI was OOM-killed" in out
    assert "captioner" not in out, "no guessing at a process that may not exist"
    assert "4.4 GiB python3 -m daemon.main" in out
    assert out.index("daemon.main") < out.index("uvicorn"), "biggest first"
    assert "start.sh" not in out, "only the top four"


def test_largest_processes_skips_kernel_threads_and_vanished_pids(tmp_path):
    from engine.failure import largest_processes
    root = _proc(tmp_path, {10: (None, []), 11: (2048, ["sleep", "9"])})
    (root / "12").mkdir()             # exited between listdir and read
    assert largest_processes(root) == [(2048 * 1024, "sleep 9")]


def test_an_unlistable_proc_still_explains_the_kill(tmp_path, monkeypatch):
    import engine.failure as f
    monkeypatch.setattr(f, "CGROUP_DIR", _cgroup(tmp_path, oom_kill=1))
    monkeypatch.setattr(f, "PROC_DIR", tmp_path / "absent")
    out = f.explain_failure(COMFY_REFUSED, 1216, 832, 241, oom_kills_before=0)
    assert "ComfyUI was OOM-killed" in out and "could not be listed" in out


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
