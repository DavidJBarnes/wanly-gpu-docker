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
