"""keyframe-server is gone, and nothing may quietly bring it back (#44).

WHY THIS IS A TEST AND NOT JUST A DELETION

The dependency survived a month after it stopped working, because the code path that used it
could not be reached and nothing said so. `KEYFRAME_URL` defaulted to `127.0.0.1:8189`, which
inside this container is our own loopback -- so `POST /free` connection-refused on every render
and was skipped in silence. #45 had made a missing collaborator a no-op, which was the right
fix for the crash and also the thing that hid the misconfiguration.

The lesson is not "delete it". It is that a dead call is invisible, so the absence has to be
asserted rather than assumed. If a VRAM-hungry neighbour returns to this card -- Qwen image
edit under wanly-console#426 -- it needs a real yield contract, and this test failing is how
someone learns that reinstating the old one is not it.
"""
import ast
import pathlib
import re

ENGINE = pathlib.Path(__file__).parent.parent / "engine" / "app.py"
SRC = ENGINE.read_text()

def _code_only(src: str) -> str:
    """Source with comments AND docstrings removed.

    Prose may discuss the retired service -- the module docstring explains at length why the
    call is gone, and that explanation is the most valuable part of this change. Only CODE
    must be free of it, so the two have to be told apart. Line-based comment stripping is not
    enough: the explanation lives in a docstring, which is a string literal, not a comment.
    """
    lines = src.splitlines()
    drop = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            expr = node.body[0]
            drop.update(range(expr.lineno, (expr.end_lineno or expr.lineno) + 1))
    return "\n".join(
        line for n, line in enumerate(lines, start=1)
        if n not in drop and not line.lstrip().startswith("#")
    )


CODE = _code_only(SRC)


def test_no_keyframe_url_remains():
    assert "KEYFRAME_URL" not in CODE, \
        "KEYFRAME_URL is back — a returning neighbour needs a contract, not this"


def test_nothing_posts_free_to_another_service():
    """The COMFY_URL /free below it stays: that frees our OWN ComfyUI, in-process, and is a
    different thing entirely. Only a foreign /free is forbidden."""
    frees = re.findall(r'requests\.post\(f?"\{(\w+)\}/free"', CODE)
    assert set(frees) <= {"COMFY_URL"}, f"posts /free to {set(frees) - {'COMFY_URL'}}"


def test_health_no_longer_advertises_it():
    """Two fields that reported on a service this container has no relationship with. They
    read as diagnostics and were in fact always None, which is worse than absent."""
    assert "keyframe_server" not in CODE
    assert "keyframe_vram_free_gb" not in CODE


def test_the_startup_banner_is_clean():
    banner = [l for l in SRC.splitlines() if "ltx-engine on {args.host}" in l]
    assert banner, "startup banner moved — this test cannot see it any more"
    tail = SRC[SRC.index(banner[0]):SRC.index(banner[0]) + 300]
    assert "keyframe" not in tail.lower()


def test_our_own_comfyui_is_still_freed():
    """The point was to remove a dependency, not the VRAM management. Deleting this would
    turn a tight-memory render into an OOM instead of a reload."""
    assert 'requests.post(f"{COMFY_URL}/free"' in CODE
    assert "MIN_FREE_GB" in CODE


def test_free_the_gpu_still_measures_before_deciding():
    assert "free = comfy_vram_free_gb()" in CODE
