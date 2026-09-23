"""#125: a swap that fails must leave the CURRENT code in place.

The 3090 crash-looped every boot on 2026-09-22 with
`python3: can't open file '/opt/engine/app.py'`. The honest swap of #121 guarded every mv
— and was still destroyed by the first one: GNU mv, on EBUSY/EXDEV, falls back to copying
file by file and unlinking the source, failing only when it reaches the bind mount that was
the problem all along. The "keeping current engine" branch printed its reassuring line over
an /opt/engine it had just hollowed out. restart=always on an emptied directory is a crash
loop that says "keeping current code" on every lap.

A rename cannot replace a rename here either (#125b): directories COPY'd by the image live
on the overlay LOWER layer, and a lower-layer directory cannot be renamed at all — measured
EXDEV on a plain, unmounted /opt/worker. os.rename would refuse every production boot. The
mv copy-fallback was the only thing that ever made moving "work", and it is exactly the
mechanism that hollows the directory out.

The fix is swap_sync.py: stage to a temp dir, then replace files IN PLACE (per-file
os.replace, atomic on overlayfs). It refuses a mount at or inside the destination BEFORE
touching anything, so a failed swap leaves the current code exactly as it was.

Part 1 tests the primitive directly (mount detection is monkeypatched; the real filesystem
semantics are not). Part 2 reproduces the real condition — a read-only bind mount INSIDE
the swap target — in a docker container, the only faithful reproduction (user namespaces do
not reproduce it: an internally-created mount is not pinned the way dockerd's external
mount is), and runs the actual fetch_engine.sh boot path end to end.

Everything in part 2 is asserted inside the one container run that did the swap: the swapped
filesystem lives in that container's writable layer, so a second `docker run` would show the
baked image again, not the aftermath.
"""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fetch_engine.sh"
SWAP_SYNC = ROOT / "swap_sync.py"


def load_swap_sync():
    spec = importlib.util.spec_from_file_location("swap_sync", SWAP_SYNC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- part 1: the primitive

class TestThePrimitive:
    def _staged(self, tmp_path, files):
        staged = tmp_path / "staged"
        for rel, text in files.items():
            p = staged / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return staged

    def test_sync_replaces_files_adds_new_and_removes_stale(self, tmp_path):
        mod = load_swap_sync()
        dst = tmp_path / "pkg"
        (dst / "sub").mkdir(parents=True)
        (dst / "app.py").write_text("OLD")
        (dst / "sub" / "gone.py").write_text("OLD")
        staged = self._staged(tmp_path, {"app.py": "NEW", "sub/deep/new.py": "NEW"})
        assert mod.main(str(staged), str(dst)) == 0
        assert (dst / "app.py").read_text() == "NEW"
        assert (dst / "sub" / "deep" / "new.py").read_text() == "NEW"
        assert not (dst / "sub" / "gone.py").exists(), "stale file survived the sync"
        assert not (dst / "sub").exists() or (dst / "sub").is_dir()

    def test_a_mount_inside_the_destination_is_refused_before_any_change(self, tmp_path,
                                                                         monkeypatch):
        """THE regression. The mount check must fire before a single file is replaced or
        removed, and with a nonzero exit — the caller keeps the current code. mv's failure
        was that the destruction and the refusal were the same operation."""
        mod = load_swap_sync()
        dst = tmp_path / "pkg"
        (dst / "recipes").mkdir(parents=True)
        (dst / "app.py").write_text("BAKED")
        staged = self._staged(tmp_path, {"app.py": "NEW"})
        monkeypatch.setattr(os.path, "ismount",
                            lambda p: os.path.realpath(p) == os.path.realpath(dst / "recipes"))
        assert mod.main(str(staged), str(dst)) == 3
        assert (dst / "app.py").read_text() == "BAKED", \
            "a refused sync must not have touched the current code"
        assert (staged / "app.py").read_text() == "NEW", "refused input left in place"

    def test_a_destination_that_is_itself_a_mount_is_refused(self, tmp_path, monkeypatch):
        mod = load_swap_sync()
        dst = tmp_path / "pkg"
        dst.mkdir()
        (dst / "app.py").write_text("BAKED")
        monkeypatch.setattr(os.path, "ismount",
                            lambda p: os.path.realpath(p) == os.path.realpath(dst))
        assert mod.main(str(self._staged(tmp_path, {"app.py": "NEW"})), str(dst)) == 3
        assert (dst / "app.py").read_text() == "BAKED"

    def test_a_file_dir_type_collision_is_refused_before_any_change(self, tmp_path):
        """Discovering a collision mid-flip would leave a half-old half-new tree — the #72
        class. The check must fire first."""
        mod = load_swap_sync()
        dst = tmp_path / "pkg"
        (dst / "model").mkdir(parents=True)
        (dst / "model" / "keep.py").write_text("BAKED")
        staged = self._staged(tmp_path, {"model": "NEW"})  # incoming FILE over a DIRECTORY
        assert mod.main(str(staged), str(dst)) == 4
        assert (dst / "model" / "keep.py").read_text() == "BAKED"

    def test_an_empty_staged_tree_removes_everything_but_never_a_mount(self, tmp_path,
                                                                       monkeypatch):
        mod = load_swap_sync()
        dst = tmp_path / "pkg"
        (dst / "recipes").mkdir(parents=True)
        (dst / "app.py").write_text("BAKED")
        staged = tmp_path / "staged"
        staged.mkdir()
        monkeypatch.setattr(os.path, "ismount",
                            lambda p: os.path.realpath(p) == os.path.realpath(dst / "recipes"))
        assert mod.main(str(staged), str(dst)) == 3
        assert (dst / "app.py").read_text() == "BAKED"


# ------------------------------------------------- part 2: the boot path, with a real mount

DOCKERFILE = """
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \\
    && rm -rf /var/lib/apt/lists/* \\
    && mkdir -p /opt/engine/recipes /opt/worker && \\
    echo "MARKER = 'baked'" > /opt/engine/app.py && \\
    echo "MARKER = 'baked'" > /opt/worker/control.py
"""
IMAGE = "wanly-swap-test"


def _docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def swap_image(tmp_path_factory):
    if shutil.which("docker") is None or _docker("info", check=False).returncode != 0:
        pytest.skip("docker unavailable")
    ctx = tmp_path_factory.mktemp("ctx")
    (ctx / "Dockerfile").write_text(DOCKERFILE)
    if _docker("build", "-q", "-t", IMAGE, str(ctx), check=False).returncode != 0:
        pytest.skip("docker build failed")
    return IMAGE


@pytest.fixture
def fake_repo(tmp_path):
    """A git remote and a SRC_DIR clone of it, plus the host dir that becomes the mount."""
    work = tmp_path / "work"
    (work / "engine").mkdir(parents=True)
    (work / "wanly_worker").mkdir()
    (work / "engine" / "app.py").write_text("MARKER = 'remote-v1'\n")
    (work / "wanly_worker" / "control.py").write_text("MARKER = 'remote-v1'\n")
    git_env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
               "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=work, check=True, capture_output=True, env=git_env)
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)],
                   check=True, capture_output=True)
    src = tmp_path / "src"
    subprocess.run(["git", "clone", "-q", str(bare), str(src)],
                   check=True, capture_output=True)
    mount = tmp_path / "recipes-host"
    mount.mkdir()
    (mount / "r.json").write_text("{}")
    return tmp_path


def _swap_in_container(image, tmp_path):
    """Run fetch_engine.sh inside the image with a read-only mount inside /opt/engine,
    then report the aftermath from INSIDE the same container.

    The git repos on the bind mount are owned by the host user and the container runs as
    root, which git refuses to read (dubious ownership). safe.directory via -c/env does
    NOT apply to a local-path REMOTE's repo; --system config does. Verified, not assumed.
    """
    script = ("git config --system --add safe.directory '*' && "
              "bash {script} 2>&1; echo RC=$?; echo ---AFTERMATH---; "
              "cat /opt/engine/app.py 2>&1; ls /opt/engine; "
              "echo ---WORKER---; cat /opt/worker/control.py; "
              "echo ---REF---; cat /tmp/code_ref").format(script=SCRIPT)
    return _docker(
        "run", "--rm",
        # rw: the script's fetch path writes into SRC_DIR; safe.directory because the
        # container runs as root against a repo owned by the host user (git >= 2.35
        # refuses that by default, and the script cannot be littered with -c flags).
        "-v", f"{tmp_path}:{tmp_path}:z",   # :z = SELinux relabel, no-op elsewhere
        "-v", f"{SCRIPT}:{SCRIPT}:ro,z",
        "-v", f"{SWAP_SYNC}:{SWAP_SYNC}:ro,z",
        "-v", f"{tmp_path}/recipes-host:/opt/engine/recipes:ro,z",
        "-e", f"ENGINE_SRC_DIR={tmp_path}/src",
        "-e", f"SWAP_SYNC={SWAP_SYNC}",
        "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory",
        "-e", "GIT_CONFIG_VALUE_0=*",
        "-e", "ENGINE_BRANCH=main",
        "-e", "ENGINE_DEST=/opt/engine",
        "-e", "WORKER_DEST=/opt/worker",
        "-e", "CODE_REF_FILE=/tmp/code_ref",
        "-e", "WANLY_IMAGE_REF=deadbeefdeadbeef",
        image, "bash", "-c", script, check=False)


def _sections(out):
    """Split the container's report into (swap_output, engine_aftermath, worker_code,
    code_ref)."""
    swap, rest = out.split("---AFTERMATH---")
    engine, rest2 = rest.split("---WORKER---")
    worker, ref = rest2.split("---REF---")
    return swap, engine, worker, ref.strip()


def test_the_mount_refuses_the_swap_and_the_current_code_survives(swap_image, fake_repo):
    """The 3090 incident end to end: loud refusal, /opt/engine/app.py still baked and
    readable, honest identity. The mv version left app.py DELETED; if this ever fails on
    the 'baked' assertion, something has reintroduced a destructive move into the swap."""
    r = _swap_in_container(swap_image, fake_repo)
    swap, engine, _worker, ref = _sections(r.stdout)
    assert "RC=0" in r.stdout, "a failed swap must fall back, never abort the boot"
    assert "SWAP FAILED" in swap
    assert "swap refused for engine" in swap
    assert "mount point(s) inside /opt/engine" in swap, \
        "the refusal must name the mount, not just fail"
    assert "baked" in engine, (
        "the refused swap destroyed the current code — this is the #125 crash loop: "
        f"/opt/engine aftermath was: {engine!r}")
    assert "recipes" in engine
    assert ref.startswith("baked-fallback"), \
        f"identity must admit the fallback, said: {ref!r}"
    assert "main @" not in ref, "must not claim the fetched sha it is not running"


def test_the_package_without_a_mount_still_takes_the_new_code(swap_image, fake_repo):
    """'Keeping current code' is per package. wanly_worker has no mount inside it; its
    swap must land even though the engine's failed beside it."""
    r = _swap_in_container(swap_image, fake_repo)
    _swap, _engine, worker, _ref = _sections(r.stdout)
    assert "remote-v1" in worker, \
        f"the unblocked package should have taken the new code, saw: {worker!r}"
