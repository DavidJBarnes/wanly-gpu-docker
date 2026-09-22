"""fetch_engine.sh — the boot path for engine/ and wanly_worker/ (#116).

Same contract as fetch_services.sh in wanly-services#38, tested the same way against a real
local git remote: fetch-and-swap updates BOTH packages, an unreachable repo boots on baked
code with a WARN, and a failed dep install ABORTS — the render stack must never come up on a
half-installed environment. The paths are env-overridable precisely so this can run off-box.
"""
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fetch_engine.sh"


def _make_remote(tmp_path):
    work = tmp_path / "remote-work"
    (work / "engine").mkdir(parents=True)
    (work / "wanly_worker").mkdir()
    (work / "engine" / "app.py").write_text("MARKER = 'remote-v1'\n")
    (work / "wanly_worker" / "control.py").write_text("MARKER = 'remote-v1'\n")
    (work / "engine" / "requirements.txt").write_text("pytest\n")
    (work / "wanly_worker" / "requirements.txt").write_text("pytest\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=work, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=work, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=work, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-qm", "v1"], cwd=work, check=True)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    return work, bare


def _make_dests(tmp_path):
    engine = tmp_path / "opt/engine"
    worker = tmp_path / "app/wanly_worker"
    for d in (engine, worker):
        d.mkdir(parents=True)
    (engine / "app.py").write_text("MARKER = 'baked'\n")
    (worker / "control.py").write_text("MARKER = 'baked'\n")
    return engine, worker


def _run(tmp_path, bare, branch="main", engine_dest=None, worker_dest=None):
    engine_dest = engine_dest or tmp_path / "opt/engine"
    worker_dest = worker_dest or tmp_path / "app/wanly_worker"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "python3"
    if not link.exists():
        link.symlink_to(sys.executable)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
             "ENGINE_REPO": str(bare), "ENGINE_BRANCH": branch,
             "WANLY_IMAGE_REF": "deadbeefdeadbeefdeadbeef",
             "ENGINE_SRC_DIR": str(tmp_path / "engine-src"),
             "ENGINE_DEST": str(engine_dest),
             "WORKER_DEST": str(worker_dest),
             "CODE_REF_FILE": str(tmp_path / "run/code_ref")},
        capture_output=True, text=True, timeout=120)


def test_fetch_updates_both_packages(tmp_path):
    _, bare = _make_remote(tmp_path)
    engine, worker = _make_dests(tmp_path)
    r = _run(tmp_path, bare)
    assert r.returncode == 0, r.stderr
    assert "engine+supervisor: main @" in r.stdout
    assert "remote-v1" in (engine / "app.py").read_text()
    assert "remote-v1" in (worker / "control.py").read_text()
    ref = tmp_path / "run/code_ref"
    assert ref.read_text().startswith("main @")
    assert not (engine.parent / "engine.new").exists()
    assert not (engine.parent / "engine.old").exists()


def test_unreachable_repo_boots_on_baked_code(tmp_path):
    engine, worker = _make_dests(tmp_path)
    r = _run(tmp_path, tmp_path / "no-such-remote.git")
    assert r.returncode == 0
    assert "BAKED" in r.stdout
    assert "baked" in (engine / "app.py").read_text()
    assert "baked" in (worker / "control.py").read_text()
    assert not (tmp_path / "run/code_ref").exists()


def test_a_second_commit_converges(tmp_path):
    """The update branch (already cloned) is different code from the init-in-place branch."""
    work, bare = _make_remote(tmp_path)
    engine, worker = _make_dests(tmp_path)
    assert _run(tmp_path, bare).returncode == 0
    (work / "engine" / "app.py").write_text("MARKER = 'remote-v2'\n")
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-qm", "v2"], cwd=work, check=True)
    subprocess.run(["git", "push", "-q", str(bare), "HEAD:refs/heads/main"], cwd=work, check=True)
    r = _run(tmp_path, bare)
    assert r.returncode == 0, r.stderr
    assert "updating code" in r.stdout
    assert "remote-v2" in (engine / "app.py").read_text()


def test_a_non_main_branch_shouts(tmp_path):
    work, bare = _make_remote(tmp_path)
    subprocess.run(["git", "push", "-q", str(bare), "HEAD:refs/heads/experiment"],
                   cwd=work, check=True)
    _make_dests(tmp_path)
    r = _run(tmp_path, bare, branch="experiment")
    assert r.returncode == 0
    assert "NOT running engine main" in r.stdout


def test_a_failed_swap_keeps_baked_code_and_says_so(tmp_path):
    """#121, the guard rather than the mount: the real 3090 failure was EBUSY moving
    /opt/engine aside because a bind mount lived inside it (proven at the docker level; CI
    cannot mount). A read-only parent makes the first filesystem step fail with EACCES,
    which drives the same guarded path: the current code must survive, the log must shout,
    and the code_ref must say baked-fallback — never the fetched sha."""
    import os
    if os.geteuid() == 0:
        import pytest
        pytest.skip("root ignores directory permissions; the EACCES lever needs a real user")
    work, bare = _make_remote(tmp_path)
    engine, worker = _make_dests(tmp_path)
    engine_parent = engine.parent
    engine_parent.chmod(0o555)
    try:
        r = _run(tmp_path, bare)
    finally:
        engine_parent.chmod(0o755)
    assert r.returncode == 0, "a failed swap must fall back, not crash-loop"
    assert "SWAP FAILED" in r.stdout
    assert "keeping current" in r.stdout or "could not stage" in r.stdout
    assert "baked" in (engine / "app.py").read_text()
    ref = (tmp_path / "run/code_ref").read_text()
    assert ref.startswith("baked-fallback")
    assert "main @" not in ref  # the honest part: never claim a sha that is not running


def test_a_failed_dep_install_aborts_the_boot(tmp_path):
    """'-zzz-not-a-pkg' fails pip's option parser without needing the network."""
    work, bare = _make_remote(tmp_path)
    (work / "engine" / "requirements.txt").write_text("-zzz-not-a-pkg\n")
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-qm", "bad-dep"], cwd=work, check=True)
    subprocess.run(["git", "push", "-q", str(bare), "HEAD:refs/heads/main"], cwd=work, check=True)
    _make_dests(tmp_path)
    r = _run(tmp_path, bare)
    assert r.returncode == 1
    assert "DEP INSTALL FAILED" in r.stdout
