"""One card per container, by UUID, and a second container on one host (wanly-gpu-docker#163).

After the 2070 goes into 3090a's box, `--device nvidia.com/gpu=all` hands both cards to both
containers and each service picks one by index. These tests guard the three parts of the fix:

  * the boot refuses unless a pinned container sees exactly its card (wanly_worker/gpu_pin.py);
  * run-worker.sh passes the pin through CDI, and refuses a bad one BEFORE removing anything;
  * a second env file (deploy/scene.env) can never replace the 3090 worker or share its row.
"""
import os
import pathlib
import shutil
import subprocess

import pytest

from wanly_worker import gpu_pin

DEPLOY = pathlib.Path(__file__).parent.parent / "deploy"
RUN = DEPLOY / "run-worker.sh"

U3090 = "GPU-522c32cc-7ac2-9dcd-f1d5-4f5390e68f0b"
U2070 = "GPU-0b2070aa-1111-2222-3333-444455556666"
C3090 = {"index": "0", "name": "NVIDIA GeForce RTX 3090", "uuid": U3090}
C2070 = {"index": "1", "name": "NVIDIA GeForce RTX 2070", "uuid": U2070}


# ------------------------------------------------------------------ the boot check


class TestTheBootCheck:
    def test_pinned_and_seeing_exactly_that_card(self):
        assert "RTX 3090" in gpu_pin.check(U3090, [C3090])

    def test_the_uuid_is_case_insensitive(self):
        gpu_pin.check(U3090.upper().replace("GPU-", "GPU-"), [C3090])

    def test_pinned_but_seeing_both_cards_refuses(self):
        """The pin never reached docker: the worker would pick a card by index."""
        with pytest.raises(gpu_pin.GpuPinError, match="sees 2 cards"):
            gpu_pin.check(U3090, [C3090, C2070])

    def test_pinned_to_one_card_and_seeing_another_refuses(self):
        with pytest.raises(gpu_pin.GpuPinError, match="Wrong card"):
            gpu_pin.check(U3090, [C2070])

    def test_pinned_and_nvidia_smi_failing_refuses(self):
        with pytest.raises(gpu_pin.GpuPinError, match="could not list"):
            gpu_pin.check(U3090, None)

    def test_unpinned_with_two_cards_warns_but_boots(self):
        """Unset keeps the old behaviour -- a box changes only when it opts in."""
        assert "WARNING" in gpu_pin.check(None, [C3090, C2070])

    def test_unpinned_single_card_is_quiet(self):
        assert "WARNING" not in gpu_pin.check(None, [C3090])

    def test_gpu_device_is_the_same_setting(self, monkeypatch):
        """The epic wrote GPU_DEVICE; the ticket GPU_UUID. Either works."""
        monkeypatch.delenv("GPU_UUID", raising=False)
        monkeypatch.setenv("GPU_DEVICE", U3090)
        assert gpu_pin.expected_uuid() == U3090

    def test_it_runs_before_anything_starts(self):
        """In the lifespan, ahead of the services: a worker must not pick a card first."""
        src = (pathlib.Path(gpu_pin.__file__).parent / "control.py").read_text()
        assert src.index("gpu_pin.enforce()") < src.index("Supervisor(registry.build(")


# ------------------------------------------------------------------ run-worker.sh


FAKE_DOCKER = r"""#!/usr/bin/env bash
echo "docker $*" >> "$LOG"
case "$1" in
  inspect)
    if [ "$2" = "-f" ]; then fmt="$3"; name="$4"; else fmt=""; name="$2"; fi
    d="$STATE/$name"
    [ -d "$d" ] || exit 1
    case "$fmt" in
      *wanly.env*) if [ -f "$d/label" ]; then cat "$d/label"; else echo "<no value>"; fi ;;
      *Config.Env*) cat "$d/env" 2>/dev/null ;;
      *) echo "sha256:0123456789abcdef" ;;
    esac ;;
  ps) ls "$STATE" ;;
  port) exit 1 ;;
  rm) echo "REMOVED $3" >> "$LOG"; rm -rf "$STATE/$3" ;;
  run)
    shift; name=""
    while [ $# -gt 0 ]; do [ "$1" = "--name" ] && name="$2"; shift; done
    mkdir -p "$STATE/$name" ;;
esac
exit 0
"""


def _stage(tmp, *, env_lines, scene_lines=None, gpus=(C3090,), cdi=(U3090,), containers=None):
    """A deploy dir with fake docker / nvidia-smi / nvidia-ctk / ss on PATH.

    containers: {name: {"label": <env file or None>, "env": "FRIENDLY_NAME=..."}} already there.
    """
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    state = tmp / "state"
    state.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    smi = "\n".join(f"GPU {c['index']}: {c['name']} (UUID: {c['uuid']})" for c in gpus)
    (bin_dir / "nvidia-smi").write_text(f"#!/usr/bin/env bash\ncat <<'X'\n{smi}\nX\n")
    ctk = "\n".join(["nvidia.com/gpu=0", *[f"nvidia.com/gpu={u}" for u in cdi], "nvidia.com/gpu=all"])
    (bin_dir / "nvidia-ctk").write_text(f"#!/usr/bin/env bash\ncat <<'X'\n{ctk}\nX\n")
    (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)

    deploy = tmp / "deploy"
    deploy.mkdir()
    shutil.copy(RUN, deploy / "run-worker.sh")
    store = tmp / "ollama"
    store.mkdir()
    base = ["QUEUE_URL=http://api.test:8001", "QUEUE_API_KEY=k",
            "IMAGE=davidjbarnes/wanly-gpu-docker:full", "SERVICES=scene-caption",
            f"OLLAMA_HOST_STORE={store}", "PRUNE_IMAGES=0"]
    (deploy / "worker.env").write_text("\n".join(base + env_lines) + "\n")
    if scene_lines is not None:
        (deploy / "scene.env").write_text("\n".join(base + scene_lines) + "\n")
    for name, spec in (containers or {}).items():
        d = state / name
        d.mkdir()
        if spec.get("label"):
            (d / "label").write_text(spec["label"] + "\n")
        (d / "env").write_text(spec.get("env", "") + "\n")
    return bin_dir, deploy, state


def _run(tmp, bin_dir, deploy, state, env_file=None):
    log = tmp / "docker.log"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", STATE=str(state), LOG=str(log),
               HOME=str(tmp))
    for k in ("GPU_UUID", "GPU_DEVICE", "NAME", "WORKER_ENV", "MODE"):
        env.pop(k, None)
    if env_file:
        env["WORKER_ENV"] = str(env_file)
    r = subprocess.run(["bash", str(deploy / "run-worker.sh")], capture_output=True, text=True,
                       env=env)
    return r, (log.read_text() if log.exists() else "")


def _run_line(log):
    return next((ln for ln in log.splitlines() if ln.startswith("docker run")), "")


class TestThePin:
    def test_unset_keeps_every_card(self, tmp_path):
        r, log = _run(tmp_path, *_stage(tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero"]))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "--device nvidia.com/gpu=all" in _run_line(log)

    def test_a_uuid_pins_one_card_through_cdi(self, tmp_path):
        r, log = _run(tmp_path, *_stage(tmp_path, env_lines=[
            "FRIENDLY_NAME=3090a.zero", f"GPU_UUID={U3090}"], gpus=(C3090, C2070), cdi=(U3090, U2070)))
        assert r.returncode == 0, r.stdout + r.stderr
        line = _run_line(log)
        assert f"--device nvidia.com/gpu={U3090}" in line
        assert f"GPU_UUID={U3090}" in line, "the container needs it for its own boot check"

    def test_gpu_device_works_too(self, tmp_path):
        r, log = _run(tmp_path, *_stage(tmp_path, env_lines=[
            "FRIENDLY_NAME=3090a.zero", f"GPU_DEVICE={U3090}"]))
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"--device nvidia.com/gpu={U3090}" in _run_line(log)

    @pytest.mark.parametrize("bad,why", [
        ("1", "not a UUID"),
        (U2070, "not a card on this host"),
    ])
    def test_a_bad_pin_is_refused_before_the_rm(self, tmp_path, bad, why):
        """A typo must not cost the running worker."""
        r, log = _run(tmp_path, *_stage(tmp_path, env_lines=[
            "FRIENDLY_NAME=3090a.zero", f"GPU_UUID={bad}"],
            containers={"wanly-gpu-docker": {"env": "FRIENDLY_NAME=3090a.zero"}}))
        assert r.returncode != 0
        assert why in r.stdout
        assert "REMOVED" not in log and "docker run" not in log

    def test_a_card_the_cdi_spec_does_not_name_is_refused(self, tmp_path):
        """docker run would fail on it AFTER the rm, leaving the box with no worker."""
        r, log = _run(tmp_path, *_stage(tmp_path, env_lines=[
            "FRIENDLY_NAME=3090a.zero", f"GPU_UUID={U2070}"], gpus=(C3090, C2070), cdi=(U3090,)))
        assert r.returncode != 0
        assert "CDI spec has no device" in r.stdout
        assert "REMOVED" not in log


class TestASecondContainer:
    SCENE = ["NAME=wanly-scene", "FRIENDLY_NAME=3090a-scene", "CONTROL_PORT=8088",
             f"GPU_UUID={U2070}"]

    def test_the_scene_container_runs_beside_the_worker(self, tmp_path):
        bin_dir, deploy, state = _stage(
            tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero"], scene_lines=self.SCENE,
            gpus=(C3090, C2070), cdi=(U3090, U2070),
            containers={"wanly-gpu-docker": {"env": "FRIENDLY_NAME=3090a.zero"}})
        r, log = _run(tmp_path, bin_dir, deploy, state, env_file=deploy / "scene.env")
        assert r.returncode == 0, r.stdout + r.stderr
        line = _run_line(log)
        assert "--name wanly-scene" in line
        assert f"--device nvidia.com/gpu={U2070}" in line
        assert "-p 8088:8081" in line and "-p 11436:11436" in line
        assert "-p 8190" not in line and "-p 8191" not in line, "no render stack ports on a captioner"
        assert f"wanly.env={deploy / 'scene.env'}" in line
        assert "REMOVED wanly-gpu-docker" not in log, "the 3090 worker must be untouched"

    def test_a_scene_env_without_name_cannot_replace_the_worker(self, tmp_path):
        """NAME defaults to wanly-gpu-docker: without this the rm would take the 3090 worker."""
        lines = [ln for ln in self.SCENE if not ln.startswith("NAME=")]
        bin_dir, deploy, state = _stage(
            tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero"], scene_lines=lines,
            gpus=(C3090, C2070), cdi=(U3090, U2070),
            containers={"wanly-gpu-docker": {"env": "FRIENDLY_NAME=3090a.zero"}})
        r, log = _run(tmp_path, bin_dir, deploy, state, env_file=deploy / "scene.env")
        assert r.returncode != 0
        assert "would replace a different worker" in r.stdout
        assert "REMOVED" not in log

    def test_a_container_made_from_another_env_file_is_not_replaced(self, tmp_path):
        bin_dir, deploy, state = _stage(
            tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero", "NAME=wanly-scene"],
            containers={"wanly-scene": {"label": "/somewhere/scene.env",
                                        "env": "FRIENDLY_NAME=3090a-scene"}})
        r, log = _run(tmp_path, bin_dir, deploy, state)
        assert r.returncode != 0
        assert "REMOVED" not in log

    def test_the_default_env_still_recreates_its_own_unlabelled_worker(self, tmp_path):
        """Every container from before this change has no label; the main worker must still
        converge with plain ./run-worker.sh and with the update timer."""
        bin_dir, deploy, state = _stage(
            tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero"],
            containers={"wanly-gpu-docker": {"env": "FRIENDLY_NAME=3090a.zero"}})
        r, log = _run(tmp_path, bin_dir, deploy, state)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "REMOVED wanly-gpu-docker" in log and "docker run" in log

    def test_two_containers_cannot_share_a_friendly_name(self, tmp_path):
        """wanly-api upserts its worker row on FRIENDLY_NAME: they would share one row (#74)."""
        lines = [ln for ln in self.SCENE if not ln.startswith("FRIENDLY_NAME=")] + \
            ["FRIENDLY_NAME=3090a.zero"]
        bin_dir, deploy, state = _stage(
            tmp_path, env_lines=["FRIENDLY_NAME=3090a.zero"], scene_lines=lines,
            gpus=(C3090, C2070), cdi=(U3090, U2070),
            containers={"wanly-gpu-docker": {"env": "FRIENDLY_NAME=3090a.zero"}})
        r, log = _run(tmp_path, bin_dir, deploy, state, env_file=deploy / "scene.env")
        assert r.returncode != 0
        assert "already registers as FRIENDLY_NAME=3090a.zero" in r.stdout
        assert "REMOVED" not in log


class TestTheExample:
    def test_scene_env_example_is_a_dedicated_captioner(self):
        s = (DEPLOY / "scene.env.example").read_text()
        assert "NAME=wanly-scene" in s
        assert "SERVICES=scene-caption\n" in s
        assert "\nSCENE_CAPTION_SHARED=" not in s, "dedicated card: the sharing flag stays off"
        assert "CONTROL_PORT=8088" in s and "PRUNE_IMAGES=0" in s
        assert s.count("QUEUE_API_KEY=\n") == 1, "never a real key in the example"

    def test_real_env_files_are_ignored(self):
        gi = (DEPLOY.parent / ".gitignore").read_text()
        assert "deploy/*.env" in gi
        assert not (DEPLOY / "scene.env").exists()

    def test_a_scene_only_box_registers_as_a_service(self):
        """It must never run the render daemon or be offered a segment."""
        from wanly_worker import registry
        assert registry.kinds_for(registry.select_mode(["scene-caption"], None)) == ["service"]
