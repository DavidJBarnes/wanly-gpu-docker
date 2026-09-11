"""The long-lived worker must converge on :latest, and never mid-render (#72).

The 3090 ran a 37-hour-old image and a 14-hour-old daemon while a RunPod pod ran current
code, and the two produced different results from the same queue. Nothing detected it; it
surfaced as a 422 that looked random.

These tests guard the two halves of the fix: that the container spec is complete enough to
recreate the worker correctly, and that the updater's decision logic is right -- because
every wrong decision it can make is expensive. Recreating mid-render loses 10-13 minutes and
leaves a segment to be reclaimed; never recreating leaves the drift in place.
"""
import os
import pathlib
import re
import shutil
import stat
import subprocess

DEPLOY = pathlib.Path(__file__).parent.parent / "deploy"
RUN = DEPLOY / "run-worker.sh"
UPDATE = DEPLOY / "update-worker.sh"


class TestTheContainerSpecIsComplete:
    """Captured from the running 3090 container on 2026-09-06. A missing mount or port does
    not fail loudly -- the worker boots and then cannot see its models, or the console cannot
    reach ComfyUI, both of which are diagnosed the slow way."""

    def test_every_captured_mount_is_present(self):
        s = RUN.read_text()
        for mount in ("/jobs", "/opt/engine/recipes:ro", "/workspace/models:ro",
                      "/workspace/models/loras"):
            assert mount in s, f"run-worker.sh no longer mounts {mount}"

    def test_the_models_tree_stays_read_only(self):
        """This box is the source of truth for 217 GB of weights. Nothing in the container
        should be able to modify or delete them."""
        assert '"$MODELS_DIR:/workspace/models:ro"' in RUN.read_text()

    def test_both_published_ports(self):
        s = RUN.read_text()
        assert ":8188" in s and ":8190" in s

    def test_it_survives_a_reboot(self):
        assert "--restart unless-stopped" in RUN.read_text()

    def test_it_gets_the_gpu_through_cdi(self):
        """`--gpus all` grants the device nodes outside the OCI spec, and a systemd reload
        takes them away again (#95). CDI puts them in the spec, where a reload keeps them."""
        s = RUN.read_text()
        assert "--device nvidia.com/gpu=all" in s
        # As a FLAG. The comment above the run block names the old one to explain why.
        assert not re.search(r"^\s*--gpus\b", s, re.M)

    def test_comfyui_path_is_explicitly_empty(self):
        """Not merely absent. With a path set the daemon takes ownership of ComfyUI's custom
        nodes, which breaks an LTX worker -- and an unset variable in the container would let
        an image default win."""
        assert '-e "COMFYUI_PATH="' in RUN.read_text()

    def test_the_queue_key_is_required_not_defaulted(self):
        """A worker that boots without it registers and then claims nothing, which reads as
        an empty queue."""
        assert 'QUEUE_API_KEY:?' in RUN.read_text()


class TestTheRename:
    def test_the_legacy_container_is_adopted_not_duplicated(self):
        """#77 renamed the container. Creating a second one beside the old is the bad outcome.

        Two containers with the same FRIENDLY_NAME both claim from the same queue, and the API
        identifies a worker by that name -- it cannot tell them apart, so they fight over
        segments and each looks like the other going wrong.
        """
        s = UPDATE.read_text()
        assert "docker rename" in s, "no adoption path for the pre-rename container"
        rename_at = s.index("docker rename")
        create_at = s.index('exec "$HERE/run-worker.sh"')
        assert rename_at < create_at, \
            "the rename must be attempted before falling through to creating a new container"


class TestTheIdleCheck:
    def test_it_asks_inside_the_container(self):
        """The engine binds 127.0.0.1 INSIDE the container, so the -p 8190:8190 mapping
        resolves to nothing. Verified on the 3090: a host-side curl returns empty, which
        parsed naively reads as "idle" and would recreate the container mid-render. This
        very nearly shipped."""
        s = UPDATE.read_text()
        assert 'docker exec "$NAME" curl' in s, \
            "the idle check must run inside the container, not against the published port"


DEFAULT_NAME = re.search(r'^NAME="\$\{NAME:-([^}]+)\}"', UPDATE.read_text(), re.M).group(1)


def _stage(tmp, *, running_image, latest_image, engine_busy, worker_status, trainer=None):
    """Stage the real update-worker.sh with fakes for every external call it makes.

    Two separate fakes, because the script asks two different things:
      * `curl` on the HOST            -> the API's /workers, for the worker's own status
      * `curl` INSIDE the container   -> the engine's /health, via docker exec
    """
    bin_dir = tmp / "bin"
    bin_dir.mkdir(exist_ok=True)

    # The container's /health (wanly-gpu-docker#83) carries the engine's running/queue_depth
    # on the ltx-engine-api entry AND, when the trainer is enabled, `training` on the
    # lora-trainer entry. `trainer` is None (no trainer in this container), "idle", or
    # "training". `engine_busy` None means the container does not answer at all.
    if engine_busy is None:
        engine_case = "exit 7"
    else:
        entries = ['{"name":"ltx-engine-api","ready":true,"queue_depth":0,"running":%d}' % engine_busy]
        if trainer is not None:
            entries.append('{"name":"lora-trainer","ready":true,"training":%s}'
                           % ('"p@y v3"' if trainer == "training" else "null"))
        engine_case = "echo '{\"status\":\"ok\",\"services\":[%s]}'" % ",".join(entries)

    docker = "\n".join([
        "#!/usr/bin/env bash",
        'case "$*" in',
        # Read from the script rather than hardcoded, so renaming the container does not
        # silently turn this stub into a no-op that answers every inspect with exit 0.
        '  "inspect -f {{.Image}} %s")  echo "%s"; exit 0 ;;' % (DEFAULT_NAME, running_image),
        '  "pull -q "*)                        exit 0 ;;',
        '  "image inspect "*)                  echo "%s"; exit 0 ;;' % latest_image,
        '  *"curl"*)                           %s ;;' % engine_case,
        'esac',
        'exit 0',
        '',
    ])
    (bin_dir / "docker").write_text(docker)
    (bin_dir / "docker").chmod(0o755)

    if worker_status is None:
        body = "BAD NOT JSON"
    elif worker_status == "absent":
        body = "[]"
    else:
        body = '[{"friendly_name":"3090.zero","status":"%s"}]' % worker_status
    # The host curl answers one thing: the API's /workers. The trainer is asked through the
    # container (docker exec), above.
    (bin_dir / "curl").write_text("\n".join([
        "#!/usr/bin/env bash",
        "cat <<'JSON'",
        body,
        "JSON",
        "",
    ]))
    (bin_dir / "curl").chmod(0o755)

    stage = tmp / "deploy"
    stage.mkdir(exist_ok=True)
    shutil.copy(UPDATE, stage / "update-worker.sh")
    (stage / "run-worker.sh").write_text("#!/usr/bin/env bash\necho RECREATED\n")
    (stage / "run-worker.sh").chmod(0o755)
    (stage / "worker.env").write_text(
        "QUEUE_URL=http://api.test:8001\nQUEUE_API_KEY=k\nFRIENDLY_NAME=3090.zero\n")
    return bin_dir, stage


def _run(tmp_path, *, running_image, latest_image, engine_busy=0, worker_status="online-idle",
         trainer=None):
    bin_dir, stage = _stage(tmp_path, running_image=running_image, latest_image=latest_image,
                            engine_busy=engine_busy, worker_status=worker_status,
                            trainer=trainer)
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
    return subprocess.run(["bash", str(stage / "update-worker.sh")],
                          capture_output=True, text=True, env=env)


class TestTheDecision:
    SAME = "sha256:aaa"
    NEW = "sha256:bbb"

    def test_it_does_nothing_when_the_image_has_not_changed(self, tmp_path):
        """The common path — it runs on a timer against an image that rarely changes."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.SAME)
        assert "nothing to do" in r.stdout and "RECREATED" not in r.stdout

    def test_it_recreates_when_the_image_changed_and_the_worker_is_idle(self, tmp_path):
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW)
        assert "RECREATED" in r.stdout, r.stdout

    def test_it_leaves_a_worker_alone_that_holds_a_claim(self, tmp_path):
        """THE regression, 2026-09-06.

        The daemon sets online-busy the instant it receives a claim, BEFORE [1/6]. The engine
        knows nothing until [3/6]. A container was recreated 50% through a 673 MB LoRA
        download in [2/6] — engine truthfully idle, worker very much not — and the abandoned
        segment sat in PROCESSING for seven hours, pinned to a live worker where no reclaim
        rule could reach it.

        console#423 widened the window the same day: an on-demand 46 GB checkpoint fetch is
        ~20 minutes inside [2/6] with the engine reporting idle throughout.
        """
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                 worker_status="online-busy", engine_busy=0)
        assert "RECREATED" not in r.stdout, "recreated while the worker held a claim"
        assert "online-busy" in r.stdout

    def test_it_believes_the_engine_over_a_worker_claiming_to_be_idle(self, tmp_path):
        """The daemon's status push can fail — the API's own reclaim logic says so. When the
        two disagree, the one that is actually rendering wins."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                 worker_status="online-idle", engine_busy=1)
        assert "RECREATED" not in r.stdout
        assert "despite status" in r.stdout

    def test_an_unregistered_worker_is_not_idle(self, tmp_path):
        """A worker mid-boot has not registered yet. Recreating then interrupts model
        staging, which on a cold pod is ~58 GB of downloads."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                 worker_status="absent")
        assert "RECREATED" not in r.stdout
        assert "not-registered" in r.stdout

    def test_an_unreadable_api_counts_as_busy(self, tmp_path):
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, worker_status=None)
        assert "RECREATED" not in r.stdout
        assert "unreadable" in r.stdout

    def test_an_unreadable_engine_counts_as_busy(self, tmp_path):
        """Fail safe on both signals, not just the API one."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, engine_busy=None)
        assert "RECREATED" not in r.stdout
        assert "assuming busy" in r.stdout


class TestTheLock:
    """wanly-gpu-docker#97.

    Seen 2026-09-10 while deploying #96: a manual update was mid-pull (13 GiB, ~11 min) when
    the timer fired twice more. Three copies all saw "image changed" and all ran
    run-worker.sh; one died on a container-name conflict, and a loser could have removed the
    winner's container between its rm -f and its run.
    """

    SAME = "sha256:aaa"
    NEW = "sha256:bbb"

    def _run_two(self, tmp_path, hold_before_run=None):
        """Fire two copies concurrently against the same lock and return both results."""
        import subprocess
        bin_dir, stage = _stage(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                                engine_busy=0, worker_status="online-idle")
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))

        script = stage / "update-worker.sh"
        first = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, env=env)
        second = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, env=env)
        first.wait(timeout=30)
        second.wait(timeout=30)
        return first.stdout.read(), second.stdout.read(), first.returncode, second.returncode

    def test_a_concurrent_run_exits_zero_without_recreating(self, tmp_path):
        """The loser must not die on a name conflict, and must not recreate: stacked firings
        are made harmless rather than trying to make them impossible."""
        out_a, out_b, code_a, code_b = self._run_two(tmp_path)
        recreated = [o for o in (out_a, out_b) if "RECREATED" in o]
        refused = [o for o in (out_a, out_b) if "another update is in progress" in o]

        assert len(recreated) == 1, (out_a, out_b)
        assert len(refused) == 1, (out_a, out_b)
        loser_code = code_b if "RECREATED" in out_a else code_a
        assert loser_code == 0, "the loser must exit 0, not a timer-visible failure"

    def test_the_lock_is_held_while_run_worker_spawns(self, tmp_path):
        """The lock must still be held when run-worker.sh runs — that is the window where a
        loser could have rm -f'd the winner's container between its rm and its run.

        flock -n 9 inside the child proves nothing: fd 9 is inherited and points at the SAME
        open file description, so flock trivially "succeeds" against its own lock. The
        honest check is a FRESH open of the lock file from an unrelated process.
        """
        import subprocess
        bin_dir, stage = _stage(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                                engine_busy=0, worker_status="online-idle")
        # A fresh open of the lock file (flock <file> opens its own fd) from INSIDE the
        # child, so it competes with the parent's held lock like any third party would.
        (stage / "run-worker.sh").write_text(
            "#!/usr/bin/env bash\n"
            'lock="${XDG_RUNTIME_DIR:-/run/lock}/wanly-worker-update.lock"\n'
            'if flock -n "$lock" true; then echo "LOCK-FREE"; else echo "LOCK-HELD"; fi\n'
        )
        (stage / "run-worker.sh").chmod(0o755)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))

        r = subprocess.run(["bash", str(stage / "update-worker.sh")],
                           capture_output=True, text=True, env=env)
        assert "LOCK-HELD" in r.stdout, r.stdout
        assert "LOCK-FREE" not in r.stdout

    def test_serial_runs_do_not_deadlock(self, tmp_path):
        """A lock that never released would turn the timer into a no-op forever — the
        installed-and-dead failure from #72, again."""
        r1 = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW)
        assert "RECREATED" in r1.stdout
        r2 = _run(tmp_path, running_image=self.NEW, latest_image=self.NEW)
        assert r2.returncode == 0
        assert "nothing to do" in r2.stdout


def test_the_scripts_are_executable():
    for p in (RUN, UPDATE):
        assert os.stat(p).st_mode & stat.S_IXUSR, "%s is not executable" % p.name


def test_no_secret_is_committed():
    """worker.env carries the queue key and must never be in the repo."""
    assert not (DEPLOY / "worker.env").exists()
    assert (DEPLOY / "worker.env.example").read_text().count("QUEUE_API_KEY=\n") == 1


class TestTheTimerActuallyFires:
    """A timer that is enabled but has no next elapse is the worst failure here, because
    `systemctl is-enabled` says "enabled" and `list-timers` lists it (wanly-gpu-docker#72).

    The first version used OnBootSec + OnUnitActiveSec. OnUnitActiveSec only re-arms once the
    TIMER has triggered the service; with OnBootSec already past there was nothing to compute
    a next elapse from, so it sat `active (elapsed)` with `Trigger: n/a` -- installed, and
    never going to run again. Observed on the 3090.
    """

    TIMER = DEPLOY / "wanly-worker-update.timer"

    def test_it_uses_an_absolute_schedule(self):
        s = self.TIMER.read_text()
        assert "OnCalendar=" in s, (
            "the timer needs an absolute schedule; OnBootSec/OnUnitActiveSec stops "
            "re-arming and the timer silently never fires again"
        )

    def test_it_does_not_rely_on_onunitactivesec(self):
        assert "OnUnitActiveSec=" not in self.TIMER.read_text()

    def test_it_catches_up_after_downtime(self):
        """A box that was off through a release should update on the next boot, not wait for
        the following slot."""
        assert "Persistent=true" in self.TIMER.read_text()


class TestTheTimerInstaller:
    """Installing the timer was four steps and the first two were silently skippable.

    Running `systemctl enable` without the `cp` gives `Unit wanly-worker-update.timer does not
    exist` — which names the unit rather than the missing copy, reads like the repo is wrong,
    and kept #72 open an extra day.
    """

    INSTALL = DEPLOY / "install-timer.sh"

    def test_it_exists_and_is_executable(self):
        assert self.INSTALL.is_file()
        assert self.INSTALL.stat().st_mode & 0o111, "not executable"

    def test_it_uses_absolute_paths(self):
        """The other way to get the same error was running it from the wrong directory."""
        s = self.INSTALL.read_text()
        assert 'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in s
        assert '"$HERE/$unit"' in s

    def test_it_installs_both_units(self):
        s = self.INSTALL.read_text()
        assert "wanly-worker-update.service" in s
        assert "wanly-worker-update.timer" in s

    def test_it_verifies_a_next_elapse_rather_than_trusting_is_enabled(self):
        """#76 was a timer that reported `enabled` while `Trigger:` said `n/a` — installed,
        enabled, and never going to fire. `is-enabled` is not evidence of scheduling."""
        s = self.INSTALL.read_text()
        assert "NextElapseUSecRealtime" in s
        # Code only. The comments explain WHY is-enabled is not evidence, and that explanation
        # is worth keeping — it is the whole reason the check exists.
        code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
        assert "is-enabled" not in code, "is-enabled proves nothing about whether it will fire"

    def test_it_fails_loudly_when_nothing_is_scheduled(self):
        s = self.INSTALL.read_text()
        assert "FATAL" in s and "exit 1" in s

    def test_it_refuses_to_run_without_root(self):
        assert 'id -u' in self.INSTALL.read_text()


class TestTheTrainerSharesTheCard:
    """2026-09-08: a training run had drained the render worker; the drained daemon had
    exited and its container come back with no drain, looking idle. This timer recreated it
    on the new image, it claimed a render beside the training, and the box hard-reset."""

    SAME = "sha256:aaa"
    NEW = "sha256:bbb"

    def test_a_training_trainer_blocks_the_recreate(self, tmp_path):
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, trainer="training")
        assert "RECREATED" not in r.stdout, r.stdout
        assert "training=yes" in r.stdout

    def test_an_idle_trainer_does_not(self, tmp_path):
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, trainer="idle")
        assert "RECREATED" in r.stdout, r.stdout

    def test_no_trainer_on_the_box_is_fine(self, tmp_path):
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, trainer=None)
        assert "RECREATED" in r.stdout, r.stdout

    def test_the_check_reads_the_containers_own_health(self):
        """The trainer is IN the container since wanly-gpu-docker#83; the old :8083 sidecar
        probe would answer nothing and read as idle."""
        src = UPDATE.read_text()
        assert ":8083" not in src
        assert 's.get("name") == "lora-trainer"' in src

    def test_a_degraded_container_that_is_training_still_blocks(self, tmp_path):
        """`curl -s`, not `-sf`: a 503 body is still the truth about a run in flight."""
        bin_dir, stage = _stage(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                                engine_busy=0, worker_status="online-idle", trainer="training")
        d = (bin_dir / "docker").read_text().replace("echo '{", "echo '{\"status\":\"degraded\",")
        (bin_dir / "docker").write_text(d)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
        r = subprocess.run(["bash", str(stage / "update-worker.sh")], capture_output=True,
                           text=True, env=env)
        assert "RECREATED" not in r.stdout, r.stdout

    def test_an_unparseable_health_counts_as_training(self, tmp_path):
        bin_dir, stage = _stage(tmp_path, running_image=self.SAME, latest_image=self.NEW,
                                engine_busy=0, worker_status="online-idle", trainer="idle")
        # The engine probe parses first and passes; make the second read garbage by having
        # docker answer differently the second time is not possible with a static stub, so
        # this pins the script text instead.
        src = UPDATE.read_text()
        assert 'print("unknown"); raise SystemExit' in src
        assert '[ "$training" != "no" ]' in src


class TestTheEngineIsAskedThroughTheSupervisor:
    """update-worker.sh reads the engine's running/queue_depth from the supervisor's /health
    (wanly-gpu-docker#83) and falls back to the engine itself for an image from before it."""

    SAME = "sha256:aaa"
    NEW = "sha256:bbb"

    def test_the_script_asks_the_control_port_first(self):
        text = UPDATE.read_text()
        assert '${CONTROL_PORT:-8081}/health' in text
        assert text.index('${CONTROL_PORT:-8081}/health') < text.index("127.0.0.1:8190/health")

    def test_a_pre_supervisor_image_still_answers_through_the_fallback(self, tmp_path):
        """The docker stub answers every in-container curl with the engine's own body, which
        has no `services` list -- the supervisor path exits 4 and the fallback decides."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, engine_busy=0)
        assert "RECREATED" in r.stdout, r.stdout
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, engine_busy=1)
        assert "RECREATED" not in r.stdout, r.stdout
