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
MODE = DEPLOY / "wanly-mode.sh"
IDLE = DEPLOY / "worker-idle.sh"


class TestTheContainerSpecIsComplete:
    """Captured from the running 3090 container on 2026-09-06. A missing mount or port does
    not fail loudly -- the worker boots and then cannot see its models, or the console cannot
    reach ComfyUI, both of which are diagnosed the slow way."""

    def test_every_captured_mount_is_present(self):
        s = RUN.read_text()
        for mount in ("/jobs", "/workspace/models:ro", "/workspace/models/loras"):
            assert mount in s, f"run-worker.sh no longer mounts {mount}"

    def test_nothing_mounts_inside_the_engine_dir(self):
        """#121: the old /opt/engine/recipes:ro mount (dead since the recipes moved to DB
        rows, wanly-api#212) made the boot swap fail on the 3090. #125 hardened the swap
        itself (swap_sync refuses mounted trees), but a mount in these paths still pins the
        box to baked code forever — run-worker.sh must not create one."""
        s = RUN.read_text()
        assert ":/opt/engine" not in s, "a mount inside the engine swap target"
        assert ":/app" not in s, "a mount inside the supervisor swap target"

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
        s = IDLE.read_text()
        assert 'docker exec "$name" curl' in s, \
            "the idle check must run inside the container, not against the published port"


DEFAULT_NAME = re.search(r'^NAME="\$\{NAME:-([^}]+)\}"', UPDATE.read_text(), re.M).group(1)


def _stage(tmp, *, running_image, latest_image, engine_busy, worker_status, trainer=None,
           container_running=True, services="ltx-engine,lora-trainer"):
    """Stage the real update-worker.sh with fakes for every external call it makes.

    Two separate fakes, because the script asks two different things:
      * `curl` on the HOST            -> the API's /workers, for the worker's own status
      * `curl` INSIDE the container   -> the engine's /health, via docker exec

    `container_running` models the #105 state: an exited container is not a busy one, and
    the updater must recreate it before any of the idle checks can walk away from it.
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

    # The #105 dead check is `docker ps -q -f "name=^/$NAME$"`: non-empty means running,
    # empty means exited. The fake always models the one container under test.
    ps_case = ('  "ps "*)  echo "fake-container-id"; exit 0 ;;' if container_running
               else '  "ps "*)  exit 0 ;;')

    docker = "\n".join([
        "#!/usr/bin/env bash",
        'case "$*" in',
        # Read from the script rather than hardcoded, so renaming the container does not
        # silently turn this stub into a no-op that answers every inspect with exit 0.
        '  "inspect -f {{.Image}} %s")  echo "%s"; exit 0 ;;' % (DEFAULT_NAME, running_image),
        # What the container was GIVEN. The gate reads this to tell "the engine is not
        # answering" from "there is no engine on this box to answer" (#131). `services`
        # None models an inspect that cannot be read at all, which must NOT be mistaken for
        # a box without the service.
        ('  "inspect -f {{range .Config.Env}}"*)  exit 1 ;;' if services is None
         else '  "inspect -f {{range .Config.Env}}"*)  echo "SERVICES=%s"; exit 0 ;;' % services),
        ps_case,
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
    # The idle gate lives beside it and is SOURCED, so a stage without it silently tests
    # nothing -- every run dies on "worker-idle.sh: No such file or directory".
    shutil.copy(IDLE, stage / "worker-idle.sh")
    (stage / "run-worker.sh").write_text("#!/usr/bin/env bash\necho RECREATED\n")
    (stage / "run-worker.sh").chmod(0o755)
    (stage / "worker.env").write_text(
        "QUEUE_URL=http://api.test:8001\nQUEUE_API_KEY=k\nFRIENDLY_NAME=3090.zero\n")
    return bin_dir, stage


def _run(tmp_path, *, running_image, latest_image, engine_busy=0, worker_status="online-idle",
         trainer=None, container_running=True, services="ltx-engine,lora-trainer"):
    bin_dir, stage = _stage(tmp_path, running_image=running_image, latest_image=latest_image,
                            engine_busy=engine_busy, worker_status=worker_status,
                            trainer=trainer, container_running=container_running,
                            services=services)
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


class TestTheDeadContainer:
    """wanly-gpu-docker#105. An nvidia driver/toolkit event kills the container (exit 255,
    `CDI device injection failed`) and docker never restarts it — RestartCount stayed 0,
    twice. A dead container must be recreated by the next timer tick: its processes are
    gone, so none of the busy checks can apply, and the digest comparison alone would say
    "nothing to do" forever."""

    SAME = "sha256:aaa"

    def test_a_dead_container_is_recreated_even_when_the_image_is_current(self, tmp_path):
        """The exact hole: image unchanged + every health check fails, which the old script
        read as busy. Dead is not busy."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.SAME,
                 container_running=False)
        assert "RECREATED" in r.stdout, r.stdout
        assert "not running" in r.stdout

    def test_a_dead_container_is_recreated_before_any_idle_check(self, tmp_path):
        """Order matters: the busy checks exist to protect a LIVE worker mid-claim. A dead
        one must not be able to reach them."""
        s = UPDATE.read_text()
        dead = s.index('docker ps -q -f "name=^/${NAME}$"')
        pull = s.index("docker pull -q")
        assert dead < pull


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
        src = IDLE.read_text()
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
        src = IDLE.read_text()
        assert 'print("unknown"); raise SystemExit' in src
        assert '[ "$training" != "no" ]' in src


class TestTheEngineIsAskedThroughTheSupervisor:
    """update-worker.sh reads the engine's running/queue_depth from the supervisor's /health
    (wanly-gpu-docker#83) and falls back to the engine itself for an image from before it."""

    SAME = "sha256:aaa"
    NEW = "sha256:bbb"

    def test_the_script_asks_the_control_port_first(self):
        text = IDLE.read_text()
        assert '${CONTROL_PORT:-8081}/health' in text
        assert text.index('${CONTROL_PORT:-8081}/health') < text.index("127.0.0.1:8190/health")

    def test_a_pre_supervisor_image_still_answers_through_the_fallback(self, tmp_path):
        """The docker stub answers every in-container curl with the engine's own body, which
        has no `services` list -- the supervisor path exits 4 and the fallback decides."""
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, engine_busy=0)
        assert "RECREATED" in r.stdout, r.stdout
        r = _run(tmp_path, running_image=self.SAME, latest_image=self.NEW, engine_busy=1)
        assert "RECREATED" not in r.stdout, r.stdout


class TestTheDevMount:
    """run-worker.sh's DEV_CODE_DIR block (#117). Staged like the updater tests: the real
    script against a fake docker that RECORDS the args of `docker run`, so what is asserted
    is what the container would actually be created with — not a substring of the source.

    The invariant under test is that a dev mount is never silent: it must pass a read-only
    mount and DEV_CODE=1 into the container (so fetch_engine.sh swaps the mount in and the
    banner/health shout), it must refuse the real queue without a second explicit flag, and a
    path that isn't a checkout must be refused outright.
    """

    def _stage(self, tmp_path, *, dev_code_dir=None, dev_allow_queue=None):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = run ]; then printf \'%s\\n\' "$@" > "%s"; echo fakecid; exit 0; fi\n'
            'if [ "$1" = inspect ]; then echo fakecid; exit 0; fi\n'
            "exit 0\n" % ("%s", bin_dir / "runargs.txt"))
        (bin_dir / "docker").chmod(0o755)
        # `ss` on the test box may genuinely have something on 8081; the script's port
        # pre-flight must see an empty answer or every test dies on a false conflict.
        (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
        (bin_dir / "ss").chmod(0o755)
        stage = tmp_path / "deploy"
        stage.mkdir()
        shutil.copy(RUN, stage / "run-worker.sh")
        jobs = tmp_path / "jobs"; jobs.mkdir()
        models = tmp_path / "models"; (models / "loras").mkdir(parents=True)
        env_text = ("QUEUE_URL=http://api.test:8001\nQUEUE_API_KEY=k\nFRIENDLY_NAME=3090.zero\n"
                    "JOBS_DIR=%s\nMODELS_DIR=%s\n" % (jobs, models))
        if dev_code_dir is not None:
            env_text += "DEV_CODE_DIR=%s\n" % dev_code_dir
        if dev_allow_queue is not None:
            env_text += "DEV_ALLOW_QUEUE=%s\n" % dev_allow_queue
        (stage / "worker.env").write_text(env_text)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]),
                   WORKER_ENV=str(stage / "worker.env"))
        r = subprocess.run(["bash", str(stage / "run-worker.sh")],
                           capture_output=True, text=True, env=env)
        args = (bin_dir / "runargs.txt").read_text() if (bin_dir / "runargs.txt").exists() else ""
        return r, args

    def _checkout(self, tmp_path):
        co = tmp_path / "checkout"
        (co / "engine").mkdir(parents=True)
        (co / "wanly_worker").mkdir()
        (co / "engine" / "app.py").write_text("")
        (co / "wanly_worker" / "control.py").write_text("")
        return co

    def test_a_dev_mount_needs_the_explicit_queue_flag(self, tmp_path):
        """Half-edited code claiming real segments is the #72 failure mode with a new
        mechanism. DEV_CODE_DIR alone must be refused, before anything is removed."""
        co = self._checkout(tmp_path)
        r, args = self._stage(tmp_path, dev_code_dir=co)
        assert r.returncode != 0
        assert "DEV_ALLOW_QUEUE" in r.stdout
        assert args == "", "must not have run docker at all"

    def test_with_both_flags_the_mount_and_the_marker_reach_the_container(self, tmp_path):
        co = self._checkout(tmp_path)
        r, args = self._stage(tmp_path, dev_code_dir=co, dev_allow_queue="1")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DEV MOUNT" in r.stdout and "NOT DEPLOYED CODE" in r.stdout
        # The fake records one docker argument per line; the mount is its own argument.
        assert "%s:/opt/dev-code:ro" % co in args.splitlines()
        assert "DEV_CODE=1" in args.splitlines()

    def test_a_path_that_is_not_the_repo_is_refused(self, tmp_path):
        not_repo = tmp_path / "random"
        not_repo.mkdir()
        r, args = self._stage(tmp_path, dev_code_dir=not_repo, dev_allow_queue="1")
        assert r.returncode != 0
        assert "not a wanly-gpu-docker checkout" in r.stdout
        assert args == ""

    def test_no_dev_code_dir_changes_nothing(self, tmp_path):
        """The default path must stay byte-for-byte the converge of #72: no /opt/dev-code,
        no DEV_CODE env. A stale env line is the risk this guards."""
        r, args = self._stage(tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "dev-code" not in args and "DEV_CODE=1" not in args
        assert args != "", "the container should still have been created"


# ---------------------------------------------------------------------------------------
# wanly-mode.sh (#131)
#
# A switch means run-worker.sh, and run-worker.sh recreates with `docker rm -f`. Every wrong
# decision here costs the same thing the updater's wrong decisions cost -- an in-flight
# segment, a model stage, a training run -- which is why both ask the same gate.
# ---------------------------------------------------------------------------------------

RENDER_LINE = "ltx-engine,lora-trainer,image-description,face-crop"
# "the container runs whatever worker.env says", as distinct from running=None, which models
# an inspect that cannot be read at all.
SAME_AS_FILE = object()


def _stage_mode(tmp, *, services=RENDER_LINE, running=SAME_AS_FILE, worker_status="online-idle",
                engine_busy=0, trainer=None, extra_env=""):
    """Stage wanly-mode.sh with the same fakes the updater gets.

    `services` is what worker.env says; `running` is what the CONTAINER was given, which is
    not the same question -- after a switch the file leads the container, and the gate reads
    the container.
    """
    running = services if running is SAME_AS_FILE else running
    bin_dir, stage = _stage(tmp, running_image="sha256:a", latest_image="sha256:a",
                            engine_busy=engine_busy, worker_status=worker_status,
                            trainer=trainer, services=running)
    shutil.copy(MODE, stage / "wanly-mode.sh")
    (stage / "wanly-mode.sh").chmod(0o755)
    # Records the line it was given, so a test can assert what would actually be created.
    (stage / "run-worker.sh").write_text(
        "#!/usr/bin/env bash\n"
        'set -a; . "$(dirname "$0")/worker.env"; set +a\n'
        'case "${MODE:-}" in caption)\n'
        '  out=""; for n in $(echo "$SERVICES" | tr "," " "); do\n'
        '    case "$n" in ltx-engine|lora-trainer) continue ;; esac\n'
        '    out="${out:+$out,}$n"; done; SERVICES="$out" ;;\n'
        'esac\n'
        'echo "RECREATED $SERVICES"\n')
    (stage / "run-worker.sh").chmod(0o755)
    (stage / "worker.env").write_text(
        "QUEUE_URL=http://api.test:8001\nQUEUE_API_KEY=k\nFRIENDLY_NAME=3090.zero\n"
        "SERVICES=%s\n#SERVICES=ltx-engine\n%s" % (services, extra_env))
    # The capability line is what a mode narrows; a caption-only box has no render stack in
    # it at all, which is a different thing from being in caption mode.
    return bin_dir, stage


def _mode(tmp, *args, **kw):
    bin_dir, stage = _stage_mode(tmp, **kw)
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
    r = subprocess.run(["bash", str(stage / "wanly-mode.sh"), *args],
                       capture_output=True, text=True, env=env)
    r.env_file = (stage / "worker.env").read_text()   # type: ignore[attr-defined]
    return r


class TestTheModeSwitch:
    def test_caption_drops_the_render_stack(self, tmp_path):
        r = _mode(tmp_path, "caption")
        assert "RECREATED image-description,face-crop" in r.stdout, r.stdout
        # MODE is the lever that gets written, not its expansion: worker.env then reads as
        # the choice that was made, and `MODE=caption ./run-worker.sh` does the same thing.
        assert "\nMODE=caption\n" in r.env_file

    def test_caption_NEVER_rewrites_the_capability_line(self, tmp_path):
        """SERVICES is what the box is equipped to do and is not a mode. Leaving it alone is
        what removes the "remember the full list to put back" failure entirely."""
        r = _mode(tmp_path, "caption")
        assert "\nSERVICES=%s\n" % RENDER_LINE in r.env_file
        assert "SERVICES_RENDER" not in r.env_file

    def test_render_is_just_MODE_ltx_engine(self, tmp_path):
        """Nothing to restore: SERVICES never moved, so going back is one word."""
        r = _mode(tmp_path, "render", services=RENDER_LINE,
                  extra_env="MODE=caption\n", running="image-description,face-crop")
        assert "\nMODE=ltx-engine\n" in r.env_file
        assert "RECREATED %s" % RENDER_LINE in r.stdout, r.stdout

    def test_caption_on_a_box_with_nothing_to_caption_with_is_refused(self, tmp_path):
        """Better than starting a container that runs nothing, boots clean and reports
        healthy -- diagnosed from another machine as "captioner unreachable"."""
        r = _mode(tmp_path, "caption", services="ltx-engine,lora-trainer")
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_a_commented_example_line_is_not_rewritten(self, tmp_path):
        """worker.env.example ships `#SERVICES=` lines. Rewriting one would uncomment it on
        the next source and silently win over the real one."""
        r = _mode(tmp_path, "caption")
        assert "#SERVICES=ltx-engine\n" in r.env_file

    def test_switching_to_the_mode_it_is_already_in_does_nothing(self, tmp_path):
        r = _mode(tmp_path, "render")
        assert "nothing to do" in r.stdout
        assert "RECREATED" not in r.stdout

    def test_status_never_recreates_anything(self, tmp_path):
        r = _mode(tmp_path, "status")
        assert r.returncode == 0
        assert "RECREATED" not in r.stdout
        assert "mode render" in r.stdout

    def test_an_unknown_argument_exits_nonzero(self, tmp_path):
        """`wanly-mode.sh caprion || echo failed` must say failed rather than report a switch
        that never happened."""
        r = _mode(tmp_path, "caprion")
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout


class TestTheModeSwitchRefusesWhileBusy:
    """run-worker.sh recreates with `docker rm -f`. Every one of these is a segment, a model
    stage or a training run that a switch would have destroyed."""

    def test_a_busy_worker_blocks_the_switch(self, tmp_path):
        r = _mode(tmp_path, "caption", worker_status="online-busy")
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_it_says_how_to_caption_without_interrupting(self, tmp_path):
        """The refusal has to name the drain, or the next move is --force."""
        r = _mode(tmp_path, "caption", worker_status="online-busy")
        assert "drain" in r.stdout

    def test_a_refusal_leaves_worker_env_untouched(self, tmp_path):
        """The gate runs BEFORE anything is written. A file left mid-switch would be applied
        by the next recreate -- the update timer's, minutes later, with nobody watching."""
        r = _mode(tmp_path, "caption", worker_status="online-busy")
        assert "\nSERVICES=%s\n" % RENDER_LINE in r.env_file
        assert "SERVICES_RENDER" not in r.env_file

    def test_a_rendering_engine_blocks_the_switch(self, tmp_path):
        r = _mode(tmp_path, "caption", engine_busy=1)
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_a_training_run_blocks_the_switch(self, tmp_path):
        r = _mode(tmp_path, "caption", trainer="training")
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_force_switches_anyway(self, tmp_path):
        r = _mode(tmp_path, "caption", "--force", worker_status="online-busy")
        assert "RECREATED image-description,face-crop" in r.stdout, r.stdout


class TestTheGateDoesNotStrandABoxInCaptionMode:
    """Found live on the 3090, 2026-09-25: `render` refused to switch back, forever.

    `online-idle` is published by the RENDER DAEMON and only by it. A container with no
    ltx-engine has no daemon and sits at a plain `online`, so requiring online-idle refused
    every switch OUT of caption mode on the absence of the very service being restored.
    """

    CAPTION = "image-description,face-crop"

    def test_plain_online_is_idle_when_there_is_no_engine_to_be_busy(self, tmp_path):
        r = _mode(tmp_path, "render", services=RENDER_LINE, running=self.CAPTION,
                  worker_status="online", engine_busy=None, trainer=None,
                  extra_env="MODE=caption\n")
        assert "RECREATED %s" % RENDER_LINE in r.stdout, r.stdout

    def test_plain_online_is_NOT_idle_when_the_engine_is_there(self, tmp_path):
        """With a render daemon on the box, `online` is a status that has not reached
        online-idle -- mid-boot, or a failed push. Not evidence of idleness."""
        r = _mode(tmp_path, "caption", worker_status="online")
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_a_missing_engine_is_not_an_unreadable_one(self, tmp_path):
        """`unknown` engine health means two different things and only one is safe: there is
        no engine on this box (nothing to ask) vs. the engine is not answering (mid-boot,
        staging 58 GB). The container's own SERVICES is what tells them apart."""
        r = _mode(tmp_path, "render", services=RENDER_LINE, running=self.CAPTION,
                  engine_busy=None, extra_env="MODE=caption\n")
        assert "RECREATED" in r.stdout, r.stdout

    def test_an_unreadable_container_is_never_exempt(self, tmp_path):
        """THE FAIL-SAFE. If the inspect cannot be read we do not know what is running, and
        answering "absent, therefore exempt" would turn every unreadable-engine case -- the
        ones that exist precisely to count as busy -- into a green light to recreate.

        `running=None` is an inspect that fails outright, beside an engine that answers
        nothing: the exact shape of a box mid-boot, staging 58 GB."""
        r = _mode(tmp_path, "caption", running=None, engine_busy=None)
        assert r.returncode == 1, r.stdout
        assert "RECREATED" not in r.stdout


class TestPauseAndResume:
    """The no-recreate path (#131 follow-up): stop claiming, let the segment finish, park.

    This is the one that answers "let me caption while jobs are queued", so the thing worth
    pinning is the WAIT -- without it `pause` returns while the render is still running, and
    wanly-api has already stopped refusing captions (it refuses only on `online-busy`, and
    `POST /drain` sets `draining` immediately). A caption then lands on a card that is ~23 of
    24 GB into a render.
    """

    @staticmethod
    def _curl(stage_bin, *, status="online-busy", engine_calls_before_idle=1):
        """A curl that answers /workers, and logs the drain calls it is asked to make."""
        (stage_bin / "curl").write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do case "$a" in *"/drain")'
            ' echo "$*" >> "%s"; echo "{}"; exit 0 ;; esac; done\n' % (stage_bin / "drain.log")
            + "echo '[{\"id\":\"w-1\",\"friendly_name\":\"3090.zero\",\"status\":\"%s\"}]'\n" % status)
        (stage_bin / "curl").chmod(0o755)

    def _run(self, tmp_path, *args, status="online-busy", engine_busy=0, services=None):
        bin_dir, stage = _stage_mode(
            tmp_path, worker_status=status, engine_busy=engine_busy,
            **({"running": services} if services is not None else {}))
        self._curl(bin_dir, status=status)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
        r = subprocess.run(["bash", str(stage / "wanly-mode.sh"), *args],
                           capture_output=True, text=True, env=env)
        log = bin_dir / "drain.log"
        r.drain = log.read_text() if log.exists() else ""   # type: ignore[attr-defined]
        return r

    def test_pause_drains_the_worker_by_its_own_id(self, tmp_path):
        """The id is looked up from FRIENDLY_NAME. Hand-assembling a UUID into a curl is why
        this path went unused."""
        r = self._run(tmp_path, "pause")
        assert "POST" in r.drain and "/workers/w-1/drain" in r.drain, r.drain

    def test_pause_reports_parked_once_the_engine_is_clear(self, tmp_path):
        r = self._run(tmp_path, "pause", engine_busy=0)
        assert r.returncode == 0
        assert "parked" in r.stdout

    def test_pause_does_NOT_claim_parked_while_the_engine_is_unreadable(self, tmp_path):
        """THE SAFETY. An unreadable engine is not a finished one, and saying "caption away"
        there is the whole hazard this command exists to close."""
        r = self._run(tmp_path, "pause", engine_busy=None)
        assert r.returncode == 1
        assert "parked" not in r.stdout
        assert "NOT safe to caption" in r.stdout

    def test_pause_still_drains_even_when_it_cannot_confirm(self, tmp_path):
        """Refusing to drain would leave the box claiming MORE work, which is worse than a
        drain it could not confirm. The drain stands; only the all-clear is withheld."""
        r = self._run(tmp_path, "pause", engine_busy=None)
        assert "/workers/w-1/drain" in r.drain

    def test_a_caption_mode_box_has_no_engine_to_wait_for(self, tmp_path):
        r = self._run(tmp_path, "pause", engine_busy=None, services="image-description")
        assert r.returncode == 0
        assert "parked" in r.stdout

    def test_resume_cancels_the_drain(self, tmp_path):
        r = self._run(tmp_path, "resume", status="draining")
        assert r.returncode == 0
        assert "DELETE" in r.drain and "/workers/w-1/drain" in r.drain, r.drain


class TestAPausedBoxIsStillSwitchable:
    """`draining` means NOT CLAIMING ANY MORE WORK -- stronger than online-idle, not weaker.

    Without this a box paused for captioning could never be switched or updated: refused
    forever on a status it was deliberately put into.
    """

    def test_a_parked_box_may_be_switched(self, tmp_path):
        r = _mode(tmp_path, "caption", worker_status="draining", engine_busy=0)
        assert "RECREATED" in r.stdout, r.stdout

    def test_but_not_while_its_last_segment_is_still_rendering(self, tmp_path):
        """What `draining` does NOT say is whether the segment already in flight has
        finished. That is the engine's signal, and it still has to pass."""
        r = _mode(tmp_path, "caption", worker_status="draining", engine_busy=1)
        assert r.returncode == 1
        assert "RECREATED" not in r.stdout

    def test_the_refusal_names_the_status_the_api_actually_reported(self, tmp_path):
        """Not the one the gate synthesised -- "despite status 'online-idle'" on a box the
        API called `draining` sends you looking for the wrong thing."""
        r = _mode(tmp_path, "caption", worker_status="draining", engine_busy=1)
        assert "draining" in r.stdout


class TestTheModeFlag:
    """MODE is a DOCKER ENV FLAG. run-worker.sh does not interpret it -- it hands it to the
    container, which decides (wanly_worker/registry.py select_mode). That is what makes
    `docker run -e MODE=caption ...` the whole switch, with no wrapper in the path."""

    def _run_worker(self, tmp_path, env_lines, **env):
        stage = tmp_path / "deploy"
        stage.mkdir(exist_ok=True)
        shutil.copy(RUN, stage / "run-worker.sh")
        (stage / "worker.env").write_text(env_lines)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "docker").write_text('#!/usr/bin/env bash\necho "docker $*"\n')
        (bin_dir / "docker").chmod(0o755)
        for stub in ("ss", "nvidia-smi"):
            (bin_dir / stub).write_text("#!/usr/bin/env bash\nexit 0\n")
            (bin_dir / stub).chmod(0o755)
        for d in ("jobs", "models", "models/loras", "ollama", "runs"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        e = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]), **env)
        return subprocess.run(["bash", str(stage / "run-worker.sh")],
                              capture_output=True, text=True, env=e)

    def _env(self, tmp_path, extra=""):
        return ("QUEUE_API_KEY=k\nQUEUE_URL=http://api.test:8001\n"
                "FRIENDLY_NAME=3090.zero\nIMAGE=x:full\n"
                "JOBS_DIR=%s/jobs\nMODELS_DIR=%s/models\n"
                "OLLAMA_HOST_STORE=%s/ollama\nLORA_RUNS_DIR=%s/runs\n"
                "SERVICES=ltx-engine,lora-trainer,image-description\n%s"
                % (tmp_path, tmp_path, tmp_path, tmp_path, extra))

    def test_MODE_reaches_the_container_as_an_env_flag(self, tmp_path):
        r = self._run_worker(tmp_path, self._env(tmp_path, "MODE=caption\n"))
        assert '-e MODE=caption' in r.stdout, r.stdout + r.stderr

    def test_it_works_as_a_one_off_env_override_with_no_file_edit(self, tmp_path):
        r = self._run_worker(tmp_path, self._env(tmp_path), MODE="caption")
        assert '-e MODE=caption' in r.stdout, r.stdout + r.stderr

    def test_SERVICES_IS_NOT_REWRITTEN(self, tmp_path):
        """The capability line goes to the container untouched. This is the whole design:
        nothing has to remember the full list in order to put it back."""
        r = self._run_worker(tmp_path, self._env(tmp_path, "MODE=caption\n"))
        assert '-e SERVICES=ltx-engine,lora-trainer,image-description' in r.stdout, r.stdout

    def test_no_MODE_passes_an_empty_flag_not_a_missing_one(self, tmp_path):
        """An unset MODE still has to be passed, or a recreate would inherit whatever the
        previous container had in an image default."""
        r = self._run_worker(tmp_path, self._env(tmp_path))
        assert '-e MODE=' in r.stdout, r.stdout + r.stderr

    def test_an_unrecognised_MODE_is_refused_BEFORE_docker_run(self, tmp_path):
        """This script's rule: refuse before `docker rm -f`, never after. A typo must not
        cost the running worker, even though the container would also reject it."""
        r = self._run_worker(tmp_path, self._env(tmp_path, "MODE=captions\n"))
        assert r.returncode != 0
        assert "not a mode" in r.stdout
        assert "docker run" not in r.stdout
        assert "rm -f" not in r.stdout
