"""The worker must come back on its own after a cold boot (wanly-gpu-docker#213).

docker restores the `unless-stopped` worker before /var/run/cdi/nvidia.yaml (tmpfs) exists,
because the toolkit's nvidia-cdi-refresh.service is ordered After=multi-user.target, which
waits for docker. The start fails with "unresolvable CDI devices" and docker never retries.
3090a lost the race on 6 boots out of 6; 3090b stayed dead until a hand `docker start`.

These tests guard the three parts of the fix:

  * wanly-cdi-spec.service writes the spec BEFORE docker, without an ordering cycle, into tmpfs
    (never a persistent /etc/cdi spec that a driver update would leave stale);
  * install-cdi-spec.sh verifies the ordering and the pinned card rather than trusting enable;
  * the update timer ticks shortly after every boot, so any other dead-at-boot worker is
    recreated in minutes, and install-timer.sh never adds a second, differently-locked timer.
"""
import os
import pathlib
import re
import shutil
import stat
import subprocess

import pytest

DEPLOY = pathlib.Path(__file__).parent.parent / "deploy"
UNIT = DEPLOY / "wanly-cdi-spec.service"
INSTALL = DEPLOY / "install-cdi-spec.sh"
TIMER = DEPLOY / "wanly-worker-update.timer"
INSTALL_TIMER = DEPLOY / "install-timer.sh"

U3090 = "GPU-522c32cc-7ac2-9dcd-f1d5-4f5390e68f0b"


def _directives(path):
    """{key: [values]} for every non-comment line, so comments that NAME a directive (to say
    why it is not used) never satisfy or fail a test."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.setdefault(k, []).append(v)
    return out


def _words(d, key):
    return " ".join(d.get(key, [])).split()


# ------------------------------------------------------------------ the unit


class TestTheSpecIsWrittenBeforeDocker:
    def test_it_is_ordered_before_docker(self):
        assert "docker.service" in _words(_directives(UNIT), "Before")

    def test_docker_pulls_it_in(self):
        """Pulled in by docker itself, so it runs whenever docker starts -- not only when
        multi-user.target happens to reach it."""
        assert "docker.service" in _words(_directives(UNIT), "WantedBy")

    def test_a_failure_does_not_stop_docker(self):
        """Wants (via WantedBy), never Requires/RequiredBy: the non-GPU containers on the box
        must still come up if the spec cannot be written."""
        d = _directives(UNIT)
        assert "RequiredBy" not in d and "Requires" not in d and "BindsTo" not in d

    def test_no_ordering_cycle_through_multi_user_target(self):
        """multi-user.target waits for docker. Anything this unit orders itself after that is
        itself after multi-user.target -- the vendor nvidia-cdi-refresh.service is -- closes a
        cycle, and systemd breaks a cycle by dropping a job from the boot."""
        after = _words(_directives(UNIT), "After")
        for bad in ("multi-user.target", "nvidia-cdi-refresh.service", "docker.service",
                    "graphical.target", "default.target"):
            assert bad not in after, f"After={bad} is an ordering cycle with docker"

    def test_it_waits_for_the_driver(self):
        after = _words(_directives(UNIT), "After")
        assert "systemd-modules-load.service" in after

    def test_the_spec_stays_on_tmpfs(self):
        """A spec in /etc/cdi survives a driver update and then describes the OLD driver's
        libraries -- injected into the worker at exactly the boot after the update."""
        d = _directives(UNIT)
        assert "NVIDIA_CTK_CDI_OUTPUT_FILE_PATH=/var/run/cdi/nvidia.yaml" in d.get("Environment", [])
        code = "\n".join(v for vs in d.values() for v in vs)
        assert "/etc/cdi" not in code

    def test_it_generates_the_way_the_vendor_unit_does(self):
        """Same command and the same override file, so the two writers of nvidia.yaml can
        never disagree about its content (identical on 3090a and 3090b, 2026-10-15)."""
        d = _directives(UNIT)
        assert d["ExecStart"] == ["/usr/bin/nvidia-ctk cdi generate"]
        assert "-/etc/nvidia-container-toolkit/nvidia-cdi-refresh.env" in d["EnvironmentFile"]

    def test_it_reruns_on_every_docker_start(self):
        """With RemainAfterExit=yes the unit would stay active, and a later `systemctl restart
        docker` would not regenerate the spec."""
        d = _directives(UNIT)
        assert d.get("Type") == ["oneshot"]
        assert d.get("RemainAfterExit", ["no"]) != ["yes"]

    def test_it_cannot_hold_docker_forever(self):
        d = _directives(UNIT)
        assert "TimeoutStartSec" in d and d["TimeoutStartSec"] != ["infinity"]

    def test_shell_dollars_are_escaped_for_systemd(self):
        """systemd expands `$i` in Exec lines itself (to nothing) unless written `$$i`; the
        wait loop would then be `[ -lt 30 ]`, an error on every pass."""
        for key in ("ExecStartPre", "ExecStart"):
            for v in _directives(UNIT).get(key, []):
                assert not re.search(r"(?<!\$)\$(?!\$)[A-Za-z_(]", v.replace("$$", "")), v


class TestTheDriverWait:
    """The ExecStartPre loop, run for real with a fake nvidia-smi and a no-op sleep."""

    def _run(self, tmp_path, fail_times):
        line = _directives(UNIT)["ExecStartPre"][0]
        m = re.match(r"^/bin/sh -c '(.*)'$", line)
        assert m, line
        script = m.group(1).replace("$$", "$")  # what systemd hands the shell
        bin_ = tmp_path / "bin"
        bin_.mkdir()
        count = tmp_path / "count"
        count.write_text("0")
        (bin_ / "nvidia-smi").write_text(
            "#!/bin/sh\n"
            f'n=$(cat {count}); n=$((n+1)); echo $n > {count}\n'
            f'[ $n -gt {fail_times} ]\n')
        (bin_ / "sleep").write_text("#!/bin/sh\nexit 0\n")
        for f in bin_.iterdir():
            f.chmod(0o755)
        env = {"PATH": f"{bin_}:/usr/bin:/bin"}
        r = subprocess.run(["/bin/sh", "-c", script], env=env, capture_output=True,
                           text=True, timeout=30)
        return r, int(count.read_text())

    def test_it_returns_as_soon_as_the_driver_answers(self, tmp_path):
        r, calls = self._run(tmp_path, fail_times=2)
        assert r.returncode == 0
        assert calls == 3
        assert "did not answer" not in r.stdout

    def test_it_gives_up_after_30_tries_and_still_lets_generate_run(self, tmp_path):
        r, calls = self._run(tmp_path, fail_times=1000)
        assert r.returncode == 0, "a failing pre-step would skip the generate entirely"
        assert calls == 30
        assert "did not answer" in r.stdout


# ------------------------------------------------------------------ the installer


class TestTheInstaller:
    """Run for real against fakes: id, install, systemctl, nvidia-ctk."""

    def _box(self, tmp_path, *, root=True, ordered=True, listed=(U3090,), pin=U3090,
             start_ok=True):
        deploy = tmp_path / "deploy"
        deploy.mkdir()
        shutil.copy(INSTALL, deploy / INSTALL.name)
        shutil.copy(UNIT, deploy / UNIT.name)
        if pin is not None:
            (deploy / "worker.env").write_text(
                f"QUEUE_API_KEY=secret$notsourced\nGPU_UUID={pin}\n")
        spec = tmp_path / "nvidia.yaml"
        bin_ = tmp_path / "bin"
        bin_.mkdir()
        log = tmp_path / "calls"
        devices = "\n".join(["nvidia.com/gpu=0", *[f"nvidia.com/gpu={u}" for u in listed],
                             "nvidia.com/gpu=all"])
        fakes = {
            "id": f"#!/bin/sh\necho {0 if root else 1000}\n",
            "install": f'#!/bin/sh\necho "install $*" >> {log}\n',
            "systemctl": (
                "#!/bin/sh\n"
                f'echo "systemctl $*" >> {log}\n'
                'case "$1 $2 $3 $4" in\n'
                '  "show docker.service -p Wants") echo "containerd.service wanly-cdi-spec.service";;\n'
                f'  "show docker.service -p After") echo "containerd.service {"wanly-cdi-spec.service" if ordered else "network-online.target"}";;\n'
                f'  start*) {"echo spec > " + str(spec) if start_ok else "exit 1"};;\n'
                "esac\n"),
            "journalctl": "#!/bin/sh\nexit 0\n",
            "nvidia-ctk": f"#!/bin/sh\ncat <<'EOF'\n{devices}\nEOF\n",
        }
        for name, body in fakes.items():
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "CDI_SPEC": str(spec)}
        r = subprocess.run(["bash", str(deploy / INSTALL.name)], env=env,
                           capture_output=True, text=True, timeout=30)
        return r, (log.read_text() if log.exists() else "")

    def test_a_good_box_installs_enables_and_runs_it(self, tmp_path):
        r, calls = self._box(tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "install -m 644" in calls and "wanly-cdi-spec.service" in calls
        assert "systemctl enable wanly-cdi-spec.service" in calls
        assert "systemctl start wanly-cdi-spec.service" in calls
        assert f"lists the pinned card nvidia.com/gpu={U3090}" in r.stdout

    def test_it_never_restarts_docker(self, tmp_path):
        """Restarting docker would take the live worker down with it."""
        _, calls = self._box(tmp_path)
        assert not re.search(r"systemctl (re)?start docker|systemctl (stop|restart) ", calls)

    def test_it_refuses_without_root(self, tmp_path):
        r, calls = self._box(tmp_path, root=False)
        assert r.returncode == 1 and "sudo" in r.stdout
        assert calls == ""

    def test_enabled_but_not_ordered_is_a_failure(self, tmp_path):
        """The whole bug is ordering. `enable` succeeding proves nothing about it."""
        r, _ = self._box(tmp_path, ordered=False)
        assert r.returncode == 1
        assert "not ordered after" in r.stdout

    def test_a_pinned_card_missing_from_the_spec_is_a_failure(self, tmp_path):
        r, _ = self._box(tmp_path, listed=())
        assert r.returncode == 1
        assert f"pins {U3090}" in r.stdout

    def test_an_unpinned_box_needs_only_gpu_all(self, tmp_path):
        r, _ = self._box(tmp_path, pin=None, listed=())
        assert r.returncode == 0, r.stdout + r.stderr

    def test_a_failed_run_is_a_failure(self, tmp_path):
        r, _ = self._box(tmp_path, start_ok=False)
        assert r.returncode == 1
        assert "failed to run" in r.stdout

    def test_it_is_executable(self):
        assert os.stat(INSTALL).st_mode & stat.S_IXUSR


# ------------------------------------------------------------------ the backstop


class TestTheTimerTicksAfterBoot:
    def test_it_ticks_shortly_after_every_boot(self):
        d = _directives(TIMER)
        assert d.get("OnBootSec"), "no boot tick: a dead-at-boot worker waits for the next slot"
        v = d["OnBootSec"][0]
        m = re.fullmatch(r"(\d+)\s*(s|sec|m|min)", v)
        assert m, v
        secs = int(m.group(1)) * (60 if m.group(2).startswith("m") else 1)
        # Not before docker has restored its containers (it would race docker's own start),
        # and not so late that it is no better than the half-hour slot.
        assert 60 <= secs <= 600, v

    def test_the_boot_tick_does_not_replace_the_absolute_schedule(self):
        """#76: OnBootSec alone (with OnUnitActiveSec) stopped scheduling after the first
        elapse. OnCalendar keeps a next elapse forever."""
        d = _directives(TIMER)
        assert d.get("OnCalendar")
        assert "OnUnitActiveSec" not in d


class TestOneTimerPerBox:
    """3090a runs the timer as a lingering USER unit. A system copy beside it locks under
    /run/lock while the user copy locks under $XDG_RUNTIME_DIR, so the two could both recreate
    the worker at once (#97). install-timer.sh must refuse rather than add one."""

    def _run(self, tmp_path, user_timer):
        home = tmp_path / "home"
        wants = home / ".config/systemd/user/timers.target.wants"
        wants.mkdir(parents=True)
        if user_timer:
            (wants / "wanly-worker-update.timer").write_text("")
        bin_ = tmp_path / "bin"
        bin_.mkdir()
        log = tmp_path / "calls"
        fakes = {
            "id": "#!/bin/sh\necho 0\n",
            "getent": f"#!/bin/sh\necho 'david:x:1000:1000::{home}:/bin/bash'\n",
            "install": f'#!/bin/sh\necho "install $*" >> {log}\n',
            "systemctl": (f'#!/bin/sh\necho "systemctl $*" >> {log}\n'
                          '[ "$1" = show ] && echo "Thu 2026-10-15 12:00:00 MDT"\nexit 0\n'),
        }
        for name, body in fakes.items():
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "SUDO_USER": "david"}
        r = subprocess.run(["bash", str(INSTALL_TIMER)], env=env, capture_output=True,
                           text=True, timeout=30)
        return r, (log.read_text() if log.exists() else "")

    def test_it_refuses_beside_a_user_timer(self, tmp_path):
        r, calls = self._run(tmp_path, user_timer=True)
        assert r.returncode == 1
        assert "USER timer" in r.stdout
        assert "install" not in calls and "enable" not in calls

    def test_it_installs_on_a_box_without_one(self, tmp_path):
        r, calls = self._run(tmp_path, user_timer=False)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "enable --now wanly-worker-update.timer" in calls
