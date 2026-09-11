"""Start the enabled services, prove each one answers, and notice when one dies.

WHY A SUPERVISOR AND NOT JUST A LONGER ENTRYPOINT

    wanly-console#426's complaint about the services this replaces is that *nothing
    supervises them*: keyframe-server had been up 32 hours with nobody watching, and the
    ollama models on the 2070 are "whatever was last pulled". The specific damage that does
    is not downtime -- it is that a broken service is indistinguishable from a healthy one
    until something far away fails. `KEYFRAME_URL` pointed at the wrong host for a month and
    the only symptom was a call that silently never ran.

    So the job here is not restarting things. It is being ABLE TO ANSWER: what was asked for,
    what actually started, and what is answering right now.

WHY A DEAD CHILD TAKES THE CONTAINER WITH IT

    Restarting a child in-process means writing backoff, crash loops and restart budgets --
    all of which Docker already has, and does better. Exiting hands the problem to
    `--restart unless-stopped`, and makes the failure visible in `docker ps` rather than
    buried in a log this process would have to be asked for.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time

from wanly_worker.service import PreflightError, Service

#: How long a service gets to start answering before the boot is called a failure. Generous:
#: ollama loads no model at startup, so this is process start and a port bind, but a cold
#: page-cache on a spinning store has been slower than it looks like it should be.
READY_TIMEOUT_S = float(os.environ.get("READY_TIMEOUT_S", "120"))


class ServiceState:
    def __init__(self, service: Service):
        self.service = service
        self.proc: asyncio.subprocess.Process | None = None
        self.ready = False
        self.error: str | None = None
        self.started_at: float | None = None

    def snapshot(self) -> dict:
        running = self.proc is not None and self.proc.returncode is None
        return {
            "name": self.service.name,
            # The SERVICES name this process belongs to: `ltx-engine` for all three of its
            # processes. What the Workers page should read as the capability.
            "group": self.service.group or self.service.name,
            "summary": self.service.summary,
            "port": self.service.port,
            "running": running,
            "ready": self.ready and running,
            "pid": self.proc.pid if self.proc else None,
            "exit_code": self.proc.returncode if self.proc else None,
            "uptime_s": round(time.time() - self.started_at) if self.started_at else None,
            "error": self.error,
            **self.service.details(),
        }


class Supervisor:
    def __init__(self, services: list[Service]):
        self.states = [ServiceState(s) for s in services]
        self.failed: str | None = None
        self._watchdog: asyncio.Task | None = None

    # ---------------------------------------------------------------- start

    async def start(self, client) -> None:
        """Preflight everything, then start everything. Raises to abort the boot.

        ALL preflights run before ANY process starts. Half-started is the worst outcome: the
        container looks alive, serves one of two services, and the missing one only surfaces
        from another host.
        """
        for st in self.states:
            try:
                st.service.preflight()
            except PreflightError as e:
                raise PreflightError(f"{st.service.name}: {e}") from e
        print(f"preflight OK for {', '.join(s.service.name for s in self.states)}",
              flush=True)

        for st in self.states:
            await self._start_one(st, client)

        self._watchdog = asyncio.create_task(self._watch())

    async def _start_one(self, st: ServiceState, client) -> None:
        svc = st.service
        env = {**os.environ, **svc.env()}
        argv = svc.command()
        timeout = svc.ready_timeout_s or READY_TIMEOUT_S
        print(f"starting {svc.name}: {' '.join(argv)}"
              + (f"  (log: {svc.log_path})" if svc.log_path else ""), flush=True)
        # A child with its own log file keeps the boot log readable; its tail is printed if it
        # dies during startup, which is what a startup failure needs to be diagnosable.
        out = open(svc.log_path, "ab") if svc.log_path else None
        try:
            st.proc = await asyncio.create_subprocess_exec(
                *argv, env=env, cwd=svc.cwd, stdout=out, stderr=subprocess.STDOUT)
        finally:
            if out is not None:
                out.close()
        st.started_at = time.time()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if st.proc.returncode is not None:
                self._print_tail(svc)
                raise RuntimeError(
                    f"{svc.name} exited with {st.proc.returncode} before it answered")
            if await svc.ready(client):
                # A probe answering is not proof that OUR process is the one answering. The
                # obvious case is a leftover server still holding the port -- on 2070.zero
                # the host ollama binds 11434, and if it were still up when this started,
                # our ollama would die of the port conflict while the probe sailed through
                # against the survivor. Confirm the child we spawned is the one alive.
                #
                # Waited on rather than polled: `returncode` is only set once the child
                # watcher has reaped, which has not necessarily happened by the time a fast
                # probe returns. A bounded wait() settles it either way, and costs a quarter
                # of a second once per service at boot.
                try:
                    await asyncio.wait_for(st.proc.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass          # still running — it really is ours
                else:
                    raise RuntimeError(
                        f"{svc.name} answered on :{svc.port} but our process had already "
                        f"exited with {st.proc.returncode} — something else holds that port")
                st.ready = True
                break
            await asyncio.sleep(1)
        else:
            self._print_tail(svc)
            raise RuntimeError(
                f"{svc.name} did not answer on :{svc.port} within {timeout:.0f}s")

        print(f"{svc.name} is answering on :{svc.port} "
              f"({time.time() - st.started_at:.1f}s)", flush=True)
        await svc.after_ready(client)
        print(f"{svc.name} ready", flush=True)

    @staticmethod
    def _print_tail(svc: Service, lines: int = 50) -> None:
        if not svc.log_path:
            return
        try:
            with open(svc.log_path, "rb") as fh:
                tail = fh.read()[-8000:].decode("utf8", "replace").splitlines()[-lines:]
            print(f"--- last {len(tail)} lines of {svc.log_path} ---", flush=True)
            for line in tail:
                print(line, flush=True)
        except OSError:
            pass

    # ------------------------------------------------------------- watchdog

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(5)
            for st in self.states:
                if st.proc is not None and st.proc.returncode is not None:
                    st.ready = False
                    st.error = f"exited with {st.proc.returncode}"
                    self.failed = st.service.name
                    print(f"!! {st.service.name} exited with {st.proc.returncode} — "
                          f"stopping the container so Docker restarts it", flush=True)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return

    # ----------------------------------------------------------------- stop

    async def stop(self) -> None:
        if self._watchdog:
            self._watchdog.cancel()
        # Reverse start order: the daemon stops claiming before the engine it drives goes.
        for st in reversed(self.states):
            if st.proc is None or st.proc.returncode is not None:
                continue
            print(f"stopping {st.service.name}", flush=True)
            st.proc.terminate()
            try:
                await asyncio.wait_for(st.proc.wait(), timeout=st.service.stop_grace_s)
            except asyncio.TimeoutError:
                print(f"{st.service.name} ignored SIGTERM — killing it", flush=True)
                st.proc.kill()
                await st.proc.wait()

    # --------------------------------------------------------------- health

    def snapshot(self) -> list[dict]:
        return [st.snapshot() for st in self.states]


def gpu_snapshot() -> dict | None:
    """What the card looks like right now. Never raises; absent is a valid answer.

    Read through nvidia-smi rather than a binding, so nothing here depends on a python CUDA
    package being installed and correct -- which is one more thing that can be subtly wrong
    while looking fine.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, total, used = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "vram_total_mib": int(total), "vram_used_mib": int(used),
                "vram_free_mib": int(total) - int(used)}
    except Exception:
        return None
