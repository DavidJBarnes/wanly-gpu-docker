"""What a service in this container has to be able to say about itself.

The container carries EVERY service; the `SERVICES` flag decides which ones actually run.
That split is deliberate. Building a separate image per combination means a matrix of images
that drift apart, and the thing that goes wrong on a box is almost never "the binary is
absent" -- it is "the binary is there and nobody started it", which a flag makes visible and
an image does not.

Each service owns four honest answers:

    preflight()   can this even work here? Raise, and the container refuses to boot.
    command()     how to start it.
    ready()       is it actually serving? Not "did the process spawn" -- ollama's process is
                  up long before it answers, and a readiness probe that watches the process
                  reports success while every request 500s.
    after_ready() what has to be true once it answers, e.g. the model being present.

preflight is separate from ready on purpose, and it is the one that earns its keep. A missing
model store or a full disk is knowable in a millisecond, and finding it out that way is worth
a great deal more than finding it out 5 GB into a pull -- which is exactly the shape of
wanly-gpu-docker#77, where a container that could not possibly work spent nine minutes
downloading before anything said so.
"""
from __future__ import annotations

import abc


class PreflightError(RuntimeError):
    """This service cannot work on this box, and we know it before starting anything."""


class Service(abc.ABC):
    #: The name used in the SERVICES flag. Lower case, no spaces.
    name: str = ""
    #: The port it serves on, published from the container.
    port: int = 0
    #: One line, shown in /health so the answer to "what is this?" is in the response.
    summary: str = ""
    #: How long this service gets to start answering. None = the supervisor's default.
    #: Per service because one number cannot cover the room: ComfyUI's cold import is
    #: minutes, and the render daemon syncs every character LoRA before it registers.
    ready_timeout_s: float | None = None
    #: Working directory for the child. None = inherit. The engine imports its sibling
    #: modules from cwd and the daemon reads its .env from cwd.
    cwd: str | None = None
    #: Where the child's stdout goes. None = the container's own stdout. ComfyUI is chatty
    #: enough to drown the boot log, and its tail is what a startup failure needs.
    log_path: str | None = None
    #: How long stop() waits after SIGTERM before SIGKILL. The daemon may be finishing a
    #: segment; a render takes up to 27 minutes.
    stop_grace_s: float = 20.0
    #: The SERVICES name this service belongs to, when a name expands to several processes.
    group: str = ""

    def preflight(self) -> None:
        """Raise PreflightError if this service cannot possibly work here."""

    @abc.abstractmethod
    def command(self) -> list[str]:
        """argv for the process to run."""

    def env(self) -> dict[str, str]:
        """Environment to add on top of the container's own."""
        return {}

    @abc.abstractmethod
    async def ready(self, client) -> bool:
        """True once the service is actually answering requests."""

    async def after_ready(self, client) -> None:
        """Anything that must hold once it answers. Raise to fail the boot."""

    def details(self) -> dict:
        """Extra per-service fields for /health. Must never raise."""
        return {}
