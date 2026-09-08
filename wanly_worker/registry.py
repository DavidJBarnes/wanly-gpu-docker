"""Which services exist, and which of them this container was asked to run.

`SERVICES` is a comma-separated list: SERVICES=ltx-engine,lora-trainer.

A NAME MAY EXPAND TO SEVERAL PROCESSES. `ltx-engine` is ComfyUI, the engine API and the
render daemon, started in that order; the flag names the capability and the registry knows
what it takes. /health lists every process; `provides` reports the names that were asked for.

AN UNKNOWN NAME IS FATAL, and that is the whole point of parsing it here rather than reading
the variable where it is used. A typo that is quietly ignored produces a container that boots
clean, reports healthy, and serves nothing -- and the caller learns about it as "captioner
unreachable" from a completely different machine. Failing at boot with the list of real names
turns a cross-host mystery into one line of log.

An EMPTY list is fatal for the same reason: a services container running no services is not a
degraded state to tolerate, it is a misconfiguration to report.
"""
from __future__ import annotations

from typing import Callable

from wanly_worker.service import Service
from wanly_worker.services.ltx_engine import ltx_engine_group

#: Every service the image can run, by the name the SERVICES flag uses. A value is a factory
#: returning the ordered list of processes that name stands for. Adding one here is the only
#: registration step.
KNOWN: dict[str, Callable[[], list[Service]]] = {
    "ltx-engine": ltx_engine_group,
}


class ConfigError(RuntimeError):
    """SERVICES named something this image cannot run."""


def parse_services(raw: str | None, known: dict | None = None) -> list[str]:
    """Turn the SERVICES flag into an ordered, de-duplicated list of known names.

    Order is preserved because services start in the order given, and a later service may
    reasonably want an earlier one already up.
    """
    catalogue = KNOWN if known is None else known
    names: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in catalogue:
            raise ConfigError(
                f"SERVICES names {name!r}, which this image does not have. "
                f"Known services: {', '.join(sorted(catalogue)) or 'none'}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ConfigError(
            f"SERVICES is empty. A services container that runs no services is a "
            f"misconfiguration, not a valid state. Known services: "
            f"{', '.join(sorted(catalogue)) or 'none'}"
        )
    return names


def build(names: list[str], known: dict | None = None) -> list[Service]:
    """The processes to run, in order. A factory may return one Service or several."""
    catalogue = KNOWN if known is None else known
    out: list[Service] = []
    for n in names:
        made = catalogue[n]()
        made = list(made) if isinstance(made, (list, tuple)) else [made]
        for svc in made:
            svc.group = svc.group or n
        out.extend(made)
    return out
