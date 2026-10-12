"""What this box is doing RIGHT NOW, while the supervisor holds its worker row (wanly-gpu-daemon#189).

A busy box must never look offline. The render daemon reports its own phase (syncing LoRAs,
with file / MB / rate / ETA) once it owns the row; this is the same vocabulary for the stretch
the SUPERVISOR owns it -- booting, and switching modes -- so the Workers page can say
"Switching edit -> render: starting comfyui" instead of a grey "Offline":

    {"phase": "switching_mode", "detail": "Switching edit → render: starting comfyui",
     "from_mode": "edit", "to_mode": "render", "service": "comfyui", "since": "..."}

Phases: booting, switching_mode, ready (plus the daemon's syncing_loras / fetching_lora).
`detail` is always a whole sentence, so a reader that does not know a phase still has
something true to show.

Listeners are woken on every change, so the registrar beats at once instead of up to 30 s
later -- a switch that takes 15 seconds would otherwise never be seen at all.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

READY = "ready"

_STATE: dict = {"phase": "booting", "detail": "Booting", "since": None}
_listeners: list[Callable[[], None]] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _notify() -> None:
    for fn in list(_listeners):
        try:
            fn()
        except Exception:                       # noqa: BLE001 -- a listener never breaks a switch
            pass


def on_change(fn: Callable[[], None]) -> None:
    if fn not in _listeners:
        _listeners.append(fn)


def set_phase(phase: str, detail: str, **fields) -> None:
    global _STATE
    _STATE = {"phase": phase, "detail": detail, "since": _now(), **fields}
    _notify()


def booting(detail: str = "Booting") -> None:
    set_phase("booting", detail)


def switching(from_mode: str, to_mode: str, step: str = "") -> None:
    set_phase("switching_mode", _switch_detail(from_mode, to_mode, step),
              from_mode=from_mode, to_mode=to_mode)


def _switch_detail(from_mode: str, to_mode: str, step: str) -> str:
    return f"Switching {from_mode} → {to_mode}" + (f": {step}" if step else "")


def step(what: str, service: str | None = None) -> None:
    """A step inside the current phase -- 'starting comfyui' -- keeping its `since`."""
    if _STATE.get("phase") == "switching_mode":
        _STATE["detail"] = _switch_detail(_STATE.get("from_mode", "?"),
                                          _STATE.get("to_mode", "?"), what)
    elif _STATE.get("phase") == "booting":
        _STATE["detail"] = f"Booting: {what}"
    else:
        return
    if service is not None:
        _STATE["service"] = service
    _notify()


def ready() -> None:
    set_phase(READY, "Ready")


def current() -> str:
    return _STATE.get("phase") or READY


def snapshot() -> dict:
    return dict(_STATE)
