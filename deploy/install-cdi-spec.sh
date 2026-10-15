#!/usr/bin/env bash
# Install (or re-install) wanly-cdi-spec.service, which writes the NVIDIA CDI spec BEFORE docker
# starts, so the worker comes back on its own after a cold boot (wanly-gpu-docker#213). Run with
# sudo, on every box that runs the worker with CDI devices (today: 3090a and 3090b).
#
# WHY THIS IS NEEDED
#     The toolkit's nvidia-cdi-refresh.service is ordered After=multi-user.target, which waits
#     for docker. So docker always restores the worker before /var/run/cdi/nvidia.yaml (tmpfs)
#     exists, the start fails with "unresolvable CDI devices", and docker never retries. The
#     unit's own comments carry the full reasoning, including why it is not a docker drop-in
#     (an ordering cycle) and not a persistent /etc/cdi spec (stale after a driver update).
#
# WHY IT VERIFIES RATHER THAN REPORTS SUCCESS
#     Same rule as install-timer.sh (#76): "enabled" is not evidence. This checks that docker
#     actually pulls the unit in and is ordered after it, runs it once, and checks the spec it
#     wrote lists the card the worker is pinned to.
#
# Running it once now rewrites /var/run/cdi/nvidia.yaml with the content it already has (the
# same command the vendor unit runs). A running container is not affected: CDI devices are
# resolved when a container is created or started, never afterwards. Docker is NOT restarted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR=/etc/systemd/system
UNIT=wanly-cdi-spec.service
# Overridable only so the tests can point it at a temp file.
SPEC="${CDI_SPEC:-/var/run/cdi/nvidia.yaml}"

if [ "$(id -u)" -ne 0 ]; then
    echo "!! run me with sudo: sudo $0"
    exit 1
fi

[ -f "$HERE/$UNIT" ] || { echo "!! $HERE/$UNIT is missing — is this checkout current?"; exit 1; }
command -v nvidia-ctk >/dev/null 2>&1 || { echo "!! nvidia-ctk is not installed — this box has no CDI to fix"; exit 1; }

install -m 644 "$HERE/$UNIT" "$UNIT_DIR/$UNIT"
echo "installed $UNIT_DIR/$UNIT"

systemctl daemon-reload
systemctl enable "$UNIT"

# The verification. Each of these is a way to be "enabled" and still lose the race at boot.
fail=0
if ! systemctl show docker.service -p Wants --value | tr ' ' '\n' | grep -qx "$UNIT"; then
    echo "!! FATAL: docker.service does not pull in $UNIT — [Install] WantedBy did not take"
    fail=1
fi
if ! systemctl show docker.service -p After --value | tr ' ' '\n' | grep -qx "$UNIT"; then
    echo "!! FATAL: docker.service is not ordered after $UNIT — it would still race"
    fail=1
fi

if ! systemctl start "$UNIT"; then
    echo "!! FATAL: $UNIT failed to run:"
    journalctl -u "$UNIT" -n 20 --no-pager || true
    fail=1
fi
if [ ! -s "$SPEC" ]; then
    echo "!! FATAL: $SPEC is missing or empty after the run"
    fail=1
fi

devices=$(nvidia-ctk cdi list 2>/dev/null || true)
echo "$devices" | grep -q "nvidia.com/gpu=all" || { echo "!! FATAL: the spec has no nvidia.com/gpu=all"; fail=1; }

# The worker's pin, when it has one. Read from worker.env without sourcing it -- it carries a
# key, and only this one value is wanted.
uuid=$(sed -n 's/^\(GPU_UUID\|GPU_DEVICE\)=\(GPU-[0-9a-fA-F-]*\).*/\2/p' "$HERE/worker.env" 2>/dev/null | head -1 || true)
if [ -n "$uuid" ]; then
    if echo "$devices" | grep -qi "nvidia.com/gpu=$uuid"; then
        echo "the spec lists the pinned card nvidia.com/gpu=$uuid"
    else
        echo "!! FATAL: worker.env pins $uuid but the spec does not list nvidia.com/gpu=$uuid"
        fail=1
    fi
fi

[ "$fail" -eq 0 ] || exit 1

echo
echo "$UNIT is installed, docker is ordered after it, and the spec is current:"
echo "$devices" | sed 's/^/    /'
echo
echo "after the next boot, check it won the race:"
echo "    journalctl -b -u $UNIT -u docker -o short-precise --no-pager | grep -E 'Generated CDI spec|Starting docker|unresolvable'"
