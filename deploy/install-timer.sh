#!/usr/bin/env bash
# Install (or re-install) the worker update timer. Run with sudo.
#
# WHY THIS IS A SCRIPT
#     The README said: cd here, `sudo cp` two units, daemon-reload, `enable --now`. Four steps,
#     and the first two are silently skippable -- run the last one alone and systemd answers
#     "Unit wanly-worker-update.timer does not exist", which names the unit rather than the
#     missing copy and reads like the repo is wrong. That happened, and #72 stayed open an
#     extra day because of it.
#
#     Absolute paths, so it does not matter what directory you are in. That was the other way
#     to get the same error.
#
# WHY IT VERIFIES RATHER THAN REPORTS SUCCESS
#     #76 was exactly this failure one level down: `is-enabled` said `enabled` while `Trigger:`
#     said `n/a`, so the timer was installed, enabled, and would never have fired. "It says
#     enabled" is not evidence that anything is scheduled. The only evidence is a next
#     elapse, so this asks for one and fails if there is not one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR=/etc/systemd/system

if [ "$(id -u)" -ne 0 ]; then
    echo "!! run me with sudo: sudo $0"
    exit 1
fi

for unit in wanly-worker-update.service wanly-worker-update.timer; do
    [ -f "$HERE/$unit" ] || { echo "!! $HERE/$unit is missing — is this checkout current?"; exit 1; }
    install -m 644 "$HERE/$unit" "$UNIT_DIR/$unit"
    echo "installed $UNIT_DIR/$unit"
done

systemctl daemon-reload
systemctl enable --now wanly-worker-update.timer

# The verification. `is-enabled` is not it.
next=$(systemctl show wanly-worker-update.timer -p NextElapseUSecRealtime --value 2>/dev/null || true)
if [ -z "$next" ] || [ "$next" = "n/a" ]; then
    echo "!! FATAL: the timer is enabled but has no next elapse — it would never fire."
    echo "!! That is #76 again. Check OnCalendar= in wanly-worker-update.timer."
    systemctl status wanly-worker-update.timer --no-pager || true
    exit 1
fi

echo
echo "timer is scheduled. Next elapse: $next"
systemctl list-timers wanly-worker-update.timer --no-pager
echo
echo "watch it with: journalctl -u wanly-worker-update.service -f"
