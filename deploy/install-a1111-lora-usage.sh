#!/usr/bin/env bash
# Install (or re-install) the A1111 LoRA-usage reporter timer (#211). Run with sudo, on the
# box that runs A1111 (3090b). Same shape as install-timer.sh: absolute paths, and it verifies
# a next elapse rather than trusting `is-enabled` (#76).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR=/etc/systemd/system
NAME=wanly-a1111-lora-usage

if [ "$(id -u)" -ne 0 ]; then
    echo "!! run me with sudo: sudo $0"
    exit 1
fi

for unit in $NAME.service $NAME.timer; do
    [ -f "$HERE/$unit" ] || { echo "!! $HERE/$unit is missing — is this checkout current?"; exit 1; }
    install -m 644 "$HERE/$unit" "$UNIT_DIR/$unit"
    echo "installed $UNIT_DIR/$unit"
done

systemctl daemon-reload
systemctl enable --now $NAME.timer

next=$(systemctl show $NAME.timer -p NextElapseUSecRealtime --value 2>/dev/null || true)
if [ -z "$next" ] || [ "$next" = "n/a" ]; then
    echo "!! FATAL: the timer is enabled but has no next elapse — it would never fire."
    systemctl status $NAME.timer --no-pager || true
    exit 1
fi

echo
echo "timer is scheduled. Next elapse: $next"
systemctl list-timers $NAME.timer --no-pager
echo
echo "first run now (walks every saved image once): sudo systemctl start $NAME.service"
echo "watch it with: journalctl -u $NAME.service -f"
