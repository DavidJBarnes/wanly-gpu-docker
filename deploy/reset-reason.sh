#!/usr/bin/env bash
# Log why this box last reset, then clear the record so the next reset gets its own (#193).
#
# WHY THIS EXISTS
#     3090a resets about once a day, half the time at idle, and the journal of every one of them
#     simply stops: no panic, no machine check, no Xid. Nothing the OS sees explains it. The AMD
#     chipset does keep a record -- FCH::PM::S5_RESET_STATUS, MMIO 0xFED80300 + 0xC0 -- of what
#     caused the last reset: thermal trip, CPU shutdown, watchdog, an uncorrected error (sync
#     flood), the reset pin, software. Kernel >= 6.16 prints and clears it at boot; this box
#     runs 6.14, which does neither, so it is done here.
#
#     An unclean end whose register carries NONE of those causes is the useful answer too: the
#     chip did not reset itself, so power was taken away from it.
#
# WHY A CONTAINER
#     /dev/mem needs root and this user has docker, not sudo -- the same route the worker uses.
#     The pinned worker image is already on the box and has python; the entrypoint is overridden,
#     so no worker starts, and there is no network.
#
# WHY IT CLEARS
#     The bits are write-1-to-clear and the hardware does not clear some of them itself, so
#     without this every reset's reason piles onto the last (the first read here held bits from
#     a month of resets). Writing the value back is exactly what the kernel does.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/wanly"
LOG="$LOG_DIR/reset-reasons.log"
mkdir -p "$LOG_DIR"

IMAGE=$(grep -m1 '^IMAGE=' "$HERE/worker.env" | cut -d= -f2-)
[ -n "$IMAGE" ] || { echo "no IMAGE in $HERE/worker.env" >&2; exit 1; }

# At boot the user manager can start before dockerd is answering. Wait for it, briefly.
for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done

# How the previous boot ended: its last journal line, and whether it shut down cleanly.
prev_end=$(journalctl -b -1 -n 1 -o short-iso --no-pager 2>/dev/null | cut -c1-25 || true)
if journalctl -b -1 -n 60 --no-pager 2>/dev/null \
        | grep -qE 'Journal stopped|Reached target.*(Shutdown|Reboot|Power-Off)|systemd-shutdown'; then
    prev_how=clean
else
    prev_how=UNCLEAN
fi

result=$(docker run --rm --privileged --network none --entrypoint python3 "$IMAGE" -c '
import mmap, os, struct

# arch/x86/kernel/cpu/amd.c s5_reset_reason_txt, upstream. Unlisted bits are reserved.
TXT = {
    0: "thermal pin BP_THERMTRIP_L was tripped",
    1: "power button was pressed for 4 seconds",
    2: "shutdown pin was tripped",
    4: "remote ASF power off command was received",
    9: "internal CPU thermal limit was tripped",
    16: "system reset pin BP_SYS_RST_L was tripped",
    17: "software issued PCI reset",
    18: "software wrote 0x4 to reset control register 0xCF9",
    19: "software wrote 0x6 to reset control register 0xCF9",
    20: "software wrote 0xE to reset control register 0xCF9",
    21: "ACPI power state transition occurred",
    22: "keyboard reset pin KB_RST_L was tripped",
    23: "internal CPU shutdown event occurred",
    24: "system failed to boot before failed boot timer expired",
    25: "hardware watchdog timer expired",
    26: "remote ASF reset command was received",
    27: "an uncorrected error caused a data fabric sync flood event",
    29: "FCH and MP1 failed warm reset handshake",
    30: "a parity error occurred",
    31: "a software sync flood event occurred",
}
PAGE, OFF = 0xFED80000, 0x3C0  # FCH_PM_BASE 0xFED80300 + FCH_PM_S5_RESET_STATUS 0xC0
fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
m = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=PAGE)
v = struct.unpack("<I", m[OFF:OFF + 4])[0]
if v == 0xFFFFFFFF:  # an error response, not a value (the kernel ignores it too)
    print("0xffffffff unreadable")
    raise SystemExit
if v:
    m[OFF:OFF + 4] = struct.pack("<I", v)  # write-1-to-clear
why = [TXT.get(i, "reserved bit %d" % i) for i in range(32) if v >> i & 1]
print("0x%08x %s" % (v, "; ".join(why) or "no reason recorded"))
')

line="$(date -Is) previous boot ended ${prev_end:-?} ($prev_how) -- reset status $result"
echo "$line" >> "$LOG"
logger -t wanly-reset-reason "$line"
echo "$line"
