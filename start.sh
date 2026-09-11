#!/bin/bash
# Boot: a GPU gate, an optional sshd, then the supervisor (wanly-gpu-docker#83).
#
# Everything that used to be phases 1-6 here -- models, daemon code, ComfyUI, the engine, the
# daemon -- is now a SERVICE under wanly_worker/, started in order by the supervisor with a
# readiness probe each and a watchdog that stops the container when one dies. What stays in
# this script is what must happen before any Python: the log tee, the "is there a GPU at all"
# gate, and sshd for a pod. The last line execs the supervisor so it is PID 1.
set -e

mkdir -p /workspace/logs
# Timestamp every line. Without it there is no way to tell a boot that is slow from one that is
# wedged: both look like a log that has stopped moving.
#
# A bash read loop rather than awk, because `awk` here is MAWK. mawk reads its input in blocks
# and only processes a block once it is full or the pipe closes -- and `fflush()` flushes its
# OUTPUT, which is not the side that is holding anything. During a 45-minute silent model
# download the boot produces a few hundred bytes, so nothing ever reached the log at all: on
# two real pods daemon.log sat at 0 bytes and the RunPod console showed nothing after the
# NVIDIA banner while the boot ran normally the whole time. Exactly the wedge/slow ambiguity
# this timestamping exists to remove, caused by the timestamping.
#
# It read as correct because a development box runs GAWK, which does not buffer this way, and
# because a test that lets the script EXIT flushes on close and passes. Reproduced in
# ubuntu:22.04 with the script still running: mawk 0 bytes, this loop 120 bytes.
#
# `read` on a pipe returns per line, so there is no buffering stage left to get this wrong.
exec > >(while IFS= read -r line; do
             printf '%s %s\n' "$(date +%H:%M:%S)" "$line"
         done | tee -a /workspace/logs/daemon.log) 2>&1

echo "=== Wanly GPU Worker (LTX 2.3) ==="
echo "image build: ${GIT_SHA:-unknown}"
echo "(this log is also at /workspace/logs/daemon.log)"
echo "SERVICES=${SERVICES:-ltx-engine}"

# ---------- 0a. Is there a GPU at all? ----------
if ! GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) \
   || [ -z "$GPU_LINE" ]; then
    if [ "${ALLOW_NO_GPU:-0}" = "1" ]; then
        echo "WARNING: no GPU visible — continuing because ALLOW_NO_GPU=1."
        echo "WARNING: this container CANNOT render. Do not expect it to claim work."
    else
        echo "!! FATAL: no GPU visible. nvidia-smi failed or reported nothing."
        echo "!! On the 3090 this means the container was started without --gpus all;"
        echo "!! use deploy/run-worker.sh rather than a hand-written docker run."
        echo "!! On a pod it means the host is broken — kill it and take another."
        echo "!! Set ALLOW_NO_GPU=1 only for a debug shell, never for a worker."
        exit 1
    fi
else
    echo "$GPU_LINE"
fi

# ---------- 0b. sshd, for a pod ----------
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    mkdir -p /run/sshd
    ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd 2>/dev/null && echo "sshd started (direct TCP on port 22)" || echo "WARN: sshd failed to start"
fi

# ---------- The supervisor ----------
# Models, daemon code, ComfyUI, the engine and the daemon are its services; see
# wanly_worker/services/ltx_engine.py. It answers /health on CONTROL_PORT.
mkdir -p /run/wanly
cd /app
exec python3 -m uvicorn wanly_worker.control:app --host 0.0.0.0 --port "${CONTROL_PORT:-8081}"
