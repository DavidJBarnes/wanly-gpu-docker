# Deploying a long-lived worker

For a box we own and keep running — today the 3090. RunPod pods do not use any of this: the
API creates them from `:latest` with `runpod_client.worker_env()`, so they are current by
construction.

## Why this directory exists

There are two independent update channels:

| what | updates on |
|---|---|
| the daemon | container **restart** — `start.sh` clones `wanly-gpu-daemon` from `main` at boot |
| `start.sh`, `download_models.sh`, the engine | image **pull + recreate** |

A pod is recreated every time it launches, so it gets both. A long-lived container only ever
gets the first, because `docker restart` reuses its image by design. That is how the 3090 came
to run a 37-hour-old image and a 14-hour-old daemon while a pod ran current code, and produce
a different result from the same queue (#72).

## First install

```bash
git clone https://github.com/DavidJBarnes/wanly-gpu-docker.git ~/wanly-gpu-docker
cd ~/wanly-gpu-docker/deploy
cp worker.env.example worker.env
$EDITOR worker.env          # QUEUE_API_KEY and the host paths
./run-worker.sh
```

## Keeping it current

```bash
sudo ~/wanly-gpu-docker/deploy/install-timer.sh
```

One command, absolute paths, and it **verifies** rather than reporting success. The four-step
version it replaces was silently skippable: run the `enable` without the `cp` and systemd says
`Unit wanly-worker-update.timer does not exist`, which names the unit rather than the missing
copy and reads like the repo is wrong. And a timer can be `enabled` with `Trigger: n/a`, which
is #76 — installed, enabled, and never going to fire. The script fails on both.

`update-worker.sh` pulls `:latest`, compares its digest against the digest the running
container was created from, and recreates only if they differ **and** the engine reports
nothing in flight. It exits 0 when it decides not to act, so a non-zero exit in the journal is
a real failure.

Check on it with:

```bash
systemctl list-timers wanly-worker-update.timer
journalctl -u wanly-worker-update.service -n 50
```

## One container per GPU (#83)

`SERVICES` in `worker.env` says what the container runs. The render stack (`ltx-engine`) is
the default and is all a RunPod pod ever runs. The 3090 runs everything:

```
IMAGE=davidjbarnes/wanly-gpu-docker:full
SERVICES=ltx-engine,lora-trainer,image-description,face-crop
LORA_RUNS_DIR=/home/david/projects/loras
OLLAMA_HOST_STORE=/usr/share/ollama/.ollama
```

`lora-trainer` and `image-description` live in the `:full` tag only; `run-worker.sh` refuses
the lean tag with them enabled, before it removes anything. The box registers ONCE, as
`render` + `trainer`, and the Workers page shows one row listing every service. A training run
drains that row: the render daemon parks, the card frees, training runs, the drain is
released. The update timer reads `training` off the container's own `/health` and leaves it
alone throughout.

## Rollback

`run-worker.sh` honours `IMAGE`, so pinning an older build is:

```bash
IMAGE=davidjbarnes/wanly-gpu-docker:<sha> ./run-worker.sh
```

Re-enable the timer afterwards or it will pull `:latest` back over the pin at the next tick.

## What it will never do

Recreate while the worker holds a claim.

**Two signals, both must say idle.** The worker's own status from the API, and the engine's
health. They cover different spans and each catches the other's blind spot:

| signal | covers | blind to |
|---|---|---|
| worker status (`online-idle`) | the whole claim — set the instant work is received, before `[1/6]` | a failed status push from the daemon |
| engine `running`/`queue_depth` | the render itself, from `[3/6]` | `[1/6]`–`[2/6]`: image and LoRA/checkpoint fetch |

The engine-only version of this cost a segment on 2026-09-06: a container was recreated 50%
through a 673 MB LoRA download in `[2/6]`, where the engine truthfully reports `running: 0`.
Because registration reuses the worker row, the abandoned segment was pinned to a live, busy
worker that no reclaim rule could reach, and it sat in `PROCESSING` for seven hours.

That window is now much wider than it was: console#423 lets a worker fetch a 46 GB checkpoint
on demand, which is roughly twenty minutes inside `[2/6]`.

Anything unreadable — the API, the engine, a worker that has not registered yet — counts as
**busy**. A box mid-boot must not be interrupted either; on a cold pod that is ~58 GB of
staging thrown away.
