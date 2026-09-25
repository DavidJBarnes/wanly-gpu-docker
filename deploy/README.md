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
released. The idle gate reads `training` off the container's own `/health` and leaves it
alone throughout.

## Render or caption, one command (#131)

The 3090 runs every service in one container, which is right — one box, one row, one GPU. But
with a job queued the box is continuously `online-busy`, and the captioner's busy guard
(`wanly-api` `app/joycaption.py`) then refuses captions rather than fighting the render for
VRAM. So after starting a job you could not batch-caption the rest of the repo until the queue
drained. The one-image-many-modes design already supported the answer; it had no lever.

```bash
~/wanly-gpu-docker/deploy/wanly-mode.sh status
~/wanly-gpu-docker/deploy/wanly-mode.sh pause      # stop claiming, finish, park  <- usually this
~/wanly-gpu-docker/deploy/wanly-mode.sh resume
~/wanly-gpu-docker/deploy/wanly-mode.sh caption    # image-description,face-crop, via a recreate
~/wanly-gpu-docker/deploy/wanly-mode.sh render     # the full line back
```

In `caption` mode **no render daemon exists**, so queued jobs simply wait — nothing is lost,
nothing is claimed, and the card is ollama's alone. `render` restores `SERVICES_RENDER`, which
the script records from whatever this box was running the first time it left render mode: the
full set differs per box, and a restore that guessed would silently drop a service.

A switch means `run-worker.sh`, which recreates with `docker rm -f`, so **it refuses while the
worker is busy** — the same three-signal gate the update timer uses, below. `--force` says you
mean it anyway, and destroys the in-flight segment or training run.

### Pause / resume — usually the better answer

**This is the one for "let me caption while jobs are queued".** No recreate, no boot, no model
re-stage, and the segment in flight **finishes** instead of being destroyed:

```bash
wanly-mode.sh pause     # finish the current segment, park, free the card
# ...queue as many jobs as you like, caption as much as you like...
wanly-mode.sh resume    # everything queued starts processing
```

While paused the worker claims nothing, so jobs pile up in the queue untouched and the card is
the captioner's. `resume` releases the drain and the backlog goes.

The daemon does the parking itself — on a live drain it finishes the segment, calls ComfyUI
`/free` and stops claiming (`wanly-gpu-daemon` `main.py:283`, #182) — so there is no separate
`/free` to remember.

**`pause` waits, and the wait is the point.** `POST /drain` sets the worker to `draining`
*immediately*, and `wanly-api` refuses an interactive caption only on `online-busy`
(`app/joycaption.py` `busy_render_beside_the_captioner`). So the instant you drain, the
captioner is un-gated — while the render is still going, on a card that sits at ~23 of 24 GB
mid-render. Captioning in that window is exactly the VRAM fight the guard exists to prevent.
`pause` blocks until the engine reports nothing in flight, and **fails loudly rather than
claiming "parked"** if it cannot read the engine.

`caption` mode is for a long captioning stretch where you would rather not have the render
stack resident at all; `pause` is for everything else.

## Rollback

`run-worker.sh` honours `IMAGE`, so pinning an older build is:

```bash
IMAGE=davidjbarnes/wanly-gpu-docker:<sha> ./run-worker.sh
```

Re-enable the timer afterwards or it will pull `:latest` back over the pin at the next tick.

## What it will never do

Recreate while the worker holds a claim.

**Three signals, all must say idle**, and they live in `worker-idle.sh` — one definition,
shared by `update-worker.sh` and `wanly-mode.sh`. Both recreate with `docker rm -f`, so both
destroy an in-flight segment if they get this wrong, and a second hand-maintained copy of
these rules is how one of them quietly stops knowing about the third. Each covers the others'
blind spot:

| signal | covers | blind to |
|---|---|---|
| worker status (`online-idle`, or `draining` — a paused box claims nothing) | the whole claim — set the instant work is received, before `[1/6]` | a failed status push from the daemon; and `draining` says nothing about the segment already in flight, which is row 2's job |
| engine `running`/`queue_depth` | the render itself, from `[3/6]` | `[1/6]`–`[2/6]`: image and LoRA/checkpoint fetch |
| trainer `training` | a training run, which *drains* the render worker and therefore looks idle | nothing else — it is the drain's blind spot, not its own |

The engine-only version of this cost a segment on 2026-09-06: a container was recreated 50%
through a 673 MB LoRA download in `[2/6]`, where the engine truthfully reports `running: 0`.
Because registration reuses the worker row, the abandoned segment was pinned to a live, busy
worker that no reclaim rule could reach, and it sat in `PROCESSING` for seven hours.

That window is now much wider than it was: console#423 lets a worker fetch a 46 GB checkpoint
on demand, which is roughly twenty minutes inside `[2/6]`.

Anything unreadable — the API, the engine, a worker that has not registered yet — counts as
**busy**. A box mid-boot must not be interrupted either; on a cold pod that is ~58 GB of
staging thrown away.

The one exception, and it is about which question is being asked rather than about risk: a
service that is **not enabled on this container** has no health to read, and "is the engine
rendering?" has no meaning on a box in caption mode. The gate checks the container's own
`SERVICES` before treating an unreadable engine or trainer as busy — otherwise `wanly-mode.sh
render` could never switch back, refusing forever on the absence of the very service it is
there to restore. The worker-status signal still covers a claim, and a container with no
`ltx-engine` holds none.
