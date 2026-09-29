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

## face-edit (console#547)

LivePortrait face edits -- expression, gaze, small head turns -- for the console's Edit dialog.
`:full` only; models are baked (no mount). wanly-api calls it at `face_edit_url`.

It claims no work, so it runs in **every** mode, and it borrows the GPU per edit only when no
neighbour needs it (`wanly_worker/services/face_edit/gpu.py`): a render in flight or an A1111
generation sends the edit to CPU (~8-10 s measured on a 6-P-core laptop CPU, 1248x1824 frame)
instead of taking VRAM a job with no fallback needs. On a free card an edit is ~1 s and the
weights go back to CPU after `FACE_EDIT_GPU_IDLE_S` (45 s) or as soon as a neighbour starts.

**The 2070** -- beside Automatic1111, where it is meant to live:

```
IMAGE=davidjbarnes/wanly-gpu-docker:full
FRIENDLY_NAME=2070.zero
QUEUE_URL=http://api.wanly22.com:8001
QUEUE_API_KEY=...
SERVICES=face-edit
FACE_EDIT_A1111_URL=http://host.docker.internal:7860
PRUNE_IMAGES=0
```

**`PRUNE_IMAGES=0` is not optional there.** `run-worker.sh` ends with `docker image prune -af`,
which removes every image no container uses -- on the 2070 that is verbatim-worker, open-webui,
ollama, postgres and ntfy, none of which has a container.

No `JOBS_DIR`/`MODELS_DIR`: those are required only with `ltx-engine` or `lora-trainer`, and a
box without them publishes no ComfyUI or engine port. `run-worker.sh` adds the
`host.docker.internal` alias whenever face-edit is enabled, so the container can see A1111 on
the host: it never uses the GPU while A1111 is generating, and asks an *idle* A1111 to unload
its checkpoint only when the card is otherwise too full (the captioner's `_yield_the_gpu` rule).

**The 3090** -- add `face-edit` to its `SERVICES` line. In render mode the card is ~23 of 24 GB,
so edits there run on CPU while a segment is in flight; in caption mode it depends on the
captioner's footprint (qwen3-vl:32b leaves ~2 GB, under the 2.5 GB bar -> CPU).

`curl -s :8085/health` reports `model_loaded`, the device, why the last edit ran where it did,
and `vram_peak_mib` -- torch's measured peak for the last GPU edit.

## image-edit and edit mode (console#548)

Qwen-Image-Edit "full mode" for the console's Edit dialog: head angles beyond LivePortrait's
±20° (three-quarter, full profile, look up/down) and free-text instructions. It needs ~20 GB of
the card and has no CPU fallback, so it runs **only in edit mode** (`registry.MODE_ONLY`): the
supervisor builds it at boot and holds it stopped; render and caption modes never run it.

```
POST :8081/mode {"mode": "edit"}    # the segment in flight finishes, then the render stack,
                                    # trainer and captioner stop and image-edit starts
POST :8081/mode {"mode": "render"}  # back; queued renders resume
```

wanly-api drives this itself (`app/full_edit.py`): it asks for edit mode when a full-mode job
arrives, waits (the console shows "3090 is rendering; edit queued"), runs the queue, and asks for
the previous mode back 90 s after the last edit. As a backstop the box returns by itself after
`EDIT_IDLE_RETURN_S` (600 s) with no edit -- an API restarted mid-queue must not leave every
render parked behind an idle model. A switch that fails to start image-edit puts the previous
mode back rather than leaving the box with nothing running. A training run is never interrupted:
wanly-api waits while the trainer reports one.

**The 3090** -- add `image-edit` to `SERVICES` (and to `SERVICES_RENDER` if the old
`wanly-mode.sh` line is still in worker.env), and mount the Qwen tree:

```
SERVICES=ltx-engine,lora-trainer,image-description,face-crop,image-edit
IMAGE_EDIT_MODELS_HOST_DIR=/home/david/models/qwen   # read-only, holds v23/Qwen-Rapid-AIO-NSFW-v23.safetensors
IMAGE_EDIT_PORT=8086
INSIGHTFACE_HOST_DIR=/home/david/.insightface       # AuraFace glintr100.onnx for the identity score
```

`download_models.sh --image-edit` (run by the preflight on the first switch) checks the
checkpoint's safetensors header against its size and fetches AuraFace into the insightface store
if it is missing. `curl -s :8086/health` reports `idle_s`, the last edit's time and VRAM peak.

Measured in the #548 spike: ~13 s per edit warm, ~23 s with the checkpoint load, ComfyUI's peak
23.1-23.7 GB with the card to itself.

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
