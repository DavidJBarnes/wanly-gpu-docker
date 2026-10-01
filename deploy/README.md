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
IMAGE_EDIT_MODELS_HOST_DIR=/home/david/models/qwen   # read-only: base/, text_encoders/, vae/ (below)
IMAGE_EDIT_PORT=8086
INSIGHTFACE_HOST_DIR=/home/david/.insightface       # AuraFace glintr100.onnx for the identity score
```

**The model is the official Qwen-Image-Edit-2511** (wanly-console#574, #157), as Comfy-Org
ships it: three files under the Qwen tree, the set the character-sheet recipe was proven with.

| folder | file | HF repo (`split_files/...`) | size |
|---|---|---|---|
| `base/` | `qwen_image_edit_2511_fp8mixed.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI` `diffusion_models/` | 20.5 GB |
| `text_encoders/` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` `text_encoders/` | 9.4 GB |
| `vae/` | `qwen_image_vae.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` `vae/` | 0.25 GB |

The tree is a read-only mount, so **stage them on the host before re-pinning**, or the
preflight refuses the switch ("bind mount from the host, and it is incomplete"). The old
Rapid-AIO `v23/` folder is no longer read; leave it or delete it. The `base/` folder may also
hold `qwen_image_edit_2511_fp8_e4m3fn.safetensors` -- a third-party all-fp8 cast, not used.

`download_models.sh --image-edit` (run by the preflight on the first switch) checks the
three files' safetensors headers against their sizes and fetches AuraFace into the insightface
store if it is missing. `curl -s :8086/health` reports `idle_s`, `model`, `settings`, the last
edit's time and VRAM peak.

**Settings changed with the model.** v23 had the Lightning accelerators baked in: 4 steps,
cfg 1, euler_ancestral/beta, ~13 s per edit warm (#548 spike). The official model runs the
template: **40 steps, CFG 4, euler/simple, AuraFlow shift 3.1, CFGNorm 1** -- ten times the
steps at two passes each, so expect an edit in minutes, not seconds. `IMAGE_EDIT_STEPS` /
`IMAGE_EDIT_CFG` in worker.env override them; **an old `IMAGE_EDIT_STEPS=4` line must go**, or
the official model runs at 4 steps and comes out as noise.

### Character sheets (console#582)

`POST :8086/turnaround` is the sheet recipe (loras/reftest-2026-09-30/sheets.py): a real face
photo plus outfit, hair and body words in; one seed's 1088x1024 front/side/back turnaround out,
**and** the 1536x1024 sheet already composed from it -- the face-detected 448 px real-face panel
beside the turnaround (`services/image_edit/sheet.py`). wanly-api asks once per seed, on the
same queue and the same edit mode as the Edit dialog's edits.

### Faces and expressions (console#569)

The Edit dialog sends every edit here now -- head angles, the expression presets (by name;
the words are `graph.EXPRESSIONS`, beside the angle words) and free text. LivePortrait is off
the dialog. `POST :8086/faces` lists the faces in an image, and `/edit` with a `face_box`
edits that face alone: crop around it, edit the crop, paste it back feathered, so nobody else
in the picture is regenerated (`services/image_edit/crop.py` says why). **A box running an
older image ignores `expression` and `face_box`**; wanly-api reads `features` off
`/health` and refuses rather than edit the whole frame, so re-pin the main 3090 for them.

## A standing image-edit box: the second 3090 (console#570)

With a second 24 GB card, Qwen runs **full-time** there and edits stop pausing renders on the
main 3090. `SERVICES=image-edit` with nothing that renders needs no mode: `select_mode` runs a
box's mode-bound services when it has nothing else. wanly-api prefers this box while its
`/health` is ok (`image_edit_standing_url`) and falls back to the main 3090's edit mode when it
is not.

It shares its card with the host's Automatic1111 (`services/image_edit/share.py`):

1. **A1111 is never interrupted.** An edit waits while A1111 is generating; wanly-api shows
   "second 3090 busy (A1111 generating)" and sends the edit only once it is not. An edit that
   still lands mid-generation waits `IMAGE_EDIT_A1111_WAIT_S` (300 s) and is then refused, not
   forced.
2. **Room is made only when needed:** with Qwen not resident and under
   `IMAGE_EDIT_MIN_FREE_MIB` (21.5 GB) free, an *idle* A1111 is asked to unload its checkpoint
   (face-edit's `yield_a1111`, reused). It reloads from RAM on its next generation.
3. **A1111 gets the card back as soon as it wants it:** with no edit running, Qwen is unloaded
   (ComfyUI `/free`) the moment A1111 starts generating, or after `IMAGE_EDIT_UNLOAD_IDLE_S`
   idle. An edit in progress is never cut short.

The one overlap it cannot prevent: generate-forever re-clicks Generate on its own timer, so
its next image can start *during* an edit and run in what Qwen leaves (slow, or an OOM for that
one image). For a run of edits, pause generate-forever -- or raise a1111-tweaks' "generate
forever" delay, which is the gap an edit starts in.

**Deploy, after the card swap** (not done yet -- the hardware is not in):

1. Driver/CUDA/CDI check on the new card: `nvidia-smi` on the host, then
   `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
2. Copy the official 2511 set from the main 3090 (~30 GB; the three files in the table above):
   `for d in base text_encoders vae; do rsync -a --info=progress2
   3090.zero:/home/david/models/qwen/$d/ /home/david/models/qwen/$d/; done`. The preflight
   checks their safetensors headers against their sizes, so a truncated copy fails loudly.
3. AuraFace: `download_models.sh --image-edit` fetches `glintr100.onnx` into the insightface
   store on first boot if missing, or copy `~/.insightface/models/auraface/` from the 3090.
   buffalo_l (face detection for `/faces`) is fetched on first use into the same store.
4. `deploy/worker.env`: the standing block in `worker.env.example` -- `SERVICES=image-edit`,
   `IMAGE_EDIT_MODELS_HOST_DIR`, `IMAGE_EDIT_A1111_URL`, `IMAGE_EDIT_UNLOAD_IDLE_S`, and
   **`PRUNE_IMAGES=0`** (the prune would delete ~40 GB of other projects' images). Drop
   `face-edit` from `SERVICES`: nothing in the console calls LivePortrait any more.
5. `PRUNE_IMAGES=0 ./deploy/run-worker.sh`, then `curl -s :8086/health` (ok,
   `shared_with_a1111: true`, `features` includes `face_box`), then one real edit.
6. wanly-api needs nothing if the box keeps the name `2070.zero`: `image_edit_standing_url`
   defaults to `http://2070.zero:8086` and is simply skipped while nothing healthy answers
   there. If the box is renamed, set `IMAGE_EDIT_STANDING_URL` in the API's env.

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

## Identity references: stage the two LoRAs BEFORE re-pinning (#156)

Images from #156 onward stage `Best_FaceID_v1.0_LoRA.safetensors` (2.47 GB) and
`Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors` (1.31 GB) as part of the required model set.
On a pod that is a download. **On the 3090 the model root is a read-only bind mount**, so
`download_models.sh` will not fetch them -- it refuses to boot with "is a bind mount from the
host, and it is incomplete". Put them in the host tree first, then re-pin:

```bash
cd /home/david/LTX-2/models/loras
for f in Best_FaceID_v1.0_LoRA.safetensors Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors; do
  curl -fL --retry 3 -o "$f.part" "https://huggingface.co/Alissonerdx/LTX-Best-Face-ID/resolve/main/$f" \
    && mv "$f.part" "$f"
done
sha256sum Best_FaceID_*.safetensors
# 7aaab2f1bff2af121e0751120ad16a3e443b4223a04b78c51740029d25f17994  Best_FaceID_v1.0_LoRA.safetensors
# 4c7804265c5e8a284c0613fb6fd63d114f029429e9837e3ea289521d7ef93ffa  Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors
```

The image also carries ComfyUI-BFSNodes pinned at bd23236 (the node the reference runs
through), and ltx-engine's `/health` lists `identity_ref` under `features`. The daemon refuses a
reference rather than sending it to an engine without that feature, so a worker still on an
older image fails sheet renders with a message saying to re-pin -- it never renders them
without the sheet.

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
