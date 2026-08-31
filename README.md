# wanly-gpu-docker

The Wanly GPU worker image. Published as `davidjbarnes/wanly-gpu-docker:latest`.

**This image runs LTX 2.3.** It was the WAN 2.2 worker until 2026-08-31; WAN has been retired
and is no longer deployable from here.

## Three processes

```
ComfyUI      :8188  background   the sampler
ltx-engine   :8190  background   graph assembly + recipe resolution
daemon              foreground   claims from wanly-api, drives the engine
```

ltx-engine sits between the daemon and ComfyUI deliberately. **It owns the graph** — it uploads
keyframes, resolves the recipe, patches the workflow and submits. The daemon does not build LTX
graphs. Every structural bug on that project came from rewriting a downloaded graph's topology
in a caller to cover a job shape it was not built for; none came from the reference workflows.

Both background services bind to **loopback**. Nothing outside the container should be
submitting graphs.

## The image is the environment; the daemon is the code

`start.sh` clones or `git pull --ff-only`s `wanly-gpu-daemon` at every boot rather than baking
it in, so a daemon fix does not need an image rebuild. It then installs only the deps added
since the image was built, and warns loudly when `daemon-requirements.txt` has drifted from the
daemon's own `requirements.txt`.

`ENGINE=ltx` is set here. The daemon itself defaults to `wan22` so that merging LTX support
could not retarget a running worker; this image is where it is turned on.

## Models are mounted, not baked

The LTX-2.3 set is ~126 GB. On the 3090 it is a read-only bind mount:

```
-v /home/david/LTX-2/models:/workspace/models:ro
-v /home/david/ltx-jobs:/jobs
```

`download_models.sh` verifies rather than downloads. A pod with no such mount needs a real
fetch step, which **is not written** — see the TODO in that script. Inventing URLs for
checkpoints that live nowhere fixed would be worse than failing with a clear message.

It also checks every `.safetensors` header against the file's actual byte count. A partial
safetensors is a **valid header over missing data**: it looks fine on disk, passes every
existence check, and fails only at load — deep in a render, on a segment already claimed. One
arrived at 14.40 of 27.16 GiB and reported nothing.

## What went when WAN did

PainterLongVideo, ReActor, FaceFusion, Frame-Interpolation, the `patches/` tree, the
CFG-aware VRAM patch to `comfy/model_base.py`, and `assert_onnx_cuda.py`.

Three long-standing traps went with them, all of which cost real time:

* FaceFusion's `install.py` replacing the pinned `onnxruntime-gpu` with a CUDA-13 wheel this
  image cannot load — the swap then ran on CPU, ~100x slower, with no error anywhere.
* ReActor's unpinned `requirements.txt` pulling transformers >= 5.x, which referenced a torch
  symbol the pinned torch did not have, so the node failed to register and ComfyUI 400'd every
  workflow using it.
* Generation defaults in the `.env` block drifting from the 3090's. Under LTX those values live
  in the recipe, which wanly-api resolves and ships inside the claim, so there is nothing left
  here to drift.

Faceswap is going away on the API and console side too — see wanly-api#209.

## Changing this image

Per the root `AGENTS.md`, the image is a first-class consumer of every change to generation, the
daemon or models:

| a change that adds or alters… | needs |
|---|---|
| a daemon Python dependency | `daemon-requirements.txt` |
| a ComfyUI node class in a workflow | the pack cloned in the `Dockerfile`, and the class in the daemon's node check |
| a model file | `download_models.sh` |
| ltx-engine behaviour | `engine/` here — it is vendored, not cloned |

**The failure mode is silence, not errors.** A missing model aborts the boot loudly, but a
*different default* just produces plausible output that is not comparable to anything.

## Traps carried over from the LTX work

* **Swapping checkpoints must move three loaders** — `CheckpointLoaderSimple`,
  `LTXAVTextEncoderLoader.ckpt_name`, `LTXVAudioVAELoader.ckpt_name`. 2.3 checkpoints are
  monoliths; changing only the first silently mixes models.
* **The validated recipe is load-bearing.** Patch values only, never topology.
* **Never `/interrupt` or clear the ComfyUI queue to tidy up.** One GPU, one queue; both kill
  whatever is rendering. Delete a specific pending item by prompt id.
* `GET /history/<id>` holds the submitted graph verbatim with literal seeds. It is the
  authoritative record of what actually rendered.
