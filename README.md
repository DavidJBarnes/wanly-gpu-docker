# wanly-gpu-docker

The GPU worker image for Wanly: ComfyUI, the custom nodes, the pinned CUDA stack, and a boot
script that stages models and hands off to [`wanly-gpu-daemon`](https://github.com/DavidJBarnes/wanly-gpu-daemon).

Published as **`davidjbarnes/wanly-gpu-docker:latest`**.

Also still published as `davidjbarnes/wanly22-runpod:latest` — the legacy name. Docker Hub cannot
rename a repository, so the transition is "publish both, migrate consumers, then drop the old one"
rather than a switch. Anything still pointing at the legacy name keeps working until then; see #36.

Originally RunPod-specific, hence the old name. It is now just *the* worker image, and is intended
to replace the hand-maintained ComfyUI install on the 3090 as well.

---

## What's in it

| | |
|---|---|
| base | `nvidia/cuda:12.4.1-devel-ubuntu22.04`, torch 2.6.0+cu124, Python 3.11 |
| ComfyUI | pinned commit `4a93a62371b6` |
| custom nodes | Frame-Interpolation (RIFE), VideoHelperSuite, ReActor, PainterLongVideo, FaceFusion — **all pinned to commits** |
| baked models | 6 FaceFusion models (~872MB): `inswapper_128`, `arcface_w600k_r50`, `retinaface_10g`, `scrfd_2.5g`, `bisenet_resnet_34`, `xseg_1` |
| staged at boot | ~39GB: Wan 2.1 VAE, umt5-xxl fp8 text encoder, Wan 2.2 i2v high/low experts (fp8 scaled), lightx2v LoRAs |

Weights live on `/workspace`; only the FaceFusion models are baked into the image, because the
node self-downloads them into its own directory (which is in the image, not on the volume) and
would otherwise re-fetch them on every container start.

## Running it

The daemon is the container's main process. It claims segments from the queue and runs them, so
**starting this image on a machine points a live worker at the real job queue.**

```bash
docker run -d --gpus all \
  -v /path/to/models:/workspace \
  -e QUEUE_URL=http://api.wanly22.com:8001 \
  -e QUEUE_API_KEY=...        # from the 3090's wanly-gpu-daemon/.env
  -e FRIENDLY_NAME=my-worker \
  -e PUBLIC_KEY="ssh-ed25519 ..."   # optional; enables sshd on 22
  davidjbarnes/wanly-gpu-docker:latest
```

To poke at the image **without** claiming jobs, override the entrypoint — do not just let it boot:

```bash
docker run --rm -it --gpus all --entrypoint bash davidjbarnes/wanly-gpu-docker:latest
```

Note `--gpus all` is required even for CPU-only inspection: ComfyUI imports `comfy_kitchen` →
triton at module load, and triton raises `0 active drivers` with no GPU visible.

### Environment

| var | default | notes |
|---|---|---|
| `QUEUE_URL` | `http://api.wanly22.com:8001` | |
| `QUEUE_API_KEY` | — | required, or the daemon can't register |
| `FRIENDLY_NAME` | `runpod-$RUNPOD_POD_ID` | shown on the console Workers page |
| `PUBLIC_KEY` | — | if set, sshd starts on port 22 |
| `UNET_HIGH_MODEL` / `UNET_LOW_MODEL` | Wan 2.2 i2v fp8 scaled | |
| `LIGHTX2V_STRENGTH_HIGH` / `_LOW` | `0.0` | usually overridden per job from the UI |
| `CFG_HIGH` / `CFG_LOW` | `1.0` | |
| `STEPS_TOTAL` / `HIGH_NOISE_STEPS` | `10` / `5` | |
| `PAINTER_MOTION_AMPLITUDE` | `1.7` | matched to the 3090 |

`start.sh` writes these into the daemon's `.env` and prints the resolved config (secrets
redacted) so a parity problem is visible in the first lines of the boot log rather than after a
day of results that quietly are not comparable to anything.

## RunPod

Verified 2026-08-07 on a 4090: 3 real jobs, identity loss +0.003 / -0.001 / +0.016 per segment,
which is parity with the 3090.

- Container disk 30GB, volume ≥60GB (models are ~39GB).
- **Network volumes work**, but they are region-locked: pin the pod's `dataCenterIds` to the
  volume's datacenter. That in turn forces Secure Cloud — Community has no 4090 inventory in any
  storage-capable datacenter. Secure is ~$0.74/hr vs ~$0.34/hr Community.
- Without a volume, every pod re-downloads ~39GB (~12 min).
- SSH via direct TCP with `PUBLIC_KEY`. RunPod's `ssh.runpod.io` proxy is interactive-only and
  ignores command arguments, so it can't be scripted.

## The dependency traps

Everything here is pinned for a reason, and most of the reasons are recorded as comments at the
relevant line. Two are worth stating up front because **they fail silently**:

**1. FaceFusion's `install.py` will replace the pinned onnxruntime.** ComfyUI runs every custom
node's `install.py` at startup. FaceFusion's skips only if `pip show onnxruntime-gpu` succeeds
*and* a `.install_complete` marker exists — and that marker is only ever written at runtime, so a
freshly built image always fails the check, uninstalls the pinned `onnxruntime-gpu==1.20.2`, and
installs the newest (a CUDA-13 wheel). The Dockerfile now writes the marker at build time.

**2. onnxruntime needs `LD_LIBRARY_PATH`.** The cuDNN libraries ship inside
`site-packages/nvidia/cudnn/lib`. torch preloads its own copies at import, so the image looks
healthy until something that *isn't* torch asks for CUDA. The Dockerfile now sets the path.

Either failure produces the same symptom: onnxruntime reports the CUDA provider unavailable,
falls back to CPU, and the faceswap runs ~100x slower — surfacing as the daemon's
"no progress from ComfyUI for 300s" watchdog. A hang, four steps from the cause.

`assert_onnx_cuda.py` runs in every build and fails it if any CUDA provider dependency cannot be
resolved. It uses `ldd` rather than loading the provider, because the CI builder has no GPU.

**Node registration, `pip list` and version strings all pass while the swap runs on CPU.** The
only check that catches it is asking onnxruntime which provider it actually chose, on a real GPU:

```bash
docker run --rm --gpus all --entrypoint python3 davidjbarnes/wanly-gpu-docker:latest -c "
import onnxruntime as ort
s = ort.InferenceSession('/app/ComfyUI/custom_nodes/Facefusion_comfyui/models/inswapper_128.onnx',
                         providers=['CUDAExecutionProvider','CPUExecutionProvider'])
print(s.get_providers())"
```

Expect `['CUDAExecutionProvider', 'CPUExecutionProvider']`. A bare `['CPUExecutionProvider']`
means the swap will run in software.

## Building

Pushing to `main` builds and pushes `:latest` and `:<sha>` under both the current and legacy names via GitHub Actions. Builds take ~20
minutes. **Test a dependency change before pushing** — a syntax check will not catch a pip
resolution failure, and the image is what production runs.

## Related

- [`wanly-gpu-daemon`](https://github.com/DavidJBarnes/wanly-gpu-daemon) — cloned fresh at boot, so daemon changes do not need an image rebuild
- [`wanly-api`](https://github.com/DavidJBarnes/wanly-api) — the queue this worker claims from
- [`wanly-console`](https://github.com/DavidJBarnes/wanly-console) — UI
