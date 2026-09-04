#!/usr/bin/env python3
"""ltx-job-runner — LTX-2.5 keyframe video generation behind an HTTP job API.

Implements DavidJBarnes/wanly-console#355. storyboard-ui POSTs a storyboard's
keyframes, gets a job_id back immediately, and polls until a video URL appears.

Shape of the thing:

  POST /job          -> {"job_id": ...}          (returns at once; work is queued)
  GET  /job/{id}     -> {"status": ..., "video": url|null, ...}
  GET  /job/{id}/video -> the mp4 bytes
  GET  /health

Two facts drive the design.

**One GPU, one job.** LTX-2.5 22B wants essentially the whole card, so jobs run
strictly one at a time through a single worker thread. Concurrency here would not
be faster, it would be an OOM.

**The GPU is already occupied.** keyframe-server's ComfyUI holds the Qwen
checkpoint resident after any edit -- measured 20.4 GB, leaving 888 MB free --
and never releases it on its own. Every job therefore calls keyframe-server's
POST /free first and waits for the card to actually come back. Without that step
LTX does not fail gracefully, it dies mid-load.
"""
import argparse
import base64
import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
import random
from pathlib import Path

import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import comfy
import purge as purge_mod
import recipe as recipe_mod
import ltx_grid

# Where the models live. In the container this is the bind mount; on a host it
# is the LTX-2 checkout. The engine no longer needs the LTX source at all —
# rendering is ComfyUI's job — so this is only ever a model root now.
MODELS_ROOT = Path(os.environ.get("MODELS_DIR",
                                 os.environ.get("LTX_HOME", "/home/david/LTX-2") + "/models"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/home/david/ltx-jobs"))
KEYFRAME_URL = os.environ.get("KEYFRAME_URL", "http://127.0.0.1:8189")
# ComfyUI, which renders. Was a subprocess against LTX's CLI until that
# interface turned out to expose no sampler, scheduler or sigma control.
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8191")
WORKFLOW = Path(os.environ.get(
    "LTX_WORKFLOW",
    Path(__file__).parent / "workflows" / "ltx23_base.api.json"))
# The transformer LoRA fusion is checked against. Read from the workflow's
# own checkpoint rather than guessed, so the check follows the graph.
TRANSFORMER = Path(os.environ.get(
    "LTX_TRANSFORMER",
    MODELS_ROOT / "ltx-2.3/diffusion_models/ltx-2.3-22b-dev.safetensors"))
MODELS = MODELS_ROOT / "ltx-2.5"
# Free at least this much VRAM before starting, or refuse rather than OOM deep
# in a model load twenty minutes later.
# Sanity floor, not a fit requirement.
#
# 18.0 was a CLI-era number: that path loaded the whole transformer, so it
# genuinely needed most of the card. ComfyUI stages dynamically instead
# ("Model LTXAV prepared for dynamic VRAM loading. 40050MB Staged") and renders
# happily with far less free, having peaked at 22.6 GB of 24.5 alongside other
# resident containers.
#
# It also counted the wrong things. ltx-engine's own ComfyUI keeps models
# resident between jobs -- 4.8 GB measured -- which is exactly what we want, and
# the wanly container holds 2.3 GB permanently. Both were being subtracted from
# a budget they do not actually take from us. The floor now only catches the
# case where something has genuinely run away with the card.
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "8.0"))
# Rendering is a ComfyUI graph; see comfy.py and workflows/.
#
#   distilled — ltx-2.5-22b-distilled-transformer. One model, no LoRA, no
#               negative prompt, no step count. Fast and what run_richmond.sh
#               used.
#   hq        — the ltx-2.5-22b-DEV transformer with the distilled LoRA applied
#               at stage 2. Takes a negative prompt and an explicit step count,
#               which is the whole reason to reach for it: the distilled model
#               gives you no lever against identity drift, and --negative-prompt
#               is where "identity change, face distortion, different person"
#               goes.
#
# `--distilled-lora` is REQUIRED by the hq parser (args.py), so hq without a
# LoRA is not a configuration that exists.
#   ltx23     — the 2.3 distilled monolith through distilled.py.
#   ltx23-hq  — the 2.3 DEV monolith through ti2vid_two_stages_hq with 2.3's own
#               distilled LoRA at stage 2. This is the one the content LoRAs on
#               this box are actually for: they record ss_sd_model_name
#               "ltx-2.3-22b-dev.safetensors", and LTX's MODELS-LTX-2.3.md says
#               to pair them with a 2.3 checkpoint.
#
# 2.3 ships MONOLITHS — one file carrying transformer, both VAEs and the text
# projection — so the CLI differs from 2.5's five split paths: it takes
# --checkpoint-path (or --distilled-checkpoint-path for distilled.py) plus
# --gemma-root pointing at a Gemma HF directory.
#   ltx23-pure — the 2.3 DEV monolith through ti2vid_one_stage, with NO
#               distillation anywhere. Both two-stage parsers make
#               --distilled-lora required=True, so "dev + HQ" still applies a
#               distilled schedule at stage 2; ti2vid_one_stage is the only path
#               that applies none. Use it when the model author says the weights
#               perform best undistilled.
#
#               The cost is real: one stage means no 2x latent upscale, so every
#               step denoises at the full target resolution rather than a quarter
#               of it, and the undistilled default is 30 steps against 8. Expect
#               it to be several times slower and much heavier on VRAM.
MODELS_23 = MODELS_ROOT / "ltx-2.3"
# Content LoRAs live here, NOT under models/ltx-2.5/loras (that holds the
# distilled LoRA a two-stage graph needs). Requests name a file in this directory
# rather than passing a path, so a browser cannot walk the filesystem.
LORA_DIR = Path(os.environ.get("LORA_DIR", MODELS_ROOT / "loras"))
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "5400"))


class ContentLora(BaseModel):
    """One motion/act LoRA in a pose's chain, with its own per-stage strengths.

    Separate strengths because stage 1 generates at half size from noise and stage 2 refines
    the 2x-upscaled latent. Lowering a competing content LoRA on stage 2 is a recorded lever
    and one flat number cannot express it.
    """

    name: str
    s1: float = Field(default=0.6, ge=0.0, le=2.0)
    s2: float = Field(default=0.6, ge=0.0, le=2.0)


class Lora(BaseModel):
    """A content LoRA by filename, resolved inside LORA_DIR.

    `strength` is the whole story until one of the per-stage fields is set. The
    two passes are not interchangeable -- stage 1 generates at half size from
    noise, stage 2 refines the 2x-upscaled latent from 0.85 -- and the community
    dev workflows exploit that: `Dev_Full-Steps` applies the distilled LoRA at
    0.3 on the base pass and 0.6 on the refinement pass. A content LoRA has the
    same asymmetry available to it, and it is the lever for the case where a
    LoRA carries the motion but degrades anatomy: keep it low where the shape is
    decided, raise it where detail is.

    Both default to `strength`, so a request that does not set them renders the
    graph exactly as before.
    """
    name: str
    strength: float = Field(default=0.6, ge=0.0, le=2.0)
    strength_stage_1: float | None = Field(default=None, ge=0.0, le=2.0)
    strength_stage_2: float | None = Field(default=None, ge=0.0, le=2.0)

    def at(self, stage: int) -> float:
        per = self.strength_stage_1 if stage == 1 else self.strength_stage_2
        return self.strength if per is None else per


class Keyframe(BaseModel):
    # data URI or http(s) URL, exactly like keyframe-server's image_urls
    image: str
    # Omit both and the recipe's defaults are applied across the whole set.
    index: int | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class JobRequest(BaseModel):
    # Empty is allowed and meaningful: with a strong conditioning image and a
    # LoRA carrying the motion, an empty prompt is a real configuration rather
    # than an oversight.
    prompt: str = ""
    # Which graph renders this job. A filename in workflows/, not a path.
    # This is what the old `pipeline` field was reaching for: the CLI had five
    # hardcoded modes, a ComfyUI graph carries its own sampler, scheduler and
    # sigmas, so the graph IS the pipeline.
    workflow: str | None = None
    # A validated recipe by name, e.g. "Missionary POV". When set, the graph is
    # built entirely from recipes.json and the free-form fields below are
    # IGNORED -- a recipe is a pinned configuration, not a starting point. The
    # resolved graph is hashed so the UI and the offline tools can be proven to
    # produce the same render.
    recipe: str | None = None
    # Which character's sheet tab the recipe comes from. Omit for the first.
    character: str | None = None
    # Base model, by the name ComfyUI lists. None keeps the graph's own.
    #
    # 2.3 checkpoints are monoliths, so this moves every loader that names the
    # file, not just the transformer one. Note the pairing rule the Sulphur 2
    # author states plainly: "Use the LoRA or the full model, never both at the
    # same time." A DISTILLED checkpoint already carries the distillation, so
    # distilled_lora_strength should be 0 against one -- and the 8/3 sigma
    # tables, which are a distilled schedule, become the right schedule again.
    checkpoint: str | None = None
    # Content LoRAs, spliced into the graph in order.
    loras: list[Lora] = Field(default_factory=list, max_length=4)
    keyframes: list[Keyframe] = Field(min_length=1, max_length=12)
    width: int = 512
    height: int = 768
    num_frames: int = 121
    frame_rate: int = 24
    seed: int | None = None
    # Steers the graph's negative encoder.
    negative_prompt: str | None = None
    # Video CRF applied to the conditioning frame before it anchors the render. None leaves
    # the workflow's own value alone; 0 is meaningful and bypasses the encode entirely.
    img_compression: int | None = Field(default=None, ge=0, le=51)
    # The pose's content LoRA -- motion and act -- chained ahead of the character LoRA on
    # both stage branches. Named explicitly rather than reusing `loras[0]`, which the recipe
    # path already spends on the CHARACTER LoRA; one list carrying two different roles by
    # index is how the wrong one gets loaded.
    #
    # None and "none" both mean "render without one", which is what every pose does today.
    # Content LoRAs STACK and are applied IN ORDER (console#410). Each carries its own
    # per-stage strengths, both defaulting to 0.6 -- exactly what resolve() hardcoded before
    # any of this was configurable -- so an entry naming only a LoRA renders at the
    # validated strength.
    #
    # Capped at 4, matching `loras` above. Order is part of the configuration: the same
    # LoRAs in a different order are a different render.
    content_loras: list[ContentLora] = Field(default_factory=list, max_length=4)
    num_inference_steps: int | None = Field(default=None, ge=1, le=50)
    # Steps per pass. None leaves that stage on the schedule the graph ships.
    #
    # The defaults are not round numbers, they are the sigma tables in the
    # workflow: 8 steps on stage 1, 3 on stage 2, shared byte-for-byte across
    # every distilled workflow in the RuneXX catalogue (stage 2's table is
    # identical in all eight, the dev ones included). Asking for a different
    # count switches that stage to BasicScheduler on linear_quadratic, which is
    # the only other regime the catalogue uses -- Dev_Full-Steps at 20,
    # DEV_Experimental_3-Pass at 35.
    #
    # Stage 1's 8 is softer than it reads: its first four steps each move sigma
    # by 0.00625, so half the budget covers 2.5% of the trajectory and the real
    # work is the last four.
    steps_stage_1: int | None = Field(default=None, ge=1, le=60)
    steps_stage_2: int | None = Field(default=None, ge=1, le=60)
    # Guidance scale, applied to both CFGGuiders. The base workflow bakes
    # cfg=1, at which the negative prompt is dead weight: the guidance sum
    # cond + cfg*(cond - uncond) never reads the uncond side. The CLI
    # defaulted to 3.0, matching the reference workflows, so the UI starts
    # there too.
    cfg: float | None = Field(default=None, ge=1.0, le=10.0)
    # Per-pass guidance. 3.0 is a STAGE 1 number: every dev workflow in the
    # catalogue does its guidance work on the base pass and leaves the refine
    # pass on a plain CFGGuider at cfg 1. Setting one value for both puts an
    # off-recipe cfg on a 3-step distilled pass. These override `cfg` per stage.
    # Swap stage 1's plain CFGGuider for the guidance stack every dev-checkpoint
    # workflow in the catalogue actually uses: a MultimodalGuider carrying
    # spatiotemporal guidance, CFG rescale and a separate audio scale, none of
    # which a CFGGuider can express. Defaults are Dev_Full-Steps' own values.
    multimodal_guidance: bool = False
    stg: float | None = Field(default=None, ge=0.0, le=10.0)
    rescale: float | None = Field(default=None, ge=0.0, le=1.0)
    stg_blocks: str | None = None
    cfg_stage_1: float | None = Field(default=None, ge=1.0, le=10.0)
    cfg_stage_2: float | None = Field(default=None, ge=1.0, le=10.0)
    # The distilled LoRA's strength. The base workflow bakes 0.5 into one
    # LoraLoaderModelOnly feeding both passes, but the DR34ML4Y author states
    # distillation "actively fights" the content LoRAs and recommends
    # 0.25-0.35 on the dev checkpoint. None leaves the graph as authored.
    distilled_lora_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    # Per-stage distilled-LoRA strength on the hq paths.
    #
    # These matter more than the names suggest. LTX defaults stage 1 to 0.25 and
    # stage 2 to 0.50, and the DR34ML4Y author states plainly that the
    # distillation LoRA and checkpoint "actively fight the nsfw training and
    # account for body horror", recommending the full dev checkpoint at 0.25-0.35
    # distill strength. The stage-2 default of 0.5 sits ABOVE that range, so the
    # out-of-the-box hq configuration is already past what the LoRA tolerates.
    # CFG guidance. "Adjust steps and CFG accordingly for both passes" is the
    # other half of that advice, and neither was reachable before.
    # CFG is not the interesting one -- the CLI already defaults to 3.0, which
    # matches the reference workflows. STG is: the LTX_2_3_HQ preset ships with
    # stg=0.0 and no stg_blocks ("STG off" per its own comment) while RuneXX's
    # LTX-2.3 Dev Full-Steps workflow runs stg=3 with rescale=0.9 on block 28.
    # So the stock hq path has spatiotemporal guidance disabled entirely.
    # Keyframes larger than the video are downscaled here, which is the point of
    # decoupling the two: the board can hold 832x1216 images so face edits work
    # on the full-resolution face, while the clip renders at whatever size the
    # GPU and the shot actually want. Upscaling is refused instead, because
    # inventing pixels to feed a conditioning frame is never what was meant.
    allow_upscale: bool = False
    # Off-grid indices are REPORTED, not refused.
    #
    # The recipe says an off-grid index "gets snapped elsewhere". No snapping
    # code exists: VideoConditionByKeyframeIndex.apply_to does
    # `positions += frame_idx` and divides by fps, so the index is an exact
    # continuous time. The reasonable worry is that a keyframe token which does
    # not coincide with a latent slot spreads its influence over the neighbours
    # instead of pinning one -- but that is inference, not measurement, and
    # working commands here use index 8. So off-grid comes back flagged on the
    # job and `strict_grid` opts into refusing it.
    strict_grid: bool = False
    snap_indices: bool = False


@dataclass
class Job:
    id: str
    req: JobRequest
    status: str = "None"          # None -> Processing -> Done | Failed
    video: str | None = None
    error: str | None = None
    placement: list[dict] = field(default_factory=list)
    prompt_id: str | None = None
    loras: list[dict] = field(default_factory=list)
    # What each pass actually ran, read back off the submitted graph rather than
    # echoed from the request -- the graph is the only thing that renders, and a
    # step count the workflow could not express is exactly the kind of mismatch
    # that goes unnoticed.
    stages: list[dict] = field(default_factory=list)
    # Things a caller should know that are not failures — same principle as
    # reporting an off-grid keyframe index rather than refusing it.
    notes: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    log_tail: list[str] = field(default_factory=list)

    def view(self, base: str) -> dict:
        d = {
            "job_id": self.id,
            "status": self.status,
            "video": f"{base}/job/{self.id}/video" if self.status == "Done" else None,
            "placement": self.placement,
            "loras": self.loras,
            "stages": self.stages,
            "notes": self.notes,
            "prompt_id": self.prompt_id,
            "queued_s": round((self.started or time.time()) - self.created, 1),
        }
        if self.started:
            d["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        if self.error:
            d["error"] = self.error
            d["log_tail"] = self.log_tail[-25:]
        return d


JOBS: dict[str, Job] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()
_LOCK = threading.Lock()

app = FastAPI(title="ltx-job-runner")


def decode_image(src: str, dest: Path):
    if src.startswith("data:"):
        dest.write_bytes(base64.b64decode(re.sub(r"^data:[^,]+,", "", src)))
    elif src.startswith(("http://", "https://")):
        dest.write_bytes(requests.get(src, timeout=120).content)
    else:
        raise HTTPException(400, "keyframe image must be a data URI or http(s) URL")


def safetensors_header(path: Path) -> dict:
    """Tensor names and metadata, without a tensor framework and without the file.

    The format is 8 bytes of little-endian u64 header length, then that many
    bytes of UTF-8 JSON. So this reads a few hundred KB off a 42 GB checkpoint
    and needs neither torch nor numpy -- `safe_open(framework="pt")` would drag
    torch into this venv purely to enumerate strings, and the whole point of the
    runner having its own venv is that LTX's stays untouched.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


_TKEYS: dict[str, set[str]] = {}


def _transformer_keys(transformer: Path) -> set[str]:
    """Weight names as the loader sees them, cached (header read, no tensors)."""
    key = str(transformer)
    if key not in _TKEYS:
        pre = "model.diffusion_model."
        _TKEYS[key] = {k[len(pre):] if k.startswith(pre) else k
                       for k in safetensors_header(transformer) if k != "__metadata__"}
    return _TKEYS[key]


def lora_coverage(lora: Path, transformer: Path) -> tuple[int, int]:
    """(fused, targeted) weights for this LoRA against this transformer.

    A LoRA whose keys do not line up fuses NOTHING and says nothing about it:
    `_affected_weight_keys` in fuse_loras.py matches purely on a
    `.lora_A.weight` naming convention, and `apply_loras` then iterates an empty
    set. No error, no warning, no log line -- the run looks completely normal and
    the LoRA simply is not there. Nothing in LTX logs LoRA loading at INFO, so
    there is otherwise no way to tell from the output of a job.

    Both sides get a prefix strip at load, which is the whole subtlety: the
    transformer loses `model.diffusion_model.` (LTXV_MODEL_COMFY_RENAMING_MAP)
    and the LoRA loses `diffusion_model.` (LTXV_LORA_COMFY_RENAMING_MAP), and
    they meet at `transformer_blocks.*`. Comparing the raw file keys reports 0%
    for every LoRA, which looks like a catastrophe and is just the wrong test.
    """
    pre = "diffusion_model."
    keys = [k[len(pre):] if k.startswith(pre) else k
            for k in safetensors_header(lora) if k != "__metadata__"]
    suffix = ".lora_A.weight"
    affected = {k[: -len(suffix)] + ".weight" for k in keys if k.endswith(suffix)}
    return len(affected & _transformer_keys(transformer)), len(affected)


def lora_base_model(lora: Path) -> str | None:
    """What the LoRA file says it was trained against, if it says anything.

    Trainers record this inconsistently -- `ss_sd_model_name`, `ss_base_model_version`,
    or nothing at all -- so absence proves nothing and this is advisory only.
    """
    meta = safetensors_header(lora).get("__metadata__") or {}
    return meta.get("ss_sd_model_name") or meta.get("ss_base_model_version")


def recorded_version(base: str | None) -> str | None:
    """LTX generation a LoRA claims, or None when the file does not really say.

    Deliberately conservative. Only "2.3" or "2.5" appearing in the recorded base
    counts; "ltx2" and "ltx2_v1" are era-ambiguous and two of the six files here
    record nothing at all. Guessing from those would produce confident warnings
    about files whose provenance is genuinely unknown, and a warning that fires
    on unknowns is one people learn to click through.
    """
    if not base:
        return None
    for v in ("2.5", "2.3"):
        if v in base:
            return v
    return None


def workflow_dir() -> Path:
    return WORKFLOW.parent


def resolve_workflow(name: str | None) -> Path:
    """A filename in workflows/, never a path — same reasoning as resolve_lora:
    these come from a browser."""
    if not name:
        return WORKFLOW
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(422, f"workflow must be a filename in {workflow_dir()}")
    p = workflow_dir() / name
    if not p.is_file():
        avail = sorted(f.name for f in workflow_dir().glob("*.json"))
        raise HTTPException(422, f"no such workflow {name!r}. Available: {avail}")
    return p


def resolve_lora(name: str) -> Path:
    """A filename inside LORA_DIR, never a path.

    Requests come from a browser, so anything path-shaped is refused outright
    rather than sanitised -- there is no legitimate reason for a LoRA reference
    to contain a separator.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(422, f"lora must be a filename in {LORA_DIR}, not a path")
    p = (LORA_DIR / name)
    if not p.is_file():
        avail = sorted(f.name for f in LORA_DIR.glob("*.safetensors"))
        raise HTTPException(422, f"no such lora {name!r}. Available: {avail}")
    return p



def _checked_content_lora(name: str | None) -> str | None:
    """Return the content LoRA name, having proven the file is actually there.

    "none" and empty both mean "render without one" and are passed through untouched --
    resolve() owns that vocabulary, and turning it into a filename lookup here is exactly
    the bug that would make `none.safetensors` a 422.
    """
    if not name or name.strip().lower() == "none":
        return name
    n = name.strip()
    resolve_lora(n if n.endswith(".safetensors") else n + ".safetensors")
    return name


def normalise(path: Path, width: int, height: int, allow_upscale: bool) -> str | None:
    """Bring a keyframe to the exact generation resolution. Returns what it did.

    Conditioning frames must arrive at the generation resolution, and LTX does
    not refuse a mismatch -- decode.py runs `resize_and_center_crop` on every
    conditioning image. So the resize happens regardless; the only question is
    whether it is done deliberately with a good filter or incidentally with
    whatever torch interpolation LTX reaches for, and whether anyone is told.
    Doing it here makes it LANCZOS, logged, and reported back on the job.

    Upscaling is refused rather than performed: a keyframe smaller than the clip
    means detail is being invented to fill a frame the model will then treat as
    ground truth.
    """
    with Image.open(path) as im:
        sw, sh = im.size
        if (sw, sh) == (width, height):
            return None
        scale = max(width / sw, height / sh)
        if scale > 1.0 and not allow_upscale:
            raise RuntimeError(
                f"{path.name} is {sw}x{sh}, smaller than the {width}x{height} clip. "
                f"Upscaling it would invent detail the model then treats as ground "
                f"truth -- render at or below the keyframe size, or pass "
                f"allow_upscale=true if you really mean it.")
        # Centre-crop to the target aspect first, then one resample. Cropping
        # after would resample content that is about to be thrown away.
        cw, ch = min(sw, round(width / scale)), min(sh, round(height / scale))
        left, top = (sw - cw) // 2, (sh - ch) // 2
        out = im.convert("RGB").crop((left, top, left + cw, top + ch)) \
                .resize((width, height), Image.LANCZOS)
    out.save(path)
    kept = (cw * ch) / (sw * sh)
    return (f"{sw}x{sh} -> {width}x{height} ({width / cw:.2f}x, "
            f"kept {kept:.0%} of frame)")


def plan(req: JobRequest) -> list[dict]:
    """Resolve every keyframe to an on-grid (index, strength). Raises 422 if it cannot."""
    n = len(req.keyframes)
    auto_idx = ltx_grid.auto_place(n, req.num_frames)
    auto_str = ltx_grid.default_strengths(n)
    out = []
    for i, kf in enumerate(req.keyframes):
        idx = auto_idx[i] if kf.index is None else kf.index
        snapped, off_grid = False, False
        if not ltx_grid.is_on_grid(idx):
            if req.strict_grid:
                raise HTTPException(422,
                    f"keyframe {i}: index {idx} is off the latent grid (0 or 1+8k); "
                    f"nearest are {ltx_grid.snap(idx - 4)} and {ltx_grid.snap(idx + 4)}.")
            if req.snap_indices:
                idx, snapped = ltx_grid.snap(idx), True
            else:
                off_grid = True
        if idx > req.num_frames:
            raise HTTPException(422,
                f"keyframe {i}: index {idx} is past num_frames={req.num_frames}")
        out.append({"index": idx,
                    "strength": auto_str[i] if kf.strength is None else kf.strength,
                    "snapped_from": kf.index if snapped else None,
                    "off_grid": off_grid})
    idxs = [o["index"] for o in out]
    if len(set(idxs)) != len(idxs):
        raise HTTPException(422, f"two keyframes share a latent slot: {idxs}. The later "
                                 f"one would silently win.")
    if idxs != sorted(idxs):
        raise HTTPException(422, f"keyframe indices must ascend, got {idxs}")
    return out


def comfy_vram_free_gb() -> float:
    """Free VRAM on the render device, as our own ComfyUI sees it."""
    r = requests.get(f"{COMFY_URL}/system_stats", timeout=60)
    r.raise_for_status()
    dev = (r.json().get("devices") or [{}])[0]
    return float(dev.get("vram_free", 0)) / 1e9


def free_the_gpu() -> float:
    """Make room for a render. Returns free GB.

    Two tenants, freed in order of what it costs to reload them.

    keyframe-server first: it holds the Qwen checkpoint (~20 GB) and never
    releases it on its own, and it is not the one about to render.

    Then, ONLY if that was not enough, our own ComfyUI. It keeps the LTX model
    resident between jobs, which is what we want -- reloading a 46 GB monolith
    costs far more than the memory is worth while jobs keep arriving. But that
    residency is OURS and reclaimable, so counting it as "something else holds
    the card" is simply wrong: measured 18.4 GB held by our own container with
    keyframe-server down at 256 MB, which failed the floor and refused a render
    that would have reused those very weights.

    Note the residency is invisible to the obvious check -- `torch_vram_total`
    reads 0.77 GB because ComfyUI stages weights outside torch's caching
    allocator under cudaMallocAsync. So there is nothing to subtract; the only
    way to know it is reclaimable is to reclaim it.

    Paying the reload only when memory is actually tight keeps the fast path
    free and pays the cost exactly when it buys something.
    """
    # keyframe-server is OPTIONAL. If it is not running there is no GPU for it to free, and
    # that is success, not failure — this used to raise, so an absent collaborator failed a
    # render that had already been claimed. Connection refused therefore falls through to
    # measuring the card directly.
    #
    # A server that IS up and refuses to yield stays a hard failure: that is a real conflict
    # over the GPU, and proceeding into a model load would OOM instead.
    free = None
    try:
        r = requests.post(f"{KEYFRAME_URL}/free", timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"keyframe-server /free -> {r.status_code} {r.text[:200]}")
        free = float(r.json().get("vram_free_gb", 0.0))
    except requests.ConnectionError:
        print(f"[gpu] keyframe-server not running at {KEYFRAME_URL} — nothing to free",
              flush=True)
    except requests.Timeout as e:
        # Up but not answering. Distinct from absent, and worth failing on: something holds
        # the card and is not letting go.
        raise RuntimeError(f"keyframe-server timed out yielding the GPU: {e}")

    if free is None:
        free = comfy_vram_free_gb()

    if free >= MIN_FREE_GB:
        return free
    try:
        print(f"[gpu] {free:.1f} GB free after keyframe-server yielded; dropping our "
              f"own resident models too", flush=True)
        requests.post(f"{COMFY_URL}/free",
                      json={"unload_models": True, "free_memory": True}, timeout=180)
        # Unloading is not synchronous with the driver's accounting -- reading
        # vram_free immediately after returned the same 5.6 GB that triggered
        # the free in the first place, and failed two renders on it. Give it a
        # few seconds to actually show up.
        for _ in range(10):
            time.sleep(2)
            free = max(free, comfy_vram_free_gb())
            if free >= MIN_FREE_GB:
                print(f"[gpu] {free:.1f} GB free after unloading", flush=True)
                return free
    except requests.RequestException as e:
        print(f"[gpu] could not free our own ComfyUI ({e})", flush=True)
    return free


DISTILLED_T = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DEV_T = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
DISTILLED_LORA = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"



def derive_size(path: Path) -> tuple[int, int]:
    """Recipe resolution is derived, not chosen: take the start frame's native
    size and round DOWN to /64, preserving aspect. 64 because the two-stage
    pipeline's own assert_resolution demands it."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return max(64, (w // 64) * 64), max(64, (h // 64) * 64)


def run_job(job: Job):
    workdir = JOBS_DIR / job.id
    workdir.mkdir(parents=True, exist_ok=True)
    job.status = "Processing"
    job.started = time.time()
    try:
        for i, kf in enumerate(job.req.keyframes):
            dest = workdir / f"kf{i + 1}.png"
            decode_image(kf.image, dest)
            # A recipe DERIVES its resolution from the start frame, so the
            # requested width/height are meaningless and must be replaced before
            # normalise() validates against them. Without this, any start frame
            # whose size differs from whatever the caller happened to send is
            # rejected as "smaller than the clip" -- which is every frame that is
            # not the caller's default.
            if i == 0 and job.req.recipe:
                job.req.width, job.req.height = derive_size(dest)
                print(f"[{job.id}] recipe: derived {job.req.width}x{job.req.height} "
                      f"from the start frame", flush=True)
            # Conditioning frames must arrive at the exact generation
            # resolution. LTX does not refuse a mismatch -- decode.py runs
            # `resize_and_center_crop(image, height, width)` on every
            # conditioning image -- so a wrong-sized keyframe is silently
            # rescaled and, if the aspect differs, silently re-framed. A
            # landscape still cropped into a portrait target can lose the
            # subject's head entirely, and nothing in the output says so.
            # storyboard-ui already normalises uploads to the board size, so
            # this only fires when something bypassed it.
            note = normalise(dest, job.req.width, job.req.height, job.req.allow_upscale)
            if note:
                job.placement[i]["resized"] = note
        (workdir / "prompt.txt").write_text(job.req.prompt)

        for lo in job.req.loras:
            # Coverage is a DIAGNOSTIC -- how many of the transformer's weights this LoRA
            # actually fuses into. It must never fail a render, and it did: TRANSFORMER
            # defaults to ltx-2.3-22b-dev.safetensors, which a worker does not download
            # (the recipe runs on sulphur_dev_bf16), so every job with a character LoRA died
            # on a pod with
            #
            #   FileNotFoundError: .../ltx-2.3/diffusion_models/ltx-2.3-22b-dev.safetensors
            #
            # while working on the 3090 purely because that box happens to keep the file.
            # Measured against the checkpoint THIS JOB renders on, not the LTX_TRANSFORMER
            # env var. With a per-pose checkpoint (console#404) those differ, and a coverage
            # number computed against the wrong file is worse than none: it looks like a
            # check while measuring something the render never touches.
            #
            # This number is the whole reason a base-model comparison is readable. Character
            # LoRAs were trained against sulphur; against another base a LoRA whose keys do
            # not line up fuses NOTHING and says nothing about it, and the render comes back
            # as the base model with none of the character in it. "fuses 0/N" is what tells
            # you that happened, rather than the output being blamed on the prompt.
            ck = (job.req.checkpoint or "").strip()
            if ck:
                if not ck.endswith(".safetensors"):
                    ck += ".safetensors"
                target = MODELS_ROOT / "ltx-2.3/diffusion_models" / ck
            else:
                target = TRANSFORMER
            try:
                hit, total = lora_coverage(resolve_lora(lo.name), target)
            except Exception as e:
                hit, total = None, None
                print(f"[{job.id}] lora coverage unavailable ({type(e).__name__}: {e}) "
                      f"-- rendering anyway", flush=True)
            job.loras.append({"name": lo.name, "strength": lo.strength,
                              "strength_stage_1": lo.at(1), "strength_stage_2": lo.at(2),
                              "fused": hit, "targeted": total})
            if hit is not None:
                print(f"[{job.id}] lora {lo.name} @{lo.at(1)}/{lo.at(2)} (stage 1/2) "
                      f"-> fuses {hit}/{total} weights against {target.name}", flush=True)
                if hit == 0:
                    # Not fatal — comparing base models is a legitimate reason to be here —
                    # but it must be impossible to miss. A 0-fusion render is the base model
                    # with none of the character in it, and it looks entirely normal.
                    print(f"[{job.id}] WARNING {lo.name} fused 0 weights against "
                          f"{target.name}: this render carries NONE of that LoRA",
                          flush=True)
                    job.notes.append(f"{lo.name} fused 0/{total} against {target.name} "
                                     f"— no character in this render")

        # keyframe-server holds ~20 GB once Qwen is loaded and never gives it
        # back on its own; LTX needs essentially the whole card. Ask it to yield
        # before taking the GPU, and refuse rather than die mid-load.
        free = free_the_gpu()
        if free < MIN_FREE_GB:
            # WARN, do not refuse.
            #
            # This floor was written for the CLI, which loaded the whole
            # transformer up front and genuinely needed the room. ComfyUI stages
            # dynamically ("Model LTXAV prepared for dynamic VRAM loading,
            # 40050MB Staged") and renders happily with far less free, having
            # peaked at 22.6 GB of 24.5 alongside other resident containers.
            #
            # Refusing has now cost more renders than it has saved: three failed
            # here while the memory in question was our own ComfyUI's resident
            # LTX weights -- the very weights the job was about to reuse. And
            # when the card really is short, the failure is an OOM that the
            # handler below already reports with actionable advice, which is a
            # better outcome than declining to try.
            print(f"[{job.id}] WARNING: only {free:.1f} GB VRAM free (floor is "
                  f"{MIN_FREE_GB}); proceeding anyway -- most of it is likely our "
                  f"own resident model, which this render will reuse", flush=True)
            job.notes.append(f"started with {free:.1f} GB free, below the "
                             f"{MIN_FREE_GB} GB floor")

        cf = comfy.Comfy(COMFY_URL)
        guides = []
        for i, kf in enumerate(job.req.keyframes, start=1):
            data = (workdir / f"kf{i}.png").read_bytes()
            name = cf.upload_image(data, f"{job.id}_kf{i}.png")
            guides.append({"name": name,
                           "index": job.placement[i - 1]["index"],
                           "strength": job.placement[i - 1]["strength"]})

        if job.req.recipe:
            # Recipe path: the graph is the validated workflow, patched with the resolved
            # configuration this request carries. Nothing is looked up -- the recipe used to
            # be read by NAME from recipes/recipes.json, which meant a pose created in the
            # console was unknown here ("unknown recipe 'Doggystyle Side v2'") and an edited
            # prompt on a seeded pose was silently ignored in favour of the file's copy.
            # No free-form patching and no guide splicing: the validated workflow conditions
            # on a single LoadImage which drives size as well.
            graph = comfy.load_workflow(resolve_workflow(recipe_mod.RECIPE_WORKFLOW))
            w, h = derive_size(workdir / "kf1.png")
            lora = job.req.loras[0] if job.req.loras else None
            graph = recipe_mod.resolve(
                graph, guides[0]["name"], w, h,
                prompt=job.req.prompt,
                negative=job.req.negative_prompt,
                checkpoint=job.req.checkpoint,
                char_lora=(lora.name if lora else None),
                char_s1=(lora.at(1) if lora else 0.8),
                char_s2=(lora.at(2) if lora else 1.5),
                # Checked here, before anything is submitted. A LoRA that is merely absent
                # should cost a 422 in the first second, not a failure deep in a render that
                # has already held the GPU — and the message names the file and lists what IS
                # present, which is what turned a past "no such lora 'pay_v2_e05'" into a
                # one-look fix. resolve_lora also refuses anything path-shaped.
                #
                # This matters more for content LoRAs than for character ones: they arrive in
                # the bucket after a worker has booted, and the worker only syncs at boot, so
                # "the pose names a LoRA this box has never heard of" is the expected failure
                # rather than an exotic one.
                content_loras=[
                    {"name": _checked_content_lora(c.name), "s1": c.s1, "s2": c.s2}
                    for c in job.req.content_loras
                ],
                img_compression=job.req.img_compression,
            )
            if job.req.num_frames:
                comfy.set_frames(graph, job.req.num_frames)
            # A recipe pins the configuration, not the draw. Without this the
            # graph's baked seed would be used for every recipe render.
            comfy.set_seed(graph, job.req.seed if job.req.seed is not None
                           else random.randint(0, 2**31 - 1))
            job.req.width, job.req.height = w, h
            gh = recipe_mod.graph_hash(graph)
            # Name the output for what it IS: <char lora>-<recipe slug>.
            char = re.sub(r"\.safetensors$", "", (lora.name if lora else "") or "")
            if char.lower() in ("", "none"):
                char = "no-char-lora"
            slug = re.sub(r"[^a-z0-9]+", "-", job.req.recipe.lower()).strip("-")
            graph["140"]["inputs"]["filename_prefix"] = f"{char}-{slug}"
            # No "as validated" claim any more. That compared this graph against the sheet's
            # baseline for the same name; with recipes in the database, whether a pose is
            # validated is a field on the row and the API and console own it. Asserting it
            # here would mean re-introducing a second source of truth to compare against,
            # which is the bug this change removes.
            # Read back from the RESOLVED GRAPH, not from the request. The request is what
            # was asked for; the graph is what will render, and those differ exactly when
            # something has gone wrong — a field silently dropped (this model has no
            # extra="forbid", so an older engine ignores unknown keys without complaint), a
            # name that failed its "none" check, a strength that fell back to a default.
            # A log that repeats the request would agree with itself in precisely the cases
            # worth catching.
            #
            # Absence is stated, never implied. "content none" and no line at all look the
            # same to a reader trying to work out whether a LoRA loaded.
            job.notes.append(
                f"recipe {job.req.recipe!r} · base {recipe_mod.base_model_note(graph)} · "
                f"{recipe_mod.lora_stack_note(graph)} · graph {gh[:12]}"
            )
            print(f"[{job.id}] recipe {job.req.recipe!r} -> {w}x{h}, base {ck}, "
                  f"{recipe_mod.lora_stack_note(graph)}, graph {gh}", flush=True)
            (workdir / "graph.json").write_text(json.dumps(graph, indent=1))
            job.stages = comfy.describe_stages(graph)
            job.prompt_id = cf.submit(graph)
            entry = cf.wait(job.prompt_id, timeout_s=JOB_TIMEOUT_S)
            fn = cf.output_video(entry)
            if not fn:
                raise RuntimeError("ComfyUI finished but produced no video")
            out = workdir / "out.mp4"
            out.write_bytes(cf.view(fn))
            job.video = str(out)
            job.status = "Done"
            print(f"[{job.id}] done in {time.time() - job.started:.0f}s -> {out}", flush=True)
            return

        graph = comfy.load_workflow(resolve_workflow(job.req.workflow))

        # Reconcile the frame count with what this graph can express BEFORE the
        # guides go in. A workflow driven by an integer seconds constant reaches
        # only 1+fps*s, so an 81-frame request renders 73 — and a guide planned
        # at index 81 against the promise would land past the end of the clip.
        actual = comfy.achievable_frames(graph, job.req.num_frames, job.req.frame_rate)
        if actual != job.req.num_frames:
            print(f"[{job.id}] {job.req.num_frames}f is not expressible by this "
                  f"graph; rendering {actual}f and replanning guides", flush=True)
            job.req.num_frames = actual
            job.placement = plan(job.req)
            guides = [{**g, "index": p["index"], "strength": p["strength"]}
                      for g, p in zip(guides, job.placement)]
        comfy.set_prompts(graph, job.req.prompt,
                          job.req.negative_prompt or None)
        comfy.set_cfg(graph, job.req.cfg, job.req.cfg_stage_1, job.req.cfg_stage_2)
        if job.req.multimodal_guidance:
            video = {}
            if job.req.cfg_stage_1 is not None:
                video["cfg"] = job.req.cfg_stage_1
            if job.req.stg is not None:
                video["stg"] = job.req.stg
            if job.req.rescale is not None:
                video["rescale"] = job.req.rescale
            applied = comfy.set_multimodal_guidance(
                graph, video=video,
                skip_blocks=job.req.stg_blocks or comfy.DEV_SKIP_BLOCKS)
            job.notes.append(f"stage 1 on multimodal guidance: {applied['video']}, "
                             f"blocks {applied['skip_blocks']}")
        if job.req.distilled_lora_strength is not None:
            comfy.set_distilled_lora(graph, job.req.distilled_lora_strength)
        if job.req.checkpoint:
            comfy.set_checkpoint(graph, job.req.checkpoint)
        comfy.set_dimensions(graph, job.req.width, job.req.height)
        comfy.set_frames(graph, job.req.num_frames)
        comfy.set_seed(graph, job.req.seed if job.req.seed is not None else 42)
        comfy.set_guides(graph, guides)
        comfy.set_loras(graph, [{"name": lo.name, "strength": lo.strength,
                                 "strength_stage_1": lo.at(1),
                                 "strength_stage_2": lo.at(2)}
                                for lo in job.req.loras])
        # After set_loras, so a stage that had to switch scheduler points its
        # BasicScheduler at that stage's own model chain rather than the shared
        # one it was hanging off before the split.
        comfy.set_steps(graph, job.req.steps_stage_1, job.req.steps_stage_2)
        (workdir / "graph.json").write_text(json.dumps(graph, indent=1))

        job.stages = comfy.describe_stages(graph)
        for st in job.stages:
            print(f"[{job.id}] stage {st['stage']}: {st['steps']} steps "
                  f"({st['schedule']}), loras {st['loras']}", flush=True)
        print(f"[{job.id}] submitting: {len(guides)} guides @ "
              f"{[g['index'] for g in guides]}, {job.req.width}x{job.req.height}, "
              f"{job.req.num_frames}f, {free:.1f} GB free", flush=True)
        job.prompt_id = cf.submit(graph)
        entry = cf.wait(job.prompt_id, timeout_s=JOB_TIMEOUT_S)

        fn = cf.output_video(entry)
        if not fn:
            raise RuntimeError("ComfyUI finished but produced no video")
        out = workdir / "out.mp4"
        out.write_bytes(cf.view(fn))
        job.video = str(out)
        job.status = "Done"
        print(f"[{job.id}] done in {time.time() - job.started:.0f}s -> {out}", flush=True)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "OOM" in msg or "out of memory" in msg.lower():
            # The card is 24 GB and shared. Say what to change rather than
            # returning torch's allocator dump.
            msg += (f" — {job.req.width}x{job.req.height} at {job.req.num_frames} "
                    f"frames did not fit. Reduce resolution or frame count; "
                    f"704x1280 at 241 frames is known to fit alongside the other "
                    f"resident containers.")
        job.status, job.error = "Failed", msg
        print(f"[{job.id}] FAILED: {job.error}", flush=True)
    finally:
        job.finished = time.time()


def worker():
    while True:
        jid = QUEUE.get()
        job = JOBS.get(jid)
        if job and job.status == "None":      # skip anything cancelled while queued
            run_job(job)
        QUEUE.task_done()


@app.post("/job")
def submit(req: JobRequest):
    req.num_frames = ltx_grid.valid_num_frames(req.num_frames)
    # 64, not 32. We always pass --spatial-upsampler-path, which makes this the
    # two-stage pipeline, and LTX's own assert_resolution(..., is_two_stage=True)
    # demands 64 (helpers.py). Validating at 32 here would let a 544-wide board
    # through to a ValueError deep inside the run, twenty minutes later.
    for name, v in (("width", req.width), ("height", req.height)):
        if v % 64:
            raise HTTPException(422,
                f"{name}={v} must be divisible by 64. The two-stage distilled "
                f"graph (spatial upsampler) requires it; only one-stage "
                f"graphs accept 32. Nearest: {(v // 64) * 64} or {(v // 64 + 1) * 64}.")
    placement = plan(req)
    job = Job(id=uuid.uuid4().hex[:12], req=req, placement=placement)
    with _LOCK:
        JOBS[job.id] = job
    QUEUE.put(job.id)
    print(f"[{job.id}] queued: {len(placement)} keyframes at "
          f"{[p['index'] for p in placement]} / {req.num_frames} frames", flush=True)
    return {"job_id": job.id, "status": job.status, "placement": placement,
            "queue_depth": QUEUE.qsize()}


@app.get("/job/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"no such job: {job_id}")
    return job.view(PUBLIC_BASE)


@app.get("/job/{job_id}/video")
def video(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.video or not Path(job.video).exists():
        raise HTTPException(404, "no video for that job (yet)")
    return FileResponse(job.video, media_type="video/mp4",
                        filename=f"ltx-{job_id}.mp4")


@app.post("/job/{job_id}/cancel")
def cancel(job_id: str):
    """Stop a job. There is one GPU and one worker, so cancelling the active job
    means interrupting ComfyUI; a queued one just never starts."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"no such job: {job_id}")
    if job.status in ("Done", "Failed", "Cancelled"):
        return job.view(PUBLIC_BASE)
    if job.status == "Processing" and job.prompt_id:
        comfy.Comfy(COMFY_URL).interrupt()
    job.status = "Cancelled"
    job.error = "cancelled by request"
    job.finished = time.time()
    print(f"[{job_id}] cancelled", flush=True)
    return job.view(PUBLIC_BASE)


@app.get("/jobs")
def list_jobs():
    return {"jobs": [j.view(PUBLIC_BASE) for j in
                     sorted(JOBS.values(), key=lambda j: -j.created)][:50]}


@app.get("/loras")
def loras():
    """Content LoRAs available to `loras: [{name, strength}]`."""
    if not LORA_DIR.is_dir():
        return {"dir": str(LORA_DIR), "loras": []}
    return {"dir": str(LORA_DIR),
            "loras": sorted(
                ({"name": f.name, "size_gb": round(f.stat().st_size / 1e9, 2),
                  "base_model": lora_base_model(f),
                  **dict(zip(("fused", "targeted"),
                             lora_coverage(f, MODELS / DISTILLED_T)))}
                 for f in LORA_DIR.glob("*.safetensors")),
                key=lambda x: x["name"])}


@app.get("/workflows")
def workflows():
    """Graphs available to `workflow`. Each carries its own sampler, scheduler
    and sigmas, which is why there is no longer a pipeline setting."""
    d = workflow_dir()
    return {"dir": str(d), "default": WORKFLOW.name,
            "workflows": sorted(f.name for f in d.glob("*.json")) if d.is_dir() else []}


@app.get("/checkpoints")
def checkpoints():
    """Base models ComfyUI can load, straight from its own schema.

    Asked of ComfyUI rather than globbed off disk: the folder mapping lives in
    extra_model_paths.yaml, and a file the mapping does not cover is invisible to
    a render no matter how present it is on the filesystem. This lists what will
    actually load.
    """
    try:
        r = requests.get(f"{COMFY_URL}/object_info/CheckpointLoaderSimple", timeout=120)
        r.raise_for_status()
        names = r.json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except Exception as e:
        raise HTTPException(503, f"could not read checkpoints from ComfyUI: {e}")
    return {"checkpoints": names}



@app.get("/recipes")
def list_recipes():
    """The validated recipes, read-only. Authoring happens in the recipe sheet;
    this endpoint never writes."""
    try:
        d = recipe_mod.load()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "character": None, "recipes": {}}
    return {"character": d.get("character"), "characters": d.get("characters", {}),
            "sources": d.get("sources", {}), "labels": d.get("labels", {}),
            "definitions": d.get("definitions", {}), "recipes": d.get("recipes", {})}

@app.post("/job/{job_id}/purge")
def purge_job(job_id: str):
    """Drop a finished job's local media once it is safely uploaded.

    Called by the daemon AFTER a successful upload, never before — the local file is the
    only copy until that upload lands, and reclaiming disk is not worth risking a render.

    Keeps graph.json and prompt.txt; see engine/purge.py for why.
    """
    return purge_mod.purge_job_dir(JOBS_DIR / job_id)


@app.post("/jobs/purge-all")
def purge_all_jobs(keep_recent: int = 5):
    """One-time sweep of everything already on disk (console#380)."""
    r = purge_mod.purge_all(JOBS_DIR, keep_recent=keep_recent)
    print(f"[purge] swept {r['dirs_purged']} job dirs, {r['files_removed']} files, "
          f"{r['freed_bytes'] / 1e9:.2f} GB", flush=True)
    return r


@app.get("/health")
def health():
    free = None
    try:
        free = requests.get(f"{KEYFRAME_URL}/health", timeout=5).json().get("vram_free_gb")
    except Exception:
        pass
    return {"status": "ok", "models_root": str(MODELS_ROOT),
            "workflow": str(WORKFLOW),
            "workflow_present": WORKFLOW.exists(),
            "comfy": COMFY_URL,
            "models_present": MODELS.exists(),
            "queue_depth": QUEUE.qsize(),
            "running": sum(1 for j in JOBS.values() if j.status == "Processing"),
            "keyframe_server": KEYFRAME_URL, "keyframe_vram_free_gb": free}


PUBLIC_BASE = ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--public-base", default=None,
                    help="base URL clients see, for building video links")
    args = ap.parse_args()
    PUBLIC_BASE = args.public_base or f"http://{args.host}:{args.port}"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f"ltx-engine on {args.host}:{args.port} | models={MODELS_ROOT} | "
          f"comfy={COMFY_URL} | keyframe-server={KEYFRAME_URL}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
