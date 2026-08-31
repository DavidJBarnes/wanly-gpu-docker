"""ComfyUI backend for ltx-engine.

Replaces the CLI subprocess this service used to shell out to. That interface
exposed no sampler, scheduler, shift, sigma or denoise control at all — grepped
for each, zero hits — and hardcoded a sampler per pipeline, so it could not
reproduce the workflows the community generates with. See
docs/ltx-2.3-comfyui-recipe.md §1.

The unit of work here is a **graph**, not a command line. A stored API-format
workflow is loaded, patched with the job's keyframes/prompt/dimensions, and
POSTed to ComfyUI.
"""
import base64
import copy
import json
import re
import time
from pathlib import Path

import requests

# Where the guides go. LTXVAddGuideMulti is the multi-keyframe primitive and its
# shape is exactly a storyboard's:
#
#   num_guides: N  ->  image_i (IMAGE) + frame_idx_i (INT) + strength_i (FLOAT)
#   outputs:           positive, negative, latent
#
# Note it returns CONDITIONING as well as a latent. Wiring only the latent
# encodes the guides but leaves them unable to steer anything — the guiders have
# to read positive/negative from this node too.
GUIDE_NODE = "LTXVAddGuideMulti"

# What the dev-checkpoint workflows schedule with once they leave the distilled
# sigma tables. `Dev_Full-Steps` runs it at 20 steps and `DEV_Experimental_3-Pass`
# at 35; the distilled variants never use it at all.
SCHEDULER = "linear_quadratic"


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # --- plumbing ---------------------------------------------------------

    def object_info(self) -> dict:
        r = requests.get(f"{self.base}/object_info", timeout=180)
        r.raise_for_status()
        return r.json()

    def upload_image(self, data: bytes, name: str) -> str:
        """Put an image in ComfyUI's input folder; returns the stored name."""
        r = requests.post(f"{self.base}/upload/image",
                          files={"image": (name, data, "image/png")},
                          data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        return r.json()["name"]

    def submit(self, graph: dict) -> str:
        r = requests.post(f"{self.base}/prompt", json={"prompt": graph}, timeout=180)
        if r.status_code != 200:
            # ComfyUI reports every node's validation errors at once, and they
            # are specific. Surfacing them beats a generic failure by a mile —
            # it is the only reliable oracle for a converted graph.
            try:
                errs = r.json().get("node_errors") or {}
                detail = "; ".join(
                    f"{nid} {e['class_type']}: {x['details'][:120]}"
                    for nid, e in errs.items() for x in e["errors"]) or r.text[:400]
            except Exception:
                detail = r.text[:400]
            raise ComfyError(f"ComfyUI rejected the graph: {detail}")
        return r.json()["prompt_id"]

    def queued(self, prompt_id: str) -> bool:
        """Whether ComfyUI still has this prompt running or pending."""
        r = requests.get(f"{self.base}/queue", timeout=30)
        if r.status_code != 200:
            return True                     # can't tell; assume alive
        q = r.json()
        for bucket in ("queue_running", "queue_pending"):
            for item in q.get(bucket) or []:
                if prompt_id in json.dumps(item):
                    return True
        return False

    def wait(self, prompt_id: str, timeout_s: int = 3600, poll: float = 5.0) -> dict:
        """Block until the prompt finishes.

        /history only gains the id once the run ends, so absence there means
        "not finished" -- but it also means "interrupted", because a cancelled
        prompt never lands in history at all. Without checking the queue too, a
        POST /interrupt leaves this polling until the full timeout: the GPU is
        idle, the render is gone, and the job still says Processing.

        The queue is checked only after history misses, and a cancelled prompt
        has to be absent from both twice running -- there is a brief window as a
        prompt completes where it has left the queue but not yet appeared in
        history, and treating that as a cancellation would fail finished jobs.
        """
        deadline = time.time() + timeout_s
        gone = 0
        while time.time() < deadline:
            r = requests.get(f"{self.base}/history/{prompt_id}", timeout=60)
            if r.status_code == 200 and (entry := r.json().get(prompt_id)):
                st = entry.get("status", {})
                # `completed` is NOT set on every terminal state. An OOM lands in
                # history as status_str "error" with completed False, so keying
                # only off `completed` polls a dead prompt until timeout — the
                # GPU idle, the render gone, the job still Processing.
                if st.get("status_str") == "error":
                    msgs = [str(m) for m in st.get("messages", [])]
                    err = next((m for m in reversed(msgs) if "execution_error" in m), "")
                    raise ComfyError(f"execution failed: {err[:400] or msgs[-1:]}")
                if st.get("completed"):
                    if st.get("status_str") != "success":
                        msgs = "; ".join(str(m)[:200] for m in st.get("messages", [])[-4:])
                        raise ComfyError(f"execution failed: {msgs}")
                    return entry
                gone = 0
            elif not self.queued(prompt_id):
                gone += 1
                if gone >= 2:
                    raise ComfyError(
                        "prompt is neither queued nor in history — interrupted "
                        "or dropped by ComfyUI")
            else:
                gone = 0
            time.sleep(poll)
        raise ComfyError(f"ComfyUI did not finish within {timeout_s}s")

    def interrupt(self) -> None:
        """Stop whatever ComfyUI is rendering. There is one GPU and one job, so
        this is unambiguous — there is nothing else it could be stopping."""
        requests.post(f"{self.base}/interrupt", timeout=30)
        requests.post(f"{self.base}/queue", json={"clear": True}, timeout=30)

    @staticmethod
    def output_video(entry: dict) -> str | None:
        """Filename of the rendered video, from a finished history entry."""
        for out in (entry.get("outputs") or {}).values():
            for item in (out.get("gifs") or []) + (out.get("videos") or []):
                if str(item.get("filename", "")).endswith((".mp4", ".webm")):
                    return item["filename"]
        return None

    def view(self, filename: str, subfolder: str = "", kind: str = "output") -> bytes:
        r = requests.get(f"{self.base}/view", timeout=300,
                         params={"filename": filename, "subfolder": subfolder, "type": kind})
        r.raise_for_status()
        return r.content


# --- graph patching -------------------------------------------------------


def _find(graph: dict, class_type: str) -> list[str]:
    return [k for k, v in graph.items() if v.get("class_type") == class_type]


def _one(graph: dict, class_type: str) -> str:
    ids = _find(graph, class_type)
    if not ids:
        raise ComfyError(f"workflow has no {class_type}")
    return ids[0]


def conditioning_slots(graph: dict) -> tuple[str, str]:
    """(positive_encoder_id, negative_encoder_id), read from the wiring.

    Never guess this. Picking the positive encoder by "whichever text is longer"
    once put an entire prompt into the NEGATIVE — the model was told to avoid
    what was asked for, and produced a slow camera drift over a still subject
    that still scored well on every automated smoothness measure.
    """
    cond = _one(graph, "LTXVConditioning")
    inputs = graph[cond]["inputs"]
    return str(inputs["positive"][0]), str(inputs["negative"][0])


def set_prompts(graph: dict, positive: str, negative: str | None) -> None:
    """Write the prompts. `negative=None` leaves the workflow's own negative
    encoder untouched — the author's quality negatives are better than an
    empty string, and the UI's negative is only worth sending when the user
    actually typed one."""
    pos, neg = conditioning_slots(graph)
    graph[pos]["inputs"]["text"] = positive
    if negative is not None:
        graph[neg]["inputs"]["text"] = negative


def set_cfg(graph: dict, cfg: float | None,
            stage_1: float | None = None, stage_2: float | None = None) -> None:
    """Guidance scale on the CFGGuiders, whole-graph or per pass.

    The base workflow bakes 1.0, at which the negative conditioning contributes
    nothing — cfg*(cond - uncond) collapses to cond, so the negative prompt is
    inert. The CLI defaulted to 3.0.

    But 3.0 is a STAGE 1 number. Every dev workflow in the RuneXX catalogue puts
    its guidance work on the base pass — `Dev_Full-Steps` and the DEV 3-pass both
    drive stage 1 through a MultimodalGuider at video cfg 3, stg 3, rescale 0.9
    on block 28 — and leaves the refine pass on a plain `CFGGuider` at **cfg 1**.
    Not one of them raises cfg on stage 2. Applying a single value to both
    guiders therefore puts an off-recipe cfg on a 3-step distilled pass, which is
    why the per-stage arguments exist.

    `cfg` sets both and is the backwards-compatible path; a per-stage value
    overrides it for that stage.
    """
    if cfg is None and stage_1 is None and stage_2 is None:
        return
    per = {}
    if cfg is not None:
        per = {1: cfg, 2: cfg}
    if stage_1 is not None:
        per[1] = stage_1
    if stage_2 is not None:
        per[2] = stage_2

    guiders = stage_guiders(graph)
    set_any = False
    for stage, value in per.items():
        node = graph.get(guiders.get(stage))
        if node and node["class_type"] == "CFGGuider":
            set_any |= _set_if_literal(node, "cfg", value)
    if not set_any:
        raise ComfyError("no stage guider is a CFGGuider with a literal cfg")


def set_distilled_lora(graph: dict, strength: float) -> None:
    """Retune the distilled LoRA. The base workflow bakes 0.5 into a single
    LoraLoaderModelOnly feeding both passes, and the DR34ML4Y author states
    distillation "actively fights" the content LoRAs — 0.25-0.35 is the
    recommended range on the dev checkpoint. Matched on the filename, never
    on position, so a reauthored graph still gets the right node."""
    touched = 0
    for v in graph.values():
        if v["class_type"] == "LoraLoaderModelOnly":
            name = str(v["inputs"].get("lora_name", ""))
            if "distilled" in name:
                v["inputs"]["strength_model"] = strength
                touched += 1
    if not touched:
        raise ComfyError("workflow has no distilled LoraLoaderModelOnly to retune")


def _set_if_literal(node: dict, field: str, value) -> bool:
    """Set an input only when it is a literal, never when it is a link.

    Overwriting a linked input with a scalar silently severs a wire. That is not
    a small edit: it removes whatever computed the value and substitutes a
    constant, so the graph still validates and still renders, just not the graph
    the author built.
    """
    cur = node["inputs"].get(field)
    if isinstance(cur, list):
        return False
    node["inputs"][field] = value
    return True


def set_dimensions(graph: dict, width: int, height: int) -> None:
    """Set the TARGET size. Stage sizes are derived and must stay derived.

    This workflow computes everything downstream of the target:

        INTConstant WIDTH/HEIGHT  ->  ImageResizeKJv2 (source to target)
                                  ->  ResizeImageMaskNode x0.5  (halve it)
                                  ->  GetImageSize
                                  ->  EmptyLTXVLatentVideo      (stage 1)
                                  ->  stage 2 upsamples x2      (back to target)

    So EmptyLTXVLatentVideo's width/height are LINKS, and forcing literals onto
    them makes stage 1 run at full size while stage 2 still doubles it — a
    960x640 request came back 1920x1280, OOM'd at 1216x832 because stage 1 was
    covering four times the intended area, and lost quality because the two-stage
    design is generate-small-then-refine-while-upscaling, not generate-full-then-
    upscale-again.

    Only the INTConstants are ours to set.
    """
    set_w = set_h = False
    for v in graph.values():
        if v["class_type"] == "INTConstant":
            title = str((v.get("_meta") or {}).get("title", "")).upper()
            if "WIDTH" in title:
                set_w |= _set_if_literal(v, "value", width)
            elif "HEIGHT" in title:
                set_h |= _set_if_literal(v, "value", height)
    # Fall back only for a workflow with no such constants, and even then only
    # where the latent's size is genuinely a literal.
    if not (set_w and set_h):
        for v in graph.values():
            if v["class_type"] == "EmptyLTXVLatentVideo":
                set_w |= _set_if_literal(v, "width", width)
                set_h |= _set_if_literal(v, "height", height)
    if not (set_w and set_h):
        raise ComfyError(
            "workflow exposes no settable width/height — it has neither titled "
            "INTConstants nor a literal EmptyLTXVLatentVideo size")


def set_guides(graph: dict, guides: list[dict]) -> None:
    """Replace the workflow's image conditioning with our keyframes.

    `guides` is [{name, index, strength}] where name is a file already uploaded
    to ComfyUI. Handles both shapes a workflow might use: an existing
    LTXVAddGuideMulti (just refill it), or single-image LTXVImgToVideoInplace
    nodes (swap each for a guide node and rewire).
    """
    existing = _find(graph, GUIDE_NODE)
    inplace = _find(graph, "LTXVImgToVideoInplace")
    if not existing and not inplace:
        raise ComfyError("workflow has no image conditioning to replace")

    # Repoint any LoadImage the workflow already had at our first keyframe.
    #
    # Not cosmetic. In this workflow the original LoadImage is load-bearing for
    # SIZE, not only conditioning:
    #
    #   LoadImage -> ImageResizeKJv2 -> ResizeImageMaskNode x0.5
    #             -> GetImageSize -> EmptyLTXVLatentVideo
    #
    # Leaving it pointed at the author's sample file fails validation outright
    # ("Invalid image file: 1757644411.webp") because that file does not exist
    # here — and if it did, the clip would be sized from the wrong image.
    for nid in _find(graph, "LoadImage"):
        if not nid.startswith("7"):
            graph[nid]["inputs"]["image"] = guides[0]["name"]

    loaders = {}
    for i, g in enumerate(guides, start=1):
        nid = f"7{i:03d}"
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": g["name"]},
                      "_meta": {"title": f"keyframe {i}"}}
        loaders[i] = nid

    def fill(node_id: str) -> None:
        inputs = graph[node_id]["inputs"]
        for k in [k for k in inputs if k.startswith("num_guides")]:
            del inputs[k]
        inputs["num_guides"] = str(len(guides))
        for i, g in enumerate(guides, start=1):
            inputs[f"num_guides.image_{i}"] = [loaders[i], 0]
            inputs[f"num_guides.frame_idx_{i}"] = g["index"]
            inputs[f"num_guides.strength_{i}"] = g["strength"]

    if existing:
        for nid in existing:
            fill(nid)
        _crop_guides(graph)
        return

    # Swap each LTXVImgToVideoInplace (latent -> latent) for a guide node
    # (conditioning + latent -> conditioning + latent). Not a drop-in: the
    # conditioning has to route through it as well.
    cond = _one(graph, "LTXVConditioning")
    for si, old_id in enumerate(inplace, start=1):
        gid = f"79{si:02d}"
        graph[gid] = {"class_type": GUIDE_NODE, "_meta": {"title": f"guides stage {si}"},
                      "inputs": {"positive": [cond, 0], "negative": [cond, 1],
                                 "vae": graph[old_id]["inputs"]["vae"],
                                 "latent": graph[old_id]["inputs"]["latent"]}}
        fill(gid)
        for k, v in graph.items():
            if k == gid:
                continue
            for name, val in list(v["inputs"].items()):
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == old_id:
                    v["inputs"][name] = [gid, 2]        # latent is output 2
        del graph[old_id]

    _route_guiders(graph)
    _crop_guides(graph)


def _crop_guides(graph: dict) -> None:
    """Strip guide tokens back out of the latent -- once per sampler stage.

    LTXVAddGuideMulti APPENDS each guide to the latent sequence; LTXVCropGuides
    removes them again. The base workflow needed neither, because
    LTXVImgToVideoInplace writes into the existing latent. Swapping in the guide
    node without the crop leaves those tokens in the sequence and they decode as
    extra frames: a 121-frame request came back 137, ending on a held,
    re-encoded copy of the keyframe.

    ONE CROP PER STAGE, not one per graph. Read off the catalogue: of the
    workflows that use LTXVAddGuide*, the dominant shape is 2 guides / 2 crops /
    2 samplers, and even the single-guide two-stage ones carry two crops. A
    single crop before the decode removes stage 2's tokens and leaves stage 1's,
    which is how 137 frames became 129 instead of 121.

    The conditioning matters too. In FML2V the stage-1 crop's positive/negative
    feed the stage-2 guide, so each stage guides against conditioning that has
    already had the previous stage's guides taken off it.
    """
    stages = stage_samplers(graph)
    guiders = stage_guiders(graph)
    crop_for: dict[int, str] = {}

    for stage in (1, 2):
        src = graph[guiders[stage]]["inputs"].get("positive")
        if not (isinstance(src, list)
                and graph.get(str(src[0]), {}).get("class_type") == GUIDE_NODE):
            continue
        guide = str(src[0])

        # The video latent coming out of this stage.
        sep = next((k for k, v in graph.items()
                    if v["class_type"] == "LTXVSeparateAVLatent"
                    and isinstance(v["inputs"].get("av_latent"), list)
                    and str(v["inputs"]["av_latent"][0]) == stages[stage]), None)
        if sep is None:
            raise ComfyError(f"stage {stage} has no LTXVSeparateAVLatent to crop after")

        existing = next((k for k, v in graph.items()
                         if v["class_type"] == "LTXVCropGuides"
                         and isinstance(v["inputs"].get("latent"), list)
                         and str(v["inputs"]["latent"][0]) == sep), None)
        if existing:
            graph[existing]["inputs"]["positive"] = [guide, 0]
            graph[existing]["inputs"]["negative"] = [guide, 1]
            crop_for[stage] = existing
            continue

        nid = _new_id(graph, "73")
        graph[nid] = {"class_type": "LTXVCropGuides",
                      "_meta": {"title": f"crop guide tokens (stage {stage})"},
                      "inputs": {"positive": [guide, 0], "negative": [guide, 1],
                                 "latent": [sep, 0]}}
        # Everything that consumed this stage's VIDEO latent now takes the
        # cropped one. Output 1 is audio and must keep coming from the split.
        for k, v in graph.items():
            if k == nid:
                continue
            for name, val in list(v["inputs"].items()):
                if isinstance(val, list) and len(val) == 2 \
                        and str(val[0]) == sep and val[1] == 0:
                    v["inputs"][name] = [nid, 2]
        crop_for[stage] = nid

    # Stage 2 guides against conditioning stage 1's crop has already cleaned.
    if 1 in crop_for and 2 in guiders:
        nxt = graph[guiders[2]]["inputs"].get("positive")
        if isinstance(nxt, list) and graph.get(str(nxt[0]), {}).get("class_type") == GUIDE_NODE:
            g2 = str(nxt[0])
            graph[g2]["inputs"]["positive"] = [crop_for[1], 0]
            graph[g2]["inputs"]["negative"] = [crop_for[1], 1]


def _route_guiders(graph: dict) -> None:
    """Point each CFGGuider at the guide node feeding its sampler's latent.

    Without this the guides are encoded into the latent but the conditioning
    never learns about them, so they do not steer.
    """
    def guide_behind(node_id, depth=0):
        if depth > 5:
            return None
        v = graph.get(str(node_id))
        if not v:
            return None
        if v["class_type"] == GUIDE_NODE:
            return str(node_id)
        for name, val in v["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and "latent" in name.lower():
                if (r := guide_behind(val[0], depth + 1)):
                    return r
        return None

    for v in graph.values():
        if v["class_type"] != "SamplerCustomAdvanced":
            continue
        gid = v["inputs"].get("guider", [None])[0]
        guide = guide_behind(v["inputs"].get("latent_image", [None])[0])
        if gid and guide:
            graph[str(gid)]["inputs"]["positive"] = [guide, 0]
            graph[str(gid)]["inputs"]["negative"] = [guide, 1]


# --- stages ---------------------------------------------------------------
#
# Everything below exists because the two passes are not interchangeable. Stage 1
# generates at half size from noise; stage 2 refines the 2x-upscaled latent
# starting at 0.85 rather than 1.0. The base workflow nonetheless runs ONE model
# chain into both CFGGuiders and a fixed sigma table into each sampler, so
# neither the step count nor a LoRA's strength could differ between them.
#
# The community dev workflows do differ: `Dev_Full-Steps` applies the distilled
# LoRA at 0.3 on stage 1 and 0.6 on stage 2, and schedules 20 steps on stage 1
# against the same 3-step table on stage 2 that every other workflow uses.


def _model_source(graph: dict, node_id: str) -> str | None:
    """The node feeding `node_id`'s MODEL input, or None when it has none."""
    v = graph.get(str(node_id), {}).get("inputs", {}).get("model")
    return str(v[0]) if isinstance(v, list) and len(v) == 2 else None


def _feeds_from(graph: dict, node_id: str, class_type: str, depth: int = 0) -> bool:
    """Whether `class_type` is upstream of `node_id` along LATENT inputs."""
    if depth > 8 or str(node_id) not in graph:
        return False
    v = graph[str(node_id)]
    if v["class_type"] == class_type:
        return True
    return any(_feeds_from(graph, val[0], class_type, depth + 1)
               for name, val in v["inputs"].items()
               if isinstance(val, list) and len(val) == 2 and "latent" in name.lower())


def stage_samplers(graph: dict) -> dict[int, str]:
    """{1: sampler_id, 2: sampler_id}, told apart by what feeds their latent.

    Stage 2 is the one denoising an upscaled latent. Never told apart by node id
    or by declaration order: the base workflow happens to declare stage 1 as 113
    and stage 2 as 119, and the FLF2V workflow declares the same roles as 13 and
    21. An id is an editor accident and re-exporting renumbers it.
    """
    ids = _find(graph, "SamplerCustomAdvanced")
    if len(ids) != 2:
        raise ComfyError(f"expected a two-stage graph, found {len(ids)} samplers")
    upscaled = [i for i in ids
                if _feeds_from(graph, i, "LTXVLatentUpsampler")]
    if len(upscaled) != 1:
        raise ComfyError("could not tell the sampler stages apart by their latents")
    return {1: next(i for i in ids if i not in upscaled), 2: upscaled[0]}


def stage_guiders(graph: dict) -> dict[int, str]:
    return {stage: str(graph[sid]["inputs"]["guider"][0])
            for stage, sid in stage_samplers(graph).items()}


def _new_id(graph: dict, prefix: str) -> str:
    n = 1
    while f"{prefix}{n:02d}" in graph:
        n += 1
    return f"{prefix}{n:02d}"


def split_model_chains(graph: dict) -> dict[int, str]:
    """Give each stage its own model chain off the shared checkpoint.

    Returns {stage: guider_id}. Idempotent, and NOT done unless something
    actually differs per stage -- an unsplit graph is the one that produced every
    verified render, and duplicating a chain that carries identical strengths
    would change the graph for no gain.

    The checkpoint itself is never duplicated. ComfyUI's ModelPatcher clones
    share the underlying weights and differ only in their patch list, so a second
    chain costs patch bookkeeping rather than another 46 GB.
    """
    guiders = stage_guiders(graph)
    src_1, src_2 = (_model_source(graph, guiders[1]), _model_source(graph, guiders[2]))
    if src_1 is None or src_2 is None:
        raise ComfyError("a stage guider has no model input to split")
    if src_1 != src_2:
        return guiders                                  # already split

    done: dict[str, str] = {}

    def clone(nid: str) -> str:
        """Copy a chain node, recursing up. Stops at the node with no MODEL
        input -- the checkpoint loader, which both stages keep sharing."""
        upstream = _model_source(graph, nid)
        if upstream is None:
            return nid
        if nid in done:
            return done[nid]
        new = _new_id(graph, "76")
        node = copy.deepcopy(graph[nid])
        title = str((node.get("_meta") or {}).get("title") or node["class_type"])
        node["_meta"] = {"title": f"{title} (stage 2)"}
        graph[new] = node
        done[nid] = new                     # before recursing, so cycles terminate
        node["inputs"]["model"] = [clone(upstream), 0]
        return new

    graph[guiders[2]]["inputs"]["model"] = [clone(src_2), 0]
    return guiders


def set_steps(graph: dict, stage_1: int | None, stage_2: int | None) -> None:
    """Step count per pass. `None` leaves that stage as the workflow authored it.

    The workflow's schedules are explicit `ManualSigmas` tables -- 8 entries plus
    a terminal zero on stage 1, 3 plus zero on stage 2 -- and those tables are
    shared byte-for-byte across every distilled workflow in the RuneXX catalogue
    (stage 2's is identical in all eight, the dev ones included). They are a
    fixed distilled schedule, not a parametric family, so a different count
    cannot mean "resample the same curve". It means the other regime the
    catalogue uses: BasicScheduler on linear_quadratic, which is what
    Dev_Full-Steps runs at 20 and DEV_Experimental_3-Pass at 35.

    So asking for the count a table already gives keeps the verified table, and
    any other count switches scheduler. The `denoise` handed to the scheduler is
    read off the table's FIRST sigma rather than assumed: stage 1 starts at 1.0
    because it denoises from noise, stage 2 at 0.85 because it is refining an
    already-upscaled latent. Hardcoding 1.0 there would re-noise stage 2 to
    scratch and throw away everything stage 1 produced.

    Stage 1's count is softer than it reads. Its first four steps each move sigma
    by 0.00625 -- 2.5% of the trajectory for half the budget -- so the eight is
    really four warm-up steps and four that do the work.
    """
    samplers = stage_samplers(graph)
    for stage, steps in ((1, stage_1), (2, stage_2)):
        if steps is None:
            continue
        sid = samplers[stage]
        src = graph[sid]["inputs"].get("sigmas")
        if not (isinstance(src, list) and len(src) == 2):
            raise ComfyError(f"stage {stage} sampler has a literal sigmas input")
        node = graph[str(src[0])]
        if node["class_type"] in ("BasicScheduler", "LTXVScheduler"):
            node["inputs"]["steps"] = steps         # already parametric
            continue
        if node["class_type"] != "ManualSigmas":
            raise ComfyError(f"stage {stage} sigmas come from {node['class_type']}, "
                             f"which exposes no step count")
        table = [float(x) for x in str(node["inputs"]["sigmas"]).split(",") if x.strip()]
        if len(table) - 1 == steps:
            continue                                # the table already is this
        new = _new_id(graph, "75")
        graph[new] = {
            "class_type": "BasicScheduler",
            "_meta": {"title": f"{SCHEDULER} x{steps} (stage {stage})"},
            "inputs": {"model": [_model_source(graph, stage_guiders(graph)[stage]), 0],
                       "scheduler": SCHEDULER, "steps": steps,
                       "denoise": round(table[0], 5)}}
        graph[sid]["inputs"]["sigmas"] = [new, 0]


def _chain_anchor(graph: dict, guider: str) -> str:
    """Where a content LoRA attaches on the chain feeding `guider`.

    The Power Lora Loader if this chain has one, otherwise its last
    LoraLoaderModelOnly. Found by walking the chain rather than by scanning the
    whole graph, because once the stages are split there are two of each and
    picking the first match would put both stages' LoRAs on stage 1's chain.
    """
    node = _model_source(graph, guider)
    fallback = None
    while node is not None:
        cls = graph[node]["class_type"]
        if cls == "Power Lora Loader (rgthree)":
            return node
        if cls == "LoraLoaderModelOnly" and fallback is None:
            fallback = node
        node = _model_source(graph, node)
    if fallback is None:
        raise ComfyError("workflow has nowhere to attach a content LoRA")
    return fallback


def _clear_content_loras(graph: dict) -> None:
    """Drop LoRA nodes a previous patch spliced in, rewiring past each.

    A graph that had already been patched once got the content LoRA spliced
    twice and applied at double strength -- invisible unless you count the nodes.
    """
    for k in [k for k in graph if k.startswith(("77", "78"))]:
        stale = graph.pop(k)
        upstream = stale["inputs"].get("model")
        for v in graph.values():
            for name, val in list(v["inputs"].items()):
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == k:
                    v["inputs"][name] = upstream


def _splice_loras(graph: dict, anchor: str, loras: list[dict], prefix: str) -> None:
    """Chain LoraLoaderModelOnly nodes in after `anchor`, repointing consumers."""
    prev = anchor
    for i, lo in enumerate(loras, start=1):
        nid = f"{prefix}{i:02d}"
        graph[nid] = {"class_type": "LoraLoaderModelOnly",
                      "_meta": {"title": f"content LoRA {i} @{lo['strength']}"},
                      "inputs": {"lora_name": lo["name"],
                                 "strength_model": lo["strength"],
                                 "model": [prev, 0]}}
        prev = nid
    for k, v in graph.items():
        if k.startswith(prefix):
            continue
        for name, val in list(v["inputs"].items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == anchor:
                v["inputs"][name] = [prev, val[1]]


def _stage_strength(lo: dict, stage: int) -> float:
    """A LoRA's strength on one pass, falling back to its single value."""
    per = lo.get(f"strength_stage_{stage}")
    return lo["strength"] if per is None else per


def describe_stages(graph: dict) -> list[dict]:
    """What each pass will actually run, read off the finished graph.

    Read back rather than echoed from the request, because the graph is the only
    thing that renders. A step count the workflow could not express, or a LoRA
    that landed on one chain when it was meant for both, is invisible in a
    request echo and obvious here.
    """
    out = []
    for stage, sid in sorted(stage_samplers(graph).items()):
        src = graph[sid]["inputs"].get("sigmas")
        node = graph[str(src[0])] if isinstance(src, list) else None
        if node is None:
            steps, schedule = None, "literal sigmas"
        elif node["class_type"] == "ManualSigmas":
            table = [x for x in str(node["inputs"]["sigmas"]).split(",") if x.strip()]
            steps, schedule = len(table) - 1, "distilled sigma table"
        else:
            steps = node["inputs"].get("steps")
            schedule = f"{node['inputs'].get('scheduler', node['class_type'])} scheduler"

        loras, seen = [], set()
        chain = _model_source(graph, stage_guiders(graph)[stage])
        while chain is not None and chain not in seen:
            seen.add(chain)
            v = graph[chain]
            if v["class_type"] == "LoraLoaderModelOnly":
                loras.append({"name": v["inputs"]["lora_name"],
                              "strength": v["inputs"]["strength_model"]})
            chain = _model_source(graph, chain)
        guider = graph.get(stage_guiders(graph)[stage], {})
        gi = guider.get("inputs", {})
        if guider.get("class_type") == "MultimodalGuider":
            src = gi.get("parameters")
            gp = graph.get(str(src[0]), {}).get("inputs", {}) if isinstance(src, list) else {}
            cfg = gp.get("cfg")
            guidance = (f"multimodal cfg {gp.get('cfg')} stg {gp.get('stg')} "
                        f"rescale {gp.get('rescale')} blocks {gi.get('skip_blocks')}")
        else:
            cfg, guidance = gi.get("cfg"), "cfg"
        out.append({"stage": stage, "steps": steps, "schedule": schedule,
                    "guidance": guidance,
                    "cfg": cfg,
                    "sampler": graph[str(graph[sid]["inputs"]["sampler"][0])]
                                    ["inputs"].get("sampler_name"),
                    "loras": list(reversed(loras))})
    return out


# Guidance parameters from RuneXX's LTX-2.3 Dev Full-Steps and DEV 3-Pass, read
# off their GuiderParameters widgets. Order is
# [modality, cfg, stg, perturb_attn, rescale, modality_scale, skip_step, cross_attn].
DEV_VIDEO_GUIDANCE = dict(cfg=3.0, stg=1.0, rescale=0.9, modality_scale=3.0)
DEV_AUDIO_GUIDANCE = dict(cfg=7.0, stg=1.0, rescale=0.7, modality_scale=3.0)
DEV_SKIP_BLOCKS = "28"


def set_multimodal_guidance(graph: dict, video: dict | None = None,
                            audio: dict | None = None,
                            skip_blocks: str = DEV_SKIP_BLOCKS) -> dict:
    """Give stage 1 the guidance stack the dev workflows actually use.

    The base workflow drives both passes with a plain `CFGGuider`, which is the
    distilled recipe. Every dev-checkpoint workflow in the catalogue does
    something else on the BASE pass: a `MultimodalGuider` fed by two
    `GuiderParameters`, carrying three things a CFGGuider has no way to express —
    spatiotemporal guidance, CFG rescale, and a separate guidance scale for
    audio (cfg 7 against video's 3).

    Stage 2 is deliberately untouched. Those same workflows leave the refine pass
    on `CFGGuider` at cfg 1, and Z6 confirmed it here by going wild at cfg 2.

    `skip_blocks` is the STG block list, "28" in both reference workflows.
    Returns what it applied, for reporting.
    """
    v = {**DEV_VIDEO_GUIDANCE, **(video or {})}
    a = {**DEV_AUDIO_GUIDANCE, **(audio or {})}
    guiders = stage_guiders(graph)
    g1 = graph[guiders[1]]
    if g1["class_type"] == "MultimodalGuider":
        return {"video": v, "audio": a, "skip_blocks": skip_blocks}   # idempotent
    if g1["class_type"] != "CFGGuider":
        raise ComfyError(f"stage 1 guider is {g1['class_type']}, not a CFGGuider")

    def params(nid: str, modality: str, cfgvals: dict, chain: str | None) -> str:
        inputs = {"modality": modality, "cfg": cfgvals["cfg"], "stg": cfgvals["stg"],
                  "perturb_attn": True, "rescale": cfgvals["rescale"],
                  "modality_scale": cfgvals["modality_scale"], "skip_step": 0,
                  "cross_attn": True}
        if chain:
            inputs["parameters"] = [chain, 0]
        graph[nid] = {"class_type": "GuiderParameters", "inputs": inputs,
                      "_meta": {"title": f"{modality} guidance"}}
        return nid

    audio_id = params(_new_id(graph, "74"), "AUDIO", a, None)
    video_id = params(_new_id(graph, "74"), "VIDEO", v, audio_id)

    # Replace stage 1's guider in place: same id, so every consumer stays wired.
    graph[guiders[1]] = {
        "class_type": "MultimodalGuider",
        "_meta": {"title": "multimodal guidance (stage 1)"},
        "inputs": {"model": g1["inputs"]["model"],
                   "positive": g1["inputs"]["positive"],
                   "negative": g1["inputs"]["negative"],
                   "parameters": [video_id, 0],
                   "skip_blocks": skip_blocks}}
    return {"video": v, "audio": a, "skip_blocks": skip_blocks}


def set_loras(graph: dict, loras: list[dict]) -> None:
    """Splice content LoRAs in after the Power Lora Loader.

    rgthree's Power Lora Loader holds its LoRAs in dynamic widgets the API
    schema does not declare, so it cannot be filled over /prompt — and shared
    workflows ship it empty. An explicit LoraLoaderModelOnly chain is the only
    way in.

    Each LoRA may carry `strength_stage_1` / `strength_stage_2`. When any of them
    differ the model chain is split so each CFGGuider gets its own, which is what
    makes a per-stage strength mean anything: the base workflow runs ONE chain
    into both guiders, so a single set of patched weights is what both passes
    sample from no matter how the strengths are labelled.

    When nothing differs the graph is left single-chained, byte-identical to what
    every verified render used. A per-stage capability is not a reason to change
    the shape of jobs that never asked for it.
    """
    if not loras:
        return
    _clear_content_loras(graph)

    if not any(_stage_strength(lo, 1) != _stage_strength(lo, 2) for lo in loras):
        anchor = (_find(graph, "Power Lora Loader (rgthree)")
                  or _find(graph, "LoraLoaderModelOnly"))
        if not anchor:
            raise ComfyError("workflow has nowhere to attach a content LoRA")
        _splice_loras(graph, anchor[0],
                      [{"name": lo["name"], "strength": _stage_strength(lo, 1)}
                       for lo in loras], "78")
        return

    guiders = split_model_chains(graph)
    for stage, prefix in ((1, "78"), (2, "77")):
        _splice_loras(graph, _chain_anchor(graph, guiders[stage]),
                      [{"name": lo["name"], "strength": _stage_strength(lo, stage)}
                       for lo in loras], prefix)


def set_checkpoint(graph: dict, name: str) -> None:
    """Swap the base model.

    LTX 2.3 ships MONOLITHS -- one file carrying the transformer, both VAEs and
    the text projection -- which is why `CheckpointLoaderSimple`,
    `LTXAVTextEncoderLoader` and `LTXVAudioVAELoader` all name the SAME file.
    Changing one and not the others gives a graph that still validates and still
    renders, with one model's transformer decoded by another model's VAE.

    So every loader currently naming the checkpoint moves together, matched on
    the value rather than the class: a workflow may load it under a node type
    this does not know about, and matching on value follows the graph instead of
    a list of class names that has to be kept up to date.

    `text_encoder` on LTXAVTextEncoderLoader is deliberately untouched -- that is
    the separate Gemma file, not the monolith.
    """
    current = graph[_one(graph, "CheckpointLoaderSimple")]["inputs"].get("ckpt_name")
    moved = 0
    for v in graph.values():
        if v["inputs"].get("ckpt_name") == current:
            v["inputs"]["ckpt_name"] = name
            moved += 1
    if not moved:
        raise ComfyError("workflow has no checkpoint loader to repoint")


def set_seed(graph: dict, seed: int) -> None:
    for v in graph.values():
        for field in ("noise_seed", "seed"):
            if field in v["inputs"] and not isinstance(v["inputs"][field], list):
                v["inputs"][field] = seed


def achievable_frames(graph: dict, frames: int, fps: int = 24) -> int:
    """The frame count this workflow can actually produce, nearest the request.

    When length is computed from an INTEGER seconds constant the reachable counts
    are `1 + fps*s`, not the `1+8k` grid the request may use. Asking for 81
    frames yields 73, and a guide placed at index 81 would then sit past the end
    of the clip — the request has to be reconciled BEFORE placement is planned,
    not after.
    """
    for v in graph.values():
        if v["class_type"] == "INTConstant":
            title = str((v.get("_meta") or {}).get("title", "")).upper()
            if "LENGTH" in title and "SECOND" in title:
                return 1 + fps * max(1, round((frames - 1) / fps))
    return frames          # workflow takes a frame count directly


def set_frames(graph: dict, frames: int, fps: int = 24) -> None:
    """Clip length, expressed however the workflow expresses it.

    This one computes frames from a duration in SECONDS:

        length = 1 + 8*(round(seconds * fps)/8)

    which is the 1+8k causal-VAE grid arrived at from the other direction. The
    latent's `length` is therefore a link, and setting it directly does nothing
    useful while severing the calculation — so the seconds constant is what has
    to move. Missing this meant every render came out at the workflow's
    hardcoded 10s no matter what the job asked for.

    A workflow that hardcodes a frame count instead gets it set directly.
    """
    seconds = max(1, round((frames - 1) / fps))
    set_any = False
    for v in graph.values():
        if v["class_type"] == "INTConstant":
            title = str((v.get("_meta") or {}).get("title", "")).upper()
            if "LENGTH" in title and "SECOND" in title:
                set_any |= _set_if_literal(v, "value", seconds)
    for v in graph.values():
        if v["class_type"] == "EmptyLTXVLatentVideo":
            set_any |= _set_if_literal(v, "length", frames)
        if v["class_type"] == "LTXVEmptyLatentAudio":
            _set_if_literal(v, "frames_number", frames)
    if not set_any:
        raise ComfyError("workflow exposes no settable clip length")


def load_workflow(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def decode_data_uri(src: str) -> bytes:
    if src.startswith("data:"):
        return base64.b64decode(re.sub(r"^data:[^,]+,", "", src))
    if src.startswith(("http://", "https://")):
        return requests.get(src, timeout=120).content
    raise ComfyError("images must be data URIs or http(s) URLs")
