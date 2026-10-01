"""Recipe resolution: a resolved (character, pose) configuration -> a ComfyUI graph.

NOTHING IS LOOKED UP HERE. Every value arrives in the request.

This module used to read recipes/recipes.json, generated from an ODS sheet, and resolve a
recipe BY NAME. That was the last of the spreadsheet, and it outlived the migration that made
recipes database rows (wanly-api#212): the API and console became DB-native while the engine
kept its own eight-name file, so a pose authored in the console rendered as

    KeyError: "unknown recipe 'Doggystyle Side v2'"

and — worse and quieter — editing a SEEDED pose changed nothing, because the engine read its
own frozen copy of the prompt instead of the one the user saved.

The rule this restores is wanly-api#207: an engine that cannot look a recipe up cannot look up
a STALE one. The daemon already sends the resolved configuration verbatim ("it is read, never
looked up"), so the file was supplying defaults for fields the caller always provides.

What is NOT decided here any more: whether a render is "as validated". That was a comparison
against the sheet's baseline. Validation is a property of the pose row now, which the API and
console own.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path

HERE = Path(__file__).parent
RECIPE_WORKFLOW = "ltx23_recipe.api.json"

# The base model a render falls back to when the caller names none (console#431).
#
# THREE PLACES ANSWER THIS AND ALL THREE MUST AGREE:
#
#   * here -- what actually renders
#   * download_models.sh's _WANTED -- what a cold container or pod actually HAS
#   * wanly-api's LTX_STACK['checkpoint'] -- what the API gates claims against
#
# They live in two repos and cannot share a constant, so the coupling is held by
# test_default_checkpoint_is_the_one_a_cold_pod_fetches() below, which reads the shell
# script. A comment would not have been enough: the failure is silent. A pod that fetched
# a different checkpoint than the default reports it through the heartbeat, the API's model
# gate then hides every default pose from it, and it claims nothing at all -- which looks
# like an empty queue rather than like a broken pod.
DEFAULT_CHECKPOINT = "10Eros_v1.5_bf16"


def _is_none(name: str | None) -> bool:
    """Is this the caller saying "no LoRA"?

    Compared with the extension STRIPPED, which is the whole point. A bare "none" was always
    excluded, but "none.safetensors" was not — and that is exactly what arrives once anything
    upstream normalises the name before sending it. ComfyUI then rejects the graph with

        9621 LoraLoaderModelOnly: lora_name: 'none.safetensors' not in [...]

    ten minutes into a claimed segment. Seen in production 2026-09-04.

    The daemon also filters this, and should. This is the last line: the engine builds the
    graph, so it is the thing that must never build a loader for a file called "none".
    """
    n = (name or "").strip().lower()
    if n.endswith(".safetensors"):
        n = n[: -len(".safetensors")]
    return n in ("", "none")


#: Node-id prefixes for the character LoRA pairs, one per slot: slot 0 is 9621/9622 (so
#: every existing single-character graph hashes exactly as it did), slot 1 is 9631/9632.
#: Content ids run 9601..9608 and stop well short of either. A third person is one more
#: entry here -- and a second `<TRIGGER>` slot everywhere upstream (console#473).
CHAR_NODE_IDS = ("962", "963")


def resolve(graph: dict, image_name: str, width: int, height: int, *,
            prompt: str, negative: str | None = None, checkpoint: str | None = None,
            char_lora: str | None = None, char_s1: float = 0.8, char_s2: float = 1.5,
            char_loras: list | None = None,
            content_loras: list | None = None,
            img_compression: int | None = None,
            text_to_video: bool = False) -> dict:
    """Patch the validated graph with this render's configuration.

    Values only, never topology — the graph template is the validated recipe and this moves
    the handful of fields that vary between renders.

    `char_loras` is the list form -- `[{name, s1, s2}, ...]`, up to `len(CHAR_NODE_IDS)`,
    in slot order -- for a shot with two people in it (console#473). `char_lora/char_s1/
    char_s2` remain as the one-character shorthand every existing caller uses, and produce
    the identical graph: a single character is `char_loras=[that one]`.

    `text_to_video` (#145) renders with NO start frame: node 290, the graph's own "Text To
    Video (no image ref)" switch, bypasses both LTXVImgToVideoInplace nodes (160/161), so
    nothing conditions on `image_name`. The LoadImage at 167 stays -- the graph derives its
    latent size from that image resized to 292/293, so the caller hands it a blank frame of
    exactly width x height and the size is the requested one. Still values only: the switch
    is a value the validated graph already carries, which is why this needs no new topology.
    """
    g = json.loads(json.dumps(graph))
    ck = checkpoint or DEFAULT_CHECKPOINT
    if not ck.endswith(".safetensors"):
        ck += ".safetensors"
    # 2.3 checkpoints are monoliths: every loader naming the file must move
    for nid in ("9500", "9501", "9502"):
        if nid in g and "ckpt_name" in g[nid].get("inputs", {}):
            g[nid]["inputs"]["ckpt_name"] = ck
    g["167"]["inputs"]["image"] = image_name
    g["292"]["inputs"]["value"] = int(width)
    g["293"]["inputs"]["value"] = int(height)
    # WRITTEN EITHER WAY, not only when True. The template ships false, so an image render
    # hashes exactly as it did; but a template re-exported with the switch left on would
    # otherwise turn every image-conditioned render into text-to-video, silently -- the start
    # frame would be uploaded, logged and ignored.
    g["290"]["inputs"]["value"] = bool(text_to_video)
    # Conditioning-frame CRF. `is not None` rather than truthiness: 0 is a real setting that
    # bypasses the encode, and `if img_compression:` would silently ignore it.
    if img_compression is not None:
        for v in g.values():
            if isinstance(v, dict) and v.get("class_type") == "LTXVPreprocess":
                v["inputs"]["img_compression"] = int(img_compression)

    g["121"]["inputs"]["text"] = prompt
    if negative:
        g["110"]["inputs"]["text"] = negative

    # one content+character chain per stage branch, mirroring how the distill
    # LoRA is already wired at 361/362
    #
    # Content LoRAs STACK (console#410): motion, act and framing are separable and a pose
    # may want several. They are applied in the order given — that order is part of the
    # configuration, not incidental, and two poses with the same LoRAs in a different order
    # will render differently.
    contents = []
    for entry in (content_loras or []):
        name = str(entry.get("name") or "").strip()
        if _is_none(name):
            # "none" is how a pose says off. Looking it up would be a filename lookup for a
            # file that does not exist.
            continue
        contents.append({
            "name": name if name.endswith(".safetensors") else name + ".safetensors",
            # 0.6 matches what this function hardcoded before any of it was configurable, so
            # an entry that names a LoRA and nothing else renders at the validated strength.
            # `is None` rather than `or`: 0 is a real setting — the LoRA loads and
            # contributes nothing, which is how you measure what it was contributing.
            "s1": float(entry["s1"]) if entry.get("s1") is not None else 0.6,
            "s2": float(entry["s2"]) if entry.get("s2") is not None else 0.6,
        })
    # The template ships with a baked content LoRA at 9601 (DR34ML4Y). Remove it before
    # building the chain, or it would sit in front of everything below.
    if "9601" in g:
        del g["9601"]
    # A character LoRA is optional. "none" renders the recipe on the checkpoint
    # alone -- useful for judging what the LoRA is actually contributing, and
    # for a shot where the start frame already carries the identity. In the list form a
    # "none" in either slot is skipped the same way, so a two-person pose can still be
    # rendered with one identity to see what the other was contributing.
    if char_loras is None:
        char_loras = [{"name": char_lora, "s1": char_s1, "s2": char_s2}]
    chars = []
    for entry in char_loras:
        name = str(entry.get("name") or "").strip()
        if _is_none(name):
            continue
        chars.append({
            "name": name if name.endswith(".safetensors") else name + ".safetensors",
            "s1": float(entry["s1"]) if entry.get("s1") is not None else 0.8,
            "s2": float(entry["s2"]) if entry.get("s2") is not None else 1.5,
        })
    if len(chars) > len(CHAR_NODE_IDS):
        raise ValueError(
            f"{len(chars)} character LoRAs; the recipe graph has room for "
            f"{len(CHAR_NODE_IDS)} (console#473)")
    # Per stage, like the character strengths beside them. This was 0.6 hardcoded for BOTH
    # stages, which is a configuration rather than a default -- stage 1 generates at half
    # size from noise and stage 2 refines the 2x-upscaled latent, so one number for both is
    # a different setup, not a simpler one. 0.6/0.6 remains the default so a caller that
    # says nothing gets exactly the graph that was validated.
    for tag, branch in {"1": "337", "2": "372"}.items():
        prev = ["301", 0]
        # Node ids: 9601/9602 for the first content LoRA (unchanged, so a single-LoRA pose
        # produces the same graph it always did), then 9603/9604, 9605/9606... Stops well
        # short of the character pair at 9621/9622 even at the cap of 4, so the two chains
        # can never collide.
        for i, c in enumerate(contents):
            cid = f"96{1 + i * 2:02d}" if tag == "1" else f"96{2 + i * 2:02d}"
            g[cid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"lora_name": c["name"],
                                 "strength_model": c["s1"] if tag == "1" else c["s2"],
                                 "model": prev},
                      # Unnumbered when there is only one, and that is not cosmetic
                      # fussiness: graph_hash includes _meta, so numbering a lone LoRA
                      # "content 1" would change the hash of every existing single-LoRA
                      # pose. The hash is the regression trail — it is what proves a render
                      # is the configuration that was signed off — and a relabel must not
                      # look like a configuration change. Verified: single-LoRA poses hash
                      # identically before and after this change.
                      "_meta": {"title": f"content {i + 1} stage {tag}" if len(contents) > 1
                                else f"content stage {tag}"}}
            prev = [cid, 0]
        # Character LoRAs LAST, closest to the sampler, one pair per person. Slot 0 keeps
        # its id and its unnumbered title so a one-person graph hashes as it always has --
        # the hash is the regression trail. Slot 1 is `char 2`, and reads off slot 0.
        for i, c in enumerate(chars):
            kid = f"{CHAR_NODE_IDS[i]}{tag}"
            g[kid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"lora_name": c["name"],
                                 "strength_model": c["s1"] if tag == "1" else c["s2"],
                                 "model": prev},
                      "_meta": {"title": f"char stage {tag}" if i == 0
                                else f"char {i + 1} stage {tag}"}}
            prev = [kid, 0]
        g[branch]["inputs"]["model"] = prev
    return g


def graph_hash(g: dict) -> str:
    """Tier-1 regression hash. Excludes the start image and output name so the
    hash tracks the RECIPE, not which fixture it happened to run against."""
    h = json.loads(json.dumps(g))
    h["167"]["inputs"]["image"] = "<fixture>"
    h["140"]["inputs"]["filename_prefix"] = "<out>"
    # The seed is a draw, not a configuration. Two renders of the same recipe at
    # different seeds are both "as validated"; only a changed PARAMETER should
    # move the hash.
    for v in h.values():
        for field in ("noise_seed", "seed"):
            if field in v.get("inputs", {}) and not isinstance(v["inputs"][field], list):
                v["inputs"][field] = "<seed>"
    return hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()


def base_model_note(graph: dict) -> str:
    """Which checkpoint this graph will actually load.

    Read from the RESOLVED GRAPH rather than the request, for the same reason
    lora_stack_note is: the request is what was asked for, the graph is what will render,
    and those differ exactly when something has gone wrong.

    2.3 checkpoints are monoliths, so every loader names the same file — 9500 is the one the
    others follow.
    """
    for nid in ("9500", "9501", "9502"):
        n = graph.get(nid)
        if n and n.get("inputs", {}).get("ckpt_name"):
            name = n["inputs"]["ckpt_name"]
            return name[: -len(".safetensors")] if name.endswith(".safetensors") else name
    return "unknown"


def lora_stack_note(graph: dict) -> str:
    """Which LoRAs this graph actually loads, per stage, as a one-line proof.

    Built by inspecting the resolved graph's LoraLoaderModelOnly nodes rather than the
    request that produced them, so the line is evidence of what will render. The node ids
    are the ones recipe.resolve() writes: 9601/9602 content, 9621/9622 character.

    Both stages are printed only when they differ. A character LoRA at 0.8/1.5 is the
    validated pair and reads better as "@0.8/1.5" than as two identical numbers repeated.
    """
    def pair(n1: str, n2: str, label: str) -> str:
        a, b = graph.get(n1), graph.get(n2)
        if not a and not b:
            return f"{label} none"
        name = (a or b)["inputs"]["lora_name"]
        s1 = a["inputs"]["strength_model"] if a else None
        s2 = b["inputs"]["strength_model"] if b else None
        # A LoRA on one stage only is legal but unusual enough to name explicitly.
        if s1 is None or s2 is None:
            stage = "stage1" if s1 is not None else "stage2"
            return f"{label} {name} @{s1 if s1 is not None else s2} ({stage} only)"
        strengths = f"{s1}" if s1 == s2 else f"{s1}/{s2}"
        return f"{label} {name} @{strengths}"

    # Every content LoRA, in the order applied — the order is part of the configuration and
    # a result cannot be tied to a chain that is only half reported.
    parts = [pair("9621", "9622", "char")]
    # A second person, only when there is one -- the common line must not grow a
    # "char2 none" nobody asked about.
    if "9631" in graph or "9632" in graph:
        parts.append(pair("9631", "9632", "char2"))
    found = []
    for i in range(4):
        n1, n2 = f"96{1 + i * 2:02d}", f"96{2 + i * 2:02d}"
        if n1 in graph or n2 in graph:
            found.append(pair(n1, n2, f"content{i + 1}"))
    # Unnumbered when there is exactly one, matching the node title convention and keeping
    # the common line readable: "content sfbehind @0.6" rather than "content1 sfbehind @0.6".
    if len(found) == 1:
        found = [pair("9601", "9602", "content")]
    parts.append(" · ".join(found) if found else "content none")
    return " · ".join(parts)


# ---------------------------------------------------------------------------------------
# Identity reference (wanly-gpu-docker#156, proven in phase 0 / #155)
# ---------------------------------------------------------------------------------------
#
# An OPTIONAL reference image -- a 1536x1024 character sheet or a face close-up -- conditioned
# into both stages through ComfyUI-BFSNodes' LTXIdentityOverlapConditioning, with the matching
# Best-Face-ID LoRA. Phase 0 rendered 36 clips through exactly this patch on wanly's own graph:
# LoRA + sheet held identity best on head turns and wide shots (k2026 mean clip median 0.587
# LoRA-only -> 0.699 with the sheet), at +33% render time (+6% for a face). Results:
# https://github.com/DavidJBarnes/wanly-gpu-docker/issues/155
#
# A SEPARATE STEP FROM resolve(), deliberately. resolve() patches values and never topology;
# this adds nodes and rewires guiders, so folding it in would break that rule for every render.
# Kept apart, a render with no reference never reaches this code at all and its graph -- and
# so its graph_hash, the regression trail -- is byte-identical to what it was before this
# existed. test_identity_ref.py pins that.

#: mode -> the Best-Face-ID LoRA trained for it (Alissonerdx/LTX-Best-Face-ID). Both files are
#: in download_models.sh's _WANTED; test_identity_ref.py holds the two lists together.
IDENTITY_LORAS = {
    "face": "Best_FaceID_v1.0_LoRA.safetensors",
    "sheet": "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors",
}
#: How the overlap node sizes the reference. The CharacterSheet LoRA was trained on 1536x1024
#: sheets, so a sheet stays at its native size; a face ref follows the render's size.
IDENTITY_RESIZE = {"face": "match_target", "sheet": "native_resolution"}
#: The caption prefix both BFID LoRAs were trained with. It goes AHEAD of the character
#: trigger, so the trigger still leads the caption proper.
IDENTITY_PROMPT_PREFIX = "ref_t2v: "

# Recipe graph ids (ltx23_recipe.api.json), per stage:
#   (preview-override node the LoRA chain feeds, distill LoRA node, the stage's latent after
#    the in-place i2v + AV concat, the sampler consuming that latent, the guiders that sample
#    with this stage's model -- the first one's conditioning is what the overlap node reads)
_IDENTITY_STAGES = {
    "1": ("337", "361", ["109", 0], "113", ("383", "129")),
    "2": ("372", "362", ["117", 0], "119", ("103",)),
}
_IDENTITY_PROMPT = "121"
_IDENTITY_VAE = ["9500", 2]
#: 9641/9642 identity LoRA, 9643/9644 overlap conditioning, 9645 LoadImage, 9646 face resize.
#: Content uses 9601.., characters 9621/9622 and 9631/9632, so nothing collides.
IDENTITY_NODE_IDS = ("9641", "9642", "9643", "9644", "9645", "9646")


def add_identity_ref(graph: dict, ref_image: str, mode: str) -> dict:
    """Return a copy of a RESOLVED recipe graph with an identity reference patched in.

    Called from app.run_job() right after resolve(), and only when the request carries both
    `identity_ref` and `identity_mode`. `ref_image` is the name ComfyUI stored the upload
    under. On both stages:

      * the BFID (face) or CharacterSheet (sheet) LoRA at 1.0, spliced AFTER the character
        LoRA(s) and BEFORE the preview-override + sulphur distill:
        301 -> content* -> char* -> IDENTITY -> 337|372 -> 361|362. LoRA deltas are additive,
        so the order is cosmetic; this position leaves resolve()'s 96xx ids and
        lora_stack_note() untouched. With no character LoRA (a sheet-only character) it
        simply follows the content chain, or the checkpoint.
      * LTXIdentityOverlapConditioning fed the post-distill model and the post-i2v latent.
        It is a model patch -- the ref is VAE-encoded inside, appended as clean tokens and
        trimmed before unpatchify -- so conditioning and latent pass through unchanged and no
        LTXVCropGuides is needed. Its MODEL feeds every guider that samples with the stage's
        model (383 and 129 on stage 1, 103 on stage 2); its LATENT feeds the sampler. The
        scheduler's ModelSamplingSD3 (368) stays on the plain model: it only reads the shift.
        source_id=2, layout=overlap, temporal offset 0, ref-CFG off -- phase 0's settings.
      * the prompt is prefixed `ref_t2v: `.

    A face ref is resized to 512x512 first, as the model card does; a sheet is passed at its
    native 1536x1024.
    """
    if mode not in IDENTITY_LORAS:
        raise ValueError(f"identity mode {mode!r}; expected one of {sorted(IDENTITY_LORAS)}")
    if not ref_image:
        raise ValueError("identity reference image name is empty")
    g = json.loads(json.dumps(graph))

    g["9645"] = {"class_type": "LoadImage", "inputs": {"image": ref_image},
                 "_meta": {"title": "identity ref"}}
    if mode == "face":
        g["9646"] = {"class_type": "ImageScale", "inputs": {
            "image": ["9645", 0], "upscale_method": "lanczos", "width": 512, "height": 512,
            "crop": "center"}, "_meta": {"title": "identity ref 512"}}
        ref = ["9646", 0]
    else:
        ref = ["9645", 0]

    for tag, (preview, distill, latent, sampler, guiders) in _IDENTITY_STAGES.items():
        lid = f"964{tag}"            # 9641 / 9642
        g[lid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": IDENTITY_LORAS[mode], "strength_model": 1.0,
                             "model": g[preview]["inputs"]["model"]},
                  "_meta": {"title": f"identity {mode} stage {tag}"}}
        g[preview]["inputs"]["model"] = [lid, 0]

        cid = f"964{2 + int(tag)}"   # 9643 / 9644
        first = g[guiders[0]]["inputs"]
        g[cid] = {"class_type": "LTXIdentityOverlapConditioning", "inputs": {
            "model": [distill, 0], "positive": first["positive"], "negative": first["negative"],
            "vae": _IDENTITY_VAE, "latent": latent, "reference_image": ref,
            "source_id": 2.0, "phase_scale": 1.0, "ref_resize_mode": IDENTITY_RESIZE[mode],
            "debug_log": False, "crop_anchor": "center", "layout": "overlap",
            "reference_guidance_scale": 1.0, "reference_temporal_offset_latents": 0},
            "_meta": {"title": f"identity overlap stage {tag}"}}
        rewired = 0
        for gid in guiders:
            if gid in g and g[gid]["inputs"].get("model") == [distill, 0]:
                g[gid]["inputs"]["model"] = [cid, 0]
                rewired += 1
        # A template re-export that moved a guider off the distill LoRA would otherwise leave
        # the reference conditioning NOTHING, and the render would look entirely normal.
        if not rewired:
            raise ValueError(f"identity ref: no stage-{tag} guider samples from {distill}; "
                             f"the recipe graph has changed shape")
        g[sampler]["inputs"]["latent_image"] = [cid, 3]

    t = g[_IDENTITY_PROMPT]["inputs"]["text"]
    if not t.startswith(IDENTITY_PROMPT_PREFIX.strip()):
        g[_IDENTITY_PROMPT]["inputs"]["text"] = IDENTITY_PROMPT_PREFIX + t
    return g


def identity_note(graph: dict) -> str:
    """Which identity reference this graph conditions on, beside lora_stack_note().

    Read off the RESOLVED graph like the LoRA line, so it is evidence of what will render.
    Absence is stated ("identity none"), never implied.
    """
    a, c = graph.get("9641"), graph.get("9643")
    if not a or not c:
        return "identity none"
    return (f"identity {a['inputs']['lora_name']} @{a['inputs']['strength_model']} "
            f"ref={graph['9645']['inputs']['image']} ({c['inputs']['ref_resize_mode']})")
