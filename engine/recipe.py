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


def resolve(graph: dict, image_name: str, width: int, height: int, *,
            prompt: str, negative: str | None = None, checkpoint: str | None = None,
            char_lora: str | None = None, char_s1: float = 0.8, char_s2: float = 1.5,
            content_loras: list | None = None,
            img_compression: int | None = None) -> dict:
    """Patch the validated graph with this render's configuration.

    Values only, never topology — the graph template is the validated recipe and this moves
    the handful of fields that vary between renders.
    """
    g = json.loads(json.dumps(graph))
    ck = checkpoint or "sulphur_dev_bf16"
    if not ck.endswith(".safetensors"):
        ck += ".safetensors"
    # 2.3 checkpoints are monoliths: every loader naming the file must move
    for nid in ("9500", "9501", "9502"):
        if nid in g and "ckpt_name" in g[nid].get("inputs", {}):
            g[nid]["inputs"]["ckpt_name"] = ck
    g["167"]["inputs"]["image"] = image_name
    g["292"]["inputs"]["value"] = int(width)
    g["293"]["inputs"]["value"] = int(height)
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
        if not name or name.lower() == "none":
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
    s1 = float(char_s1)
    s2 = float(char_s2)
    # A character LoRA is optional. "none" renders the recipe on the checkpoint
    # alone -- useful for judging what the LoRA is actually contributing, and
    # for a shot where the start frame already carries the identity.
    char_name = (char_lora or "").strip()
    want_char = char_name.lower() not in ("", "none")
    # Per stage, like the character strengths beside them. This was 0.6 hardcoded for BOTH
    # stages, which is a configuration rather than a default -- stage 1 generates at half
    # size from noise and stage 2 refines the 2x-upscaled latent, so one number for both is
    # a different setup, not a simpler one. 0.6/0.6 remains the default so a caller that
    # says nothing gets exactly the graph that was validated.
    for tag, (branch, strength) in {"1": ("337", s1), "2": ("372", s2)}.items():
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
        if want_char:
            kid = f"962{tag}"
            char = char_name if char_name.endswith(".safetensors") else char_name + ".safetensors"
            g[kid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"lora_name": char, "strength_model": strength, "model": prev},
                      "_meta": {"title": f"char stage {tag}"}}
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
