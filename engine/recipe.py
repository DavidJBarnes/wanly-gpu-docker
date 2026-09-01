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
            content_lora: str | None = None, img_compression: int | None = None) -> dict:
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
    content = (content_lora or "none").strip()
    has_content = content.lower() not in ("", "none")
    if "9601" in g:
        del g["9601"]
    s1 = float(char_s1)
    s2 = float(char_s2)
    # A character LoRA is optional. "none" renders the recipe on the checkpoint
    # alone -- useful for judging what the LoRA is actually contributing, and
    # for a shot where the start frame already carries the identity.
    char_name = (char_lora or "").strip()
    want_char = char_name.lower() not in ("", "none")
    for tag, (branch, strength) in {"1": ("337", s1), "2": ("372", s2)}.items():
        prev = ["301", 0]
        if has_content:
            cid = f"960{tag}"
            name = content if content.endswith(".safetensors") else content + ".safetensors"
            g[cid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"lora_name": name, "strength_model": 0.6, "model": prev},
                      "_meta": {"title": f"content stage {tag}"}}
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
