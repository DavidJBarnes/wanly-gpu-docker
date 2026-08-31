"""Recipe resolution: a validated (character, pose) config -> a ComfyUI graph.

The authoring path is the recipe sheet (ODS) -> recipes.json. This module is the
ONLY place that turns that data into a graph, so the UI and the offline tools
produce byte-identical results and share one regression hash.

Values from the sheet are ASSERTED, not loosely parsed: if a shared definition
stops saying what this code implements, resolution fails loudly rather than
quietly rendering something else.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path

HERE = Path(__file__).parent
RECIPES = Path(__file__).parent / "recipes" / "recipes.json"
RECIPE_WORKFLOW = "ltx23_recipe.api.json"


def load() -> dict:
    return json.loads(RECIPES.read_text())


def _assert_definitions(r: dict, d: dict) -> None:
    dist = d[r["distill"]]
    assert "sulphur_distill_lora_condsafe" in dist and "0.3" in dist and "0.6" in dist, \
        f"distill definition drifted: {dist}"
    steps = d[r["steps"]]
    assert "20 steps" in steps and "0.85, 0.7250, 0.4219, 0.0" in steps, \
        f"steps definition drifted: {steps}"
    guid = d[r["guidance"]]
    for tok in ("cfg 3", "stg 1", "rescale 0.9", "block 28", "cfg 1"):
        assert tok in guid, f"guidance definition drifted, missing {tok!r}"
    assert r["frames"].startswith("241"), f"frames drifted: {r['frames']}"


def resolve(graph: dict, recipe_name: str, image_name: str, width: int, height: int,
            data: dict | None = None, overrides: dict | None = None,
            character: str | None = None) -> dict:
    data = data or load()
    book = (data.get("characters", {}).get(character) if character
            else None) or data
    if recipe_name not in book["recipes"]:
        raise KeyError(f"unknown recipe {recipe_name!r}"
                       + (f" for {character!r}" if character else ""))
    r = dict(book["recipes"][recipe_name])
    d = data["definitions"]
    _assert_definitions(r, d)
    o = overrides or {}

    g = json.loads(json.dumps(graph))
    ck = r["checkpoint"]
    if not ck.endswith(".safetensors"):
        ck += ".safetensors"
    # 2.3 checkpoints are monoliths: every loader naming the file must move
    for nid in ("9500", "9501", "9502"):
        if nid in g and "ckpt_name" in g[nid].get("inputs", {}):
            g[nid]["inputs"]["ckpt_name"] = ck
    g["167"]["inputs"]["image"] = image_name
    g["292"]["inputs"]["value"] = int(width)
    g["293"]["inputs"]["value"] = int(height)
    g["121"]["inputs"]["text"] = o.get("prompt") or d[r["prompt"]]
    g["110"]["inputs"]["text"] = o.get("negative") or d[r["negative"]]

    # one content+character chain per stage branch, mirroring how the distill
    # LoRA is already wired at 361/362
    content = (r.get("content_lora") or "none").strip()
    has_content = content.lower() not in ("", "none")
    if "9601" in g:
        del g["9601"]
    s1 = float(o.get("char_s1", r["char_s1"]))
    s2 = float(o.get("char_s2", r["char_s2"]))
    # A character LoRA is optional. "none" renders the recipe on the checkpoint
    # alone -- useful for judging what the LoRA is actually contributing, and
    # for a shot where the start frame already carries the identity.
    char_name = (o.get("char_lora") or r["char_lora"] or "").strip()
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


def char_lora_for(recipe_name: str, character: str | None = None,
                  data: dict | None = None) -> str:
    """The character LoRA a recipe uses, for naming output."""
    data = data or load()
    book = (data.get("characters", {}).get(character) if character else None) or data
    return (book.get("recipes", {}).get(recipe_name) or {}).get("char_lora", "")
