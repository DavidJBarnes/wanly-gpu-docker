#!/usr/bin/env python3
"""Point the converted workflow at our image, prompt, LoRA and orientation.

Everything here is a correction to the AUTHOR'S defaults, not to the conversion:
the workflow ships landscape, with a placeholder positive prompt, and with an
empty Power Lora Loader.
"""
import json, sys

api = json.load(open("ltx23_api.json"))
ui = {n["id"]: n for n in json.load(
    open("LTX-2.3_-_I2V_T2V_Basic_for_checkpoint_models.json"))["nodes"]}

POSITIVE = ("m15510n4ry. a beautiful blonde woman in white lingerie lies on her back on a "
            "dark bed, legs raised and spread. A man off camera thrusts into her "
            "repeatedly in a steady rhythm; her breasts and thighs jiggle with each "
            "impact. She arches her back, tilts her head and moans, holding eye contact "
            "with the camera. Handheld point of view, natural indoor light. "
            "Audio: her rhythmic moaning and breathing, skin against skin.")
NEGATIVE = ("static, still image, frozen, no motion, camera pan, slideshow, "
            "identity change, different person, face distortion, warped anatomy, "
            "extra limbs, deformed hands, blurry, low quality")

# 1. The prompt must go to the POSITIVE encoder. LTXVConditioning node 107 wires
#    positive <- 121 and negative <- 110. Encoder 121 takes its text over a link
#    from PrimitiveStringMultiline 352, which shipped holding the author's
#    placeholder ("Make this image come alive with fluid motion."). Writing to
#    110 -- as a "longest string is the prompt" guess did -- puts the whole
#    prompt in the NEGATIVE, which asks the model to avoid the very thing wanted.
cond = next(v for v in api.values() if v["class_type"] == "LTXVConditioning")
pos_id = str(cond["inputs"]["positive"][0])
neg_id = str(cond["inputs"]["negative"][0])
print(f"positive encoder = {pos_id}, negative encoder = {neg_id}")

api[pos_id]["inputs"]["text"] = POSITIVE      # replaces the link to 352
api[neg_id]["inputs"]["text"] = NEGATIVE
print(f"  wrote prompt to {pos_id}, negative to {neg_id}")

# 2. Orientation. The workflow is authored landscape (WIDTH=1280, HEIGHT=736);
#    our source is portrait 768x1344, so those constants swap. Both stay /32.
for nid, n in api.items():
    if n["class_type"] == "INTConstant":
        t = (ui.get(int(nid)) or {}).get("title")
        if t == "WIDTH":
            n["inputs"]["value"] = 736
            print(f"  {nid} WIDTH  -> 736")
        elif t == "HEIGHT":
            n["inputs"]["value"] = 1280
            print(f"  {nid} HEIGHT -> 1280")

# 3. The content LoRA. rgthree's Power Lora Loader carries its LoRAs in dynamic
#    widgets that the API schema does not declare, so it cannot be filled over
#    /prompt -- and the author ships it empty. Splice an explicit
#    LoraLoaderModelOnly in after it instead.
plora = next((k for k, v in api.items()
              if v["class_type"] == "Power Lora Loader (rgthree)"), None)
if plora:
    new_id = "9001"
    api[new_id] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"lora_name": "DR34ML4Y_LT3X_V3.safetensors",
                              "strength_model": 0.6,
                              "model": [plora, 0]},
                   "_meta": {"title": "content LoRA (spliced in)"}}
    moved = 0
    for k, v in api.items():
        if k in (new_id, plora):
            continue
        for name, val in v["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == plora:
                v["inputs"][name] = [new_id, val[1]]
                moved += 1
    print(f"  spliced DR34ML4Y @0.6 after Power Lora Loader {plora}; "
          f"repointed {moved} consumer(s)")

# 4. Ours, not the author's.
api["167"]["inputs"]["image"] = "00008-2776248146-swapped.png"
api["330"]["inputs"]["vae_name"] = "pixel_space"
api["366"]["inputs"]["text_encoder"] = "gemma_3_12B_it_fp8_scaled.safetensors"

json.dump(api, open("ltx23_api_patched.json", "w"), indent=1)
print(f"wrote ltx23_api_patched.json ({len(api)} nodes)")
