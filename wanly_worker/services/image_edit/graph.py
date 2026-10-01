"""The Qwen-Image-Edit ComfyUI graph and the head-angle recipe (wanly-console#548).

PURE: no I/O, no ComfyUI, no torch -- everything here is decided before a request reaches the
card, and is tested without one.

THE MODEL IS THE OFFICIAL Qwen-Image-Edit-2511 (wanly-console#574, wanly-gpu-docker#157), as
Comfy-Org ships it for ComfyUI: three files, loaded separately -- the fp8mixed transformer
(UNETLoader), the Qwen2.5-VL 7B text encoder (CLIPLoader, type qwen_image) and the Qwen-Image VAE.
It replaced Phr00t's Qwen-Rapid-AIO-NSFW-v23, a community merge with the Lightning accelerators
baked in (4 steps, cfg 1): v23 turned heads into a different person -- younger, smoother skin,
another nose and jaw (#574) -- while keyframe-server's good results came from the official
model. The settings are the official template's, the ones the character-sheet recipe
(loras/reftest-2026-09-30/sheets.py) was proven with: 40 steps, CFG 4, euler/simple,
ModelSamplingAuraFlow shift 3.1, CFGNorm 1, reference latents by `index_timestep_zero`.
Ten times v23's steps, at two passes each for real CFG: an edit costs minutes, not seconds.

THE GRAPH: UNETLoader -> ModelSamplingAuraFlow -> CFGNorm -> KSampler, conditioned by
TextEncodeQwenImageEditPlus (the source as image1, positive and empty-prompt negative alike)
through FluxKontextMultiReferenceLatentMethod. Every node is core ComfyUI. Topology is fixed;
only values move.

TWO GRAPHS, ONE MODEL. `build_workflow` is the Edit dialog's (an angle, an expression, free
text, at the source's own size). `turnaround_workflow` is the character-sheet recipe
(wanly-console#582, #585): one photo of the person in, a 1088x1024 front/side/back turnaround
out.

THE HEAD-ANGLE RECIPE IS PROMPT-ONLY, chosen by the #548 spike (36 edits, 3 real portraits):
image-space wording plus the framing pin reached three-quarter and full profile in the asked
direction on every source, framing held. fal's Qwen-Image-Edit-2511-Multiple-Angles LoRA -- the
one community angle LoRA matching this base -- was tested at 0.9 and rejected: it moves the
CAMERA, not the head (zooms out ~40%, invents bodies and a second person), mostly fails to turn
the head at three-quarter, and its "left side view" landed on both sides. `build_workflow` keeps
a generic `lora` argument; nothing here passes one.

FRAMING PINS, from keyframe-server's docs/pipeline-notes.md section 2 ("Edit models re-frame when
you ask for something off-frame"), all three reused:
  1. the latent is the SOURCE's own size (capped at MAX_MP, multiple of 16) -- never the model's
     favourite 1 MP square, which re-frames by construction;
  2. the instruction forbids re-framing, explicitly and redundantly (FRAMING_PIN);
  3. only changes visible inside the current frame are asked for.
"""
from __future__ import annotations

import math
import os

#: The three files, relative to the model tree's folders (service.paths_yaml). They must match
#: download_models.sh's --image-edit _WANTED; test_image_edit.py holds the two together.
UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
#: folder in the model tree -> file. What the preflight checks and the graph loads.
MODEL_FILES = {"base": UNET, "text_encoders": TEXT_ENCODER, "vae": VAE}
#: What a result says it was made with (/health, every response, the sheet's provenance).
MODEL = "Qwen-Image-Edit-2511 (Comfy-Org fp8mixed)"
#: Kept under its old name: wanly-api records it per edit, and the console shows it.
CHECKPOINT = UNET

SAMPLER = os.environ.get("IMAGE_EDIT_SAMPLER", "euler")
SCHEDULER = os.environ.get("IMAGE_EDIT_SCHEDULER", "simple")
STEPS = int(os.environ.get("IMAGE_EDIT_STEPS", "40"))
CFG = float(os.environ.get("IMAGE_EDIT_CFG", "4.0"))
SHIFT = float(os.environ.get("IMAGE_EDIT_SHIFT", "3.1"))
CFG_NORM = float(os.environ.get("IMAGE_EDIT_CFG_NORM", "1.0"))
#: Output ceiling. Compute scales with pixels (keyframe-server: 0.39 MP ~6 s, 1.55 MP ~21 s,
#: 7 MP ~186 s) and a phone photo fed in raw is 12 MP.
MAX_MP = float(os.environ.get("IMAGE_EDIT_MAX_MP", "1.2"))

FRAMING_PIN = (
    "Keep everything else in the photograph exactly the same: the same person and identity, "
    "the same facial expression, the same hair, clothes and background, the same lighting, "
    "the same camera position, distance, crop and zoom. Do not zoom out."
)

#: The pin for every edit that CHANGES the expression -- an expression preset, or a free-text
#: instruction (console#569: the dialog's "describe the change" box is now how expressions are
#: asked for). It is FRAMING_PIN without "the same facial expression": appended to "make her
#: smile", that clause asked for the opposite of the edit. A head angle alone keeps FRAMING_PIN
#: word for word -- that is the wording the #548 spike measured.
EDIT_PIN = (
    "Keep everything else in the photograph exactly the same: the same person and identity, "
    "the same hair, clothes and background, the same lighting, "
    "the same camera position, distance, crop and zoom. Do not zoom out."
)

#: Largest head turn / tilt the angle recipe accepts, in degrees. 90 is a full profile; past it
#: the face is turning away from the camera, which is not a head-angle edit any more.
MAX_YAW = 90.0
MAX_PITCH = 45.0


def latent_size(width: int, height: int, max_mp: float = MAX_MP) -> tuple[int, int]:
    """The source's own size, scaled down to `max_mp` if larger, floored to multiples of 16.

    Never scaled UP: a small source generated at 1 MP comes back re-framed, which is the
    failure the pin exists for, and invents detail nobody asked for.
    """
    w, h = int(width), int(height)
    mp = (w * h) / 1e6
    if mp > max_mp:
        s = math.sqrt(max_mp / mp)
        w, h = int(w * s), int(h * s)
    return max(16, (w // 16) * 16), max(16, (h // 16) * 16)


# ---------------------------------------------------------------------------- head angle
#
# DIRECTION IS THE IMAGE'S, AND MATCHES PHASE 1. Negative yaw turns the face toward the LEFT
# EDGE OF THE PICTURE -- the viewer's left, the subject's right. That is what LivePortrait's
# rotate_yaw < 0 does (checked on Kelly-2000 sel_008: -15 points the nose image-left), so a
# "left" preset means the same thing whichever engine the console routes it to. Positive pitch
# tilts the head UP (chin raised).
#
# The words are the image's too, because the model is only ever shown the image: "toward the
# left edge of the image" is unambiguous where "her left" depends on who is asking.


def _yaw_words(yaw: float) -> str | None:
    a = abs(yaw)
    if a < 5:
        return None
    side = "left" if yaw < 0 else "right"
    if a >= 70:
        return (f"Turn the person's head {round(a)} degrees into a full side profile, the nose "
                f"pointing toward the {side} edge of the image.")
    if a >= 35:
        return (f"Turn the person's head about {round(a)} degrees so the face is in "
                f"three-quarter view, pointing toward the {side} edge of the image.")
    return (f"Turn the person's head about {round(a)} degrees toward the {side} edge of the "
            f"image.")


def _pitch_words(pitch: float) -> str | None:
    a = abs(pitch)
    if a < 5:
        return None
    if pitch > 0:
        return (f"Tilt the person's head back about {round(a)} degrees so the chin is raised "
                f"and the face looks up.")
    return (f"Tilt the person's head forward about {round(a)} degrees so the chin is lowered "
            f"and the face looks down.")


def angle_prompt(yaw: float, pitch: float) -> str:
    """The instruction for a head angle, framing pin included. ValueError if there is nothing
    to do or the angle is out of range -- the caller turns that into a 422."""
    if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
        raise ValueError(f"head angle out of range: yaw {yaw} (max ±{MAX_YAW:g}), "
                         f"pitch {pitch} (max ±{MAX_PITCH:g})")
    parts = [p for p in (_yaw_words(yaw), _pitch_words(pitch)) if p]
    if not parts:
        raise ValueError("nothing to apply: a head angle under 5 degrees is not a change")
    return " ".join(parts + [FRAMING_PIN])


def instruction_prompt(instruction: str) -> str:
    """A free-text instruction with the edit pin appended, unless the caller already pinned
    it. The pin is what keeps an outfit change from zooming out to show the shoes.

    EDIT_PIN, not FRAMING_PIN (console#569): the free-text box is how an expression is asked
    for now, and FRAMING_PIN's "the same facial expression" contradicted every one of those."""
    text = instruction.strip()
    if not text:
        raise ValueError("nothing to apply: the instruction is empty")
    if FRAMING_PIN in text or EDIT_PIN in text:
        return text
    return f"{text.rstrip('.')}. {EDIT_PIN}"


# --------------------------------------------------------------------------- expressions
#
# THE EXPRESSION PRESETS AS QWEN INSTRUCTIONS (console#569). LivePortrait is retired from the
# Edit dialog -- it "drops every detail" -- so the dialog's Smile / Big laugh / ... buttons are
# instructions here, written the way the head-angle words are: say what the face DOES,
# explicitly, and scope the change ("change only the facial expression") so the model does not
# take it as licence to redraw the rest. The gaze presets are image-space like the angles: "the
# left edge of the image" is what yaw < 0 means too.
#
# NOT YET MEASURED. The head-angle wording was chosen from 36 renders (#548); these were
# written from the same rules but are unrendered. The wording lives here, beside the model, so
# retuning one is a change to this table and nothing else -- wanly-api sends only the name.
#
# name -> (label, instruction). The names are the old LivePortrait preset names where one
# existed, so a saved file's `_edit-smile_` tag reads the same across the switch.

EXPRESSIONS: dict[str, tuple[str, str]] = {
    "smile": ("Smile", (
        "Change only the person's facial expression to a natural, gentle smile: the corners "
        "of the mouth turned up and the cheeks slightly lifted.")),
    "big_laugh": ("Big laugh", (
        "Change only the person's facial expression to a big, open-mouthed laugh: the mouth "
        "wide open showing the teeth, the cheeks raised and the eyes creased with laughter.")),
    "surprised": ("Surprised", (
        "Change only the person's facial expression to surprise: the eyebrows raised high, the "
        "eyes wide open and the mouth open in a small oval.")),
    "eyes_closed": ("Eyes closed", (
        "Close the person's eyes gently, the eyelids fully shut as if resting. Change nothing "
        "else about the face.")),
    "sad": ("Sad", (
        "Change only the person's facial expression to sadness: the inner ends of the eyebrows "
        "raised, the corners of the mouth turned down and the lips pressed slightly together.")),
    "angry": ("Angry", (
        "Change only the person's facial expression to anger: the eyebrows drawn down and "
        "together, the eyes narrowed and the lips pressed firmly together.")),
    "serious": ("Serious", (
        "Change only the person's facial expression to a calm, serious look: no smile, the "
        "mouth closed and relaxed, a steady gaze.")),
    "speaking": ("Speaking", (
        "Change only the person's mouth so they look caught mid-sentence while talking: the "
        "lips naturally parted and the mouth slightly open.")),
    "look_left": ("Eyes left", (
        "Keep the head exactly where it is and move only the eyes, so the person looks toward "
        "the left edge of the image.")),
    "look_right": ("Eyes right", (
        "Keep the head exactly where it is and move only the eyes, so the person looks toward "
        "the right edge of the image.")),
    "look_up": ("Eyes up", (
        "Keep the head exactly where it is and move only the eyes, so the person looks "
        "upward.")),
    "look_down": ("Eyes down", (
        "Keep the head exactly where it is and move only the eyes, so the person looks "
        "downward.")),
}


def expression_words(name: str) -> str:
    """The instruction for an expression preset. ValueError (-> 422) for an unknown name."""
    try:
        return EXPRESSIONS[name][1]
    except KeyError:
        raise ValueError(f"unknown expression {name!r}; known: {', '.join(EXPRESSIONS)}") \
            from None


def compose_prompt(yaw: float = 0.0, pitch: float = 0.0, expression: str | None = None,
                   instruction: str | None = None) -> str:
    """The prompt for any mix of a head angle, an expression preset and free text.

    A head angle ALONE is angle_prompt verbatim -- the measured recipe, FRAMING_PIN and all.
    Anything that changes the expression, alone or with an angle, ends in EDIT_PIN instead.
    ValueError when there is nothing to do or a value is out of range.
    """
    angle = abs(yaw) >= 5 or abs(pitch) >= 5
    text = (instruction or "").strip()
    if not expression and not text:
        if not angle and (yaw or pitch):
            raise ValueError("nothing to apply: a head angle under 5 degrees is not a change")
        return angle_prompt(yaw, pitch)
    if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
        raise ValueError(f"head angle out of range: yaw {yaw} (max ±{MAX_YAW:g}), "
                         f"pitch {pitch} (max ±{MAX_PITCH:g})")
    parts = [p for p in (_yaw_words(yaw), _pitch_words(pitch)) if p]
    if expression:
        parts.append(expression_words(expression))
    if text:
        if not parts:
            return instruction_prompt(text)
        parts.append(text.rstrip(".") + ".")
    return " ".join(parts + [EDIT_PIN])


# ------------------------------------------------------------------------------- graph


def settings_note(steps: int = STEPS, cfg: float = CFG) -> str:
    """The sampler settings in words, for a result's record."""
    return (f"{steps} steps, cfg {cfg:g}, {SAMPLER}/{SCHEDULER}, AuraFlow shift {SHIFT:g}, "
            f"CFGNorm {CFG_NORM:g}, index_timestep_zero")


def _qwen(wf: dict, image: list, prompt: str, latent: list, seed: int, steps: int, cfg: float,
          denoise: float = 1.0, lora: str | None = None, lora_strength: float = 0.0) -> dict:
    """The official 2511 graph around an image node and a latent node already in `wf`."""
    wf["1"] = {"class_type": "UNETLoader",
               "inputs": {"unet_name": UNET, "weight_dtype": "default"}}
    wf["7"] = {"class_type": "CLIPLoader",
               "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image", "device": "default"}}
    wf["8"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    wf["3"] = {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"prompt": prompt, "clip": ["7", 0], "vae": ["8", 0], "image1": image}}
    wf["4"] = {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"prompt": "", "clip": ["7", 0], "vae": ["8", 0], "image1": image}}
    wf["31"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["3", 0],
                           "reference_latents_method": "index_timestep_zero"}}
    wf["41"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["4", 0],
                           "reference_latents_method": "index_timestep_zero"}}
    model: list = ["1", 0]
    if lora and lora_strength:
        wf["20"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": lora,
                               "strength_model": float(lora_strength)}}
        model = ["20", 0]
    wf["10"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": SHIFT}}
    wf["11"] = {"class_type": "CFGNorm", "inputs": {"model": ["10", 0], "strength": CFG_NORM}}
    wf["2"] = {"class_type": "KSampler",
               "inputs": {"model": ["11", 0], "positive": ["31", 0], "negative": ["41", 0],
                          "latent_image": latent, "seed": int(seed), "steps": int(steps),
                          "cfg": float(cfg), "sampler_name": SAMPLER, "scheduler": SCHEDULER,
                          "denoise": float(denoise)}}
    wf["5"] = {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["8", 0]}}
    return wf


def build_workflow(source_name: str, width: int, height: int, prompt: str, seed: int,
                   steps: int = STEPS, denoise: float = 1.0,
                   lora: str | None = None, lora_strength: float = 0.0,
                   cfg: float = CFG) -> dict:
    """The Edit dialog's graph: the source at the latent's size, edited in place.

    `denoise` < 1 seeds the sampler with the SOURCE's latent instead of noise, so only part of
    the picture is redrawn. It is for small adjustments; a head turn needs 1.0, because a
    turned head is not a light repaint of a frontal one.
    """
    wf: dict = {
        "101": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "9": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    latent: list = ["9", 0]
    if denoise < 1.0:
        # The encoder emits a latent at the image's own size, so the source must already be
        # the latent's size -- the caller resizes it before upload.
        wf["110"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["101", 0], "vae": ["8", 0]}}
        latent = ["110", 0]
        del wf["9"]
    _qwen(wf, ["101", 0], prompt, latent, seed, steps, cfg, denoise, lora, lora_strength)
    wf["6"] = {"class_type": "SaveImage",
               "inputs": {"images": ["5", 0], "filename_prefix": "image-edit"}}
    return wf


# ---------------------------------------------------------------------------- turnaround
#
# THE CHARACTER-SHEET RECIPE, ONE-PHOTO FORM (wanly-console#582, #585). One photo of the person --
# full body or most of it, in the outfit -- is image 1; a photoreal front / side / back
# full-body turnaround on white comes out, 1088x1024: the right-hand part of the 1536x1024 sheet
# (sheet.py puts a face panel cropped from the SAME photo beside it). The photo goes through
# FluxKontextImageScale (the model's own preferred size), not at its own size: the output is a
# new picture, not an edit of this one, so there is no framing to pin.
#
# WHY ONE PHOTO, AND NO BODY WORDS (#585). Qwen-Image-Edit-2511 keeps image 1 faithfully and
# mostly ignores everything else: a "She has an athletic build." sentence (#582's BODY field)
# and a body photo given as image 2 both came back with the face photo's implied build. Build is
# only controllable when image 1 IS the body -- so the photo is the whole person, and the prompt
# asks to keep "body shape, build and proportions from image 1". There is no body field.
#
# The wording is loras/phase0-2026-10-01/character_sheet_one_input.json's, verbatim for a woman
# with hair and outfit given (test_image_edit.py checks it word for word); a man gets "he/his",
# and `subject` names who image 1 shows.

TURNAROUND_W, TURNAROUND_H = 1088, 1024


def _clause(text: str | None) -> str:
    return (text or "").strip().rstrip(".").strip()


def turnaround_prompt(outfit: str, hair: str | None = None, gender: str = "female",
                      subject: str | None = None) -> str:
    """The turnaround instruction. ValueError (-> 422) without an outfit."""
    outfit = _clause(outfit)
    if not outfit:
        raise ValueError("a turnaround needs an outfit")
    he, his = ("he", "his") if gender == "male" else ("she", "her")
    who = f"the {_clause(subject) or ('man' if gender == 'male' else 'woman')} in image 1"
    hair = _clause(hair) or f"{his} hair exactly as in image 1"
    return (f"Create a photorealistic full-body character turnaround of {who} on a plain pure "
            f"white studio background. Three full-body views of the same person side by side, "
            f"left to right: a front view facing the camera, a side view facing 90 degrees to "
            f"the right, and a back view facing completely away from the camera. Each view "
            f"shows {his} whole body from head to toe with no cropping, standing upright with "
            f"arms relaxed at {his} sides, feet visible. Keep {his} exact face, facial "
            f"features, skin tone, body shape, build and proportions from image 1, and {hair}. "
            f"{he.capitalize()} wears {outfit}, identical in all three views. Soft even studio "
            f"lighting, equal white spacing between the views, no text, no labels, no borders.")


def turnaround_workflow(photo_name: str, prompt: str, seed: int, steps: int = STEPS,
                        cfg: float = CFG) -> dict:
    """The one-input workflow's graph: the photo, scaled by FluxKontextImageScale, as image1;
    a 1088x1024 empty latent."""
    wf: dict = {
        "101": {"class_type": "LoadImage", "inputs": {"image": photo_name}},
        "102": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["101", 0]}},
        "9": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": TURNAROUND_W, "height": TURNAROUND_H, "batch_size": 1}},
    }
    _qwen(wf, ["102", 0], prompt, ["9", 0], seed, steps, cfg)
    wf["6"] = {"class_type": "SaveImage",
               "inputs": {"images": ["5", 0], "filename_prefix": "turnaround"}}
    return wf
