"""The Qwen-Image-Edit ComfyUI graph and the head-angle recipe (wanly-console#548).

PURE: no I/O, no ComfyUI, no torch -- everything here is decided before a request reaches the
card, and is tested without one.

THE GRAPH is keyframe-server's `build_workflow` (3090:~/keyframe-server/server.py), which ran
this checkpoint in production: CheckpointLoaderSimple -> TextEncodeQwenImageEditPlus (the source
as image1) -> KSampler -> VAEDecode. Phr00t's Rapid-AIO merges transformer, VAE and text encoder
into one file with the Lightning accelerators baked in -- hence 4 steps at cfg 1, and the
author's euler_ancestral/beta for v23. Topology is fixed; only values move.

THE CHECKPOINT'S BASE IS Qwen-Image-Edit-2511. Verified, not assumed: v23's own metadata carries
the merge graph (UNETLoader qwen_image_edit_2511_bf16) and the `__index_timestep_zero__` marker
ComfyUI's model_detection keys 2511 on, and Phr00t's changelog says v20 onward is "100% Qwen Edit
2511".

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

CHECKPOINT = os.environ.get("IMAGE_EDIT_CKPT", "Qwen-Rapid-AIO-NSFW-v23.safetensors")
SAMPLER = os.environ.get("IMAGE_EDIT_SAMPLER", "euler_ancestral")
SCHEDULER = os.environ.get("IMAGE_EDIT_SCHEDULER", "beta")
STEPS = int(os.environ.get("IMAGE_EDIT_STEPS", "4"))
#: Output ceiling. Compute scales with pixels (keyframe-server: 0.39 MP ~6 s, 1.55 MP ~21 s,
#: 7 MP ~186 s) and a phone photo fed in raw is 12 MP.
MAX_MP = float(os.environ.get("IMAGE_EDIT_MAX_MP", "1.2"))

FRAMING_PIN = (
    "Keep everything else in the photograph exactly the same: the same person and identity, "
    "the same facial expression, the same hair, clothes and background, the same lighting, "
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
    """A free-text instruction with the framing pin appended, unless the caller already pinned
    it. The pin is what keeps an outfit change from zooming out to show the shoes."""
    text = instruction.strip()
    if not text:
        raise ValueError("nothing to apply: the instruction is empty")
    if FRAMING_PIN in text:
        return text
    return f"{text.rstrip('.')}. {FRAMING_PIN}"


# ------------------------------------------------------------------------------- graph


def build_workflow(source_name: str, width: int, height: int, prompt: str, seed: int,
                   steps: int = STEPS, denoise: float = 1.0,
                   lora: str | None = None, lora_strength: float = 0.0) -> dict:
    """ComfyUI API-format graph, keyframe-server's topology.

    `denoise` < 1 seeds the sampler with the SOURCE's latent instead of noise, so only part of
    the picture is redrawn. It is for small adjustments; a head turn needs 1.0, because a
    turned head is not a light repaint of a frontal one.
    """
    model: list = ["1", 0]
    wf: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "9": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "101": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "4": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"prompt": "", "clip": ["1", 1], "vae": ["1", 2]}},
        "3": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"prompt": prompt, "clip": ["1", 1], "vae": ["1", 2],
                         "image1": ["101", 0]}},
    }
    if lora and lora_strength:
        wf["20"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": lora,
                               "strength_model": float(lora_strength)}}
        model = ["20", 0]
    latent: list = ["9", 0]
    if denoise < 1.0:
        # The encoder emits a latent at the image's own size, so the source must already be
        # the latent's size -- the caller resizes it before upload.
        wf["110"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["101", 0], "vae": ["1", 2]}}
        latent = ["110", 0]
        del wf["9"]
    wf["2"] = {"class_type": "KSampler",
               "inputs": {"model": model, "positive": ["3", 0], "negative": ["4", 0],
                          "latent_image": latent, "seed": int(seed), "steps": int(steps),
                          "cfg": 1.0, "sampler_name": SAMPLER, "scheduler": SCHEDULER,
                          "denoise": float(denoise)}}
    wf["5"] = {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 2]}}
    wf["6"] = {"class_type": "SaveImage",
               "inputs": {"images": ["5", 0], "filename_prefix": "image-edit"}}
    return wf
