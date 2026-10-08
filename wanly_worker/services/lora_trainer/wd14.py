"""WD14 tagging for SDXL character runs (#175): the captions the aio LoRAs were trained on.

Wanly's stored dataset captions are qwen sentences, written for LTX. An SDXL booru-tag base
(Lustify) is prompted -- and was trained, for aio -- in tags, so an SDXL run re-captions its
own staged images here rather than using them.

RUN AS A SCRIPT, in the sd-scripts venv: `python wd14.py <image_dir> <trigger>`. That venv has
onnxruntime and this package's runtime does not need it, so the heavy imports live in `main`
and the caption rules stay importable (and testable) anywhere.

THE PREPROCESSING IS THE CONTRACT. From ~/projects/loras/training-how-to.md §4, which matches
sd-scripts' finetune/tag_images_by_wd14_tagger.py: BGR, raw 0-255 floats, padded square with
WHITE and then resized to 448. Any one of those wrong and the tagger does not fail -- it returns
plausible garbage ("solo, simple_background, black_background, dark") for every image. That
signature is checked: a run whose captions all collapse to it is refused.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

THRESHOLD = 0.35

#: Category headers and meta-tags (how-to §4).
EXCLUDE_TAGS = {
    "general", "sensitive", "questionable", "explicit",
    "no_humans", "1girl", "1boy", "2girls", "3girls",
    "multiple_girls", "general_focus", "comic", "sketch",
    "monochrome", "greyscale",
}

#: FIXED CHARACTER FEATURES (how-to §5): the trigger is supposed to carry these, so a caption
#: that names them gives the model somewhere else to put the identity. EXACT tokens --
#: `breast_hold` and `hair_over_shoulder` describe a pose and stay.
STRIP_TAGS = {
    "blonde_hair", "brown_hair", "black_hair", "red_hair", "long_hair", "short_hair",
    "multicolored_hair",
    "blue_eyes", "green_eyes", "brown_eyes", "grey_eyes",
    "breasts", "large_breasts", "medium_breasts", "small_breasts", "nipples", "dark_nipples",
    "areolae", "breasts_apart",
    "freckles", "lips", "nose", "mole", "mole_on_breast", "mole_under_mouth", "forehead",
    "glasses",
}

#: WORN GLASSES ARE A FIXED FEATURE TOO (#202): Joana wears hers in every render. Tagged, they
#: bind to `glasses` / `brown-framed_eyewear` instead of the trigger, and prompting the tag back
#: draws SDXL's generic frames pasted over a face the LoRA learned bare. So `glasses` and every
#: frame-style `*_eyewear` tag go -- but not these, which describe the SHOT, not the face.
#: `no_eyewear` most of all: it marks a bare-faced shot, the one place glasses are NOT on.
#: Tinted lenses (`tinted_eyewear`, `red-tinted_eyewear`, ...) are sunglasses by another name.
#: `sunglasses` and `eyewear_removed` / `eyewear_on_head` don't end in `_eyewear` and stay too.
#: Names checked against the tagger's own selected_tags.csv on 3090a.
KEEP_EYEWEAR_TAGS = {"adjusting_eyewear", "holding_eyewear", "hand_on_eyewear",
                     "looking_over_eyewear", "no_eyewear"}


def _stripped(tag: str) -> bool:
    if tag in STRIP_TAGS:
        return True
    if tag in KEEP_EYEWEAR_TAGS or tag.endswith("tinted_eyewear"):
        return False
    return tag.endswith("_eyewear")

#: What broken preprocessing produces for every image (how-to §4).
GARBAGE = {"solo", "simple_background", "black_background", "dark", "negative_space"}


def caption(trigger: str, tags: list[str]) -> str:
    """`<trigger>, tag, tag, ...` -- trigger FIRST, because keep_tokens and the prompt both
    assume it is. Missing trigger = dead LoRA (it killed a v8)."""
    kept = [t for t in tags if t not in EXCLUDE_TAGS and not _stripped(t) and t != trigger]
    return ", ".join([trigger, *kept])


def looks_broken(captions: list[str]) -> bool:
    """Every caption made only of the broken-preprocessing tags. One such image is a dark
    photo; ALL of them is the tagger seeing noise."""
    if not captions:
        return False
    for c in captions:
        tags = {t.strip() for t in c.split(",")[1:] if t.strip()}
        if not tags <= GARBAGE:
            return False
    return True


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def main(image_dir: str, trigger: str, model_dir: str) -> int:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    md = Path(model_dir)
    with (md / "selected_tags.csv").open(encoding="utf-8") as fh:
        names = [row["name"] for row in csv.DictReader(fh)]
    session = ort.InferenceSession(str(md / "model.onnx"), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name

    written = []
    #: What the TAGGER said, behind a one-token prefix: a pair's prefix ("k3lly, 1girl", #184)
    #: is several tags, and they must not count as evidence the tagger saw anything.
    tagged = []
    for path in _images(Path(image_dir)):
        a = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        h, w = a.shape[:2]
        size = max(h, w)
        px, py = size - w, size - h
        a = np.pad(a, ((py // 2, py - py // 2), (px // 2, px - px // 2), (0, 0)),
                   mode="constant", constant_values=255)
        a = np.array(Image.fromarray(a[:, :, ::-1]).resize((448, 448), Image.LANCZOS))[:, :, ::-1]
        probs = session.run([out], {inp: np.expand_dims(a.astype(np.float32), 0)})[0][0]
        tags = [names[i] for i, p in enumerate(probs) if p > THRESHOLD]
        text = caption(trigger, tags)
        tagged.append(caption("_", tags))
        path.with_suffix(".txt").write_text(text + "\n")
        written.append(text)
        print(f"{path.name}: {text}", flush=True)

    if looks_broken(tagged):
        print("every caption is the broken-preprocessing signature "
              f"({', '.join(sorted(GARBAGE))}) -- refusing to train on them", flush=True)
        return 3
    print(f"tagged {len(written)} images", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: wd14.py <image_dir> <trigger> <wd14_model_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
