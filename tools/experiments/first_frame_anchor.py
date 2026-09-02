#!/usr/bin/env python3
"""wanly-gpu-docker#48 — does the render hold the start frame, and what moves the needle?

WHY THIS EXISTS IN THE REPO AND NOT IN /tmp
    A curation script was lost once to /tmp being cleared and had to be rebuilt from
    scratch. Anything that produces a number someone will later cite belongs here.

WHAT WENT WRONG LAST TIME
    The earlier #48 run measured each clip against ITS OWN FRAME 0. That cannot tell
    "holds the anchor" from "moves less" — a frozen clip scores perfectly — which is why
    img_compression's measured -34% never changed the default. This measures against the
    START IMAGE, and reports motion alongside, so a variant that wins by freezing is
    visible as exactly that.

TWO NUMBERS, ALWAYS TOGETHER
    anchor_i = mean|frame_i - start_image|     does it hold?
    motion_i = mean|frame_i - frame_{i-1}|     is it still moving?

    A variant only wins if anchor improves AND motion does not collapse. Reporting one
    without the other is what made the previous result unusable.

VALUES ONLY
    Patches node 162 (LTXVPreprocess.img_compression) and node 360 (ManualSigmas, stage 2).
    No topology change — the locked recipe's structure is not ours to edit here. Note that
    the real cause #1 on the ticket, the absence of LTXVAddGuide, CANNOT be tested this way:
    expect "less bad", not "fixed".

RUN (inside the engine container, which is where ComfyUI is):
    docker cp first_frame_anchor.py wanly-ltx:/tmp/
    docker exec wanly-ltx python3 /tmp/first_frame_anchor.py \
        --kf /jobs/f5dfcffcd980/kf1.png --out /jobs/_exp48
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt")

from engine import comfy as comfy_mod          # noqa: E402
from engine import recipe as recipe_mod        # noqa: E402

# The reference render: job d10c3e8f, "p@y — p@y - insertion". Held fixed across variants
# so the only thing that differs is the lever under test.
PROMPT = (
    "p@y, this is a video of a young brunette girl seated on a sofa wearing a tube top and "
    "jeans. Her hands are at her side and she is looking at the viewer with a patient look "
    "on her face. From the perspective of a handheld camera, a naked mans penis emerges from "
    "the bottom of the screen. The man walks towards the girl. His large penis with veins "
    "enters from the foreground. As he gets close to her he"
)
NEGATIVE = (
    "static, still image, frozen, no motion, slideshow, identity change, different person, "
    "face distortion, warped anatomy, extra limbs, deformed hands, merged limbs"
)
CHAR_LORA, CHAR_S1, CHAR_S2 = "pay_v2_e05", 0.8, 1.5
CONTENT_LORA, CONTENT_S1, CONTENT_S2 = "ltxdeepthroat_v01", 1.0, 1.0
WIDTH, HEIGHT, FRAMES, FPS = 1216, 832, 241, 24

# One fixed seed for every variant. Without this, run-to-run variation swamps the lever —
# the effect being chased is a few points of mean difference.
SEED = 2779862524661820 % (2 ** 31 - 1)

STAGE2_SIGMAS_NODE = "360"      # ManualSigmas, stage 2: "0.85, 0.7250, 0.4219, 0.0"
PREPROCESS_NODE = "162"         # LTXVPreprocess.img_compression

VARIANTS = [
    # label            img_compression   stage-2 sigmas
    ("A_control",      18,               None),
    ("B_imgcomp4",     4,                None),
    ("C_sigma065",     18,               "0.65, 0.55, 0.32, 0.0"),
    ("D_both",         4,                "0.65, 0.55, 0.32, 0.0"),
]


def build(graph_template: dict, image_name: str, img_compression: int, sigmas: str | None) -> dict:
    g = recipe_mod.resolve(
        graph_template, image_name, WIDTH, HEIGHT,
        prompt=PROMPT, negative=NEGATIVE,
        char_lora=CHAR_LORA, char_s1=CHAR_S1, char_s2=CHAR_S2,
        content_lora=CONTENT_LORA, content_s1=CONTENT_S1, content_s2=CONTENT_S2,
        img_compression=img_compression,
    )
    comfy_mod.set_frames(g, FRAMES)
    comfy_mod.set_seed(g, SEED)
    if sigmas is not None:
        node = g.get(STAGE2_SIGMAS_NODE)
        if node is None or node["class_type"] != "ManualSigmas":
            raise SystemExit(f"node {STAGE2_SIGMAS_NODE} is not ManualSigmas — graph changed")
        node["inputs"]["sigmas"] = sigmas
    return g


def measure(mp4: str, start_png: str, limit: int = 60):
    """Per-frame anchor fidelity AND motion, both against explicit references."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(mp4)
    frames = []
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (128, 128)).astype(np.float32))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {mp4}")

    # The start IMAGE, never frames[0]. frames[0] is already a render and is part of what
    # is being measured — using it as the reference is the bug in the previous run.
    ref = cv2.resize(cv2.imread(start_png), (128, 128)).astype(np.float32)
    base = ref.reshape(-1, 3).mean(0)

    rows = []
    for i, f in enumerate(frames):
        shift = f.reshape(-1, 3).mean(0) - base
        rows.append({
            "frame": i,
            "t": round(i / FPS, 3),
            "anchor": round(float(np.abs(f - ref).mean()), 3),
            "motion": round(float(np.abs(f - frames[i - 1]).mean()), 3) if i else 0.0,
            "dB": round(float(shift[0]), 2),
            "dG": round(float(shift[1]), 2),
            "dR": round(float(shift[2]), 2),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", required=True, help="the START IMAGE, i.e. the workdir's kf1.png")
    ap.add_argument("--out", default="/jobs/_exp48")
    ap.add_argument("--only", default="", help="comma-separated variant labels")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cf = comfy_mod.Comfy(os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    # load_workflow takes a Path, not a str.
    template = comfy_mod.load_workflow(Path(
        os.environ.get("LTX_WORKFLOW_RECIPE", "/opt/engine/workflows/ltx23_recipe.api.json")))

    image_name = cf.upload_image(open(args.kf, "rb").read(), "exp48_start.png")
    print(f"start image uploaded as {image_name}", flush=True)

    wanted = [v for v in VARIANTS if not args.only or v[0] in args.only.split(",")]
    summary = []

    for label, imgc, sigmas in wanted:
        print(f"\n=== {label}: img_compression={imgc} sigmas={sigmas or 'default (0.85...)'}",
              flush=True)
        g = build(template, image_name, imgc, sigmas)
        with open(os.path.join(args.out, f"{label}.graph.json"), "w") as fh:
            json.dump(g, fh, indent=1)

        t0 = time.monotonic()
        pid = cf.submit(g)
        entry = cf.wait(pid, timeout_s=3600)
        name = comfy_mod.Comfy.output_video(entry)
        if not name:
            print(f"  {label}: no video in the history entry — skipped", flush=True)
            continue
        mp4 = os.path.join(args.out, f"{label}.mp4")
        with open(mp4, "wb") as fh:
            fh.write(cf.view(name))
        print(f"  rendered in {time.monotonic() - t0:.0f}s -> {mp4}", flush=True)

        rows = measure(mp4, args.kf)
        with open(os.path.join(args.out, f"{label}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        early = [r for r in rows if r["frame"] <= 12]
        late = [r for r in rows if 13 <= r["frame"] <= 40]
        summary.append({
            "variant": label,
            "img_compression": imgc,
            "stage2_sigmas": sigmas or "0.85, 0.7250, 0.4219, 0.0",
            "anchor_f0": rows[0]["anchor"],
            "anchor_mean_0_12": round(sum(r["anchor"] for r in early) / len(early), 3),
            "anchor_mean_13_40": round(sum(r["anchor"] for r in late) / len(late), 3) if late else None,
            "motion_mean_1_40": round(sum(r["motion"] for r in rows[1:41]) / max(len(rows[1:41]), 1), 3),
        })
        print("  " + json.dumps(summary[-1]), flush=True)

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        if summary:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    print("\n=== SUMMARY (anchor lower is better; motion must NOT collapse) ===", flush=True)
    for s in summary:
        print("  %-14s anchor f0=%-7s 0-12=%-7s 13-40=%-7s | motion=%s" % (
            s["variant"], s["anchor_f0"], s["anchor_mean_0_12"],
            s["anchor_mean_13_40"], s["motion_mean_1_40"]), flush=True)
    print("\nA variant only wins if anchor improves AND motion holds. A large anchor gain "
          "with a large motion drop is a frozen clip, not a fixed one.", flush=True)


if __name__ == "__main__":
    main()
