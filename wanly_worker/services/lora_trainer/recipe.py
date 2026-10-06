"""The training recipe, and the only copy of it (wanly-services#7).

It came from published LTX-2.3 practice and beat this project's own first attempt -- 622 images,
rank 64, 3000 steps -- on identity in a quarter of the time. Every number here was paid for:

    images        50        published guidance is "20-50 usually enough"; 622 was worse
    rank/alpha    32/32     rank 64 gave minimal improvement, confirmed here
    LR            1e-4      7e-5 was worse
    steps         1200      strong at 750-1000; further just makes it brittle
    repeats       10        50 x 10 = 500 samples/epoch, so 1200 steps = 2.4 epochs
    resolution    1024      a CEILING, not a target -- see bucket_no_upscale below
    preset        video_sa_ca_ff   NOT the default t2v
    profile       --nf4_base, no block swap, no sampling  -> ~14 GB VRAM measured

TWO THINGS THAT LOOK LIKE MISTAKES AND ARE NOT.

`video_sa_ca_ff` rather than the default `t2v`: the dataset is stills and has no audio, and t2v
creates audio-branch weights that never receive a training signal.

`bucket_no_upscale` makes `resolution` a ceiling. Nothing is upscaled to reach 1024, so a
dataset of face crops trains at its native sizes -- p@y's 13 images trained at a median of about
396px and produced a usable identity anyway. Do not spend a day upscaling crops to chase the
bucket. The real sizes are visible ONLY in the latent cache filenames, never in the config.

`num_repeats` is fixed, so EPOCHS SCALE INVERSELY WITH DATASET SIZE. 13 images x 10 = 130
samples/epoch, so 1200 steps is 9.2 epochs rather than 2.4 -- four times the passes over each
image at identical flags. Not wrong, but know which experiment you are running.

And it is fixed PER GROUP, not per run (#145). A regularization group rides at fewer repeats
than the identity it protects, so an epoch is the SUM of images x repeats over every group --
never total images x 10. See `samples_per_epoch`.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Where the trainer checkout and its venv live in the image. Overridable because the same code
#: has to run against a host install during development.
TRAINER_DIR = os.environ.get("TRAINER_DIR", "/opt/ltx-trainer")
TRAINER_PYTHON = os.environ.get("TRAINER_PYTHON", f"{TRAINER_DIR}/venv/bin/python")
TRAINER_ACCELERATE = os.environ.get("TRAINER_ACCELERATE", f"{TRAINER_DIR}/venv/bin/accelerate")

#: Mounted read-only from the host, exactly as the ollama store is: the image carries code, the
#: mount carries weights. The base checkpoint alone is 43 GB.
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")
#: THE BASE A JOB TRAINS AGAINST IS THE JOB'S CHOICE (#145): `config.base_checkpoint` names it,
#: and `checkpoint_path` resolves it. This is only the FALLBACK, for a job whose config predates
#: that field (a retried legacy row) and for the boot-time preflight.
#:
#: 10Eros, not ltx-2.3-22b-dev. The LoRAs render on 10Eros (engine/recipe.py's
#: DEFAULT_CHECKPOINT), and training against the base they will actually be loaded onto is the
#: point. Verified before the switch: same key set and shapes as dev, same config metadata,
#: byte-identical VAE and connectors, and `--ltx_version 2.3 --ltx_version_check_mode error`
#: passes on it -- so no other flag changes. LTX_BASE_CKPT still overrides it.
DEFAULT_BASE_CHECKPOINT = "10Eros_v1.5_bf16"
CKPT = os.environ.get(
    "LTX_BASE_CKPT",
    f"{MODELS_DIR}/ltx-2.3/diffusion_models/{DEFAULT_BASE_CHECKPOINT}.safetensors")
GEMMA = os.environ.get("LTX_GEMMA", f"{MODELS_DIR}/ltx-2.3/text_encoders/gemma-3-12b-it")



def checkpoint_path(name: str | None) -> str:
    """The file a job's `config.base_checkpoint` names, or the fallback when it names none.

    A BARE NAME, the way the engine and the API spell a checkpoint everywhere else
    ("10Eros_v1.5_bf16"); `.safetensors` is tolerated and not doubled. It resolves inside the
    models mount and nowhere else -- anything path-shaped is refused rather than followed,
    because the name arrives from the queue and a claim must not be able to point the trainer
    at an arbitrary file.

    Whether the file EXISTS is the preflight's question, not this one's: resolving and checking
    separately is what lets the preflight say which name it was given and where it looked.
    """
    n = (name or "").strip()
    if not n:
        return CKPT
    if n.endswith(".safetensors"):
        n = n[: -len(".safetensors")]
    if "/" in n or "\\" in n or n.startswith(".") or not n:
        raise ValueError(f"base_checkpoint {name!r} is not a bare checkpoint name")
    return f"{MODELS_DIR}/ltx-2.3/diffusion_models/{n}.safetensors"


#: Where run directories live. One per character per version, never shared -- reusing one is how
#: two versions' images, caches and checkpoints got mixed.
RUNS_DIR = os.environ.get("LORA_RUNS_DIR", "/loras")

DEFAULTS = {
    "network_dim": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "lora_target_preset": "video_sa_ca_ff",
    "num_repeats": 10,
    "seed": 42,
    "steps": 1200,
    "resolution": 1024,
}


#: THE CLIP GROUP (#189): real 5-10 s clips of the character, trained beside the stills so the
#: LoRA sees the face in motion -- expressions changing, head turns -- which no still carries.
#:
#: PROVISIONAL, EVERY NUMBER. None of these has been measured; they are starting points until a
#: run on the 3090 (seconds per step and peak VRAM) sets them. Expect a video step to be much
#: slower than a still one. Why each starts where it does:
#:
#:     resolution     512    NOT the stills' 1024. A 49-frame sample at the 1024 ceiling is far
#:                           past the ~14 GB the stills recipe measured. Still a ceiling
#:                           (bucket_no_upscale is in [general]), and the API caps clips at 768.
#:     target_frames  49     8n+1, the frame counts LTX's VAE packs into whole latent frames.
#:                           Anything else is padded by repeating the last frame -- a frozen
#:                           tail the LoRA would learn as motion.
#:     windows        3      frame_extraction "uniform" + frame_sample K spreads K 49-frame
#:                           windows evenly across the clip, so a 10 s clip trains on its start,
#:                           middle and end rather than its first two seconds only.
#:     num_repeats    5      half the stills' 10: each clip already yields `windows` samples.
#:
#: NO RESAMPLING. The API normalizes every clip to 25 fps, which is musubi's LTX2 target_fps
#: default, so no source_fps is written and no frame is dropped or duplicated.
CLIP_DEFAULTS = {
    "resolution": 512,
    "target_frames": 49,
    "windows": 3,
    "num_repeats": 5,
}
#: What a clip group's files are, and what a stills group's are not. musubi picks files by
#: extension per directory kind, so a file of the wrong kind is silently skipped -- and its
#: caption with it -- rather than refused.
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}


def run_dir(character: str, version: int, arch: str = "ltx") -> Path:
    """One directory per VERSION. Never per character.

    Sharing one meant a v2 with fewer images training on v1's leftovers while the console
    reported the new count, reusing v1's latent cache, and overwriting the first few of v1's
    checkpoints while leaving the rest beside them, indistinguishable.

    And one per ARCH (#175): an LTX v1 and an SDXL v1 of the same character are different
    runs, and the stage step's rmtree would otherwise delete one to start the other.
    """
    if arch == "sdxl":
        return Path(RUNS_DIR) / character / f"sdxl-v{version}"
    return Path(RUNS_DIR) / character / f"ltx23b-v{version}"


def dataset_toml(run: Path, groups: list[dict] | None = None) -> str:
    """The dataset config. One `[[datasets]]` entry per identity group.

    `groups` absent (or one entry) is the single-identity shape every run before #102
    wrote, byte-identical — the toml is the regression trail and a single run must not
    look different. A joint run (#102) writes one entry per group, each with its OWN
    data + cache directory and its own num_repeats: the repeats balance the two sets
    (Payton 55 images vs David 50), and per-dataset dirs keep the caches separate
    because a cache is latents for specific images + captions — sharing one across
    groups would pair group 1's latents with group 0's cached text outputs.
    """
    if not groups:
        groups = [{"data": f"{run}/data", "cache": f"{run}/cache",
                   "num_repeats": DEFAULTS["num_repeats"]}]
    entries = "\n\n".join(_clip_entry(g) if g.get("kind") == "clip" else
                           f'[[datasets]]\n'
                           f'image_directory = "{g["data"]}"\n'
                           f'cache_directory = "{g["cache"]}"\n'
                           f'resolution = [{DEFAULTS["resolution"]}, {DEFAULTS["resolution"]}]\n'
                           f'num_repeats = {g["num_repeats"]}'
                           for g in groups)
    return f"""[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

{entries}
"""


def _clip_entry(g: dict) -> str:
    """A clip group's `[[datasets]]` entry (#189): musubi's video dataset, `windows` uniform
    49-frame samples per clip. See CLIP_DEFAULTS for why each number is what it is."""
    res = CLIP_DEFAULTS["resolution"]
    return (f'[[datasets]]\n'
            f'video_directory = "{g["data"]}"\n'
            f'cache_directory = "{g["cache"]}"\n'
            f'resolution = [{res}, {res}]\n'
            f'num_repeats = {g["num_repeats"]}\n'
            f'target_frames = [{CLIP_DEFAULTS["target_frames"]}]\n'
            f'frame_extraction = "uniform"\n'
            f'frame_sample = {g.get("windows") or CLIP_DEFAULTS["windows"]}')


def cache_latents_cmd(run: Path, ckpt: str | None = None) -> list[str]:
    """`ckpt` is the resolved base (`checkpoint_path`). The three stages must name the SAME
    file: latents cached against one base and trained against another is a run that
    completes and learns nothing useful, with no error anywhere."""
    return [
        TRAINER_PYTHON, "-m", "musubi_tuner.ltx2_cache_latents",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", ckpt or CKPT,
        "--ltx2_mode", "video", "--device", "cuda", "--vae_dtype", "bf16",
    ]


def cache_text_cmd(run: Path, ckpt: str | None = None) -> list[str]:
    return [
        TRAINER_PYTHON, "-m", "musubi_tuner.ltx2_cache_text_encoder_outputs",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", ckpt or CKPT,
        "--gemma_root", GEMMA, "--gemma_load_in_8bit",
        "--ltx2_mode", "video", "--device", "cuda",
        "--mixed_precision", "bf16", "--batch_size", "1",
    ]


def train_cmd(run: Path, character: str, version: int, config: dict,
              ckpt: str | None = None) -> list[str]:
    """The training invocation. `config` overrides DEFAULTS, and the job carries a snapshot of
    it -- so a change to DEFAULTS cannot retroactively alter a job that is already queued.

    `ckpt` is the resolved base; absent, it is resolved from the same `config` the rest of the
    flags come from, so a caller cannot get a base that disagrees with the job's."""
    c = {**DEFAULTS, **(config or {})}
    ckpt = ckpt or checkpoint_path(c.get("base_checkpoint"))
    return [
        TRAINER_ACCELERATE, "launch",
        "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
        f"{TRAINER_DIR}/src/musubi_tuner/ltx2_train_network.py",
        "--mixed_precision", "bf16",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", ckpt,
        "--ltx_version", "2.3", "--ltx_version_check_mode", "error",
        "--ltx2_mode", "video", "--nf4_base", "--quantize_device", "cuda",
        "--gradient_checkpointing", "--sdpa",
        "--network_module", "networks.lora_ltx2",
        "--lora_target_preset", str(c["lora_target_preset"]),
        "--network_dim", str(c["network_dim"]),
        "--network_alpha", str(c["network_alpha"]),
        "--learning_rate", str(c["learning_rate"]),
        "--optimizer_type", "AdamW8bit",
        "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "50",
        "--timestep_sampling", "shifted_logit_normal",
        "--max_train_steps", str(c["steps"]),
        "--save_every_n_epochs", "1",
        # A run stopped early resumes from its last epoch rather than restarting. Fifty minutes
        # is long enough that this matters.
        "--save_state", "--save_state_on_train_end",
        "--seed", str(c["seed"]),
        "--output_dir", f"{run}/output",
        # The REAL version, not a hardcoded _v2. That literal is why p@y and l@ura, both first
        # versions, both produced NAME_v2-0000NN.
        "--output_name", f"{character}_v{version}",
    ]


def samples_per_epoch(groups: list[tuple]) -> int:
    """One epoch, in samples (= steps at batch_size 1): SUM over groups of images x repeats.

    `groups` is [(image_count, num_repeats), ...]; a repeats of None is DEFAULTS'. Per group
    because the repeats are (#145) -- a 50-image identity at 10 beside a 200-image
    regularization pool at 1 is 700 samples, and "250 images x 10" would put the epoch at 2500
    and label every checkpoint with a step it was never written at.

    A CLIP GROUP (#189) passes a third element, its `windows`: musubi cuts that many samples
    from every clip, so it counts clips x windows x repeats -- the API's arithmetic, which is
    what keeps the step each checkpoint is labelled with in agreement on both sides. A pair
    (every caller before #189) is windows 1.
    """
    total = 0
    for g in groups:
        n, r = g[0], g[1]
        w = g[2] if len(g) > 2 and g[2] else 1
        total += n * (r or DEFAULTS["num_repeats"]) * w
    return max(1, total)


def estimated_epochs(image_count: int, steps: int, repeats: int | None = None, *,
                     per_epoch: int | None = None) -> int:
    """num_repeats is fixed, so this is not steps/1200 -- it is how many passes over each image
    the run will actually make.

    `per_epoch` is the real figure from `samples_per_epoch` and wins when given: a run whose
    groups repeat differently has no single `repeats` to multiply by. `image_count x repeats`
    remains for a caller (or a persisted job) that only knows the one number."""
    if not per_epoch:
        per_epoch = image_count * (repeats or DEFAULTS["num_repeats"])
    return max(1, steps // max(1, per_epoch))


# ------------------------------------------------------------------------------------------ SDXL
#
# SDXL CHARACTER LoRAs (#175), for the START IMAGES. David generates those in SDXL, and the start
# image is where identity mostly comes from, so this is the same trainer pointed at a second
# architecture -- not a second trainer. `config.arch == "sdxl"` selects it; anything else is LTX,
# which is what every job before #175 is.
#
# THE RECIPE IS THE "aio" RUNS, UNCHANGED (k3lly_aio-1_e8 / k3lly_aio-2_e12, 2026-03-31 and
# 04-02, trained by hand with ~/projects/loras/train_character.sh on 3090a). Every number below
# was read back out of those files' own ss_* metadata, not from the script, because the
# metadata is what actually ran:
#
#     base          BigaspV2Lustify   sha256 a23c0f4f... (ss_new_sd_model_hash)
#     sd-scripts    1a3ec9e           (ss_sd_scripts_commit_hash)
#     rank/alpha    128/64
#     LR            unet 8e-5, text encoder 2e-5, AdamW8bit, cosine_with_restarts x1, warmup 100
#     repeats       8
#     noise         min_snr_gamma 5, noise_offset 0.0357
#     resolution    1024 buckets, bucket_no_upscale -- a ceiling, as on the LTX side
#     clip_skip     1 (Lustify; it would be 2 on a Pony base)
#
# NO MUSUBI. kohya sd-scripts in its own venv, pinned to the versions that trained aio (torch
# 2.5.1+cu124, xformers 0.0.29.post1). It caches latents itself, so the two cache stages the
# LTX path runs do not exist here.
SDXL_DIR = os.environ.get("SDXL_TRAINER_DIR", "/opt/sd-scripts")
SDXL_PYTHON = os.environ.get("SDXL_TRAINER_PYTHON", f"{SDXL_DIR}/venv/bin/python")
SDXL_ACCELERATE = os.environ.get("SDXL_TRAINER_ACCELERATE", f"{SDXL_DIR}/venv/bin/accelerate")
#: Under the same read-only models mount: <host models>/sdxl/<name>.safetensors.
SDXL_MODELS_DIR = f"{MODELS_DIR}/sdxl"
SDXL_DEFAULT_BASE_CHECKPOINT = "BigaspV2Lustify"
#: The WD14 ConvNext v2 tagger (SmilingWolf/wd-v1-4-convnext-tagger-v2): model.onnx and
#: selected_tags.csv. On the mount, not in the image -- it is 390 MB of weights.
WD14_DIR = os.environ.get("WD14_DIR", f"{SDXL_MODELS_DIR}/wd14")

SDXL_DEFAULTS = {
    "network_dim": 128,
    "network_alpha": 64,
    "learning_rate": 8e-5,
    "text_encoder_lr": 2e-5,
    "num_repeats": 8,
    "seed": 42,
    "resolution": 1024,
    "noise_offset": 0.0357,
    "min_snr_gamma": 5.0,
    "clip_skip": 1,
    "lr_warmup_steps": 100,
}

#: Per epoch, for the disk gate. aio's rank-128 files are 1.82 GB in fp32; saved fp16 (see
#: sdxl_train_cmd) that halves. No resume state is written. Rounded up.
SDXL_GB_PER_EPOCH = 1.0


def arch_of(config: dict | None) -> str:
    """`sdxl` or `ltx`. Absent, or anything unrecognised, is LTX: that is what every job before
    #175 was, and a retried legacy row must train exactly as it was created to."""
    return "sdxl" if (config or {}).get("arch") == "sdxl" else "ltx"


def defaults_for(arch: str) -> dict:
    return SDXL_DEFAULTS if arch == "sdxl" else DEFAULTS


def sdxl_checkpoint_path(name: str | None) -> str:
    """`checkpoint_path`, for SDXL: a bare name, resolved under the mount's sdxl/ and nowhere
    else. The same refusal of anything path-shaped, for the same reason -- the name arrives
    from the queue."""
    n = (name or "").strip() or SDXL_DEFAULT_BASE_CHECKPOINT
    if n.endswith(".safetensors"):
        n = n[: -len(".safetensors")]
    if "/" in n or "\\" in n or n.startswith(".") or not n:
        raise ValueError(f"base_checkpoint {name!r} is not a bare checkpoint name")
    return f"{SDXL_MODELS_DIR}/{n}.safetensors"


def base_checkpoint_for(config: dict | None) -> str:
    """The base FILE a job trains against, whichever arch it is."""
    name = (config or {}).get("base_checkpoint")
    if arch_of(config) == "sdxl":
        return sdxl_checkpoint_path(name)
    return checkpoint_path(name)


def sdxl_dataset_toml(run: Path, num_repeats: int, keep_tokens: int = 1,
                      subsets: list[dict] | None = None) -> str:
    """kohya's dataset config.

    SOLO (no `subsets`): ONE subset, byte-identical to what every run before #184 wrote --
    no regularization group, as aio had none. shuffle_caption off and keep_tokens 1, as aio:
    the trigger is the first tag and stays there. caption_dropout 0.

    A PAIR (#184) passes `subsets`, one per group ({data, num_repeats, keep_tokens}): each
    group's own repeats, and keep_tokens covering its whole prefix ("k3lly, d@vid, 1girl,
    1boy" is 4) so nothing the identity binds to is ever shuffled or dropped."""
    if subsets is None:
        body = f"""[[datasets]]
  [[datasets.subsets]]
    image_dir = "{run}/data"
    num_repeats = {num_repeats}
"""
    else:
        body = "[[datasets]]\n" + "\n".join(
            f"""  [[datasets.subsets]]
    image_dir = "{s['data']}"
    num_repeats = {s['num_repeats']}
    keep_tokens = {s['keep_tokens']}
""" for s in subsets)
    return f"""[general]
enable_bucket = true
bucket_no_upscale = true
resolution = {SDXL_DEFAULTS['resolution']}
caption_extension = ".txt"
batch_size = 1
flip_aug = false
color_aug = false
keep_tokens = {keep_tokens}
shuffle_caption = false
caption_dropout_rate = 0.0

{body}"""


def sdxl_train_cmd(run: Path, character: str, version: int, config: dict,
                   ckpt: str | None = None) -> list[str]:
    """The aio invocation, from train_character.sh, with the step count the job asked for.

    --max_train_steps rather than --max_train_epochs: the API sends steps (epochs x samples per
    epoch, so they come out whole), and the progress reader anchors on the step total. The last
    epoch is written WITHOUT a number -- kohya skips the numbered save on the final epoch -- and
    that unnumbered file is the `final` checkpoint, as on the LTX side."""
    c = {**SDXL_DEFAULTS, **(config or {})}
    ckpt = ckpt or sdxl_checkpoint_path(c.get("base_checkpoint"))
    return [
        SDXL_ACCELERATE, "launch",
        "--num_cpu_threads_per_process", "4", "--mixed_precision", "bf16",
        f"{SDXL_DIR}/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={ckpt}",
        f"--dataset_config={run}/dataset.toml",
        f"--output_dir={run}/output",
        # The REAL version, as on the LTX side.
        f"--output_name={character}_v{version}",
        "--save_model_as=safetensors",
        # fp16 ON DISK -- the one departure from aio, and not a training one. aio saved fp32
        # (1.8 GB a file), which at this box's ~0.6 MB/s uplink is ~50 min per checkpoint to S3.
        # Training math is unchanged (bf16 either way) and A1111 casts on load.
        "--save_precision=fp16",
        "--save_every_n_epochs=1",
        f"--max_train_steps={c['steps']}",
        f"--learning_rate={c['learning_rate']}",
        f"--unet_lr={c['learning_rate']}",
        f"--text_encoder_lr={c['text_encoder_lr']}",
        "--lr_scheduler=cosine_with_restarts",
        f"--lr_warmup_steps={c['lr_warmup_steps']}",
        "--lr_scheduler_num_cycles=1",
        "--network_module=networks.lora",
        f"--network_dim={c['network_dim']}",
        f"--network_alpha={c['network_alpha']}",
        "--optimizer_type=AdamW8bit",
        "--mixed_precision=bf16",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--gradient_checkpointing",
        "--max_data_loader_n_workers=2",
        f"--seed={c['seed']}",
        "--max_token_length=225",
        "--xformers",
        "--bucket_no_upscale",
        f"--clip_skip={c['clip_skip']}",
        f"--min_snr_gamma={c['min_snr_gamma']}",
        f"--noise_offset={c['noise_offset']}",
        f"--logging_dir={run}/logs/tb",
    ]
