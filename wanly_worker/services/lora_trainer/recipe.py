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
CKPT = os.environ.get(
    "LTX_BASE_CKPT",
    f"{MODELS_DIR}/ltx-2.3/diffusion_models/ltx-2.3-22b-dev.safetensors")
GEMMA = os.environ.get("LTX_GEMMA", f"{MODELS_DIR}/ltx-2.3/text_encoders/gemma-3-12b-it")

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


def run_dir(character: str, version: int) -> Path:
    """One directory per VERSION. Never per character.

    Sharing one meant a v2 with fewer images training on v1's leftovers while the console
    reported the new count, reusing v1's latent cache, and overwriting the first few of v1's
    checkpoints while leaving the rest beside them, indistinguishable.
    """
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
    entries = "\n\n".join(
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


def cache_latents_cmd(run: Path) -> list[str]:
    return [
        TRAINER_PYTHON, "-m", "musubi_tuner.ltx2_cache_latents",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", CKPT,
        "--ltx2_mode", "video", "--device", "cuda", "--vae_dtype", "bf16",
    ]


def cache_text_cmd(run: Path) -> list[str]:
    return [
        TRAINER_PYTHON, "-m", "musubi_tuner.ltx2_cache_text_encoder_outputs",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", CKPT,
        "--gemma_root", GEMMA, "--gemma_load_in_8bit",
        "--ltx2_mode", "video", "--device", "cuda",
        "--mixed_precision", "bf16", "--batch_size", "1",
    ]


def train_cmd(run: Path, character: str, version: int, config: dict) -> list[str]:
    """The training invocation. `config` overrides DEFAULTS, and the job carries a snapshot of
    it -- so a change to DEFAULTS cannot retroactively alter a job that is already queued."""
    c = {**DEFAULTS, **(config or {})}
    return [
        TRAINER_ACCELERATE, "launch",
        "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
        f"{TRAINER_DIR}/src/musubi_tuner/ltx2_train_network.py",
        "--mixed_precision", "bf16",
        "--dataset_config", f"{run}/dataset.toml",
        "--ltx2_checkpoint", CKPT,
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


def estimated_epochs(image_count: int, steps: int, repeats: int | None = None) -> int:
    """num_repeats is fixed, so this is not steps/1200 -- it is how many passes over each image
    the run will actually make. A joint run's count is BOTH groups' images summed (the
    caller passes the total), with the repeats that apply to them."""
    per_epoch = max(1, image_count * (repeats or DEFAULTS["num_repeats"]))
    return max(1, steps // per_epoch)
