"""LivePortrait's ExpressionEditor, run in-process -- no ComfyUI.

keyframe-server ran this node inside a ComfyUI server and talked to it over HTTP: upload the
image, submit a three-node graph, poll /history, fetch the output. That was the price of
sharing a ComfyUI with Qwen. Here it would buy nothing and cost a lot: a second ComfyUI on a
card that already has one (the 3090), or a whole ComfyUI on a box that has none (the 2070), an
unpinned ComfyUI master underneath, and a server that keeps the models on the GPU until told
otherwise -- the opposite of what a card shared with A1111 needs.

So the node's own code is imported and called directly, UNCHANGED, at a pinned commit
(Dockerfile: ALP_COMMIT). It touches ComfyUI in exactly two places, and both are shimmed below:

    folder_paths      where the models live, and a temp dir for its crop preview
    comfy.utils       load_torch_file (a safetensors read) and a ProgressBar

Everything that matters -- the keypoint maths in calc_fe, the 256x256 warp/decode, the stitched
composite behind the face mask that leaves every pixel outside it bit-identical -- is the
node's code, byte for byte, which is what keyframe-server measured.

WHAT IS OURS:
  * the device. The node decides once, at first use, via a module global (`cur_device`). We set
    it per request (gpu.choose) and move the five networks with it, so the same process edits
    on the GPU when the card is free and on the CPU when a neighbour needs it.
  * the face detector stays on CPU. ultralytics' `device=""` means "the GPU if there is one",
    which would park a second model and a CUDA context on the card for a ~50 ms job.
  * a no-face preflight using the node's own detector and its own 30 px rule, so a faceless
    image is a clean 422. The node itself would silently warp the WHOLE FRAME as if it were a
    face ("Failed to detect face!!" and carry on). keyframe-server preflit with YuNet, which
    picks a different face on crowded frames than the node does; asking the node's detector
    means the preflight and the edit can never disagree.
  * which face. The node edits the one nearest the horizontal centre; to edit another, app.py
    hands it a crop planned by faces.py, and `edit(expect_box=...)` re-asks the same detector
    on that crop so a wrong plan is refused rather than warping the wrong person (#553).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types

import numpy as np

from wanly_worker.services.face_edit import faces as picking
from wanly_worker.services.face_edit.expression import Expression

#: The pinned ComfyUI-AdvancedLivePortrait checkout (Dockerfile).
NODE_DIR = os.environ.get("FACE_EDIT_NODE_DIR", "/opt/face-edit/ComfyUI-AdvancedLivePortrait")
#: liveportrait/*.safetensors and ultralytics/face_yolov8n.pt, baked into the image.
MODELS_DIR = os.environ.get("FACE_EDIT_MODELS_DIR", "/opt/face-edit/models")
SCRATCH = os.environ.get("FACE_EDIT_SCRATCH", "/tmp/face-edit")

LIVEPORTRAIT_MODELS = (
    "appearance_feature_extractor", "motion_extractor", "warping_module",
    "spade_generator", "stitching_retargeting_module",
)
DETECTOR_MODEL = "face_yolov8n.pt"
#: The node's detector skips boxes narrower than this (nodes.py detect_face).
MIN_FACE_PX = picking.MIN_FACE_PX


def model_paths() -> list[str]:
    return ([os.path.join(MODELS_DIR, "liveportrait", f"{m}.safetensors")
             for m in LIVEPORTRAIT_MODELS]
            + [os.path.join(MODELS_DIR, "ultralytics", DETECTOR_MODEL)])


def missing_models() -> list[str]:
    return [p for p in model_paths() if not os.path.isfile(p)]


class NoFace(ValueError):
    """The node's own detector found nothing it would warp."""


class NotIsolated(ValueError):
    """On the crop it was given, the node would edit a face other than the chosen one."""


def _install_shims() -> None:
    """The two ComfyUI modules the node imports, reduced to what it calls.

    Installed unconditionally: this runs in the face-edit child process, which never imports
    the real ComfyUI, and a half-real folder_paths (ComfyUI on sys.path by accident) would
    resolve models against ComfyUI's tree instead of the baked ones.
    """
    import torch

    for d in ("output", "temp"):
        os.makedirs(os.path.join(SCRATCH, d), exist_ok=True)

    fp = types.ModuleType("folder_paths")
    fp.models_dir = MODELS_DIR
    fp.output_directory = os.path.join(SCRATCH, "output")
    fp.get_folder_paths = lambda name: [os.path.join(MODELS_DIR, name)]
    fp.get_temp_directory = lambda: os.path.join(SCRATCH, "temp")
    fp.get_save_image_path = lambda *a, **k: None
    sys.modules["folder_paths"] = fp

    def load_torch_file(path, *a, **k):
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)

    class ProgressBar:
        def __init__(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

        def update_absolute(self, *a, **k):
            pass

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = load_torch_file
    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils


def _import_node():
    """Import the node package from its directory (the name has hyphens, so not `import`)."""
    name = "advanced_live_portrait"
    if f"{name}.nodes" in sys.modules:
        return sys.modules[f"{name}.nodes"]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(NODE_DIR, "__init__.py"), submodule_search_locations=[NODE_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return sys.modules[f"{name}.nodes"]


class Engine:
    """One pipeline per process, one edit at a time (the caller holds `lock`)."""

    def __init__(self):
        self.nodes = None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.load_error: str | None = None
        self.loaded_in_s: float | None = None
        self.last_used = 0.0
        #: torch's own peak for the most recent GPU edit, in MiB. The measurement the VRAM
        #: estimate in gpu.py is waiting for.
        self.vram_peak_mib: int | None = None

    @property
    def loaded(self) -> bool:
        return self.nodes is not None

    def load(self) -> None:
        """Weights into RAM, on CPU. Moved to the GPU per request, and back when idle."""
        if self.nodes is not None:
            return
        t0 = time.time()
        try:
            # ultralytics writes a settings file at import and falls back to a noisy warning
            # when the directory is missing; YOLO_OFFLINE stops its update/analytics checks,
            # which would otherwise make an edit wait on the network.
            os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(SCRATCH, "ultralytics"))
            os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
            os.environ.setdefault("YOLO_OFFLINE", "True")
            missing = missing_models()
            if missing:
                raise FileNotFoundError(f"models missing: {', '.join(missing)}")
            import torch

            _install_shims()
            nodes = _import_node()
            nodes.cur_device = torch.device("cpu")
            eng = nodes.g_engine
            # Overwrite one preview file rather than writing preview1..N forever: the node
            # saves a crop preview to the temp dir on every run, for ComfyUI's UI.
            eng.get_temp_img_name = lambda: "expression_edit_preview.png"
            det = eng.get_detect_model()

            def bboxes_on_cpu(image_rgb):
                pred = det(image_rgb, conf=0.7, device="cpu", verbose=False)
                return pred[0].boxes.xyxy.cpu().numpy()

            eng.get_face_bboxes = bboxes_on_cpu
            eng.get_pipeline()
            self.nodes = nodes
            self.device = "cpu"
            self.load_error = None
            self.loaded_in_s = round(time.time() - t0, 1)
        except Exception as e:  # reported by /health and by every edit until it works
            self.load_error = f"{type(e).__name__}: {e}"
            raise

    # ------------------------------------------------------------------ device

    def _modules(self):
        p = self.nodes.g_engine.pipeline
        return (p.appearance_feature_extractor, p.motion_extractor, p.warping_module,
                p.spade_generator, p.stitching_retargeting_module["stitching"])

    def to(self, device: str) -> None:
        if device == self.device:
            return
        import torch

        for m in self._modules():
            m.to(device)
        self.nodes.cur_device = torch.device(device)
        self.device = device
        if device == "cpu" and torch.cuda.is_available():
            # Hand the blocks back to the driver, not just to torch's cache: the neighbour
            # (A1111, ComfyUI, ollama) is a different process and cannot use torch's cache.
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------- edit

    def detect(self, rgb: np.ndarray) -> list[list[float]]:
        """The node's detector's raw boxes, in its order -- the order its tie-break uses."""
        return [list(map(float, b[:4])) for b in self.nodes.g_engine.get_face_bboxes(rgb)]

    def faces(self, rgb: np.ndarray) -> int:
        return len(picking.valid(self.detect(rgb)))

    def edit(self, rgb: np.ndarray, exp: Expression, *, face_pad: float,
             src_ratio: float, expect_box: list[float] | None = None) -> np.ndarray:
        """HxWx3 uint8 RGB in, the node's full-frame composite out (same size, same dtype).

        `expect_box` (in rgb's coordinates) is the face a crop was planned around: the node
        must pick it on these pixels or nothing is warped (NotIsolated). Call with `lock` held
        and the device already chosen.
        """
        import torch

        raw = self.detect(rgb)
        if not picking.valid(raw):
            raise NoFace("no face detected")
        if expect_box is not None and not picking.isolates(raw, rgb.shape[1], expect_box):
            raise NotIsolated("the chosen face is not the one the node picks on its crop")
        src = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
        on_gpu = self.device == "cuda"
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        out = self.nodes.ExpressionEditor().run(
            **exp.model_dump(),
            src_ratio=src_ratio,
            # No driving image in phase 1; these are the node's defaults and inert without one.
            sample_ratio=1.0, sample_parts="OnlyExpression",
            crop_factor=face_pad, src_image=src,
        )
        if on_gpu:
            self.vram_peak_mib = int(torch.cuda.max_memory_allocated() / 2 ** 20)
        self.last_used = time.time()
        # result[0] is the FULL-FRAME composite. The node's "ui" preview is the face crop only
        # -- keyframe-server's first gotcha, and the reason nothing here reads it.
        img = out["result"][0][0].cpu().numpy()
        # The node built this as uint8/255 (pil2tensor), so rounding recovers its bytes exactly.
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
