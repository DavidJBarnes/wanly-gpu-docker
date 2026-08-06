# Facefusion_comfyui patches

`Facefusion_comfyui` will not work out of the box for this pipeline. These carry the changes
that were living, uncommitted, in the 3090's working tree — the setup only worked because of
edits that existed on exactly one machine and in no repository.

Applied in the Dockerfile after the pinned clone.

## facefusion_comfyui.patch

**`content_filter/content_filter.py`** — one line, `return False`, disabling the NSFW gate.
Stock, the filter blocks this content outright. Note the gate is two files: this detector plus
a self-hash in `core.py`, so a future upgrade that changes the file re-breaks it and the
constant has to be recomputed. Do not substitute someone else's patched fork.

**`facefusion_api/nodes/base.py`** — imports `bytesio_to_image_tensor` / `tensor_to_bytesio`
from a local shim instead of `comfy_api_nodes.util`, which does not exist in the ComfyUI build
we pin. Without this the node fails to IMPORT, so every faceswap workflow dies at prompt
validation rather than at runtime.

**`facefusion_api/swap_local.py`** — uncomments three `print` statements. Diagnostic only, but
they are the only visibility into which face the swap actually selected, and selection is
positional (see below), so keep them.

## comfy_compat.py

The shim `base.py` imports. Copied to `facefusion_api/nodes/comfy_compat.py`; it is untracked
upstream, so it cannot travel in the patch.

## Worth knowing

Face selection in this node is POSITIONAL, not identity-based: `reference_image` and
`reference_face_distance` are declared as node inputs and never passed to the swap, and the
SAME index selects both the source and the target face. It works today because the subject sits
at position 0 under `left-right` ordering. `utils.py` already contains
`find_similar_faces(reference_face, ...)`; it is simply not wired into `swap_local.py`.
