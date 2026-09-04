"""The recipe resolver: values only, never topology.

`tools/known-good/m-series-validated.graph.json` is the validated DR34 i2v motion recipe.
The rule from CLAUDE.md is that resolve() patches VALUES and never structure — so the tests
that matter here are the ones that would catch a change in what gets rendered, not a change
in what gets returned.

The content LoRA strength was hardcoded 0.6 for both stages until console#395 made it a
per-pose setting. Everything below exists to prove that making it configurable did not move
the default.
"""
import hashlib
import json
import pathlib

import pytest

from engine import recipe as recipe_mod

WORKFLOW = pathlib.Path(__file__).parent.parent / "engine/workflows/ltx23_recipe.api.json"


@pytest.fixture
def graph():
    return json.loads(WORKFLOW.read_text())


def _hash(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


BASE = dict(image_name="kf1.png", width=832, height=1216, prompt="a prompt")


def test_a_pose_with_no_content_lora_is_the_graph_that_was_validated(graph):
    """Every existing pose is this case, so this hash is the regression line.

    If it moves, every validated render moved with it — and the symptom would be output
    that is subtly different rather than an error.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2")
    # Pinned against the resolver as it stood BEFORE content strengths were configurable
    # (verified by running both versions side by side over four configurations). A change
    # here is a change to every validated render, so it should require deleting this line.
    assert _hash(g) == "f3c7605015e2515aba55392078f26ff78e32d553df9cb5367e0ebd3851e3e7ea"
    # And the property behind the hash, stated so a legitimate re-pin still has to hold it.
    assert "9601" not in g
    assert "9602" not in g


def test_the_default_strengths_are_the_old_hardcode(graph):
    """0.6 on both stages. A caller that names a LoRA and no strengths must get the
    graph the hardcode produced, not a new number chosen while making it configurable."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1")
    assert g["9601"]["inputs"]["strength_model"] == 0.6
    assert g["9602"]["inputs"]["strength_model"] == 0.6


def test_the_two_stages_take_different_strengths(graph):
    """The whole point of the change. Stage 1 decides shape from noise; stage 2 refines the
    2x-upscaled latent. A LoRA that carries motion but degrades anatomy wants to be low
    where shape is decided and higher where detail is."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1",
                           content_s1=0.3, content_s2=1.1)
    assert g["9601"]["inputs"]["strength_model"] == 0.3
    assert g["9602"]["inputs"]["strength_model"] == 1.1


def test_a_content_strength_of_zero_is_honoured(graph):
    """0 loads the LoRA and gives it no weight, which is how you measure its contribution.

    It must not be treated as "unset" and replaced with 0.6 — the whole measurement would
    then be of the wrong configuration.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1",
                           content_s1=0.0, content_s2=0.0)
    assert g["9601"]["inputs"]["strength_model"] == 0.0
    assert g["9602"]["inputs"]["strength_model"] == 0.0


def test_the_content_lora_is_chained_ahead_of_the_character_lora(graph):
    """Order is the topology, and it is not ours to change: content -> character -> branch.

    If these ever swap, both LoRAs still load and the render still succeeds — at different
    weights against different base latents. That is the kind of change nothing catches.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1")
    # stage 1: content 9601 feeds char 9621, which feeds branch 337
    assert g["9621"]["inputs"]["model"] == ["9601", 0]
    assert g["337"]["inputs"]["model"] == ["9621", 0]
    # stage 2: same shape on 372
    assert g["9622"]["inputs"]["model"] == ["9602", 0]
    assert g["372"]["inputs"]["model"] == ["9622", 0]


@pytest.mark.parametrize("value", ["none", "NONE", "", None])
def test_none_in_any_spelling_renders_without_a_content_lora(graph, value):
    """"none" is how the stack says off, and it must not become a filename lookup."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_lora=value)
    assert "9601" not in g and "9602" not in g


def test_the_extension_is_added_when_missing(graph):
    """The console stores bare names; ComfyUI wants the filename."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1")
    assert g["9601"]["inputs"]["lora_name"] == "sfbehind_LTX2_3_v0_1.safetensors"
    g2 = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                            content_lora="sfbehind_LTX2_3_v0_1.safetensors")
    assert g2["9601"]["inputs"]["lora_name"] == "sfbehind_LTX2_3_v0_1.safetensors"


# ---------------------------------------------------------------------------------------
# The log line that proves which LoRAs actually loaded.
#
# Built from the RESOLVED GRAPH, not the request. Those differ exactly when something has
# gone wrong — a field silently dropped (the engine request model has no extra="forbid", so
# an older engine ignores unknown keys without complaint), a "none" check that missed, a
# strength that fell back to a default. A log echoing the request would agree with itself
# in precisely the cases worth catching.
# ---------------------------------------------------------------------------------------
# Lives in recipe.py, not app.py: it reads the graph resolve() produced, and app.py
# imports comfy, which a pure test run has no reason to need.
from engine.recipe import lora_stack_note  # noqa: E402


def test_the_note_names_both_loras_and_their_strengths(graph):
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", char_s1=0.8, char_s2=1.5,
                           content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.35, content_s2=1.25)
    note = lora_stack_note(g)
    assert "char k3lly2026_v2.safetensors @0.8/1.5" in note
    assert "content sfbehind_LTX2_3_v0_1.safetensors @0.35/1.25" in note


def test_absence_is_stated_not_implied(graph):
    """"content none" and no mention at all read identically to someone asking whether a
    LoRA loaded. Every pose today is this case, so it is the line that will be read most."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2")
    assert "content none" in lora_stack_note(g)


def test_no_character_lora_is_also_stated(graph):
    g = recipe_mod.resolve(graph, **BASE, char_lora="none",
                           content_lora="sfbehind_LTX2_3_v0_1")
    note = lora_stack_note(g)
    assert "char none" in note
    assert "content sfbehind_LTX2_3_v0_1.safetensors" in note


def test_equal_strengths_are_not_printed_twice(graph):
    """@0.6 rather than @0.6/0.6 — the common case should be the short one."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1")
    assert "content sfbehind_LTX2_3_v0_1.safetensors @0.6" in lora_stack_note(g)
    assert "@0.6/0.6" not in lora_stack_note(g)


def test_a_zero_strength_is_visible_in_the_log(graph):
    """A LoRA loaded at 0 contributes nothing, and the log must not make that look like a
    LoRA doing its job — this is the line someone reads when the output looks unchanged."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.0, content_s2=0.0)
    assert "content sfbehind_LTX2_3_v0_1.safetensors @0.0" in lora_stack_note(g)


def test_the_note_names_the_base_model(graph):
    """A render's record must say which checkpoint it ran on, or two base models cannot be
    told apart afterwards — which is the entire point of making it per-pose."""
    from engine.recipe import base_model_note
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           checkpoint="10Eros_v1.5_bf16")
    assert base_model_note(g) == "10Eros_v1.5_bf16"


def test_the_default_base_model_is_named_too(graph):
    """Not just overrides. A render on the default must say so explicitly, or "no mention"
    and "the default" become indistinguishable in a log."""
    from engine.recipe import base_model_note
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2")
    assert base_model_note(g) == "sulphur_dev_bf16"


def test_the_note_works_without_any_lora(graph):
    """`ck` used to be defined inside the LoRA loop, so a pose with no character LoRA would
    have raised NameError building its own log line. Reading the graph avoids that."""
    from engine.recipe import base_model_note
    g = recipe_mod.resolve(graph, **BASE, char_lora="none", checkpoint="ltx-2.3-22b-dev")
    assert base_model_note(g) == "ltx-2.3-22b-dev"
