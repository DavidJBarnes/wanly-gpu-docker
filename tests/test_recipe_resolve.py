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
import re

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
    """The regression line for the RESOLVER: if this hash moves, output moved with it, and
    the symptom would be renders that are subtly different rather than an error.

    The checkpoint is named EXPLICITLY (console#431). It used to be left to the default,
    which quietly made this hash a pin on two separate things — how the resolver patches a
    graph, and which base model happens to be default. When the default moved to 10Eros the
    hash moved with it, and a tripwire that fires on an intended policy change is one that
    gets re-pinned by reflex until it stops guarding anything.

    Verified at the time of the switch: naming sulphur here reproduces the pinned hash
    byte-for-byte, and the only difference against the new default is ckpt_name on nodes
    9500/9501/9502. The resolver itself did not move.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           checkpoint="sulphur_dev_bf16")
    # Pinned against the resolver as it stood BEFORE content strengths were configurable
    # (verified by running both versions side by side over four configurations). A change
    # here is a change to every validated render, so it should require deleting this line.
    assert _hash(g) == "f3c7605015e2515aba55392078f26ff78e32d553df9cb5367e0ebd3851e3e7ea"
    # And the property behind the hash, stated so a legitimate re-pin still has to hold it.
    assert "9601" not in g
    assert "9602" not in g


def test_the_default_changes_nothing_but_the_checkpoint(graph):
    """What moving the default is allowed to do, stated as a test (console#431).

    Exactly three loaders change and nothing else. 2.3 checkpoints are monoliths, so all
    three must move together — a default that reached two of them would load two different
    base models in one render.
    """
    explicit = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                                  checkpoint="sulphur_dev_bf16")
    default = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2")
    differing = {k for k in explicit if explicit[k] != default[k]}
    assert differing == {"9500", "9501", "9502"}
    for nid in differing:
        assert default[nid]["inputs"]["ckpt_name"] == \
            f"{recipe_mod.DEFAULT_CHECKPOINT}.safetensors"


def test_the_default_strengths_are_the_old_hardcode(graph):
    """0.6 on both stages. A caller that names a LoRA and no strengths must get the
    graph the hardcode produced, not a new number chosen while making it configurable."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    assert g["9601"]["inputs"]["strength_model"] == 0.6
    assert g["9602"]["inputs"]["strength_model"] == 0.6


def test_the_two_stages_take_different_strengths(graph):
    """The whole point of the change. Stage 1 decides shape from noise; stage 2 refines the
    2x-upscaled latent. A LoRA that carries motion but degrades anatomy wants to be low
    where shape is decided and higher where detail is."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.3, "s2": 1.1}])
    assert g["9601"]["inputs"]["strength_model"] == 0.3
    assert g["9602"]["inputs"]["strength_model"] == 1.1


def test_a_content_strength_of_zero_is_honoured(graph):
    """0 loads the LoRA and gives it no weight, which is how you measure its contribution.

    It must not be treated as "unset" and replaced with 0.6 — the whole measurement would
    then be of the wrong configuration.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.0, "s2": 0.0}])
    assert g["9601"]["inputs"]["strength_model"] == 0.0
    assert g["9602"]["inputs"]["strength_model"] == 0.0


def test_the_content_lora_is_chained_ahead_of_the_character_lora(graph):
    """Order is the topology, and it is not ours to change: content -> character -> branch.

    If these ever swap, both LoRAs still load and the render still succeeds — at different
    weights against different base latents. That is the kind of change nothing catches.
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    # stage 1: content 9601 feeds char 9621, which feeds branch 337
    assert g["9621"]["inputs"]["model"] == ["9601", 0]
    assert g["337"]["inputs"]["model"] == ["9621", 0]
    # stage 2: same shape on 372
    assert g["9622"]["inputs"]["model"] == ["9602", 0]
    assert g["372"]["inputs"]["model"] == ["9622", 0]


@pytest.mark.parametrize("value", ["none", "NONE", "", None])
def test_none_in_any_spelling_renders_without_a_content_lora(graph, value):
    """"none" is how the stack says off, and it must not become a filename lookup."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_loras=[{"name": value}] if value is not None else [])
    assert "9601" not in g and "9602" not in g


def test_the_extension_is_added_when_missing(graph):
    """The console stores bare names; ComfyUI wants the filename."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    assert g["9601"]["inputs"]["lora_name"] == "sfbehind_LTX2_3_v0_1.safetensors"
    g2 = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                            content_loras=[{"name": "sfbehind_LTX2_3_v0_1.safetensors"}])
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
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.35, "s2": 1.25}])
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
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    note = lora_stack_note(g)
    assert "char none" in note
    assert "content sfbehind_LTX2_3_v0_1.safetensors" in note


def test_equal_strengths_are_not_printed_twice(graph):
    """@0.6 rather than @0.6/0.6 — the common case should be the short one."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1"}])
    assert "content sfbehind_LTX2_3_v0_1.safetensors @0.6" in lora_stack_note(g)
    assert "@0.6/0.6" not in lora_stack_note(g)


def test_a_zero_strength_is_visible_in_the_log(graph):
    """A LoRA loaded at 0 contributes nothing, and the log must not make that look like a
    LoRA doing its job — this is the line someone reads when the output looks unchanged."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "sfbehind_LTX2_3_v0_1", "s1": 0.0, "s2": 0.0}])
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
    assert base_model_note(g) == recipe_mod.DEFAULT_CHECKPOINT


def test_default_checkpoint_is_the_one_a_cold_pod_fetches():
    """The default and the download list must name the same checkpoint (console#431).

    They are separate files in separate languages and nothing but this test connects them.
    It is here rather than in a comment because the failure is silent and expensive: a pod
    that fetched sulphur while the default is 10Eros boots green, reports sulphur through
    its heartbeat, and is then refused every default pose by the API's model gate. It
    claims nothing, and an idle pod is indistinguishable from an empty queue.

    Parsed out of the _WANTED array rather than grepped for the bare name, so that a stale
    mention of the checkpoint in a nearby comment cannot make this pass.
    """
    script = (pathlib.Path(__file__).parent.parent / "download_models.sh").read_text()
    rows = re.findall(r'^\s*"([^"]+\|[^"]*)"\s*$', script, re.MULTILINE)
    fetched = {r.split("|")[1] for r in rows if r.split("|")[0].endswith("diffusion_models")}
    assert fetched == {f"{recipe_mod.DEFAULT_CHECKPOINT}.safetensors"}, (
        f"download_models.sh fetches {fetched}, but the engine defaults to "
        f"{recipe_mod.DEFAULT_CHECKPOINT!r}"
    )


def test_the_note_works_without_any_lora(graph):
    """`ck` used to be defined inside the LoRA loop, so a pose with no character LoRA would
    have raised NameError building its own log line. Reading the graph avoids that."""
    from engine.recipe import base_model_note
    g = recipe_mod.resolve(graph, **BASE, char_lora="none", checkpoint="ltx-2.3-22b-dev")
    assert base_model_note(g) == "ltx-2.3-22b-dev"


# ---------------------------------------------------------------------------------------
# Stacking (console#410). Motion, act and framing are separable, so a pose may want several.
# ---------------------------------------------------------------------------------------


def test_several_content_loras_chain_in_order(graph):
    """Order is part of the configuration, not incidental. The same LoRAs in a different
    order are a different render, so the chain must follow the list exactly."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_loras=[
        {"name": "first", "s1": 0.3, "s2": 0.9},
        {"name": "second", "s1": 0.5, "s2": 1.0},
    ])
    assert g["9601"]["inputs"]["lora_name"] == "first.safetensors"
    assert g["9603"]["inputs"]["lora_name"] == "second.safetensors"
    # first -> second -> char -> branch
    assert g["9601"]["inputs"]["model"] == ["301", 0]
    assert g["9603"]["inputs"]["model"] == ["9601", 0]
    assert g["9621"]["inputs"]["model"] == ["9603", 0]
    assert g["337"]["inputs"]["model"] == ["9621", 0]


def test_each_lora_keeps_its_own_per_stage_strengths(graph):
    """The whole reason for chaining rather than using the rgthree loader at node 301: that
    node is shared across both stages and cannot express a per-stage difference."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_loras=[
        {"name": "a", "s1": 0.3, "s2": 0.9},
        {"name": "b", "s1": 0.5, "s2": 1.2},
    ])
    assert (g["9601"]["inputs"]["strength_model"], g["9602"]["inputs"]["strength_model"]) == (0.3, 0.9)
    assert (g["9603"]["inputs"]["strength_model"], g["9604"]["inputs"]["strength_model"]) == (0.5, 1.2)


def test_content_ids_never_reach_the_character_pair(graph):
    """At the cap of 4 the content chain ends at 9608; the character LoRA lives at
    9621/9622. If those ever collided one would silently overwrite the other."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": f"c{i}"} for i in range(4)])
    assert g["9607"]["inputs"]["lora_name"] == "c3.safetensors"
    assert "9608" in g
    assert g["9621"]["inputs"]["lora_name"] == "k3lly2026_v2.safetensors"


def test_a_none_entry_in_the_list_is_skipped_not_looked_up(graph):
    """"none" is how a pose says off. Looked up it would be a filename lookup for a file
    that does not exist, failing the segment ten minutes into a claim."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_loras=[
        {"name": "none"}, {"name": "real"},
    ])
    assert g["9601"]["inputs"]["lora_name"] == "real.safetensors"
    assert "9603" not in g


def test_the_note_lists_every_content_lora_in_order(graph):
    """A result cannot be tied to a chain that is only half reported."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2", content_loras=[
        {"name": "first", "s1": 0.3, "s2": 0.9},
        {"name": "second", "s1": 0.5, "s2": 1.0},
    ])
    note = lora_stack_note(g)
    assert "content1 first.safetensors @0.3/0.9" in note
    assert "content2 second.safetensors @0.5/1.0" in note
    assert note.index("content1") < note.index("content2")


def test_a_single_lora_is_not_numbered(graph):
    """graph_hash includes _meta, so numbering a lone LoRA "content 1" would change the
    hash of every existing single-LoRA pose — and that hash is the regression trail."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": "only"}])
    assert g["9601"]["_meta"]["title"] == "content stage 1"


@pytest.mark.parametrize("spelling", ["none", "NONE", "none.safetensors",
                                      "None.safetensors", " none ", ""])
def test_no_character_lora_in_any_spelling(graph, spelling):
    """Seen in production 2026-09-04:

        9621 LoraLoaderModelOnly: lora_name: 'none.safetensors' not in [...]

    A bare "none" was always excluded. "none.safetensors" was not — and that is exactly what
    arrives once anything upstream normalises the name before sending it. ComfyUI then
    rejects the whole graph, ten minutes into a claimed segment.

    The engine builds the graph, so it is the last thing that can refuse to build a loader
    for a file called "none".
    """
    g = recipe_mod.resolve(graph, **BASE, char_lora=spelling)
    assert "9621" not in g and "9622" not in g


@pytest.mark.parametrize("spelling", ["none", "none.safetensors", "NONE.SafeTensors"])
def test_no_content_lora_in_any_spelling(graph, spelling):
    """Same trap on the content chain, which normalises names the same way."""
    g = recipe_mod.resolve(graph, **BASE, char_lora="k3lly2026_v2",
                           content_loras=[{"name": spelling}])
    assert "9601" not in g and "9602" not in g


def test_every_line_that_names_the_base_model_reads_it_from_the_graph():
    """Both the segment note and its stdout twin report the base model, and they must derive
    it from the same place. They did not.

    The note was fixed to read the resolved graph; the print beside it kept using a local
    `ck` assigned INSIDE the `for lo in job.req.loras` loop — so a render with NO character
    LoRA never bound it and the segment died on its own log line:

        UnboundLocalError: local variable 'ck' referenced before assignment

    after the graph had already been built. Seen in production 2026-09-04, on the very
    "no character" path that had just been made selectable.

    Two lines stating one fact from two sources is the bug. This pins every such line to the
    graph, which is the only thing that knows what will actually render.
    """
    import re

    src = open("engine/app.py").read()
    naming = [ln.strip() for ln in src.split("\n") if re.search(r"base \{", ln)]
    assert naming, "expected at least one line naming the base model"
    for ln in naming:
        assert "base_model_note(graph)" in ln, (
            f"this line derives the base model some other way: {ln}")


# ---------------------------------------------------------------- two people (console#473)

TWO = [{"name": "pay_v2_e05", "s1": 0.8, "s2": 1.5},
       {"name": "david_v1_final", "s1": 0.7, "s2": 1.2}]


def test_a_second_character_chains_after_the_first_on_both_stages(graph):
    g = recipe_mod.resolve(graph, **BASE, char_loras=TWO)
    assert g["9621"]["inputs"]["lora_name"] == "pay_v2_e05.safetensors"
    assert g["9631"]["inputs"]["lora_name"] == "david_v1_final.safetensors"
    assert g["9631"]["inputs"]["model"] == ["9621", 0]
    assert g["337"]["inputs"]["model"] == ["9631", 0]
    assert g["9632"]["inputs"]["model"] == ["9622", 0]
    assert g["372"]["inputs"]["model"] == ["9632", 0]
    assert g["9631"]["inputs"]["strength_model"] == 0.7
    assert g["9632"]["inputs"]["strength_model"] == 1.2
    assert g["9631"]["_meta"]["title"] == "char 2 stage 1"


def test_a_single_character_graph_is_unchanged_by_the_list_form(graph):
    """The hash is the regression trail; the list form of one character must not move it."""
    scalar = recipe_mod.resolve(graph, **BASE, char_lora="pay_v2_e05", char_s1=0.8, char_s2=1.5)
    listed = recipe_mod.resolve(graph, **BASE, char_loras=[TWO[0]])
    assert recipe_mod.graph_hash(scalar) == recipe_mod.graph_hash(listed)
    assert "9631" not in listed and "9632" not in listed


def test_the_second_character_never_collides_with_content_ids(graph):
    contents = [{"name": f"c{i}", "s1": 0.6, "s2": 0.6} for i in range(4)]
    g = recipe_mod.resolve(graph, **BASE, char_loras=TWO, content_loras=contents)
    ids = {"9601", "9602", "9603", "9604", "9605", "9606", "9607", "9608",
           "9621", "9622", "9631", "9632"}
    assert ids <= set(g)
    # content 4 -> char 1 -> char 2 -> branch, on stage 1
    assert g["9621"]["inputs"]["model"] == ["9607", 0]
    assert g["9631"]["inputs"]["model"] == ["9621", 0]
    assert g["337"]["inputs"]["model"] == ["9631", 0]


def test_the_note_names_both_characters(graph):
    g = recipe_mod.resolve(graph, **BASE, char_loras=TWO)
    note = recipe_mod.lora_stack_note(g)
    assert "char pay_v2_e05.safetensors @0.8/1.5" in note
    assert "char2 david_v1_final.safetensors @0.7/1.2" in note


def test_one_person_notes_do_not_grow_a_char2(graph):
    g = recipe_mod.resolve(graph, **BASE, char_loras=[TWO[0]])
    assert "char2" not in recipe_mod.lora_stack_note(g)


@pytest.mark.parametrize("spelling", ["none", "None", "", "none.safetensors"])
def test_a_none_in_the_second_slot_is_skipped(graph, spelling):
    g = recipe_mod.resolve(graph, **BASE, char_loras=[TWO[0], {"name": spelling}])
    assert "9631" not in g
    assert g["337"]["inputs"]["model"] == ["9621", 0]


def test_a_none_in_the_first_slot_still_places_the_second_at_its_own_id(graph):
    """Slot ids are positional after filtering: the one remaining person takes slot 0, so
    its graph is the ordinary one-person graph rather than a lone 9631."""
    g = recipe_mod.resolve(graph, **BASE, char_loras=[{"name": "none"}, TWO[1]])
    assert g["9621"]["inputs"]["lora_name"] == "david_v1_final.safetensors"
    assert "9631" not in g


def test_more_than_two_characters_is_refused(graph):
    with pytest.raises(ValueError, match="room for 2"):
        recipe_mod.resolve(graph, **BASE, char_loras=TWO + [{"name": "third"}])


def test_the_id_table_is_the_only_cap():
    import inspect
    src = inspect.getsource(recipe_mod.resolve)
    assert "len(CHAR_NODE_IDS)" in src
    assert recipe_mod.CHAR_NODE_IDS == ("962", "963")
