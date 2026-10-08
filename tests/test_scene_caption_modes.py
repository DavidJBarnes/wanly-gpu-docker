"""Scene captions only in chosen modes (wanly-gpu-docker#199).

Until the 2070 is in, 3090b's one card is shared by everything, scene captions included.
JoyCaption yields to image-edit per edit (SCENE_CAPTION_SHARED) but knows nothing about a
render, so beside a 23 GB LTX render it would OOM the card. SCENE_CAPTION_MODES=edit keeps it
to the mode it can share. Unset keeps it in every mode: right for its own card.
"""
import pytest

from wanly_worker import control, registry
from wanly_worker.registry import ConfigError, scene_caption_modes, select_mode

#: 3090b's interim line: render and edit, with scene captions riding on edit.
THE_3090B = ["ltx-engine", "image-edit", "face-crop", "scene-caption"]


@pytest.fixture
def edit_only(monkeypatch):
    monkeypatch.setenv("SCENE_CAPTION_MODES", "edit")


class TestUnsetIsEveryMode:
    def test_unset_keeps_scene_caption_everywhere(self, monkeypatch):
        """The 2070 case: its own card, nothing to collide with."""
        monkeypatch.delenv("SCENE_CAPTION_MODES", raising=False)
        for mode in ("render", "motion", "edit"):
            assert "scene-caption" in select_mode(THE_3090B, mode), mode

    def test_blank_is_unset(self):
        assert scene_caption_modes("") is None
        assert scene_caption_modes(" , ") is None


class TestEditOnly:
    def test_render_runs_no_scene_captioner(self, edit_only):
        """The OOM this exists to prevent: JoyCaption beside a render."""
        assert select_mode(THE_3090B, "render") == ["ltx-engine", "face-crop"]

    def test_edit_runs_image_edit_and_scene_captions(self, edit_only):
        """The pairing that ran on 3090b for days; the SHARED yield handles the card."""
        assert select_mode(THE_3090B, "edit") == ["image-edit", "face-crop", "scene-caption"]

    def test_unset_mode_is_render(self, edit_only):
        assert "scene-caption" not in select_mode(THE_3090B, None)

    def test_old_names_are_the_same_modes(self, monkeypatch):
        monkeypatch.setenv("SCENE_CAPTION_MODES", "image-edit")
        assert "scene-caption" in select_mode(THE_3090B, "edit")
        assert "scene-caption" not in select_mode(THE_3090B, "ltx-engine")

    def test_several_modes(self, monkeypatch):
        monkeypatch.setenv("SCENE_CAPTION_MODES", "edit, motion")
        assert scene_caption_modes("edit, motion") == {"edit", "motion"}
        assert "scene-caption" in select_mode(THE_3090B + ["image-description"], "motion")

    def test_other_services_untouched(self, edit_only):
        """Only scene-caption is scoped; the mode rules for everything else are #164's."""
        monkeypatch_free = select_mode(["ltx-engine", "image-edit", "face-crop"], "render")
        assert select_mode(THE_3090B, "render") == monkeypatch_free


class TestATypoRefuses:
    def test_unknown_mode_is_a_config_error(self, monkeypatch):
        """A typo would otherwise drop scene captions from every mode and look exactly like a
        captioner that is down."""
        monkeypatch.setenv("SCENE_CAPTION_MODES", "edti")
        with pytest.raises(ConfigError, match="edti"):
            select_mode(THE_3090B, "render")


class TestTheAvailableModesDoNotChange:
    def test_modes_list_is_the_same_with_and_without(self, monkeypatch):
        """Scoping scene captions must not add or remove a mode the console offers."""
        monkeypatch.setattr(control, "_equipped", THE_3090B)
        monkeypatch.delenv("SCENE_CAPTION_MODES", raising=False)
        before = control._available_modes()
        monkeypatch.setenv("SCENE_CAPTION_MODES", "edit")
        assert control._available_modes() == before
        assert "edit" in before and "render" in before
        assert registry.MODES  # sanity: the catalogue itself is untouched
