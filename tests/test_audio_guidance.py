"""Stage 1's audio guidance cfg (wanly-gpu-docker#114).

Audio quality with the 10eros checkpoint was often bad, and the audio branch was running
cfg 7.0 — RuneXX's dev-workflow value — against video's 3.0. Cfg 7 is a plausible
overcooking regime for audio; the shipped default dropped to 5.0, and when recent renders
still sounded poor, to 3.0 (matching video's). These tests pin the value on the node that
actually gets sent: the AUDIO GuiderParameters, which is the FIRST of the two chained
under the MultimodalGuider (VIDEO chains on top of it).

The paired old-vs-new review happens by ear on real segments; a unit test can only stop
the constant from drifting back by accident.
"""
from engine import comfy


def _two_stage_graph():
    """The stage topology of the base workflow: two CFGGuiders, stage 2's latent
    coming through the upsampler."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "103": {"class_type": "CFGGuider",
                "inputs": {"cfg": 1, "model": ["1", 0], "positive": ["9", 0],
                           "negative": ["9", 1]}},
        "129": {"class_type": "CFGGuider",
                "inputs": {"cfg": 1, "model": ["1", 0], "positive": ["9", 0],
                           "negative": ["9", 1]}},
        "117": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["1", 0]}},
        "113": {"class_type": "SamplerCustomAdvanced",
                "inputs": {"guider": ["129", 0], "latent_image": ["109", 0]}},
        "119": {"class_type": "SamplerCustomAdvanced",
                "inputs": {"guider": ["103", 0], "latent_image": ["117", 0]}},
    }


def _guider_params(graph, modality):
    return [n for n in graph.values()
            if n["class_type"] == "GuiderParameters"
            and n["inputs"]["modality"] == modality]


def test_the_shipped_audio_cfg_is_3():
    assert comfy.DEV_AUDIO_GUIDANCE["cfg"] == 3.0


def test_the_audio_node_carries_the_shipped_cfg():
    graph = _two_stage_graph()
    comfy.set_multimodal_guidance(graph)
    (audio,) = _guider_params(graph, "AUDIO")
    assert audio["inputs"]["cfg"] == comfy.DEV_AUDIO_GUIDANCE["cfg"]


def test_video_and_audio_scales_stay_separate():
    """One value for both modalities would silently undo the whole point of the
    MultimodalGuider — the video cfg must not ride along on the audio node."""
    graph = _two_stage_graph()
    comfy.set_multimodal_guidance(graph, video={"cfg": 4.0})
    (audio,) = _guider_params(graph, "AUDIO")
    (video,) = _guider_params(graph, "VIDEO")
    assert audio["inputs"]["cfg"] == 3.0
    assert video["inputs"]["cfg"] == 4.0


def test_an_explicit_audio_override_still_wins():
    graph = _two_stage_graph()
    applied = comfy.set_multimodal_guidance(graph, audio={"cfg": 7.0})
    (audio,) = _guider_params(graph, "AUDIO")
    assert applied["audio"]["cfg"] == 7.0
    assert audio["inputs"]["cfg"] == 7.0
