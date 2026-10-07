"""Four modes per 3090 -- render / train / motion / edit -- with a verified unload (#164).

A 3090 holds one big model at a time, so a mode IS the choice of tenant. Three things here are
the ticket:

  * WHICH SERVICES EACH MODE RUNS, pinned. A mode that quietly gains a tenant is the OOM.
  * THE UNLOAD IS CHECKED, not trusted. A 20 GB model left resident by the last mode is how a
    switch becomes an OOM; the switch reads the card and refuses with the numbers.
  * NOTHING THAT ALREADY SAYS `ltx-engine` OR `caption` BREAKS. wanly-api and the console compare
    against those spellings until wanly-api#392 / wanly-console#589.
"""
import asyncio

import pytest
from fastapi import HTTPException

from wanly_worker import control, registry
from wanly_worker.registry import ConfigError, select_mode

#: 3090a's capability line, and 3090b's once it is a full worker.
THE_3090 = ["ltx-engine", "lora-trainer", "image-description", "face-crop", "image-edit"]


# ----------------------------------------------------------------------- the modes


class TestWhatEachModeRuns:
    def test_render(self):
        # The trainer still rides along on its drain until #165 moves it to train mode only.
        assert select_mode(THE_3090, "render") == ["ltx-engine", "lora-trainer", "face-crop"]

    def test_train(self):
        """The trainer and nothing else that claims work, and never the 32B captioner."""
        assert select_mode(THE_3090, "train") == ["lora-trainer", "face-crop"]

    def test_motion(self):
        assert select_mode(THE_3090, "motion") == ["image-description", "face-crop"]

    def test_edit(self):
        assert select_mode(THE_3090, "edit") == ["face-crop", "image-edit"]

    def test_scene_caption_and_face_crop_stay_in_every_mode(self):
        for mode in registry.MODES:
            names = select_mode(THE_3090 + ["scene-caption"], mode)
            assert "face-crop" in names and "scene-caption" in names, mode

    def test_no_mode_has_two_tenants(self):
        """The whole point: at most one of the 14-21 GB tenants per mode."""
        tenants = {"ltx-engine", "image-description", "image-edit"}
        for mode in registry.MODES:
            assert len(tenants & set(select_mode(THE_3090, mode))) <= 1, mode

    def test_train_needs_the_trainer(self):
        with pytest.raises(ConfigError, match="lora-trainer"):
            select_mode(["ltx-engine", "image-description", "face-crop"], "train")


class TestTheOldNamesStillWork:
    @pytest.mark.parametrize("old,new", [
        ("ltx-engine", "render"), ("engine", "render"), ("caption", "motion"),
        ("image-description", "motion"), ("image-edit", "edit"), ("full-edit", "edit"),
        ("trainer", "train"), ("lora-trainer", "train"), ("", "render"), (None, "render"),
    ])
    def test_alias(self, old, new):
        assert registry.canonical_mode(old) == new
        assert select_mode(THE_3090, old) == select_mode(THE_3090, new)

    def test_the_reported_spelling_is_the_one_the_api_compares(self):
        """wanly-api's captioner refusal keys on `ltx-engine`; the console toggle on `caption`."""
        assert [registry.legacy_name(m) for m in registry.MODES] == \
            ["ltx-engine", "train", "caption", "edit"]
        assert registry.legacy_name(None) is None


# ----------------------------------------------------------------------- the switch


class _Sup:
    def __init__(self):
        self.applied = []

    def snapshot(self):
        return []

    async def apply(self, groups, client):
        self.applied.append(list(groups))


@pytest.fixture
def box(monkeypatch):
    s = _Sup()
    monkeypatch.setattr(control, "_sup", s)
    monkeypatch.setattr(control, "_client", object())
    monkeypatch.setattr(control, "_equipped", list(THE_3090))
    monkeypatch.setattr(control, "_mode", "render")
    monkeypatch.setattr(control, "_active", select_mode(THE_3090, "render"))
    monkeypatch.setattr(control, "_mode_lock", asyncio.Lock())
    monkeypatch.setattr(control, "_pending", None)
    monkeypatch.setattr(control, "_mode_error", None)
    monkeypatch.setattr(control, "_queue", None)
    monkeypatch.setattr(control, "_last_unload", None)
    monkeypatch.setattr(control, "_training_now", lambda: None)
    monkeypatch.setattr(control, "MODE_SWITCH_UNLOAD_TIMEOUT_S", 0.0)

    async def no_free():
        return None
    monkeypatch.setattr(control, "_free_comfyui", no_free)
    from wanly_worker.services.lora_trainer import gpu
    monkeypatch.setattr(gpu, "SOLE_TENANT", False)
    return s


def _vram(monkeypatch, *readings):
    """The card reads these in turn, then the last one forever."""
    seq = list(readings)

    def read():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    monkeypatch.setattr(control, "_vram_used", read)


def _settle(mode):
    async def go():
        await control.set_mode(control.ModeRequest(mode=mode))
        if control._mode_task is not None:
            await control._mode_task
        if control._edit_watch is not None:
            control._edit_watch.cancel()
        return control._mode, control._mode_error
    return asyncio.run(go())


class TestTheUnloadIsVerified:
    def test_stop_then_check_then_start(self, box, monkeypatch):
        _vram(monkeypatch, 21000, 900)
        mode, err = _settle("train")
        assert (mode, err) == ("train", None)
        assert box.applied == [["lora-trainer", "face-crop"], ["lora-trainer", "face-crop"]]
        assert control._last_unload["found_mib"] == 21000
        assert control._last_unload["after_mib"] == 900
        assert control._last_unload["ok"] is True

    def test_a_card_that_did_not_empty_is_refused_and_the_box_put_back(self, box, monkeypatch):
        """Starting the new tenant beside 14 GB the old one left behind IS the OOM."""
        _vram(monkeypatch, 21000, 14000)
        mode, err = _settle("motion")
        assert mode == "render"
        assert "14000 MiB" in err and "not starting motion" in err
        assert box.applied[-1] == select_mode(THE_3090, "render"), "left the box running nothing"
        assert ["image-description", "face-crop"] not in box.applied, \
            "started the captioner on a card that did not empty"
        assert control._last_unload["ok"] is False

    def test_the_limit_is_configurable_for_a_card_with_another_tenant(self, box, monkeypatch):
        monkeypatch.setattr(control, "MODE_SWITCH_VRAM_MAX_MIB", 12000)
        _vram(monkeypatch, 21000, 9000)
        assert _settle("motion") == ("motion", None)

    def test_an_unreadable_card_proceeds_loudly(self, box, monkeypatch, capsys):
        _vram(monkeypatch, None)
        assert _settle("motion") == ("motion", None)
        assert "NOT verified" in capsys.readouterr().out
        assert control._last_unload["ok"] is None

    def test_comfyui_is_asked_to_free_when_render_leaves(self, box, monkeypatch):
        _vram(monkeypatch, 21000, 900)
        freed = []

        async def free():
            freed.append(True)
        monkeypatch.setattr(control, "_free_comfyui", free)
        _settle("edit")
        assert freed

    def test_a_switch_that_stops_nothing_does_not_check(self, box, monkeypatch):
        """A standing box going between modes that keep everything has nothing to unload."""
        monkeypatch.setattr(control, "_equipped", ["image-edit", "scene-caption"])
        monkeypatch.setattr(control, "_active", ["image-edit", "scene-caption"])
        monkeypatch.setattr(control, "_mode", "edit")
        _vram(monkeypatch, 20000)
        assert _settle("render") == ("render", None)
        assert control._last_unload is None


class TestNeverMidRun:
    def test_a_switch_during_training_is_a_409_naming_the_run(self, box, monkeypatch):
        monkeypatch.setattr(control, "_training_now", lambda: "Joana v3 (step 65/1260)")
        with pytest.raises(HTTPException) as e:
            asyncio.run(control.set_mode(control.ModeRequest(mode="motion")))
        assert e.value.status_code == 409
        assert "Joana v3" in e.value.detail
        assert box.applied == []

    def test_asking_for_the_current_mode_during_training_is_not_an_error(self, box, monkeypatch):
        monkeypatch.setattr(control, "_training_now", lambda: "Joana v3 (step 65/1260)")
        body = asyncio.run(control.set_mode(control.ModeRequest(mode="ltx-engine")))
        assert body["changed"] is False

    def test_a_run_claimed_between_accept_and_switch_stops_the_switch(self, box, monkeypatch):
        state = {"n": 0}

        def training():
            state["n"] += 1
            return None if state["n"] == 1 else "Joana v3 (step 1/1260)"
        monkeypatch.setattr(control, "_training_now", training)
        _vram(monkeypatch, 900)
        mode, err = _settle("motion")
        assert mode == "render" and "Joana v3" in err
        assert box.applied == []


class TestTrainMode:
    def test_the_trainer_has_the_card_to_itself_in_train_mode(self, box, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu
        _vram(monkeypatch, 21000, 900)
        _settle("train")
        assert gpu.SOLE_TENANT is True
        _settle("render")
        assert gpu.SOLE_TENANT is False

    def test_the_poller_claims_only_with_the_trainer_running_and_no_switch(self, box,
                                                                          monkeypatch):
        assert control._trainer_may_claim() is True             # render: rides along
        monkeypatch.setattr(control, "_pending", "motion")
        assert control._trainer_may_claim() is False            # a switch is under way
        monkeypatch.setattr(control, "_pending", None)
        monkeypatch.setattr(control, "_active", select_mode(THE_3090, "motion"))
        assert control._trainer_may_claim() is False            # motion: the captioner's card

    def test_in_train_mode_the_worker_id_is_the_live_registrar(self, box, monkeypatch):
        """The daemon's id file still names the row it deleted on its way out."""
        class Q:
            worker_id = "live-row"

            @staticmethod
            def render_daemon_registers():
                return False
        monkeypatch.setattr(control, "_queue", Q())
        assert control._worker_id() == "live-row"


class TestTheModeIsReportedBothWays:
    def test_post_mode_answers_in_both_spellings(self, box, monkeypatch):
        _vram(monkeypatch, 900)
        body = asyncio.run(control.set_mode(control.ModeRequest(mode="caption")))
        control._mode_task.cancel()
        assert (body["mode"], body["pending"]) == ("ltx-engine", "caption")
        assert (body["mode_name"], body["pending_mode_name"]) == ("render", "motion")

    def test_health_lists_the_modes_this_box_can_enter(self, box, monkeypatch):
        monkeypatch.setattr(control, "_equipped", ["ltx-engine", "lora-trainer", "face-crop"])
        assert control._available_modes() == ["render", "train", "motion"]
        monkeypatch.setattr(control, "_equipped", list(THE_3090))
        assert control._available_modes() == list(registry.MODES)


# ----------------------------------------------------------------------- the trainer side


class TestThePollerGate:
    def test_no_claim_outside_a_mode_that_trains(self, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as P
        from wanly_worker.services.lora_trainer import app as trainer_app
        asked = []

        class C:
            async def get(self, url, **kw):
                asked.append(url)
                raise AssertionError("claimed outside a mode that trains")

        p = P.Poller(C(), lambda: "w-1", may_claim=lambda: False)

        async def nothing():
            return None
        monkeypatch.setattr(p, "check_publish_requests", nothing)
        monkeypatch.setattr(trainer_app.STORE, "claim_slot", lambda: True)
        asyncio.run(p.tick())
        assert asked == []


class TestAcquireInTrainMode:
    def test_nothing_to_drain_and_the_card_is_checked(self, monkeypatch):
        """No daemon is running, so there is nothing of ours to drain -- and looking for one
        anyway would find another box's render worker and drain THAT."""
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "SOLE_TENANT", True)
        monkeypatch.setattr(gpu, "vram_used_mib", lambda: 900)

        class C:
            async def get(self, *a, **k):
                raise AssertionError("looked for a render worker to drain in train mode")
        assert asyncio.run(gpu.acquire(C(), "3090")) == ""

    def test_a_card_taken_since_the_switch_is_refused(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "SOLE_TENANT", True)
        monkeypatch.setattr(gpu, "vram_used_mib", lambda: 20000)
        with pytest.raises(RuntimeError, match="train mode"):
            asyncio.run(gpu.acquire(object(), "3090"))
