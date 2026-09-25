"""/health has to be actionable, not decorative.

The point of this endpoint is that a timer, a probe or a one-line `curl -sf` can act on it
without parsing prose. That means the STATUS CODE carries the verdict -- the same reason
wanly-gpu-docker's update timer treats an unreadable worker status as busy rather than idle.
A /health that is always 200 can only be read by a human who already suspected something.
"""
import json

import pytest

from wanly_worker import control


class _FakeSup:
    def __init__(self, services):
        self._services = services

    def snapshot(self):
        return self._services


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    # The runner has no GPU and the 2070 does; neither should decide whether this passes.
    monkeypatch.setattr(control, "gpu_snapshot", lambda: None)


def _get():
    import asyncio
    resp = asyncio.run(control.health())
    return resp.status_code, json.loads(bytes(resp.body))


def test_all_ready_is_200_ok(monkeypatch):
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True},
    ]))
    code, body = _get()
    assert code == 200
    assert body["status"] == "ok"
    assert body["services"][0]["name"] == "joycaption"


def test_one_service_down_is_503(monkeypatch):
    """Partially up is the state that must not read as healthy: wanly-api would keep sending
    captions to a port that answers nothing."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True},
        {"name": "other", "ready": False, "running": False},
    ]))
    code, body = _get()
    assert code == 503
    assert body["status"] == "degraded"


def test_no_services_at_all_is_503(monkeypatch):
    """Before the lifespan has started anything, or after it gave up. Not 'ok, zero services'."""
    monkeypatch.setattr(control, "_sup", _FakeSup([]))
    code, _ = _get()
    assert code == 503


def test_it_reports_which_build_it_is(monkeypatch):
    """wanly-gpu-docker#72 was fourteen hours of two boxes running different code with no way
    to tell them apart. The fix was making the build say so; same here."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "joycaption", "ready": True, "running": True}]))
    _, body = _get()
    assert "build" in body


def test_health_reports_running_code_separately_from_the_image(tmp_path, monkeypatch):
    """#116: engine/ and wanly_worker/ are fetched at boot, so the image's sha no longer
    answers 'which code'. Both refs, or #72 is back with a new mechanism."""
    ref = tmp_path / "code_ref"
    ref.write_text("main @ abc1234\n")
    monkeypatch.setattr(control, "CODE_REF_FILE", str(ref))
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "ltx-engine-api", "ready": True, "running": True}]))
    code, body = _get()
    assert code == 200
    assert body["code"] == "main @ abc1234"


def test_no_code_ref_file_means_baked_code(tmp_path, monkeypatch):
    """The fetch failed before writing the ref (or this image predates #116) — say baked,
    don't omit the field."""
    monkeypatch.setattr(control, "CODE_REF_FILE", str(tmp_path / "absent"))
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "ltx-engine-api", "ready": True, "running": True}]))
    _, body = _get()
    assert body["code"].startswith("baked")


def test_gpu_snapshot_never_raises(monkeypatch):
    """It is diagnostic. A missing or broken nvidia-smi must not take /health down with it."""
    import wanly_worker.supervisor as sup_mod
    monkeypatch.setattr(sup_mod.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert sup_mod.gpu_snapshot() is None


# ---------------------------------------------------------------------------------------
# POST /mode (#131): change what the box is doing WITHOUT recreating the container.
#
# MODE as an env var is create-time only, so every flip through it costs a boot and a model
# re-stage. That is long enough that nobody flips, which is how a box stays locked into one
# job. These pin the in-place path.
# ---------------------------------------------------------------------------------------

import asyncio

from fastapi import HTTPException


class _RecordingSup:
    """Records what apply() was asked for; that IS the contract of the route."""

    def __init__(self):
        self.applied = []

    def snapshot(self):
        return []

    async def apply(self, groups, client):
        self.applied.append(list(groups))


@pytest.fixture
def sup(monkeypatch):
    s = _RecordingSup()
    monkeypatch.setattr(control, "_sup", s)
    monkeypatch.setattr(control, "_client", object())
    monkeypatch.setattr(control, "_equipped",
                        ["ltx-engine", "lora-trainer", "image-description", "face-crop"])
    monkeypatch.setattr(control, "_mode", "ltx-engine")
    monkeypatch.setattr(control, "_mode_lock", asyncio.Lock())
    monkeypatch.setattr(control, "_pending", None)
    monkeypatch.setattr(control, "_mode_error", None)
    return s


def _post(mode):
    """Just the accept. The switch itself runs in a task -- see _post_and_settle."""
    async def go():
        out = await control.set_mode(control.ModeRequest(mode=mode))
        task = control._mode_task
        if task is not None and not task.done():
            task.cancel()
        return out
    return asyncio.run(go())


def _post_and_settle(mode):
    """Accept AND let the background switch finish, returning (accept, mode, pending)."""
    async def go():
        out = await control.set_mode(control.ModeRequest(mode=mode))
        if control._mode_task is not None:
            await control._mode_task
        return out, control._mode, control._pending
    return asyncio.run(go())


def test_caption_stops_everything_that_claims_work(sup):
    body, mode, pending = _post_and_settle("caption")
    assert sup.applied == [["image-description", "face-crop"]]
    assert body["changed"] is True
    assert mode == "caption" and pending is None


def test_the_request_RETURNS_before_the_switch_runs(sup):
    """THE WHOLE POINT. Stopping the render daemon lets the segment in flight finish --
    up to ~27 minutes by design, so nothing is destroyed. Waiting for that inside the
    request is what makes a working switch look like a failed one: the caller times out
    and reports an error while the box does exactly what was asked."""
    body = _post("caption")
    assert body["pending"] == "caption"
    assert body["mode"] == "ltx-engine", "it claimed the new mode before the switch ran"
    assert sup.applied == [], "the switch ran inside the request"


def test_a_second_request_for_the_same_switch_is_not_an_error(sup, monkeypatch):
    """A UI that re-sends while it waits must not be told it failed for asking twice."""
    monkeypatch.setattr(control, "_pending", "caption")
    body = asyncio.run(control.set_mode(control.ModeRequest(mode="caption")))
    assert body["pending"] == "caption"
    assert body["changed"] is False


def test_switching_to_a_DIFFERENT_mode_mid_switch_is_a_409(sup, monkeypatch):
    monkeypatch.setattr(control, "_pending", "caption")
    with pytest.raises(HTTPException) as e:
        _post("ltx-engine")
    assert e.value.status_code == 409


def test_a_failed_switch_is_reported_not_swallowed(sup, monkeypatch):
    """By the time it fails there is no request left to raise to, so /health carries it."""
    async def boom(groups, client):
        raise RuntimeError("ComfyUI would not stop")
    monkeypatch.setattr(sup, "apply", boom)
    _post_and_settle("caption")
    assert control._pending is None
    assert "would not stop" in (control._mode_error or "")
    assert control._mode == "ltx-engine", "a failed switch moved the mode anyway"


def test_going_back_starts_the_whole_capability_line_again(sup, monkeypatch):
    monkeypatch.setattr(control, "_mode", "caption")
    _, mode, pending = _post_and_settle("ltx-engine")
    assert sup.applied == [["ltx-engine", "lora-trainer", "image-description", "face-crop"]]
    assert mode == "ltx-engine" and pending is None


def test_asking_for_the_mode_it_is_already_in_changes_nothing(sup):
    """Not merely harmless: apply() would stop and restart services, so a no-op that was
    not a no-op would cost a model reload every time the UI re-sent the current state."""
    body = _post("ltx-engine")
    assert sup.applied == []
    assert body["changed"] is False


def test_an_alias_is_the_same_mode_not_a_change(sup):
    body = _post("render")
    assert sup.applied == []
    assert body["changed"] is False


def test_an_unknown_mode_is_a_400_with_the_real_names(sup):
    with pytest.raises(HTTPException) as e:
        _post("captions")
    assert e.value.status_code == 400
    assert "caption" in str(e.value.detail)
    assert sup.applied == []


def test_a_mode_that_would_leave_nothing_running_is_a_400(sup, monkeypatch):
    """Better than a container that runs nothing, reports healthy and is diagnosed from
    another machine as "captioner unreachable"."""
    monkeypatch.setattr(control, "_equipped", ["ltx-engine", "lora-trainer"])
    with pytest.raises(HTTPException) as e:
        _post("caption")
    assert e.value.status_code == 400
    assert sup.applied == []


def test_before_startup_it_is_503_not_a_crash(monkeypatch):
    monkeypatch.setattr(control, "_sup", None)
    monkeypatch.setattr(control, "_client", None)
    with pytest.raises(HTTPException) as e:
        _post("caption")
    assert e.value.status_code == 503


def test_a_service_stopped_on_purpose_does_not_make_health_degraded(monkeypatch):
    """Every box in caption mode would otherwise answer 503, and every probe that keys on
    the status code would call it dead."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "ltx-engine-api", "ready": False, "running": False, "stopped": True},
        {"name": "image-description", "ready": True, "running": True, "stopped": False},
    ]))
    code, body = _get()
    assert code == 200
    assert body["status"] == "ok"


def test_health_says_what_the_box_can_do_and_what_it_is_doing(monkeypatch):
    """So a caller can offer the other mode without knowing anything about this container."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "image-description", "ready": True, "running": True, "stopped": False}]))
    monkeypatch.setattr(control, "_equipped", ["ltx-engine", "image-description"])
    monkeypatch.setattr(control, "_mode", "caption")
    monkeypatch.setattr(control, "_pending", None)
    monkeypatch.setattr(control, "_mode_error", None)
    _, body = _get()
    assert body["equipped"] == ["ltx-engine", "image-description"]
    assert body["mode"] == "caption"


def test_health_carries_the_switch_in_progress(monkeypatch):
    """So the caller shows "switching" rather than the mode it is not in yet."""
    monkeypatch.setattr(control, "_sup", _FakeSup([
        {"name": "image-description", "ready": True, "running": True, "stopped": False}]))
    monkeypatch.setattr(control, "_mode", "ltx-engine")
    monkeypatch.setattr(control, "_pending", "caption")
    monkeypatch.setattr(control, "_mode_error", None)
    _, body = _get()
    assert body["mode"] == "ltx-engine"
    assert body["pending_mode"] == "caption"


class TestTheCaptionModelAroundASwitch:
    """Warm on the way in, drop on the way out, and the ORDER of the drop is what matters:
    a 20 GB captioner still resident when ComfyUI starts is 20 GB the renderer does not
    have -- the same collision as captioning beside a render, pointed the other way."""

    @pytest.fixture
    def imgsvc(self, monkeypatch):
        from wanly_worker.services.image_description import service as s
        calls = []

        async def warm(client, model=s.MODEL, keep_alive=s.PIN):
            calls.append(("warm", keep_alive))
            return True

        async def release(client, model=s.MODEL):
            calls.append(("release", s.DROP))
            return True

        monkeypatch.setattr(s, "warm", warm)
        monkeypatch.setattr(s, "release", release)
        return calls

    def test_entering_caption_mode_warms_the_model(self, sup, imgsvc):
        _post_and_settle("caption")
        assert ("warm", -1) in imgsvc, "the first caption would pay the 88s cold load"

    def test_leaving_caption_mode_drops_it_BEFORE_the_stack_starts(self, sup, imgsvc,
                                                                   monkeypatch):
        monkeypatch.setattr(control, "_mode", "caption")
        order = []

        async def apply(groups, client):
            order.append("apply")
        monkeypatch.setattr(sup, "apply", apply)

        from wanly_worker.services.image_description import service as s

        async def release(client, model=s.MODEL):
            order.append("release")
            return True
        monkeypatch.setattr(s, "release", release)

        _post_and_settle("ltx-engine")
        assert order == ["release", "apply"], \
            "ComfyUI started while the captioner still held the card"

    def test_entering_render_mode_from_render_mode_drops_nothing(self, sup, imgsvc,
                                                                 monkeypatch):
        """Nothing was pinned, so there is nothing to drop -- and an unload here would be a
        pointless round trip on every no-op."""
        monkeypatch.setattr(control, "_mode", "caption")
        _post_and_settle("ltx-engine")
        before = list(imgsvc)
        monkeypatch.setattr(control, "_mode", "ltx-engine")
        _post_and_settle("ltx-engine")
        assert list(imgsvc) == before

    def test_the_pin_stops_when_the_box_leaves_caption_mode(self, sup, imgsvc, monkeypatch):
        """A pin outliving caption mode would re-pin a 20 GB model onto a rendering card
        every five minutes."""
        _post_and_settle("caption")
        assert control._pin_task is not None
        monkeypatch.setattr(control, "_mode", "caption")
        _post_and_settle("ltx-engine")
        assert control._pin_task is None
