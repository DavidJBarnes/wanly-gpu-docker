"""Face detection and cropping (face-crop; in this image since wanly-gpu-docker#83).

The detection is insightface's and not worth re-testing. What is worth testing is everything
around it, because each piece encodes something the manual pipeline got wrong at least once:

  * face 0 is the LARGEST, not the right one. In a two-person photo the biggest face may be the
    wrong woman, and on p@y an 887px crop of the wrong person survived a by-eye cull.
  * the cos floor is 0.4 and it is what catches that. The raw p@y pool produced a selected
    minimum of -0.042 — a different person entirely, in a set that looked fine.
  * scoring is against the MEAN of a reference set, never a single image: one reference is one
    lighting and one angle.
"""
import pytest

from wanly_worker.services.face_crop import detect as fd


class TestTheGate:
    def test_the_floor_is_buffalo_ls_same_person_range(self):
        assert fd.COS_FLOOR == 0.4

    def test_cosine_of_a_vector_with_itself_is_one(self):
        v = [0.6, 0.8]
        assert fd.cosine(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_faces_score_negative(self):
        """-0.042 is a real number from p@y's pool: a different person entirely."""
        assert fd.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-5)

    def test_a_missing_embedding_scores_below_any_floor(self):
        """An image with no detectable face must never pass the gate by accident."""
        assert fd.cosine([], [1.0, 0.0]) < fd.COS_FLOOR
        assert fd.cosine([1.0, 0.0], []) < fd.COS_FLOOR


class TestTheReferenceMean:
    def test_it_averages_and_renormalises(self):
        """Scored against the mean, never a single image — one reference is one lighting and
        one angle."""
        mu = fd.reference_mean([[1.0, 0.0], [0.0, 1.0]])
        assert fd.cosine(mu, mu) == pytest.approx(1.0, abs=1e-5)
        assert mu[0] == pytest.approx(mu[1], abs=1e-6)

    def test_empty_embeddings_are_ignored_not_counted(self):
        """An image the detector found nothing in must not drag the mean toward zero."""
        assert fd.reference_mean([[1.0, 0.0], []]) == pytest.approx([1.0, 0.0], abs=1e-6)

    def test_no_usable_reference_yields_nothing(self):
        assert fd.reference_mean([[], []]) == []
        assert fd.reference_mean([]) == []


class TestCroppingRules:
    def test_the_crop_is_padded(self):
        """The crops that trained well were not tight to the jaw — a face needs some hair and
        chin to be recognisable."""
        assert fd.PAD > 0

    def test_a_borderline_detection_is_kept_rather_than_dropped(self):
        """The cull is a human's job. A low threshold shows a doubtful face; a high one hides
        it and the set is quietly smaller than the person thinks."""
        assert fd.MIN_DET_SCORE <= 0.5

    def test_faces_are_returned_largest_first_and_indexed(self):
        import inspect
        # Sorted in the finder detect() and measure() share (#206), so "face 0" means the same
        # face to a crop and to a measurement.
        assert "faces.sort" in inspect.getsource(fd._find)
        assert "_find(" in inspect.getsource(fd.measure)
        assert "index=i" in inspect.getsource(fd.detect)

    def test_the_crop_is_square(self):
        """A varying aspect ratio buckets unpredictably, and bucket_no_upscale means the bucket
        is whatever the image already is."""
        import inspect
        assert "side = max(" in inspect.getsource(fd.detect)

    def test_the_crop_is_clamped_not_letterboxed(self):
        """A black border is a feature the model will happily learn."""
        import inspect
        src = inspect.getsource(fd.detect)
        assert "max(0, int(" in src and "min(w, int(" in src


class TestTheService:
    def test_it_is_registered_under_the_flag_name(self):
        from wanly_worker import registry
        assert "face-crop" in registry.KNOWN

    def test_it_binds_all_interfaces(self):
        """Unlike the trainer, wanly-api calls this one across the network — the same way it
        calls joycaption."""
        from wanly_worker.services.face_crop import FaceCrop
        assert "0.0.0.0" in FaceCrop().command()

    def test_details_never_raises(self):
        from wanly_worker.services.face_crop import FaceCrop
        assert isinstance(FaceCrop().details(), dict)

    def test_the_model_cache_is_mounted(self):
        """buffalo_l is ~300 MB and is fetched on first use; in the container it would be
        re-downloaded on every recreate."""
        import pathlib
        s = (pathlib.Path(__file__).parent.parent / "deploy/run-worker.sh").read_text()
        assert "/root/.insightface" in s


class TestTheModelIsLoadedBeforeAnythingAsksForIt:
    """insightface fetches ~300 MB on first use, and first use was inside a request. On a box
    that had never run this, the first crop paid for that download inside wanly-api's HTTP call
    and blew its timeout — which surfaces as `face-crop unreachable` in the console, pointing at
    the network rather than at a model that simply was not there yet. The retry would have
    worked, which is the worst kind of bug to be told about."""

    def test_the_model_is_preloaded_at_startup(self):
        import inspect
        from wanly_worker.services.face_crop import app as mod
        src = inspect.getsource(mod)
        assert 'on_event("startup")' in src
        assert "_analyser" in src

    def test_the_preload_does_not_block_the_event_loop(self):
        """The supervisor's readiness probe times out otherwise, and kills a service that is
        doing exactly what it should be."""
        import inspect
        from wanly_worker.services.face_crop import app as mod
        src = inspect.getsource(mod._warm)
        assert "asyncio.to_thread" in src
        assert "asyncio.create_task" in src

    def test_a_failed_preload_is_not_fatal(self):
        """A container that will not start is worse than one that retries the load on the
        first crop and reports the real error."""
        import inspect
        from wanly_worker.services.face_crop import app as mod
        assert "except Exception" in inspect.getsource(mod._warm)

    def test_health_separates_up_from_ready_to_crop(self):
        from wanly_worker.services.face_crop import detect as fd
        # is_loaded is the distinction; a bare 200 said "ready" while a 300 MB download was
        # still owed.
        assert callable(fd.is_loaded)
        assert isinstance(fd.is_loaded(), bool)

    def test_health_reports_it(self):
        import inspect
        from wanly_worker.services.face_crop import app as mod
        assert "model_loaded" in inspect.getsource(mod.health)


class TestACropHasToFitThroughTheWire:
    """Fourteen group photos produced a ~150 MB response as full-resolution lossless PNG. That
    is minutes on a home uplink, so wanly-api's 300s read timed out while this service had
    already logged `200 OK` — the console said "cropping failed" and both sides' logs looked
    fine."""

    def test_a_crop_is_capped_at_the_trainer_ceiling(self):
        """`resolution` is a ceiling with bucket_no_upscale, so a larger crop is downscaled
        during latent caching anyway. Shipping it across the internet first buys nothing."""
        from wanly_worker.services.face_crop import detect as fd
        from wanly_worker.services.lora_trainer import recipe
        assert fd.MAX_EDGE == recipe.DEFAULTS["resolution"]

    def test_the_cap_is_applied_by_downscaling_not_by_cropping_further(self):
        """Cropping tighter would cut off the hair and chin PAD exists to keep."""
        import inspect
        from wanly_worker.services.face_crop import detect as fd
        src = inspect.getsource(fd.detect)
        assert "cv2.resize" in src
        assert "INTER_AREA" in src  # the right filter for downscaling

    def test_crops_are_jpeg_not_lossless_png(self):
        """The sources are already JPEG out of a phone, so a q95 re-encode is invisible — and
        PNG on photographic content is an order of magnitude larger for no gain a trainer can
        use."""
        import inspect
        from wanly_worker.services.face_crop import detect as fd
        src = inspect.getsource(fd.detect)
        assert '".jpg"' in src
        assert '".png"' not in src
        assert fd.IMAGE_FORMAT == "jpeg"

    def test_the_response_says_which_format_it_is(self):
        """So wanly-api names the object it writes correctly instead of assuming .png."""
        from wanly_worker.services.face_crop.app import CropFace
        assert "format" in CropFace.model_fields

    def test_the_default_format_is_the_old_contract(self):
        """A caller that predates this field assumed png, and must keep working."""
        from wanly_worker.services.face_crop.app import CropFace
        f = CropFace(source_index=0, face_index=0, png_b64="x", width=1, det_score=1.0, yaw=0.0)
        assert f.format == "jpeg"


class TestATightCropIsRetriedPadded:
    """#178: SCRFD missed a 440x440 face crop (det 0.00) that it found at 0.89 with a border.
    The retry happens ONLY when nothing was found -- padding moves embeddings of faces that
    already detect (raw vs padded cosine down to 0.48), so a working image must not be padded."""

    class _Face:
        def __init__(self, bbox, score=0.9):
            import numpy as np
            self.bbox = np.array(bbox, dtype=float)
            self.det_score = score
            self.normed_embedding = np.array([0.6, 0.8])
            self.pose = None

    def _img(self, w=440, h=440):
        import cv2
        import numpy as np
        ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
        return buf.tobytes()

    def _fake(self, monkeypatch, found_plain):
        calls = []
        face = self._Face

        class App:
            def get(self, img):
                calls.append(img.shape[:2])
                padded = img.shape[0] > 440
                if found_plain and not padded:
                    return [face([40, 40, 400, 400])]
                # In the padded frame (220 px border) the face sits 220 px further in.
                return [face([260, 260, 620, 620])] if padded else []
        monkeypatch.setattr(fd, "_analyser", lambda: App())
        return calls

    def test_no_face_plain_is_found_padded_and_mapped_back(self, monkeypatch):
        calls = self._fake(monkeypatch, found_plain=False)
        faces = fd.detect(self._img())
        assert len(faces) == 1
        assert calls == [(440, 440), (880, 880)]   # plain, then one padded retry
        # The crop is cut from the ORIGINAL: no wider than the 440 px image, no border.
        assert faces[0].width <= 440
        assert faces[0].embedding == [0.6, 0.8]

    def test_an_image_that_already_works_is_never_padded(self, monkeypatch):
        calls = self._fake(monkeypatch, found_plain=True)
        assert len(fd.detect(self._img())) == 1
        assert calls == [(440, 440)]

    def test_the_retry_can_be_turned_off(self, monkeypatch):
        calls = self._fake(monkeypatch, found_plain=False)
        monkeypatch.setattr(fd, "RETRY_PAD", 0.0)
        assert fd.detect(self._img()) == []
        assert calls == [(440, 440)]


class TestHeadAndShoulders:
    """#187: a portrait framing beside the tight face crop -- just above the hairline down to
    the collarbone and upper chest, 4:5. The face crop must not move a pixel."""

    # A 100 px face box (brow to chin) in the middle of a big photo.
    BOX = (450.0, 300.0, 530.0, 400.0)

    def test_it_reaches_above_the_hairline_and_down_to_the_chest(self):
        left, top, right, bottom = fd.head_shoulders_box(*self.BOX, 2000, 2000)
        assert top == 300 - 60                     # 0.6 face heights above the box
        assert bottom == 400 + 150                 # 1.5 face heights below the chin
        assert (right - left) / (bottom - top) == pytest.approx(0.8, abs=0.01)
        assert (left + right) / 2 == pytest.approx(490, abs=1)   # centred on the face

    def test_at_an_edge_it_slides_and_keeps_its_shape(self):
        """Clamping would cut a shoulder off and change the aspect crop to crop."""
        left, top, right, bottom = fd.head_shoulders_box(10, 10, 90, 110, 2000, 2000)
        assert (left, top) == (0, 0)
        assert (right - left) / (bottom - top) == pytest.approx(0.8, abs=0.01)
        assert bottom - top == 310

    def test_a_small_photo_shrinks_from_the_bottom_keeping_the_head(self):
        # 310 px of window in a 250 px tall photo: the head stays, the chest goes.
        left, top, right, bottom = fd.head_shoulders_box(*self.BOX, 2000, 250 + 300 - 60)
        assert top <= 240 and bottom <= 490
        assert (right - left) / (bottom - top) == pytest.approx(0.8, abs=0.01)
        assert 0 <= left and right <= 2000

    def test_it_never_leaves_the_image(self):
        for box, (w, h) in [((0, 0, 200, 300), (220, 320)), ((900, 900, 1000, 1000), (1000, 1000))]:
            left, top, right, bottom = fd.head_shoulders_box(*box, w, h)
            assert 0 <= left < right <= w and 0 <= top < bottom <= h

    def _fake_detector(self, monkeypatch):
        import numpy as np

        class F:
            bbox = np.array(self.BOX)
            det_score = 0.9
            normed_embedding = np.array([0.6, 0.8])
            pose = None

        class App:
            def get(self, img):
                return [F()]
        monkeypatch.setattr(fd, "_analyser", lambda: App())

    def _img(self):
        import cv2
        import numpy as np
        ok, buf = cv2.imencode(".jpg", np.full((1000, 1000, 3), 128, np.uint8))
        return buf.tobytes()

    def test_detect_cuts_a_portrait_and_the_face_crop_is_unchanged(self, monkeypatch):
        import cv2
        import numpy as np
        self._fake_detector(monkeypatch)
        face = fd.detect(self._img())[0]
        portrait = fd.detect(self._img(), "head_shoulders")[0]
        shape = lambda f: cv2.imdecode(np.frombuffer(f.png, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        assert shape(face) == (140, 140)           # square, 1.4 x the 100 px box, as ever
        h, w = shape(portrait)
        assert (h, w) == (310, 248)
        assert portrait.embedding == face.embedding   # same face, same score

    def test_an_unknown_framing_is_refused(self):
        with pytest.raises(ValueError):
            fd.detect(b"", "full_body")

    def test_the_service_takes_and_echoes_it(self, monkeypatch):
        """An older service ignores `framing` and returns face crops; the echo is how the API
        tells. Absent on the request means face, as before."""
        import asyncio
        import base64
        from wanly_worker.services.face_crop import app as svc
        self._fake_detector(monkeypatch)
        assert svc.CropRequest(images=["x"]).framing == "face"
        req = svc.CropRequest(images=[base64.b64encode(self._img()).decode()],
                              framing="head_shoulders")
        out = asyncio.run(svc.crop(req))
        assert out.framing == "head_shoulders" and out.faces[0].width == 248


# ------------------------------------------------------------------ #206: measure + upscale

class _FakeFace:
    def __init__(self, bbox, score=0.9, pose=None):
        import numpy as np
        self.bbox = np.array(bbox, dtype=float)
        self.det_score = score
        self.normed_embedding = np.array([0.6, 0.8])
        self.pose = None if pose is None else np.array(pose, dtype=float)


def _jpeg(w, h):
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    return buf.tobytes()


def _detector(monkeypatch, faces):
    class App:
        def get(self, img):
            return list(faces)
    monkeypatch.setattr(fd, "_analyser", lambda: App())


class TestMeasureAtTrainingSize:
    """wanly-api#431: 29 of Joana v3's 48 faces were under 250 px AT TRAINING SIZE. The trainer
    scales an image down to the 1024^2 area and never up (bucket_no_upscale), so the number
    that matters is the face's height after that -- not in the photograph."""

    def test_a_phone_photo_is_scaled_down_to_the_training_area(self):
        # 1080x1440: the v3 phone shots.
        assert fd.train_scale(1080, 1440) == pytest.approx((1024 * 1024 / (1080 * 1440)) ** 0.5)

    def test_a_small_image_is_never_scaled_up(self):
        """A 254 px close-up trains at 254 px -- the whole reason upscaling exists."""
        assert fd.train_scale(254, 373) == 1.0
        assert fd.train_scale(1024, 1024) == 1.0

    def test_the_area_matches_the_trainer(self):
        from wanly_worker.services.lora_trainer import recipe
        assert fd.TRAIN_EDGE == recipe.DEFAULTS["resolution"] == recipe.SDXL_DEFAULTS["resolution"]

    def test_it_reports_face_height_at_training_size_and_pose(self, monkeypatch):
        # A 300 px face (y 100..400) in a 2048x2048 photo trains at 150 px.
        _detector(monkeypatch, [_FakeFace([100, 100, 300, 400], pose=[5.0, -20.0, 2.0])])
        m = fd.measure(_jpeg(2048, 2048))
        assert (m["width"], m["height"], m["train_scale"]) == (2048, 2048, 0.5)
        face = m["faces"][0]
        assert face["face_h"] == 300 and face["face_px_at_train"] == 150
        # insightface's pose is (pitch, yaw, roll); yaw is the one detect() has always read.
        assert (face["pitch"], face["yaw"], face["roll"]) == (5.0, -20.0, 2.0)
        assert face["box"] == [100, 100, 300, 400]

    def test_faces_come_back_largest_first(self, monkeypatch):
        _detector(monkeypatch, [_FakeFace([0, 0, 50, 50]), _FakeFace([0, 0, 200, 200])])
        assert [f["face_h"] for f in fd.measure(_jpeg(500, 500))["faces"]] == [200, 50]

    def test_no_face_is_an_empty_list_and_no_pose_is_none(self, monkeypatch):
        _detector(monkeypatch, [])
        monkeypatch.setattr(fd, "RETRY_PAD", 0.0)
        assert fd.measure(_jpeg(400, 300))["faces"] == []
        _detector(monkeypatch, [_FakeFace([0, 0, 100, 100])])
        assert fd.measure(_jpeg(400, 300))["faces"][0]["yaw"] is None

    def test_a_face_found_on_the_padded_retry_is_measured_in_the_original_frame(self, monkeypatch):
        """#178's retry border must not inflate the box -- or the image size."""
        class App:
            def get(self, img):
                return [_FakeFace([300, 300, 500, 500])] if img.shape[0] > 400 else []
        monkeypatch.setattr(fd, "_analyser", lambda: App())
        m = fd.measure(_jpeg(400, 400))   # 200 px border on each side when padded
        assert m["width"] == 400 and m["faces"][0]["box"] == [100, 100, 300, 300]

    def test_bytes_that_are_not_an_image_measure_as_none(self):
        assert fd.measure(b"not an image") is None


class TestUpscalePlanning:
    """Which images Real-ESRGAN touches, and how far. Pure arithmetic, tested without torch."""

    def test_a_small_crop_is_brought_to_the_ceiling_on_its_long_edge(self):
        from wanly_worker.services.face_crop import upscale as up
        assert up.plan(254, 373) == pytest.approx(1024 / 373)

    def test_one_already_near_the_ceiling_is_left_alone(self):
        """900 -> 1024 buys the trainer nothing and costs a model pass and a re-encode."""
        from wanly_worker.services.face_crop import upscale as up
        assert up.plan(720, 900) == 1.0
        assert up.plan(2000, 3000) == 1.0

    def test_the_target_is_the_trainer_ceiling(self):
        from wanly_worker.services.face_crop import upscale as up
        assert up.TARGET_EDGE == fd.MAX_EDGE == 1024

    def test_the_denoise_blend_keeps_mostly_the_wdn_model(self):
        """normal * 0.2 + wdn * 0.8 is the setting validated as keeping freckles. Backwards, it
        is the plastic skin this exists to avoid."""
        from wanly_worker.services.face_crop import upscale as up
        assert up.DENOISE == 0.2
        out = up.blend({"w": 1.0}, {"w": 0.0}, up.DENOISE)
        assert out["w"] == pytest.approx(0.2)

    def test_the_skin_smoothing_rrdb_model_is_not_used(self):
        import inspect
        from wanly_worker.services.face_crop import upscale as up
        assert up.NORMAL == "realesr-general-x4v3.pth" and up.WDN == "realesr-general-wdn-x4v3.pth"
        assert "RRDB" not in inspect.getsource(up._build)

    def test_one_native_pass_at_most(self):
        """A second 4x pass renders 16x internally for detail a tiny face does not have."""
        from wanly_worker.services.face_crop import upscale as up
        assert up.MAX_PASSES == 1

    def test_the_weights_are_baked_into_the_image_and_checksummed(self):
        import pathlib
        from wanly_worker.services.face_crop import upscale as up
        df = (pathlib.Path(__file__).parent.parent / "Dockerfile").read_text()
        assert f"FACE_UPSCALE_MODELS_DIR={up.MODELS_DIR}" in df
        for name in (up.NORMAL, up.WDN):
            assert name.removesuffix(".pth") in df
        assert "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292" in df
        assert "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca" in df

    def test_missing_weights_say_so(self, monkeypatch, tmp_path):
        from wanly_worker.services.face_crop import upscale as up
        monkeypatch.setattr(up, "MODELS_DIR", tmp_path)
        assert up.available() is False


class TestTilingHasNoSeams:
    """Tiles are feathered, not hard-cut. With a net that is exactly nearest-neighbour 4x, the
    tiled result must equal the untiled one to the last pixel."""

    def test_tiled_equals_whole(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        from wanly_worker.services.face_crop import upscale as up
        net = lambda x: F.interpolate(x, scale_factor=4, mode="nearest")
        x = torch.rand(1, 3, 300, 520)
        monkeypatch.setattr(up, "TILE", 128)
        assert torch.allclose(up._run_tiled(net, x), net(x), atol=1e-5)


class TestCropWithUpscale:
    def _fake_up(self, monkeypatch):
        import cv2
        from wanly_worker.services.face_crop import upscale as up
        calls = []

        def fake(img, target):
            calls.append((img.shape[:2], target))
            f = up.plan(img.shape[1], img.shape[0], target)
            if f == 1.0:
                return img
            return cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)))
        monkeypatch.setattr(up, "upscale_bgr", fake)
        monkeypatch.setattr(up, "available", lambda: True)
        return calls

    def _shape(self, face):
        import cv2
        import numpy as np
        return cv2.imdecode(np.frombuffer(face.png, np.uint8), cv2.IMREAD_COLOR).shape[:2]

    def test_a_small_crop_is_upscaled_to_the_ceiling(self, monkeypatch):
        calls = self._fake_up(monkeypatch)
        _detector(monkeypatch, [_FakeFace([450, 300, 530, 400])])   # 100 px face
        face = fd.detect(_jpeg(1000, 1000), "head_shoulders", upscale=True)[0]
        assert face.upscaled is True
        assert max(self._shape(face)) == 1024
        assert calls[0][1] == fd.MAX_EDGE
        # The embedding is the original detection's, not re-run on invented pixels.
        assert face.embedding == [0.6, 0.8]

    def test_without_the_flag_nothing_changes(self, monkeypatch):
        calls = self._fake_up(monkeypatch)
        _detector(monkeypatch, [_FakeFace([450, 300, 530, 400])])
        face = fd.detect(_jpeg(1000, 1000), "head_shoulders")[0]
        assert face.upscaled is False and self._shape(face) == (310, 248) and calls == []

    def test_a_crop_already_large_is_not_upscaled(self, monkeypatch):
        self._fake_up(monkeypatch)
        _detector(monkeypatch, [_FakeFace([1000, 600, 1400, 1100])])   # 500 px face
        face = fd.detect(_jpeg(3000, 3000), "head_shoulders", upscale=True)[0]
        assert face.upscaled is False and max(self._shape(face)) == 1024

    def test_the_service_echoes_upscale(self, monkeypatch):
        """An older service ignores `upscale` and sends plain crops; the echo is how wanly-api
        tells, the same way it tells for framing."""
        import asyncio
        import base64
        from wanly_worker.services.face_crop import app as svc
        self._fake_up(monkeypatch)
        _detector(monkeypatch, [_FakeFace([450, 300, 530, 400])])
        assert svc.CropRequest(images=["x"]).upscale is False
        req = svc.CropRequest(images=[base64.b64encode(_jpeg(1000, 1000)).decode()],
                              framing="head_shoulders", upscale=True)
        out = asyncio.run(svc.crop(req))
        assert out.upscale is True and out.faces[0].upscaled is True

    def test_upscale_without_weights_is_refused_not_ignored(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        from wanly_worker.services.face_crop import app as svc
        from wanly_worker.services.face_crop import upscale as up
        monkeypatch.setattr(up, "available", lambda: False)
        with pytest.raises(HTTPException) as e:
            asyncio.run(svc.crop(svc.CropRequest(images=["x"], upscale=True)))
        assert e.value.status_code == 503


class TestTheEndpoints:
    def test_health_names_what_this_build_can_do(self, monkeypatch):
        """wanly-api checks this before a "Fix small faces" run, so an older service is caught
        before any work, not half way through."""
        import asyncio
        from wanly_worker.services.face_crop import app as svc
        from wanly_worker.services.face_crop import upscale as up
        monkeypatch.setattr(up, "available", lambda: True)
        h = asyncio.run(svc.health())
        assert {"measure", "upscale", "head_shoulders"} <= set(h["features"])
        assert h["upscale_ready"] is True and h["train_edge"] == fd.TRAIN_EDGE

    def test_new_code_in_an_old_image_does_not_claim_upscale(self, monkeypatch):
        """The code is fetched at boot, the weights come with the image: after a restart on an
        image built before #206 this code runs with no weights, and must say so."""
        import asyncio
        from wanly_worker.services.face_crop import app as svc
        from wanly_worker.services.face_crop import upscale as up
        monkeypatch.setattr(up, "available", lambda: False)
        h = asyncio.run(svc.health())
        assert "measure" in h["features"] and "upscale" not in h["features"]

    def test_measure_is_parallel_to_the_images(self, monkeypatch):
        import asyncio
        import base64
        from wanly_worker.services.face_crop import app as svc
        _detector(monkeypatch, [_FakeFace([0, 0, 100, 200])])
        b = lambda x: base64.b64encode(x).decode()
        out = asyncio.run(svc.measure(svc.ImagesRequest(images=[b(_jpeg(500, 500)),
                                                               b(b"junk")])))
        assert out["results"][0]["faces"][0]["face_px_at_train"] == 200
        assert out["results"][1] is None

    def test_whole_image_upscale_skips_what_is_already_big_enough(self, monkeypatch):
        import asyncio
        import base64
        import cv2
        import numpy as np
        from wanly_worker.services.face_crop import app as svc
        TestCropWithUpscale()._fake_up(monkeypatch)
        b = lambda x: base64.b64encode(x).decode()
        out = asyncio.run(svc.upscale(svc.ImagesRequest(images=[b(_jpeg(254, 373)),
                                                               b(_jpeg(900, 900))])))
        small, big = out["images"]
        assert small["upscaled"] is True and (small["width"], small["height"]) == (697, 1024)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(small["b64"]), np.uint8), 1)
        assert img.shape[:2] == (1024, 697)
        assert big["upscaled"] is False and big["b64"] is None


# ------------------------------------------------------------------ wanly-api#436: pair framing

class TestPairFraming:
    """A composition set is captioned with BOTH triggers. A one-face crop under that caption
    teaches the pair LoRA that one face is both people (wanly-api#430), so the composition
    fix crops both people: the union of the two largest faces, hair to chin round each (#209)."""

    # Two 100 px faces side by side, a little apart, in a big photo.
    A = (400.0, 500.0, 480.0, 600.0)
    B = (700.0, 520.0, 780.0, 620.0)

    def _contains(self, win, *boxes):
        left, top, right, bottom = win
        return all(left <= b[0] and top <= b[1] and b[2] <= right and b[3] <= bottom
                   for b in boxes)

    def test_it_holds_both_faces_hair_to_chin(self):
        win = fd.pair_box(self.A, self.B, 3000, 3000)
        left, top, right, bottom = win
        assert self._contains(win, self.A, self.B)
        assert top == 500 - 45                                   # 0.45 face heights above
        assert left == 400 - 24 and right == 780 + 24            # 0.3 face widths either side
        # 428 x 190 hair to chin is past 2:1, so it is deepened to exactly 2:1 -- downward.
        assert (right - left, bottom - top) == (428, 214)

    def test_each_face_gets_room_in_its_own_size(self):
        """The smaller face's hair room is its own: sizing both by the bigger one is area the
        smaller face pays for."""
        big, small = (400.0, 500.0, 560.0, 700.0), (600.0, 560.0, 680.0, 660.0)
        left, top, right, bottom = fd.pair_box(big, small, 3000, 3000)
        assert top == 500 - 90 and left == 400 - 48 and right == 680 + 24

    def test_a_wide_pair_is_deepened_downward_not_cropped(self):
        far = (2000.0, 500.0, 2080.0, 600.0)
        left, top, right, bottom = fd.pair_box(self.A, far, 4000, 4000)
        assert (right - left) / (bottom - top) == pytest.approx(2.0, abs=0.01)
        assert top == 455 and self._contains((left, top, right, bottom), self.A, far)

    def test_a_stacked_pair_is_widened_to_a_portrait(self):
        below = (420.0, 900.0, 500.0, 1000.0)
        left, top, right, bottom = fd.pair_box(self.A, below, 3000, 3000)
        assert (right - left) / (bottom - top) == pytest.approx(0.8, abs=0.01)
        assert self._contains((left, top, right, bottom), self.A, below)

    def test_it_never_leaves_the_photo_and_never_drops_a_face(self):
        cases = [((5, 5, 105, 105), (300, 10, 400, 110), (420, 300)),       # tight photo
                 ((0, 0, 80, 100), (900, 880, 1000, 1000), (1000, 1000)),     # opposite corners
                 ((-10, 50, 90, 150), (600, 40, 700, 140), (700, 400))]       # box past the edge
        # And a sweep of side-by-side pairs, sizes and gaps, near every edge of a phone photo.
        for fh in (120, 200, 600):
            for gap in (0.0, 0.5, 1.5, 3.0):
                for x0, y0 in ((0, 0), (1500, 2000), (3000 - 2.6 * fh - gap * fh, 4000 - fh)):
                    a = (x0, y0, x0 + 0.8 * fh, y0 + fh)
                    bx = a[2] + gap * fh
                    cases.append((a, (bx, y0 + 0.1 * fh, bx + 0.7 * fh, y0 + 0.95 * fh),
                                  (3000, 4000)))
        for a, b, (w, h) in cases:
            win = fd.pair_box(a, b, w, h)
            left, top, right, bottom = win
            assert 0 <= left < right <= w and 0 <= top < bottom <= h
            clamp = lambda x: (max(0, x[0]), max(0, x[1]), min(w, x[2]), min(h, x[3]))
            assert self._contains(win, clamp(a), clamp(b))

    def test_detect_makes_one_crop_of_both(self, monkeypatch):
        _detector(monkeypatch, [_FakeFace(list(self.A)), _FakeFace(list(self.B))])
        out = fd.detect(_jpeg(2000, 2000), "pair")
        assert len(out) == 1 and out[0].index == 0
        assert out[0].width == 804 - 376

    def test_fewer_than_two_faces_is_no_crop_at_all(self, monkeypatch):
        """Never a one-face crop under a two-person caption."""
        _detector(monkeypatch, [_FakeFace(list(self.A))])
        assert fd.detect(_jpeg(2000, 2000), "pair") == []

    def test_the_service_takes_echoes_and_advertises_it(self, monkeypatch):
        import asyncio
        import base64
        from wanly_worker.services.face_crop import app as svc
        _detector(monkeypatch, [_FakeFace(list(self.A))])
        one = base64.b64encode(_jpeg(2000, 2000)).decode()
        out = asyncio.run(svc.crop(svc.CropRequest(images=[one], framing="pair")))
        assert out.framing == "pair" and out.faces == [] and out.no_face == [0]
        assert out.too_far_apart == []                  # one face is not "too far apart"
        assert "pair" in asyncio.run(svc.health())["features"]


# ------------------------------------------------------------------ #209: tight enough to train

def _side_by_side(fa, fb, gap, w=3000, h=4000):
    """Two faces (heights fa, fb; width 0.8 of height) `gap` face-heights apart, a little
    offset in height, centred in a w x h photo -- the shape of a two-person phone photo."""
    fw_a, fw_b = 0.8 * fa, 0.8 * fb
    x0 = w / 2 - (fw_a + gap * fa + fw_b) / 2
    a = (x0, 1500.0, x0 + fw_a, 1500.0 + fa)
    bx = a[2] + gap * fa
    return a, (bx, 1530.0, bx + fw_b, 1530.0 + fb)


class TestPairFaceSizeAtTraining:
    """#209: on DavidJoana the first pair framing took the smaller face from a median 183 px to
    210 at training size; 5 of 24 crops reached 250. The trainer takes ~TRAIN_EDGE^2 of area,
    so what decides it is the faces' share of the crop -- and, for a wide crop, being delivered
    at that area rather than at a 1024 long edge."""

    @pytest.fixture
    def fake_up(self, monkeypatch):
        import cv2
        from wanly_worker.services.face_crop import upscale as up

        def fake(img, target):
            f = up.plan(img.shape[1], img.shape[0], target)
            if f == 1.0:
                return img
            return cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)))
        monkeypatch.setattr(up, "upscale_bgr", fake)
        monkeypatch.setattr(up, "available", lambda: True)

    def _trained_smaller_face(self, monkeypatch, a, b, w=3000, h=4000):
        """END TO END: the crop detect() actually returns, then the trainer's own scaling --
        the number wanly-api's next /measure would put on the badge."""
        import cv2
        import numpy as np
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        face = fd.detect(_jpeg(w, h), "pair", upscale=True)[0]
        out_h, out_w = cv2.imdecode(np.frombuffer(face.png, np.uint8), 1).shape[:2]
        small = min(a[3] - a[1], b[3] - b[1])
        return small * (out_w / face.width) * fd.train_scale(out_w, out_h)

    @pytest.mark.parametrize("fa,fb,gap", [
        (200, 200, 0.1),     # cheek to cheek
        (220, 180, 0.3),
        (180, 180, 0.5),
        (210, 190, 1.0),
        (200, 180, 2.0),     # a couple of face-heights apart
    ])
    def test_typical_side_by_side_faces_clear_250(self, monkeypatch, fake_up, fa, fb, gap):
        a, b = _side_by_side(fa, fb, gap)
        assert fd.pair_face_px(fd.pair_box(a, b, 3000, 4000), min(fa, fb)) >= 250
        assert self._trained_smaller_face(monkeypatch, a, b) >= 250 * 0.99   # JPEG/int rounding

    def test_it_beats_the_first_cut_by_far(self):
        """The same cheek-to-cheek pair through #208's window (0.6 / 1.5 / 0.8 fh, 3:2, a 1024
        long edge) trained the smaller face at 306 px; the tight window must do much better."""
        a, b = _side_by_side(200, 200, 0.1)
        assert fd.pair_face_px(fd.pair_box(a, b, 3000, 4000), 200) > 1.5 * 306

    def test_a_wide_crop_is_delivered_at_the_training_area_not_a_1024_edge(self, monkeypatch,
                                                                           fake_up):
        """A 2:1 crop at 1024x512 is half the area the trainer would take, every face in it
        1/sqrt(2) the size it could be."""
        import cv2
        import numpy as np
        a, b = _side_by_side(200, 200, 2.5)
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        face = fd.detect(_jpeg(3000, 4000), "pair", upscale=True)[0]
        out_h, out_w = cv2.imdecode(np.frombuffer(face.png, np.uint8), 1).shape[:2]
        assert out_w / out_h == pytest.approx(2.0, abs=0.01)
        assert out_w * out_h == pytest.approx(fd.TRAIN_EDGE ** 2, rel=0.01)
        assert fd.train_scale(out_w, out_h) == pytest.approx(1.0, abs=0.002)  # trainer keeps it

    def test_a_big_pair_crop_is_scaled_down_to_the_area_too(self, monkeypatch):
        import cv2
        import numpy as np
        a, b = _side_by_side(600, 600, 2.0)
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        face = fd.detect(_jpeg(3000, 4000), "pair")[0]
        out_h, out_w = cv2.imdecode(np.frombuffer(face.png, np.uint8), 1).shape[:2]
        assert out_w * out_h <= fd.TRAIN_EDGE ** 2 and out_w > fd.MAX_EDGE

    def test_the_solo_framings_still_cap_the_long_edge(self, monkeypatch):
        """Area sizing is the pair's alone; the square and portrait crops are unchanged."""
        import cv2
        import numpy as np
        _detector(monkeypatch, [_FakeFace([1000, 600, 1400, 1100])])
        face = fd.detect(_jpeg(3000, 3000), "head_shoulders")[0]
        shape = cv2.imdecode(np.frombuffer(face.png, np.uint8), 1).shape[:2]
        assert max(shape) == fd.MAX_EDGE


class TestPairTooFarApart:
    """#209: two faces many face-widths apart leave each a sliver of a crop that is mostly the
    room between them. Below PAIR_MIN_PX for the smaller face, the photo is reported, not
    cropped -- in `no_face`, which is what wanly-api#440 already treats as "no crop"."""

    def test_far_apart_faces_are_refused_with_the_number(self, monkeypatch):
        a, b = _side_by_side(200, 200, 6.0)
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        with pytest.raises(fd.PairTooFarApart) as e:
            fd.detect(_jpeg(3000, 4000), "pair")
        assert e.value.px < fd.PAIR_MIN_PX

    def test_the_threshold_is_wanly_apis_small_face_line(self):
        assert fd.PAIR_MIN_PX == 250

    def test_it_is_judged_on_the_smaller_face(self):
        """A big face beside a small one, close: the big one would clear 250, the small one
        does not -- and the small one is the person the pair LoRA would learn small."""
        big, small = (1000.0, 1500.0, 1400.0, 2000.0), (1450.0, 1700.0, 1530.0, 1800.0)
        win = fd.pair_box(big, small, 3000, 4000)
        assert fd.pair_face_px(win, 500) >= 250 > fd.pair_face_px(win, 100)

    def test_zero_turns_the_rule_off(self, monkeypatch):
        a, b = _side_by_side(200, 200, 6.0)
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        monkeypatch.setattr(fd, "PAIR_MIN_PX", 0.0)
        assert len(fd.detect(_jpeg(3000, 4000), "pair")) == 1

    def test_the_service_reports_it_where_the_api_already_looks(self, monkeypatch):
        """In `no_face` (the API's "no crop for this photo", unchanged) AND in `too_far_apart`
        (why). Other images in the batch are unaffected."""
        import asyncio
        import base64
        from wanly_worker.services.face_crop import app as svc
        far = _side_by_side(200, 200, 6.0)
        near = _side_by_side(200, 200, 0.2)
        jpg = base64.b64encode(_jpeg(3000, 4000)).decode()
        seq = iter([list(far), list(near)])

        class App:
            def get(self, img):
                return [_FakeFace(list(box)) for box in next(seq)]
        monkeypatch.setattr(fd, "_analyser", lambda: App())
        out = asyncio.run(svc.crop(svc.CropRequest(images=[jpg, jpg], framing="pair")))
        assert out.no_face == [0] and out.too_far_apart == [0]
        assert [f.source_index for f in out.faces] == [1]

    def test_other_framings_never_raise_it(self, monkeypatch):
        a, b = _side_by_side(200, 200, 6.0)
        _detector(monkeypatch, [_FakeFace(list(a)), _FakeFace(list(b))])
        assert len(fd.detect(_jpeg(3000, 4000), "head_shoulders")) == 2
