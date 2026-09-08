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
        src = inspect.getsource(fd.detect)
        assert "faces.sort" in src
        assert "index=i" in src

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
