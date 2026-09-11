"""The character-LoRA trainer (wanly-services#7; in this image since wanly-gpu-docker#83).

Three things here are worth more than the rest, because each is a failure that has already cost
this project real time:

  * THE RECIPE. Every number in it was paid for -- 622 images and rank 64 lost to 50 and 32.
    If a port of the pipeline quietly changes one, the LoRAs get worse and nothing says so.
  * THE DRAIN RECONCILER. A drain survives worker re-registration by design, so a trainer that
    dies holding one stops the render queue permanently with nothing pointing at the cause.
  * PROGRESS PARSING. The trainer writes a carriage-return progress bar, so a line-oriented
    reader shows nothing for fifty minutes and a healthy run looks hung.
"""
import asyncio
from pathlib import Path

import re

import pytest


def _real_safetensors(path):
    """A tiny but genuine safetensors file: u64 header length, JSON header, then bytes."""
    import json
    import struct
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _runs_dir(monkeypatch, tmp_path):
    """Point every module's run directory at a temp dir. Importing this package must never
    require the /loras mount -- a service that cannot be imported without its bind mount cannot
    be tested at all."""
    from wanly_worker.services.lora_trainer import recipe
    monkeypatch.setattr(recipe, "RUNS_DIR", str(tmp_path))
    yield tmp_path


class TestTheRecipe:
    """These numbers are the product of a lot of GPU hours. A port that changes one silently
    produces worse LoRAs and says nothing."""

    def test_the_values_that_were_paid_for(self):
        from wanly_worker.services.lora_trainer.recipe import DEFAULTS
        assert DEFAULTS["network_dim"] == 32, "rank 64 gave minimal improvement"
        assert DEFAULTS["network_alpha"] == 32
        assert DEFAULTS["learning_rate"] == 1e-4, "7e-5 was worse"
        assert DEFAULTS["steps"] == 1200
        assert DEFAULTS["num_repeats"] == 10
        assert DEFAULTS["seed"] == 42, "a free seed makes checkpoints incomparable"

    def test_the_preset_is_not_the_default_t2v(self):
        """The dataset is stills with no audio; t2v creates audio-branch weights that never
        receive a training signal."""
        from wanly_worker.services.lora_trainer.recipe import DEFAULTS, train_cmd
        assert DEFAULTS["lora_target_preset"] == "video_sa_ca_ff"
        cmd = train_cmd(Path("/tmp/r"), "x", 1, {})
        assert cmd[cmd.index("--lora_target_preset") + 1] == "video_sa_ca_ff"

    def test_bucket_no_upscale_is_on(self):
        """It makes `resolution` a ceiling rather than a target. Face crops train at native
        size and produce a usable identity anyway."""
        from wanly_worker.services.lora_trainer.recipe import dataset_toml
        assert "bucket_no_upscale = true" in dataset_toml(Path("/tmp/r"))

    def test_the_output_name_carries_the_real_version(self):
        """A hardcoded _v2 is why p@y and l@ura, both first versions, both produced
        NAME_v2-0000NN."""
        from wanly_worker.services.lora_trainer.recipe import train_cmd
        cmd = train_cmd(Path("/tmp/r"), "p@y", 3, {})
        assert cmd[cmd.index("--output_name") + 1] == "p@y_v3"

    def test_state_is_saved_so_a_killed_run_resumes(self):
        from wanly_worker.services.lora_trainer.recipe import train_cmd
        assert "--save_state" in train_cmd(Path("/tmp/r"), "x", 1, {})

    def test_a_run_directory_is_per_version_not_per_character(self):
        """Sharing one meant a v2 training on v1's leftover images while reporting the new
        count."""
        from wanly_worker.services.lora_trainer.recipe import run_dir
        assert run_dir("p@y", 1) != run_dir("p@y", 2)
        assert str(run_dir("p@y", 2)).endswith("/p@y/ltx23b-v2")

    def test_epochs_scale_inversely_with_dataset_size(self):
        """num_repeats is fixed, so 1200 steps is 2.4 epochs on 50 images and 9 on 13 -- four
        times the passes over each image at identical flags."""
        from wanly_worker.services.lora_trainer.recipe import estimated_epochs
        assert estimated_epochs(50, 1200) == 2
        assert estimated_epochs(13, 1200) == 9

    def test_a_job_config_can_override_a_default_but_the_default_stands_alone(self):
        from wanly_worker.services.lora_trainer.recipe import train_cmd
        cmd = train_cmd(Path("/tmp/r"), "x", 1, {"steps": 800})
        assert cmd[cmd.index("--max_train_steps") + 1] == "800"
        cmd = train_cmd(Path("/tmp/r"), "x", 1, {})
        assert cmd[cmd.index("--max_train_steps") + 1] == "1200"


class TestProgress:
    def test_it_reads_a_carriage_return_bar(self, tmp_path):
        """The whole point. `tail -n` on this shows nothing for fifty minutes."""
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(
            b"noise" * 400 + b"\r 45%|####  | 540/1200 [10:00<12:00, 2.35s/it]")
        assert read_progress(tmp_path)[:3] == (540, 1200, 2.35)

    def test_no_log_yet_is_zeroes_not_a_crash(self, tmp_path):
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        assert read_progress(tmp_path) == (0, 0, 0.0, None)

    def test_a_small_bracketed_pair_is_not_mistaken_for_the_step_counter(self, tmp_path):
        """The bar also carries "[01:23<45:67]"-shaped pairs."""
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(b"\r  0%| | 0/12 [00:00<?]")
        done, total, _, _ = read_progress(tmp_path)
        assert (done, total) == (0, 0), "a 12-step total is not a real run"

    def test_it_ignores_a_bar_that_is_not_the_training_bar(self, tmp_path):
        """A training log carries several. The first real run reported step 1747/5947 for a
        1200-step job -- a model-loading bar -- and then overwrote the job's own step count
        with 5947. Anchoring on the configured total is what tells them apart."""
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(
            b"\rloading 1747/5947\r 45%|##  | 540/1200 [10:00<12:00, 2.35s/it]")
        assert read_progress(tmp_path, expect_total=1200)[:3] == (540, 1200, 2.35)

    def test_a_loading_bar_alone_reports_nothing(self, tmp_path):
        """Better to say "no progress yet" than to report someone else's."""
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(b"\rloading 1747/5947")
        assert read_progress(tmp_path, expect_total=1200) == (0, 0, 0.0, None)

    def test_the_job_step_count_is_never_taken_from_the_log(self, tmp_path):
        import inspect
        from wanly_worker.services.lora_trainer import pipeline
        src = inspect.getsource(pipeline.train)
        assert "job.steps, job.rate_s_per_it = done, total, rate" not in src
        assert "job.step, job.rate_s_per_it = done, rate" in src


class TestPreflight:
    def test_a_lean_image_is_named_as_the_cause(self, monkeypatch):
        """`:latest` carries the render stack only. Saying "no trainer" without saying why
        sends the reader to the wrong place."""
        from wanly_worker.service import PreflightError
        from wanly_worker.services.lora_trainer import recipe
        from wanly_worker.services.lora_trainer.service import LoraTrainer
        monkeypatch.setattr(recipe, "TRAINER_PYTHON", "/nope/python")
        with pytest.raises(PreflightError) as e:
            LoraTrainer().preflight()
        assert "WITH_TRAINER=1" in str(e.value) and ":full" in str(e.value)

    def test_a_missing_model_mount_is_named_as_a_mount(self, monkeypatch, tmp_path):
        from wanly_worker.service import PreflightError
        from wanly_worker.services.lora_trainer import recipe
        from wanly_worker.services.lora_trainer.service import LoraTrainer
        fake = tmp_path / "python"
        fake.write_text("")
        monkeypatch.setattr(recipe, "TRAINER_PYTHON", str(fake))
        monkeypatch.setattr(recipe, "CKPT", "/nope/ckpt.safetensors")
        with pytest.raises(PreflightError, match="bind-mounted"):
            LoraTrainer().preflight()


class TestTheDrainReconciler:
    """A drain survives worker re-registration by design -- wanly-api's
    `reregistered_drain_state` says cancelling one is an explicit action. So a trainer that dies
    holding one leaves the render queue stopped, permanently, with nothing pointing at why."""

    def test_a_finished_job_still_holding_a_drain_is_found(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import jobs as J
        monkeypatch.setattr(J, "STATE_DIR", tmp_path / ".trainer")
        store = J.Store()
        j = J.Job(id="a", character="p@y", trigger="p@y", version=2, steps=1200)
        j.phase, j.drained_worker_id = "failed", "w-1"
        store.add(j)
        assert store.orphaned_drains() == [("a", "w-1")]

    def test_a_running_job_holding_a_drain_is_left_alone(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import jobs as J
        monkeypatch.setattr(J, "STATE_DIR", tmp_path / ".trainer")
        store = J.Store()
        j = J.Job(id="a", character="p@y", trigger="p@y", version=2, steps=1200)
        j.phase, j.drained_worker_id = "training", "w-1"
        store.add(j)
        assert store.orphaned_drains() == []

    def test_state_survives_a_restart(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import jobs as J
        monkeypatch.setattr(J, "STATE_DIR", tmp_path / ".trainer")
        s1 = J.Store()
        j = J.Job(id="a", character="p@y", trigger="p@y", version=2, steps=1200)
        j.phase, j.drained_worker_id = "failed", "w-1"
        s1.add(j)
        assert J.Store().orphaned_drains() == [("a", "w-1")]

    def test_reconcile_releases_and_clears(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu, jobs as J
        monkeypatch.setattr(J, "STATE_DIR", tmp_path / ".trainer")
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        store = J.Store()
        j = J.Job(id="a", character="p@y", trigger="p@y", version=2, steps=1200)
        j.phase, j.drained_worker_id = "completed", "w-1"
        store.add(j)

        deleted = []

        class _C:
            async def delete(self, url, headers=None, timeout=None):
                deleted.append(url)
                class R: status_code = 204
                return R()

        _run(gpu.reconcile(_C(), store))
        assert deleted == ["http://api/workers/w-1/drain"]
        assert store.orphaned_drains() == []

    def test_a_corrupt_state_file_does_not_stop_the_container(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import jobs as J
        d = tmp_path / ".trainer"
        d.mkdir()
        (d / "bad.json").write_text("{not json")
        monkeypatch.setattr(J, "STATE_DIR", d)
        assert J.Store().all() == []


class TestAcquire:
    def test_it_refuses_to_train_beside_a_busy_worker(self, monkeypatch):
        """Sharing the card is the OOM this exists to prevent -- and the drain it is holding
        has to come off before it gives up."""
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "DRAIN_TIMEOUT_S", 0)
        monkeypatch.setattr(gpu, "RENDER_WORKER_NAME", "3090.zero")
        monkeypatch.setattr(gpu, "vram_used_mib", lambda: 23000)
        released = []

        class _C:
            async def get(self, url, headers=None, timeout=None):
                class R:
                    status_code = 200
                    @staticmethod
                    def json():
                        return [{"id": "w-1", "kind": "render", "status": "online-busy",
                                 "hostname": "3090", "friendly_name": "3090.zero"}]
                    @staticmethod
                    def raise_for_status(): pass
                return R()
            async def post(self, url, json=None, headers=None, timeout=None):
                class R:
                    status_code = 200
                    @staticmethod
                    def raise_for_status(): pass
                return R()
            async def delete(self, url, headers=None, timeout=None):
                released.append(url)
                class R: status_code = 204
                return R()

        with pytest.raises(RuntimeError, match="did not free the GPU"):
            _run(gpu.acquire(_C(), "3090"))
        assert released, "gave up while still holding the drain"

    def test_no_render_worker_at_all_means_nothing_to_drain(self, monkeypatch):
        """A trainer on a box with no render worker takes the card without ceremony."""
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "RENDER_WORKER_NAME", "")

        class _C:
            async def get(self, url, headers=None, timeout=None):
                class R:
                    @staticmethod
                    def json(): return [{"id": "s", "kind": "service", "status": "online"}]
                    @staticmethod
                    def raise_for_status(): pass
                return R()

        assert _run(gpu.acquire(_C(), "3090")) == ""


class TestTheService:
    def test_it_is_registered_under_the_flag_name(self):
        from wanly_worker import registry
        assert "lora-trainer" in registry.KNOWN

    def test_it_binds_loopback_only(self):
        """Unlike joycaption, nothing outside the container should reach the trainer directly:
        the console goes through wanly-api and the CLI through the control port."""
        from wanly_worker.services.lora_trainer import LoraTrainer
        assert "127.0.0.1" in LoraTrainer().command()

    def test_details_never_raises(self, monkeypatch):
        """It feeds /health, which must answer even when the trainer is broken."""
        from wanly_worker.services.lora_trainer import LoraTrainer
        assert isinstance(LoraTrainer().details(), dict)


class TestOneAtATime:
    def test_a_second_job_is_refused_while_one_runs(self, tmp_path, monkeypatch):
        """Two concurrent runs would not be twice as fast, they would be an OOM."""
        from wanly_worker.services.lora_trainer import jobs as J
        monkeypatch.setattr(J, "STATE_DIR", tmp_path / ".trainer")
        store = J.Store()
        assert store.claim_slot() is True
        store.add(J.Job(id="a", character="x", trigger="x", version=1, steps=1200))
        assert store.claim_slot() is False


class TestTheImageSplit:
    """Two tags from one Dockerfile. The split is payload, not behaviour — SERVICES still
    decides what runs — but a trainer deployed against the lean tag would fail in preflight
    after a pull, which is a slow way to learn a one-line mistake."""

    import pathlib
    ROOT = pathlib.Path(__file__).parent.parent

    def test_the_trainer_layer_is_opt_in(self):
        d = (self.ROOT / "Dockerfile").read_text()
        assert "ARG WITH_TRAINER=0" in d, "the default must be lean"
        assert 'if [ "$WITH_TRAINER" = "1" ]' in d

    def test_the_trainer_commit_is_pinned(self):
        """A moving branch would hand a different trainer to every build."""
        d = (self.ROOT / "Dockerfile").read_text()
        assert "ARG TRAINER_COMMIT=" in d
        assert 'checkout "$TRAINER_COMMIT"' in d

    def test_python_311_specifically(self):
        """The base is 22.04/3.10; the venv this reproduces is 3.11.9."""
        d = (self.ROOT / "Dockerfile").read_text()
        assert "python3.11" in d and "deadsnakes" in d

    def test_torch_is_the_proven_cu128_build(self):
        d = (self.ROOT / "Dockerfile").read_text()
        assert "torch==2.8.0" in d and "cu128" in d

    def test_both_tags_are_built_in_one_run(self):
        w = (self.ROOT / ".github/workflows/build-push.yml").read_text()
        assert "wanly-gpu-docker:latest" in w and "wanly-gpu-docker:full" in w
        assert "wanly-gpu-docker:next-full" in w, "the soak branch needs a full tag too"
        assert "WITH_TRAINER=1" in w

    def test_the_trainer_port_is_not_exposed(self):
        """It binds loopback; the console goes through wanly-api and the CLI through the
        control port. Exposing 8082 would only invite a caller that bypasses both."""
        d = (self.ROOT / "Dockerfile").read_text()
        assert not re.search(r"EXPOSE[^\n]*\b8082\b", d)

    def test_deploy_refuses_a_trainer_on_the_lean_image_before_removing_anything(self):
        s = (self.ROOT / "deploy/run-worker.sh").read_text()
        head, _, tail = s.partition('docker rm -f "$NAME"')
        assert "built WITHOUT them" in head, "the refusal must come before the running container is removed"

    def test_deploy_mounts_are_per_service(self):
        """A pod-shaped render box has no ollama store and no run directories; requiring them
        everywhere would mean inventing empty directories to satisfy a check."""
        s = (self.ROOT / "deploy/run-worker.sh").read_text()
        assert "WANT_OLLAMA" in s and "WANT_TRAINER" in s
        assert "/workspace/models:ro" in s, "the models tree must be read-only"
        assert '-v "$LORA_RUNS_DIR:/loras"' in s


class TestFindingTheWorkerToDrain:
    """"The render worker on my host" is not inferable from inside a container.

    Every worker registers the hostname it sees, which is its own container id: the render
    worker on 3090.zero reported `4b6d0f9c3b3b` while the trainer beside it reported
    `54a6e63cdd96`. Matching those was the first attempt, it silently found nothing, and the
    trainer took the card without draining. It only got away with it because the worker was
    idle at the time.
    """

    @staticmethod
    def _client(rows):
        class _C:
            async def get(self, url, headers=None, timeout=None):
                class R:
                    @staticmethod
                    def json(): return rows
                    @staticmethod
                    def raise_for_status(): pass
                return R()
        return _C()

    def _rows(self, *names):
        return [{"id": n, "kind": "render", "status": "online-idle", "friendly_name": n,
                 "hostname": "some-container-id"} for n in names]

    def test_an_explicit_name_is_matched(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "RENDER_WORKER_NAME", "3090.zero")
        w = _run(gpu.find_render_worker(self._client(self._rows("other", "3090.zero"))))
        assert w["friendly_name"] == "3090.zero"

    def test_one_render_worker_is_unambiguous(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "RENDER_WORKER_NAME", "")
        w = _run(gpu.find_render_worker(self._client(self._rows("3090.zero"))))
        assert w["friendly_name"] == "3090.zero"

    def test_several_and_no_name_refuses_to_guess(self, monkeypatch):
        """Draining the wrong box stops a queue for no reason."""
        from wanly_worker.services.lora_trainer import gpu
        monkeypatch.setattr(gpu, "QUEUE_URL", "http://api")
        monkeypatch.setattr(gpu, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(gpu, "RENDER_WORKER_NAME", "")
        with pytest.raises(RuntimeError, match="Refusing to guess"):
            _run(gpu.find_render_worker(self._client(self._rows("a", "b"))))

    def test_hostname_is_not_consulted(self):
        """It is the container id, so it can only ever produce a false negative."""
        import inspect
        from wanly_worker.services.lora_trainer import gpu
        src = inspect.getsource(gpu.find_render_worker)
        assert 'w.get("hostname")' not in src


class TestTheImageCarriesOpenCVsLibraries:
    """The trainer imports cv2 through musubi_tuner.dataset.image_video_dataset. A CUDA *runtime*
    base carries no GL, so stage 1 died with "ImportError: libGL.so.1" -- after the image was
    built, pushed, pulled, and a job claimed."""

    def test_libgl_is_installed_in_the_trainer_layer(self):
        import pathlib
        d = (pathlib.Path(__file__).parent.parent / "Dockerfile").read_text()
        assert "libgl1" in d and "libglib2.0-0" in d


class TestAFailureIsReported:
    """A failure before the first training step left the API row CLAIMED forever.

    on_progress was only called from inside the training loop, so a job that died while staging
    -- as the first real run did, on a missing libGL -- never told anyone. The orphan reclaim
    would then requeue it, the trainer would claim it again and fail again: a loop that looks
    like a queue quietly not moving.
    """

    def test_the_terminal_state_is_reported_from_the_finally(self):
        import inspect
        from wanly_worker.services.lora_trainer import app as mod
        src = inspect.getsource(mod._run)
        finally_block = src.rsplit("finally:", 1)[1]
        assert "on_progress(job)" in finally_block

    def test_a_failing_report_does_not_mask_the_real_failure(self):
        """Losing the report is bad; replacing the training error with an HTTP error is worse."""
        import inspect
        from wanly_worker.services.lora_trainer import app as mod
        src = inspect.getsource(mod._run)
        assert "could not report the final state" in src

    def test_the_poller_maps_failed_to_the_api_status(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._report)
        assert '"failed": "failed"' in src and '"error_message"' in src


class TestSharedMemory:
    """Docker gives a container 64 MB of /dev/shm and PyTorch's DataLoader workers pass tensors
    through it. The first real training run died forty minutes in with

        RuntimeError: unable to write to file </torch_311_...>: No space left on device (28)
        DataLoader worker ... killed by signal: Bus error

    which names shared memory only in the second message and reads like a full disk in the
    first."""

    def test_the_deploy_raises_shm_for_a_trainer(self):
        import pathlib
        s = (pathlib.Path(__file__).parent.parent / "deploy/run-worker.sh").read_text()
        assert "--shm-size" in s
        assert 'WANT_TRAINER" = "1" ] && echo 8g' in s

    def test_a_render_only_box_is_left_at_the_default(self):
        """It has no DataLoader and no reason for 8 GB of shm."""
        import pathlib
        s = (pathlib.Path(__file__).parent.parent / "deploy/run-worker.sh").read_text()
        assert "echo 64m" in s


class TestCancellingActuallyStopsTheRun:
    """Cancelling did nothing, in both directions.

    The container's own endpoint wrote `phase = "cancelled"`, which `_run` overwrote within
    seconds as it moved through staging/training/collecting -- and nothing read it anyway. A job
    claimed from the queue was worse still: the console can only write the API row, so the
    container never heard about the cancel at all.
    """

    def test_the_flag_is_separate_from_the_phase(self):
        """A phase written from outside is overwritten by the run; a flag is not."""
        from wanly_worker.services.lora_trainer.jobs import Job
        j = Job(id="j", character="pay", trigger="p@y", version=2, steps=1200)
        assert j.cancel_requested is False

    def test_the_endpoint_sets_the_flag(self):
        import inspect
        from wanly_worker.services.lora_trainer import app as mod
        src = inspect.getsource(mod.cancel)
        assert "job.cancel_requested = True" in src

    def test_the_training_loop_checks_the_flag(self):
        import inspect
        from wanly_worker.services.lora_trainer import pipeline as mod
        src = inspect.getsource(mod.train)
        assert "job.cancel_requested" in src
        assert "proc.terminate()" in src

    def test_it_terminates_before_it_kills(self):
        """SIGKILL leaves accelerate's children holding the CUDA context, and the next run OOMs
        against a card that looks free."""
        import inspect
        from wanly_worker.services.lora_trainer import pipeline as mod
        src = inspect.getsource(mod.train)
        assert src.index("proc.terminate()") < src.index("proc.kill()")

    def test_a_cancelled_run_is_not_reported_as_failed(self):
        import inspect
        from wanly_worker.services.lora_trainer import app as mod, pipeline
        assert issubclass(pipeline.Cancelled, pipeline.PipelineError)
        src = inspect.getsource(mod._run)
        # The narrower handler must come first, or `except Exception` swallows it.
        assert src.index("except pipeline.Cancelled") < src.index("except Exception")

    def test_the_api_reply_is_how_a_claimed_job_learns_it_was_cancelled(self):
        """Nothing in the API reaches into this box, so the row returned by the progress report
        is the only channel — and the container's own /cancel cannot reach a claimed job."""
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._patch)
        assert 'r.json().get("status") == "cancelled"' in src
        assert "job.cancel_requested = True" in src


class TestCheckpointsGoStraightToS3:
    """The run that "completed" on 2026-09-07 produced five checkpoints and three reached the
    bucket. They went through wanly-api, one 650 MB multipart at a time, AFTER training -- so
    the console said "running" at 100% for the hour that took, two did not survive, the job
    said completed anyway, and the character row kept pointing at last week's file."""

    def _poller(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        return mod.Poller(client=None, worker_id_getter=lambda: "w")

    def _job(self):
        from wanly_worker.services.lora_trainer.jobs import Job
        return Job(id="j1", character="p@y", trigger="p@y", version=2, steps=1200,
                   remote_id="remote-1")

    def _write(self, tmp_path, job, names):
        from wanly_worker.services.lora_trainer import recipe
        out = recipe.run_dir(job.character, job.version) / "output"
        out.mkdir(parents=True, exist_ok=True)
        for n in names:
            (out / n).write_bytes(b"x" * 16)

    def test_the_final_checkpoint_sorts_last(self, tmp_path, monkeypatch):
        """It has no epoch number, so it sorts FIRST alphabetically -- and it is the one with
        the most training in it."""
        p, job = self._poller(tmp_path, monkeypatch), self._job()
        self._write(tmp_path, job, ["p@y_v2.comfy.safetensors", "p@y_v2-000002.comfy.safetensors",
                                    "p@y_v2-000001.comfy.safetensors", "p@y_v2-000001.safetensors"])
        names = [f.name for f in p._local_checkpoints(job)]
        assert names == ["p@y_v2-000001.comfy.safetensors", "p@y_v2-000002.comfy.safetensors",
                         "p@y_v2.comfy.safetensors"]

    def test_the_final_checkpoint_is_labelled_final_not_left_bare(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._publish)
        assert '{"final": "true"}' in src

    def test_a_growing_file_is_not_taken(self, tmp_path, monkeypatch):
        """The trainer writes in place. A checkpoint whose size changed since the last look is
        still being written, and uploading it would publish half a LoRA."""
        from wanly_worker.services.lora_trainer import poller as mod
        monkeypatch.setattr(mod, "CHECKPOINT_SETTLE_S", 0)
        p, job = self._poller(tmp_path, monkeypatch), self._job()
        self._write(tmp_path, job, ["p@y_v2-000001.comfy.safetensors"])
        f = p._local_checkpoints(job)[0]
        assert not p._settled(f)          # first sight: nothing to compare against
        f.write_bytes(b"x" * 32)
        assert not p._settled(f)          # grew
        assert p._settled(f)              # unchanged since last look, window elapsed

    def test_uploads_start_during_training_not_after(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._report)
        assert 'if job.phase in ("training", "collecting"):' in src
        assert "self._sweep_checkpoints(job)" in src.split('if job.phase == "completed"')[0]

    def test_completed_is_not_said_until_everything_is_in_the_bucket(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._report)
        after = src.split('if job.phase == "completed"')[1]
        assert after.index("await self._drain_uploads(job)") < after.index("await self._patch(job, body)")

    def test_a_missing_checkpoint_fails_the_job_and_says_where_the_files_are(self, tmp_path, monkeypatch):
        """A green row with the final checkpoint absent is exactly what this replaces."""
        from wanly_worker.services.lora_trainer import poller as mod
        monkeypatch.setattr(mod, "CHECKPOINT_SETTLE_S", 0)
        p, job = self._poller(tmp_path, monkeypatch), self._job()
        job.phase = "completed"
        self._write(tmp_path, job, ["p@y_v2-000001.comfy.safetensors", "p@y_v2.comfy.safetensors"])
        # e01 made it; the final did not.
        p._published["j1"] = {str(p._local_checkpoints(job)[0]): "s3://ltx-loras/character/pay_v2_e01.safetensors"}
        p._failed["j1"] = [str(p._local_checkpoints(job)[1])]
        sent = []

        async def fake_patch(j, body):
            sent.append(body)
        p._patch = fake_patch

        async def no_drain(j):
            return None
        p._drain_uploads = no_drain
        _run(p._report(job))
        final = sent[-1]
        assert final["status"] == "failed"
        assert "p@y_v2.comfy.safetensors" in final["error_message"]
        assert "/output" in final["error_message"]

    def test_everything_published_is_completed(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        monkeypatch.setattr(mod, "CHECKPOINT_SETTLE_S", 0)
        p, job = self._poller(tmp_path, monkeypatch), self._job()
        job.phase = "completed"
        self._write(tmp_path, job, ["p@y_v2-000001.comfy.safetensors", "p@y_v2.comfy.safetensors"])
        p._published["j1"] = {str(f): "s3://x" for f in p._local_checkpoints(job)}
        sent = []

        async def fake_patch(j, body):
            sent.append(body)
        p._patch = fake_patch

        async def no_drain(j):
            return None
        p._drain_uploads = no_drain
        _run(p._report(job))
        assert sent[-1]["status"] == "completed"
        assert "2 checkpoint(s) published" in sent[-1]["progress_log"]

    def test_the_file_is_streamed_not_read_whole(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._publish)
        assert "content=self._counted(job, path)" in src
        assert "async for chunk in _file_chunks(path):" in inspect.getsource(mod.Poller._counted)
        assert '"Content-Length": str(size)' in src

    def test_one_upload_at_a_time_and_the_final_jumps_the_queue(self, tmp_path, monkeypatch):
        """The uplink is slower than the trainer, so the queue is behind when training ends.
        The final checkpoint is the one the character will point at; it must not wait behind
        four epochs that nobody has chosen yet."""
        p, job = self._poller(tmp_path, monkeypatch), self._job()
        self._write(tmp_path, job, ["p@y_v2-000001.comfy.safetensors", "p@y_v2-000002.comfy.safetensors",
                                    "p@y_v2.comfy.safetensors"])
        order = []

        async def fake_publish(j, path):
            order.append(path.name)
        p._publish = fake_publish
        p._queue["j1"] = list(p._local_checkpoints(job))
        _run(p._upload_worker(job))
        assert order == ["p@y_v2.comfy.safetensors", "p@y_v2-000001.comfy.safetensors",
                         "p@y_v2-000002.comfy.safetensors"]

    def test_progress_is_reported_while_the_queue_drains(self):
        """After training the console showed "running" at 100% with "5 checkpoint(s)" for the
        hour the uploads took, which is indistinguishable from a trainer that has hung."""
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._drain_uploads)
        assert 'await self._patch(job, {"progress_log": status})' in src


class TestARestartFailsWhatItInterrupted:
    """A job persisted mid-run counted as active after a restart: the one-at-a-time slot was
    taken forever, its drain was not an orphan to the reconciler, and the API row stayed
    RUNNING with a progress log -- the one shape the orphan reclaim leaves alone."""

    def _store_with(self, tmp_path, monkeypatch, **fields):
        import json
        from wanly_worker.services.lora_trainer import jobs as mod
        monkeypatch.setattr(mod, "STATE_DIR", tmp_path / ".trainer")
        (tmp_path / ".trainer").mkdir()
        base = dict(id="j1", character="p@y", trigger="p@y", version=2, steps=1200,
                    remote_id="r1", phase="training", drained_worker_id="w1")
        base.update(fields)
        (tmp_path / ".trainer" / "j1.json").write_text(json.dumps(base))
        return mod.Store()

    def test_a_mid_run_job_is_failed_on_load(self, tmp_path, monkeypatch):
        store = self._store_with(tmp_path, monkeypatch)
        job = store.get("j1")
        assert job.phase == "failed"
        assert "restarted" in job.error and "training" in job.error

    def test_its_drain_becomes_an_orphan_the_reconciler_will_release(self, tmp_path, monkeypatch):
        store = self._store_with(tmp_path, monkeypatch)
        assert store.orphaned_drains() == [("j1", "w1")]

    def test_the_slot_is_free_again(self, tmp_path, monkeypatch):
        store = self._store_with(tmp_path, monkeypatch)
        assert store.claim_slot()

    def test_it_is_handed_over_for_reporting_once(self, tmp_path, monkeypatch):
        store = self._store_with(tmp_path, monkeypatch)
        assert [j.id for j in store.failed_by_restart()] == ["j1"]
        assert store.failed_by_restart() == []

    def test_a_finished_job_is_left_alone(self, tmp_path, monkeypatch):
        store = self._store_with(tmp_path, monkeypatch, phase="completed", drained_worker_id="")
        assert store.get("j1").phase == "completed"
        assert store.failed_by_restart() == []

    def test_startup_reports_it_to_the_api(self):
        import inspect
        from wanly_worker.services.lora_trainer import app as mod
        src = inspect.getsource(mod.lifespan)
        assert "STORE.failed_by_restart()" in src
        assert '"status": "failed"' in src


class TestTheDrainOutlastsA720pRender:
    def test_the_default_matches_the_daemons_own_drain_wait(self):
        """1800 s was under a 720x1056 render (~1780 s measured), so a training job arriving
        just after one started gave up a minute before the card would have been free."""
        import inspect
        from wanly_worker.services.lora_trainer import gpu as mod
        assert 'os.environ.get("DRAIN_TIMEOUT_S", "3600")' in inspect.getsource(mod)


class TestTheWaitForTheCardIsReported:
    """Between the claim and the first training step nothing was reported, so the API row
    sat at "claimed" with no progress through staging and a drain wait that can last a
    720p render -- which the console shows as queued and the orphan reclaim, after twenty
    minutes, puts back in the queue."""

    def test_staging_is_reported_before_the_drain(self):
        import inspect
        from wanly_worker.services.lora_trainer import app as mod
        src = inspect.getsource(mod._run)
        stage = src.index("run = await pipeline.stage(job, groups)")
        acquire = src.index("await gpu.acquire(client, HOSTNAME")
        assert "await on_progress(job)" in src[stage:acquire]

    def test_every_poll_of_the_wait_reports(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(mod, "DRAIN_TIMEOUT_S", 1)
        monkeypatch.setattr(mod, "vram_used_mib", lambda: 20000)
        monkeypatch.setattr(mod.asyncio, "sleep", _noop_sleep)
        seen = []

        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return [{"id": "w1", "friendly_name": "3090.zero",
                                     "kind": "render", "status": "online-busy"}]

        class _C:
            async def get(self, *a, **k): return _R()
            async def post(self, *a, **k): return _R()
            async def delete(self, *a, **k): return _R()

        async def on_wait(msg): seen.append(msg)
        with pytest.raises(RuntimeError):
            _run(mod.acquire(_C(), "h", on_wait=on_wait))
        assert seen and "finish rendering" in seen[0] and "20000 MiB" in seen[0]


async def _noop_sleep(_s):
    return None


class TestReleasingADrainOnAWorkerThatRebooted:
    """On a box with a restart policy a drained daemon exits, the container comes back and
    registers a NEW row; the old id 404s. That is the drain having worked, and it was being
    logged as the queue-stopped-forever case with a DELETE to run by hand."""

    def test_a_404_is_success_not_an_alarm(self, monkeypatch, capsys):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")

        class _R:
            status_code = 404

        class _C:
            async def delete(self, *a, **k): return _R()

        assert _run(mod.release(_C(), "old-id")) is True
        out = capsys.readouterr().out
        assert "re-registered" in out and "COULD NOT RELEASE" not in out


class TestTheLossIsRead:
    def test_avr_loss_comes_off_the_bar(self, tmp_path):
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(
            b"steps:  45%|####  | 540/1200 [20:00<25:00, 2.35s/it, avr_loss=0.734, loss_a=n/a]\r"
            b"steps:  46%|####  | 552/1200 [20:30<24:30, 2.35s/it, avr_loss=0.712, loss_a=n/a]")
        done, total, rate, loss = read_progress(tmp_path, expect_total=1200)
        assert (done, total) == (552, 1200)
        assert loss == 0.712

    def test_no_loss_yet_is_none(self, tmp_path):
        from wanly_worker.services.lora_trainer.pipeline import read_progress
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "03_train.log").write_bytes(b"loading: 100%|####| 5/5")
        assert read_progress(tmp_path, expect_total=1200)[3] is None


class TestOnlyWantedCheckpointsGoUp:
    """Only the final by default; every epoch with "all"; anything else when the console asks
    for it by name. An epoch nobody asked for is not "missing" at the end."""

    def _setup(self, tmp_path, monkeypatch, policy):
        from wanly_worker.services.lora_trainer import poller as mod
        from wanly_worker.services.lora_trainer.jobs import Job
        from wanly_worker.services.lora_trainer import recipe
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        p = mod.Poller(client=None, worker_id_getter=lambda: "w")
        job = Job(id="j1", character="p@y", trigger="p@y", version=2, steps=1200, images=27,
                  remote_id="r1")
        p._policy["j1"] = policy
        out = recipe.run_dir(job.character, job.version) / "output"
        out.mkdir(parents=True)
        for n in ("p@y_v2-000001.comfy.safetensors", "p@y_v2-000002.comfy.safetensors",
                  "p@y_v2.comfy.safetensors"):
            (out / n).write_bytes(b"x" * 16)
        return p, job

    def test_final_only_by_default(self, tmp_path, monkeypatch):
        p, job = self._setup(tmp_path, monkeypatch, "final")
        wanted = [f.name for f in p._local_checkpoints(job) if p._wanted(job, f)]
        assert wanted == ["p@y_v2.comfy.safetensors"]
        assert [f.name for f in p._unpublished(job)] == ["p@y_v2.comfy.safetensors"]

    def test_all_means_all(self, tmp_path, monkeypatch):
        p, job = self._setup(tmp_path, monkeypatch, "all")
        assert len([f for f in p._local_checkpoints(job) if p._wanted(job, f)]) == 3

    def test_a_request_by_name_adds_one(self, tmp_path, monkeypatch):
        p, job = self._setup(tmp_path, monkeypatch, "final")
        p._requested["j1"] = {"e02"}
        wanted = sorted(f.name for f in p._local_checkpoints(job) if p._wanted(job, f))
        assert wanted == ["p@y_v2-000002.comfy.safetensors", "p@y_v2.comfy.safetensors"]

    def test_epochs_are_reported_with_step_and_loss(self, tmp_path, monkeypatch):
        p, job = self._setup(tmp_path, monkeypatch, "final")
        job.loss_log = [[100, 0.9], [270, 0.8], [500, 0.75], [1200, 0.7]]
        epochs = p._epochs(job)
        assert [e["label"] for e in epochs] == ["e01", "e02", "final"]
        assert epochs[0]["step"] == 270 and epochs[0]["loss"] == 0.8
        assert epochs[2]["step"] == 1200 and epochs[2]["loss"] == 0.7

    def test_finished_jobs_are_polled_for_requests(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        assert "await self.check_publish_requests()" in inspect.getsource(mod.Poller.tick)
        src = inspect.getsource(mod.Poller.check_publish_requests)
        assert 'row.get("publish_requests")' in src and "self._sweep_checkpoints(job)" in src


class TestAFileGoingUpIsNotQueuedAgain:
    """Popped from the queue when its upload starts, the final checkpoint was neither queued
    nor published for twelve minutes, and the ten-second sweep during the drain put it
    straight back: it went up twice on the first end-to-end run."""

    def test_an_inflight_file_is_skipped_by_the_sweep(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        from wanly_worker.services.lora_trainer.jobs import Job
        from wanly_worker.services.lora_trainer import recipe
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(mod, "CHECKPOINT_SETTLE_S", 0)
        p = mod.Poller(client=None, worker_id_getter=lambda: "w")
        job = Job(id="j1", character="p@y", trigger="p@y", version=2, steps=100, remote_id="r1")
        out = recipe.run_dir(job.character, job.version) / "output"
        out.mkdir(parents=True)
        f = out / "p@y_v2.comfy.safetensors"
        _real_safetensors(f)   # a file with a header, or the sweep refuses it (2026-09-08)
        async def idle(_job):
            return None
        p._upload_worker = idle

        async def go():
            p._settled(f)                       # first sight
            p._inflight.add(str(f))             # "uploading"
            p._sweep_checkpoints(job)
            assert p._queue.get("j1", []) == []
            p._inflight.discard(str(f))
            p._sweep_checkpoints(job)
            assert p._queue["j1"] == [f]
        _run(go())

    def test_the_worker_marks_and_clears_it(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller._upload_worker)
        assert "self._inflight.add(str(path))" in src and "self._inflight.discard(str(path))" in src


class TestAStaleWorkerNameDoesNotMeanTrainBesideIt:
    """The render worker was renamed in the console; RENDER_WORKER_NAME still said the old
    name; the trainer found nothing to drain and OOM'd twenty seconds in, loading the text
    encoder beside 13.5 GB of idle engine."""

    def _client(self, names):
        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return [{"id": n, "friendly_name": n, "kind": "render", "status": "online-idle"}
                        for n in names]
        class _C:
            async def get(self, *a, **k): return _R()
            async def post(self, *a, **k): return _R()
            async def delete(self, *a, **k): return _R()
        return _C()

    def test_the_only_render_worker_is_used_when_the_name_is_stale(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "RENDER_WORKER_NAME", "3090.zero")
        w = _run(mod.find_render_worker(self._client(["ltx-engine-1"]), "h"))
        assert w["friendly_name"] == "ltx-engine-1"

    def test_two_and_a_stale_name_is_still_nothing(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "RENDER_WORKER_NAME", "3090.zero")
        assert _run(mod.find_render_worker(self._client(["a", "b"]), "h")) is None

    def test_a_busy_card_with_nothing_to_drain_refuses(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(mod, "RENDER_WORKER_NAME", "3090.zero")
        monkeypatch.setattr(mod, "vram_used_mib", lambda: 13542)
        with pytest.raises(RuntimeError, match="Refusing to train beside"):
            _run(mod.acquire(self._client(["a", "b"]), "h"))

    def test_a_free_card_with_nothing_to_drain_proceeds(self, monkeypatch):
        from wanly_worker.services.lora_trainer import gpu as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        monkeypatch.setattr(mod, "RENDER_WORKER_NAME", "")
        monkeypatch.setattr(mod, "vram_used_mib", lambda: 300)
        assert _run(mod.acquire(self._client([]), "h")) == ""


class TestUploadProgressIsInTheStatusLine:
    """The trainer streams the file itself, so it knows how far the upload is; "uploading
    final" for eighteen minutes with no number looked like a hang."""

    def _poller(self, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        monkeypatch.setattr(mod, "QUEUE_URL", "http://api")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "k")
        return mod.Poller(client=None, worker_id_getter=lambda: "w")

    def test_the_drain_line_carries_a_percentage(self, monkeypatch):
        from wanly_worker.services.lora_trainer.jobs import Job
        p = self._poller(monkeypatch)
        job = Job(id="j1", character="p@y", trigger="p@y", version=2, steps=100)
        p._current["j1"] = "final"
        p._sent["j1"] = (262144000, 654444968)
        assert p._upload_status(job).startswith("uploading final — 40% of 624 MB")

    def test_the_training_line_gets_a_suffix_while_an_upload_overlaps(self, monkeypatch):
        from wanly_worker.services.lora_trainer.jobs import Job
        p = self._poller(monkeypatch)
        job = Job(id="j1", character="p@y", trigger="p@y", version=2, steps=100)
        assert p._upload_suffix(job) == ""
        p._current["j1"] = "e02"
        p._sent["j1"] = (100, 400)
        assert p._upload_suffix(job) == " · uploading e02 25%"

    def test_the_counter_keeps_score(self, tmp_path, monkeypatch):
        from wanly_worker.services.lora_trainer.jobs import Job
        p = self._poller(monkeypatch)
        job = Job(id="j1", character="p@y", trigger="p@y", version=2, steps=100)
        f = tmp_path / "x.bin"
        f.write_bytes(b"x" * 1000)
        p._sent["j1"] = (0, 1000)

        async def go():
            n = 0
            async for _ in p._counted(job, f):
                n += 1
            return n
        assert _run(go()) >= 1
        assert p._sent["j1"] == (1000, 1000)


class TestPreflightSeesTheCard:
    """After the host rebooted the container came back with nvidia-smi happy and torch saying
    "No CUDA GPUs are available"; the first job died twenty seconds after the claim."""

    def test_a_missing_device_is_a_preflight_error_that_names_the_fix(self, monkeypatch, tmp_path):
        from wanly_worker.services.lora_trainer import service as mod
        from wanly_worker.services.lora_trainer import recipe
        from wanly_worker.service import PreflightError
        py = tmp_path / "python"; py.write_text("#!/bin/sh\nexit 3\n"); py.chmod(0o755)
        ck = tmp_path / "ckpt"; ck.write_text("x"); gm = tmp_path / "gemma"; gm.mkdir()
        monkeypatch.setattr(recipe, "TRAINER_PYTHON", str(py))
        monkeypatch.setattr(recipe, "CKPT", str(ck))
        monkeypatch.setattr(recipe, "GEMMA", str(gm))
        with pytest.raises(PreflightError, match="run-worker.sh"):
            mod.LoraTrainer().preflight()

    def test_the_probe_asks_torch_not_nvidia_smi(self):
        import inspect
        from wanly_worker.services.lora_trainer import service as mod
        assert "torch.cuda.is_available()" in inspect.getsource(mod.LoraTrainer.preflight)


class TestARunTheRestartInterruptedMidUploadIsFinished:
    """Training had finished but the container died before "completed" reached the API; the
    row said running with "0 of 1 in the bucket" forever. Me v2, 2026-09-08 15:25 reboot."""

    def test_the_watch_loop_completes_it_once_everything_wanted_is_up(self):
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller.check_publish_requests)
        assert 'row.get("status") in ("running", "claimed")' in src
        assert 'self._patch(job, {"status": "completed"' in src
        # only after the sweep, so a missing final is queued before it is judged
        assert src.index("self._sweep_checkpoints(job)") < src.index('"status": "completed"')


class TestATickWithNoWorkerIdIsLoud:
    """The first night the trainer polled with no worker id and silently returned: the worker
    row was green and heartbeating, the service answered ready, the store was empty -- and
    Payton v1 sat pending for an hour while the poll tick said nothing to no one. Every other
    part of that failure can be seen by proxy; this line is the only thing that says the queue
    is not empty, the box just cannot ask."""

    def test_a_tick_with_no_id_says_so_and_claims_nothing(self, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        said = []
        monkeypatch.setattr(mod, "_log", lambda m: said.append(m))
        p = mod.Poller(client=None, worker_id_getter=lambda: None)

        class C:
            async def get(self, *a, **k):
                raise AssertionError("the poll must never reach the API without an id")
        p._client = C()

        async def go():
            await p.tick()
        _run(go())
        assert said and "no worker id" in said[0]

    def test_the_warning_repeats_on_its_interval_not_every_tick(self, monkeypatch):
        import time as time_mod
        from wanly_worker.services.lora_trainer import poller as mod
        said = []
        monkeypatch.setattr(mod, "_log", lambda m: said.append(m))
        p = mod.Poller(client=None, worker_id_getter=lambda: None)

        async def go():
            await p.tick()
            await p.tick()
            await p.tick()
        real_time = time_mod.time
        t = [1000.0]

        def fake_time():
            return t[0]
        monkeypatch.setattr(mod.time, "time", fake_time)
        _run(go())
        assert len(said) == 1
        # Past the interval, and it is said again -- three silent hours must not be possible.
        t[0] = 1000.0 + mod.NO_WORKER_ID_LOG_S + 1
        _run(go())
        assert len(said) == 2

    def test_a_tick_with_an_id_stays_quiet(self, monkeypatch):
        from wanly_worker.services.lora_trainer import poller as mod
        said = []
        monkeypatch.setattr(mod, "_log", lambda m: said.append(m))
        p = mod.Poller(client=lambda *a, **k: None, worker_id_getter=lambda: "w")
        # tick would now call check_publish_requests against a disabled env: QUEUE_URL unset.
        monkeypatch.setattr(mod, "QUEUE_URL", "")
        monkeypatch.setattr(mod, "QUEUE_API_KEY", "")
        app_mod = __import__("wanly_worker.services.lora_trainer.app", fromlist=["STORE"])

        class NowhereStore:
            def claim_slot(self):
                return False
        monkeypatch.setattr(app_mod, "STORE", NowhereStore())

        async def go():
            await p.tick()
        _run(go())
        assert said == []


class TestTheJointRun:
    """A joint two-identity run (wanly-gpu-docker#102): ONE LoRA trained on both identity
    groups at once. Per-group captions and dirs, a two-entry dataset toml, and the
    single-identity shape byte-unchanged — the toml is the regression trail."""

    def test_single_identity_toml_is_unchanged(self):
        from wanly_worker.services.lora_trainer.recipe import dataset_toml
        toml = dataset_toml(Path("/tmp/r"))
        assert toml.count("[[datasets]]") == 1
        # The bare dirs, exactly as before #102.
        assert 'image_directory = "/tmp/r/data"' in toml
        assert 'cache_directory = "/tmp/r/cache"' in toml
        # No numbered dirs in the single-identity shape.
        assert "data0" not in toml and "data1" not in toml

    def test_joint_toml_has_one_entry_per_group(self):
        from wanly_worker.services.lora_trainer.recipe import dataset_toml
        toml = dataset_toml(Path("/tmp/r"), groups=[
            {"data": "/tmp/r/data", "cache": "/tmp/r/cache", "num_repeats": 10},
            {"data": "/tmp/r/data1", "cache": "/tmp/r/cache1", "num_repeats": 10},
        ])
        assert toml.count("[[datasets]]") == 2
        assert 'image_directory = "/tmp/r/data1"' in toml
        assert 'cache_directory = "/tmp/r/cache1"' in toml

    def test_stage_writes_per_group_captions(self, monkeypatch, tmp_path):
        """Group 1's images caption GROUP 1's trigger — group 0's would bind the second
        face to the wrong person's token, and the interference this run exists to escape
        would arrive through the captions instead."""
        import asyncio
        from wanly_worker.services.lora_trainer import pipeline
        from wanly_worker.services.lora_trainer.jobs import Job

        monkeypatch.setattr(pipeline.recipe, "RUNS_DIR", str(tmp_path))
        job = Job(id="j1", character="pay", trigger="p@y", version=1, steps=1200)
        img0 = [(f"a{i}.jpg", b"x") for i in range(2)]
        img1 = [(f"b{i}.jpg", b"y") for i in range(2)]
        run = asyncio.run(pipeline.stage(job, [
            {"images": img0, "caption": "p@y, woman", "num_repeats": 10},
            {"images": img1, "caption": "d@vid, man", "num_repeats": 10},
        ]))
        assert (run / "data" / "sel_000.txt").read_text() == "p@y, woman\n"
        assert (run / "data1" / "sel_000.txt").read_text() == "d@vid, man\n"
        toml = (run / "dataset.toml").read_text()
        assert toml.count("[[datasets]]") == 2
        assert 'image_directory = "%s/data1"' % run in toml

    def test_joint_poller_map_carries_the_second_group(self):
        """The claim's second_* land in the TrainRequest; absent stays absent."""
        import inspect
        from wanly_worker.services.lora_trainer import poller as mod
        src = inspect.getsource(mod.Poller.tick)
        assert "second_download_urls" in src
        assert "second_identity" in src
