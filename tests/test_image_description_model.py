"""Rebuilding joycaption:beta-one from public bytes (#3).

The single most valuable test here is the first one: it hashes the layers actually shipped in
this repo and compares them to the digests the table declares. Those 1,070 bytes are the only
part of the captioner with no upstream — the llama-3 template, the uncensored system prompt,
and temperature 0.4 / top_p 0.9 — and if they are edited or corrupted the model still builds,
still answers, and quietly captions differently from everything console#405 was tuned against.
"""
import hashlib
import json

import pytest

from wanly_worker.services.image_description import model as jm


def test_the_layers_shipped_in_this_repo_match_their_declared_digests():
    """Guards the bytes, not the filenames. A whitespace edit to the system prompt changes the
    model's behaviour and would otherwise be invisible."""
    for blob in [*jm.BLOBS, jm.CONFIG]:
        if blob.remote:
            continue
        src = jm.local_source(blob)
        assert src.is_file(), f"{blob.kind} layer missing from the image at {src}"
        data = src.read_bytes()
        assert len(data) == blob.size, f"{blob.kind}: {len(data)} bytes, declared {blob.size}"
        assert hashlib.sha256(data).hexdigest() == blob.digest, \
            f"{blob.kind} does not match its declared digest — it has been modified"


def test_the_config_lists_exactly_the_layers_we_ship():
    """ollama's config carries diff_ids for every layer. If the table and the config disagree,
    ollama has a model whose parts are not the parts we think they are."""
    cfg = json.loads(jm.local_source(jm.CONFIG).read_bytes())
    assert cfg["rootfs"]["diff_ids"] == [f"sha256:{b.digest}" for b in jm.BLOBS]


def test_the_two_fetched_layers_are_the_gguf_files():
    remote = [b for b in jm.BLOBS if b.remote]
    assert len(remote) == 2
    assert all(b.filename.endswith(".gguf") for b in remote)
    # The digest IS the HF LFS oid; that equality is the whole basis for not hosting this.
    model = next(b for b in remote if b.kind == "model")
    assert model.digest == "e8ae55dd07e61d541ab741d6ed63e7810192cea65d7ef8cda69b2a99fb06dc15"
    assert model.size == 4920735936


def test_the_upstream_is_pinned_to_a_commit():
    """A moving branch would hand us different bytes with no warning. The digest check would
    catch it, but only after transferring 4.6 GB."""
    assert len(jm.HF_REVISION) == 40 and jm.HF_REVISION.isalnum()
    assert jm.HF_REVISION in jm.BASE_URL


class TestPresence:
    def test_a_truncated_blob_does_not_count_as_present(self, tmp_path):
        """The failure this exists for: a short blob satisfies exists(), passes everything that
        is not a length check, and dies at load — inside somebody's caption."""
        blob = jm.BLOBS[0]
        p = tmp_path / "models" / "blobs"
        p.mkdir(parents=True)
        (p / f"sha256-{blob.digest}").write_bytes(b"x" * 100)
        assert not jm.have(str(tmp_path), blob)
        assert blob in jm.missing(str(tmp_path))

    def test_a_full_length_blob_counts(self, tmp_path):
        blob = jm.BLOBS[2]        # the 254-byte template, cheap to write
        p = tmp_path / "models" / "blobs"
        p.mkdir(parents=True)
        (p / f"sha256-{blob.digest}").write_bytes(b"x" * blob.size)
        assert jm.have(str(tmp_path), blob)


class TestTheManifest:
    def test_it_carries_no_absolute_paths(self):
        """The original's `from` fields point into 2070.zero's store and mean nothing on any
        other machine."""
        assert "from" not in json.dumps(jm.manifest())
        assert "/usr/share/ollama" not in json.dumps(jm.manifest())

    def test_it_lists_every_layer_in_order(self):
        m = jm.manifest()
        assert [l["digest"] for l in m["layers"]] == [f"sha256:{b.digest}" for b in jm.BLOBS]
        assert m["config"]["digest"] == f"sha256:{jm.CONFIG.digest}"


class TestProvisioning:
    """The download path, with a fake transport — no network, no 4.6 GB."""

    class _Resp:
        def __init__(self, body, status=200, declared=None):
            self.status_code = status
            self._body = body
            self.headers = {"content-length": str(declared if declared is not None
                                                  else len(body))}

        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def aiter_bytes(self, n):
            for i in range(0, len(self._body), n):
                yield self._body[i:i + n]

    class _Client:
        def __init__(self, resp): self._resp = resp
        def stream(self, *a, **k): return self._resp

    def _store(self, tmp_path):
        (tmp_path / "models" / "blobs").mkdir(parents=True)
        return str(tmp_path)

    def _one_remote_blob(self, monkeypatch, body):
        """Reduce the table to a single small remote layer so the test is about the mechanism."""
        blob = jm.Blob("model", "application/vnd.ollama.image.model",
                       hashlib.sha256(body).hexdigest(), len(body), "tiny.gguf")
        monkeypatch.setattr(jm, "BLOBS", [blob])
        return blob

    def test_a_good_download_lands_under_its_digest(self, tmp_path, monkeypatch):
        body = b"pretend gguf" * 1000
        blob = self._one_remote_blob(monkeypatch, body)
        store = self._store(tmp_path)
        import asyncio
        asyncio.run(jm.provision(store, self._Client(self._Resp(body)), log=lambda *_: None))
        landed = tmp_path / "models" / "blobs" / f"sha256-{blob.digest}"
        assert landed.read_bytes() == body
        assert (tmp_path / "models" / jm.MANIFEST_PATH).is_file()

    def test_a_corrupted_download_is_rejected_and_leaves_nothing(self, tmp_path, monkeypatch):
        body = b"pretend gguf" * 1000
        blob = self._one_remote_blob(monkeypatch, body)
        store = self._store(tmp_path)
        tampered = b"x" + body[1:]
        import asyncio
        with pytest.raises(jm.ProvisionError, match="hashed to"):
            asyncio.run(jm.provision(store, self._Client(self._Resp(tampered)),
                                     log=lambda *_: None))
        blobs = tmp_path / "models" / "blobs"
        assert not (blobs / f"sha256-{blob.digest}").exists()
        assert not list(blobs.glob("*.part")), "left a partial behind"
        # And crucially no manifest: a half-model that ollama lists and cannot load is worse
        # than no model at all.
        assert not (tmp_path / "models" / jm.MANIFEST_PATH).exists()

    def test_a_wrong_length_upstream_is_caught_before_hashing(self, tmp_path, monkeypatch):
        body = b"pretend gguf" * 1000
        self._one_remote_blob(monkeypatch, body)
        store = self._store(tmp_path)
        import asyncio
        with pytest.raises(jm.ProvisionError, match="upstream, expected"):
            asyncio.run(jm.provision(store, self._Client(self._Resp(body, declared=999)),
                                     log=lambda *_: None))

    def test_an_http_error_is_reported_not_written(self, tmp_path, monkeypatch):
        self._one_remote_blob(monkeypatch, b"body")
        store = self._store(tmp_path)
        import asyncio
        with pytest.raises(jm.ProvisionError, match="HTTP 404"):
            asyncio.run(jm.provision(store, self._Client(self._Resp(b"", status=404)),
                                     log=lambda *_: None))

    def test_it_refuses_when_the_disk_cannot_hold_it(self, tmp_path, monkeypatch):
        self._one_remote_blob(monkeypatch, b"body")
        monkeypatch.setattr(jm, "BUILD_HEADROOM_GB", 10 ** 9)
        store = self._store(tmp_path)
        import asyncio
        with pytest.raises(jm.ProvisionError, match="free"):
            asyncio.run(jm.provision(store, self._Client(self._Resp(b"body")),
                                     log=lambda *_: None))
