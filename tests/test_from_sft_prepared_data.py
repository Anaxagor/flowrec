"""Prepared CPU snapshots preserve batches and never rerun preprocessing."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp7-from-sft"))
import prepared_data as prepared
import from_sft_runtime as runtime
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.datasets import SequenceDataset

SPEC = importlib.util.spec_from_file_location("prepared_example_helpers", ROOT / "tests/test_tensorized_batches.py")
examples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(examples)


@contextmanager
def no_lock(_path):
    yield


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Exercise real dataset/collator classes with ragged histories, missing
    # semantic IDs and NaN payloads. Raw preprocessing is tested separately
    # by the actual Instruments roundtrip smoke.
    cfg = AmazonReviewDataConfig(reviews_path=tmp_path / "reviews.parquet", batch_size=2,
        eval_batch_size=2, num_workers=0, persistent_workers=False, pin_memory=False,
        include_raw_in_batch=False, max_history_len=5, user_hash_mod=1)
    dm = AmazonReviewDataModule(cfg)
    dm.id2item = examples.ID2ITEM
    dm.item2id = {raw: i for i, raw in enumerate(dm.id2item)}
    dm.semantic_ids, dm.sid_len = examples.SEMANTIC_IDS, 3
    frame = examples.example_frame()
    for split in ("train", "val", "test"):
        data = frame.copy()
        data["split"] = split
        setattr(dm, split + "_dataset", SequenceDataset(data, split=split))
    dm._is_setup = True
    session = SimpleNamespace(cache_root=tmp_path, config={}, config_path=tmp_path / "config.json",
        source={"source_id": "tiny_source"}, input_identity={"source": "sft-only", "data": "frozen"},
        current_cfg=cfg)
    session.base = SimpleNamespace(control=SimpleNamespace(file_lock=no_lock),
        tiger_runner=SimpleNamespace(build_data_config=lambda *_args, **_kwargs: session.current_cfg))
    monkeypatch.setattr(runtime, "input_identity", lambda session: session.input_identity)
    monkeypatch.setattr(prepared, "code_identity", lambda: {"data.py": "original-code-hash"})
    return session, dm


def assert_batches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, equal_nan=True, rtol=0, atol=0)
        else:
            assert actual[key] == value


def test_roundtrip_preserves_batches_and_arrays_without_setup(fixture, monkeypatch):
    session, dm = fixture
    before = next(iter(dm.val_dataloader()))
    dm.train_dataset[0]  # Disposable worker records must not be persisted.
    original_records = dm.train_dataset._records
    artifact = prepared.save_prepared_data(session, dm)
    assert artifact.is_file() and prepared.snapshot_ready(session)
    assert dm.train_dataset._records is original_records
    monkeypatch.setattr(AmazonReviewDataModule, "setup", lambda *_a, **_k: pytest.fail("setup called on reload"))
    loaded = prepared.load_prepared_data(session)
    assert loaded._is_setup and loaded.train_dataset._records is None
    assert loaded.semantic_ids == dm.semantic_ids and loaded.item2id == dm.item2id
    for split in ("train", "val", "test"):
        original, restored = getattr(dm, split + "_dataset"), getattr(loaded, split + "_dataset")
        pd.testing.assert_frame_equal(restored.examples, original.examples)
        assert hasattr(restored, "_tensorized_arrays_cache")
    assert_batches_equal(next(iter(loaded.val_dataloader())), before)
    manifest = prepared.check_prepared_data(session, verify_hashes=True)
    assert manifest["buffer_ranges"] and manifest["buffers"]["bytes"] > 0
    assert prepared.save_prepared_data(session, loaded) == artifact


def test_reloaded_operational_loader_settings_do_not_change_examples(fixture):
    session, dm = fixture
    prepared.save_prepared_data(session, dm)
    session.current_cfg = replace(dm.cfg, batch_size=3, eval_batch_size=1, test_batch_size=4,
        num_workers=2, persistent_workers=True, pin_memory=True, train_drop_last=True)
    loaded = prepared.load_prepared_data(session)
    assert loaded.cfg == session.current_cfg
    assert len(loaded.train_dataset) == len(dm.train_dataset)
    assert len(loaded.val_dataloader()) == len(dm.val_dataset)
    assert len(loaded.test_dataloader()) == (len(dm.test_dataset) + 3) // 4
    # Avoid spawning workers in this unit test; compare actual tensor content
    # using the changed batch protocol in both original and restored modules.
    loaded.cfg = dm.cfg = replace(session.current_cfg, num_workers=0, persistent_workers=False)
    assert_batches_equal(next(iter(loaded.test_dataloader())), next(iter(dm.test_dataloader())))


def test_mapped_tensor_mutations_are_private_to_process_and_cache_stays_valid(fixture):
    session, dm = fixture
    prepared.save_prepared_data(session, dm)
    loaded = prepared.load_prepared_data(session)
    array = loaded.train_dataset._tensorized_arrays_cache.history_items
    old = array[0].item()
    array[0] = -99
    again = prepared.load_prepared_data(session)
    assert again.train_dataset._tensorized_arrays_cache.history_items[0].item() == old
    assert dm.train_dataset._tensorized_arrays_cache.history_items[0].item() == old
    prepared.check_prepared_data(session, verify_hashes=True)


def test_missing_snapshot_returns_none_and_partial_save_never_publishes(fixture, monkeypatch):
    session, dm = fixture
    assert prepared.load_prepared_data(session) is None
    assert not prepared.snapshot_ready(session)
    monkeypatch.setattr(prepared._StreamingPickler, "dump", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        prepared.save_prepared_data(session, dm)
    assert not prepared.snapshot_ready(session)
    assert not list(prepared.cache_directory(session).glob("*.pkl"))
    assert not list(prepared.cache_directory(session).glob("*.buffers"))


def test_changed_source_or_preprocessing_code_is_rejected_before_unpickle(fixture, monkeypatch):
    session, dm = fixture
    prepared.save_prepared_data(session, dm)
    session.input_identity["data"] = "changed"
    with pytest.raises(RuntimeError, match="inputs/code/runtime changed"):
        prepared.load_prepared_data(session)
    session.input_identity["data"] = "frozen"
    monkeypatch.setattr(prepared, "code_identity", lambda: {"data.py": "new-code-hash"})
    with pytest.raises(RuntimeError, match="inputs/code/runtime changed"):
        prepared.snapshot_ready(session)


def test_changed_input_cfg_cannot_be_mistaken_for_worker_override(fixture):
    session, dm = fixture
    prepared.save_prepared_data(session, dm)
    session.current_cfg = replace(dm.cfg, max_history_len=50)
    with pytest.raises(RuntimeError, match="input configuration differs"):
        prepared.load_prepared_data(session)


def test_corrupted_artifact_is_rejected_without_raw_fallback(fixture):
    session, dm = fixture
    artifact = prepared.save_prepared_data(session, dm)
    artifact.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="artifact changed"):
        prepared.load_prepared_data(session)


def test_explicit_digest_audit_detects_same_size_same_mtime_replacement(fixture):
    session, dm = fixture
    prepared.save_prepared_data(session, dm)
    manifest = prepared.check_prepared_data(session)
    path = prepared.cache_directory(session) / manifest["buffers"]["file"]
    stat = path.stat()
    with path.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepared.check_prepared_data(session, verify_hashes=True)
