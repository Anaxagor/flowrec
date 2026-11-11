"""Compact CE snapshots preserve selected batches without raw-data rebuilds."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))
import ce_yambda_prepared_data as prepared
from flowrec.data.datasets import DatasetStats, SequenceDataset
from flowrec.data.indexed_batches import create_indexed_parts
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

SPEC = importlib.util.spec_from_file_location("ce_prepared_examples", ROOT / "tests/test_tensorized_batches.py")
examples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(examples)


@pytest.fixture
def fixture(tmp_path):
    cfg = YambdaDataConfig(interactions_path=tmp_path / "unavailable.parquet", dataset_size="500m",
        train_target_policy="last_n", train_last_n=128, max_history_len=4,
        batch_size=2, eval_batch_size=2, num_workers=0, pin_memory=False,
        persistent_workers=False, include_raw_in_batch=False, user_hash_mod=1000)
    dm = YambdaDataModule(cfg)
    dm.id2item = examples.ID2ITEM
    dm.item2id = {raw: i for i, raw in enumerate(dm.id2item)}
    dm.id2user = ["unused", "u1", "u2", "u3"]
    dm.user2id = {raw: i for i, raw in enumerate(dm.id2user)}
    dm.semantic_ids, dm.sid_len = examples.SEMANTIC_IDS, 3
    dm.train_item_counts_by_raw_id = {"A1": 7, "A2": 4, "A3": 9, "A4": 2, "A5": 8}
    dm.interactions = pd.DataFrame({"private_raw_payload": ["must not be serialized"]})
    for split in ("train", "val", "test"):
        frame = examples.example_frame()
        frame["split"] = split
        setattr(dm, split + "_dataset", SequenceDataset(frame, split))
    dm.stats = DatasetStats(num_users=4, num_items=5, num_train_examples=4,
        num_val_examples=4, num_test_examples=4, num_interactions=30)
    dm._is_setup = True
    return tmp_path / "snapshot", dm, {"source": "trial4", "subset": {"n": 50, "seed": 42}}


def _batch(dm, subset, positions, *, tensorized=True):
    _, collate = create_indexed_parts(dm, subset, tensorized=tensorized)
    return collate(positions)


@pytest.mark.parametrize("subset", [None, [2, 0, 2], [3, 1]])
def test_selected_training_and_full_heldout_roundtrip(fixture, monkeypatch, subset):
    directory, dm, identity = fixture
    count = len(dm.train_dataset) if subset is None else len(subset)
    positions = list(range(count))
    expected = _batch(dm, subset, positions)
    expected_eval = next(iter(dm.val_dataloader()))
    audit = {"all_fingerprints_equal": True, "full_pool": len(dm.train_dataset)}
    manifest = prepared.prepare(directory, dm, subset, identity, audit=audit)
    monkeypatch.setattr(YambdaDataModule, "setup", lambda *_a, **_k: pytest.fail("raw setup called"))
    loaded = prepared.load(directory, dm.cfg, identity)
    loaded.setup()
    assert manifest["counts"] == {"train": count, "val": 4, "test": 4}
    assert loaded._ce_prepared_audit == audit
    assert loaded._ce_prepared_full_train_count == 4
    assert loaded._ce_prepared_source_indices.tolist() == (list(range(4)) if subset is None else subset)
    assert loaded.item2id == dm.item2id and loaded.semantic_ids == dm.semantic_ids
    assert loaded.train_item_counts_by_raw_id == dm.train_item_counts_by_raw_id
    assert loaded.interactions.empty and loaded.interaction_store is None
    assert loaded.summary()["num_train_examples"] == count
    assert loaded.summary()["prepared_full_train_examples"] == 4
    for split in ("train", "val", "test"):
        dataset = getattr(loaded, split + "_dataset")
        assert all(dtype.kind != "O" for dtype in dataset.examples.dtypes)
        assert "history_item_ids" not in dataset.examples
        assert dataset._records is None
        assert dataset._ce_prepared
    examples.assert_batches_equal(expected, _batch(loaded, None, positions))
    examples.assert_batches_equal(expected, _batch(loaded, None, positions, tensorized=False))
    examples.assert_batches_equal(expected_eval, next(iter(loaded.val_dataloader())))
    examples.assert_batches_equal(next(iter(dm.test_dataloader())), next(iter(loaded.test_dataloader())))
    # Single-example fallback reconstructs only the requested row, never a
    # list of dictionaries for every training example.
    loaded.train_dataset[0]
    assert loaded.train_dataset._records is None
    prepared.check(directory, identity, verify_hashes=True)


def test_operational_settings_and_seed_can_change_without_rebuilding(fixture):
    directory, dm, identity = fixture
    prepared.prepare(directory, dm, [2, 0], identity)
    cfg = replace(dm.cfg, batch_size=3, eval_batch_size=1, test_batch_size=3,
                  num_workers=2, seed=43, pin_memory=True, persistent_workers=True)
    loaded = prepared.load(directory, cfg, identity)
    assert loaded.cfg == cfg
    assert len(loaded.train_dataset) == 2
    with pytest.raises(RuntimeError, match="input configuration differs"):
        prepared.load(directory, replace(cfg, max_history_len=50), identity)


def test_non_deterministic_protocol_seed_cannot_be_changed(fixture):
    directory, dm, identity = fixture
    dm.cfg = replace(dm.cfg, train_target_policy="all")
    prepared.prepare(directory, dm, None, identity)
    with pytest.raises(RuntimeError, match="input configuration differs"):
        prepared.load(directory, replace(dm.cfg, seed=43), identity)


def test_mapped_tensors_are_private_and_preparation_does_not_mutate_source(fixture):
    directory, dm, identity = fixture
    before = dm.train_dataset.examples.copy(deep=True)
    prepared.prepare(directory, dm, [2, 0], identity)
    assert not hasattr(dm.train_dataset, "_tensorized_arrays_cache")
    loaded = prepared.load(directory, dm.cfg, identity)
    array = loaded.train_dataset._tensorized_arrays_cache.history_items
    value = array[0].item()
    array[0] = -999
    again = prepared.load(directory, dm.cfg, identity)
    assert again.train_dataset._tensorized_arrays_cache.history_items[0].item() == value
    pd.testing.assert_frame_equal(dm.train_dataset.examples, before)
    prepared.check(directory, identity, verify_hashes=True)


def test_missing_cache_never_rebuilds_and_failed_save_does_not_commit(fixture, monkeypatch):
    directory, dm, identity = fixture
    assert prepared.load(directory, dm.cfg, identity) is None
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        prepared.prepare(directory, dm, [1, 2], identity)
    assert not (directory / "manifest.json").exists()
    assert list(directory.glob("*.pkl")) == [] and list(directory.glob("*.pt")) == []


def test_identity_or_subset_change_and_corrupt_artifacts_are_rejected(fixture):
    directory, dm, identity = fixture
    manifest = prepared.prepare(directory, dm, [1, 2], identity)
    with pytest.raises(RuntimeError, match="inputs changed"):
        prepared.load(directory, dm.cfg, {**identity, "subset": {"n": 15, "seed": 42}})
    tensor_path = directory / manifest["artifacts"]["tensors"]["file"]
    tensor_path.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="artifact changed"):
        prepared.load(directory, dm.cfg, identity)


def test_full_checksum_audit_detects_same_size_same_mtime_replacement(fixture):
    directory, dm, identity = fixture
    manifest = prepared.prepare(directory, dm, None, identity)
    path = directory / manifest["artifacts"]["tensors"]["file"]
    stat = path.stat()
    with path.open("r+b") as stream:
        byte = stream.read(1); stream.seek(0); stream.write(bytes([byte[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepared.check(directory, identity, verify_hashes=True)


def test_metadata_checksum_is_always_checked_before_pickle_load(fixture, monkeypatch):
    directory, dm, identity = fixture
    manifest = prepared.prepare(directory, dm, None, identity)
    path = directory / manifest["artifacts"]["metadata"]["file"]
    stat = path.stat()
    with path.open("r+b") as stream:
        byte = stream.read(1); stream.seek(0); stream.write(bytes([byte[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(prepared.pickle, "load", lambda *_a, **_k: pytest.fail("untrusted pickle loaded"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepared.load(directory, dm.cfg, identity)


@pytest.mark.parametrize("indices", [[-1], [4], [[1]], [0, 1, 5]])
def test_out_of_range_or_nonflat_subset_rejected(fixture, indices):
    directory, dm, identity = fixture
    with pytest.raises(ValueError, match="one-dimensional valid source subset"):
        prepared.prepare(directory, dm, indices, identity)


def test_existing_snapshot_reuses_committed_generation(fixture):
    directory, dm, identity = fixture
    first = prepared.prepare(directory, dm, [3, 1], identity)
    assert prepared.prepare(directory, dm, [3, 1], identity) == first
    assert len(list(directory.glob("*.pt"))) == 1
    with pytest.raises(RuntimeError, match="selected source indices differ"):
        prepared.prepare(directory, dm, [1, 3], identity)
