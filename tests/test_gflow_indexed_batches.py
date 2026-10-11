"""Reward-cache indices and batch semantics survive tensorized loading."""

from __future__ import annotations

import multiprocessing
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.collators import DenseCausalCollator, SequenceBatchCollator
from flowrec.data.datasets import SequenceDataset
from flowrec.data.indexed_batches import (
    ExamplePositions,
    IndexedRecordCollator,
    IndexedTensorCollator,
    create_indexed_parts,
)


ID2ITEM = ["<PAD>", "a", "b", "c", "missing"]
SEMANTIC_IDS = {"a": [0, 4, 8], "b": [2, 5, 9], "c": [3, 6, 10]}


def example_frame():
    rows = []
    for i, history in enumerate(([1, 4, 2], [], [2], [3, 2, 1, 4, 1])):
        rows.append(dict(
            raw_user_id=f"user-{i % 2}", user_id=i % 2 + 1,
            history_item_ids=list(history),
            history_raw_items=[ID2ITEM[item] for item in history],
            history_ratings=[float("nan") if j == 1 else float(j) for j in range(len(history))],
            history_verified_purchase=[None if j % 2 else "true" for j in range(len(history))],
            target_raw_item=ID2ITEM[i + 1], target_item_id=i + 1,
            target_rating=float("nan") if i == 2 else float(i),
            target_timestamp=100 + i, target_position=len(history),
            target_verified_purchase=None if i == 1 else bool(i % 2),
            split="train",
        ))
    # Exercise the legacy default-fill semantics for optional history data.
    rows[2]["history_ratings"] = []
    rows[2]["history_verified_purchase"] = []
    return pd.DataFrame(rows)


class TinyDataModule:
    def __init__(self, *, enabled=True, include_raw=False, padding="left", semantic=True):
        self.train_dataset = SequenceDataset(example_frame(), "train")
        self.cfg = SimpleNamespace(tensorized_batches=enabled)
        self.id2item = ID2ITEM
        self.include_raw = include_raw
        self.padding = padding
        self.semantic = semantic

    def _make_collator(self):
        return SequenceBatchCollator(
            max_history_len=3, history_padding=self.padding,
            semantic_ids=SEMANTIC_IDS if self.semantic else None,
            sid_len=3 if self.semantic else None,
            include_raw=self.include_raw, user_hash_mod=97,
        )


def legacy_reference(dm, subset, positions):
    records = []
    for position in positions:
        source = position if subset is None else subset[position]
        record = dict(dm.train_dataset.examples.iloc[source])
        record["example_index"] = position
        records.append(record)
    return dm._make_collator()(records)


def assert_batch_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("subset", [None, [3, 0, 3, 1, 2]])
@pytest.mark.parametrize("padding", ["left", "right"])
@pytest.mark.parametrize("semantic", [False, True])
def test_tensorized_batches_match_original_loader(subset, padding, semantic):
    dm = TinyDataModule(padding=padding, semantic=semantic)
    dataset, collate = create_indexed_parts(dm, subset)
    assert isinstance(dataset, ExamplePositions)
    assert isinstance(collate, IndexedTensorCollator)
    positions = [len(dataset) - 1, 0, 2, 0]
    batch = collate([dataset[index] for index in positions])
    assert_batch_equal(batch, legacy_reference(dm, subset, positions))
    assert batch["example_indices"].tolist() == positions
    assert dm.train_dataset._records is None
    assert not hasattr(dataset, "source")
    assert not hasattr(collate, "dataset")


def test_full_split_uses_no_identity_mapping():
    dm = TinyDataModule()
    dataset, collate = create_indexed_parts(dm, None)
    assert len(dataset) == len(dm.train_dataset)
    assert collate.source_indices is None


@pytest.mark.parametrize("include_raw,enabled", [(True, True), (False, False)])
def test_record_fallback_keeps_subset_positions(include_raw, enabled):
    dm = TinyDataModule(include_raw=include_raw, enabled=enabled)
    subset = [3, 0, 3, 1]
    dataset, collate = create_indexed_parts(dm, subset)
    assert isinstance(collate, IndexedRecordCollator)
    positions = [3, 2, 0]
    assert_batch_equal(collate(positions), legacy_reference(dm, subset, positions))
    assert len(dataset) == len(subset)


class CustomCollator:
    def __call__(self, records):
        return {"custom_targets": torch.tensor([record["target_item_id"] for record in records])}


def test_custom_collator_falls_back_and_receives_original_records():
    dm = TinyDataModule()
    _, collate = create_indexed_parts(dm, [3, 0], collator=CustomCollator())
    assert isinstance(collate, IndexedRecordCollator)
    batch = collate([1, 0])
    assert batch["custom_targets"].tolist() == [1, 4]
    assert batch["example_indices"].tolist() == [1, 0]


def test_explicit_tensorized_opt_out():
    dm = TinyDataModule()
    _, collate = create_indexed_parts(dm, [3, 0], tensorized=False)
    assert isinstance(collate, IndexedRecordCollator)
    assert_batch_equal(collate([1, 0]), legacy_reference(dm, [3, 0], [1, 0]))


def test_dense_collator_keeps_legacy_batch_and_subset_indices():
    dm = TinyDataModule()
    frame = dm.train_dataset.examples
    frame["label_item_ids"] = frame["history_item_ids"].apply(list)
    frame["label_mask"] = frame["history_item_ids"].apply(lambda row: [True] * len(row))
    dense = DenseCausalCollator(max_seq_len=3, include_raw=False)
    _, collate = create_indexed_parts(dm, [3, 0], collator=dense)
    records = [dict(frame.iloc[index], example_index=position) for position, index in [(1, 0), (0, 3)]]
    assert isinstance(collate, IndexedRecordCollator)
    assert_batch_equal(collate([1, 0]), dense(records))


def test_empty_subset_is_empty_without_record_materialization():
    dm = TinyDataModule()
    dataset, _ = create_indexed_parts(dm, [])
    assert len(dataset) == 0
    assert dm.train_dataset._records is None


@pytest.mark.parametrize("subset", [[-1], [4]])
def test_out_of_bounds_subset_fails_early(subset):
    with pytest.raises(IndexError, match="outside"):
        create_indexed_parts(TinyDataModule(), subset)


def test_worker_payload_does_not_serialize_source_records():
    dm = TinyDataModule()
    dataset, collate = create_indexed_parts(dm, [3, 1])
    # A source-only unpicklable value makes accidentally retaining its frame
    # fail here. Flat arrays and the collator remain serializable for spawn.
    dm.train_dataset.examples["unused_payload"] = [lambda: None] * 4
    restored_dataset, restored_collate = pickle.loads(pickle.dumps((dataset, collate)))
    assert len(restored_dataset) == 2
    assert_batch_equal(restored_collate([1, 0]), legacy_reference(dm, [3, 1], [1, 0]))


@pytest.mark.parametrize("context", [None, "spawn", "fork"])
def test_shuffled_dataloader_reproducible_with_workers(context):
    if context is not None and context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} is unavailable")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dm = TinyDataModule()
        subset = [3, 0, 3, 1, 2, 0, 1]
        dataset, collate = create_indexed_parts(dm, subset)

        def load_once():
            kwargs = {} if context is None else {"multiprocessing_context": context, "timeout": 30}
            loader = DataLoader(
                dataset, batch_size=3, shuffle=True, collate_fn=collate,
                num_workers=0 if context is None else 2,
                generator=torch.Generator().manual_seed(44), **kwargs,
            )
            return list(loader)

        batches = load_once()
        repeated = load_once()
        seen = []
        for batch, same in zip(batches, repeated):
            assert_batch_equal(batch, same)
            positions = batch["example_indices"].tolist()
            seen.extend(positions)
            assert_batch_equal(batch, legacy_reference(dm, subset, positions))
        assert sorted(seen) == list(range(len(subset)))
        assert seen != list(range(len(subset)))
        assert dm.train_dataset._records is None
    finally:
        torch.set_num_threads(previous_threads)
