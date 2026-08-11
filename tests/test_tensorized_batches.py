"""Equivalence tests: TensorizedSequenceCollator vs SequenceBatchCollator.

The vectorized fast path must produce batches identical to the legacy
per-example collator — same keys, dtypes, shapes, and values (including NaN
payloads and missing-semantic-ID rows) — so that TIGER/SFT, GFlow
(example_indices), reward models (history_ratings / target_ratings), and
candidate samplers (target_token_ids) behave the same on either path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.collators import SequenceBatchCollator
from flowrec.data.datasets import SequenceDataset
from flowrec.data.tensor_batches import (
    SequenceExampleArrays,
    TensorizedSequenceDataset,
    maybe_tensorized_parts,
)


ID2ITEM = ["<PAD>", "A1", "A2", "A3", "A4", "A5"]
# A4 deliberately has no semantic ID (missing-item path); token 0 is valid.
SEMANTIC_IDS = {
    "A1": [1, 259, 517],
    "A2": [0, 256, 512],
    "A3": [7, 300, 700],
    "A5": [200, 400, 600],
}


def example_frame() -> pd.DataFrame:
    rows = [
        # Ragged histories, NaN ratings, None/str verified flags, item without
        # semantic ID (A4), and a repeated user for the hashing path.
        dict(
            raw_user_id="u1",
            user_id=1,
            history_raw_items=["A1", "A2", "A4"],
            history_item_ids=[1, 2, 4],
            target_raw_item="A3",
            target_item_id=3,
            target_rating=5.0,
            target_timestamp=1000,
            target_verified_purchase=True,
            history_ratings=[4.0, float("nan"), 1.0],
            history_verified_purchase=[True, None, "false"],
            split="train",
            target_position=3,
        ),
        dict(
            raw_user_id="u2",
            user_id=2,
            history_raw_items=["A5"],
            history_item_ids=[5],
            target_raw_item="A4",  # target without semantic ID
            target_item_id=4,
            target_rating=float("nan"),
            target_timestamp=2000,
            target_verified_purchase=None,
            history_ratings=[float("nan")],
            history_verified_purchase=[float("nan")],
            split="train",
            target_position=1,
        ),
        dict(
            raw_user_id="u1",
            user_id=1,
            history_raw_items=["A2", "A3", "A5", "A1", "A2"],
            history_item_ids=[2, 3, 5, 1, 2],
            target_raw_item="A1",
            target_item_id=1,
            target_rating=3.0,
            target_timestamp=3000,
            target_verified_purchase=False,
            history_ratings=[],  # legacy default-fill path
            history_verified_purchase=[],
            split="train",
            target_position=5,
        ),
        dict(
            raw_user_id="u3",
            user_id=3,
            history_raw_items=["A3", "A4", "A1", "A5"],
            history_item_ids=[3, 4, 1, 5],
            target_raw_item="A2",
            target_item_id=2,
            target_rating=2.0,
            target_timestamp=4000,
            target_verified_purchase="true",
            history_ratings=[1.0, 2.0, 3.0, 4.0],
            history_verified_purchase=[False, True, False, True],
            split="train",
            target_position=4,
        ),
    ]
    return pd.DataFrame(rows)


def make_collator(**overrides) -> SequenceBatchCollator:
    kwargs = dict(
        max_history_len=4,
        history_padding="left",
        pad_item_id=0,
        pad_token_id=0,
        semantic_ids=SEMANTIC_IDS,
        sid_len=3,
        include_raw=False,
        user_hash_mod=1000,
    )
    kwargs.update(overrides)
    return SequenceBatchCollator(**kwargs)


def assert_batches_equal(legacy: dict, fast: dict) -> None:
    assert set(legacy) == set(fast), (
        f"key mismatch: {sorted(set(legacy) ^ set(fast))}"
    )
    for key, expected in legacy.items():
        actual = fast[key]
        assert torch.is_tensor(expected) == torch.is_tensor(actual), key
        if not torch.is_tensor(expected):
            assert expected == actual, key
            continue
        assert expected.dtype == actual.dtype, f"{key}: dtype"
        assert expected.shape == actual.shape, f"{key}: shape"
        if expected.is_floating_point():
            both_nan = torch.isnan(expected) & torch.isnan(actual)
            assert torch.equal(
                expected[~both_nan], actual[~both_nan]
            ), f"{key}: values"
        else:
            assert torch.equal(expected, actual), f"{key}: values"


def legacy_batch(dataset: SequenceDataset, collator, indices) -> dict:
    return collator([dataset[i] for i in indices])


def fast_batch(dataset: SequenceDataset, collator, indices) -> dict:
    parts = maybe_tensorized_parts(dataset, collator, id2item=ID2ITEM, enabled=True)
    assert parts is not None
    _, fast_collate = parts
    return fast_collate(list(indices))


@pytest.mark.parametrize("padding", ["left", "right"])
@pytest.mark.parametrize("hash_mod", [None, 1000])
@pytest.mark.parametrize("with_semantic", [True, False])
def test_batches_identical(padding, hash_mod, with_semantic):
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator(
        history_padding=padding,
        user_hash_mod=hash_mod,
        semantic_ids=SEMANTIC_IDS if with_semantic else None,
        sid_len=3 if with_semantic else None,
    )
    for indices in ([0, 1, 2, 3], [2], [3, 0], [1, 1, 2]):
        assert_batches_equal(
            legacy_batch(dataset, collator, indices),
            fast_batch(dataset, collator, indices),
        )


def test_histories_longer_than_width_are_clipped_identically():
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator(max_history_len=2)
    for indices in ([0, 1, 2, 3], [2]):
        assert_batches_equal(
            legacy_batch(dataset, collator, indices),
            fast_batch(dataset, collator, indices),
        )


def test_include_raw_falls_back_to_legacy():
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator(include_raw=True)
    assert (
        maybe_tensorized_parts(dataset, collator, id2item=ID2ITEM, enabled=True) is None
    )


def test_disabled_flag_falls_back_to_legacy():
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator()
    assert (
        maybe_tensorized_parts(dataset, collator, id2item=ID2ITEM, enabled=False) is None
    )


def test_dataloader_end_to_end_matches_legacy():
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator()
    parts = maybe_tensorized_parts(dataset, collator, id2item=ID2ITEM, enabled=True)
    assert parts is not None
    fast_dataset, fast_collate = parts
    assert isinstance(fast_dataset, TensorizedSequenceDataset)
    # loader.dataset keeps exposing the examples frame for downstream scripts.
    assert fast_dataset.examples is dataset.examples

    legacy_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collator)
    fast_loader = DataLoader(
        fast_dataset, batch_size=3, shuffle=False, collate_fn=fast_collate
    )
    for legacy, fast in zip(legacy_loader, fast_loader, strict=True):
        assert_batches_equal(legacy, fast)


def test_arrays_cached_per_dataset():
    dataset = SequenceDataset(example_frame(), split="train")
    first = SequenceExampleArrays.for_dataset(dataset)
    second = SequenceExampleArrays.for_dataset(dataset)
    assert first is second


def test_feature_length_mismatch_raises():
    frame = example_frame()
    frame.at[0, "history_ratings"] = [1.0, 2.0]  # 2 values vs 3 history items
    dataset = SequenceDataset(frame, split="train")
    with pytest.raises(ValueError, match="history_ratings"):
        SequenceExampleArrays.for_dataset(dataset)


def test_nan_payloads_survive():
    dataset = SequenceDataset(example_frame(), split="train")
    collator = make_collator()
    fast = fast_batch(dataset, collator, [1])
    assert math.isnan(float(fast["target_ratings"][0]))
    # NaN rating sits at the last (left-padded) history position.
    assert bool(torch.isnan(fast["history_ratings"][0]).any())
