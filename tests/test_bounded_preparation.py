"""Chunked preparation must preserve all original source examples and hashes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp7-from-sft"))
sys.path.insert(0, str(ROOT / "scripts"))
import bounded_preparation as bounded
import gpt2_matched_finetune as matched
from flowrec.data import sequence_builder as builder


def events():
    rows = []
    for user, length in enumerate((6, 18, 170, 205)):
        for position in range(length):
            rows.append({"raw_user_id": f"u{user}", "raw_item_id": f"i{position % 17}",
                         "timestamp": 1_700_000_000 + position,
                         "rating": np.nan if position % 13 == 0 else float(position % 5),
                         "verified_purchase": None if position % 9 == 0 else position % 2 == 0})
    return builder.factorize_interaction_ids(pd.DataFrame(rows))[0]


CONFIGS = [
    builder.SequenceSplitConfig(max_history_len=50),
    builder.SequenceSplitConfig(max_history_len=50, train_target_policy="last_n", train_last_n=128,
                                train_window_len=50, train_window_stride=1),
    builder.SequenceSplitConfig(max_history_len=None, train_target_policy="sampled", train_targets_per_user=11),
    builder.SequenceSplitConfig(max_history_len=5, train_target_policy="hybrid",
                                train_hybrid_recent_n=7, train_hybrid_random_n=8),
]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("chunk_size", [1, 7, 100, 1000])
def test_complete_splits_match_original_exactly(config, chunk_size):
    data = events()
    before = deepcopy(data)
    original = builder._attach_history_lists
    expected = builder.build_sequence_splits(data, cfg=config)
    with bounded.bounded_history_build(chunk_size):
        actual = builder.build_sequence_splits(data, cfg=config)
    assert builder._attach_history_lists is original
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(actual, split), getattr(expected, split), check_exact=True)
    pd.testing.assert_frame_equal(data, before, check_exact=True)


@pytest.mark.parametrize("unsorted", [False, True])
def test_mixed_empty_histories_indices_and_list_independence(unsorted):
    data = builder._prepare_events(events())
    if unsorted:
        data = data.sample(frac=1, random_state=42)
    examples = pd.DataFrame({"user_id": [2, 0, 2, 3, 0, 1],
                             "history_start_idx": [0, 1, 2, 4, 0, 2],
                             "history_end_idx": [3, 1, 5, 4, 0, 7],
                             "preserved": ["a", "b", "c", "d", "e", "f"]}, index=[9, 1, 1, 8, -1, 4])
    expected = examples.copy()
    builder._attach_history_lists(expected, data)
    actual = examples.copy()
    with bounded.bounded_history_build(2):
        builder._attach_history_lists(actual, data)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    histories = [value for column in bounded.HISTORY_COLUMNS for value in actual[column]]
    assert all(isinstance(value, list) for value in histories)
    assert len({id(value) for value in histories}) == len(histories)


@pytest.mark.parametrize("count", [0, 8])
def test_all_empty_histories_preserve_original_types(count):
    examples = pd.DataFrame({"user_id": [0] * count, "history_start_idx": [0] * count,
                             "history_end_idx": [0] * count})
    expected = examples.copy()
    builder._attach_history_lists(expected, events())
    with bounded.bounded_history_build(3):
        builder._attach_history_lists(examples, events())
    pd.testing.assert_frame_equal(examples, expected, check_exact=True)


def test_intermediate_calls_are_bounded_and_sorted_events_are_sliced(monkeypatch):
    calls = []
    original = builder._attach_history_lists
    def record(examples, relevant):
        calls.append((len(examples), len(relevant)))
        assert len(examples) <= 5
        return original(examples, relevant)
    data = events()
    monkeypatch.setattr(builder, "_attach_history_lists", record)
    with bounded.bounded_history_build(5):
        builder.build_sequence_splits(data, cfg=CONFIGS[1])
    assert len(calls) > 20
    # The four-row validation/test splits intentionally use the original
    # single-call fast path; the large train split never re-hashes all events.
    assert max(n for _, n in calls[:-2]) < len(data)


def test_matches_saved_source_audit_hashes():
    data = events()
    cfg = CONFIGS[1]
    expected = builder.build_sequence_splits(data, cfg=cfg)

    def dm(splits):
        item2id = dict(zip(data.raw_item_id, data.item_id))
        return SimpleNamespace(train_dataset=SimpleNamespace(examples=splits.train),
            val_dataset=SequenceView(splits.val), test_dataset=SequenceView(splits.test),
            num_items=len(item2id), item2id=item2id,
            semantic_ids={key: [i, 1] for i, key in enumerate(sorted(item2id))})

    def frame_hash(frame, histories):
        columns = ["raw_user_id", "target_position", "target_raw_item", "target_timestamp"]
        subset = frame[columns].copy()
        if histories:
            subset["history_item_ids"] = frame.history_item_ids.map(tuple)
        return hashlib.sha256(pd.util.hash_pandas_object(subset, index=False).to_numpy().tobytes()).hexdigest()

    reference = dm(expected)
    audit = {"num_train_targets": len(expected.train), "num_items": reference.num_items,
             "num_val_examples": len(expected.val), "num_test_examples": len(expected.test),
             "train_targets_sha256": frame_hash(expected.train, False),
             "train_contexts_sha256": frame_hash(expected.train, True),
             "validation_sha256": frame_hash(expected.val, True),
             "test_sha256": frame_hash(expected.test, True),
             "catalog_sha256": hashlib.sha256(json.dumps(sorted(reference.item2id.items())).encode()).hexdigest(),
             "semantic_ids_sha256": hashlib.sha256(json.dumps(sorted(reference.semantic_ids.items())).encode()).hexdigest()}
    with bounded.bounded_history_build(17):
        actual = builder.build_sequence_splits(data, cfg=cfg)
    result = matched.audit_prefix_data(dm(actual), {"dataset": "yambda_500m_listens", "matched_data_audit": audit})
    assert result["all_fingerprints_equal"]


class SequenceView:
    def __init__(self, frame):
        self.examples = frame

    def __len__(self):
        return len(self.examples)


def test_context_restores_after_errors_and_nested_use():
    original = builder._attach_history_lists
    with bounded.bounded_history_build(3):
        outer = builder._attach_history_lists
        with bounded.bounded_history_build(2):
            assert builder._attach_history_lists is not outer
        assert builder._attach_history_lists is outer
        with pytest.raises(RuntimeError, match="failure"):
            with bounded.bounded_history_build(1):
                raise RuntimeError("failure")
        assert builder._attach_history_lists is outer
    assert builder._attach_history_lists is original
    with bounded.bounded_history_build(0):
        assert builder._attach_history_lists is original


@pytest.mark.parametrize("value", [-1, 2.5, True, "10"])
def test_invalid_chunk_size(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        with bounded.bounded_history_build(value):
            pass
