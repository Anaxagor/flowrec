"""Training metadata projection must preserve examples without decoding payloads."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.dense_prefix import example_digest
from flowrec.data.preprocessors import load_metadata_frame


def _metadata_with_fallbacks():
    return pd.DataFrame({
        "parent_asin": ["parent-a", None, None, None, None, "parent-a", "", None],
        "asin": ["asin-a", "asin-b", None, None, None, "asin-z", "asin-empty", None],
        "raw_item_id": ["raw-a", "raw-b", "raw-c", None, None, "raw-z", "raw-empty", None],
        "item_id": ["item-a", "item-b", "item-c", "item-d", None, "item-z", "item-empty", None],
        "title": [f"title-{i}" for i in range(8)],
        # A bounded version of the thousands-of-fields Beauty2023 struct.
        "details": [{f"field-{j}": "value" if j == i else None for j in range(256)}
                    for i in range(8)],
    })


@pytest.mark.parametrize("item_key, expected", [
    ("parent_asin", ["parent-a", "asin-b", "raw-c", "item-d", ""]),
    ("asin", ["asin-a", "asin-b", "raw-c", "item-d", "asin-z", "asin-empty"]),
])
def test_parquet_projects_ids_before_decoding_wide_metadata(tmp_path, monkeypatch, item_key, expected):
    path = tmp_path / "metadata.parquet"
    _metadata_with_fallbacks().to_parquet(path)
    reference = load_metadata_frame(path, item_key=item_key)
    read_parquet = pd.read_parquet
    reads = []

    def projected_read(*args, **kwargs):
        columns = kwargs.get("columns")
        assert columns is not None, "The full metadata payload must never be decoded"
        assert set(columns) == {"parent_asin", "asin", "raw_item_id", "item_id"}
        assert len(columns) == len(set(columns))
        reads.append(list(columns))
        return read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", projected_read)
    projected = load_metadata_frame(path, item_key=item_key, ids_only=True)
    assert reads
    assert projected.columns.tolist() == ["raw_item_id"]
    assert projected.raw_item_id.tolist() == expected
    pd.testing.assert_frame_equal(projected, reference[["raw_item_id"]])
    assert "details" in reference and "title" in reference


@pytest.mark.parametrize("available_key", ["parent_asin", "asin", "raw_item_id", "item_id", None])
def test_missing_metadata_id_columns_preserve_fallback_behavior(tmp_path, available_key):
    frame = pd.DataFrame({"title": ["first", "second", "duplicate", "missing"]})
    if available_key is not None:
        frame[available_key] = ["item-b", "item-a", "item-b", None]
    path = tmp_path / "metadata.parquet"
    frame.to_parquet(path)
    reference = load_metadata_frame(path, item_key="parent_asin")
    projected = load_metadata_frame(path, item_key="parent_asin", ids_only=True)
    pd.testing.assert_frame_equal(projected, reference[["raw_item_id"]])
    assert projected.raw_item_id.tolist() == ([] if available_key is None else ["item-b", "item-a"])


def test_dataframe_projection_preserves_input_and_first_occurrence():
    frame = _metadata_with_fallbacks()
    before = frame.copy(deep=True)
    reference = load_metadata_frame(frame, item_key="parent_asin")
    projected = load_metadata_frame(frame, item_key="parent_asin", ids_only=True)
    pd.testing.assert_frame_equal(projected, reference[["raw_item_id"]])
    pd.testing.assert_frame_equal(frame, before)
    assert reference.loc[reference.raw_item_id.eq("parent-a"), "title"].tolist() == ["title-0"]


def _amazon_config(tmp_path):
    rows = [
        {"user_id": str(user), "asin": str(i % 6), "timestamp": 1000 + i,
         "rating": 5.0, "verified_purchase": True}
        for user, length in enumerate((6, 10, 8)) for i in range(length)
    ]
    rows.extend([
        # This item has semantic IDs but is excluded by metadata filtering.
        {"user_id": "0", "asin": "outside-metadata", "timestamp": 1002,
         "rating": 5.0, "verified_purchase": True},
        # This item has metadata but lacks semantic IDs.
        {"user_id": "1", "asin": "outside-semantic-ids", "timestamp": 1003,
         "rating": 5.0, "verified_purchase": True},
        {"user_id": "1", "asin": "2", "timestamp": 1004,
         "rating": 1.0, "verified_purchase": False},
        # An excluded auxiliary event must stay excluded as well.
        {"user_id": "2", "asin": "outside-metadata", "timestamp": 1004,
         "rating": 2.0, "verified_purchase": False},
    ])
    reviews = tmp_path / "reviews.parquet"
    pd.DataFrame(rows).to_parquet(reviews)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame([
        {"parent_asin": str(i) if i % 2 == 0 else None, "asin": str(i),
         "title": f"Product {i}", "details": {"color": "red", "unused": None}}
        for i in range(6)
    ] + [
        {"parent_asin": "outside-semantic-ids", "asin": None,
         "title": "No semantic IDs", "details": None},
        {"parent_asin": "0", "asin": "alternate-duplicate",
         "title": "Duplicate item", "details": None},
    ]).to_parquet(metadata)
    sids = tmp_path / "sids.json"
    sids.write_text(json.dumps({**{str(i): [i // 3, i % 3] for i in range(6)},
                               "outside-metadata": [2, 0]}))
    return AmazonReviewDataConfig(
        reviews_path=reviews, metadata_path=metadata, semantic_id_path=sids,
        metadata_item_key="parent_asin", filter_reviews_to_metadata=True,
        positive_criterion="verified_purchase == True", min_user_interactions=3,
        min_item_interactions=0, max_history_len=3, filter_cold_items_after_split=False,
        train_example_layout="prefix_target", train_prefix_from_dense=True,
        history_padding="left", include_raw_in_batch=False, num_workers=0,
        persistent_workers=False, processed_cache_dir=tmp_path / "processed",
    )


def test_projected_metadata_preserves_dense_prefix_data_and_reward_inputs(tmp_path):
    full = AmazonReviewDataModule(_amazon_config(tmp_path))
    projected = AmazonReviewDataModule(replace(full.cfg, metadata_ids_only=True))
    assert full._processed_cache_path() != projected._processed_cache_path()
    full.setup()
    projected.setup()

    assert full.item2id == projected.item2id
    assert full.user2id == projected.user2id
    assert full.semantic_ids == projected.semantic_ids
    assert "outside-metadata" not in projected.item2id
    assert "outside-semantic-ids" not in projected.item2id
    assert len(projected.train_dataset) == 15
    for split in ("train", "val", "test"):
        left = getattr(full, f"{split}_dataset").examples
        right = getattr(projected, f"{split}_dataset").examples
        pd.testing.assert_frame_equal(left, right)
        assert example_digest(left) == example_digest(right)
    pd.testing.assert_frame_equal(full.interactions, projected.interactions)
    assert full.dense_prefix_audit == projected.dense_prefix_audit
    assert full.stats == projected.stats
    assert len(full.interaction_store) == len(projected.interaction_store) == 1
    pd.testing.assert_frame_equal(full.interaction_store.events, projected.interaction_store.events)
    assert full.interaction_store.stats == projected.interaction_store.stats
    assert full._metadata_by_item
    assert projected._metadata_by_item == {}
    assert projected.metadata.columns.tolist() == ["raw_item_id"]
    assert projected._processed_cache_path().is_file()

    reloaded = AmazonReviewDataModule(projected.cfg)
    reloaded.setup()
    assert reloaded._metadata_by_item == {}
    assert reloaded.dense_prefix_audit == projected.dense_prefix_audit
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(reloaded, f"{split}_dataset").examples,
                                      getattr(projected, f"{split}_dataset").examples)
