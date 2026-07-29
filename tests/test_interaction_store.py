"""Positive/auxiliary partition contract and the InteractionLogStore."""

import numpy as np
import pandas as pd
import pytest

from flowrec.data.interaction_store import (
    InteractionLogStore,
    build_interaction_store,
    partition_interactions,
)


def _yambda_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_user_id": ["u1", "u1", "u1", "u2", "u2"],
            "raw_item_id": ["a", "a", "b", "a", "c"],
            "timestamp": [10, 20, 30, 40, 50],
            "event_type": ["listen", "like", "listen", "listen", "dislike"],
            "is_organic": pd.array([True, None, False, True, False], dtype="boolean"),
            "played_ratio_pct": [90.0, np.nan, 20.0, 55.0, np.nan],
            "track_length_seconds": [180.0, np.nan, 200.0, 180.0, np.nan],
            "order_idx": [0, 1, 2, 3, 4],
        }
    )


class TestPartition:
    def test_none_criterion_keeps_everything_positive(self) -> None:
        frame = _yambda_frame()
        positives, auxiliary = partition_interactions(frame, None)
        assert positives is frame
        assert auxiliary.empty
        assert list(auxiliary.columns) == list(frame.columns)

    def test_criterion_strips_positives_and_keeps_payload_in_auxiliary(self) -> None:
        positives, auxiliary = partition_interactions(
            _yambda_frame(), "event_type == 'listen' and played_ratio_pct >= 50"
        )
        assert list(positives.columns) == ["raw_user_id", "raw_item_id", "timestamp", "order_idx"]
        assert len(positives) == 2  # (u1, a, 90) and (u2, a, 55)
        assert positives["raw_item_id"].tolist() == ["a", "a"]

        assert len(auxiliary) == 3  # like, low listen, dislike
        assert "played_ratio_pct" in auxiliary.columns
        assert sorted(auxiliary["event_type"]) == ["dislike", "like", "listen"]

    def test_na_criterion_rows_are_not_positive(self) -> None:
        # Likes have NaN played_ratio_pct; the comparison must not select them.
        positives, auxiliary = partition_interactions(_yambda_frame(), "played_ratio_pct >= 50")
        assert len(positives) == 2
        assert (auxiliary["event_type"] == "like").sum() == 1

    def test_bad_criterion_raises_with_columns(self) -> None:
        with pytest.raises(ValueError, match="raw_user_id"):
            partition_interactions(_yambda_frame(), "no_such_column >= 1")

    def test_zero_match_criterion_raises_with_null_diagnostics(self) -> None:
        # An all-null column (dataset export lacking the field) is the usual
        # cause of a criterion matching nothing; the error must say so.
        frame = _yambda_frame().assign(verified_purchase=pd.Series([None] * 5, dtype=object))
        with pytest.raises(ValueError, match="verified_purchase: 100% null"):
            partition_interactions(frame, "verified_purchase == True")


def _store() -> InteractionLogStore:
    _, auxiliary = partition_interactions(
        _yambda_frame(), "event_type == 'listen' and played_ratio_pct >= 50"
    )
    user2id = {"u1": 0, "u2": 1}
    item2id = {"a": 1, "b": 2}  # "c" not in catalog
    return build_interaction_store(
        auxiliary,
        user2id=user2id,
        item2id=item2id,
        num_users=2,
        item_vocab_size=3,
    )


class TestStore:
    def test_build_maps_ids_and_counts_drops(self) -> None:
        store = _store()
        assert len(store) == 2  # like(u1,a), low-listen(u1,b); dislike(u2,c) dropped
        assert store.stats.num_events_total == 3
        assert store.stats.num_events_dropped_unknown_item == 1
        assert store.counts_by_type() == {"like": 1, "listen": 1}

    def test_has_pair(self) -> None:
        store = _store()
        result = store.has_pair("like", [0, 0, 1], [1, 2, 1])
        assert result.tolist() == [True, False, False]

    def test_items_for_user(self) -> None:
        store = _store()
        assert store.items_for_user("like", 0).tolist() == [1]
        assert store.items_for_user("like", 1).tolist() == []
        assert store.items_for_user("listen", 0).tolist() == [2]

    def test_item_and_user_counts(self) -> None:
        store = _store()
        assert store.item_counts("like").tolist() == [0, 1, 0]
        assert store.user_counts("like").tolist() == [1, 0]

    def test_payload_for_pairs(self) -> None:
        store = _store()
        values = store.payload_for_pairs("listen", [0, 0], [2, 1], "played_ratio_pct")
        assert values[0] == 20.0
        assert np.isnan(values[1])

    def test_payload_agg_over_duplicate_pairs(self) -> None:
        events = pd.DataFrame(
            {
                "user_id": [0, 0, 0],
                "item_id": [1, 1, 1],
                "event_type": ["listen"] * 3,
                "timestamp": [1, 2, 3],
                "played_ratio_pct": [10.0, 90.0, 40.0],
            }
        )
        store = InteractionLogStore(events, num_users=1, item_vocab_size=2)
        assert store.payload_for_pairs("listen", [0], [1], "played_ratio_pct", agg="last")[0] == 40.0
        assert store.payload_for_pairs("listen", [0], [1], "played_ratio_pct", agg="max")[0] == 90.0
        assert store.payload_for_pairs("listen", [0], [1], "played_ratio_pct", agg="mean")[0] == pytest.approx(140 / 3)

    def test_filter_by_time(self) -> None:
        store = _store()
        early = store.filter_by_time(20)
        assert len(early) == 1  # only the like at ts=20
        assert early.counts_by_type() == {"like": 1}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        store = _store()
        store.save(tmp_path / "store")
        loaded = InteractionLogStore.load(tmp_path / "store")
        assert len(loaded) == len(store)
        assert loaded.item_vocab_size == store.item_vocab_size
        assert loaded.stats.num_events_dropped_unknown_item == 1
        assert loaded.has_pair("like", [0], [1]).tolist() == [True]


def _yambda_rows(rng) -> pd.DataFrame:
    rows = []
    for u in range(12):
        n = int(rng.integers(10, 40))
        for i in range(n):
            item = int(rng.integers(0, 25))
            rows.append(
                {
                    "uid": u,
                    "item_id": item,
                    "timestamp": 1_000 + i,
                    "event_type": "listen",
                    "is_organic": int(rng.integers(0, 2)),
                    "played_ratio_pct": int(rng.integers(0, 101)),
                    "track_length_seconds": 200,
                }
            )
            if rng.random() < 0.1:
                rows.append(
                    {
                        "uid": u,
                        "item_id": item,
                        "timestamp": 1_000 + i,
                        "event_type": "like",
                        "is_organic": int(rng.integers(0, 2)),
                        "played_ratio_pct": None,
                        "track_length_seconds": None,
                    }
                )
    return pd.DataFrame(rows)


def test_yambda_datamodule_partitions_into_store(tmp_path) -> None:
    from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

    rng = np.random.default_rng(0)
    interactions_path = tmp_path / "yambda.parquet"
    _yambda_rows(rng).to_parquet(interactions_path)

    cfg = YambdaDataConfig(
        interactions_path=interactions_path,
        positive_criterion="event_type == 'listen' and played_ratio_pct >= 50",
        min_user_interactions=3,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
    )
    dm = YambdaDataModule(cfg)
    dm.setup()

    # Positives are minimal: no payload columns reach the training frame.
    for column in ("event_type", "is_organic", "played_ratio_pct", "track_length_seconds"):
        assert column not in dm.interactions.columns
    assert {"raw_user_id", "raw_item_id", "timestamp"} <= set(dm.interactions.columns)

    store = dm.interaction_store
    assert store is not None and len(store) > 0
    assert set(store.event_types()) <= {"like", "listen"}
    assert "played_ratio_pct" in store.payload_columns
    # Store is keyed by the final catalog.
    assert store.events["item_id"].max() < dm.item_vocab_size
    assert store.events["user_id"].max() < dm.num_users
    # Low-played listens land in the store, not in training positives.
    listen_pct = store.payload_for_pairs(
        "listen",
        store.events["user_id"].to_numpy(),
        store.events["item_id"].to_numpy(),
        "played_ratio_pct",
        agg="max",
    )
    assert np.nanmax(listen_pct) < 50

    summary = dm.summary()
    assert summary["positive_criterion"] == cfg.positive_criterion
    assert summary["interaction_store"]["num_events"] == len(store)

    batch = next(iter(dm.train_dataloader()))
    assert "history_item_ids" in batch


def test_amazon_datamodule_partitions_into_store(tmp_path) -> None:
    from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule

    rng = np.random.default_rng(1)
    rows = []
    for u in range(10):
        for i in range(int(rng.integers(8, 20))):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "asin": f"i{int(rng.integers(0, 20))}",
                    "timestamp": 1_000 + i,
                    "rating": float(rng.integers(1, 6)),
                    "verified_purchase": bool(rng.integers(0, 2)),
                }
            )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(rows).to_parquet(reviews_path)

    cfg = AmazonReviewDataConfig(
        reviews_path=reviews_path,
        item_key="asin",
        positive_criterion="rating >= 4.0",
        min_user_interactions=3,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
    )
    dm = AmazonReviewDataModule(cfg)
    dm.setup()

    assert "rating" not in dm.interactions.columns
    store = dm.interaction_store
    assert store is not None and len(store) > 0
    assert store.event_types() == ["review"]
    # Every stored review is sub-threshold.
    ratings = store.payload_for_pairs(
        "review",
        store.events["user_id"].to_numpy(),
        store.events["item_id"].to_numpy(),
        "rating",
        agg="max",
    )
    assert np.nanmax(ratings) < 4.0

    batch = next(iter(dm.train_dataloader()))
    assert "history_item_ids" in batch


def _dm_with_exclusive_negative_item(tmp_path, *, embeddings_for_all: bool = True):
    """Datamodule where item 'i100' is touched only in sub-threshold listens."""
    from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

    rows = []
    for u in range(8):
        for i in range(6):
            rows.append(
                {
                    "uid": u,
                    "item_id": f"i{i}",
                    "timestamp": 1_000 + i,
                    "event_type": "listen",
                    "played_ratio_pct": 95,
                }
            )
        rows.append(
            {
                "uid": u,
                "item_id": "i100",
                "timestamp": 2_000,
                "event_type": "listen",
                "played_ratio_pct": 10,
            }
        )
    interactions_path = tmp_path / "yambda.parquet"
    pd.DataFrame(rows).to_parquet(interactions_path)

    cfg = YambdaDataConfig(
        interactions_path=interactions_path,
        positive_criterion="event_type == 'listen' and played_ratio_pct >= 50",
        min_user_interactions=3,
        min_item_interactions=0,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
    )
    dm = YambdaDataModule(cfg)
    dm.setup()
    return dm


class TestItemScopeForTokenizer:
    def test_store_keeps_out_of_catalog_item_ids(self, tmp_path) -> None:
        dm = _dm_with_exclusive_negative_item(tmp_path)
        store = dm.interaction_store
        assert store.out_of_catalog_item_ids == {"i100"}
        assert store.stats.num_events_dropped_unknown_item == 8
        assert store.to_summary_dict()["num_out_of_catalog_items"] == 1

        store.save(tmp_path / "store")
        from flowrec.data.interaction_store import InteractionLogStore

        loaded = InteractionLogStore.load(tmp_path / "store")
        assert loaded.out_of_catalog_item_ids == {"i100"}
        assert loaded.filter_by_time(10_000).out_of_catalog_item_ids == {"i100"}

    def test_item_ids_in_data_scopes(self, tmp_path) -> None:
        dm = _dm_with_exclusive_negative_item(tmp_path)
        positives = dm.item_ids_in_data()
        assert "i100" not in positives
        both = dm.item_ids_in_data(include_auxiliary=True)
        assert both == positives | {"i100"}

    def test_embedding_store_subset_to_datamodule(self, tmp_path) -> None:
        import torch
        from flowrec.item_representation.base import ItemEmbeddingStore

        dm = _dm_with_exclusive_negative_item(tmp_path)
        catalog_and_more = [f"i{i}" for i in range(6)] + ["i100", "i900", "i901"]
        emb_store = ItemEmbeddingStore(
            item_ids=catalog_and_more,
            embeddings=torch.randn(len(catalog_and_more), 8),
        )

        with_aux = emb_store.subset_to_datamodule(dm, include_auxiliary=True)
        assert set(with_aux.item_ids) == set(dm.item_ids_in_data()) | {"i100"}

        # Default scope is the catalog: exclusively-negative items are removed
        # in preprocessing and never sampled, so the tokenizer skips them.
        positives_only = emb_store.subset_to_datamodule(dm)
        assert "i100" not in positives_only.item_ids
        assert "i900" not in with_aux.item_ids

    def test_load_yambda_embeddings_for_datamodule(self, tmp_path) -> None:
        from flowrec.item_representation.yambda import (
            load_yambda_audio_embedding_store_for_datamodule,
        )

        dm = _dm_with_exclusive_negative_item(tmp_path)
        all_items = [f"i{i}" for i in range(6)] + ["i100", "i900"]
        emb = pd.DataFrame(
            {
                "item_id": all_items,
                "normalized_embed": [np.random.rand(4).tolist() for _ in all_items],
            }
        )
        emb_path = tmp_path / "embeddings.parquet"
        emb.to_parquet(emb_path)

        store = load_yambda_audio_embedding_store_for_datamodule(
            dm, embeddings_path=emb_path, include_auxiliary=True
        )
        assert set(store.item_ids) == set(dm.item_ids_in_data(include_auxiliary=True))
        assert "i900" not in store.item_ids

        positives_only = load_yambda_audio_embedding_store_for_datamodule(
            dm, embeddings_path=emb_path
        )
        assert set(positives_only.item_ids) == set(dm.item_ids_in_data())

        with pytest.raises(ValueError, match="embeddings_path"):
            load_yambda_audio_embedding_store_for_datamodule(dm)
