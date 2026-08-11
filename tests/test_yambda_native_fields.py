"""Yambda canonical frame keeps native multi-event fields and supports
embedding-coverage item filtering."""

import numpy as np
import pandas as pd
import torch

from flowrec.data.yambda import (
    YambdaDataConfig,
    YambdaDataModule,
    load_embedding_item_ids,
    normalize_yambda_flat_frame,
    normalize_yambda_sequential_frame,
)


def _flat_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "item_id": [10, 11, 10, 12],
            "timestamp": [100, 200, 300, 400],
            "event_type": ["listen", "like", "listen", None],
            "is_organic": [1, 0, None, 1],
            "played_ratio_pct": [80, None, 40, None],
            "track_length_seconds": [180, 180, 240, None],
        }
    )


def test_flat_frame_keeps_native_event_fields() -> None:
    out = normalize_yambda_flat_frame(_flat_frame())

    assert list(out.columns) == [
        "raw_user_id",
        "raw_item_id",
        "timestamp",
        "event_type",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
        "order_idx",
    ]
    assert "rating" not in out.columns and "verified_purchase" not in out.columns

    assert out["event_type"].tolist() == ["listen", "like", "listen", "listen"]
    assert bool(out["is_organic"].iloc[0]) is True
    assert bool(out["is_organic"].iloc[1]) is False
    assert pd.isna(out["is_organic"].iloc[2])
    assert out["played_ratio_pct"].iloc[0] == 80.0
    assert np.isnan(out["played_ratio_pct"].iloc[1])
    assert out["track_length_seconds"].iloc[0] == 180.0
    assert np.isnan(out["track_length_seconds"].iloc[3])


def test_flat_frame_filters_still_apply() -> None:
    frame = _flat_frame()
    organic = normalize_yambda_flat_frame(frame, organic_filter="organic")
    assert organic["raw_item_id"].tolist() == ["10", "12"]

    recommendation = normalize_yambda_flat_frame(frame, organic_filter="recommendation")
    assert recommendation["raw_item_id"].tolist() == ["11"]

    # listen@40 is dropped; the row with missing event_type defaults to
    # "listen" with unknown pct and is dropped as well; the like passes.
    pct = normalize_yambda_flat_frame(frame, min_played_ratio_pct=50)
    assert pct["raw_item_id"].tolist() == ["10", "11"]


def test_sequential_frame_keeps_native_event_fields() -> None:
    seq = pd.DataFrame(
        [
            {
                "uid": 7,
                "item_ids": [1, 2],
                "timestamps": [10, 20],
                "event_types": ["listen", "like"],
                "is_organic": [1, 0],
                "played_ratio_pcts": [90, None],
            }
        ]
    )
    out = normalize_yambda_sequential_frame(seq)
    assert out["event_type"].tolist() == ["listen", "like"]
    assert out["played_ratio_pct"].iloc[0] == 90.0
    assert bool(out["is_organic"].iloc[0]) is True
    assert bool(out["is_organic"].iloc[1]) is False


def test_load_embedding_item_ids_parquet_and_pt(tmp_path) -> None:
    emb = pd.DataFrame(
        {
            "item_id": [10, 11],
            "embed": [np.random.rand(4).tolist(), np.random.rand(4).tolist()],
        }
    )
    emb_path = tmp_path / "embeddings.parquet"
    emb.to_parquet(emb_path)
    assert load_embedding_item_ids(emb_path) == {"10", "11"}

    pt_path = tmp_path / "store.pt"
    torch.save({"item_ids": ["10", "12"], "embeddings": torch.zeros(2, 4)}, pt_path)
    assert load_embedding_item_ids(pt_path) == {"10", "12"}


def test_datamodule_drops_items_missing_embedding(tmp_path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(12):
        n = int(rng.integers(8, 40))
        for i in range(n):
            rows.append(
                {
                    "uid": u,
                    "item_id": int(rng.integers(0, 30)),
                    "timestamp": 1_000 + i,
                    "event_type": "listen",
                    "is_organic": int(rng.integers(0, 2)),
                    "played_ratio_pct": int(rng.integers(0, 101)),
                    "track_length_seconds": 200,
                }
            )
    interactions_path = tmp_path / "yambda.parquet"
    pd.DataFrame(rows).to_parquet(interactions_path)

    # Items 20..29 have no embedding and must be dropped.
    emb = pd.DataFrame(
        {
            "item_id": list(range(0, 20)),
            "embed": [np.random.rand(4).tolist() for _ in range(20)],
        }
    )
    embeddings_path = tmp_path / "embeddings.parquet"
    emb.to_parquet(embeddings_path)

    cfg = YambdaDataConfig(
        interactions_path=interactions_path,
        embeddings_path=embeddings_path,
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

    assert all(int(raw_item) < 20 for raw_item in dm.item_ids_in_data())
    assert dm.summary()["num_embedding_items"] == 20
    for column in ("event_type", "is_organic", "played_ratio_pct", "track_length_seconds"):
        assert column in dm.interactions.columns

    batch = next(iter(dm.train_dataloader()))
    assert "history_item_ids" in batch


def test_parquet_window_pushdown_matches_full_load(tmp_path) -> None:
    """The arrow-level window pushdown must yield the exact same canonical
    frame as loading everything into pandas and filtering there."""
    from flowrec.data.yambda import load_yambda_interactions_frame

    rng = np.random.default_rng(3)
    day = 86_400
    rows = []
    for u in range(6):
        for d in range(0, 200, 2):  # events spanning ~200 days
            rows.append(
                {
                    "uid": u,
                    "item_id": int(rng.integers(0, 30)),
                    "timestamp": d * day + int(rng.integers(0, day)),
                    "event_type": "listen",
                    "played_ratio_pct": int(rng.integers(0, 101)),
                }
            )
    path = tmp_path / "yambda.parquet"
    pd.DataFrame(rows).to_parquet(path)

    pushed = load_yambda_interactions_frame(path, hf_recent_months=2)
    # max_rows disables pushdown; with max_rows == len the semantics are
    # identical to the historical full-load path.
    full = load_yambda_interactions_frame(path, hf_recent_months=2, max_rows=len(rows))

    assert len(pushed) == len(full) > 0
    assert len(pushed) < len(rows)  # the window actually trimmed something
    pd.testing.assert_frame_equal(
        pushed.sort_values("order_idx").reset_index(drop=True).drop(columns=["order_idx"]),
        full.sort_values("order_idx").reset_index(drop=True).drop(columns=["order_idx"]),
    )
