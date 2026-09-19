"""VK-LSVD datamodule: real upstream schema, provided weekly split, both
training layouts, positive/auxiliary partition, and the catalog-restricted
embedding loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flowrec.data.popularity import training_item_counts_by_raw_id
from flowrec.data.vk_lsvd import (
    VK_FEEDBACK_FLAGS,
    VKLSVDDataConfig,
    VKLSVDDataModule,
    _read_vk_tabular_frame,
    build_provided_sequence_splits,
    load_vk_lsvd_embedding_item_ids,
    normalize_vk_lsvd_frame,
    vk_lsvd_local_input_files,
)
from flowrec.item_representation.vk_lsvd import (
    VKLSVDItemEmbeddingBuilder,
    VKLSVDItemEmbeddingConfig,
    load_vk_lsvd_item_embedding_store_for_datamodule,
)
from scripts import sasrec_config_runner as runner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vk_lsvd_smoke_assets import (  # noqa: E402
    TEST_WEEK,
    TRAIN_WEEKS,
    VALIDATION_WEEK,
    build_smoke_subsample,
)


def _upstream_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": np.array([7, 7, 8], dtype=np.uint32),
            "item_id": np.array([100, 101, 100], dtype=np.uint32),
            "place": np.array([0, 1, 0], dtype=np.uint8),
            "platform": np.array([2, 2, 1], dtype=np.uint8),
            "agent": np.array([0, 0, 1], dtype=np.uint8),
            "timespent": np.array([0, 30, 255], dtype=np.uint8),
            "like": [False, True, False],
            "dislike": [False, False, True],
            "share": [False, False, False],
            "bookmark": [False, True, False],
            "click_on_author": [False, False, False],
            "open_comments": [False, False, True],
        }
    )


def _config(subsample: Path, **overrides) -> VKLSVDDataConfig:
    values = dict(
        interactions_root=subsample,
        split_strategy="provided",
        min_user_interactions=2,
        min_item_interactions=1,
        max_history_len=10,
        batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
        filter_cold_items_after_split=False,
    )
    values.update(overrides)
    return VKLSVDDataConfig(**values)


@pytest.fixture(scope="module")
def smoke_assets(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("vk_lsvd")
    return build_smoke_subsample(root, n_users=120, n_items=80, events_per_user=(3, 9))


# ---------------------------------------------------------------------------
# Normalization of the upstream schema
# ---------------------------------------------------------------------------


def test_normalize_upstream_schema_without_timestamp_column() -> None:
    cfg = VKLSVDDataConfig(week_timestamp_stride=1_000)
    out = normalize_vk_lsvd_frame(_upstream_frame(), split="validation", week=25, cfg=cfg)

    assert list(out.columns) == [
        "raw_user_id",
        "raw_item_id",
        "timestamp",
        "rating",
        "order_idx",
        "flowrec_split",
        "flowrec_week",
        "timespent",
        *VK_FEEDBACK_FLAGS,
    ]
    assert out["raw_user_id"].tolist() == ["7", "7", "8"]
    assert out["raw_item_id"].tolist() == ["100", "101", "100"]
    # Synthetic timestamps keep the file order and never interleave weeks.
    assert out["timestamp"].tolist() == [25_000, 25_001, 25_002]
    assert out["flowrec_split"].tolist() == ["val"] * 3
    assert out["flowrec_week"].tolist() == [25] * 3
    assert out["timespent"].tolist() == [0.0, 30.0, 255.0]
    assert out["like"].dtype == bool and out["like"].tolist() == [False, True, False]
    assert out["dislike"].tolist() == [False, False, True]
    # weighted_feedback: timespent/255 + like + bookmark ...; zero -> default.
    assert out["rating"].iloc[0] == pytest.approx(1.0)
    assert out["rating"].iloc[1] == pytest.approx(30 / 255 + 1.0 + 1.0)
    assert out["rating"].iloc[2] == pytest.approx(1.0 - 1.0 + 0.25)


def test_rating_modes_and_scope_filters() -> None:
    frame = _upstream_frame()
    binary = normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig(rating_mode="binary"))
    assert binary["rating"].tolist() == [1.0, 1.0, 1.0]
    timespent = normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig(rating_mode="timespent"))
    assert timespent["rating"].tolist() == [0.0, 30.0, 255.0]

    watched = normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig(min_timespent=1))
    assert watched["raw_item_id"].tolist() == ["101", "100"]
    positive = normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig(require_positive_feedback=True))
    assert len(positive) == 2

    with pytest.raises(ValueError, match="rating_mode"):
        normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig(rating_mode="nope"))  # type: ignore[arg-type]


def test_bool_flags_accept_numeric_and_string_exports() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [1, 2, 3, 4],
            "timespent": [1, 1, 1, 1],
            "like": [1, 0, 1.0, 0.0],
            "dislike": ["true", "false", "1", None],
            "share": pd.array([True, None, False, True], dtype="boolean"),
        }
    )
    out = normalize_vk_lsvd_frame(frame, cfg=VKLSVDDataConfig())
    assert out["like"].tolist() == [True, False, True, False]
    assert out["dislike"].tolist() == [True, False, True, False]
    assert out["share"].tolist() == [True, False, False, True]
    assert out["bookmark"].tolist() == [False] * 4


def test_explicit_timestamp_column_wins_over_synthetic() -> None:
    frame = _upstream_frame()
    frame["timestamp"] = [10, 20, 30]
    out = normalize_vk_lsvd_frame(frame, week=3, cfg=VKLSVDDataConfig())
    assert out["timestamp"].tolist() == [10, 20, 30]


def test_parquet_reader_prunes_columns_and_reads_only_the_head(smoke_assets) -> None:
    subsample, _ = smoke_assets
    path = subsample / "train" / "week_00.parquet"
    head = _read_vk_tabular_frame(path, max_rows=5, columns=("user_id", "item_id", "timespent"))
    assert len(head) == 5
    assert list(head.columns) == ["user_id", "item_id", "timespent"]
    full = _read_vk_tabular_frame(path)
    assert "place" in full.columns and len(full) > 5
    assert _read_vk_tabular_frame(path, max_rows=0).empty


# ---------------------------------------------------------------------------
# Provided weekly split
# ---------------------------------------------------------------------------


def test_local_input_files_and_prepare_data_guard(smoke_assets, tmp_path: Path) -> None:
    subsample, _ = smoke_assets
    files = vk_lsvd_local_input_files(_config(subsample))
    assert [path.name for path in files] == [
        *(f"week_{week:02}.parquet" for week in TRAIN_WEEKS),
        f"week_{VALIDATION_WEEK:02}.parquet",
        f"week_{TEST_WEEK:02}.parquet",
    ]

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="interactions_root"):
        VKLSVDDataModule(_config(empty_root)).setup()
    with pytest.raises(ValueError, match="No VK-LSVD interaction files were configured"):
        VKLSVDDataModule(_config(None)).setup()

    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        VKLSVDDataModule(_config(None, train_paths=[missing])).setup()


def test_provided_split_prefix_target_examples_follow_the_week_labels(smoke_assets) -> None:
    subsample, _ = smoke_assets
    data = VKLSVDDataModule(_config(subsample, global_eval_target_mode="all"))
    data.setup()

    interactions = data.interactions
    week_of = {
        (u, int(p)): int(w)
        for u, p, w in zip(interactions["user_id"], interactions["position"], interactions["flowrec_week"])
    }
    train = data.train_dataset.examples
    val = data.val_dataset.examples
    test = data.test_dataset.examples
    assert len(train) and len(val) and len(test)
    assert all(
        week_of[(int(u), int(p))] in TRAIN_WEEKS
        for u, p in zip(train["user_id"], train["target_position"])
    )
    assert all(
        week_of[(int(u), int(p))] == VALIDATION_WEEK
        for u, p in zip(val["user_id"], val["target_position"])
    )
    assert all(
        week_of[(int(u), int(p))] == TEST_WEEK
        for u, p in zip(test["user_id"], test["target_position"])
    )
    # Every validation-week event with history is a target under mode "all".
    val_rows = interactions.loc[interactions["flowrec_split"].eq("val") & interactions["position"].ge(1)]
    assert len(val) == len(val_rows)
    # Rolling history: a validation target sees the train-week events.
    assert (val["history_item_ids"].str.len() > 0).all()

    stride = data.cfg.week_timestamp_stride
    assert data.global_val_cutoff == VALIDATION_WEEK * stride
    assert data.global_test_cutoff == TEST_WEEK * stride
    summary = data.summary()
    assert summary["dataset"] == "vk_lsvd"
    assert summary["split_strategy"] == "provided"
    assert summary["train_example_layout"] == "prefix_target"
    assert summary["interaction_store"]["num_events"] == 0

    # Train-count popularity for the provided split, both entry points.
    counts = training_item_counts_by_raw_id(
        interactions,
        split_strategy="provided",
        global_val_cutoff=None,
        item2id=data.item2id,
    )
    expected = interactions.loc[interactions["flowrec_split"].eq("train"), "raw_item_id"].value_counts()
    assert counts == {str(k): int(v) for k, v in expected.items()}
    assert data.train_item_counts_by_raw_id == counts
    popularity = runner.training_item_popularity(data)
    assert popularity.shape == (data.item_vocab_size,)
    assert float(popularity.sum()) == float(expected.sum())
    assert popularity[0] == 0.0


def test_last_per_user_and_window_start_eval_options(smoke_assets) -> None:
    subsample, _ = smoke_assets
    data = VKLSVDDataModule(
        _config(subsample, global_eval_target_mode="last_per_user", global_history_policy="window_start")
    )
    data.setup()
    val = data.val_dataset.examples
    assert val["user_id"].is_unique
    interactions = data.interactions
    train_len = interactions.loc[interactions["flowrec_split"].eq("train")].groupby("user_id").size()
    # window_start: the history ends where the validation week begins.
    for user, history in zip(val["user_id"], val["history_item_ids"]):
        n_train = int(train_len.get(int(user), 0))
        assert len(history) == min(n_train, data.cfg.max_history_len)


def test_train_target_policies_apply_to_the_provided_split(smoke_assets) -> None:
    subsample, _ = smoke_assets
    all_targets = VKLSVDDataModule(_config(subsample))
    all_targets.setup()
    last = VKLSVDDataModule(_config(subsample, train_target_policy="last"))
    last.setup()
    last_n = VKLSVDDataModule(_config(subsample, train_target_policy="last_n", train_last_n=2))
    last_n.setup()
    assert last.train_dataset.examples["user_id"].is_unique
    assert len(last.train_dataset.examples) < len(last_n.train_dataset.examples) < len(all_targets.train_dataset.examples)
    # Same catalog and eval splits regardless of the train policy.
    assert last.item2id == all_targets.item2id
    assert len(last.val_dataset) == len(all_targets.val_dataset)


def test_split_labels_contradicting_time_order_raise() -> None:
    frame = pd.DataFrame(
        {
            "raw_user_id": ["u", "u", "u"],
            "raw_item_id": ["1", "2", "3"],
            "timestamp": [1, 2, 3],
            "rating": [1.0, 1.0, 1.0],
            "order_idx": [0, 1, 2],
            "flowrec_split": ["train", "val", "train"],
            "flowrec_week": [0, 25, 0],
            "user_id": [0, 0, 0],
            "item_id": [1, 2, 3],
            "position": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="contradict"):
        build_provided_sequence_splits(frame, cfg=VKLSVDDataConfig())


# ---------------------------------------------------------------------------
# Dense-causal layouts (SASRec left padding, GPT-2 right padding + tokens)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padding", ["left", "right"])
def test_dense_causal_train_region_is_the_train_weeks(smoke_assets, padding: str) -> None:
    subsample, _ = smoke_assets
    data = VKLSVDDataModule(
        _config(subsample, train_example_layout="dense_causal", history_padding=padding, train_window_stride=5)
    )
    data.setup()

    train = data.train_dataset.examples
    assert "label_item_ids" in train.columns and len(train)
    interactions = data.interactions
    train_end = interactions.loc[interactions["flowrec_split"].eq("train")].groupby("user_id").size()
    for user, position in zip(train["user_id"], train["target_position"]):
        assert int(position) < int(train_end.get(int(user), 0))

    batch = next(iter(data.train_dataloader()))
    for key in ("input_item_ids", "attention_mask", "label_item_ids", "label_mask"):
        assert key in batch
    assert batch["input_item_ids"].shape[1] == data.cfg.max_history_len
    assert int(batch["label_mask"].sum()) > 0
    assert "input_token_ids" not in batch

    val_batch = next(iter(data.val_dataloader()))
    assert "history_item_ids" in val_batch and "input_item_ids" not in val_batch
    assert data.summary()["train_example_layout"] == "dense_causal"


def test_dense_causal_collator_emits_semantic_token_stream(smoke_assets) -> None:
    subsample, _ = smoke_assets
    data = VKLSVDDataModule(
        _config(
            subsample,
            train_example_layout="dense_causal",
            history_padding="right",
            user_hash_mod=1,
        )
    )
    data.setup()
    rng = np.random.default_rng(0)
    data.add_semantic_ids(
        {item: [int(x) for x in rng.integers(0, 8, size=3)] for item in data.item_ids_in_data()}
    )

    batch = next(iter(data.train_dataloader()))
    for key in ("input_token_ids", "token_attention_mask", "label_token_ids", "token_label_mask", "hashed_user_ids"):
        assert key in batch, key
    assert batch["input_token_ids"].shape[1] == batch["label_token_ids"].shape[1]

    val_batch = next(iter(data.val_dataloader()))
    assert val_batch["history_token_ids"].shape[-1] == 3
    assert "target_token_ids" in val_batch


def test_tensorized_eval_batches_match_the_legacy_collator(smoke_assets) -> None:
    subsample, _ = smoke_assets
    fast = VKLSVDDataModule(_config(subsample, tensorized_batches=True))
    fast.setup()
    slow = VKLSVDDataModule(_config(subsample, tensorized_batches=False))
    slow.setup()
    fast_batch = next(iter(fast.val_dataloader()))
    slow_batch = next(iter(slow.val_dataloader()))
    for key in ("history_item_ids", "history_mask", "target_item_ids", "user_ids"):
        assert torch.equal(fast_batch[key], slow_batch[key]), key


# ---------------------------------------------------------------------------
# Positive/auxiliary partition and embedding coverage
# ---------------------------------------------------------------------------


def test_positive_criterion_partitions_into_catalog_and_payload_store(smoke_assets) -> None:
    subsample, _ = smoke_assets
    criterion = "timespent >= 1 and not dislike"
    data = VKLSVDDataModule(_config(subsample, positive_criterion=criterion))
    data.setup()

    store = data.interaction_store
    assert store is not None and len(store) > 0
    assert store.event_types() == ["interaction"]
    for column in ("timespent", "dislike", "like", "flowrec_split"):
        assert column in store.events.columns
    assert not bool(((store.events["timespent"] >= 1) & ~store.events["dislike"]).any())

    # Positives alone define the catalog; auxiliary-only items are exposed.
    positives = data.interactions
    assert (positives["flowrec_split"].isin(["train", "val", "test"])).all()
    in_data = data.item_ids_in_data()
    with_aux = data.item_ids_in_data(include_auxiliary=True)
    assert in_data <= with_aux
    assert data.summary()["positive_criterion"] == criterion
    assert data.summary()["interaction_store"]["num_events"] == len(store)


def test_embedding_coverage_filter_and_catalog_restricted_store(smoke_assets, tmp_path: Path) -> None:
    subsample, npz_path = smoke_assets
    ids = load_vk_lsvd_embedding_item_ids(npz_path)
    assert isinstance(ids, np.ndarray) and ids.dtype == np.int64 and len(ids) == 80

    # Drop half of the catalog from the embedding file.
    partial = tmp_path / "partial.npz"
    with np.load(npz_path) as data_npz:
        keep = data_npz["item_id"] % 2 == 0
        np.savez(partial, item_id=data_npz["item_id"][keep], embedding=data_npz["embedding"][keep])

    full = VKLSVDDataModule(_config(subsample))
    full.setup()
    filtered = VKLSVDDataModule(_config(subsample, embeddings_path=partial))
    filtered.setup()
    assert all(int(item) % 2 == 0 for item in filtered.item_ids_in_data())
    assert filtered.num_items < full.num_items
    assert filtered.summary()["num_embedding_items"] == 40

    store = load_vk_lsvd_item_embedding_store_for_datamodule(filtered, embedding_dim=16)
    assert set(store.item_ids) == filtered.item_ids_in_data()
    assert store.embeddings.shape == (filtered.num_items, 16)
    assert store.embeddings.dtype == torch.float32
    assert store.metadata["num_source_items"] == 40

    # Semantic-ID + embedding filters intersect.
    semantic = {item: [1, 2, 3] for item in list(full.item_ids_in_data())[:20]}
    import json

    sid_path = tmp_path / "semantic_ids.json"
    sid_path.write_text(json.dumps(semantic))
    both = VKLSVDDataModule(_config(subsample, embeddings_path=partial, semantic_id_path=sid_path))
    both.setup()
    assert both.item_ids_in_data() <= (set(semantic) & filtered.item_ids_in_data())


def test_npz_builder_filters_on_arrays_and_slices_components(smoke_assets) -> None:
    _, npz_path = smoke_assets
    store = VKLSVDItemEmbeddingBuilder(
        VKLSVDItemEmbeddingConfig(embeddings_path=npz_path, embedding_dim=8, allowed_items={"3", "5", "999"})
    ).build()
    assert store.item_ids == ["3", "5"]
    assert store.embeddings.shape == (2, 8)
    assert store.metadata["num_skipped_allowed"] == 78

    by_array = VKLSVDItemEmbeddingBuilder(
        VKLSVDItemEmbeddingConfig(embeddings_path=npz_path, allowed_items=np.array([5, 3]), max_items=1)
    ).build()
    assert by_array.item_ids == ["3"] and by_array.embeddings.shape == (1, 64)

    with pytest.raises(ValueError, match="No VK-LSVD item embeddings"):
        VKLSVDItemEmbeddingBuilder(
            VKLSVDItemEmbeddingConfig(embeddings_path=npz_path, allowed_items={"nope"})
        ).build()
