"""Tests for the unified exp1 SASRec runner (run_sasrec_all.py)."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from flowrec.models.base import RecModelOutput
from scripts import sasrec_config_runner as common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "exp1-standard-params" / "run_sasrec_all.py"
)
SPEC = importlib.util.spec_from_file_location("run_sasrec_all", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_all = importlib.util.module_from_spec(SPEC)
# The dataclasses in the script resolve their module through sys.modules.
sys.modules[SPEC.name] = run_all
SPEC.loader.exec_module(run_all)


def _plan(group_key: str, requested: list[str] | None = None):
    plans = run_all.build_group_plans(run_all.DEFAULT_CONFIG_DIR, requested)
    return next(plan for plan in plans if plan.group.key == group_key)


def test_default_selection_covers_all_five_exp1_datasets() -> None:
    plans = run_all.build_group_plans(run_all.DEFAULT_CONFIG_DIR, None)

    assert [plan.group.key for plan in plans] == ["amazon", "amazon_2023", "yambda"]
    assert [name for plan in plans for name in plan.dataset_names] == [
        "beauty",
        "instruments",
        "beauty_2023",
        "yambda_50m_listens",
        "yambda_500m_listens",
    ]
    assert run_all.combined_metrics_summary_path(plans) == (
        PROJECT_ROOT
        / "artifacts"
        / "exp1-standard_params"
        / run_all.COMBINED_METRICS_SUMMARY_NAME
    )


def test_explicit_dataset_selection_spans_groups() -> None:
    plans = run_all.build_group_plans(
        run_all.DEFAULT_CONFIG_DIR, ["beauty", "yambda_500m_listens"]
    )
    assert [(plan.group.key, plan.dataset_names) for plan in plans] == [
        ("amazon", ["beauty"]),
        ("yambda", ["yambda_500m_listens"]),
    ]

    with pytest.raises(KeyError, match="unknown_dataset"):
        run_all.build_group_plans(run_all.DEFAULT_CONFIG_DIR, ["unknown_dataset"])


def test_small_amazon_group_resolves_dense_left_padded_three_seed_config() -> None:
    plan = _plan("amazon")

    assert plan.dataset_names == ["beauty", "instruments"]
    assert common.resolve_seeds(
        argparse.Namespace(seeds=None), plan.trainer_config
    ) == [42, 43, 44]

    for dataset_name in plan.dataset_names:
        cfg = run_all.build_amazon_data_config(
            dataset_name, plan.dataset_config, seed=43
        )
        assert cfg.train_example_layout == "dense_causal"
        assert cfg.history_padding == "left"
        assert cfg.seed == 43
        assert cfg.semantic_id_path is None

    checkpoint_root = common.configured_path(
        plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
    )
    assert checkpoint_root == (
        PROJECT_ROOT / "checkpoints/exp1-standard_params/sasrec"
    )


def test_amazon_2023_group_resolves_dense_left_padded_three_seed_config() -> None:
    plan = _plan("amazon_2023")

    assert plan.dataset_names == ["beauty_2023"]
    assert common.resolve_seeds(
        argparse.Namespace(seeds=None), plan.trainer_config
    ) == [42, 43, 44]

    cfg = run_all.build_amazon_data_config("beauty_2023", plan.dataset_config, seed=43)
    assert cfg.reviews_path == (
        PROJECT_ROOT / "data/raw/dataframe_format/reviews_beauty_2023.parquet"
    )
    assert cfg.metadata_path == (
        PROJECT_ROOT / "data/raw/dataframe_format/meta_beauty_2023.parquet"
    )
    assert cfg.item_key == "parent_asin"
    assert cfg.metadata_item_key == "parent_asin"
    assert cfg.train_example_layout == "dense_causal"
    assert cfg.history_padding == "left"
    assert cfg.seed == 43

    checkpoint_root = common.configured_path(
        plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
    )
    assert checkpoint_root == (
        PROJECT_ROOT / "checkpoints/exp1-standard_params/sasrec/amazon-2023"
    )
    assert plan.trainer_config["metrics_summary_name"] == (
        "sasrec_amazon_2023_metrics_summary.csv"
    )


def test_yambda_group_resolves_requested_50m_and_500m_parameters() -> None:
    plan = _plan("yambda")

    assert plan.dataset_names == ["yambda_50m_listens", "yambda_500m_listens"]
    assert common.resolve_seeds(
        argparse.Namespace(seeds=None), plan.trainer_config
    ) == [42, 43, 44]

    for name, size in (
        ("yambda_50m_listens", "50m"),
        ("yambda_500m_listens", "500m"),
    ):
        cfg = run_all.build_yambda_data_config(name, plan.dataset_config, seed=44)
        assert cfg.use_hf is False
        assert cfg.dataset_type == "flat"
        assert cfg.dataset_size == size
        assert cfg.event_file == "multi_event"
        assert cfg.interactions_path == (
            PROJECT_ROOT / f"data/raw/yambda/flat/{size}/multi_event.parquet"
        )
        assert cfg.embeddings_path == PROJECT_ROOT / "data/raw/yambda/embeddings.parquet"
        assert cfg.drop_items_missing_embedding is True
        assert cfg.positive_criterion == (
            "event_type == 'listen' and played_ratio_pct >= 50"
        )
        assert cfg.min_played_ratio_pct is None
        assert cfg.split_strategy == "leave_one_out"
        assert cfg.train_example_layout == "dense_causal"
        assert cfg.train_window_len is None
        assert cfg.train_window_stride is None
        assert cfg.deduplicate_consecutive is True
        assert cfg.filter_cold_items_after_split is True
        assert cfg.min_user_interactions == 5
        assert cfg.min_item_interactions == 5
        assert cfg.max_history_len == 50
        assert cfg.history_padding == "left"
        assert cfg.seed == 44

    checkpoint_root = common.configured_path(
        plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
    )
    assert checkpoint_root == (
        PROJECT_ROOT / "checkpoints/exp1-standard_params/sasrec/yambda"
    )
    summary_template = plan.trainer_config["metrics_summary_name"]
    assert summary_template == "sasrec_{dataset}_metrics_summary.csv"
    artifacts_root = common.configured_path(
        plan.trainer_config, "artifacts_dir", project_root=PROJECT_ROOT
    )
    summary_50m = common.dataset_metrics_summary_path(
        artifacts_root,
        summary_template,
        dataset_name="yambda_50m_listens",
    )
    summary_500m = common.dataset_metrics_summary_path(
        artifacts_root,
        summary_template,
        dataset_name="yambda_500m_listens",
    )
    assert summary_50m != summary_500m
    assert summary_50m.name == "sasrec_yambda_50m_listens_metrics_summary.csv"
    assert summary_500m.name == "sasrec_yambda_500m_listens_metrics_summary.csv"


def test_every_dataset_trains_with_the_same_dense_causal_layout() -> None:
    plans = run_all.build_group_plans(run_all.DEFAULT_CONFIG_DIR, None)

    run_all.validate_train_layout(plans)

    drifted = copy.deepcopy(plans)
    drifted[-1].dataset_config["sasrec"]["train_example_layout"] = "prefix_target"
    with pytest.raises(ValueError, match="identically"):
        run_all.validate_train_layout(drifted)


def test_yambda_runner_preflights_all_local_inputs(tmp_path: Path) -> None:
    plan = _plan("yambda")
    dataset = copy.deepcopy(plan.dataset_config)
    embeddings = tmp_path / "embeddings.parquet"
    interactions_50m = tmp_path / "50m.parquet"
    interactions_500m = tmp_path / "500m.parquet"
    for path in (embeddings, interactions_50m, interactions_500m):
        path.touch()

    dataset["embeddings_path"] = str(embeddings)
    dataset["datasets"]["yambda_50m_listens"]["interactions_path"] = str(
        interactions_50m
    )
    dataset["datasets"]["yambda_500m_listens"]["interactions_path"] = str(
        interactions_500m
    )
    names = plan.dataset_names

    run_all.validate_yambda_inputs(names, dataset)

    interactions_500m.unlink()
    with pytest.raises(FileNotFoundError, match="500m.parquet"):
        run_all.validate_yambda_inputs(names, dataset)


def test_experiment_directories_include_cluster_job_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {"same": "config"}
    monkeypatch.setenv("RUN_ID", "1001")
    first = common.create_run_dir(tmp_path, "sasrec_yambda_50m", config)
    monkeypatch.setenv("RUN_ID", "1002")
    second = common.create_run_dir(tmp_path, "sasrec_yambda_500m", config)

    assert first != second
    assert "_1001_" in first.name
    assert "_1002_" in second.name


class _StaticModel(torch.nn.Module):
    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, batch):
        return RecModelOutput(scores=self.scores[batch["row_ids"]])


def test_common_runner_writes_all_four_metric_families_for_both_splits(
    tmp_path: Path,
) -> None:
    scores = torch.full((1, 21), -torch.inf)
    scores[0, 1:] = torch.arange(20, 0, -1)
    metrics = common.evaluate_ranking_metrics(
        _StaticModel(scores),
        [
            {
                "row_ids": torch.tensor([0]),
                "target_item_ids": torch.tensor([1]),
                "history_item_ids": torch.tensor([[2]]),
                "history_mask": torch.tensor([[True]]),
            }
        ],
        device="cpu",
        item_popularity=torch.arange(21, dtype=torch.float64),
        num_catalog_items=20,
    )
    summaries = [
        {
            "dataset": {"name": "example"},
            "seed": 42,
            "validation_metrics": metrics,
            "test_metrics": metrics,
            "best_checkpoint": "best.pt",
            "experiment_dir": "run",
        }
    ]
    output = tmp_path / "metrics.csv"
    common.write_metrics_summary(output, summaries)

    with open(output, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["split"] for row in rows] == ["validation", "test"]
    for row in rows:
        for k in (5, 10, 20):
            assert row[f"HR@{k}"] != ""
            assert row[f"NDCG@{k}"] != ""
            assert row[f"Coverage@{k}"] != ""
            assert row[f"ARP@{k}"] != ""
