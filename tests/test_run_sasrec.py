"""Tests for the shared SASRec runner machinery in scripts/sasrec_config_runner."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from flowrec.models.base import RecModelOutput
from scripts import sasrec_config_runner as common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"


def _configs():
    dataset = common.load_yaml(CONFIG_DIR / "dataset_amazon.yaml")
    model = common.load_yaml(CONFIG_DIR / "sasrec_model.yaml")
    trainer = common.load_yaml(CONFIG_DIR / "trainer_sasrec.yaml")
    return dataset, model, trainer


def test_exp1_sasrec_config_resolves_three_seed_output_roots() -> None:
    _, _, trainer_cfg = _configs()
    args = argparse.Namespace(seeds=None)

    assert common.resolve_seeds(args, trainer_cfg) == [42, 43, 44]
    checkpoint_root = common.configured_path(
        trainer_cfg, "checkpoints_dir", project_root=PROJECT_ROOT
    )
    assert checkpoint_root == (
        PROJECT_ROOT / "checkpoints" / "exp1-standard_params" / "sasrec"
    )
    assert common.configured_path(
        trainer_cfg, "artifacts_dir", project_root=PROJECT_ROOT
    ) == PROJECT_ROOT / "artifacts" / "exp1-standard_params"


def test_exp1_sasrec_model_method_optimizer_and_trainer_are_config_driven(
    tmp_path: Path,
) -> None:
    _, model_cfg, trainer_cfg = _configs()
    data = SimpleNamespace(
        item_vocab_size=101,
        cfg=SimpleNamespace(max_history_len=20, train_window_len=None),
    )

    model = common.build_model(model_cfg, data)
    assert model.d_model == model_cfg["d_model"]
    assert model.nhead == model_cfg["nhead"]
    assert model.num_layers == model_cfg["num_layers"]
    assert model.cfg.initializer_range == model_cfg["initializer_range"]
    assert model.cfg.layer_norm_eps == model_cfg["layer_norm_eps"]

    method = common.build_method(model_cfg)
    assert method.cfg.to_dict()["item_topk"] == model_cfg["sft"]["item_topk"]

    args = argparse.Namespace(max_epochs=None, max_steps=None)
    trainer = common.build_trainer_config(trainer_cfg, tmp_path, args)
    assert trainer.max_epochs == trainer_cfg["max_epochs"]
    assert trainer.monitor == trainer_cfg["monitor"]
    assert trainer.save_best is True
    assert trainer.save_last is False
    assert Path(trainer.checkpoint_dir) == tmp_path

    optimizer = common.optimizer_config(trainer_cfg)
    assert optimizer["lr"] == trainer_cfg["lr"]
    assert optimizer["weight_decay"] == trainer_cfg["weight_decay"]
    assert optimizer["betas"] == tuple(trainer_cfg["betas"])
    assert optimizer["eps"] == trainer_cfg["eps"]


class _StaticRankingModel(torch.nn.Module):
    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_scores", scores)

    def forward(self, batch):
        return RecModelOutput(scores=self._scores[batch["row_ids"]])


def test_training_popularity_excludes_leave_one_out_validation_and_test() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [0] * 5 + [1] * 5,
            "position": list(range(5)) * 2,
            "raw_item_id": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "a",
                "c",
                "c",
                "d",
                "e",
            ],
        }
    )
    data = SimpleNamespace(
        interactions=interactions,
        cfg=SimpleNamespace(split_strategy="leave_one_out", pad_item_id=0),
        item2id={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        item_vocab_size=6,
    )

    popularity = common.training_item_popularity(data)

    assert popularity.tolist() == [0.0, 2.0, 1.0, 3.0, 0.0, 0.0]


def test_ranking_metrics_include_exact_hr_ndcg_coverage_and_arp() -> None:
    # Row 0 ranks item IDs 1..20; row 1 ranks them 20..1.
    scores = torch.full((2, 21), -torch.inf)
    scores[0, 1:] = torch.arange(20, 0, -1)
    scores[1, 1:] = torch.arange(1, 21)
    model = _StaticRankingModel(scores)
    dataloader = [
        {
            "row_ids": torch.tensor([0, 1]),
            "target_item_ids": torch.tensor([2, 15]),
            "history_item_ids": torch.tensor([[1], [20]]),
            "history_mask": torch.ones(2, 1, dtype=torch.bool),
        }
    ]
    popularity = torch.arange(21, dtype=torch.float64) * 10.0

    metrics = common.evaluate_ranking_metrics(
        model,
        dataloader,
        device="cpu",
        item_popularity=popularity,
        num_catalog_items=20,
    )

    assert metrics["num_examples"] == 2.0
    assert metrics["HR@5"] == 0.5
    assert math.isclose(
        metrics["NDCG@5"],
        0.5 / math.log2(3.0),
        rel_tol=1e-6,
    )
    assert metrics["Coverage@5"] == 0.5
    assert metrics["ARP@5"] == 105.0

    assert metrics["HR@10"] == 1.0
    expected_ndcg_10 = (
        1.0 / math.log2(3.0) + 1.0 / math.log2(7.0)
    ) / 2.0
    assert math.isclose(metrics["NDCG@10"], expected_ndcg_10, rel_tol=1e-6)
    assert metrics["Coverage@10"] == 1.0
    assert metrics["ARP@10"] == 105.0

    assert metrics["HR@20"] == 1.0
    assert metrics["Coverage@20"] == 1.0
    assert metrics["ARP@20"] == 105.0


def test_metrics_summary_writes_validation_and_test_for_every_run(
    tmp_path: Path,
) -> None:
    metric_values = {
        "num_examples": 7.0,
        **{
            metric: float(index)
            for index, metric in enumerate(
                metric
                for k in common.DEFAULT_RANKING_KS
                for metric in (
                    f"HR@{k}",
                    f"NDCG@{k}",
                    f"Coverage@{k}",
                    f"ARP@{k}",
                )
            )
        },
    }
    summaries = [
        {
            "dataset": {"name": "beauty"},
            "seed": seed,
            "validation_metrics": metric_values,
            "test_metrics": metric_values,
            "best_checkpoint": f"seed_{seed}/best.pt",
            "experiment_dir": f"seed_{seed}/experiment",
        }
        for seed in (42, 43)
    ]
    output_path = tmp_path / "artifacts" / "sasrec_metrics_summary.csv"

    common.write_metrics_summary(output_path, summaries)

    with open(output_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert {(row["seed"], row["split"]) for row in rows} == {
        ("42", "validation"),
        ("42", "test"),
        ("43", "validation"),
        ("43", "test"),
    }
    for row in rows:
        for k in common.DEFAULT_RANKING_KS:
            assert row[f"HR@{k}"] != ""
            assert row[f"NDCG@{k}"] != ""
            assert row[f"Coverage@{k}"] != ""
            assert row[f"ARP@{k}"] != ""


def test_exp1_sasrec_baseline_is_exact_ce_and_sce_variant_changes_only_the_objective() -> None:
    _, model_cfg, trainer_cfg = _configs()

    base = common.resolve_model_variant(model_cfg, None)
    assert "variants" not in base
    assert base["sft"]["item_loss"] == "cross_entropy"
    assert "scalable_cross_entropy" not in base["sft"]
    assert common.build_method(base).cfg.item_loss == "cross_entropy"
    assert common.available_variants(model_cfg) == ["sce"]

    sce = common.resolve_model_variant(model_cfg, "sce")
    architecture = lambda cfg: {k: v for k, v in cfg.items() if k != "sft"}
    assert architecture(sce) == architecture(base)
    assert sce["sft"]["item_loss"] == "scalable_cross_entropy"
    # Metric settings are inherited from the shared sft block.
    assert sce["sft"]["item_topk"] == model_cfg["sft"]["item_topk"]
    # "auto" bucket sizes must be resolved from the dataset before use.
    try:
        common.build_method(sce)
    except ValueError as error:
        assert "auto" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("unresolved 'auto' bucket sizes must be rejected")

    # Paper setting: n_buckets = bucket_size_x = int(2 * sqrt(batch * window)).
    for batch_size, window, expected in ((128, 20, 101), (128, 50, 160)):
        data = SimpleNamespace(
            cfg=SimpleNamespace(
                batch_size=batch_size, max_history_len=window, train_window_len=None
            )
        )
        resolved = common.resolve_sce_bucket_sizes(sce, data)
        method = common.build_method(resolved)
        assert method.cfg.item_loss == "scalable_cross_entropy"
        buckets = method.cfg.scalable_cross_entropy
        assert buckets.n_buckets == buckets.bucket_size_x == expected
        assert expected == int(2 * math.sqrt(batch_size * window))
        assert buckets.bucket_size_y == 1024
    # Exact-CE configs pass through untouched.
    assert common.resolve_sce_bucket_sizes(base, data) == base

    try:
        common.resolve_model_variant(model_cfg, "nope")
    except KeyError as error:
        assert "sce" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("unknown variants must be rejected")

    # build_model accepts the raw YAML (with the variants block) unchanged.
    data = SimpleNamespace(
        item_vocab_size=101,
        cfg=SimpleNamespace(max_history_len=20, train_window_len=None),
    )
    assert common.build_model(model_cfg, data).d_model == model_cfg["d_model"]

    # The trainer variant only relocates outputs; every hyperparameter is kept.
    baseline = common.apply_trainer_variant(trainer_cfg, None)
    variant = common.apply_trainer_variant(trainer_cfg, "sce")
    assert "variants" not in baseline and "variants" not in variant
    for key in ("checkpoints_dir", "experiments_dir", "artifacts_dir"):
        assert variant[key] != baseline[key]
        assert "sasrec_sce" in variant[key]
    for key in baseline:
        if key not in ("checkpoints_dir", "experiments_dir", "artifacts_dir"):
            assert variant[key] == baseline[key], key
    for bad in (
        {**trainer_cfg, "variants": {"sce": {**trainer_cfg["variants"]["sce"], "lr": 1.0}}},
        {**trainer_cfg, "variants": {"sce": {"checkpoints_dir": "elsewhere"}}},
        {**trainer_cfg, "variants": {"sce": {**trainer_cfg["variants"]["sce"], "artifacts_dir": trainer_cfg["artifacts_dir"]}}},
        {**trainer_cfg, "variants": {}},
    ):
        try:
            common.apply_trainer_variant(bad, "sce")
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"invalid trainer variant accepted: {bad['variants']}")
