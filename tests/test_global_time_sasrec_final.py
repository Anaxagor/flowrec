"""Final global-time full-CE runs: split parity, checkpoint selection and metrics."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from test_global_time_representation import _assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "global_sasrec_final", PROJECT_ROOT / "scripts/global-time/run_sasrec_tuned_all.py"
)
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


def _tuned(config_dir, name):
    best = SimpleNamespace(params={
        **run.tuning.base.EXP1_BASELINE_PARAMS,
        "d_model": 8, "num_layers": 1, "nhead": 1, "lr": 0.001,
    })
    config = run.tuning.tuned_config(best, run.tuning.TUNING_SPECS[name])
    path = config_dir / f"sasrec_model_{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _args(tmp_path, config_dir, name):
    return [
        "--datasets", name, "--config-dir", str(config_dir),
        "--checkpoints-dir", str(tmp_path / "checkpoints"),
        "--experiments-dir", str(tmp_path / "experiments"),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--device", "cpu", "--num-workers", "0", "--max-epochs", "1",
        "--no-progress-bar",
    ]


def _ranking_assets(tmp_path, name):
    """Expand the cold-item fixture to 24 warm items for real top-20 scoring."""
    family = "yambda" if name.startswith("yambda") else "amazon"
    config_dir = _assets(tmp_path, name, family)
    path = tmp_path / "interactions.parquet"
    frame = pd.read_parquet(path)
    warm = frame.loc[frame.timestamp.eq(1) & frame.item_id.eq(1)]
    extra = []
    item_key = "parent_asin" if name == "beauty_2023" else "asin"
    for item in range(100, 122):
        copy = warm.assign(item_id=item)
        if family == "amazon":
            copy[item_key] = str(item)
        extra.append(copy)
    pd.concat([frame, *extra], ignore_index=True).to_parquet(path)
    path = tmp_path / ("metadata.parquet" if family == "amazon" else "embeddings.parquet")
    metadata = pd.read_parquet(path)
    key = item_key if family == "amazon" else "item_id"
    copies = [metadata.iloc[:1].assign(**{key: str(item) if family == "amazon" else item})
              for item in range(100, 122)]
    pd.concat([metadata, *copies], ignore_index=True).to_parquet(path)
    return config_dir


def test_defaults_and_array_mapping():
    args = run.parse_args([])
    assert args.datasets == ["beauty", "instruments", "beauty_2023", "yambda_50m_listens"]
    assert args.seeds == [42, 43, 44]
    assert args.checkpoints_dir == PROJECT_ROOT / "checkpoints/exp2/global_split/initial/sasrec"
    for path in (args.config_dir, args.tuned_config_dir,
                 args.experiments_dir, args.artifacts_dir):
        assert "global-time" in path.parts
    for i, dataset in enumerate(args.datasets):
        task = run.parse_args(["--array-task-id", str(i)])
        assert task.datasets == [dataset] and task.seeds == [42, 43, 44]
    for invalid in (["--array-task-id", "4"], ["--seeds", "42", "42"],
                    ["--array-task-id", "0", "--datasets", "beauty"]):
        with pytest.raises(SystemExit):
            run.parse_args(invalid)




def test_tuned_config_loader_uses_global_export_and_rejects_other_objectives(tmp_path):
    path = _tuned(tmp_path, "beauty")
    model, trainer = run.load_tuned_config("beauty", tmp_path)
    assert model["d_model"] == 8
    assert model["sft"]["item_loss"] == "cross_entropy"
    assert trainer["lr"] == 0.001
    assert not (set(model) & run.base.TRAINER_KEYS)
    config = yaml.safe_load(path.read_text())
    config["sft"]["item_loss"] = "scalable_cross_entropy"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires full cross_entropy"):
        run.load_tuned_config("beauty", tmp_path)
    with pytest.raises(FileNotFoundError, match="tuning_sasrec.py"):
        run.load_tuned_config("instruments", tmp_path)


def test_shared_loader_preserves_legacy_default_directory(tmp_path, monkeypatch):
    _tuned(tmp_path, "beauty")
    monkeypatch.setattr(run.base, "CONFIG_DIR", tmp_path)
    assert run.base.load_tuned_config("beauty") == run.load_tuned_config("beauty", tmp_path)
    # An explicit global config directory must never fall back to legacy files.
    with pytest.raises(FileNotFoundError):
        run.base.load_tuned_config("beauty", config_dir=tmp_path / "missing_global_configs")


@pytest.mark.parametrize("changes", [{"split_strategy": "leave_one_out"},
                                     {"filter_cold_items_after_split": False},
                                     {"cold_item_train_source": "targets"}])
def test_final_runner_rejects_incompatible_split_before_training(tmp_path, changes):
    config_dir = _assets(tmp_path, "beauty")
    _tuned(config_dir, "beauty")
    path = config_dir / "dataset_amazon.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(changes)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires global_time"):
        run.main(_args(tmp_path, config_dir, "beauty"))
    assert not (tmp_path / "experiments").exists()


def test_preflight_checks_all_winners_before_training(tmp_path, monkeypatch):
    config_dir = _assets(tmp_path, "beauty")
    _tuned(config_dir, "beauty")

    def unexpected_training(**kwargs):
        raise AssertionError("All selected winners must be checked before training")

    monkeypatch.setattr(run.runner, "run_sweep", unexpected_training)
    args = _args(tmp_path, config_dir, "beauty")
    args.insert(2, "instruments")
    with pytest.raises(FileNotFoundError, match="sasrec_model_instruments.yaml"):
        run.main(args)


@pytest.mark.parametrize("name", run.DATASETS)
def test_three_seeds_train_and_reload_best_before_test(tmp_path, monkeypatch, name):
    torch.set_num_threads(1)
    config_dir = _ranking_assets(tmp_path, name)
    _tuned(config_dir, name)
    group, cfg = run.tuning.resolve_data_config(name, run.tuning.Runtime(config_dir=config_dir))
    reference = run.tuning.build_datamodule(group, cfg)
    assert len(reference.item2id) == 24
    assert "3" not in reference.item2id
    assert len(reference.val_dataset) == len(reference.test_dataset) == 1

    # Train/validation/test examples and catalogs stay fixed across seeds;
    # only training randomness (including loader shuffle) changes.
    for seed in (43, 44):
        data = run.tuning.build_datamodule(group, replace(cfg, seed=seed))
        assert data.item2id == reference.item2id
        assert data.train_dataloader().generator.initial_seed() == seed
        for split in ("train", "val", "test"):
            pd.testing.assert_frame_equal(getattr(data, f"{split}_dataset").examples,
                                          getattr(reference, f"{split}_dataset").examples)

    events = []
    original_load = run.runner.Trainer.load_checkpoint
    original_evaluate = run.runner.evaluate_ranking_metrics

    def load_checkpoint(self, path, **kwargs):
        assert kwargs == {"load_optimizer": False, "load_scheduler": False}
        events.append(("reload", str(path)))
        return original_load(self, path, **kwargs)

    def evaluate(model, dataloader, **kwargs):
        events.append(("evaluate", None))
        return original_evaluate(model, dataloader, **kwargs)

    monkeypatch.setattr(run.runner.Trainer, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(run.runner, "evaluate_ranking_metrics", evaluate)
    # Final training must neither consume nor replace the tuning checkpoint.
    winner = tmp_path / "checkpoints" / name / "best_model.pt"
    winner.parent.mkdir(parents=True)
    winner.write_text("tuning winner")
    run.main(_args(tmp_path, config_dir, name))
    assert winner.read_text() == "tuning winner"
    assert [event[0] for event in events] == ["reload", "evaluate", "evaluate"] * 3

    summary = json.loads((winner.parent / "training_summary.json").read_text())
    assert summary["seeds"] == [42, 43, 44]
    assert [result["seed"] for result in summary["runs"]] == [42, 43, 44]
    weights = []
    for result in summary["runs"]:
        seed = result["seed"]
        path = winner.parent / f"seed_{seed}" / "best.pt"
        assert result["best_checkpoint"] == str(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights.append(next(iter(checkpoint["model_state_dict"].values())))
        resolved = json.loads((Path(result["experiment_dir"]) / "resolved_config.json").read_text())
        assert resolved["seed"] == resolved["data"]["seed"] == seed
        assert resolved["data"]["split_strategy"] == "global_time"
        assert resolved["data"]["filter_cold_items_after_split"] is True
        assert resolved["method"]["item_loss"] == "cross_entropy"
        spec = run.tuning.TUNING_SPECS[name]
        assert resolved["trainer"]["early_stopping_patience"] == spec.patience
        assert resolved["trainer"]["gradient_accumulation_steps"] == spec.grad_accum
        assert resolved["trainer"]["monitor"] == "val/sft/ndcg@10"
        assert result["test_metrics"]["num_examples"] == 1
    assert not torch.equal(weights[0], weights[1])

    artifacts = tmp_path / "artifacts"
    with (artifacts / f"sasrec_{name}_metrics_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["seed"], r["split"]) for r in rows] == [
        (str(seed), split) for seed in (42, 43, 44) for split in ("validation", "test")
    ]
    with (artifacts / f"sasrec_{name}_metrics_mean_std.csv").open() as handle:
        aggregate = list(csv.DictReader(handle))
    assert len(aggregate) == 24
    for row in aggregate:
        values = [float(r[row["metric"]]) for r in rows if r["split"] == row["split"]]
        assert int(row["num_seeds"]) == 3
        assert float(row["mean"]) == pytest.approx(statistics.fmean(values))
        assert float(row["std"]) == pytest.approx(statistics.stdev(values))
    assert not (artifacts / run.base.COMBINED_METRICS_SUMMARY_NAME).exists()


def test_multi_dataset_invocation_writes_combined_and_partial_seed_statistics(tmp_path):
    torch.set_num_threads(1)
    config_dir = _ranking_assets(tmp_path, "beauty")
    path = config_dir / "dataset_amazon.yaml"
    config = yaml.safe_load(path.read_text())
    config["datasets"]["instruments"] = dict(config["datasets"]["beauty"])
    path.write_text(yaml.safe_dump(config))
    for name in ("beauty", "instruments"):
        _tuned(config_dir, name)
    args = _args(tmp_path, config_dir, "beauty")
    args.insert(2, "instruments")
    run.main([*args, "--seeds", "42"])
    artifacts = tmp_path / "artifacts"
    with (artifacts / run.base.COMBINED_METRICS_SUMMARY_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["dataset"], r["split"]) for r in rows] == [
        (name, split) for name in ("beauty", "instruments") for split in ("validation", "test")
    ]
    with (artifacts / "sasrec_all_metrics_mean_std.csv").open() as handle:
        aggregate = list(csv.DictReader(handle))
    assert len(aggregate) == 48
    assert {row["dataset"] for row in aggregate} == {"beauty", "instruments"}
    assert all(row["num_seeds"] == "1" and row["std"] == "" for row in aggregate)
