"""Three-seed SCE final runs select on validation and score the global test split."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from test_global_time_representation import _assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "global_sasrec_sce_final", PROJECT_ROOT / "scripts/global-time/run_sasrec_sce_tuned_all.py"
)
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)
DATASET = "yambda_500m_listens"


def _tuned(config_dir):
    tuning = run.tuning
    hparams = {**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS,
               "d_model": 8, "nhead": 1, "num_layers": 1, "lr": 0.001}
    return tuning.write_tuned_config(
        dataset_name=DATASET, hparams=hparams,
        sce=tuning.resolve_sce_config(hparams, bucket_base=8).to_dict(), bucket_base=8,
        best_value=0.2, best_trial_number=0, best_epoch=0,
        trials_completed=1, trials_planned=1, study_name="smoke_global_time",
        spec=tuning.TUNING_SPECS[DATASET],
        rt=tuning.Runtime(tuned_config_dir=config_dir, required_split_strategy="global_time"),
    )


def _config(tmp_path):
    config_dir = _assets(tmp_path, DATASET, "yambda")
    path = config_dir / "dataset_yambda.yaml"
    config = yaml.safe_load(path.read_text())
    config["datasets"][DATASET]["dataset_size"] = "500m"
    path.write_text(yaml.safe_dump(config))
    # Retain the cold-last-target fixture, expanding to 24 warm catalog
    # items so the real evaluator can compute top-20 recommendations.
    path = tmp_path / "interactions.parquet"
    frame = pd.read_parquet(path)
    warm = frame.loc[frame.timestamp.eq(1) & frame.item_id.eq(1)]
    pd.concat([frame, *(warm.assign(item_id=i) for i in range(100, 122))],
              ignore_index=True).to_parquet(path)
    path = tmp_path / "embeddings.parquet"
    embeddings = pd.read_parquet(path)
    pd.concat([embeddings, *(embeddings.iloc[:1].assign(item_id=i) for i in range(100, 122))],
              ignore_index=True).to_parquet(path)
    _tuned(config_dir)
    return config_dir


def _args(tmp_path, config_dir):
    return ["--config-dir", str(config_dir),
            "--checkpoints-dir", str(tmp_path / "checkpoints"),
            "--experiments-dir", str(tmp_path / "experiments"),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--device", "cpu", "--num-workers", "0", "--max-epochs", "1", "--no-progress-bar"]


def test_defaults_match_sce_winner_and_global_time_paths():
    args = run.parse_args([])
    assert args.datasets == [DATASET] and args.seeds == [42, 43, 44]
    for path in (args.config_dir, args.tuned_config_dir, args.checkpoints_dir,
                 args.experiments_dir, args.artifacts_dir):
        assert "global-time" in path.parts
    for invalid in (["--seeds", "42"], ["--seeds", "42", "43", "43"],
                    ["--max-epochs", "0"], ["--num-workers", "-1"]):
        with pytest.raises(SystemExit):
            run.parse_args(invalid)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "scripts/scheduler/run_sasrec_sce_global_time_tuned_all.submit-worker").is_file(),
    reason="scheduler scripts are gitignored",
)
def test_three_seed_launcher():
    launcher = (PROJECT_ROOT / "scripts/scheduler/run_sasrec_sce_global_time_tuned_all.submit-worker").read_text()
    assert "# execution option --account=research-project" in launcher
    assert "--seeds 42 43 44" in launcher
    assert "scripts/global-time/run_sasrec_sce_tuned_all.py" in launcher
    assert "# execution option --array" not in launcher  # One writer for the three-seed summaries.


@pytest.mark.parametrize("changes", [
    {"split_strategy": "leave_one_out"},
    {"global_history_policy": "train_only"},
    {"global_eval_target_mode": "all"},
    {"filter_cold_items_after_split": False},
    {"cold_item_train_source": "targets"},
])
def test_invalid_split_fails_before_training(tmp_path, changes):
    config_dir = _config(tmp_path)
    path = config_dir / "dataset_yambda.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(changes)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires"):
        run.main(_args(tmp_path, config_dir))
    assert not (tmp_path / "experiments").exists()


def test_missing_global_winner_has_no_fallback_and_wrong_loss_is_rejected(tmp_path):
    config_dir = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="Tuned SCE config not found"):
        run.main([*_args(tmp_path, config_dir), "--tuned-config-dir", str(tmp_path / "missing")])
    path = config_dir / f"sasrec_sce_model_{DATASET}.yaml"
    config = yaml.safe_load(path.read_text())
    config["sft"]["item_loss"] = "cross_entropy"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="does not train with scalable_cross_entropy"):
        run.main(_args(tmp_path, config_dir))
    assert not (tmp_path / "experiments").exists()


def test_three_seeds_reload_validation_best_and_write_test_mean_std(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config_dir = _config(tmp_path)
    events, snapshots = [], []
    original_load = run.runner.Trainer.load_checkpoint
    original_evaluate = run.runner.evaluate_ranking_metrics
    original_experiment = run.runner.run_experiment

    def experiment(name, **kwargs):
        data = kwargs["data"]
        assert set(data.item2id) == {"1", "2", *(str(i) for i in range(100, 122))}
        assert len(data.val_dataset) == len(data.test_dataset) == 1
        assert data.train_dataloader().generator.initial_seed() == kwargs["seed"]
        snapshots.append(data)
        return original_experiment(name, **kwargs)

    def load_checkpoint(self, path, **kwargs):
        assert kwargs == {"load_optimizer": False, "load_scheduler": False}
        events.append(("reload", str(path)))
        return original_load(self, path, **kwargs)

    def evaluate(model, dataloader, **kwargs):
        events.append(("evaluate", None))
        return original_evaluate(model, dataloader, **kwargs)

    monkeypatch.setattr(run.runner, "run_experiment", experiment)
    monkeypatch.setattr(run.runner.Trainer, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(run.runner, "evaluate_ranking_metrics", evaluate)
    winner = tmp_path / "checkpoints" / DATASET / "best_model.pt"
    winner.parent.mkdir(parents=True)
    winner.write_text("tuning winner")
    run.main(_args(tmp_path, config_dir))
    assert winner.read_text() == "tuning winner"
    assert [event[0] for event in events] == ["reload", "evaluate", "evaluate"] * 3
    assert [event[1] for event in events if event[0] == "reload"] == [
        str(winner.parent / f"seed_{seed}" / "best.pt") for seed in (42, 43, 44)
    ]
    for data in snapshots[1:]:
        assert data.item2id == snapshots[0].item2id
        for split in ("train", "val", "test"):
            pd.testing.assert_frame_equal(getattr(data, f"{split}_dataset").examples,
                                          getattr(snapshots[0], f"{split}_dataset").examples)

    summary = json.loads((winner.parent / "training_summary.json").read_text())
    assert [result["seed"] for result in summary["runs"]] == [42, 43, 44]
    weights = []
    for result in summary["runs"]:
        seed = result["seed"]
        checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
        weights.append(next(iter(checkpoint["model_state_dict"].values())))
        resolved = json.loads((Path(result["experiment_dir"]) / "resolved_config.json").read_text())
        assert resolved["seed"] == resolved["data"]["seed"] == seed
        assert resolved["data"]["split_strategy"] == "global_time"
        assert (resolved["data"]["global_val_timestamp"], resolved["data"]["global_test_timestamp"]) == (10, 20)
        assert resolved["method"]["item_loss"] == "scalable_cross_entropy"
        assert resolved["method"]["scalable_cross_entropy"]["n_buckets"] == 8
        assert resolved["variant"] == "sce"
        assert resolved["trainer"]["monitor"] == "val/sft/ndcg@10"
        assert resolved["trainer"]["early_stopping_patience"] == run.tuning.TUNING_SPECS[DATASET].patience
        assert result["test_metrics"]["num_examples"] == 1
    assert not torch.equal(weights[0], weights[1])

    artifacts = tmp_path / "artifacts"
    with (artifacts / f"sasrec_sce_{DATASET}_metrics_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["seed"], r["split"]) for r in rows] == [
        (str(seed), split) for seed in (42, 43, 44) for split in ("validation", "test")
    ]
    with (artifacts / run.AGGREGATE_SUMMARY_NAME.format(dataset=DATASET)).open() as handle:
        aggregate = list(csv.DictReader(handle))
    assert len(aggregate) == 24
    for row in aggregate:
        values = [float(r[row["metric"]]) for r in rows if r["split"] == row["split"]]
        assert int(row["num_seeds"]) == 3
        assert float(row["mean"]) == pytest.approx(statistics.fmean(values))
        assert float(row["std"]) == pytest.approx(statistics.stdev(values))
    assert not (artifacts / run.base.COMBINED_METRICS_SUMMARY_NAME).exists()
