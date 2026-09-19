"""Tests for the three-seed runner of the tuned SASRec + SCE models
(run_sasrec_sce_tuned_all.py) and the evaluator's per-seed mode.

Unit tests cover the tuned-YAML split and the trainer regime; the smoke test
tunes a one-trial study on the synthetic Yambda dataset, trains the tuned
config for three seeds with a step cap on CPU, checks the seed checkpoints
and metric tables, and re-scores the seed checkpoints with the evaluator.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yambda_smoke_assets import SMOKE_DATASET, build_smoke_config_dir  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts" / "exp2-tuned-params"
SPEC = importlib.util.spec_from_file_location("run_sasrec_sce_tuned_all", SCRIPTS / "run_sasrec_sce_tuned_all.py")
assert SPEC is not None and SPEC.loader is not None
run_tuned = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_tuned
SPEC.loader.exec_module(run_tuned)

METRIC_COLUMNS = [f"{m}@{k}" for k in (5, 10, 20) for m in ("HR", "NDCG", "Coverage", "ARP")]


def _write_tuned_yaml(tmp_path: Path, dataset: str = "yambda_50m_listens") -> Path:
    sce = run_tuned.sce_tuning
    rt = sce.Runtime(tuned_config_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
    hparams = {**sce.tuning.EXP1_BASELINE_PARAMS, **sce.SCE_BASELINE_PARAMS, "lr": 5e-4}
    return sce.write_tuned_config(
        dataset_name=dataset, hparams=hparams,
        sce=sce.resolve_sce_config(hparams, bucket_base=160).to_dict(), bucket_base=160,
        best_value=0.1, best_trial_number=3, best_epoch=7, trials_completed=10, trials_planned=50,
        study_name="s", spec=sce.TUNING_SPECS[dataset], rt=rt,
    )


def test_tuned_yaml_splits_into_model_and_trainer_and_requires_sce(tmp_path: Path) -> None:
    path = _write_tuned_yaml(tmp_path)
    model_config, trainer = run_tuned.load_tuned_config("yambda_50m_listens", tuned_config_dir=tmp_path)
    assert model_config["d_model"] == 128 and model_config["num_layers"] == 3
    assert model_config["sft"]["item_loss"] == "scalable_cross_entropy"
    assert model_config["sft"]["scalable_cross_entropy"] == {"n_buckets": 160, "bucket_size_x": 160, "bucket_size_y": 1024, "mix_x": False}
    assert set(trainer) == {"lr", "weight_decay", "betas", "eps", "max_grad_norm", "gradient_accumulation_steps"}
    assert trainer["lr"] == pytest.approx(5e-4) and trainer["gradient_accumulation_steps"] == 16
    assert not (set(model_config) & run_tuned.TRAINER_KEYS)

    with pytest.raises(FileNotFoundError, match="tuning_sasrec_sce.py"):
        run_tuned.load_tuned_config("beauty_2023", tuned_config_dir=tmp_path)
    # A full-CE YAML (or an unresolved 'auto') is rejected.
    cfg = yaml.safe_load(path.read_text())
    cfg["sft"]["item_loss"] = "cross_entropy"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="scalable_cross_entropy"):
        run_tuned.load_tuned_config("yambda_50m_listens", tuned_config_dir=tmp_path)
    cfg["sft"]["item_loss"] = "scalable_cross_entropy"
    cfg["sft"]["scalable_cross_entropy"]["n_buckets"] = "auto"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="n_buckets"):
        run_tuned.load_tuned_config("yambda_50m_listens", tuned_config_dir=tmp_path)


def test_trainer_config_mirrors_the_tuning_regime() -> None:
    spec = run_tuned.sce_tuning.TUNING_SPECS["yambda_500m_listens"]
    cfg = run_tuned.build_trainer_config(
        "yambda_500m_listens", "yambda", {"lr": 1e-3, "weight_decay": 1e-2, "gradient_accumulation_steps": 16}
    )
    assert cfg["max_epochs"] == spec.max_epochs and cfg["early_stopping_patience"] == spec.patience
    assert cfg["monitor"] == "val/sft/ndcg@10" and cfg["seeds"] == [42, 43, 44]
    assert cfg["checkpoints_dir"] == "checkpoints/exp2-tuned_params/sasrec_sce"
    assert cfg["artifacts_dir"] == "artifacts/exp2-tuned-params/sasrec_sce"
    assert cfg["metrics_summary_name"] == "sasrec_sce_{dataset}_metrics_summary.csv"
    assert cfg["save_interaction_store"] is False
    # The exp1 runner accepts every key (no unknown trainer keys).
    import argparse
    trainer = run_tuned.runner.build_trainer_config(cfg, Path("/tmp/x"), argparse.Namespace(max_epochs=None, max_steps=None))
    assert trainer.early_stopping_patience == spec.patience


def test_smoke_trains_three_seeds_and_evaluates_them(tmp_path: Path) -> None:
    config_dir = build_smoke_config_dir(tmp_path)
    tune_ckpt, tuned_dir = tmp_path / "tune_ckpt", tmp_path / "tuned"
    run_ckpt, experiments, artifacts = tmp_path / "ckpt", tmp_path / "experiments", tmp_path / "artifacts"

    proc = subprocess.run([sys.executable, str(SCRIPTS / "tuning_sasrec_sce.py"),
                           "--datasets", SMOKE_DATASET, "--spec", "yambda_50m_listens",
                           "--config-dir", str(config_dir), "--checkpoint-root", str(tune_ckpt),
                           "--artifacts-dir", str(artifacts), "--tuned-config-dir", str(tuned_dir),
                           "--n-trials", "1", "--max-epochs", "1", "--train-batches-per-epoch", "2",
                           "--max-eval-batches", "2", "--num-workers", "0", "--device", "cpu", "--no-progress-bar"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    proc = subprocess.run([sys.executable, str(SCRIPTS / "run_sasrec_sce_tuned_all.py"),
                           "--datasets", SMOKE_DATASET, "--spec", "yambda_50m_listens",
                           "--config-dir", str(config_dir), "--tuned-config-dir", str(tuned_dir),
                           "--checkpoints-dir", str(run_ckpt), "--experiments-dir", str(experiments),
                           "--artifacts-dir", str(artifacts), "--device", "cpu", "--max-steps", "3",
                           "--num-workers", "0"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    for seed in (42, 43, 44):
        assert (run_ckpt / SMOKE_DATASET / f"seed_{seed}" / "best.pt").exists()
    summary = json.loads((run_ckpt / SMOKE_DATASET / "training_summary.json").read_text())
    assert sorted(r["seed"] for r in summary["runs"]) == [42, 43, 44]
    assert len(list(experiments.glob(f"*sasrec_{SMOKE_DATASET}_seed-*"))) == 3

    with (artifacts / f"sasrec_sce_{SMOKE_DATASET}_metrics_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["seed"], r["split"]) for r in rows] == [(str(s), split) for s in (42, 43, 44) for split in ("validation", "test")]
    for row in rows:
        assert row["dataset"] == SMOKE_DATASET
        for column in METRIC_COLUMNS:
            assert float(row[column]) >= 0.0
    assert not (artifacts / run_tuned.COMBINED_METRICS_SUMMARY_NAME).exists()
    # Runs record the SCE objective they trained with.
    run_dir = next(experiments.glob(f"*sasrec_{SMOKE_DATASET}_seed-42*"))
    text = "".join(p.read_text() for p in run_dir.glob("*.json"))
    assert "scalable_cross_entropy" in text and '"variant": "sce"' in text

    # The evaluator re-scores the seed checkpoints without retraining.
    proc = subprocess.run([sys.executable, str(SCRIPTS / "evaluate_sasrec_sce_tuned.py"),
                           "--datasets", SMOKE_DATASET, "--seeds", "42", "43", "44",
                           "--config-dir", str(config_dir), "--checkpoint-root", str(run_ckpt),
                           "--artifacts-dir", str(artifacts), "--tuned-config-dir", str(tuned_dir),
                           "--num-workers", "0", "--device", "cpu", "--max-eval-batches", "2"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    with (artifacts / f"sasrec_sce_{SMOKE_DATASET}_seeds_eval_metrics_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["seed"], r["split"]) for r in rows] == [(str(s), split) for s in (42, 43, 44) for split in ("validation", "test")]
    assert all(row["best_checkpoint"].endswith(f"seed_{row['seed']}/best.pt") for row in rows)
    seeds_json = json.loads((artifacts / f"sasrec_sce_{SMOKE_DATASET}_seeds_eval_summary.json").read_text())
    assert [run["seed"] for run in seeds_json["runs"]] == [42, 43, 44]
    assert all(run["checkpoint_kind"] == f"seed_{run['seed']}/best.pt" for run in seeds_json["runs"])
