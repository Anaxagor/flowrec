"""Tests for the tuned SASRec + SCE evaluator (evaluate_sasrec_sce_tuned.py).

The smoke test tunes a one-trial study on the synthetic Yambda dataset to
obtain a ``best_model.pt``, then evaluates it and checks the CSV/JSON
artifacts; unit tests cover the checkpoint/seed helpers.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yambda_smoke_assets import SMOKE_DATASET, build_smoke_config_dir  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts" / "exp2-tuned-params"
SPEC = importlib.util.spec_from_file_location("evaluate_sasrec_sce_tuned", SCRIPTS / "evaluate_sasrec_sce_tuned.py")
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)

METRIC_COLUMNS = [f"{m}@{k}" for k in (5, 10, 20) for m in ("HR", "NDCG", "Coverage", "ARP")]


def test_training_seed_is_derived_from_the_best_trial_record() -> None:
    assert evaluate.training_seed(None) is None
    assert evaluate.training_seed({"best_trial_number": 7}) == evaluate.sce_tuning.SEED + 7
    reranked = {
        "best_trial_number": 3,
        "rerank_ranking": [
            {"trial": 9, "seed_results": [{"seed": 42, "value": 0.9}]},
            {"trial": 3, "seed_results": [{"seed": 42, "value": 0.1}, {"seed": 44, "value": 0.3}, {"seed": 43, "value": 0.2}]},
        ],
    }
    assert evaluate.training_seed(reranked) == 44


def test_checkpoint_vocab_must_match_the_datamodule() -> None:
    from flowrec.models.sasrec import SASRec, SASRecConfig

    model = SASRec(SASRecConfig(item_vocab_size=12, max_history_len=4, d_model=8, nhead=2, num_layers=1))
    payload = {"model_config": model.config_dict(), "model_state_dict": model.state_dict()}
    rebuilt = evaluate.build_model_from_checkpoint(payload, SimpleNamespace(item_vocab_size=12))
    assert rebuilt.config_dict() == model.config_dict()
    for a, b in zip(rebuilt.state_dict().values(), model.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="item_vocab_size"):
        evaluate.build_model_from_checkpoint(payload, SimpleNamespace(item_vocab_size=13))
    with pytest.raises(KeyError, match="model_config"):
        evaluate.build_model_from_checkpoint({"model_state_dict": {}}, SimpleNamespace(item_vocab_size=12))


def test_smoke_evaluates_a_tuned_checkpoint_and_writes_artifacts(tmp_path: Path) -> None:
    config_dir = build_smoke_config_dir(tmp_path)
    ckpt_root, artifacts, tuned_dir = tmp_path / "ckpt", tmp_path / "artifacts", tmp_path / "tuned"
    common = ["--config-dir", str(config_dir), "--checkpoint-root", str(ckpt_root),
              "--artifacts-dir", str(artifacts), "--tuned-config-dir", str(tuned_dir),
              "--num-workers", "0", "--device", "cpu"]

    # Without a checkpoint the evaluator fails fast, or skips with the flag.
    proc = subprocess.run([sys.executable, str(SCRIPTS / "evaluate_sasrec_sce_tuned.py"),
                           "--datasets", SMOKE_DATASET, *common],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0 and "Missing tuned SCE checkpoints" in proc.stderr

    proc = subprocess.run([sys.executable, str(SCRIPTS / "tuning_sasrec_sce.py"),
                           "--datasets", SMOKE_DATASET, "--spec", "yambda_50m_listens", *common,
                           "--n-trials", "1", "--max-epochs", "1", "--train-batches-per-epoch", "2",
                           "--max-eval-batches", "2", "--no-progress-bar"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    checkpoint = ckpt_root / SMOKE_DATASET / "best_model.pt"
    assert checkpoint.exists()

    proc = subprocess.run([sys.executable, str(SCRIPTS / "evaluate_sasrec_sce_tuned.py"),
                           "--datasets", SMOKE_DATASET, *common, "--max-eval-batches", "3"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    with (artifacts / f"sasrec_sce_{SMOKE_DATASET}_eval_metrics_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["validation", "test"]
    for row in rows:
        assert row["dataset"] == SMOKE_DATASET
        assert row["seed"] == str(evaluate.sce_tuning.SEED)  # trial 0 of the search
        assert row["best_checkpoint"] == str(checkpoint)
        assert float(row["num_examples"]) > 0
        for column in METRIC_COLUMNS:
            value = float(row[column])
            assert value >= 0.0
            if column.startswith(("HR", "NDCG", "Coverage")):
                assert value <= 1.0

    summary = json.loads((artifacts / f"sasrec_sce_{SMOKE_DATASET}_eval_summary.json").read_text())
    run = summary["run"]
    assert run["item_loss"] == "scalable_cross_entropy"
    assert set(run["scalable_cross_entropy"]) == {"n_buckets", "bucket_size_x", "bucket_size_y", "mix_x"}
    assert run["tuning"]["best_trial_number"] == 0 and run["tuning"]["best_value_val_ndcg10"] is not None
    assert run["model"]["item_vocab_size"] == run["dataset"]["item_vocab_size"]
    assert run["tuned_config"] == str(tuned_dir / f"sasrec_sce_model_{SMOKE_DATASET}.yaml")
    assert set(run["validation_metrics"]) >= set(METRIC_COLUMNS) and set(run["test_metrics"]) >= set(METRIC_COLUMNS)
    assert summary["checkpoint"] == str(checkpoint) and summary["ranking_ks"] == [5, 10, 20]
    # A single dataset never writes the combined table.
    assert not (artifacts / evaluate.COMBINED_EVAL_SUMMARY_NAME).exists()
