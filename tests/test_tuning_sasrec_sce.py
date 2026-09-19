"""Tests for the exp2 SASRec + SCE Optuna tuning script (tuning_sasrec_sce.py).

Unit tests cover the search-space containment of the incumbents, the SCE
size resolution (paper formula x multipliers, memory cap) and the tuned-YAML
emitter; the smoke test runs the whole study loop twice (search, then resume)
on a synthetic Yambda-shaped dataset on CPU.
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
import yaml

from flowrec.ft_methods.sft.config import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yambda_smoke_assets import build_smoke_config_dir  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "exp2-tuned-params" / "tuning_sasrec_sce.py"
SPEC = importlib.util.spec_from_file_location("tuning_sasrec_sce", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


def _dm(batch_size: int, window: int) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(batch_size=batch_size, max_history_len=window, train_window_len=None))


# ---------------------------------------------------------------------------
# Search space and incumbents
# ---------------------------------------------------------------------------

def test_every_space_contains_the_exp1_baseline_with_sce_defaults() -> None:
    tuning.validate_specs()
    assert set(tuning.TUNING_SPECS) == set(tuning.DATASETS_TO_TUNE)
    baseline = {**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS}
    for spec in tuning.TUNING_SPECS.values():
        assert tuning.params_in_space(baseline, spec)
    # Trainer regime matches the full-CE tuning for the shared datasets.
    for name in ("beauty_2023", "yambda_50m_listens"):
        base = tuning.tuning.TUNING_SPECS[name]
        sce = tuning.TUNING_SPECS[name]
        assert (sce.grad_accum, sce.n_trials, sce.patience, sce.lr) == (base.grad_accum, base.n_trials, base.patience, base.lr)
    assert tuning.TUNING_SPECS["yambda_500m_listens"].n_trials < tuning.TUNING_SPECS["yambda_50m_listens"].n_trials


def test_incumbents_include_the_full_ce_tuned_configs_and_lie_in_the_space() -> None:
    for name in tuning.DATASETS_TO_TUNE:
        incumbents = tuning.incumbent_params(name)
        labels = [label for label, _ in incumbents]
        assert "exp1 baseline" in labels[0]
        source = tuning.CE_TUNED_PRIOR[name]
        if (tuning.CONFIG_DIR / f"sasrec_model_{source}.yaml").is_file():
            assert any(f"sasrec_model_{source}.yaml" in label for label in labels)
        for _, params in incumbents:
            assert set(tuning.ARCH_PARAM_NAMES) | set(tuning.SCE_PARAM_NAMES) <= set(params)
            assert tuning.params_in_space(params, tuning.TUNING_SPECS[name]), (name, params)
    assert tuning.CE_TUNED_PRIOR["yambda_500m_listens"] == "yambda_50m_listens"


def test_params_outside_the_space_are_rejected() -> None:
    spec = tuning.TUNING_SPECS["yambda_50m_listens"]
    good = {**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS}
    assert tuning.params_in_space(good, spec)
    assert not tuning.params_in_space({**good, "d_model": 48}, spec)
    assert not tuning.params_in_space({**good, "bucket_size_y": 100}, spec)
    assert not tuning.params_in_space({**good, "n_buckets_mult": 3.0}, spec)
    assert not tuning.params_in_space({k: v for k, v in good.items() if k != "mix_x"}, spec)


# ---------------------------------------------------------------------------
# SCE size resolution
# ---------------------------------------------------------------------------

def test_sce_sizes_follow_the_paper_formula_times_the_multipliers() -> None:
    assert tuning.sce_bucket_base(_dm(128, 50)) == 160
    assert tuning.sce_bucket_base(_dm(128, 20)) == 101
    cfg = tuning.resolve_sce_config(
        {"n_buckets_mult": 2.0, "bucket_size_x_mult": 0.5, "bucket_size_y": 512, "mix_x": True},
        bucket_base=160,
    )
    assert (cfg.n_buckets, cfg.bucket_size_x, cfg.bucket_size_y, cfg.mix_x) == (320, 80, 512, True)
    assert tuning.bucket_logits(cfg) == 320 * 80 * 512
    baseline = tuning.resolve_sce_config(tuning.SCE_BASELINE_PARAMS, bucket_base=101)
    assert (baseline.n_buckets, baseline.bucket_size_x, baseline.bucket_size_y) == (101, 101, 1024)
    # Every space admits the baseline combination and rejects the largest one.
    for spec in tuning.TUNING_SPECS.values():
        assert tuning.bucket_logits(baseline) <= spec.max_bucket_logits
        largest = tuning.resolve_sce_config(
            {"n_buckets_mult": max(spec.n_buckets_mult), "bucket_size_x_mult": max(spec.bucket_size_x_mult),
             "bucket_size_y": max(spec.bucket_size_y), "mix_x": False},
            bucket_base=160,
        )
        assert tuning.bucket_logits(largest) > spec.max_bucket_logits


# ---------------------------------------------------------------------------
# Objective robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (FloatingPointError("Non-finite loss at step 7: nan"), "non_finite_loss"),
        (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), "cuda_oom"),
    ],
)
def test_diverged_or_oom_trials_are_pruned_instead_of_killing_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
) -> None:
    import optuna

    spec = tuning.TUNING_SPECS["yambda_50m_listens"]
    rt = tuning.Runtime(checkpoint_root=tmp_path / "ckpt", artifacts_dir=tmp_path / "artifacts",
                        tuned_config_dir=tmp_path / "tuned", device="cpu", use_amp=False)

    def failing_train_once(**kwargs):
        raise error

    monkeypatch.setattr(tuning, "train_once", failing_train_once)
    objective = tuning.make_objective(
        dataset_name="yambda_50m_listens", dm=_dm(128, 50), dataset_dir=tmp_path / "ckpt" / "y",
        spec=spec, rt=rt, study_name="s", trials_planned=5,
    )
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS})
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned):
        objective(trial)
    assert trial.user_attrs["pruned_reason"] == reason
    assert trial.user_attrs["sce"] == {"n_buckets": 160, "bucket_size_x": 160, "bucket_size_y": 1024, "mix_x": False}
    # Unrelated errors still propagate (a genuine bug must not be swallowed).
    # Enqueue the baseline again so the sampled trial passes the bucket cap
    # and actually reaches train_once.
    monkeypatch.setattr(tuning, "train_once", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    study.enqueue_trial({**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS})
    with pytest.raises(RuntimeError, match="boom"):
        objective(study.ask())


# ---------------------------------------------------------------------------
# Tuned YAML
# ---------------------------------------------------------------------------

def test_tuned_yaml_round_trips_into_an_sce_sft_config(tmp_path: Path) -> None:
    rt = tuning.Runtime(tuned_config_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
    hparams = {**tuning.tuning.EXP1_BASELINE_PARAMS, **tuning.SCE_BASELINE_PARAMS, "lr": 1e-5}
    sce = tuning.resolve_sce_config(hparams, bucket_base=160).to_dict()
    path = tuning.write_tuned_config(
        dataset_name="yambda_50m_listens", hparams=hparams, sce=sce, bucket_base=160,
        best_value=0.1, best_trial_number=3, best_epoch=7, trials_completed=10, trials_planned=50,
        study_name="s", spec=tuning.TUNING_SPECS["yambda_50m_listens"], rt=rt,
    )
    assert path == tmp_path / "sasrec_sce_model_yambda_50m_listens.yaml"
    cfg = yaml.safe_load(path.read_text())
    for key in ("d_model", "nhead", "num_layers", "dropout", "pos_drop", "lr", "weight_decay",
                "betas", "eps", "max_grad_norm", "gradient_accumulation_steps", "sft"):
        assert key in cfg, key
    assert cfg["lr"] == pytest.approx(1e-5) and isinstance(cfg["lr"], float)
    assert cfg["gradient_accumulation_steps"] == tuning.TUNING_SPECS["yambda_50m_listens"].grad_accum
    sft = SFTConfig.from_dict(cfg["sft"])
    assert sft.item_loss == "scalable_cross_entropy"
    assert sft.scalable_cross_entropy.to_dict() == {"n_buckets": 160, "bucket_size_x": 160, "bucket_size_y": 1024, "mix_x": False}
    assert sft.item_topk == (1, 5, 10, 20)


# ---------------------------------------------------------------------------
# End-to-end smoke run on synthetic data
# ---------------------------------------------------------------------------

def test_smoke_study_writes_checkpoint_artifacts_and_tuned_yaml_then_resumes(tmp_path: Path) -> None:
    config_dir = build_smoke_config_dir(tmp_path)
    ckpt_root, artifacts, tuned_dir = tmp_path / "ckpt", tmp_path / "artifacts", tmp_path / "tuned"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--datasets", "yambda_smoke", "--spec", "yambda_50m_listens",
        "--config-dir", str(config_dir), "--checkpoint-root", str(ckpt_root),
        "--artifacts-dir", str(artifacts), "--tuned-config-dir", str(tuned_dir),
        "--n-trials", "2", "--max-epochs", "2", "--train-batches-per-epoch", "3",
        "--max-eval-batches", "2", "--num-workers", "0", "--device", "cpu", "--no-progress-bar",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    # Without a full-CE tuned config in --tuned-config-dir only the exp1
    # incumbent is enqueued; the second trial is sampled.
    assert "Enqueued incumbent: exp1 baseline" in proc.stdout
    assert "SCE bucket base int(2*sqrt(32 x 50)) = 80" in proc.stdout

    dataset_dir = ckpt_root / "yambda_smoke"
    for name in ("optuna_journal.log", "best_model.pt", "best_trial.json", "all_trials.json",
                 tuning.EPOCH_LOG_FILENAME):
        assert (dataset_dir / name).exists(), name
    assert not (dataset_dir / "trials").exists() or not any((dataset_dir / "trials").iterdir())

    best = json.loads((artifacts / "sasrec_sce_yambda_smoke_best_trial.json").read_text())
    assert best == json.loads((dataset_dir / "best_trial.json").read_text())
    assert best["item_loss"] == "scalable_cross_entropy"
    assert best["trials_completed"] == 2 and best["trials_planned"] == 2
    assert best["sce_bucket_base"] == 80
    assert set(best["scalable_cross_entropy"]) == {"n_buckets", "bucket_size_x", "bucket_size_y", "mix_x"}
    assert best["best_value_val_ndcg10"] >= 0.0 and best["best_epoch"] in (0, 1)
    assert Path(best["checkpoint"]) == dataset_dir / "best_model.pt"

    with (artifacts / "sasrec_sce_yambda_smoke_trials.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(reader.fieldnames) == len(set(reader.fieldnames)), reader.fieldnames
    assert {row["trial"] for row in rows} == {"0", "1"}
    assert all(row["state"] in {"COMPLETE", "PRUNED"} for row in rows)
    assert rows[0]["n_buckets"] == "80" and rows[0]["bucket_size_x"] == "80"

    tuned = yaml.safe_load((tuned_dir / "sasrec_sce_model_yambda_smoke.yaml").read_text())
    sft = SFTConfig.from_dict(tuned["sft"])
    assert sft.item_loss == "scalable_cross_entropy"
    assert sft.scalable_cross_entropy.to_dict() == best["scalable_cross_entropy"]
    assert tuned["d_model"] == best["best_params"]["d_model"]

    # A second invocation resumes the finished study without training again.
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "Resuming study: 2/2 finished trials." in proc.stdout
    assert "trial 2" not in proc.stdout
    assert json.loads((artifacts / "sasrec_sce_yambda_smoke_best_trial.json").read_text())["best_trial_number"] == best["best_trial_number"]

    # --rerank retrains the top configs with three seeds and rewrites every
    # best-trial output with the mean-selected winner.
    proc = subprocess.run(cmd + ["--rerank", "--rerank-top-k", "2"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    rerank = json.loads((artifacts / "sasrec_sce_yambda_smoke_rerank_summary.json").read_text())
    assert rerank == json.loads((dataset_dir / "rerank_summary.json").read_text())
    assert rerank["rerank_seeds"] == list(tuning.RERANK_SEEDS)
    assert 1 <= len(rerank["rerank_ranking"]) <= 2
    assert all(len(r["seed_results"]) == 3 for r in rerank["rerank_ranking"])
    reranked_best = json.loads((artifacts / "sasrec_sce_yambda_smoke_best_trial.json").read_text())
    assert "re-rank" in reranked_best["selection"]
    assert reranked_best["best_trial_number"] == rerank["best_trial_number"]
    assert (dataset_dir / "best_model.pt").exists() and not (dataset_dir / "rerank").exists()
    tuned = yaml.safe_load((tuned_dir / "sasrec_sce_model_yambda_smoke.yaml").read_text())
    assert SFTConfig.from_dict(tuned["sft"]).scalable_cross_entropy.to_dict() == reranked_best["scalable_cross_entropy"]
