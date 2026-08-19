"""Tests for the exp2 TIGER grid tuning script (tuning_tiger_yambda_500m.py).

Unit tests cover the plan construction (incumbent-centred grids, OFAT vs
full designs, canonical trial names, combined-trial derivation) and the
tuned-YAML emitter; the smoke test runs the whole worker loop on a synthetic
Yambda-shaped dataset on CPU (two epochs of three batches per trial).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "exp2-tuned-params" / "tuning_tiger_yambda_500m.py"
SPEC = importlib.util.spec_from_file_location("tuning_tiger_yambda_500m", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tuning = importlib.util.module_from_spec(SPEC)
# The dataclasses in the script resolve their module through sys.modules.
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _incumbent() -> "tuning.TrialConfig":
    return tuning.TrialConfig(
        model={
            "embedding_dim": 128, "num_heads": 6, "num_encoder_layers": 2, "num_decoder_layers": 2,
            "dim_feedforward": 512, "d_kv": 32, "dropout": 0.1, "user_ids_count": 1,
            "codebook_size": 256, "gradient_checkpointing": False, "eval_num_beams": 20,
            "val_num_beams": 1, "constrain_valid_sequences": True, "ks": [5, 10, 20],
            "sft": {"compute_train_metrics": True, "generation_topk": [1]},
        },
        trainer={"lr": 0.0018511868508356536, "weight_decay": 1e-2, "max_grad_norm": 2.0,
                 "betas": [0.9, 0.999], "eps": 1e-8, "amp_dtype": "float16", "use_amp": True},
        batch_size=2048,
        grad_accum=1,
    )


def test_grids_always_contain_the_incumbent_first() -> None:
    inc = _incumbent()
    grids = tuning.resolve_grids(["lr", "dropout", "embedding_dim"], inc, {"lr": (1e-3, 0.0018511868508356536, 3e-3)})
    assert grids["lr"][0] == pytest.approx(inc.trainer["lr"])
    assert grids["lr"] == pytest.approx([0.0018511868508356536, 1e-3, 3e-3])  # incumbent deduplicated
    assert grids["dropout"] == [0.1, 0.0, 0.2]
    assert grids["embedding_dim"] == [128, 256]


def test_ofat_plan_is_baseline_plus_single_deviations() -> None:
    inc = _incumbent()
    params = list(tuning.DEFAULT_PARAMS)
    grids = tuning.resolve_grids(params, inc, {})
    plan = tuning.build_plan(design="ofat", params=params, grids=grids, incumbent=inc, extra_trials=[])
    names = [t.name for t in plan]
    assert names[0] == "baseline"
    assert len(plan) == 1 + sum(len(grids[p]) - 1 for p in params) == 10
    assert "lr-0.001" in names and "embedding_dim-256" in names and "num_layers-4" in names
    assert all(len(t.overrides) <= 1 for t in plan)


def test_full_plan_uses_canonical_names_shared_with_ofat() -> None:
    inc = _incumbent()
    grids = tuning.resolve_grids(["lr", "dropout"], inc, {"lr": (3e-3,), "dropout": (0.0,)})
    plan = tuning.build_plan(
        design="full", params=["lr", "dropout"], grids=grids, incumbent=inc,
        extra_trials=[{"lr": 1e-3, "dropout": 0.1}],  # dropout at the incumbent value is dropped from the name
    )
    assert [t.name for t in plan] == ["baseline", "dropout-0", "lr-0.003", "lr-0.003__dropout-0", "lr-0.001"]
    assert plan[-1].role == "extra"


def test_trial_config_applies_tied_architecture_axes() -> None:
    inc = _incumbent()
    cfg = tuning.Trial({"embedding_dim": 256, "num_layers": 4, "lr": 3e-3}).config(inc)
    assert cfg.model["embedding_dim"] == 256 and cfg.model["dim_feedforward"] == 1024
    assert cfg.model["num_encoder_layers"] == cfg.model["num_decoder_layers"] == 4
    assert cfg.trainer["lr"] == pytest.approx(3e-3)
    # The incumbent is untouched.
    assert inc.model["embedding_dim"] == 128 and inc.trainer["lr"] == pytest.approx(0.0018511868508356536)


def test_combined_trial_merges_winners_above_the_margin() -> None:
    inc = _incumbent()
    params = ["lr", "dropout", "d_kv"]
    grids = tuning.resolve_grids(params, inc, {"lr": (3e-3,), "dropout": (0.0,), "d_kv": (64,)})

    def result(name: str, value: float, overrides: dict) -> dict:
        return {"trial": name, "status": "complete", "best_value": value, "overrides": overrides}

    results = {
        "baseline": result("baseline", 0.100, {}),
        "lr-0.003": result("lr-0.003", 0.104, {"lr": 3e-3}),          # +4%  -> in
        "dropout-0": result("dropout-0", 0.1005, {"dropout": 0.0}),  # +0.5% -> below the 1% margin
        "d_kv-64": result("d_kv-64", 0.090, {"d_kv": 64}),           # worse -> out
    }
    effects = tuning.parameter_effects(params=params, grids=grids, results=results, incumbent=inc)
    assert effects["lr"]["winner_value"] == pytest.approx(3e-3)
    assert effects["dropout"]["winner_value"] == pytest.approx(0.0)
    assert effects["d_kv"]["winner_value"] == 32
    combined = tuning.derive_combined_trial(effects, incumbent=inc, min_rel_improvement=0.01)
    assert combined is not None and combined.overrides == {"lr": 3e-3} and combined.role == "combined"
    combined = tuning.derive_combined_trial(effects, incumbent=inc, min_rel_improvement=0.0)
    assert combined is not None and combined.name == "lr-0.003__dropout-0"


def test_yaml_block_round_trips_types() -> None:
    text = tuning._yaml_block({"lr": 1.85e-3, "eps": 1e-8, "ks": [5, 10, 20], "flag": True,
                               "sft": {"generation_topk": [1], "name": "val/sft/ndcg@1"}})
    loaded = yaml.safe_load(text)
    assert loaded == {"lr": 1.85e-3, "eps": 1e-8, "ks": [5, 10, 20], "flag": True,
                      "sft": {"generation_topk": [1], "name": "val/sft/ndcg@1"}}
    assert "{" not in text  # block style, no flow mappings


# ---------------------------------------------------------------------------
# End-to-end smoke run on synthetic data
# ---------------------------------------------------------------------------

def _build_smoke_assets(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    rows = []
    n_items = 300
    for user in range(400):
        n = int(rng.integers(12, 60))
        t0 = 1_700_000_000 + int(rng.integers(0, 10_000))
        for i in range(n):
            rows.append({
                "uid": user, "item_id": int(rng.zipf(1.3) % n_items), "timestamp": t0 + i * 60,
                "event_type": "listen", "is_organic": int(rng.integers(0, 2)),
                "played_ratio_pct": int(rng.integers(30, 101)), "track_length_seconds": 200,
            })
    interactions = root / "multi_event.parquet"
    pd.DataFrame(rows).to_parquet(interactions)
    embeddings = root / "embeddings.parquet"
    pd.DataFrame({"item_id": list(range(n_items)),
                  "embed": [rng.random(4).tolist() for _ in range(n_items)]}).to_parquet(embeddings)

    sid_dir = root / "semantic_ids" / "yambda_smoke"
    sid_dir.mkdir(parents=True)
    sids, seen = {}, {}
    for item in range(n_items):
        code = (int(rng.integers(0, 256)), 256 + int(rng.integers(0, 256)), 512 + int(rng.integers(0, 256)))
        rank = seen.get(code, 0)
        seen[code] = rank + 1
        sids[str(item)] = [*code, 768 + rank]
    (sid_dir / "semantic_ids.json").write_text(json.dumps(sids), encoding="utf-8")

    cfg = yaml.safe_load((PROJECT_ROOT / "configs/exp2-tuned-params/dataset_yambda.yaml").read_text())
    cfg["datasets"] = {"yambda_smoke": {"dataset_size": "50m", "interactions_path": str(interactions)}}
    cfg["embeddings_path"] = str(embeddings)
    cfg["hf_recent_months"] = None
    cfg["tiger"] = {"train_target_policy": "last_n", "train_last_n": 16, "batch_size": 32,
                    "eval_batch_size": 32, "test_eval_batch_size": 8, "num_workers": 0,
                    "pin_memory": False, "persistent_workers": False}
    dataset_yaml = root / "dataset_yambda_smoke.yaml"
    dataset_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"dataset_yaml": dataset_yaml, "semantic_ids_dir": root / "semantic_ids"}


def test_smoke_sweep_writes_results_summary_and_tuned_yaml(tmp_path: Path) -> None:
    assets = _build_smoke_assets(tmp_path)
    ckpt_root, artifacts, tuned = tmp_path / "ckpt", tmp_path / "artifacts", tmp_path / "tuned.yaml"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", "yambda_smoke",
        "--dataset-config", str(assets["dataset_yaml"]),
        "--semantic-ids-dir", str(assets["semantic_ids_dir"]),
        "--checkpoint-root", str(ckpt_root), "--artifacts-dir", str(artifacts),
        "--tuned-config-path", str(tuned),
        "--epochs-per-trial", "2", "--train-batches-per-epoch", "3", "--max-eval-batches", "2",
        "--num-workers", "0", "--device", "cpu",
        "--params", "lr", "dropout", "--grid", "lr=3e-3", "--grid", "dropout=0.0",
        "--val-sample-fraction", "0.5", "--val-num-beams", "4", "--val-topk", "1", "4",
        "--monitor", "val/sft/ndcg@4", "--trial-hours-estimate", "0", "--fail-fast",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    trials_dir = ckpt_root / "yambda_smoke" / "trials"
    for name in ("baseline", "lr-0.003", "dropout-0"):
        result = json.loads((trials_dir / name / "result.json").read_text())
        assert result["status"] == "complete"
        assert result["epochs_completed"] == 2 and len(result["curve"]) == 2
        assert result["monitor"] == "val/sft/ndcg@4" and result["best_value"] is not None
        assert (trials_dir / name / "best.pt").exists()
        assert not (trials_dir / name / "CLAIMED").exists()
    assert json.loads((trials_dir / "lr-0.003" / "result.json").read_text())["hparams"]["lr"] == pytest.approx(3e-3)

    summary = json.loads((ckpt_root / "yambda_smoke" / "tuning_summary.json").read_text())
    assert summary["trials_complete"] == 3 and summary["winner"] in {"baseline", "lr-0.003", "dropout-0"}
    assert set(summary["effects"]) == {"lr", "dropout"}
    assert (ckpt_root / "yambda_smoke" / "best_trial.json").exists()

    with (artifacts / "tiger_yambda_smoke_tuning_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["trial"] for row in rows} == {"baseline", "lr-0.003", "dropout-0"}

    tuned_cfg = yaml.safe_load(tuned.read_text())
    for key in ("embedding_dim", "num_heads", "num_encoder_layers", "num_decoder_layers", "dim_feedforward",
                "d_kv", "dropout", "lr", "weight_decay", "betas", "eps", "max_grad_norm",
                "gradient_accumulation_steps", "batch_size", "sft"):
        assert key in tuned_cfg, key
    winner = summary["winner"]
    assert tuned_cfg["lr"] == pytest.approx(3e-3 if winner == "lr-0.003" else 0.0018511868508356536)

    # A second invocation finds nothing to claim and must not reload the data.
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "skipping the datamodule setup" in proc.stdout
