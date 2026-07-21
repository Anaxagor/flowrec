#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_BASE_CONFIGS = {
    "beauty": PROJECT_ROOT / "configs/sweeps/sft_loss_stage2_20260617/beauty/b_sftloss_s2_09_h128_lr5e4_bs128_ga1_gc05.json",
    "mi": PROJECT_ROOT / "configs/sweeps/sft_loss_stage2_20260617/mi/m_sftloss_s2_23_h192_lr5e4_bs512_ga1_gc1.json",
}

ANCHORS = {
    "beauty": [
        {"embedding_dim": 128, "lr": 3e-4, "train_batch_size": 128, "gradient_accumulation_steps": 4, "max_grad_norm": 1.0},
        {"embedding_dim": 128, "lr": 5e-4, "train_batch_size": 128, "gradient_accumulation_steps": 1, "max_grad_norm": 0.5},
        {"embedding_dim": 192, "lr": 5e-4, "train_batch_size": 512, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0},
    ],
    "mi": [
        {"embedding_dim": 128, "lr": 3e-4, "train_batch_size": 128, "gradient_accumulation_steps": 4, "max_grad_norm": 1.0},
        {"embedding_dim": 192, "lr": 5e-4, "train_batch_size": 512, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0},
        {"embedding_dim": 128, "lr": 5e-4, "train_batch_size": 128, "gradient_accumulation_steps": 1, "max_grad_norm": 0.5},
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one source-style SFT TIGER Optuna trial.")
    p.add_argument("--dataset", choices=sorted(DATASET_BASE_CONFIGS), required=True)
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////path/study.db")
    p.add_argument("--trials-dir", type=Path, default=PROJECT_ROOT / "configs/sweeps/sft_optuna_sourcepad_20260624")
    p.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "experiments/tiger/sft_optuna_sourcepad_20260624")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--n-startup-trials", type=int, default=15)
    p.add_argument("--init-study", action="store_true", help="Create study and enqueue anchors, then exit.")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def storage(url: str) -> optuna.storages.RDBStorage:
    return optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def create_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        group=True,
    )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage(args.storage),
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_anchors(study: optuna.Study, dataset: str) -> None:
    existing = study.get_trials(deepcopy=False)
    if existing:
        print(f"Study already has {len(existing)} trials; not enqueueing anchors again.", flush=True)
        return
    for params in ANCHORS[dataset]:
        study.enqueue_trial(params)
    print(f"Enqueued {len(ANCHORS[dataset])} anchor trials for {dataset}.", flush=True)


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 160, 192, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    train_batch_size = trial.suggest_categorical("train_batch_size", [128, 256, 512])
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0, 2.0])
    if train_batch_size * grad_accum > 1024:
        raise optuna.TrialPruned(f"effective batch too large: {train_batch_size}x{grad_accum}")
    return {
        "embedding_dim": int(embedding_dim),
        "dim_feedforward": int(embedding_dim) * 4,
        "lr": float(lr),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "max_grad_norm": float(max_grad_norm),
    }


def lr_tag(lr: float) -> str:
    text = f"{lr:.2e}".replace(".00", "")
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def run_name(dataset: str, trial_number: int, params: dict[str, Any]) -> str:
    prefix = "b" if dataset == "beauty" else "m"
    gc = str(params["max_grad_norm"]).replace(".", "p")
    return (
        f"{prefix}_optuna_srcpad_t{trial_number:03d}"
        f"_h{params['embedding_dim']}"
        f"_lr{lr_tag(params['lr'])}"
        f"_bs{params['train_batch_size']}"
        f"_ga{params['gradient_accumulation_steps']}"
        f"_gc{gc}"
    )


def build_trial_config(args: argparse.Namespace, trial: optuna.Trial, params: dict[str, Any]) -> tuple[Path, str]:
    base = load_json(DATASET_BASE_CONFIGS[args.dataset])
    name = run_name(args.dataset, trial.number, params)
    base["experiment_name"] = name
    base["seed"] = int(args.seed + trial.number)

    dataset_cfg = base.setdefault("dataset", {})
    dataset_cfg.update(
        {
            "split_strategy": "leave_one_out",
            "train_all_prefixes": True,
            "train_target_policy": "all",
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "max_history_len": 20,
            "history_padding": "left",
            "filter_cold_items_after_split": False,
            "global_history_policy": "rolling",
            "global_eval_target_mode": "last_per_user",
            "offset_semantic_ids_by_one": True,
        }
    )

    model_cfg = base.setdefault("model", {})
    model_cfg.update(
        {
            "embedding_dim": params["embedding_dim"],
            "dim_feedforward": params["dim_feedforward"],
            "num_heads": 6,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "d_kv": 32,
            "dropout": 0.1,
            "num_beams": 100,
            "top_k": 20,
        }
    )

    dataloader_cfg = base.setdefault("dataloader", {})
    dataloader_cfg.update(
        {
            "train_batch_size": params["train_batch_size"],
            "validation_batch_size": 128,
            "train_drop_last": False,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
        }
    )

    base.setdefault("method", {})["name"] = "sft"
    optimizer_cfg = base.setdefault("optimizer", {})
    optimizer_cfg.update({"lr": params["lr"], "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8})

    training_cfg = base.setdefault("training", {})
    training_cfg.update(
        {
            "max_epochs": 220,
            "early_stopping_patience": 20,
            "early_stopping_mode": "best",
            "monitor": "val/sft/ndcg@10",
            "monitor_mode": "max",
            "validate_every_epochs": 2,
            "validate_every_steps": None,
            "evaluate_generation": True,
            "generation_num_beams": 10,
            "generation_num_return_sequences": 10,
            "generation_topk": [5, 10],
            "generation_constrain_valid_sequences": True,
            "test_evaluate_generation": True,
            "test_generation_num_beams": 100,
            "test_generation_num_return_sequences": 20,
            "test_generation_topk": [5, 10, 20],
            "val_sample_fraction": 0.25,
            "val_sample_seed": int(args.seed + trial.number * 17),
            "val_sample_resample": False,
            "gradient_accumulation_steps": params["gradient_accumulation_steps"],
            "max_grad_norm": params["max_grad_norm"],
            "log_every_steps": 200,
            "use_amp": True,
            "amp_dtype": "float16",
            "save_best": True,
            "save_last": True,
            "scheduler_step_on": "none",
            "min_delta": 0.0,
        }
    )

    trial_cfg_path = args.trials_dir / args.dataset / f"{name}.json"
    write_json(trial_cfg_path, base)
    return trial_cfg_path, name


def read_objective(run_dir: Path) -> tuple[float, dict[str, Any]]:
    summary_path = run_dir / "final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final summary: {summary_path}")
    summary = load_json(summary_path)
    value = summary.get("best_metric")
    if value is None:
        value = summary.get("validation_metrics", {}).get("val/sft/ndcg@10")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Invalid objective value in {summary_path}: {value!r}")
    return float(value), summary


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    params = sample_params(trial)
    cfg_path, name = build_trial_config(args, trial, params)
    exp_root = args.experiments_dir / args.dataset
    run_dir = exp_root / name
    trial.set_user_attr("config_path", str(cfg_path))
    trial.set_user_attr("run_dir", str(run_dir))
    for key, value in params.items():
        trial.set_user_attr(key, value)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_tiger.py"),
        "--config",
        str(cfg_path),
        "--experiments-dir",
        str(exp_root),
        "--run-name",
        name,
        "--plain-run-dir",
        "--device",
        args.device,
    ]
    print("Running", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    value, summary = read_objective(run_dir)
    trial.set_user_attr("best_epoch", summary.get("best_epoch"))
    trial.set_user_attr("epochs_completed", summary.get("epochs_completed"))
    for key in ("test/sft/ndcg@10", "test/sft/ndcg@5", "test/sft/ndcg@20", "test/sft/hr@10", "test/sft/generation_coverage@10"):
        if key in summary.get("test_metrics", {}):
            trial.set_user_attr(key, summary["test_metrics"][key])
    return value


def main() -> None:
    args = parse_args()
    study = create_study(args)
    if args.init_study:
        enqueue_anchors(study, args.dataset)
        return
    study.optimize(lambda trial: objective(trial, args), n_trials=1, gc_after_trial=True)
    print(f"Best value so far: {study.best_value:.6f}", flush=True)
    print(f"Best params so far: {study.best_params}", flush=True)


if __name__ == "__main__":
    main()
