#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_BASE_CONFIGS = {
    "beauty": PROJECT_ROOT
    / "configs/sweeps/sft_loss_stage2_20260617/beauty/b_sftloss_s2_09_h128_lr5e4_bs128_ga1_gc05.json",
    "mi": PROJECT_ROOT
    / "configs/sweeps/sft_loss_stage2_20260617/mi/m_sftloss_s2_23_h192_lr5e4_bs512_ga1_gc1.json",
}

ANCHORS = {
    "beauty": [
        {
            "embedding_dim": 128,
            "lr": 3e-4,
            "batch_grad": "128x4",
            "max_grad_norm": 1.0,
        },
        {
            "embedding_dim": 128,
            "lr": 5e-4,
            "batch_grad": "128x1",
            "max_grad_norm": 0.5,
        },
        {
            "embedding_dim": 192,
            "lr": 5e-4,
            "batch_grad": "512x1",
            "max_grad_norm": 1.0,
        },
    ],
    "mi": [
        {
            "embedding_dim": 128,
            "lr": 3e-4,
            "batch_grad": "128x4",
            "max_grad_norm": 1.0,
        },
        {
            "embedding_dim": 192,
            "lr": 5e-4,
            "batch_grad": "512x1",
            "max_grad_norm": 1.0,
        },
        {
            "embedding_dim": 128,
            "lr": 5e-4,
            "batch_grad": "128x1",
            "max_grad_norm": 0.5,
        },
    ],
}

BATCH_GRAD_CHOICES = ["128x1", "128x2", "128x4", "256x1", "256x2", "256x4", "512x1", "512x2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one source-style SFT TIGER Optuna trial.")
    parser.add_argument("--dataset", choices=sorted(DATASET_BASE_CONFIGS), required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=PROJECT_ROOT / "configs/sweeps/sft_optuna_source_style_20260624",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/tiger/sft_optuna_source_style_20260624",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--n-startup-trials", type=int, default=15)
    parser.add_argument("--init-study", action="store_true")
    return parser.parse_args()


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        group=True,
    )
    storage = optuna.storages.RDBStorage(
        url=args.storage,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_anchors(study: optuna.Study, dataset: str) -> None:
    if study.get_trials(deepcopy=False):
        print("Study already has trials; not enqueueing anchors again.", flush=True)
        return
    for params in ANCHORS[dataset]:
        study.enqueue_trial(params)
    print(f"Enqueued {len(ANCHORS[dataset])} anchors for {dataset}.", flush=True)


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    embedding_dim = int(trial.suggest_categorical("embedding_dim", [128, 160, 192, 256]))
    lr = float(trial.suggest_float("lr", 1e-4, 1e-3, log=True))
    batch_grad = str(trial.suggest_categorical("batch_grad", BATCH_GRAD_CHOICES))
    max_grad_norm = float(trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0, 2.0]))
    batch_size, grad_accum = (int(part) for part in batch_grad.split("x", 1))
    return {
        "embedding_dim": embedding_dim,
        "dim_feedforward": embedding_dim * 4,
        "lr": lr,
        "batch_grad": batch_grad,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_grad_norm": max_grad_norm,
    }


def lr_tag(lr: float) -> str:
    return f"{lr:.2e}".replace(".00", "").replace("+", "").replace("-", "m").replace(".", "p")


def run_name(dataset: str, trial_number: int, params: dict[str, Any]) -> str:
    prefix = "b" if dataset == "beauty" else "m"
    gc = str(params["max_grad_norm"]).replace(".", "p")
    return (
        f"{prefix}_srcstyle_t{trial_number:03d}"
        f"_h{params['embedding_dim']}"
        f"_lr{lr_tag(params['lr'])}"
        f"_bg{params['batch_grad']}"
        f"_gc{gc}"
    )


def build_trial_config(args: argparse.Namespace, trial: optuna.Trial, params: dict[str, Any]) -> tuple[Path, str]:
    config = load_json(DATASET_BASE_CONFIGS[args.dataset])
    name = run_name(args.dataset, trial.number, params)
    config["experiment_name"] = name
    config["seed"] = int(args.seed + trial.number)

    config.setdefault("dataset", {}).update(
        {
            "split_strategy": "leave_one_out",
            "train_all_prefixes": True,
            "train_target_policy": "all",
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "max_history_len": 20,
            "deduplicate_consecutive": False,
            "history_padding": "left",
            "filter_cold_items_after_split": False,
            "offset_semantic_ids_by_one": True,
        }
    )

    config.setdefault("model", {}).update(
        {
            "embedding_dim": params["embedding_dim"],
            "dim_feedforward": params["dim_feedforward"],
            "num_heads": 6,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "d_kv": 32,
            "dropout": 0.1,
            "num_beams": 20,
            "top_k": 20,
        }
    )

    config.setdefault("dataloader", {}).update(
        {
            "train_batch_size": params["train_batch_size"],
            "validation_batch_size": 64,
            "train_drop_last": False,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": False,
        }
    )

    config.setdefault("method", {})["name"] = "sft"
    config.setdefault("optimizer", {}).update(
        {
            "lr": params["lr"],
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )

    config.setdefault("training", {}).update(
        {
            "max_epochs": 300,
            "early_stopping_patience": 40,
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
            "gradient_checkpointing": True,
            "log_every_steps": 200,
            "use_amp": True,
            "amp_dtype": "float16",
            "save_best": True,
            "save_last": True,
            "scheduler_step_on": "none",
            "min_delta": 0.0,
        }
    )

    config_path = args.trials_dir / args.dataset / f"{name}.json"
    write_json(config_path, config)
    return config_path, name


def read_objective(run_dir: Path) -> tuple[float, dict[str, Any]]:
    summary_path = run_dir / "final_summary.json"
    summary = load_json(summary_path)
    value = summary.get("best_metric")
    if value is None:
        value = summary.get("validation_metrics", {}).get("val/sft/ndcg@10")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Invalid objective value in {summary_path}: {value!r}")
    return float(value), summary


def main() -> None:
    args = parse_args()
    study = create_study(args)
    if args.init_study:
        with file_lock(args.lock_path):
            enqueue_anchors(study, args.dataset)
        return

    with file_lock(args.lock_path):
        trial = study.ask()
        params = sample_params(trial)
        config_path, name = build_trial_config(args, trial, params)
        run_dir = args.experiments_dir / args.dataset / name
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("run_dir", str(run_dir))
        for key, value in params.items():
            trial.set_user_attr(key, value)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_tiger.py"),
        "--config",
        str(config_path),
        "--experiments-dir",
        str(args.experiments_dir / args.dataset),
        "--run-name",
        name,
        "--plain-run-dir",
        "--device",
        args.device,
    ]
    print("Running", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        value, summary = read_objective(run_dir)
        with file_lock(args.lock_path):
            trial.set_user_attr("best_epoch", summary.get("best_epoch"))
            trial.set_user_attr("epochs_completed", summary.get("epochs_completed"))
            test_metrics = summary.get("test_metrics", {})
            for key in (
                "test/sft/ndcg@10",
                "test/sft/ndcg@5",
                "test/sft/ndcg@20",
                "test/sft/recall@10",
                "test/sft/generation_coverage@10",
            ):
                if key in test_metrics:
                    trial.set_user_attr(key, test_metrics[key])
            study.tell(trial, value)
    except Exception as exc:
        with file_lock(args.lock_path):
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise exc

    print(f"Trial {trial.number} value={value:.6f}", flush=True)
    print(f"Best value so far: {study.best_value:.6f}", flush=True)
    print(f"Best params so far: {study.best_params}", flush=True)


if __name__ == "__main__":
    main()
