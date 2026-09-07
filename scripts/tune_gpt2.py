from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_gpt2 import (  # noqa: E402
    GenerationMetricAdapter,
    SUPPORTED_DATASETS,
    build_data_config,
    build_data_module,
    build_experiment_config,
    build_method,
    build_model,
    build_semantic_popularity_lookup,
    build_test_dataloader,
    build_trainer_config,
    load_and_validate_semantic_metadata,
    load_yaml,
    load_yaml_optional,
    prepare_semantic_ids,
    resolve_path,
    restore_generation_config,
    seed_everything,
    override_generation_for_test,
    write_json,
)
from flowrec.training.loggers import to_jsonable  # noqa: E402
from flowrec.training.trainer import Trainer  # noqa: E402

MONITOR = "val/sft/ndcg@10"
EMBEDDING_DIMS = (128, 192, 256)
NUM_HEADS = (4, 8)
NUM_LAYERS = (3, 4, 6, 8)
FFN_MULTIPLIERS = (1, 2, 3, 4)
WEIGHT_DECAYS = (0.0, 1e-4, 1e-3, 1e-2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Tune FlowRec GPT-2 on one dataset and one fixed seed.")
    p.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="yambda_50m_listens",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config-dir", type=Path, default=Path("configs/exp1-standard_params"))
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/gpt2_tuning"))
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None, help="Optuna storage URL; default is SQLite in the study dir.")
    p.add_argument("--device", default=None, help="e.g. cuda or cuda:0")
    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument(
        "--timeout-hours",
        type=float,
        default=0.0,
        help="Optional Optuna timeout; <=0 disables it.",
    )
    p.add_argument("--final-test-only", action="store_true")
    p.add_argument("--skip-final-test", action="store_true")
    return p.parse_args()


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_dim": trial.suggest_categorical("embedding_dim", list(EMBEDDING_DIMS)),
        "num_heads": trial.suggest_categorical("num_heads", list(NUM_HEADS)),
        "num_layers": trial.suggest_categorical("num_layers", list(NUM_LAYERS)),
        "ffn_multiplier": trial.suggest_categorical("ffn_multiplier", list(FFN_MULTIPLIERS)),
        "dropout": trial.suggest_float("dropout", 0.0, 0.20),
        "lr": trial.suggest_float("lr", 2e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", list(WEIGHT_DECAYS)),
    }


def apply_trial_params(config: dict[str, Any], params: dict[str, Any]) -> None:
    d_model = int(params["embedding_dim"])
    num_heads = int(params["num_heads"])
    if d_model % num_heads != 0:
        raise ValueError(f"embedding_dim={d_model} must be divisible by num_heads={num_heads}.")

    dropout = float(params["dropout"])
    config["model"].update(
        {
            "embedding_dim": d_model,
            "num_heads": num_heads,
            "num_layers": int(params["num_layers"]),
            "dim_feedforward": d_model * int(params["ffn_multiplier"]),
            "dropout": dropout,
            "attention_dropout": dropout,
            "residual_dropout": dropout,
            "embedding_dropout": dropout,
        }
    )
    config["optimizer"]["lr"] = float(params["lr"])
    config["optimizer"]["weight_decay"] = float(params["weight_decay"])


def configure_training(config: dict[str, Any], max_epochs: int) -> None:
    config["training"].update(
        {
            "max_epochs": int(max_epochs),
            "validate_every_epochs": 3,
            "validate_every_steps": None,
            "monitor": MONITOR,
            "monitor_mode": "max",
            "save_best": True,
            "save_last": True,
            # Six validation events = up to 18 epochs without improvement.
            "early_stopping_patience": 6,
            "early_stopping_mode": "best",
            "min_delta": 1e-5,
        }
    )


def build_base_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_dir = resolve_path(args.config_dir, base=PROJECT_ROOT)
    model_config_path = resolve_path(
        args.model_config or (config_dir / "gpt2_model.yaml"), base=PROJECT_ROOT
    )
    config = build_experiment_config(
        dataset_name=args.dataset,
        seed=args.seed,
        model_yaml=load_yaml(model_config_path),
        amazon_yaml=load_yaml(config_dir / "dataset_amazon.yaml"),
        yambda_yaml=load_yaml(config_dir / "dataset_yambda.yaml"),
        trainer_yaml=load_yaml_optional(config_dir / "trainer.yaml"),
        method_override="sft",
    )
    configure_training(config, args.max_epochs)
    config["seed"] = int(args.seed)
    return config, model_config_path


def setup_data(config: dict[str, Any], model_config_path: Path):
    data = build_data_module(build_data_config(config, config_path=model_config_path))
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("GPT-2 tuning requires semantic IDs.")
    load_and_validate_semantic_metadata(data, config)
    prepare_semantic_ids(data, config)
    return data


def make_method(config: dict[str, Any], model, data, train_loader, model_config_path: Path):
    method = build_method(
        config, model, data=data, train_loader=train_loader, config_path=model_config_path
    )
    return GenerationMetricAdapter(
        method,
        popularity_by_semantic_id=build_semantic_popularity_lookup(data),
        ks=tuple(int(k) for k in config["training"]["generation_topk"]),
    )


def best_checkpoint(fit_result: Any, trial_dir: Path) -> Path:
    raw = getattr(fit_result, "best_checkpoint_path", None)
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    candidates = sorted((trial_dir / "checkpoints").glob("*best*.pt"))
    if not candidates:
        raise RuntimeError(f"No best checkpoint was saved in {trial_dir / 'checkpoints'}")
    return candidates[-1]


def baseline_params(config: dict[str, Any]) -> dict[str, Any] | None:
    model = config["model"]
    optimizer = config["optimizer"]
    d_model = int(model["embedding_dim"])
    d_ff = int(model["dim_feedforward"])
    if d_ff % d_model != 0:
        return None

    params = {
        "embedding_dim": d_model,
        "num_heads": int(model["num_heads"]),
        "num_layers": int(model["num_layers"]),
        "ffn_multiplier": d_ff // d_model,
        "dropout": float(model.get("dropout", 0.1)),
        "lr": float(optimizer.get("lr", 3e-4)),
        "weight_decay": float(optimizer.get("weight_decay", 0.0)),
    }
    if (
        params["embedding_dim"] not in EMBEDDING_DIMS
        or params["num_heads"] not in NUM_HEADS
        or params["num_layers"] not in NUM_LAYERS
        or params["ffn_multiplier"] not in FFN_MULTIPLIERS
        or params["weight_decay"] not in WEIGHT_DECAYS
        or not 0.0 <= params["dropout"] <= 0.20
        or not 2e-4 <= params["lr"] <= 5e-3
    ):
        return None
    return params


def run_trial(
    trial: optuna.Trial,
    *,
    base_config: dict[str, Any],
    data,
    model_config_path: Path,
    output_dir: Path,
    device: str | None,
    seed: int,
    dataset: str,
) -> float:
    seed_everything(seed)
    params = suggest_params(trial)
    config = json.loads(json.dumps(base_config))
    apply_trial_params(config, params)

    trial_dir = output_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json(trial_dir / "params.json", params)

    model = trainer = None
    try:
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        model = build_model(config, data)
        method = make_method(config, model, data, train_loader, model_config_path)
        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=build_trainer_config(config, trial_dir),
            device=device,
            lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            betas=tuple(config["optimizer"].get("betas", (0.9, 0.999))),
            eps=float(config["optimizer"].get("eps", 1e-8)),
        )

        started = time.perf_counter()
        fit_result = trainer.fit(train_loader, val_loader)
        elapsed = time.perf_counter() - started

        metric = trainer.state.best_metric
        if metric is None:
            raise RuntimeError(f"Trainer did not produce {MONITOR} for trial {trial.number}.")
        checkpoint = best_checkpoint(fit_result, trial_dir)

        best_epoch = None if trainer.state.best_epoch is None else int(trainer.state.best_epoch)
        trial.set_user_attr("dataset", dataset)
        trial.set_user_attr("seed", seed)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_checkpoint", str(checkpoint))
        trial.set_user_attr("elapsed_sec", float(elapsed))
        write_json(
            trial_dir / "result.json",
            {
                "trial": trial.number,
                "dataset": dataset,
                "seed": seed,
                "best_val_ndcg10": float(metric),
                "best_epoch": best_epoch,
                "best_checkpoint": str(checkpoint),
                "elapsed_sec": elapsed,
                "params": params,
            },
        )
        return float(metric)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"CUDA OOM: {exc}") from exc
        raise
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_final_test(
    study: optuna.Study,
    *,
    base_config: dict[str, Any],
    data,
    model_config_path: Path,
    output_dir: Path,
    device: str | None,
    seed: int,
    dataset: str,
) -> dict[str, Any]:
    best = study.best_trial
    checkpoint_raw = best.user_attrs.get("best_checkpoint")
    if not checkpoint_raw or not Path(checkpoint_raw).exists():
        raise FileNotFoundError(f"Best-trial checkpoint is missing: {checkpoint_raw}")

    seed_everything(seed)
    config = json.loads(json.dumps(base_config))
    apply_trial_params(config, dict(best.params))
    final_dir = output_dir / "best_trial_test"
    final_dir.mkdir(parents=True, exist_ok=True)

    model = trainer = None
    try:
        train_loader = data.train_dataloader()
        model = build_model(config, data)
        method = make_method(config, model, data, train_loader, model_config_path)
        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=build_trainer_config(config, final_dir),
            device=device,
            lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            betas=tuple(config["optimizer"].get("betas", (0.9, 0.999))),
            eps=float(config["optimizer"].get("eps", 1e-8)),
        )
        trainer.load_checkpoint(
            Path(checkpoint_raw),
            load_optimizer=False,
            load_scheduler=False,
        )

        training_cfg = config.get("training", {})
        active_generation_cfg = getattr(method, "cfg", None)

        test_generation_num_beams = training_cfg.get(
            "test_generation_num_beams",
            getattr(active_generation_cfg, "generation_num_beams", None),
        )
        test_generation_num_return_sequences = training_cfg.get(
            "test_generation_num_return_sequences",
            getattr(active_generation_cfg, "generation_num_return_sequences", None),
        )
        test_generation_topk = training_cfg.get(
            "test_generation_topk",
            getattr(active_generation_cfg, "generation_topk", None),
        )
        test_evaluate_generation = training_cfg.get(
            "test_evaluate_generation",
            getattr(active_generation_cfg, "evaluate_generation", None),
        )

        old_generation_cfg = override_generation_for_test(
            method,
            evaluate_generation=test_evaluate_generation,
            num_beams=test_generation_num_beams,
            num_return_sequences=test_generation_num_return_sequences,
            topk=test_generation_topk,
        )
        try:
            test_metrics = trainer.test(build_test_dataloader(data, config))
        finally:
            restore_generation_config(method, old_generation_cfg)

        result = {
            "dataset": dataset,
            "seed": seed,
            "best_trial": best.number,
            "best_val_ndcg10": float(best.value),
            "best_epoch": best.user_attrs.get("best_epoch"),
            "best_checkpoint": checkpoint_raw,
            "best_params": dict(best.params),
            "test_metrics": to_jsonable(test_metrics),
        }
        write_json(output_dir / "best_result.json", result)
        return result
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    study_name = args.study_name or f"gpt2_{args.dataset}_seed{args.seed}"
    output_dir = resolve_path(args.output_dir, base=PROJECT_ROOT) / args.dataset / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config, model_config_path = build_base_config(args)
    data = setup_data(base_config, model_config_path)

    storage = args.storage or f"sqlite:///{(output_dir / 'study.db').resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=int(args.sampler_seed), n_startup_trials=4, multivariate=True
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Evaluate the dataset's current YAML configuration once as a reference,
    # provided it lies inside this tuning search space.
    if len(study.trials) == 0:
        reference = baseline_params(base_config)
        if reference is not None:
            study.enqueue_trial(reference)

    if not args.final_test_only and args.n_trials > 0:
        timeout = None if args.timeout_hours <= 0 else args.timeout_hours * 3600.0
        study.optimize(
            lambda trial: run_trial(
                trial,
                base_config=base_config,
                data=data,
                model_config_path=model_config_path,
                output_dir=output_dir,
                device=args.device,
                seed=int(args.seed),
                dataset=args.dataset,
            ),
            n_trials=args.n_trials,
            timeout=timeout,
            gc_after_trial=True,
        )

    if args.skip_final_test:
        return
    if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        raise RuntimeError("No completed Optuna trials are available for final test.")

    result = run_final_test(
        study,
        base_config=base_config,
        data=data,
        model_config_path=model_config_path,
        output_dir=output_dir,
        device=args.device,
        seed=int(args.seed),
        dataset=args.dataset,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
