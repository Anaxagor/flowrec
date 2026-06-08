from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.loggers import (
    CompositeExperimentLogger,
    CometExperimentLogger,
    JsonlExperimentLogger,
    to_jsonable,
)
from flowrec.training.trainer import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.run_sasrec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec with FlowRec.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sasrec_beauty.json",
        help="Path to a JSON experiment config.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "sasrec",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_json(config_path)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.max_epochs
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    run_name = args.run_name or str(config.get("experiment_name", "sasrec"))
    run_dir = create_run_dir(args.experiments_dir, run_name, config)
    setup_logging(run_dir)

    data_cfg = build_data_config(config, config_path=config_path)
    data = AmazonReviewDataModule(data_cfg)
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = build_model(config, data)
    method = build_method(config)
    trainer_cfg = build_trainer_config(config, run_dir)

    resolved = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "data": asdict(data_cfg),
        "data_summary": data.summary(),
        "model": model.config_dict(),
        "method": method.cfg.to_dict(),
        "trainer": trainer_cfg.to_dict(),
        "optimizer": config.get("optimizer", {}),
    }
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "cli_args.json", vars(args))

    logger = build_experiment_logger(
        config,
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_path,
        resolved_config=resolved,
        cli_args=vars(args),
    )
    try:
        LOGGER.info("Run directory: %s", run_dir)
        LOGGER.info("Config: %s", config_path)
        LOGGER.info("Data summary: %s", json.dumps(data.summary(), sort_keys=True))
        LOGGER.info(
            "Model parameters: total=%s trainable=%s",
            f"{model.num_parameters():,}",
            f"{model.num_parameters(trainable_only=True):,}",
        )

        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=trainer_cfg,
            device=args.device,
            lr=float(config.get("optimizer", {}).get("lr", 3e-4)),
            weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.0)),
            betas=tuple(config.get("optimizer", {}).get("betas", (0.9, 0.999))),
            eps=float(config.get("optimizer", {}).get("eps", 1e-8)),
            logger=logger,
        )

        fit_result = trainer.fit(train_loader, val_loader)
        test_metrics: dict[str, float] = {}
        checkpoint_for_test = fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        if checkpoint_for_test is not None:
            trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)
            test_metrics = trainer.test(test_loader)

        summary = {
            "dataset": {**data_config_paths(data_cfg), **data.summary()},
            "model": model.config_dict(),
            "epochs_completed": int(trainer.state.epoch + 1),
            "steps_completed": int(trainer.state.global_step),
            "best_metric": trainer.state.best_metric,
            "best_epoch": trainer.state.best_epoch,
            "validation_metrics": fit_result.val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": fit_result.best_checkpoint_path,
            "last_checkpoint": fit_result.last_checkpoint_path,
            "experiment_dir": str(run_dir),
        }
        write_json(run_dir / "final_summary.json", summary)
        write_text_summary(run_dir / "final_summary.md", summary)
        logger({"type": "final_summary", "global_step": int(trainer.state.global_step), **summary_to_metrics(summary)})
        for name in ("resolved_config.json", "cli_args.json", "final_summary.json", "final_summary.md"):
            logger.log_asset(run_dir / name, file_name=name, step=int(trainer.state.global_step))
        LOGGER.info("Training finished. Final summary: %s", json.dumps(summary, sort_keys=True))
    finally:
        logger.close()


def build_data_config(config: Mapping[str, Any], *, config_path: Path) -> AmazonReviewDataConfig:
    data = dict(config.get("data", {}))
    dataset = dict(config.get("dataset", {}))
    dataloader = dict(config.get("dataloader", {}))
    base = config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT
    max_sequence_length = int(dataset.get("max_sequence_length", 20))
    preprocessing_preset = dataset.get("preprocessing_preset", "default")

    return AmazonReviewDataConfig(
        reviews_path=resolve_path(data["reviews_path"], base=base),
        metadata_path=resolve_optional_path(data.get("metadata_path"), base=base),
        semantic_id_path=resolve_optional_path(data.get("semantic_id_path"), base=base),
        item_key=dataset.get("item_key", "asin"),
        preprocessing_preset=preprocessing_preset,
        min_user_interactions=int(dataset.get("min_user_interactions", 5)),
        min_item_interactions=int(dataset.get("min_item_interactions", 5)),
        use_verified_only=bool(dataset.get("use_verified_only", False)),
        deduplicate_consecutive=bool(dataset.get("deduplicate_consecutive", False)),
        deduplicate_user_items=bool(dataset.get("deduplicate_user_items", False)),
        deduplicate_user_items_keep=dataset.get("deduplicate_user_items_keep", "first"),
        split_strategy=dataset.get("split_strategy", "leave_one_out"),
        min_history_len=int(dataset.get("min_history_len", 1)),
        train_all_prefixes=bool(dataset.get("train_all_prefixes", False)),
        train_target_policy=dataset.get("train_target_policy", "last_n"),
        train_targets_per_user=int(dataset.get("train_targets_per_user", 256)),
        train_stride=int(dataset.get("train_stride", 10)),
        train_last_n=int(dataset.get("train_last_n", 20)),
        train_hybrid_recent_n=int(dataset.get("train_hybrid_recent_n", 256)),
        train_hybrid_random_n=int(dataset.get("train_hybrid_random_n", 256)),
        global_val_timestamp=(
            int(dataset["global_val_timestamp"])
            if dataset.get("global_val_timestamp") is not None
            else None
        ),
        global_test_timestamp=(
            int(dataset["global_test_timestamp"])
            if dataset.get("global_test_timestamp") is not None
            else None
        ),
        global_val_ratio=float(dataset.get("global_val_ratio", 0.1)),
        global_test_ratio=float(dataset.get("global_test_ratio", 0.1)),
        global_history_policy=dataset.get("global_history_policy", "rolling"),
        global_eval_target_mode=dataset.get("global_eval_target_mode", "last_per_user"),
        max_history_len=int(dataset.get("max_history_len", max(1, max_sequence_length - 1))),
        history_padding=dataset.get("history_padding", "right"),
        batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 256))),
        eval_batch_size=int(
            dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 256))
        ),
        num_workers=int(dataloader.get("num_workers", 4)),
        pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(dataloader.get("persistent_workers", True)),
        train_drop_last=bool(dataloader.get("train_drop_last", False)),
        include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
        offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
        drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
        filter_cold_items_after_split=bool(
            dataset.get(
                "filter_cold_items_after_split",
                False if preprocessing_preset == "letter_amazon18" else True,
            )
        ),
        cold_item_train_source=dataset.get("cold_item_train_source", "observed"),
        seed=int(config.get("seed", 42)),
    )


def build_model(config: Mapping[str, Any], data: AmazonReviewDataModule) -> SASRec:
    model_cfg = dict(config.get("model", {}))
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(data.item_vocab_size),
            max_history_len=int(data.cfg.max_history_len),
            d_model=int(model_cfg.get("d_model", model_cfg.get("embedding_dim", 64))),
            nhead=int(model_cfg.get("nhead", model_cfg.get("num_heads", 2))),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            pos_drop=float(model_cfg.get("pos_drop", model_cfg.get("dropout", 0.3))),
        )
    )
    model.pad_item_id = int(model_cfg.get("pad_item_id", 0))
    return model


def build_method(config: Mapping[str, Any]) -> SFTMethod:
    method_cfg = dict(config.get("method", {}))
    return SFTMethod(
        SFTConfig(
            compute_train_metrics=bool(method_cfg.get("compute_train_metrics", False)),
            compute_eval_metrics=bool(method_cfg.get("compute_eval_metrics", True)),
            compute_item_metrics=bool(method_cfg.get("compute_item_metrics", True)),
            compute_token_metrics=False,
            evaluate_generation=False,
            item_topk=tuple(int(k) for k in method_cfg.get("item_topk", [5, 10, 20])),
            pad_item_id=int(method_cfg.get("pad_item_id", 0)),
        )
    )


def build_trainer_config(config: Mapping[str, Any], run_dir: Path) -> TrainerConfig:
    training = dict(config.get("training", {}))
    allowed = {
        "max_epochs",
        "max_steps",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "skip_non_finite_loss",
        "use_amp",
        "amp_dtype",
        "scheduler_step_on",
        "log_every_steps",
        "progress_bar",
        "validate_every_epochs",
        "validate_every_steps",
        "run_validation_at_start",
        "save_best",
        "save_last",
        "save_every_epochs",
        "monitor",
        "monitor_mode",
        "early_stopping_patience",
        "min_delta",
        "prefix_method_metrics_with_stage",
        "max_train_batches",
        "max_eval_batches",
    }
    data = {key: value for key, value in training.items() if key in allowed}
    data["checkpoint_dir"] = run_dir / "checkpoints"
    return TrainerConfig.from_dict(data)


def build_experiment_logger(
    config: Mapping[str, Any],
    *,
    run_name: str,
    run_dir: Path,
    config_path: Path,
    resolved_config: Mapping[str, Any],
    cli_args: Mapping[str, Any],
) -> CompositeExperimentLogger:
    loggers = [JsonlExperimentLogger(run_dir / "training_log.jsonl")]
    comet_logger = CometExperimentLogger.from_config(
        dict(config.get("comet", {})),
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_path,
        resolved_config=resolved_config,
        cli_args=cli_args,
    )
    if comet_logger is not None:
        loggers.append(comet_logger)
    return CompositeExperimentLogger(loggers)


def data_config_paths(cfg: AmazonReviewDataConfig) -> dict[str, str | None]:
    return {
        "reviews_path": str(cfg.reviews_path),
        "metadata_path": str(cfg.metadata_path) if cfg.metadata_path is not None else None,
    }


def summary_to_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epochs_completed": summary.get("epochs_completed"),
        "steps_completed": summary.get("steps_completed"),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
    }
    for group in ("validation_metrics", "test_metrics"):
        metrics = summary.get(group)
        if isinstance(metrics, Mapping):
            payload.update(metrics)
    return payload


def create_run_dir(experiments_dir: Path, run_name: str, config: Mapping[str, Any]) -> Path:
    experiments_dir = resolve_path(experiments_dir, base=PROJECT_ROOT)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:48] or "run"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    run_dir = experiments_dir / f"{stamp}_{safe_name}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def write_text_summary(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# SASRec Run Summary",
        "",
        f"- Dataset: {summary['dataset'].get('reviews_path')}",
        f"- Epochs completed: {summary['epochs_completed']}",
        f"- Steps completed: {summary['steps_completed']}",
        f"- Best metric: {summary['best_metric']}",
        f"- Best checkpoint: {summary['best_checkpoint']}",
        f"- Last checkpoint: {summary['last_checkpoint']}",
        f"- Experiment folder: {summary['experiment_dir']}",
        "",
        "## Validation Metrics",
        json.dumps(to_jsonable(summary.get("validation_metrics", {})), indent=2, sort_keys=True),
        "",
        "## Test Metrics",
        json.dumps(to_jsonable(summary.get("test_metrics", {})), indent=2, sort_keys=True),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_optional_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    return resolve_path(value, base=base)


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
