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
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.base import FlowRecMethod
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.preference.config import DPOConfig
from flowrec.ft_methods.preference.method import DPOMethod
from flowrec.ft_methods.grpo.method import GRPOMethod
from flowrec.ft_methods.grpo.config import GRPOConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.tiger import TigerModel
from flowrec.training.trainer import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.run_tiger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TIGER with FlowRec.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiger_beauty.json",
        help="Path to a JSON experiment config.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Short name for the run folder.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "tiger",
        help="Directory where immutable TIGER experiment folders are created.",
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

    run_name = args.run_name or str(config.get("experiment_name", "tiger"))
    run_dir = create_run_dir(args.experiments_dir, run_name, config)
    setup_logging(run_dir)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Config: %s", config_path)

    data_cfg = build_data_config(config, config_path=config_path)
    data = build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("TIGER training requires data.semantic_id_path.")

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = build_model(config, data)
    method = build_method(config, model)
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
        logger=JsonlExperimentLogger(run_dir / "training_log.jsonl"),
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

    LOGGER.info("Training finished. Final summary: %s", json.dumps(summary, sort_keys=True))


def build_data_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> AmazonReviewDataConfig | YambdaDataConfig:
    data = dict(config.get("data", {}))
    dataset = dict(config.get("dataset", {}))
    dataloader = dict(config.get("dataloader", {}))
    model = dict(config.get("model", {}))
    base = config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT
    max_sequence_length = int(dataset.get("max_sequence_length", 20))
    source = str(data.get("source", dataset.get("source", "amazon"))).lower()

    if source == "yambda":
        return YambdaDataConfig(
            interactions_path=resolve_optional_path(data.get("interactions_path"), base=base),
            semantic_id_path=resolve_optional_path(data.get("semantic_id_path"), base=base),
            use_hf=bool(data.get("use_hf", False)),
            hf_dataset_name=str(data.get("hf_dataset_name", "yandex/yambda")),
            dataset_type=dataset.get("dataset_type", data.get("dataset_type", "flat")),
            dataset_size=dataset.get("dataset_size", data.get("dataset_size", "50m")),
            event_file=dataset.get("event_file", data.get("event_file", "listens")),
            streaming=bool(data.get("streaming", dataset.get("streaming", False))),
            max_rows=(
                int(data["max_rows"])
                if data.get("max_rows") is not None
                else (
                    int(dataset["max_rows"])
                    if dataset.get("max_rows") is not None
                    else None
                )
            ),
            min_user_interactions=int(dataset.get("min_user_interactions", 5)),
            min_item_interactions=int(dataset.get("min_item_interactions", 0)),
            deduplicate_consecutive=bool(dataset.get("deduplicate_consecutive", False)),
            keep_raw_interactions=bool(dataloader.get("include_raw_in_batch", False)),
            organic_filter=dataset.get("organic_filter", "any"),
            min_played_ratio_pct=(
                float(dataset["min_played_ratio_pct"])
                if dataset.get("min_played_ratio_pct") is not None
                else None
            ),
            allowed_event_types=(
                tuple(str(x) for x in dataset["allowed_event_types"])
                if dataset.get("allowed_event_types") is not None
                else None
            ),
            split_strategy=dataset.get("split_strategy", "leave_one_out"),
            min_history_len=int(dataset.get("min_history_len", 1)),
            train_all_prefixes=bool(dataset.get("train_all_prefixes", True)),
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
            global_eval_target_mode=dataset.get("global_eval_target_mode", "all"),
            max_history_len=int(dataset.get("max_history_len", max(1, max_sequence_length - 1))),
            batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 256))),
            eval_batch_size=int(
                dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 512))
            ),
            num_workers=int(dataloader.get("num_workers", 4)),
            pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
            persistent_workers=bool(dataloader.get("persistent_workers", True)),
            train_drop_last=bool(dataloader.get("train_drop_last", False)),
            include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
            offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
            drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
            user_hash_mod=int(model["user_ids_count"]) if model.get("user_ids_count") is not None else None,
            seed=int(config.get("seed", 42)),
        )

    if source != "amazon":
        raise ValueError(f"Unknown data.source={source!r}. Expected 'amazon' or 'yambda'.")

    return AmazonReviewDataConfig(
        reviews_path=resolve_path(data["reviews_path"], base=base),
        metadata_path=resolve_optional_path(data.get("metadata_path"), base=base),
        semantic_id_path=resolve_optional_path(data.get("semantic_id_path"), base=base),
        item_key=dataset.get("item_key", "parent_asin"),
        min_user_interactions=int(dataset.get("min_user_interactions", 5)),
        min_item_interactions=int(dataset.get("min_item_interactions", 0)),
        split_strategy=dataset.get("split_strategy", "leave_one_out"),
        min_history_len=int(dataset.get("min_history_len", 1)),
        train_all_prefixes=bool(dataset.get("train_all_prefixes", True)),
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
        global_eval_target_mode=dataset.get("global_eval_target_mode", "all"),
        max_history_len=int(dataset.get("max_history_len", max(1, max_sequence_length - 1))),
        batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 256))),
        eval_batch_size=int(
            dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 512))
        ),
        num_workers=int(dataloader.get("num_workers", 4)),
        pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(dataloader.get("persistent_workers", True)),
        train_drop_last=bool(dataloader.get("train_drop_last", False)),
        include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
        offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
        drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
        user_hash_mod=int(model["user_ids_count"]) if model.get("user_ids_count") is not None else None,
        seed=int(config.get("seed", 42)),
    )


def build_data_module(
    cfg: AmazonReviewDataConfig | YambdaDataConfig,
) -> AmazonReviewDataModule | YambdaDataModule:
    if isinstance(cfg, AmazonReviewDataConfig):
        return AmazonReviewDataModule(cfg)
    if isinstance(cfg, YambdaDataConfig):
        return YambdaDataModule(cfg)
    raise TypeError(f"Unsupported data config: {type(cfg)!r}")


def data_config_paths(cfg: AmazonReviewDataConfig | YambdaDataConfig) -> dict[str, str | None]:
    fields = ("reviews_path", "metadata_path", "interactions_path", "semantic_id_path")
    return {
        key: str(getattr(cfg, key)) if getattr(cfg, key, None) is not None else None
        for key in fields
        if hasattr(cfg, key)
    }


def build_model(
    config: Mapping[str, Any],
    data: AmazonReviewDataModule | YambdaDataModule,
) -> TigerModel:
    model_cfg = dict(config.get("model", {}))
    num_codebooks = int(config.get("dataset", {}).get("num_codebooks", data.sid_len or 0))
    if data.sid_len != num_codebooks:
        LOGGER.warning(
            "Configured num_codebooks=%s but semantic IDs have length=%s; using semantic ID length.",
            num_codebooks,
            data.sid_len,
        )
    top_k = int(model_cfg.get("top_k", model_cfg.get("num_return_sequences", 20)))

    return TigerModel(
        codebook_size=int(model_cfg.get("codebook_size", 256)),
        sem_id_len=int(data.sid_len or num_codebooks),
        user_ids_count=int(model_cfg.get("user_ids_count", 2000)),
        semantic_id_mapping=data.semantic_ids,
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
        num_decoder_layers=int(model_cfg.get("num_decoder_layers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_beams=int(model_cfg.get("num_beams", 100)),
        num_return_sequences=top_k,
        d_kv=int(model_cfg.get("d_kv", 64)),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-6)),
        activation=str(model_cfg.get("activation", "relu")),
        initializer_range=float(model_cfg.get("initializer_range", 0.02)),
    )

    checkpoint_path = model_cfg.get("checkpoint_path")
    if checkpoint_path:
        LOGGER.info("Loading initial model checkpoint from %s", checkpoint_path)
        model.load_checkpoint(Path(checkpoint_path), map_location="cpu")

    return model


def build_method(config: Mapping[str, Any], model: TigerModel) -> FlowRecMethod:
    training = dict(config.get("training", {}))
    method_cfg = dict(config.get("method", {}))
    method_name = str(method_cfg.get("name", "sft")).lower()
    top_k = int(config.get("model", {}).get("top_k", model.config_dict()["num_return_sequences"]))

    if method_name == "sft":
        sft_cfg = SFTConfig(
            compute_item_metrics=False,
            evaluate_generation=bool(training.get("evaluate_generation", True)),
            generation_topk=tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
            generation_num_beams=int(config.get("model", {}).get("num_beams", model.config_dict()["num_beams"])),
            generation_num_return_sequences=top_k,
            generation_constrained=bool(training.get("generation_constrained", True)),
        )
        return SFTMethod(sft_cfg)

    if method_name == "dpo":
        defaults = {
            "compute_train_metrics": True,
            "compute_eval_metrics": True,
            "compute_token_metrics": True,
            "evaluate_generation": bool(training.get("evaluate_generation", True)),
            "generation_topk": tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
            "generation_num_beams": int(
                config.get("model", {}).get("num_beams", model.config_dict()["num_beams"])
            ),
            "generation_num_return_sequences": top_k,
            "generation_constrained": bool(training.get("generation_constrained", True)),
        }
        defaults.update(method_cfg)
        return DPOMethod(DPOConfig.from_dict(defaults), model=model)
    
    if method_name == "grpo":
        defaults = {
            "compute_train_metrics": True,
            "compute_eval_metrics": True,
            "compute_token_metrics": True,
            "evaluate_generation": bool(training.get("evaluate_generation", True)),
            "generation_topk": tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
            "generation_num_beams": int(
                config.get("model", {}).get("num_beams", model.config_dict()["num_beams"])
            ),
            "generation_num_return_sequences": top_k,
            "generation_constrained": bool(training.get("generation_constrained", True)),
        }
    defaults.update(method_cfg)
    return GRPOMethod(GRPOConfig.from_dict(defaults), model=model)

    if method_name != "gflow":
        raise ValueError(f"Unknown method.name={method_name!r}. Expected 'sft', 'grpo', 'dpo', or 'gflow'.")

    defaults = {
        "evaluate_generation": bool(training.get("evaluate_generation", True)),
        "generation_topk": tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
        "generation_num_beams": int(
            config.get("model", {}).get("num_beams", model.config_dict()["num_beams"])
        ),
        "generation_num_return_sequences": top_k,
        "generation_constrained": bool(training.get("generation_constrained", True)),
        "logz_vocab_size": int(model.config_dict()["semantic_vocab_size"]),
    }
    defaults.update(method_cfg)
    return GFlowMethod(GFlowConfig.from_dict(defaults))


def build_trainer_config(config: Mapping[str, Any], run_dir: Path) -> TrainerConfig:
    training = dict(config.get("training", {}))
    checkpoint_dir = run_dir / "checkpoints"

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
    trainer_data = {key: value for key, value in training.items() if key in allowed}
    trainer_data["checkpoint_dir"] = checkpoint_dir
    return TrainerConfig.from_dict(trainer_data)


class JsonlExperimentLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, payload: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
        event_type = payload.get("type", "metric")
        brief = {k: v for k, v in payload.items() if k != "type" and isinstance(v, (int, float, str))}
        LOGGER.info("%s %s", event_type, brief)


def create_run_dir(experiments_dir: Path, run_name: str, config: Mapping[str, Any]) -> Path:
    experiments_dir = resolve_path(experiments_dir, base=PROJECT_ROOT)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:48] or "run"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    run_dir = experiments_dir / f"{stamp}_{safe_name}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "run.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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
    dataset = summary["dataset"]
    dataset_path = dataset.get("reviews_path") or dataset.get("interactions_path")
    lines = [
        "# TIGER Run Summary",
        "",
        f"- Dataset: {dataset_path}",
        f"- Semantic IDs: {dataset.get('semantic_id_path')}",
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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    main()
