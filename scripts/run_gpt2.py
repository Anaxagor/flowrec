from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.candidates.item_model import (
    ItemModelSemanticCandidateConfig,
    ItemModelSemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import FlowRecMethod
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.rewards.mix import MultiplicativeRewardModel
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
from flowrec.training.loggers import (
    CompositeExperimentLogger,
    CometExperimentLogger,
    JsonlExperimentLogger,
    to_jsonable,
)
from flowrec.training.trainer import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.run_gpt2")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 recommender with FlowRec.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gpt2_beauty_sft.json",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Short name for the run folder.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "gpt2",
        help="Directory where immutable GPT-2 experiment folders are created.",
    )
    parser.add_argument(
        "--plain-run-dir",
        action="store_true",
        help="Use run-name as the exact run folder name without timestamp/hash.",
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

    run_name = args.run_name or str(config.get("experiment_name", "gpt2"))
    run_dir = create_run_dir(args.experiments_dir, run_name, config, plain=bool(args.plain_run_dir))
    setup_logging(run_dir)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Config: %s", config_path)

    data_cfg = build_data_config(config, config_path=config_path)
    data = build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("GPT-2 training requires data.semantic_id_path.")
    prepare_semantic_ids(data, config)

    init_checkpoint_path = config.get("training", {}).get("init_checkpoint_path")
    init_checkpoint_strict = bool(config.get("training", {}).get("init_checkpoint_strict", True))

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = build_model(config, data)
    if data.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support "
            f"train_example_layout={data.cfg.train_example_layout!r}; "
            f"supported: {model.supported_train_layouts}."
        )
    if init_checkpoint_path is not None:
        resolved_init_checkpoint = resolve_experiment_path(init_checkpoint_path, config_path=config_path)
        LOGGER.info("Loading initial GPT-2 model weights from %s", resolved_init_checkpoint)
        load_model_state_dict(model, resolved_init_checkpoint, strict=init_checkpoint_strict)

    method = build_method(
        config,
        model,
        data=data,
        train_loader=train_loader,
        config_path=config_path,
    )
    trainer_cfg = build_trainer_config(config, run_dir)

    resolved = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "data": asdict(data_cfg),
        "data_summary": data.summary(),
        "model": model.config_dict(),
        "method": resolved_method_config(method, config),
        "trainer": trainer_cfg.to_dict(),
        "optimizer": config.get("optimizer", {}),
    }
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "cli_args.json", vars(args))
    experiment_logger = build_experiment_logger(
        config,
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_path,
        resolved_config=resolved,
        cli_args=vars(args),
    )

    LOGGER.info("Data summary: %s", json.dumps(data.summary(), sort_keys=True))
    LOGGER.info(
        "Model parameters: total=%s trainable=%s",
        f"{model.num_parameters():,}",
        f"{model.num_parameters(trainable_only=True):,}",
    )

    try:
        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=trainer_cfg,
            device=args.device,
            lr=float(config.get("optimizer", {}).get("lr", 3e-4)),
            weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.0)),
            betas=tuple(config.get("optimizer", {}).get("betas", (0.9, 0.999))),
            eps=float(config.get("optimizer", {}).get("eps", 1e-8)),
            logger=experiment_logger,
        )
        # Measure training time
        fit_started_at = time.perf_counter()
        LOGGER.info("Trainer.fit started (training plus periodic validation)")

        fit_result = trainer.fit(train_loader, val_loader)

        fit_elapsed_sec = time.perf_counter() - fit_started_at

        # Finish training time measurement and log the results
        LOGGER.info(
            "Trainer.fit finished (training plus periodic validation): %.2f minutes (%.2f seconds)",
            fit_elapsed_sec / 60.0,
            fit_elapsed_sec,
        )

        # Loading the best checkpoint also restores its trainer state. Keep the
        # actual completed-training state for the final summary.
        completed_epochs = int(trainer.state.epoch + 1)
        completed_steps = int(trainer.state.global_step)

        test_metrics: dict[str, float] = {}
        checkpoint_for_test = fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        if checkpoint_for_test is not None:
            trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)
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
            test_evaluate_generation = training_cfg.get("test_evaluate_generation")
            if test_evaluate_generation is None:
                test_evaluate_generation = bool(training_cfg.get("evaluate_generation", True))
                if test_generation_num_beams is not None:
                    test_evaluate_generation = True
            old_generation_cfg = override_generation_for_test(
                method,
                evaluate_generation=test_evaluate_generation,
                num_beams=test_generation_num_beams,
                num_return_sequences=test_generation_num_return_sequences,
                topk=test_generation_topk,
            )
            try:
                LOGGER.info(
                    "Using test_evaluate_generation=%s and test_generation_num_beams=%s for final test.",
                    test_evaluate_generation,
                    test_generation_num_beams,
                )
                test_metrics = trainer.test(test_loader)
            finally:
                restore_generation_config(method, old_generation_cfg)

        summary = {
            "dataset": {**data_config_paths(data_cfg), **data.summary()},
            "model": model.config_dict(),
            "epochs_completed": completed_epochs,
            "steps_completed": completed_steps,
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
        log_final_summary(experiment_logger, summary, step=completed_steps)
        log_run_artifacts(experiment_logger, run_dir, step=completed_steps)

        LOGGER.info("Training finished. Final summary: %s", json.dumps(summary, sort_keys=True))
    finally:
        close_logger(experiment_logger)


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
    max_history_len = int(dataset.get("max_history_len", max(1, max_sequence_length - 1)))
    train_example_layout = str(dataset.get("train_example_layout", "prefix_target"))
    train_window_len = resolve_train_window_len(
        dataset,
        train_example_layout=train_example_layout,
        max_history_len=max_history_len,
        max_sequence_length=max_sequence_length,
    )
    source = str(data.get("source", dataset.get("source", "amazon"))).lower()

    if source == "yambda":
        hf_recent_months = dataset.get("hf_recent_months")
        return YambdaDataConfig(
            interactions_path=resolve_optional_path(data.get("interactions_path"), base=base),
            semantic_id_path=resolve_optional_path(data.get("semantic_id_path"), base=base),
            use_hf=bool(data.get("use_hf", False)),
            hf_dataset_name=str(data.get("hf_dataset_name", "yandex/yambda")),
            hf_start_date=dataset.get("hf_start_date"),
            hf_end_date=dataset.get("hf_end_date"),
            hf_recent_months=int(hf_recent_months) if hf_recent_months is not None else None,
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
            train_example_layout=train_example_layout,
            train_window_len=train_window_len,
            train_window_stride=(
                int(dataset["train_window_stride"])
                if dataset.get("train_window_stride") is not None
                else None
            ),
            train_all_prefixes=bool(dataset.get("train_all_prefixes", True)),
            train_target_policy=dataset.get("train_target_policy", "all"),
            train_targets_per_user=int(dataset.get("train_targets_per_user", 256)),
            train_stride=int(dataset.get("train_stride", 10)),
            train_last_n=int(dataset.get("train_last_n", 512)),
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
            global_eval_target_mode=dataset.get("global_eval_target_mode", "all"),
            max_history_len=max_history_len,
            history_padding=dataset.get("history_padding", "left"),
            batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 256))),
            eval_batch_size=int(
                dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 512))
            ),
            num_workers=int(dataloader.get("num_workers", 4)),
            pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
            persistent_workers=bool(dataloader.get("persistent_workers", True)),
            train_drop_last=bool(dataloader.get("train_drop_last", False)),
            include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
            filter_cold_items_after_split=bool(dataset.get("filter_cold_items_after_split", True)),
            cold_item_train_source=dataset.get("cold_item_train_source", "observed"),
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
        train_example_layout=train_example_layout,
        train_window_len=train_window_len,
        train_window_stride=(
            int(dataset["train_window_stride"])
            if dataset.get("train_window_stride") is not None
            else None
        ),
        train_all_prefixes=bool(dataset.get("train_all_prefixes", True)),
        train_target_policy=dataset.get("train_target_policy", "all"),
        train_targets_per_user=int(dataset.get("train_targets_per_user", 256)),
        train_stride=int(dataset.get("train_stride", 10)),
        train_last_n=int(dataset.get("train_last_n", 512)),
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
        global_eval_target_mode=dataset.get("global_eval_target_mode", "all"),
        max_history_len=max_history_len,
        history_padding=dataset.get("history_padding", "left"),
        batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 256))),
        eval_batch_size=int(
            dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 512))
        ),
        num_workers=int(dataloader.get("num_workers", 4)),
        pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(dataloader.get("persistent_workers", True)),
        train_drop_last=bool(dataloader.get("train_drop_last", False)),
        include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
        filter_cold_items_after_split=bool(dataset.get("filter_cold_items_after_split", True)),
        cold_item_train_source=dataset.get("cold_item_train_source", "observed"),
        offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
        drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
        user_hash_mod=int(model["user_ids_count"]) if model.get("user_ids_count") is not None else None,
        seed=int(config.get("seed", 42)),
    )


def resolve_train_window_len(
    dataset: Mapping[str, Any],
    *,
    train_example_layout: str,
    max_history_len: int,
    max_sequence_length: int,
) -> int | None:
    raw_value = dataset.get("train_window_len")
    if train_example_layout != "dense_causal":
        return int(raw_value) if raw_value is not None else None

    if raw_value is None:
        return max_history_len

    value = int(raw_value)
    # Older GPT-2 configs used max_sequence_length=history+target as the dense
    # input-window length. DenseCausalCollator adds the final target itself.
    if value == max_sequence_length and value == max_history_len + 1:
        LOGGER.warning(
            "Interpreting legacy train_window_len=%s as total sequence length; "
            "using %s history items instead.",
            value,
            max_history_len,
        )
        return max_history_len
    if value <= 0:
        raise ValueError("dataset.train_window_len must be positive.")
    if value > max_history_len:
        raise ValueError(
            "For dense_causal training, train_window_len is the number of input "
            f"history items and must not exceed max_history_len ({max_history_len}); "
            f"got {value}."
        )
    return value


def prepare_semantic_ids(
    data: AmazonReviewDataModule | YambdaDataModule,
    config: Mapping[str, Any],
) -> None:
    """Validate and optionally compact the post-filtered semantic vocabulary."""
    if data.semantic_ids is None:
        raise ValueError("Semantic IDs are required.")

    mapping = {str(item): [int(token) for token in tokens] for item, tokens in data.semantic_ids.items()}
    if not mapping:
        raise ValueError("The post-filtered semantic-ID catalog is empty.")

    lengths = {len(tokens) for tokens in mapping.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"Semantic IDs must have one positive fixed length, got {sorted(lengths)}.")
    sid_len = next(iter(lengths))
    if any(token < 0 for tokens in mapping.values() for token in tokens):
        raise ValueError("Semantic IDs must be non-negative.")

    rows = torch.as_tensor(list(mapping.values()), dtype=torch.long)
    unique_rows = int(torch.unique(rows, dim=0).size(0))
    if unique_rows != rows.size(0):
        raise ValueError(
            "Semantic IDs are not unique after collision resolution: "
            f"{unique_rows} unique IDs for {rows.size(0)} items."
        )

    model_cfg = config.get("model", {})
    compact = bool(model_cfg.get("compact_semantic_vocabulary", True))
    old_max = int(rows.max().item())
    if compact:
        used_tokens = sorted({int(token) for tokens in mapping.values() for token in tokens})
        # Keep 0 available for dataloader padding; actual semantic tokens start at 1.
        token_remap = {token: new_token for new_token, token in enumerate(used_tokens, start=1)}
        mapping = {
            item: [token_remap[token] for token in tokens]
            for item, tokens in mapping.items()
        }
        data.semantic_ids = mapping
        data.sid_len = sid_len
        LOGGER.info(
            "Compacted semantic vocabulary: max_token %s -> %s, used_tokens=%s.",
            old_max,
            max(token_remap.values()),
            len(token_remap),
        )

    rows = torch.as_tensor(list(mapping.values()), dtype=torch.long)
    position_stats = []
    for position in range(sid_len):
        values, counts = torch.unique(rows[:, position], return_counts=True)
        position_stats.append(
            {
                "position": position,
                "unique_tokens": int(values.numel()),
                "largest_token_share": float(counts.max().item() / rows.size(0)),
            }
        )
    LOGGER.info(
        "Semantic catalog: items=%s sid_len=%s vocab_max=%s position_stats=%s",
        rows.size(0),
        sid_len,
        int(rows.max().item()),
        json.dumps(position_stats),
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
) -> GPT2RecModel:
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = dict(config.get("dataset", {}))

    if data.sid_len is None or data.semantic_ids is None:
        raise ValueError(
            "GPT-2 requires semantic IDs. Set data.semantic_id_path "
            "to a valid semantic_ids.json file."
        )

    configured_codebooks = int(dataset_cfg.get("num_codebooks", data.sid_len))
    if data.sid_len != configured_codebooks:
        LOGGER.warning(
            "Configured num_codebooks=%s but semantic IDs have length=%s; "
            "using the loaded semantic-ID length.",
            configured_codebooks,
            data.sid_len,
        )

    embedding_dim = int(model_cfg.get("embedding_dim", model_cfg.get("n_embd", 128)))
    num_heads = int(model_cfg.get("num_heads", model_cfg.get("n_head", 4)))
    if embedding_dim % num_heads != 0:
        raise ValueError(
            "GPT-2 embedding_dim must be divisible by num_heads: "
            f"{embedding_dim} % {num_heads} != 0."
        )

    top_k = int(model_cfg.get("top_k", model_cfg.get("num_return_sequences", 20)))
    max_history_len = int(dataset_cfg.get("max_history_len", 20))
    train_window_len = int(dataset_cfg.get("train_window_len") or max_history_len)
    dense_train_positions = 1 + (train_window_len + 1) * int(data.sid_len) - 1
    generation_positions = 1 + max_history_len * int(data.sid_len) + int(data.sid_len)
    required_positions = max(dense_train_positions, generation_positions)
    max_position_embeddings = int(
        model_cfg.get(
            "max_position_embeddings",
            model_cfg.get("n_positions", required_positions),
        )
    )
    if max_position_embeddings < required_positions:
        raise ValueError(
            "model.max_position_embeddings is too small for the configured "
            "history/window and semantic-ID length: "
            f"{max_position_embeddings} < {required_positions}."
        )

    model = GPT2RecModel(
        codebook_size=int(model_cfg.get("codebook_size", 256)),
        sem_id_len=int(data.sid_len),
        user_ids_count=int(model_cfg.get("user_ids_count", 2000)),
        semantic_id_mapping=data.semantic_ids,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=int(model_cfg.get("num_layers", model_cfg.get("n_layer", 4))),
        dim_feedforward=int(model_cfg.get("dim_feedforward", model_cfg.get("n_inner", 512))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attention_dropout=float(
            model_cfg.get("attention_dropout", model_cfg.get("dropout", 0.1))
        ),
        residual_dropout=float(
            model_cfg.get("residual_dropout", model_cfg.get("dropout", 0.1))
        ),
        embedding_dropout=float(
            model_cfg.get("embedding_dropout", model_cfg.get("dropout", 0.1))
        ),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-5)),
        initializer_range=float(model_cfg.get("initializer_range", 0.02)),
        max_position_embeddings=max_position_embeddings,
        num_beams=int(model_cfg.get("num_beams", 20)),
        num_return_sequences=top_k,
        fast_greedy_generation=bool(model_cfg.get("fast_greedy_generation", True)),
    )

    if bool(config.get("training", {}).get("gradient_checkpointing", False)):
        model.model.gradient_checkpointing_enable()
        model.model.config.use_cache = False
        LOGGER.info("Enabled GPT-2 gradient checkpointing.")

    return model


def build_method(
    config: Mapping[str, Any],
    model: GPT2RecModel,
    *,
    data: AmazonReviewDataModule | YambdaDataModule,
    train_loader,
    config_path: Path,
) -> FlowRecMethod:
    training = dict(config.get("training", {}))
    method_cfg = dict(config.get("method", {}))
    method_name = str(method_cfg.get("name", "sft")).lower()
    top_k = int(config.get("model", {}).get("top_k", model.config_dict()["num_return_sequences"]))
    default_generation_num_beams = int(
        config.get("model", {}).get("num_beams", model.config_dict()["num_beams"])
    )
    generation_num_beams = int(
        training.get("generation_num_beams", default_generation_num_beams)
    )
    generation_num_return_sequences = int(
        training.get("generation_num_return_sequences", min(top_k, generation_num_beams))
    )
    generation_topk = tuple(
        int(k)
        for k in training.get(
            "generation_topk",
            [5, 10, min(20, generation_num_return_sequences)],
        )
    )
    validate_generation_config(
        num_beams=generation_num_beams,
        num_return_sequences=generation_num_return_sequences,
        topk=generation_topk,
    )

    if method_name == "sft":
        sft_cfg = SFTConfig(
            compute_item_metrics=False,
            evaluate_generation=bool(training.get("evaluate_generation", True)),
            generation_topk=generation_topk,
            generation_num_beams=generation_num_beams,
            generation_num_return_sequences=generation_num_return_sequences,
            generation_constrain_valid_sequences=bool(
                training.get("generation_constrain_valid_sequences", True)
            ),
            generation_exclude_seen_items=bool(
                training.get("generation_exclude_seen_items", False)
            ),
        )
        return SFTMethod(sft_cfg)

    if method_name != "gflow":
        raise ValueError(f"Unknown method.name={method_name!r}. Expected 'sft' or 'gflow'.")

    reward_model_name = str(method_cfg.pop("reward_model", "rating")).lower()
    candidate_sampler_name = str(method_cfg.get("candidate_sampler", "random")).lower()
    reward_cfg = dict(method_cfg.pop("reward", {}))
    svd_cfg = dict(method_cfg.pop("svd", {}))
    sasrec_cfg = dict(method_cfg.pop("sasrec", {}))
    sampler_cfg = dict(method_cfg)
    for sampler_only_key in (
        "easy_negative_steps",
        "exclude_history_in_sampling",
        "sasrec_oversample_factor",
        "keep_scorer_eval",
        "sampling_temperature",
        "candidate_pool_size",
        "score_chunk_size",
        "pool_strategy",
    ):
        method_cfg.pop(sampler_only_key, None)

    defaults = {
        "evaluate_generation": bool(training.get("evaluate_generation", True)),
        "generation_topk": generation_topk,
        "generation_num_beams": generation_num_beams,
        "generation_num_return_sequences": generation_num_return_sequences,
        "generation_constrain_valid_sequences": bool(
            training.get("generation_constrain_valid_sequences", True)
        ),
        "generation_exclude_seen_items": bool(
            training.get("generation_exclude_seen_items", False)
        ),
        "logz_vocab_size": int(model.config_dict()["semantic_vocab_size"]),
    }
    defaults.update(method_cfg)
    gflow_cfg = GFlowConfig.from_dict(defaults)
    needs_sasrec_reward = reward_config_requires_model(reward_model_name, reward_cfg, "sasrec")
    needs_svd_reward = reward_config_requires_model(reward_model_name, reward_cfg, "svd")
    sasrec_model = (
        build_sasrec_teacher_model(data, sasrec_cfg, config_path=config_path)
        if needs_sasrec_reward
        or candidate_sampler_name in {"sasrec", "sasrec_score", "model_score", "item_model"}
        else None
    )
    svd_model = (
        build_svd_scoring_model(
            data=data,
            train_loader=train_loader,
            seed=int(config.get("seed", 42)),
            svd_cfg=svd_cfg,
        )
        if needs_svd_reward or candidate_sampler_name == "svd"
        else None
    )
    reward_model = build_reward_model(
        reward_model_name,
        data=data,
        train_loader=train_loader,
        config_path=config_path,
        seed=int(config.get("seed", 42)),
        reward_cfg=reward_cfg,
        svd_cfg=svd_cfg,
        svd_model=svd_model,
        sasrec_model=sasrec_model,
        reward_epsilon=float(gflow_cfg.reward_epsilon),
    )
    if (
        isinstance(reward_model, MixedRewardModel)
        and reward_model.uses_candidate_set_normalization
        and not gflow_cfg.reward_cache_path
    ):
        LOGGER.warning(
            "Live mixed reward uses candidate-set normalization (%s). "
            "For TB this is not a fixed terminal reward; prefer precomputed reward_cache_path "
            "or component normalizers=none.",
            reward_model.normalizers,
        )
    candidate_sampler = build_candidate_sampler(
        candidate_sampler_name,
        model=model,
        data=data,
        svd_model=svd_model,
        sasrec_model=sasrec_model,
        method_cfg=sampler_cfg,
    )
    return GFlowMethod(gflow_cfg, reward_model=reward_model, candidate_sampler=candidate_sampler)



def validate_generation_config(
    *,
    num_beams: int,
    num_return_sequences: int,
    topk: tuple[int, ...],
) -> None:
    if num_beams <= 0:
        raise ValueError("generation_num_beams must be positive.")
    if num_return_sequences <= 0:
        raise ValueError("generation_num_return_sequences must be positive.")
    if num_return_sequences > num_beams:
        raise ValueError(
            "generation_num_return_sequences cannot exceed generation_num_beams "
            "when do_sample=False."
        )
    if not topk or any(k <= 0 for k in topk):
        raise ValueError("generation_topk must contain positive integers.")
    if max(topk) > num_return_sequences:
        raise ValueError(
            "The largest generation_topk value cannot exceed "
            "generation_num_return_sequences: "
            f"max(topk)={max(topk)}, num_return_sequences={num_return_sequences}."
        )

def override_generation_for_test(
    method: FlowRecMethod,
    *,
    evaluate_generation: Any = None,
    num_beams: Any = None,
    num_return_sequences: Any = None,
    topk: Any = None,
) -> tuple[Any, Any] | None:
    updates = {}
    if evaluate_generation is not None:
        updates["evaluate_generation"] = bool(evaluate_generation)
    if num_beams is not None:
        updates["generation_num_beams"] = int(num_beams)
    if num_return_sequences is not None:
        updates["generation_num_return_sequences"] = int(num_return_sequences)
    if topk is not None:
        updates["generation_topk"] = tuple(int(k) for k in topk)
    if not updates:
        return None

    current_cfg = getattr(method, "cfg", None)
    resolved_num_beams = int(
        updates.get(
            "generation_num_beams",
            getattr(current_cfg, "generation_num_beams"),
        )
    )
    resolved_num_return_sequences = int(
        updates.get(
            "generation_num_return_sequences",
            getattr(current_cfg, "generation_num_return_sequences"),
        )
    )
    resolved_topk = tuple(
        int(k)
        for k in updates.get(
            "generation_topk",
            getattr(current_cfg, "generation_topk"),
        )
    )
    validate_generation_config(
        num_beams=resolved_num_beams,
        num_return_sequences=resolved_num_return_sequences,
        topk=resolved_topk,
    )

    if isinstance(method, SFTMethod):
        old_cfg = method.cfg
        method.cfg = replace(method.cfg, **updates)
        return old_cfg, None

    if isinstance(method, GFlowMethod):
        old_cfg = method.cfg
        old_sft_cfg = method.sft_method.cfg
        method.cfg = replace(method.cfg, **updates)
        method.sft_method.cfg = replace(method.sft_method.cfg, **updates)
        return old_cfg, old_sft_cfg

    return None


def restore_generation_config(
    method: FlowRecMethod,
    old_generation_cfg: tuple[Any, Any] | None,
) -> None:
    if old_generation_cfg is None:
        return

    old_cfg, old_sft_cfg = old_generation_cfg
    if isinstance(method, SFTMethod):
        method.cfg = old_cfg
    elif isinstance(method, GFlowMethod):
        method.cfg = old_cfg
        if old_sft_cfg is not None:
            method.sft_method.cfg = old_sft_cfg


def build_svd_scoring_model(
    *,
    data: AmazonReviewDataModule | YambdaDataModule,
    train_loader,
    seed: int,
    svd_cfg: Mapping[str, Any],
):
    from flowrec.models.svd import SVDConfig, SVDModel

    fit_cfg = dict(svd_cfg.get("fit", {}))
    model_cfg = {key: value for key, value in svd_cfg.items() if key != "fit"}
    svd_model = SVDModel(
        SVDConfig(
            num_users=int(data.num_users),
            num_items=int(data.item_vocab_size),
            n_components=int(model_cfg.get("n_components", 256)),
            pad_item_id=int(model_cfg.get("pad_item_id", 0)),
            input_transform=model_cfg.get("input_transform", "binary"),
            user_representation=model_cfg.get("user_representation", "history_foldin"),
            foldin_reg=float(model_cfg.get("foldin_reg", 1e-2)),
            exclude_history_in_generate=bool(model_cfg.get("exclude_history_in_generate", False)),
            score_chunk_size=(
                int(model_cfg["score_chunk_size"])
                if model_cfg.get("score_chunk_size") is not None
                else None
            ),
            random_state=int(model_cfg.get("random_state", seed)),
        )
    )
    LOGGER.info("Fitting SVD scoring model from the training dataloader.")
    svd_model.fit_from_dataloader(
        train_loader,
        count_targets=bool(fit_cfg.get("count_targets", True)),
        count_history=bool(fit_cfg.get("count_history", True)),
        use_ratings=bool(fit_cfg.get("use_ratings", False)),
    )
    return svd_model


class MixedRewardModel(torch.nn.Module):
    def __init__(
        self,
        components: list[torch.nn.Module],
        *,
        weights: list[float] | None = None,
        normalizers: list[str] | None = None,
        reward_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("MixedRewardModel requires at least one component.")
        self.components = torch.nn.ModuleList(components)
        self.weights = [float(w) for w in (weights or [1.0] * len(components))]
        self.normalizers = [str(n) for n in (normalizers or ["batch_max"] * len(components))]
        self.reward_epsilon = float(reward_epsilon)
        if len(self.weights) != len(self.components) or len(self.normalizers) != len(self.components):
            raise ValueError("weights and normalizers must match the number of reward components.")

    @property
    def uses_candidate_set_normalization(self) -> bool:
        return any(str(mode).lower() not in {"none", "identity"} for mode in self.normalizers)

    def forward(self, batch, trajectories) -> torch.Tensor:
        mixed: torch.Tensor | None = None
        for model, weight, normalizer in zip(self.components, self.weights, self.normalizers):
            rewards = model(batch, trajectories).float()
            if rewards.ndim != 2:
                raise ValueError(f"Reward components must return [B, N], got {tuple(rewards.shape)}.")
            term = self._normalize(rewards, normalizer) * weight
            mixed = term if mixed is None else mixed + term
        if mixed is None:
            raise RuntimeError("MixedRewardModel has no reward components.")
        return mixed.clamp_min(self.reward_epsilon)

    def _normalize(self, rewards: torch.Tensor, mode: str) -> torch.Tensor:
        mode = mode.lower()
        eps = self.reward_epsilon
        rewards = rewards.clamp_min(eps)
        if mode in {"none", "identity"}:
            return rewards
        if mode in {"batch_max", "max"}:
            denom = rewards.detach().amax().clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"batch_mean", "mean"}:
            denom = rewards.detach().mean().clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"batch_minmax", "minmax"}:
            min_reward = rewards.detach().amin()
            denom = (rewards.detach().amax() - min_reward).clamp_min(eps)
            scaled = (rewards - min_reward) / denom
            if torch.isclose(denom, rewards.new_tensor(eps)):
                scaled = torch.ones_like(rewards)
            return scaled.clamp_min(eps)
        if mode in {"row_max", "per_row_max"}:
            denom = rewards.detach().amax(dim=1, keepdim=True).clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"row_mean", "per_row_mean"}:
            denom = rewards.detach().mean(dim=1, keepdim=True).clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"row_minmax", "per_row_minmax"}:
            min_reward = rewards.detach().amin(dim=1, keepdim=True)
            denom = (rewards.detach().amax(dim=1, keepdim=True) - min_reward).clamp_min(eps)
            scaled = (rewards - min_reward) / denom
            scaled = torch.where(denom <= eps, torch.ones_like(scaled), scaled)
            return scaled.clamp_min(eps)
        raise ValueError(
            f"Unknown reward normalizer {mode!r}. "
            "Expected none, batch_max, batch_mean, batch_minmax, row_max, row_mean, or row_minmax."
        )


def reward_config_requires_model(name: str, reward_cfg: Mapping[str, Any], model_name: str) -> bool:
    normalized_name = str(name).lower()
    if normalized_name == model_name:
        return True
    if normalized_name not in {"mix", "mixed", "mixture", "sum", "multiply", "multiplicative", "product"}:
        return False
    return any(
        reward_config_requires_model(
            component_reward_name(component),
            component_reward_config(component),
            model_name,
        )
        for component in reward_components(reward_cfg)
    )


def reward_components(reward_cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_components = reward_cfg.get("components", reward_cfg.get("rewards"))
    if raw_components is None:
        raise ValueError("Mixed rewards require reward.components (or reward.rewards).")
    components: list[Mapping[str, Any]] = []
    for component in raw_components:
        if isinstance(component, str):
            components.append({"name": component})
        elif isinstance(component, Mapping):
            components.append(dict(component))
        else:
            raise TypeError(f"Reward component must be a string or object, got {type(component)!r}.")
    return components


def component_reward_name(component: Mapping[str, Any]) -> str:
    name = component.get("name", component.get("reward_model", component.get("model")))
    if name is None:
        raise ValueError(f"Reward component is missing name/reward_model: {component!r}")
    return str(name).lower()


def component_reward_config(component: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(component.get("reward", {}))
    reserved = {"name", "reward_model", "model", "weight", "normalize", "normalization", "reward"}
    for key, value in component.items():
        if key not in reserved:
            config.setdefault(key, value)
    return config


def build_reward_model(
    name: str,
    *,
    data: AmazonReviewDataModule | YambdaDataModule,
    train_loader,
    config_path: Path,
    seed: int,
    reward_cfg: Mapping[str, Any],
    svd_cfg: Mapping[str, Any],
    svd_model,
    sasrec_model: SASRec | None,
    reward_epsilon: float,
    allow_default_rating: bool = True,
):
    name = str(name).lower()
    if name in {"rating", "ratings"}:
        if allow_default_rating:
            return None
        from flowrec.rewards.rating import RatingRewardConfig, RatingRewardModel

        reward_data = {"reward_epsilon": reward_epsilon, **dict(reward_cfg)}
        return RatingRewardModel(RatingRewardConfig(**reward_data))

    if name in {"mix", "mixed", "mixture", "sum"}:
        components = reward_components(reward_cfg)
        default_normalizer = str(reward_cfg.get("normalize", reward_cfg.get("normalization", "batch_max")))
        models: list[torch.nn.Module] = []
        weights: list[float] = []
        normalizers: list[str] = []
        for component in components:
            models.append(
                build_reward_model(
                    component_reward_name(component),
                    data=data,
                    train_loader=train_loader,
                    config_path=config_path,
                    seed=seed,
                    reward_cfg=component_reward_config(component),
                    svd_cfg=svd_cfg,
                    svd_model=svd_model,
                    sasrec_model=sasrec_model,
                    reward_epsilon=reward_epsilon,
                    allow_default_rating=False,
                )
            )
            weights.append(float(component.get("weight", 1.0)))
            normalizers.append(str(component.get("normalize", component.get("normalization", default_normalizer))))
        return MixedRewardModel(
            models,
            weights=weights,
            normalizers=normalizers,
            reward_epsilon=reward_epsilon,
        )

    if name in {"multiply", "multiplicative", "product"}:
        components = reward_components(reward_cfg)
        models = []
        weights = []
        for component in components:
            models.append(
                build_reward_model(
                    component_reward_name(component),
                    data=data,
                    train_loader=train_loader,
                    config_path=config_path,
                    seed=seed,
                    reward_cfg=component_reward_config(component),
                    svd_cfg=svd_cfg,
                    svd_model=svd_model,
                    sasrec_model=sasrec_model,
                    reward_epsilon=reward_epsilon,
                    allow_default_rating=False,
                )
            )
            weights.append(float(component.get("weight", 1.0)))
        return MultiplicativeRewardModel(
            models,
            weights=weights,
            reward_epsilon=reward_epsilon,
        )

    mapping = item_id_by_semantic_row(data)
    reward_data = {"reward_epsilon": reward_epsilon, **dict(reward_cfg)}

    if name in {"similarity", "semantic_similarity"}:
        from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel

        return SimilarityRewardModel(SimilarityRewardConfig(**reward_data))

    if name == "svd":
        from flowrec.rewards.svd import SVDRewardConfig, SVDRewardModel

        if svd_model is None:
            svd_model = build_svd_scoring_model(
                data=data,
                train_loader=train_loader,
                seed=seed,
                svd_cfg=svd_cfg,
            )
        return SVDRewardModel(
            svd_model,
            SVDRewardConfig(**reward_data),
            item_id_by_semantic_row=mapping,
        )

    if name == "sasrec":
        if sasrec_model is None:
            raise ValueError("SASRec reward requires a loaded SASRec teacher model.")
        return SASRecRewardModel(
            sasrec_model,
            SASRecRewardConfig(**reward_data),
            item_id_by_semantic_row=mapping,
        )

    if name in {"amazon_log", "amazonlog"}:
        from flowrec.rewards.amazon_log import (
            AmazonLogRewardConfig,
            AmazonLogRewardModel,
            compute_item_mean_ratings,
        )

        log_cfg = AmazonLogRewardConfig(**reward_data)
        item_mean_rating = None
        global_mean_rating = None
        if log_cfg.reward_mode == "item_centered":
            interactions = getattr(data, "interactions", None)
            if interactions is None or len(interactions) == 0:
                raise ValueError(
                    "amazon_log reward with reward_mode='item_centered' needs "
                    "data.interactions to compute item mean ratings. "
                    "Call data.setup() first."
                )
            # Item means come from the full interaction log; shrinkage keeps
            # single-rating (incl. held-out) contributions small.
            item_mean_rating, global_mean_rating = compute_item_mean_ratings(
                interactions["item_id"].to_numpy(),
                interactions["rating"].to_numpy(dtype="float32"),
                num_items=int(data.item_vocab_size),
                shrinkage=log_cfg.item_mean_shrinkage,
            )
        return AmazonLogRewardModel(
            log_cfg,
            item_id_by_semantic_row=mapping,
            item_mean_rating=item_mean_rating,
            global_mean_rating=global_mean_rating,
        )

    raise ValueError(
        f"Unknown method.reward_model={name!r}. "
        "Expected rating, similarity, svd, sasrec, amazon_log, mix, or multiply."
    )


def build_candidate_sampler(
    name: str,
    *,
    model: GPT2RecModel,
    data: AmazonReviewDataModule | YambdaDataModule,
    svd_model,
    sasrec_model: SASRec | None,
    method_cfg: Mapping[str, Any],
):
    if name in {"random", "generated"}:
        return None

    if name in {"sasrec", "sasrec_score", "model_score", "item_model"}:
        if sasrec_model is None:
            raise ValueError("method.sasrec.checkpoint_path is required for SASRec candidate sampling.")
        item_id_to_row = restrict_item_id_mapping_to_scorer_vocab(
            item_id_to_semantic_row(data),
            sasrec_model,
        )
        return ItemModelSemanticIDCandidateSampler(
            sasrec_model,
            item_semantic_ids=model.item_semantic_ids.detach().cpu(),
            item_id_to_semantic_row=item_id_to_row,
            cfg=ItemModelSemanticCandidateConfig(
                num_negative_samples=int(method_cfg.get("num_negative_samples", 1)),
                include_positive=bool(method_cfg.get("include_positive", True)),
                avoid_positive=bool(method_cfg.get("avoid_positive", True)),
                oversample_factor=int(method_cfg.get("sasrec_oversample_factor", 4)),
                easy_negative_steps=int(method_cfg.get("easy_negative_steps", 0)),
                exclude_history=bool(method_cfg.get("exclude_history_in_sampling", True)),
                keep_scorer_eval=bool(method_cfg.get("keep_scorer_eval", True)),
                sampling_temperature=float(method_cfg.get("sampling_temperature", 1.0)),
                candidate_pool_size=int(method_cfg.get("candidate_pool_size", 1024)),
                score_chunk_size=int(method_cfg.get("score_chunk_size", 50_000)),
                pool_strategy=str(method_cfg.get("pool_strategy", "random_pool")),
                sampler_name="sasrec_semantic_id",
            ),
        )

    if name == "svd":
        if svd_model is None:
            raise ValueError("SVD candidate sampling requires method.svd config.")
        item_id_to_row = item_id_to_semantic_row(data)
        return ItemModelSemanticIDCandidateSampler(
            svd_model,
            item_semantic_ids=model.item_semantic_ids.detach().cpu(),
            item_id_to_semantic_row=item_id_to_row,
            cfg=ItemModelSemanticCandidateConfig(
                num_negative_samples=int(method_cfg.get("num_negative_samples", 1)),
                include_positive=bool(method_cfg.get("include_positive", True)),
                avoid_positive=bool(method_cfg.get("avoid_positive", True)),
                easy_negative_steps=int(method_cfg.get("easy_negative_steps", 0)),
                exclude_history=bool(method_cfg.get("exclude_history_in_sampling", True)),
                keep_scorer_eval=bool(method_cfg.get("keep_scorer_eval", True)),
                sampling_temperature=float(method_cfg.get("sampling_temperature", 1.0)),
                candidate_pool_size=int(method_cfg.get("candidate_pool_size", 1024)),
                score_chunk_size=int(method_cfg.get("score_chunk_size", 50_000)),
                pool_strategy=str(method_cfg.get("pool_strategy", "random_pool")),
                sampler_name="svd_semantic_id",
            ),
        )

    raise ValueError(
        f"Unknown method.candidate_sampler={name!r}. "
        "Expected random, generated, sasrec, sasrec_score, model_score, item_model, or svd."
    )


def build_sasrec_teacher_model(
    data: AmazonReviewDataModule | YambdaDataModule,
    sasrec_cfg: Mapping[str, Any],
    *,
    config_path: Path,
) -> SASRec:
    checkpoint = sasrec_cfg.get("checkpoint_path")
    if checkpoint is None:
        raise ValueError("method.sasrec.checkpoint_path is required for SASRec reward/sampling.")
    model_data = dict(sasrec_cfg.get("model", {}))
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(model_data.get("item_vocab_size", data.item_vocab_size)),
            max_history_len=int(model_data.get("max_history_len", getattr(data.cfg, "max_history_len", 20))),
            d_model=int(model_data.get("d_model", model_data.get("embedding_dim", 64))),
            nhead=int(model_data.get("nhead", model_data.get("num_heads", 2))),
            num_layers=int(model_data.get("num_layers", 2)),
            dropout=float(model_data.get("dropout", 0.3)),
            pos_drop=float(model_data.get("pos_drop", model_data.get("dropout", 0.3))),
        )
    )
    model.pad_item_id = int(model_data.get("pad_item_id", 0))
    load_model_state_dict(model, resolve_experiment_path(checkpoint, config_path=config_path))
    return model


def restrict_item_id_mapping_to_scorer_vocab(mapping: torch.Tensor, scorer_model) -> torch.Tensor:
    item_vocab_size = getattr(scorer_model, "item_vocab_size", None)
    if item_vocab_size is None:
        cfg = getattr(scorer_model, "cfg", None)
        item_vocab_size = getattr(cfg, "item_vocab_size", None)
    if item_vocab_size is None:
        return mapping
    limit = int(item_vocab_size)
    if limit >= mapping.numel():
        return mapping
    restricted = mapping.clone()
    restricted[max(limit, 0):] = -1
    return restricted


def item_id_by_semantic_row(data: AmazonReviewDataModule | YambdaDataModule) -> torch.Tensor | None:
    semantic_ids = getattr(data, "semantic_ids", None)
    item2id = getattr(data, "item2id", None)
    if not semantic_ids or not item2id:
        return None
    return torch.tensor(
        [int(item2id.get(raw_item_id, -1)) for raw_item_id in sorted(semantic_ids)],
        dtype=torch.long,
    )


def item_id_to_semantic_row(data: AmazonReviewDataModule | YambdaDataModule) -> torch.Tensor:
    semantic_ids = getattr(data, "semantic_ids", None)
    item2id = getattr(data, "item2id", None)
    if not semantic_ids or not item2id:
        return torch.empty(0, dtype=torch.long)
    out = torch.full((int(data.item_vocab_size),), -1, dtype=torch.long)
    for row, raw_item_id in enumerate(sorted(semantic_ids)):
        item_id = item2id.get(raw_item_id)
        if item_id is not None and 0 <= int(item_id) < out.numel():
            out[int(item_id)] = int(row)
    return out


def load_model_state_dict(model: torch.nn.Module, path: Path, *, strict: bool = True) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        if "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=strict)


def resolve_experiment_path(value: str | Path, *, config_path: Path) -> Path:
    base = config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT
    return resolve_path(value, base=base)


def resolved_method_config(method: FlowRecMethod, config: Mapping[str, Any]) -> dict[str, Any]:
    out = method.cfg.to_dict()
    requested = dict(config.get("method", {}))
    for key in (
        "reward_model",
        "reward",
        "svd",
        "sasrec",
        "easy_negative_steps",
        "exclude_history_in_sampling",
        "sasrec_oversample_factor",
    ):
        if key in requested:
            out[key] = requested[key]
    return out


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
        "early_stopping_mode",
        "min_delta",
        "prefix_method_metrics_with_stage",
        "max_train_batches",
        "max_eval_batches",
        "val_sample_fraction",
        "val_sample_seed",
        "val_sample_resample",
    }
    trainer_data = {key: value for key, value in training.items() if key in allowed}
    trainer_data["checkpoint_dir"] = checkpoint_dir
    return TrainerConfig.from_dict(trainer_data)


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


def log_final_summary(
    logger: CompositeExperimentLogger,
    summary: Mapping[str, Any],
    *,
    step: int,
) -> None:
    payload: dict[str, Any] = {
        "type": "final_summary",
        "global_step": step,
        "epochs_completed": summary.get("epochs_completed"),
        "steps_completed": summary.get("steps_completed"),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
    }
    for group in ("validation_metrics", "test_metrics"):
        metrics = summary.get(group)
        if isinstance(metrics, Mapping):
            payload.update(metrics)
    logger(payload)


def log_run_artifacts(logger: CompositeExperimentLogger, run_dir: Path, *, step: int) -> None:
    for name in ("resolved_config.json", "cli_args.json", "final_summary.json", "final_summary.md"):
        logger.log_asset(run_dir / name, file_name=name, step=step)


def close_logger(logger: CompositeExperimentLogger) -> None:
    logger.close()


def create_run_dir(
    experiments_dir: Path,
    run_name: str,
    config: Mapping[str, Any],
    *,
    plain: bool = False,
) -> Path:
    experiments_dir = resolve_path(experiments_dir, base=PROJECT_ROOT)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:80] or "run"
    if plain:
        run_dir = experiments_dir / safe_name
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        run_dir = experiments_dir / f"{stamp}_{safe_name[:48]}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "run.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
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
        "# GPT-2 Run Summary",
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


if __name__ == "__main__":
    main()
