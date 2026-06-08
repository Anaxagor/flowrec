from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import sys
from dataclasses import asdict, replace
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
from flowrec.candidates.item_model import (
    ItemModelSemanticCandidateConfig,
    ItemModelSemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import FlowRecMethod
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import FlowRecModel
from flowrec.models.gpt2_semantic import GPT2SemanticModel
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.models.tiger import TigerModel
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
from flowrec.training.loggers import (
    CompositeExperimentLogger,
    CometExperimentLogger,
    JsonlExperimentLogger,
    to_jsonable,
)
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
    init_checkpoint_path = resolve_init_checkpoint_path(config, config_path=config_path)
    if init_checkpoint_path is not None:
        strict = bool(config.get("training", {}).get("init_checkpoint_strict", True))
        LOGGER.info("Initializing model from checkpoint: %s", init_checkpoint_path)
        load_model_state_dict(model, init_checkpoint_path, strict=strict)

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
        "scheduler": config.get("scheduler", config.get("lr_scheduler", {})),
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
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

    scheduler_factory = build_scheduler_factory(config, trainer_cfg, train_loader)

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
            scheduler=scheduler_factory,
            logger=experiment_logger,
        )

        fit_result = trainer.fit(train_loader, val_loader)
        test_metrics: dict[str, float] = {}
        checkpoint_for_test = fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        if checkpoint_for_test is not None:
            trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)
            test_generation_num_beams = config.get("training", {}).get("test_generation_num_beams")
            old_generation_cfg = override_generation_num_beams(method, test_generation_num_beams)
            try:
                if test_generation_num_beams is not None:
                    LOGGER.info("Using test_generation_num_beams=%s for final test.", test_generation_num_beams)
                test_metrics = trainer.test(test_loader)
            finally:
                restore_generation_config(method, old_generation_cfg)

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
        log_final_summary(experiment_logger, summary, step=int(trainer.state.global_step))
        log_run_artifacts(experiment_logger, run_dir, step=int(trainer.state.global_step))

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
    preprocessing_preset = dataset.get("preprocessing_preset", "default")
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
            max_history_len=int(dataset.get("max_history_len", max(1, max_sequence_length - 1))),
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
        preprocessing_preset=preprocessing_preset,
        min_user_interactions=int(dataset.get("min_user_interactions", 5)),
        min_item_interactions=int(dataset.get("min_item_interactions", 0)),
        use_verified_only=bool(dataset.get("use_verified_only", False)),
        keep_raw_reviews=bool(dataloader.get("include_raw_in_batch", False)),
        deduplicate_consecutive=bool(dataset.get("deduplicate_consecutive", False)),
        deduplicate_user_items=bool(dataset.get("deduplicate_user_items", False)),
        deduplicate_user_items_keep=dataset.get("deduplicate_user_items_keep", "first"),
        split_strategy=dataset.get("split_strategy", "leave_one_out"),
        min_history_len=int(dataset.get("min_history_len", 1)),
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
        max_history_len=int(dataset.get("max_history_len", max(1, max_sequence_length - 1))),
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
        offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
        drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
        user_hash_mod=int(model["user_ids_count"]) if model.get("user_ids_count") is not None else None,
        filter_cold_items_after_split=bool(
            dataset.get(
                "filter_cold_items_after_split",
                False if preprocessing_preset == "letter_amazon18" else True,
            )
        ),
        cold_item_train_source=dataset.get("cold_item_train_source", "observed"),
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
) -> FlowRecModel:
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    num_codebooks = int(dataset_cfg.get("num_codebooks", data.sid_len or 0))
    if data.sid_len != num_codebooks:
        LOGGER.warning(
            "Configured num_codebooks=%s but semantic IDs have length=%s; using semantic ID length.",
            num_codebooks,
            data.sid_len,
        )
    top_k = int(model_cfg.get("top_k", model_cfg.get("num_return_sequences", 20)))
    architecture = str(
        model_cfg.get("architecture", model_cfg.get("model_type", model_cfg.get("type", "tiger")))
    ).lower()

    if architecture in {"gpt2", "gpt2_semantic", "decoder_only", "causal_lm"}:
        return GPT2SemanticModel(
            codebook_size=int(model_cfg.get("codebook_size", 256)),
            sem_id_len=int(data.sid_len or num_codebooks),
            user_ids_count=int(model_cfg.get("user_ids_count", 2000)),
            semantic_id_mapping=data.semantic_ids,
            embedding_dim=int(model_cfg.get("embedding_dim", model_cfg.get("n_embd", 384))),
            num_heads=int(model_cfg.get("num_heads", model_cfg.get("n_head", 6))),
            num_layers=int(
                model_cfg.get(
                    "num_layers",
                    model_cfg.get("num_decoder_layers", model_cfg.get("n_layer", 4)),
                )
            ),
            dim_feedforward=int(model_cfg.get("dim_feedforward", model_cfg.get("n_inner", 1024))),
            dropout=float(model_cfg.get("dropout", 0.1)),
            num_beams=int(model_cfg.get("num_beams", 100)),
            num_return_sequences=top_k,
            max_history_len=int(dataset_cfg.get("max_history_len", dataset_cfg.get("max_sequence_length", 20))),
            user_token_position=str(model_cfg.get("user_token_position", "prefix")),
            tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", True)),
            train_eos_token=bool(model_cfg.get("train_eos_token", True)),
            layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-5)),
            activation_function=str(model_cfg.get("activation_function", "gelu_new")),
            initializer_range=float(model_cfg.get("initializer_range", 0.02)),
            max_position_embeddings=(
                int(model_cfg["max_position_embeddings"])
                if "max_position_embeddings" in model_cfg
                else None
            ),
        )

    if architecture not in {"tiger", "t5", "t5_tiger"}:
        raise ValueError(f"Unknown model architecture: {architecture!r}.")

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
        user_token_position=str(model_cfg.get("user_token_position", "suffix")),
        tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", False)),
        train_eos_token=bool(model_cfg.get("train_eos_token", False)),
        d_kv=int(model_cfg.get("d_kv", 64)),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-6)),
        activation=str(model_cfg.get("activation", "relu")),
        initializer_range=float(model_cfg.get("initializer_range", 0.02)),
    )


def build_method(
    config: Mapping[str, Any],
    model: FlowRecModel,
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

    if method_name == "sft":
        sft_cfg = SFTConfig(
            compute_item_metrics=False,
            evaluate_generation=bool(training.get("evaluate_generation", True)),
            generation_topk=tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
            generation_num_beams=generation_num_beams,
            generation_num_return_sequences=top_k,
            generation_constrained=bool(training.get("generation_constrained", True)),
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
        "generation_topk": tuple(training.get("generation_topk", [5, 10, min(20, top_k)])),
        "generation_num_beams": generation_num_beams,
        "generation_num_return_sequences": top_k,
        "generation_constrained": bool(training.get("generation_constrained", True)),
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
    candidate_sampler = build_candidate_sampler(
        candidate_sampler_name,
        model=model,
        data=data,
        svd_model=svd_model,
        sasrec_model=sasrec_model,
        method_cfg=sampler_cfg,
    )
    return GFlowMethod(gflow_cfg, reward_model=reward_model, candidate_sampler=candidate_sampler)


def override_generation_num_beams(
    method: FlowRecMethod,
    num_beams: Any,
) -> tuple[Any, Any] | None:
    if num_beams is None:
        return None

    beams = int(num_beams)
    if isinstance(method, SFTMethod):
        old_cfg = method.cfg
        method.cfg = replace(method.cfg, generation_num_beams=beams)
        return old_cfg, None

    if isinstance(method, GFlowMethod):
        old_cfg = method.cfg
        old_sft_cfg = method.sft_method.cfg
        method.cfg = replace(method.cfg, generation_num_beams=beams)
        method.sft_method.cfg = replace(method.sft_method.cfg, generation_num_beams=beams)
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

    def forward(self, batch: Batch, trajectories) -> torch.Tensor:
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
        raise ValueError(
            f"Unknown reward normalizer {mode!r}. "
            "Expected none, batch_max, batch_mean, or batch_minmax."
        )


def reward_config_requires_model(name: str, reward_cfg: Mapping[str, Any], model_name: str) -> bool:
    normalized_name = str(name).lower()
    if normalized_name == model_name:
        return True
    if normalized_name not in {"mix", "mixed", "mixture", "sum"}:
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

    raise ValueError(
        f"Unknown method.reward_model={name!r}. "
        "Expected rating, similarity, svd, sasrec, or mix."
    )


def build_candidate_sampler(
    name: str,
    *,
    model: TigerModel,
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
        item_id_to_row = item_id_to_semantic_row(data)
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
            item_vocab_size=int(data.item_vocab_size),
            max_history_len=int(getattr(data.cfg, "max_history_len", 20)),
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


def resolve_init_checkpoint_path(config: Mapping[str, Any], *, config_path: Path) -> Path | None:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        return None
    checkpoint = training.get("init_checkpoint_path")
    if checkpoint is None:
        return None
    return resolve_experiment_path(checkpoint, config_path=config_path)


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
        "keep_scorer_eval",
        "sampling_temperature",
        "candidate_pool_size",
        "score_chunk_size",
        "pool_strategy",
    ):
        if key in requested:
            out[key] = requested[key]
    return out


def build_scheduler_factory(
    config: Mapping[str, Any],
    trainer_cfg: TrainerConfig,
    train_loader,
):
    scheduler_cfg = dict(config.get("scheduler", config.get("lr_scheduler", {})) or {})
    scheduler_type = str(
        scheduler_cfg.get("type", scheduler_cfg.get("name", "none"))
    ).lower()
    if scheduler_type in {"", "none", "constant"}:
        return None

    try:
        num_batches = len(train_loader)
    except TypeError:
        num_batches = None
    if not num_batches:
        raise ValueError("A finite train_loader length is required for scheduled TIGER training.")

    accumulation = max(1, int(trainer_cfg.gradient_accumulation_steps))
    steps_per_epoch = max(1, math.ceil(int(num_batches) / accumulation))
    if trainer_cfg.max_steps is not None:
        total_steps = max(1, math.ceil(int(trainer_cfg.max_steps) / accumulation))
    else:
        total_steps = max(1, steps_per_epoch * int(trainer_cfg.max_epochs))

    warmup_steps = scheduler_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(float(scheduler_cfg.get("warmup_ratio", 0.0)) * total_steps)
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    if trainer_cfg.scheduler_step_on == "none":
        LOGGER.warning(
            "Scheduler %s was configured but training.scheduler_step_on='none'; "
            "scheduler will not step unless scheduler_step_on is changed.",
            scheduler_type,
        )

    LOGGER.info(
        "Using %s LR scheduler: total_optimizer_steps=%s warmup_steps=%s steps_per_epoch=%s.",
        scheduler_type,
        total_steps,
        warmup_steps,
        steps_per_epoch,
    )

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if warmup_steps > 0 and current_step <= warmup_steps:
            return max(1e-8, current_step / warmup_steps)

        if scheduler_type == "cosine":
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (current_step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        if scheduler_type in {"inverse_sqrt", "inv_sqrt", "inverse-square-root"}:
            if warmup_steps <= 0:
                constant_steps = int(scheduler_cfg.get("constant_steps", 0))
                if current_step <= constant_steps:
                    return 1.0
                return max(min_lr_ratio, math.sqrt(max(1, constant_steps or 1) / current_step))
            return max(min_lr_ratio, math.sqrt(warmup_steps / current_step))

        raise ValueError(
            f"Unknown scheduler type {scheduler_type!r}. Expected cosine, inverse_sqrt, or none."
        )

    def factory(optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return factory


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


if __name__ == "__main__":
    main()
