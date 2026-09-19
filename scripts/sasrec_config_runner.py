from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import yaml

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


LOGGER = logging.getLogger("flowrec.sasrec_config_runner")
DEFAULT_RANKING_KS = (5, 10, 20)

DataFactory = Callable[[str, int], Any]


def run_sweep(
    *,
    dataset_names: list[str],
    seeds: list[int],
    data_factory: DataFactory,
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    config_dir: Path,
    project_root: Path,
    args: argparse.Namespace,
    on_run_complete: Callable[[Mapping[str, Any]], None] | None = None,
    variant: str | None = None,
) -> list[Mapping[str, Any]]:
    checkpoint_root = configured_path(
        trainer_config, "checkpoints_dir", project_root=project_root
    )
    experiments_root = configured_path(
        trainer_config, "experiments_dir", project_root=project_root
    )
    artifacts_root = configured_path(
        trainer_config, "artifacts_dir", project_root=project_root
    )
    summary_name_template = str(
        trainer_config.get("metrics_summary_name", "sasrec_metrics_summary.csv")
    )
    ranking_ks = resolve_ranking_ks(trainer_config)

    summaries: list[Mapping[str, Any]] = []
    summaries_by_dataset: dict[str, list[Mapping[str, Any]]] = {
        dataset_name: [] for dataset_name in dataset_names
    }
    for dataset_name in dataset_names:
        for seed in seeds:
            data = data_factory(dataset_name, seed)
            summary = run_experiment(
                dataset_name,
                seed=seed,
                data=data,
                dataset_config=dataset_config,
                model_config=model_config,
                trainer_config=trainer_config,
                config_dir=config_dir,
                checkpoint_root=checkpoint_root,
                experiments_root=experiments_root,
                ranking_ks=ranking_ks,
                args=args,
                variant=variant,
            )
            summaries.append(summary)
            summaries_by_dataset[dataset_name].append(summary)
            # Preserve completed results if a later dataset/seed is interrupted.
            metrics_summary_path = dataset_metrics_summary_path(
                artifacts_root,
                summary_name_template,
                dataset_name=dataset_name,
            )
            write_metrics_summary(
                metrics_summary_path,
                summaries_by_dataset[dataset_name],
                ks=ranking_ks,
            )
            write_training_summary(
                checkpoint_root / safe_component(dataset_name) / "training_summary.json",
                config_dir=config_dir,
                dataset_names=[dataset_name],
                seeds=seeds,
                ranking_ks=ranking_ks,
                metrics_summary_path=metrics_summary_path,
                summaries=summaries_by_dataset[dataset_name],
            )
            if on_run_complete is not None:
                on_run_complete(summary)
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # A non-templated name intentionally represents one combined sweep. Do not
    # create this shared file for templated per-dataset jobs, which may run
    # concurrently on separate cluster nodes.
    if "{dataset}" not in summary_name_template:
        metrics_summary_path = artifacts_root / summary_name_template
        write_metrics_summary(metrics_summary_path, summaries, ks=ranking_ks)
        write_training_summary(
            checkpoint_root / "training_summary.json",
            config_dir=config_dir,
            dataset_names=dataset_names,
            seeds=seeds,
            ranking_ks=ranking_ks,
            metrics_summary_path=metrics_summary_path,
            summaries=summaries,
        )
    for dataset_name in dataset_names:
        LOGGER.info(
            "[%s] saved validation/test metrics summary to %s",
            dataset_name,
            dataset_metrics_summary_path(
                artifacts_root,
                summary_name_template,
                dataset_name=dataset_name,
            ),
        )
    return summaries


def run_experiment(
    dataset_name: str,
    *,
    seed: int,
    data: Any,
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    config_dir: Path,
    checkpoint_root: Path,
    experiments_root: Path,
    ranking_ks: tuple[int, ...],
    args: argparse.Namespace,
    variant: str | None = None,
) -> Mapping[str, Any]:
    seed_everything(seed)
    # "auto" SCE bucket sizes depend on this dataset's batch size and window.
    model_config = resolve_sce_bucket_sizes(model_config, data)

    run_name = f"sasrec_{safe_component(dataset_name)}_seed-{seed}"
    full_config = {
        "dataset_name": dataset_name,
        "variant": variant,
        "seed": seed,
        "dataset": dict(dataset_config),
        "model": dict(model_config),
        "trainer": dict(trainer_config),
    }
    run_dir = create_run_dir(experiments_root, run_name, full_config)
    setup_logging(run_dir)
    checkpoint_dir = checkpoint_root / safe_component(dataset_name) / f"seed_{seed}"

    data.setup()
    if (
        bool(trainer_config.get("save_interaction_store", False))
        and data.interaction_store is not None
        and len(data.interaction_store)
    ):
        data.interaction_store.save(run_dir / "interaction_store")

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = build_model(model_config, data)
    if data.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support "
            f"train_example_layout={data.cfg.train_example_layout!r}; "
            f"supported: {model.supported_train_layouts}."
        )
    method = build_method(model_config)
    train_cfg = build_trainer_config(trainer_config, checkpoint_dir, args)
    device = resolve_device(args.device or trainer_config.get("device", "cuda"))

    resolved = {
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "variant": variant,
        "seed": seed,
        "device": device,
        "data": asdict(data.cfg),
        "data_summary": data.summary(),
        "model": model.config_dict(),
        "method": method.cfg.to_dict(),
        "trainer": train_cfg.to_dict(),
        "optimizer": optimizer_config(trainer_config),
        "ranking_ks": list(ranking_ks),
    }
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "cli_args.json", vars(args))

    logger = build_experiment_logger(
        trainer_config,
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_dir,
        resolved_config=resolved,
        cli_args=vars(args),
    )
    try:
        LOGGER.info("Dataset: %s", dataset_name)
        LOGGER.info("Run directory: %s", run_dir)
        LOGGER.info("Checkpoint directory: %s", checkpoint_dir)
        LOGGER.info("Data summary: %s", json.dumps(data.summary(), sort_keys=True))
        LOGGER.info(
            "Model parameters: total=%s trainable=%s",
            f"{model.num_parameters():,}",
            f"{model.num_parameters(trainable_only=True):,}",
        )

        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=train_cfg,
            device=device,
            **optimizer_config(trainer_config),
            logger=logger,
        )
        fit_result = trainer.fit(train_loader, val_loader)

        checkpoint_for_eval = (
            fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        )
        if checkpoint_for_eval is not None:
            trainer.load_checkpoint(
                checkpoint_for_eval,
                load_optimizer=False,
                load_scheduler=False,
            )

        popularity = training_item_popularity(data)
        evaluation_kwargs = {
            "device": trainer.device,
            "item_popularity": popularity,
            "num_catalog_items": int(data.num_items),
            "ks": ranking_ks,
            "pad_item_id": int(data.cfg.pad_item_id),
            "exclude_history": bool(model.cfg.exclude_history_in_generate),
        }
        validation_metrics = evaluate_ranking_metrics(
            model, val_loader, **evaluation_kwargs
        )
        test_metrics = evaluate_ranking_metrics(
            model, test_loader, **evaluation_kwargs
        )

        summary = {
            "seed": seed,
            "dataset": {
                "name": dataset_name,
                **data_config_paths(data.cfg),
                **data.summary(),
            },
            "model": model.config_dict(),
            "epochs_completed": int(trainer.state.epoch + 1),
            "steps_completed": int(trainer.state.global_step),
            "best_metric": trainer.state.best_metric,
            "best_epoch": trainer.state.best_epoch,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "optimization_validation_metrics": fit_result.val_metrics,
            "best_checkpoint": fit_result.best_checkpoint_path,
            "last_checkpoint": fit_result.last_checkpoint_path,
            "experiment_dir": str(run_dir),
        }
        write_json(run_dir / "final_summary.json", summary)
        write_text_summary(run_dir / "final_summary.md", summary)
        logger(
            {
                "type": "final_summary",
                "global_step": int(trainer.state.global_step),
                **summary_to_metrics(summary),
            }
        )
        for name in (
            "resolved_config.json",
            "cli_args.json",
            "final_summary.json",
            "final_summary.md",
        ):
            logger.log_asset(
                run_dir / name,
                file_name=name,
                step=int(trainer.state.global_step),
            )
        LOGGER.info("Training finished: %s", json.dumps(to_jsonable(summary), sort_keys=True))
        return summary
    finally:
        logger.close()


# Objective variants. ``variants: {name: {sft: ...}}`` in the SASRec model YAML
# overrides only the SFT objective; ``variants: {name: {checkpoints_dir, ...}}``
# in each trainer YAML relocates that variant's outputs. The exp1 scripts select
# one with ``--variant``, so e.g. scalable cross-entropy trains every dataset
# with the baseline's architecture and hyperparameters while never overwriting
# the full-CE checkpoints and metrics.
VARIANTS_KEY = "variants"
MODEL_VARIANT_KEYS = frozenset({"sft"})
TRAINER_VARIANT_OUTPUT_KEYS = ("checkpoints_dir", "experiments_dir", "artifacts_dir")
TRAINER_VARIANT_OPTIONAL_KEYS = frozenset({"metrics_summary_name"})
SCE_AUTO = "auto"


def available_variants(config: Mapping[str, Any]) -> list[str]:
    variants = config.get(VARIANTS_KEY) or {}
    if not isinstance(variants, Mapping):
        raise TypeError(f"{VARIANTS_KEY!r} must map variant names to override blocks.")
    return sorted(str(name) for name in variants)


def _variant_profile(
    config: Mapping[str, Any],
    variant: str | None,
    *,
    source: str,
) -> Mapping[str, Any] | None:
    if variant is None:
        return None
    available = available_variants(config)
    if variant not in available:
        raise KeyError(
            f"{source} does not define variant {variant!r}; available: {available}."
        )
    profile = config[VARIANTS_KEY][variant] or {}
    if not isinstance(profile, Mapping):
        raise TypeError(f"{source} variant {variant!r} must be a mapping.")
    return profile


def resolve_model_variant(
    model_config: Mapping[str, Any],
    variant: str | None,
) -> dict[str, Any]:
    """Apply the model YAML's ``variants[variant].sft`` block and drop ``variants``."""
    resolved = {key: value for key, value in model_config.items() if key != VARIANTS_KEY}
    profile = _variant_profile(model_config, variant, source="The SASRec model config")
    if profile is None:
        return resolved
    unknown = sorted(set(profile) - MODEL_VARIANT_KEYS)
    if unknown:
        raise KeyError(
            f"Model variant {variant!r} may only override {sorted(MODEL_VARIANT_KEYS)}; "
            f"got {unknown}. The architecture must stay the baseline's."
        )
    sft = dict(resolved.get("sft") or {})
    for key, value in dict(profile.get("sft") or {}).items():
        if isinstance(value, Mapping) and isinstance(sft.get(key), Mapping):
            sft[key] = {**sft[key], **value}
        else:
            sft[key] = value
    resolved["sft"] = sft
    return resolved


def apply_trainer_variant(
    trainer_config: Mapping[str, Any],
    variant: str | None,
) -> dict[str, Any]:
    """Relocate outputs to the trainer YAML's ``variants[variant]`` folders.

    Only output locations may change: every hyperparameter is kept, and each
    folder must differ from the baseline's so the baseline is never overwritten.
    """
    resolved = {key: value for key, value in trainer_config.items() if key != VARIANTS_KEY}
    profile = _variant_profile(trainer_config, variant, source="The SASRec trainer config")
    if profile is None:
        return resolved
    allowed = set(TRAINER_VARIANT_OUTPUT_KEYS) | TRAINER_VARIANT_OPTIONAL_KEYS
    unknown = sorted(set(profile) - allowed)
    if unknown:
        raise KeyError(
            f"Trainer variant {variant!r} may only relocate outputs "
            f"{sorted(allowed)}; got {unknown}. Hyperparameters must stay the baseline's."
        )
    missing = [key for key in TRAINER_VARIANT_OUTPUT_KEYS if key not in profile]
    if missing:
        raise KeyError(
            f"Trainer variant {variant!r} must define {list(TRAINER_VARIANT_OUTPUT_KEYS)}; "
            f"missing {missing}."
        )
    for key in TRAINER_VARIANT_OUTPUT_KEYS:
        if str(profile[key]) == str(resolved.get(key)):
            raise ValueError(
                f"Trainer variant {variant!r} keeps the baseline {key}={profile[key]!r}; "
                "variants must write to separate folders."
            )
    resolved.update(dict(profile))
    return resolved


def sce_bucket_count(batch_size: int, window_len: int) -> int:
    """Paper setting for SCE (Mezentsev et al., RecSys'24):
    ``n_buckets = bucket_size_x = int(2 * sqrt(batch_size * window_len))``."""
    return max(1, int(2 * math.sqrt(int(batch_size) * int(window_len))))


def resolve_sce_bucket_sizes(
    model_config: Mapping[str, Any],
    data: Any,
) -> dict[str, Any]:
    """Replace ``"auto"`` SCE bucket sizes with the paper formula for this dataset."""
    sft = dict(model_config.get("sft") or {})
    if str(sft.get("item_loss", "cross_entropy")).lower() != "scalable_cross_entropy":
        return dict(model_config)
    sce = dict(sft.get("scalable_cross_entropy") or {})
    auto_keys = [
        key
        for key, value in sce.items()
        if isinstance(value, str) and value.strip().lower() == SCE_AUTO
    ]
    if not auto_keys:
        return dict(model_config)
    unsupported = sorted(set(auto_keys) - {"n_buckets", "bucket_size_x"})
    if unsupported:
        raise ValueError(
            f"{SCE_AUTO!r} is only defined for n_buckets and bucket_size_x; got {unsupported}."
        )
    window_len = int(getattr(data.cfg, "train_window_len", None) or data.cfg.max_history_len)
    count = sce_bucket_count(int(data.cfg.batch_size), window_len)
    for key in auto_keys:
        sce[key] = count
    sft["scalable_cross_entropy"] = sce
    return {**model_config, "sft": sft}


def build_model(model_config: Mapping[str, Any], data: Any) -> SASRec:
    max_positions = max(
        int(data.cfg.max_history_len),
        int(data.cfg.train_window_len or 0),
    )
    field_names = {field.name for field in fields(SASRecConfig)}
    unknown = sorted(set(model_config) - field_names - {"sft", VARIANTS_KEY})
    if unknown:
        raise KeyError(f"Unknown SASRec model config keys: {unknown}")
    values = {
        key: value for key, value in model_config.items() if key in field_names
    }
    values.update(
        item_vocab_size=int(data.item_vocab_size),
        max_history_len=max_positions,
    )
    return SASRec(SASRecConfig(**values))


def build_method(model_config: Mapping[str, Any]) -> SFTMethod:
    return SFTMethod(SFTConfig.from_dict(dict(model_config.get("sft", {}))))


def build_trainer_config(
    trainer_config: Mapping[str, Any],
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> TrainerConfig:
    allowed = {field.name for field in fields(TrainerConfig)} - {"checkpoint_dir"}
    runtime_keys = {
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "device",
        "seeds",
        "checkpoints_dir",
        "experiments_dir",
        "artifacts_dir",
        "metrics_summary_name",
        "ranking_ks",
        "save_interaction_store",
        "comet",
        VARIANTS_KEY,
    }
    unknown = sorted(set(trainer_config) - allowed - runtime_keys)
    if unknown:
        raise KeyError(f"Unknown SASRec trainer config keys: {unknown}")

    values = {
        key: value for key, value in trainer_config.items() if key in allowed
    }
    values["use_amp"] = bool(trainer_config.get("use_amp", False)) and torch.cuda.is_available()
    if getattr(args, "max_epochs", None) is not None:
        values["max_epochs"] = int(args.max_epochs)
    if getattr(args, "max_steps", None) is not None:
        values["max_steps"] = int(args.max_steps)
    values["checkpoint_dir"] = checkpoint_dir
    return TrainerConfig.from_dict(values)


def optimizer_config(trainer_config: Mapping[str, Any]) -> dict[str, Any]:
    betas = trainer_config.get("betas", (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"Optimizer betas must contain two values; got {betas!r}.")
    return {
        "lr": float(trainer_config["lr"]),
        "weight_decay": float(trainer_config["weight_decay"]),
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(trainer_config.get("eps", 1e-8)),
    }


def resolve_seeds(
    args: argparse.Namespace,
    trainer_config: Mapping[str, Any],
) -> list[int]:
    configured = (
        args.seeds
        if getattr(args, "seeds", None) is not None
        else trainer_config.get("seeds")
    )
    if configured is None:
        raise KeyError("The trainer config must define seeds.")
    seeds = [int(seed) for seed in configured]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"SASRec training requires three distinct seeds; got {seeds}.")
    return seeds


def resolve_ranking_ks(trainer_config: Mapping[str, Any]) -> tuple[int, ...]:
    values = tuple(int(k) for k in trainer_config.get("ranking_ks", DEFAULT_RANKING_KS))
    if values != DEFAULT_RANKING_KS:
        raise ValueError(
            f"This experiment requires ranking_ks={DEFAULT_RANKING_KS}; got {values}."
        )
    return values


def training_item_popularity(data: Any) -> torch.Tensor:
    """Count item interactions in the training partition without split leakage."""

    frame = data.interactions
    if frame.empty:
        return torch.zeros(data.item_vocab_size, dtype=torch.float64)

    if data.cfg.split_strategy == "leave_one_out":
        user_lengths = frame.groupby("user_id", sort=False)["position"].transform("size")
        train_mask = frame["position"].lt(user_lengths - 2)
    elif data.cfg.split_strategy == "global_time":
        if data.global_val_cutoff is None:
            raise RuntimeError("global_time data is missing its validation cutoff.")
        train_mask = frame["timestamp"].lt(int(data.global_val_cutoff))
    elif data.cfg.split_strategy == "provided":
        # VK-LSVD keeps the source split label on every interaction row.
        if "flowrec_split" not in frame.columns:
            raise RuntimeError("provided-split data is missing its flowrec_split column.")
        train_mask = frame["flowrec_split"].eq("train")
    else:
        raise ValueError(f"Unsupported split_strategy={data.cfg.split_strategy!r}.")

    train_item_ids = (
        frame.loc[train_mask, "raw_item_id"]
        .astype(str)
        .map(data.item2id)
        .dropna()
        .astype(np.int64)
        .to_numpy()
    )
    if train_item_ids.size == 0:
        return torch.zeros(data.item_vocab_size, dtype=torch.float64)

    counts = torch.bincount(
        torch.from_numpy(train_item_ids), minlength=int(data.item_vocab_size)
    ).to(torch.float64)
    counts[int(data.cfg.pad_item_id)] = 0.0
    return counts


@torch.no_grad()
def evaluate_ranking_metrics(
    model: SASRec,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device,
    item_popularity: torch.Tensor,
    num_catalog_items: int,
    ks: tuple[int, ...] = DEFAULT_RANKING_KS,
    pad_item_id: int = 0,
    exclude_history: bool = False,
) -> dict[str, float]:
    """Compute full-catalog HR, NDCG, Coverage, and train-count ARP."""

    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"ks must contain positive integers; got {ks}.")
    if num_catalog_items <= 0:
        raise ValueError("num_catalog_items must be positive.")

    model.eval()
    device_t = torch.device(device)
    popularity = item_popularity.detach().cpu().to(torch.float64)
    item_vocab_size = int(popularity.numel())
    max_k = max(ks)
    if item_vocab_size - 1 < max_k:
        raise ValueError(
            f"Cannot compute top-{max_k} with only {item_vocab_size - 1} "
            "non-padding item IDs."
        )

    total = 0
    hit_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    coverage = {
        k: torch.zeros(item_vocab_size, dtype=torch.bool) for k in ks
    }
    arp_sum = {k: 0.0 for k in ks}
    arp_count = {k: 0 for k in ks}

    for raw_batch in dataloader:
        batch = {
            key: value.to(device_t, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        output = model(batch)
        if output.scores is None:
            raise RuntimeError("SASRec returned no item scores during evaluation.")
        scores = output.scores.clone()
        scores[:, pad_item_id] = -torch.inf

        if exclude_history:
            history = batch["history_item_ids"].long()
            history_mask = batch["history_mask"].bool()
            valid_seen = (
                history_mask
                & history.ge(0)
                & history.lt(item_vocab_size)
                & history.ne(pad_item_id)
            )
            if valid_seen.any():
                rows = torch.arange(scores.size(0), device=device_t).unsqueeze(1)
                scores[rows.expand_as(history)[valid_seen], history[valid_seen]] = -torch.inf
            scores[:, pad_item_id] = -torch.inf

        recommendations = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu()
        targets = batch["target_item_ids"].detach().cpu().long()
        valid_targets = (
            targets.ne(pad_item_id)
            & targets.ge(0)
            & targets.lt(item_vocab_size)
        )
        if not valid_targets.any():
            continue

        recommendations = recommendations[valid_targets]
        targets = targets[valid_targets]
        total += int(targets.numel())

        for k in ks:
            topk = recommendations[:, :k]
            matches = topk.eq(targets.unsqueeze(1))
            hits = matches.any(dim=1)
            hit_sum[k] += float(hits.sum().item())
            ranks = matches.to(torch.float32).argmax(dim=1)
            if hits.any():
                ndcg_sum[k] += float(
                    (1.0 / torch.log2(ranks[hits].to(torch.float32) + 2.0)).sum().item()
                )

            valid_recommendations = (
                topk.ne(pad_item_id)
                & topk.ge(0)
                & topk.lt(item_vocab_size)
            )
            item_ids = topk[valid_recommendations]
            if item_ids.numel():
                coverage[k][item_ids.unique()] = True
                arp_sum[k] += float(popularity[item_ids].sum().item())
                arp_count[k] += int(item_ids.numel())

    if total == 0:
        raise RuntimeError("No valid evaluation examples found.")

    metrics: dict[str, float] = {"num_examples": float(total)}
    for k in ks:
        metrics[f"HR@{k}"] = hit_sum[k] / total
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / total
        metrics[f"Coverage@{k}"] = (
            float(coverage[k].sum().item()) / float(num_catalog_items)
        )
        metrics[f"ARP@{k}"] = arp_sum[k] / max(arp_count[k], 1)
    return metrics


def write_metrics_summary(
    path: Path,
    summaries: Iterable[Mapping[str, Any]],
    *,
    ks: tuple[int, ...] = DEFAULT_RANKING_KS,
) -> None:
    metric_keys = [
        metric
        for k in ks
        for metric in (f"HR@{k}", f"NDCG@{k}", f"Coverage@{k}", f"ARP@{k}")
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        dataset = summary.get("dataset", {})
        dataset_name = dataset.get("name") if isinstance(dataset, Mapping) else None
        for split, group_name in (
            ("validation", "validation_metrics"),
            ("test", "test_metrics"),
        ):
            metrics = summary.get(group_name, {})
            metrics = metrics if isinstance(metrics, Mapping) else {}
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": summary.get("seed"),
                    "split": split,
                    "num_examples": metrics.get("num_examples"),
                    **{key: metrics.get(key) for key in metric_keys},
                    "best_checkpoint": summary.get("best_checkpoint"),
                    "experiment_dir": summary.get("experiment_dir"),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "split",
        "num_examples",
        *metric_keys,
        "best_checkpoint",
        "experiment_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(to_jsonable(row) for row in rows)


def dataset_metrics_summary_path(
    artifacts_root: Path,
    name_template: str,
    *,
    dataset_name: str,
) -> Path:
    safe_dataset = safe_component(dataset_name)
    if "{dataset}" in name_template:
        file_name = name_template.format(dataset=safe_dataset)
    else:
        file_name = name_template
    path = Path(file_name)
    if path.is_absolute() or path.name != file_name:
        raise ValueError(
            "metrics_summary_name must be a file name, optionally containing "
            "the {dataset} placeholder."
        )
    return artifacts_root / path


def write_training_summary(
    path: Path,
    *,
    config_dir: Path,
    dataset_names: list[str],
    seeds: list[int],
    ranking_ks: tuple[int, ...],
    metrics_summary_path: Path,
    summaries: Iterable[Mapping[str, Any]],
) -> None:
    write_json(
        path,
        {
            "config_dir": str(config_dir),
            "datasets": dataset_names,
            "seeds": seeds,
            "ranking_ks": list(ranking_ks),
            "metrics_summary": str(metrics_summary_path),
            "runs": list(summaries),
        },
    )


def build_experiment_logger(
    trainer_config: Mapping[str, Any],
    *,
    run_name: str,
    run_dir: Path,
    config_path: Path,
    resolved_config: Mapping[str, Any],
    cli_args: Mapping[str, Any],
) -> CompositeExperimentLogger:
    loggers = [JsonlExperimentLogger(run_dir / "training_log.jsonl")]
    comet_logger = CometExperimentLogger.from_config(
        dict(trainer_config.get("comet", {})),
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_path,
        resolved_config=resolved_config,
        cli_args=cli_args,
    )
    if comet_logger is not None:
        loggers.append(comet_logger)
    return CompositeExperimentLogger(loggers)


def data_config_paths(cfg: Any) -> dict[str, str | None]:
    names = (
        "reviews_path",
        "metadata_path",
        "interactions_path",
        "interactions_root",
        "embeddings_path",
        "semantic_id_path",
    )
    return {
        name: str(getattr(cfg, name)) if getattr(cfg, name, None) is not None else None
        for name in names
        if hasattr(cfg, name)
    }


def summary_to_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epochs_completed": summary.get("epochs_completed"),
        "steps_completed": summary.get("steps_completed"),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
    }
    for split, group_name in (
        ("validation", "validation_metrics"),
        ("test", "test_metrics"),
    ):
        metrics = summary.get(group_name)
        if isinstance(metrics, Mapping):
            payload.update({f"{split}/{key}": value for key, value in metrics.items()})
    return payload


def configured_path(
    trainer_config: Mapping[str, Any],
    key: str,
    *,
    project_root: Path,
) -> Path:
    value = trainer_config.get(key)
    if value is None:
        raise KeyError(f"The trainer config must define {key}.")
    return resolve_path(value, base=project_root)


def resolve_device(value: str | torch.device) -> str:
    configured = str(value)
    if configured.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if configured == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return configured


def create_run_dir(
    experiments_dir: Path,
    run_name: str,
    config: Mapping[str, Any],
) -> Path:
    safe_name = safe_component(run_name)[:64] or "run"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    execution_id = safe_component(
        os.environ.get("RUN_ID", f"pid-{os.getpid()}")
    )
    digest = hashlib.sha1(
        json.dumps(to_jsonable(config), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    run_dir = experiments_dir / f"{stamp}_{safe_name}_{execution_id}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "dataset"


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


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must contain a mapping: {path}")
    return loaded


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2, sort_keys=True)


def write_text_summary(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# SASRec Run Summary",
        "",
        f"- Dataset: {summary['dataset'].get('name')}",
        f"- Seed: {summary['seed']}",
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
    return None if value is None else resolve_path(value, base=base)


def resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


# Dataset-config keys that hold local input locations. The scalar keys cover
# every datamodule family (Amazon reviews/metadata, Yambda interactions and
# embeddings, VK-LSVD root/embeddings, semantic IDs); the list keys are the
# VK-LSVD explicit per-split file lists.
INPUT_PATH_KEYS: tuple[str, ...] = (
    "reviews_path",
    "metadata_path",
    "interactions_path",
    "interactions_root",
    "embeddings_path",
    "semantic_id_path",
)
INPUT_PATH_LIST_KEYS: tuple[str, ...] = ("train_paths", "val_paths", "test_paths")


def resolve_input_paths(values: dict[str, Any], *, base: Path) -> dict[str, Any]:
    """Resolve every input-path key in a dataset-config mapping, in place."""
    for key in INPUT_PATH_KEYS:
        value = values.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            values[key] = [resolve_path(item, base=base) for item in value]
        else:
            values[key] = resolve_path(value, base=base)
    for key in INPUT_PATH_LIST_KEYS:
        value = values.get(key)
        if value is not None:
            values[key] = [resolve_path(item, base=base) for item in value]
    return values


def required_input_files(cfg: Any) -> list[Path]:
    """Local files a datamodule config depends on, for pre-flight checks.

    VK-LSVD configs expand their weekly split directories into the individual
    Parquet files; the other families list their configured scalar paths.
    """
    from flowrec.data.vk_lsvd import VKLSVDDataConfig, vk_lsvd_local_input_files

    paths: list[Path] = []
    for key in ("reviews_path", "metadata_path", "embeddings_path", "semantic_id_path"):
        value = getattr(cfg, key, None)
        if value is not None:
            paths.append(Path(value))
    if isinstance(cfg, VKLSVDDataConfig):
        paths.extend(vk_lsvd_local_input_files(cfg))
    else:
        value = getattr(cfg, "interactions_path", None)
        if value is not None:
            paths.append(Path(value))
    return paths
