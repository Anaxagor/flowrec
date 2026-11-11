from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.candidates.sampler import (
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
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
    write_safe_markdown_summary,
)
from flowrec.training.trainer import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.run_gpt2")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DEFAULT_STANDARD_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "gpt2"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_KS = (5, 10, 20)
YAMBDA_ALIAS = "yambda_50m_listens"
SUPPORTED_DATASETS = (
    "beauty",
    "instruments",
    "yambda_50m_listens",
    "yambda_500m_listens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the FlowRec GPT-2 recommender on exp1-standard_params with a "
            "multi-seed sweep and validation/test generation metrics."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="beauty",
        choices=[*SUPPORTED_DATASETS, "yambda", "all"],
        help=(
            "Dataset to train. 'yambda' is an alias for yambda_50m_listens; "
            "'all' runs Beauty, Instruments, Yambda-50M and Yambda-500M."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run. Defaults to gpt2_model.yaml or 42 43 44.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_STANDARD_CONFIG_DIR,
        help="Directory containing dataset_amazon.yaml, dataset_yambda.yaml and trainer.yaml.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="GPT-2 YAML. Defaults to <config-dir>/gpt2_model.yaml.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Root directory for GPT-2 experiment folders.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--method", choices=["sft", "gflow"], default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--plain-run-dir",
        action="store_true",
        help="Do not add timestamp/hash to the sweep directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = resolve_path(args.config_dir, base=PROJECT_ROOT)
    model_config_path = resolve_path(
        args.model_config or (config_dir / "gpt2_model.yaml"),
        base=PROJECT_ROOT,
    )

    model_yaml = load_yaml(model_config_path)
    amazon_yaml = load_yaml(config_dir / "dataset_amazon.yaml")
    yambda_yaml = load_yaml(config_dir / "dataset_yambda.yaml")
    trainer_yaml = load_yaml_optional(config_dir / "trainer.yaml")

    datasets = resolve_dataset_names(args.dataset)
    configured_seeds = model_yaml.get("defaults", {}).get("seeds", DEFAULT_SEEDS)
    seeds = tuple(int(seed) for seed in (args.seeds or configured_seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got {seeds}.")

    for dataset_name in datasets:
        sweep_config_preview = build_experiment_config(
            dataset_name=dataset_name,
            seed=seeds[0],
            model_yaml=model_yaml,
            amazon_yaml=amazon_yaml,
            yambda_yaml=yambda_yaml,
            trainer_yaml=trainer_yaml,
            method_override=args.method,
        )
        if args.max_epochs is not None:
            sweep_config_preview.setdefault("training", {})["max_epochs"] = int(args.max_epochs)
        if args.max_steps is not None:
            sweep_config_preview.setdefault("training", {})["max_steps"] = int(args.max_steps)

        run_name = args.run_name or f"gpt2_{dataset_name}_{sweep_config_preview['method']['name']}"
        sweep_dir = create_sweep_dir(
            experiments_dir=args.experiments_dir,
            dataset_name=dataset_name,
            run_name=run_name,
            config=sweep_config_preview,
            plain=bool(args.plain_run_dir),
        )
        setup_logging(sweep_dir)
        LOGGER.info("Dataset sweep directory: %s", sweep_dir)
        LOGGER.info("Seeds: %s", seeds)

        seed_results: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            seed_dir = sweep_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            config = build_experiment_config(
                dataset_name=dataset_name,
                seed=seed,
                model_yaml=model_yaml,
                amazon_yaml=amazon_yaml,
                yambda_yaml=yambda_yaml,
                trainer_yaml=trainer_yaml,
                method_override=args.method,
            )
            if args.max_epochs is not None:
                config.setdefault("training", {})["max_epochs"] = int(args.max_epochs)
            if args.max_steps is not None:
                config.setdefault("training", {})["max_steps"] = int(args.max_steps)

            LOGGER.info("Starting dataset=%s seed=%s", dataset_name, seed)
            seed_results[seed] = run_one_seed(
                config=config,
                dataset_name=dataset_name,
                seed=seed,
                seed_dir=seed_dir,
                device=args.device,
                model_config_path=model_config_path,
                standard_config_dir=config_dir,
            )

        setup_logging(sweep_dir)
        aggregate = aggregate_seed_metrics(seed_results, ks=resolve_ks(model_yaml))
        write_json(sweep_dir / "aggregate_test_metrics.json", aggregate)
        write_aggregate_csv(sweep_dir / "aggregate_test_metrics.csv", aggregate)
        write_aggregate_markdown(sweep_dir / "aggregate_summary.md", dataset_name, aggregate)
        write_json(
            sweep_dir / "sweep_config.json",
            {
                "dataset": dataset_name,
                "seeds": list(seeds),
                "model_config_path": str(model_config_path),
                "standard_config_dir": str(config_dir),
                "seed_dirs": {str(seed): result["run_dir"] for seed, result in seed_results.items()},
            },
        )
        LOGGER.info("Finished dataset=%s. Aggregate metrics: %s", dataset_name, json.dumps(aggregate))


def run_one_seed(
    *,
    config: dict[str, Any],
    dataset_name: str,
    seed: int,
    seed_dir: Path,
    device: str | None,
    model_config_path: Path,
    standard_config_dir: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    setup_logging(seed_dir)

    LOGGER.info("Run directory: %s", seed_dir)
    LOGGER.info("Dataset: %s seed=%s", dataset_name, seed)

    data_cfg = build_data_config(config, config_path=model_config_path)
    data = build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("GPT-2 training requires semantic IDs.")

    semantic_metadata = load_and_validate_semantic_metadata(data, config)
    prepare_semantic_ids(data, config)

    # Dataloaders are intentionally created after any optional semantic-ID
    # transformation so collators and model see exactly the same token mapping.
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = build_test_dataloader(data, config)

    model = build_model(config, data)
    if data.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support "
            f"train_example_layout={data.cfg.train_example_layout!r}; "
            f"supported: {model.supported_train_layouts}."
        )

    init_checkpoint_path = config.get("training", {}).get("init_checkpoint_path")
    init_checkpoint_strict = bool(config.get("training", {}).get("init_checkpoint_strict", True))
    if init_checkpoint_path is not None:
        resolved_init_checkpoint = resolve_experiment_path(
            init_checkpoint_path,
            config_path=model_config_path,
        )
        LOGGER.info("Loading initial GPT-2 model weights from %s", resolved_init_checkpoint)
        load_model_state_dict(model, resolved_init_checkpoint, strict=init_checkpoint_strict)

    method = build_method(
        config,
        model,
        data=data,
        train_loader=train_loader,
        config_path=model_config_path,
    )
    popularity_lookup = build_semantic_popularity_lookup(data)
    method = GenerationMetricAdapter(
        method,
        popularity_by_semantic_id=popularity_lookup,
        ks=tuple(int(k) for k in config["training"]["generation_topk"]),
    )

    trainer_cfg = build_trainer_config(config, seed_dir)
    resolved = {
        "dataset_name": dataset_name,
        "seed": seed,
        "run_dir": str(seed_dir),
        "standard_config_dir": str(standard_config_dir),
        "model_config_path": str(model_config_path),
        "data": asdict(data.cfg),
        "data_summary": data.summary(),
        "semantic_id_metadata": semantic_metadata,
        "model": model.config_dict(),
        "method": resolved_method_config(method, config),
        "trainer": trainer_cfg.to_dict(),
        "optimizer": config.get("optimizer", {}),
        "evaluation": {
            "ks": list(config["training"]["generation_topk"]),
            "arp_definition": "mean train interaction count of recommended catalog items",
            "ranking_metrics_only_on": ["validation", "test"],
        },
        "source_config": config.get("source_config", {}),
    }
    write_json(seed_dir / "resolved_config.json", resolved)
    write_json(seed_dir / "cli_args.json", {"dataset": dataset_name, "seed": seed, "device": device})
    if semantic_metadata:
        write_json(seed_dir / "semantic_id_metadata.json", semantic_metadata)

    experiment_logger = build_experiment_logger(
        config,
        run_name=f"{dataset_name}_seed_{seed}",
        run_dir=seed_dir,
        config_path=model_config_path,
        resolved_config=resolved,
        cli_args={"dataset": dataset_name, "seed": seed},
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
            device=device,
            lr=float(config.get("optimizer", {}).get("lr", 3e-4)),
            weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.0)),
            betas=tuple(config.get("optimizer", {}).get("betas", (0.9, 0.999))),
            eps=float(config.get("optimizer", {}).get("eps", 1e-8)),
            logger=experiment_logger,
        )

        fit_started_at = time.perf_counter()
        fit_result = trainer.fit(train_loader, val_loader)
        fit_elapsed_sec = time.perf_counter() - fit_started_at
        LOGGER.info("Trainer.fit finished in %.2f minutes", fit_elapsed_sec / 60.0)

        completed_epochs = int(trainer.state.epoch + 1)
        completed_steps = int(trainer.state.global_step)
        checkpoint_for_eval = fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        if checkpoint_for_eval is not None:
            trainer.load_checkpoint(
                checkpoint_for_eval,
                load_optimizer=False,
                load_scheduler=False,
            )

        # Re-evaluate the best checkpoint on the complete validation and test
        # loaders. This gives the final per-seed metrics saved to disk and avoids
        # reporting metrics from the last (possibly non-best) epoch.
        validation_metrics = trainer.evaluate(val_loader, stage="val")
        test_metrics = trainer.test(test_loader)
        write_json(seed_dir / "validation_metrics.json", validation_metrics)
        write_json(seed_dir / "test_metrics.json", test_metrics)

        summary = {
            "dataset": {**data_config_paths(data.cfg), **data.summary()},
            "dataset_name": dataset_name,
            "seed": seed,
            "model": model.config_dict(),
            "epochs_completed": completed_epochs,
            "steps_completed": completed_steps,
            "fit_elapsed_sec": fit_elapsed_sec,
            "best_metric": trainer.state.best_metric,
            "best_epoch": trainer.state.best_epoch,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": fit_result.best_checkpoint_path,
            "last_checkpoint": fit_result.last_checkpoint_path,
            "run_dir": str(seed_dir),
            "experiment_dir": str(seed_dir),
        }
        write_json(seed_dir / "final_summary.json", summary)
        write_text_summary(seed_dir / "final_summary.md", summary)
        log_final_summary(experiment_logger, summary, step=completed_steps)
        log_run_artifacts(experiment_logger, seed_dir, step=completed_steps)
        LOGGER.info("Seed %s finished. Test metrics: %s", seed, json.dumps(test_metrics, sort_keys=True))
        return summary
    finally:
        close_logger(experiment_logger)


class GenerationMetricAdapter(FlowRecMethod):
    """Add HitRate aliases and ARP without changing the generic Trainer.

    SFTMethod already generates top-k semantic IDs on validation/test and stores
    them in MethodOutput.extra. The Trainer separately computes exact dataset-level
    catalog coverage from the same generated IDs. This adapter only adds:
      * HitRate@K as an explicit alias of single-target Recall@K;
      * ARP@K from train-only item popularity.

    It delegates every objective/loss operation to the wrapped method, so SFT and
    GFlow behaviour is otherwise unchanged.
    """

    method_name = "generation_metric_adapter"

    def __init__(
        self,
        base_method: FlowRecMethod,
        *,
        popularity_by_semantic_id: Mapping[tuple[int, ...], float],
        ks: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.base_method = base_method
        self._popularity_by_semantic_id = dict(popularity_by_semantic_id)
        self._ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))

    @property
    def cfg(self):
        return self.base_method.cfg

    def compute_loss(
        self,
        model,
        batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        out = self.base_method.compute_loss(model, batch, stage=stage, step=step)
        if stage == "train":
            return out

        generated = out.extra.get("generated_token_ids")
        if not torch.is_tensor(generated):
            return out
        if generated.ndim == 2:
            generated = generated.unsqueeze(1)
        if generated.ndim != 3:
            return out

        prefix = out.extra.get("generation_metric_prefix")
        if not isinstance(prefix, str) or not prefix:
            prefix = "sft"
        prefix = prefix.rstrip("/")

        for k in self._ks:
            kk = min(int(k), int(generated.size(1)))
            if kk <= 0:
                continue

            # Single-target next-item evaluation: Recall@K == HitRate@K.
            recall_key = f"{prefix}/recall@{k}"
            generation_recall_key = f"{prefix}/generation_recall@{k}"
            if recall_key in out.metrics:
                out.metrics[f"{prefix}/hitrate@{k}"] = out.metrics[recall_key]
            elif generation_recall_key in out.metrics:
                out.metrics[f"{prefix}/hitrate@{k}"] = out.metrics[generation_recall_key]

            out.metrics[f"{prefix}/arp@{k}"] = self._arp(generated[:, :kk, :])

        return out

    def _arp(self, generated: torch.Tensor) -> torch.Tensor:
        rows = generated.detach().cpu().long().reshape(-1, generated.size(-1)).tolist()
        if not rows:
            return generated.new_zeros((), dtype=torch.float32)
        values = [
            float(self._popularity_by_semantic_id.get(tuple(int(x) for x in row), 0.0))
            for row in rows
        ]
        return torch.tensor(
            sum(values) / max(len(values), 1),
            dtype=torch.float32,
            device=generated.device,
        )


def build_semantic_popularity_lookup(
    data: AmazonReviewDataModule | YambdaDataModule,
) -> dict[tuple[int, ...], float]:
    """Map semantic-ID tuples to train-only interaction popularity counts."""
    if data.semantic_ids is None:
        return {}
    interactions = getattr(data, "interactions", None)
    if interactions is None or len(interactions) == 0:
        LOGGER.warning("Cannot compute ARP popularity: datamodule has no interactions frame.")
        return {tuple(tokens): 0.0 for tokens in data.semantic_ids.values()}

    frame = interactions
    split_strategy = str(getattr(data.cfg, "split_strategy", "leave_one_out"))
    if split_strategy == "leave_one_out":
        if "position" not in frame.columns or "user_id" not in frame.columns:
            raise ValueError("ARP for leave_one_out requires interactions columns user_id and position.")
        last_position = frame.groupby("user_id")["position"].transform("max")
        train_frame = frame.loc[frame["position"] <= (last_position - 2)]
    elif split_strategy == "global_time":
        cutoff = getattr(data, "global_val_cutoff", None)
        if cutoff is None:
            cutoff = getattr(data.cfg, "global_val_timestamp", None)
        if cutoff is None:
            raise ValueError("ARP for global_time requires a validation cutoff.")
        train_frame = frame.loc[frame["timestamp"] < int(cutoff)]
    else:
        raise ValueError(f"Unsupported split_strategy for ARP: {split_strategy!r}")

    counts = train_frame["raw_item_id"].astype(str).value_counts().to_dict()
    return {
        tuple(int(token) for token in tokens): float(counts.get(str(raw_item), 0))
        for raw_item, tokens in data.semantic_ids.items()
    }


def resolve_dataset_names(name: str) -> tuple[str, ...]:
    normalized = str(name).lower()
    if normalized == "all":
        return SUPPORTED_DATASETS
    if normalized == "yambda":
        return (YAMBDA_ALIAS,)
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}.")
    return (normalized,)


def resolve_ks(model_yaml: Mapping[str, Any]) -> tuple[int, ...]:
    defaults = model_yaml.get("defaults", {})
    raw = defaults.get("ks", DEFAULT_KS)
    ks = tuple(sorted({int(k) for k in raw if int(k) > 0}))
    if not ks:
        raise ValueError("gpt2_model.yaml must define at least one positive K.")
    return ks


def build_experiment_config(
    *,
    dataset_name: str,
    seed: int,
    model_yaml: Mapping[str, Any],
    amazon_yaml: Mapping[str, Any],
    yambda_yaml: Mapping[str, Any],
    trainer_yaml: Mapping[str, Any],
    method_override: str | None,
) -> dict[str, Any]:
    defaults = dict(model_yaml.get("defaults", {}))
    profiles = model_yaml.get("datasets", {})
    if dataset_name not in profiles:
        raise KeyError(f"gpt2_model.yaml has no profile for {dataset_name!r}.")
    profile = dict(profiles[dataset_name])
    ks = resolve_ks(model_yaml)

    model_cfg = deep_merge(dict(defaults.get("model", {})), dict(profile.get("model", {})))
    optimizer_cfg = deep_merge(
        {"weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        dict(defaults.get("optimizer", {})),
    )
    optimizer_cfg = deep_merge(optimizer_cfg, dict(profile.get("optimizer", {})))

    raw_trainer = dict(trainer_yaml.get("trainer", trainer_yaml)) if trainer_yaml else {}
    training_cfg = deep_merge(
        {
            "max_epochs": 100,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "skip_non_finite_loss": False,
            "use_amp": True,
            "amp_dtype": "bfloat16",
            "gradient_checkpointing": False,
            "log_every_steps": 50,
            "progress_bar": True,
            "validate_every_epochs": 1,
            "run_validation_at_start": False,
            "save_best": True,
            "save_last": True,
            "monitor": "val/sft/ndcg@10",
            "monitor_mode": "max",
            "early_stopping_patience": 10,
            "early_stopping_mode": "best",
            "min_delta": 1e-5,
            "prefix_method_metrics_with_stage": True,
        },
        raw_trainer,
    )
    training_cfg = deep_merge(training_cfg, dict(defaults.get("training", {})))
    training_cfg = deep_merge(training_cfg, dict(profile.get("training", {})))
    training_cfg.update(
        {
            "evaluate_generation": True,
            "generation_topk": list(ks),
            "generation_num_beams": max(ks),
            "generation_num_return_sequences": max(ks),
            "generation_constrain_valid_sequences": True,
            "generation_exclude_seen_items": False,
            "test_evaluate_generation": True,
            "test_generation_topk": list(ks),
            "test_generation_num_beams": max(ks),
            "test_generation_num_return_sequences": max(ks),
        }
    )

    method_name = str(method_override or profile.get("method", defaults.get("method", "sft"))).lower()
    if method_name not in {"sft", "gflow"}:
        raise ValueError(f"Unsupported GPT-2 method {method_name!r}.")
    method_cfg = {"name": method_name}
    method_cfg.update(dict(defaults.get("method_config", {})))
    method_cfg.update(dict(profile.get("method_config", {})))

    semantic_dir = profile.get("semantic_ids_dir")
    if semantic_dir is None:
        semantic_dir = (
            PROJECT_ROOT
            / "flowrec"
            / "datasets"
            / "semantic_ids"
            / "exp1-standard_params"
            / dataset_name
        )
    semantic_dir = resolve_path(semantic_dir, base=PROJECT_ROOT)
    semantic_path = semantic_dir / "semantic_ids.json"

    if dataset_name in {"beauty", "instruments"}:
        standard = amazon_yaml
        source = "amazon"
    else:
        standard = yambda_yaml
        source = "yambda"

    dataset_specific = dict(standard.get("datasets", {}).get(dataset_name, {}))
    if not dataset_specific:
        raise KeyError(f"Standard dataset config has no entry for {dataset_name!r}.")
    shared = {
        key: value
        for key, value in standard.items()
        if key not in {"datasets", "dataset", "sasrec", "tiger", "gpt2"}
    }
    dataset_cfg = deep_merge(shared, dataset_specific)

    # Model-family-specific data-loader overrides live in the standard dataset
    # config. This keeps shared dataset semantics in one place while allowing
    # GPT-2 to use different physical train/validation/test batch sizes.
    dataset_cfg = deep_merge(dataset_cfg, dict(standard.get("gpt2", {})))

    dataset_cfg.update(
        {
            "train_example_layout": "dense_causal",
            "train_window_len": None,
            "train_window_stride": None,
            "history_padding": "right",
            "include_raw_in_batch": False,
        }
    )
    config = {
        "seed": int(seed),
        "data": {
            "source": source,
            "semantic_id_path": str(semantic_path),
            **dataset_specific,
        },
        "dataset": dataset_cfg,
        "dataloader": {
            "batch_size": int(dataset_cfg.get("batch_size", 128)),
            "train_batch_size": int(dataset_cfg.get("batch_size", 128)),
            "eval_batch_size": int(dataset_cfg.get("eval_batch_size", 64)),
            "validation_batch_size": int(dataset_cfg.get("eval_batch_size", 64)),
            # test_eval_batch_size is intentionally runner-only: the generic
            # datamodules expose one eval_batch_size for both val and test.
            "test_eval_batch_size": int(
                dataset_cfg.get(
                    "test_eval_batch_size",
                    dataset_cfg.get("eval_batch_size", 64),
                )
            ),
            "num_workers": int(dataset_cfg.get("num_workers", 0)),
            "pin_memory": bool(dataset_cfg.get("pin_memory", True)),
            "persistent_workers": bool(dataset_cfg.get("persistent_workers", False)),
            "train_drop_last": bool(dataset_cfg.get("train_drop_last", False)),
            "include_raw_in_batch": False,
        },
        "model": {
            **model_cfg,
            "num_beams": max(ks),
            "top_k": max(ks),
            "compact_semantic_vocabulary": bool(model_cfg.get("compact_semantic_vocabulary", False)),
        },
        "method": method_cfg,
        "training": training_cfg,
        "optimizer": optimizer_cfg,
        "comet": dict(defaults.get("comet", {})),
        "source_config": {
            "dataset_standard": dataset_cfg,
            "gpt2_profile": profile,
        },
    }

    return config


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_data_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> AmazonReviewDataConfig | YambdaDataConfig:
    data = dict(config.get("data", {}))
    dataset = dict(config.get("dataset", {}))
    dataloader = dict(config.get("dataloader", {}))
    model = dict(config.get("model", {}))
    base = PROJECT_ROOT
    max_history_len = int(dataset.get("max_history_len", 20))
    train_example_layout = str(dataset.get("train_example_layout", "dense_causal"))
    train_window_len = dataset.get("train_window_len")
    if train_example_layout == "dense_causal" and train_window_len is None:
        train_window_len = max_history_len
    if train_window_len is not None:
        train_window_len = int(train_window_len)
    train_window_stride = dataset.get("train_window_stride")
    if train_window_stride is not None:
        train_window_stride = int(train_window_stride)

    source = str(data.get("source", dataset.get("source", "amazon"))).lower()
    semantic_id_path = resolve_optional_path(data.get("semantic_id_path"), base=base)

    if source == "yambda":
        hf_recent_months = dataset.get("hf_recent_months")
        allowed_event_types = dataset.get("allowed_event_types")
        return YambdaDataConfig(
            interactions_path=resolve_optional_path(data.get("interactions_path"), base=base),
            semantic_id_path=semantic_id_path,
            use_hf=bool(dataset.get("use_hf", data.get("use_hf", False))),
            hf_dataset_name=str(dataset.get("hf_dataset_name", data.get("hf_dataset_name", "yandex/yambda"))),
            hf_start_date=dataset.get("hf_start_date"),
            hf_end_date=dataset.get("hf_end_date"),
            hf_recent_months=int(hf_recent_months) if hf_recent_months is not None else None,
            dataset_type=dataset.get("dataset_type", data.get("dataset_type", "flat")),
            dataset_size=dataset.get("dataset_size", data.get("dataset_size", "50m")),
            event_file=dataset.get("event_file", data.get("event_file", "listens")),
            streaming=bool(dataset.get("streaming", data.get("streaming", False))),
            max_rows=int(dataset["max_rows"]) if dataset.get("max_rows") is not None else None,
            min_user_interactions=int(dataset.get("min_user_interactions", 5)),
            min_item_interactions=int(dataset.get("min_item_interactions", 0)),
            deduplicate_consecutive=bool(dataset.get("deduplicate_consecutive", False)),
            keep_raw_interactions=bool(dataloader.get("include_raw_in_batch", False)),
            organic_filter=dataset.get("organic_filter", "any"),
            positive_criterion=(
                str(dataset["positive_criterion"])
                if dataset.get("positive_criterion") is not None
                else None
            ),
            min_played_ratio_pct=(
                float(dataset["min_played_ratio_pct"])
                if dataset.get("min_played_ratio_pct") is not None
                else None
            ),
            allowed_event_types=(
                tuple(str(x) for x in allowed_event_types)
                if allowed_event_types is not None
                else None
            ),
            split_strategy=dataset.get("split_strategy", "leave_one_out"),
            min_history_len=int(dataset.get("min_history_len", 1)),
            train_example_layout=train_example_layout,
            train_window_len=train_window_len,
            train_window_stride=train_window_stride,
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
            history_padding=dataset.get("history_padding", "right"),
            batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 128))),
            eval_batch_size=int(dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 64))),
            num_workers=int(dataloader.get("num_workers", 8)),
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
        semantic_id_path=semantic_id_path,
        item_key=dataset.get("item_key", data.get("item_key", "asin")),
        filter_reviews_to_metadata=bool(dataset.get("filter_reviews_to_metadata", True)),
        positive_criterion=dataset.get("positive_criterion"),
        min_user_interactions=int(dataset.get("min_user_interactions", 5)),
        min_item_interactions=int(dataset.get("min_item_interactions", 0)),
        deduplicate_consecutive=bool(dataset.get("deduplicate_consecutive", False)),
        split_strategy=dataset.get("split_strategy", "leave_one_out"),
        min_history_len=int(dataset.get("min_history_len", 1)),
        train_example_layout=train_example_layout,
        train_window_len=train_window_len,
        train_window_stride=train_window_stride,
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
        history_padding=dataset.get("history_padding", "right"),
        batch_size=int(dataloader.get("train_batch_size", dataloader.get("batch_size", 128))),
        eval_batch_size=int(dataloader.get("validation_batch_size", dataloader.get("eval_batch_size", 64))),
        num_workers=int(dataloader.get("num_workers", 0)),
        pin_memory=bool(dataloader.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(dataloader.get("persistent_workers", False)),
        train_drop_last=bool(dataloader.get("train_drop_last", False)),
        include_raw_in_batch=bool(dataloader.get("include_raw_in_batch", False)),
        filter_cold_items_after_split=bool(dataset.get("filter_cold_items_after_split", False)),
        cold_item_train_source=dataset.get("cold_item_train_source", "observed"),
        offset_semantic_ids_by_one=bool(dataset.get("offset_semantic_ids_by_one", True)),
        drop_items_missing_semantic_id=bool(dataset.get("drop_items_missing_semantic_id", True)),
        user_hash_mod=int(model["user_ids_count"]) if model.get("user_ids_count") is not None else None,
        seed=int(config.get("seed", 42)),
    )


def prepare_semantic_ids(
    data: AmazonReviewDataModule | YambdaDataModule,
    config: Mapping[str, Any],
) -> None:
    """Validate the post-filtered semantic catalog; compaction is opt-in."""
    if data.semantic_ids is None:
        raise ValueError("Semantic IDs are required.")

    mapping = {str(item): [int(token) for token in tokens] for item, tokens in data.semantic_ids.items()}
    if not mapping:
        raise ValueError("The post-filtered semantic-ID catalog is empty.")

    lengths = {len(tokens) for tokens in mapping.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(
            "GPT2RecModel requires fixed-length semantic IDs after collision resolution; "
            f"got lengths={sorted(lengths)}."
        )
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
    compact = bool(model_cfg.get("compact_semantic_vocabulary", False))
    old_max = int(rows.max().item())
    if compact:
        used_tokens = sorted({int(token) for tokens in mapping.values() for token in tokens})
        token_remap = {token: new_token for new_token, token in enumerate(used_tokens, start=1)}
        mapping = {item: [token_remap[token] for token in tokens] for item, tokens in mapping.items()}
        data.semantic_ids = mapping
        data.sid_len = sid_len
        LOGGER.warning(
            "Compacted semantic vocabulary at runtime: max_token %s -> %s. "
            "For reproducible published experiments prefer compact_semantic_vocabulary=false.",
            old_max,
            max(token_remap.values()),
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


def load_and_validate_semantic_metadata(
    data: AmazonReviewDataModule | YambdaDataModule,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_path = getattr(data.cfg, "semantic_id_path", None)
    if semantic_path is None:
        return {}
    metadata_path = Path(semantic_path).parent / "semantic_id_metadata.json"
    if not metadata_path.exists():
        LOGGER.warning("semantic_id_metadata.json not found next to %s", semantic_path)
        return {}

    metadata = load_json(metadata_path)
    collision = metadata.get("collision_stats", {})
    if collision:
        before = int(collision.get("num_unique_before", collision.get("num_items", 0)))
        after = int(collision.get("num_unique_after", before))
        num_items = int(collision.get("num_items", after))
        if after != num_items:
            raise ValueError(
                "Semantic metadata reports unresolved collisions: "
                f"num_unique_after={after}, num_items={num_items}."
            )

    rqvae = metadata.get("rqvae_config", {})
    configured_codebook = config.get("model", {}).get("codebook_size")
    metadata_codebook = rqvae.get("codebook_size")
    if configured_codebook is not None and metadata_codebook is not None:
        if int(configured_codebook) != int(metadata_codebook):
            LOGGER.warning(
                "GPT-2 profile codebook_size=%s differs from semantic metadata codebook_size=%s. "
                "The loaded semantic_ids.json remains the source of truth for GPT2RecModel vocabulary.",
                configured_codebook,
                metadata_codebook,
            )
    return metadata


def build_data_module(
    cfg: AmazonReviewDataConfig | YambdaDataConfig,
) -> AmazonReviewDataModule | YambdaDataModule:
    if isinstance(cfg, AmazonReviewDataConfig):
        return AmazonReviewDataModule(cfg)
    if isinstance(cfg, YambdaDataConfig):
        return YambdaDataModule(cfg)
    raise TypeError(f"Unsupported data config: {type(cfg)!r}")


def build_test_dataloader(
    data: AmazonReviewDataModule | YambdaDataModule,
    config: Mapping[str, Any],
):
    """Build the test loader with an optional runner-only batch-size override.

    The datamodule configs intentionally keep a single ``eval_batch_size`` field.
    GPT-2 generation can be much more memory-intensive at final test time than
    in-training validation, so ``dataloader.test_eval_batch_size`` is resolved by
    the runner. We temporarily replace the datamodule config only while the
    DataLoader object is constructed, then restore it immediately.
    """
    dataloader_cfg = dict(config.get("dataloader", {}))
    validation_batch_size = int(
        dataloader_cfg.get(
            "validation_batch_size",
            dataloader_cfg.get("eval_batch_size", data.cfg.eval_batch_size),
        )
    )
    test_batch_size = int(
        dataloader_cfg.get("test_eval_batch_size", validation_batch_size)
    )
    if test_batch_size <= 0:
        raise ValueError(
            f"dataloader.test_eval_batch_size must be positive, got {test_batch_size}."
        )

    if test_batch_size == int(data.cfg.eval_batch_size):
        return data.test_dataloader()

    original_cfg = data.cfg
    try:
        data.cfg = replace(original_cfg, eval_batch_size=test_batch_size)
        return data.test_dataloader()
    finally:
        data.cfg = original_cfg


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
    if data.sid_len is None or data.semantic_ids is None:
        raise ValueError("GPT-2 requires semantic IDs.")

    embedding_dim = int(model_cfg.get("embedding_dim", model_cfg.get("n_embd", 128)))
    num_heads = int(model_cfg.get("num_heads", model_cfg.get("n_head", 4)))
    if embedding_dim % num_heads != 0:
        raise ValueError(
            "GPT-2 embedding_dim must be divisible by num_heads: "
            f"{embedding_dim} % {num_heads} != 0."
        )

    max_history_len = int(data.cfg.max_history_len)
    train_window_len = int(
        data.cfg.train_window_len if data.cfg.train_window_len is not None else max_history_len
    )
    dense_train_positions = 1 + (train_window_len + 1) * int(data.sid_len) - 1
    generation_positions = 1 + max_history_len * int(data.sid_len) + int(data.sid_len)
    required_positions = max(dense_train_positions, generation_positions)
    max_position_embeddings = int(
        model_cfg.get("max_position_embeddings", model_cfg.get("n_positions", required_positions))
    )
    if max_position_embeddings < required_positions:
        raise ValueError(
            "model.max_position_embeddings is too small for the resolved data config: "
            f"{max_position_embeddings} < {required_positions}."
        )

    top_k = int(model_cfg.get("top_k", 20))
    model = GPT2RecModel(
        codebook_size=int(model_cfg.get("codebook_size", 256)),
        sem_id_len=int(data.sid_len),
        semantic_id_mapping=data.semantic_ids,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=int(model_cfg.get("num_layers", model_cfg.get("n_layer", 4))),
        dim_feedforward=int(model_cfg.get("dim_feedforward", model_cfg.get("n_inner", 512))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attention_dropout=float(model_cfg.get("attention_dropout", model_cfg.get("dropout", 0.1))),
        residual_dropout=float(model_cfg.get("residual_dropout", model_cfg.get("dropout", 0.1))),
        embedding_dropout=float(model_cfg.get("embedding_dropout", model_cfg.get("dropout", 0.1))),
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


def aggregate_seed_metrics(
    seed_results: Mapping[int, Mapping[str, Any]],
    *,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    per_seed: dict[str, dict[str, float]] = {}
    for seed, result in seed_results.items():
        test_metrics = result.get("test_metrics", {})
        per_seed[str(seed)] = canonical_ranking_metrics(test_metrics, ks=ks)

    metric_names = [
        f"{name}@{k}"
        for name in ("NDCG", "HitRate", "Coverage", "ARP")
        for k in ks
    ]
    aggregate: dict[str, Any] = {
        "seeds": sorted(int(seed) for seed in seed_results),
        "per_seed": per_seed,
        "metrics": {},
    }
    for metric in metric_names:
        values = [per_seed[str(seed)][metric] for seed in sorted(seed_results)]
        mean = float(statistics.fmean(values))
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregate["metrics"][metric] = {
            "mean": mean,
            "std": std,
            "display": f"{mean:.6f} ± {std:.6f}",
            "values": {str(seed): per_seed[str(seed)][metric] for seed in sorted(seed_results)},
        }
    return aggregate


def canonical_ranking_metrics(
    metrics: Mapping[str, Any],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"NDCG@{k}"] = _required_metric(
            metrics,
            [f"test/sft/ndcg@{k}", f"test/sft/generation_ndcg@{k}"],
        )
        out[f"HitRate@{k}"] = _required_metric(
            metrics,
            [f"test/sft/hitrate@{k}", f"test/sft/recall@{k}", f"test/sft/generation_recall@{k}"],
        )
        out[f"Coverage@{k}"] = _required_metric(
            metrics,
            [f"test/sft/generation_coverage@{k}"],
        )
        out[f"ARP@{k}"] = _required_metric(metrics, [f"test/sft/arp@{k}"])
    return out


def _required_metric(metrics: Mapping[str, Any], candidates: list[str]) -> float:
    for key in candidates:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"Missing required metric. Tried keys: {candidates}. Available: {sorted(metrics)}")


def write_aggregate_csv(path: Path, aggregate: Mapping[str, Any]) -> None:
    rows = []
    for metric, stats in aggregate.get("metrics", {}).items():
        rows.append({"metric": metric, "mean": stats["mean"], "std": stats["std"], "display": stats["display"]})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "display"])
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_markdown(path: Path, dataset_name: str, aggregate: Mapping[str, Any]) -> None:
    lines = [f"# GPT-2 aggregate test metrics — {dataset_name}", "", "| Metric | Mean ± std |", "|---|---:|"]
    for metric, stats in aggregate.get("metrics", {}).items():
        lines.append(f"| {metric} | {stats['display']} |")
    lines.extend(["", f"Seeds: {', '.join(str(x) for x in aggregate.get('seeds', []))}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def create_sweep_dir(
    *,
    experiments_dir: Path,
    dataset_name: str,
    run_name: str,
    config: Mapping[str, Any],
    plain: bool,
) -> Path:
    root = resolve_path(experiments_dir, base=PROJECT_ROOT) / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:80] or "gpt2"
    if plain:
        run_dir = root / safe_name
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        digest = hashlib.sha1(json.dumps(to_jsonable(config), sort_keys=True).encode("utf-8")).hexdigest()[:8]
        run_dir = root / f"{stamp}_{safe_name[:48]}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_yaml(path: Path) -> dict[str, Any]:
    path = resolve_path(path, base=PROJECT_ROOT)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def load_yaml_optional(path: Path) -> dict[str, Any]:
    path = resolve_path(path, base=PROJECT_ROOT)
    if not path.exists():
        LOGGER.warning("Optional trainer config not found: %s; using runner defaults.", path)
        return {}
    return load_yaml(path)



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
        "num_negatives",
        "candidate_scorer",
        "curriculum_steps",
        "temperature",
        "exclude_history",
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

    candidate_scorer_name = str(sampler_cfg.get("candidate_scorer", "sasrec")).lower()
    needs_sasrec_sampler = (
        candidate_sampler_name in {"sasrec", "sasrec_score", "model_score", "item_model"}
        or (
            candidate_sampler_name in {"curriculum", "top_n"}
            and candidate_scorer_name == "sasrec"
        )
    )
    needs_svd_sampler = (
        candidate_sampler_name == "svd"
        or (
            candidate_sampler_name in {"curriculum", "top_n"}
            and candidate_scorer_name == "svd"
        )
    )

    sasrec_model = (
        build_sasrec_teacher_model(data, sasrec_cfg, config_path=config_path)
        if needs_sasrec_reward or needs_sasrec_sampler
        else None
    )
    svd_model = (
        build_svd_scoring_model(
            data=data,
            train_loader=train_loader,
            seed=int(config.get("seed", 42)),
            svd_cfg=svd_cfg,
        )
        if needs_svd_reward or needs_svd_sampler
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
    """Build the unified post-refactor semantic-ID candidate sampler."""

    name = str(name).lower()
    item_id_to_row = item_id_to_semantic_row(data)
    scorer = None

    if name == "generated":
        raise ValueError(
            "method.candidate_sampler='generated' is no longer supported. "
            "Use random, curriculum, top_n, sasrec, or svd."
        )

    if name == "random":
        mode = "random"
        default_exclude_history = False

    elif name in {"sasrec", "sasrec_score", "model_score", "item_model"}:
        if sasrec_model is None:
            raise ValueError(
                "method.sasrec.checkpoint_path is required for SASRec candidate sampling."
            )
        mode = "curriculum"
        scorer = sasrec_model
        item_id_to_row = restrict_item_id_mapping_to_scorer_vocab(
            item_id_to_row,
            sasrec_model,
        )
        default_exclude_history = True

    elif name == "svd":
        if svd_model is None:
            raise ValueError("SVD candidate sampling requires method.svd config.")
        mode = "curriculum"
        scorer = svd_model
        default_exclude_history = True

    elif name in {"curriculum", "top_n"}:
        mode = name
        scorer_name = str(method_cfg.get("candidate_scorer", "sasrec")).lower()

        if scorer_name == "sasrec":
            if sasrec_model is None:
                raise ValueError(
                    f"candidate_sampler={name!r} with candidate_scorer='sasrec' "
                    "requires method.sasrec.checkpoint_path."
                )
            scorer = sasrec_model
            item_id_to_row = restrict_item_id_mapping_to_scorer_vocab(
                item_id_to_row,
                sasrec_model,
            )
        elif scorer_name == "svd":
            if svd_model is None:
                raise ValueError(
                    f"candidate_sampler={name!r} with candidate_scorer='svd' "
                    "requires method.svd config."
                )
            scorer = svd_model
        else:
            raise ValueError(
                f"Unknown method.candidate_scorer={scorer_name!r}. "
                "Expected 'sasrec' or 'svd'."
            )
        default_exclude_history = True

    else:
        raise ValueError(
            f"Unknown method.candidate_sampler={name!r}. "
            "Expected random, curriculum, top_n, sasrec, sasrec_score, "
            "model_score, item_model, or svd."
        )

    num_negatives = int(
        method_cfg.get("num_negatives", method_cfg.get("num_negative_samples", 1))
    )
    curriculum_steps = int(
        method_cfg.get("curriculum_steps", method_cfg.get("easy_negative_steps", 0))
    )
    temperature = float(
        method_cfg.get("temperature", method_cfg.get("sampling_temperature", 1.0))
    )
    exclude_history = bool(
        method_cfg.get(
            "exclude_history",
            method_cfg.get(
                "exclude_history_in_sampling",
                default_exclude_history,
            ),
        )
    )

    return SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode=mode,
            num_negatives=num_negatives,
            include_positive=bool(method_cfg.get("include_positive", True)),
            exclude_history=exclude_history,
            curriculum_steps=curriculum_steps,
            temperature=temperature,
        ),
        item_semantic_ids=model.item_semantic_ids.detach().cpu(),
        item_id_to_semantic_row=item_id_to_row,
        scorer=scorer,
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
        "candidate_sampler",
        "candidate_scorer",
        "num_negatives",
        "num_negative_samples",
        "curriculum_steps",
        "easy_negative_steps",
        "temperature",
        "sampling_temperature",
        "exclude_history",
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
    write_safe_markdown_summary(path, summary, model_name="GPT-2")


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

