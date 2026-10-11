"""Train config-driven GPT-2 with the SFT loss on every exp1 dataset.

Covers the two small Amazon datasets (Beauty 2014, Instruments 2018), the
large Amazon Beauty 2023 dataset, the two Yambda datasets (50M and 500M
listens), and the VK-LSVD subsamples (provided weekly split), running three
seeds per dataset. GPT-2 consumes the exp1 semantic
IDs built by train_tokenizer_all.py from
``datasets/semantic_ids/exp1-standard_params/<dataset>/semantic_ids.json``.
Model parameters come from ``gpt2_model.yaml`` (per-dataset architecture and
lr, adopted from the externally tuned GPT-2 configs) and trainer parameters
from ``trainer_gpt2.yaml``; the dataset YAMLs define the catalogs (their
``sasrec:``/``tiger:`` blocks are ignored — GPT-2 always trains on the
dense_causal layout with right padding, forced by this runner exactly like
the SASRec runner forces its own layout).

Dataset and dataloader parameters come exclusively from the dataset YAMLs.
Their ``gpt2:`` sub-blocks (mirroring the ``sasrec:``/``tiger:`` blocks)
carry only the GPT-2 batching overrides — batch_size for training,
eval_batch_size for the in-training 20-beam validation, the runner-only
test_eval_batch_size for the final 20-beam evaluation, and dataloader
workers; every other data parameter (history length, filters, positive
criterion, the Yambda embedding filter) keeps the shared dataset-YAML values,
so GPT-2 sees exactly the item universe and splits TIGER and SASRec see.

After training, the best checkpoint (monitored on
``val/sft/generation_ndcg@10``) is evaluated on the validation and test
splits with beam search, and HR, NDCG, Coverage, and ARP at k in {5, 10, 20}
are written per seed to
``artifacts/exp1-standard_params/gpt2_sft_<dataset>_metrics_summary.csv``
(plus a combined ``gpt2_sft_all_metrics_summary.csv`` for multi-dataset
invocations), matching the SASRec and TIGER exp1 summary formats.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.vk_lsvd import VKLSVDDataConfig, VKLSVDDataModule
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.training.trainer import Trainer, TrainerConfig
from scripts import sasrec_config_runner as runner


LOGGER = runner.LOGGER
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
DEFAULT_SEMANTIC_IDS_DIR = (
    PROJECT_ROOT / "datasets" / "semantic_ids" / "exp1-standard_params"
)
MODEL_CONFIG_NAME = "gpt2_model.yaml"
TRAINER_CONFIG_NAME = "trainer_gpt2.yaml"
COMBINED_METRICS_SUMMARY_NAME = "gpt2_sft_all_metrics_summary.csv"
# GPT-2 is a decoder-only generator; exp1 trains it on dense causal windows.
REQUIRED_TRAIN_LAYOUT = "dense_causal"
# Final 20-beam evaluation batch size when a dataset YAML has no gpt2 block.
DEFAULT_TEST_EVAL_BATCH_SIZE = 64
# Runtime keys this runner reads from trainer_gpt2.yaml. An older checkout of
# the config (without them) still runs with these standard exp1 values instead
# of failing a queued Slurm task.
TRAINER_RUNTIME_DEFAULTS: dict[str, Any] = {
    "seeds": [42, 43, 44],
    "checkpoints_dir": "checkpoints/exp1-standard_params/gpt2",
    "experiments_dir": "experiments/gpt2",
    "artifacts_dir": "artifacts/exp1-standard_params",
    "metrics_summary_name": "gpt2_sft_{dataset}_metrics_summary.csv",
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "save_best": True,
    "save_last": True,
}


@dataclass(frozen=True)
class DatasetGroup:
    """One exp1 dataset family sharing a dataset YAML and a datamodule."""

    key: str
    data_config_name: str
    config_cls: type
    data_module: type


# Training order: large datasets first (Yambda 50M/500M, the VK-LSVD
# subsamples, then Beauty 2023), then the small Amazon datasets.
GROUPS: tuple[DatasetGroup, ...] = (
    DatasetGroup("yambda", "dataset_yambda.yaml", YambdaDataConfig, YambdaDataModule),
    DatasetGroup("vk_lsvd", "dataset_vk_lsvd.yaml", VKLSVDDataConfig, VKLSVDDataModule),
    DatasetGroup("amazon_2023", "dataset_amazon_2023.yaml", AmazonReviewDataConfig, AmazonReviewDataModule),
    DatasetGroup("amazon", "dataset_amazon.yaml", AmazonReviewDataConfig, AmazonReviewDataModule),
)


@dataclass(frozen=True)
class GroupPlan:
    group: DatasetGroup
    dataset_config: dict[str, Any]
    dataset_names: list[str]


@dataclass(frozen=True)
class BatchPlan:
    """Batch sizes for one run, fixed by the dataset YAML's gpt2 block
    (CLI overrides win); accumulation comes from trainer_gpt2.yaml."""

    train_batch_size: int
    gradient_accumulation_steps: int
    val_batch_size: int
    test_batch_size: int

    def to_dict(self) -> dict[str, int]:
        return {
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "val_batch_size": self.val_batch_size,
            "test_batch_size": self.test_batch_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GPT-2 with the SFT loss on the exp1 datasets (Amazon "
            "Beauty 2014, Instruments 2018, Beauty 2023, Yambda 50M, Yambda "
            "500M, and the VK-LSVD subsamples) for three seeds each and "
            "report validation/test HR, NDCG, Coverage, and ARP."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys to train on (any mix of yambda_50m_listens, "
            "yambda_500m_listens, vk_lsvd_ur0.01_ir0.01, vk_lsvd_ur0.01_ip0.01, "
            "beauty_2023, beauty, instruments) or 'all'; defaults to every "
            "configured dataset, large datasets first."
        ),
    )
    parser.add_argument(
        "--semantic-ids-dir",
        type=Path,
        default=DEFAULT_SEMANTIC_IDS_DIR,
        help="Folder with <dataset>/semantic_ids.json from train_tokenizer_all.py.",
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Defaults to the trainer YAML's three seeds; may be an explicit "
            "subset, e.g. one seed per Slurm array task (single-seed runs "
            "write seed-suffixed summary files so concurrent tasks never "
            "overwrite each other)."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override for the dataset YAML gpt2 block's batch_size.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override for the dataset YAML gpt2 block's eval_batch_size.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Override for the dataset YAML gpt2 block's test_eval_batch_size (final 20-beam evaluation).",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override.")
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug override for both in-training validation and the final evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    semantic_ids_dir = runner.resolve_path(args.semantic_ids_dir, base=PROJECT_ROOT)
    model_config = runner.load_yaml(config_dir / MODEL_CONFIG_NAME)
    trainer_config = apply_trainer_defaults(
        runner.load_yaml(config_dir / TRAINER_CONFIG_NAME)
    )
    plans = build_group_plans(config_dir, args.datasets)
    seeds = resolve_seeds(args, trainer_config)
    ranking_ks = resolve_ranking_ks(model_config)

    # Fail before any multi-day training if a later dataset's inputs (raw data
    # or the step-2 semantic IDs) are absent, a config drifted away from the
    # dense_causal layout, or a selected dataset has no gpt2_model.yaml
    # profile.
    validate_inputs(plans, model_config, semantic_ids_dir=semantic_ids_dir)

    checkpoint_root = runner.configured_path(
        trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
    )
    artifacts_root = runner.configured_path(
        trainer_config, "artifacts_dir", project_root=PROJECT_ROOT
    )
    summary_name_template = str(
        trainer_config.get("metrics_summary_name", "gpt2_sft_{dataset}_metrics_summary.csv")
    )

    # Single-dataset invocations (e.g. one Slurm array task per dataset) skip
    # the shared combined summary so concurrent jobs never overwrite it.
    selected_total = sum(len(plan.dataset_names) for plan in plans)
    combined_summaries: list[Mapping[str, Any]] = []

    for plan in plans:
        for dataset_name in plan.dataset_names:
            dataset_summaries: list[Mapping[str, Any]] = []
            for seed in seeds:
                summary = run_experiment(
                    dataset_name,
                    group=plan.group,
                    dataset_config=plan.dataset_config,
                    model_config=model_config,
                    trainer_config=trainer_config,
                    config_dir=config_dir,
                    semantic_ids_dir=semantic_ids_dir,
                    checkpoint_root=checkpoint_root,
                    ranking_ks=ranking_ks,
                    seed=seed,
                    args=args,
                )
                dataset_summaries.append(summary)
                combined_summaries.append(summary)
                # Preserve completed results if a later dataset/seed is
                # interrupted.
                metrics_summary_path = seed_scoped(
                    runner.dataset_metrics_summary_path(
                        artifacts_root, summary_name_template, dataset_name=dataset_name
                    ),
                    seeds,
                )
                runner.write_metrics_summary(
                    metrics_summary_path, dataset_summaries, ks=ranking_ks
                )
                runner.write_training_summary(
                    seed_scoped(
                        checkpoint_root
                        / runner.safe_component(dataset_name)
                        / "training_summary.json",
                        seeds,
                    ),
                    config_dir=config_dir,
                    dataset_names=[dataset_name],
                    seeds=seeds,
                    ranking_ks=ranking_ks,
                    metrics_summary_path=metrics_summary_path,
                    summaries=dataset_summaries,
                )
                if selected_total > 1:
                    runner.write_metrics_summary(
                        artifacts_root / COMBINED_METRICS_SUMMARY_NAME,
                        combined_summaries,
                        ks=ranking_ks,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            LOGGER.info(
                "[%s] saved validation/test metrics summary to %s",
                dataset_name,
                seed_scoped(
                    runner.dataset_metrics_summary_path(
                        artifacts_root, summary_name_template, dataset_name=dataset_name
                    ),
                    seeds,
                ),
            )


def resolve_seeds(
    args: argparse.Namespace, trainer_config: Mapping[str, Any]
) -> list[int]:
    """Explicit --seeds may be any distinct non-empty subset (e.g. one seed
    per Slurm array task); without it the trainer YAML must define the full
    three exp1 seeds."""
    if args.seeds is None:
        return runner.resolve_seeds(args, trainer_config)
    seeds = [int(seed) for seed in args.seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must be non-empty and distinct; got {seeds}")
    return seeds


def seed_scoped(path: Path, seeds: list[int]) -> Path:
    """Suffix summary files with the seed for single-seed invocations, so
    concurrent per-seed Slurm tasks for one dataset never overwrite each
    other's results."""
    if len(seeds) != 1:
        return path
    return path.with_name(f"{path.stem}_seed-{seeds[0]}{path.suffix}")


def run_experiment(
    dataset_name: str,
    *,
    group: DatasetGroup,
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    config_dir: Path,
    semantic_ids_dir: Path,
    checkpoint_root: Path,
    ranking_ks: tuple[int, ...],
    seed: int,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    runner.seed_everything(seed)

    profile = model_profile(model_config, dataset_name)
    run_name = f"gpt2_sft_{runner.safe_component(dataset_name)}_seed-{seed}"
    full_config = {
        "dataset_name": dataset_name,
        "seed": seed,
        "dataset": dict(dataset_config),
        "model": dict(model_config),
        "trainer": dict(trainer_config),
    }
    experiments_root = runner.configured_path(
        trainer_config, "experiments_dir", project_root=PROJECT_ROOT
    )
    run_dir = runner.create_run_dir(experiments_root, run_name, full_config)
    runner.setup_logging(run_dir)
    checkpoint_dir = checkpoint_root / runner.safe_component(dataset_name) / f"seed_{seed}"
    device = runner.resolve_device(args.device or trainer_config.get("device", "cuda"))

    data_cfg, test_eval_batch_size = build_data_config(
        group,
        dataset_name,
        dataset_config,
        seed=seed,
        semantic_ids_dir=semantic_ids_dir,
        user_hash_mod=user_ids_count(model_config, profile),
    )
    batch_plan = resolve_batch_plan(
        args, trainer_config, data_cfg=data_cfg, test_eval_batch_size=test_eval_batch_size
    )
    LOGGER.info("[%s seed=%s] batch plan: %s", dataset_name, seed, batch_plan.to_dict())
    data = group.data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError(f"Dataset {dataset_name!r} produced no semantic IDs.")

    model = build_model(model_config, profile, data)
    method = build_method(model_config)
    use_amp = bool(trainer_config.get("use_amp", False)) and device.startswith("cuda")

    data.cfg = replace(
        data.cfg,
        batch_size=batch_plan.train_batch_size,
        eval_batch_size=batch_plan.val_batch_size,
    )

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    if data.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support "
            f"train_example_layout={data.cfg.train_example_layout!r}; "
            f"supported: {model.supported_train_layouts}."
        )

    train_cfg = build_trainer_config(
        trainer_config, profile, checkpoint_dir, args, use_amp=use_amp
    )
    resolved = {
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "seed": seed,
        "device": device,
        "batch_plan": batch_plan.to_dict(),
        "data": runner.to_jsonable(
            {field.name: getattr(data.cfg, field.name) for field in fields(data.cfg)}
        ),
        "data_summary": data.summary(),
        "model": model.config_dict(),
        "method": method.cfg.to_dict(),
        "trainer": train_cfg.to_dict(),
        "optimizer": optimizer_config(trainer_config, profile),
        "ranking_ks": list(ranking_ks),
    }
    runner.write_json(run_dir / "resolved_config.json", resolved)
    runner.write_json(run_dir / "cli_args.json", vars(args))

    logger = runner.build_experiment_logger(
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
            **optimizer_config(trainer_config, profile),
            logger=logger,
        )
        fit_result = trainer.fit(train_loader, val_loader)

        checkpoint_for_eval = (
            fit_result.best_checkpoint_path or fit_result.last_checkpoint_path
        )
        if checkpoint_for_eval is not None:
            trainer.load_checkpoint(
                checkpoint_for_eval, load_optimizer=False, load_scheduler=False
            )

        # The final HR/NDCG/Coverage/ARP evaluation generates with the test
        # beam width, so it uses the smaller configured evaluation batch size.
        data.cfg = replace(data.cfg, eval_batch_size=batch_plan.test_batch_size)
        lookup = SemanticIdItemLookup(data.semantic_ids, data.item2id)
        evaluation_kwargs = {
            "device": trainer.device,
            "lookup": lookup,
            "item_popularity": runner.training_item_popularity(data),
            "num_catalog_items": int(data.num_items),
            "ks": ranking_ks,
            "num_beams": int(model_config.get("eval_num_beams", 20)),
            "constrain_valid_sequences": bool(
                model_config.get("constrain_valid_sequences", True)
            ),
            "max_batches": args.max_eval_batches,
        }
        validation_metrics = evaluate_generation_ranking_metrics(
            model, data.val_dataloader(), **evaluation_kwargs
        )
        test_metrics = evaluate_generation_ranking_metrics(
            model, data.test_dataloader(), **evaluation_kwargs
        )

        summary = {
            "seed": seed,
            "dataset": {
                "name": dataset_name,
                **runner.data_config_paths(data.cfg),
                **data.summary(),
            },
            "model": model.config_dict(),
            "batch_plan": batch_plan.to_dict(),
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
        runner.write_json(run_dir / "final_summary.json", summary)
        runner.write_text_summary(run_dir / "final_summary.md", summary)
        logger(
            {
                "type": "final_summary",
                "global_step": int(trainer.state.global_step),
                **runner.summary_to_metrics(summary),
            }
        )
        LOGGER.info(
            "Training finished: %s", json.dumps(runner.to_jsonable(summary), sort_keys=True)
        )
        return summary
    finally:
        logger.close()


def apply_trainer_defaults(trainer_config: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(trainer_config)
    missing = sorted(set(TRAINER_RUNTIME_DEFAULTS) - set(merged))
    for key in missing:
        merged[key] = TRAINER_RUNTIME_DEFAULTS[key]
    if missing:
        LOGGER.warning(
            "trainer_gpt2.yaml does not define %s; using the standard exp1 "
            "defaults. Pull the updated configs/exp1-standard_params to "
            "silence this warning.",
            missing,
        )
    return merged


def build_group_plans(
    config_dir: Path,
    requested: list[str] | None,
) -> list[GroupPlan]:
    explicit: list[str] | None = None
    if requested is not None and "all" not in requested:
        explicit = list(dict.fromkeys(requested))

    plans: list[GroupPlan] = []
    matched: set[str] = set()
    for group in GROUPS:
        dataset_config = runner.load_yaml(config_dir / group.data_config_name)
        available = list(dataset_config.get("datasets", {}))
        if explicit is None:
            names = available
        else:
            names = [name for name in available if name in explicit]
            matched.update(names)
        if names:
            plans.append(GroupPlan(group, dataset_config, names))

    if explicit is not None:
        unknown = [name for name in explicit if name not in matched]
        if unknown:
            available_all = [
                name
                for group in GROUPS
                for name in runner.load_yaml(config_dir / group.data_config_name).get(
                    "datasets", {}
                )
            ]
            raise KeyError(f"Unknown dataset(s) {unknown}; available: {available_all}")
    if not plans:
        raise ValueError("No datasets selected.")
    return plans


def model_profile(model_config: Mapping[str, Any], dataset_name: str) -> dict[str, Any]:
    profiles = model_config.get("datasets", {})
    if dataset_name not in profiles:
        raise KeyError(
            f"gpt2_model.yaml has no profile for {dataset_name!r}; "
            f"available: {sorted(profiles)}"
        )
    return dict(profiles[dataset_name])


def user_ids_count(
    model_config: Mapping[str, Any], profile: Mapping[str, Any]
) -> int:
    model_defaults = dict(model_config.get("defaults", {}).get("model", {}))
    model_values = {**model_defaults, **dict(profile.get("model", {}))}
    return int(model_values.get("user_ids_count", 1))


def build_data_config(
    group: DatasetGroup,
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    seed: int,
    semantic_ids_dir: Path,
    user_hash_mod: int,
) -> tuple[Any, int]:
    """Merge shared + per-dataset YAML values into the datamodule config.

    Returns the config plus the runner-only test_eval_batch_size (final
    20-beam evaluation). The ``sasrec:`` and ``tiger:`` sub-blocks are
    ignored — GPT-2 trains on the dense_causal layout with right padding,
    forced below — and the dataset YAML's ``gpt2:`` sub-blocks (shared and
    per-dataset) are merged instead, carrying the GPT-2 batch sizes and
    workers.
    """
    entry = dict(dataset_config["datasets"][dataset_name])
    entry.pop("sasrec", None)
    entry.pop("tiger", None)
    gpt2_values = dict(dataset_config.get("gpt2", {}))
    gpt2_values.update(dict(entry.pop("gpt2", {})))
    test_eval_batch_size = int(
        gpt2_values.pop("test_eval_batch_size", DEFAULT_TEST_EVAL_BATCH_SIZE)
    )
    if not gpt2_values:
        LOGGER.warning(
            "%s defines no gpt2 block for %s; GPT-2 falls back to the "
            "shared (SASRec-era) dataloader values.",
            group.data_config_name,
            dataset_name,
        )

    field_names = {field.name for field in fields(group.config_cls)}
    shared = {
        key: value for key, value in dataset_config.items() if key in field_names
    }
    unknown = sorted(set(entry) - field_names)
    if unknown:
        raise KeyError(
            f"Unknown {group.config_cls.__name__} dataset config keys: {unknown}"
        )
    unknown_gpt2 = sorted(set(gpt2_values) - field_names)
    if unknown_gpt2:
        raise KeyError(
            f"Unknown {group.config_cls.__name__} keys in the "
            f"{group.data_config_name} gpt2 block: {unknown_gpt2}"
        )
    values = {**shared, **entry, **gpt2_values}
    values["seed"] = int(seed)
    matched = bool(values.get("train_dense_match_prefix", False))
    if not matched:
        values["user_hash_mod"] = int(user_hash_mod)
        values["semantic_id_path"] = semantic_id_path_for(semantic_ids_dir, dataset_name)
    else:
        # A matched export records TIGER's exact preprocessing, including
        # nullable user hashing and the semantic-ID mapping used in the study.
        values.setdefault("user_hash_mod", user_hash_mod)
        values.setdefault("semantic_id_path", semantic_id_path_for(semantic_ids_dir, dataset_name))
    # GPT-2's dense-causal materialization: non-overlapping windows of
    # max_history_len items (null window settings default to that) with right
    # padding, matching how the model consumes token sequences. These are
    # model-family constants, not tunables, so they never come from the YAML.
    values["train_example_layout"] = REQUIRED_TRAIN_LAYOUT
    if matched:
        values.setdefault("train_window_len", values["max_history_len"])
        values.setdefault("train_window_stride", 1)
    else:
        values["train_window_len"] = None
        values["train_window_stride"] = None
    values["history_padding"] = "right"
    values["include_raw_in_batch"] = False
    if values.get("event_file") == "multievent":
        values["event_file"] = "multi_event"
    if values.get("allowed_event_types") is not None:
        values["allowed_event_types"] = tuple(values["allowed_event_types"])
    for key in ("train_weeks", "val_weeks", "test_weeks"):
        if values.get(key) is not None:
            values[key] = tuple(int(week) for week in values[key])
    runner.resolve_input_paths(values, base=PROJECT_ROOT)

    cfg = group.config_cls(**values)
    if cfg.train_example_layout != REQUIRED_TRAIN_LAYOUT:
        raise ValueError(
            f"Dataset {dataset_name!r} resolves "
            f"train_example_layout={cfg.train_example_layout!r}; GPT-2 requires "
            f"{REQUIRED_TRAIN_LAYOUT!r}."
        )
    return cfg, test_eval_batch_size


def semantic_id_path_for(semantic_ids_dir: Path, dataset_name: str) -> Path:
    return semantic_ids_dir / runner.safe_component(dataset_name) / "semantic_ids.json"


def validate_inputs(
    plans: list[GroupPlan],
    model_config: Mapping[str, Any],
    *,
    semantic_ids_dir: Path,
) -> None:
    missing: list[Path] = []
    for plan in plans:
        for dataset_name in plan.dataset_names:
            model_profile(model_config, dataset_name)
            cfg, _ = build_data_config(
                plan.group,
                dataset_name,
                plan.dataset_config,
                seed=0,
                semantic_ids_dir=semantic_ids_dir,
                user_hash_mod=1,
            )
            if getattr(cfg, "use_hf", False):
                raise ValueError(
                    "The exp1 runs are offline-only; set use_hf=false and "
                    "configure the local input paths for every Yambda / "
                    "VK-LSVD dataset."
                )
            try:
                required = runner.required_input_files(cfg)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Missing input files for {dataset_name!r}: {exc}"
                ) from exc
            missing.extend(path for path in required if not path.is_file())
    if missing:
        formatted = "\n".join(f"  - {path}" for path in dict.fromkeys(missing))
        raise FileNotFoundError(
            "Missing input files (raw data and/or semantic IDs from "
            f"train_tokenizer_all.py):\n{formatted}"
        )


def build_model(
    model_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    data: Any,
) -> GPT2RecModel:
    ks = resolve_ranking_ks(model_config)
    semantic_id_mapping = {
        item: data.semantic_ids[item]
        for item in data.id2item[1:]
        if item in data.semantic_ids
    }
    if not semantic_id_mapping:
        raise ValueError("No catalog item has a semantic ID.")

    model_defaults = dict(model_config.get("defaults", {}).get("model", {}))
    values = {**model_defaults, **dict(profile.get("model", {}))}
    embedding_dim = int(values.get("embedding_dim", 128))
    num_heads = int(values.get("num_heads", 4))
    if embedding_dim % num_heads != 0:
        raise ValueError(
            "GPT-2 embedding_dim must be divisible by num_heads: "
            f"{embedding_dim} % {num_heads} != 0."
        )

    # Position budget for the resolved data config: dense-causal training
    # windows (user token + (window + 1) items x sid_len tokens, next-token
    # shifted) and generation (user token + full history + one generated ID).
    max_history_len = int(data.cfg.max_history_len)
    sid_len = int(data.sid_len)
    dense_train_positions = 1 + (max_history_len + 1) * sid_len - 1
    generation_positions = 1 + max_history_len * sid_len + sid_len
    required_positions = max(dense_train_positions, generation_positions)

    dropout = float(values.get("dropout", 0.1))
    model = GPT2RecModel(
        codebook_size=int(values.get("codebook_size", 256)),
        sem_id_len=sid_len,
        user_ids_count=int(values.get("user_ids_count", 1)),
        semantic_id_mapping=semantic_id_mapping,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=int(values.get("num_layers", 4)),
        dim_feedforward=int(values.get("dim_feedforward", 512)),
        dropout=dropout,
        attention_dropout=float(values.get("attention_dropout", dropout)),
        residual_dropout=float(values.get("residual_dropout", dropout)),
        embedding_dropout=float(values.get("embedding_dropout", dropout)),
        layer_norm_eps=float(values.get("layer_norm_eps", 1e-5)),
        initializer_range=float(values.get("initializer_range", 0.02)),
        max_position_embeddings=int(
            values.get("max_position_embeddings", required_positions)
        ),
        num_beams=int(model_config.get("eval_num_beams", 20)),
        num_return_sequences=max(ks),
        fast_greedy_generation=bool(values.get("fast_greedy_generation", True)),
    )
    if bool(model_config.get("gradient_checkpointing", False)):
        model.model.gradient_checkpointing_enable()
        model.model.config.use_cache = False
        LOGGER.info("Enabled GPT-2 gradient checkpointing.")
    return model


def build_method(model_config: Mapping[str, Any]) -> SFTMethod:
    sft = dict(model_config.get("sft", {}))
    val_num_beams = int(model_config.get("val_num_beams", 20))
    sft.setdefault("generation_topk", [5, 10, 20])
    sft.setdefault("generation_num_beams", val_num_beams)
    sft.setdefault("generation_num_return_sequences", max(sft["generation_topk"]))
    sft.setdefault(
        "generation_constrain_valid_sequences",
        bool(model_config.get("constrain_valid_sequences", True)),
    )
    # exp1 never excludes seen items at evaluation, for any model.
    sft.setdefault("generation_exclude_seen_items", False)
    return SFTMethod(SFTConfig.from_dict(sft))


def build_trainer_config(
    trainer_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    checkpoint_dir: Path,
    args: argparse.Namespace,
    *,
    use_amp: bool,
) -> TrainerConfig:
    allowed = {field.name for field in fields(TrainerConfig)} - {"checkpoint_dir"}
    runtime_keys = {
        "weight_decay",
        "betas",
        "eps",
        "device",
        "seeds",
        "checkpoints_dir",
        "experiments_dir",
        "artifacts_dir",
        "metrics_summary_name",
        "comet",
    }
    unknown = sorted(set(trainer_config) - allowed - runtime_keys)
    if unknown:
        raise KeyError(f"Unknown GPT-2 trainer config keys: {unknown}")

    values = {key: value for key, value in trainer_config.items() if key in allowed}
    # Per-dataset trainer overrides from the gpt2_model.yaml profile (the
    # Yambda profiles use a fixed-sample in-training validation).
    overrides = dict(profile.get("training", {}))
    unknown_overrides = sorted(set(overrides) - allowed)
    if unknown_overrides:
        raise KeyError(
            f"Unknown TrainerConfig keys in the gpt2_model.yaml training "
            f"override: {unknown_overrides}"
        )
    values.update(overrides)
    values["use_amp"] = use_amp
    if args.max_epochs is not None:
        values["max_epochs"] = int(args.max_epochs)
    if args.max_steps is not None:
        values["max_steps"] = int(args.max_steps)
    if args.max_eval_batches is not None:
        values["max_eval_batches"] = int(args.max_eval_batches)
    values["checkpoint_dir"] = checkpoint_dir
    return TrainerConfig.from_dict(values)


def optimizer_config(
    trainer_config: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """AdamW settings: shared values from trainer_gpt2.yaml, lr from the
    dataset's gpt2_model.yaml profile (tuned jointly with its architecture)."""
    optimizer = dict(profile.get("optimizer", {}))
    if "lr" not in optimizer:
        raise KeyError(
            "The gpt2_model.yaml dataset profile must define optimizer.lr."
        )
    betas = trainer_config.get("betas", (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"Optimizer betas must contain two values; got {betas!r}.")
    return {
        "lr": float(optimizer["lr"]),
        "weight_decay": float(trainer_config.get("weight_decay", 1e-2)),
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(trainer_config.get("eps", 1e-8)),
    }


def resolve_ranking_ks(model_config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = model_config.get("defaults", {}).get("ks", runner.DEFAULT_RANKING_KS)
    values = tuple(int(k) for k in raw)
    if values != runner.DEFAULT_RANKING_KS:
        raise ValueError(
            f"This experiment requires ks={runner.DEFAULT_RANKING_KS}; got {values}."
        )
    return values


# ---------------------------------------------------------------------------
# Batch sizing (fixed by the dataset YAML gpt2 blocks)
# ---------------------------------------------------------------------------


def resolve_batch_plan(
    args: argparse.Namespace,
    trainer_config: Mapping[str, Any],
    *,
    data_cfg: Any,
    test_eval_batch_size: int,
) -> BatchPlan:
    """Batch sizes from the dataset YAML's gpt2 block; CLI overrides win.

    batch_size feeds the training loader, eval_batch_size the in-training
    20-beam validation, and test_eval_batch_size the final 20-beam val/test
    evaluation.
    """
    return BatchPlan(
        train_batch_size=int(args.train_batch_size or data_cfg.batch_size),
        gradient_accumulation_steps=int(
            trainer_config.get("gradient_accumulation_steps", 1)
        ),
        val_batch_size=int(args.val_batch_size or data_cfg.eval_batch_size),
        test_batch_size=int(args.test_batch_size or test_eval_batch_size),
    )


# ---------------------------------------------------------------------------
# HR / NDCG / Coverage / ARP evaluation for generated semantic IDs
# ---------------------------------------------------------------------------


class SemanticIdItemLookup:
    """Vectorized semantic-ID-tuple -> datamodule item-ID lookup.

    Semantic-ID tuples are packed into int64 keys (base = max token + 1) and
    resolved through a sorted tensor + searchsorted, so decoding beam-search
    output stays cheap even for the Yambda catalogs.
    """

    def __init__(
        self,
        semantic_ids: Mapping[str, list[int]],
        item2id: Mapping[str, int],
    ) -> None:
        self.sid_len = len(next(iter(semantic_ids.values())))
        max_token = max(max(tokens) for tokens in semantic_ids.values())
        self.base = int(max_token) + 1
        if self.base**self.sid_len >= 2**63:
            raise ValueError(
                f"Cannot pack semantic IDs (base={self.base}, len={self.sid_len}) "
                "into int64 keys."
            )

        keys: list[int] = []
        values: list[int] = []
        for raw_item, tokens in semantic_ids.items():
            item_id = item2id.get(str(raw_item))
            if item_id is None:
                continue
            key = 0
            for token in tokens:
                key = key * self.base + int(token)
            keys.append(key)
            values.append(int(item_id))
        if not keys:
            raise ValueError("No semantic ID maps to a catalog item.")

        keys_t = torch.tensor(keys, dtype=torch.int64)
        values_t = torch.tensor(values, dtype=torch.int64)
        order = keys_t.argsort()
        self._keys = keys_t[order]
        self._values = values_t[order]

    def lookup(self, generated: torch.Tensor) -> torch.Tensor:
        """Map generated tokens [..., sid_len] to item IDs; -1 when unknown."""
        flat = generated.detach().cpu().to(torch.int64).reshape(-1, generated.size(-1))
        in_range = (flat >= 0).all(dim=1) & (flat < self.base).all(dim=1)
        packed = torch.zeros(flat.size(0), dtype=torch.int64)
        for level in range(self.sid_len):
            packed = packed * self.base + flat[:, level].clamp(0, self.base - 1)

        positions = torch.searchsorted(self._keys, packed).clamp(max=self._keys.numel() - 1)
        found = in_range & self._keys[positions].eq(packed)
        item_ids = torch.where(found, self._values[positions], torch.tensor(-1))
        return item_ids.view(generated.shape[:-1])


def _generate_ranking_recommendations(
    model: GPT2RecModel,
    batch: Mapping[str, Any],
    *,
    lookup: SemanticIdItemLookup,
    num_beams: int,
    num_return_sequences: int,
    constrain_valid_sequences: bool,
    max_batch_size: int | None,
) -> tuple[torch.Tensor, int | None]:
    """Retry only failed generation chunks, preserving row order and beams.

    Decoded recommendations live on CPU. The returned cap is reused for the
    rest of the split, so every loader batch need not rediscover the GPU limit.
    """
    size = int(batch["target_item_ids"].shape[0])
    start = 0
    recommendations = []
    while start < size:
        stop = min(size, start + (max_batch_size or size))
        chunk = {
            key: value[start:stop]
            if ((torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == size)
                or (isinstance(value, (list, tuple)) and len(value) == size))
            else value
            for key, value in batch.items()
        }
        try:
            generated = model.generate(
                chunk,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                constrain_valid_sequences=constrain_valid_sequences,
            )
        except torch.cuda.OutOfMemoryError:
            if stop - start <= 1:
                raise
            max_batch_size = max(1, (stop - start) // 2)
        else:
            recommendations.append(lookup.lookup(generated))
            del generated
            start = stop
            continue

        # Leave the except block first: the traceback owns the failed forward's
        # activations. Free those references before releasing cached GPU memory.
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.warning(
            "GPT-2 evaluation CUDA OOM at batch=%d, beams=%d; retrying the same "
            "examples with generation batch=%d",
            stop - start, num_beams, max_batch_size,
        )

    return torch.cat(recommendations, dim=0), max_batch_size


@torch.no_grad()
def evaluate_generation_ranking_metrics(
    model: GPT2RecModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device,
    lookup: SemanticIdItemLookup,
    item_popularity: torch.Tensor,
    num_catalog_items: int,
    ks: tuple[int, ...],
    num_beams: int,
    constrain_valid_sequences: bool,
    pad_item_id: int = 0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Beam-search HR, NDCG, Coverage, and train-count ARP for GPT-2.

    Mirrors run_tiger_sft_all.evaluate_generation_ranking_metrics: HR/NDCG
    compare the decoded top-k items against the held-out target, Coverage
    counts distinct recommended catalog items, and ARP averages
    training-interaction counts over the recommended items that decode to
    catalog entries. CUDA OOM halves only the generation batch, leaving the
    beam count, evaluated examples, and loader-level max_batches unchanged.
    """
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"ks must contain positive integers; got {ks}.")
    if num_catalog_items <= 0:
        raise ValueError("num_catalog_items must be positive.")

    model.eval()
    device_t = torch.device(device)
    popularity = item_popularity.detach().cpu().to(torch.float64)
    item_vocab_size = int(popularity.numel())
    max_k = max(ks)

    total = 0
    hit_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    covered = {k: torch.zeros(item_vocab_size, dtype=torch.bool) for k in ks}
    arp_sum = {k: 0.0 for k in ks}
    arp_count = {k: 0 for k in ks}
    generation_batch_size = None

    for batch_idx, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device_t, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        recommendations, generation_batch_size = _generate_ranking_recommendations(
            model,
            batch,
            lookup=lookup,
            num_beams=num_beams,
            num_return_sequences=max_k,
            constrain_valid_sequences=constrain_valid_sequences,
            max_batch_size=generation_batch_size,
        )
        targets = batch["target_item_ids"].detach().cpu().long()
        valid_targets = (
            targets.ne(pad_item_id) & targets.ge(0) & targets.lt(item_vocab_size)
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

            valid_recommendations = topk.ne(pad_item_id) & topk.ge(0)
            item_ids = topk[valid_recommendations]
            if item_ids.numel():
                covered[k][item_ids.unique()] = True
                arp_sum[k] += float(popularity[item_ids].sum().item())
                arp_count[k] += int(item_ids.numel())

    if total == 0:
        raise RuntimeError("No valid evaluation examples found.")

    metrics: dict[str, float] = {"num_examples": float(total)}
    for k in ks:
        metrics[f"HR@{k}"] = hit_sum[k] / total
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / total
        metrics[f"Coverage@{k}"] = float(covered[k].sum().item()) / float(num_catalog_items)
        metrics[f"ARP@{k}"] = arp_sum[k] / max(arp_count[k], 1)
    return metrics


if __name__ == "__main__":
    main()
