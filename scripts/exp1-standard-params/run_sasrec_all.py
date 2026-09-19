"""Train config-driven SASRec on every exp1 dataset with a single entry point.

Covers the two small Amazon datasets (Beauty 2014, Instruments 2018), the
large Amazon Beauty 2023 dataset, the two Yambda datasets (50M and 500M
listens), and the VK-LSVD subsamples (provided weekly split), running three
seeds per dataset. SASRec trains identically on every dataset: dense-causal
windows with left padding and full-catalog cross-entropy; the script refuses
configs that resolve to any other training layout. ``--variant sce`` trains the same datasets with scalable cross-entropy
(SCE) instead, keeping every other hyperparameter and writing checkpoints,
experiments, and metrics to the separate folders each trainer YAML defines
under ``variants:``, so the full-CE baseline is never overwritten. Best
checkpoints are saved under
``checkpoints/exp1-standard_params/sasrec/`` and validation/test HR, NDCG,
Coverage, and ARP for every completed run are combined in
``artifacts/exp1-standard_params/sasrec_all_metrics_summary.csv``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.vk_lsvd import VKLSVDDataConfig, VKLSVDDataModule
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from scripts import sasrec_config_runner as runner


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
MODEL_CONFIG_NAME = "sasrec_model.yaml"
COMBINED_METRICS_SUMMARY_NAME = "sasrec_all_metrics_summary.csv"
# The exp1 experiment trains SASRec the same way on every dataset.
REQUIRED_TRAIN_LAYOUT = "dense_causal"


@dataclass(frozen=True)
class DatasetGroup:
    """One exp1 dataset family sharing data/trainer YAMLs and a datamodule."""

    key: str
    data_config_name: str
    trainer_config_name: str
    data_module: type
    build_data_config: Callable[..., Any]
    validate_inputs: Callable[[list[str], Mapping[str, Any]], None]


@dataclass(frozen=True)
class GroupPlan:
    group: DatasetGroup
    dataset_config: dict[str, Any]
    trainer_config: dict[str, Any]
    dataset_names: list[str]


def build_amazon_data_config(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    seed: int,
) -> AmazonReviewDataConfig:
    values = merge_dataset_values(
        AmazonReviewDataConfig, dataset_name, dataset_config, seed=seed
    )
    if "reviews_path" not in values:
        raise KeyError(f"Dataset {dataset_name!r} does not define reviews_path.")
    runner.resolve_input_paths(values, base=PROJECT_ROOT)
    return AmazonReviewDataConfig(**values)


def build_yambda_data_config(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    seed: int,
) -> YambdaDataConfig:
    values = merge_dataset_values(
        YambdaDataConfig, dataset_name, dataset_config, seed=seed
    )
    # Accept the spelling from informal experiment notes while always passing
    # the actual Hugging Face filename stem expected by the datamodule.
    if values.get("event_file") == "multievent":
        values["event_file"] = "multi_event"
    if values.get("allowed_event_types") is not None:
        values["allowed_event_types"] = tuple(values["allowed_event_types"])
    runner.resolve_input_paths(values, base=PROJECT_ROOT)
    return YambdaDataConfig(**values)


VK_LSVD_SOURCE_KEYS = (
    "interactions_root",
    "interactions_path",
    "train_paths",
    "val_paths",
    "test_paths",
)


def build_vk_lsvd_data_config(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    seed: int,
) -> VKLSVDDataConfig:
    values = merge_dataset_values(
        VKLSVDDataConfig, dataset_name, dataset_config, seed=seed
    )
    if not values.get("use_hf") and not any(values.get(key) for key in VK_LSVD_SOURCE_KEYS):
        raise KeyError(
            f"Dataset {dataset_name!r} does not define interactions_root "
            "(or explicit train/val/test paths)."
        )
    for key in ("train_weeks", "val_weeks", "test_weeks"):
        if values.get(key) is not None:
            values[key] = tuple(int(week) for week in values[key])
    runner.resolve_input_paths(values, base=PROJECT_ROOT)
    return VKLSVDDataConfig(**values)


def merge_dataset_values(
    config_cls: type,
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    entry = dict(dataset_config["datasets"][dataset_name])
    sasrec_data = dict(dataset_config.get("sasrec", {}))
    sasrec_data.update(dict(entry.pop("sasrec", {})))

    field_names = {field.name for field in fields(config_cls)}
    shared = {
        key: value for key, value in dataset_config.items() if key in field_names
    }
    unknown = sorted(set(entry) - field_names)
    if unknown:
        raise KeyError(
            f"Unknown {config_cls.__name__} dataset config keys: {unknown}"
        )
    unknown_sasrec = sorted(set(sasrec_data) - field_names)
    if unknown_sasrec:
        raise KeyError(
            f"Unknown {config_cls.__name__} SASRec data config keys: {unknown_sasrec}"
        )
    return {**shared, **entry, **sasrec_data, "seed": int(seed)}


def validate_amazon_inputs(
    dataset_names: list[str],
    dataset_config: Mapping[str, Any],
) -> None:
    missing: list[Path] = []
    for dataset_name in dataset_names:
        cfg = build_amazon_data_config(dataset_name, dataset_config, seed=0)
        for path in (cfg.reviews_path, cfg.metadata_path):
            if path is not None and not Path(path).is_file():
                missing.append(Path(path))
    raise_if_missing(missing, source="Amazon review")


def validate_yambda_inputs(
    dataset_names: list[str],
    dataset_config: Mapping[str, Any],
) -> None:
    missing: list[Path] = []
    for dataset_name in dataset_names:
        cfg = build_yambda_data_config(dataset_name, dataset_config, seed=0)
        if cfg.use_hf:
            raise ValueError(
                "The exp1 Yambda runs are offline-only; set use_hf=false "
                "and configure interactions_path for every dataset."
            )
        for path in (cfg.interactions_path, cfg.embeddings_path):
            if path is None or not path.is_file():
                missing.append(path or Path("<not configured>"))
    raise_if_missing(missing, source="Yambda")


def validate_vk_lsvd_inputs(
    dataset_names: list[str],
    dataset_config: Mapping[str, Any],
) -> None:
    missing: list[Path] = []
    for dataset_name in dataset_names:
        cfg = build_vk_lsvd_data_config(dataset_name, dataset_config, seed=0)
        if cfg.use_hf:
            raise ValueError(
                "The exp1 VK-LSVD runs are offline-only; set use_hf=false "
                "and configure interactions_root for every dataset."
            )
        try:
            required = runner.required_input_files(cfg)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing local VK-LSVD input files for {dataset_name!r}: {exc}"
            ) from exc
        missing.extend(path for path in required if not path.is_file())
    raise_if_missing(missing, source="VK-LSVD")


def raise_if_missing(missing: list[Path], *, source: str) -> None:
    if not missing:
        return
    formatted = "\n".join(f"  - {path}" for path in dict.fromkeys(missing))
    raise FileNotFoundError(
        f"Missing local {source} input files. Download/copy them before "
        f"starting training:\n{formatted}"
    )


# Training order: small Amazon first, then Amazon 2023, Yambda, VK-LSVD.
GROUPS: tuple[DatasetGroup, ...] = (
    DatasetGroup(
        key="amazon",
        data_config_name="dataset_amazon.yaml",
        trainer_config_name="trainer_sasrec.yaml",
        data_module=AmazonReviewDataModule,
        build_data_config=build_amazon_data_config,
        validate_inputs=validate_amazon_inputs,
    ),
    DatasetGroup(
        key="amazon_2023",
        data_config_name="dataset_amazon_2023.yaml",
        trainer_config_name="trainer_sasrec_amazon_2023.yaml",
        data_module=AmazonReviewDataModule,
        build_data_config=build_amazon_data_config,
        validate_inputs=validate_amazon_inputs,
    ),
    DatasetGroup(
        key="yambda",
        data_config_name="dataset_yambda.yaml",
        trainer_config_name="trainer_sasrec_yambda.yaml",
        data_module=YambdaDataModule,
        build_data_config=build_yambda_data_config,
        validate_inputs=validate_yambda_inputs,
    ),
    DatasetGroup(
        key="vk_lsvd",
        data_config_name="dataset_vk_lsvd.yaml",
        trainer_config_name="trainer_sasrec_vk_lsvd.yaml",
        data_module=VKLSVDDataModule,
        build_data_config=build_vk_lsvd_data_config,
        validate_inputs=validate_vk_lsvd_inputs,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train config-driven SASRec on the exp1 datasets (Amazon Beauty "
            "2014, Instruments 2018, Beauty 2023, Yambda 50M, Yambda 500M, "
            "and the VK-LSVD subsamples) for three seeds each and report "
            "validation/test ranking, coverage, and popularity metrics."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys to train on (any mix of beauty, instruments, "
            "beauty_2023, yambda_50m_listens, yambda_500m_listens, "
            "vk_lsvd_ur0.01_ir0.01, vk_lsvd_ur0.01_ip0.01) or 'all'; "
            "defaults to every configured dataset."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Objective variant defined under `variants:` in sasrec_model.yaml "
            "and the trainer YAMLs (e.g. sce). Trains every selected dataset "
            "with the variant's sft block and the baseline hyperparameters, "
            "writing checkpoints/metrics to the variant's separate folders."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    model_config = runner.resolve_model_variant(
        runner.load_yaml(config_dir / MODEL_CONFIG_NAME), args.variant
    )
    plans = build_group_plans(config_dir, args.datasets, variant=args.variant)

    # Fail before any multi-day training if a later group's inputs are absent
    # or a config drifted away from the uniform training layout.
    validate_train_layout(plans)
    for plan in plans:
        plan.group.validate_inputs(plan.dataset_names, plan.dataset_config)

    # Single-dataset invocations (e.g. one Slurm array task per dataset) skip
    # the shared combined summary so concurrent jobs never overwrite it; the
    # per-dataset summaries from run_sweep already cover them.
    selected_total = sum(len(plan.dataset_names) for plan in plans)
    combined_path = (
        combined_metrics_summary_path(plans) if selected_total > 1 else None
    )
    combined_summaries: list[Mapping[str, Any]] = []

    def record_run(summary: Mapping[str, Any]) -> None:
        combined_summaries.append(summary)
        runner.write_metrics_summary(combined_path, combined_summaries)

    for plan in plans:
        runner.run_sweep(
            dataset_names=plan.dataset_names,
            seeds=runner.resolve_seeds(args, plan.trainer_config),
            data_factory=make_data_factory(plan),
            dataset_config=plan.dataset_config,
            model_config=model_config,
            trainer_config=plan.trainer_config,
            config_dir=config_dir,
            project_root=PROJECT_ROOT,
            args=args,
            on_run_complete=record_run if combined_path is not None else None,
            variant=args.variant,
        )

    if combined_path is not None:
        runner.LOGGER.info(
            "Saved combined validation/test metrics summary to %s", combined_path
        )


def build_group_plans(
    config_dir: Path,
    requested: list[str] | None,
    *,
    variant: str | None = None,
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
            selection = str(dataset_config.get("dataset", "all"))
            names = available if selection == "all" else [selection]
            missing = [name for name in names if name not in available]
            if missing:
                raise KeyError(
                    f"{group.data_config_name} selects unknown dataset(s) "
                    f"{missing}; available: {available}"
                )
        else:
            names = [name for name in available if name in explicit]
            matched.update(names)
        if not names:
            continue
        trainer_config = runner.apply_trainer_variant(
            runner.load_yaml(config_dir / group.trainer_config_name), variant
        )
        plans.append(GroupPlan(group, dataset_config, trainer_config, names))

    if explicit is not None:
        unknown = [name for name in explicit if name not in matched]
        if unknown:
            available_all = [
                name
                for group in GROUPS
                for name in runner.load_yaml(
                    config_dir / group.data_config_name
                ).get("datasets", {})
            ]
            raise KeyError(
                f"Unknown dataset(s) {unknown}; available: {available_all}"
            )
    if not plans:
        raise ValueError("No datasets selected.")
    return plans


def validate_train_layout(plans: list[GroupPlan]) -> None:
    for plan in plans:
        for dataset_name in plan.dataset_names:
            cfg = plan.group.build_data_config(
                dataset_name, plan.dataset_config, seed=0
            )
            layout = getattr(cfg, "train_example_layout", None)
            if layout != REQUIRED_TRAIN_LAYOUT:
                raise ValueError(
                    f"Dataset {dataset_name!r} resolves "
                    f"train_example_layout={layout!r}; the exp1 experiment "
                    f"trains SASRec identically on every dataset and requires "
                    f"{REQUIRED_TRAIN_LAYOUT!r}."
                )


def combined_metrics_summary_path(plans: list[GroupPlan]) -> Path:
    roots = {
        runner.configured_path(
            plan.trainer_config, "artifacts_dir", project_root=PROJECT_ROOT
        )
        for plan in plans
    }
    if len(roots) != 1:
        raise ValueError(
            "All exp1 trainer configs must share one artifacts_dir; got "
            f"{sorted(str(root) for root in roots)}."
        )
    return roots.pop() / COMBINED_METRICS_SUMMARY_NAME


def make_data_factory(plan: GroupPlan) -> runner.DataFactory:
    def factory(dataset_name: str, seed: int) -> Any:
        data_config = plan.group.build_data_config(
            dataset_name, plan.dataset_config, seed=seed
        )
        return plan.group.data_module(data_config)

    return factory


if __name__ == "__main__":
    main()
