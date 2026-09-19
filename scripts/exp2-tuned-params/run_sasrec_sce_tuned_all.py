"""Train the Optuna-tuned SASRec + scalable cross-entropy (SCE) models for the
final exp2 runs: three seeds per dataset.

For each dataset tuned by ``tuning_sasrec_sce.py`` (Amazon Beauty 2023,
Yambda 50M listens, Yambda 500M listens) this script reads the tuned
hyperparameters from ``configs/exp2-tuned-params/sasrec_sce_model_<dataset>.yaml``
(architecture, AdamW settings, and the resolved SCE bucket block), trains
SASRec with SCE for seeds 42/43/44 under the regime the dataset was tuned in
(monitor ``val/sft/ndcg@10``, epoch cap, patience, accumulation), evaluates
the best checkpoint of every run on the validation and test splits
(full-catalog HR, NDCG, Coverage, and ARP at 5/10/20), and writes:

- ``artifacts/exp2-tuned-params/sasrec_sce/sasrec_sce_<dataset>_metrics_summary.csv``
  per-dataset validation/test metric table (one row per seed and split)
- ``artifacts/exp2-tuned-params/sasrec_sce/sasrec_sce_all_metrics_summary.csv``
  combined table (only for multi-dataset invocations, so concurrent
  single-dataset cluster jobs never overwrite it)
- ``checkpoints/exp2-tuned_params/sasrec_sce/<dataset>/seed_<seed>/best.pt``
  best checkpoints, next to the tuning outputs (``best_model.pt`` etc.)
- ``experiments/exp2-tuned-params/sasrec_sce/<run>/`` resolved configs, logs,
  and ``final_summary.json`` per run

Validation and test always score the full catalog exactly; SCE only replaces
the training objective. ``evaluate_sasrec_sce_tuned.py --seeds 42 43 44``
re-scores the seed checkpoints later without retraining.

Usage::

    python scripts/exp2-tuned-params/run_sasrec_sce_tuned_all.py             # all three
    python scripts/exp2-tuned-params/run_sasrec_sce_tuned_all.py --datasets beauty_2023

On the cluster one Slurm array task per dataset::

    sbatch scripts/slurm/run_sasrec_sce_tuned_all_array.sbatch
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.models.sasrec import SASRecConfig
from scripts import sasrec_config_runner as runner


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
# Search-space budgets (epoch cap, patience, accumulation), dataset groups,
# and the tuned-config path layout are shared with the tuning script.
sce_tuning = _load_module("exp2_tuning_sasrec_sce", _HERE / "tuning_sasrec_sce.py")
tuning = sce_tuning.tuning

CONFIG_DIR = sce_tuning.CONFIG_DIR
CHECKPOINTS_DIR = "checkpoints/exp2-tuned_params/sasrec_sce"
EXPERIMENTS_DIR = "experiments/exp2-tuned-params/sasrec_sce"
ARTIFACTS_DIR = "artifacts/exp2-tuned-params/sasrec_sce"
METRICS_SUMMARY_NAME = "sasrec_sce_{dataset}_metrics_summary.csv"
COMBINED_METRICS_SUMMARY_NAME = "sasrec_sce_all_metrics_summary.csv"

DATASETS = sce_tuning.DATASETS_TO_TUNE
SEEDS = tuning.RERANK_SEEDS  # (42, 43, 44), the exp1/exp2 protocol seeds
ITEM_LOSS = sce_tuning.ITEM_LOSS
VARIANT = "sce"

# Keys of the tuned YAML that belong to the optimizer/trainer, not to
# SASRecConfig; everything else (plus the sft block) is the model config.
TRAINER_KEYS = {
    "lr",
    "weight_decay",
    "betas",
    "eps",
    "max_grad_norm",
    "gradient_accumulation_steps",
}


@dataclass(frozen=True)
class OutputDirs:
    """Output roots, project-root relative unless absolute."""

    checkpoints_dir: str = CHECKPOINTS_DIR
    experiments_dir: str = EXPERIMENTS_DIR
    artifacts_dir: str = ARTIFACTS_DIR


def load_tuned_config(
    dataset_name: str,
    *,
    tuned_config_dir: Path = CONFIG_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a tuned SCE YAML into (model config, optimizer/trainer overrides)."""

    path = sce_tuning.tuned_config_path(dataset_name, tuned_config_dir=tuned_config_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Tuned SCE config not found: {path}. Run the SCE tuning first "
            "(scripts/exp2-tuned-params/tuning_sasrec_sce.py or the "
            "scripts/slurm/tuning_sasrec_sce_array.sbatch workers)."
        )
    raw = runner.load_yaml(path)

    model_fields = {field.name for field in fields(SASRecConfig)}
    model_config = {
        key: value for key, value in raw.items() if key in model_fields or key == "sft"
    }
    trainer_overrides = {key: raw[key] for key in TRAINER_KEYS if key in raw}
    unknown = sorted(set(raw) - set(model_config) - TRAINER_KEYS)
    if unknown:
        raise KeyError(f"Unknown keys in {path.name}: {unknown}")
    for key in ("lr", "weight_decay"):
        if key not in trainer_overrides:
            raise KeyError(f"{path.name} does not define {key}.")

    sft = dict(model_config.get("sft") or {})
    if str(sft.get("item_loss", "")).lower() != ITEM_LOSS:
        raise ValueError(
            f"{path.name} does not train with {ITEM_LOSS} "
            f"(sft.item_loss={sft.get('item_loss')!r}); this runner is for the "
            "SCE-tuned configs only."
        )
    sce = dict(sft.get("scalable_cross_entropy") or {})
    for key in ("n_buckets", "bucket_size_x", "bucket_size_y"):
        value = sce.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{path.name} sft.scalable_cross_entropy.{key} must be a positive "
                f"integer; got {value!r} (the tuning script writes resolved sizes)."
            )
    return model_config, trainer_overrides


def build_trainer_config(
    dataset_name: str,
    group_key: str,
    tuned: Mapping[str, Any],
    *,
    spec_name: str | None = None,
    dirs: OutputDirs = OutputDirs(),
) -> dict[str, Any]:
    """exp1-style trainer settings under the regime the dataset was tuned in."""

    spec = sce_tuning.TUNING_SPECS[spec_name or dataset_name]
    return {
        # Tuned optimizer (AdamW) and gradient settings.
        "lr": float(tuned["lr"]),
        "weight_decay": float(tuned["weight_decay"]),
        "betas": list(tuned.get("betas", (0.9, 0.999))),
        "eps": float(tuned.get("eps", 1e-8)),
        "max_grad_norm": float(tuned.get("max_grad_norm", tuning.MAX_GRAD_NORM)),
        "gradient_accumulation_steps": int(
            tuned.get("gradient_accumulation_steps", spec.grad_accum)
        ),
        # Same budget and early stopping as the tuning trials.
        "max_epochs": spec.max_epochs,
        "max_steps": None,
        "monitor": tuning.MONITOR,
        "monitor_mode": "max",
        "early_stopping_patience": spec.patience,
        "early_stopping_mode": "best",
        "min_delta": 0.0,
        "validate_every_epochs": 1,
        "validate_every_steps": None,
        "run_validation_at_start": False,
        "skip_non_finite_loss": False,
        "log_every_steps": tuning.LOG_EVERY_STEPS,
        "progress_bar": True,
        "save_best": True,
        "save_last": False,
        "use_amp": True,
        "amp_dtype": tuning.AMP_DTYPE,
        # Runtime and outputs.
        "device": "cuda",
        "seeds": list(SEEDS),
        "ranking_ks": list(runner.DEFAULT_RANKING_KS),
        "checkpoints_dir": dirs.checkpoints_dir,
        "experiments_dir": dirs.experiments_dir,
        "artifacts_dir": dirs.artifacts_dir,
        # {dataset} keeps concurrent per-dataset cluster jobs from rewriting
        # the same metrics file.
        "metrics_summary_name": METRICS_SUMMARY_NAME,
        # Mirrors exp1/exp2: the small-Amazon auxiliary store is cheap and
        # feeds the reward pipeline; the large stores are not duplicated.
        "save_interaction_store": group_key == "amazon",
        "comet": {"enabled": False},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Optuna-tuned SASRec + SCE models on beauty_2023, "
            "yambda_50m_listens and yambda_500m_listens for three seeds each and "
            "report validation/test ranking, coverage, and popularity metrics."
        )
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=list(DATASETS),
        help=(
            "Dataset keys to train (default: all three). A key outside the "
            "default set must be defined in a dataset YAML under --config-dir "
            "and paired with --spec."
        ),
    )
    parser.add_argument(
        "--spec", type=str, default=None, choices=list(sce_tuning.TUNING_SPECS),
        help="Tuning regime (epoch cap, patience) to apply to every dataset (default: the dataset's own).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Three distinct seeds (default: 42 43 44).")
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override the dataset YAML's DataLoader workers.")
    # Paths (defaults are the cluster protocol locations).
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                        help="Directory with the exp2 dataset YAMLs.")
    parser.add_argument("--tuned-config-dir", type=Path, default=CONFIG_DIR,
                        help="Where sasrec_sce_model_<dataset>.yaml is read from.")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path(CHECKPOINTS_DIR))
    parser.add_argument("--experiments-dir", type=Path, default=Path(EXPERIMENTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    tuned_config_dir = runner.resolve_path(args.tuned_config_dir, base=PROJECT_ROOT)
    dirs = OutputDirs(
        checkpoints_dir=str(args.checkpoints_dir),
        experiments_dir=str(args.experiments_dir),
        artifacts_dir=str(args.artifacts_dir),
    )

    # Fail before any multi-hour training if a tuned config, an input file,
    # or the required training layout is missing for any selected dataset.
    plans: list[tuple[str, Any, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for dataset_name in args.datasets:
        spec_name = args.spec or dataset_name
        if spec_name not in sce_tuning.TUNING_SPECS:
            raise KeyError(
                f"No tuning regime for dataset {dataset_name!r}; pass --spec "
                f"{list(sce_tuning.TUNING_SPECS)} to reuse one."
            )
        group, dataset_config = sce_tuning.find_group(dataset_name, config_dir=config_dir)
        model_config, tuned_trainer = load_tuned_config(
            dataset_name, tuned_config_dir=tuned_config_dir
        )
        trainer_config = build_trainer_config(
            dataset_name, group.key, tuned_trainer, spec_name=spec_name, dirs=dirs
        )

        data_cfg = group.build_data_config(dataset_name, dataset_config, seed=0)
        layout = getattr(data_cfg, "train_example_layout", None)
        if layout != sce_tuning.REQUIRED_TRAIN_LAYOUT:
            raise ValueError(
                f"Dataset {dataset_name!r} resolves train_example_layout={layout!r}; "
                f"the exp2 runs require {sce_tuning.REQUIRED_TRAIN_LAYOUT!r}."
            )
        group.validate_inputs([dataset_name], dataset_config)
        plans.append((dataset_name, group, dataset_config, model_config, trainer_config))

    # Single-dataset invocations (e.g. one cluster job per dataset) skip the
    # shared combined summary so concurrent jobs never overwrite it.
    combined_path = (
        runner.resolve_path(dirs.artifacts_dir, base=PROJECT_ROOT) / COMBINED_METRICS_SUMMARY_NAME
        if len(plans) > 1
        else None
    )
    combined_summaries: list[Mapping[str, Any]] = []

    def record_run(summary: Mapping[str, Any]) -> None:
        combined_summaries.append(summary)
        runner.write_metrics_summary(combined_path, combined_summaries)

    for dataset_name, group, dataset_config, model_config, trainer_config in plans:

        def factory(
            name: str,
            seed: int,
            _group: Any = group,
            _dataset_config: dict[str, Any] = dataset_config,
        ) -> Any:
            data_cfg = _group.build_data_config(name, _dataset_config, seed=seed)
            if args.num_workers is not None:
                data_cfg = replace(data_cfg, num_workers=int(args.num_workers), persistent_workers=False)
            return _group.data_module(data_cfg)

        runner.run_sweep(
            dataset_names=[dataset_name],
            seeds=runner.resolve_seeds(args, trainer_config),
            data_factory=factory,
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_config=trainer_config,
            config_dir=config_dir,
            project_root=PROJECT_ROOT,
            args=args,
            on_run_complete=record_run if combined_path is not None else None,
            variant=VARIANT,
        )

    if combined_path is not None:
        runner.LOGGER.info(
            "Saved combined validation/test metrics summary to %s", combined_path
        )


if __name__ == "__main__":
    main()
