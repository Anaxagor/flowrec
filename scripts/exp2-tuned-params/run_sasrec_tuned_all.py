"""Train the Optuna-tuned SASRec models for the final exp2 runs.

For each of the four exp2 datasets (Amazon Beauty 2014, Instruments 2018,
Beauty 2023, Yambda 50M listens) this script reads the tuned hyperparameters
from ``configs/exp2-tuned-params/sasrec_model_<dataset>.yaml`` (written by
``tuning_sasrec.py``), trains SASRec for three seeds, evaluates the best
checkpoint of every run on the validation and test splits (full-catalog HR,
NDCG, Coverage, and ARP at 5/10/20), and writes:

- ``artifacts/exp2-tuned-params/sasrec_<dataset>_metrics_summary.csv``
  per-dataset validation/test metric table (one row per seed and split)
- ``artifacts/exp2-tuned-params/sasrec_all_metrics_summary.csv``
  combined table (only for multi-dataset invocations, so concurrent
  single-dataset cluster jobs never overwrite it)
- ``checkpoints/exp2-tuned_params/sasrec/<dataset>/seed_<seed>/``
  best checkpoints, next to the tuning outputs
- ``experiments/exp2-tuned-params/sasrec/<run>/`` resolved configs, logs,
  and ``final_summary.json`` per run

Preprocessing comes from the same exp2 dataset YAMLs used during tuning, and
the trainer mirrors the tuning regime (monitor, epoch cap, patience, and the
tuned optimizer/accumulation settings), so the tuned hyperparameters are
applied under exactly the conditions they were selected in.

Usage::

    python scripts/exp2-tuned-params/run_sasrec_tuned_all.py             # all four
    python scripts/exp2-tuned-params/run_sasrec_tuned_all.py --datasets beauty
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch

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
run_all = _load_module(
    "exp1_run_sasrec_all", _HERE.parent / "exp1-standard-params" / "run_sasrec_all.py"
)
tuning = _load_module("exp2_tuning_sasrec", _HERE / "tuning_sasrec.py")


CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
CHECKPOINTS_DIR = "checkpoints/exp2-tuned_params/sasrec"
EXPERIMENTS_DIR = "experiments/exp2-tuned-params/sasrec"
ARTIFACTS_DIR = "artifacts/exp2-tuned-params"
COMBINED_METRICS_SUMMARY_NAME = "sasrec_all_metrics_summary.csv"

DATASETS = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens")
SEEDS = (42, 43, 44)

AMP_DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

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


def find_group(dataset_name: str) -> tuple[Any, dict[str, Any]]:
    for group in run_all.GROUPS:
        dataset_config = runner.load_yaml(CONFIG_DIR / group.data_config_name)
        if dataset_name in dataset_config.get("datasets", {}):
            return group, dataset_config
    raise KeyError(
        f"Dataset {dataset_name!r} is not defined in any dataset YAML under "
        f"{CONFIG_DIR}."
    )


def load_tuned_config(dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a tuned YAML into (model config, optimizer/trainer overrides)."""

    path = CONFIG_DIR / f"sasrec_model_{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Tuned config not found: {path}. Run the exp2 tuning first "
            "(scripts/exp2-tuned-params/tuning_sasrec.py or the "
            "scripts/slurm/tuning_sasrec_*.sbatch arrays)."
        )
    raw = runner.load_yaml(path)

    model_fields = {field.name for field in fields(SASRecConfig)}
    model_config = {
        key: value
        for key, value in raw.items()
        if key in model_fields or key == "sft"
    }
    trainer_overrides = {key: raw[key] for key in TRAINER_KEYS if key in raw}
    unknown = sorted(set(raw) - set(model_config) - TRAINER_KEYS)
    if unknown:
        raise KeyError(f"Unknown keys in {path.name}: {unknown}")
    for key in ("lr", "weight_decay"):
        if key not in trainer_overrides:
            raise KeyError(f"{path.name} does not define {key}.")
    return model_config, trainer_overrides


def build_trainer_config(
    dataset_name: str,
    group_key: str,
    tuned: Mapping[str, Any],
) -> dict[str, Any]:
    """exp1-style trainer settings under the regime the dataset was tuned in."""

    spec = tuning.TUNING_SPECS[dataset_name]
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
        "log_every_steps": 200,
        "progress_bar": True,
        "save_best": True,
        "save_last": False,
        "use_amp": True,
        "amp_dtype": AMP_DTYPE,
        # Runtime and outputs.
        "device": "cuda",
        "seeds": list(SEEDS),
        "ranking_ks": [5, 10, 20],
        "checkpoints_dir": CHECKPOINTS_DIR,
        "experiments_dir": EXPERIMENTS_DIR,
        "artifacts_dir": ARTIFACTS_DIR,
        # {dataset} keeps concurrent per-dataset cluster jobs from rewriting
        # the same metrics file.
        "metrics_summary_name": "sasrec_{dataset}_metrics_summary.csv",
        # Mirrors exp1: the small-Amazon auxiliary store is cheap and feeds
        # the reward pipeline; the large stores are not duplicated per run.
        "save_interaction_store": group_key == "amazon",
        "comet": {"enabled": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Optuna-tuned SASRec models on the exp2 datasets for "
            "three seeds each and report validation/test ranking, coverage, "
            "and popularity metrics."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Subset of datasets to train (default: all four).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Fail before any multi-hour training if a tuned config, an input file,
    # or the required training layout is missing for any selected dataset.
    plans: list[tuple[str, Any, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for dataset_name in args.datasets:
        group, dataset_config = find_group(dataset_name)
        model_config, tuned_trainer = load_tuned_config(dataset_name)
        trainer_config = build_trainer_config(dataset_name, group.key, tuned_trainer)

        data_cfg = group.build_data_config(dataset_name, dataset_config, seed=0)
        layout = getattr(data_cfg, "train_example_layout", None)
        if layout != run_all.REQUIRED_TRAIN_LAYOUT:
            raise ValueError(
                f"Dataset {dataset_name!r} resolves train_example_layout="
                f"{layout!r}; the exp2 runs require "
                f"{run_all.REQUIRED_TRAIN_LAYOUT!r}."
            )
        group.validate_inputs([dataset_name], dataset_config)
        plans.append(
            (dataset_name, group, dataset_config, model_config, trainer_config)
        )

    # Single-dataset invocations (e.g. one cluster job per dataset) skip the
    # shared combined summary so concurrent jobs never overwrite it.
    combined_path = (
        runner.resolve_path(ARTIFACTS_DIR, base=PROJECT_ROOT)
        / COMBINED_METRICS_SUMMARY_NAME
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
            return _group.data_module(data_cfg)

        runner.run_sweep(
            dataset_names=[dataset_name],
            seeds=runner.resolve_seeds(args, trainer_config),
            data_factory=factory,
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_config=trainer_config,
            config_dir=CONFIG_DIR,
            project_root=PROJECT_ROOT,
            args=args,
            on_run_complete=record_run if combined_path is not None else None,
        )

    if combined_path is not None:
        runner.LOGGER.info(
            "Saved combined validation/test metrics summary to %s", combined_path
        )


if __name__ == "__main__":
    main()
