"""Retrain global-time, full-CE SASRec winners for seeds 42, 43 and 44.

Read the YAMLs exported by the global-time tuner, initialize a fresh model
for each seed, select the best checkpoint using validation NDCG@10, then
evaluate validation/test HR, NDCG, Coverage and ARP at 5/10/20. Data loading,
training and evaluation reuse the existing exp2 pipeline. Only the config
and output locations change; global split/cold-item guards match the tuner.

One worker owns a dataset and runs all three seeds sequentially,
so per-dataset summaries have a single writer. Multi-dataset invocations
also produce combined tables. Run after global-time tuning has finished::

    python scripts/global-time/run_sasrec_tuned_all.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sasrec_config_runner as runner


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module(
    "_global_time_sasrec_final_base",
    PROJECT_ROOT / "scripts/exp2-tuned-params/run_sasrec_tuned_all.py",
)
tuning = _load_module(
    "_global_time_sasrec_final_tuning", Path(__file__).with_name("tuning_sasrec.py")
)

DATASETS = tuning.DATASETS
SEEDS = base.SEEDS
CONFIG_DIR = tuning.CONFIG_DIR
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/exp2/global_split/initial/sasrec"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments/global-time/exp2-tuned-params/sasrec"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params"
METRICS_SUMMARY_NAME = "sasrec_{dataset}_metrics_summary.csv"
AGGREGATE_SUMMARY_NAME = "sasrec_{dataset}_metrics_mean_std.csv"


def load_tuned_config(dataset_name: str, config_dir: Path):
    model, trainer = base.load_tuned_config(dataset_name, config_dir=config_dir)
    if runner.build_method(model).cfg.item_loss != "cross_entropy":
        raise ValueError(f"{dataset_name}: this runner requires full cross_entropy tuned configs")
    return model, trainer


def write_mean_std(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    """Per-dataset/split sample statistics; expose the count for partial runs."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for summary in summaries:
        name = summary["dataset"]["name"]
        for split in ("validation", "test"):
            metrics = summary[f"{split}_metrics"]
            for k in runner.DEFAULT_RANKING_KS:
                for metric in ("HR", "NDCG", "Coverage", "ARP"):
                    key = f"{metric}@{k}"
                    groups.setdefault((name, split, key), []).append(float(metrics[key]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "split", "metric", "num_seeds", "mean", "std"]
        )
        writer.writeheader()
        for (name, split, metric), values in groups.items():
            writer.writerow({
                "dataset": name, "split": split, "metric": metric,
                "num_seeds": len(values), "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
            })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    selection.add_argument(
        "--array-task-id", type=int, choices=range(len(DATASETS)),
        help="Dataset index; each task trains all requested seeds (default: 42/43/44).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--tuned-config-dir", type=Path, help="Defaults to --config-dir.")
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override; use separate outputs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override; use separate outputs.")
    args = parser.parse_args(argv)
    if args.array_task_id is not None:
        args.datasets = [DATASETS[args.array_task_id]]
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Datasets and seeds must not contain duplicates.")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be nonnegative.")
    for key in ("max_epochs", "max_steps"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    for key in ("config_dir", "checkpoints_dir", "experiments_dir", "artifacts_dir"):
        setattr(args, key, runner.resolve_path(getattr(args, key), base=PROJECT_ROOT))
    args.tuned_config_dir = runner.resolve_path(
        args.tuned_config_dir or args.config_dir, base=PROJECT_ROOT
    )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rt = tuning.Runtime(config_dir=args.config_dir, num_workers=args.num_workers)
    plans = []
    # Validate every requested config/input before starting expensive training.
    # Load each winner once so all its seeds use the same hyperparameters.
    for name in args.datasets:
        model_config, tuned_trainer = load_tuned_config(name, args.tuned_config_dir)
        group, data_cfg = tuning.resolve_data_config(name, rt)
        dataset_config = runner.load_yaml(args.config_dir / group.data_config_name)
        group.validate_inputs([name], dataset_config)
        trainer_config = base.build_trainer_config(name, group.key, tuned_trainer)
        trainer_config.update(
            checkpoints_dir=str(args.checkpoints_dir),
            experiments_dir=str(args.experiments_dir),
            artifacts_dir=str(args.artifacts_dir),
            metrics_summary_name=METRICS_SUMMARY_NAME,
            seeds=args.seeds, device=args.device,
            progress_bar=not args.no_progress_bar,
        )
        plans.append((name, group, data_cfg, dataset_config, model_config, trainer_config))

    combined: list[Mapping[str, Any]] = []
    for name, group, data_cfg, dataset_config, model_config, trainer_config in plans:
        completed: list[Mapping[str, Any]] = []

        def factory(_name, seed, _group=group, _cfg=data_cfg):
            # Pinned global cutoffs/dense windows are seed-independent; the
            # dataloader shuffle and model initialization use the run's seed.
            data = tuning.build_datamodule(_group, replace(_cfg, seed=seed))
            if not len(data.test_dataset):
                raise ValueError(f"{_name}: no test targets remain after cold-item filtering")
            return data

        def record_run(summary):
            completed.append(summary)
            write_mean_std(
                args.artifacts_dir / AGGREGATE_SUMMARY_NAME.format(dataset=name), completed
            )
            # Per-dataset array tasks never write shared aggregate files.
            if len(plans) > 1:
                combined.append(summary)
                runner.write_metrics_summary(
                    args.artifacts_dir / base.COMBINED_METRICS_SUMMARY_NAME, combined
                )
                write_mean_std(
                    args.artifacts_dir / AGGREGATE_SUMMARY_NAME.format(dataset="all"), combined
                )

        runner.run_sweep(
            dataset_names=[name], seeds=args.seeds, data_factory=factory,
            dataset_config=dataset_config, model_config=model_config,
            trainer_config=trainer_config, config_dir=args.config_dir,
            project_root=PROJECT_ROOT, args=args, on_run_complete=record_run,
        )


if __name__ == "__main__":
    main()
