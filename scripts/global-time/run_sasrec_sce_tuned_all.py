"""Train the global-time SCE winner and evaluate its best checkpoints on test.

Defaults to Yambda 500M and seeds 42/43/44. Read the tuned YAML written by
the global-time SCE search (or its optional reranking), train a fresh model
for each seed, select checkpoints by validation NDCG@10, and score the full
catalog on validation/test. Report HR, NDCG, Coverage and ARP at 5/10/20,
both per seed and as mean/sample standard deviation across the three seeds.

One job runs all seeds sequentially and writes summaries after each seed.
Run after all tuning/reranking workers have finished::

    python scripts/global-time/run_sasrec_sce_tuned_all.py --device cuda
    sbatch scripts/slurm/run_sasrec_sce_global_time_tuned_all.sbatch
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

_SPEC = importlib.util.spec_from_file_location(
    "_global_time_sce_final_base",
    PROJECT_ROOT / "scripts/exp2-tuned-params/run_sasrec_sce_tuned_all.py",
)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)
tuning = base.sce_tuning

DATASETS = ("yambda_500m_listens",)
SEEDS = base.SEEDS
CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp2-tuned-params"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/global-time/exp2-tuned_params/sasrec_sce"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments/global-time/exp2-tuned-params/sasrec_sce"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params/sasrec_sce"
AGGREGATE_SUMMARY_NAME = "sasrec_sce_{dataset}_metrics_mean_std.csv"


def write_mean_std(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    """Sample statistics for completed seeds, retaining partial results."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for summary in summaries:
        name = summary["dataset"]["name"]
        for split in ("validation", "test"):
            for k in runner.DEFAULT_RANKING_KS:
                for metric in ("HR", "NDCG", "Coverage", "ARP"):
                    key = f"{metric}@{k}"
                    groups.setdefault((name, split, key), []).append(
                        float(summary[f"{split}_metrics"][key])
                    )
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
    parser.add_argument("--datasets", nargs="+", choices=base.DATASETS, default=list(DATASETS))
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
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Datasets must not contain duplicates.")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        parser.error("Provide three distinct seeds (default: 42 43 44).")
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
    rt = tuning.Runtime(config_dir=args.config_dir, required_split_strategy="global_time")
    dirs = base.OutputDirs(
        checkpoints_dir=str(args.checkpoints_dir), experiments_dir=str(args.experiments_dir),
        artifacts_dir=str(args.artifacts_dir),
    )
    plans = []
    # Snapshot every winner and validate all inputs before training any seed.
    for name in args.datasets:
        model_config, tuned_trainer = base.load_tuned_config(name, tuned_config_dir=args.tuned_config_dir)
        group, dataset_config = tuning.find_group(name, config_dir=args.config_dir)
        data_cfg = group.build_data_config(name, dataset_config, seed=tuning.SEED)
        tuning.validate_data_config(name, data_cfg, rt)
        group.validate_inputs([name], dataset_config)
        if args.num_workers is not None:
            data_cfg = replace(data_cfg, num_workers=args.num_workers, persistent_workers=False)
        trainer_config = base.build_trainer_config(name, group.key, tuned_trainer, dirs=dirs)
        trainer_config.update(seeds=args.seeds, device=args.device, progress_bar=not args.no_progress_bar)
        plans.append((name, group, data_cfg, dataset_config, model_config, trainer_config))

    combined: list[Mapping[str, Any]] = []
    for name, group, data_cfg, dataset_config, model_config, trainer_config in plans:
        completed: list[Mapping[str, Any]] = []

        def factory(_name, seed, _group=group, _cfg=data_cfg):
            # Cutoffs and catalogs are fixed; initialization and shuffle vary.
            data = _group.data_module(replace(_cfg, seed=seed))
            data.setup()
            if any(not len(getattr(data, f"{split}_dataset")) for split in ("train", "val", "test")):
                raise ValueError(f"{_name}: an empty split remains after global-time cold-item filtering")
            if data.num_items < max(runner.DEFAULT_RANKING_KS):
                raise ValueError(f"{_name}: too few training items for top-{max(runner.DEFAULT_RANKING_KS)} metrics")
            return data

        def record_run(summary):
            completed.append(summary)
            write_mean_std(args.artifacts_dir / AGGREGATE_SUMMARY_NAME.format(dataset=name), completed)
            if len(plans) > 1:
                combined.append(summary)
                runner.write_metrics_summary(args.artifacts_dir / base.COMBINED_METRICS_SUMMARY_NAME, combined)
                write_mean_std(args.artifacts_dir / AGGREGATE_SUMMARY_NAME.format(dataset="all"), combined)

        runner.run_sweep(
            dataset_names=[name], seeds=args.seeds, data_factory=factory,
            dataset_config=dataset_config, model_config=model_config,
            trainer_config=trainer_config, config_dir=args.config_dir,
            project_root=PROJECT_ROOT, args=args, on_run_complete=record_run,
            variant=base.VARIANT,
        )


if __name__ == "__main__":
    main()
