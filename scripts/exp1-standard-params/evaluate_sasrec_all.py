"""Evaluate trained exp1 SASRec checkpoints on the validation/test splits.

Loads the best (or, with --checkpoint last, the last) checkpoints written by
run_sasrec_all.py from the current primary model
``checkpoints/exp1/sasrec/<dataset>/best.pt`` for every requested evaluation seed.
Explicit last-checkpoint requests use current experiment snapshots.
Reruns the final full-catalog evaluation without
retraining: HR, NDCG, Coverage, and ARP at k in {5, 10, 20} on both the
validation and test splits, computed with the exact datamodule, model
builder, and metric code the training runner uses.

Results go to ``artifacts/exp1-standard_params`` as
``sasrec_<dataset>_eval_metrics_summary.csv`` per dataset (plus a combined
``sasrec_all_eval_metrics_summary.csv`` for multi-dataset invocations) in the
same format as the training summaries, alongside a
``sasrec_<dataset>_eval_summary.json`` with the full per-seed payloads. The
``_eval_`` names keep the training-time summaries intact.

Checkpoints missing for a requested dataset/seed abort before any datamodule
setup; pass --skip-missing to evaluate whatever has finished training (e.g.
while some Yambda 500M seeds are still queued).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sasrec_all as sasrec
from scripts import sasrec_config_runner as runner
from scripts.checkpoint_runs import seed_checkpoint_path


LOGGER = runner.LOGGER
EVAL_SUMMARY_NAME_TEMPLATE = "sasrec_{dataset}_eval_metrics_summary.csv"
EVAL_JSON_NAME_TEMPLATE = "sasrec_{dataset}_eval_summary.json"
COMBINED_EVAL_SUMMARY_NAME = "sasrec_all_eval_metrics_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained exp1 SASRec checkpoints on the validation "
            "and test splits and write HR/NDCG/Coverage/ARP summaries to the "
            "artifacts folder."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=sasrec.DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys to evaluate (any mix of beauty, instruments, "
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
            "Objective variant the checkpoints were trained with "
            "(run_sasrec_all.py --variant, e.g. sce); reads and writes the "
            "variant's separate folders."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Defaults to the trainer YAMLs' shared artifacts_dir.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Current primary best.pt (default), or each seed's current experiment last.pt.",
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Defaults to the trainer YAMLs' three seeds; single-seed runs "
            "write seed-suffixed summary files so concurrent tasks never "
            "overwrite each other."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Skip dataset/seed pairs whose checkpoint is absent (with a "
            "warning) instead of aborting before the first evaluation."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override for the dataset YAML's sasrec eval_batch_size.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug override: evaluate at most this many batches per split.",
    )
    return parser.parse_args()


def main() -> None:
    # Unlike the training runner there is no per-run setup_logging, so give
    # the shared LOGGER a console handler.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    model_config = runner.resolve_model_variant(
        runner.load_yaml(config_dir / sasrec.MODEL_CONFIG_NAME), args.variant
    )
    plans = sasrec.build_group_plans(config_dir, args.datasets, variant=args.variant)
    sasrec.validate_train_layout(plans)
    for plan in plans:
        plan.group.validate_inputs(plan.dataset_names, plan.dataset_config)

    artifacts_root = (
        runner.resolve_path(args.artifacts_dir, base=PROJECT_ROOT)
        if args.artifacts_dir is not None
        else shared_artifacts_root(plans)
    )

    # Fail before any (potentially slow) datamodule setup if a checkpoint is
    # absent — unless --skip-missing drops those dataset/seed pairs instead.
    work: list[tuple[sasrec.GroupPlan, str, list[int]]] = []
    missing: list[Path] = []
    for plan in plans:
        seeds = resolve_seeds(args, plan.trainer_config)
        checkpoint_root = runner.configured_path(
            plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
        )
        for dataset_name in plan.dataset_names:
            available: list[int] = []
            for seed in seeds:
                path = checkpoint_path_for(
                    checkpoint_root, dataset_name, seed, which=args.checkpoint
                )
                if path.is_file():
                    available.append(seed)
                else:
                    missing.append(path)
            if available:
                work.append((plan, dataset_name, available))
    if missing and not args.skip_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing {args.checkpoint} checkpoints (run run_sasrec_all.py "
            f"first, or pass --skip-missing to evaluate the rest):\n{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing checkpoint %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset/seed has a checkpoint.")

    selected_total = len(work)
    combined_summaries: list[Mapping[str, Any]] = []

    for plan, dataset_name, seeds in work:
        checkpoint_root = runner.configured_path(
            plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
        )
        ranking_ks = runner.resolve_ranking_ks(plan.trainer_config)
        dataset_summaries: list[Mapping[str, Any]] = []
        for seed in seeds:
            summary = evaluate_experiment(
                dataset_name,
                plan=plan,
                model_config=model_config,
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
                    artifacts_root,
                    EVAL_SUMMARY_NAME_TEMPLATE,
                    dataset_name=dataset_name,
                ),
                seeds,
            )
            runner.write_metrics_summary(
                metrics_summary_path, dataset_summaries, ks=ranking_ks
            )
            runner.write_json(
                seed_scoped(
                    runner.dataset_metrics_summary_path(
                        artifacts_root,
                        EVAL_JSON_NAME_TEMPLATE,
                        dataset_name=dataset_name,
                    ),
                    seeds,
                ),
                {
                    "config_dir": str(config_dir),
                    "variant": args.variant,
                    "checkpoint_root": str(checkpoint_root),
                    "checkpoint": args.checkpoint,
                    "seeds": seeds,
                    "ranking_ks": list(ranking_ks),
                    "metrics_summary": str(metrics_summary_path),
                    "runs": [runner.to_jsonable(s) for s in dataset_summaries],
                },
            )
            if selected_total > 1:
                runner.write_metrics_summary(
                    artifacts_root / COMBINED_EVAL_SUMMARY_NAME,
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
                    artifacts_root,
                    EVAL_SUMMARY_NAME_TEMPLATE,
                    dataset_name=dataset_name,
                ),
                seeds,
            ),
        )


def resolve_seeds(
    args: argparse.Namespace, trainer_config: Mapping[str, Any]
) -> list[int]:
    """Explicit --seeds may be any distinct non-empty subset (evaluation has
    no three-seed requirement); without it the trainer YAML must define the
    full three exp1 seeds."""
    if args.seeds is None:
        return runner.resolve_seeds(args, trainer_config)
    seeds = [int(seed) for seed in args.seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must be non-empty and distinct; got {seeds}")
    return seeds


def shared_artifacts_root(plans: list[sasrec.GroupPlan]) -> Path:
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
    return roots.pop()


def seed_scoped(path: Path, seeds: list[int]) -> Path:
    """Suffix summary files with the seed for single-seed invocations, so
    concurrent per-seed workers for one dataset never overwrite each
    other's results."""
    if len(seeds) != 1:
        return path
    return path.with_name(f"{path.stem}_seed-{seeds[0]}{path.suffix}")


def checkpoint_path_for(
    checkpoint_root: Path, dataset_name: str, seed: int, *, which: str
) -> Path:
    return seed_checkpoint_path(
        PROJECT_ROOT, checkpoint_root, runner.safe_component(dataset_name), seed, which=which
    )


def limited(dataloader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        return dataloader
    return islice(iter(dataloader), max_batches)


def evaluate_experiment(
    dataset_name: str,
    *,
    plan: sasrec.GroupPlan,
    model_config: Mapping[str, Any],
    checkpoint_root: Path,
    ranking_ks: tuple[int, ...],
    seed: int,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    runner.seed_everything(seed)
    checkpoint_path = checkpoint_path_for(
        checkpoint_root, dataset_name, seed, which=args.checkpoint
    )
    device = runner.resolve_device(
        args.device or plan.trainer_config.get("device", "cuda")
    )
    LOGGER.info(
        "[%s seed=%s] evaluating %s on %s",
        dataset_name,
        seed,
        checkpoint_path,
        device,
    )

    data = plan.group.data_module(
        plan.group.build_data_config(dataset_name, plan.dataset_config, seed=seed)
    )
    data.setup()
    if args.eval_batch_size is not None:
        data.cfg = replace(data.cfg, eval_batch_size=int(args.eval_batch_size))

    model = runner.build_model(model_config, data)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict")
    saved_model_config = payload.get("model_config")
    if saved_model_config is not None and saved_model_config != model.config_dict():
        LOGGER.warning(
            "[%s seed=%s] checkpoint model config differs from the one built "
            "from %s; saved=%s built=%s",
            dataset_name,
            seed,
            sasrec.MODEL_CONFIG_NAME,
            json.dumps(runner.to_jsonable(saved_model_config), sort_keys=True),
            json.dumps(runner.to_jsonable(model.config_dict()), sort_keys=True),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    evaluation_kwargs = {
        "device": device,
        "item_popularity": runner.training_item_popularity(data),
        "num_catalog_items": int(data.num_items),
        "ks": ranking_ks,
        "pad_item_id": int(data.cfg.pad_item_id),
        "exclude_history": bool(model.cfg.exclude_history_in_generate),
    }
    validation_metrics = runner.evaluate_ranking_metrics(
        model,
        limited(data.val_dataloader(), args.max_eval_batches),
        **evaluation_kwargs,
    )
    test_metrics = runner.evaluate_ranking_metrics(
        model,
        limited(data.test_dataloader(), args.max_eval_batches),
        **evaluation_kwargs,
    )

    trainer_state = payload.get("trainer_state", {}) or {}
    summary = {
        "seed": seed,
        "dataset": {
            "name": dataset_name,
            **runner.data_config_paths(data.cfg),
            **data.summary(),
        },
        "model": model.config_dict(),
        "checkpoint_kind": args.checkpoint,
        "best_metric": trainer_state.get("best_metric"),
        "best_epoch": trainer_state.get("best_epoch"),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        # write_metrics_summary reads this column for the CSV rows.
        "best_checkpoint": str(checkpoint_path),
    }
    LOGGER.info(
        "[%s seed=%s] validation=%s test=%s",
        dataset_name,
        seed,
        json.dumps(validation_metrics, sort_keys=True),
        json.dumps(test_metrics, sort_keys=True),
    )
    return summary


if __name__ == "__main__":
    main()
