"""Evaluate trained exp1 TIGER SFT checkpoints on the validation/test splits.

Loads the best (or, with --checkpoint last, the last) checkpoints written by
run_tiger_sft_all.py from the current primary model
``checkpoints/exp1/tiger/<dataset>/best.pt`` for every requested evaluation seed.
Explicit last-checkpoint requests use current experiment snapshots.
Reruns the final 20-beam evaluation without retraining:
HR, NDCG, Coverage, and ARP at k in {5, 10, 20} on both the validation and
test splits, computed with the exact datamodule, model builder, and metric
code the training runner uses.

Results go to ``artifacts/exp1-standard_params`` as
``tiger_sft_<dataset>_eval_metrics_summary.csv`` per dataset (plus a combined
``tiger_sft_all_eval_metrics_summary.csv`` for multi-dataset invocations) in
the same format as the training summaries, alongside a
``tiger_sft_<dataset>_eval_summary.json`` with the full per-seed payloads.
The ``_eval_`` names keep the training-time summaries intact.

Checkpoints missing for a requested dataset/seed abort before any datamodule
setup; pass --skip-missing to evaluate whatever has finished training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

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

import run_tiger_sft_all as sft
from scripts import sasrec_config_runner as runner
from scripts.checkpoint_runs import seed_checkpoint_path


LOGGER = runner.LOGGER
EVAL_SUMMARY_NAME_TEMPLATE = "tiger_sft_{dataset}_eval_metrics_summary.csv"
EVAL_JSON_NAME_TEMPLATE = "tiger_sft_{dataset}_eval_summary.json"
COMBINED_EVAL_SUMMARY_NAME = "tiger_sft_all_eval_metrics_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained exp1 TIGER SFT checkpoints on the "
            "validation and test splits and write HR/NDCG/Coverage/ARP "
            "summaries to the artifacts folder."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=sft.DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys to evaluate (any mix of yambda_50m_listens, "
            "yambda_500m_listens, vk_lsvd_ur0.01_ir0.01, vk_lsvd_ur0.01_ip0.01, "
            "beauty_2023, beauty, instruments) or 'all'; "
            "defaults to every configured dataset."
        ),
    )
    parser.add_argument(
        "--semantic-ids-dir",
        type=Path,
        default=sft.DEFAULT_SEMANTIC_IDS_DIR,
        help="Folder with <dataset>/semantic_ids.json from train_tokenizer_all.py.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Defaults to the trainer YAML's checkpoints_dir.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Defaults to the trainer YAML's artifacts_dir.",
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
            "Defaults to the trainer YAML's three seeds; single-seed runs "
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
        "--test-batch-size",
        type=int,
        default=None,
        help=(
            "Override for the dataset YAML's tiger test_eval_batch_size; used "
            "for the 20-beam evaluation of both splits."
        ),
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
    semantic_ids_dir = runner.resolve_path(args.semantic_ids_dir, base=PROJECT_ROOT)
    model_config = runner.load_yaml(config_dir / sft.MODEL_CONFIG_NAME)
    trainer_config = sft.apply_trainer_defaults(
        runner.load_yaml(config_dir / sft.TRAINER_CONFIG_NAME)
    )
    plans = sft.build_group_plans(config_dir, args.datasets)
    seeds = sft.resolve_seeds(args, trainer_config)
    ranking_ks = sft.resolve_ranking_ks(model_config)
    sft.validate_inputs(plans, args, semantic_ids_dir=semantic_ids_dir)

    checkpoint_root = (
        runner.resolve_path(args.checkpoint_root, base=PROJECT_ROOT)
        if args.checkpoint_root is not None
        else runner.configured_path(
            trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT
        )
    )
    artifacts_root = (
        runner.resolve_path(args.artifacts_dir, base=PROJECT_ROOT)
        if args.artifacts_dir is not None
        else runner.configured_path(
            trainer_config, "artifacts_dir", project_root=PROJECT_ROOT
        )
    )

    # Fail before any (potentially slow) datamodule setup if a checkpoint is
    # absent — unless --skip-missing drops those dataset/seed pairs instead,
    # so every seed with a finished training run still gets evaluated.
    work: list[tuple[sft.GroupPlan, str, list[int]]] = []
    missing: list[Path] = []
    for plan in plans:
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
            f"Missing {args.checkpoint} checkpoints (run run_tiger_sft_all.py "
            f"first, or pass --skip-missing to evaluate the rest):\n{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing checkpoint %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset/seed has a checkpoint.")

    selected_total = len(work)
    combined_summaries: list[Mapping[str, Any]] = []

    for plan, dataset_name, dataset_seeds in work:
        dataset_summaries: list[Mapping[str, Any]] = []
        for seed in dataset_seeds:
            summary = evaluate_experiment(
                dataset_name,
                group=plan.group,
                dataset_config=plan.dataset_config,
                model_config=model_config,
                trainer_config=trainer_config,
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
            metrics_summary_path = sft.seed_scoped(
                runner.dataset_metrics_summary_path(
                    artifacts_root,
                    EVAL_SUMMARY_NAME_TEMPLATE,
                    dataset_name=dataset_name,
                ),
                dataset_seeds,
            )
            runner.write_metrics_summary(
                metrics_summary_path, dataset_summaries, ks=ranking_ks
            )
            runner.write_json(
                sft.seed_scoped(
                    runner.dataset_metrics_summary_path(
                        artifacts_root,
                        EVAL_JSON_NAME_TEMPLATE,
                        dataset_name=dataset_name,
                    ),
                    dataset_seeds,
                ),
                {
                    "config_dir": str(config_dir),
                    "checkpoint_root": str(checkpoint_root),
                    "checkpoint": args.checkpoint,
                    "seeds": dataset_seeds,
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
            sft.seed_scoped(
                runner.dataset_metrics_summary_path(
                    artifacts_root,
                    EVAL_SUMMARY_NAME_TEMPLATE,
                    dataset_name=dataset_name,
                ),
                dataset_seeds,
            ),
        )


def checkpoint_path_for(
    checkpoint_root: Path, dataset_name: str, seed: int, *, which: str
) -> Path:
    return seed_checkpoint_path(
        PROJECT_ROOT, checkpoint_root, runner.safe_component(dataset_name), seed, which=which
    )


def evaluate_experiment(
    dataset_name: str,
    *,
    group: sft.DatasetGroup,
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    semantic_ids_dir: Path,
    checkpoint_root: Path,
    ranking_ks: tuple[int, ...],
    seed: int,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    runner.seed_everything(seed)
    checkpoint_path = checkpoint_path_for(
        checkpoint_root, dataset_name, seed, which=args.checkpoint
    )
    device = runner.resolve_device(args.device or trainer_config.get("device", "cuda"))

    data_cfg, test_eval_batch_size = sft.build_data_config(
        group,
        dataset_name,
        dataset_config,
        seed=seed,
        semantic_ids_dir=semantic_ids_dir,
        user_hash_mod=int(model_config.get("user_ids_count", 1)),
    )
    test_batch_size = int(args.test_batch_size or test_eval_batch_size)
    LOGGER.info(
        "[%s seed=%s] evaluating %s on %s with eval_batch_size=%s",
        dataset_name,
        seed,
        checkpoint_path,
        device,
        test_batch_size,
    )
    data = group.data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError(f"Dataset {dataset_name!r} produced no semantic IDs.")

    model = sft.build_model(model_config, data)
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
            sft.MODEL_CONFIG_NAME,
            json.dumps(runner.to_jsonable(saved_model_config), sort_keys=True),
            json.dumps(runner.to_jsonable(model.config_dict()), sort_keys=True),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    # Both final evaluations generate with the test beam width, so they use
    # the smaller configured evaluation batch size (as in run_tiger_sft_all).
    data.cfg = replace(data.cfg, eval_batch_size=test_batch_size)
    lookup = sft.SemanticIdItemLookup(data.semantic_ids, data.item2id)
    evaluation_kwargs = {
        "device": device,
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
    validation_metrics = sft.evaluate_generation_ranking_metrics(
        model, data.val_dataloader(), **evaluation_kwargs
    )
    test_metrics = sft.evaluate_generation_ranking_metrics(
        model, data.test_dataloader(), **evaluation_kwargs
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
