"""Evaluate trained exp1 GPT-2 SFT checkpoints on the validation/test splits.

Loads the best (or, with --checkpoint last, the last) checkpoints written by
run_gpt2_sft_all.py from the current primary model
``checkpoints/exp1/gpt2/<dataset>/best.pt`` for every requested evaluation seed.
Explicit last-checkpoint requests use current experiment snapshots.
Reruns the final 20-beam evaluation without retraining:
HR, NDCG, Coverage, and ARP at k in {5, 10, 20} on both the validation and
test splits, computed with the exact datamodule, model builder, and metric
code the training runner uses (so GPT-2 is scored on the same item universe
and splits as TIGER and SASRec).

Results go to ``artifacts/exp1-standard_params`` as
``gpt2_sft_<dataset>_eval_metrics_summary.csv`` per dataset (plus a combined
``gpt2_sft_all_eval_metrics_summary.csv`` for multi-dataset invocations) in
the same format as the training summaries, alongside a
``gpt2_sft_<dataset>_eval_summary.json`` with the full per-seed payloads and a
``gpt2_sft_<dataset>_eval_metrics_aggregate.csv`` with the mean/std over the
evaluated seeds. The ``_eval_`` names keep the training-time summaries intact.

Single-seed invocations (one worker per dataset/seed pair) write
seed-suffixed files; ``--merge-seed-summaries`` then folds those per-seed
JSON payloads into the per-dataset tables without touching a GPU or a
datamodule.

Checkpoints missing for a requested dataset/seed abort before any datamodule
setup; pass --skip-missing to evaluate whatever has finished training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import replace
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

import run_gpt2_sft_all as sft
from scripts import sasrec_config_runner as runner
from scripts.checkpoint_runs import seed_checkpoint_path


LOGGER = runner.LOGGER
EVAL_SUMMARY_NAME_TEMPLATE = "gpt2_sft_{dataset}_eval_metrics_summary.csv"
EVAL_AGGREGATE_NAME_TEMPLATE = "gpt2_sft_{dataset}_eval_metrics_aggregate.csv"
EVAL_JSON_NAME_TEMPLATE = "gpt2_sft_{dataset}_eval_summary.json"
COMBINED_EVAL_SUMMARY_NAME = "gpt2_sft_all_eval_metrics_summary.csv"
COMBINED_EVAL_AGGREGATE_NAME = "gpt2_sft_all_eval_metrics_aggregate.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained exp1 GPT-2 SFT checkpoints on the "
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
            "Defaults to the trainer YAML's three seeds; may be an explicit "
            "subset, e.g. one seed per process (single-seed runs "
            "write seed-suffixed summary files so concurrent tasks never "
            "overwrite each other)."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Skip dataset/seed pairs whose checkpoint (or, with "
            "--merge-seed-summaries, per-seed summary) is absent with a "
            "warning instead of aborting before the first evaluation."
        ),
    )
    parser.add_argument(
        "--merge-seed-summaries",
        action="store_true",
        help=(
            "Do not evaluate anything: fold the seed-suffixed "
            "<dataset>_eval_summary_seed-<seed>.json payloads written by "
            "per-seed invocations into the per-dataset CSV/JSON/aggregate "
            "files (and the combined tables for several datasets)."
        ),
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help=(
            "Override for the dataset YAML's gpt2 test_eval_batch_size; used "
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

    if args.merge_seed_summaries:
        merge_seed_summaries(
            plans,
            seeds=seeds,
            artifacts_root=artifacts_root,
            ranking_ks=ranking_ks,
            skip_missing=args.skip_missing,
        )
        return

    sft.validate_inputs(plans, model_config, semantic_ids_dir=semantic_ids_dir)

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
            f"Missing {args.checkpoint} checkpoints (run run_gpt2_sft_all.py "
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
            write_dataset_outputs(
                artifacts_root,
                dataset_name,
                dataset_summaries,
                seeds=dataset_seeds,
                ranking_ks=ranking_ks,
                metadata={
                    "config_dir": str(config_dir),
                    "checkpoint_root": str(checkpoint_root),
                    "checkpoint": args.checkpoint,
                },
            )
            if selected_total > 1:
                write_combined_outputs(
                    artifacts_root, combined_summaries, ranking_ks=ranking_ks
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
    profile = sft.model_profile(model_config, dataset_name)
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
        user_hash_mod=sft.user_ids_count(model_config, profile),
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

    model = sft.build_model(model_config, profile, data)
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
    # the smaller configured evaluation batch size (as in run_gpt2_sft_all).
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


# ---------------------------------------------------------------------------
# Artifact writers (per-seed table, per-dataset aggregate, JSON payload)
# ---------------------------------------------------------------------------


def write_dataset_outputs(
    artifacts_root: Path,
    dataset_name: str,
    summaries: list[Mapping[str, Any]],
    *,
    seeds: list[int],
    ranking_ks: tuple[int, ...],
    metadata: Mapping[str, Any],
    scoped: bool = True,
) -> None:
    """Write the per-dataset CSV (one row per seed and split), the mean/std
    aggregate CSV, and the JSON payload; single-seed invocations get
    seed-suffixed names so concurrent array tasks never collide (merged
    outputs always use the plain names: scoped=False)."""

    def output_path(template: str) -> Path:
        path = runner.dataset_metrics_summary_path(
            artifacts_root, template, dataset_name=dataset_name
        )
        return sft.seed_scoped(path, seeds) if scoped else path

    metrics_summary_path = output_path(EVAL_SUMMARY_NAME_TEMPLATE)
    runner.write_metrics_summary(metrics_summary_path, summaries, ks=ranking_ks)
    write_metrics_aggregate(
        output_path(EVAL_AGGREGATE_NAME_TEMPLATE), summaries, ks=ranking_ks
    )
    runner.write_json(
        output_path(EVAL_JSON_NAME_TEMPLATE),
        {
            **dict(metadata),
            "seeds": seeds,
            "ranking_ks": list(ranking_ks),
            "metrics_summary": str(metrics_summary_path),
            "runs": [runner.to_jsonable(s) for s in summaries],
        },
    )


def write_combined_outputs(
    artifacts_root: Path,
    summaries: list[Mapping[str, Any]],
    *,
    ranking_ks: tuple[int, ...],
) -> None:
    runner.write_metrics_summary(
        artifacts_root / COMBINED_EVAL_SUMMARY_NAME, summaries, ks=ranking_ks
    )
    write_metrics_aggregate(
        artifacts_root / COMBINED_EVAL_AGGREGATE_NAME, summaries, ks=ranking_ks
    )


def write_metrics_aggregate(
    path: Path,
    summaries: Iterable[Mapping[str, Any]],
    *,
    ks: tuple[int, ...],
) -> None:
    """Mean and (population) std of every metric over the seeds of each
    dataset and split — one row per (dataset, split)."""
    metric_keys = [
        metric
        for k in ks
        for metric in (f"HR@{k}", f"NDCG@{k}", f"Coverage@{k}", f"ARP@{k}")
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for summary in summaries:
        dataset = summary.get("dataset", {})
        dataset_name = dataset.get("name") if isinstance(dataset, Mapping) else None
        for split, group_name in (
            ("validation", "validation_metrics"),
            ("test", "test_metrics"),
        ):
            metrics = summary.get(group_name, {})
            if isinstance(metrics, Mapping):
                grouped.setdefault((str(dataset_name), split), []).append(
                    {"seed": summary.get("seed"), **metrics}
                )

    rows: list[dict[str, Any]] = []
    for (dataset_name, split), entries in grouped.items():
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "split": split,
            "num_seeds": len(entries),
            "seeds": " ".join(str(entry["seed"]) for entry in entries),
        }
        for key in metric_keys:
            values = [float(entry[key]) for entry in entries if entry.get(key) is not None]
            row[f"{key}_mean"] = statistics.fmean(values) if values else None
            row[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "num_seeds",
        "seeds",
        *(f"{key}_{stat}" for key in metric_keys for stat in ("mean", "std")),
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runner.to_jsonable(row) for row in rows)


# ---------------------------------------------------------------------------
# Merging per-seed array-task outputs
# ---------------------------------------------------------------------------


def merge_seed_summaries(
    plans: list[sft.GroupPlan],
    *,
    seeds: list[int],
    artifacts_root: Path,
    ranking_ks: tuple[int, ...],
    skip_missing: bool,
) -> None:
    """Fold the seed-suffixed JSON payloads of per-seed evaluation tasks into
    the per-dataset CSV/JSON/aggregate files (plus the combined tables when
    several datasets are selected). No GPU or datamodule is needed."""
    work: list[tuple[str, list[int], list[Mapping[str, Any]]]] = []
    missing: list[Path] = []
    for plan in plans:
        for dataset_name in plan.dataset_names:
            summaries: list[Mapping[str, Any]] = []
            found: list[int] = []
            for seed in seeds:
                path = sft.seed_scoped(
                    runner.dataset_metrics_summary_path(
                        artifacts_root, EVAL_JSON_NAME_TEMPLATE, dataset_name=dataset_name
                    ),
                    [seed],
                )
                if not path.is_file():
                    missing.append(path)
                    continue
                with open(path, encoding="utf-8") as file:
                    payload = json.load(file)
                runs = [
                    run for run in payload.get("runs", []) if int(run.get("seed", -1)) == seed
                ]
                if not runs:
                    raise ValueError(f"{path} contains no run for seed {seed}.")
                summaries.extend(runs)
                found.append(seed)
            if summaries:
                work.append((dataset_name, found, summaries))
    if missing and not skip_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing per-seed evaluation summaries (run the per-seed "
            "evaluations first, or pass --skip-missing to merge the rest):\n"
            f"{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing per-seed summary %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset/seed has a per-seed summary.")

    combined: list[Mapping[str, Any]] = []
    for dataset_name, found, summaries in work:
        metadata = {
            "merged_from_seed_summaries": True,
            "checkpoint": summaries[0].get("checkpoint_kind"),
        }
        write_dataset_outputs(
            artifacts_root,
            dataset_name,
            summaries,
            seeds=found,
            ranking_ks=ranking_ks,
            metadata=metadata,
            scoped=False,
        )
        combined.extend(summaries)
        LOGGER.info(
            "[%s] merged seeds %s into %s",
            dataset_name,
            found,
            runner.dataset_metrics_summary_path(
                artifacts_root, EVAL_SUMMARY_NAME_TEMPLATE, dataset_name=dataset_name
            ),
        )
    if len(work) > 1:
        write_combined_outputs(artifacts_root, combined, ranking_ks=ranking_ks)
        LOGGER.info(
            "merged %d datasets into %s",
            len(work),
            artifacts_root / COMBINED_EVAL_SUMMARY_NAME,
        )


if __name__ == "__main__":
    main()
