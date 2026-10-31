"""Evaluate the Optuna-tuned SASRec + scalable cross-entropy (SCE) models.

For each dataset tuned by ``tuning_sasrec_sce.py`` (Amazon Beauty 2023,
Yambda 50M listens, Yambda 500M listens) this script loads the incumbent
best checkpoint ``checkpoints/exp2/loo_split/initial/sasrec_sce/<dataset>/best.pt``, rebuilds the datamodule from the exp2 dataset YAMLs with the
tuning split seed (so validation/test are the splits the model was selected
on), scores the full catalog on both splits, and writes to
``artifacts/exp2-tuned-params/sasrec_sce/``:

- ``sasrec_sce_<dataset>_eval_metrics_summary.csv``
      validation/test HR, NDCG, Coverage, and ARP at 5/10/20 (the shared
      metrics table layout)
- ``sasrec_sce_<dataset>_eval_summary.json``
      the same metrics plus the model architecture, the SCE block the model
      was trained with, and the tuning study's best-trial record
- ``sasrec_sce_all_eval_metrics_summary.csv``
      combined table (multi-dataset invocations only, so concurrent
      single-dataset jobs never overwrite it)

With ``--seeds 42 43 44`` the script scores the current primary for each
evaluation seed and writes
``sasrec_sce_<dataset>_seeds_eval_metrics_summary.csv`` / ``_seeds_eval_summary.json``
(seed-suffixed for single-seed invocations, like the exp1 evaluator) and the
combined ``sasrec_sce_all_seeds_eval_metrics_summary.csv``.
An explicit experiment ``--checkpoint-root`` selects its per-seed run snapshots.

The model is built from the architecture stored inside the checkpoint, so
weights and configuration cannot drift apart; the tuned YAML
``configs/exp2-tuned-params/sasrec_sce_model_<dataset>.yaml`` is only
cross-checked (a mismatch is reported, not fatal). The re-evaluated
validation NDCG@10 is printed next to the study's best value as a sanity
check.

Usage::

    python scripts/exp2-tuned-params/evaluate_sasrec_sce_tuned.py            # all three
    python scripts/exp2-tuned-params/evaluate_sasrec_sce_tuned.py --datasets yambda_50m_listens
    python scripts/exp2-tuned-params/evaluate_sasrec_sce_tuned.py --skip-missing
    python scripts/exp2-tuned-params/evaluate_sasrec_sce_tuned.py --seeds 42 43 44   # final runs

On the cluster one Slurm array task per dataset::

    sbatch scripts/slurm/evaluate_sasrec_sce_tuned_array.sbatch
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import fields, replace
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.models.sasrec import SASRec, SASRecConfig
from scripts import sasrec_config_runner as runner
from scripts.checkpoint_runs import seed_checkpoint_path, incumbent_checkpoint_path


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
# Datamodule construction (exp2 dataset YAMLs, tuning split seed) and the
# output layout are shared with the tuning script.
sce_tuning = _load_module("exp2_tuning_sasrec_sce", _HERE / "tuning_sasrec_sce.py")

LOGGER = runner.LOGGER

DATASETS = sce_tuning.DATASETS_TO_TUNE
CHECKPOINT_NAME = "best.pt"          # tuning incumbent
SEED_CHECKPOINT_NAME = "best.pt"           # per-seed final runs
EVAL_SUMMARY_NAME_TEMPLATE = "sasrec_sce_{dataset}_eval_metrics_summary.csv"
EVAL_JSON_NAME_TEMPLATE = "sasrec_sce_{dataset}_eval_summary.json"
COMBINED_EVAL_SUMMARY_NAME = "sasrec_sce_all_eval_metrics_summary.csv"
SEEDS_EVAL_SUMMARY_NAME_TEMPLATE = "sasrec_sce_{dataset}_seeds_eval_metrics_summary.csv"
SEEDS_EVAL_JSON_NAME_TEMPLATE = "sasrec_sce_{dataset}_seeds_eval_summary.json"
COMBINED_SEEDS_EVAL_SUMMARY_NAME = "sasrec_sce_all_seeds_eval_metrics_summary.csv"
RANKING_KS = runner.DEFAULT_RANKING_KS
TUNING_MONITOR_KEY = "NDCG@10"


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def dataset_dir_for(checkpoint_root: Path, dataset_name: str) -> Path:
    return checkpoint_root / runner.safe_component(dataset_name)


def checkpoint_path_for(
    checkpoint_root: Path,
    dataset_name: str,
    *,
    seed: int | None = None,
    name: str | None = None,
) -> Path:
    """Read the current primary best model; last snapshots remain run-specific."""

    dataset = runner.safe_component(dataset_name)
    if seed is None:
        if name is not None:
            return checkpoint_root / dataset / name
        return incumbent_checkpoint_path(PROJECT_ROOT, checkpoint_root, dataset)
    if name is not None and name not in {"best.pt", "last.pt"}:
        return checkpoint_root / dataset / f"seed_{int(seed)}" / name
    return seed_checkpoint_path(
        PROJECT_ROOT, checkpoint_root, dataset, int(seed),
        which=Path(name or SEED_CHECKPOINT_NAME).stem,
    )


def seed_scoped(path: Path, seeds: list[int]) -> Path:
    """Suffix summary files with the seed for single-seed invocations, so
    concurrent per-seed jobs for one dataset never overwrite each other."""

    if len(seeds) != 1:
        return path
    return path.with_name(f"{path.stem}_seed-{seeds[0]}{path.suffix}")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        LOGGER.warning("Could not read %s: %s", path, error)
        return None


def tuned_yaml_architecture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (SASRecConfig keys, sft block) of a tuned SCE YAML, or empties."""

    if not path.is_file():
        return {}, {}
    raw = runner.load_yaml(path)
    model_fields = {field.name for field in fields(SASRecConfig)}
    architecture = {key: value for key, value in raw.items() if key in model_fields}
    sft = dict(raw.get("sft") or {})
    return architecture, sft


def training_seed(best_trial: Mapping[str, Any] | None) -> int | None:
    """Seed the promoted checkpoint was trained with, derived from best_trial.json."""

    if not best_trial:
        return None
    trial_number = best_trial.get("best_trial_number")
    ranking = best_trial.get("rerank_ranking")
    if ranking:
        for entry in ranking:
            if entry.get("trial") == trial_number and entry.get("seed_results"):
                best = max(entry["seed_results"], key=lambda r: r.get("value", float("-inf")))
                return int(best["seed"])
        return None
    if trial_number is None:
        return None
    return int(sce_tuning.SEED) + int(trial_number)


def build_model_from_checkpoint(payload: Mapping[str, Any], data: Any) -> SASRec:
    saved_config = payload.get("model_config")
    if not isinstance(saved_config, Mapping):
        raise KeyError("checkpoint does not contain model_config")
    if "model_state_dict" not in payload:
        raise KeyError("checkpoint does not contain model_state_dict")
    if int(saved_config.get("item_vocab_size", -1)) != int(data.item_vocab_size):
        raise ValueError(
            "checkpoint item_vocab_size "
            f"{saved_config.get('item_vocab_size')} != datamodule item_vocab_size "
            f"{data.item_vocab_size}; the dataset YAML or split differs from the "
            "one the model was tuned on."
        )
    model = SASRec(SASRecConfig(**dict(saved_config)))
    model.load_state_dict(payload["model_state_dict"])
    return model


def limited(dataloader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        return dataloader
    return islice(iter(dataloader), max_batches)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_dataset(
    dataset_name: str,
    *,
    rt: Any,
    checkpoint_path: Path,
    dataset_dir: Path,
    device: str,
    eval_batch_size: int | None,
    max_eval_batches: int | None,
    seed: int | None = None,
) -> dict[str, Any]:
    runner.seed_everything(sce_tuning.SEED)
    best_trial = load_json_if_exists(dataset_dir / "best_trial.json")
    yaml_path = sce_tuning.tuned_config_path(dataset_name, tuned_config_dir=rt.tuned_config_dir)
    yaml_architecture, yaml_sft = tuned_yaml_architecture(yaml_path)

    LOGGER.info("[%s seed=%s] evaluating %s on %s", dataset_name, seed, checkpoint_path, device)
    # The tuning incumbent was selected on the split-seed-42 partition; the
    # final runs train and are evaluated on their own seed's partition.
    data = sce_tuning.build_datamodule(dataset_name, rt, split_seed=seed)
    if eval_batch_size is not None:
        data.cfg = replace(data.cfg, eval_batch_size=int(eval_batch_size))

    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(payload, data)
    saved_config = dict(payload["model_config"])
    if yaml_architecture:
        mismatched = sorted(
            key for key, value in yaml_architecture.items()
            if key not in ("item_vocab_size", "max_history_len") and saved_config.get(key) != value
        )
        if mismatched:
            LOGGER.warning(
                "[%s] %s differs from the checkpoint architecture on %s (checkpoint "
                "values are used); saved=%s yaml=%s",
                dataset_name, yaml_path.name, mismatched,
                json.dumps({k: saved_config.get(k) for k in mismatched}, sort_keys=True),
                json.dumps({k: yaml_architecture.get(k) for k in mismatched}, sort_keys=True),
            )
    model.to(torch.device(device))
    model.eval()

    evaluation_kwargs = {
        "device": device,
        "item_popularity": runner.training_item_popularity(data),
        "num_catalog_items": int(data.num_items),
        "ks": RANKING_KS,
        "pad_item_id": int(data.cfg.pad_item_id),
        "exclude_history": bool(model.cfg.exclude_history_in_generate),
    }
    validation_metrics = runner.evaluate_ranking_metrics(
        model, limited(data.val_dataloader(), max_eval_batches), **evaluation_kwargs
    )
    test_metrics = runner.evaluate_ranking_metrics(
        model, limited(data.test_dataloader(), max_eval_batches), **evaluation_kwargs
    )

    trainer_state = payload.get("trainer_state", {}) or {}
    tuning_record = {
        key: best_trial.get(key)
        for key in (
            "study_name", "selection", "best_value_val_ndcg10", "best_trial_number",
            "best_epoch", "trials_completed", "trials_planned", "best_params",
            "sce_bucket_base", "updated_on",
        )
    } if best_trial else {}
    sce_config = (
        dict(best_trial.get("scalable_cross_entropy") or {}) if best_trial
        else dict(yaml_sft.get("scalable_cross_entropy") or {})
    )
    summary = {
        "seed": int(seed) if seed is not None else training_seed(best_trial),
        "dataset": {
            "name": dataset_name,
            **runner.data_config_paths(data.cfg),
            **data.summary(),
        },
        "model": model.config_dict(),
        "item_loss": sce_tuning.ITEM_LOSS,
        "scalable_cross_entropy": sce_config,
        "tuned_config": str(yaml_path) if yaml_path.is_file() else None,
        "tuning": tuning_record,
        "checkpoint_kind": (
            f"{checkpoint_path.parent.name}/{checkpoint_path.name}"
            if checkpoint_path.parent.name.startswith("seed_") else checkpoint_path.name
        ),
        "best_metric": trainer_state.get("best_metric"),
        "best_epoch": trainer_state.get("best_epoch"),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        # write_metrics_summary reads this column for the CSV rows.
        "best_checkpoint": str(checkpoint_path),
    }

    LOGGER.info(
        "[%s] validation=%s test=%s",
        dataset_name,
        json.dumps(validation_metrics, sort_keys=True),
        json.dumps(test_metrics, sort_keys=True),
    )
    tuned_value = tuning_record.get("best_value_val_ndcg10") if tuning_record else None
    if (
        seed is None and tuned_value is not None
        and TUNING_MONITOR_KEY in validation_metrics and max_eval_batches is None
    ):
        LOGGER.info(
            "[%s] validation %s re-evaluated=%.6f vs tuning best=%.6f (%s)",
            dataset_name, TUNING_MONITOR_KEY, float(validation_metrics[TUNING_MONITOR_KEY]),
            float(tuned_value), tuning_record.get("selection"),
        )

    del model, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the tuned SASRec + SCE checkpoints on the validation and "
            "test splits and write HR/NDCG/Coverage/ARP summaries to the "
            "artifacts folder."
        )
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=list(DATASETS),
        help="Dataset keys to evaluate (default: all three tuned datasets).",
    )
    parser.add_argument("--config-dir", type=Path, default=sce_tuning.CONFIG_DIR,
                        help="Directory with the exp2 dataset YAMLs.")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints/exp2/loo_split/initial/sasrec_sce",
                        help="Primary model root (best.pt, best_trial.json); explicit tuning roots also supported.")
    parser.add_argument("--artifacts-dir", type=Path, default=sce_tuning.ARTIFACTS_DIR,
                        help="Where the evaluation CSV/JSON summaries are written.")
    parser.add_argument("--tuned-config-dir", type=Path, default=sce_tuning.CONFIG_DIR,
                        help="Where sasrec_sce_model_<dataset>.yaml is cross-checked.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Evaluation seeds for the current primary. An explicit experiment "
            "--checkpoint-root instead selects its <dataset>/seed_<seed>/best.pt."
        ),
    )
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Checkpoint filename override; defaults to primary best.pt (explicit tuning roots: best_model.pt).")
    parser.add_argument("--device", type=str, default=None,
                        help="Runtime override (default: cuda when available, else cpu).")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Override for the dataset YAML's eval_batch_size.")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Debug override: evaluate at most this many batches per split.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override the dataset YAML's DataLoader workers.")
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Skip datasets whose checkpoint is absent (with a warning) instead of aborting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    device = runner.resolve_device(args.device) if args.device else sce_tuning.tuning.DEVICE
    rt = sce_tuning.Runtime(
        config_dir=runner.resolve_path(args.config_dir, base=PROJECT_ROOT),
        checkpoint_root=runner.resolve_path(args.checkpoint_root, base=PROJECT_ROOT),
        artifacts_dir=runner.resolve_path(args.artifacts_dir, base=PROJECT_ROOT),
        tuned_config_dir=runner.resolve_path(args.tuned_config_dir, base=PROJECT_ROOT),
        device=device,
        num_workers=args.num_workers,
    )

    # Fail before any (potentially slow) datamodule setup if a checkpoint is
    # absent, unless --skip-missing drops those datasets instead.
    seeds: list[int] | None = None
    if args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"--seeds must be non-empty and distinct; got {seeds}")

    work: list[tuple[str, int | None, Path]] = []
    missing: list[Path] = []
    for dataset_name in args.datasets:
        for seed in (seeds if seeds is not None else [None]):
            path = checkpoint_path_for(
                rt.checkpoint_root, dataset_name, seed=seed, name=args.checkpoint_name
            )
            if path.is_file():
                work.append((dataset_name, seed, path))
            else:
                missing.append(path)
    if missing and not args.skip_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing tuned SCE checkpoints (run tuning_sasrec_sce.py first, or pass "
            f"--skip-missing to evaluate the rest):\n{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing checkpoint %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset has a tuned SCE checkpoint.")

    rt.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        csv_template, json_template, combined_name = (
            EVAL_SUMMARY_NAME_TEMPLATE, EVAL_JSON_NAME_TEMPLATE, COMBINED_EVAL_SUMMARY_NAME
        )
    else:
        csv_template, json_template, combined_name = (
            SEEDS_EVAL_SUMMARY_NAME_TEMPLATE, SEEDS_EVAL_JSON_NAME_TEMPLATE,
            COMBINED_SEEDS_EVAL_SUMMARY_NAME,
        )
    datasets_in_work = list(dict.fromkeys(name for name, _, _ in work))
    combined: list[Mapping[str, Any]] = []
    for dataset_name in datasets_in_work:
        dataset_summaries: list[Mapping[str, Any]] = []
        dataset_seeds = [seed for name, seed, _ in work if name == dataset_name and seed is not None]
        metrics_summary_path = runner.dataset_metrics_summary_path(
            rt.artifacts_dir, csv_template, dataset_name=dataset_name
        )
        json_path = runner.dataset_metrics_summary_path(
            rt.artifacts_dir, json_template, dataset_name=dataset_name
        )
        if seeds is not None:
            metrics_summary_path = seed_scoped(metrics_summary_path, dataset_seeds)
            json_path = seed_scoped(json_path, dataset_seeds)

        for name, seed, checkpoint_path in work:
            if name != dataset_name:
                continue
            summary = evaluate_dataset(
                dataset_name,
                rt=rt,
                checkpoint_path=checkpoint_path,
                dataset_dir=dataset_dir_for(rt.checkpoint_root, dataset_name),
                device=device,
                eval_batch_size=args.eval_batch_size,
                max_eval_batches=args.max_eval_batches,
                seed=seed,
            )
            dataset_summaries.append(summary)
            combined.append(summary)

            # Preserve completed results if a later seed/dataset is interrupted.
            runner.write_metrics_summary(metrics_summary_path, dataset_summaries, ks=RANKING_KS)
            runner.write_json(
                json_path,
                {
                    "config_dir": str(rt.config_dir),
                    "checkpoint_root": str(rt.checkpoint_root),
                    "seeds": dataset_seeds or None,
                    "ranking_ks": list(RANKING_KS),
                    "max_eval_batches": args.max_eval_batches,
                    "metrics_summary": str(metrics_summary_path),
                    **(
                        {"checkpoint": str(checkpoint_path), "run": runner.to_jsonable(summary)}
                        if seeds is None
                        else {"runs": [runner.to_jsonable(s) for s in dataset_summaries]}
                    ),
                },
            )
            if len(datasets_in_work) > 1:
                runner.write_metrics_summary(
                    rt.artifacts_dir / combined_name, combined, ks=RANKING_KS
                )
        LOGGER.info(
            "[%s] saved validation/test metrics to %s and %s",
            dataset_name, metrics_summary_path, json_path,
        )

    if len(datasets_in_work) > 1:
        LOGGER.info("Saved combined summary to %s", rt.artifacts_dir / combined_name)


if __name__ == "__main__":
    main()
