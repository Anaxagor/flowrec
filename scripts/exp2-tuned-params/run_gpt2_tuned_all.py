"""Train and evaluate the Optuna-tuned GPT-2 models for the final exp2 runs.

For each exp2 dataset (Amazon Beauty 2014, Instruments 2018, Beauty 2023,
Yambda 50M listens, Yambda 500M listens) this script reads the tuned
configuration from ``configs/exp2-tuned-params/gpt2_model_<dataset>.yaml``
(written by ``tuning_gpt2.py``), trains GPT-2 with the SFT loss for three
seeds, evaluates the best checkpoint of every run on the validation and test
splits with constrained 20-beam generation -- HR, NDCG, Coverage, and ARP at
5/10/20, the exp1 metric set computed with the exp1 runner's own evaluation
code -- and writes:

- ``artifacts/exp2-tuned-params/gpt2_<dataset>_metrics_summary.csv``
  one row per seed and split
- ``artifacts/exp2-tuned-params/gpt2_<dataset>_metrics_aggregate.csv``
  mean / std over the seeds, one row per split
- ``artifacts/exp2-tuned-params/gpt2_<dataset>_summary.json``
  the full per-seed payloads (data summary, model config, both metric sets)
- ``artifacts/exp2-tuned-params/gpt2_all_metrics_summary.csv`` and
  ``gpt2_all_metrics_aggregate.csv`` (multi-dataset invocations only, so
  concurrent single-dataset cluster jobs never overwrite them)
- ``checkpoints/exp2-tuned_params/gpt2/<dataset>/seed_<seed>/`` best and
  last checkpoints, next to the tuning outputs of that dataset
- ``experiments/exp2-tuned-params/gpt2/<run>/`` resolved configs, logs,
  ``final_summary.json`` per run

Regime. Training happens through the exp1 runner (``run_gpt2_sft_all.py``:
same datamodule construction, model builder, trainer, and evaluation loop),
under the regime the hyperparameters were selected in, as the SASRec exp2
runner does: the tuned YAML's ``tuning_regime`` block supplies the monitor
(``val/sft/generation_ndcg@10``), epoch cap, early-stopping patience,
min_delta, gradient clipping / accumulation, and AMP dtype; its dataset
profile supplies the architecture, the tuned lr and weight_decay (the tuned
value overrides ``trainer_gpt2.yaml``'s shared one), and the validation
subsample used for early stopping (Yambda 10 %, Beauty 2023 25 %). The
remaining trainer keys come from ``configs/exp1-standard_params/
trainer_gpt2.yaml``; batching from the ``gpt2:`` blocks of the exp2 dataset
YAMLs; semantic IDs from ``datasets/semantic_ids/exp2-tuned-params`` -- all
exactly as during tuning. The final validation/test evaluation always scores
the FULL splits.

Modes::

    # train three seeds per dataset and evaluate (default)
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py                 # all five
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py --datasets beauty
    # one seed per cluster task; seed-suffixed outputs never collide
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py --datasets yambda_50m_listens --seeds 43
    # fold the per-seed outputs into the plain per-dataset tables (no GPU)
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py --datasets yambda_50m_listens --merge-seed-summaries
    # re-score existing seed checkpoints without retraining (gpt2_<dataset>_eval_* files)
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py --evaluate-only
    # score the search incumbent checkpoints/exp2-tuned_params/gpt2/<dataset>/best_model.pt
    # (gpt2_<dataset>_incumbent_eval_* files)
    python scripts/exp2-tuned-params/run_gpt2_tuned_all.py --evaluate-only --checkpoint incumbent
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sasrec_config_runner as runner


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_EXP1_SCRIPTS = _HERE.parent / "exp1-standard-params"
# Registered under its plain module name so evaluate_gpt2_sft_all's
# ``import run_gpt2_sft_all`` resolves to this very module object.
run_all = _load_module("run_gpt2_sft_all", _EXP1_SCRIPTS / "run_gpt2_sft_all.py")
evaluate_all = _load_module(
    "evaluate_gpt2_sft_all", _EXP1_SCRIPTS / "evaluate_gpt2_sft_all.py"
)

LOGGER = runner.LOGGER

CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
EXP1_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
DEFAULT_SEMANTIC_IDS_DIR = (
    PROJECT_ROOT / "datasets" / "semantic_ids" / "exp2-tuned-params"
)
CHECKPOINTS_DIR = "checkpoints/exp2-tuned_params/gpt2"
EXPERIMENTS_DIR = "experiments/exp2-tuned-params/gpt2"
ARTIFACTS_DIR = "artifacts/exp2-tuned-params"

DATASETS = (
    "beauty",
    "instruments",
    "beauty_2023",
    "yambda_50m_listens",
    "yambda_500m_listens",
)
SEEDS = (42, 43, 44)
# The datamodule seed the tuning used; the search incumbent is scored on it.
TUNING_SEED = 42
INCUMBENT_CHECKPOINT_NAME = "best_model.pt"

# Keys of a tuned YAML's tuning_regime block that map onto TrainerConfig /
# optimizer settings of the final runs.
REGIME_TRAINER_KEYS = {
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "max_grad_norm": "max_grad_norm",
    "betas": "betas",
    "eps": "eps",
    "min_delta": "min_delta",
    "amp_dtype": "amp_dtype",
    "early_stopping_patience": "early_stopping_patience",
    "max_epochs": "max_epochs",
    "monitor": "monitor",
}

# Older exports lack these keys. New matched studies record them so final
# training cannot silently change validation cadence or sample selection.
OPTIONAL_REGIME_TRAINER_KEYS = (
    "validate_every_epochs", "max_steps", "use_amp", "val_sample_fraction",
    "val_sample_seed", "val_sample_resample",
)


@dataclass(frozen=True)
class OutputNames:
    """Artifact file-name templates of one mode ({dataset} placeholder)."""

    summary: str
    aggregate: str
    payload: str
    combined_summary: str
    combined_aggregate: str


TRAIN_OUTPUTS = OutputNames(
    summary="gpt2_{dataset}_metrics_summary.csv",
    aggregate="gpt2_{dataset}_metrics_aggregate.csv",
    payload="gpt2_{dataset}_summary.json",
    combined_summary="gpt2_all_metrics_summary.csv",
    combined_aggregate="gpt2_all_metrics_aggregate.csv",
)
# --evaluate-only keeps the training-time tables intact.
EVAL_OUTPUTS = OutputNames(
    summary="gpt2_{dataset}_eval_metrics_summary.csv",
    aggregate="gpt2_{dataset}_eval_metrics_aggregate.csv",
    payload="gpt2_{dataset}_eval_summary.json",
    combined_summary="gpt2_all_eval_metrics_summary.csv",
    combined_aggregate="gpt2_all_eval_metrics_aggregate.csv",
)
INCUMBENT_OUTPUTS = OutputNames(
    summary="gpt2_{dataset}_incumbent_eval_metrics_summary.csv",
    aggregate="gpt2_{dataset}_incumbent_eval_metrics_aggregate.csv",
    payload="gpt2_{dataset}_incumbent_eval_summary.json",
    combined_summary="gpt2_all_incumbent_eval_metrics_summary.csv",
    combined_aggregate="gpt2_all_incumbent_eval_metrics_aggregate.csv",
)


@dataclass(frozen=True)
class Paths:
    config_dir: Path          # exp2 dataset YAMLs
    tuned_config_dir: Path    # gpt2_model_<dataset>.yaml
    semantic_ids_dir: Path
    checkpoint_root: Path
    experiments_root: Path
    artifacts_root: Path


@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    group: Any                     # run_gpt2_sft_all.DatasetGroup
    dataset_config: dict[str, Any]
    model_config: dict[str, Any]   # the tuned YAML (gpt2_model.yaml layout)
    trainer_config: dict[str, Any]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def find_group(dataset_name: str, config_dir: Path) -> tuple[Any, dict[str, Any]]:
    for group in run_all.GROUPS:
        dataset_config = runner.load_yaml(config_dir / group.data_config_name)
        if dataset_name in dataset_config.get("datasets", {}):
            return group, dataset_config
    raise KeyError(
        f"Dataset {dataset_name!r} is not defined in any dataset YAML under "
        f"{config_dir}."
    )


def tuned_config_path(dataset_name: str, tuned_config_dir: Path) -> Path:
    return tuned_config_dir / f"gpt2_model_{dataset_name}.yaml"


def load_tuned_config(dataset_name: str, tuned_config_dir: Path) -> dict[str, Any]:
    """Load ``gpt2_model_<dataset>.yaml`` and check it carries what the final
    runs need: the dataset profile (model + tuned optimizer) and the regime
    block the tuner recorded."""

    path = tuned_config_path(dataset_name, tuned_config_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Tuned config not found: {path}. Run the exp2 GPT-2 tuning first "
            "(scripts/exp2-tuned-params/tuning_gpt2.py or the "
            "scripts/scheduler/tuning_gpt2_*.submit-worker arrays); the tuner writes this "
            "file whenever a trial improves the study."
        )
    doc = runner.load_yaml(path)
    profile = run_all.model_profile(doc, dataset_name)
    optimizer = dict(profile.get("optimizer") or {})
    for key in ("lr", "weight_decay"):
        if key not in optimizer:
            raise KeyError(f"{path.name}: datasets.{dataset_name}.optimizer.{key} is missing.")
    if not dict(profile.get("model") or {}):
        raise KeyError(f"{path.name}: datasets.{dataset_name}.model is missing.")
    regime = doc.get("tuning_regime")
    if not isinstance(regime, Mapping):
        raise KeyError(
            f"{path.name} has no tuning_regime block; regenerate it with the "
            "current tuning_gpt2.py (the block records the epoch cap, patience, "
            "and monitor the final runs must reproduce)."
        )
    missing = sorted(set(REGIME_TRAINER_KEYS) - set(regime))
    if missing:
        raise KeyError(f"{path.name}: tuning_regime lacks {missing}.")
    run_all.resolve_ranking_ks(doc)
    return doc


def build_trainer_config(
    dataset_name: str,
    model_config: Mapping[str, Any],
    paths: Paths,
) -> dict[str, Any]:
    """exp1 trainer_gpt2.yaml under the regime the dataset was tuned in.

    Only keys the exp1 runner accepts are produced (TrainerConfig fields plus
    its runtime keys), so ``run_gpt2_sft_all.build_trainer_config`` validates
    the result exactly as it validates the exp1 file.
    """

    base = run_all.apply_trainer_defaults(
        runner.load_yaml(EXP1_CONFIG_DIR / run_all.TRAINER_CONFIG_NAME)
    )
    # Objective variants (a SASRec-side mechanism) are not part of the GPT-2
    # regime; never forward such a block to the exp1 runner's key whitelist.
    base.pop("variants", None)
    regime = model_config["tuning_regime"]
    profile = run_all.model_profile(model_config, dataset_name)

    trainer = dict(base)
    for regime_key, trainer_key in REGIME_TRAINER_KEYS.items():
        value = regime[regime_key]
        if trainer_key == "betas":
            value = [float(value[0]), float(value[1])]
        trainer[trainer_key] = value
    trainer["monitor_mode"] = "max"
    trainer["early_stopping_mode"] = "best"
    trainer["validate_every_epochs"] = 1
    for key in OPTIONAL_REGIME_TRAINER_KEYS:
        if key in regime:
            trainer[key] = regime[key]
    trainer["validate_every_steps"] = None
    # The tuned weight_decay replaces the shared exp1 value (the runner reads
    # weight_decay from the trainer config and lr from the profile).
    trainer["weight_decay"] = float(profile["optimizer"]["weight_decay"])
    trainer["seeds"] = list(SEEDS)
    trainer["checkpoints_dir"] = str(paths.checkpoint_root)
    trainer["experiments_dir"] = str(paths.experiments_root)
    trainer["artifacts_dir"] = str(paths.artifacts_root)
    trainer["metrics_summary_name"] = TRAIN_OUTPUTS.summary
    return trainer


def build_plans(args: argparse.Namespace, paths: Paths) -> list[DatasetPlan]:
    """Resolve every selected dataset and fail before any training if a tuned
    config, an input file, or the dense_causal layout is missing."""

    plans: list[DatasetPlan] = []
    for dataset_name in args.datasets:
        group, dataset_config = find_group(dataset_name, paths.config_dir)
        model_config = load_tuned_config(dataset_name, paths.tuned_config_dir)
        if "tuning_data_config" in model_config:
            recorded = model_config["tuning_data_config"]
            if not isinstance(recorded, Mapping) or not recorded.get("train_dense_match_prefix"):
                raise ValueError("tuning_data_config must record the matched dense training protocol")
            # Ignore generic exp2 preprocessing: the search may use TIGER's
            # different SID offsets, embedding filters and target selection.
            dataset_config = {"datasets": {dataset_name: dict(recorded)}, "gpt2": {}}
            original_root = runner.resolve_path(Path(CHECKPOINTS_DIR), base=PROJECT_ROOT)
            if paths.checkpoint_root.resolve() == original_root.resolve():
                raise ValueError(
                    "Matched training requires a separate --checkpoints-dir; "
                    "use checkpoints/exp2-tuned_params/gpt2_tiger_matched to preserve old runs"
                )
        trainer_config = build_trainer_config(dataset_name, model_config, paths)
        group_plan = run_all.GroupPlan(group, dataset_config, [dataset_name])
        run_all.validate_inputs(
            [group_plan], model_config, semantic_ids_dir=paths.semantic_ids_dir
        )
        plans.append(
            DatasetPlan(dataset_name, group, dataset_config, model_config, trainer_config)
        )
    return plans


# ---------------------------------------------------------------------------
# Evaluation of an existing checkpoint (seed runs and the search incumbent)
# ---------------------------------------------------------------------------

def checkpoint_path_for(
    checkpoint_root: Path, dataset_name: str, seed: int | None, *, which: str
) -> Path:
    dataset_dir = checkpoint_root / runner.safe_component(dataset_name)
    if which == "incumbent":
        return dataset_dir / INCUMBENT_CHECKPOINT_NAME
    if seed is None:
        raise ValueError("Seed checkpoints need a seed.")
    return dataset_dir / f"seed_{seed}" / f"{which}.pt"


def evaluate_checkpoint(
    plan: DatasetPlan,
    *,
    checkpoint_path: Path,
    seed: int | None,
    paths: Paths,
    ranking_ks: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Score one checkpoint on the full validation and test splits with the
    exp1 runner's evaluation loop (constrained beam search, HR/NDCG/Coverage/
    ARP). ``seed=None`` scores the search incumbent."""

    data_seed = TUNING_SEED if seed is None else int(seed)
    runner.seed_everything(data_seed)
    model_config = plan.model_config
    profile = run_all.model_profile(model_config, plan.dataset_name)
    device = runner.resolve_device(
        args.device or plan.trainer_config.get("device", "cuda")
    )

    data_cfg, test_eval_batch_size = run_all.build_data_config(
        plan.group,
        plan.dataset_name,
        plan.dataset_config,
        seed=data_seed,
        semantic_ids_dir=paths.semantic_ids_dir,
        user_hash_mod=run_all.user_ids_count(model_config, profile),
    )
    test_batch_size = int(args.test_batch_size or test_eval_batch_size)
    LOGGER.info(
        "[%s %s] evaluating %s on %s with eval_batch_size=%s",
        plan.dataset_name,
        "incumbent" if seed is None else f"seed={seed}",
        checkpoint_path,
        device,
        test_batch_size,
    )
    data = plan.group.data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError(f"Dataset {plan.dataset_name!r} produced no semantic IDs.")

    model = run_all.build_model(model_config, profile, data)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict")
    saved_model_config = payload.get("model_config")
    if saved_model_config is not None and saved_model_config != model.config_dict():
        LOGGER.warning(
            "[%s] checkpoint model config differs from the tuned YAML's: saved=%s built=%s",
            plan.dataset_name,
            json.dumps(runner.to_jsonable(saved_model_config), sort_keys=True),
            json.dumps(runner.to_jsonable(model.config_dict()), sort_keys=True),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    # Both final evaluations generate with the test beam width, so they use
    # the smaller configured evaluation batch size (as in run_gpt2_sft_all).
    data.cfg = replace(data.cfg, eval_batch_size=test_batch_size)
    lookup = run_all.SemanticIdItemLookup(data.semantic_ids, data.item2id)
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
    validation_metrics = run_all.evaluate_generation_ranking_metrics(
        model, data.val_dataloader(), **evaluation_kwargs
    )
    test_metrics = run_all.evaluate_generation_ranking_metrics(
        model, data.test_dataloader(), **evaluation_kwargs
    )

    trainer_state = payload.get("trainer_state", {}) or {}
    summary = {
        "seed": "incumbent" if seed is None else seed,
        "dataset": {
            "name": plan.dataset_name,
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
    if seed is None:
        best_trial_json = checkpoint_path.parent / "best_trial.json"
        if best_trial_json.is_file():
            summary["tuning_best_trial"] = json.loads(
                best_trial_json.read_text(encoding="utf-8")
            )
    LOGGER.info(
        "[%s %s] validation=%s test=%s",
        plan.dataset_name,
        summary["seed"],
        json.dumps(validation_metrics, sort_keys=True),
        json.dumps(test_metrics, sort_keys=True),
    )
    return summary


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def scoped_path(
    artifacts_root: Path, template: str, dataset_name: str, seeds: list[int] | None
) -> Path:
    path = runner.dataset_metrics_summary_path(
        artifacts_root, template, dataset_name=dataset_name
    )
    return run_all.seed_scoped(path, seeds) if seeds is not None else path


def write_dataset_outputs(
    artifacts_root: Path,
    dataset_name: str,
    summaries: list[Mapping[str, Any]],
    *,
    seeds: list[int] | None,
    ranking_ks: tuple[int, ...],
    metadata: Mapping[str, Any],
    names: OutputNames,
) -> Path:
    """Per-dataset CSV (one row per seed and split), mean/std aggregate, and
    the JSON payload. Single-seed invocations get seed-suffixed names so
    concurrent array tasks never collide; ``seeds=None`` writes plain names
    (merged outputs and the incumbent)."""

    summary_path = scoped_path(artifacts_root, names.summary, dataset_name, seeds)
    runner.write_metrics_summary(summary_path, summaries, ks=ranking_ks)
    evaluate_all.write_metrics_aggregate(
        scoped_path(artifacts_root, names.aggregate, dataset_name, seeds),
        summaries,
        ks=ranking_ks,
    )
    runner.write_json(
        scoped_path(artifacts_root, names.payload, dataset_name, seeds),
        {
            **dict(metadata),
            "seeds": seeds,
            "ranking_ks": list(ranking_ks),
            "metrics_summary": str(summary_path),
            "runs": [runner.to_jsonable(s) for s in summaries],
        },
    )
    return summary_path


def write_combined_outputs(
    artifacts_root: Path,
    summaries: list[Mapping[str, Any]],
    *,
    ranking_ks: tuple[int, ...],
    names: OutputNames,
) -> None:
    runner.write_metrics_summary(
        artifacts_root / names.combined_summary, summaries, ks=ranking_ks
    )
    evaluate_all.write_metrics_aggregate(
        artifacts_root / names.combined_aggregate, summaries, ks=ranking_ks
    )


def merge_seed_summaries(
    dataset_names: list[str],
    *,
    seeds: list[int],
    artifacts_root: Path,
    ranking_ks: tuple[int, ...],
    names: OutputNames,
    skip_missing: bool,
) -> None:
    """Fold the seed-suffixed JSON payloads of per-seed cluster tasks into
    the plain per-dataset CSV/JSON/aggregate files (plus the combined tables
    for several datasets). Needs neither a GPU nor a datamodule."""

    work: list[tuple[str, list[int], list[Mapping[str, Any]]]] = []
    missing: list[Path] = []
    for dataset_name in dataset_names:
        summaries: list[Mapping[str, Any]] = []
        found: list[int] = []
        for seed in seeds:
            path = scoped_path(artifacts_root, names.payload, dataset_name, [seed])
            if not path.is_file():
                missing.append(path)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs = [r for r in payload.get("runs", []) if int(r.get("seed", -1)) == seed]
            if not runs:
                raise ValueError(f"{path} contains no run for seed {seed}.")
            summaries.extend(runs)
            found.append(seed)
        if summaries:
            work.append((dataset_name, found, summaries))
    if missing and not skip_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing per-seed summaries (run the per-seed tasks first, or pass "
            f"--skip-missing to merge the rest):\n{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing per-seed summary %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset/seed has a per-seed summary.")

    combined: list[Mapping[str, Any]] = []
    for dataset_name, found, summaries in work:
        path = write_dataset_outputs(
            artifacts_root,
            dataset_name,
            summaries,
            seeds=None,
            ranking_ks=ranking_ks,
            metadata={"merged_from_seed_summaries": True, "merged_seeds": found},
            names=names,
        )
        combined.extend(summaries)
        LOGGER.info("[%s] merged seeds %s into %s", dataset_name, found, path)
    if len(work) > 1:
        write_combined_outputs(artifacts_root, combined, ranking_ks=ranking_ks, names=names)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def train_and_evaluate(
    plans: list[DatasetPlan],
    *,
    seeds: list[int],
    paths: Paths,
    args: argparse.Namespace,
) -> None:
    combined: list[Mapping[str, Any]] = []
    for plan in plans:
        ranking_ks = run_all.resolve_ranking_ks(plan.model_config)
        dataset_summaries: list[Mapping[str, Any]] = []
        for seed in seeds:
            summary = run_all.run_experiment(
                plan.dataset_name,
                group=plan.group,
                dataset_config=plan.dataset_config,
                model_config=plan.model_config,
                trainer_config=plan.trainer_config,
                config_dir=paths.config_dir,
                semantic_ids_dir=paths.semantic_ids_dir,
                checkpoint_root=paths.checkpoint_root,
                ranking_ks=ranking_ks,
                seed=seed,
                args=args,
            )
            dataset_summaries.append(summary)
            combined.append(summary)
            # Preserve completed results if a later dataset/seed is interrupted.
            summary_path = write_dataset_outputs(
                paths.artifacts_root,
                plan.dataset_name,
                dataset_summaries,
                seeds=seeds,
                ranking_ks=ranking_ks,
                metadata={
                    "mode": "train",
                    "tuned_config": str(
                        tuned_config_path(plan.dataset_name, paths.tuned_config_dir)
                    ),
                    "config_dir": str(paths.config_dir),
                    "semantic_ids_dir": str(paths.semantic_ids_dir),
                    "checkpoint_root": str(paths.checkpoint_root),
                },
                names=TRAIN_OUTPUTS,
            )
            runner.write_training_summary(
                run_all.seed_scoped(
                    paths.checkpoint_root
                    / runner.safe_component(plan.dataset_name)
                    / "training_summary.json",
                    seeds,
                ),
                config_dir=paths.tuned_config_dir,
                dataset_names=[plan.dataset_name],
                seeds=seeds,
                ranking_ks=ranking_ks,
                metrics_summary_path=summary_path,
                summaries=dataset_summaries,
            )
            if len(plans) > 1:
                write_combined_outputs(
                    paths.artifacts_root, combined, ranking_ks=ranking_ks, names=TRAIN_OUTPUTS
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        LOGGER.info(
            "[%s] saved validation/test metrics summary to %s",
            plan.dataset_name,
            scoped_path(paths.artifacts_root, TRAIN_OUTPUTS.summary, plan.dataset_name, seeds),
        )


def evaluate_only(
    plans: list[DatasetPlan],
    *,
    seeds: list[int],
    paths: Paths,
    args: argparse.Namespace,
) -> None:
    incumbent = args.checkpoint == "incumbent"
    names = INCUMBENT_OUTPUTS if incumbent else EVAL_OUTPUTS

    # Fail before any (slow) datamodule setup if a checkpoint is absent,
    # unless --skip-missing drops those pairs instead.
    work: list[tuple[DatasetPlan, list[int | None]]] = []
    missing: list[Path] = []
    for plan in plans:
        candidates: list[int | None] = [None] if incumbent else list(seeds)
        available = [
            s
            for s in candidates
            if checkpoint_path_for(
                paths.checkpoint_root, plan.dataset_name, s, which=args.checkpoint
            ).is_file()
        ]
        missing.extend(
            checkpoint_path_for(paths.checkpoint_root, plan.dataset_name, s, which=args.checkpoint)
            for s in candidates
            if s not in available
        )
        if available:
            work.append((plan, available))
    if missing and not args.skip_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing {args.checkpoint} checkpoints (train the seeds first, or pass "
            f"--skip-missing to evaluate the rest):\n{formatted}"
        )
    for path in missing:
        LOGGER.warning("Skipping missing checkpoint %s", path)
    if not work:
        raise FileNotFoundError("No requested dataset/seed has a checkpoint.")

    combined: list[Mapping[str, Any]] = []
    for plan, available in work:
        ranking_ks = run_all.resolve_ranking_ks(plan.model_config)
        dataset_summaries: list[Mapping[str, Any]] = []
        for seed in available:
            summary = evaluate_checkpoint(
                plan,
                checkpoint_path=checkpoint_path_for(
                    paths.checkpoint_root, plan.dataset_name, seed, which=args.checkpoint
                ),
                seed=seed,
                paths=paths,
                ranking_ks=ranking_ks,
                args=args,
            )
            dataset_summaries.append(summary)
            combined.append(summary)
            write_dataset_outputs(
                paths.artifacts_root,
                plan.dataset_name,
                dataset_summaries,
                seeds=None if incumbent else [s for s in available if s is not None],
                ranking_ks=ranking_ks,
                metadata={
                    "mode": "evaluate",
                    "checkpoint": args.checkpoint,
                    "tuned_config": str(
                        tuned_config_path(plan.dataset_name, paths.tuned_config_dir)
                    ),
                    "checkpoint_root": str(paths.checkpoint_root),
                    "semantic_ids_dir": str(paths.semantic_ids_dir),
                },
                names=names,
            )
            if len(work) > 1:
                write_combined_outputs(
                    paths.artifacts_root, combined, ranking_ks=ranking_ks, names=names
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Optuna-tuned GPT-2 (SFT) models on the exp2 datasets "
            "for three seeds each and report validation/test HR, NDCG, "
            "Coverage, and ARP; or re-score existing checkpoints."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Subset of datasets (default: all five).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Defaults to 42 43 44; may be a subset, e.g. one seed per cluster "
            "task (single-seed runs write seed-suffixed files; fold them with "
            "--merge-seed-summaries)."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument(
        "--semantic-ids-dir",
        type=Path,
        default=DEFAULT_SEMANTIC_IDS_DIR,
        help="Folder with <dataset>/semantic_ids.json (default: the exp2 tokenizer output).",
    )
    modes = parser.add_argument_group("modes")
    modes.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip training; score existing checkpoints on the full validation "
            "and test splits (gpt2_<dataset>_eval_* files)."
        ),
    )
    modes.add_argument(
        "--checkpoint",
        choices=("best", "last", "incumbent"),
        default="best",
        help=(
            "Which checkpoint --evaluate-only scores: the seed runs' best.pt "
            "(default) or last.pt, or 'incumbent' for the search's "
            "best_model.pt per dataset (gpt2_<dataset>_incumbent_eval_* files)."
        ),
    )
    modes.add_argument(
        "--skip-missing",
        action="store_true",
        help="With --evaluate-only / --merge-seed-summaries: skip absent seeds instead of aborting.",
    )
    modes.add_argument(
        "--merge-seed-summaries",
        action="store_true",
        help=(
            "Do not train or evaluate: fold the seed-suffixed per-seed payloads "
            "(training ones, or the eval ones with --evaluate-only) into the "
            "plain per-dataset tables and the combined tables."
        ),
    )
    debug = parser.add_argument_group("overrides (debug / smoke tests)")
    debug.add_argument("--train-batch-size", type=int, default=None)
    debug.add_argument("--val-batch-size", type=int, default=None)
    debug.add_argument("--test-batch-size", type=int, default=None)
    debug.add_argument("--max-epochs", type=int, default=None)
    debug.add_argument("--max-steps", type=int, default=None)
    debug.add_argument("--max-eval-batches", type=int, default=None)
    debug.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="Dataset YAML dir.")
    debug.add_argument("--tuned-config-dir", type=Path, default=CONFIG_DIR)
    debug.add_argument("--checkpoints-dir", type=Path, default=Path(CHECKPOINTS_DIR))
    debug.add_argument("--experiments-dir", type=Path, default=Path(EXPERIMENTS_DIR))
    debug.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    args = parser.parse_args()
    if args.checkpoint == "incumbent" and not args.evaluate_only:
        parser.error("--checkpoint incumbent requires --evaluate-only")
    if args.merge_seed_summaries and args.checkpoint == "incumbent":
        parser.error("--merge-seed-summaries does not apply to the incumbent (no seeds)")
    return args


def resolve_paths(args: argparse.Namespace) -> Paths:
    def resolve(value: Path) -> Path:
        return runner.resolve_path(value, base=PROJECT_ROOT)

    return Paths(
        config_dir=resolve(args.config_dir),
        tuned_config_dir=resolve(args.tuned_config_dir),
        semantic_ids_dir=resolve(args.semantic_ids_dir),
        checkpoint_root=resolve(args.checkpoints_dir),
        experiments_root=resolve(args.experiments_dir),
        artifacts_root=resolve(args.artifacts_dir),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    paths = resolve_paths(args)
    seeds = [int(s) for s in (args.seeds or SEEDS)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must be non-empty and distinct; got {seeds}")

    if args.merge_seed_summaries:
        merge_seed_summaries(
            list(args.datasets),
            seeds=seeds,
            artifacts_root=paths.artifacts_root,
            ranking_ks=runner.DEFAULT_RANKING_KS,
            names=EVAL_OUTPUTS if args.evaluate_only else TRAIN_OUTPUTS,
            skip_missing=args.skip_missing,
        )
        return

    plans = build_plans(args, paths)
    for plan in plans:
        LOGGER.info(
            "[%s] tuned config %s | model %s | optimizer %s | regime %s",
            plan.dataset_name,
            tuned_config_path(plan.dataset_name, paths.tuned_config_dir),
            json.dumps(run_all.model_profile(plan.model_config, plan.dataset_name)["model"], sort_keys=True),
            json.dumps(run_all.model_profile(plan.model_config, plan.dataset_name)["optimizer"], sort_keys=True),
            json.dumps(runner.to_jsonable(plan.model_config["tuning_regime"]), sort_keys=True),
        )

    if args.evaluate_only:
        evaluate_only(plans, seeds=seeds, paths=paths, args=args)
    else:
        train_and_evaluate(plans, seeds=seeds, paths=paths, args=args)


if __name__ == "__main__":
    main()
