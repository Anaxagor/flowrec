"""Optuna tuning of SASRec trained with scalable cross-entropy (SCE) for exp2.

Tunes SASRec on Yambda 50M listens, Yambda 500M listens, and Amazon Beauty
2023 with ``sft.item_loss: scalable_cross_entropy`` (Mezentsev et al.,
RecSys'24) by maximizing ``val/sft/ndcg@10`` on the leave-one-out validation
split. Validation and test always score the full catalog exactly; SCE only
replaces the training objective.

The search space is the full-CE space of ``tuning_sasrec.py`` (architecture
and AdamW parameters per dataset) extended with the SCE parameters:

- ``n_buckets_mult`` / ``bucket_size_x_mult``  multipliers of the paper
  setting ``int(2 * sqrt(batch_size * window))`` (160 for Yambda at 128 x 50,
  101 for Beauty 2023 at 128 x 20), resolved per dataset at runtime;
- ``bucket_size_y``                             items kept per bucket;
- ``mix_x``                                     random-mixture bucket vectors.

Trials whose bucket logits ``n_buckets * bucket_size_x * bucket_size_y``
exceed the per-dataset cap are pruned before training, and CUDA OOM prunes a
trial instead of killing the worker. Every study starts from two incumbents
enqueued by worker 0: the exp1 baseline with the exp1 ``--variant sce``
bucket settings, and the full-CE Optuna winner of the same dataset
(``configs/exp2-tuned-params/sasrec_model_<dataset>.yaml``; Yambda 500M uses
the 50M winner as its prior) combined with those SCE settings.

Preprocessing is not tuned: datamodules come from the exp2 dataset YAMLs
through the same config-merging path as ``run_sasrec_all.py``, so the tuned
values stay valid for the final multi-seed runs. Datamodules, model, epoch
reporting and incumbent locking are shared with ``tuning_sasrec.py``.

Outputs, per dataset:

``checkpoints/exp2-tuned_params/sasrec_sce/<dataset>/``
    ``optuna_journal.log``      resumable study (NFS-safe journal; rerunning
                                continues until the planned trial count)
    ``best_model.pt``           best-epoch weights of the incumbent best trial
    ``best_trial.json``         incumbent metadata (params, resolved SCE
                                sizes, value, epoch)
    ``epoch_val_ndcg10.jsonl``  per-epoch monitor values for every trial
    ``all_trials.json``         full study dump after the study ends
``artifacts/exp2-tuned-params/sasrec_sce/``
    ``sasrec_sce_<dataset>_best_trial.json``   best-trial results
    ``sasrec_sce_<dataset>_trials.csv``        one row per trial
    ``sasrec_sce_tuning_summary.json``         all datasets (multi-dataset
                                               invocations only)
``configs/exp2-tuned-params/sasrec_sce_model_<dataset>.yaml``
    tuned hyperparameters incl. the resolved SCE block, in the layout of the
    full-CE tuned configs.

The trial budget lives in the shared study, so several workers started with
the same ``--datasets`` (one per scheduler array task; ``--worker-id`` defaults
to ``$WORKER_ID``) cooperate on one journal. After the search,
``--rerank`` retrains the top configurations with three full seeds each and
rewrites the outputs with the best mean.

Usage::

    python scripts/exp2-tuned-params/tuning_sasrec_sce.py                 # all three
    python scripts/exp2-tuned-params/tuning_sasrec_sce.py --datasets beauty_2023
    # one worker of a parallel study (repeat with different --worker-id):
    python scripts/exp2-tuned-params/tuning_sasrec_sce.py \\
        --datasets yambda_500m_listens --worker-id 3
    # after the search finished: 3-seed re-rank of the top configs
    python scripts/exp2-tuned-params/tuning_sasrec_sce.py \\
        --datasets yambda_50m_listens --rerank

On the cluster one two-dimensional scheduler array (datasets x workers, task =
dataset_index * workers + worker) runs the search, and a per-dataset array
runs the re-rank::

    submit-worker scripts/scheduler/tuning_sasrec_sce_array.submit-worker
    submit-worker scripts/scheduler/tuning_sasrec_sce_rerank_array.submit-worker
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, fields, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.ft_methods.sft.scalable_cross_entropy import ScalableCrossEntropyConfig
from flowrec.models.sasrec import SASRec
from flowrec.training.trainer import Trainer, TrainerConfig
from scripts import sasrec_config_runner as runner


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
# Shared with the full-CE tuning: dataset groups, model builder, search-space
# base class and per-dataset ranges, epoch reporter, promotion lock.
tuning = _load_module("exp2_tuning_sasrec", _HERE / "tuning_sasrec.py")
run_all = tuning.run_all


# ---------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "exp2-tuned_params" / "sasrec_sce"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "exp2-tuned-params" / "sasrec_sce"

DATASETS_TO_TUNE = ("beauty_2023", "yambda_50m_listens", "yambda_500m_listens")

ITEM_LOSS = "scalable_cross_entropy"
MONITOR = tuning.MONITOR
SEED = tuning.SEED
RERANK_SEEDS = tuning.RERANK_SEEDS
REQUIRED_TRAIN_LAYOUT = tuning.REQUIRED_TRAIN_LAYOUT
EPOCH_LOG_FILENAME = tuning.EPOCH_LOG_FILENAME

ARCH_PARAM_NAMES = ("d_model", "nhead", "num_layers", "dropout", "pos_drop", "lr", "weight_decay")
SCE_PARAM_NAMES = ("n_buckets_mult", "bucket_size_x_mult", "bucket_size_y", "mix_x")

# The exp1 ``--variant sce`` bucket settings (paper formula x1, bucket_size_y
# 1024, no mixing). Combined with every incumbent architecture.
SCE_BASELINE_PARAMS: dict[str, Any] = {
    "n_buckets_mult": 1.0,
    "bucket_size_x_mult": 1.0,
    "bucket_size_y": 1024,
    "mix_x": False,
}

# Which full-CE tuned config seeds each study. Yambda 500M has no tuned
# full-CE config of its own; the 50M winner is the closest prior.
CE_TUNED_PRIOR = {
    "beauty_2023": "beauty_2023",
    "yambda_50m_listens": "yambda_50m_listens",
    "yambda_500m_listens": "yambda_50m_listens",
}

TUNED_CONFIG_DATA_YAML = {
    "beauty_2023": "dataset_amazon_2023.yaml",
    "yambda_50m_listens": "dataset_yambda.yaml",
    "yambda_500m_listens": "dataset_yambda.yaml",
}


@dataclass(frozen=True)
class SCETuningSpec(tuning.TuningSpec):
    """Full-CE search space plus the SCE bucketing parameters."""

    n_buckets_mult: tuple[float, ...]
    bucket_size_x_mult: tuple[float, ...]
    bucket_size_y: tuple[int, ...]
    mix_x: tuple[bool, ...]
    # Upper bound on n_buckets * bucket_size_x * bucket_size_y. Larger
    # combinations are pruned before training: they cost more memory than
    # the full-catalog logits SCE is meant to avoid.
    max_bucket_logits: int


def _sce_spec(base: tuning.TuningSpec, **overrides: Any) -> SCETuningSpec:
    values = {field.name: getattr(base, field.name) for field in fields(tuning.TuningSpec)}
    values.update(overrides)
    return SCETuningSpec(
        **values,
        n_buckets_mult=(0.5, 1.0, 2.0, 4.0),
        bucket_size_x_mult=(0.5, 1.0, 2.0, 4.0),
        bucket_size_y=(256, 512, 1024, 2048, 4096),
        mix_x=(False, True),
        max_bucket_logits=2**29,
    )


TUNING_SPECS: dict[str, SCETuningSpec] = {
    # Same architecture/optimizer ranges and budgets as the full-CE search;
    # beauty_2023 and yambda_50m keep the ranges that produced their tuned
    # full-CE configs, so the enqueued incumbents lie inside the space.
    "beauty_2023": _sce_spec(
        tuning.TUNING_SPECS["beauty_2023"],
        notes="large sparse Amazon 2023 with SCE; medium capacity, moderate dropout",
    ),
    "yambda_50m_listens": _sce_spec(
        tuning.TUNING_SPECS["yambda_50m_listens"],
        notes="dense large-scale music log with SCE; wide/deep models, light dropout",
    ),
    # Yambda 500M reuses the 50M space (same catalog window and batching)
    # with a smaller budget: epochs are several times longer, so the study
    # runs fewer trials, stops earlier, and prunes sooner.
    "yambda_500m_listens": _sce_spec(
        tuning.TUNING_SPECS["yambda_50m_listens"],
        n_trials=30,
        n_startup_trials=5,
        patience=4,
        pruner_warmup_epochs=2,
        notes="Yambda 500M with SCE; 50M ranges, reduced trial budget",
    ),
}


@dataclass(frozen=True)
class Runtime:
    """Paths and runtime overrides (defaults reproduce the cluster protocol)."""

    config_dir: Path = CONFIG_DIR
    checkpoint_root: Path = CHECKPOINT_ROOT
    artifacts_dir: Path = ARTIFACTS_DIR
    tuned_config_dir: Path = CONFIG_DIR
    device: str = tuning.DEVICE
    use_amp: bool = tuning.USE_AMP
    amp_dtype: str = tuning.AMP_DTYPE
    progress_bar: bool = tuning.PROGRESS_BAR
    n_trials: int | None = None
    max_epochs: int | None = None
    train_batches_per_epoch: int | None = None
    max_eval_batches: int | None = None
    num_workers: int | None = None


# ---------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------

def suggest_hparams(trial: optuna.Trial, spec: SCETuningSpec) -> dict[str, Any]:
    hparams = tuning.suggest_hparams(trial, spec)
    hparams["n_buckets_mult"] = trial.suggest_categorical(
        "n_buckets_mult", list(spec.n_buckets_mult)
    )
    hparams["bucket_size_x_mult"] = trial.suggest_categorical(
        "bucket_size_x_mult", list(spec.bucket_size_x_mult)
    )
    hparams["bucket_size_y"] = trial.suggest_categorical(
        "bucket_size_y", list(spec.bucket_size_y)
    )
    hparams["mix_x"] = trial.suggest_categorical("mix_x", list(spec.mix_x))
    return hparams


def params_in_space(params: Mapping[str, Any], spec: SCETuningSpec) -> bool:
    """Whether ``params`` (architecture + optimizer + SCE) lies in ``spec``."""

    def close_in(value: float, choices: Iterable[float]) -> bool:
        return any(abs(float(value) - float(c)) <= 1e-9 * max(1.0, abs(float(c))) for c in choices)

    try:
        return (
            int(params["d_model"]) in spec.d_model
            and int(params["nhead"]) in spec.nhead
            and spec.num_layers[0] <= int(params["num_layers"]) <= spec.num_layers[1]
            and spec.dropout[0] <= float(params["dropout"]) <= spec.dropout[1]
            and spec.pos_drop[0] <= float(params["pos_drop"]) <= spec.pos_drop[1]
            and spec.lr[0] <= float(params["lr"]) <= spec.lr[1]
            and spec.weight_decay[0] <= float(params["weight_decay"]) <= spec.weight_decay[1]
            and close_in(params["n_buckets_mult"], spec.n_buckets_mult)
            and close_in(params["bucket_size_x_mult"], spec.bucket_size_x_mult)
            and int(params["bucket_size_y"]) in spec.bucket_size_y
            and bool(params["mix_x"]) in spec.mix_x
        )
    except (KeyError, TypeError, ValueError):
        return False


def incumbent_params(
    dataset_name: str,
    *,
    tuned_config_dir: Path = CONFIG_DIR,
) -> list[tuple[str, dict[str, Any]]]:
    """Configurations enqueued before the search: exp1 baseline and the
    full-CE Optuna winner (or its prior), each with the exp1 SCE settings."""

    incumbents = [
        (
            "exp1 baseline architecture with the exp1 --variant sce buckets",
            {**tuning.EXP1_BASELINE_PARAMS, **SCE_BASELINE_PARAMS},
        )
    ]
    source = CE_TUNED_PRIOR.get(dataset_name, dataset_name)
    path = tuned_config_dir / f"sasrec_model_{source}.yaml"
    if path.is_file():
        config = runner.load_yaml(path)
        arch = {key: config[key] for key in ARCH_PARAM_NAMES if key in config}
        if len(arch) == len(ARCH_PARAM_NAMES):
            incumbents.append(
                (f"full-CE tuned config {path.name} with the exp1 --variant sce buckets",
                 {**arch, **SCE_BASELINE_PARAMS}),
            )
    return incumbents


def validate_specs(*, tuned_config_dir: Path = CONFIG_DIR) -> None:
    """Every search space must contain the exp1 incumbent; the full-CE
    tuned incumbent is only enqueued when it lies in the space."""

    for name, spec in TUNING_SPECS.items():
        baseline = {**tuning.EXP1_BASELINE_PARAMS, **SCE_BASELINE_PARAMS}
        if not params_in_space(baseline, spec):
            raise ValueError(
                f"TUNING_SPECS[{name!r}] excludes the exp1 baseline configuration "
                f"{baseline}; widen the space so the search contains the incumbent."
            )
        if spec.max_bucket_logits <= 0:
            raise ValueError(f"TUNING_SPECS[{name!r}].max_bucket_logits must be positive.")


# ---------------------------------------------------------------------
# SCE resolution
# ---------------------------------------------------------------------

def sce_bucket_base(dm: Any) -> int:
    """Paper setting ``int(2 * sqrt(batch_size * window))`` for this datamodule."""

    window = int(dm.cfg.train_window_len or dm.cfg.max_history_len)
    return runner.sce_bucket_count(int(dm.cfg.batch_size), window)


def resolve_sce_config(
    hparams: Mapping[str, Any],
    *,
    bucket_base: int,
) -> ScalableCrossEntropyConfig:
    return ScalableCrossEntropyConfig(
        n_buckets=max(1, round(bucket_base * float(hparams["n_buckets_mult"]))),
        bucket_size_x=max(1, round(bucket_base * float(hparams["bucket_size_x_mult"]))),
        bucket_size_y=int(hparams["bucket_size_y"]),
        mix_x=bool(hparams["mix_x"]),
    )


def bucket_logits(cfg: ScalableCrossEntropyConfig) -> int:
    return int(cfg.n_buckets) * int(cfg.bucket_size_x) * int(cfg.bucket_size_y)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def find_group(dataset_name: str, *, config_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Locate the run_sasrec_all dataset group that owns ``dataset_name``."""

    for group in run_all.GROUPS:
        path = config_dir / group.data_config_name
        if not path.is_file():
            continue
        dataset_config = runner.load_yaml(path)
        if dataset_name in dataset_config.get("datasets", {}):
            return group, dataset_config
    raise KeyError(
        f"Dataset {dataset_name!r} is not defined in any dataset YAML under {config_dir}."
    )


def build_datamodule(dataset_name: str, rt: Runtime, *, split_seed: int | None = None) -> Any:
    group, dataset_config = find_group(dataset_name, config_dir=rt.config_dir)
    group.validate_inputs([dataset_name], dataset_config)

    # The split seed is fixed for the whole study so every trial sees the
    # identical train/val partition; only trial seeds vary. The final
    # three-seed runs pass their own seed (they train on that partition).
    data_cfg = group.build_data_config(
        dataset_name, dataset_config, seed=SEED if split_seed is None else int(split_seed)
    )
    if rt.num_workers is not None:
        data_cfg = replace(data_cfg, num_workers=int(rt.num_workers), persistent_workers=False)
    layout = getattr(data_cfg, "train_example_layout", None)
    if layout != REQUIRED_TRAIN_LAYOUT:
        raise ValueError(
            f"Dataset {dataset_name!r} resolves train_example_layout={layout!r}; "
            f"exp2 SCE tuning requires {REQUIRED_TRAIN_LAYOUT!r}."
        )

    dm = group.data_module(data_cfg)
    dm.setup()
    print(f"\n[{dataset_name}] data summary:")
    print(json.dumps(dm.summary(), indent=2, sort_keys=True, default=str))
    print(
        f"[{dataset_name}] SCE bucket base int(2*sqrt({dm.cfg.batch_size} x "
        f"{int(dm.cfg.train_window_len or dm.cfg.max_history_len)})) = {sce_bucket_base(dm)}"
    )
    return dm


# ---------------------------------------------------------------------
# Method / trainer
# ---------------------------------------------------------------------

def build_method(dm: Any, sce_cfg: ScalableCrossEntropyConfig) -> SFTMethod:
    return SFTMethod(
        SFTConfig(
            loss_weight=1.0,
            metric_prefix="sft",
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_item_metrics=True,
            # Only the tuning metric during search; the final runs evaluate
            # the full [1, 5, 10, 20] cutoffs.
            item_topk=(10,),
            compute_token_metrics=False,
            evaluate_generation=False,
            pad_item_id=int(dm.cfg.pad_item_id),
            item_loss=ITEM_LOSS,
            scalable_cross_entropy=sce_cfg,
        )
    )


def build_trainer(
    *,
    model: SASRec,
    method: SFTMethod,
    hparams: Mapping[str, Any],
    spec: SCETuningSpec,
    rt: Runtime,
    checkpoint_dir: Path,
    logger: Any,
) -> Trainer:
    cfg = TrainerConfig(
        max_epochs=int(rt.max_epochs or spec.max_epochs),
        gradient_accumulation_steps=spec.grad_accum,
        max_grad_norm=tuning.MAX_GRAD_NORM,
        use_amp=rt.use_amp,
        amp_dtype=rt.amp_dtype,
        log_every_steps=tuning.LOG_EVERY_STEPS,
        progress_bar=rt.progress_bar,
        validate_every_epochs=1,
        checkpoint_dir=checkpoint_dir,
        # The trial result is the best epoch, not the last epoch.
        save_best=True,
        save_last=False,
        monitor=MONITOR,
        monitor_mode="max",
        early_stopping_patience=spec.patience,
        min_delta=0.0,
        early_stopping_mode="best",
        prefix_method_metrics_with_stage=True,
        # Debug caps (None for real tuning).
        max_train_batches=rt.train_batches_per_epoch,
        max_eval_batches=rt.max_eval_batches,
    )
    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=cfg,
        device=rt.device,
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        betas=tuning.BETAS,
        eps=tuning.EPS,
        logger=logger,
    )


def train_once(
    *,
    dm: Any,
    hparams: Mapping[str, Any],
    spec: SCETuningSpec,
    rt: Runtime,
    checkpoint_dir: Path,
    seed: int,
    logger: Any = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Train one SCE-SASRec with ``hparams`` and return its best-epoch result.

    Shared by the search objective and the --rerank stage. The caller owns
    ``checkpoint_dir`` cleanup.
    """

    sce_cfg = resolve_sce_config(hparams, bucket_base=sce_bucket_base(dm))
    model: SASRec | None = None
    trainer: Trainer | None = None
    try:
        runner.seed_everything(seed)
        model = tuning.build_model(dm, hparams)
        method = build_method(dm, sce_cfg)
        trainer = build_trainer(
            model=model,
            method=method,
            hparams=hparams,
            spec=spec,
            rt=rt,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
        print(
            f"{log_prefix} params: {json.dumps(dict(hparams), sort_keys=True)} "
            f"| sce: {json.dumps(sce_cfg.to_dict(), sort_keys=True)} "
            f"| trainable params: {model.num_parameters(trainable_only=True):,}",
            flush=True,
        )

        fit_result = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        value = trainer.state.best_metric
        if value is None:
            value = fit_result.val_metrics.get(MONITOR)
        if value is None or not math.isfinite(float(value)):
            # The trainer's metric averager drops non-finite values, so a
            # trial without a finite monitor value diverged (NaN/inf loss).
            raise FloatingPointError(
                f"{log_prefix} finished without a finite {MONITOR} value."
            )

        return {
            "value": float(value),
            "best_epoch": (
                int(trainer.state.best_epoch)
                if trainer.state.best_epoch is not None
                else None
            ),
            "epochs_completed": int(trainer.state.epoch + 1),
            "checkpoint": (
                Path(fit_result.best_checkpoint_path)
                if fit_result.best_checkpoint_path
                else None
            ),
            "sce": sce_cfg.to_dict(),
        }
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Outputs: tuned config, checkpoint-side metadata, artifacts
# ---------------------------------------------------------------------

def tuned_config_path(dataset_name: str, *, tuned_config_dir: Path) -> Path:
    return tuned_config_dir / f"sasrec_sce_model_{dataset_name}.yaml"


def write_tuned_config(
    *,
    dataset_name: str,
    hparams: Mapping[str, Any],
    sce: Mapping[str, Any],
    bucket_base: int,
    best_value: float,
    best_trial_number: int,
    best_epoch: int | None,
    trials_completed: int,
    trials_planned: int,
    study_name: str,
    spec: SCETuningSpec,
    rt: Runtime,
    selection: str = "best single-seed trial of the search",
) -> Path:
    path = tuned_config_path(dataset_name, tuned_config_dir=rt.tuned_config_dir)
    data_yaml = TUNED_CONFIG_DATA_YAML.get(dataset_name, "dataset_*.yaml")
    yaml_float = tuning.yaml_float
    text = f"""\
# SASRec + scalable cross-entropy (SCE) hyperparameters for {dataset_name}
# (exp2, Optuna-tuned). Produced by scripts/exp2-tuned-params/
# tuning_sasrec_sce.py: TPE sampler + median pruner maximizing {MONITOR}
# on the leave-one-out validation split. Preprocessing parameters come from
# configs/exp2-tuned-params/{data_yaml} and were not tuned. Best checkpoint:
# checkpoints/exp2-tuned_params/sasrec_sce/{dataset_name}/best_model.pt
#
# study: {study_name}
# tuned on: {date.today().isoformat()}
# selection: {selection}
# trials completed: {trials_completed} (planned: {trials_planned})
# best trial: {best_trial_number}, best epoch: {best_epoch}
# best {MONITOR}: {best_value:.6f}

# Architecture (SASRecConfig); item_vocab_size and max_history_len are
# derived from the datamodule at runtime.
d_model: {int(hparams["d_model"])}
nhead: {int(hparams["nhead"])}
num_layers: {int(hparams["num_layers"])}
dropout: {yaml_float(hparams["dropout"])}
pos_drop: {yaml_float(hparams["pos_drop"])}
initializer_range: 0.02
layer_norm_eps: 1.0e-8
pad_item_id: 0
exclude_history_in_generate: false

# Optimizer (AdamW)
lr: {yaml_float(hparams["lr"])}
weight_decay: {yaml_float(hparams["weight_decay"])}
betas: [0.9, 0.999]
eps: 1.0e-8

# Fixed trainer settings the tuning ran with (same as the full-CE tuning).
max_grad_norm: {yaml_float(tuning.MAX_GRAD_NORM)}
gradient_accumulation_steps: {spec.grad_accum}

# SFT objective and metrics. The SCE sizes are the tuned multipliers
# (n_buckets x{yaml_float(hparams["n_buckets_mult"])}, bucket_size_x x{yaml_float(hparams["bucket_size_x_mult"])})
# of the paper setting int(2 * sqrt(batch_size * window)) = {bucket_base} for this
# dataset, resolved to integers. Validation/test always use the exact
# full-catalog loss.
sft:
  item_loss: {ITEM_LOSS}
  scalable_cross_entropy:
    n_buckets: {int(sce["n_buckets"])}
    bucket_size_x: {int(sce["bucket_size_x"])}
    bucket_size_y: {int(sce["bucket_size_y"])}
    mix_x: {"true" if bool(sce["mix_x"]) else "false"}
  compute_train_metrics: false
  compute_eval_metrics: true
  compute_item_metrics: true
  compute_token_metrics: false
  evaluate_generation: false
  item_topk: [1, 5, 10, 20]
  pad_item_id: 0
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def best_trial_payload(
    *,
    dataset_name: str,
    study_name: str,
    hparams: Mapping[str, Any],
    sce: Mapping[str, Any],
    bucket_base: int,
    value: float,
    trial_number: int,
    best_epoch: int | None,
    trials_completed: int,
    trials_planned: int,
    checkpoint: Path,
    tuned_config: Path,
    selection: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset": dataset_name,
        "study_name": study_name,
        "item_loss": ITEM_LOSS,
        "metric": MONITOR,
        "selection": selection,
        "best_value_val_ndcg10": float(value),
        "best_trial_number": int(trial_number),
        "best_epoch": best_epoch,
        "best_params": dict(hparams),
        "scalable_cross_entropy": dict(sce),
        "sce_bucket_base": int(bucket_base),
        "trials_completed": int(trials_completed),
        "trials_planned": int(trials_planned),
        "checkpoint": str(checkpoint),
        "tuned_config": str(tuned_config),
        "updated_on": date.today().isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_best_trial_files(dataset_dir: Path, rt: Runtime, payload: Mapping[str, Any]) -> Path:
    """best_trial.json next to the checkpoint plus the artifacts copy."""

    text = json.dumps(payload, indent=2, sort_keys=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "best_trial.json").write_text(text, encoding="utf-8")
    rt.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact = rt.artifacts_dir / f"sasrec_sce_{payload['dataset']}_best_trial.json"
    artifact.write_text(text, encoding="utf-8")
    return artifact


def write_trials_csv(
    *,
    dataset_name: str,
    study: optuna.Study,
    bucket_base: int,
    rt: Runtime,
) -> Path:
    """One row per trial with raw params and the resolved SCE sizes."""

    rt.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = rt.artifacts_dir / f"sasrec_sce_{dataset_name}_trials.csv"
    # bucket_size_y and mix_x are used as sampled; only the multipliers
    # resolve to new integer columns.
    sce_columns = ("n_buckets", "bucket_size_x")
    columns = [
        "dataset", "trial", "state", "val_ndcg10", "best_epoch", "epochs_completed",
        "pruned_reason", *ARCH_PARAM_NAMES, *SCE_PARAM_NAMES, *sce_columns, "bucket_logits",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for t in study.trials:
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "trial": t.number,
                "state": t.state.name,
                "val_ndcg10": t.value,
                "best_epoch": t.user_attrs.get("best_epoch"),
                "epochs_completed": t.user_attrs.get("epochs_completed"),
                "pruned_reason": t.user_attrs.get("pruned_reason"),
            }
            row.update({key: t.params.get(key) for key in (*ARCH_PARAM_NAMES, *SCE_PARAM_NAMES)})
            if all(key in t.params for key in SCE_PARAM_NAMES):
                sce = resolve_sce_config(t.params, bucket_base=bucket_base)
                row.update({key: value for key, value in sce.to_dict().items() if key in sce_columns})
                row["bucket_logits"] = bucket_logits(sce)
            writer.writerow(row)
    return path


def write_all_trials_json(dataset_dir: Path, study: optuna.Study) -> Path:
    path = dataset_dir / "all_trials.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs,
                }
                for t in study.trials
            ],
            f,
            indent=2,
            sort_keys=True,
        )
    return path


# ---------------------------------------------------------------------
# Incumbent promotion
# ---------------------------------------------------------------------

def promote_if_incumbent(
    *,
    dataset_name: str,
    dataset_dir: Path,
    trial: optuna.Trial,
    hparams: Mapping[str, Any],
    result: Mapping[str, Any],
    bucket_base: int,
    spec: SCETuningSpec,
    rt: Runtime,
    study_name: str,
    trials_planned: int,
) -> None:
    """Persist model, params and artifacts as soon as a trial beats the best."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tuning.promotion_lock(dataset_dir):
        best_json = dataset_dir / "best_trial.json"
        previous_value = float("-inf")
        if best_json.exists():
            try:
                previous = json.loads(best_json.read_text(encoding="utf-8"))
                previous_value = float(previous.get("best_value_val_ndcg10", float("-inf")))
            except (json.JSONDecodeError, TypeError, ValueError):
                previous_value = float("-inf")
        value = float(result["value"])
        if value <= previous_value:
            return

        checkpoint_path = result.get("checkpoint")
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            shutil.copy2(checkpoint_path, dataset_dir / "best_model.pt")
        else:
            print(
                f"[{dataset_name} | trial {trial.number}] WARNING: no best-epoch "
                "checkpoint found; best_model.pt not updated.",
                flush=True,
            )

        config_path = write_tuned_config(
            dataset_name=dataset_name,
            hparams=hparams,
            sce=result["sce"],
            bucket_base=bucket_base,
            best_value=value,
            best_trial_number=trial.number,
            best_epoch=result["best_epoch"],
            trials_completed=trial.number + 1,
            trials_planned=trials_planned,
            study_name=study_name,
            spec=spec,
            rt=rt,
        )
        artifact = write_best_trial_files(
            dataset_dir,
            rt,
            best_trial_payload(
                dataset_name=dataset_name,
                study_name=study_name,
                hparams=hparams,
                sce=result["sce"],
                bucket_base=bucket_base,
                value=value,
                trial_number=trial.number,
                best_epoch=result["best_epoch"],
                trials_completed=trial.number + 1,
                trials_planned=trials_planned,
                checkpoint=dataset_dir / "best_model.pt",
                tuned_config=config_path,
                selection="best single-seed trial of the search (incumbent while the search runs)",
            ),
        )
        print(
            f"[{dataset_name} | trial {trial.number}] new best {MONITOR}={value:.6f}; "
            f"updated best_model.pt, {config_path} and {artifact}",
            flush=True,
        )


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

def make_objective(
    *,
    dataset_name: str,
    dm: Any,
    dataset_dir: Path,
    spec: SCETuningSpec,
    rt: Runtime,
    study_name: str,
    trials_planned: int,
):
    bucket_base = sce_bucket_base(dm)

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial, spec)
        trial.set_user_attr("dataset", dataset_name)

        sce_cfg = resolve_sce_config(hparams, bucket_base=bucket_base)
        trial.set_user_attr("sce", sce_cfg.to_dict())
        if bucket_logits(sce_cfg) > spec.max_bucket_logits:
            trial.set_user_attr("pruned_reason", "bucket_logits_cap")
            print(
                f"[{dataset_name} | trial {trial.number}] bucket logits "
                f"{bucket_logits(sce_cfg):,} exceed the cap {spec.max_bucket_logits:,}; pruning.",
                flush=True,
            )
            raise optuna.TrialPruned("bucket logits exceed max_bucket_logits")

        trial_dir = dataset_dir / "trials" / f"trial_{trial.number}"
        try:
            reporter = tuning.EpochValReporter(
                trial=trial,
                dataset_name=dataset_name,
                log_path=dataset_dir / EPOCH_LOG_FILENAME,
            )
            result = train_once(
                dm=dm,
                hparams=hparams,
                spec=spec,
                rt=rt,
                checkpoint_dir=trial_dir,
                seed=SEED + trial.number,
                logger=reporter,
                log_prefix=f"[{dataset_name} | trial {trial.number}]",
            )
            trial.set_user_attr("best_epoch", result["best_epoch"])
            trial.set_user_attr("epochs_completed", result["epochs_completed"])

            promote_if_incumbent(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                trial=trial,
                hparams=hparams,
                result=result,
                bucket_base=bucket_base,
                spec=spec,
                rt=rt,
                study_name=study_name,
                trials_planned=trials_planned,
            )
            print(
                f"[{dataset_name} | trial {trial.number}] done: "
                f"best_epoch={result['best_epoch']} {MONITOR}={result['value']:.6f}",
                flush=True,
            )
            return result["value"]

        except FloatingPointError as exc:
            # The trainer raises on a non-finite loss (skip_non_finite_loss is
            # off): the configuration diverged. Record it as pruned so the
            # study counts it and the worker keeps running.
            trial.set_user_attr("pruned_reason", "non_finite_loss")
            print(
                f"[{dataset_name} | trial {trial.number}] diverged ({exc}); pruning.",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned("non-finite loss") from exc

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                trial.set_user_attr("pruned_reason", "cuda_oom")
                print(f"[{dataset_name} | trial {trial.number}] CUDA OOM; pruning.", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA out of memory") from exc
            raise

        finally:
            # The incumbent checkpoint was already copied to best_model.pt.
            shutil.rmtree(trial_dir, ignore_errors=True)

    return objective


# ---------------------------------------------------------------------
# Per-dataset tuning
# ---------------------------------------------------------------------

FINISHED_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def study_name_for(dataset_name: str) -> str:
    return f"sasrec_sce_{dataset_name}_val_ndcg10_exp2"


def tune_one_dataset(
    dataset_name: str,
    *,
    rt: Runtime,
    spec_name: str | None = None,
    worker_id: int = 0,
) -> dict[str, Any]:
    spec = TUNING_SPECS[spec_name or dataset_name]
    n_trials = int(rt.n_trials or spec.n_trials)
    dataset_dir = rt.checkpoint_root / runner.safe_component(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    study_name = study_name_for(dataset_name)
    journal_path = dataset_dir / "optuna_journal.log"

    print("\n" + "=" * 100)
    print(f"Tuning SASRec + SCE on {dataset_name}")
    print("=" * 100)
    print(f"Config dir:      {rt.config_dir}")
    print(f"Output dir:      {dataset_dir}")
    print(f"Artifacts dir:   {rt.artifacts_dir}")
    print(f"Study:           {study_name} (journal: {journal_path})")
    print(f"Worker:          {worker_id}")
    print(f"Device:          {rt.device} (amp={rt.use_amp}, dtype={rt.amp_dtype})")
    print(f"Budget:          {n_trials} trials, <= {rt.max_epochs or spec.max_epochs} epochs, "
          f"patience {spec.patience}")
    print(f"Space notes:     {spec.notes}")
    print(f"SCE space:       n_buckets x{list(spec.n_buckets_mult)}, "
          f"bucket_size_x x{list(spec.bucket_size_x_mult)}, "
          f"bucket_size_y {list(spec.bucket_size_y)}, mix_x {list(spec.mix_x)}, "
          f"cap {spec.max_bucket_logits:,} bucket logits")

    dm = build_datamodule(dataset_name, rt)
    bucket_base = sce_bucket_base(dm)

    sampler = optuna.samplers.TPESampler(
        # Distinct seeds keep parallel workers from proposing identical
        # startup configurations.
        seed=SEED + 9973 * worker_id,
        n_startup_trials=spec.n_startup_trials,
        multivariate=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=spec.n_startup_trials,
        n_warmup_steps=spec.pruner_warmup_epochs,
        interval_steps=1,
    )
    # Journal storage is safe for concurrent workers on shared (NFS)
    # filesystems, unlike SQLite.
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    if worker_id == 0:
        # Seed the search with the incumbents so tuning can only improve on
        # them in expectation. skip_if_exists keeps resumed and re-submitted
        # runs from enqueueing them twice.
        for label, params in incumbent_params(dataset_name, tuned_config_dir=rt.tuned_config_dir):
            if params_in_space(params, spec):
                study.enqueue_trial(dict(params), skip_if_exists=True)
                print(f"Enqueued incumbent: {label}: {json.dumps(params, sort_keys=True)}")
            else:
                print(f"WARNING: incumbent outside the search space, not enqueued: {label}: "
                      f"{json.dumps(params, sort_keys=True)}", flush=True)

    already_done = len([t for t in study.trials if t.state in FINISHED_STATES])
    if already_done:
        print(f"Resuming study: {already_done}/{n_trials} finished trials.")

    if already_done < n_trials:
        # The budget is enforced against the shared study, not this process:
        # every worker keeps pulling trials until the study holds n_trials
        # finished ones, so any number of workers can cooperate.
        study.optimize(
            make_objective(
                dataset_name=dataset_name,
                dm=dm,
                dataset_dir=dataset_dir,
                spec=spec,
                rt=rt,
                study_name=study_name,
                trials_planned=n_trials,
            ),
            n_trials=None,
            gc_after_trial=True,
            show_progress_bar=False,
            callbacks=[MaxTrialsCallback(n_trials, states=FINISHED_STATES)],
        )

    best = study.best_trial
    trials_completed = len([t for t in study.trials if t.state in FINISHED_STATES])
    best_sce = resolve_sce_config(best.params, bucket_base=bucket_base).to_dict()

    # Rewrite the tuned config and best-trial files with final statistics.
    config_path = write_tuned_config(
        dataset_name=dataset_name,
        hparams=best.params,
        sce=best_sce,
        bucket_base=bucket_base,
        best_value=float(study.best_value),
        best_trial_number=best.number,
        best_epoch=best.user_attrs.get("best_epoch"),
        trials_completed=trials_completed,
        trials_planned=n_trials,
        study_name=study_name,
        spec=spec,
        rt=rt,
    )
    payload = best_trial_payload(
        dataset_name=dataset_name,
        study_name=study_name,
        hparams=best.params,
        sce=best_sce,
        bucket_base=bucket_base,
        value=float(study.best_value),
        trial_number=best.number,
        best_epoch=best.user_attrs.get("best_epoch"),
        trials_completed=trials_completed,
        trials_planned=n_trials,
        checkpoint=dataset_dir / "best_model.pt",
        tuned_config=config_path,
        selection="best single-seed trial of the search",
    )
    artifact = write_best_trial_files(dataset_dir, rt, payload)
    trials_csv = write_trials_csv(dataset_name=dataset_name, study=study, bucket_base=bucket_base, rt=rt)
    write_all_trials_json(dataset_dir, study)

    print(f"\n[{dataset_name}] best {MONITOR}: {study.best_value:.6f} (trial {best.number})")
    print(json.dumps(best.params, indent=2, sort_keys=True))
    print(f"[{dataset_name}] resolved SCE: {json.dumps(best_sce, sort_keys=True)}")
    print(f"[{dataset_name}] tuned config: {config_path}")
    print(f"[{dataset_name}] best model:   {dataset_dir / 'best_model.pt'}")
    print(f"[{dataset_name}] artifacts:    {artifact}, {trials_csv}")

    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**payload, "trials_csv": str(trials_csv)}


# ---------------------------------------------------------------------
# Multi-seed re-ranking of the top search configurations
# ---------------------------------------------------------------------

def rerank_top_configs(
    dataset_name: str,
    *,
    rt: Runtime,
    top_k: int,
    spec_name: str | None = None,
) -> dict[str, Any]:
    """Retrain the top-k distinct search configs on RERANK_SEEDS and pick the
    winner by mean validation NDCG@10, rewriting every best-trial output."""

    spec = TUNING_SPECS[spec_name or dataset_name]
    n_trials = int(rt.n_trials or spec.n_trials)
    dataset_dir = rt.checkpoint_root / runner.safe_component(dataset_name)
    journal_path = dataset_dir / "optuna_journal.log"
    if not journal_path.exists():
        raise FileNotFoundError(
            f"No study journal at {journal_path}; run the search for {dataset_name!r} first."
        )

    study = optuna.load_study(
        study_name=study_name_for(dataset_name),
        storage=JournalStorage(JournalFileBackend(str(journal_path))),
    )
    finished = [t for t in study.trials if t.state in FINISHED_STATES]
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(f"Study for {dataset_name!r} has no completed trials.")
    if len(finished) < n_trials:
        print(
            f"[{dataset_name} | rerank] WARNING: study holds {len(finished)}/{n_trials} "
            "finished trials; re-ranking an unfinished search.",
            flush=True,
        )

    seen: set[str] = set()
    candidates: list[optuna.trial.FrozenTrial] = []
    for t in sorted(complete, key=lambda t: t.value, reverse=True):
        key = json.dumps(t.params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(t)
        if len(candidates) == top_k:
            break

    print(
        f"[{dataset_name} | rerank] re-training top {len(candidates)} configs x "
        f"{len(RERANK_SEEDS)} seeds (search values: {[round(t.value, 6) for t in candidates]})",
        flush=True,
    )
    dm = build_datamodule(dataset_name, rt)
    bucket_base = sce_bucket_base(dm)

    results: list[dict[str, Any]] = []
    rerank_root = dataset_dir / "rerank"
    for t in candidates:
        seed_results: list[dict[str, Any]] = []
        failed = False
        for seed in RERANK_SEEDS:
            checkpoint_dir = rerank_root / f"trial_{t.number}" / f"seed_{seed}"
            try:
                out = train_once(
                    dm=dm,
                    hparams=t.params,
                    spec=spec,
                    rt=rt,
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                    log_prefix=f"[{dataset_name} | rerank trial {t.number} | seed {seed}]",
                )
            except (RuntimeError, FloatingPointError) as exc:
                if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                    raise
                reason = "CUDA OOM" if isinstance(exc, RuntimeError) else f"diverged ({exc})"
                print(
                    f"[{dataset_name} | rerank trial {t.number} | seed {seed}] "
                    f"{reason}; dropping this config.",
                    flush=True,
                )
                failed = True
                break
            seed_results.append(
                {
                    "seed": seed,
                    "value": out["value"],
                    "best_epoch": out["best_epoch"],
                    "checkpoint": str(out["checkpoint"]) if out["checkpoint"] else None,
                    "sce": out["sce"],
                }
            )
            print(
                f"[{dataset_name} | rerank trial {t.number} | seed {seed}] "
                f"{MONITOR}={out['value']:.6f}",
                flush=True,
            )

        if failed or not seed_results:
            continue
        values = [r["value"] for r in seed_results]
        mean = sum(values) / len(values)
        std = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            if len(values) > 1
            else 0.0
        )
        results.append(
            {
                "trial": t.number,
                "params": t.params,
                "sce": seed_results[0]["sce"],
                "search_value": float(t.value),
                "seed_results": seed_results,
                "mean_val_ndcg10": mean,
                "std_val_ndcg10": std,
            }
        )
        print(
            f"[{dataset_name} | rerank trial {t.number}] mean {MONITOR}={mean:.6f} "
            f"+- {std:.6f} (search value {t.value:.6f})",
            flush=True,
        )

    if not results:
        raise RuntimeError(f"Re-ranking produced no results for {dataset_name!r}.")
    results.sort(key=lambda r: r["mean_val_ndcg10"], reverse=True)
    winner = results[0]
    best_seed = max(winner["seed_results"], key=lambda r: r["value"])

    selection = (
        f"mean {MONITOR} over seeds {list(RERANK_SEEDS)} across the "
        f"top-{len(results)} search configs (re-rank)"
    )
    if best_seed["checkpoint"] and Path(best_seed["checkpoint"]).exists():
        shutil.copy2(best_seed["checkpoint"], dataset_dir / "best_model.pt")
    config_path = write_tuned_config(
        dataset_name=dataset_name,
        hparams=winner["params"],
        sce=winner["sce"],
        bucket_base=bucket_base,
        best_value=winner["mean_val_ndcg10"],
        best_trial_number=winner["trial"],
        best_epoch=best_seed["best_epoch"],
        trials_completed=len(finished),
        trials_planned=n_trials,
        study_name=study.study_name,
        spec=spec,
        rt=rt,
        selection=selection,
    )
    payload = best_trial_payload(
        dataset_name=dataset_name,
        study_name=study.study_name,
        hparams=winner["params"],
        sce=winner["sce"],
        bucket_base=bucket_base,
        value=winner["mean_val_ndcg10"],
        trial_number=winner["trial"],
        best_epoch=best_seed["best_epoch"],
        trials_completed=len(finished),
        trials_planned=n_trials,
        checkpoint=dataset_dir / "best_model.pt",
        tuned_config=config_path,
        selection=selection,
        extra={
            "std_val_ndcg10": winner["std_val_ndcg10"],
            "rerank_seeds": list(RERANK_SEEDS),
            "rerank_ranking": results,
        },
    )
    artifact = write_best_trial_files(dataset_dir, rt, payload)
    summary_text = json.dumps(payload, indent=2, sort_keys=True)
    (dataset_dir / "rerank_summary.json").write_text(summary_text, encoding="utf-8")
    (rt.artifacts_dir / f"sasrec_sce_{dataset_name}_rerank_summary.json").write_text(
        summary_text, encoding="utf-8"
    )
    shutil.rmtree(rerank_root, ignore_errors=True)

    print(
        f"\n[{dataset_name} | rerank] winner: trial {winner['trial']} with mean "
        f"{MONITOR}={winner['mean_val_ndcg10']:.6f} +- {winner['std_val_ndcg10']:.6f}",
        flush=True,
    )
    print(f"[{dataset_name} | rerank] tuned config: {config_path}")
    print(f"[{dataset_name} | rerank] artifacts:    {artifact}")

    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune SASRec trained with scalable cross-entropy (SCE) using Optuna "
            "(val/sft/ndcg@10) on beauty_2023, yambda_50m_listens and "
            "yambda_500m_listens."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS_TO_TUNE),
        help=(
            "Dataset keys to tune (default: all three). A key outside the "
            "default set must be defined in a dataset YAML under --config-dir "
            "and paired with --spec."
        ),
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        choices=list(TUNING_SPECS),
        help="Search space/budget to use for every dataset (default: the dataset's own).",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=int(os.environ.get("WORKER_ID", "0") or 0),
        help=(
            "Index of this worker within a multi-worker study (defaults to "
            "$WORKER_ID). Workers started with the same --datasets "
            "share the study journal and global trial budget; the ID only "
            "diversifies the sampler seed. Worker 0 enqueues the incumbents."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Skip searching; retrain the top configurations of the finished "
            "study with three full seeds each and rewrite the outputs with "
            "the best mean. Run as a single job per dataset."
        ),
    )
    parser.add_argument("--rerank-top-k", type=int, default=3,
                        help="How many distinct top search configs to re-rank (default: 3).")
    # Paths (defaults are the cluster protocol locations).
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                        help="Directory with the exp2 dataset YAMLs.")
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT,
                        help="Per-dataset study journals and best checkpoints.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR,
                        help="Best-trial JSON and per-trial CSV per dataset.")
    parser.add_argument("--tuned-config-dir", type=Path, default=CONFIG_DIR,
                        help="Where sasrec_sce_model_<dataset>.yaml is written and the "
                             "full-CE sasrec_model_<dataset>.yaml incumbents are read.")
    # Debug overrides (smoke tests); leave unset for real tuning.
    parser.add_argument("--n-trials", type=int, default=None, help="Override the planned trial count.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override the epoch cap per trial.")
    parser.add_argument("--train-batches-per-epoch", type=int, default=None,
                        help="Debug cap on training batches per epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Debug cap on validation batches.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override the dataset YAML's DataLoader workers.")
    parser.add_argument("--device", type=str, default=None,
                        help="Runtime override (default: cuda when available, else cpu).")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args(argv)


def runtime_from_args(args: argparse.Namespace) -> Runtime:
    device = runner.resolve_device(args.device) if args.device else tuning.DEVICE
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    return Runtime(
        config_dir=runner.resolve_path(args.config_dir, base=PROJECT_ROOT),
        checkpoint_root=runner.resolve_path(args.checkpoint_root, base=PROJECT_ROOT),
        artifacts_dir=runner.resolve_path(args.artifacts_dir, base=PROJECT_ROOT),
        tuned_config_dir=runner.resolve_path(args.tuned_config_dir, base=PROJECT_ROOT),
        device=device,
        use_amp=use_cuda,
        amp_dtype=tuning.AMP_DTYPE,
        progress_bar=not args.no_progress_bar,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        train_batches_per_epoch=args.train_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rt = runtime_from_args(args)
    validate_specs(tuned_config_dir=rt.tuned_config_dir)
    for dataset_name in args.datasets:
        if (args.spec or dataset_name) not in TUNING_SPECS:
            raise KeyError(
                f"No search space for dataset {dataset_name!r}; pass --spec "
                f"{list(TUNING_SPECS)} to reuse one."
            )
    runner.seed_everything(SEED)
    rt.checkpoint_root.mkdir(parents=True, exist_ok=True)
    rt.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.rerank:
        for dataset_name in args.datasets:
            rerank_top_configs(dataset_name, rt=rt, top_k=args.rerank_top_k, spec_name=args.spec)
        return

    summaries: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        summaries[dataset_name] = tune_one_dataset(
            dataset_name, rt=rt, spec_name=args.spec, worker_id=args.worker_id
        )

    print("\n" + "=" * 100)
    print("Finished tuning.")
    print(json.dumps(summaries, indent=2, sort_keys=True))

    # Single-dataset invocations (array tasks / parallel workers) skip the
    # shared combined summary so concurrent jobs never overwrite it; the
    # per-dataset artifacts already cover them.
    if len(args.datasets) > 1:
        text = json.dumps(summaries, indent=2, sort_keys=True)
        for path in (
            rt.checkpoint_root / "tuning_summary.json",
            rt.artifacts_dir / "sasrec_sce_tuning_summary.json",
        ):
            path.write_text(text, encoding="utf-8")
            print(f"Saved combined summary to: {path}")


if __name__ == "__main__":
    main()
