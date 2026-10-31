"""Optuna hyperparameter tuning of SASRec for the exp2 (tuned-params) runs.

Tunes SASRec independently on the four exp2 datasets — Amazon Beauty 2014,
Amazon Instruments 2018, Amazon Beauty 2023, and Yambda 50M listens — by
maximizing ``val/sft/ndcg@10`` on the leave-one-out validation split.

Preprocessing parameters are NOT tuned: every datamodule is built from the
dataset YAMLs in ``configs/exp2-tuned-params`` through the exact same
config-merging code path as ``run_sasrec_all.py``, so tuned hyperparameters
remain valid for the final multi-seed training runs.

Per dataset the script maintains, under
``experiments/exp2-tuning/sasrec/<dataset>/``:

- ``optuna_journal.log``      resumable study storage (NFS-safe journal;
                              re-running continues until the planned trial
                              count is reached)
- ``best_model.pt``           best-epoch weights of the incumbent best trial,
                              promoted as soon as a trial improves the study
- ``best_trial.json``         incumbent metadata (params, value, epoch)
- ``epoch_val_ndcg10.jsonl``  per-epoch monitor values for every trial
- ``all_trials.json``         full study dump written after the study ends

and writes the winning hyperparameters to
``configs/exp2-tuned-params/sasrec_model_<dataset>.yaml`` (architecture +
optimizer keys, mirroring the exp1 ``sasrec_model.yaml`` layout).

The trial budget lives in the shared study, not in the process: several
processes started with the same ``--datasets`` value cooperate on one study
and together stop once it holds the planned number of finished trials. That
is how the large datasets parallelize trials across a Slurm array — one
worker per array task, all sharing the journal — while the small datasets
run as one worker each:

- beauty, instruments        100 trials, single worker per dataset
                             (scripts/slurm/tuning_sasrec_array.sbatch)
- beauty_2023                50 trials, worker array
                             (scripts/slurm/tuning_sasrec_beauty_2023_array.sbatch)
- yambda_50m_listens         50 trials, worker array
                             (scripts/slurm/tuning_sasrec_yambda_50m_array.sbatch)

Two safeguards address the findings from the first exp2 round (where tuning
on Instruments failed to beat the exp1 baseline): every study starts from
the exp1 baseline configuration (enqueued as the first trial, so the search
always contains the incumbent it is trying to beat), and after a search
reaches its budget, ``--rerank`` retrains the top configurations with three
full seeds each and rewrites the tuned YAML with the best mean — countering
the winner's curse of selecting on single-seed validation scores.

Usage::

    python scripts/exp2-tuned-params/tuning_sasrec.py                # all four
    python scripts/exp2-tuned-params/tuning_sasrec.py --datasets beauty
    # one worker of a parallel study (repeat with different --worker-id):
    python scripts/exp2-tuned-params/tuning_sasrec.py \
        --datasets beauty_2023 --worker-id 3
    # after the search finished: 3-seed re-rank of the top configs
    # (single job, not a worker array):
    python scripts/exp2-tuned-params/tuning_sasrec.py \
        --datasets instruments --rerank
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.util
import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping

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
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.trainer import Trainer, TrainerConfig
from scripts import sasrec_config_runner as runner


def _load_run_all_module() -> Any:
    """Import run_sasrec_all.py (hyphenated directory, so load by path)."""

    path = (
        Path(__file__).resolve().parents[1]
        / "exp1-standard-params"
        / "run_sasrec_all.py"
    )
    spec = importlib.util.spec_from_file_location("exp1_run_sasrec_all", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_all = _load_run_all_module()


# ---------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
CHECKPOINT_ROOT = PROJECT_ROOT / "experiments" / "exp2-tuning" / "sasrec"

DATASETS_TO_TUNE = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens")

SEED = 42

# Seeds for the --rerank stage: the top search configs are retrained on all
# of these and ranked by their mean validation NDCG@10.
RERANK_SEEDS = (42, 43, 44)

# The exp1 "standard" configuration (configs/exp1-standard_params/
# sasrec_model.yaml + trainer_sasrec.yaml). Enqueued as the first trial of
# every study: the search must contain the incumbent it is trying to beat —
# the first exp2 round excluded it on Instruments (num_layers was capped at
# 2) and failed to improve on it. validate_specs() enforces containment.
EXP1_BASELINE_PARAMS = {
    "d_model": 128,
    "nhead": 2,
    "num_layers": 3,
    "dropout": 0.3,
    "pos_drop": 0.3,
    "lr": 3e-4,
    "weight_decay": 1e-2,
}
MONITOR = "val/sft/ndcg@10"
REQUIRED_TRAIN_LAYOUT = "dense_causal"

# SASRec launches follow the tiger batching criteria: effective batch 2048
# everywhere. The per-step batch size comes from the dataset YAMLs (capped by
# the dense-causal full-catalog logits), and each spec's grad_accum tops it
# up to 2048: beauty/instruments 2048x1, beauty_2023 512x4, yambda 128x16.
MAX_GRAD_NORM = 1.0
BETAS = (0.9, 0.999)
EPS = 1e-8
LOG_EVERY_STEPS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
PROGRESS_BAR = True

EPOCH_LOG_FILENAME = "epoch_val_ndcg10.jsonl"


# ---------------------------------------------------------------------
# Per-dataset search spaces
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TuningSpec:
    """Search space and trial budget for one dataset."""

    d_model: tuple[int, ...]
    nhead: tuple[int, ...]          # every choice must divide every d_model
    num_layers: tuple[int, int]     # inclusive range
    dropout: tuple[float, float]
    pos_drop: tuple[float, float]
    lr: tuple[float, float]         # log-uniform
    weight_decay: tuple[float, float]  # log-uniform
    n_trials: int
    n_startup_trials: int
    max_epochs: int
    patience: int
    pruner_warmup_epochs: int
    # Together with the dataset YAML's batch_size this must give an effective
    # batch of 2048, matching the tiger batching criteria.
    grad_accum: int
    notes: str


# Ranges follow the accumulated SASRec tuning experience: on the small 5-core
# Amazon sets the original paper and later reproductions (SASRec+ with full
# cross-entropy, BERT4Rec replication studies) converge to small hidden sizes
# with heavy dropout, while large dense implicit-feedback logs reward more
# capacity, more depth, and much weaker regularization.
TUNING_SPECS: dict[str, TuningSpec] = {
    # Beauty 2014, 5-core: ~22k users / ~12k items / ~0.2M reviews, mean
    # history ~9, >99.9% sparse. Models overfit within a few epochs, so the
    # space is biased toward small capacity and strong dropout; lr around
    # 1e-3 is the known sweet spot for full-CE SASRec here.
    "beauty": TuningSpec(
        d_model=(32, 64, 128),
        nhead=(1, 2, 4),
        num_layers=(1, 3),
        dropout=(0.3, 0.7),
        pos_drop=(0.1, 0.6),
        lr=(1e-4, 3e-3),
        weight_decay=(1e-6, 3e-2),
        n_trials=100,
        n_startup_trials=12,
        max_epochs=300,
        patience=10,
        pruner_warmup_epochs=10,
        grad_accum=1,  # batch 2048 from the YAML
        notes="small sparse 5-core Amazon; small models, heavy dropout",
    ),
    # Instruments 2018, 5-core: the smallest set (~25k users / ~10k items),
    # sequences even shorter than Beauty. Same regime as Beauty. The first
    # exp2 round capped depth at 2 blocks, but the 3-layer exp1 baseline
    # matched that round's tuned result, so the cap was removed and dropout
    # reaches down to 0.2 to cover the baseline's neighborhood.
    "instruments": TuningSpec(
        d_model=(32, 64, 128),
        nhead=(1, 2, 4),
        num_layers=(1, 3),
        dropout=(0.2, 0.7),
        pos_drop=(0.1, 0.6),
        lr=(1e-4, 3e-3),
        weight_decay=(1e-6, 3e-2),
        n_trials=100,
        n_startup_trials=12,
        max_epochs=300,
        patience=10,
        pruner_warmup_epochs=10,
        grad_accum=1,  # batch 2048 from the YAML
        notes="smallest sparse 5-core Amazon; shallow models, heavy dropout",
    ),
    # Beauty 2023 (verified purchases): one to two orders of magnitude more
    # users and a ~100k+ item catalog. Still sparse per user, but the extra
    # data supports larger embeddings and only moderate dropout; weight decay
    # matters more than dropout at this scale, hence the tighter upper bound.
    "beauty_2023": TuningSpec(
        d_model=(64, 128, 256),
        nhead=(2, 4, 8),
        num_layers=(1, 3),
        dropout=(0.1, 0.5),
        pos_drop=(0.0, 0.4),
        lr=(1e-4, 3e-3),
        weight_decay=(1e-6, 1e-2),
        n_trials=50,
        n_startup_trials=8,
        max_epochs=300,
        patience=8,
        pruner_warmup_epochs=6,
        grad_accum=4,  # batch 512 from the YAML -> effective 2048
        notes="large sparse Amazon 2023; medium capacity, moderate dropout",
    ),
    # Yambda 50M listens: dense music-streaming log with long histories
    # (max_history_len 50 from the dataset YAML) and a huge catalog. Dense
    # implicit feedback favors wide/deep models and little dropout; epochs
    # are expensive, so patience is short and the pruner starts early. lr
    # floor is higher — tiny rates cannot traverse this data volume before
    # early stopping kicks in.
    "yambda_50m_listens": TuningSpec(
        d_model=(128, 256),
        nhead=(2, 4, 8),
        num_layers=(2, 4),
        dropout=(0.0, 0.3),
        pos_drop=(0.0, 0.3),
        lr=(2e-4, 3e-3),
        weight_decay=(1e-6, 1e-2),
        n_trials=50,
        n_startup_trials=5,
        max_epochs=300,
        patience=5,
        pruner_warmup_epochs=3,
        grad_accum=16,  # batch 128 from the YAML -> effective 2048
        notes="dense large-scale music log; wide/deep models, light dropout",
    ),
}

TUNED_CONFIG_DATA_YAML = {
    "beauty": "dataset_amazon.yaml",
    "instruments": "dataset_amazon.yaml",
    "beauty_2023": "dataset_amazon_2023.yaml",
    "yambda_50m_listens": "dataset_yambda.yaml",
}


def validate_specs() -> None:
    """Every search space must contain the exp1 incumbent configuration."""

    base = EXP1_BASELINE_PARAMS
    for name, spec in TUNING_SPECS.items():
        contained = (
            base["d_model"] in spec.d_model
            and base["nhead"] in spec.nhead
            and spec.num_layers[0] <= base["num_layers"] <= spec.num_layers[1]
            and spec.dropout[0] <= base["dropout"] <= spec.dropout[1]
            and spec.pos_drop[0] <= base["pos_drop"] <= spec.pos_drop[1]
            and spec.lr[0] <= base["lr"] <= spec.lr[1]
            and spec.weight_decay[0] <= base["weight_decay"] <= spec.weight_decay[1]
        )
        if not contained:
            raise ValueError(
                f"TUNING_SPECS[{name!r}] excludes the exp1 baseline "
                f"configuration {base}; widen the space so the search "
                "contains the incumbent."
            )


def suggest_hparams(trial: optuna.Trial, spec: TuningSpec) -> dict[str, Any]:
    return {
        "d_model": trial.suggest_categorical("d_model", list(spec.d_model)),
        "nhead": trial.suggest_categorical("nhead", list(spec.nhead)),
        "num_layers": trial.suggest_int("num_layers", *spec.num_layers),
        "dropout": trial.suggest_float("dropout", *spec.dropout),
        "pos_drop": trial.suggest_float("pos_drop", *spec.pos_drop),
        "lr": trial.suggest_float("lr", *spec.lr, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *spec.weight_decay, log=True),
    }


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def find_group(dataset_name: str) -> tuple[Any, dict[str, Any]]:
    """Locate the run_sasrec_all dataset group that owns ``dataset_name``."""

    for group in run_all.GROUPS:
        dataset_config = runner.load_yaml(CONFIG_DIR / group.data_config_name)
        if dataset_name in dataset_config.get("datasets", {}):
            return group, dataset_config
    raise KeyError(
        f"Dataset {dataset_name!r} is not defined in any dataset YAML under "
        f"{CONFIG_DIR}."
    )


def build_datamodule(dataset_name: str) -> Any:
    group, dataset_config = find_group(dataset_name)
    group.validate_inputs([dataset_name], dataset_config)

    # The split seed is fixed for the whole study so every trial sees the
    # identical train/val partition; only trial seeds vary.
    data_cfg = group.build_data_config(dataset_name, dataset_config, seed=SEED)
    layout = getattr(data_cfg, "train_example_layout", None)
    if layout != REQUIRED_TRAIN_LAYOUT:
        raise ValueError(
            f"Dataset {dataset_name!r} resolves train_example_layout={layout!r}; "
            f"exp2 tuning requires {REQUIRED_TRAIN_LAYOUT!r}."
        )

    dm = group.data_module(data_cfg)
    dm.setup()
    print(f"\n[{dataset_name}] data summary:")
    print(json.dumps(dm.summary(), indent=2, sort_keys=True, default=str))
    return dm


# ---------------------------------------------------------------------
# Model / method / trainer
# ---------------------------------------------------------------------

def build_model(dm: Any, hparams: Mapping[str, Any]) -> SASRec:
    cfg = SASRecConfig(
        item_vocab_size=int(dm.item_vocab_size),
        # Positional embeddings must also cover dense training windows.
        max_history_len=max(
            int(dm.cfg.max_history_len), int(dm.cfg.train_window_len or 0)
        ),
        d_model=int(hparams["d_model"]),
        nhead=int(hparams["nhead"]),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams["dropout"]),
        pos_drop=float(hparams["pos_drop"]),
        initializer_range=0.02,
        layer_norm_eps=1e-8,
        pad_item_id=int(dm.cfg.pad_item_id),
        exclude_history_in_generate=False,
    )
    model = SASRec(cfg)
    if dm.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"SASRec does not support train_example_layout="
            f"{dm.cfg.train_example_layout!r}; supported: "
            f"{model.supported_train_layouts}."
        )
    return model


def build_method(dm: Any) -> SFTMethod:
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
        )
    )


def build_trainer(
    *,
    model: SASRec,
    method: SFTMethod,
    hparams: Mapping[str, Any],
    spec: TuningSpec,
    checkpoint_dir: Path,
    logger: Any,
) -> Trainer:
    cfg = TrainerConfig(
        max_epochs=spec.max_epochs,
        gradient_accumulation_steps=spec.grad_accum,
        max_grad_norm=MAX_GRAD_NORM,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        log_every_steps=LOG_EVERY_STEPS,
        progress_bar=PROGRESS_BAR,
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
    )
    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=cfg,
        device=DEVICE,
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        betas=BETAS,
        eps=EPS,
        logger=logger,
    )


# ---------------------------------------------------------------------
# Per-epoch reporting and pruning
# ---------------------------------------------------------------------

class EpochValReporter:
    """Trainer logger hook: record per-epoch NDCG@10 and prune bad trials."""

    def __init__(self, *, trial: optuna.Trial, dataset_name: str, log_path: Path):
        self.trial = trial
        self.dataset_name = dataset_name
        self.log_path = log_path
        self.epoch = 0

    def __call__(self, payload: dict[str, Any]) -> None:
        if payload.get("type") != "val_end" or MONITOR not in payload:
            return
        self.epoch += 1
        value = float(payload[MONITOR])

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "dataset": self.dataset_name,
                        "trial": self.trial.number,
                        "epoch": self.epoch,
                        "metric": MONITOR,
                        "val_ndcg10": value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(
            f"[{self.dataset_name} | trial {self.trial.number} | "
            f"epoch {self.epoch}] {MONITOR}={value:.6f}",
            flush=True,
        )

        self.trial.report(value, step=self.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at epoch {self.epoch} with {MONITOR}={value:.6f}"
            )


# ---------------------------------------------------------------------
# Incumbent promotion and config output
# ---------------------------------------------------------------------

def yaml_float(value: float) -> str:
    """Format a float so PyYAML re-reads it as a float (1e-05 would not)."""

    text = f"{float(value):.6g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        text = f"{mantissa}e{exponent}"
    elif "." not in text:
        text += ".0"
    return text


def write_tuned_config(
    *,
    dataset_name: str,
    hparams: Mapping[str, Any],
    best_value: float,
    best_trial_number: int,
    best_epoch: int | None,
    trials_completed: int,
    trials_planned: int,
    study_name: str,
    grad_accum: int,
    selection: str = "best single-seed trial of the search",
) -> Path:
    path = CONFIG_DIR / f"sasrec_model_{dataset_name}.yaml"
    data_yaml = TUNED_CONFIG_DATA_YAML[dataset_name]
    text = f"""\
# SASRec hyperparameters for {dataset_name} (exp2, Optuna-tuned).
# Produced by scripts/exp2-tuned-params/tuning_sasrec.py: TPE sampler +
# median pruner maximizing {MONITOR} on the leave-one-out validation split.
# Preprocessing parameters come from configs/exp2-tuned-params/{data_yaml}
# and were not tuned. Best checkpoint:
# experiments/exp2-tuning/sasrec/{dataset_name}/best_model.pt
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

# Fixed trainer settings the tuning ran with. gradient_accumulation_steps
# tops the dataset YAML's batch_size up to the tiger-matching effective 2048.
max_grad_norm: {yaml_float(MAX_GRAD_NORM)}
gradient_accumulation_steps: {grad_accum}

# SFT objective and metrics (matches exp1 sasrec_model.yaml)
sft:
  compute_train_metrics: false
  compute_eval_metrics: true
  compute_item_metrics: true
  compute_token_metrics: false
  evaluate_generation: false
  item_topk: [1, 5, 10, 20]
  pad_item_id: 0
"""
    path.write_text(text, encoding="utf-8")
    return path


@contextmanager
def promotion_lock(dataset_dir: Path) -> Iterator[None]:
    """Serialize incumbent promotion across parallel workers via flock."""

    lock_path = dataset_dir / ".promote.lock"
    try:
        lock_file = lock_path.open("w")
    except OSError:
        yield
        return
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError:
            print(
                f"[promotion] WARNING: could not lock {lock_path}; "
                "promoting without a lock.",
                flush=True,
            )
        yield
    finally:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def promote_if_incumbent(
    *,
    dataset_name: str,
    dataset_dir: Path,
    trial: optuna.Trial,
    hparams: Mapping[str, Any],
    value: float,
    best_epoch: int | None,
    checkpoint_path: Path | None,
    spec: TuningSpec,
    study_name: str,
) -> None:
    """Persist the model/params as soon as a trial beats the stored best."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with promotion_lock(dataset_dir):
        _promote_if_incumbent_locked(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            trial=trial,
            hparams=hparams,
            value=value,
            best_epoch=best_epoch,
            checkpoint_path=checkpoint_path,
            spec=spec,
            study_name=study_name,
        )


def _promote_if_incumbent_locked(
    *,
    dataset_name: str,
    dataset_dir: Path,
    trial: optuna.Trial,
    hparams: Mapping[str, Any],
    value: float,
    best_epoch: int | None,
    checkpoint_path: Path | None,
    spec: TuningSpec,
    study_name: str,
) -> None:
    best_json = dataset_dir / "best_trial.json"
    if best_json.exists():
        try:
            previous = json.loads(best_json.read_text(encoding="utf-8"))
            previous_value = float(previous.get("best_value_val_ndcg10", float("-inf")))
        except (json.JSONDecodeError, TypeError, ValueError):
            previous_value = float("-inf")
    else:
        previous_value = float("-inf")

    if value <= previous_value:
        return

    if checkpoint_path is not None and checkpoint_path.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, dataset_dir / "best_model.pt")
    else:
        print(
            f"[{dataset_name} | trial {trial.number}] WARNING: no best-epoch "
            "checkpoint found; best_model.pt not updated.",
            flush=True,
        )

    best_json.parent.mkdir(parents=True, exist_ok=True)
    best_json.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "study_name": study_name,
                "metric": MONITOR,
                "best_value_val_ndcg10": value,
                "best_trial_number": trial.number,
                "best_epoch": best_epoch,
                "best_params": dict(hparams),
                "checkpoint": str(dataset_dir / "best_model.pt"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    config_path = write_tuned_config(
        dataset_name=dataset_name,
        hparams=hparams,
        best_value=value,
        best_trial_number=trial.number,
        best_epoch=best_epoch,
        trials_completed=trial.number + 1,
        trials_planned=spec.n_trials,
        study_name=study_name,
        grad_accum=spec.grad_accum,
    )
    print(
        f"[{dataset_name} | trial {trial.number}] new best "
        f"{MONITOR}={value:.6f}; updated best_model.pt and {config_path}",
        flush=True,
    )


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

def train_once(
    *,
    dm: Any,
    hparams: Mapping[str, Any],
    spec: TuningSpec,
    checkpoint_dir: Path,
    seed: int,
    logger: Any = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Train one SASRec with ``hparams`` and return its best-epoch result.

    Shared by the search objective and the --rerank stage so both train
    under exactly the same regime. The caller owns ``checkpoint_dir``
    cleanup.
    """

    model: SASRec | None = None
    trainer: Trainer | None = None
    try:
        runner.seed_everything(seed)
        model = build_model(dm, hparams)
        method = build_method(dm)
        trainer = build_trainer(
            model=model,
            method=method,
            hparams=hparams,
            spec=spec,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
        print(
            f"{log_prefix} params: {json.dumps(dict(hparams), sort_keys=True)} "
            f"| trainable params: "
            f"{model.num_parameters(trainable_only=True):,}",
            flush=True,
        )

        fit_result = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

        value = trainer.state.best_metric
        if value is None:
            value = fit_result.val_metrics.get(MONITOR)
        if value is None:
            raise RuntimeError(f"{log_prefix} finished without a {MONITOR} value.")

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
        }
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_objective(*, dataset_name: str, dm: Any, dataset_dir: Path, spec: TuningSpec, study_name: str):
    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial, spec)
        trial_dir = dataset_dir / "trials" / f"trial_{trial.number}"

        try:
            reporter = EpochValReporter(
                trial=trial,
                dataset_name=dataset_name,
                log_path=dataset_dir / EPOCH_LOG_FILENAME,
            )
            result = train_once(
                dm=dm,
                hparams=hparams,
                spec=spec,
                checkpoint_dir=trial_dir,
                seed=SEED + trial.number,
                logger=reporter,
                log_prefix=f"[{dataset_name} | trial {trial.number}]",
            )
            value = result["value"]
            best_epoch = result["best_epoch"]

            trial.set_user_attr("dataset", dataset_name)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("epochs_completed", result["epochs_completed"])

            promote_if_incumbent(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                trial=trial,
                hparams=hparams,
                value=value,
                best_epoch=best_epoch,
                checkpoint_path=result["checkpoint"],
                spec=spec,
                study_name=study_name,
            )

            print(
                f"[{dataset_name} | trial {trial.number}] done: "
                f"best_epoch={best_epoch} {MONITOR}={value:.6f}",
                flush=True,
            )
            return value

        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                print(
                    f"[{dataset_name} | trial {trial.number}] CUDA OOM; pruning.",
                    flush=True,
                )
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

def tune_one_dataset(dataset_name: str, *, worker_id: int = 0) -> dict[str, Any]:
    spec = TUNING_SPECS[dataset_name]
    dataset_dir = CHECKPOINT_ROOT / runner.safe_component(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"sasrec_{dataset_name}_val_ndcg10_exp2"
    journal_path = dataset_dir / "optuna_journal.log"

    print("\n" + "=" * 100)
    print(f"Tuning SASRec on {dataset_name}")
    print("=" * 100)
    print(f"Config dir:      {CONFIG_DIR}")
    print(f"Output dir:      {dataset_dir}")
    print(f"Study:           {study_name} (journal: {journal_path})")
    print(f"Worker:          {worker_id}")
    print(f"Device:          {DEVICE} (amp={USE_AMP}, dtype={AMP_DTYPE})")
    print(f"Budget:          {spec.n_trials} trials, <= {spec.max_epochs} epochs, "
          f"patience {spec.patience}")
    print(f"Space notes:     {spec.notes}")

    dm = build_datamodule(dataset_name)

    sampler = optuna.samplers.TPESampler(
        # Distinct seeds keep parallel workers from proposing identical
        # startup configurations.
        seed=SEED + 9973 * worker_id,
        n_startup_trials=spec.n_startup_trials,
        multivariate=True,
        # Treat running trials as if finished at their current value so
        # concurrent workers avoid near-duplicate suggestions. No effect
        # when a study runs single-worker.
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
        # Seed the search with the exp1 incumbent configuration so tuning
        # can only improve on it in expectation. skip_if_exists keeps
        # resumed and re-submitted runs from enqueueing it twice.
        study.enqueue_trial(dict(EXP1_BASELINE_PARAMS), skip_if_exists=True)

    finished_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    )
    already_done = len([t for t in study.trials if t.state in finished_states])
    if already_done:
        print(f"Resuming study: {already_done}/{spec.n_trials} finished trials.")

    if already_done < spec.n_trials:
        # The budget is enforced against the shared study, not this process:
        # every worker keeps pulling trials until the study holds n_trials
        # finished ones, so any number of workers can cooperate without
        # overshooting by more than the trials still in flight.
        study.optimize(
            make_objective(
                dataset_name=dataset_name,
                dm=dm,
                dataset_dir=dataset_dir,
                spec=spec,
                study_name=study_name,
            ),
            n_trials=None,
            gc_after_trial=True,
            show_progress_bar=False,
            callbacks=[MaxTrialsCallback(spec.n_trials, states=finished_states)],
        )

    best = study.best_trial
    trials_completed = len(
        [t for t in study.trials if t.state in finished_states]
    )

    # Rewrite the tuned config with final study statistics.
    config_path = write_tuned_config(
        dataset_name=dataset_name,
        hparams=best.params,
        best_value=float(study.best_value),
        best_trial_number=best.number,
        best_epoch=best.user_attrs.get("best_epoch"),
        trials_completed=trials_completed,
        trials_planned=spec.n_trials,
        study_name=study_name,
        grad_accum=spec.grad_accum,
    )

    with (dataset_dir / "all_trials.json").open("w", encoding="utf-8") as f:
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

    summary = {
        "dataset": dataset_name,
        "best_value_val_ndcg10": float(study.best_value),
        "best_trial_number": best.number,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "best_params": best.params,
        "trials_completed": trials_completed,
        "checkpoint": str(dataset_dir / "best_model.pt"),
        "tuned_config": str(config_path),
    }
    print(f"\n[{dataset_name}] best {MONITOR}: {study.best_value:.6f} "
          f"(trial {best.number})")
    print(json.dumps(best.params, indent=2, sort_keys=True))
    print(f"[{dataset_name}] tuned config: {config_path}")
    print(f"[{dataset_name}] best model:   {dataset_dir / 'best_model.pt'}")

    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------
# Multi-seed re-ranking of the top search configurations
# ---------------------------------------------------------------------

def rerank_top_configs(dataset_name: str, *, top_k: int) -> dict[str, Any]:
    """Retrain the top-k distinct search configs on RERANK_SEEDS and pick
    the winner by mean validation NDCG@10.

    Single-seed trial values are noisy on the small datasets, so the search
    argmax overfits the validation split (winner's curse). This stage
    rewrites the tuned YAML, best_model.pt, and best_trial.json with the
    config whose mean over three full seeds is best.
    """

    spec = TUNING_SPECS[dataset_name]
    dataset_dir = CHECKPOINT_ROOT / runner.safe_component(dataset_name)
    journal_path = dataset_dir / "optuna_journal.log"
    if not journal_path.exists():
        raise FileNotFoundError(
            f"No study journal at {journal_path}; run the search for "
            f"{dataset_name!r} first."
        )

    study = optuna.load_study(
        study_name=f"sasrec_{dataset_name}_val_ndcg10_exp2",
        storage=JournalStorage(JournalFileBackend(str(journal_path))),
    )
    finished = [
        t
        for t in study.trials
        if t.state
        in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(f"Study for {dataset_name!r} has no completed trials.")
    if len(finished) < spec.n_trials:
        print(
            f"[{dataset_name} | rerank] WARNING: study holds "
            f"{len(finished)}/{spec.n_trials} finished trials; re-ranking "
            "an unfinished search.",
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
        f"[{dataset_name} | rerank] re-training top {len(candidates)} configs "
        f"x {len(RERANK_SEEDS)} seeds (search values: "
        f"{[round(t.value, 6) for t in candidates]})",
        flush=True,
    )
    dm = build_datamodule(dataset_name)

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
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                    log_prefix=(
                        f"[{dataset_name} | rerank trial {t.number} | seed {seed}]"
                    ),
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(
                    f"[{dataset_name} | rerank trial {t.number} | seed {seed}] "
                    "CUDA OOM; dropping this config.",
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
                "search_value": float(t.value),
                "seed_results": seed_results,
                "mean_val_ndcg10": mean,
                "std_val_ndcg10": std,
            }
        )
        print(
            f"[{dataset_name} | rerank trial {t.number}] mean {MONITOR}="
            f"{mean:.6f} +- {std:.6f} (search value {t.value:.6f})",
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
        best_value=winner["mean_val_ndcg10"],
        best_trial_number=winner["trial"],
        best_epoch=best_seed["best_epoch"],
        trials_completed=len(finished),
        trials_planned=spec.n_trials,
        study_name=study.study_name,
        grad_accum=spec.grad_accum,
        selection=selection,
    )
    summary = {
        "dataset": dataset_name,
        "selection": selection,
        "winner": winner,
        "ranking": results,
        "tuned_config": str(config_path),
        "checkpoint": str(dataset_dir / "best_model.pt"),
    }
    (dataset_dir / "rerank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (dataset_dir / "best_trial.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "study_name": study.study_name,
                "metric": MONITOR,
                "selection": selection,
                "best_value_val_ndcg10": winner["mean_val_ndcg10"],
                "std_val_ndcg10": winner["std_val_ndcg10"],
                "best_trial_number": winner["trial"],
                "best_epoch": best_seed["best_epoch"],
                "best_params": winner["params"],
                "checkpoint": str(dataset_dir / "best_model.pt"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(rerank_root, ignore_errors=True)

    print(
        f"\n[{dataset_name} | rerank] winner: trial {winner['trial']} with "
        f"mean {MONITOR}={winner['mean_val_ndcg10']:.6f} "
        f"+- {winner['std_val_ndcg10']:.6f}",
        flush=True,
    )
    print(f"[{dataset_name} | rerank] tuned config: {config_path}")
    print(f"[{dataset_name} | rerank] summary: {dataset_dir / 'rerank_summary.json'}")

    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune SASRec with Optuna (val/sft/ndcg@10) on the exp2 datasets: "
            "beauty, instruments, beauty_2023, yambda_50m_listens."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS_TO_TUNE),
        choices=list(DATASETS_TO_TUNE),
        help="Subset of datasets to tune (default: all four).",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help=(
            "Index of this worker within a multi-worker study (e.g. the "
            "Slurm array task ID). Workers started with the same --datasets "
            "share the study journal and global trial budget; the ID only "
            "diversifies the sampler seed."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Skip searching; instead retrain the top configurations of the "
            "finished study with three full seeds each and rewrite the "
            "tuned YAML/checkpoint with the best mean. Run as a single job "
            "per dataset, not a worker array."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=3,
        help="How many distinct top search configs to re-rank (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_specs()
    runner.seed_everything(SEED)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.rerank:
        for dataset_name in args.datasets:
            rerank_top_configs(dataset_name, top_k=args.rerank_top_k)
        return

    summaries: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        summaries[dataset_name] = tune_one_dataset(
            dataset_name, worker_id=args.worker_id
        )

    print("\n" + "=" * 100)
    print("Finished tuning.")
    print(json.dumps(summaries, indent=2, sort_keys=True))

    # Single-dataset invocations (array tasks / parallel workers) skip the
    # shared combined summary so concurrent jobs never overwrite it; the
    # per-dataset outputs under each dataset dir already cover them.
    if len(args.datasets) > 1:
        summary_path = CHECKPOINT_ROOT / "tuning_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, sort_keys=True)
        print(f"Saved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
