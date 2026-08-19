#!/usr/bin/env python
"""Grid tuning of TIGER (SFT) on Yambda 500M listens for the exp2 tuned-params runs.

Why a grid and not Optuna
-------------------------
One Yambda-500M epoch costs ~20-30 min on an A100 (the exp1 seed-42 run:
4,886 optimizer steps of batch 2048 per epoch, ~27 min/epoch, 50 epochs in
the 24 h wall) and the datamodule setup alone takes ~50 min and ~1.3 TB RSS.
A TPE study of tens of trials x full training is out of budget, so this
script sweeps a SMALL, EXPLICIT grid around the configuration that produced
the current checkpoint::

    checkpoints/exp1-standard_params/tiger/yambda_500m_listens/seed_42/best.pt
    <- configs/exp1-standard_params/tiger_model.yaml
       + configs/exp1-standard_params/trainer_tiger.yaml
       + the tiger: block of configs/exp2-tuned-params/dataset_yambda.yaml

training every grid point for the same short epoch budget and comparing them
on the same validation subset.

Design
------
* Incumbent = the current parameters, read from the configs at runtime (never
  hard-coded), so the sweep is always centred on what produced the checkpoint.
  Every grid axis contains the incumbent value by construction.
* One-factor-at-a-time (``--design ofat``, default): the plan is the baseline
  plus every single-parameter deviation from the grid — the SUM, not the
  product, of the grid sizes (10 trials with the default grid).
  ``--design full`` runs the cartesian product of the selected ``--params``
  when a small interaction study is affordable.
* Equal budget: every trial trains ``--epochs-per-trial`` (default 8) epochs
  from scratch with the same seed and is scored by its best ``--monitor``
  value (default ``val/sft/ndcg@10`` from constrained 10-beam generation on a
  FIXED 25% validation subsample: the reported metric, and far less noisy than
  1-beam ndcg@1 at this budget). The exp1 checkpoint had reached ~88% of its
  epoch-50 NDCG@1 by epoch 8, so 8 epochs separate configurations without
  paying for convergence. Raising ``--epochs-per-trial`` later EXTENDS the
  finished trials from their ``last.pt`` instead of retraining them.
* Combine: once every OFAT trial has a result, the per-parameter winners that
  beat the baseline by >= ``--min-rel-improvement`` are merged into one
  ``combined`` trial (same budget) — the usual follow-up that captures
  additive gains without a full grid.
* Cluster-friendly: workers (Slurm array tasks) share the plan on disk and
  claim trials atomically (flock + heartbeat), so any number of workers can
  cooperate; results are per-trial JSON files, so re-submitting resumes the
  sweep, and an interrupted trial resumes from its ``last.pt`` (24 h walls vs
  4-8 h trials). Each worker pays the datamodule setup once and then trains
  as many trials as fit in ``--time-budget-hours``.

Default grid (incumbent value marked with *; alternatives are the values
tried on top of it)::

    lr             1e-3, 1.85e-3*, 3e-3    optimizer step (AdamW, no schedule)
    dropout        0.0, 0.1*, 0.2          T5 dropout_rate
    embedding_dim  128*, 256               d_model; dim_feedforward keeps the
                                           incumbent d_ff/d_model ratio (4x)
    num_layers     2*, 4                   encoder = decoder depth
    weight_decay   0.0, 1e-2*, 1e-1        AdamW decoupled decay
    d_kv           32*, 64                 per-head attention width
    batch_size     (opt-in)                --grid batch_size=1024,4096
    max_grad_norm  (opt-in)                --grid max_grad_norm=1.0

Preprocessing (history length, last-128 target policy, filters, semantic IDs)
is NOT tuned: the datamodule is built through the exact config-merging path
of run_tiger_sft_all.py from the exp2 dataset YAML, so tuned values remain
valid for the final multi-seed runs.

Outputs (``<root>`` = checkpoints/exp2-tuned_params/tiger/yambda_500m_listens)::

    <root>/plan.json                       the trial plan (names, overrides, hparams)
    <root>/trials/<trial>/CLAIMED          worker claim (heartbeat = mtime)
    <root>/trials/<trial>/result.json      per-trial result (curve, best value/epoch, timing)
    <root>/trials/<trial>/epoch_log.jsonl  per-epoch records (train/val metrics, timing)
    <root>/trials/<trial>/best.pt|last.pt  best-epoch / resume-or-extend checkpoints
    <root>/best_trial.json                 winner + params (rewritten at every aggregation)
    <root>/tuning_summary.json             all trials, per-parameter effects, combined derivation
    artifacts/exp2-tuned-params/tiger_yambda_500m_listens_tuning_summary.csv
    configs/exp2-tuned-params/tiger_model_yambda_500m_listens.yaml   tuned config
        (tiger_model.yaml layout + optimizer/trainer keys, like sasrec_model_<ds>.yaml)

Usage::

    # one worker (e.g. one Slurm array task; see scripts/slurm/tuning_tiger_yambda_500m_array.sbatch)
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --device cuda --worker-id 0
    # print the plan / status without touching data
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --list-plan
    # aggregate finished trials, (re)write summary + tuned YAML, no GPU
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --stage report
    # a small interaction grid instead of OFAT
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --design full --params lr dropout
    # extra grid values / opt-in axes / an explicit extra point
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --grid lr=5e-4,1e-3,3e-3 --grid batch_size=4096
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --extra-trial lr=3e-3,dropout=0.0,num_layers=4
    # after killed jobs: forget failed/OOM results, or drop dead claims explicitly
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --retry-failed --reset-claims
    # wiring check on synthetic data (see tests/test_tuning_tiger_yambda_500m.py)
    python scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py --dataset yambda_smoke \
        --dataset-config <tmp>/dataset_yambda_smoke.yaml --semantic-ids-dir <tmp>/semantic_ids \
        --checkpoint-root <tmp>/ckpt --artifacts-dir <tmp>/artifacts --tuned-config-path <tmp>/tuned.yaml \
        --epochs-per-trial 2 --train-batches-per-epoch 3 --max-eval-batches 2 --num-workers 0 --device cpu

Follow-up: the winning configuration goes into the tuned YAML above; the final
exp2 multi-seed training runs it under the exp1 trainer schedule (max_epochs
500, patience 10). The winner's ``best.pt`` (8 epochs) can also warm-start
that run to save ~4 h.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib.util
import itertools
import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch
import yaml

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.ft_methods.sft.method import SFTMethod  # noqa: E402
from flowrec.training.trainer import Trainer, TrainerConfig  # noqa: E402
from scripts import sasrec_config_runner as runner  # noqa: E402


def _load_module(name: str, path: Path) -> Any:
    """Import a script from a hyphenated directory by path."""

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The exp1 TIGER runner owns the datamodule/model construction the checkpoint
# was trained with; reuse it so tuning trials see the identical data path.
tiger_all = _load_module(
    "exp1_run_tiger_sft_all",
    Path(__file__).resolve().parents[1] / "exp1-standard-params" / "run_tiger_sft_all.py",
)


# ---------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------

DATASET = "yambda_500m_listens"

EXP1_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
EXP2_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
DEFAULT_MODEL_CONFIG = EXP1_CONFIG_DIR / "tiger_model.yaml"
DEFAULT_TRAINER_CONFIG = EXP1_CONFIG_DIR / "trainer_tiger.yaml"
DEFAULT_DATASET_CONFIG = EXP2_CONFIG_DIR / "dataset_yambda.yaml"
DEFAULT_SEMANTIC_IDS_DIR = PROJECT_ROOT / "datasets" / "semantic_ids" / "exp2-tuned-params"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "exp2-tuned_params" / "tiger"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "exp2-tuned-params"
# The checkpoint whose configuration is the incumbent (documentation +
# optional cross-check of the architecture keys).
INCUMBENT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/exp1-standard_params/tiger/yambda_500m_listens/seed_42/best.pt"
)

SEED = 42
DEFAULT_MONITOR = "val/sft/ndcg@10"
DEFAULT_VAL_NUM_BEAMS = 10
DEFAULT_VAL_TOPK = (1, 5, 10)
DEFAULT_VAL_SAMPLE_FRACTION = 0.25
DEFAULT_EPOCHS_PER_TRIAL = 8
DEFAULT_MIN_REL_IMPROVEMENT = 0.01

BASELINE_NAME = "baseline"
CLAIM_FILENAME = "CLAIMED"
RESULT_FILENAME = "result.json"
EPOCH_LOG_FILENAME = "epoch_log.jsonl"
STATUS_COMPLETE = "complete"
STATUS_OOM = "oom"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------

@dataclass
class TrialConfig:
    """Mutable per-trial copy of the incumbent configuration."""

    model: dict[str, Any]
    trainer: dict[str, Any]
    batch_size: int
    grad_accum: int

    def hparams(self) -> dict[str, Any]:
        """Flat view of the tunable keys (for tables and the tuned YAML)."""

        return {
            "lr": float(self.trainer["lr"]),
            "weight_decay": float(self.trainer["weight_decay"]),
            "max_grad_norm": float(self.trainer["max_grad_norm"]),
            "dropout": float(self.model["dropout"]),
            "embedding_dim": int(self.model["embedding_dim"]),
            "dim_feedforward": int(self.model["dim_feedforward"]),
            "num_encoder_layers": int(self.model["num_encoder_layers"]),
            "num_decoder_layers": int(self.model["num_decoder_layers"]),
            "num_heads": int(self.model["num_heads"]),
            "d_kv": int(self.model["d_kv"]),
            "batch_size": int(self.batch_size),
            "gradient_accumulation_steps": int(self.grad_accum),
        }


@dataclass(frozen=True)
class GridParam:
    """One tunable axis: how to read the incumbent value and how to apply a value."""

    name: str
    alternatives: tuple[Any, ...]
    kind: type
    read: Callable[[TrialConfig], Any]
    apply: Callable[[TrialConfig, Any, TrialConfig], None]
    doc: str


def _apply_embedding_dim(cfg: TrialConfig, value: Any, incumbent: TrialConfig) -> None:
    ratio = int(incumbent.model["dim_feedforward"]) / int(incumbent.model["embedding_dim"])
    cfg.model["embedding_dim"] = int(value)
    cfg.model["dim_feedforward"] = int(round(int(value) * ratio))


def _read_num_layers(cfg: TrialConfig) -> int:
    enc = int(cfg.model["num_encoder_layers"])
    dec = int(cfg.model["num_decoder_layers"])
    if enc != dec:
        raise ValueError(
            "The num_layers grid axis ties encoder and decoder depth, but the "
            f"incumbent has num_encoder_layers={enc} != num_decoder_layers={dec}; "
            "drop num_layers from --params or align the incumbent config."
        )
    return enc


def _apply_num_layers(cfg: TrialConfig, value: Any, incumbent: TrialConfig) -> None:
    cfg.model["num_encoder_layers"] = int(value)
    cfg.model["num_decoder_layers"] = int(value)


def _model_param(name: str, alternatives: tuple[Any, ...], kind: type, doc: str) -> GridParam:
    return GridParam(
        name=name,
        alternatives=alternatives,
        kind=kind,
        read=lambda cfg: kind(cfg.model[name]),
        apply=lambda cfg, value, _inc: cfg.model.__setitem__(name, kind(value)),
        doc=doc,
    )


def _trainer_param(name: str, alternatives: tuple[Any, ...], kind: type, doc: str) -> GridParam:
    return GridParam(
        name=name,
        alternatives=alternatives,
        kind=kind,
        read=lambda cfg: kind(cfg.trainer[name]),
        apply=lambda cfg, value, _inc: cfg.trainer.__setitem__(name, kind(value)),
        doc=doc,
    )


def _apply_batch_size(cfg: TrialConfig, value: Any, incumbent: TrialConfig) -> None:
    cfg.batch_size = int(value)


# Ordered as the OFAT plan runs them: the axes most likely to matter for a
# T5-style generator on 10M+ training rows first (step size, regularization,
# capacity), then the cheaper secondary axes.
GRID_PARAMS: dict[str, GridParam] = {
    "lr": _trainer_param(
        "lr", (1e-3, 3e-3), float,
        "AdamW step size (no schedule); ~x0.5 / x1.6 around the incumbent 1.85e-3",
    ),
    "dropout": _model_param(
        "dropout", (0.0, 0.2), float,
        "T5 dropout_rate; the incumbent's val loss sits below its train loss, so 0.0 is a serious candidate",
    ),
    "embedding_dim": GridParam(
        name="embedding_dim",
        alternatives=(256,),
        kind=int,
        read=lambda cfg: int(cfg.model["embedding_dim"]),
        apply=_apply_embedding_dim,
        doc="d_model; dim_feedforward keeps the incumbent d_ff/d_model ratio (512/128 = 4x)",
    ),
    "num_layers": GridParam(
        name="num_layers",
        alternatives=(4,),
        kind=int,
        read=_read_num_layers,
        apply=_apply_num_layers,
        doc="encoder = decoder depth (the TIGER paper uses 4+4)",
    ),
    "weight_decay": _trainer_param(
        "weight_decay", (0.0, 1e-1), float,
        "AdamW decoupled weight decay",
    ),
    "d_kv": _model_param(
        "d_kv", (64,), int,
        "per-head attention width (the TIGER paper uses 64)",
    ),
    # Opt-in axes (empty default grids): --grid batch_size=1024,4096 etc.
    "batch_size": GridParam(
        name="batch_size",
        alternatives=(),
        kind=int,
        read=lambda cfg: int(cfg.batch_size),
        apply=_apply_batch_size,
        doc="train loader batch (effective batch = batch_size x gradient_accumulation_steps)",
    ),
    "max_grad_norm": _trainer_param(
        "max_grad_norm", (), float,
        "gradient clipping norm",
    ),
}
DEFAULT_PARAMS = tuple(name for name, spec in GRID_PARAMS.items() if spec.alternatives)


# ---------------------------------------------------------------------
# Incumbent
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Incumbent:
    """The current configuration (what produced the checkpoint) + the datamodule config."""

    config: TrialConfig
    data_cfg: Any                       # YambdaDataConfig
    test_eval_batch_size: int
    group: Any                          # tiger_all.DatasetGroup
    dataset_config: dict[str, Any]
    sources: dict[str, str]


def load_incumbent(args: argparse.Namespace) -> Incumbent:
    model_config = runner.load_yaml(args.model_config)
    trainer_config = tiger_all.apply_trainer_defaults(runner.load_yaml(args.trainer_config))
    dataset_config = runner.load_yaml(args.dataset_config)

    group = next((g for g in tiger_all.GROUPS if g.key == "yambda"), None)
    if group is None:
        raise RuntimeError("run_tiger_sft_all.py no longer defines the yambda dataset group.")
    if args.dataset not in dataset_config.get("datasets", {}):
        raise KeyError(
            f"Dataset {args.dataset!r} is not defined in {args.dataset_config}; "
            f"available: {list(dataset_config.get('datasets', {}))}"
        )

    data_cfg, test_eval_batch_size = tiger_all.build_data_config(
        group,
        args.dataset,
        dataset_config,
        seed=SEED,
        semantic_ids_dir=args.semantic_ids_dir,
        user_hash_mod=int(model_config.get("user_ids_count", 1)),
    )
    if args.num_workers is not None:
        data_cfg = replace(
            data_cfg,
            num_workers=int(args.num_workers),
            persistent_workers=bool(int(args.num_workers) > 0),
        )

    config = TrialConfig(
        model=dict(model_config),
        trainer=dict(trainer_config),
        batch_size=int(data_cfg.batch_size),
        grad_accum=int(trainer_config.get("gradient_accumulation_steps", 1)),
    )
    incumbent = Incumbent(
        config=config,
        data_cfg=data_cfg,
        test_eval_batch_size=int(test_eval_batch_size),
        group=group,
        dataset_config=dataset_config,
        sources={
            "model_config": str(args.model_config),
            "trainer_config": str(args.trainer_config),
            "dataset_config": str(args.dataset_config),
            "semantic_id_path": str(data_cfg.semantic_id_path),
            "checkpoint": str(INCUMBENT_CHECKPOINT),
        },
    )
    cross_check_incumbent_checkpoint(config)
    return incumbent


def cross_check_incumbent_checkpoint(config: TrialConfig) -> None:
    """Warn when the config YAMLs drifted from the checkpoint's own sidecar."""

    sidecar = INCUMBENT_CHECKPOINT.with_suffix(INCUMBENT_CHECKPOINT.suffix + ".json")
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    saved_model = payload.get("model_config", {})
    saved_trainer = payload.get("trainer_config", {})
    drift = []
    for key in ("embedding_dim", "num_heads", "num_encoder_layers", "num_decoder_layers",
                "dim_feedforward", "dropout", "codebook_size", "user_ids_count"):
        if key in saved_model and saved_model[key] != config.model.get(key):
            drift.append(f"model.{key}: yaml={config.model.get(key)!r} checkpoint={saved_model[key]!r}")
    for key in ("max_grad_norm", "gradient_accumulation_steps"):
        if key in saved_trainer and saved_trainer[key] != config.trainer.get(key):
            drift.append(f"trainer.{key}: yaml={config.trainer.get(key)!r} checkpoint={saved_trainer[key]!r}")
    if drift:
        print(
            "[incumbent] WARNING: the config YAMLs differ from the checkpoint "
            f"sidecar {sidecar}:\n  - " + "\n  - ".join(drift) + "\n  The sweep is "
            "centred on the YAML values.",
            flush=True,
        )


def clone_config(config: TrialConfig) -> TrialConfig:
    return TrialConfig(
        model=dict(config.model),
        trainer=dict(config.trainer),
        batch_size=int(config.batch_size),
        grad_accum=int(config.grad_accum),
    )


# ---------------------------------------------------------------------
# Trial plan
# ---------------------------------------------------------------------

def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class Trial:
    """A grid point, identified by its NON-incumbent overrides only, so the same
    configuration gets the same name (and result file) under every design."""

    overrides: dict[str, Any] = field(default_factory=dict)
    role: str = "sweep"

    @property
    def name(self) -> str:
        if not self.overrides:
            return BASELINE_NAME
        return "__".join(f"{key}-{fmt_value(value)}" for key, value in self.overrides.items())

    def config(self, incumbent: TrialConfig) -> TrialConfig:
        cfg = clone_config(incumbent)
        for key, value in self.overrides.items():
            GRID_PARAMS[key].apply(cfg, value, incumbent)
        return cfg


def resolve_grids(
    params: list[str],
    incumbent: TrialConfig,
    grid_overrides: Mapping[str, tuple[Any, ...]],
) -> dict[str, list[Any]]:
    """Per-parameter value lists, always containing the incumbent value first."""

    grids: dict[str, list[Any]] = {}
    for name in params:
        spec = GRID_PARAMS[name]
        current = spec.read(incumbent)
        alternatives = grid_overrides.get(name, spec.alternatives)
        values: list[Any] = [current]
        for value in alternatives:
            value = spec.kind(value)
            if not _same_value(value, current) and not any(_same_value(value, v) for v in values):
                values.append(value)
        grids[name] = values
    return grids


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def build_plan(
    *,
    design: str,
    params: list[str],
    grids: Mapping[str, list[Any]],
    incumbent: TrialConfig,
    extra_trials: list[dict[str, Any]],
) -> list[Trial]:
    trials: list[Trial] = [Trial({})]
    if design == "ofat":
        for name in params:
            for value in grids[name][1:]:
                trials.append(Trial({name: value}))
    elif design == "full":
        for combo in itertools.product(*[grids[name] for name in params]):
            overrides = {
                name: value
                for name, value in zip(params, combo)
                if not _same_value(value, GRID_PARAMS[name].read(incumbent))
            }
            trials.append(Trial(overrides))
    else:
        raise ValueError(f"Unknown design {design!r}")

    for overrides in extra_trials:
        trials.append(Trial(_canonical_overrides(overrides, incumbent), role="extra"))

    unique: dict[str, Trial] = {}
    for trial in trials:
        unique.setdefault(trial.name, trial)
    return list(unique.values())


def _canonical_overrides(overrides: Mapping[str, Any], incumbent: TrialConfig) -> dict[str, Any]:
    """Drop incumbent-valued keys and order keys as GRID_PARAMS does."""

    out: dict[str, Any] = {}
    for name in GRID_PARAMS:
        if name in overrides:
            value = GRID_PARAMS[name].kind(overrides[name])
            if not _same_value(value, GRID_PARAMS[name].read(incumbent)):
                out[name] = value
    unknown = sorted(set(overrides) - set(GRID_PARAMS))
    if unknown:
        raise KeyError(f"Unknown grid parameter(s) {unknown}; known: {list(GRID_PARAMS)}")
    return out


def parse_trial_spec(spec: str) -> dict[str, Any]:
    """'lr=3e-3,dropout=0.0' -> {'lr': 3e-3, 'dropout': 0.0}."""

    out: dict[str, Any] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Trial spec item {item!r} must look like name=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in GRID_PARAMS:
            raise argparse.ArgumentTypeError(f"Unknown grid parameter {key!r}; known: {list(GRID_PARAMS)}")
        out[key] = GRID_PARAMS[key].kind(value.strip())
    return out


def parse_grid_spec(spec: str) -> tuple[str, tuple[Any, ...]]:
    """'lr=5e-4,1e-3,3e-3' -> ('lr', (5e-4, 1e-3, 3e-3))."""

    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--grid expects name=v1,v2,...; got {spec!r}")
    key, values = spec.split("=", 1)
    key = key.strip()
    if key not in GRID_PARAMS:
        raise argparse.ArgumentTypeError(f"Unknown grid parameter {key!r}; known: {list(GRID_PARAMS)}")
    parsed = tuple(GRID_PARAMS[key].kind(v.strip()) for v in values.split(",") if v.strip())
    return key, parsed


# ---------------------------------------------------------------------
# On-disk state: claims and results
# ---------------------------------------------------------------------

@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            print(f"[lock] WARNING: could not flock {path}; continuing without a lock.", flush=True)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


class TrialStore:
    """Per-trial directories with atomic claims (flock) and JSON results."""

    def __init__(self, root: Path, *, worker_id: int, stale_claim_hours: float) -> None:
        self.root = root
        self.trials_dir = root / "trials"
        self.lock_path = root / ".claims.lock"
        self.worker_id = worker_id
        self.stale_claim_seconds = float(stale_claim_hours) * 3600.0
        self._last_heartbeat = 0.0

    def trial_dir(self, trial: Trial) -> Path:
        return self.trials_dir / trial.name

    def result_path(self, trial: Trial) -> Path:
        return self.trial_dir(trial) / RESULT_FILENAME

    def claim_path(self, trial: Trial) -> Path:
        return self.trial_dir(trial) / CLAIM_FILENAME

    def load_result(self, trial: Trial) -> dict[str, Any] | None:
        path = self.result_path(trial)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def needs_work(self, trial: Trial, *, epochs_per_trial: int) -> bool:
        """No result yet, or a complete result with fewer epochs than now
        requested whose last.pt can be extended."""

        result = self.load_result(trial)
        if result is None:
            return True
        if result.get("status") != STATUS_COMPLETE:
            return False
        if int(result.get("epochs_completed", 0)) >= int(epochs_per_trial):
            return False
        return (self.trial_dir(trial) / "last.pt").exists()

    def status(self, trial: Trial, *, epochs_per_trial: int | None = None) -> str:
        result = self.load_result(trial)
        if result is not None:
            if epochs_per_trial is not None and self.needs_work(trial, epochs_per_trial=epochs_per_trial):
                return "extendable"
            return str(result.get("status", STATUS_COMPLETE))
        claim = self.claim_path(trial)
        if claim.exists():
            return "running" if not self._is_stale(claim) else "stale-claim"
        if (self.trial_dir(trial) / "last.pt").exists():
            return "partial"
        return "pending"

    def _is_stale(self, claim: Path) -> bool:
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            return True
        return age > self.stale_claim_seconds

    def claim_next(
        self, plan: list[Trial], *, epochs_per_trial: int, only: set[str] | None = None
    ) -> Trial | None:
        """Atomically claim the first trial that needs work and has no live claim."""

        with file_lock(self.lock_path):
            for trial in plan:
                if only is not None and trial.name not in only:
                    continue
                if not self.needs_work(trial, epochs_per_trial=epochs_per_trial):
                    continue
                claim = self.claim_path(trial)
                if claim.exists() and not self._is_stale(claim):
                    continue
                self._write_claim(trial)
                return trial
        return None

    def _write_claim(self, trial: Trial) -> None:
        claim = self.claim_path(trial)
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.write_text(
            json.dumps(
                {
                    "worker_id": self.worker_id,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_array": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "claimed_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._last_heartbeat = time.time()

    def heartbeat(self, trial: Trial, *, min_interval: float = 60.0) -> None:
        now = time.time()
        if now - self._last_heartbeat < min_interval:
            return
        claim = self.claim_path(trial)
        try:
            if claim.exists():
                os.utime(claim, None)
            else:
                self._write_claim(trial)
        except OSError:
            pass
        self._last_heartbeat = now

    def release(self, trial: Trial) -> None:
        try:
            self.claim_path(trial).unlink()
        except FileNotFoundError:
            pass

    def write_result(self, trial: Trial, payload: Mapping[str, Any]) -> Path:
        path = self.result_path(trial)
        tmp = path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(runner.to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def retry_failed(self, plan: list[Trial]) -> int:
        removed = 0
        with file_lock(self.lock_path):
            for trial in plan:
                result = self.load_result(trial)
                if result is not None and result.get("status") in (STATUS_OOM, STATUS_FAILED):
                    self.result_path(trial).unlink()
                    removed += 1
        return removed

    def reset_claims(self, plan: list[Trial]) -> int:
        removed = 0
        with file_lock(self.lock_path):
            for trial in plan:
                claim = self.claim_path(trial)
                if claim.exists() and not self.result_path(trial).exists():
                    claim.unlink()
                    removed += 1
        return removed


# ---------------------------------------------------------------------
# Training one trial
# ---------------------------------------------------------------------

class TimeBudgetExceeded(RuntimeError):
    """Raised from the epoch hook when the next epoch would not fit the wall time."""


class TrialLogger:
    """Trainer logger hook: per-epoch rows, claim heartbeat, wall-time guard."""

    def __init__(
        self,
        *,
        store: TrialStore,
        trial: Trial,
        monitor: str,
        started_at: float,
        time_budget_hours: float | None,
    ) -> None:
        self.store = store
        self.trial = trial
        self.monitor = monitor
        self.started_at = started_at
        self.time_budget_seconds = (
            float(time_budget_hours) * 3600.0 if time_budget_hours is not None else None
        )
        self.log_path = store.trial_dir(trial) / EPOCH_LOG_FILENAME

    def __call__(self, payload: Mapping[str, Any]) -> None:
        kind = payload.get("type")
        self.store.heartbeat(self.trial)
        if kind == "train_step":
            print(
                f"[{self.trial.name}] epoch {payload.get('epoch')} step {payload.get('global_step')} "
                f"train/loss={payload.get('train/loss', float('nan')):.4f}",
                flush=True,
            )
            return
        if kind == "val_end":
            value = payload.get(self.monitor)
            shown = f"{float(value):.6f}" if value is not None else "n/a"
            print(
                f"[{self.trial.name}] validation: {self.monitor}={shown} "
                f"val/loss={payload.get('val/loss', float('nan')):.4f}",
                flush=True,
            )
            return
        if kind == "epoch_end":
            epoch = int(payload.get("epoch", -1)) + 1
            minutes = float(payload.get("epoch_time_min", 0.0))
            elapsed = time.time() - self.started_at
            # The trainer writes last.pt BEFORE appending this record to its
            # state history, so a resumed trial would lose its last epoch;
            # persist the full record here and rebuild the curve from it.
            record = {k: v for k, v in payload.items() if k != "type"}
            record["trial"] = self.trial.name
            record["time"] = datetime.now().isoformat(timespec="seconds")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(runner.to_jsonable(record), sort_keys=True) + "\n")
            print(
                f"[{self.trial.name}] epoch {epoch} done in {minutes:.1f} min "
                f"(worker elapsed {elapsed / 3600:.2f} h)",
                flush=True,
            )
            if (
                self.time_budget_seconds is not None
                and elapsed + minutes * 60.0 > self.time_budget_seconds
            ):
                raise TimeBudgetExceeded(
                    f"next epoch (~{minutes:.1f} min) would exceed the "
                    f"{self.time_budget_seconds / 3600:.2f} h worker budget"
                )


def build_method(incumbent_model: Mapping[str, Any], args: argparse.Namespace) -> SFTMethod:
    """The incumbent's SFT block with the tuning validation generation settings."""

    sft = dict(incumbent_model.get("sft", {}))
    topk = tuple(int(k) for k in args.val_topk)
    sft["evaluate_generation"] = True
    sft["generation_topk"] = topk
    sft["generation_num_beams"] = int(args.val_num_beams)
    sft["generation_num_return_sequences"] = max(topk)
    sft["generation_constrain_valid_sequences"] = bool(
        incumbent_model.get("constrain_valid_sequences", True)
    )
    return SFTMethod(SFTConfig.from_dict(sft))


def resolve_val_batch_size(args: argparse.Namespace, incumbent: Incumbent) -> int:
    """Keep batch x beams at the footprint the incumbent validation/test used."""

    if args.val_batch_size is not None:
        return int(args.val_batch_size)
    beams = int(args.val_num_beams)
    if beams <= 1:
        return int(incumbent.data_cfg.eval_batch_size)
    test_beams = int(incumbent.config.model.get("eval_num_beams", 20))
    return max(1, int(incumbent.test_eval_batch_size) * test_beams // beams)


def build_trainer_config(
    *,
    trial_cfg: TrialConfig,
    trial_dir: Path,
    args: argparse.Namespace,
    use_amp: bool,
) -> TrainerConfig:
    trainer = trial_cfg.trainer
    fraction = float(args.val_sample_fraction)
    return TrainerConfig(
        max_epochs=int(args.epochs_per_trial),
        gradient_accumulation_steps=int(trial_cfg.grad_accum),
        max_grad_norm=float(trainer["max_grad_norm"]),
        use_amp=use_amp,
        amp_dtype=str(trainer.get("amp_dtype", "float16")),
        log_every_steps=int(trainer.get("log_every_steps", 200)),
        progress_bar=bool(args.progress_bar),
        validate_every_epochs=1,
        checkpoint_dir=trial_dir,
        save_best=True,
        save_last=True,
        monitor=str(args.monitor),
        monitor_mode="max",
        early_stopping_patience=args.patience,
        min_delta=0.0,
        early_stopping_mode="best",
        prefix_method_metrics_with_stage=True,
        max_train_batches=args.train_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        val_sample_fraction=fraction if fraction < 1.0 else None,
        val_sample_seed=int(args.val_sample_seed),
        # A FIXED subset: identical across epochs, trials, workers and resumes.
        val_sample_resample=False,
    )


def train_trial(
    *,
    trial: Trial,
    incumbent: Incumbent,
    data: Any,
    store: TrialStore,
    args: argparse.Namespace,
    worker_started_at: float,
    device: str,
) -> dict[str, Any] | None:
    """Train one grid point; returns the result payload, or None when the
    worker's wall-time budget stopped the trial (state left resumable)."""

    trial_cfg = trial.config(incumbent.config)
    trial_dir = store.trial_dir(trial)
    trial_dir.mkdir(parents=True, exist_ok=True)
    val_batch_size = resolve_val_batch_size(args, incumbent)
    use_amp = bool(trial_cfg.trainer.get("use_amp", False)) and device.startswith("cuda")

    data.cfg = replace(
        data.cfg,
        batch_size=int(trial_cfg.batch_size),
        eval_batch_size=int(val_batch_size),
        seed=SEED,
    )

    runner.seed_everything(SEED)
    model = None
    trainer = None
    started = time.time()
    resumed_from: str | None = None
    try:
        model = tiger_all.build_model(trial_cfg.model, data)
        method = build_method(incumbent.config.model, args)
        train_cfg = build_trainer_config(
            trial_cfg=trial_cfg, trial_dir=trial_dir, args=args, use_amp=use_amp
        )
        logger = TrialLogger(
            store=store,
            trial=trial,
            monitor=str(args.monitor),
            started_at=worker_started_at,
            time_budget_hours=args.time_budget_hours,
        )
        trainer = Trainer.from_optimizer_config(
            model=model,
            method=method,
            cfg=train_cfg,
            device=device,
            **runner.optimizer_config(trial_cfg.trainer),
            logger=logger,
        )

        last_ckpt = trial_dir / "last.pt"
        epoch_log = trial_dir / EPOCH_LOG_FILENAME
        if args.resume and last_ckpt.exists():
            trainer.load_checkpoint(last_ckpt, load_optimizer=True, load_scheduler=False)
            trainer.state.epoch += 1  # last.pt holds a COMPLETED epoch
            trainer.state.should_stop = False
            resumed_from = str(last_ckpt)
            print(
                f"[{trial.name}] resuming from {last_ckpt} at epoch {trainer.state.epoch + 1}/"
                f"{args.epochs_per_trial} (best so far {trainer.state.best_metric})",
                flush=True,
            )
        else:
            # Fresh start: drop leftovers of an earlier attempt.
            for stale in (last_ckpt, last_ckpt.with_suffix(".pt.json"), epoch_log,
                          trial_dir / "best.pt", trial_dir / "best.pt.json"):
                _unlink_quietly(stale)

        num_params = int(model.num_parameters(trainable_only=True))
        print(
            f"[{trial.name}] overrides={json.dumps(trial.overrides, sort_keys=True)} "
            f"hparams={json.dumps(trial_cfg.hparams(), sort_keys=True)} "
            f"trainable_params={num_params:,} val_batch={val_batch_size} beams={args.val_num_beams}",
            flush=True,
        )
        if trainer.state.epoch >= int(args.epochs_per_trial):
            print(f"[{trial.name}] all {args.epochs_per_trial} epochs already trained; finalizing.", flush=True)
            fit_result = None
        else:
            try:
                fit_result = trainer.fit(data.train_dataloader(), data.val_dataloader())
            except TimeBudgetExceeded as exc:
                print(f"[{trial.name}] stopped by the worker time budget: {exc}. "
                      f"Progress is in {last_ckpt}; the trial resumes on the next run.", flush=True)
                return None

        history = merge_epoch_records(trainer.state.history, epoch_log)
        curve = [_curve_row(record, str(args.monitor)) for record in history if "val" in record]
        best_value = trainer.state.best_metric
        best_epoch = trainer.state.best_epoch
        if best_value is None and curve:
            best_row = max(curve, key=lambda r: r["value"] if r["value"] is not None else float("-inf"))
            best_value, best_epoch = best_row["value"], best_row["epoch"] - 1
        if best_value is None:
            raise RuntimeError(f"[{trial.name}] finished without a {args.monitor} value.")

        best_ckpt = trial_dir / "best.pt"
        if fit_result is not None and fit_result.best_checkpoint_path:
            best_ckpt = Path(fit_result.best_checkpoint_path)
        payload = {
            "trial": trial.name,
            "role": trial.role,
            "status": STATUS_COMPLETE,
            "overrides": dict(trial.overrides),
            "hparams": trial_cfg.hparams(),
            "model_config": model.config_dict(),
            "num_parameters": num_params,
            "monitor": str(args.monitor),
            "best_value": float(best_value),
            "best_epoch": int(best_epoch) + 1 if best_epoch is not None else None,
            "final_value": curve[-1]["value"] if curve else None,
            "epochs_completed": len(history),
            "epochs_planned": int(args.epochs_per_trial),
            "steps_completed": int(trainer.state.global_step),
            "curve": curve,
            "mean_epoch_min": (
                sum(float(r.get("epoch_time_min", 0.0)) for r in history) / len(history)
                if history else None
            ),
            "wall_hours_this_run": (time.time() - started) / 3600.0,
            "resumed_from": resumed_from,
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
            "last_checkpoint": str(trial_dir / "last.pt") if (trial_dir / "last.pt").exists() else None,
            "validation": {
                "num_beams": int(args.val_num_beams),
                "topk": [int(k) for k in args.val_topk],
                "sample_fraction": float(args.val_sample_fraction),
                "sample_seed": int(args.val_sample_seed),
                "batch_size": int(val_batch_size),
            },
            "seed": SEED,
            "device": device,
            "worker": {"worker_id": store.worker_id, "host": socket.gethostname(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        store.write_result(trial, payload)
        if args.drop_last_checkpoint:
            _unlink_quietly(trial_dir / "last.pt")
            _unlink_quietly(trial_dir / "last.pt.json")
        print(
            f"[{trial.name}] done: best {args.monitor}={float(best_value):.6f} at epoch "
            f"{payload['best_epoch']} ({payload['epochs_completed']}/{args.epochs_per_trial} epochs, "
            f"{payload['wall_hours_this_run']:.2f} h)",
            flush=True,
        )
        return payload

    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            print(f"[{trial.name}] CUDA out of memory; recording the trial as {STATUS_OOM}.", flush=True)
            payload = {
                "trial": trial.name,
                "role": trial.role,
                "status": STATUS_OOM,
                "overrides": dict(trial.overrides),
                "hparams": trial_cfg.hparams(),
                "monitor": str(args.monitor),
                "best_value": None,
                "error": str(exc)[:2000],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            store.write_result(trial, payload)
            return payload
        raise
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_epoch_records(history: list[dict[str, Any]], epoch_log: Path) -> list[dict[str, Any]]:
    """Per-epoch records from the trainer state plus the persisted epoch log
    (which also holds the epoch that last.pt was written before)."""

    by_epoch: dict[int, dict[str, Any]] = {}
    for record in history:
        if "epoch" in record:
            by_epoch[int(record["epoch"])] = dict(record)
    if epoch_log.exists():
        for line in epoch_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in record:
                by_epoch[int(record["epoch"])] = record
    return [by_epoch[key] for key in sorted(by_epoch)]


def _curve_row(record: Mapping[str, Any], monitor: str) -> dict[str, Any]:
    val = dict(record.get("val", {}))
    train = dict(record.get("train", {}))
    return {
        "epoch": int(record.get("epoch", -1)) + 1,
        "value": val.get(monitor),
        "val_loss": val.get("val/loss"),
        "train_loss": train.get("train/loss"),
        "epoch_time_min": record.get("epoch_time_min"),
        "val": {k: v for k, v in val.items()},
    }


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------
# Aggregation: effects, combined trial, summary files, tuned YAML
# ---------------------------------------------------------------------

def collect_results(store: TrialStore, plan: list[Trial]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for trial in plan:
        result = store.load_result(trial)
        if result is not None:
            results[trial.name] = result
    # Trials from other designs / earlier grids that share the store.
    if store.trials_dir.exists():
        for path in sorted(store.trials_dir.glob(f"*/{RESULT_FILENAME}")):
            name = path.parent.name
            if name not in results:
                try:
                    results[name] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
    return results


def _complete(results: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: r for name, r in results.items()
        if r.get("status") == STATUS_COMPLETE and r.get("best_value") is not None
    }


def parameter_effects(
    *,
    params: list[str],
    grids: Mapping[str, list[Any]],
    results: Mapping[str, dict[str, Any]],
    incumbent: TrialConfig,
) -> dict[str, dict[str, Any]]:
    """Per-parameter table from the single-deviation (OFAT) trials."""

    complete = _complete(results)
    baseline = complete.get(BASELINE_NAME)
    baseline_value = float(baseline["best_value"]) if baseline else None
    effects: dict[str, dict[str, Any]] = {}
    for name in params:
        rows = []
        for value in grids[name]:
            trial = Trial({} if _same_value(value, GRID_PARAMS[name].read(incumbent)) else {name: value})
            result = complete.get(trial.name)
            status = results.get(trial.name, {}).get("status") if trial.name in results else None
            rows.append(
                {
                    "value": value,
                    "trial": trial.name,
                    "best_value": float(result["best_value"]) if result else None,
                    "best_epoch": result.get("best_epoch") if result else None,
                    "status": status or "pending",
                    "rel_vs_baseline": (
                        (float(result["best_value"]) - baseline_value) / baseline_value
                        if result and baseline_value
                        else None
                    ),
                }
            )
        scored = [r for r in rows if r["best_value"] is not None]
        winner = max(scored, key=lambda r: r["best_value"]) if scored else None
        effects[name] = {
            "incumbent": GRID_PARAMS[name].read(incumbent),
            "rows": rows,
            "winner_value": winner["value"] if winner else None,
            "winner_trial": winner["trial"] if winner else None,
            "complete": all(
                r["best_value"] is not None or r["status"] in (STATUS_OOM, STATUS_FAILED) for r in rows
            ),
        }
    return effects


def derive_combined_trial(
    effects: Mapping[str, dict[str, Any]],
    *,
    incumbent: TrialConfig,
    min_rel_improvement: float,
) -> Trial | None:
    """Merge the per-parameter winners that beat the baseline by the margin."""

    overrides: dict[str, Any] = {}
    for name, effect in effects.items():
        winner = effect.get("winner_value")
        if winner is None or _same_value(winner, effect["incumbent"]):
            continue
        row = next(r for r in effect["rows"] if _same_value(r["value"], winner))
        rel = row.get("rel_vs_baseline")
        if rel is not None and rel >= float(min_rel_improvement):
            overrides[name] = winner
    if not overrides:
        return None
    return Trial(_canonical_overrides(overrides, incumbent), role="combined")


def rank_results(results: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    complete = _complete(results)
    return sorted(complete.values(), key=lambda r: float(r["best_value"]), reverse=True)


def aggregate(
    *,
    store: TrialStore,
    plan: list[Trial],
    params: list[str],
    grids: Mapping[str, list[Any]],
    incumbent: Incumbent,
    args: argparse.Namespace,
    write_tuned_config: bool = True,
) -> dict[str, Any]:
    """Rebuild summary JSON/CSV (and the tuned YAML) from the results on disk."""

    with file_lock(store.root / ".aggregate.lock"):
        results = collect_results(store, plan)
        effects = parameter_effects(params=params, grids=grids, results=results, incumbent=incumbent.config)
        ranking = rank_results(results)
        combined = derive_combined_trial(
            effects, incumbent=incumbent.config, min_rel_improvement=args.min_rel_improvement
        )
        baseline = _complete(results).get(BASELINE_NAME)
        baseline_value = float(baseline["best_value"]) if baseline else None
        winner = ranking[0] if ranking else None

        summary = {
            "dataset": args.dataset,
            "design": args.design,
            "params": list(params),
            "grids": {name: list(values) for name, values in grids.items()},
            "incumbent": {
                "hparams": incumbent.config.hparams(),
                "sources": incumbent.sources,
            },
            "protocol": {
                "monitor": args.monitor,
                "epochs_per_trial": int(args.epochs_per_trial),
                "seed": SEED,
                "val_num_beams": int(args.val_num_beams),
                "val_topk": [int(k) for k in args.val_topk],
                "val_sample_fraction": float(args.val_sample_fraction),
                "val_sample_seed": int(args.val_sample_seed),
                "train_batches_per_epoch": args.train_batches_per_epoch,
                "min_rel_improvement": float(args.min_rel_improvement),
            },
            "plan": [
                {"trial": t.name, "role": t.role, "overrides": t.overrides,
                 "status": store.status(t, epochs_per_trial=int(args.epochs_per_trial))}
                for t in plan
            ],
            "trials_planned": len(plan),
            "trials_complete": len(_complete(results)),
            "baseline_value": baseline_value,
            "effects": effects,
            "combined_trial": (
                {"trial": combined.name, "overrides": combined.overrides,
                 "status": store.status(combined)}
                if combined else None
            ),
            "ranking": [
                {
                    "trial": r["trial"],
                    "role": r.get("role"),
                    "best_value": r["best_value"],
                    "best_epoch": r.get("best_epoch"),
                    "final_value": r.get("final_value"),
                    "rel_vs_baseline": (
                        (float(r["best_value"]) - baseline_value) / baseline_value
                        if baseline_value else None
                    ),
                    "hparams": r.get("hparams"),
                    "num_parameters": r.get("num_parameters"),
                    "mean_epoch_min": r.get("mean_epoch_min"),
                    "epochs_completed": r.get("epochs_completed"),
                }
                for r in ranking
            ],
            "winner": ranking[0]["trial"] if ranking else None,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        runner.write_json(store.root / "tuning_summary.json", summary)
        write_summary_csv(args.artifacts_dir / f"tiger_{args.dataset}_tuning_summary.csv", summary, results)

        if winner is not None:
            winner_trial = Trial(dict(winner.get("overrides", {})), role=str(winner.get("role", "sweep")))
            winner_cfg = winner_trial.config(incumbent.config)
            best_trial = {
                "dataset": args.dataset,
                "metric": args.monitor,
                "best_value": float(winner["best_value"]),
                "best_epoch": winner.get("best_epoch"),
                "baseline_value": baseline_value,
                "rel_vs_baseline": summary["ranking"][0]["rel_vs_baseline"],
                "trial": winner["trial"],
                "overrides": winner.get("overrides", {}),
                "hparams": winner_cfg.hparams(),
                "checkpoint": winner.get("best_checkpoint"),
                "trials_complete": summary["trials_complete"],
                "trials_planned": summary["trials_planned"],
            }
            runner.write_json(store.root / "best_trial.json", best_trial)
            if write_tuned_config:
                path = write_tuned_yaml(
                    path=args.tuned_config_path,
                    winner=winner,
                    winner_cfg=winner_cfg,
                    incumbent=incumbent,
                    summary=summary,
                    args=args,
                )
                summary["tuned_config"] = str(path)
                runner.write_json(store.root / "tuning_summary.json", summary)
        print_summary(summary)
        return summary


CSV_COLUMNS = [
    "trial", "role", "status", "best_value", "best_epoch", "final_value", "rel_vs_baseline_pct",
    "lr", "weight_decay", "dropout", "embedding_dim", "dim_feedforward", "num_encoder_layers",
    "num_decoder_layers", "num_heads", "d_kv", "batch_size", "gradient_accumulation_steps",
    "max_grad_norm", "num_parameters", "epochs_completed", "epochs_planned", "mean_epoch_min",
    "steps_completed", "curve",
]


def write_summary_csv(path: Path, summary: Mapping[str, Any], results: Mapping[str, dict[str, Any]]) -> None:
    baseline_value = summary.get("baseline_value")
    rows = []
    for name, r in results.items():
        hp = dict(r.get("hparams", {}))
        best = r.get("best_value")
        rows.append(
            {
                "trial": name,
                "role": r.get("role"),
                "status": r.get("status"),
                "best_value": best,
                "best_epoch": r.get("best_epoch"),
                "final_value": r.get("final_value"),
                "rel_vs_baseline_pct": (
                    100.0 * (float(best) - baseline_value) / baseline_value
                    if best is not None and baseline_value else None
                ),
                **{key: hp.get(key) for key in CSV_COLUMNS[7:20]},
                "num_parameters": r.get("num_parameters"),
                "epochs_completed": r.get("epochs_completed"),
                "epochs_planned": r.get("epochs_planned"),
                "mean_epoch_min": r.get("mean_epoch_min"),
                "steps_completed": r.get("steps_completed"),
                "curve": ";".join(
                    f"{row['value']:.6f}" if row.get("value") is not None else "nan"
                    for row in r.get("curve", [])
                ),
            }
        )
    rows.sort(key=lambda row: (row["best_value"] is None, -(row["best_value"] or 0.0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary: Mapping[str, Any]) -> None:
    monitor = summary["protocol"]["monitor"]
    print("\n" + "=" * 100)
    print(f"TIGER grid tuning on {summary['dataset']} — {summary['trials_complete']}/"
          f"{summary['trials_planned']} planned trials complete "
          f"({summary['protocol']['epochs_per_trial']} epochs each, monitor {monitor})")
    print("=" * 100)
    print("Plan status:")
    for item in summary["plan"]:
        print(f"  {item['status']:>12}  {item['trial']}")
    if summary["ranking"]:
        print(f"\nRanking (best {monitor} within the epoch budget):")
        print(f"  {'trial':<48} {'best':>10} {'ep':>3} {'vs base':>9}  {'min/ep':>7}")
        for r in summary["ranking"]:
            rel = r.get("rel_vs_baseline")
            rel_s = f"{100 * rel:+.2f}%" if rel is not None else "n/a"
            ep_min = r.get("mean_epoch_min")
            print(f"  {r['trial']:<48} {r['best_value']:>10.6f} {str(r.get('best_epoch')):>3} "
                  f"{rel_s:>9}  {ep_min if ep_min is None else f'{ep_min:.1f}':>7}")
    print("\nPer-parameter effects (single-deviation trials):")
    for name, effect in summary["effects"].items():
        cells = []
        for row in effect["rows"]:
            mark = "*" if _same_value(row["value"], effect["incumbent"]) else ""
            val = f"{row['best_value']:.6f}" if row["best_value"] is not None else row["status"]
            cells.append(f"{fmt_value(row['value'])}{mark}: {val}")
        win = effect.get("winner_value")
        print(f"  {name:<14} " + " | ".join(cells) + (f"   -> {fmt_value(win)}" if win is not None else ""))
    combined = summary.get("combined_trial")
    if combined:
        note = ""
        if len(combined.get("overrides", {})) <= 1:
            note = "  (a single per-parameter winner: identical to that sweep trial, nothing extra to train)"
        print(f"\nCombined trial: {combined['trial']}  [{combined['status']}]{note}")
    else:
        print("\nCombined trial: none (no per-parameter winner beats the baseline by the margin yet)")
    if summary.get("winner"):
        print(f"\nWinner: {summary['winner']}")
    if summary.get("tuned_config"):
        print(f"Tuned config: {summary['tuned_config']}")
    print(flush=True)


def _yaml_scalar(value: Any) -> str:
    """YAML scalar / flow-sequence text that PyYAML re-reads with the same type."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, (int, float, str)):
        return yaml.safe_dump(value, default_flow_style=True).replace("\n...\n", "\n").strip()
    raise TypeError(f"Unsupported YAML value {value!r}")


def _yaml_lines(mapping: Mapping[str, Any], indent: int = 0) -> list[str]:
    """Block-style mapping lines with inline (flow) sequences, e.g. ``ks: [5, 10, 20]``."""

    lines: list[str] = []
    pad = " " * indent
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            lines.extend(_yaml_lines(value, indent + 2))
        else:
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    return lines


def _yaml_block(mapping: Mapping[str, Any]) -> str:
    return "\n".join(_yaml_lines(runner.to_jsonable(dict(mapping)))) + "\n"


def write_tuned_yaml(
    *,
    path: Path,
    winner: Mapping[str, Any],
    winner_cfg: TrialConfig,
    incumbent: Incumbent,
    summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    """tiger_model.yaml layout + optimizer/trainer keys (like sasrec_model_<ds>.yaml)."""

    inc = incumbent.config
    model = winner_cfg.model
    trainer = winner_cfg.trainer
    protocol = summary["protocol"]
    baseline_value = summary.get("baseline_value")
    rel = summary["ranking"][0].get("rel_vs_baseline") if summary.get("ranking") else None
    changed = dict(winner.get("overrides", {}))

    def tag(key: str, current: Any, base: Any) -> str:
        return "" if _same_value(current, base) else f"   # tuned (incumbent {fmt_value(base)})"

    arch_keys = ["embedding_dim", "num_heads", "num_encoder_layers", "num_decoder_layers",
                 "dim_feedforward", "d_kv", "dropout", "user_ids_count", "codebook_size",
                 "gradient_checkpointing"]
    arch_lines = []
    for key in arch_keys:
        if key not in model:
            continue
        arch_lines.append(_yaml_block({key: model[key]}).rstrip() + tag(key, model[key], inc.model.get(key)))
    gen_keys = ["eval_num_beams", "val_num_beams", "constrain_valid_sequences", "ks"]
    gen = {key: model[key] for key in gen_keys if key in model}
    optim = {
        "lr": float(trainer["lr"]),
        "weight_decay": float(trainer["weight_decay"]),
        "betas": [float(b) for b in trainer.get("betas", (0.9, 0.999))],
        "eps": float(trainer.get("eps", 1e-8)),
    }
    optim_lines = []
    for key, value in optim.items():
        base = inc.trainer.get(key, value)
        optim_lines.append(_yaml_block({key: value}).rstrip() + (tag(key, value, base) if key in ("lr", "weight_decay") else ""))
    train_lines = [
        _yaml_block({"max_grad_norm": float(trainer["max_grad_norm"])}).rstrip()
        + tag("max_grad_norm", float(trainer["max_grad_norm"]), float(inc.trainer["max_grad_norm"])),
        _yaml_block({"gradient_accumulation_steps": int(winner_cfg.grad_accum)}).rstrip(),
        _yaml_block({"batch_size": int(winner_cfg.batch_size)}).rstrip()
        + tag("batch_size", int(winner_cfg.batch_size), int(inc.batch_size)),
    ]
    sft = model.get("sft", {})

    header = f"""\
# TIGER hyperparameters for {args.dataset} (exp2, grid-tuned).
# Produced by scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py: {summary['design']}
# grid around the exp1 checkpoint configuration ({incumbent.sources['checkpoint']}),
# every trial trained for {protocol['epochs_per_trial']} epochs from scratch (seed {protocol['seed']})
# and scored by its best {protocol['monitor']} from constrained {protocol['val_num_beams']}-beam
# generation on a fixed {protocol['val_sample_fraction']:g} validation subsample (seed {protocol['val_sample_seed']}).
# Preprocessing comes from configs/exp2-tuned-params/dataset_yambda.yaml and was not tuned.
#
# tuned on: {date.today().isoformat()}
# trials complete: {summary['trials_complete']} (planned: {summary['trials_planned']})
# winner: {winner['trial']}  (changes vs incumbent: {json.dumps(changed, sort_keys=True) if changed else 'none'})
# best {protocol['monitor']}: {float(winner['best_value']):.6f} at epoch {winner.get('best_epoch')}
# baseline (incumbent) {protocol['monitor']}: {f'{baseline_value:.6f}' if baseline_value is not None else 'n/a'}"""
    if rel is not None:
        header += f"  ({100 * rel:+.2f}%)"
    header += f"""

# Architecture (TigerModel); sem_id_len is derived from the datamodule at
# runtime. Keys without a "tuned" tag are the incumbent values.
{chr(10).join(arch_lines)}

# Generation / decoding / evaluation cutoffs (unchanged)
{_yaml_block(gen).rstrip()}

# Optimizer (AdamW)
{chr(10).join(optim_lines)}

# Trainer settings the tuning ran with (batch_size is the train-loader batch
# from the dataset YAML tiger block; the final run keeps the exp1
# trainer_tiger.yaml schedule: max_epochs 500, patience 10 on val/sft/ndcg@1).
{chr(10).join(train_lines)}

# SFT objective and metrics (unchanged)
{_yaml_block({"sft": sft}).rstrip()}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header, encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------

def _has_claimable_work(
    *,
    store: TrialStore,
    plan: list[Trial],
    params: list[str],
    grids: Mapping[str, list[Any]],
    incumbent: Incumbent,
    args: argparse.Namespace,
    stages: list[str],
    only: set[str] | None,
) -> bool:
    epochs = int(args.epochs_per_trial)

    def claimable(trials: list[Trial]) -> bool:
        for trial in trials:
            if only is not None and trial.name not in only:
                continue
            if not store.needs_work(trial, epochs_per_trial=epochs):
                continue
            claim = store.claim_path(trial)
            if claim.exists() and not store._is_stale(claim):
                continue
            return True
        return False

    if "sweep" in stages and claimable(plan):
        return True
    if "combine" in stages:
        results = collect_results(store, plan)
        effects = parameter_effects(params=params, grids=grids, results=results, incumbent=incumbent.config)
        pending = [name for name, e in effects.items() if not e["complete"]]
        if pending and not args.allow_partial_combine:
            return False
        combined = derive_combined_trial(
            effects, incumbent=incumbent.config, min_rel_improvement=args.min_rel_improvement
        )
        if combined is not None and not any(t.name == combined.name for t in plan):
            return claimable([combined])
    return False

def run_worker(
    *,
    args: argparse.Namespace,
    incumbent: Incumbent,
    plan: list[Trial],
    params: list[str],
    grids: Mapping[str, list[Any]],
    store: TrialStore,
    worker_started_at: float,
) -> bool:
    """Train trials until none is left or the time budget is spent.

    Returns True when the last thing this worker did was aggregate (so the
    caller need not print the summary again)."""

    only = set(args.trials) if args.trials else None
    aggregated = False
    device = runner.resolve_device(args.device)
    if args.device.startswith("cuda") and device != args.device:
        print(f"[worker {args.worker_id}] WARNING: CUDA unavailable, running on {device}.", flush=True)

    stages = ["sweep", "combine"] if args.stage == "all" else [args.stage]

    # Do not pay the ~50 min / 1.3 TB datamodule setup when nothing is claimable
    # (e.g. a re-submitted array after the sweep finished).
    if not _has_claimable_work(
        store=store, plan=plan, params=params, grids=grids, incumbent=incumbent, args=args,
        stages=stages, only=only,
    ):
        print(f"[worker {args.worker_id}] nothing left to claim for stage(s) {stages}; "
              "skipping the datamodule setup.", flush=True)
        return False

    missing = [
        str(path)
        for path in (
            getattr(incumbent.data_cfg, "interactions_path", None),
            getattr(incumbent.data_cfg, "embeddings_path", None),
            getattr(incumbent.data_cfg, "semantic_id_path", None),
        )
        if path is not None and not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing input files:\n  - " + "\n  - ".join(missing))

    print(f"[worker {args.worker_id}] loading the {args.dataset} datamodule "
          f"(this is the slow, memory-heavy step) ...", flush=True)
    setup_started = time.time()
    data = incumbent.group.data_module(incumbent.data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError(f"Dataset {args.dataset!r} produced no semantic IDs.")
    print(f"[worker {args.worker_id}] datamodule ready in {(time.time() - setup_started) / 60:.1f} min; "
          f"summary: {json.dumps(runner.to_jsonable(data.summary()), sort_keys=True)}", flush=True)

    trial_hours: list[float] = []

    for stage in stages:
        while True:
            if stage == "sweep":
                candidates = plan
            else:
                results = collect_results(store, plan)
                effects = parameter_effects(params=params, grids=grids, results=results, incumbent=incumbent.config)
                pending = [name for name, e in effects.items() if not e["complete"]]
                if pending and not args.allow_partial_combine:
                    print(f"[worker {args.worker_id}] combine stage: waiting for OFAT results on "
                          f"{pending}; the last worker to finish them will run it.", flush=True)
                    break
                combined = derive_combined_trial(
                    effects, incumbent=incumbent.config, min_rel_improvement=args.min_rel_improvement
                )
                if combined is None:
                    print(f"[worker {args.worker_id}] combine stage: no per-parameter winner beats the "
                          f"baseline by {100 * args.min_rel_improvement:.1f}%; nothing to combine.", flush=True)
                    break
                if store.load_result(combined) is not None or any(t.name == combined.name for t in plan):
                    print(f"[worker {args.worker_id}] combine stage: {combined.name} already in the plan/results.", flush=True)
                    break
                candidates = [combined]

            elapsed_h = (time.time() - worker_started_at) / 3600.0
            estimate_h = max(trial_hours) if trial_hours else float(args.trial_hours_estimate)
            if args.time_budget_hours is not None and elapsed_h + estimate_h > float(args.time_budget_hours):
                print(f"[worker {args.worker_id}] {elapsed_h:.2f} h elapsed + ~{estimate_h:.2f} h per trial "
                      f"exceeds the {args.time_budget_hours} h budget; not claiming another trial.", flush=True)
                return aggregated

            trial = store.claim_next(candidates, epochs_per_trial=int(args.epochs_per_trial), only=only)
            if trial is None:
                print(f"[worker {args.worker_id}] no unclaimed {stage} trial left.", flush=True)
                break
            print(f"\n{'-' * 100}\n[worker {args.worker_id}] claimed {stage} trial: {trial.name}\n{'-' * 100}", flush=True)
            started = time.time()
            try:
                payload = train_trial(
                    trial=trial, incumbent=incumbent, data=data, store=store, args=args,
                    worker_started_at=worker_started_at, device=device,
                )
            except Exception:
                traceback.print_exc()
                store.write_result(trial, {
                    "trial": trial.name, "role": trial.role, "status": STATUS_FAILED,
                    "overrides": dict(trial.overrides), "monitor": str(args.monitor),
                    "best_value": None, "error": traceback.format_exc()[-4000:],
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                })
                store.release(trial)
                if args.fail_fast:
                    raise
                aggregated = False
                continue
            store.release(trial)
            if payload is None:
                # Time budget hit mid-trial: state is resumable, stop this worker.
                aggregate(store=store, plan=plan, params=params, grids=grids, incumbent=incumbent, args=args)
                return True
            trial_hours.append((time.time() - started) / 3600.0)
            aggregate(store=store, plan=plan, params=params, grids=grids, incumbent=incumbent, args=args)
            aggregated = True

    del data
    gc.collect()
    return aggregated


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Grid tuning of TIGER (SFT) on Yambda 500M around the exp1 checkpoint "
            "configuration: OFAT (or full) grid, equal short epoch budget per "
            "trial, cooperative workers, tuned YAML output."
        )
    )
    # Plan
    p.add_argument("--design", choices=("ofat", "full"), default="ofat",
                   help="ofat: baseline + single-parameter deviations (default); full: cartesian product of --params")
    p.add_argument("--params", nargs="+", default=list(DEFAULT_PARAMS), choices=list(GRID_PARAMS),
                   help=f"Grid axes to sweep, in plan order (default: {' '.join(DEFAULT_PARAMS)})")
    p.add_argument("--grid", action="append", default=[], type=parse_grid_spec, metavar="NAME=V1,V2",
                   help="Override the alternatives of one axis, e.g. --grid lr=5e-4,1e-3,3e-3 (repeatable)")
    p.add_argument("--extra-trial", action="append", default=[], type=parse_trial_spec, metavar="NAME=V,NAME=V",
                   help="Add an explicit grid point, e.g. --extra-trial lr=3e-3,dropout=0.0 (repeatable)")
    p.add_argument("--trials", nargs="+", default=None,
                   help="Only run these trial names (see --list-plan); default: every unclaimed trial")
    p.add_argument("--stage", choices=("all", "sweep", "combine", "report"), default="all",
                   help="all: sweep, then the combined trial once the OFAT results are in; report: aggregate only (no data, no GPU)")
    p.add_argument("--min-rel-improvement", type=float, default=DEFAULT_MIN_REL_IMPROVEMENT,
                   help="Relative gain over the baseline a per-parameter winner needs to enter the combined trial")
    p.add_argument("--allow-partial-combine", action="store_true",
                   help="Derive the combined trial from the OFAT results available so far")
    p.add_argument("--list-plan", action="store_true", help="Print the plan and per-trial status, then exit")
    p.add_argument("--reset-claims", action="store_true",
                   help="Delete claims of unfinished trials before starting (use after killed jobs)")
    p.add_argument("--retry-failed", action="store_true",
                   help="Forget failed/OOM results so those trials are trained again")

    # Budget per trial
    p.add_argument("--epochs-per-trial", type=int, default=DEFAULT_EPOCHS_PER_TRIAL)
    p.add_argument("--patience", type=int, default=None,
                   help="Optional early stopping inside a trial (validations without improvement); default: none")
    p.add_argument("--train-batches-per-epoch", type=int, default=None,
                   help="Cap the training batches per epoch (sub-epoch budgets / smoke tests)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="Debug cap for validation batches")

    # Validation / selection metric
    p.add_argument("--monitor", default=DEFAULT_MONITOR,
                   help="Selection metric (default val/sft/ndcg@10; the checkpoint's own regime is "
                        "--monitor val/sft/ndcg@1 --val-num-beams 1 --val-topk 1 --val-sample-fraction 1.0)")
    p.add_argument("--val-num-beams", type=int, default=DEFAULT_VAL_NUM_BEAMS)
    p.add_argument("--val-topk", type=int, nargs="+", default=list(DEFAULT_VAL_TOPK))
    p.add_argument("--val-sample-fraction", type=float, default=DEFAULT_VAL_SAMPLE_FRACTION,
                   help="Fixed validation subsample used by every trial (1.0 = full split)")
    p.add_argument("--val-sample-seed", type=int, default=0)
    p.add_argument("--val-batch-size", type=int, default=None,
                   help="Validation batch; default keeps batch x beams at the incumbent's footprint")

    # Worker / cluster
    p.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0))
    p.add_argument("--time-budget-hours", type=float, default=23.5,
                   help="Wall-time budget of this worker incl. datamodule setup; trials that would not fit are not started, "
                        "and a running trial stops after the last epoch that fits (resumable). Use a negative value to disable.")
    p.add_argument("--trial-hours-estimate", type=float, default=4.5,
                   help="Assumed trial duration before this worker has measured one")
    p.add_argument("--stale-claim-hours", type=float, default=1.0,
                   help="A claim without heartbeat for this long is treated as dead and re-claimed")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Do not resume a partial trial from its last.pt")
    p.add_argument("--drop-last-checkpoint", action="store_true",
                   help="Delete last.pt when a trial completes (default keeps it, so a later run with a larger "
                        "--epochs-per-trial can extend finished trials instead of retraining them)")
    p.add_argument("--fail-fast", action="store_true", help="Re-raise unexpected trial errors instead of recording them")
    p.add_argument("--progress-bar", action="store_true")

    # Assets and paths
    p.add_argument("--dataset", default=DATASET, help="Dataset key of the dataset YAML (default yambda_500m_listens)")
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG,
                   help="Incumbent TIGER model YAML (default: exp1 tiger_model.yaml = the checkpoint's)")
    p.add_argument("--trainer-config", type=Path, default=DEFAULT_TRAINER_CONFIG,
                   help="Incumbent trainer YAML (default: exp1 trainer_tiger.yaml)")
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG,
                   help="Dataset YAML with the tiger: block (default: exp2 dataset_yambda.yaml)")
    p.add_argument("--semantic-ids-dir", type=Path, default=DEFAULT_SEMANTIC_IDS_DIR)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                   help="Tuning state lives under <checkpoint-root>/<dataset>/")
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--tuned-config-path", type=Path, default=None,
                   help="Default configs/exp2-tuned-params/tiger_model_<dataset>.yaml")
    p.add_argument("--num-workers", type=int, default=None, help="Dataloader workers override")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    for path_attr in ("model_config", "trainer_config", "dataset_config", "semantic_ids_dir",
                      "checkpoint_root", "artifacts_dir"):
        setattr(args, path_attr, runner.resolve_path(getattr(args, path_attr), base=PROJECT_ROOT))
    if args.tuned_config_path is None:
        args.tuned_config_path = EXP2_CONFIG_DIR / f"tiger_model_{runner.safe_component(args.dataset)}.yaml"
    else:
        args.tuned_config_path = runner.resolve_path(args.tuned_config_path, base=PROJECT_ROOT)
    if args.time_budget_hours is not None and args.time_budget_hours < 0:
        args.time_budget_hours = None
    if args.epochs_per_trial < 1:
        p.error("--epochs-per-trial must be >= 1")
    if not 0.0 < args.val_sample_fraction <= 1.0:
        p.error("--val-sample-fraction must be in (0, 1]")
    if max(args.val_topk) > args.val_num_beams:
        p.error("--val-num-beams must be >= max(--val-topk)")
    if "loss" in args.monitor:
        p.error("only higher-is-better monitors (ndcg/recall/hit-rate) are supported; got --monitor " + args.monitor)
    monitor_k = args.monitor.rsplit("@", 1)[-1]
    if monitor_k.isdigit() and int(monitor_k) not in args.val_topk:
        p.error(f"--monitor {args.monitor} needs {monitor_k} in --val-topk {args.val_topk}")
    seen = set()
    args.params = [x for x in args.params if not (x in seen or seen.add(x))]
    args.grid_overrides = {name: values for name, values in args.grid}
    return args


def main() -> None:
    worker_started_at = time.time()
    args = parse_args()
    runner.seed_everything(SEED)

    incumbent = load_incumbent(args)
    grids = resolve_grids(args.params, incumbent.config, args.grid_overrides)
    plan = build_plan(
        design=args.design, params=args.params, grids=grids,
        incumbent=incumbent.config, extra_trials=args.extra_trial,
    )
    root = args.checkpoint_root / runner.safe_component(args.dataset)
    root.mkdir(parents=True, exist_ok=True)
    store = TrialStore(root, worker_id=args.worker_id, stale_claim_hours=args.stale_claim_hours)
    runner.write_json(
        root / "plan.json",
        {
            "dataset": args.dataset,
            "design": args.design,
            "params": args.params,
            "grids": grids,
            "incumbent": {"hparams": incumbent.config.hparams(), "sources": incumbent.sources},
            "trials": [
                {"trial": t.name, "role": t.role, "overrides": t.overrides,
                 "hparams": t.config(incumbent.config).hparams()}
                for t in plan
            ],
            "protocol": {
                "monitor": args.monitor, "epochs_per_trial": args.epochs_per_trial, "seed": SEED,
                "val_num_beams": args.val_num_beams, "val_topk": args.val_topk,
                "val_sample_fraction": args.val_sample_fraction, "val_sample_seed": args.val_sample_seed,
            },
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("=" * 100)
    print(f"TIGER grid tuning on {args.dataset} ({args.design}; worker {args.worker_id})")
    print("=" * 100)
    print(f"Incumbent:       {json.dumps(incumbent.config.hparams(), sort_keys=True)}")
    print(f"  from:          {incumbent.sources['model_config']}, {incumbent.sources['trainer_config']}")
    print(f"Grids:           {json.dumps(grids)}")
    print(f"Budget:          {args.epochs_per_trial} epochs/trial, monitor {args.monitor} "
          f"({args.val_num_beams} beams, val subsample {args.val_sample_fraction:g})")
    print(f"State dir:       {root}")
    print(f"Plan ({len(plan)} trials):")
    for trial in plan:
        status = store.status(trial, epochs_per_trial=int(args.epochs_per_trial))
        print(f"  {status:>12}  {trial.name:<48} {json.dumps(trial.overrides, sort_keys=True)}")
    print(flush=True)

    if args.reset_claims:
        removed = store.reset_claims(plan)
        print(f"Removed {removed} unfinished claim(s).", flush=True)
    if args.retry_failed:
        removed = store.retry_failed(plan)
        print(f"Forgot {removed} failed/OOM result(s); those trials will be re-trained.", flush=True)
    if args.list_plan:
        return
    if args.stage == "report":
        aggregate(store=store, plan=plan, params=args.params, grids=grids, incumbent=incumbent, args=args)
        return

    aggregated = run_worker(
        args=args, incumbent=incumbent, plan=plan, params=args.params, grids=grids,
        store=store, worker_started_at=worker_started_at,
    )
    if not aggregated:
        aggregate(store=store, plan=plan, params=args.params, grids=grids, incumbent=incumbent, args=args)
    print(f"[worker {args.worker_id}] finished after {(time.time() - worker_started_at) / 3600:.2f} h.", flush=True)


if __name__ == "__main__":
    main()
