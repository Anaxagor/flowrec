"""Optuna hyperparameter tuning of GPT-2 (SFT) for the exp2 (tuned-params) runs.

Tunes the decoder-only GPT-2 recommender independently on the exp2 datasets
-- Amazon Beauty 2014, Amazon Instruments 2018, Amazon Beauty 2023, Yambda
50M listens, and Yambda 500M listens -- by maximizing
``val/sft/generation_ndcg@10`` (constrained 20-beam generation, i.e. the
metric the final evaluation reports) on the leave-one-out validation split.

The logic mirrors ``tuning_sasrec.py``: TPE sampler + median pruner, an
NFS-safe journal shared by cooperating workers, promotion of the incumbent
best model/config as soon as a trial improves the study, the exp1 baseline
enqueued as the first trial of every study (so the search always contains
the configuration it is trying to beat), and a ``--rerank`` stage that
retrains the top configurations with three full seeds each.

What is tuned
-------------
Architecture and optimizer only::

    embedding_dim   GPT-2 n_embd
    num_heads       n_head (every choice divides every embedding_dim)
    num_layers      n_layer
    ff_mult         dim_feedforward = ff_mult * embedding_dim
    dropout         one value fanned out to attention/residual/embedding
                    dropout, as in the tuned reference configs exp1 adopted
    lr              AdamW step size
    weight_decay    AdamW decoupled weight decay

Everything else is the exp1 GPT-2 regime and is NOT tuned. The datamodule is
built from the dataset YAMLs in ``configs/exp2-tuned-params`` through the
exact config-merging code path of ``run_gpt2_sft_all.py`` (dense_causal
windows, right padding, the ``gpt2:`` batch sizes -- Amazon 256, Yambda 512
-- with gradient accumulation 1), so tuned hyperparameters remain valid for
the final multi-seed runs. Semantic IDs come from
``datasets/semantic_ids/exp2-tuned-params/<dataset>/semantic_ids.json`` (the
exp2 tokenizer output the exp2 TIGER runs use; ``--semantic-ids-dir``
overrides). ``user_ids_count`` stays 1; betas, eps, max_grad_norm,
min_delta, and bf16 AMP come from ``configs/exp1-standard_params/
trainer_gpt2.yaml``; validation decodes with ``val_num_beams`` from
``gpt2_model.yaml``. The exp1 per-dataset configuration (``gpt2_model.yaml``
profile + the shared ``weight_decay``) is read at startup, enqueued as trial
0 of every study, and every search space is checked to contain it.

The large datasets monitor a FIXED validation subsample (Yambda 10 % as in
the exp1 profiles, Beauty 2023 25 %) because every-epoch 20-beam generation
over the full split would dominate the trial cost; the final exp2 runs still
evaluate the full splits.

Outputs
-------
Per dataset under ``checkpoints/exp2-tuned_params/gpt2/<dataset>/``:

- ``optuna_journal.log``                 resumable study storage (re-running
                                         continues until the planned trial
                                         count is reached)
- ``best_model.pt``                      best-epoch weights of the incumbent
- ``best_trial.json``                    incumbent metadata (params, value)
- ``epoch_val_generation_ndcg10.jsonl``  per-epoch monitor values, all trials
- ``all_trials.json``                    full study dump after the study ends
- ``rerank_summary.json``                the --rerank ranking

and the winning configuration in
``configs/exp2-tuned-params/gpt2_model_<dataset>.yaml`` -- the
``gpt2_model.yaml`` layout with a single ``datasets:`` profile (model /
optimizer / training blocks), so the exp2 GPT-2 runner can consume it the
way ``run_gpt2_sft_all.py`` consumes the exp1 file.

Trial budget and the two array modes
------------------------------------
The trial budget lives in the shared study, not in the process: every
worker started with the same ``--datasets`` value pulls trials from the same
journal until the study holds the planned number of finished trials. The
same command line therefore serves both Slurm array shapes:

- datasets as the array dimension, one worker per dataset
  (``scripts/slurm/tuning_gpt2_array.sbatch``: task i -> dataset i);
- workers as the array dimension for one dataset, trials in parallel
  (``scripts/slurm/tuning_gpt2_workers_array.sbatch``: task i ->
  ``--worker-id i``, dataset from ``GPT2_TUNE_DATASET``).

Planned budgets (TUNING_SPECS): beauty / instruments 100 trials,
beauty_2023 60, yambda_50m_listens 40, yambda_500m_listens 20. Studies
resume after a time-out; workers are stateless. The journal storage has no
heartbeat, so a trial whose worker was killed by the time limit stays
RUNNING forever: pass ``--fail-stale-trials`` on the resubmission (only when
no worker of that study is alive) to mark such trials FAIL. The exp1
baseline is re-enqueued until a finished trial holds it, so a killed
baseline attempt is retried.

Usage::

    python scripts/exp2-tuned-params/tuning_gpt2.py                     # all five
    python scripts/exp2-tuned-params/tuning_gpt2.py --datasets beauty
    # one worker of a parallel study (repeat with different --worker-id):
    python scripts/exp2-tuned-params/tuning_gpt2.py \
        --datasets yambda_50m_listens --worker-id 3
    # after the search finished: 3-seed re-rank of the top configs
    # (single job per dataset, not a worker array):
    python scripts/exp2-tuned-params/tuning_gpt2.py --datasets instruments --rerank
    # smoke test that touches neither the real studies nor configs/:
    python scripts/exp2-tuned-params/tuning_gpt2.py --datasets beauty \
        --n-trials 2 --max-epochs 1 --debug-max-batches 2 \
        --output-root /tmp/gpt2_tune_smoke
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
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping

import optuna
import torch
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.training.trainer import Trainer, TrainerConfig
from scripts import sasrec_config_runner as runner


def _load_run_all_module() -> Any:
    """Import run_gpt2_sft_all.py (hyphenated directory, so load by path)."""

    path = (
        Path(__file__).resolve().parents[1]
        / "exp1-standard-params"
        / "run_gpt2_sft_all.py"
    )
    spec = importlib.util.spec_from_file_location("exp1_run_gpt2_sft_all", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_all = _load_run_all_module()


# ---------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
EXP1_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "exp2-tuned_params" / "gpt2"
DEFAULT_SEMANTIC_IDS_DIR = (
    PROJECT_ROOT / "datasets" / "semantic_ids" / "exp2-tuned-params"
)

DATASETS_TO_TUNE = (
    "beauty",
    "instruments",
    "beauty_2023",
    "yambda_50m_listens",
    "yambda_500m_listens",
)

SEED = 42

# Seeds for the --rerank stage: the top search configs are retrained on all
# of these and ranked by their mean validation generation NDCG@10.
RERANK_SEEDS = (42, 43, 44)

# The in-training validation decodes with beam search, so the generation
# metric is monitored (val/sft/ndcg@k would be the teacher-forced token
# score, which is not what the final evaluation reports).
MONITOR = "val/sft/generation_ndcg@10"
MONITOR_K = 10
REQUIRED_TRAIN_LAYOUT = "dense_causal"
LOG_EVERY_STEPS = 200
# Fixed validation subsamples never resample: every trial scores the same
# users.
VAL_SAMPLE_SEED = 0

SEARCH_PARAMS = (
    "embedding_dim",
    "num_heads",
    "num_layers",
    "ff_mult",
    "dropout",
    "lr",
    "weight_decay",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROGRESS_BAR = True

EPOCH_LOG_FILENAME = "epoch_val_generation_ndcg10.jsonl"


# ---------------------------------------------------------------------
# Per-dataset search spaces
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TuningSpec:
    """Search space and trial budget for one dataset."""

    embedding_dim: tuple[int, ...]
    num_heads: tuple[int, ...]        # every choice must divide every embedding_dim
    num_layers: tuple[int, int]       # inclusive range
    ff_mult: tuple[int, ...]          # dim_feedforward = ff_mult * embedding_dim
    dropout: tuple[float, float]      # one value for attn/resid/embd pdrop
    lr: tuple[float, float]           # log-uniform
    weight_decay: tuple[float, float]  # log-uniform
    n_trials: int
    n_startup_trials: int
    max_epochs: int
    patience: int
    pruner_warmup_epochs: int
    # Fixed validation subsample (fraction of val users) for the monitor;
    # None = the full split, as in the exp1 Amazon profiles.
    val_sample_fraction: float | None
    notes: str


# Ranges follow the GPT-2-as-recommender experience behind the exp1
# profiles: on the small 5-core Amazon sets the externally tuned Beauty
# config landed at a 256-wide, 3-block model with moderate dropout and an lr
# below 3e-4, so those spaces stay centred there but reach down to 64-wide
# models (token-level GPT-2 needs more width than SASRec's 32-128 because it
# supervises ~1k semantic tokens, 4 per item) and up to heavier dropout;
# the large catalogs reward width/depth with little dropout and a larger lr,
# as the tuned Yambda profile (512 x 6, dropout 0.2, lr 1.2e-3) shows.
# Weight decay spans the AdamW-for-transformers range 1e-4..1e-1 everywhere
# (exp1 used a flat 1e-2).
TUNING_SPECS: dict[str, TuningSpec] = {
    # Beauty 2014, 5-core: ~22k users / ~12k items, mean history ~9, so
    # 80-token dense windows are mostly padding and models overfit within a
    # few tens of epochs. Small-to-medium widths, shallow stacks, and a wide
    # dropout range; the lr floor is low because the tuned incumbent already
    # sits at 2.8e-4.
    "beauty": TuningSpec(
        embedding_dim=(64, 128, 256, 384),
        num_heads=(2, 4, 8),
        num_layers=(1, 4),
        ff_mult=(2, 3, 4),
        dropout=(0.05, 0.4),
        lr=(5e-5, 2e-3),
        weight_decay=(1e-4, 1e-1),
        n_trials=100,
        n_startup_trials=15,
        max_epochs=300,
        patience=10,
        pruner_warmup_epochs=8,
        val_sample_fraction=None,
        notes="small sparse 5-core Amazon; small/medium width, wide dropout range",
    ),
    # Instruments 2018, 5-core: the smallest set (~25k users / ~10k items),
    # even shorter sequences. Same regime as Beauty; the exp1 profile is an
    # untuned interpolation (256 x 4, dropout 0.1, lr 3e-4), so the space
    # deliberately covers both shallower and more regularized models.
    "instruments": TuningSpec(
        embedding_dim=(64, 128, 256, 384),
        num_heads=(2, 4, 8),
        num_layers=(1, 4),
        ff_mult=(2, 3, 4),
        dropout=(0.05, 0.4),
        lr=(5e-5, 2e-3),
        weight_decay=(1e-4, 1e-1),
        n_trials=100,
        n_startup_trials=15,
        max_epochs=300,
        patience=10,
        pruner_warmup_epochs=8,
        val_sample_fraction=None,
        notes="smallest sparse 5-core Amazon; shallow models, wide dropout range",
    ),
    # Beauty 2023 (verified purchases): one to two orders of magnitude more
    # users and a ~100k+ item catalog behind ~1k semantic tokens. The extra
    # data supports wider/deeper stacks with only light dropout; the exp1
    # profile is again an untuned interpolation. Full-split 20-beam
    # validation every epoch is too slow at this size, so the monitor uses a
    # fixed 25 % of the validation users.
    "beauty_2023": TuningSpec(
        embedding_dim=(128, 256, 512),
        num_heads=(4, 8),
        num_layers=(2, 6),
        ff_mult=(2, 3, 4),
        dropout=(0.0, 0.3),
        lr=(1e-4, 3e-3),
        weight_decay=(1e-4, 1e-1),
        n_trials=60,
        n_startup_trials=10,
        max_epochs=200,
        patience=8,
        pruner_warmup_epochs=5,
        val_sample_fraction=0.25,
        notes="large sparse Amazon 2023; medium/large width, light dropout",
    ),
    # Yambda 50M listens: dense music log, 200-token windows (history 50 x
    # sid_len 4), ~355k items. Dense implicit feedback favors wide and deep
    # models with little dropout and a larger lr (the tuned profile: 512 x 6,
    # dropout 0.2, lr 1.2e-3). Epochs are expensive, so the epoch cap and
    # patience are short and the pruner starts early; the monitor uses the
    # exp1 profile's fixed 10 % validation sample.
    "yambda_50m_listens": TuningSpec(
        embedding_dim=(256, 512, 768),
        num_heads=(4, 8),
        num_layers=(4, 8),
        ff_mult=(2, 3, 4),
        dropout=(0.0, 0.3),
        lr=(2e-4, 3e-3),
        weight_decay=(1e-4, 1e-1),
        n_trials=40,
        n_startup_trials=8,
        max_epochs=60,
        patience=5,
        pruner_warmup_epochs=3,
        val_sample_fraction=0.1,
        notes="dense large-scale music log; wide/deep models, light dropout",
    ),
    # Yambda 500M listens: ten times the data of 50M, so a single trial is a
    # multi-day training. The space is a tight neighbourhood of the tuned
    # profile (which exp1 shares between the two Yambda sizes) and the budget
    # is small; run it as a worker array and expect it to need several
    # resubmissions. The 50M winner is the cheaper prior if this study cannot
    # be afforded.
    "yambda_500m_listens": TuningSpec(
        embedding_dim=(256, 512),
        num_heads=(4, 8),
        num_layers=(4, 6),
        ff_mult=(2, 4),
        dropout=(0.0, 0.3),
        lr=(3e-4, 3e-3),
        weight_decay=(1e-3, 1e-1),
        n_trials=20,
        n_startup_trials=5,
        max_epochs=30,
        patience=4,
        pruner_warmup_epochs=2,
        val_sample_fraction=0.1,
        notes="largest dense log; tight neighbourhood of the tuned profile, small budget",
    ),
}

TUNED_CONFIG_DATA_YAML = {
    "beauty": "dataset_amazon.yaml",
    "instruments": "dataset_amazon.yaml",
    "beauty_2023": "dataset_amazon_2023.yaml",
    "yambda_50m_listens": "dataset_yambda.yaml",
    "yambda_500m_listens": "dataset_yambda.yaml",
}


# ---------------------------------------------------------------------
# Fixed regime (everything the search does not tune) from the exp1 configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FixedRegime:
    """The exp1 GPT-2 settings the search holds fixed, plus the per-dataset
    exp1 baselines expressed in search-space coordinates."""

    model_config: dict[str, Any]       # configs/exp1-standard_params/gpt2_model.yaml
    grad_accum: int
    max_grad_norm: float | None
    betas: tuple[float, float]
    eps: float
    min_delta: float
    use_amp: bool
    amp_dtype: str
    val_num_beams: int
    constrain_valid_sequences: bool
    user_ids_count: int
    baselines: dict[str, dict[str, Any]]   # dataset -> SEARCH_PARAMS values
    codebook_sizes: dict[str, int]         # dataset -> profile codebook_size


def load_regime() -> FixedRegime:
    model_config = runner.load_yaml(EXP1_CONFIG_DIR / run_all.MODEL_CONFIG_NAME)
    trainer_config = run_all.apply_trainer_defaults(
        runner.load_yaml(EXP1_CONFIG_DIR / run_all.TRAINER_CONFIG_NAME)
    )
    # Fails loudly if the exp1 defaults ever drift away from ks=(5, 10, 20).
    run_all.resolve_ranking_ks(model_config)

    betas = trainer_config.get("betas", (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"Optimizer betas must contain two values; got {betas!r}.")

    amp_dtype = str(trainer_config.get("amp_dtype", "bfloat16"))
    use_amp = bool(trainer_config.get("use_amp", True)) and torch.cuda.is_available()
    if (
        use_amp
        and amp_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = "float16"

    shared_weight_decay = float(trainer_config.get("weight_decay", 1e-2))
    baselines: dict[str, dict[str, Any]] = {}
    codebook_sizes: dict[str, int] = {}
    for dataset_name in DATASETS_TO_TUNE:
        baseline, codebook_size = exp1_baseline(
            model_config, dataset_name, weight_decay=shared_weight_decay
        )
        baselines[dataset_name] = baseline
        codebook_sizes[dataset_name] = codebook_size

    max_grad_norm = trainer_config.get("max_grad_norm", 1.0)
    return FixedRegime(
        model_config=dict(model_config),
        grad_accum=int(trainer_config.get("gradient_accumulation_steps", 1)),
        max_grad_norm=None if max_grad_norm is None else float(max_grad_norm),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(trainer_config.get("eps", 1e-8)),
        min_delta=float(trainer_config.get("min_delta", 0.0)),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        val_num_beams=int(model_config.get("val_num_beams", 20)),
        constrain_valid_sequences=bool(
            model_config.get("constrain_valid_sequences", True)
        ),
        user_ids_count=run_all.user_ids_count(model_config, {}),
        baselines=baselines,
        codebook_sizes=codebook_sizes,
    )


def exp1_baseline(
    model_config: Mapping[str, Any], dataset_name: str, *, weight_decay: float
) -> tuple[dict[str, Any], int]:
    """The exp1 profile of ``dataset_name`` in search-space coordinates."""

    profile = run_all.model_profile(model_config, dataset_name)
    model_defaults = dict(model_config.get("defaults", {}).get("model", {}))
    values = {**model_defaults, **dict(profile.get("model", {}))}

    embedding_dim = int(values["embedding_dim"])
    dim_feedforward = int(values["dim_feedforward"])
    if dim_feedforward % embedding_dim:
        raise ValueError(
            f"exp1 gpt2_model.yaml profile {dataset_name!r} has "
            f"dim_feedforward={dim_feedforward} that is not a multiple of "
            f"embedding_dim={embedding_dim}; the search parameterizes the FFN "
            "width as ff_mult * embedding_dim."
        )
    dropout = float(values["dropout"])
    for key in ("attention_dropout", "residual_dropout", "embedding_dropout"):
        if abs(float(values.get(key, dropout)) - dropout) > 1e-12:
            raise ValueError(
                f"exp1 gpt2_model.yaml profile {dataset_name!r} sets {key}="
                f"{values[key]} != dropout={dropout}; the search uses a single "
                "dropout value, so the baseline cannot be represented."
            )
    optimizer = dict(profile.get("optimizer", {}))
    if "lr" not in optimizer:
        raise KeyError(
            f"exp1 gpt2_model.yaml profile {dataset_name!r} defines no optimizer.lr."
        )
    baseline = {
        "embedding_dim": embedding_dim,
        "num_heads": int(values["num_heads"]),
        "num_layers": int(values["num_layers"]),
        "ff_mult": dim_feedforward // embedding_dim,
        "dropout": dropout,
        "lr": float(optimizer["lr"]),
        "weight_decay": float(optimizer.get("weight_decay", weight_decay)),
    }
    return baseline, int(values.get("codebook_size", 256))


def validate_specs(regime: FixedRegime) -> None:
    """Every search space must be well-formed and contain its exp1 incumbent."""

    for name, spec in TUNING_SPECS.items():
        for heads in spec.num_heads:
            bad = [d for d in spec.embedding_dim if d % heads]
            if bad:
                raise ValueError(
                    f"TUNING_SPECS[{name!r}]: num_heads={heads} does not divide "
                    f"embedding_dim choices {bad}."
                )
        base = regime.baselines[name]
        contained = (
            base["embedding_dim"] in spec.embedding_dim
            and base["num_heads"] in spec.num_heads
            and spec.num_layers[0] <= base["num_layers"] <= spec.num_layers[1]
            and base["ff_mult"] in spec.ff_mult
            and spec.dropout[0] <= base["dropout"] <= spec.dropout[1]
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
        "embedding_dim": trial.suggest_categorical("embedding_dim", list(spec.embedding_dim)),
        "num_heads": trial.suggest_categorical("num_heads", list(spec.num_heads)),
        "num_layers": trial.suggest_int("num_layers", *spec.num_layers),
        "ff_mult": trial.suggest_categorical("ff_mult", list(spec.ff_mult)),
        "dropout": trial.suggest_float("dropout", *spec.dropout),
        "lr": trial.suggest_float("lr", *spec.lr, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *spec.weight_decay, log=True),
    }


# ---------------------------------------------------------------------
# Run context (paths and debug overrides shared by every stage)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    regime: FixedRegime
    semantic_ids_dir: Path
    output_root: Path
    config_out_dir: Path
    n_trials_override: int | None = None
    max_epochs_override: int | None = None
    debug_max_batches: int | None = None
    config_dir: Path = CONFIG_DIR
    required_split_strategy: str | None = None
    device: str = DEVICE
    progress_bar: bool = PROGRESS_BAR
    num_workers: int | None = None

    def n_trials(self, spec: TuningSpec) -> int:
        return int(self.n_trials_override or spec.n_trials)

    def max_epochs(self, spec: TuningSpec) -> int:
        return int(self.max_epochs_override or spec.max_epochs)

    def dataset_dir(self, dataset_name: str) -> Path:
        return self.output_root / runner.safe_component(dataset_name)

    def tuned_config_path(self, dataset_name: str) -> Path:
        return self.config_out_dir / f"gpt2_model_{dataset_name}.yaml"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def find_group(dataset_name: str, *, config_dir: Path = CONFIG_DIR) -> tuple[Any, dict[str, Any]]:
    """Locate the run_gpt2_sft_all dataset group that owns ``dataset_name``."""

    filename = TUNED_CONFIG_DATA_YAML[dataset_name]
    group = next(group for group in run_all.GROUPS if group.data_config_name == filename)
    dataset_config = runner.load_yaml(config_dir / filename)
    if dataset_name in dataset_config.get("datasets", {}):
        return group, dataset_config
    raise KeyError(
        f"Dataset {dataset_name!r} is not defined in any dataset YAML under "
        f"{config_dir}."
    )


def resolve_data_config(dataset_name: str, ctx: RunContext) -> tuple[Any, Any]:
    group, dataset_config = find_group(dataset_name, config_dir=ctx.config_dir)
    entry = dataset_config["datasets"][dataset_name]
    if "gpt2" not in dataset_config and "gpt2" not in entry:
        # The runner only warns and falls back to the SASRec-era loader
        # values; a silent batch-size change would invalidate the tuning.
        raise KeyError(
            f"{ctx.config_dir / group.data_config_name} defines no gpt2: batching "
            f"block for {dataset_name!r}; add one (see the exp1 dataset YAMLs)."
        )

    # The split seed is fixed for the whole study so every trial sees the
    # identical train/val partition; only trial seeds vary.
    data_cfg, _test_eval_batch_size = run_all.build_data_config(
        group,
        dataset_name,
        dataset_config,
        seed=SEED,
        semantic_ids_dir=ctx.semantic_ids_dir,
        user_hash_mod=ctx.regime.user_ids_count,
    )
    if ctx.num_workers is not None:
        data_cfg = replace(data_cfg, num_workers=ctx.num_workers, persistent_workers=False)
    validate_data_config(dataset_name, data_cfg, ctx)
    return group, data_cfg


def validate_data_config(dataset_name: str, data_cfg: Any, ctx: RunContext) -> None:
    if ctx.required_split_strategy is not None and data_cfg.split_strategy != ctx.required_split_strategy:
        raise ValueError(f"{dataset_name}: requires split_strategy={ctx.required_split_strategy!r}")
    if ctx.required_split_strategy == "global_time":
        if (
            data_cfg.global_val_timestamp is None
            or data_cfg.global_test_timestamp is None
            or data_cfg.global_val_timestamp >= data_cfg.global_test_timestamp
            or data_cfg.global_history_policy != "rolling"
            or data_cfg.global_eval_target_mode != "last_per_user"
            or not data_cfg.filter_cold_items_after_split
            or data_cfg.cold_item_train_source != "observed"
            or data_cfg.drop_items_missing_semantic_id
        ):
            raise ValueError(
                f"{dataset_name}: global-time GPT-2 tuning requires ordered pinned cutoffs, "
                "rolling histories, last_per_user targets, train-observed cold-item filtering "
                "and drop_items_missing_semantic_id=false"
            )


def build_datamodule(dataset_name: str, ctx: RunContext) -> Any:
    group, data_cfg = resolve_data_config(dataset_name, ctx)
    if getattr(data_cfg, "use_hf", False):
        raise ValueError(
            "The exp2 runs are offline-only; set use_hf=false and configure "
            "interactions_path for every Yambda dataset."
        )
    missing = []
    for key in (
        "reviews_path",
        "metadata_path",
        "interactions_path",
        "embeddings_path",
        "semantic_id_path",
    ):
        path = getattr(data_cfg, key, None)
        if path is not None and not Path(path).is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"[{dataset_name}] missing input files:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    layout = getattr(data_cfg, "train_example_layout", None)
    if layout != REQUIRED_TRAIN_LAYOUT:
        raise ValueError(
            f"Dataset {dataset_name!r} resolves train_example_layout={layout!r}; "
            f"GPT-2 tuning requires {REQUIRED_TRAIN_LAYOUT!r}."
        )

    dm = group.data_module(data_cfg)
    dm.setup()
    if dm.semantic_ids is None or dm.sid_len is None:
        raise ValueError(f"Dataset {dataset_name!r} produced no semantic IDs.")
    if ctx.required_split_strategy == "global_time":
        missing_ids = dm.item2id.keys() - dm.semantic_ids.keys()
        if missing_ids:
            raise ValueError(f"{dataset_name}: semantic IDs miss {len(missing_ids)} training items")
        if not dm.num_items or not len(dm.train_dataset) or not len(dm.val_dataset):
            raise ValueError(f"{dataset_name}: empty training catalog, training rows or validation targets")
        dataset_dir = ctx.dataset_dir(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with promotion_lock(dataset_dir):
            runner.write_json(dataset_dir / "data_summary.json", dm.summary())
    print(f"\n[{dataset_name}] data summary:")
    print(json.dumps(dm.summary(), indent=2, sort_keys=True, default=str))
    print(
        f"[{dataset_name}] batch plan: train {dm.cfg.batch_size} x "
        f"grad_accum {ctx.regime.grad_accum}, validation {dm.cfg.eval_batch_size} "
        f"({ctx.regime.val_num_beams} beams)"
    )
    return dm


# ---------------------------------------------------------------------
# Model / method / trainer
# ---------------------------------------------------------------------

def model_block(hparams: Mapping[str, Any], *, codebook_size: int) -> dict[str, Any]:
    """The gpt2_model.yaml ``model:`` block for a search point."""

    embedding_dim = int(hparams["embedding_dim"])
    dropout = float(hparams["dropout"])
    return {
        "embedding_dim": embedding_dim,
        "codebook_size": int(codebook_size),
        "num_heads": int(hparams["num_heads"]),
        "num_layers": int(hparams["num_layers"]),
        "dim_feedforward": int(hparams["ff_mult"]) * embedding_dim,
        "dropout": dropout,
        "attention_dropout": dropout,
        "residual_dropout": dropout,
        "embedding_dropout": dropout,
    }


def training_block(spec: TuningSpec) -> dict[str, Any]:
    """Per-dataset TrainerConfig overrides (gpt2_model.yaml ``training:``)."""

    if spec.val_sample_fraction is None:
        return {}
    return {
        "val_sample_fraction": float(spec.val_sample_fraction),
        "val_sample_seed": VAL_SAMPLE_SEED,
        "val_sample_resample": False,
    }


def build_profile(
    hparams: Mapping[str, Any], spec: TuningSpec, *, codebook_size: int
) -> dict[str, Any]:
    """A gpt2_model.yaml dataset profile for one search point."""

    return {
        "model": model_block(hparams, codebook_size=codebook_size),
        "optimizer": {
            "lr": float(hparams["lr"]),
            "weight_decay": float(hparams["weight_decay"]),
        },
        "training": training_block(spec),
    }


def build_model(
    ctx: RunContext, dm: Any, hparams: Mapping[str, Any], spec: TuningSpec, dataset_name: str
) -> GPT2RecModel:
    # Built through the exp1 runner so the tuned model (position budget,
    # semantic catalog, user token, beam defaults) is constructed exactly
    # like the final runs construct it.
    profile = build_profile(
        hparams, spec, codebook_size=ctx.regime.codebook_sizes[dataset_name]
    )
    model = run_all.build_model(ctx.regime.model_config, profile, dm)
    if dm.cfg.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"GPT2RecModel does not support train_example_layout="
            f"{dm.cfg.train_example_layout!r}; supported: "
            f"{model.supported_train_layouts}."
        )
    return model


def build_method(ctx: RunContext) -> SFTMethod:
    return SFTMethod(
        SFTConfig(
            loss_weight=1.0,
            metric_prefix="sft",
            compute_train_metrics=False,
            compute_eval_metrics=True,
            # GPT-2 has no item logits; token metrics are the teacher-forced
            # score and not the tuning target.
            compute_item_metrics=False,
            compute_token_metrics=False,
            evaluate_generation=True,
            # Only the tuning cutoff during search; the final runs evaluate
            # the full [5, 10, 20] cutoffs.
            generation_topk=(MONITOR_K,),
            generation_num_beams=ctx.regime.val_num_beams,
            generation_num_return_sequences=MONITOR_K,
            generation_constrain_valid_sequences=ctx.regime.constrain_valid_sequences,
            # exp1/exp2 never exclude seen items at evaluation, for any model.
            generation_exclude_seen_items=False,
        )
    )


def build_trainer(
    *,
    ctx: RunContext,
    model: GPT2RecModel,
    method: SFTMethod,
    hparams: Mapping[str, Any],
    spec: TuningSpec,
    checkpoint_dir: Path,
    logger: Any,
) -> Trainer:
    regime = ctx.regime
    cfg = TrainerConfig(
        max_epochs=ctx.max_epochs(spec),
        gradient_accumulation_steps=regime.grad_accum,
        max_grad_norm=regime.max_grad_norm,
        use_amp=regime.use_amp,
        amp_dtype=regime.amp_dtype,
        log_every_steps=LOG_EVERY_STEPS,
        progress_bar=ctx.progress_bar,
        validate_every_epochs=1,
        checkpoint_dir=checkpoint_dir,
        # The trial result is the best epoch, not the last epoch.
        save_best=True,
        save_last=False,
        monitor=MONITOR,
        monitor_mode="max",
        early_stopping_patience=spec.patience,
        min_delta=regime.min_delta,
        early_stopping_mode="best",
        prefix_method_metrics_with_stage=True,
        max_train_batches=ctx.debug_max_batches,
        max_eval_batches=ctx.debug_max_batches,
        **training_block(spec),
    )
    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=cfg,
        device=ctx.device,
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        betas=regime.betas,
        eps=regime.eps,
        logger=logger,
    )


# ---------------------------------------------------------------------
# Per-epoch reporting and pruning
# ---------------------------------------------------------------------

class EpochValReporter:
    """Trainer logger hook: record per-epoch generation NDCG@10 and prune."""

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
                        "val_generation_ndcg10": value,
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

class _TunedConfigDumper(yaml.SafeDumper):
    """Block-style mappings (like gpt2_model.yaml) with inline scalar lists."""


def _represent_inline_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_TunedConfigDumper.add_representer(list, _represent_inline_list)


def dump_yaml(doc: Mapping[str, Any]) -> str:
    return yaml.dump(
        dict(doc),
        Dumper=_TunedConfigDumper,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_tuned_config(
    *,
    ctx: RunContext,
    dataset_name: str,
    hparams: Mapping[str, Any],
    best_value: float,
    best_trial_number: int,
    best_epoch: int | None,
    trials_completed: int,
    trials_planned: int,
    study_name: str,
    selection: str = "best single-seed trial of the search",
) -> Path:
    """Write ``gpt2_model_<dataset>.yaml`` in the gpt2_model.yaml layout."""

    regime = ctx.regime
    spec = TUNING_SPECS[dataset_name]
    path = ctx.tuned_config_path(dataset_name)
    data_yaml = TUNED_CONFIG_DATA_YAML[dataset_name]
    model_config = regime.model_config

    if spec.val_sample_fraction is None:
        val_scope = "the full validation split"
    else:
        val_scope = (
            f"a fixed {spec.val_sample_fraction:g} sample of the validation "
            f"users (seed {VAL_SAMPLE_SEED})"
        )
    baseline = regime.baselines[dataset_name]
    split = ctx.required_split_strategy or "from dataset config"
    producer = (
        "scripts/global-time/tuning_gpt2.py"
        if ctx.required_split_strategy == "global_time"
        else "scripts/exp2-tuned-params/tuning_gpt2.py"
    )
    header = f"""\
# GPT-2 hyperparameters for {dataset_name} (exp2, Optuna-tuned).
# Produced by {producer}: TPE sampler + median
# pruner maximizing {MONITOR} (constrained {regime.val_num_beams}-beam
# generation on {val_scope}). Validation split: {split}.
# Preprocessing and batching come from {_relative(ctx.config_dir / data_yaml)}
# (gpt2: block) and semantic IDs from
# {_relative(ctx.semantic_ids_dir)}/{runner.safe_component(dataset_name)}/semantic_ids.json;
# neither was tuned. Layout mirrors configs/exp1-standard_params/gpt2_model.yaml
# with a single datasets: profile. optimizer.weight_decay is tuned here and
# overrides the shared trainer_gpt2.yaml value; tuning_regime records the fixed
# trainer settings the search ran with (informational, from trainer_gpt2.yaml).
# Best checkpoint: {_relative(ctx.dataset_dir(dataset_name) / "best_model.pt")}
#
# study: {study_name}
# tuned on: {date.today().isoformat()}
# selection: {selection}
# trials completed: {trials_completed} (planned: {trials_planned})
# best trial: {best_trial_number}, best epoch: {best_epoch}
# best {MONITOR}: {best_value:.6f}
# exp1 baseline (trial 0): {json.dumps(baseline, sort_keys=True)}
# search-space point: {json.dumps(dict(hparams), sort_keys=True)}

"""
    doc = {
        "defaults": {
            "ks": list(run_all.resolve_ranking_ks(model_config)),
            "model": dict(model_config.get("defaults", {}).get("model", {})),
        },
        "eval_num_beams": int(model_config.get("eval_num_beams", 20)),
        "val_num_beams": regime.val_num_beams,
        "constrain_valid_sequences": regime.constrain_valid_sequences,
        "sft": dict(model_config.get("sft", {})),
        "datasets": {
            dataset_name: build_profile(
                hparams, spec, codebook_size=regime.codebook_sizes[dataset_name]
            )
        },
        "tuning_regime": {
            "gradient_accumulation_steps": regime.grad_accum,
            "max_grad_norm": regime.max_grad_norm,
            "betas": list(regime.betas),
            "eps": regime.eps,
            "min_delta": regime.min_delta,
            "amp_dtype": regime.amp_dtype,
            "early_stopping_patience": spec.patience,
            "max_epochs": ctx.max_epochs(spec),
            "monitor": MONITOR,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + dump_yaml(doc), encoding="utf-8")
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
    ctx: RunContext,
    dataset_name: str,
    trial: optuna.Trial,
    hparams: Mapping[str, Any],
    value: float,
    best_epoch: int | None,
    checkpoint_path: Path | None,
    study_name: str,
) -> None:
    """Persist the model/params as soon as a trial beats the stored best."""

    dataset_dir = ctx.dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with promotion_lock(dataset_dir):
        _promote_if_incumbent_locked(
            ctx=ctx,
            dataset_name=dataset_name,
            trial=trial,
            hparams=hparams,
            value=value,
            best_epoch=best_epoch,
            checkpoint_path=checkpoint_path,
            study_name=study_name,
        )


def _promote_if_incumbent_locked(
    *,
    ctx: RunContext,
    dataset_name: str,
    trial: optuna.Trial,
    hparams: Mapping[str, Any],
    value: float,
    best_epoch: int | None,
    checkpoint_path: Path | None,
    study_name: str,
) -> None:
    dataset_dir = ctx.dataset_dir(dataset_name)
    spec = TUNING_SPECS[dataset_name]
    best_json = dataset_dir / "best_trial.json"
    if best_json.exists():
        try:
            previous = json.loads(best_json.read_text(encoding="utf-8"))
            previous_value = float(
                previous.get("best_value_val_generation_ndcg10", float("-inf"))
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            previous_value = float("-inf")
    else:
        previous_value = float("-inf")

    if value <= previous_value:
        return

    if checkpoint_path is not None and checkpoint_path.exists():
        shutil.copy2(checkpoint_path, dataset_dir / "best_model.pt")
    else:
        print(
            f"[{dataset_name} | trial {trial.number}] WARNING: no best-epoch "
            "checkpoint found; best_model.pt not updated.",
            flush=True,
        )

    best_json.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "study_name": study_name,
                "metric": MONITOR,
                "best_value_val_generation_ndcg10": value,
                "best_trial_number": trial.number,
                "best_epoch": best_epoch,
                "best_params": dict(hparams),
                "model": model_block(
                    hparams, codebook_size=ctx.regime.codebook_sizes[dataset_name]
                ),
                "checkpoint": str(dataset_dir / "best_model.pt"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    config_path = write_tuned_config(
        ctx=ctx,
        dataset_name=dataset_name,
        hparams=hparams,
        best_value=value,
        best_trial_number=trial.number,
        best_epoch=best_epoch,
        trials_completed=trial.number + 1,
        trials_planned=ctx.n_trials(spec),
        study_name=study_name,
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
    ctx: RunContext,
    dm: Any,
    dataset_name: str,
    hparams: Mapping[str, Any],
    spec: TuningSpec,
    checkpoint_dir: Path,
    seed: int,
    logger: Any = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Train one GPT-2 with ``hparams`` and return its best-epoch result.

    Shared by the search objective and the --rerank stage so both train
    under exactly the same regime. The caller owns ``checkpoint_dir``
    cleanup.
    """

    model: GPT2RecModel | None = None
    trainer: Trainer | None = None
    try:
        runner.seed_everything(seed)
        model = build_model(ctx, dm, hparams, spec, dataset_name)
        method = build_method(ctx)
        trainer = build_trainer(
            ctx=ctx,
            model=model,
            method=method,
            hparams=hparams,
            spec=spec,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
        print(
            f"{log_prefix} params: {json.dumps(dict(hparams), sort_keys=True)} "
            f"| dim_feedforward: {int(hparams['ff_mult']) * int(hparams['embedding_dim'])} "
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


def make_objective(*, ctx: RunContext, dataset_name: str, dm: Any, spec: TuningSpec, study_name: str):
    dataset_dir = ctx.dataset_dir(dataset_name)

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
                ctx=ctx,
                dm=dm,
                dataset_name=dataset_name,
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
            trial.set_user_attr(
                "dim_feedforward",
                int(hparams["ff_mult"]) * int(hparams["embedding_dim"]),
            )

            promote_if_incumbent(
                ctx=ctx,
                dataset_name=dataset_name,
                trial=trial,
                hparams=hparams,
                value=value,
                best_epoch=best_epoch,
                checkpoint_path=result["checkpoint"],
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

FINISHED_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
)


# Trials in these states already cover their parameters: finished ones are
# results, running/waiting ones will become results. FAIL trials (a worker
# killed by the time limit, cleaned up with --fail-stale-trials) do not.
COVERING_STATES = FINISHED_STATES + (
    optuna.trial.TrialState.RUNNING,
    optuna.trial.TrialState.WAITING,
)


def study_name_for(dataset_name: str, *, split_strategy: str | None = None) -> str:
    suffix = f"_{split_strategy}" if split_strategy else ""
    return f"gpt2_{dataset_name}_val_generation_ndcg10_exp2{suffix}"


def validate_study_protocol(study: optuna.Study, dataset_name: str, ctx: RunContext) -> None:
    """Keep resumed global-time searches and reranking on the same protocol."""
    if ctx.required_split_strategy is None:
        return
    _, data_cfg = resolve_data_config(dataset_name, ctx)
    # Worker count and device precision may change across resubmissions.
    data = asdict(data_cfg)
    for key in ("num_workers", "persistent_workers", "pin_memory"):
        data.pop(key, None)
    spec = asdict(TUNING_SPECS[dataset_name])
    spec.pop("n_trials")  # Allow extending the search budget.
    regime = asdict(ctx.regime)
    regime.pop("amp_dtype")
    regime.pop("use_amp")
    protocol = runner.to_jsonable({
        "version": 1, "data_config": data, "spec": spec, "regime": regime,
        "max_epochs": ctx.max_epochs(TUNING_SPECS[dataset_name]),
        "debug_max_batches": ctx.debug_max_batches, "monitor": MONITOR, "seed": SEED,
    })
    with promotion_lock(ctx.dataset_dir(dataset_name)):
        previous = study.user_attrs.get("protocol")
        trials = study.get_trials(deepcopy=False)
        # A missing SID file can abort setup after the baseline was queued,
        # before any trial starts. Allow correcting just that input path so
        # the failed launch can resume without deleting its journal.
        sid_path_only = previous is not None and {
            **previous,
            "data_config": {
                **previous.get("data_config", {}),
                "semantic_id_path": protocol["data_config"]["semantic_id_path"],
            },
        } == protocol
        can_correct_sid_path = sid_path_only and all(
            trial.state == optuna.trial.TrialState.WAITING for trial in trials
        )
        if previous != protocol and (previous is not None or trials) and not can_correct_sid_path:
            raise ValueError("Existing study uses a different split/training protocol; use a new output root")
        study.set_user_attr("protocol", protocol)
        runner.write_json(ctx.dataset_dir(dataset_name) / "study_config.json", protocol)


def count_finished(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=FINISHED_STATES))


def trial_has_params(trial: optuna.trial.FrozenTrial, params: Mapping[str, Any]) -> bool:
    """True if ``trial`` was enqueued with / suggested exactly ``params``."""

    trial_params = trial.system_attrs.get("fixed_params", trial.params)
    if set(trial_params) != set(params):
        return False
    for name, value in params.items():
        existing = trial_params[name]
        if isinstance(value, bool) or isinstance(existing, bool):
            if bool(value) != bool(existing):
                return False
        elif isinstance(value, (int, float)) and isinstance(existing, (int, float)):
            if float(value) != float(existing):
                return False
        elif value != existing:
            return False
    return True


def ensure_incumbent_enqueued(study: optuna.Study, params: Mapping[str, Any]) -> bool:
    """Enqueue ``params`` unless a finished, running, or waiting trial holds them.

    Optuna's own ``skip_if_exists`` also counts FAIL trials, so a baseline
    attempt killed by the time limit would never be retried. Returns True if
    a trial was enqueued.
    """

    for trial in study.get_trials(deepcopy=False, states=COVERING_STATES):
        if trial_has_params(trial, params):
            return False
    study.enqueue_trial(dict(params), skip_if_exists=False)
    return True


def fail_stale_running_trials(study: optuna.Study) -> list[int]:
    """Mark every RUNNING trial FAIL; for resubmissions after all workers died."""

    failed: list[int] = []
    for trial in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
        failed.append(trial.number)
    return failed


def tune_one_dataset(
    ctx: RunContext,
    dataset_name: str,
    *,
    worker_id: int = 0,
    fail_stale_trials: bool = False,
) -> dict[str, Any]:
    spec = TUNING_SPECS[dataset_name]
    n_trials = ctx.n_trials(spec)
    dataset_dir = ctx.dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    study_name = study_name_for(dataset_name, split_strategy=ctx.required_split_strategy)
    journal_path = dataset_dir / "optuna_journal.log"
    regime = ctx.regime

    print("\n" + "=" * 100)
    print(f"Tuning GPT-2 on {dataset_name}")
    print("=" * 100)
    print(f"Config dir:      {ctx.config_dir}")
    print(f"Semantic IDs:    {ctx.semantic_ids_dir}")
    print(f"Output dir:      {dataset_dir}")
    print(f"Tuned config:    {ctx.tuned_config_path(dataset_name)}")
    print(f"Study:           {study_name} (journal: {journal_path})")
    print(f"Worker:          {worker_id}")
    print(f"Device:          {ctx.device} (amp={regime.use_amp}, dtype={regime.amp_dtype})")
    print(f"Budget:          {n_trials} trials, <= {ctx.max_epochs(spec)} epochs, "
          f"patience {spec.patience}")
    print(f"Monitor:         {MONITOR} ({regime.val_num_beams} beams, "
          f"val sample {spec.val_sample_fraction if spec.val_sample_fraction else 'full'})")
    print(f"exp1 baseline:   {json.dumps(regime.baselines[dataset_name], sort_keys=True)}")
    print(f"Space notes:     {spec.notes}")
    if ctx.debug_max_batches is not None:
        print(f"DEBUG:           max {ctx.debug_max_batches} train/eval batches per epoch")

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
    validate_study_protocol(study, dataset_name, ctx)

    if fail_stale_trials:
        failed = fail_stale_running_trials(study)
        if failed:
            print(
                f"Marked RUNNING trials {failed} as FAIL (--fail-stale-trials); "
                "their workers are assumed dead."
            )

    if worker_id == 0:
        # Seed the search with the exp1 incumbent configuration so tuning
        # can only improve on it in expectation. Re-submitted runs do not
        # duplicate it, but a FAILED attempt is retried.
        if ensure_incumbent_enqueued(study, regime.baselines[dataset_name]):
            print("Enqueued the exp1 baseline configuration as the next trial.")

    already_done = count_finished(study)
    if already_done:
        print(f"Resuming study: {already_done}/{n_trials} finished trials.")

    if already_done < n_trials:
        # The datamodule is built only when trials remain: a resubmitted
        # worker of a finished study must not pay the (Yambda: ~1 h) setup
        # just to rewrite the summary files below.
        dm = build_datamodule(dataset_name, ctx)
        try:
            # The budget is enforced against the shared study, not this
            # process: every worker keeps pulling trials until the study
            # holds n_trials finished ones, so any number of workers can
            # cooperate without overshooting by more than the trials still
            # in flight.
            study.optimize(
                make_objective(
                    ctx=ctx,
                    dataset_name=dataset_name,
                    dm=dm,
                    spec=spec,
                    study_name=study_name,
                ),
                n_trials=None,
                gc_after_trial=True,
                show_progress_bar=False,
                callbacks=[MaxTrialsCallback(n_trials, states=FINISHED_STATES)],
            )
        finally:
            del dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Study already holds its trial budget; skipping the search.")

    complete = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    if not any(t.value is not None for t in complete):
        raise RuntimeError(
            f"Study {study_name} holds no completed trial (every trial was "
            "pruned or failed); inspect the journal and the worker logs."
        )
    best = study.best_trial
    trials_completed = count_finished(study)

    # Final study statistics, serialized with the promotion lock so workers
    # finishing at the same time never interleave writes.
    with promotion_lock(dataset_dir):
        config_path = write_tuned_config(
            ctx=ctx,
            dataset_name=dataset_name,
            hparams=best.params,
            best_value=float(study.best_value),
            best_trial_number=best.number,
            best_epoch=best.user_attrs.get("best_epoch"),
            trials_completed=trials_completed,
            trials_planned=n_trials,
            study_name=study_name,
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
        "best_value_val_generation_ndcg10": float(study.best_value),
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
    return summary


# ---------------------------------------------------------------------
# Multi-seed re-ranking of the top search configurations
# ---------------------------------------------------------------------

def rerank_top_configs(ctx: RunContext, dataset_name: str, *, top_k: int) -> dict[str, Any]:
    """Retrain the top-k distinct search configs on RERANK_SEEDS and pick
    the winner by mean validation generation NDCG@10.

    Single-seed trial values are noisy, so the search argmax overfits the
    validation split (winner's curse). This stage rewrites the tuned YAML,
    best_model.pt, and best_trial.json with the config whose mean over three
    full seeds is best.
    """

    spec = TUNING_SPECS[dataset_name]
    n_trials = ctx.n_trials(spec)
    dataset_dir = ctx.dataset_dir(dataset_name)
    journal_path = dataset_dir / "optuna_journal.log"
    if not journal_path.exists():
        raise FileNotFoundError(
            f"No study journal at {journal_path}; run the search for "
            f"{dataset_name!r} first."
        )

    study = optuna.load_study(
        study_name=study_name_for(dataset_name, split_strategy=ctx.required_split_strategy),
        storage=JournalStorage(JournalFileBackend(str(journal_path))),
    )
    validate_study_protocol(study, dataset_name, ctx)
    finished = [t for t in study.trials if t.state in FINISHED_STATES]
    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(f"Study for {dataset_name!r} has no completed trials.")
    if len(finished) < n_trials:
        print(
            f"[{dataset_name} | rerank] WARNING: study holds "
            f"{len(finished)}/{n_trials} finished trials; re-ranking "
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
    dm = build_datamodule(dataset_name, ctx)

    results: list[dict[str, Any]] = []
    rerank_root = dataset_dir / "rerank"
    for t in candidates:
        seed_results: list[dict[str, Any]] = []
        failed = False
        for seed in RERANK_SEEDS:
            checkpoint_dir = rerank_root / f"trial_{t.number}" / f"seed_{seed}"
            try:
                out = train_once(
                    ctx=ctx,
                    dm=dm,
                    dataset_name=dataset_name,
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
                "mean_val_generation_ndcg10": mean,
                "std_val_generation_ndcg10": std,
            }
        )
        print(
            f"[{dataset_name} | rerank trial {t.number}] mean {MONITOR}="
            f"{mean:.6f} +- {std:.6f} (search value {t.value:.6f})",
            flush=True,
        )

    if not results:
        raise RuntimeError(f"Re-ranking produced no results for {dataset_name!r}.")
    results.sort(key=lambda r: r["mean_val_generation_ndcg10"], reverse=True)
    winner = results[0]
    best_seed = max(winner["seed_results"], key=lambda r: r["value"])

    selection = (
        f"mean {MONITOR} over seeds {list(RERANK_SEEDS)} across the "
        f"top-{len(results)} search configs (re-rank)"
    )
    if best_seed["checkpoint"] and Path(best_seed["checkpoint"]).exists():
        shutil.copy2(best_seed["checkpoint"], dataset_dir / "best_model.pt")
    config_path = write_tuned_config(
        ctx=ctx,
        dataset_name=dataset_name,
        hparams=winner["params"],
        best_value=winner["mean_val_generation_ndcg10"],
        best_trial_number=winner["trial"],
        best_epoch=best_seed["best_epoch"],
        trials_completed=len(finished),
        trials_planned=n_trials,
        study_name=study.study_name,
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
                "best_value_val_generation_ndcg10": winner["mean_val_generation_ndcg10"],
                "std_val_generation_ndcg10": winner["std_val_generation_ndcg10"],
                "best_trial_number": winner["trial"],
                "best_epoch": best_seed["best_epoch"],
                "best_params": winner["params"],
                "model": model_block(
                    winner["params"],
                    codebook_size=ctx.regime.codebook_sizes[dataset_name],
                ),
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
        f"mean {MONITOR}={winner['mean_val_generation_ndcg10']:.6f} "
        f"+- {winner['std_val_generation_ndcg10']:.6f}",
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Tune GPT-2 (SFT) with Optuna ({MONITOR}) on the exp2 datasets: "
            + ", ".join(DATASETS_TO_TUNE)
            + "."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS_TO_TUNE),
        choices=list(DATASETS_TO_TUNE),
        help="Subset of datasets to tune (default: all five).",
    )
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help=(
            "Index of this worker within a multi-worker study (e.g. the "
            "Slurm array task ID). Workers started with the same --datasets "
            "share the study journal and global trial budget; the ID only "
            "diversifies the sampler seed. Worker 0 enqueues the exp1 baseline."
        ),
    )
    parser.add_argument(
        "--semantic-ids-dir",
        type=Path,
        default=DEFAULT_SEMANTIC_IDS_DIR,
        help=(
            "Folder with <dataset>/semantic_ids.json (default: %(default)s)."
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
    parser.add_argument(
        "--fail-stale-trials",
        action="store_true",
        help=(
            "At startup mark every RUNNING trial of the study as FAIL. Use it "
            "when resubmitting after the time limit killed all workers: the "
            "journal storage has no heartbeat, so trials of dead workers stay "
            "RUNNING forever, bias the sampler, and (for the enqueued exp1 "
            "baseline) would block a retry. Never pass it while other workers "
            "of the same study are alive."
        ),
    )
    debug = parser.add_argument_group(
        "debug / smoke-test overrides (never use for the real studies)"
    )
    debug.add_argument(
        "--output-root",
        type=Path,
        default=CHECKPOINT_ROOT,
        help=(
            "Where per-dataset study dirs go (default: %(default)s). "
            "With a non-default value the tuned YAMLs are "
            "written under <output-root>/configs unless --config-out-dir is set."
        ),
    )
    debug.add_argument(
        "--config-out-dir",
        type=Path,
        default=None,
        help="Where gpt2_model_<dataset>.yaml is written (default: the entry point's config dir).",
    )
    debug.add_argument("--n-trials", type=int, default=None, help="Override the planned trial count.")
    debug.add_argument("--max-epochs", type=int, default=None, help="Override the epoch cap.")
    debug.add_argument(
        "--debug-max-batches",
        type=int,
        default=None,
        help="Cap train and validation batches per epoch.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_context(args: argparse.Namespace, regime: FixedRegime) -> RunContext:
    for name in ("n_trials", "max_epochs", "debug_max_batches", "rerank_top_k"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.worker_id < 0 or (args.num_workers is not None and args.num_workers < 0):
        raise ValueError("--worker-id and --num-workers must be nonnegative")
    output_root = runner.resolve_path(args.output_root, base=PROJECT_ROOT)
    if args.config_out_dir is not None:
        config_out_dir = runner.resolve_path(args.config_out_dir, base=PROJECT_ROOT)
    elif output_root.resolve() == CHECKPOINT_ROOT.resolve():
        config_out_dir = CONFIG_DIR
    else:
        # A redirected study (smoke test) must not overwrite the real tuned
        # configs.
        config_out_dir = output_root / "configs"
    return RunContext(
        regime=replace(regime, use_amp=regime.use_amp and str(args.device).startswith("cuda")),
        semantic_ids_dir=runner.resolve_path(args.semantic_ids_dir, base=PROJECT_ROOT),
        output_root=output_root,
        config_out_dir=config_out_dir,
        n_trials_override=args.n_trials,
        max_epochs_override=args.max_epochs,
        debug_max_batches=args.debug_max_batches,
        config_dir=runner.resolve_path(args.config_dir, base=PROJECT_ROOT),
        device=args.device,
        progress_bar=not args.no_progress_bar,
        num_workers=args.num_workers,
    )


def run(args: argparse.Namespace, ctx: RunContext) -> None:
    validate_specs(ctx.regime)
    runner.seed_everything(SEED)
    ctx.output_root.mkdir(parents=True, exist_ok=True)

    if args.rerank:
        for dataset_name in args.datasets:
            rerank_top_configs(ctx, dataset_name, top_k=args.rerank_top_k)
        return

    summaries: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        summaries[dataset_name] = tune_one_dataset(
            ctx,
            dataset_name,
            worker_id=args.worker_id,
            fail_stale_trials=args.fail_stale_trials,
        )

    print("\n" + "=" * 100)
    print("Finished tuning.")
    print(json.dumps(summaries, indent=2, sort_keys=True))

    # Single-dataset invocations (array tasks / parallel workers) skip the
    # shared combined summary so concurrent jobs never overwrite it; the
    # per-dataset outputs under each dataset dir already cover them.
    if len(args.datasets) > 1:
        summary_path = ctx.output_root / "tuning_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, sort_keys=True)
        print(f"Saved combined summary to: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args, build_context(args, load_regime()))


if __name__ == "__main__":
    main()
