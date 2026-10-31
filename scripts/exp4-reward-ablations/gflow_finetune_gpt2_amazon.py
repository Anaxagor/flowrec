#!/usr/bin/env python
"""GFlow fine-tuning ablation of the exp2 TUNED GPT-2 checkpoints on Amazon
and Yambda. Defaults retain Beauty 2014 / Instruments 2018; --datasets also
accepts beauty_2023, yambda_50m_listens and yambda_500m_listens.

The three added datasets store reward factors using precompute_reward_factors.py
and ComposedFactorizedRewardCache (the TIGER implementation). Yambda 50M uses
each user's last 128 targets; Yambda 500M samples 15 of those targets, seed 42.
Both subsets are applied after constructing the checkpoint's full catalog.
The base-matched wrapper uses all prefixes for all three Amazon datasets.

GPT-2 counterpart of the yambda_50m TIGER grid
(gflow_finetune_yambda50m.py) — same protocol (configs/exp2-tuned-params/
finetune_gflow.json), same fixed gradient-norm calibration of the gflow
weight, same curriculum sampler — but a full reward-composition grid:

  datasets (2)        beauty, instruments
  reward combos (7)   subsets of {sasrec, interaction, similarity}, sizes 1-3
  similarity (2)      every combo containing `similarity` is run once per
                      --similarity-modes variant:
                        sim_hier  hierarchical prefix-gated similarity over
                                  the RQ-VAE CODEBOOK embeddings (exp1
                                  tokenizer model.pt; exact match on the
                                  collision suffix)
                        sim_embed cosine similarity between the item CONTENT
                                  embeddings the tokenizer quantizes
                                  (datasets/embeddings/<ds>_content_embeddings.pt,
                                  (cos + 1) / 2)
                      (`exact` = positionwise token agreement is available
                      but not in the default grid)
                      -> 3 similarity-free + 2 x 4 similarity combos = 11
  weights (per combo) every assignment of --weight-levels (default 1 2) to
                      the combo's components whose smallest weight is the
                      first level: 1 / 3 / 7 assignments for 1 / 2 / 3
                      components (33 weighted combos in total). The global
                      scale of the weight vector is redundant with the power
                      (multiply) or drops out of the balance losses (mixed),
                      so only weight RATIOS are enumerated.
  reward powers (4)   {0.25, 1, 4, 16}: final reward = R_composed ** power
  objectives (3)      tb, db, subtb (GPT-2 scores prefixes through the
                      per-level fallback, so db/subtb cost sid_len scoring
                      passes per batch instead of one)
  num negatives (4)   {1, 3, 10, 100}, curriculum sampler over the SASRec
                      teacher (easy -> hard over --curriculum-steps)

  = 33 x 4 x 3 x 4 = 1584 runs per dataset, 3168 in total (single seed).
  Every dimension is a CLI list, so sub-grids are one flag away
  (e.g. --objectives tb --powers 1 4 --similarity-modes hierarchical).

Reward composition (--composition):
  multiply (default)  R = prod_i R_i ** w_i           (MultiplicativeRewardModel
                      semantics: weights are exponents = linear coefficients
                      in log-reward space, the quantity TB consumes)
  mixed               R = sum_i w_i * R_i / max_row(R_i)  (the additive
                      per-example max-normalized mixture of the exp4 TIGER
                      Amazon ablation and gflow_finetune_gpt2_beauty.py)
In both modes the power is applied AFTER composition and logZ = log sum over
the full catalog of the powered reward.

Reward caches for Beauty / Instruments: unlike the yambda grid (one dense cache per combo, with power
sums stored at build time), this script precomputes ONE dense full-catalog
cache PER COMPONENT per dataset (sasrec, interaction, sim_hier, sim_embed: 4
caches, ~2.3 GB float16 each for beauty) with
scripts/precompute_reward_cache.py, and composes weights / power /
composition mode at lookup time (ComposedRewardCache, float32, on the training
device). Every (combo, weights, power, composition, objective, K) run
therefore shares the same four caches, and the weight / objective grids cost
no extra cache builds.

Per dataset (fixed):
  - Init: the exp2 Optuna-tuned GPT-2 checkpoint
    checkpoints/exp2/loo_split/initial/gpt2/<ds>/best.pt.
    The architecture is rebuilt from the checkpoint's own model_config
    (or its optional .json sidecar) and the tuned dropout from best_trial.json,
    through run_gpt2_sft_all.build_model, then strict-loaded. The promoted
    checkpoints use the exp2 semantic-ID tables
    (datasets/semantic_ids/exp2-tuned-params/<ds>/, level-offset,
    token = code + 1 + level * 256), restricted to the training catalog.
    Their matching RQ-VAE weights live in <ds>/rqvae/model.pt for Beauty and
    Instruments, and <ds>/model.pt for the other datasets. The row-table
    check and the teacher-forced-NLL sanity check refuse any other pairing.
  - Data: the corresponding exp2 dataset preprocessing, catalog and held-out
    examples. Training expands active labels from the base dense-causal windows
    into one prefix/target per row, preserving each window's context boundary.
    Amazon uses all labels (window/stride 20); Yambda uses window/stride 50,
    then last 128 targets/user (50M), or 15 sampled from those 128 (500M,
    subset seed 42). SFT-only and GFlow share this pipeline and cache indices.
    --train-last-n 0 means no additional target reduction. Cache identities
    include the dense-window context; prior sliding-prefix caches are invalid.
  - SASRec teacher (curriculum sampler + sasrec reward): the exp2 tuned
    checkpoints/exp2/loo_split/initial/sasrec/<ds>/best.pt.
  - Reward components: sasrec (sigmoid-transformed teacher scores),
    interaction (store-backed evidence modulator: neutral 1.0 without
    evidence; on beauty every 2014 review is verified so the auxiliary store
    is empty and only positives carry evidence), sim_hier / sim_embed (the
    two similarity variants above, over the exp1 tokenizer's codebooks and
    the 384-d MiniLM content embeddings it was trained on).
  - Training: lr 3e-5, batch 256 x accum 8 (micro-batch scaled down for
    K=100 so batch x (K+1) <= 4096), monitor constrained beam-5
    val/sft/ndcg@5 every 5 epochs, <= 500 epochs, patience 10; then beam-20
    HR / NDCG / Coverage / ARP @ {5, 10, 20} on validation + test
    (run_gpt2_sft_all.evaluate_generation_ranking_metrics), upserted into
    artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_dense_prefix_metrics_summary.csv.
    The untouched init checkpoint is evaluated once per dataset as
    variant="init".

Outputs:
  experiments/exp4-gflow-gpt2-amazon-dense-prefix/<ds>/<combo>/w<weights>/pow_<p>/<obj>_k<K>/seed_<s>/
      resolved_config.json calibration.json training_history.json
      final_metrics.json result.json checkpoints/best.pt
      (<combo> names the concrete components, e.g. sasrec_x_sim_embed)
  experiments/exp4-gflow-gpt2-amazon-dense-prefix/<ds>/baseline/init_metrics.json
  artifacts/exp4-reward_ablation/gflow_gpt2_amazon_dense_prefix/<ds>/reward_cache/<component>/
      (sasrec, interaction, sim_hier, sim_embed[, sim_exact])

Usage:
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py --list
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py --index $SLURM_ARRAY_TASK_ID
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py \
      --cache-only --datasets beauty --components sim_embed      # one cache
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py \
      --datasets instruments --combos sasrec_x_similarity --powers 1 \
      --similarity-modes embedding_cosine --objectives tb \
      --num-candidates 3 --weight-levels 1                       # filtered sweep
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py \
      --smoke --device cpu --num-workers 0 --datasets beauty --index 0
The grid is dataset-major, then combo-major: consecutive --index values share
a dataset's datamodule, teacher, and component caches.
  sbatch scripts/slurm/gflow_finetune_gpt2_amazon_array.sbatch   # index mode

Cooperative workers (the recommended cluster mode — the Slurm array dimension
is WORKERS, sized to the per-user job limit, not grid entries):
  bash scripts/slurm/gflow_finetune_gpt2_amazon_submit.sh          # sizes + submits
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py \
      --worker-id $SLURM_ARRAY_TASK_ID --worker-count 24 --time-budget-hours 23.5
Each worker walks the (rotated) grid, atomically claims one pending run at a
time (<run_dir>/CLAIMED, flock-serialized, heartbeat = mtime; claims without
a heartbeat for --stale-claim-hours are treated as dead and re-claimed),
builds a dataset's datamodule/teacher/caches once, and stops claiming when
the next run would not fit --time-budget-hours. Failed runs (result.json
status=failed) are skipped by workers unless --retry-failed. Login-node
actions (no data loading): --status, --cache-status, --reset-claims.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# pyarrow.dataset's native module must load before torch on Windows;
# pandas.read_parquet imports it lazily and crashes otherwise.
import pyarrow.dataset  # noqa: F401
import torch

faulthandler.enable()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (
    str(PROJECT_ROOT),
    str(SCRIPTS_DIR),
    str(SCRIPT_DIR),
    str(SCRIPTS_DIR / "exp1-standard-params"),
    str(SCRIPTS_DIR / "exp2-tuned-params"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
import run_gpt2_sft_all as gpt2_runner  # noqa: E402  (scripts/exp1-standard-params)
import sft_finetune_control as control  # noqa: E402  (scripts/exp2-tuned-params)
import gflow_finetune_yambda50m as gflow_ablation  # noqa: E402  (this directory)
import gflow_finetune_yambda500m as factor_runner  # noqa: E402
import gflow_finetune_gpt2_beauty as gpt2_beauty  # noqa: E402  (scripts/)
import gpt2_matched_finetune as matched_source  # noqa: E402
from checkpoint_paths import initial_asset_path  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON, GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.ft_methods.gflow.factorized_cache import (  # noqa: E402
    ComposedFactorizedRewardCache, FactorizedRewardCache,
)
from flowrec.training import Trainer, TrainerConfig  # noqa: E402

runner = gpt2_runner.runner  # scripts/sasrec_config_runner (exp1 shared helpers)
GPT2GFlowMethod = gpt2_beauty.GPT2GFlowMethod


# --------------------------------------------------------------------------- #
# Assets and grid
# --------------------------------------------------------------------------- #
EXP2_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
EXP2_DATASET_YAML = EXP2_CONFIG_DIR / "dataset_amazon.yaml"
EXP1_GPT2_MODEL_YAML = PROJECT_ROOT / "configs" / "exp1-standard_params" / "gpt2_model.yaml"
# Semantic IDs and RQ-VAE of the promoted exp2 checkpoints.
SEMANTIC_IDS_ROOT = PROJECT_ROOT / "datasets" / "semantic_ids" / "exp2-tuned-params"
GPT2_ROOT = "checkpoints/exp2/loo_split/initial/gpt2"
SASREC_ROOT = "checkpoints/exp2/loo_split/initial/sasrec"
# Promoted tables are level-offset (token = code + 1 + level * 256); the
# hierarchical similarity reward de-offsets levels with this stride.
LEVEL_STRIDE = 256
# Item content embeddings (sim_embed reward).
EMBEDDINGS_ROOT = PROJECT_ROOT / "datasets" / "embeddings"

DATASETS: dict[str, dict[str, Any]] = {
    "beauty": {
        "gpt2_dir": f"{GPT2_ROOT}/beauty",
        "sasrec_checkpoint": f"{SASREC_ROOT}/beauty/best.pt",
        "sasrec_trial": f"{SASREC_ROOT}/beauty/best_trial.json",
        # exp2 tuning recorded full-val NDCG@10 0.0527; the 3-batch sanity
        # subset scores ~0.026 (a subset artifact, verified 2026-09-03).
        "teacher_ndcg_threshold": 0.02,
        "teacher_ndcg_reference": 0.0527,
    },
    "instruments": {
        "gpt2_dir": f"{GPT2_ROOT}/instruments",
        "sasrec_checkpoint": f"{SASREC_ROOT}/instruments/best.pt",
        "sasrec_trial": f"{SASREC_ROOT}/instruments/best_trial.json",
        # exp2 tuning recorded full-val NDCG@10 0.138.
        "teacher_ndcg_threshold": 0.04,
        "teacher_ndcg_reference": 0.138,
    },
}

# The new datasets use the same promoted exp2 layout. Their reward matrices
# are too large to persist: only per-example factors and shared tables are saved.
for _dataset, _teacher, _threshold in (
    ("beauty_2023", "sasrec", 0.01),
    ("yambda_50m_listens", "sasrec", 0.02),
    ("yambda_500m_listens", "sasrec_sce", 0.02),
):
    _teacher_dir = f"checkpoints/exp2/loo_split/initial/{_teacher}/{_dataset}"
    DATASETS[_dataset] = {
        "gpt2_dir": f"{GPT2_ROOT}/{_dataset}",
        "sasrec_checkpoint": f"{_teacher_dir}/best.pt",
        "sasrec_trial": f"{_teacher_dir}/best_trial.json",
        "teacher_ndcg_threshold": _threshold,
        "factorized": True,
    }
DEFAULT_DATASETS = ("beauty", "instruments")
MATCHED_PROTOCOL = EXP2_CONFIG_DIR / "finetune_base_matched.json"


def dataset_yaml_path(dataset: str) -> Path:
    if dataset.startswith("yambda_"):
        return EXP2_CONFIG_DIR / "dataset_yambda.yaml"
    if dataset == "beauty_2023":
        return EXP2_CONFIG_DIR / "dataset_amazon_2023.yaml"
    return EXP2_DATASET_YAML


def rqvae_model_path(dataset: str) -> Path:
    directory = SEMANTIC_IDS_ROOT / dataset
    return directory / "rqvae" / "model.pt" if dataset in DEFAULT_DATASETS else directory / "model.pt"


def embeddings_path(dataset: str) -> Path:
    if dataset.startswith("yambda_"):
        return PROJECT_ROOT / "data/raw/yambda/embeddings.parquet"
    return EMBEDDINGS_ROOT / f"{dataset}_content_embeddings.pt"

# Abstract combo shapes are subsets of these three reward families; the
# `similarity` slot is instantiated per similarity mode (concrete component
# keys below), so a combo like sasrec_x_similarity yields one run per mode.
COMPONENT_ORDER = ("sasrec", "interaction", "similarity")
# similarity mode -> concrete component key (cache dir name, combo key, CSV)
SIMILARITY_MODES = {
    "hierarchical": "sim_hier",       # RQ-VAE codebook-embedding geometry
    "embedding_cosine": "sim_embed",  # cosine between item content embeddings
    "exact": "sim_exact",             # positionwise token agreement (opt-in)
}
DEFAULT_SIMILARITY_MODES = ("hierarchical", "embedding_cosine")
CONCRETE_COMPONENTS = ("sasrec", "interaction", *SIMILARITY_MODES.values())
DEFAULT_WEIGHT_LEVELS = (1.0, 2.0)
DEFAULT_POWERS = (0.25, 1.0, 4.0, 16.0)
DEFAULT_NUM_NEGATIVES = (1, 3, 10, 100)
ALL_OBJECTIVES = ("tb", "db", "subtb")
DEFAULT_OBJECTIVES = ALL_OBJECTIVES
COMPOSITIONS = ("multiply", "mixed")
# Source of the K negatives: teacher = SASRec curriculum (legacy run dirs / CSV
# rows); random = uniform over the catalog; policy = the GPT-2 model being
# fine-tuned samples them itself. random / policy load no teacher unless a
# combo contains the sasrec reward.
SAMPLERS = ("teacher", "random", "policy")
DEFAULT_SAMPLER = "teacher"

PROTOCOL_CONFIG = EXP2_CONFIG_DIR / "finetune_gflow.json"
DEFAULT_METRICS_CSV_NAME = "gflow_finetune_gpt2_amazon_dense_prefix_metrics_summary.csv"


def all_combos() -> list[tuple[str, ...]]:
    return gflow_ablation.all_combos()


def combo_key(combo: Sequence[str]) -> str:
    return "_x_".join(combo)


def weight_assignments(num_components: int, levels: Sequence[float]) -> list[tuple[float, ...]]:
    """Every assignment of ``levels`` to the components whose minimum is the
    smallest level (weight ratios only: a common factor is a reward power in
    multiply mode and cancels in the balance losses in mixed mode)."""
    ordered = sorted({float(v) for v in levels})
    if not ordered or ordered[0] <= 0:
        raise ValueError(f"weight levels must be positive, got {levels!r}")
    return [
        assignment
        for assignment in itertools.product(ordered, repeat=int(num_components))
        if min(assignment) == ordered[0]
    ]


def weight_tag(weights: Sequence[float]) -> str:
    return "-".join(gflow_ablation.power_tag(w) for w in weights)


def protocol_defaults() -> dict[str, Any]:
    """finetune_gflow.json shared + gflow defaults, then the `gpt2_amazon`
    override block (currently max_candidate_batch=512: on 32 GB V100s every
    GPT-2 micro-batch scoring more than 512 candidate sequences ran out of
    memory in the first sweep)."""
    defaults = dict(gflow_ablation.protocol_defaults())
    if PROTOCOL_CONFIG.exists():
        payload = json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
        defaults.update({k: v for k, v in payload.get("gpt2_amazon", {}).items() if k != "note"})
    return defaults


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    D = protocol_defaults()  # configs/exp2-tuned-params/finetune_gflow.json (+ gpt2_amazon)
    p = argparse.ArgumentParser(
        "GFlow fine-tuning ablation of the exp2 tuned GPT-2 checkpoints on "
        "beauty / instruments: reward combos x weights x powers x negatives, "
        "gradient-norm-calibrated gflow weight."
    )
    combo_keys = [combo_key(c) for c in all_combos()]
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DEFAULT_DATASETS))
    p.add_argument("--dataset-profile", choices=["legacy", "matched"], default="legacy",
                   help="Apply per-dataset fine-tuning settings from finetune_base_matched.json")
    p.add_argument("--yambda50-base", choices=["legacy", "trial36"], default="legacy",
                   help="Yambda 50M init: legacy promoted model or explicitly matched trial 36 best.pt")
    p.add_argument("--yambda500-base", choices=["legacy", "trial4"], default="legacy",
                   help="Yambda 500M init: legacy promoted model or explicitly matched trial 4 best.pt")
    p.add_argument("--yambda500-targets", choices=["last128", "sampled50"], default="last128",
                   help="Trial 4 targets: all last 128, or a fixed sample of up to 50 per user "
                        "from that pool, without replacement and with subset seed 42")
    p.add_argument("--prefix-from-dense", action="store_true", default=True,
                   help="Derive prefixes from base GPT-2 dense windows (always enabled)")
    p.add_argument("--audit-data-only", action="store_true",
                   help="Build data and write the dense/prefix split audit, without models or rewards")
    p.add_argument("--combos", nargs="+", choices=combo_keys, default=combo_keys)
    p.add_argument("--component-combos", nargs="+", default=None,
                   help="Concrete component combos ('interaction', 'sim_hier_x_sim_embed', "
                        "'interaction_x_sim_hier_x_sim_embed', ...) run INSTEAD of --combos x "
                        "--similarity-modes; the only way to put both similarity rewards into one "
                        f"combo. Components: {list(CONCRETE_COMPONENTS)}")
    p.add_argument("--samplers", nargs="+", choices=list(SAMPLERS), default=[DEFAULT_SAMPLER],
                   help="Grid dimension: source of the K negatives. teacher = SASRec curriculum "
                        "(default, legacy run dirs); random = uniform over the catalog; policy = "
                        "sampled without replacement from the model being fine-tuned, always "
                        "on-policy (stochastic beam search constrained to valid catalog items; no "
                        "curriculum, temperature 1)")
    p.add_argument("--policy-extra-beams", type=int, default=None,
                   help="policy sampler: beams beyond K (default K, i.e. 2K beams)")
    p.add_argument("--weight-levels", type=float, nargs="+", default=list(DEFAULT_WEIGHT_LEVELS),
                   help="Weight levels assigned to the components of each combo; "
                        "only assignments whose minimum is the first level are run "
                        "(1 / 3 / 7 assignments for 1 / 2 / 3 components with two levels). "
                        "A single level = equal weights only.")
    p.add_argument("--powers", type=float, nargs="+", default=list(DEFAULT_POWERS))
    p.add_argument("--num-candidates", type=int, nargs="+", default=list(DEFAULT_NUM_NEGATIVES),
                   help="Curriculum negatives per example (grid dimension)")
    p.add_argument("--objectives", nargs="+", choices=list(ALL_OBJECTIVES),
                   default=list(DEFAULT_OBJECTIVES),
                   help="GFlow balance losses (grid dimension): trajectory (tb), "
                        "detailed (db), sub-trajectory (subtb) balance")
    p.add_argument("--similarity-modes", nargs="+", choices=list(SIMILARITY_MODES),
                   default=list(DEFAULT_SIMILARITY_MODES),
                   help="Similarity reward variants (grid dimension for combos with "
                        "`similarity`): hierarchical = RQ-VAE codebook-embedding geometry "
                        "(sim_hier), embedding_cosine = cosine between item content "
                        "embeddings (sim_embed), exact = token agreement (sim_exact)")
    p.add_argument("--composition", choices=list(COMPOSITIONS), default="multiply",
                   help="multiply: R = prod R_i ** w_i; mixed: R = sum w_i * R_i / rowmax(R_i)")
    p.add_argument("--components", nargs="+", choices=list(CONCRETE_COMPONENTS), default=None,
                   help="With --cache-only: build only these component caches "
                        "(default: every component used by the selected combos and "
                        "similarity modes)")
    p.add_argument("--seed", type=int, default=int(D.get("seed", 42)))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Grid selection (slurm arrays).
    p.add_argument("--list", action="store_true", help="Print the run grid with indices and exit")
    p.add_argument("--index", type=str, default=None,
                   help="Run one grid entry ('137') or a contiguous inclusive range "
                        "('135-139') sequentially (see --list); an end past the grid is clamped")
    p.add_argument("--force", action="store_true",
                   help="Re-run entries whose result.json already reports completed")

    # Cooperative workers (Slurm array = workers; see the module docstring).
    p.add_argument("--worker-id", type=int, default=None,
                   help="Run as cooperative worker N over the selected grid (claims runs "
                        "one at a time; mutually exclusive with --index)")
    p.add_argument("--worker-count", type=int, default=None,
                   help="Number of workers in the array (rotates each worker's starting "
                        "point over the grid); default $SLURM_ARRAY_TASK_COUNT or 1")
    p.add_argument("--time-budget-hours", type=float, default=None,
                   help="Worker wall-time budget: stop claiming when elapsed + the longest "
                        "run this worker has completed (or --run-hours-estimate) would "
                        "exceed it")
    p.add_argument("--run-hours-estimate", type=float, default=1.0,
                   help="Per-run duration assumed before this worker has completed a run "
                        "(with --resume: always, since an interrupted run is continued "
                        "rather than lost)")
    p.add_argument("--stale-claim-hours", type=float, default=3.0,
                   help="A claim without heartbeat for this long is treated as dead")
    p.add_argument("--retry-failed", action="store_true",
                   help="Workers also claim runs whose result.json says failed "
                        "(index mode always re-runs them)")
    p.add_argument("--status", action="store_true",
                   help="Print completed/failed/running/stale/pending counts for the "
                        "selected grid and exit (no data loading)")
    p.add_argument("--cache-status", action="store_true",
                   help="Print which component caches the selected grid needs and whether "
                        "they exist (MISSING <slot> <dataset> <component> lines; exit 1 if "
                        "any is missing) and exit")
    p.add_argument("--reset-claims", action="store_true",
                   help="Release every claim of a non-completed run in the selected grid "
                        "and exit. Only when no worker of this grid is running (live "
                        "claims are otherwise protected by the heartbeat)")

    # Data scale.
    p.add_argument("--train-last-n", type=int, default=0,
                   help="Fine-tune on the last N train examples per user (subset only; "
                        "the datamodule layout and catalog are untouched). 0 = the full "
                        "eligible history (every prefix for Amazon; last 128 for Yambda). Applies to the GFlow runs "
                        "and the SFT-only control alike; the component caches are built "
                        "for the same subset, so changing it needs another --cache-root")

    p.add_argument("--train-random-n", type=int, default=0,
                   help="Sample N prefix examples per user; exclusive with --train-last-n")
    p.add_argument("--train-subset-seed", type=int, default=0)
    p.add_argument("--yambda50-all-targets", action="store_true",
                   help="SFT-only Yambda 50M diagnostic: expand every base dense label instead of "
                        "only the last 128; appends all-targets suffixes to output/cache/CSV paths")

    # Reward pieces.
    p.add_argument("--sasrec-transform", default=str(D.get("sasrec_transform", "sigmoid")))
    p.add_argument("--sasrec-temperature", type=float, default=float(D.get("sasrec_temperature", 1.0)))

    # Sampler (fixed easy-negative phase).
    p.add_argument("--curriculum-steps", type=int, default=int(D.get("curriculum_steps", 3000)))
    p.add_argument("--curriculum-epochs", type=float, default=None,
                   help="Define the easy->hard ramp in EPOCHS over the fine-tuning examples "
                        "(overrides --curriculum-steps, which counts micro-steps: its length in "
                        "passes changes with the data scope and with the per-K micro-batch)")
    p.add_argument("--sampler-temperature", type=float, default=float(D.get("sampler_temperature", 1.0)))

    # Gradient-norm calibration of the gflow weight.
    p.add_argument("--gflow-weight-mode", choices=["fixed_grad_ratio", "constant"],
                   default=str(D.get("gflow_weight_mode", "fixed_grad_ratio")))
    p.add_argument("--gflow-target-grad-ratio", type=float,
                   default=float(D.get("gflow_target_grad_ratio", 1.0)),
                   help="rho_target: ||lambda*g_GFlow|| / ||g_SFT|| at init (1.0 = equal norms)")
    p.add_argument("--rho-targets", type=float, nargs="+", default=None,
                   help="Make rho_target a GRID dimension (e.g. 0.1 1 10); default: "
                        "just --gflow-target-grad-ratio. rho=1.0 keeps the plain run "
                        "dir name, other values get a _rho<r> suffix.")
    p.add_argument("--gflow-constant-weight", type=float, default=0.1,
                   help="Used only with --gflow-weight-mode constant")
    p.add_argument("--gflow-calibration-batches", type=int,
                   default=int(D.get("gflow_calibration_batches", 5)))
    p.add_argument("--gflow-calibration-eps", type=float,
                   default=float(D.get("gflow_calibration_eps", 1e-12)))
    p.add_argument("--gflow-calibration-seed", type=int,
                   default=int(D.get("gflow_calibration_seed", 1234)))

    # Reward cache (one per component per dataset).
    p.add_argument("--cache-batch-size", type=int, default=32)
    p.add_argument("--cache-format", choices=["auto", "dense", "factorized"], default="auto",
                   help="auto uses factors for Beauty 2023 and both Yambda datasets")
    p.add_argument("--cache-verify-batches", type=int, default=1)
    p.add_argument("--cache-dtype", choices=["float16", "float32"],
                   default=str(D.get("cache_dtype", "float16")))
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--require-existing-caches", action="store_true",
                   help="Load existing caches/factors only; fail instead of precomputing missing ones")
    p.add_argument("--cache-only", action="store_true",
                   help="Build/verify the component reward caches for the selected "
                        "datasets, then exit (run before the main array)")
    p.add_argument("--max-candidate-batch", type=int,
                   default=int(D.get("max_candidate_batch", 4096)),
                   help="Upper bound on batch * (num_negatives + 1) scored candidate sequences "
                        "per micro-batch (grad accumulation keeps the effective batch); the "
                        "gpt2_amazon block of finetune_gflow.json sets 512 for 32 GB GPUs")
    p.add_argument("--oom-retries", type=int, default=3,
                   help="On CUDA out-of-memory, halve --max-candidate-batch (and the final "
                        "evaluation batch) and restart the run, up to this many times")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                   help="Continue an unfinished run from its checkpoints/last.pt (written "
                        "every epoch) instead of restarting it: weights, optimizer, AMP "
                        "scaler, best value, patience counter, and the curriculum step are "
                        "restored; the calibrated lambda comes from calibration.json and the "
                        "micro-batch cap from the interrupted attempt. Makes wall-time kills "
                        "harmless for runs longer than one worker's remaining time. Ignored "
                        "with --force and after an OOM retry (which restarts the run)")
    p.add_argument("--sft-only", action="store_true",
                   help="Control arm: fine-tune with the SFT loss only (no rewards, sampler, "
                        "or calibration) under the identical trainer protocol, one run per "
                        "selected dataset (<output-root>/<dataset>/sft_only/seed_<s>, CSV "
                        "variant sft_only); ignores the grid dimensions")

    # Training (SFT-control protocol).
    p.add_argument("--max-epochs", type=int, default=int(D.get("max_epochs", 500)))
    p.add_argument("--validate-every", type=int, default=int(D.get("validate_every", 5)))
    p.add_argument("--patience", type=int, default=int(D.get("patience", 10)))
    p.add_argument("--min-epochs", type=int, default=0,
                   help="Early-stopping warm-up: validations of the first N epochs never count "
                        "towards --patience, so a run (and, identically, the SFT-only control) "
                        "cannot be stopped before the curriculum has reached full difficulty")
    p.add_argument("--lr", type=float, default=float(D.get("lr", 3e-5)))
    p.add_argument("--weight-decay", type=float, default=float(D.get("weight_decay", 1e-2)))
    p.add_argument("--batch-size", type=int, default=int(D.get("batch_size", 256)))
    p.add_argument("--grad-accum", type=int, default=int(D.get("grad_accum", 8)))
    p.add_argument("--max-grad-norm", type=float, default=float(D.get("max_grad_norm", 0.5)))
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"],
                   default=str(D.get("amp_dtype", "float16")))
    p.add_argument("--val-batch-size", type=int, default=int(D.get("val_batch_size", 256)))
    p.add_argument("--val-sample-fraction", type=float, default=None)
    p.add_argument("--val-sample-resample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--uniform-micro-batch", action=argparse.BooleanOptionalAction, default=False,
                   help="One micro-batch x accumulation for EVERY run of the selected grid "
                        "and for the SFT-only control: the setting the largest selected "
                        "--num-candidates needs under --max-candidate-batch (K=3, cap 512: "
                        "128 x 16). Default: each K gets its own micro-batch and the "
                        "control runs at --batch-size x --grad-accum; the effective batch "
                        "is --batch-size * --grad-accum either way")
    p.add_argument("--validate-at-start", dest="validate_at_start",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Score the init checkpoint before the first epoch: it seeds the best "
                        "value / best.pt and is the reference for early-stopping patience "
                        "(finetune_gflow.json run_validation_at_start, the TIGER fine-tuning "
                        "protocol). Off by default here because the last-8 sweep ran "
                        "without it")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--loader-prefetch-factor", type=int, default=2,
                   help="Number of training minibatches prepared ahead per DataLoader worker")
    p.add_argument("--reward-prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="Prepare compact factor-cache rows in training workers; dense caches keep their existing path")
    p.add_argument("--tensorized-train-loader", action=argparse.BooleanOptionalAction, default=True,
                   help="Assemble indexed training batches from compact sequence arrays")

    # Final evaluation (exp1 GPT-2 protocol: beam-20 HR/NDCG/Coverage/ARP).
    p.add_argument("--test-ks", type=int, nargs="+", default=list(D.get("test_ks", [5, 10, 20])))
    p.add_argument("--eval-num-beams", type=int, default=int(D.get("eval_num_beams", 20)))
    p.add_argument("--test-batch-size", type=int, default=128,
                   help="Beam-20 final evaluation batch (exp1 gpt2 test_eval_batch_size)")

    # Output.
    p.add_argument("--output-root", type=Path, default=Path("experiments/exp4-gflow-gpt2-amazon-dense-prefix"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("artifacts/exp4-reward_ablation/gflow_gpt2_amazon_dense_prefix"),
                   help="Component caches live at <cache-root>/<dataset>/reward_cache/<component>")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/exp4-reward_ablation"))
    p.add_argument("--metrics-csv", type=Path, default=None,
                   help="Metrics summary CSV (default: <artifacts-dir>/"
                        f"{DEFAULT_METRICS_CSV_NAME}). Runs with a different data subset / "
                        "protocol must not share a CSV: rows are keyed by the grid "
                        "identity only")

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0,
                   help="Max teacher-forced val NLL/token for the init checkpoint "
                        "(a random init scores ~log(vocab) ~ 6.65)")
    p.add_argument("--skip-init-baseline", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: ~8-batch caches, 1 epoch, unshuffled")
    args = p.parse_args(argv)
    if args.num_workers < 0 or args.loader_prefetch_factor < 1:
        p.error("--num-workers must be nonnegative and --loader-prefetch-factor positive")
    if args.require_existing_caches and args.force_cache:
        p.error("--require-existing-caches cannot be combined with --force-cache")
    if args.worker_id is not None and args.index is not None:
        p.error("--worker-id and --index are mutually exclusive")
    if args.worker_id is not None and args.worker_id < 0:
        p.error("--worker-id must be non-negative")
    if args.worker_count is None:
        args.worker_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1") or 1)
    if args.worker_count < 1:
        p.error("--worker-count must be positive")
    if any(float(w) <= 0 for w in args.weight_levels):
        p.error("--weight-levels must be positive")
    if any(float(x) <= 0 for x in args.powers):
        p.error("--powers must be positive")
    if any(int(k) < 0 for k in args.num_candidates):
        p.error("--num-candidates must be non-negative")
    if args.eval_num_beams < 1:
        p.error("--eval-num-beams must be positive")
    if any(k < 1 for k in args.test_ks):
        p.error("--test-ks must contain positive integers")
    if max(args.test_ks) > args.eval_num_beams:
        p.error("--eval-num-beams must be at least max(--test-ks): "
                "5-beam evaluation requires --test-ks 5; @10/@20 require more beams")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if args.yambda500_targets == "sampled50" and args.yambda500_base != "trial4":
        p.error("--yambda500-targets sampled50 requires --yambda500-base trial4")
    if args.yambda500_base == "trial4":
        if args.datasets != ["yambda_500m_listens"] or args.yambda50_base != "legacy":
            p.error("--yambda500-base trial4 requires only Yambda 500M and cannot select trial36")
        sampled = args.yambda500_targets == "sampled50"
        required = [("--train-last-n", args.train_last_n, 0 if sampled else 128),
                    ("--train-random-n", args.train_random_n, 50 if sampled else 0)]
        if sampled:
            required.append(("--train-subset-seed", args.train_subset_seed, 42))
        for option, value, expected in required:
            if any(token.split("=", 1)[0] == option for token in raw_argv) and value != expected:
                p.error(f"--yambda500-targets {args.yambda500_targets} requires {option} {expected}")
    if args.yambda50_all_targets:
        if args.yambda50_base == "trial36":
            p.error("--yambda50-all-targets is incompatible with matched trial 36 last-128 targets")
        if args.datasets != ["yambda_50m_listens"] or not (args.sft_only or args.audit_data_only):
            p.error("--yambda50-all-targets requires only Yambda 50M and --sft-only or --audit-data-only")
        if args.train_last_n != 0 or args.train_random_n != 0:
            p.error("--yambda50-all-targets cannot be combined with target subsampling")
        # A wrapper also supplies paths in raw_argv, so suffix its defaults too.
        # Already-suffixed explicit paths remain stable on resubmission.
        for field in ("output_root", "cache_root"):
            path = Path(getattr(args, field))
            if not path.name.endswith("-all-targets"):
                setattr(args, field, path.with_name(path.name + "-all-targets"))
        csv = metrics_csv_path(args)
        if not csv.stem.endswith("_all_targets"):
            csv = csv.with_name(csv.stem + "_all_targets" + csv.suffix)
        args.metrics_csv = csv
    args.dataset_overrides = [key for key in DATASET_ARGUMENTS
                              if any(arg.split("=", 1)[0] in
                                     ("--" + key.replace("_", "-"), "--no-" + key.replace("_", "-"))
                                     for arg in raw_argv)]
    return dataset_args(args, args.datasets[0])


DATASET_ARGUMENTS = ("train_last_n", "train_random_n", "train_subset_seed", "batch_size",
                     "grad_accum", "val_batch_size", "test_batch_size", "val_sample_fraction",
                     "val_sample_resample", "max_candidate_batch")


def dataset_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    """Resolve a fresh namespace per dataset; explicit CLI options always win."""
    resolved = argparse.Namespace(**vars(args))
    matched = getattr(args, "dataset_profile", "legacy") == "matched"
    is_yambda = dataset.startswith("yambda_")
    if matched or is_yambda:
        protocol = load_json(MATCHED_PROTOCOL)
        block = {"train_random_n": 0, "train_subset_seed": 0,
                 "max_candidate_batch": protocol_defaults()["max_candidate_batch"],
                 **protocol["datasets"][dataset]}
        if matched:
            # Model-specific batch/evaluation settings apply equally to the
            # GPT-2 SFT control and GFlow, without changing TIGER's defaults.
            arm = protocol["arms"]["gpt2_sft_only_and_gflow"]
            block.update(arm.get("dataset_overrides", {}).get(dataset, {}))
        for key in DATASET_ARGUMENTS:
            use_dataset_default = matched or key in ("train_last_n", "train_random_n", "train_subset_seed")
            if use_dataset_default and key in block and key not in args.dataset_overrides:
                setattr(resolved, key, block[key])
    # Keep the full dense checkpoint catalog and held-out rows. Select the
    # Yambda tail from dense labels, then sample the 500M subset from that tail.
    resolved.train_pool_last_n = (int(runner.load_yaml(dataset_yaml_path(dataset))["tiger"]["train_last_n"])
                                 if is_yambda else None)
    if dataset == "yambda_50m_listens" and getattr(args, "yambda50_all_targets", False):
        resolved.train_pool_last_n = None
    if dataset == "yambda_500m_listens" and matched_source.selected(dataset, args):
        # Build/audit the checkpoint's complete last-128 pool before selecting
        # the fine-tuning subset. Sampling must not alter histories or catalog.
        sampled = getattr(args, "yambda500_targets", "last128") == "sampled50"
        resolved.train_pool_last_n = 128
        resolved.train_last_n = 0 if sampled else 128
        resolved.train_random_n = 50 if sampled else 0
        if sampled:
            resolved.train_subset_seed = 42
        if "val_sample_resample" not in args.dataset_overrides:
            resolved.val_sample_resample = False
        if "val_sample_fraction" not in args.dataset_overrides:
            resolved.val_sample_fraction = 0.25
    # The new dense protocol already selected one full-context target per
    # row. GFlow uses its equivalent ordinary prefixes, not legacy windows.
    resolved.prefix_from_dense = not matched_source.selected(dataset, args)
    return resolved


def abspath(rel: str | Path) -> Path:
    return initial_asset_path(PROJECT_ROOT, rel)


def metrics_csv_path(args: argparse.Namespace) -> Path:
    if args.metrics_csv is not None:
        return abspath(args.metrics_csv)
    return abspath(args.artifacts_dir) / DEFAULT_METRICS_CSV_NAME


def cache_last_n(args: argparse.Namespace) -> int | None:
    """--per-user-last-n of the component caches (the value their manifests
    record): None = no additional tail reduction of the eligible history."""
    last_n = int(args.train_last_n)
    return last_n if last_n > 0 else None


def finetune_subset_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {**factor_runner.subset_spec(args),
            "per_user_pool_last_n": getattr(args, "train_pool_last_n", None)}


def describe_finetune_subset(args: argparse.Namespace) -> str:
    description = factor_runner.describe_subset(finetune_subset_spec(args))
    if getattr(args, "train_pool_last_n", None) is not None:
        description += f" within each user's last {args.train_pool_last_n} examples"
    return description


def finetune_indices(train_dataset, args: argparse.Namespace) -> list[int]:
    """Shared training/cache subset, keeping the original catalog and indices."""
    from precompute_reward_cache import subset_indices

    indices = subset_indices(train_dataset, **finetune_subset_spec(args))
    return list(range(len(train_dataset))) if indices is None else indices


def curriculum_steps_for_run(args: argparse.Namespace, run_batch: int, num_examples: int) -> int:
    """Sampler micro-steps of the easy->hard ramp: --curriculum-epochs passes
    over the run's fine-tuning examples at its micro-batch, else the
    --curriculum-steps protocol constant."""
    if args.curriculum_epochs is not None:
        return gflow_ablation.curriculum_steps_for_epochs(args.curriculum_epochs, num_examples, run_batch)
    return int(args.curriculum_steps)


def run_batch_settings(args: argparse.Namespace, num_negatives: int | None) -> tuple[int, int]:
    """(micro-batch, grad accumulation) of one run; the effective batch is
    always --batch-size * --grad-accum. ``num_negatives=None`` is the SFT-only
    control. --uniform-micro-batch pins every run, control included, to the
    setting of the largest selected K, so no arm differs in micro-batch."""
    if args.uniform_micro_batch:
        batch, accum = gflow_ablation.batch_settings_for_k(max(int(k) for k in args.num_candidates), args)
    elif num_negatives is None:
        return int(args.batch_size), int(args.grad_accum)
    else:
        batch, accum = gflow_ablation.batch_settings_for_k(int(num_negatives), args)
    effective = int(args.batch_size) * int(args.grad_accum)
    if batch * accum != effective:
        # Pick a divisor so candidate-memory scaling preserves the protocol's
        # effective batch exactly (e.g. 256 x 8 -> 32 x 64 at K=10, cap=512).
        batch = next(n for n in range(batch, 0, -1) if effective % n == 0)
        accum = effective // batch
    return batch, accum


load_json = gflow_ablation.load_json
write_json = gflow_ablation.write_json


# --------------------------------------------------------------------------- #
# Composed reward cache: weights / power / composition at lookup time
# --------------------------------------------------------------------------- #
class ComposedRewardCache:
    """Drop-in for GFlowMethod's ``_PrecomputedRewardCache`` that composes
    per-component full-catalog caches at lookup time.

    Each component cache stores the RAW component reward for every
    (example, catalog item) — built by precompute_reward_cache.py with a
    single-component reward config. The lookup returns, like the base class,
    ``(selected [B, N], reward_sum [B], full [B, num_items] | None)`` with

        multiply:  R = prod_i clamp(R_i) ** w_i
        mixed:     R = sum_i w_i * R_i / max_row(R_i)

    then ``R ** power`` and ``reward_sum = sum_items R ** power`` (the exact
    full-catalog logZ), all in float32 on the requested device. Candidate rows
    < 0 resolve to the example's cached target row exactly as in the base
    class (each component cache performs the substitution).
    """

    def __init__(
        self,
        components: Sequence[tuple[str, Path | str, float]],
        *,
        composition: str,
        power: float,
        reward_epsilon: float = GFLOW_REWARD_EPSILON,
    ) -> None:
        if not components:
            raise ValueError("ComposedRewardCache requires at least one component cache.")
        if composition not in COMPOSITIONS:
            raise ValueError(f"composition must be one of {COMPOSITIONS}, got {composition!r}")
        if float(power) <= 0.0:
            raise ValueError("power must be positive.")
        self.names = [str(name) for name, _, _ in components]
        self.caches = [_PrecomputedRewardCache(path, power=1.0) for _, path, _ in components]
        self.weights = [float(w) for _, _, w in components]
        if any(w <= 0.0 for w in self.weights):
            raise ValueError(f"component weights must be positive, got {self.weights}")
        self.composition = str(composition)
        self.power = float(power)
        self.reward_epsilon = float(reward_epsilon)
        sizes = {c.num_examples for c in self.caches}
        if len(sizes) != 1:
            raise ValueError(
                f"Component caches disagree on num_examples: "
                f"{dict(zip(self.names, [c.num_examples for c in self.caches]))}"
            )
        widths = {int(c._num_items) for c in self.caches}
        if len(widths) != 1:
            raise ValueError(f"Component caches disagree on the catalogue width: {widths}")
        self._num_examples = sizes.pop()
        self._num_items = widths.pop()

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def num_items(self) -> int:
        return self._num_items

    def describe(self) -> dict[str, Any]:
        terms = [f"{n}^{w:g}" if self.composition == "multiply" else f"{w:g}*rownorm({n})"
                 for n, w in zip(self.names, self.weights)]
        joiner = " * " if self.composition == "multiply" else " + "
        return {
            "composition": self.composition,
            "components": self.names,
            "weights": self.weights,
            "power": self.power,
            "form": "(" + joiner.join(terms) + f") ** {self.power:g}",
        }

    def select_candidates(
        self,
        indices: torch.Tensor,
        candidate_rows: torch.Tensor,
        device: torch.device,
        *,
        include_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        eps = self.reward_epsilon
        combined_sel: torch.Tensor | None = None
        combined_full: torch.Tensor | None = None
        for cache, weight in zip(self.caches, self.weights):
            sel, _, full = cache.select_candidates(
                indices, candidate_rows, device=device, include_full=True
            )
            assert full is not None
            sel = sel.float().clamp_min(eps)
            full = full.float().clamp_min(eps)
            if self.composition == "multiply":
                term_sel = sel if weight == 1.0 else sel.pow(weight)
                term_full = full if weight == 1.0 else full.pow(weight)
                combined_sel = term_sel if combined_sel is None else combined_sel * term_sel
                combined_full = term_full if combined_full is None else combined_full * term_full
            else:  # mixed: per-example max normalization, weighted sum
                row_max = full.amax(dim=1, keepdim=True).clamp_min(eps)
                term_sel = (sel / row_max).clamp_min(eps) * weight
                term_full = (full / row_max).clamp_min(eps) * weight
                combined_sel = term_sel if combined_sel is None else combined_sel + term_sel
                combined_full = term_full if combined_full is None else combined_full + term_full
        assert combined_sel is not None and combined_full is not None
        combined_sel = combined_sel.clamp_min(eps)
        combined_full = combined_full.clamp_min(eps)
        if self.power != 1.0:
            combined_sel = combined_sel.pow(self.power).clamp_min(eps)
            combined_full = combined_full.pow(self.power).clamp_min(eps)
        reward_sum = combined_full.sum(dim=1)
        return combined_sel, reward_sum, (combined_full if include_full else None)


def make_method(
    base_cfg: GFlowConfig, sampler, sft_cfg: SFTConfig, cache: ComposedRewardCache
) -> GPT2GFlowMethod:
    method = GPT2GFlowMethod(
        base_cfg, candidate_sampler=sampler,
        sft_method=gflow_ablation.AliasedSFTMethod(sft_cfg),
    )
    method._reward_cache = cache  # composed at lookup time (see ComposedRewardCache)
    return method


# --------------------------------------------------------------------------- #
# Config assembly (single source of truth for datamodule + caches + teacher)
# --------------------------------------------------------------------------- #
def similarity_key(mode: str) -> str:
    try:
        return SIMILARITY_MODES[str(mode)]
    except KeyError:
        raise ValueError(f"Unknown similarity mode {mode!r}; expected one of "
                         f"{list(SIMILARITY_MODES)}") from None


def similarity_mode_of(key: str) -> str | None:
    for mode, concrete in SIMILARITY_MODES.items():
        if concrete == key:
            return mode
    return None


def concrete_components(combo: Sequence[str], similarity_mode: str | None) -> tuple[str, ...]:
    """Abstract combo shape -> concrete component keys (similarity -> sim_*)."""
    out = []
    for name in combo:
        if name == "similarity":
            if similarity_mode is None:
                raise ValueError("A combo with `similarity` needs a similarity mode.")
            out.append(similarity_key(similarity_mode))
        else:
            out.append(name)
    return tuple(out)


def component_spec(key: str, dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    """run_tiger reward-registry spec for one CONCRETE component key. `name`
    is the registry name (sim_* keys all build the `similarity` reward)."""
    if key == "sasrec":
        return {
            "name": "sasrec",
            "weight": 1.0,
            "reward": {
                "transform": str(args.sasrec_transform),
                "temperature": float(args.sasrec_temperature),
            },
        }
    if key == "interaction":
        # Engagement scoring + evidence-scored positives (module defaults).
        return {"name": "interaction", "weight": 1.0, "reward": {}}
    mode = similarity_mode_of(key)
    if mode == "hierarchical":
        # token_offset comes from the datamodule's offset_semantic_ids_by_one
        # (1 here); the collision suffix beyond the RQ levels is exact-match.
        reward = {
            "mode": "hierarchical",
            "level_stride": LEVEL_STRIDE,
            "rqvae_model_path": str(rqvae_model_path(dataset)),
        }
    elif mode == "embedding_cosine":
        # (cos(z_item, z_target) + 1) / 2 over the normalized content embeddings.
        reward = {
            "mode": "embedding_cosine",
            "embeddings_path": str(embeddings_path(dataset)),
        }
    elif mode == "exact":
        reward = {"mode": "exact"}
    else:
        raise ValueError(f"Unknown reward component {key!r}")
    return {"name": "similarity", "weight": 1.0, "reward": reward}


def component_keys_for(args: argparse.Namespace) -> list[str]:
    """Concrete component keys the selected similarity modes make available."""
    return ["sasrec", "interaction", *[similarity_key(m) for m in args.similarity_modes]]


def sasrec_method_cfg(dataset: str) -> dict[str, Any]:
    spec = DATASETS[dataset]
    trial = abspath(spec["sasrec_trial"])
    payload = load_json(trial) if trial.is_file() else {}
    params = payload.get("best_params", payload)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
    return {
        "checkpoint_path": str(abspath(spec["sasrec_checkpoint"])),
        "model": {k: params[k] for k in model_keys if k in params},
    }


def build_run_config(dataset: str, args: argparse.Namespace, user_ids_count: int) -> dict[str, Any]:
    """run_tiger-schema config driving the fine-tuning datamodule, the SASRec
    teacher, and (per component, see cache_config) the reward-cache
    precompute. Mirrors the existing exp2 dataset YAML preprocessing the GPT-2
    tuner used, on the prefix_target layout (GFlow needs one target per
    example); the catalog is identical to the checkpoint's."""
    yaml_cfg = runner.load_yaml(dataset_yaml_path(dataset))
    entry = dict(yaml_cfg["datasets"][dataset])
    source = "yambda" if dataset.startswith("yambda_") else "amazon"
    if source == "amazon" and not bool(yaml_cfg.get("filter_reviews_to_metadata", True)):
        raise ValueError("filter_reviews_to_metadata=false is not supported by run_tiger.build_data_config")
    # Build the checkpoint's dense windows and catalog before expanding labels
    # into prefixes. The datamodule retains the dense validation/test rows.
    values = {k: v for k, v in yaml_cfg.items() if k not in {"datasets", "gpt2", "tiger", "sasrec"}}
    values.update({k: v for k, v in entry.items() if k not in {"gpt2", "tiger", "sasrec"}})
    values.update(train_example_layout="prefix_target", train_target_policy="all",
                  train_all_prefixes=True, history_padding="left", drop_items_missing_semantic_id=True)
    if source == "amazon":
        # SFT/GFlow and reward precompute use existing semantic IDs. Preserve
        # metadata catalog filtering without decoding unused text/details.
        values["metadata_ids_only"] = True
    if getattr(args, "prefix_from_dense", False):
        # The exp2 GPT-2 runner fixes both window options to None (H / H).
        # Catalog and held-out examples are built before the tail is selected.
        values.update(train_prefix_from_dense=True, train_window_len=None, train_window_stride=None)
        if source == "yambda" and args.train_pool_last_n is not None:
            values.update(train_target_policy="last_n", train_last_n=int(args.train_pool_last_n))
    data = {"source": source, "semantic_id_path": str(SEMANTIC_IDS_ROOT / dataset / "semantic_ids.json")}
    for key in ("reviews_path", "metadata_path", "interactions_path", "embeddings_path"):
        if values.get(key):
            data[key] = str(abspath(values.pop(key)))
    result = {
        "experiment_name": f"gflow_gpt2_{dataset}",
        "seed": int(args.seed),
        "data": data,
        "dataset": values,
        "dataloader": {
            "train_batch_size": int(args.batch_size),
            "batch_size": int(args.batch_size),
            "validation_batch_size": int(args.val_batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": True,
            "persistent_workers": int(args.num_workers) > 0,
            "train_drop_last": False,
        },
        # user_ids_count feeds user_hash_mod (1: one shared user token, as in
        # the tuned run).
        "model": {"use_user_token": True, "user_ids_count": int(user_ids_count)},
        "method": {
            "name": "gflow",
            "reward_model": "multiply" if args.composition == "multiply" else "mixed",
            "reward_epsilon": GFLOW_REWARD_EPSILON,
            # Informational for the run (never built as one reward model): all
            # concrete components the grid can draw on. component_keys is the
            # parallel list of concrete keys (specs carry registry names only,
            # because run_tiger folds unknown spec keys into the reward config).
            "component_keys": component_keys_for(args),
            "reward": {
                "components": [component_spec(key, dataset, args)
                               for key in component_keys_for(args)]
            },
            "sasrec": sasrec_method_cfg(dataset),
        },
    }
    if matched_source.selected(dataset, args):
        recorded = matched_source.load_recorded(PROJECT_ROOT, dataset)
        data, values = matched_source.input_config(recorded, PROJECT_ROOT)
        result["data"], result["dataset"] = data, values
        # Dataset seed is part of the frozen pretraining protocol; the
        # fine-tuning seed still controls optimizer/shuffle/model randomness.
        result["seed"] = int(recorded["data_config"]["seed"])
        user_hash_mod = recorded["data_config"].get("user_hash_mod")
        result["model"] = {"use_user_token": user_hash_mod is not None,
                           "user_ids_count": user_hash_mod}
        result["initial_model"] = {"source": "gpt2_tiger_matched", "dataset": dataset,
                                   "trial": matched_source.trial_number(dataset),
                                   "checkpoint": str(tuned_checkpoint_paths(dataset, args)["checkpoint"])}
    return result


def component_spec_from_config(run_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    keys = list(run_config["method"]["component_keys"])
    return dict(run_config["method"]["reward"]["components"][keys.index(key)])


def cache_config(run_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """The precompute config for ONE concrete component: the same data/teacher,
    but a single-component reward so the cache stores the raw component values."""
    config = json.loads(json.dumps(run_config))
    spec = component_spec_from_config(run_config, key)
    config["experiment_name"] = f"{run_config['experiment_name']}_{key}_cache"
    config["method"] = {
        "name": "gflow",
        "reward_model": spec["name"],
        "reward_epsilon": GFLOW_REWARD_EPSILON,
        "reward": dict(spec.get("reward", {})),
        "sasrec": dict(run_config["method"]["sasrec"]),
    }
    return config


def component_cache_dir(dataset: str, key: str, args: argparse.Namespace) -> Path:
    layout = "reward_factors" if uses_factor_cache(dataset, args) else "reward_cache"
    return abspath(args.cache_root) / dataset / layout / key


def uses_factor_cache(dataset: str, args: argparse.Namespace) -> bool:
    mode = getattr(args, "cache_format", "auto")
    return mode == "factorized" or (mode == "auto" and bool(DATASETS[dataset].get("factorized")))


def compose_reward_cache(entry: Mapping[str, Any], cache_dirs: Mapping[str, Path],
                         args: argparse.Namespace):
    cls = ComposedFactorizedRewardCache if uses_factor_cache(entry["dataset"], args) else ComposedRewardCache
    return cls([(key, cache_dirs[key], weight) for key, weight in zip(entry["components"], entry["weights"])],
               composition=str(entry["composition"]), power=float(entry["power"]))


def validate_inputs(dataset: str, args: argparse.Namespace, *, need_teacher: bool = True,
                    components: Sequence[str] | None = None) -> None:
    spec = DATASETS[dataset]
    paths = tuned_checkpoint_paths(dataset, args)
    yaml_path = dataset_yaml_path(dataset)
    required = [paths["checkpoint"], paths.get("resolved_config", paths.get("best_trial")),
                SEMANTIC_IDS_ROOT / dataset / "semantic_ids.json", yaml_path, EXP1_GPT2_MODEL_YAML]
    if need_teacher:
        required.append(abspath(spec["sasrec_checkpoint"]))
        if matched_source.selected(dataset, args):
            import run_sasrec_all

            required.append(abspath(spec["sasrec_checkpoint"]).parent / "training_summary.json")
            teacher_config = run_sasrec_all.build_yambda_data_config(
                dataset, runner.load_yaml(yaml_path), seed=42)
            required.extend(Path(path) for path in
                            (teacher_config.interactions_path, teacher_config.embeddings_path) if path)
    keys = component_keys_for(args) if components is None else components
    if "sim_hier" in keys:
        required.append(rqvae_model_path(dataset))
    if "sim_embed" in keys:
        required.append(embeddings_path(dataset))
    if matched_source.selected(dataset, args):
        recorded = matched_source.load_recorded(PROJECT_ROOT, dataset)
        entry, _ = matched_source.input_config(recorded, PROJECT_ROOT)
    else:
        yaml_cfg = runner.load_yaml(yaml_path)
        entry = {**yaml_cfg, **yaml_cfg["datasets"][dataset]}
    for key in ("reviews_path", "metadata_path", "interactions_path", "embeddings_path"):
        if entry.get(key):
            required.append(abspath(entry[key]))
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"[{dataset}] missing inputs:\n{formatted}")
    if matched_source.selected(dataset, args):
        recorded = matched_source.load_recorded(PROJECT_ROOT, dataset)
        data, _ = matched_source.input_config(recorded, PROJECT_ROOT)
        missing = [Path(value) for key, value in data.items()
                   if key.endswith("_path") and value and not Path(value).is_file()]
        if missing:
            raise FileNotFoundError(f"[{dataset}] missing recorded trial-{matched_source.trial_number(dataset)} inputs: {missing}")


# --------------------------------------------------------------------------- #
# Model construction from the tuned checkpoint
# --------------------------------------------------------------------------- #
ARCH_KEYS = ("codebook_size", "sem_id_len", "embedding_dim", "num_heads", "num_layers",
             "dim_feedforward", "max_position_embeddings", "semantic_vocab_size",
             "unified_vocab_size")


def tuned_checkpoint_paths(dataset: str, args: argparse.Namespace | None = None) -> dict[str, Path]:
    if matched_source.selected(dataset, args):
        return matched_source.trial_paths(PROJECT_ROOT, dataset)
    gpt2_dir = abspath(DATASETS[dataset]["gpt2_dir"])
    if dataset in matched_source.TRIALS:
        # Canonical Yambda bases now hold the selected matched trials. Legacy
        # profiles must keep their original weights and preprocessing together.
        gpt2_dir = gpt2_dir / "full_history"
        return {
            "checkpoint": abspath(gpt2_dir / "best.pt"),
            "sidecar": abspath(gpt2_dir / "best.pt.json"),
            "best_trial": gpt2_dir / "best_trial.json",
        }
    return {
        "checkpoint": gpt2_dir / "best.pt",
        "sidecar": gpt2_dir / "best.pt.json",
        "best_trial": abspath(gpt2_dir / "best_trial.json"),
    }


def tuned_profile(dataset: str, args: argparse.Namespace | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """gpt2_model.yaml-style profile for the tuned checkpoint: architecture
    from its own model_config (the promoted checkpoint need not have a
    sidecar), dropout from the tuner's best_trial.json best_params."""
    paths = tuned_checkpoint_paths(dataset, args)
    if dataset == "yambda_500m_listens" and matched_source.selected(dataset, args):
        # A finite earlier best.pt may survive the later NaN. Inspect actual
        # weights on CPU; a valid JSON sidecar cannot establish their safety.
        metadata = matched_source.load_finite_checkpoint(paths["checkpoint"])
        if paths["sidecar"].is_file():
            sidecar = load_json(paths["sidecar"])
            if sidecar.get("model_config") != metadata.get("model_config"):
                raise RuntimeError("Trial 4 best.pt model_config differs from its sidecar")
    else:
        metadata = (load_json(paths["sidecar"]) if paths["sidecar"].is_file()
                    else torch.load(paths["checkpoint"], map_location="cpu", weights_only=False))
    model_config = metadata["model_config"]
    if matched_source.selected(dataset, args):
        recorded = matched_source.load_recorded(PROJECT_ROOT, dataset)
        if model_config != recorded["model_config"]:
            raise RuntimeError(f"Trial {matched_source.trial_number(dataset)} best.pt model_config differs from its resolved_config.json")
        params = recorded["hparams"]
    else:
        best = load_json(paths["best_trial"])
        params = best.get("best_params", {})
    if "dropout" not in params:
        raise KeyError(f"{paths['best_trial']} lacks best_params.dropout")
    dropout = float(params["dropout"])
    profile = {
        "model": {
            "embedding_dim": int(model_config["embedding_dim"]),
            "codebook_size": int(model_config["codebook_size"]),
            "num_heads": int(model_config["num_heads"]),
            "num_layers": int(model_config["num_layers"]),
            "dim_feedforward": int(model_config["dim_feedforward"]),
            "max_position_embeddings": int(model_config["max_position_embeddings"]),
            "dropout": dropout,
            "attention_dropout": dropout,
            "residual_dropout": dropout,
            "embedding_dropout": dropout,
        }
    }
    return profile, model_config


def build_gpt2_from_tuned_checkpoint(dataset: str, dm, args: argparse.Namespace | None = None) -> tuple[Any, dict[str, Any]]:
    """Rebuild the tuned architecture through run_gpt2_sft_all.build_model
    (exp1 defaults: shared user token, fast greedy generation, beam 20),
    verify it against the saved config, then strict-load the weights."""
    profile, sidecar_cfg = tuned_profile(dataset, args)
    model_yaml = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    model = gpt2_runner.build_model(model_yaml, profile, dm)
    built = model.config_dict()
    mismatch = {k: (built.get(k), sidecar_cfg.get(k)) for k in ARCH_KEYS
                if k in sidecar_cfg and built.get(k) != sidecar_cfg.get(k)}
    if mismatch:
        raise RuntimeError(
            f"[{dataset}] rebuilt GPT-2 differs from the checkpoint model_config "
            f"(built, saved): {mismatch}. The datamodule catalog / semantic "
            "table does not match the tuned checkpoint."
        )
    gpt2_beauty.load_gpt2_checkpoint(model, tuned_checkpoint_paths(dataset, args)["checkpoint"])
    gpt2_beauty.check_semantic_row_table(model, dm)
    return model, profile


# --------------------------------------------------------------------------- #
# Reward caches (one per component per dataset)
# --------------------------------------------------------------------------- #
def _smoke_examples(args: argparse.Namespace) -> int:
    # Must cover every example the smoke training pass touches: 8 unshuffled
    # train batches plus the (unshuffled, in-smoke) calibration batch.
    return int(args.batch_size) * 8


def ensure_component_cache(
    dataset: str, key: str, run_config: Mapping[str, Any], config_dir: Path,
    num_subset_examples: int, args: argparse.Namespace, *, prepared_data: Any = None,
) -> dict[str, Any]:
    factorized = uses_factor_cache(dataset, args)
    cache_dir = component_cache_dir(dataset, key, args)
    manifest_path = cache_dir / "manifest.json"
    config_path = config_dir / f"cache_config_{key}.json"
    spec = component_spec_from_config(run_config, key)
    # The factor builder and the training loader must use the identical subset.
    subset = finetune_subset_spec(args)
    factor_name = "similarity" if key == "sim_hier" else key

    def check(manifest):
        from flowrec.data.dense_prefix import dense_prefix_signature

        expected_context = dense_prefix_signature(run_config["dataset"])
        if manifest.get("training_context") != expected_context:
            raise RuntimeError(f"Reward cache at {cache_dir} has a different training_context. "
                               "Rebuild with --force-cache.")
        for field, expected in subset.items():
            if manifest.get(field) != expected:
                raise RuntimeError(f"Reward cache at {cache_dir} has {field}={manifest.get(field)!r}, "
                                   f"expected {expected!r}. Rebuild with --force-cache.")
        if factorized:
            factor_runner.check_component_manifest(
                manifest, cache_dir, factor_name, num_subset_examples, args,
                similarity_mode=similarity_mode_of(key))
            return manifest
        return _check_component_manifest(manifest, cache_dir, key, spec, num_subset_examples, args)

    write_json(config_path, cache_config(run_config, key))
    with control.file_lock(cache_dir.parent / f"{cache_dir.name}.build.lock"):
        if manifest_path.exists() and not args.force_cache:
            return check(load_json(manifest_path))
        if args.require_existing_caches:
            raise RuntimeError(f"Required reward cache is missing at {cache_dir}; "
                               "--require-existing-caches forbids recalculation.")
        cmd = [sys.executable, str(SCRIPTS_DIR / ("precompute_reward_factors.py" if factorized
                                                else "precompute_reward_cache.py")),
               "--config", str(config_path), "--output-dir", str(cache_dir), "--split", "train",
               "--batch-size", str(int(args.cache_batch_size)), "--num-workers", str(int(args.num_workers)),
               "--device", str(args.device)]
        if factorized:
            cmd += ["--hidden-dtype", str(args.cache_dtype), "--verify-batches", str(args.cache_verify_batches)]
        else:
            cmd += ["--dtype", str(args.cache_dtype)]
        if cache_last_n(args) is not None:
            cmd += ["--per-user-last-n", str(cache_last_n(args))]
        if int(args.train_random_n) > 0:
            cmd += ["--per-user-random-n", str(args.train_random_n),
                    "--per-user-random-seed", str(args.train_subset_seed)]
        if subset["per_user_pool_last_n"] is not None:
            cmd += ["--per-user-pool-last-n", str(subset["per_user_pool_last_n"])]
        if args.smoke:
            cmd += ["--max-examples", str(_smoke_examples(args))]
        if factorized and prepared_data is not None:
            import random
            import numpy as np
            import precompute_reward_factors

            print(f"[{dataset}/{key}] precomputing reward factors with the already prepared data "
                  "in this process (no second dataset load)", flush=True)
            # Preserve the caller's RNG just as the former subprocess did.
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            try:
                with torch.random.fork_rng(devices=cuda_devices):
                    precompute_reward_factors.main(cmd[2:], prepared_data=prepared_data)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        else:
            print(f"[{dataset}/{key}] precomputing reward {'factors' if factorized else 'cache'}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        write_json(cache_dir / "component.json", {"key": key, "spec": spec})
        manifest = load_json(manifest_path)
    return check(manifest)


def _check_component_manifest(
    manifest: Mapping[str, Any], cache_dir: Path, key: str, spec: Mapping[str, Any],
    num_subset_examples: int, args: argparse.Namespace,
) -> dict[str, Any]:
    names = [c.get("name") for c in manifest.get("components", [])]
    if names != [spec["name"]]:
        raise RuntimeError(
            f"Reward cache at {cache_dir} stores components {names}, expected "
            f"[{spec['name']!r}] (a single raw component). Rebuild with --force-cache."
        )
    marker_path = cache_dir / "component.json"
    if marker_path.exists():
        marker = load_json(marker_path)
        stored_mode = marker.get("spec", {}).get("reward", {}).get("mode")
        if marker.get("key") != key or stored_mode != spec.get("reward", {}).get("mode"):
            raise RuntimeError(
                f"Reward cache at {cache_dir} was built for component "
                f"{marker.get('key')!r} (mode {stored_mode!r}), not {key!r}. "
                "Rebuild with --force-cache."
            )
    # Cache rows are positions in the fine-tuning subset, so a cache of another
    # subset is silently misaligned (verify_cache_alignment only sees the
    # first batch, where a last-N subset and the full history can agree).
    stored_last_n = manifest.get("per_user_last_n")
    stored_last_n = int(stored_last_n) if stored_last_n else None
    if stored_last_n != cache_last_n(args):
        def describe(n: int | None) -> str:
            return "the full user history" if n is None else f"each user's last {n} examples"
        raise RuntimeError(
            f"Reward cache at {cache_dir} covers {describe(stored_last_n)}, but this run "
            f"fine-tunes on {describe(cache_last_n(args))} (--train-last-n "
            f"{args.train_last_n}). Use a separate --cache-root or rebuild with --force-cache."
        )
    cached = int(manifest.get("num_examples", 0))
    expected = _smoke_examples(args) if args.smoke else num_subset_examples
    if cached < min(expected, num_subset_examples):
        raise RuntimeError(
            f"Reward cache at {cache_dir} covers {cached} examples but the fine-tuning "
            f"subset has {num_subset_examples}. Rebuild with --force-cache (config drift?)."
        )
    return dict(manifest)


def components_for(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Concrete component keys used by the selected grid entries, in canonical order."""
    used = {key for entry in entries for key in entry["components"]}
    return [key for key in CONCRETE_COMPONENTS if key in used]


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def parse_component_combo(key: str) -> tuple[str, ...]:
    """'interaction_x_sim_hier' -> concrete components in canonical order."""
    remaining, found = str(key), []
    for name in sorted(CONCRETE_COMPONENTS, key=len, reverse=True):   # sim_embed before shorter names
        if name in remaining.split("_x_"):
            found.append(name)
    parts = remaining.split("_x_")
    if not parts or sorted(parts) != sorted(found) or len(set(parts)) != len(parts):
        raise SystemExit(f"--component-combos {key!r}: expected components of {list(CONCRETE_COMPONENTS)} "
                         "joined by '_x_'")
    return tuple(name for name in CONCRETE_COMPONENTS if name in parts)


def combo_shapes(args: argparse.Namespace) -> list[tuple[tuple[str, ...], str | None, tuple[str, ...]]]:
    """(combo, similarity mode, concrete components) of every reward shape."""
    explicit = getattr(args, "component_combos", None)
    if explicit:
        shapes = []
        for key in explicit:
            components = parse_component_combo(key)
            modes = [similarity_mode_of(c) for c in components if similarity_mode_of(c)]
            shapes.append((components, "+".join(modes) if modes else None, components))
        return shapes
    combos = [c for c in all_combos() if combo_key(c) in set(args.combos)]
    return [(tuple(combo), mode, concrete_components(combo, mode))
            for combo in combos
            # Combos with `similarity` fan out over the similarity modes.
            for mode in (list(args.similarity_modes) if "similarity" in combo else [None])]


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    rhos = args.rho_targets or [args.gflow_target_grad_ratio]
    grid: list[dict[str, Any]] = []
    for dataset in args.datasets:  # dataset-major: shared datamodule/teacher/caches
        for sampler in (getattr(args, "samplers", None) or [DEFAULT_SAMPLER]):
            for combo, mode, components in combo_shapes(args):
                for weights in weight_assignments(len(combo), args.weight_levels):
                    for power in args.powers:
                        for rho in rhos:
                            for objective in args.objectives:
                                for k in args.num_candidates:
                                    grid.append({
                                        "dataset": str(dataset),
                                        "sampler": str(sampler),
                                        "combo": tuple(combo),
                                        "similarity_mode": mode,
                                        "components": components,
                                        "weights": tuple(float(w) for w in weights),
                                        "composition": str(args.composition),
                                        "power": float(power),
                                        "rho": float(rho),
                                        "objective": str(objective),
                                        "num_negatives": int(k),
                                    })
    return grid


def _sampler_suffix(entry: Mapping[str, Any]) -> str:
    # The teacher sampler keeps the plain layout so finished runs are found.
    sampler = str(entry.get("sampler", DEFAULT_SAMPLER))
    return "" if sampler == DEFAULT_SAMPLER else f"_neg-{sampler}"


def needs_teacher(entries: Sequence[Mapping[str, Any]]) -> bool:
    """The SASRec teacher model is loaded only for the teacher sampler or a
    combo with the sasrec reward."""
    return any(str(e.get("sampler", DEFAULT_SAMPLER)) == DEFAULT_SAMPLER or "sasrec" in tuple(e["components"])
               for e in entries)


def _rho_suffix(rho: float) -> str:
    return "" if float(rho) == 1.0 else f"_rho{gflow_ablation.power_tag(rho)}"


def _composition_dir(composition: str) -> str:
    # multiply is the default composition and keeps the plain layout.
    return "" if composition == "multiply" else f"mixed_"


def run_dir_for(output_root: Path, entry: Mapping[str, Any], seed: int) -> Path:
    return (
        output_root / entry["dataset"]
        / f"{_composition_dir(entry['composition'])}{combo_key(entry['components'])}{_sampler_suffix(entry)}"
        / f"w{weight_tag(entry['weights'])}" / f"pow_{gflow_ablation.power_tag(entry['power'])}"
        / f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}"
        / f"seed_{seed}"
    )


def entry_tag(entry: Mapping[str, Any]) -> str:
    return (f"{entry['dataset']}/{_composition_dir(entry['composition'])}"
            f"{combo_key(entry['components'])}{_sampler_suffix(entry)}"
            f"/w{weight_tag(entry['weights'])}/p{gflow_ablation.power_tag(entry['power'])}/"
            f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}")


def entry_identity(entry: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "dataset": entry["dataset"],
        "combo": combo_key(entry["components"]),
        "sampler": str(entry.get("sampler", DEFAULT_SAMPLER)),
        "similarity_mode": entry.get("similarity_mode") or "-",
        "weights": weight_tag(entry["weights"]),
        "composition": entry["composition"],
        "power": float(entry["power"]),
        "objective": entry["objective"],
        "num_negatives": int(entry["num_negatives"]),
        "seed": int(seed),
        "target_grad_ratio": float(entry["rho"]),
    }


# --------------------------------------------------------------------------- #
# Run status + cooperative-worker claims (<run_dir>/CLAIMED, flock-serialized)
# --------------------------------------------------------------------------- #
CLAIM_FILENAME = "CLAIMED"
RESULT_FILENAME = "result.json"
RUN_STATUSES = ("completed", "failed", "running", "stale-claim", "pending")


def load_result(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / RESULT_FILENAME
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def run_completed(run_dir: Path) -> bool:
    result = load_result(run_dir)
    return bool(result) and result.get("status") == "completed"


def check_existing_training_context(run_dir: Path, args: argparse.Namespace) -> None:
    """Do not skip or overwrite an old sliding-prefix run in a custom directory."""
    path = run_dir / "resolved_config.json"
    if not path.exists():
        if run_completed(run_dir) and (getattr(args, "prefix_from_dense", False)
                                      or getattr(args, "yambda500_targets", "last128") == "sampled50"):
            raise RuntimeError(f"Cannot verify training context at {run_dir}; choose a separate --output-root")
        return
    saved = load_json(path)
    if bool(saved.get("cli_args", {}).get("prefix_from_dense", False)) != bool(args.prefix_from_dense):
        raise RuntimeError(f"Different training context at {run_dir}; choose a separate --output-root")
    if bool(saved.get("cli_args", {}).get("yambda50_all_targets", False)) != bool(args.yambda50_all_targets):
        raise RuntimeError(f"Different target selection at {run_dir}; choose a separate --output-root")
    if saved.get("cli_args", {}).get("yambda500_targets", "last128") != getattr(args, "yambda500_targets", "last128"):
        raise RuntimeError(f"Different target selection at {run_dir}; choose a separate --output-root")
    for field in ("yambda50_base", "yambda500_base"):
        if saved.get("cli_args", {}).get(field, "legacy") != getattr(args, field, "legacy"):
            raise RuntimeError(f"Different initial checkpoint at {run_dir}; choose a separate --output-root")


class ClaimStore:
    """Atomic per-run claims for cooperative workers (the yambda-500m tuner's
    TrialStore pattern): a CLAIMED file inside the run dir, written under one
    flock so two workers never claim the same run; its mtime is the heartbeat
    (refreshed from the trainer logger), so a claim left behind by a killed
    worker becomes stale after ``stale_claim_hours`` and is re-claimable."""

    def __init__(self, output_root: Path, *, worker_id: int | None, stale_claim_hours: float) -> None:
        self.output_root = output_root
        self.lock_path = output_root / ".claims.lock"
        self.worker_id = worker_id
        self.stale_claim_seconds = float(stale_claim_hours) * 3600.0
        self._last_heartbeat = 0.0

    @staticmethod
    def claim_path(run_dir: Path) -> Path:
        return run_dir / CLAIM_FILENAME

    def _is_stale(self, claim: Path) -> bool:
        try:
            return time.time() - claim.stat().st_mtime > self.stale_claim_seconds
        except OSError:
            return True

    def status_of(self, run_dir: Path) -> str:
        result = load_result(run_dir)
        if result is not None and result.get("status") == "completed":
            return "completed"
        claim = self.claim_path(run_dir)
        if claim.exists():
            return "running" if not self._is_stale(claim) else "stale-claim"
        if result is not None and result.get("status") == "failed":
            return "failed"
        return "pending"

    def try_claim(self, run_dir: Path, *, retry_failed: bool) -> bool:
        with control.file_lock(self.lock_path):
            status = self.status_of(run_dir)
            if status in {"completed", "running"} or (status == "failed" and not retry_failed):
                return False
            self._write_claim(run_dir)
            return True

    def _write_claim(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.claim_path(run_dir).write_text(json.dumps({
            "worker_id": self.worker_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "claimed_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        self._last_heartbeat = time.time()

    def heartbeat(self, run_dir: Path, *, min_interval: float = 60.0) -> None:
        now = time.time()
        if now - self._last_heartbeat < min_interval:
            return
        claim = self.claim_path(run_dir)
        try:
            if claim.exists():
                os.utime(claim, None)
            else:
                self._write_claim(run_dir)
        except OSError:
            pass
        self._last_heartbeat = now

    def release(self, run_dir: Path) -> None:
        try:
            self.claim_path(run_dir).unlink()
        except FileNotFoundError:
            pass

    def reset_claims(self, run_dirs: Sequence[Path]) -> int:
        removed = 0
        with control.file_lock(self.lock_path):
            for run_dir in run_dirs:
                claim = self.claim_path(run_dir)
                if claim.exists() and not run_completed(run_dir):
                    claim.unlink()
                    removed += 1
        return removed


def heartbeat_logger(base_logger, store: ClaimStore | None, run_dir: Path):
    """Wrap the status logger so every trainer event refreshes the claim."""
    if store is None:
        return base_logger

    def logger(payload: dict[str, Any]) -> None:
        store.heartbeat(run_dir)
        base_logger(payload)

    return logger


def performance_logger(base_logger, *, tag: str, run_dir: Path,
                       micro_batch: int, device: torch.device):
    """Expose steady-state training speed and allocator usage between epochs.

    Rates exclude validation/setup: the first train event of each epoch only
    anchors the timer. GPU peaks are process-wide (including calibration and
    evaluation); allocated/reserved memory is PyTorch's, not GPU utilization.
    """
    previous = None

    def logger(payload: dict[str, Any]) -> None:
        nonlocal previous
        base_logger(payload)
        event = payload.get("type")
        if event not in {"train_step", "epoch_end"}:
            previous = None
            return
        record = {key: payload[key] for key in (
            "type", "epoch", "batch_idx", "global_step", "optimizer_step", "epoch_time_sec",
            "train/data_wait_sec", "train/step_wall_sec", "train/reward_prepare_sec", "train/examples_per_sec",
        ) if key in payload}
        record["micro_batch"] = micro_batch
        if event == "train_step":
            now = time.perf_counter()
            epoch, batch = int(payload["epoch"]), int(payload["batch_idx"])
            if previous is not None and previous[0] == epoch and batch > previous[1]:
                seconds = now - previous[2]
                if seconds > 0:
                    record["micro_steps_per_sec"] = (batch - previous[1]) / seconds
                    record["examples_per_sec"] = micro_batch * record["micro_steps_per_sec"]
            previous = (epoch, batch, now)
        else:
            previous = None
        if device.type == "cuda":
            for name, fn in (
                ("allocated_gib", torch.cuda.memory_allocated),
                ("reserved_gib", torch.cuda.memory_reserved),
                ("peak_allocated_gib", torch.cuda.max_memory_allocated),
                ("peak_reserved_gib", torch.cuda.max_memory_reserved),
            ):
                record[name] = fn(device) / 2**30
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "performance.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        details = " | ".join(f"{key}={record[key]:.3f}" for key in (
            "examples_per_sec", "epoch_time_sec", "allocated_gib", "reserved_gib",
            "peak_allocated_gib", "peak_reserved_gib",
        ) if key in record)
        print(f"[{tag}] performance {event} epoch={int(payload['epoch']) + 1} "
              f"step={payload['global_step']}" + (f" | {details}" if details else ""), flush=True)

    return logger


# Cache-phase slot order shared with the Slurm launchers (array index ->
# (dataset, component)); sim_exact is opt-in and built with --cache-only.
CACHE_SLOT_COMPONENTS = ("sasrec", "interaction", "sim_hier", "sim_embed")
CACHE_SLOTS = [(ds, key) for ds in DATASETS for key in CACHE_SLOT_COMPONENTS]


def print_status(grid: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, dict[str, int]]:
    output_root = abspath(args.output_root)
    store = ClaimStore(output_root, worker_id=None, stale_claim_hours=float(args.stale_claim_hours))
    counts: dict[str, dict[str, int]] = {}
    for entry in grid:
        status = store.status_of(run_dir_for(output_root, entry, int(args.seed)))
        counts.setdefault(entry["dataset"], {s: 0 for s in RUN_STATUSES})[status] += 1
    total = {s: 0 for s in RUN_STATUSES}
    for dataset, per in counts.items():
        print(f"{dataset:<12} " + " ".join(f"{s}={per[s]}" for s in RUN_STATUSES))
        for s in RUN_STATUSES:
            total[s] += per[s]
    print("total        " + " ".join(f"{s}={total[s]}" for s in RUN_STATUSES) + f" grid={len(grid)}")
    # Workers can pick up pending + stale-claim (+ failed with --retry-failed).
    claimable = total["pending"] + total["stale-claim"] + (total["failed"] if args.retry_failed else 0)
    print(f"claimable={claimable}")
    return counts


def print_cache_status(grid: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> int:
    missing = 0
    for dataset in args.datasets:
        entries = [e for e in grid if e["dataset"] == dataset]
        if not entries:
            continue
        for key in components_for(entries):
            cache_dir = component_cache_dir(dataset, key, args)
            exists = (cache_dir / "manifest.json").exists()
            slot = CACHE_SLOTS.index((dataset, key)) if (dataset, key) in CACHE_SLOTS else -1
            print(f"{'OK     ' if exists else 'MISSING'} {slot:2d} {dataset} {key} {cache_dir}")
            missing += 0 if exists else 1
    print(f"missing={missing}")
    return missing


# --------------------------------------------------------------------------- #
# Metrics CSV (file-locked upsert, one row per run x split)
# --------------------------------------------------------------------------- #
CSV_KEY = ("dataset", "combo", "weights", "composition", "power", "objective",
           "num_negatives", "seed", "variant", "split", "target_grad_ratio", "sampler")


def upsert_metrics_csv(csv_path: Path, new_rows: list[dict[str, Any]]) -> None:
    import csv as _csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with control.file_lock(csv_path.with_suffix(".lock")):
        rows: list[dict[str, Any]] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = [dict(r) for r in _csv.DictReader(fh)]

        def with_sampler(row: Mapping[str, Any]) -> dict[str, Any]:
            # Rows written before the sampler dimension existed.
            row = dict(row)
            if not row.get("sampler"):
                row["sampler"] = DEFAULT_SAMPLER if row.get("variant") == "gflow" else "-"
            return row

        def key_of(row: Mapping[str, Any]) -> tuple[str, ...]:
            return tuple(str(row.get(k, "")) for k in CSV_KEY)

        merged = {key_of(r): r for r in map(with_sampler, rows)}
        for row in map(with_sampler, new_rows):
            merged[key_of(row)] = {k: ("" if v is None else v) for k, v in row.items()}
        all_rows = list(merged.values())
        fieldnames = list(CSV_KEY) + sorted({k for r in all_rows for k in r} - set(CSV_KEY))
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(all_rows, key=key_of):
                writer.writerow({k: row.get(k, "") for k in fieldnames})


# --------------------------------------------------------------------------- #
# Calibration (exp4 protocol through the GPT-2 method + composed cache)
# --------------------------------------------------------------------------- #
def reward_cache_for_prefetch(cache, args: argparse.Namespace):
    """Workers read raw factors only; legacy dense reward caches stay in-process."""
    if args.reward_prefetch and isinstance(cache, (FactorizedRewardCache, ComposedFactorizedRewardCache)):
        return cache
    return None


def input_pipeline_settings(args: argparse.Namespace, *, reward_cache=None) -> dict[str, Any]:
    return {
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.loader_prefetch_factor),
        "tensorized_train_loader": bool(args.tensorized_train_loader),
        "reward_prefetch": reward_cache_for_prefetch(reward_cache, args) is not None,
    }


def calibrate_gflow_weight(
    *, model, base_gflow_cfg: GFlowConfig, sampler, sft_cfg: SFTConfig,
    cache: ComposedRewardCache | ComposedFactorizedRewardCache,
    dm, subset_indices: list[int], device: torch.device, args: argparse.Namespace,
    tag: str, target_ratio: float, batch_size: int,
) -> dict[str, Any]:
    """lambda = rho_target / median_b(||g_GFlow,b|| / ||g_SFT,b||) at the init
    checkpoint, before any optimizer step (gflow_finetune_yambda50m protocol:
    sequential full-precision gradients, identical RNG per pass, step=0 =
    easy curriculum phase, weights/optimizer/loader order untouched). The
    reference uses the same SFT method directly, avoiding an otherwise
    discarded GFlow forward at zero weight. Each graph is freed before the
    next forward, while gradient norms/cosine retain the two-pass protocol.
    """
    from flowrec.ft_methods.calibration import calibrate_sequential_variants

    sft_variant = gflow_ablation.AliasedSFTMethod(sft_cfg)
    gflow_variant = make_method(
        replace(base_gflow_cfg, sft_weight=0.0, gflow_weight=1.0), sampler, sft_cfg, cache)

    params = [p for p in model.parameters() if p.requires_grad]
    loader = gflow_ablation.indexed_train_loader(
        dm, subset_indices, shuffle=not args.smoke, batch_size=int(batch_size),
        seed=int(args.gflow_calibration_seed), num_workers=0,
        reward_cache=reward_cache_for_prefetch(cache, args),
        prefetch_factor=int(args.loader_prefetch_factor),
        tensorized=bool(args.tensorized_train_loader),
    )
    num_batches = 1 if args.smoke else int(args.gflow_calibration_batches)

    was_training = model.training
    model.train()

    try:
        result = calibrate_sequential_variants(
            model, sft_variant, gflow_variant, loader, params,
            target_ratio=float(target_ratio), eps=float(args.gflow_calibration_eps),
            seed=int(args.gflow_calibration_seed), step=0,
            num_batches=num_batches, device=device,
        )
    finally:
        model.train(was_training)
        runner.seed_everything(int(args.seed))  # calibration must not shift training
    result["implementation"] = "sequential_variants_fp32"
    print(
        f"[{tag}] calibration: ||g_SFT||~{result['sft_grad_norm_median']:.4g} "
        f"||g_GFlow||~{result['gflow_grad_norm_median']:.4g} "
        f"r0={result['raw_grad_ratio_median']:.4g} cos={result['grad_cosine_median']:.3f} "
        f"rho_target={result['target_grad_ratio']} -> lambda={result['fixed_gflow_weight']:.6g}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# Init baseline (once per dataset, file-locked)
# --------------------------------------------------------------------------- #
def ensure_init_baseline(
    *, dataset: str, dm, device: torch.device, args: argparse.Namespace,
    output_root: Path, csv_path: Path,
) -> None:
    baseline_path = output_root / dataset / "baseline" / "init_metrics.json"
    if baseline_path.exists() or args.skip_init_baseline:
        return
    with control.file_lock(baseline_path.with_suffix(".lock")):
        if baseline_path.exists():
            return
        print(f"[{dataset}] evaluating the untouched init checkpoint ...", flush=True)
        model, _ = build_gpt2_from_tuned_checkpoint(dataset, dm, args)
        model.to(device).eval()
        max_eval = 2 if args.smoke else args.max_eval_batches
        splits = gpt2_beauty.evaluate_splits(model, dm, args, device, max_eval,
                                             splits=("validation", "test"))
        payload = {
            "dataset": dataset,
            "checkpoint": str(tuned_checkpoint_paths(dataset, args)["checkpoint"]),
            **splits,
        }
        write_json(baseline_path, payload)
        identity = {"dataset": dataset, "combo": "-", "weights": "-", "composition": "-",
                    "power": "-", "objective": "-", "num_negatives": "-",
                    "seed": int(args.seed), "variant": "init", "target_grad_ratio": "-"}
        upsert_metrics_csv(csv_path, gflow_ablation.metric_rows(
            identity, splits, {"status": "completed"}))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def light_validation_sft_cfg() -> SFTConfig:
    """In-training validation stays light: constrained beam-5 generation and
    only the NDCG@5 monitor (val/sft/ndcg@5 via the alias). The full metric
    set runs on validation + test at the end."""
    return SFTConfig(
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=False,
        compute_item_metrics=False,
        evaluate_generation=True,
        generation_topk=(5,),
        generation_num_beams=5,
        generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
        generation_exclude_seen_items=False,
    )


def build_trainer(model, method, *, tag: str, run_dir: Path, run_accum: int,
                  args: argparse.Namespace, device: torch.device,
                  claims: ClaimStore | None = None) -> Trainer:
    """The SFT-control fine-tuning protocol (lr, clipping, AMP, monitor,
    early stopping) — shared by the GFlow runs and the SFT-only control."""
    max_epochs = 1 if args.smoke else int(args.max_epochs)
    max_train_batches = 8 if args.smoke else args.max_train_batches
    max_eval_batches = 2 if args.smoke else args.max_eval_batches
    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=int(run_accum),
            max_grad_norm=float(args.max_grad_norm),
            use_amp=device.type == "cuda",
            amp_dtype=str(args.amp_dtype),
            validate_every_epochs=int(args.validate_every),
            log_every_steps=200,
            monitor="val/sft/ndcg@5",
            monitor_mode="max",
            early_stopping_patience=int(args.patience),
            early_stopping_min_epochs=int(args.min_epochs),
            run_validation_at_start=bool(args.validate_at_start),
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            val_sample_fraction=args.val_sample_fraction,
            val_sample_resample=args.val_sample_resample,
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        logger=heartbeat_logger(performance_logger(
            gflow_ablation.make_status_logger(tag), tag=tag, run_dir=run_dir,
            micro_batch=int(args.batch_size) * int(args.grad_accum) // int(run_accum),
            device=device,
        ), claims, run_dir),
    )


def resumable_checkpoint(run_dir: Path, identity: Mapping[str, Any], num_finetune_examples: int,
                         args: argparse.Namespace) -> Path | None:
    """checkpoints/last.pt of an interrupted attempt of THIS run (same grid
    identity and fine-tuning subset per resolved_config.json), else None."""
    if not args.resume or args.force:
        return None
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    resolved_path = run_dir / "resolved_config.json"
    if not last_ckpt.exists() or not resolved_path.exists():
        return None
    resolved = load_json(resolved_path)
    same_run = all(resolved.get(k) == v for k, v in identity.items())
    same_data = (resolved.get("data", {}).get("num_finetune_examples") == int(num_finetune_examples))
    same_data = same_data and (bool(resolved.get("cli_args", {}).get("prefix_from_dense", False))
                              == bool(getattr(args, "prefix_from_dense", False)))
    same_data = same_data and (bool(resolved.get("cli_args", {}).get("yambda50_all_targets", False))
                              == bool(getattr(args, "yambda50_all_targets", False)))
    same_data = same_data and (resolved.get("cli_args", {}).get("yambda500_targets", "last128")
                              == getattr(args, "yambda500_targets", "last128"))
    same_data = same_data and all(
        resolved.get("cli_args", {}).get(field, "legacy") == getattr(args, field, "legacy")
        for field in ("yambda50_base", "yambda500_base"))
    if getattr(args, "train_pool_last_n", None) is not None:
        expected_subset = {"finetune_per_user_pool_last_n": args.train_pool_last_n,
                           "finetune_per_user_last_n": cache_last_n(args),
                           "finetune_per_user_random_n": args.train_random_n,
                           "finetune_subset_seed": args.train_subset_seed}
        same_data = same_data and all(resolved.get("data", {}).get(k) == v for k, v in expected_subset.items())
    if not (same_run and same_data):
        return None
    try:  # torch.save is not atomic: a worker killed mid-write leaves a truncated file
        torch.load(last_ckpt, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint means "start over"
        print(f"[resume] {last_ckpt} is unreadable ({type(exc).__name__}: {exc}); "
              "restarting the run from the init checkpoint", flush=True)
        return None
    return last_ckpt


def resume_trainer(trainer: Trainer, last_ckpt: Path, *, tag: str) -> int:
    """Restore an interrupted run; returns the last finished epoch. last.pt is
    written at the END of an epoch with state.epoch = that epoch, and fit()
    starts at state.epoch, so advance it — otherwise the finished epoch (and,
    on a validation epoch, its patience update) would be repeated. The
    history record of that last epoch is written after last.pt and is the
    only thing a resume loses."""
    trainer.load_checkpoint(last_ckpt, load_optimizer=True, load_scheduler=True)
    finished = int(trainer.state.epoch)
    trainer.state.epoch = finished + 1
    print(f"[{tag}] resumed from {last_ckpt}: {finished + 1} epochs done, best "
          f"{trainer.state.best_metric} (epoch {trainer.state.best_epoch}), bad validations "
          f"{trainer.state.bad_epochs}, stop={trainer.state.should_stop}", flush=True)
    return finished


def resume_sampler_curriculum(sampler: SemanticIDCandidateSampler, trainer: Trainer,
                              last_ckpt: Path, *, tag: str) -> None:
    """Keep the same easy/hard mixture when micro-batch changes on resume.

    The trainer restores its actual cumulative micro-steps. An epoch-based
    ramp may now contain fewer steps, so translate sampler progress only.
    The saved offset also makes repeated resumes (and batch decreases) safe.
    """
    if sampler.cfg.mode != "curriculum":
        return
    saved = load_json(last_ckpt.parent.parent / "resolved_config.json").get("sampler", {})
    old_steps = int(saved.get("curriculum_steps", sampler.cfg.curriculum_steps))
    new_steps = int(sampler.cfg.curriculum_steps)
    if old_steps <= 0 or new_steps <= 0:
        return
    step = int(trainer.state.global_step)
    old_offset = int(saved.get("curriculum_step_offset", 0))
    new_offset = round((step + old_offset) * new_steps / old_steps) - step
    sampler.cfg = replace(sampler.cfg, curriculum_step_offset=new_offset)
    print(f"[{tag}] resumed curriculum: steps={old_steps}->{new_steps}, "
          f"offset={new_offset}, alpha={sampler._curriculum_alpha(step):.6g}", flush=True)


def fit_and_evaluate(trainer: Trainer, model, dm, subset_indices: list[int], *, run_batch: int,
                     run_dir: Path, args: argparse.Namespace, device: torch.device,
                     resumed_from_epoch: int | None = None, train_loader=None, reward_cache=None):
    """Fine-tune on the subset (finetune_indices), reload best.pt, and run the
    beam-20 metric set on validation + test. ``train_loader`` replaces the
    indexed prefix-example loader (the dense-layout control passes its own)."""
    max_eval_batches = 2 if args.smoke else args.max_eval_batches
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    resumed = resumed_from_epoch is not None
    if resumed and (trainer.state.should_stop or trainer.state.epoch >= trainer.cfg.max_epochs):
        # Interrupted during the final evaluation: training had already ended.
        fit = SimpleNamespace(state=trainer.state, best_checkpoint_path=None,
                              last_checkpoint_path=str(last_path))
        fit.state.epoch = int(resumed_from_epoch)
    else:
        # The datamodule survives OOM retries; update the actual validation
        # loader's configuration as well as the attempt's CLI arguments.
        dm.cfg = replace(dm.cfg, eval_batch_size=int(args.val_batch_size))
        if train_loader is None:
            train_loader = gflow_ablation.indexed_train_loader(
                dm, subset_indices, shuffle=not args.smoke,
                batch_size=int(run_batch), seed=int(args.seed),
                reward_cache=reward_cache_for_prefetch(reward_cache, args),
                prefetch_factor=int(args.loader_prefetch_factor),
                tensorized=bool(args.tensorized_train_loader),
            )
        fit = trainer.fit(train_loader, dm.val_dataloader())
    write_json(run_dir / "training_history.json", {"history": fit.state.history})
    # fit() only reports a best.pt it wrote itself; after a resume the best
    # may predate the interruption.
    if fit.best_checkpoint_path is None and resumed and best_path.exists():
        fit = SimpleNamespace(state=fit.state, best_checkpoint_path=str(best_path),
                              last_checkpoint_path=fit.last_checkpoint_path)
    checkpoint_for_eval = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_eval:
        trainer.load_checkpoint(checkpoint_for_eval, load_optimizer=False, load_scheduler=False)
    splits = gpt2_beauty.evaluate_splits(model, dm, args, device, max_eval_batches,
                                         splits=("validation", "test"))
    write_json(run_dir / "final_metrics.json", splits)
    return fit, splits


def build_sampler(name: str, *, model, dm, sasrec, num_negatives: int, curriculum_steps: int,
                  args: argparse.Namespace) -> SemanticIDCandidateSampler:
    """K negatives + the positive, never the target or a batch-history item
    (same three samplers as the TIGER script, gflow_finetune_yambda500m.py)."""
    common = dict(num_negatives=int(num_negatives), include_positive=True, exclude_history=True)
    table = model.item_semantic_ids.detach()
    mapping = tiger_runner.item_id_to_semantic_row(dm)
    if name == "teacher":
        if sasrec is None:
            raise RuntimeError("the teacher sampler needs the SASRec teacher model")
        return SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="curriculum", curriculum_steps=int(curriculum_steps),
                                   temperature=float(args.sampler_temperature), **common),
            item_semantic_ids=table,
            item_id_to_semantic_row=tiger_runner.restrict_item_id_mapping_to_scorer_vocab(mapping, sasrec),
            scorer=sasrec,
        )
    if name == "random":
        return SemanticIDCandidateSampler(CandidateSamplerConfig(mode="random", **common),
                                          item_semantic_ids=table, item_id_to_semantic_row=mapping)
    if name == "policy":
        return SemanticIDCandidateSampler(
            # always on-policy (no curriculum, temperature 1), valid catalog items only
            CandidateSamplerConfig(mode="policy",
                                   policy_extra_beams=getattr(args, "policy_extra_beams", None), **common),
            item_semantic_ids=table, item_id_to_semantic_row=mapping,
        )
    raise ValueError(f"Unknown sampler {name!r}; expected one of {SAMPLERS}")


def run_single(
    *, entry: Mapping[str, Any], run_config: Mapping[str, Any], dm, subset_indices: list[int],
    sasrec, cache_dirs: Mapping[str, Path], run_dir: Path, csv_path: Path,
    args: argparse.Namespace, claims: ClaimStore | None = None, attempt: int = 0,
    resume_attempt: int = 0,
) -> dict[str, Any]:
    """``attempt`` counts OOM retries; only attempt ``resume_attempt`` (the
    first one of this process, see execute_entry) may resume from last.pt."""
    tag = entry_tag(entry)
    dataset = entry["dataset"]
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    runner.seed_everything(int(args.seed))
    model, profile = build_gpt2_from_tuned_checkpoint(dataset, dm, args)
    model.to(device)

    run_batch, run_accum = run_batch_settings(args, int(entry["num_negatives"]))
    if run_batch != int(args.batch_size):
        print(f"[{tag}] candidate-memory scaling: batch {args.batch_size} -> {run_batch}, "
              f"grad_accum {args.grad_accum} -> {run_accum}", flush=True)
    curriculum_steps = curriculum_steps_for_run(args, run_batch, len(subset_indices))
    sampler_name = str(entry.get("sampler", DEFAULT_SAMPLER))
    if args.curriculum_epochs is not None and sampler_name == "teacher":
        print(f"[{tag}] curriculum: {args.curriculum_epochs:g} epochs of {len(subset_indices)} examples "
              f"= {curriculum_steps} micro-steps at batch {run_batch}", flush=True)

    sampler = build_sampler(sampler_name, model=model, dm=dm, sasrec=sasrec,
                            num_negatives=int(entry["num_negatives"]),
                            curriculum_steps=curriculum_steps, args=args)

    sft_cfg = light_validation_sft_cfg()
    cache = compose_reward_cache(entry, cache_dirs, args)
    base_gflow_cfg = GFlowConfig(
        objective=str(entry["objective"]),
        sft_weight=1.0,
        gflow_weight=1.0,  # placeholder; calibrated below
        # Informational: the composed cache is injected directly (make_method);
        # the power is applied there, so the config keeps power 1.
        reward_cache_path=str(cache_dirs[entry["components"][0]]),
        reward_cache_power=1.0,
        tb_loss_weighting="uniform",
        candidate_sampler={"teacher": "sasrec"}.get(sampler_name, sampler_name),
        num_negative_samples=int(entry["num_negatives"]),
        easy_negative_steps=sampler.cfg.curriculum_steps,
        candidate_sampling_temperature=float(args.sampler_temperature),
    )

    calibration_path = run_dir / "calibration.json"
    last_ckpt = resumable_checkpoint(run_dir, entry_identity(entry, int(args.seed)),
                                     len(subset_indices), args) if attempt == resume_attempt else None
    if last_ckpt is not None and not calibration_path.exists():
        last_ckpt = None
    if last_ckpt is not None:
        # The weights moved since the calibration: keep the run's lambda.
        calibration = load_json(calibration_path)
        lambda_gflow = float(calibration["fixed_gflow_weight"])
        print(f"[{tag}] resuming: lambda={lambda_gflow:.6g} from {calibration_path}", flush=True)
    elif args.gflow_weight_mode == "fixed_grad_ratio":
        calibration = calibrate_gflow_weight(
            model=model, base_gflow_cfg=base_gflow_cfg, sampler=sampler, sft_cfg=sft_cfg,
            cache=cache, dm=dm, subset_indices=subset_indices, device=device, args=args,
            tag=tag, target_ratio=float(entry["rho"]), batch_size=run_batch,
        )
        lambda_gflow = float(calibration["fixed_gflow_weight"])
    else:
        calibration = {"mode": "constant", "fixed_gflow_weight": float(args.gflow_constant_weight)}
        lambda_gflow = float(args.gflow_constant_weight)
    write_json(calibration_path, calibration)

    method = make_method(replace(base_gflow_cfg, gflow_weight=lambda_gflow), sampler, sft_cfg, cache)
    trainer = build_trainer(model, method, tag=tag, run_dir=run_dir, run_accum=run_accum,
                            args=args, device=device, claims=claims)
    resumed_from_epoch = resume_trainer(trainer, last_ckpt, tag=tag) if last_ckpt is not None else None
    if last_ckpt is not None:
        resume_sampler_curriculum(sampler, trainer, last_ckpt, tag=tag)
    batching = {"batch_size": run_batch, "gradient_accumulation": run_accum,
                "effective_batch": run_batch * run_accum,
                "max_candidate_batch": int(args.max_candidate_batch), "oom_retries": int(attempt),
                "val_batch_size": int(args.val_batch_size), "test_batch_size": int(args.test_batch_size)}

    resolved = {
        **entry_identity(entry, int(args.seed)),
        "variant": "gflow",
        "model": model.config_dict(),
        "model_profile": profile,
        "init_checkpoint": str(tuned_checkpoint_paths(dataset, args)["checkpoint"]),
        "sasrec_checkpoint": run_config["method"]["sasrec"]["checkpoint_path"],
        "gflow": method.cfg.to_dict(),
        "gflow_weight_mode": str(args.gflow_weight_mode),
        "rho_target": float(entry["rho"]),
        "calibration": {k: v for k, v in calibration.items() if k != "per_batch"},
        "sampler": {
            "type": {"teacher": "sasrec_curriculum", "random": "uniform_catalog",
                     "policy": "model_stochastic_beam_search"}[sampler_name],
            "name": sampler_name, "mode": sampler.cfg.mode,
            "num_negatives": int(entry["num_negatives"]),
            "include_positive": True, "exclude_history": True,
            "curriculum_steps": sampler.cfg.curriculum_steps,
            "curriculum_step_offset": sampler.cfg.curriculum_step_offset,
            "curriculum_epochs": args.curriculum_epochs if sampler_name == "teacher" else None,
            "temperature": float(sampler.cfg.temperature),
            "on_policy": sampler_name == "policy", "valid_item_constraint": True,
            "policy_extra_beams": sampler.cfg.policy_extra_beams if sampler_name == "policy" else None,
        },
        "reward": {
            **cache.describe(),
            "power_applied": type(cache).__name__,
            "combo_shape": combo_key(entry["combo"]),
            "similarity_mode": entry.get("similarity_mode"),
            "component_specs": {key: component_spec_from_config(run_config, key)
                                for key in entry["components"]},
            "cache_dirs": {key: str(cache_dirs[key]) for key in entry["components"]},
        },
        "batching": batching,
        "input_pipeline": input_pipeline_settings(args, reward_cache=cache),
        "data": {
            **runner.data_config_paths(dm.cfg),
            "train_example_layout": "prefix_target",
            "finetune_per_user_last_n": cache_last_n(args),
            "finetune_subset": describe_finetune_subset(args),
            "finetune_per_user_pool_last_n": args.train_pool_last_n,
            "finetune_per_user_random_n": args.train_random_n,
            "finetune_subset_seed": args.train_subset_seed,
            "num_train_examples": len(dm.train_dataset),
            "num_finetune_examples": len(subset_indices),
            **dm.summary(),
        },
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "resumed_from_epoch": resumed_from_epoch,
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] fine-tuning: lambda={lambda_gflow:.6g}, epochs<={trainer.cfg.max_epochs}, "
          f"micro-batch {run_batch} x accum {run_accum}", flush=True)
    print(f"[{tag}] input pipeline: {json.dumps(resolved['input_pipeline'], sort_keys=True)}", flush=True)
    fit, splits = fit_and_evaluate(trainer, model, dm, subset_indices, run_batch=run_batch,
                                   run_dir=run_dir, args=args, device=device,
                                   resumed_from_epoch=resumed_from_epoch, reward_cache=cache)

    result = {
        **entry_identity(entry, int(args.seed)),
        "status": "completed",
        "resumed_from_epoch": resumed_from_epoch,
        "lambda_gflow": lambda_gflow,
        "raw_grad_ratio_median": calibration.get("raw_grad_ratio_median"),
        "grad_cosine_median": calibration.get("grad_cosine_median"),
        "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1),
        "elapsed_sec": time.time() - started,
        "batching": batching,
        "best_checkpoint": fit.best_checkpoint_path,
        "run_dir": str(run_dir),
        "splits": splits,
    }
    write_json(run_dir / "result.json", result)

    extra = {
        "status": "completed", "lambda_gflow": lambda_gflow,
        "raw_grad_ratio_median": result["raw_grad_ratio_median"],
        "grad_cosine_median": result["grad_cosine_median"],
        "best_epoch": result["best_epoch"], "best_val_ndcg5": result["best_val_ndcg5"],
        "epochs_completed": result["epochs_completed"], "run_dir": str(run_dir),
        "micro_batch": run_batch, "grad_accum": run_accum,
        "max_candidate_batch": int(args.max_candidate_batch), "oom_retries": int(attempt),
        "elapsed_sec": round(result["elapsed_sec"], 1),
    }
    upsert_metrics_csv(csv_path, gflow_ablation.metric_rows(
        {**entry_identity(entry, int(args.seed)), "variant": "gflow"}, splits, extra))

    del model, method, trainer, sampler, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def select_grid(grid: list[dict[str, Any]], index: str | None) -> list[dict[str, Any]]:
    if index is None:
        return grid
    text = str(index)
    try:
        if "-" in text:
            start, end = (int(x) for x in text.split("-", 1))
        else:
            start = end = int(text)
    except ValueError:
        raise SystemExit(f"--index must be 'N' or 'A-B', got {text!r}") from None
    if start < 0 or start > end:
        raise SystemExit(f"--index range {text!r} is invalid")
    if start >= len(grid):
        print(f"--index {text}: start beyond the grid ({len(grid)} entries) — nothing to do")
        return []
    end = min(end, len(grid) - 1)
    print(f"--index {text}: running {end - start + 1} grid entr{'y' if start == end else 'ies'} "
          f"[{start}..{end}]", flush=True)
    return grid[start:end + 1]


@dataclass
class DatasetContext:
    dataset: str
    run_config: dict[str, Any]
    dm: Any
    subset_indices: list[int]
    sasrec: Any
    cache_dirs: dict[str, Path]


def matched_teacher_translation(dataset: str, dm, args: argparse.Namespace) -> Path:
    """Reconstruct the teacher's own catalog and translate by raw item ID.

    The matched GPT-2 catalog is smaller than the legacy SASRec catalog.
    Equal-sized/overlapping integer IDs are not evidence of matching items.
    A locked shared artifact avoids rebuilding the teacher data per worker.
    """
    import run_sasrec_all
    from flowrec.data.yambda import YambdaDataModule

    checkpoint = abspath(DATASETS[dataset]["sasrec_checkpoint"])
    summary_path = checkpoint.parent / "training_summary.json"
    summary = load_json(summary_path)
    recorded_runs = [run["dataset"] for run in summary.get("runs", [])
                     if run.get("dataset", {}).get("name") == dataset]
    if not recorded_runs:
        raise ValueError(f"Cannot verify original SASRec preprocessing from {summary_path}")
    teacher_config = run_sasrec_all.build_yambda_data_config(
        dataset, runner.load_yaml(dataset_yaml_path(dataset)), seed=42)
    teacher_config = replace(teacher_config, num_workers=0, persistent_workers=False)
    stat = checkpoint.stat()
    catalog_hash = hashlib.sha256(json.dumps(sorted(dm.item2id.items())).encode()).hexdigest()
    identity = {"checkpoint": str(checkpoint), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "data_config": runner.to_jsonable(asdict(teacher_config)),
                "target_catalog_sha256": catalog_hash,
                "recorded_summary": recorded_runs[0]}
    directory = abspath(args.cache_root) / dataset / "source_assets"
    path = directory / "teacher_id_translation.json"
    with control.file_lock(path.with_suffix(".lock")):
        if path.is_file() and load_json(path).get("identity") == identity:
            return path
        teacher_dm = YambdaDataModule(teacher_config)
        teacher_dm.setup()
        actual_summary = teacher_dm.summary()
        # The promoted seed/trial models share preprocessing; all listed
        # summaries must agree with the reconstructed catalog and splits.
        for recorded in recorded_runs:
            for key in ("num_items", "num_users", "num_interactions", "num_train_examples",
                        "num_val_examples", "num_test_examples", "item_vocab_size"):
                if key in recorded and actual_summary.get(key) != recorded[key]:
                    raise RuntimeError(f"SASRec source-data reconstruction differs for {key}: "
                                       f"{actual_summary.get(key)!r} != {recorded[key]!r}")
        table = [0] * int(dm.item_vocab_size)
        missing = []
        for raw_item, item_id in dm.item2id.items():
            if int(item_id) == 0:
                continue
            source_id = teacher_dm.item2id.get(raw_item)
            if source_id is None or int(source_id) <= 0:
                missing.append(raw_item)
            else:
                table[int(item_id)] = int(source_id)
        if missing:
            raise RuntimeError(f"SASRec cannot score {len(missing)} matched catalog items; "
                               f"missing raw item IDs include {missing[:5]}")
        payload = {"identity": identity, "target_catalog_sha256": catalog_hash,
                   "teacher_item_vocab_size": int(teacher_dm.item_vocab_size),
                   "teacher_catalog_sha256": hashlib.sha256(
                       json.dumps(sorted(teacher_dm.item2id.items())).encode()).hexdigest(),
                   "item_id_translation": table}
        write_json(path, payload)
        del teacher_dm
    return path


def build_dataset_context(
    dataset: str, components: Sequence[str], *, args: argparse.Namespace,
    device: torch.device, output_root: Path, csv_path: Path, need_teacher: bool = True,
) -> DatasetContext | None:
    """Datamodule + sanity checks + SASRec teacher + component caches (+ the
    init baseline) for one dataset. Returns None after the cache phase when
    --cache-only. ``need_teacher=False`` (SFT-only control) skips the teacher
    and the caches."""
    validate_inputs(dataset, args, need_teacher=need_teacher, components=components)
    model_yaml = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    profile, _ = tuned_profile(dataset, args)
    user_hash_mod = gpt2_runner.user_ids_count(model_yaml, profile)
    run_config = build_run_config(dataset, args, user_hash_mod)
    config_dir = output_root / dataset
    config_path = config_dir / "data_reward_config.json"
    write_json(config_path, run_config)

    data_cfg = tiger_runner.build_data_config(dict(run_config), config_path=config_path)
    dm = tiger_runner.build_data_module(data_cfg)
    setup_started = time.perf_counter()
    dm.setup()
    print(f"[{dataset}] data setup: {time.perf_counter() - setup_started:.1f} seconds", flush=True)
    if dm.semantic_ids is None or dm.sid_len is None:
        raise ValueError(f"[{dataset}] datamodule produced no semantic IDs.")
    store = dm.interaction_store
    subset_indices = finetune_indices(dm.train_dataset, args)
    if matched_source.selected(dataset, args):
        audit = matched_source.audit_prefix_data(dm, matched_source.load_recorded(PROJECT_ROOT, dataset))
        write_json(config_dir / "matched_prefix_audit.json", {
            **audit, "num_finetune_examples": len(subset_indices),
            "target_selection": finetune_subset_spec(args),
        })
        print(f"[{dataset}] trial {matched_source.trial_number(dataset)} targets, full histories, catalog, semantic IDs and held-out data match",
              flush=True)
    elif getattr(args, "prefix_from_dense", False):
        write_json(config_dir / "dense_prefix_audit.json", {
            **dm.dense_prefix_audit, "training_context": dm.summary()["training_context"],
            "num_finetune_examples": len(subset_indices), "target_selection": finetune_subset_spec(args),
        })
        print(f"[{dataset}] dense/prefix validation and test examples match exactly", flush=True)
    print(
        f"[{dataset}] catalog={int(dm.item_vocab_size)} ids, train={len(dm.train_dataset)} "
        f"prefix examples (fine-tune subset: {len(subset_indices)} = "
        f"{describe_finetune_subset(args)}), "
        f"aux store={len(store) if store is not None else 0} events",
        flush=True,
    )

    # Refuse to fine-tune a checkpoint that does not match this datamodule.
    probe, _ = build_gpt2_from_tuned_checkpoint(dataset, dm, args)
    probe.to(device)
    if not args.skip_sanity_check:
        nll = gpt2_beauty.sanity_check_init_nll(
            probe, dm, device, threshold=float(args.sanity_nll_threshold))
        if not math.isfinite(nll):
            raise FloatingPointError(f"[{dataset}] init sanity produced non-finite val NLL: {nll}")
        print(f"[{dataset}] init sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sasrec = None
    if need_teacher:
        if matched_source.selected(dataset, args):
            translation_path = matched_teacher_translation(dataset, dm, args)
            run_config["method"]["sasrec"]["item_id_translation_path"] = str(translation_path)
            write_json(config_path, run_config)
        sasrec = tiger_runner.build_sasrec_teacher_model(
            dm, dict(run_config["method"]["sasrec"]), config_path=config_path
        )
        sasrec.to(device).eval()
        for param in sasrec.parameters():
            param.requires_grad_(False)
        if not args.skip_sanity_check:
            spec = DATASETS[dataset]
            teacher_ndcg = gflow_ablation.sanity_check_teacher(
                sasrec, dm, device, threshold=float(spec["teacher_ndcg_threshold"]))
            print(f"[{dataset}] SASRec teacher sanity OK: val NDCG@10 {teacher_ndcg:.4f} "
                  f"(3-batch subset)", flush=True)

    # Component caches (raw rewards; composition happens at lookup time).
    cache_dirs: dict[str, Path] = {}
    for key in components:
        cache_dir = component_cache_dir(dataset, key, args)
        manifest = ensure_component_cache(
            dataset, key, run_config, config_dir, len(subset_indices), args, prepared_data=dm)
        if uses_factor_cache(dataset, args):
            factor_runner.verify_cache_alignment(
                dm, subset_indices, {key: FactorizedRewardCache(cache_dir)},
                tensorized=bool(args.tensorized_train_loader),
            )
        else:
            gflow_ablation.verify_cache_alignment(dm, subset_indices, cache_dir)
        cache_dirs[key] = cache_dir
        print(f"[{dataset}/{key}] reward cache ready: {manifest['num_examples']} "
              f"examples x {manifest.get('num_catalog_items')} items at {cache_dir}", flush=True)
    if args.cache_only:
        print(f"[{dataset}] --cache-only: skipping the runs", flush=True)
        return None

    ensure_init_baseline(dataset=dataset, dm=dm, device=device, args=args,
                         output_root=output_root, csv_path=csv_path)
    return DatasetContext(dataset, run_config, dm, subset_indices, sasrec, cache_dirs)


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def oom_retry_args(args: argparse.Namespace, num_negatives: int | None) -> argparse.Namespace | None:
    """Reduce the actual micro-batch, retaining the effective optimizer batch.

    A loose candidate cap must not produce a retry with the same training
    batch. Validation/final evaluation shrink too, down to one example.
    ``None`` selects the SFT-only control, including non-uniform launches.
    """
    batch, _ = run_batch_settings(args, num_negatives)
    cap = int(args.max_candidate_batch)
    val_batch = int(args.val_batch_size)
    test_batch = int(args.test_batch_size)
    if batch <= 1 and val_batch <= 1 and test_batch <= 1:
        return None
    retry = argparse.Namespace(**vars(args))
    if num_negatives is None and not args.uniform_micro_batch:
        effective = int(args.batch_size) * int(args.grad_accum)
        new_batch = next(n for n in range(max(1, batch // 2), 0, -1) if effective % n == 0)
        retry.batch_size = new_batch
        retry.grad_accum = effective // new_batch
    else:
        k = max(int(k) for k in args.num_candidates) if args.uniform_micro_batch else int(num_negatives)
        retry.max_candidate_batch = max(k + 1, min(cap // 2, max(1, batch // 2) * (k + 1)))
    retry.val_batch_size = max(1, val_batch // 2)
    retry.test_batch_size = max(1, test_batch // 2)
    return retry


def interrupted_attempt_args(
    run_dir: Path, identity: Mapping[str, Any], num_finetune_examples: int,
    args: argparse.Namespace,
) -> tuple[int, argparse.Namespace]:
    """(attempt, args) a --resume of this run starts from: when the interrupted
    attempt had already shrunk the micro-batch cap after CUDA OOMs, continue
    with that cap — restarting at the configured one would OOM again and the
    retry would discard last.pt."""
    if resumable_checkpoint(run_dir, identity, num_finetune_examples, args) is None:
        return 0, args
    batching = load_json(run_dir / "resolved_config.json").get("batching", {})
    prior_cap = int(batching.get("max_candidate_batch") or args.max_candidate_batch)
    prior_retries = int(batching.get("oom_retries") or 0)
    if prior_retries <= 0:
        return 0, args
    resumed = argparse.Namespace(**vars(args))
    resumed.max_candidate_batch = min(prior_cap, int(args.max_candidate_batch))
    if identity.get("combo") == "-" and not args.uniform_micro_batch:
        effective = int(args.batch_size) * int(args.grad_accum)
        prior_batch = min(int(batching.get("batch_size") or args.batch_size), int(args.batch_size))
        resumed.batch_size = next(n for n in range(prior_batch, 0, -1) if effective % n == 0)
        resumed.grad_accum = effective // resumed.batch_size
    for key in ("val_batch_size", "test_batch_size"):
        configured = int(getattr(args, key))
        saved = int(batching.get(key) or max(1, configured // (2 ** prior_retries)))
        setattr(resumed, key, min(configured, saved))
    return prior_retries, resumed


def execute_entry(
    entry: Mapping[str, Any], ctx: DatasetContext, run_dir: Path, *, args: argparse.Namespace,
    csv_path: Path, claims: ClaimStore | None = None,
) -> dict[str, Any]:
    """run_single with the grid's failure policy: a CUDA out-of-memory halves
    the candidate micro-batch cap (grad accumulation keeps the effective batch)
    and restarts the run up to --oom-retries times; any other failure — or an
    OOM that cannot shrink further — is recorded (result.json status=failed +
    CSV rows) and never aborts the grid."""
    attempt, attempt_args = interrupted_attempt_args(
        run_dir, entry_identity(entry, int(args.seed)), len(ctx.subset_indices), args)
    resume_attempt = attempt
    while True:
        retry_args: argparse.Namespace | None = None
        try:
            return run_single(
                entry=entry, run_config=ctx.run_config, dm=ctx.dm, subset_indices=ctx.subset_indices,
                sasrec=ctx.sasrec, cache_dirs=ctx.cache_dirs, run_dir=run_dir, csv_path=csv_path,
                args=attempt_args, claims=claims, attempt=attempt, resume_attempt=resume_attempt,
            )
        except Exception as exc:  # noqa: BLE001 - keep the grid going
            if is_cuda_oom(exc) and attempt < int(args.oom_retries):
                retry_args = oom_retry_args(attempt_args, int(entry["num_negatives"]))
            if retry_args is not None:
                print(f"[{entry_tag(entry)}] CUDA out of memory at max_candidate_batch="
                      f"{attempt_args.max_candidate_batch}; retry {attempt + 1}/{args.oom_retries} "
                      f"with {retry_args.max_candidate_batch} (validation batch "
                      f"{retry_args.val_batch_size}, test batch "
                      f"{retry_args.test_batch_size})", flush=True)
            else:
                print(f"[{entry_tag(entry)}] FAILED: {exc}", flush=True)
                traceback.print_exc()
                identity = entry_identity(entry, int(args.seed))
                result = {**identity, "status": "failed", "error": str(exc), "run_dir": str(run_dir),
                          "oom_retries": attempt,
                          "max_candidate_batch": int(attempt_args.max_candidate_batch)}
                write_json(run_dir / RESULT_FILENAME, result)
                upsert_metrics_csv(csv_path, [
                    {**identity, "variant": "gflow", "split": split, "status": "failed",
                     "error": str(exc)[:300], "oom_retries": attempt,
                     "max_candidate_batch": int(attempt_args.max_candidate_batch)}
                    for split in ("validation", "test")
                ])
                if torch.device(args.device).type == "cuda":
                    torch.cuda.empty_cache()
                return result
        # OOM retry: the exception (and the failed attempt's model / trainer
        # it references) is released here, so the GPU memory can be reclaimed.
        attempt += 1
        attempt_args = retry_args
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


# --------------------------------------------------------------------------- #
# SFT-only control (same protocol, no GFlow term)
# --------------------------------------------------------------------------- #
def sft_control_run_dir(output_root: Path, dataset: str, seed: int) -> Path:
    return output_root / dataset / "sft_only" / f"seed_{seed}"


def sft_control_identity(dataset: str, seed: int) -> dict[str, Any]:
    return {"dataset": dataset, "combo": "-", "similarity_mode": "-", "weights": "-",
            "composition": "-", "power": "-", "objective": "-", "num_negatives": "-",
            "seed": int(seed), "target_grad_ratio": "-", "sampler": "-"}


def run_sft_control(dataset: str, ctx: DatasetContext, run_dir: Path, *, args: argparse.Namespace,
                    csv_path: Path, attempt: int = 0, resume_attempt: int = 0) -> dict[str, Any]:
    """The no-reward counterpart of run_single: the same init, data subset,
    optimizer, effective batch (256 x 8; the grid's own micro-batch with
    --uniform-micro-batch), monitor, early stopping, and final evaluation,
    with loss = L_SFT only. Attributes the grid's gains to the
    GFlow term rather than to continued SFT at the fine-tuning lr."""
    tag = f"{dataset}/sft_only"
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runner.seed_everything(int(args.seed))
    model, profile = build_gpt2_from_tuned_checkpoint(dataset, ctx.dm, args)
    model.to(device)
    method = gflow_ablation.AliasedSFTMethod(light_validation_sft_cfg())
    run_batch, run_accum = run_batch_settings(args, None)
    trainer = build_trainer(model, method, tag=tag, run_dir=run_dir, run_accum=run_accum,
                            args=args, device=device)
    identity = sft_control_identity(dataset, int(args.seed))
    last_ckpt = (resumable_checkpoint(run_dir, identity, len(ctx.subset_indices), args)
                 if attempt == resume_attempt else None)
    resumed_from_epoch = resume_trainer(trainer, last_ckpt, tag=tag) if last_ckpt is not None else None
    write_json(run_dir / "resolved_config.json", {
        **identity, "variant": "sft_only", "model": model.config_dict(), "model_profile": profile,
        "init_checkpoint": str(tuned_checkpoint_paths(dataset, args)["checkpoint"]),
        "loss": "L_SFT only (gflow_weight 0, no sampler / reward cache / calibration)",
        "batching": {"batch_size": run_batch, "gradient_accumulation": run_accum,
                     "effective_batch": run_batch * run_accum,
                     "max_candidate_batch": int(args.max_candidate_batch), "oom_retries": int(attempt),
                     "val_batch_size": int(args.val_batch_size), "test_batch_size": int(args.test_batch_size)},
        "input_pipeline": input_pipeline_settings(args),
        "data": {**runner.data_config_paths(ctx.dm.cfg), "train_example_layout": "prefix_target",
                 "finetune_per_user_last_n": cache_last_n(args),
                 "finetune_subset": describe_finetune_subset(args),
                 "finetune_per_user_pool_last_n": args.train_pool_last_n,
                 "finetune_per_user_random_n": args.train_random_n,
                 "finetune_subset_seed": args.train_subset_seed,
                 "num_train_examples": len(ctx.dm.train_dataset),
                 "num_finetune_examples": len(ctx.subset_indices), **ctx.dm.summary()},
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })
    print(f"[{tag}] SFT-only fine-tuning: epochs<={trainer.cfg.max_epochs}, "
          f"batch {run_batch} x accum {run_accum}", flush=True)
    print(f"[{tag}] input pipeline: {json.dumps(input_pipeline_settings(args), sort_keys=True)}", flush=True)
    fit, splits = fit_and_evaluate(trainer, model, ctx.dm, ctx.subset_indices, run_batch=run_batch,
                                   run_dir=run_dir, args=args, device=device,
                                   resumed_from_epoch=resumed_from_epoch)
    result = {
        **identity, "variant": "sft_only", "status": "completed",
        "resumed_from_epoch": resumed_from_epoch,
        "best_epoch": fit.state.best_epoch, "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1), "elapsed_sec": time.time() - started,
        "best_checkpoint": fit.best_checkpoint_path, "run_dir": str(run_dir), "splits": splits,
        "oom_retries": int(attempt),
    }
    write_json(run_dir / RESULT_FILENAME, result)
    upsert_metrics_csv(csv_path, gflow_ablation.metric_rows(
        {**identity, "variant": "sft_only"}, splits,
        {"status": "completed", "best_epoch": result["best_epoch"],
         "best_val_ndcg5": result["best_val_ndcg5"], "epochs_completed": result["epochs_completed"],
         "run_dir": str(run_dir), "micro_batch": run_batch, "grad_accum": run_accum,
         "oom_retries": int(attempt), "max_candidate_batch": int(args.max_candidate_batch),
         "elapsed_sec": round(result["elapsed_sec"], 1)}))
    del model, method, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def execute_sft_control(dataset: str, ctx: DatasetContext, run_dir: Path, *,
                        args: argparse.Namespace, csv_path: Path) -> dict[str, Any]:
    """SFT control with the same OOM backoff and resume policy as GFlow."""
    attempt, attempt_args = interrupted_attempt_args(
        run_dir, sft_control_identity(dataset, int(args.seed)), len(ctx.subset_indices), args)
    resume_attempt = attempt
    while True:
        try:
            return run_sft_control(dataset, ctx, run_dir, args=attempt_args, csv_path=csv_path,
                                   attempt=attempt, resume_attempt=resume_attempt)
        except Exception as exc:
            if not is_cuda_oom(exc) or attempt >= int(args.oom_retries):
                raise
            retry_args = oom_retry_args(attempt_args, None)
            if retry_args is None:
                raise
            batch, accum = run_batch_settings(retry_args, None)
            print(f"[{dataset}/sft_only] CUDA out of memory; retry {attempt + 1}/{args.oom_retries} "
                  f"with batch {batch} x accum {accum}, validation {retry_args.val_batch_size}, "
                  f"test {retry_args.test_batch_size}", flush=True)
        # Exit the exception scope before freeing memory; tracebacks retain
        # the failed trainer and its CUDA tensors until then.
        attempt += 1
        attempt_args = retry_args
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


def run_sft_controls(args: argparse.Namespace, *, device: torch.device, output_root: Path,
                     csv_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        args = dataset_args(args, dataset)
        run_dir = sft_control_run_dir(output_root, dataset, int(args.seed))
        check_existing_training_context(run_dir, args)
        if run_completed(run_dir) and not args.force:
            print(f"[{dataset}/sft_only] already completed — skipping (--force to redo)", flush=True)
            continue
        ctx = build_dataset_context(dataset, [], args=args, device=device, output_root=output_root,
                                    csv_path=csv_path, need_teacher=False)
        assert ctx is not None
        try:
            results.append(execute_sft_control(dataset, ctx, run_dir, args=args, csv_path=csv_path))
        except Exception as exc:  # noqa: BLE001 - record and continue with the next dataset
            print(f"[{dataset}/sft_only] FAILED: {exc}", flush=True)
            traceback.print_exc()
            identity = sft_control_identity(dataset, int(args.seed))
            result = {**identity, "variant": "sft_only", "status": "failed", "error": str(exc),
                      "run_dir": str(run_dir)}
            write_json(run_dir / RESULT_FILENAME, result)
            upsert_metrics_csv(csv_path, [
                {**identity, "variant": "sft_only", "split": split, "status": "failed",
                 "error": str(exc)[:300]} for split in ("validation", "test")])
            results.append(result)
        release_context(ctx, device)
    for row in results:
        line = f"{row['dataset']:<12} sft_only {row['status']:<10}"
        if row.get("status") == "completed":
            line += (f" best_val_ndcg@5={row.get('best_val_ndcg5')} "
                     f"test_NDCG@10={row['splits']['test'].get('NDCG@10'):.5f}")
        print(line, flush=True)
    print(f"metrics CSV: {csv_path}", flush=True)
    return results


def release_context(ctx: DatasetContext | None, device: torch.device) -> None:
    if ctx is None:
        return
    del ctx.dm, ctx.sasrec
    if device.type == "cuda":
        torch.cuda.empty_cache()


def print_summary(all_results: Sequence[Mapping[str, Any]], csv_path: Path) -> None:
    print("\n=== summary ===", flush=True)
    for row in all_results:
        line = (f"{row['dataset']:<12} {row['combo']:<36} w={row['weights']:<8} "
                f"p={row['power']:<6g} {row['objective']:<6} K={row['num_negatives']:<5} "
                f"{row['status']:<10}")
        if row.get("status") == "completed":
            lam = row.get("lambda_gflow")
            best = row.get("best_val_ndcg5")
            test_ndcg5 = row.get("splits", {}).get("test", {}).get("NDCG@5")
            if isinstance(lam, float):
                line += f" lambda={lam:.4g}"
            if isinstance(best, float):
                line += f" best_val_ndcg@5={best:.5f}"
            if isinstance(test_ndcg5, float):
                line += f" test_NDCG@5={test_ndcg5:.5f}"
        print(line, flush=True)
    print(f"metrics CSV: {csv_path}", flush=True)


def rotate(items: list[Any], offset: int) -> list[Any]:
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def run_worker(
    grid: Sequence[Mapping[str, Any]], *, args: argparse.Namespace, device: torch.device,
    output_root: Path, csv_path: Path,
) -> list[dict[str, Any]]:
    """Claim and run grid entries until none is claimable or the time budget
    is spent. Worker w starts at a rotated position of the grid (datasets and
    entries), so concurrent workers spread over the grid instead of racing
    for the same entry, and builds each dataset context only after its first
    successful claim."""
    wid = int(args.worker_id)
    count = max(1, int(args.worker_count))
    tag = f"worker {wid}/{count}"
    claims = ClaimStore(output_root, worker_id=wid, stale_claim_hours=float(args.stale_claim_hours))
    started = time.time()
    run_hours: list[float] = []
    results: list[dict[str, Any]] = []

    datasets = rotate([d for d in args.datasets if any(e["dataset"] == d for e in grid)], wid)
    for dataset in datasets:
        args = dataset_args(args, dataset)
        entries = [e for e in grid if e["dataset"] == dataset]
        needed = components_for(entries)
        entries = rotate(entries, (wid * len(entries)) // count)
        ctx: DatasetContext | None = None
        budget_hit = False
        for entry in entries:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            check_existing_training_context(run_dir, args)
            if claims.status_of(run_dir) in {"completed", "running"}:
                continue
            elapsed_h = (time.time() - started) / 3600.0
            # With --resume a run cut off by the wall time is continued by the
            # next worker, so only the setup time needs to fit the budget.
            estimate_h = (max(run_hours) if run_hours and not args.resume
                          else float(args.run_hours_estimate))
            if args.time_budget_hours is not None and elapsed_h + estimate_h > float(args.time_budget_hours):
                print(f"[{tag}] {elapsed_h:.2f} h elapsed + ~{estimate_h:.2f} h per run exceeds "
                      f"the {args.time_budget_hours} h budget; not claiming another run.", flush=True)
                budget_hit = True
                break
            if not claims.try_claim(run_dir, retry_failed=bool(args.retry_failed)):
                continue
            print(f"\n[{tag}] claimed {entry_tag(entry)}", flush=True)
            run_started = time.time()
            try:
                if ctx is None:
                    ctx = build_dataset_context(dataset, needed, args=args, device=device,
                                                output_root=output_root, csv_path=csv_path,
                                                need_teacher=needs_teacher(entries))
                    assert ctx is not None
                result = execute_entry(entry, ctx, run_dir, args=args, csv_path=csv_path, claims=claims)
            except Exception as exc:  # noqa: BLE001 - context build failed: release and stop
                claims.release(run_dir)
                print(f"[{tag}] dataset setup for {dataset} FAILED: {exc}", flush=True)
                traceback.print_exc()
                raise
            finally:
                claims.release(run_dir)
            run_hours.append((time.time() - run_started) / 3600.0)
            results.append(result)
            write_json(output_root / f"worker_{wid}_summary.json", {"runs": results})
            print(f"[{tag}] {entry_tag(entry)} -> {result['status']} in {run_hours[-1]:.2f} h "
                  f"(worker elapsed {(time.time() - started) / 3600:.2f} h)", flush=True)
        release_context(ctx, device)
        if budget_hit:
            break
    print(f"[{tag}] done: {len(results)} runs in {(time.time() - started) / 3600:.2f} h", flush=True)
    return results


def audit_data(args: argparse.Namespace) -> None:
    """Audit the shared SFT/GFlow input pipeline, including final target selection."""
    for dataset in args.datasets:
        selected = dataset_args(args, dataset)
        config = build_run_config(dataset, selected, 1)
        directory = abspath(args.output_root) / dataset
        path = directory / "data_reward_config.json"
        write_json(path, config)
        data = tiger_runner.build_data_module(tiger_runner.build_data_config(config, config_path=path))
        data.setup()
        indices = finetune_indices(data.train_dataset, selected)
        if matched_source.selected(dataset, selected):
            report = directory / "matched_prefix_audit.json"
            audit = matched_source.audit_prefix_data(data, matched_source.load_recorded(PROJECT_ROOT, dataset))
        else:
            report = directory / "dense_prefix_audit.json"
            audit = {**data.dense_prefix_audit, "training_context": data.summary()["training_context"]}
        write_json(report, {**audit, "num_finetune_examples": len(indices),
                            "target_selection": finetune_subset_spec(selected)})
        print(f"[{dataset}] data audit passed; {len(indices)} training targets: {report}", flush=True)
        del data


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit_data_only:
        audit_data(args)
        return
    grid = build_grid(args)

    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d}  {entry_tag(entry)}")
        per_dataset = {ds: sum(1 for e in grid if e["dataset"] == ds) for ds in args.datasets}
        print(f"total: {len(grid)} runs ({', '.join(f'{k}={v}' for k, v in per_dataset.items())})")
        return
    if args.status:
        print_status(grid, args)
        return
    if args.cache_status:
        raise SystemExit(1 if print_cache_status(grid, args) else 0)
    if args.reset_claims:
        output_root = abspath(args.output_root)
        store = ClaimStore(output_root, worker_id=None, stale_claim_hours=float(args.stale_claim_hours))
        removed = store.reset_claims([run_dir_for(output_root, e, int(args.seed)) for e in grid])
        print(f"released {removed} claim(s) under {output_root}")
        return

    runner.seed_everything(int(args.seed))
    device = torch.device(runner.resolve_device(args.device))
    args.device = str(device)
    output_root = abspath(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_csv_path(args)

    if args.sft_only:
        run_sft_controls(args, device=device, output_root=output_root, csv_path=csv_path)
        return

    if args.worker_id is not None:
        results = run_worker(grid, args=args, device=device, output_root=output_root, csv_path=csv_path)
        print_summary(results, csv_path)
        return

    grid = select_grid(grid, args.index)
    if not grid and not args.cache_only:
        return

    # Index mode: group the selected runs by dataset so the datamodule /
    # teacher / caches are built once per dataset.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    if args.cache_only:
        for dataset in args.datasets:
            by_dataset[dataset] = []
    for entry in grid:
        by_dataset.setdefault(entry["dataset"], []).append(entry)

    all_results: list[dict[str, Any]] = []
    for dataset, entries in by_dataset.items():
        args = dataset_args(args, dataset)
        pending = []
        for entry in entries:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            check_existing_training_context(run_dir, args)
            if run_completed(run_dir) and not args.force:
                print(f"[{entry_tag(entry)}] already completed — skipping (--force to redo)", flush=True)
                continue
            pending.append(entry)
        if not pending and not args.cache_only:
            continue

        print(f"\n=== dataset: {dataset} ({len(pending)} runs) ===", flush=True)
        if args.cache_only:
            components = list(args.components or component_keys_for(args))
            unavailable = sorted(set(components) - set(component_keys_for(args)))
            if unavailable:
                raise SystemExit(
                    f"--components {unavailable} need their similarity mode in "
                    f"--similarity-modes (current: {args.similarity_modes})")
        else:
            components = components_for(pending)
        ctx = build_dataset_context(dataset, components, args=args, device=device,
                                    output_root=output_root, csv_path=csv_path,
                                    need_teacher=("sasrec" in components) if args.cache_only
                                    else needs_teacher(pending))
        if ctx is None:
            continue
        for entry in pending:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            all_results.append(execute_entry(entry, ctx, run_dir, args=args, csv_path=csv_path))
            write_json(output_root / "summary.json", {"runs": all_results})
        release_context(ctx, device)

    print_summary(all_results, csv_path)


if __name__ == "__main__":
    main()
