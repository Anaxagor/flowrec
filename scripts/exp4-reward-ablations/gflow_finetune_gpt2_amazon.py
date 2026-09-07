#!/usr/bin/env python
"""GFlow fine-tuning ablation of the exp2 TUNED GPT-2 checkpoints on the small
Amazon datasets (Beauty 2014, Instruments 2018).

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

Reward caches: unlike the yambda grid (one dense cache per combo, with power
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
    checkpoints/exp2-tuned_params/initial/gpt2/<ds>/checkpoints/best.pt.
    The architecture is rebuilt from the checkpoint's own sidecar
    model_config (best.pt.json) and the tuned dropout from best_result.json,
    through run_gpt2_sft_all.build_model, then strict-loaded. NOTE: both
    tuned checkpoints were trained with the exp1 semantic-ID tables
    (datasets/semantic_ids/exp1-standard_params/<ds>/, level-offset,
    token = code + 1 + level * 256), NOT the exp2 tables: the beauty
    checkpoint's item_semantic_ids buffer equals the exp1 table row for row,
    and the instruments buffer (8,556 catalog items, max token 779) is the
    exp1 table restricted to the verified-purchase catalog. The row-table
    check and the teacher-forced-NLL sanity check refuse any other pairing.
  - Data: the exp2 dataset_amazon.yaml preprocessing (positive_criterion
    "verified_purchase == True", 5-core, max_history_len 20) that the tuner
    used, materialized on the prefix_target layout (one target per example
    = one reward-cache row); the catalog is identical to the checkpoint's.
    Fine-tuning runs on each user's last --train-last-n train examples
    (protocol default 8; nearly the full set on these datasets).
  - SASRec teacher (curriculum sampler + sasrec reward): the exp2 tuned
    checkpoints/exp2-tuned_params/initial/sasrec/<ds>/best_model.pt.
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
    artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_metrics_summary.csv.
    The untouched init checkpoint is evaluated once per dataset as
    variant="init".

Outputs:
  experiments/exp4-gflow-gpt2-amazon/<ds>/<combo>/w<weights>/pow_<p>/<obj>_k<K>/seed_<s>/
      resolved_config.json calibration.json training_history.json
      final_metrics.json result.json checkpoints/best.pt
      (<combo> names the concrete components, e.g. sasrec_x_sim_embed)
  experiments/exp4-gflow-gpt2-amazon/<ds>/baseline/init_metrics.json
  artifacts/exp4-reward_ablation/gflow_gpt2_amazon/<ds>/reward_cache/<component>/
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
import itertools
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace
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
import gflow_finetune_gpt2_beauty as gpt2_beauty  # noqa: E402  (scripts/)
from precompute_reward_cache import per_user_tail_indices  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON, GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402

runner = gpt2_runner.runner  # scripts/sasrec_config_runner (exp1 shared helpers)
GPT2GFlowMethod = gpt2_beauty.GPT2GFlowMethod


# --------------------------------------------------------------------------- #
# Assets and grid
# --------------------------------------------------------------------------- #
EXP2_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"
EXP2_DATASET_YAML = EXP2_CONFIG_DIR / "dataset_amazon.yaml"
EXP1_GPT2_MODEL_YAML = PROJECT_ROOT / "configs" / "exp1-standard_params" / "gpt2_model.yaml"
# Semantic IDs the tuned checkpoints were trained with (see the module
# docstring): the exp1 tokenizer tables + their RQ-VAE model.pt.
SEMANTIC_IDS_ROOT = PROJECT_ROOT / "datasets" / "semantic_ids" / "exp1-standard_params"
GPT2_ROOT = "checkpoints/exp2-tuned_params/initial/gpt2"
SASREC_ROOT = "checkpoints/exp2-tuned_params/initial/sasrec"
# exp1 tables are level-offset (token = code + 1 + level * 256); the
# hierarchical similarity reward de-offsets levels with this stride.
LEVEL_STRIDE = 256
# Content embeddings the exp1 RQ-VAE tokenizer quantized (sim_embed reward).
EMBEDDINGS_ROOT = PROJECT_ROOT / "datasets" / "embeddings"

DATASETS: dict[str, dict[str, Any]] = {
    "beauty": {
        "gpt2_dir": f"{GPT2_ROOT}/beauty",
        "sasrec_checkpoint": f"{SASREC_ROOT}/beauty/best_model.pt",
        "sasrec_trial": f"{SASREC_ROOT}/beauty/best_trial.json",
        # exp2 tuning recorded full-val NDCG@10 0.0527; the 3-batch sanity
        # subset scores ~0.026 (a subset artifact, verified 2026-09-03).
        "teacher_ndcg_threshold": 0.02,
        "teacher_ndcg_reference": 0.0527,
    },
    "instruments": {
        "gpt2_dir": f"{GPT2_ROOT}/instruments",
        "sasrec_checkpoint": f"{SASREC_ROOT}/instruments/best_model.pt",
        "sasrec_trial": f"{SASREC_ROOT}/instruments/best_trial.json",
        # exp2 tuning recorded full-val NDCG@10 0.138.
        "teacher_ndcg_threshold": 0.04,
        "teacher_ndcg_reference": 0.138,
    },
}

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

PROTOCOL_CONFIG = EXP2_CONFIG_DIR / "finetune_gflow.json"


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    D = gflow_ablation.protocol_defaults()  # configs/exp2-tuned-params/finetune_gflow.json
    p = argparse.ArgumentParser(
        "GFlow fine-tuning ablation of the exp2 tuned GPT-2 checkpoints on "
        "beauty / instruments: reward combos x weights x powers x negatives, "
        "gradient-norm-calibrated gflow weight."
    )
    combo_keys = [combo_key(c) for c in all_combos()]
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--combos", nargs="+", choices=combo_keys, default=combo_keys)
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
                   help="Per-run duration assumed before this worker has completed a run")
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
    p.add_argument("--train-last-n", type=int, default=int(D.get("train_last_n", 8)),
                   help="Fine-tune on the last N train examples per user (subset only; "
                        "the datamodule layout and catalog are untouched)")

    # Reward pieces.
    p.add_argument("--sasrec-transform", default=str(D.get("sasrec_transform", "sigmoid")))
    p.add_argument("--sasrec-temperature", type=float, default=float(D.get("sasrec_temperature", 1.0)))

    # Sampler (fixed easy-negative phase).
    p.add_argument("--curriculum-steps", type=int, default=int(D.get("curriculum_steps", 3000)))
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
    p.add_argument("--cache-dtype", choices=["float16", "float32"],
                   default=str(D.get("cache_dtype", "float16")))
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true",
                   help="Build/verify the component reward caches for the selected "
                        "datasets, then exit (run before the main array)")
    p.add_argument("--max-candidate-batch", type=int,
                   default=int(D.get("max_candidate_batch", 4096)),
                   help="Upper bound on batch * (num_negatives + 1) during training")

    # Training (SFT-control protocol).
    p.add_argument("--max-epochs", type=int, default=int(D.get("max_epochs", 500)))
    p.add_argument("--validate-every", type=int, default=int(D.get("validate_every", 5)))
    p.add_argument("--patience", type=int, default=int(D.get("patience", 10)))
    p.add_argument("--lr", type=float, default=float(D.get("lr", 3e-5)))
    p.add_argument("--weight-decay", type=float, default=float(D.get("weight_decay", 1e-2)))
    p.add_argument("--batch-size", type=int, default=int(D.get("batch_size", 256)))
    p.add_argument("--grad-accum", type=int, default=int(D.get("grad_accum", 8)))
    p.add_argument("--max-grad-norm", type=float, default=float(D.get("max_grad_norm", 0.5)))
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"],
                   default=str(D.get("amp_dtype", "float16")))
    p.add_argument("--val-batch-size", type=int, default=int(D.get("val_batch_size", 256)))
    p.add_argument("--num-workers", type=int, default=4)

    # Final evaluation (exp1 GPT-2 protocol: beam-20 HR/NDCG/Coverage/ARP).
    p.add_argument("--test-ks", type=int, nargs="+", default=list(D.get("test_ks", [5, 10, 20])))
    p.add_argument("--eval-num-beams", type=int, default=int(D.get("eval_num_beams", 20)))
    p.add_argument("--test-batch-size", type=int, default=128,
                   help="Beam-20 final evaluation batch (exp1 gpt2 test_eval_batch_size)")

    # Output.
    p.add_argument("--output-root", type=Path, default=Path("experiments/exp4-gflow-gpt2-amazon"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("artifacts/exp4-reward_ablation/gflow_gpt2_amazon"),
                   help="Component caches live at <cache-root>/<dataset>/reward_cache/<component>")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/exp4-reward_ablation"))

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
    return args


def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
            "rqvae_model_path": str(SEMANTIC_IDS_ROOT / dataset / "model.pt"),
        }
    elif mode == "embedding_cosine":
        # (cos(z_item, z_target) + 1) / 2 over the normalized content embeddings.
        reward = {
            "mode": "embedding_cosine",
            "embeddings_path": str(EMBEDDINGS_ROOT / f"{dataset}_content_embeddings.pt"),
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
    payload = load_json(abspath(spec["sasrec_trial"]))
    params = payload.get("best_params", payload)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
    return {
        "checkpoint_path": str(abspath(spec["sasrec_checkpoint"])),
        "model": {k: params[k] for k in model_keys if k in params},
    }


def build_run_config(dataset: str, args: argparse.Namespace, user_ids_count: int) -> dict[str, Any]:
    """run_tiger-schema config driving the fine-tuning datamodule, the SASRec
    teacher, and (per component, see cache_config) the reward-cache
    precompute. Mirrors the exp2 dataset_amazon.yaml preprocessing the GPT-2
    tuner used, on the prefix_target layout (GFlow needs one target per
    example); the catalog is identical to the checkpoint's."""
    yaml_cfg = runner.load_yaml(EXP2_DATASET_YAML)
    entry = dict(yaml_cfg["datasets"][dataset])
    if not bool(yaml_cfg.get("filter_reviews_to_metadata", True)):
        raise ValueError(
            "dataset_amazon.yaml sets filter_reviews_to_metadata=false, which "
            "run_tiger.build_data_config cannot express."
        )
    criterion = entry.get("positive_criterion", yaml_cfg.get("positive_criterion"))
    return {
        "experiment_name": f"gflow_gpt2_{dataset}",
        "seed": int(args.seed),
        "data": {
            "reviews_path": str(abspath(entry["reviews_path"])),
            "metadata_path": str(abspath(entry["metadata_path"])),
            "semantic_id_path": str(SEMANTIC_IDS_ROOT / dataset / "semantic_ids.json"),
        },
        "dataset": {
            "item_key": entry.get("item_key", "asin"),
            "positive_criterion": criterion,
            "split_strategy": yaml_cfg.get("split_strategy", "leave_one_out"),
            "min_history_len": int(yaml_cfg.get("min_history_len", 1)),
            "min_user_interactions": int(yaml_cfg.get("min_user_interactions", 5)),
            "min_item_interactions": int(yaml_cfg.get("min_item_interactions", 5)),
            "max_history_len": int(yaml_cfg.get("max_history_len", 20)),
            "deduplicate_consecutive": bool(yaml_cfg.get("deduplicate_consecutive", False)),
            "filter_cold_items_after_split": bool(yaml_cfg.get("filter_cold_items_after_split", False)),
            "cold_item_train_source": yaml_cfg.get("cold_item_train_source", "observed"),
            "offset_semantic_ids_by_one": bool(yaml_cfg.get("offset_semantic_ids_by_one", True)),
            "drop_items_missing_semantic_id": True,
            "train_example_layout": "prefix_target",
            "train_all_prefixes": bool(yaml_cfg.get("train_all_prefixes", True)),
            "train_target_policy": "all",
            "history_padding": "left",
        },
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
    return abspath(args.cache_root) / dataset / "reward_cache" / key


def validate_inputs(dataset: str, args: argparse.Namespace) -> None:
    spec = DATASETS[dataset]
    gpt2_dir = abspath(spec["gpt2_dir"])
    required = [
        gpt2_dir / "checkpoints" / "best.pt",
        gpt2_dir / "checkpoints" / "best.pt.json",
        gpt2_dir / "best_result.json",
        abspath(spec["sasrec_checkpoint"]),
        abspath(spec["sasrec_trial"]),
        SEMANTIC_IDS_ROOT / dataset / "semantic_ids.json",
        EXP2_DATASET_YAML,
        EXP1_GPT2_MODEL_YAML,
    ]
    if "hierarchical" in args.similarity_modes:
        required.append(SEMANTIC_IDS_ROOT / dataset / "model.pt")
    if "embedding_cosine" in args.similarity_modes:
        required.append(EMBEDDINGS_ROOT / f"{dataset}_content_embeddings.pt")
    yaml_cfg = runner.load_yaml(EXP2_DATASET_YAML)
    entry = yaml_cfg["datasets"][dataset]
    required += [abspath(entry["reviews_path"]), abspath(entry["metadata_path"])]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"[{dataset}] missing inputs:\n{formatted}")


# --------------------------------------------------------------------------- #
# Model construction from the tuned checkpoint
# --------------------------------------------------------------------------- #
ARCH_KEYS = ("codebook_size", "sem_id_len", "embedding_dim", "num_heads", "num_layers",
             "dim_feedforward", "max_position_embeddings", "semantic_vocab_size",
             "unified_vocab_size")


def tuned_checkpoint_paths(dataset: str) -> dict[str, Path]:
    gpt2_dir = abspath(DATASETS[dataset]["gpt2_dir"])
    return {
        "checkpoint": gpt2_dir / "checkpoints" / "best.pt",
        "sidecar": gpt2_dir / "checkpoints" / "best.pt.json",
        "best_result": gpt2_dir / "best_result.json",
    }


def tuned_profile(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """gpt2_model.yaml-style profile for the tuned checkpoint: architecture
    from the checkpoint's sidecar model_config, dropout (absent from the
    sidecar) from the tuner's best_result.json best_params."""
    paths = tuned_checkpoint_paths(dataset)
    model_config = load_json(paths["sidecar"])["model_config"]
    best = load_json(paths["best_result"])
    params = best.get("best_params", {})
    if "dropout" not in params:
        raise KeyError(f"{paths['best_result']} lacks best_params.dropout")
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


def build_gpt2_from_tuned_checkpoint(dataset: str, dm) -> tuple[Any, dict[str, Any]]:
    """Rebuild the tuned architecture through run_gpt2_sft_all.build_model
    (exp1 defaults: shared user token, fast greedy generation, beam 20),
    verify it against the sidecar, then strict-load the weights."""
    profile, sidecar_cfg = tuned_profile(dataset)
    model_yaml = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    model = gpt2_runner.build_model(model_yaml, profile, dm)
    built = model.config_dict()
    mismatch = {k: (built.get(k), sidecar_cfg.get(k)) for k in ARCH_KEYS
                if k in sidecar_cfg and built.get(k) != sidecar_cfg.get(k)}
    if mismatch:
        raise RuntimeError(
            f"[{dataset}] rebuilt GPT-2 differs from the checkpoint sidecar "
            f"(built, sidecar): {mismatch}. The datamodule catalog / semantic "
            "table does not match the tuned checkpoint."
        )
    gpt2_beauty.load_gpt2_checkpoint(model, tuned_checkpoint_paths(dataset)["checkpoint"])
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
    num_subset_examples: int, args: argparse.Namespace,
) -> dict[str, Any]:
    cache_dir = component_cache_dir(dataset, key, args)
    manifest_path = cache_dir / "manifest.json"
    config_path = config_dir / f"cache_config_{key}.json"
    spec = component_spec_from_config(run_config, key)
    write_json(config_path, cache_config(run_config, key))
    # Serialize concurrent array tasks: the first builds, the rest wait on the
    # lock and find the manifest.
    with control.file_lock(cache_dir.parent / f"{cache_dir.name}.build.lock"):
        if manifest_path.exists() and not args.force_cache:
            manifest = load_json(manifest_path)
            return _check_component_manifest(manifest, cache_dir, key, spec,
                                             num_subset_examples, args)
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "precompute_reward_cache.py"),
            "--config", str(config_path),
            "--output-dir", str(cache_dir),
            "--split", "train",
            "--per-user-last-n", str(int(args.train_last_n)),
            "--batch-size", str(int(args.cache_batch_size)),
            "--num-workers", str(int(args.num_workers)),
            "--device", str(args.device),
            "--dtype", str(args.cache_dtype),
        ]
        if args.smoke:
            cmd += ["--max-examples", str(_smoke_examples(args))]
        print(f"[{dataset}/{key}] precomputing reward cache: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        # The manifest records only the registry name; the marker pins the
        # concrete component (e.g. which similarity mode) this cache holds.
        write_json(cache_dir / "component.json", {"key": key, "spec": spec})
        manifest = load_json(manifest_path)
    return _check_component_manifest(manifest, cache_dir, key, spec, num_subset_examples, args)


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
    stored_last_n = manifest.get("per_user_last_n")
    if stored_last_n is not None and int(stored_last_n) != int(args.train_last_n):
        raise RuntimeError(
            f"Reward cache at {cache_dir} was built with --per-user-last-n "
            f"{stored_last_n}, but this run uses {args.train_last_n}. Rebuild with --force-cache."
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
def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = [c for c in all_combos() if combo_key(c) in set(args.combos)]
    rhos = args.rho_targets or [args.gflow_target_grad_ratio]
    grid: list[dict[str, Any]] = []
    for dataset in args.datasets:  # dataset-major: shared datamodule/teacher/caches
        for combo in combos:
            # Combos with `similarity` fan out over the similarity modes.
            modes = list(args.similarity_modes) if "similarity" in combo else [None]
            for mode in modes:
                components = concrete_components(combo, mode)
                for weights in weight_assignments(len(combo), args.weight_levels):
                    for power in args.powers:
                        for rho in rhos:
                            for objective in args.objectives:
                                for k in args.num_candidates:
                                    grid.append({
                                        "dataset": str(dataset),
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


def _rho_suffix(rho: float) -> str:
    return "" if float(rho) == 1.0 else f"_rho{gflow_ablation.power_tag(rho)}"


def _composition_dir(composition: str) -> str:
    # multiply is the default composition and keeps the plain layout.
    return "" if composition == "multiply" else f"mixed_"


def run_dir_for(output_root: Path, entry: Mapping[str, Any], seed: int) -> Path:
    return (
        output_root / entry["dataset"]
        / f"{_composition_dir(entry['composition'])}{combo_key(entry['components'])}"
        / f"w{weight_tag(entry['weights'])}" / f"pow_{gflow_ablation.power_tag(entry['power'])}"
        / f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}"
        / f"seed_{seed}"
    )


def entry_tag(entry: Mapping[str, Any]) -> str:
    return (f"{entry['dataset']}/{_composition_dir(entry['composition'])}"
            f"{combo_key(entry['components'])}"
            f"/w{weight_tag(entry['weights'])}/p{gflow_ablation.power_tag(entry['power'])}/"
            f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}")


def entry_identity(entry: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "dataset": entry["dataset"],
        "combo": combo_key(entry["components"]),
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
           "num_negatives", "seed", "variant", "split", "target_grad_ratio")


def upsert_metrics_csv(csv_path: Path, new_rows: list[dict[str, Any]]) -> None:
    import csv as _csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with control.file_lock(csv_path.with_suffix(".lock")):
        rows: list[dict[str, Any]] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = [dict(r) for r in _csv.DictReader(fh)]

        def key_of(row: Mapping[str, Any]) -> tuple[str, ...]:
            return tuple(str(row.get(k, "")) for k in CSV_KEY)

        merged = {key_of(r): r for r in rows}
        for row in new_rows:
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
def calibrate_gflow_weight(
    *, model, base_gflow_cfg: GFlowConfig, sampler, sft_cfg: SFTConfig, cache: ComposedRewardCache,
    dm, subset_indices: list[int], device: torch.device, args: argparse.Namespace,
    tag: str, target_ratio: float, batch_size: int,
) -> dict[str, Any]:
    """lambda = rho_target / median_b(||g_GFlow,b|| / ||g_SFT,b||) at the init
    checkpoint, before any optimizer step (gflow_finetune_yambda50m protocol:
    two method variants isolate the unweighted losses through the training
    code path, identical RNG per pass, step=0 = easy curriculum phase, full
    precision, weights/optimizer/loader order untouched)."""
    sft_variant = make_method(
        replace(base_gflow_cfg, sft_weight=1.0, gflow_weight=0.0), sampler, sft_cfg, cache)
    gflow_variant = make_method(
        replace(base_gflow_cfg, sft_weight=0.0, gflow_weight=1.0), sampler, sft_cfg, cache)

    params = [p for p in model.parameters() if p.requires_grad]
    loader = gflow_ablation.indexed_train_loader(
        dm, subset_indices, shuffle=not args.smoke, batch_size=int(batch_size),
        seed=int(args.gflow_calibration_seed), num_workers=0,
    )
    num_batches = 1 if args.smoke else int(args.gflow_calibration_batches)

    was_training = model.training
    model.train()

    def loss_pairs():
        it = iter(loader)
        for b in range(num_batches):
            batch = gflow_ablation.move_batch(next(it), device)
            pair = []
            for variant in (sft_variant, gflow_variant):
                seed = int(args.gflow_calibration_seed) + 1000 * b
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                out = variant.compute_loss(model, batch, stage="train", step=0)
                pair.append(out.loss)
            yield pair[0], pair[1]

    result = gflow_ablation.calibrate_from_loss_pairs(
        loss_pairs(), params, target_ratio=float(target_ratio),
        eps=float(args.gflow_calibration_eps),
    )
    model.train(was_training)
    runner.seed_everything(int(args.seed))  # calibration must not shift training
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
        model, _ = build_gpt2_from_tuned_checkpoint(dataset, dm)
        model.to(device).eval()
        max_eval = 2 if args.smoke else args.max_eval_batches
        splits = gpt2_beauty.evaluate_splits(model, dm, args, device, max_eval,
                                             splits=("validation", "test"))
        payload = {
            "dataset": dataset,
            "checkpoint": str(tuned_checkpoint_paths(dataset)["checkpoint"]),
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
def run_single(
    *, entry: Mapping[str, Any], run_config: Mapping[str, Any], dm, subset_indices: list[int],
    sasrec, cache_dirs: Mapping[str, Path], run_dir: Path, csv_path: Path,
    args: argparse.Namespace, claims: ClaimStore | None = None,
) -> dict[str, Any]:
    tag = entry_tag(entry)
    dataset = entry["dataset"]
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    runner.seed_everything(int(args.seed))
    model, profile = build_gpt2_from_tuned_checkpoint(dataset, dm)
    model.to(device)

    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="curriculum",
            num_negatives=int(entry["num_negatives"]),
            include_positive=True,
            exclude_history=True,
            curriculum_steps=int(args.curriculum_steps),
            temperature=float(args.sampler_temperature),
        ),
        item_semantic_ids=model.item_semantic_ids.detach(),
        item_id_to_semantic_row=tiger_runner.restrict_item_id_mapping_to_scorer_vocab(
            tiger_runner.item_id_to_semantic_row(dm), sasrec
        ),
        scorer=sasrec,
    )

    # In-training validation stays light: constrained beam-5 generation and
    # only the NDCG@5 monitor (val/sft/ndcg@5 via the alias). The full metric
    # set runs on validation + test at the end.
    sft_cfg = SFTConfig(
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
    cache = ComposedRewardCache(
        [(key, cache_dirs[key], w) for key, w in zip(entry["components"], entry["weights"])],
        composition=str(entry["composition"]),
        power=float(entry["power"]),
    )
    base_gflow_cfg = GFlowConfig(
        objective=str(entry["objective"]),
        sft_weight=1.0,
        gflow_weight=1.0,  # placeholder; calibrated below
        # Informational: the composed cache is injected directly (make_method);
        # the power is applied there, so the config keeps power 1.
        reward_cache_path=str(cache_dirs[entry["components"][0]]),
        reward_cache_power=1.0,
        tb_loss_weighting="uniform",
        candidate_sampler="sasrec",
        num_negative_samples=int(entry["num_negatives"]),
        easy_negative_steps=int(args.curriculum_steps),
        candidate_sampling_temperature=float(args.sampler_temperature),
    )

    run_batch, run_accum = gflow_ablation.batch_settings_for_k(int(entry["num_negatives"]), args)
    if run_batch != int(args.batch_size):
        print(f"[{tag}] candidate-memory scaling: batch {args.batch_size} -> {run_batch}, "
              f"grad_accum {args.grad_accum} -> {run_accum}", flush=True)

    if args.gflow_weight_mode == "fixed_grad_ratio":
        calibration = calibrate_gflow_weight(
            model=model, base_gflow_cfg=base_gflow_cfg, sampler=sampler, sft_cfg=sft_cfg,
            cache=cache, dm=dm, subset_indices=subset_indices, device=device, args=args,
            tag=tag, target_ratio=float(entry["rho"]), batch_size=run_batch,
        )
        lambda_gflow = float(calibration["fixed_gflow_weight"])
    else:
        calibration = {"mode": "constant", "fixed_gflow_weight": float(args.gflow_constant_weight)}
        lambda_gflow = float(args.gflow_constant_weight)
    write_json(run_dir / "calibration.json", calibration)

    method = make_method(replace(base_gflow_cfg, gflow_weight=lambda_gflow), sampler, sft_cfg, cache)

    max_epochs = 1 if args.smoke else int(args.max_epochs)
    max_train_batches = 8 if args.smoke else args.max_train_batches
    max_eval_batches = 2 if args.smoke else args.max_eval_batches
    trainer = Trainer.from_optimizer_config(
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
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        logger=heartbeat_logger(gflow_ablation.make_status_logger(tag), claims, run_dir),
    )

    resolved = {
        **entry_identity(entry, int(args.seed)),
        "variant": "gflow",
        "model": model.config_dict(),
        "model_profile": profile,
        "init_checkpoint": str(tuned_checkpoint_paths(dataset)["checkpoint"]),
        "sasrec_checkpoint": run_config["method"]["sasrec"]["checkpoint_path"],
        "gflow": method.cfg.to_dict(),
        "gflow_weight_mode": str(args.gflow_weight_mode),
        "rho_target": float(entry["rho"]),
        "calibration": {k: v for k, v in calibration.items() if k != "per_batch"},
        "sampler": {
            "type": "sasrec_curriculum", "mode": "curriculum",
            "num_negatives": int(entry["num_negatives"]),
            "include_positive": True, "exclude_history": True,
            "curriculum_steps": int(args.curriculum_steps),
            "temperature": float(args.sampler_temperature),
        },
        "reward": {
            **cache.describe(),
            "power_applied": "ComposedRewardCache (component caches store raw rewards)",
            "combo_shape": combo_key(entry["combo"]),
            "similarity_mode": entry.get("similarity_mode"),
            "component_specs": {key: component_spec_from_config(run_config, key)
                                for key in entry["components"]},
            "cache_dirs": {key: str(cache_dirs[key]) for key in entry["components"]},
        },
        "batching": {"batch_size": run_batch, "gradient_accumulation": run_accum,
                     "effective_batch": run_batch * run_accum},
        "data": {
            **runner.data_config_paths(dm.cfg),
            "train_example_layout": "prefix_target",
            "finetune_per_user_last_n": int(args.train_last_n),
            "num_train_examples": len(dm.train_dataset),
            "num_finetune_examples": len(subset_indices),
            **dm.summary(),
        },
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] fine-tuning: lambda={lambda_gflow:.6g}, epochs<={max_epochs}", flush=True)
    fit = trainer.fit(
        gflow_ablation.indexed_train_loader(
            dm, subset_indices, shuffle=not args.smoke,
            batch_size=int(run_batch), seed=int(args.seed),
        ),
        dm.val_dataloader(),
    )
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    checkpoint_for_eval = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_eval:
        trainer.load_checkpoint(checkpoint_for_eval, load_optimizer=False, load_scheduler=False)

    splits = gpt2_beauty.evaluate_splits(model, dm, args, device, max_eval_batches,
                                         splits=("validation", "test"))
    write_json(run_dir / "final_metrics.json", splits)

    result = {
        **entry_identity(entry, int(args.seed)),
        "status": "completed",
        "lambda_gflow": lambda_gflow,
        "raw_grad_ratio_median": calibration.get("raw_grad_ratio_median"),
        "grad_cosine_median": calibration.get("grad_cosine_median"),
        "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1),
        "elapsed_sec": time.time() - started,
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


def build_dataset_context(
    dataset: str, components: Sequence[str], *, args: argparse.Namespace,
    device: torch.device, output_root: Path, csv_path: Path,
) -> DatasetContext | None:
    """Datamodule + sanity checks + SASRec teacher + component caches (+ the
    init baseline) for one dataset. Returns None after the cache phase when
    --cache-only."""
    validate_inputs(dataset, args)
    model_yaml = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    profile, _ = tuned_profile(dataset)
    user_hash_mod = gpt2_runner.user_ids_count(model_yaml, profile)
    run_config = build_run_config(dataset, args, user_hash_mod)
    config_dir = output_root / dataset
    config_path = config_dir / "data_reward_config.json"
    write_json(config_path, run_config)

    data_cfg = tiger_runner.build_data_config(dict(run_config), config_path=config_path)
    dm = tiger_runner.build_data_module(data_cfg)
    dm.setup()
    if dm.semantic_ids is None or dm.sid_len is None:
        raise ValueError(f"[{dataset}] datamodule produced no semantic IDs.")
    store = dm.interaction_store
    subset_indices = per_user_tail_indices(dm.train_dataset, int(args.train_last_n))
    print(
        f"[{dataset}] catalog={int(dm.item_vocab_size)} ids, train={len(dm.train_dataset)} "
        f"prefix examples (fine-tune subset: {len(subset_indices)}), "
        f"aux store={len(store) if store is not None else 0} events",
        flush=True,
    )

    # Refuse to fine-tune a checkpoint that does not match this datamodule.
    probe, _ = build_gpt2_from_tuned_checkpoint(dataset, dm)
    probe.to(device)
    if not args.skip_sanity_check:
        nll = gpt2_beauty.sanity_check_init_nll(
            probe, dm, device, threshold=float(args.sanity_nll_threshold))
        print(f"[{dataset}] init sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
              f"(3-batch subset; tuning recorded ~{spec['teacher_ndcg_reference']})", flush=True)

    # Component caches (raw rewards; composition happens at lookup time).
    cache_dirs: dict[str, Path] = {}
    for key in components:
        cache_dir = component_cache_dir(dataset, key, args)
        manifest = ensure_component_cache(
            dataset, key, run_config, config_dir, len(subset_indices), args)
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


def execute_entry(
    entry: Mapping[str, Any], ctx: DatasetContext, run_dir: Path, *, args: argparse.Namespace,
    csv_path: Path, claims: ClaimStore | None = None,
) -> dict[str, Any]:
    """run_single with the grid's failure policy: a failing run is recorded
    (result.json status=failed + CSV rows) and never aborts the grid."""
    try:
        return run_single(
            entry=entry, run_config=ctx.run_config, dm=ctx.dm, subset_indices=ctx.subset_indices,
            sasrec=ctx.sasrec, cache_dirs=ctx.cache_dirs, run_dir=run_dir, csv_path=csv_path,
            args=args, claims=claims,
        )
    except Exception as exc:  # noqa: BLE001 - keep the grid going
        print(f"[{entry_tag(entry)}] FAILED: {exc}", flush=True)
        traceback.print_exc()
        identity = entry_identity(entry, int(args.seed))
        result = {**identity, "status": "failed", "error": str(exc), "run_dir": str(run_dir)}
        write_json(run_dir / RESULT_FILENAME, result)
        upsert_metrics_csv(csv_path, [
            {**identity, "variant": "gflow", "split": split,
             "status": "failed", "error": str(exc)[:300]}
            for split in ("validation", "test")
        ])
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        return result


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
        entries = [e for e in grid if e["dataset"] == dataset]
        needed = components_for(entries)
        entries = rotate(entries, (wid * len(entries)) // count)
        ctx: DatasetContext | None = None
        budget_hit = False
        for entry in entries:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            if claims.status_of(run_dir) in {"completed", "running"}:
                continue
            elapsed_h = (time.time() - started) / 3600.0
            estimate_h = max(run_hours) if run_hours else float(args.run_hours_estimate)
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
                                                output_root=output_root, csv_path=csv_path)
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
    csv_path = abspath(args.artifacts_dir) / "gflow_finetune_gpt2_amazon_metrics_summary.csv"

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
        pending = []
        for entry in entries:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
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
                                    output_root=output_root, csv_path=csv_path)
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
