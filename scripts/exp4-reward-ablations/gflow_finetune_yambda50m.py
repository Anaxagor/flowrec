#!/usr/bin/env python
"""GFlow fine-tuning ablation on the Yambda 50m exp2 TIGER checkpoint.

Grid (420 runs by default, single seed):
  reward combos (7)   subsets of {sasrec, interaction, similarity}, sizes 1-3,
                      composed multiplicatively (MultiplicativeRewardModel);
  reward powers (5)   {0.25, 1, 5, 16, 32}, applied at cache-load time via
                      GFlowConfig.reward_cache_power;
  objectives (3)      {tb, db, subtb};
  num negatives (4)   {3, 10, 100, 1000}, always the curriculum sampler over
                      SASRec-teacher scores (easy -> hard).

Per run:
  - TIGER init: the promoted yambda_50m SFT checkpoint
    (checkpoints/exp2-tuned_params/initial/tiger/yambda_50m/checkpoints/best.pt),
    guarded by the teacher-forced-NLL sanity check.
  - Data: the checkpoint's exact positive stream, reproduced via
    positive_criterion="event_type == 'listen' and played_ratio_pct >= 50"
    instead of the destructive scope filters, so likes/dislikes/partial
    listens land in the auxiliary store for the interaction (engagement)
    reward. The datamodule keeps the checkpoint's full example layout
    (train_last_n=128, catalog 112,943 items — the post-split cold-item
    filter derives the catalog from the train example layout, so reducing
    train_last_n in the config would shrink the catalog and break the
    checkpoint load). Fine-tuning instead runs on a deterministic per-user
    tail SUBSET of that layout (--train-last-n, default 8 -> ~70k examples),
    which keeps each dense full-catalog reward cache at ~16 GB (float16)
    instead of ~224 GB.
  - Reward cache: one cache per combo (shared by all powers, objectives, and
    K), built by scripts/precompute_reward_cache.py with
    --store-power-sums for every power in the grid. logZ and DB/SubTB prefix
    flows are analytical from the cache (main-branch GFlowMethod contract).
  - GFlow weight: fixed gradient-norm calibration (the "fixed_grad_ratio"
    rule): lambda = rho_target / median_b(||g_GFlow,b|| / ||g_SFT,b||),
    measured once at the init checkpoint with torch.autograd.grad over the
    shared TIGER parameters (this analytical setup has no GFlow-only
    parameters), full precision, before any optimizer update, then frozen.
    rho_target defaults to 1.0 (equal initial gradient norms). Gradient
    cosine is logged as a diagnostic only.
  - Sampler: curriculum over SASRec scores, exclude_history=True, easy
    phase fixed at --curriculum-steps 3000 batch-steps (~11 epochs at
    last_n=8 / batch 256): long enough that the joint loss starts on easy
    negatives right after calibration, short enough that most of training
    runs at full difficulty and the phase is identical for every K.
  - Training: lr 3e-5, batch 256 x accum 8 (effective 2048, the SFT-control
    protocol), monitor val/sft/ndcg@5 (constrained beam-5), <= 500 epochs,
    patience 10.
  - After fine-tuning: beam-20 HR / NDCG / Coverage / ARP @ {5, 10, 20} on
    validation + test (sft_finetune_control's CatalogIndex protocol), saved
    per run and upserted into
    artifacts/exp4-reward_ablation/gflow_finetune_yambda_50m_metrics_summary.csv.
    The untouched init checkpoint is evaluated once as variant="init".

scheduler usage:
  python scripts/exp4-reward-ablations/gflow_finetune_yambda50m.py --list
  python scripts/exp4-reward-ablations/gflow_finetune_yambda50m.py --index $WORKER_ID
  # or filtered sweeps:
  python scripts/exp4-reward-ablations/gflow_finetune_yambda50m.py \
      --combos sasrec_x_interaction --powers 1 --objectives tb --num-candidates 10
Consecutive --index values share a reward cache (combo-major order), so array
chunks of 60 stay on one cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR), str(SCRIPTS_DIR / "exp2-tuned-params")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
import sft_finetune_control as control  # noqa: E402
from precompute_reward_cache import per_user_tail_indices  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import MethodOutput  # noqa: E402
from flowrec.ft_methods.calibration import (  # noqa: E402
    calibrate_from_loss_pairs,  # re-exported: the GPT-2 scripts and tests import it from here
    gradient_vector,
    variant_loss_pairs,
)
from flowrec.ft_methods.gflow.config import GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.method import GFlowMethod  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.ft_methods.sft.method import SFTMethod  # noqa: E402
from flowrec.models.base import Batch, FlowRecModel  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Assets and grid
# --------------------------------------------------------------------------- #
DATASET = "yambda_50m_listens"
SFT_CONFIG = "configs/exp2-tuned-params/tiger_yambda_50m_listens_sft.json"
TIGER_CHECKPOINT = "checkpoints/exp2-tuned_params/initial/tiger/yambda_50m/checkpoints/best.pt"
SASREC_CHECKPOINT = "checkpoints/exp2-tuned_params/initial/sasrec/yambda_50m_listens/best_model.pt"
SASREC_TRIAL = "checkpoints/exp2-tuned_params/initial/sasrec/yambda_50m_listens/best_trial.json"
RQVAE_MODEL = "datasets/semantic_ids/exp2-tuned-params/yambda_50m_listens/model.pt"
# Same positive stream as the checkpoint's allowed_event_types=[listen] +
# min_played_ratio_pct=50, but the non-criterion events feed the aux store.
POSITIVE_CRITERION = "event_type == 'listen' and played_ratio_pct >= 50"
# yambda_50m semantic IDs are level-offset (token = code + level*256, suffix
# at 768+) with NO +1 offset (offset_semantic_ids_by_one=false).
LEVEL_STRIDE = 256

# Canonical component order; combo keys join names with "_x_".
COMPONENT_ORDER = ("sasrec", "interaction", "similarity")


def component_spec(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "sasrec":
        return {
            "name": "sasrec",
            "weight": 1.0,
            "reward": {
                "transform": str(args.sasrec_transform),
                "temperature": float(args.sasrec_temperature),
            },
        }
    if name == "interaction":
        # Engagement scoring + evidence-scored positives (the module defaults).
        return {"name": "interaction", "weight": 1.0, "reward": {}}
    if name == "similarity":
        return {
            "name": "similarity",
            "weight": 1.0,
            "reward": {
                "mode": "hierarchical",
                "level_stride": LEVEL_STRIDE,
                "rqvae_model_path": str(abspath(RQVAE_MODEL)),
            },
        }
    raise ValueError(f"Unknown reward component {name!r}")


def all_combos() -> list[tuple[str, ...]]:
    """Ordered subsets of COMPONENT_ORDER, sizes 1-3, canonical order."""
    return [
        ("sasrec",), ("interaction",), ("similarity",),
        ("sasrec", "interaction"), ("sasrec", "similarity"),
        ("interaction", "similarity"),
        ("sasrec", "interaction", "similarity"),
    ]


def combo_key(combo: tuple[str, ...]) -> str:
    return "_x_".join(combo)


DEFAULT_POWERS = (0.25, 1.0, 5.0, 16.0, 32.0)
DEFAULT_OBJECTIVES = ("tb", "db", "subtb")
DEFAULT_NUM_NEGATIVES = (3, 10, 100, 1000)

# The fine-tuning protocol (shared with the SFT-only control) lives in this
# config; parse_args loads it as the argparse defaults so the script and the
# documented protocol cannot drift (tests/test_finetune_protocol_configs.py
# additionally asserts parity with sft_finetune_control's defaults).
PROTOCOL_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_gflow.json"


def protocol_defaults() -> dict[str, Any]:
    if not PROTOCOL_CONFIG.exists():
        return {}
    payload = json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    return {**payload.get("shared_with_sft_control", {}), **payload.get("gflow", {})}


def parse_args() -> argparse.Namespace:
    D = protocol_defaults()
    p = argparse.ArgumentParser(
        "GFlow fine-tuning ablation on yambda_50m: reward combos x powers x "
        "objectives x curriculum negative counts, gradient-norm-calibrated "
        "gflow weight."
    )
    combo_keys = [combo_key(c) for c in all_combos()]
    p.add_argument("--combos", nargs="+", choices=combo_keys, default=combo_keys)
    p.add_argument("--powers", type=float, nargs="+", default=list(DEFAULT_POWERS))
    p.add_argument("--objectives", nargs="+", choices=list(DEFAULT_OBJECTIVES),
                   default=list(DEFAULT_OBJECTIVES))
    p.add_argument("--num-candidates", type=int, nargs="+", default=list(DEFAULT_NUM_NEGATIVES))
    p.add_argument("--seed", type=int, default=int(D.get("seed", 42)))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Grid selection (scheduler arrays).
    p.add_argument("--list", action="store_true", help="Print the run grid with indices and exit")
    p.add_argument("--index", type=str, default=None,
                   help="Run one grid entry ('137') or a contiguous inclusive range "
                        "('135-139') sequentially (see --list). Ranges let one scheduler "
                        "array task cover several entries when the QOS job-submit "
                        "limit is tighter than the grid; an end past the grid is "
                        "clamped, so the last block may be short.")
    p.add_argument("--force", action="store_true",
                   help="Re-run entries whose result.json already reports completed")

    # Data scale (cache tractability).
    p.add_argument("--train-last-n", type=int, default=int(D.get("train_last_n", 8)),
                   help="Fine-tune on the last N train examples per user — a subset of "
                        "the checkpoint's last_n=128 layout, so the catalog is untouched. "
                        "8 keeps each dense reward cache ~16 GB")

    # Reward pieces. sigmoid keeps the sasrec component bounded in (0, 1):
    # with reward powers up to 32 applied in float32 at cache load, an
    # unbounded softplus reward (~logit magnitude, >16 is common) would
    # overflow pow(32) to inf and poison logZ.
    p.add_argument("--sasrec-transform", default=str(D.get("sasrec_transform", "sigmoid")))
    p.add_argument("--sasrec-temperature", type=float, default=float(D.get("sasrec_temperature", 1.0)))

    # Sampler (fixed easy-negative phase).
    p.add_argument("--curriculum-steps", type=int, default=int(D.get("curriculum_steps", 3000)),
                   help="Batch-steps to ramp curriculum negatives easy -> hard (fixed for all runs)")
    p.add_argument("--sampler-temperature", type=float, default=float(D.get("sampler_temperature", 1.0)))

    # Gradient-norm calibration of the gflow weight.
    p.add_argument("--gflow-weight-mode", choices=["fixed_grad_ratio", "constant"],
                   default=str(D.get("gflow_weight_mode", "fixed_grad_ratio")))
    p.add_argument("--gflow-target-grad-ratio", type=float,
                   default=float(D.get("gflow_target_grad_ratio", 1.0)),
                   help="rho_target: ||lambda*g_GFlow|| / ||g_SFT|| at init (1.0 = equal norms)")
    p.add_argument("--rho-targets", type=float, nargs="+", default=None,
                   help="Make rho_target a GRID dimension (e.g. 0.1 1 10 = weak / equal / "
                        "strong gflow influence). Multiplies the grid size and changes "
                        "--index mapping — check --list. Default: just "
                        "--gflow-target-grad-ratio, keeping the legacy 420-run mapping. "
                        "rho=1.0 runs keep their legacy run dirs, so completed ones are "
                        "skipped; other rho values get a _rho<r> dir suffix.")
    p.add_argument("--gflow-constant-weight", type=float, default=0.1,
                   help="Used only with --gflow-weight-mode constant")
    p.add_argument("--gflow-calibration-batches", type=int,
                   default=int(D.get("gflow_calibration_batches", 5)))
    p.add_argument("--gflow-calibration-eps", type=float,
                   default=float(D.get("gflow_calibration_eps", 1e-12)))
    p.add_argument("--gflow-calibration-seed", type=int,
                   default=int(D.get("gflow_calibration_seed", 1234)))

    # Reward cache.
    p.add_argument("--cache-batch-size", type=int, default=32,
                   help="Full-catalog reward eval batch for the cache build")
    p.add_argument("--cache-dtype", choices=["float16", "float32"],
                   default=str(D.get("cache_dtype", "float16")))
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true",
                   help="Build/verify the reward caches for the selected combos, then exit "
                        "(run one task per combo before the main array)")

    # Candidate-scoring memory cap: activation memory scales with
    # batch * (K+1) scored candidate sequences. 256 x 1001 allocates ~37 GiB
    # in one tensor (instant OOM on 32 GB); 256 x 101 accumulates to ~31 GiB.
    # The train/calibration batch is scaled down so batch * (K+1) stays under
    # this cap, with gradient accumulation keeping the effective batch fixed.
    p.add_argument("--max-candidate-batch", type=int,
                   default=int(D.get("max_candidate_batch", 4096)),
                   help="Upper bound on batch * (num_negatives + 1) during training")

    # Training (SFT-control protocol).
    p.add_argument("--max-epochs", type=int, default=int(D.get("max_epochs", 500)))
    p.add_argument("--validate-every", type=int, default=int(D.get("validate_every", 5)),
                   help="Validate every N epochs; patience counts validations "
                        "(SFT-control protocol: 5)")
    p.add_argument("--patience", type=int, default=int(D.get("patience", 10)))
    p.add_argument("--validate-at-start", dest="validate_at_start",
                   action=argparse.BooleanOptionalAction,
                   default=bool(D.get("run_validation_at_start", True)),
                   help="Score the init checkpoint before the first epoch: it seeds the best "
                        "value / best.pt and is the reference for early-stopping patience, so a "
                        "run that never improves stops after `patience` validations with the "
                        "init as its best (SFT-control protocol)")
    p.add_argument("--lr", type=float, default=float(D.get("lr", 3e-5)))
    p.add_argument("--weight-decay", type=float, default=float(D.get("weight_decay", 1e-2)))
    p.add_argument("--batch-size", type=int, default=int(D.get("batch_size", 256)))
    p.add_argument("--grad-accum", type=int, default=int(D.get("grad_accum", 8)))
    p.add_argument("--max-grad-norm", type=float, default=float(D.get("max_grad_norm", 0.5)),
                   help="Gradient clipping (SFT-control protocol: 0.5)")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"],
                   default=str(D.get("amp_dtype", "float16")),
                   help="AMP dtype on CUDA (SFT-control protocol: float16)")
    p.add_argument("--val-batch-size", type=int, default=int(D.get("val_batch_size", 256)))
    p.add_argument("--num-workers", type=int, default=8)

    # Final evaluation.
    p.add_argument("--test-ks", type=int, nargs="+", default=list(D.get("test_ks", [5, 10, 20])))
    p.add_argument("--eval-num-beams", type=int, default=int(D.get("eval_num_beams", 20)))

    # Output.
    p.add_argument("--output-root", type=Path, default=Path("experiments/exp4-gflow-yambda50m"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("artifacts/exp4-reward_ablation/gflow_yambda50m/reward_cache"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/exp4-reward_ablation"))
    p.add_argument("--promote-root", type=str, default="",
                   help="Copy best.pt per run here; empty (default) disables promotion "
                        "(420 runs would copy ~10 GB of checkpoints)")

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0)
    p.add_argument("--skip-init-baseline", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: ~256-example cache, 1 epoch, unshuffled")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def power_tag(power: float) -> str:
    return format(float(power), "g").replace(".", "p").replace("-", "m")


class IndexedDataset(Dataset):
    """Wrap a SequenceDataset so every record carries its dataset index (the
    link between a training example and its reward-cache row)."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = dict(self.base[idx])
        record["example_index"] = int(idx)
        return record


class IndexedCollator:
    def __init__(self, base) -> None:
        self.base = base

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base(examples)
        batch["example_indices"] = torch.as_tensor(
            [int(ex["example_index"]) for ex in examples], dtype=torch.long
        )
        return batch


def indexed_train_loader(dm, subset_indices: list[int] | None, *, shuffle: bool,
                         batch_size: int, seed: int,
                         num_workers: int | None = None,
                         reward_cache=None, prefetch_factor: int = 2,
                         tensorized: bool = True) -> DataLoader:
    """Train loader over the per-user tail subset; ``example_indices`` are
    SUBSET positions, matching the rows of a cache built with the same
    --per-user-last-n."""
    from flowrec.data.indexed_batches import create_indexed_parts
    from flowrec.ft_methods.gflow.prefetch import RewardPrefetchCollator

    dataset, collator = create_indexed_parts(dm, subset_indices, tensorized=tensorized)
    if reward_cache is not None:
        collator = RewardPrefetchCollator(collator, reward_cache.make_row_reader())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    workers = dm.cfg.num_workers if num_workers is None else int(num_workers)
    if workers < 0 or int(prefetch_factor) < 1:
        raise ValueError("num_workers must be nonnegative and prefetch_factor positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=dm.cfg.pin_memory,
        persistent_workers=bool(dm.cfg.persistent_workers) and workers > 0,
        drop_last=False,
        collate_fn=collator,
        generator=generator if shuffle else None,
        **({"prefetch_factor": int(prefetch_factor)} if workers > 0 else {}),
    )


class AliasedSFTMethod(SFTMethod):
    """SFT whose generation_ndcg@K is aliased to ndcg@K so the trainer can
    monitor ``val/sft/ndcg@5``."""

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        out = super().compute_loss(model, batch, stage=stage, step=step)
        metrics = dict(out.metrics)
        prefix = self.cfg.metric_prefix.rstrip("/")
        for k in self.cfg.generation_topk:
            for src, dst in (
                (f"{prefix}/generation_recall@{k}", f"{prefix}/recall@{k}"),
                (f"{prefix}/generation_ndcg@{k}", f"{prefix}/ndcg@{k}"),
            ):
                if src in metrics:
                    metrics[dst] = metrics[src]
        return MethodOutput(loss=out.loss, metrics=metrics, model_output=out.model_output, extra=out.extra)


# --------------------------------------------------------------------------- #
# Fixed gradient-norm calibration of the gflow weight
# --------------------------------------------------------------------------- #
# The maths (gradient_vector / calibrate_from_loss_pairs) and the two-variant
# pairing (variant_loss_pairs) live in flowrec.ft_methods.calibration, shared
# with the RL methods; this script only assembles the two GFlow variants.
def calibrate_gflow_weight(
    *,
    model,
    base_gflow_cfg: GFlowConfig,
    sampler,
    sft_cfg: SFTConfig,
    dm,
    subset_indices: list[int],
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
    target_ratio: float,
    batch_size: int,
) -> dict[str, Any]:
    """One-time calibration at the init checkpoint, before any optimizer step.

    Two GFlowMethod variants isolate the unweighted losses through the exact
    training code path: (sft_weight, gflow_weight) = (1, 0) gives L_SFT and
    (0, 1) gives L_GFlow. Both share the sampler and the mmap'd reward cache,
    and each pass re-seeds the RNG identically, so dropout masks and sampled
    candidates match across the pair. step=0 keeps the curriculum sampler in
    its initial (easy) phase — the gradient the run actually starts with.
    Full precision (no autocast/GradScaler); model weights, optimizer state,
    and the training dataloader order are untouched (a separate deterministic
    calibration loader is used, and the RNG is re-seeded after).
    """
    sft_variant = GFlowMethod(
        replace(base_gflow_cfg, sft_weight=1.0, gflow_weight=0.0),
        candidate_sampler=sampler,
        sft_method=AliasedSFTMethod(sft_cfg),
    )
    gflow_variant = GFlowMethod(
        replace(base_gflow_cfg, sft_weight=0.0, gflow_weight=1.0),
        candidate_sampler=sampler,
        sft_method=AliasedSFTMethod(sft_cfg),
    )
    gflow_variant._reward_cache = sft_variant._get_reward_cache()  # share the mmap

    params = [p for p in model.parameters() if p.requires_grad]
    # Smoke caches cover only the first examples, so calibrate unshuffled there.
    loader = indexed_train_loader(
        dm, subset_indices, shuffle=not args.smoke, batch_size=int(batch_size),
        seed=int(args.gflow_calibration_seed), num_workers=0,
    )
    num_batches = 1 if args.smoke else int(args.gflow_calibration_batches)

    was_training = model.training
    model.train()

    loss_pairs = variant_loss_pairs(
        model, sft_variant, gflow_variant, loader,
        seed=int(args.gflow_calibration_seed), step=0,
        num_batches=num_batches, device=device,
    )
    result = calibrate_from_loss_pairs(
        loss_pairs, params,
        target_ratio=float(target_ratio),
        eps=float(args.gflow_calibration_eps),
    )
    model.train(was_training)
    # Restore the run's RNG so calibration does not shift training.
    tiger_runner.seed_everything(int(args.seed))

    print(
        f"[{tag}] calibration: ||g_SFT||~{result['sft_grad_norm_median']:.4g} "
        f"||g_GFlow||~{result['gflow_grad_norm_median']:.4g} "
        f"r0={result['raw_grad_ratio_median']:.4g} "
        f"cos={result['grad_cosine_median']:.3f} "
        f"rho_target={result['target_grad_ratio']} "
        f"-> lambda={result['fixed_gflow_weight']:.6g}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# Config assembly + reward cache
# --------------------------------------------------------------------------- #
def sasrec_method_cfg() -> dict[str, Any]:
    payload = load_json(abspath(SASREC_TRIAL))
    params = payload.get("best_params", payload)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
    return {
        "checkpoint_path": str(abspath(SASREC_CHECKPOINT)),
        "model": {k: params[k] for k in model_keys if k in params},
    }


def build_run_config(combo: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
    """One config drives the datamodule, the reward registry, and the cache
    precompute — a single source of truth for reward semantics."""
    config = copy.deepcopy(load_json(abspath(SFT_CONFIG)))
    config["experiment_name"] = f"gflow_yambda50m_{combo_key(combo)}"
    config["seed"] = int(args.seed)
    config["data"]["interactions_path"] = str(abspath(config["data"]["interactions_path"]))
    config["data"]["semantic_id_path"] = str(abspath(config["data"]["semantic_id_path"]))

    dataset_cfg = dict(config.get("dataset", {}))
    # Criterion split instead of destructive scope filters: identical positive
    # stream/catalog, but the aux store gets likes/dislikes/partial listens.
    # train_last_n stays at the checkpoint's 128 — the catalog depends on the
    # train example layout; fine-tuning subsets the examples instead.
    dataset_cfg["allowed_event_types"] = None
    dataset_cfg["min_played_ratio_pct"] = None
    dataset_cfg["positive_criterion"] = POSITIVE_CRITERION
    config["dataset"] = dataset_cfg

    dataloader = dict(config.get("dataloader", {}))
    dataloader["train_batch_size"] = int(args.batch_size)
    dataloader["batch_size"] = int(args.batch_size)
    dataloader["validation_batch_size"] = int(args.val_batch_size)
    dataloader["num_workers"] = int(args.num_workers)
    dataloader["persistent_workers"] = int(args.num_workers) > 0
    config["dataloader"] = dataloader

    config["method"] = {
        "name": "gflow",
        "reward_model": "multiply",
        "reward_epsilon": 1e-6,
        "reward": {"components": [component_spec(name, args) for name in combo]},
        "sasrec": sasrec_method_cfg(),
    }
    return config


def _smoke_examples(args: argparse.Namespace) -> int:
    # Must cover every example the smoke training pass touches: 8 unshuffled
    # train batches plus the (unshuffled, in-smoke) calibration batch.
    return int(args.batch_size) * 8


def ensure_reward_cache(
    combo: tuple[str, ...], config_path: Path, cache_dir: Path,
    num_subset_examples: int, args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    # Serialize concurrent array tasks: the first builds, the rest wait on the
    # lock and find the manifest.
    with control.file_lock(cache_dir.parent / f"{cache_dir.name}.build.lock"):
        if manifest_path.exists() and not args.force_cache:
            manifest = load_json(manifest_path)
            return _check_cache_manifest(manifest, cache_dir, num_subset_examples, args)
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
            "--store-power-sums", *[format(float(p), "g") for p in args.powers],
        ]
        if args.smoke:
            cmd += ["--max-examples", str(_smoke_examples(args))]
        print(f"[{combo_key(combo)}] precomputing reward cache: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        manifest = load_json(manifest_path)
    return _check_cache_manifest(manifest, cache_dir, num_subset_examples, args)


def _check_cache_manifest(
    manifest: dict[str, Any], cache_dir: Path, num_subset_examples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stored_last_n = manifest.get("per_user_last_n")
    if stored_last_n is not None and int(stored_last_n) != int(args.train_last_n):
        raise RuntimeError(
            f"Reward cache at {cache_dir} was built with --per-user-last-n "
            f"{stored_last_n}, but this run uses {args.train_last_n}. "
            "Rebuild with --force-cache."
        )
    cached = int(manifest.get("num_examples", 0))
    expected = _smoke_examples(args) if args.smoke else num_subset_examples
    if cached < min(expected, num_subset_examples):
        raise RuntimeError(
            f"Reward cache at {cache_dir} covers {cached} examples but the fine-tuning "
            f"subset has {num_subset_examples}. Rebuild with --force-cache (config drift?)."
        )
    stored_powers = {float(p) for p in manifest.get("stored_power_sums", [])}
    missing = [p for p in args.powers if float(p) != 1.0 and float(p) not in stored_powers]
    if missing:
        raise RuntimeError(
            f"Reward cache at {cache_dir} lacks power sums for {missing} "
            f"(stored: {sorted(stored_powers)}). Rebuild with --force-cache."
        )
    return manifest


def verify_cache_alignment(dm, subset_indices: list[int], cache_dir: Path) -> None:
    """Spot-check that subset example order matches the cache rows.

    The cache stores per-example target_item_ids; compare the first batch of
    an unshuffled indexed loader against them. Catches config drift between
    the cache build and this run (the tie-ordering trap this repo knows).
    """
    manifest = load_json(cache_dir / "manifest.json")
    shard = manifest["shards"][0]
    shard_path = Path(shard["path"])
    if not shard_path.is_absolute():
        shard_path = cache_dir / shard_path
    payload = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=True)
    cached_targets = payload["target_item_ids"].reshape(-1)

    loader = indexed_train_loader(dm, subset_indices, shuffle=False,
                                  batch_size=min(256, cached_targets.numel()),
                                  seed=0, num_workers=0)
    batch = next(iter(loader))
    n = min(int(batch["target_item_ids"].numel()), cached_targets.numel())
    ours = batch["target_item_ids"].reshape(-1)[:n].long()
    theirs = cached_targets[:n].long()
    agree = float((ours == theirs).float().mean().item())
    if agree < 1.0:
        raise RuntimeError(
            f"Reward cache at {cache_dir} disagrees with the datamodule on "
            f"{(1 - agree) * 100:.1f}% of the first {n} targets — the cache was "
            "built from a different data configuration/order. Rebuild with --force-cache."
        )


@torch.no_grad()
def sanity_check_teacher(
    sasrec, dm, device: torch.device, *, batches: int = 3, threshold: float = 0.02
) -> float:
    """Full-catalog val NDCG@10 of the (possibly id-translated) SASRec teacher.

    The yambda_50m tuning recorded val NDCG@10 ~0.117; a wrong id translation
    scores near zero. Guards the pre-filter/post-filter id mapping end to end.
    """
    sasrec.eval()
    total, ndcg = 0, 0.0
    for i, raw_batch in enumerate(dm.val_dataloader()):
        if i >= batches:
            break
        batch = move_batch(raw_batch, device)
        scores = sasrec.score_items(batch)  # [B, dm.item_vocab_size]
        target = batch["target_item_ids"].long()
        target_scores = scores.gather(1, target.unsqueeze(1))
        rank = scores.gt(target_scores).sum(dim=1)  # 0-based rank of the target
        hit = rank < 10
        if hit.any():
            ndcg += float((1.0 / torch.log2(rank[hit].float() + 2.0)).sum().item())
        total += int(target.size(0))
    value = ndcg / max(total, 1)
    if value < threshold:
        raise RuntimeError(
            f"SASRec teacher sanity check FAILED: val NDCG@10 {value:.4f} < "
            f"{threshold} (tuning recorded ~0.117). The teacher's item-id "
            "space does not match the datamodule; refusing to sample/score "
            "with a misaligned teacher."
        )
    return value


# --------------------------------------------------------------------------- #
# Metrics CSV (file-locked upsert, one row per run x split)
# --------------------------------------------------------------------------- #
CSV_KEY = ("dataset", "combo", "power", "objective", "num_negatives", "seed",
           "variant", "split", "target_grad_ratio")


def upsert_metrics_csv(
    csv_path: Path, new_rows: list[dict[str, Any]], *,
    key: tuple[str, ...] | None = None,
    legacy_defaults: Any = None,
) -> None:
    """File-locked upsert. ``key`` extends the row identity (default CSV_KEY);
    ``legacy_defaults(values)`` may fill key columns that older rows lack, so
    reruns replace them instead of sitting next to them."""
    import csv as _csv

    CSV_KEY = tuple(key) if key is not None else globals()["CSV_KEY"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(".lock")
    with control.file_lock(lock_path):
        rows: list[dict[str, Any]] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = [dict(r) for r in _csv.DictReader(fh)]

        def key_of(row: Mapping[str, Any]) -> tuple[str, ...]:
            values = {k: str(row.get(k, "")) for k in CSV_KEY}
            # Rows written before rho became a grid dimension lack the column;
            # every such run used rho=1.0, so normalize — reruns then replace
            # their stale failed rows instead of sitting next to them.
            if not values["target_grad_ratio"] and values["variant"] == "gflow":
                values["target_grad_ratio"] = "1.0"
            if legacy_defaults is not None:
                legacy_defaults(values)
            return tuple(values[k] for k in CSV_KEY)

        merged = {key_of(r): r for r in rows}
        for row in new_rows:
            merged[key_of(row)] = {k: ("" if v is None else v) for k, v in row.items()}
        all_rows = list(merged.values())
        fieldnames = list(CSV_KEY) + sorted(
            {k for r in all_rows for k in r} - set(CSV_KEY)
        )
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(all_rows, key=key_of):
                writer.writerow({k: row.get(k, "") for k in fieldnames})


def metric_rows(
    identity: Mapping[str, Any], splits: Mapping[str, Mapping[str, float]],
    extra: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in splits.items():
        rows.append({
            **identity, "split": split,
            **{k: v for k, v in metrics.items()},
            **extra,
        })
    return rows


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = [c for c in all_combos() if combo_key(c) in set(args.combos)]
    rhos = args.rho_targets or [args.gflow_target_grad_ratio]
    grid = []
    for combo in combos:  # combo-major: consecutive indices share a cache
        for power in args.powers:
            for rho in rhos:
                for objective in args.objectives:
                    for k in args.num_candidates:
                        grid.append({
                            "combo": combo,
                            "power": float(power),
                            "rho": float(rho),
                            "objective": str(objective),
                            "num_negatives": int(k),
                        })
    return grid


def _rho_suffix(rho: float) -> str:
    # rho=1.0 keeps the legacy layout so already-completed runs are found.
    return "" if float(rho) == 1.0 else f"_rho{power_tag(rho)}"


def run_dir_for(output_root: Path, entry: Mapping[str, Any], seed: int) -> Path:
    return (
        output_root / combo_key(entry["combo"]) / f"pow_{power_tag(entry['power'])}"
        / f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}"
        / f"seed_{seed}"
    )


def entry_tag(entry: Mapping[str, Any]) -> str:
    return (f"{combo_key(entry['combo'])}/p{power_tag(entry['power'])}/"
            f"{entry['objective']}_k{entry['num_negatives']}{_rho_suffix(entry['rho'])}")


def batch_settings_for_k(num_negatives: int, args: argparse.Namespace) -> tuple[int, int]:
    """(batch_size, grad_accum) keeping batch*(K+1) <= max_candidate_batch and
    the effective batch (batch * accum) constant.

    K=3/10 keep 256x8; K=100 -> 40x51; K=1000 -> 4x512. Note per-epoch compute
    is inherently ~(K+1)-proportional (every example scores K+1 candidate
    sequences), so K=1000 epochs cost ~90x a K=10 epoch regardless of batching.
    """
    base_batch = int(args.batch_size)
    effective = base_batch * int(args.grad_accum)
    batch = max(1, min(base_batch, int(args.max_candidate_batch) // (int(num_negatives) + 1)))
    accum = max(1, effective // batch)
    return batch, accum


def curriculum_steps_for_epochs(epochs: float, num_examples: int, run_batch: int) -> int:
    """Sampler micro-steps of ``epochs`` passes over the fine-tuning examples.

    The curriculum sampler counts one step per MICRO-batch, so a ramp given in
    micro-steps changes its length in passes with the data scope (3000 steps:
    ~11 epochs of the 50m last-8 subset, <1 epoch of the full layout), with
    the micro-batch and therefore with K. Defined in epochs it is invariant
    to all three.
    """
    steps_per_epoch = -(-int(num_examples) // max(int(run_batch), 1))
    return max(0, int(round(float(epochs) * steps_per_epoch)))


# --------------------------------------------------------------------------- #
# Init baseline (once, file-locked)
# --------------------------------------------------------------------------- #
def ensure_init_baseline(
    *, config: Mapping[str, Any], config_path: Path, dm, index, device: torch.device,
    args: argparse.Namespace, output_root: Path, csv_path: Path,
) -> None:
    baseline_path = output_root / "baseline" / "init_metrics.json"
    if baseline_path.exists() or args.skip_init_baseline:
        return
    with control.file_lock(baseline_path.with_suffix(".lock")):
        if baseline_path.exists():
            return
        print("[init] evaluating the untouched init checkpoint ...", flush=True)
        model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
        control.load_tiger_checkpoint(model, abspath(TIGER_CHECKPOINT))
        model.to(device).eval()
        max_eval = 2 if args.smoke else args.max_eval_batches
        splits = {
            "validation": control.evaluate_split(
                model, dm.val_dataloader(), index,
                ks=tuple(args.test_ks), num_beams=int(args.eval_num_beams), max_batches=max_eval),
            "test": control.evaluate_split(
                model, dm.test_dataloader(), index,
                ks=tuple(args.test_ks), num_beams=int(args.eval_num_beams), max_batches=max_eval),
        }
        payload = {"dataset": DATASET, "checkpoint": str(abspath(TIGER_CHECKPOINT)), **splits}
        write_json(baseline_path, payload)
        identity = {"dataset": DATASET, "combo": "-", "power": "-", "objective": "-",
                    "num_negatives": "-", "seed": int(args.seed), "variant": "init"}
        upsert_metrics_csv(csv_path, metric_rows(identity, splits, {"status": "completed"}))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def make_status_logger(tag: str):
    keys = (
        "train/loss", "train/gflow/sft_loss", "train/gflow/gflow_loss",
        "val/loss", "val/sft/ndcg@5",
    )

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") not in {"train_step", "epoch_end", "val_end", "test_end", "non_finite_loss_skipped"}:
            return
        parts = [f"{k}={payload[k]:.5g}" for k in keys if isinstance(payload.get(k), (int, float))]
        if payload.get("type") == "train_step":
            parts.insert(0, f"step={payload.get('global_step')}")
            for key in ("train/data_wait_sec", "train/step_wall_sec", "train/examples_per_sec",
                        "train/reward_prepare_sec"):
                if isinstance(payload.get(key), (int, float)):
                    parts.append(f"{key}={payload[key]:.5g}")
        for block_key in ("train", "val"):
            block = payload.get(block_key, {})
            if isinstance(block, Mapping):
                parts += [f"{k}={block[k]:.5g}" for k in keys if isinstance(block.get(k), (int, float))]
        epoch = payload.get("epoch")
        header = f"[{tag}] {payload.get('type')}" + (
            f" epoch={int(epoch) + 1}" if isinstance(epoch, int) else ""
        )
        print(header + (" | " + " | ".join(parts) if parts else ""), flush=True)

    return logger


def run_single(
    *, entry: Mapping[str, Any], config: Mapping[str, Any], config_path: Path,
    dm, subset_indices: list[int], sasrec, index, cache_dir: Path, run_dir: Path,
    csv_path: Path, args: argparse.Namespace, init_baseline_path: Path | None = None,
) -> dict[str, Any]:
    tag = entry_tag(entry)
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tiger_runner.seed_everything(int(args.seed))
    model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    control.load_tiger_checkpoint(model, abspath(TIGER_CHECKPOINT))
    model.to(device)

    mapping = tiger_runner.item_id_to_semantic_row(dm)
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
            mapping, sasrec
        ),
        scorer=sasrec,
    )

    gen_topk = (5,)
    sft_cfg = SFTConfig(
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        compute_item_metrics=False,
        evaluate_generation=True,
        generation_topk=gen_topk,
        generation_num_beams=5,
        generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
    )
    base_gflow_cfg = GFlowConfig(
        objective=str(entry["objective"]),
        sft_weight=1.0,
        gflow_weight=1.0,  # placeholder; calibrated below
        reward_cache_path=str(cache_dir),
        reward_cache_power=float(entry["power"]),
        tb_loss_weighting="uniform",
        candidate_sampler="sasrec",
        num_negative_samples=int(entry["num_negatives"]),
        easy_negative_steps=int(args.curriculum_steps),
        candidate_sampling_temperature=float(args.sampler_temperature),
    )

    run_batch, run_accum = batch_settings_for_k(int(entry["num_negatives"]), args)
    if run_batch != int(args.batch_size):
        print(
            f"[{tag}] candidate-memory scaling: batch {args.batch_size} -> {run_batch}, "
            f"grad_accum {args.grad_accum} -> {run_accum} "
            f"(batch x (K+1) = {run_batch * (int(entry['num_negatives']) + 1)} "
            f"<= {args.max_candidate_batch})",
            flush=True,
        )

    if args.gflow_weight_mode == "fixed_grad_ratio":
        calibration = calibrate_gflow_weight(
            model=model, base_gflow_cfg=base_gflow_cfg, sampler=sampler,
            sft_cfg=sft_cfg, dm=dm, subset_indices=subset_indices,
            device=device, args=args, tag=tag,
            target_ratio=float(entry["rho"]), batch_size=run_batch,
        )
        lambda_gflow = float(calibration["fixed_gflow_weight"])
    else:
        calibration = {"mode": "constant", "fixed_gflow_weight": float(args.gflow_constant_weight)}
        lambda_gflow = float(args.gflow_constant_weight)
    write_json(run_dir / "calibration.json", calibration)

    method = GFlowMethod(
        replace(base_gflow_cfg, gflow_weight=lambda_gflow),
        candidate_sampler=sampler,
        sft_method=AliasedSFTMethod(sft_cfg),
    )

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
            run_validation_at_start=bool(args.validate_at_start),
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        logger=make_status_logger(tag),
    )

    resolved = {
        "dataset": DATASET,
        "combo": combo_key(entry["combo"]),
        "reward_power": float(entry["power"]),
        "objective": str(entry["objective"]),
        "num_negatives": int(entry["num_negatives"]),
        "seed": int(args.seed),
        "gflow": method.cfg.to_dict(),
        "gflow_weight_mode": str(args.gflow_weight_mode),
        "rho_target": float(entry["rho"]),
        "batching": {"batch_size": run_batch, "gradient_accumulation": run_accum,
                     "effective_batch": run_batch * run_accum},
        "calibration": {k: v for k, v in calibration.items() if k != "per_batch"},
        "sampler": {
            "type": "sasrec_curriculum", "mode": "curriculum",
            "num_negatives": int(entry["num_negatives"]),
            "include_positive": True, "exclude_history": True,
            "curriculum_steps": int(args.curriculum_steps),
            "temperature": float(args.sampler_temperature),
        },
        "reward": {
            "form": "(" + " * ".join(entry["combo"]) + f") ** {entry['power']}",
            "power_applied": "reward_cache_power (cache stores the unpowered product)",
            "components": config["method"]["reward"]["components"],
        },
        "data": {
            "positive_criterion": POSITIVE_CRITERION,
            "datamodule_train_last_n": 128,
            "finetune_per_user_last_n": int(args.train_last_n),
            "num_datamodule_train_examples": len(dm.train_dataset),
            "num_finetune_examples": len(subset_indices),
        },
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "sft_checkpoint": str(abspath(TIGER_CHECKPOINT)),
        "cache_dir": str(cache_dir),
        "item_vocab_size": int(dm.item_vocab_size),
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] fine-tuning: lambda={lambda_gflow:.6g}, epochs<={max_epochs}", flush=True)
    fit = trainer.fit(
        indexed_train_loader(
            dm, subset_indices, shuffle=not args.smoke,
            batch_size=int(run_batch), seed=int(args.seed)
        ),
        dm.val_dataloader(),
    )
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    checkpoint_for_test = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_test:
        trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)

    # best_epoch -1: no epoch beat the init checkpoint, so best.pt is a copy of
    # the init weights and their beam-20 full-split metrics are the init
    # baseline; reuse it instead of re-measuring the same weights.
    splits = None
    final_source = "evaluated"
    if fit.state.best_epoch == -1 and init_baseline_path is not None and not args.smoke:
        try:
            baseline = load_json(init_baseline_path)
        except Exception:  # noqa: BLE001 - missing/unreadable: evaluate
            baseline = {}
        recorded = str(baseline.get("checkpoint", ""))
        if recorded.endswith(TIGER_CHECKPOINT) and all(k in baseline for k in ("validation", "test")):
            splits = {"validation": dict(baseline["validation"]), "test": dict(baseline["test"])}
            final_source = "init_baseline"
            print(f"[{tag}] no epoch beat the init checkpoint; final metrics = init baseline "
                  f"({init_baseline_path})", flush=True)
    if splits is None:
        splits = {
            "validation": control.evaluate_split(
                model, dm.val_dataloader(), index,
                ks=tuple(args.test_ks), num_beams=int(args.eval_num_beams),
                max_batches=max_eval_batches),
            "test": control.evaluate_split(
                model, dm.test_dataloader(), index,
                ks=tuple(args.test_ks), num_beams=int(args.eval_num_beams),
                max_batches=max_eval_batches),
        }
    write_json(run_dir / "final_metrics.json", {**splits, "source": final_source})

    if args.promote_root and fit.best_checkpoint_path:
        promote_dir = (
            abspath(args.promote_root) / DATASET /
            f"gflow_{combo_key(entry['combo'])}_p{power_tag(entry['power'])}_"
            f"{entry['objective']}_k{entry['num_negatives']}"
        )
        promote_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit.best_checkpoint_path, promote_dir / "best.pt")
        sidecar = Path(str(fit.best_checkpoint_path) + ".json")
        if sidecar.exists():
            shutil.copy2(sidecar, promote_dir / "best.pt.json")
        write_json(promote_dir / "final_metrics.json", splits)

    result = {
        "dataset": DATASET,
        "combo": combo_key(entry["combo"]),
        "power": float(entry["power"]),
        "objective": str(entry["objective"]),
        "num_negatives": int(entry["num_negatives"]),
        "seed": int(args.seed),
        "status": "completed",
        "lambda_gflow": lambda_gflow,
        "target_grad_ratio": calibration.get("target_grad_ratio"),
        "raw_grad_ratio_median": calibration.get("raw_grad_ratio_median"),
        "grad_cosine_median": calibration.get("grad_cosine_median"),
        "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1),
        "final_metrics_source": final_source,
        "elapsed_sec": time.time() - started,
        "run_dir": str(run_dir),
        "splits": splits,
    }
    write_json(run_dir / "result.json", result)

    identity = {
        "dataset": DATASET, "combo": result["combo"], "power": result["power"],
        "objective": result["objective"], "num_negatives": result["num_negatives"],
        "seed": result["seed"], "variant": "gflow",
    }
    extra = {
        "status": "completed", "lambda_gflow": lambda_gflow,
        "target_grad_ratio": result["target_grad_ratio"],
        "raw_grad_ratio_median": result["raw_grad_ratio_median"],
        "grad_cosine_median": result["grad_cosine_median"],
        "best_epoch": result["best_epoch"], "best_val_ndcg5": result["best_val_ndcg5"],
        "epochs_completed": result["epochs_completed"], "run_dir": str(run_dir),
    }
    upsert_metrics_csv(csv_path, metric_rows(identity, splits, extra))

    del model, method, trainer, sampler
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    grid = build_grid(args)

    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d}  {entry_tag(entry)}")
        print(f"total: {len(grid)} runs")
        return

    if args.index is not None:
        text = str(args.index)
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
            return
        end = min(end, len(grid) - 1)
        grid = grid[start:end + 1]
        print(f"--index {text}: running {len(grid)} grid entr{'y' if len(grid) == 1 else 'ies'} "
              f"[{start}..{end}]", flush=True)

    tiger_runner.seed_everything(int(args.seed))
    device = torch.device(args.device)
    output_root = abspath(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = abspath(args.artifacts_dir) / "gflow_finetune_yambda_50m_metrics_summary.csv"

    # Group the selected runs by combo so the datamodule / teacher / cache
    # are built once per combo.
    by_combo: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for entry in grid:
        by_combo.setdefault(entry["combo"], []).append(entry)

    all_results: list[dict[str, Any]] = []
    for combo, entries in by_combo.items():
        ckey = combo_key(combo)
        pending = []
        for entry in entries:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            done = run_dir / "result.json"
            if done.exists() and not args.force:
                try:
                    if load_json(done).get("status") == "completed":
                        print(f"[{entry_tag(entry)}] already completed — skipping (--force to redo)",
                              flush=True)
                        continue
                except Exception:  # noqa: BLE001 - unreadable marker: re-run
                    pass
            pending.append(entry)
        if not pending:
            continue

        print(f"\n=== combo: {ckey} ({len(pending)} runs) ===", flush=True)
        config = build_run_config(combo, args)
        config_path = output_root / ckey / "data_reward_config.json"
        write_json(config_path, config)

        data_cfg = tiger_runner.build_data_config(dict(config), config_path=config_path)
        dm = tiger_runner.build_data_module(data_cfg)
        dm.setup()
        store = dm.interaction_store
        subset_indices = per_user_tail_indices(dm.train_dataset, int(args.train_last_n))
        print(
            f"[{ckey}] catalog={dm.item_vocab_size} items, train={len(dm.train_dataset)} "
            f"examples (fine-tune subset: {len(subset_indices)}), "
            f"aux store={len(store) if store is not None else 0} events",
            flush=True,
        )

        # Refuse to fine-tune a checkpoint that does not match this datamodule.
        probe = tiger_runner.build_model(dict(config), dm, config_path=config_path)
        control.load_tiger_checkpoint(probe, abspath(TIGER_CHECKPOINT))
        probe.to(device)
        if not args.skip_sanity_check:
            nll = control.sanity_check_checkpoint(probe, dm, device, float(args.sanity_nll_threshold))
            print(f"[{ckey}] init sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

        sasrec = tiger_runner.build_sasrec_teacher_model(
            dm, dict(config["method"]["sasrec"]), config_path=config_path
        )
        sasrec.to(device).eval()
        for param in sasrec.parameters():
            param.requires_grad_(False)
        if not args.skip_sanity_check:
            teacher_ndcg = sanity_check_teacher(sasrec, dm, device)
            print(f"[{ckey}] SASRec teacher sanity OK: val NDCG@10 {teacher_ndcg:.4f}", flush=True)

        cache_dir = abspath(args.cache_root) / ckey
        manifest = ensure_reward_cache(combo, config_path, cache_dir, len(subset_indices), args)
        verify_cache_alignment(dm, subset_indices, cache_dir)
        print(
            f"[{ckey}] reward cache ready: {manifest['num_examples']} examples, "
            f"power sums {manifest.get('stored_power_sums')}, "
            f"components={[c['name'] for c in manifest['components']]}",
            flush=True,
        )
        if args.cache_only:
            print(f"[{ckey}] --cache-only: skipping the runs", flush=True)
            continue

        index = control.CatalogIndex(
            tiger_runner.build_model(dict(config), dm, config_path=config_path), dm
        )
        ensure_init_baseline(
            config=config, config_path=config_path, dm=dm, index=index, device=device,
            args=args, output_root=output_root, csv_path=csv_path,
        )

        for entry in pending:
            run_dir = run_dir_for(output_root, entry, int(args.seed))
            try:
                result = run_single(
                    entry=entry, config=config, config_path=config_path, dm=dm,
                    subset_indices=subset_indices, sasrec=sasrec, index=index,
                    cache_dir=cache_dir, run_dir=run_dir, csv_path=csv_path, args=args,
                    init_baseline_path=output_root / "baseline" / "init_metrics.json",
                )
            except Exception as exc:  # noqa: BLE001 - keep the grid going
                import traceback

                print(f"[{entry_tag(entry)}] FAILED: {exc}", flush=True)
                traceback.print_exc()
                result = {
                    "dataset": DATASET, "combo": combo_key(entry["combo"]),
                    "power": float(entry["power"]), "objective": entry["objective"],
                    "num_negatives": int(entry["num_negatives"]), "seed": int(args.seed),
                    "target_grad_ratio": float(entry["rho"]),
                    "status": "failed", "error": str(exc), "run_dir": str(run_dir),
                }
                identity = {k: result[k] for k in
                            ("dataset", "combo", "power", "objective", "num_negatives",
                             "seed", "target_grad_ratio")}
                upsert_metrics_csv(csv_path, [
                    {**identity, "variant": "gflow", "split": split,
                     "status": "failed", "error": str(exc)[:300]}
                    for split in ("validation", "test")
                ])
            all_results.append(result)
            write_json(output_root / "summary.json", {"runs": all_results})

    print("\n=== summary ===", flush=True)
    for row in all_results:
        line = (f"{row['combo']:<36} p={row['power']:<6g} {row['objective']:<6} "
                f"K={row['num_negatives']:<5} {row['status']:<10}")
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


if __name__ == "__main__":
    main()
