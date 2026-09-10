#!/usr/bin/env python
"""GFlow fine-tuning of the exp1 GPT-2 SFT checkpoint on Amazon Beauty 2014.

Single-run counterpart of the exp4 TIGER/yambda ablation
(scripts/exp4-reward-ablations/gflow_finetune_yambda50m.py) for the GPT-2
model: loss = L_SFT + lambda * L_GFlow with lambda from one-time
gradient-norm calibration. The fine-tuning protocol (lr, effective batch,
epochs, patience, monitor, curriculum length, calibration settings) is loaded
from configs/exp2-tuned-params/finetune_gflow.json exactly like the exp4
runner, so the two stay comparable.

This run's fixed choices:
  - Init: the exp1 GPT-2 beauty checkpoint
    (checkpoints/exp1-standard_params/gpt2/beauty/seed_<seed>/best.pt),
    rebuilt with run_gpt2_sft_all.build_model from the exp1 gpt2_model.yaml
    profile and strict-loaded (the GPT-2 load path re-validates the semantic
    catalog).
  - Data: the exp1 beauty preprocessing (dataset_amazon.yaml values; identical
    in exp2) with the exp1 semantic IDs, but materialized on the
    prefix_target layout: GFlow needs one (history, target) example per
    reward-cache row, and GPT-2 supports both layouts. The catalog is
    layout-independent for beauty (no cold-item filtering), which the strict
    checkpoint load, an item_semantic_ids row-table check, and a
    teacher-forced NLL sanity check all verify at startup. Fine-tuning runs
    on the per-user last --train-last-n examples (protocol default 8; beauty
    has ~6 train examples per user, so this keeps nearly the full set).
  - Reward: MIXED (additive) over the three standard components -
    sasrec (sigmoid-transformed teacher scores) + interaction (evidence-scored
    positives; beauty's auxiliary store is empty because every 2014 review
    matches the verified-purchase criterion, so negatives sit at the neutral
    unobserved reward) + similarity (mode='hierarchical' by default:
    prefix-gated codebook-geometry similarity over the exp1 RQ-VAE levels,
    token_offset=1 / level_stride=256 for the exp1 level-offset tables; the
    collision-suffix level uses exact match). Components are per-example
    max-normalized and summed by the cache builder (reward_model='mixed'),
    one dense full-catalog cache per similarity mode, reused across reruns.
  - Sampler: CURRICULUM over the exp2 Optuna-tuned beauty SASRec teacher
    (checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_model.pt),
    exclude_history=True, easy -> hard over --curriculum-steps batch-steps.
  - Negatives: --num-negatives 3 (plus the positive: 4 scored candidates per
    example, far under --max-candidate-batch, so batch/accum stay at the
    protocol 256 x 8).
  - GPT-2 vs TIGER: GFlowMethod's candidate scoring threads TIGER encoder
    state through the batch; decoder-only GPT-2 rebuilds the user+history
    prompt from the batch itself, so GPT2GFlowMethod overrides that hook and
    candidates are scored through the generic FlowRecModel.score_items ->
    token_logprobs path. The default objective is 'tb' (one scoring pass);
    'db'/'subtb' work through the per-level fallback but cost sid_len passes.
  - Validation during fine-tuning is deliberately light: constrained beam-5
    generation scoring only NDCG@5 (the early-stopping monitor
    val/sft/ndcg@5) - no token metrics, no beam-20 pass. The full metric set
    - beam-20 HR/NDCG/Coverage/ARP at {5, 10, 20} via
    run_gpt2_sft_all.evaluate_generation_ranking_metrics, matching
    artifacts/exp1-standard_params/gpt2_sft_beauty_metrics_summary.csv - is
    computed for best.pt on the TEST split only (--eval-validation adds the
    validation split back).

Outputs:
  experiments/gflow-finetune-gpt2/<stamp>_gflow_gpt2_beauty_.../
      resolved_config.json, calibration.json, training_history.json,
      final_summary.json, checkpoints/best.pt
  artifacts/gflow_finetune/gpt2_beauty/reward_cache/mixed_.../  (reused)
  artifacts/gflow_finetune/gflow_gpt2_beauty_metrics_summary_seed-<seed>.csv

Usage:
  python scripts/gflow_finetune_gpt2_beauty.py --device cuda
  python scripts/gflow_finetune_gpt2_beauty.py --smoke --num-workers 0  # wiring check
  python scripts/gflow_finetune_gpt2_beauty.py --cache-only             # build the cache, exit
  sbatch scripts/slurm/gflow_finetune_gpt2_beauty.sbatch                # Slurm launch
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

# pyarrow.dataset's native module must load before torch on Windows;
# pandas.read_parquet imports it lazily and crashes otherwise.
import pyarrow.dataset  # noqa: F401
import torch

faulthandler.enable()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (
    str(PROJECT_ROOT),
    str(SCRIPT_DIR),
    str(SCRIPT_DIR / "exp1-standard-params"),
    str(SCRIPT_DIR / "exp4-reward-ablations"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
import run_gpt2_sft_all as gpt2_runner  # noqa: E402  (scripts/exp1-standard-params)
import gflow_finetune_yambda50m as gflow_ablation  # noqa: E402  (scripts/exp4-reward-ablations)
from precompute_reward_cache import per_user_tail_indices  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.gflow.config import GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.method import GFlowMethod  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.models.base import Batch  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402

runner = gpt2_runner.runner  # scripts/sasrec_config_runner (exp1 shared helpers)


# --------------------------------------------------------------------------- #
# Fixed assets
# --------------------------------------------------------------------------- #
DATASET = "beauty"
EXP1_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
EXP1_DATASET_YAML = EXP1_CONFIG_DIR / "dataset_amazon.yaml"
EXP1_GPT2_MODEL_YAML = EXP1_CONFIG_DIR / "gpt2_model.yaml"
SEMANTIC_IDS = PROJECT_ROOT / "datasets" / "semantic_ids" / "exp1-standard_params" / "beauty" / "semantic_ids.json"
RQVAE_MODEL = SEMANTIC_IDS.parent / "model.pt"
GPT2_CHECKPOINT_TEMPLATE = "checkpoints/exp1-standard_params/gpt2/beauty/seed_{seed}/best.pt"
SASREC_CHECKPOINT = "checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_model.pt"
SASREC_TRIAL = "checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_trial.json"
REWARD_COMPONENTS = ("sasrec", "interaction", "similarity")
# exp1 semantic IDs are level-offset (token = code + 1 + level * 256, suffix
# levels beyond the RQ codebooks) - the hierarchical similarity decode.
LEVEL_STRIDE = 256


def parse_args() -> argparse.Namespace:
    D = gflow_ablation.protocol_defaults()  # configs/exp2-tuned-params/finetune_gflow.json
    p = argparse.ArgumentParser(
        "GFlow fine-tuning of the exp1 GPT-2 beauty checkpoint: mixed "
        "(sasrec + interaction + similarity) reward, curriculum negative "
        "sampling, gradient-norm-calibrated gflow weight."
    )
    p.add_argument("--seed", type=int, default=int(D.get("seed", 42)),
                   help="Training seed; also selects the exp1 init checkpoint seed dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help=f"GPT-2 init weights (default: {GPT2_CHECKPOINT_TEMPLATE} for --seed)")
    p.add_argument("--sasrec-checkpoint", type=Path, default=Path(SASREC_CHECKPOINT))
    p.add_argument("--sasrec-trial", type=Path, default=Path(SASREC_TRIAL))

    # GFlow objective and candidates.
    p.add_argument("--objective", choices=["tb", "db", "subtb"], default="tb")
    p.add_argument("--num-negatives", type=int, default=3,
                   help="Curriculum-sampled negatives per example (this run's protocol: 3)")
    p.add_argument("--reward-power", type=float, default=1.0,
                   help="Exponent on the cached mixed reward (GFlowConfig.reward_cache_power)")
    p.add_argument("--curriculum-steps", type=int, default=int(D.get("curriculum_steps", 3000)),
                   help="Batch-steps to ramp curriculum negatives easy -> hard")
    p.add_argument("--sampler-temperature", type=float, default=float(D.get("sampler_temperature", 1.0)))
    p.add_argument("--sasrec-transform", default=str(D.get("sasrec_transform", "sigmoid")))
    p.add_argument("--sasrec-temperature", type=float, default=float(D.get("sasrec_temperature", 1.0)))
    p.add_argument("--similarity-mode", choices=["hierarchical", "exact"], default="hierarchical",
                   help="Similarity reward component: prefix-gated RQ-VAE codebook "
                        "geometry (hierarchical) or positionwise token agreement (exact)")

    # Data scale.
    p.add_argument("--train-last-n", type=int, default=int(D.get("train_last_n", 8)),
                   help="Fine-tune on the last N train examples per user (subset only; "
                        "the datamodule layout and catalog are untouched)")
    p.add_argument("--max-candidate-batch", type=int, default=int(D.get("max_candidate_batch", 4096)),
                   help="Upper bound on batch * (num_negatives + 1) during training")

    # Gradient-norm calibration of the gflow weight.
    p.add_argument("--gflow-weight-mode", choices=["fixed_grad_ratio", "constant"],
                   default=str(D.get("gflow_weight_mode", "fixed_grad_ratio")))
    p.add_argument("--gflow-target-grad-ratio", type=float,
                   default=float(D.get("gflow_target_grad_ratio", 1.0)))
    p.add_argument("--gflow-constant-weight", type=float, default=0.1,
                   help="Used only with --gflow-weight-mode constant")
    p.add_argument("--gflow-calibration-batches", type=int,
                   default=int(D.get("gflow_calibration_batches", 5)))
    p.add_argument("--gflow-calibration-eps", type=float,
                   default=float(D.get("gflow_calibration_eps", 1e-12)))
    p.add_argument("--gflow-calibration-seed", type=int,
                   default=int(D.get("gflow_calibration_seed", 1234)))

    # Reward cache.
    p.add_argument("--cache-batch-size", type=int, default=32)
    p.add_argument("--cache-dtype", choices=["float16", "float32"],
                   default=str(D.get("cache_dtype", "float16")))
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true",
                   help="Build/verify the reward cache, then exit")

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

    # Final evaluation (exp1 GPT-2 protocol).
    p.add_argument("--test-ks", type=int, nargs="+", default=list(D.get("test_ks", [5, 10, 20])))
    p.add_argument("--eval-num-beams", type=int, default=int(D.get("eval_num_beams", 20)))
    p.add_argument("--test-batch-size", type=int, default=128,
                   help="Beam-20 final evaluation batch (exp1 gpt2 test_eval_batch_size)")
    p.add_argument("--eval-validation", action="store_true",
                   help="Also run the full beam-20 metric set on the validation split "
                        "at the end (default: test only)")
    p.add_argument("--eval-init", action="store_true",
                   help="Also evaluate the untouched init checkpoint (the exp1 summary "
                        "CSV already contains these numbers, so this is off by default)")

    # Output.
    p.add_argument("--output-root", type=Path, default=Path("experiments/gflow-finetune-gpt2"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("artifacts/gflow_finetune/gpt2_beauty/reward_cache"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/gflow_finetune"))

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0,
                   help="Max teacher-forced val NLL/token for the init checkpoint "
                        "(random init would be ~log(776) ~ 6.65)")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: ~8-batch cache, 1 epoch, unshuffled")
    args = p.parse_args()

    if args.init_checkpoint is None:
        args.init_checkpoint = Path(GPT2_CHECKPOINT_TEMPLATE.format(seed=int(args.seed)))
    # Fields the reused exp4 helpers (ensure_reward_cache, batch_settings_for_k)
    # read from the namespace.
    args.powers = [float(args.reward_power)]
    return args


def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------- #
# GFlow method for a decoder-only model
# --------------------------------------------------------------------------- #
class GPT2GFlowMethod(GFlowMethod):
    """GFlowMethod whose candidate scoring works without TIGER encoder state.

    GPT-2 rebuilds the user+history prompt from the batch on every scoring
    call (FlowRecModel.score_items -> GPT2RecModel.token_logprobs), so there
    is no encoder activation to thread through the scoring batch.
    """

    def _candidate_scoring_batch(self, batch: Batch, model_output: Any) -> Batch:
        del model_output
        return dict(batch)


# --------------------------------------------------------------------------- #
# Config assembly (single source of truth for datamodule + reward cache)
# --------------------------------------------------------------------------- #
def similarity_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.similarity_mode == "hierarchical":
        # token_offset is set by build_reward_model from the datamodule's
        # offset_semantic_ids_by_one (1 for beauty); the collision-suffix
        # level beyond the RQ codebooks falls back to exact match.
        return {
            "mode": "hierarchical",
            "level_stride": LEVEL_STRIDE,
            "rqvae_model_path": str(RQVAE_MODEL),
        }
    return {"mode": "exact"}


def build_run_config(args: argparse.Namespace, user_ids_count: int) -> dict[str, Any]:
    """run_tiger-schema config driving BOTH the fine-tuning datamodule and the
    reward-cache precompute subprocess, mirroring the exp1 beauty
    preprocessing (dataset_amazon.yaml) with the exp1 semantic IDs. Only the
    train example layout differs from the exp1 GPT-2 run: prefix_target
    instead of dense_causal, because GFlow needs one target per example."""
    yaml_cfg = runner.load_yaml(EXP1_DATASET_YAML)
    entry = dict(yaml_cfg["datasets"][DATASET])

    # run_tiger.build_data_config cannot express filter_reviews_to_metadata;
    # it relies on the AmazonReviewDataConfig default (True). Refuse to run if
    # the exp1 YAML ever diverges, instead of silently building a different
    # catalog than the checkpoint's.
    if not bool(yaml_cfg.get("filter_reviews_to_metadata", True)):
        raise ValueError(
            "dataset_amazon.yaml sets filter_reviews_to_metadata=false, which "
            "this script cannot pass through run_tiger.build_data_config."
        )

    sasrec_trial = json.loads(abspath(args.sasrec_trial).read_text(encoding="utf-8"))
    sasrec_params = sasrec_trial.get("best_params", sasrec_trial)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")

    return {
        "experiment_name": "gflow_finetune_gpt2_beauty",
        "seed": int(args.seed),
        "data": {
            "reviews_path": str(abspath(entry["reviews_path"])),
            "metadata_path": str(abspath(entry["metadata_path"])),
            "semantic_id_path": str(SEMANTIC_IDS),
        },
        "dataset": {
            "item_key": entry.get("item_key", "asin"),
            "positive_criterion": yaml_cfg.get("positive_criterion"),
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
        # user_ids_count feeds user_hash_mod so hashed user tokens match the
        # exp1 GPT-2 run (user_ids_count=1: one shared user token).
        "model": {"use_user_token": True, "user_ids_count": int(user_ids_count)},
        "method": {
            "name": "gflow",
            "reward_model": "mixed",
            "reward_epsilon": 1e-6,
            "reward": {
                "components": [
                    {
                        "name": "sasrec",
                        "weight": 1.0,
                        "reward": {
                            "transform": str(args.sasrec_transform),
                            "temperature": float(args.sasrec_temperature),
                        },
                    },
                    # Engagement scoring + evidence-scored positives (module
                    # defaults; beauty's auxiliary store is empty, see header).
                    {"name": "interaction", "weight": 1.0, "reward": {}},
                    {"name": "similarity", "weight": 1.0, "reward": similarity_spec(args)},
                ]
            },
            "sasrec": {
                "checkpoint_path": str(abspath(args.sasrec_checkpoint)),
                "model": {k: sasrec_params[k] for k in model_keys if k in sasrec_params},
            },
        },
    }


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        abspath(args.init_checkpoint),
        abspath(args.sasrec_checkpoint),
        abspath(args.sasrec_trial),
        SEMANTIC_IDS,
        EXP1_DATASET_YAML,
        EXP1_GPT2_MODEL_YAML,
    ]
    if args.similarity_mode == "hierarchical":
        required.append(RQVAE_MODEL)
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing inputs:\n{formatted}")


# --------------------------------------------------------------------------- #
# Model construction + sanity checks
# --------------------------------------------------------------------------- #
def build_gpt2_from_exp1_profile(dm) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    model_config = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    profile = gpt2_runner.model_profile(model_config, DATASET)
    model = gpt2_runner.build_model(model_config, profile, dm)
    return model, model_config, profile


def load_gpt2_checkpoint(model, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if state_dict is None:
        raise KeyError(f"{path} does not contain model_state_dict (not a Trainer checkpoint?)")
    model.load_state_dict(state_dict, strict=True)


def check_semantic_row_table(model, dm) -> None:
    """The reward cache, the sampler mapping, and the exact-flow index all use
    run_tiger's row convention: data.semantic_ids sorted by raw item id. The
    GPT-2 catalog is built from the id2item subset; if semantic_ids.json
    carried extra non-catalog items the two row spaces would silently
    diverge, so require exact equality."""
    expected = torch.as_tensor(
        [dm.semantic_ids[key] for key in sorted(dm.semantic_ids)], dtype=torch.long
    )
    actual = model.item_semantic_ids.detach().cpu().long()
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise RuntimeError(
            f"GPT-2 semantic catalog ({tuple(actual.shape)}) does not match the "
            f"datamodule's semantic-ID table ({tuple(expected.shape)}); the "
            "reward-cache rows would be misaligned. Check semantic_id_path."
        )


@torch.no_grad()
def sanity_check_init_nll(model, dm, device: torch.device, *, batches: int = 3,
                          threshold: float = 6.0) -> float:
    """Teacher-forced val NLL/token of the loaded init checkpoint. A wrong
    catalog/config load scores near the uniform ~log(776) ~ 6.65."""
    was_training = model.training
    model.eval()
    total_logprob, total_tokens = 0.0, 0
    for i, raw_batch in enumerate(dm.val_dataloader()):
        if i >= batches:
            break
        batch = gflow_ablation.move_batch(dict(raw_batch), device)
        logprobs = model.token_logprobs(batch)
        mask = batch.get("target_token_mask")
        if torch.is_tensor(mask):
            mask = mask.bool().to(logprobs.device)
        else:
            mask = torch.ones_like(logprobs, dtype=torch.bool)
        total_logprob += float(logprobs.masked_fill(~mask, 0.0).sum().item())
        total_tokens += int(mask.sum().item())
    model.train(was_training)
    nll = -total_logprob / max(total_tokens, 1)
    if nll > threshold:
        raise RuntimeError(
            f"GPT-2 init sanity check FAILED: teacher-forced val NLL {nll:.3f} > "
            f"{threshold}. The checkpoint does not match this datamodule."
        )
    return nll


# --------------------------------------------------------------------------- #
# Gradient-norm calibration (exp4 protocol, GPT-2 method variants)
# --------------------------------------------------------------------------- #
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
) -> dict[str, Any]:
    """One-time calibration at the init checkpoint, before any optimizer step.

    Mirrors gflow_finetune_yambda50m.calibrate_gflow_weight: two method
    variants isolate the unweighted losses through the exact training code
    path - (sft_weight, gflow_weight) = (1, 0) gives L_SFT and (0, 1) gives
    L_GFlow - sharing the sampler and the mmap'd reward cache, with the RNG
    re-seeded identically per pass so dropout masks and sampled candidates
    match across the pair. step=0 keeps the curriculum sampler in its initial
    (easy) phase. Full precision; weights, optimizer state, and the training
    loader order are untouched.
    """
    sft_variant = GPT2GFlowMethod(
        replace(base_gflow_cfg, sft_weight=1.0, gflow_weight=0.0),
        candidate_sampler=sampler,
        sft_method=gflow_ablation.AliasedSFTMethod(sft_cfg),
    )
    gflow_variant = GPT2GFlowMethod(
        replace(base_gflow_cfg, sft_weight=0.0, gflow_weight=1.0),
        candidate_sampler=sampler,
        sft_method=gflow_ablation.AliasedSFTMethod(sft_cfg),
    )
    gflow_variant._reward_cache = sft_variant._get_reward_cache()  # share the mmap

    params = [p for p in model.parameters() if p.requires_grad]
    loader = gflow_ablation.indexed_train_loader(
        dm, subset_indices, shuffle=not args.smoke, batch_size=int(args.batch_size),
        seed=int(args.gflow_calibration_seed), num_workers=0,
    )
    num_batches = 1 if args.smoke else int(args.gflow_calibration_batches)

    was_training = model.training
    model.train()

    loss_pairs = gflow_ablation.variant_loss_pairs(
        model, sft_variant, gflow_variant, loader,
        seed=int(args.gflow_calibration_seed), step=0,
        num_batches=num_batches, device=device,
    )
    result = gflow_ablation.calibrate_from_loss_pairs(
        loss_pairs, params,
        target_ratio=float(args.gflow_target_grad_ratio),
        eps=float(args.gflow_calibration_eps),
    )
    model.train(was_training)
    # Restore the run's RNG so calibration does not shift training.
    runner.seed_everything(int(args.seed))

    print(
        f"[calibration] ||g_SFT||~{result['sft_grad_norm_median']:.4g} "
        f"||g_GFlow||~{result['gflow_grad_norm_median']:.4g} "
        f"r0={result['raw_grad_ratio_median']:.4g} "
        f"cos={result['grad_cosine_median']:.3f} "
        f"rho_target={result['target_grad_ratio']} "
        f"-> lambda={result['fixed_gflow_weight']:.6g}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# Final evaluation (exp1 GPT-2 protocol)
# --------------------------------------------------------------------------- #
def evaluate_splits(model, dm, args: argparse.Namespace, device: torch.device,
                    max_eval_batches: int | None,
                    splits: tuple[str, ...]) -> dict[str, dict[str, float]]:
    ks = tuple(int(k) for k in args.test_ks)
    lookup = gpt2_runner.SemanticIdItemLookup(dm.semantic_ids, dm.item2id)
    evaluation_kwargs = {
        "device": device,
        "lookup": lookup,
        "item_popularity": runner.training_item_popularity(dm),
        "num_catalog_items": int(dm.num_items),
        "ks": ks,
        "num_beams": int(args.eval_num_beams),
        "constrain_valid_sequences": True,
        "max_batches": max_eval_batches,
    }
    loaders = {"validation": dm.val_dataloader, "test": dm.test_dataloader}
    previous_eval_batch = dm.cfg.eval_batch_size
    dm.cfg = replace(dm.cfg, eval_batch_size=int(args.test_batch_size))
    try:
        return {
            split: gpt2_runner.evaluate_generation_ranking_metrics(
                model, loaders[split](), **evaluation_kwargs
            )
            for split in splits
        }
    finally:
        dm.cfg = replace(dm.cfg, eval_batch_size=previous_eval_batch)


def write_split_metrics_csv(path: Path, summary: Mapping[str, Any],
                            splits: Mapping[str, Mapping[str, float]],
                            ks: tuple[int, ...]) -> None:
    """exp1-column CSV (sasrec_config_runner.write_metrics_summary layout),
    but only for the splits that were actually evaluated - the shared helper
    would emit an empty validation row for a test-only run."""
    import csv

    metric_keys = [
        metric
        for k in ks
        for metric in (f"HR@{k}", f"NDCG@{k}", f"Coverage@{k}", f"ARP@{k}")
    ]
    fieldnames = ["dataset", "seed", "split", "num_examples", *metric_keys,
                  "best_checkpoint", "experiment_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, metrics in splits.items():
            writer.writerow(runner.to_jsonable({
                "dataset": DATASET,
                "seed": summary.get("seed"),
                "split": split,
                "num_examples": metrics.get("num_examples"),
                **{key: metrics.get(key) for key in metric_keys},
                "best_checkpoint": summary.get("best_checkpoint"),
                "experiment_dir": summary.get("experiment_dir"),
            }))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    validate_inputs(args)
    started = time.time()
    device = torch.device(runner.resolve_device(args.device))
    runner.seed_everything(int(args.seed))

    model_yaml = runner.load_yaml(EXP1_GPT2_MODEL_YAML)
    profile = gpt2_runner.model_profile(model_yaml, DATASET)
    user_hash_mod = gpt2_runner.user_ids_count(model_yaml, profile)
    config = build_run_config(args, user_hash_mod)

    run_name = (
        f"gflow_gpt2_beauty_mixed_{args.objective}_k{int(args.num_negatives)}_seed-{int(args.seed)}"
    )
    run_dir = runner.create_run_dir(abspath(args.output_root), run_name, config)
    runner.setup_logging(run_dir)
    config_path = run_dir / "data_reward_config.json"
    gflow_ablation.write_json(config_path, config)
    print(f"Run directory: {run_dir}", flush=True)

    # Data (prefix_target layout; catalog identical to the exp1 checkpoint's).
    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    dm = tiger_runner.build_data_module(data_cfg)
    dm.setup()
    if dm.semantic_ids is None or dm.sid_len is None:
        raise ValueError("The beauty datamodule produced no semantic IDs.")
    subset_indices = per_user_tail_indices(dm.train_dataset, int(args.train_last_n))
    print(
        f"[data] catalog={int(dm.item_vocab_size)} ids, train={len(dm.train_dataset)} "
        f"prefix examples (fine-tune subset: {len(subset_indices)})",
        flush=True,
    )

    # GPT-2 init (exp1 profile + checkpoint), verified against this datamodule.
    model, _, _ = build_gpt2_from_exp1_profile(dm)
    load_gpt2_checkpoint(model, abspath(args.init_checkpoint))
    check_semantic_row_table(model, dm)
    model.to(device)
    if not args.skip_sanity_check:
        nll = sanity_check_init_nll(
            model, dm, device, threshold=float(args.sanity_nll_threshold)
        )
        print(f"[init] sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)

    # SASRec teacher (exp2 tuned beauty), frozen; drives the curriculum
    # sampler and the sasrec reward component.
    sasrec = tiger_runner.build_sasrec_teacher_model(
        dm, dict(config["method"]["sasrec"]), config_path=config_path
    )
    sasrec.to(device).eval()
    for param in sasrec.parameters():
        param.requires_grad_(False)
    if not args.skip_sanity_check:
        teacher_ndcg = gflow_ablation.sanity_check_teacher(sasrec, dm, device, threshold=0.02)
        print(f"[teacher] sanity OK: val NDCG@10 {teacher_ndcg:.4f} (tuning recorded ~0.053)",
              flush=True)

    # Mixed reward cache (one per component set and similarity mode; reused
    # across reruns - the mode is part of the directory name because the
    # manifest alone cannot tell an exact cache from a hierarchical one).
    cache_dir = abspath(args.cache_root) / (
        "mixed_" + "_".join(REWARD_COMPONENTS) + f"_sim-{args.similarity_mode}"
    )
    manifest = gflow_ablation.ensure_reward_cache(
        REWARD_COMPONENTS, config_path, cache_dir, len(subset_indices), args
    )
    gflow_ablation.verify_cache_alignment(dm, subset_indices, cache_dir)
    print(
        f"[cache] ready: {manifest['num_examples']} examples, "
        f"components={[c['name'] for c in manifest['components']]}",
        flush=True,
    )
    if args.cache_only:
        print("[cache] --cache-only: done", flush=True)
        return

    # Curriculum sampler over the SASRec teacher.
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="curriculum",
            num_negatives=int(args.num_negatives),
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
    # only the NDCG@5 monitor (val/sft/ndcg@5 via the alias) - no token
    # metrics, no beam-20 pass. The full metric set runs on test at the end.
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
    base_gflow_cfg = GFlowConfig(
        objective=str(args.objective),
        sft_weight=1.0,
        gflow_weight=1.0,  # placeholder; calibrated below
        reward_cache_path=str(cache_dir),
        reward_cache_power=float(args.reward_power),
        tb_loss_weighting="uniform",
        candidate_sampler="sasrec",
        num_negative_samples=int(args.num_negatives),
        easy_negative_steps=int(args.curriculum_steps),
        candidate_sampling_temperature=float(args.sampler_temperature),
    )

    run_batch, run_accum = gflow_ablation.batch_settings_for_k(int(args.num_negatives), args)
    if run_batch != int(args.batch_size):
        print(f"[batching] candidate-memory scaling: batch {args.batch_size} -> {run_batch}, "
              f"grad_accum {args.grad_accum} -> {run_accum}", flush=True)

    if args.gflow_weight_mode == "fixed_grad_ratio":
        calibration = calibrate_gflow_weight(
            model=model, base_gflow_cfg=base_gflow_cfg, sampler=sampler,
            sft_cfg=sft_cfg, dm=dm, subset_indices=subset_indices,
            device=device, args=args,
        )
        lambda_gflow = float(calibration["fixed_gflow_weight"])
    else:
        calibration = {"mode": "constant", "fixed_gflow_weight": float(args.gflow_constant_weight)}
        lambda_gflow = float(args.gflow_constant_weight)
    gflow_ablation.write_json(run_dir / "calibration.json", calibration)

    method = GPT2GFlowMethod(
        replace(base_gflow_cfg, gflow_weight=lambda_gflow),
        candidate_sampler=sampler,
        sft_method=gflow_ablation.AliasedSFTMethod(sft_cfg),
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
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        logger=gflow_ablation.make_status_logger(run_name),
    )

    resolved = {
        "dataset": DATASET,
        "model": model.config_dict(),
        "variant": "gflow_mixed_curriculum",
        "objective": str(args.objective),
        "num_negatives": int(args.num_negatives),
        "reward_power": float(args.reward_power),
        "similarity_mode": str(args.similarity_mode),
        "seed": int(args.seed),
        "init_checkpoint": str(abspath(args.init_checkpoint)),
        "sasrec_checkpoint": str(abspath(args.sasrec_checkpoint)),
        "gflow": method.cfg.to_dict(),
        "gflow_weight_mode": str(args.gflow_weight_mode),
        "calibration": {k: v for k, v in calibration.items() if k != "per_batch"},
        "sampler": {
            "type": "sasrec_curriculum",
            "mode": "curriculum",
            "num_negatives": int(args.num_negatives),
            "include_positive": True,
            "exclude_history": True,
            "curriculum_steps": int(args.curriculum_steps),
            "temperature": float(args.sampler_temperature),
        },
        "reward": {
            "form": "(" + " + ".join(REWARD_COMPONENTS) + f") ** {args.reward_power}",
            "components": config["method"]["reward"]["components"],
        },
        "batching": {
            "batch_size": run_batch,
            "gradient_accumulation": run_accum,
            "effective_batch": run_batch * run_accum,
        },
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
        "cache_dir": str(cache_dir),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    gflow_ablation.write_json(run_dir / "resolved_config.json", resolved)

    final_splits = ("validation", "test") if args.eval_validation else ("test",)
    init_metrics = None
    if args.eval_init:
        print("[init] evaluating the untouched init checkpoint ...", flush=True)
        init_metrics = evaluate_splits(model, dm, args, device, max_eval_batches,
                                       splits=final_splits)
        gflow_ablation.write_json(run_dir / "init_metrics.json", init_metrics)

    print(f"[train] lambda={lambda_gflow:.6g}, epochs<={max_epochs}", flush=True)
    fit = trainer.fit(
        gflow_ablation.indexed_train_loader(
            dm, subset_indices, shuffle=not args.smoke,
            batch_size=int(run_batch), seed=int(args.seed),
        ),
        dm.val_dataloader(),
    )
    gflow_ablation.write_json(run_dir / "training_history.json", {"history": fit.state.history})

    checkpoint_for_eval = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_eval is not None:
        trainer.load_checkpoint(checkpoint_for_eval, load_optimizer=False, load_scheduler=False)

    splits = evaluate_splits(model, dm, args, device, max_eval_batches,
                             splits=final_splits)

    ks = tuple(int(k) for k in args.test_ks)
    summary = {
        "seed": int(args.seed),
        "dataset": {"name": DATASET, **runner.data_config_paths(dm.cfg), **dm.summary()},
        "model": model.config_dict(),
        "variant": "gflow_mixed_curriculum",
        "objective": str(args.objective),
        "num_negatives": int(args.num_negatives),
        "reward_power": float(args.reward_power),
        "similarity_mode": str(args.similarity_mode),
        "lambda_gflow": lambda_gflow,
        "raw_grad_ratio_median": calibration.get("raw_grad_ratio_median"),
        "grad_cosine_median": calibration.get("grad_cosine_median"),
        "epochs_completed": int(trainer.state.epoch + 1),
        "steps_completed": int(trainer.state.global_step),
        "best_metric": trainer.state.best_metric,
        "best_epoch": trainer.state.best_epoch,
        "init_checkpoint": str(abspath(args.init_checkpoint)),
        "init_metrics": init_metrics,
        "validation_metrics": splits.get("validation"),
        "test_metrics": splits["test"],
        "best_checkpoint": fit.best_checkpoint_path,
        "last_checkpoint": fit.last_checkpoint_path,
        "experiment_dir": str(run_dir),
        "elapsed_sec": time.time() - started,
    }
    gflow_ablation.write_json(run_dir / "final_summary.json", summary)
    runner.write_text_summary(run_dir / "final_summary.md", summary)

    # Seed-scoped so concurrent per-seed Slurm array tasks never overwrite
    # each other (each invocation runs exactly one seed).
    csv_path = abspath(args.artifacts_dir) / (
        f"gflow_gpt2_beauty_metrics_summary_seed-{int(args.seed)}.csv"
    )
    write_split_metrics_csv(csv_path, summary, splits, ks)

    print("\n=== summary ===", flush=True)
    print(
        f"lambda={lambda_gflow:.4g} best_val_ndcg@5={trainer.state.best_metric} "
        f"epochs={summary['epochs_completed']}",
        flush=True,
    )
    for split, metrics in splits.items():
        rendered = " ".join(f"{key}={metrics[key]:.5f}" for key in
                            (f"HR@{max(ks)}", f"NDCG@{max(ks)}", f"Coverage@{max(ks)}"))
        print(f"{split}: {rendered}", flush=True)
    print(f"metrics CSV: {csv_path}", flush=True)
    print(f"run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
