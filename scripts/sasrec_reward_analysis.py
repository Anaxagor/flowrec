#!/usr/bin/env python
"""Diagnose the SASRec-reward GFlow fine-tuning: negative-sampling diversity
and recommendation collapse.

Context: fine-tuning TIGER with the SASRec reward (see
scripts/rewards_ablation.py) degrades ranking and, most visibly, coverage.
Hypothesis: the curriculum negative sampler concentrates on a small set of
high-SASRec-score items, and the TB loss teaches the model to put its mass
there.

Two modes:

``--mode train`` (default) - recreate the rewards_ablation fine-tuning
    pipeline (SFT checkpoint init, SFT + TB loss with precomputed logZ, SASRec
    curriculum sampler, 3 negatives, standard-params configs) and rerun it
    with validation on ``val/sft/ndcg@K`` (default K=5, full val set) EVERY
    epoch, while instrumenting the run so that every negative actually drawn
    during training is recorded. The logZ table is recomputed each run to
    match the requested reward transform/temperature. Per epoch this yields:
      - item diversity of the real negatives (unique ratio, catalog coverage,
        entropy, Gini, top-1% concentration),
      - reward diversity (level, spread, within-context spread, SASRec-score
        percentile of the negatives),
      - per-example-user infographics built from that user's actual training
        draws,
      - collapse dynamics: the item distribution of the top-1 validation
        recommendations at every epoch.
    Afterwards the freshly trained best/last checkpoints are compared against
    the SFT baseline: full test-set top-K generation, concentration metrics,
    and a table of the items recommendations collapse into, cross-referenced
    with SASRec percentile, popularity, and how often each was drawn as a
    (hard) negative during this very run.

``--mode replay`` - no training: replay the sampler offline against an
    existing fine-tuned run (the sampler depends only on the frozen SASRec
    teacher and the global step, so per-epoch sampling can be simulated
    exactly), then run the same collapse analysis on that run's checkpoints.

Run (from repo root, torch env):
    python scripts/sasrec_reward_analysis.py --dataset beauty
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for entry in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rewards_ablation import (
    CHROME,
    TB_SAFE_SASREC_TRANSFORMS,
    AliasedSFTMethod,
    _style_axis,
    build_curriculum_sampler,
    build_datamodule,
    build_gflow_method,
    build_reward_model,
    build_tiger_model,
    find_sasrec_checkpoint,
    indexed_train_loader,
    item_id_to_semantic_row,
    load_exp_configs,
    load_or_compute_logz,
    load_sasrec_teacher,
    load_sft_checkpoint,
    make_status_logger,
    move_batch,
    path_from_repo,
    restrict_mapping_to_scorer_vocab,
    sasrec_alignment_check,
    sasrec_reward_key,
    select_example_indices,
    semantic_item_tokens,
    semantic_row_to_item_id,
    set_seed,
    write_json,
)
from flowrec.candidates.item_model import (
    ItemModelSemanticCandidateConfig,
    ItemModelSemanticIDCandidateSampler,
)
from flowrec.data.datamodule import AmazonReviewDataModule
from flowrec.ft_methods.base import MethodOutput
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.sasrec import SASRec
from flowrec.models.tiger import TigerModel
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
from flowrec.training import Trainer, TrainerConfig

# Fixed categorical colors (dataviz reference palette): color follows the model.
MODEL_COLORS = {
    "sft_baseline": "#2a78d6",    # blue
    "finetuned_best": "#4a3aa7",  # violet
    "finetuned_last": "#e34948",  # red
    "sasrec_teacher": "#eda100",  # yellow
}
PHASE_COLORS = {"easy": "#1baf7a", "hard": "#e34948"}
EASY_SHADE = "#1baf7a"


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #
def gini_coefficient(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    cum = np.cumsum(x) / total
    return float((n + 1 - 2 * cum.sum()) / n)


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def effective_items(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(np.exp(-(p * np.log(p)).sum()))


def top_share(counts: np.ndarray, top_n: int) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(counts)[::-1][:top_n].sum() / total)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sampling_stats(
    counts: np.ndarray,
    rewards: np.ndarray,
    ctx_std: np.ndarray,
    percentiles: np.ndarray,
    *,
    num_items: int,
) -> dict[str, float]:
    total = int(counts.sum())
    unique = int((counts > 0).sum())
    top1pct_n = max(1, int(round(0.01 * num_items)))
    return {
        "total_negatives": total,
        "unique_items": unique,
        "unique_ratio": unique / max(total, 1),
        "catalog_coverage": unique / max(num_items, 1),
        "norm_entropy": normalized_entropy(counts),
        "gini": gini_coefficient(counts),
        "top1pct_share": top_share(counts, top1pct_n),
        "reward_mean": float(rewards.mean()) if rewards.size else float("nan"),
        "reward_std": float(rewards.std()) if rewards.size else float("nan"),
        "reward_median": float(np.median(rewards)) if rewards.size else float("nan"),
        "ctx_reward_std_mean": float(ctx_std.mean()) if ctx_std.size else float("nan"),
        "pct_mean": float(percentiles.mean()) if percentiles.size else float("nan"),
        "pct_median": float(np.median(percentiles)) if percentiles.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Run discovery / metadata
# --------------------------------------------------------------------------- #
def discover_run_dir(dataset: str, explicit: str | None) -> Path | None:
    if explicit is not None:
        run_dir = path_from_repo(explicit)
        return run_dir if run_dir.exists() else None
    root = path_from_repo("experiments/rewards_ablation")
    candidates = [
        p for p in root.glob(f"*_rewards_ablation/{dataset}/sasrec") if (p / "checkpoints").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"run_dir": str(run_dir)}
    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        sampler_cfg = resolved.get("sampler", {})
        meta["easy_negative_steps"] = int(sampler_cfg.get("easy_negative_steps", 2000))
        meta["num_negative_samples"] = int(sampler_cfg.get("num_negative_samples", 3))
        meta["num_random_negatives"] = int(sampler_cfg.get("num_random_negatives", 0))
        meta["include_positive"] = bool(sampler_cfg.get("include_positive", True))
        meta["reward"] = resolved.get("reward")
        reward_options = resolved.get("reward_options", {})
        meta["reward_transform"] = str(reward_options.get("sasrec_transform", "softplus"))
        meta["reward_temperature"] = float(reward_options.get("sasrec_temperature", 1.0))
    history_path = run_dir / "training_history.json"
    if history_path.exists():
        meta["history"] = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        meta["epochs_completed"] = len(meta["history"])
    best_sidecar = run_dir / "checkpoints" / "best.pt.json"
    if best_sidecar.exists():
        state = json.loads(best_sidecar.read_text(encoding="utf-8")).get("trainer_state", {})
        meta["best_epoch"] = state.get("best_epoch")
    return meta


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
def build_probe_batches(dm: AmazonReviewDataModule, n_batches: int, seed: int) -> list[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed + 1234)
    loader = DataLoader(
        dm.train_dataset,
        batch_size=dm.cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=dm._make_collator(),
    )
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= n_batches:
            break
        batches.append(batch)
    return batches


def build_replay_sampler(
    sasrec: SASRec,
    item_tokens: torch.Tensor,
    item_to_row: torch.Tensor,
    *,
    num_negatives: int,
    easy_negative_steps: int,
    num_random_negatives: int = 0,
    include_positive: bool = True,
) -> ItemModelSemanticIDCandidateSampler:
    return ItemModelSemanticIDCandidateSampler(
        sasrec,
        item_semantic_ids=item_tokens.cpu(),
        item_id_to_semantic_row=restrict_mapping_to_scorer_vocab(item_to_row, sasrec),
        cfg=ItemModelSemanticCandidateConfig(
            num_negative_samples=num_negatives,
            include_positive=bool(include_positive),
            avoid_positive=True,
            easy_negative_steps=easy_negative_steps,
            num_random_negatives=int(num_random_negatives),
            exclude_history=True,
            keep_scorer_eval=True,
            sampling_temperature=1.0,
            candidate_pool_size=-1,
            score_chunk_size=50_000,
            pool_strategy="full_catalog",
            sampler_name="sasrec_semantic_id",
        ),
    )


@torch.no_grad()
def negative_percentiles(
    sasrec: SASRec, batch: Batch, neg_ids: torch.Tensor
) -> torch.Tensor:
    """SASRec score percentile of each negative within the user's full catalog scores."""
    full_scores = sasrec.score_items(batch).float()  # [B, V]
    neg_scores = full_scores.gather(1, neg_ids.clamp_min(0))
    return full_scores.unsqueeze(1).lt(neg_scores.unsqueeze(2)).float().mean(dim=2)


def split_negative_candidates(
    trajectories, rewards: torch.Tensor | None = None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract negative candidates via the is_positive mask.

    Works whether or not the positive is part of the candidate set
    (``--tb-exclude-positive``); assumes a uniform per-row candidate layout,
    which holds for the samplers used here.
    """
    candidate_ids = trajectories.extra.get("candidate_item_ids")
    if not torch.is_tensor(candidate_ids):
        return None, None
    neg_mask = ~trajectories.is_positive.to(candidate_ids.device).bool()
    n_neg = int(neg_mask[0].sum())
    if n_neg == 0:
        return None, None
    bsz = candidate_ids.size(0)
    neg_ids = candidate_ids[neg_mask].reshape(bsz, n_neg)
    neg_rewards = rewards[neg_mask].reshape(bsz, n_neg) if torch.is_tensor(rewards) else None
    return neg_ids, neg_rewards


# --------------------------------------------------------------------------- #
# Mode "train": instrumented re-run of the rewards_ablation fine-tuning
# --------------------------------------------------------------------------- #
class SamplingRecorder:
    """Accumulate the negatives actually drawn during fine-tuning, per epoch.

    Fed by RecordingRewardModel (train batches) and RecordingSFTMethod
    (validation top-1 recommendations); flushed by the trainer logger on every
    ``epoch_end`` event.
    """

    def __init__(
        self,
        *,
        item_vocab_size: int,
        num_items: int,
        example_infos: list[dict[str, Any]],
        sasrec: SASRec,
        catalog_lookup: Mapping[tuple[int, ...], int],
        row_to_item: torch.Tensor,
        pct_batches_per_epoch: int = 8,
    ) -> None:
        self.vocab = int(item_vocab_size)
        self.num_items = int(num_items)
        self.sasrec = sasrec
        self.catalog_lookup = catalog_lookup
        self.row_to_item = row_to_item.cpu().numpy()
        self.pct_batches_per_epoch = int(pct_batches_per_epoch)
        self.user_index = {int(info["user_id"]): idx for idx, info in enumerate(example_infos)}
        self._example_ids = np.array(sorted(self.user_index), dtype=np.int64)

        self.epoch_rows: list[dict[str, Any]] = []
        self.val_rows: list[dict[str, Any]] = []
        self.per_epoch_user_data: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        self.hard_neg_counts = np.zeros(self.vocab, dtype=np.int64)
        self._reset()

    def _reset(self) -> None:
        self._counts = torch.zeros(self.vocab, dtype=torch.long)
        self._rewards: list[np.ndarray] = []
        self._ctx_std: list[np.ndarray] = []
        self._pcts: list[np.ndarray] = []
        self._pct_batches = 0
        self._batches = 0
        self._easy_batches = 0
        self._user_items: dict[int, list[np.ndarray]] = defaultdict(list)
        self._user_rewards: dict[int, list[np.ndarray]] = defaultdict(list)
        self._val_counts = np.zeros(self.vocab, dtype=np.int64)
        self._val_users = 0
        self._val_invalid = 0

    @torch.no_grad()
    def record_train(self, batch: Batch, trajectories, rewards: torch.Tensor) -> None:
        neg_ids, neg_rewards = split_negative_candidates(trajectories, rewards)
        if neg_ids is None:
            return
        neg_ids = neg_ids.detach()
        neg_rewards = neg_rewards.detach().float()
        valid = neg_ids.ge(0)

        flat = neg_ids[valid].reshape(-1).cpu()
        self._counts.index_add_(0, flat, torch.ones_like(flat))
        self._rewards.append(neg_rewards[valid].cpu().numpy())
        self._ctx_std.append(neg_rewards.std(dim=1, unbiased=False).cpu().numpy())
        self._batches += 1
        if trajectories.extra.get("sampler_phase") == "easy":
            self._easy_batches += 1

        if self._pct_batches < self.pct_batches_per_epoch:
            pct = negative_percentiles(self.sasrec, batch, neg_ids)
            self._pcts.append(pct[valid].cpu().numpy())
            self._pct_batches += 1

        user_ids = batch.get("user_ids")
        if torch.is_tensor(user_ids) and self.user_index:
            uids = user_ids.detach().cpu().numpy()
            mask = np.isin(uids, self._example_ids)
            if mask.any():
                neg_cpu = neg_ids.cpu().numpy()
                rew_cpu = neg_rewards.cpu().numpy()
                for i in np.nonzero(mask)[0]:
                    uidx = self.user_index[int(uids[i])]
                    self._user_items[uidx].append(neg_cpu[i])
                    self._user_rewards[uidx].append(rew_cpu[i])

    def record_val_generated(self, generated: torch.Tensor | None) -> None:
        if not torch.is_tensor(generated):
            return
        gen = generated.detach().cpu().long()
        if gen.ndim == 2:
            gen = gen.unsqueeze(1)
        top1 = gen[:, 0, :]
        for row in top1.tolist():
            catalog_row = self.catalog_lookup.get(tuple(row), -1)
            item_id = int(self.row_to_item[catalog_row]) if catalog_row >= 0 else -1
            if item_id >= 0:
                self._val_counts[item_id] += 1
            else:
                self._val_invalid += 1
        self._val_users += int(top1.size(0))

    def end_epoch(self, epoch: int) -> None:
        counts = self._counts.numpy().copy()
        rewards = np.concatenate(self._rewards) if self._rewards else np.zeros(0)
        ctx_std = np.concatenate(self._ctx_std) if self._ctx_std else np.zeros(0)
        pcts = np.concatenate(self._pcts) if self._pcts else np.zeros(0)
        stats = sampling_stats(counts, rewards, ctx_std, pcts, num_items=self.num_items)
        frac_easy = self._easy_batches / max(self._batches, 1)
        phase = "easy" if frac_easy >= 0.5 else "hard"
        if phase == "hard":
            self.hard_neg_counts += counts
        self.epoch_rows.append({"epoch": epoch, "phase": phase, "frac_easy": frac_easy, **stats})

        self.per_epoch_user_data[epoch] = {
            uidx: {
                "item_ids": np.stack(self._user_items[uidx]) if self._user_items[uidx] else np.zeros((0, 1), dtype=np.int64),
                "rewards": np.stack(self._user_rewards[uidx]) if self._user_rewards[uidx] else np.zeros((0, 1)),
            }
            for uidx in self.user_index.values()
        }

        if self._val_users > 0:
            self.val_rows.append(
                {
                    "epoch": epoch,
                    "val_users": self._val_users,
                    "val_top1_coverage": float((self._val_counts > 0).sum() / max(self.num_items, 1)),
                    "val_top1_norm_entropy": normalized_entropy(self._val_counts),
                    "val_top1_effective_items": effective_items(self._val_counts),
                    "val_top1_top10_share": top_share(self._val_counts, 10),
                    "val_top1_invalid_rate": self._val_invalid / max(self._val_users, 1),
                }
            )
        print(
            f"[recorder] epoch {epoch + 1} ({phase}, frac_easy={frac_easy:.2f}): "
            f"unique_ratio={stats['unique_ratio']:.3f} coverage={stats['catalog_coverage']:.3f} "
            f"top1%={stats['top1pct_share']:.3f} reward_mean={stats['reward_mean']:.3f} "
            f"pct={stats['pct_mean']:.3f}"
            + (
                f" | val_top1_cov={self.val_rows[-1]['val_top1_coverage']:.4f} "
                f"val_top10_share={self.val_rows[-1]['val_top1_top10_share']:.3f}"
                if self._val_users > 0
                else ""
            ),
            flush=True,
        )
        self._reset()


class RecordingRewardModel(torch.nn.Module):
    """Pass-through SASRec reward that reports every training draw to the recorder."""

    def __init__(self, base: SASRecRewardModel, recorder: SamplingRecorder) -> None:
        super().__init__()
        self.base = base
        self._recorder = recorder

    def forward(self, batch: Batch, trajectories) -> torch.Tensor:
        rewards = self.base(batch, trajectories)
        if self.training:  # training_step only; validation stays unrecorded here
            self._recorder.record_train(batch, trajectories, rewards)
        return rewards


class RecordingSFTMethod(AliasedSFTMethod):
    """AliasedSFTMethod that reports validation top-1 generations to the recorder."""

    def __init__(self, cfg, recorder: SamplingRecorder) -> None:
        super().__init__(cfg)
        self._recorder = recorder

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        out = super().compute_loss(model, batch, stage=stage, step=step)
        if stage == "val":
            self._recorder.record_val_generated(out.extra.get("generated_token_ids"))
        return out


def run_instrumented_finetune(
    *,
    args: argparse.Namespace,
    cfgs: Mapping[str, Mapping[str, Any]],
    dm: AmazonReviewDataModule,
    dataset: str,
    sasrec: SASRec,
    base_reward: SASRecRewardModel,
    reward_temperature: float,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    example_infos: list[dict[str, Any]],
    catalog_lookup: Mapping[tuple[int, ...], int],
    device: torch.device,
    out_dir: Path,
    easy_steps: int,
) -> tuple[SamplingRecorder, Path, dict[str, Any]]:
    trainer_yaml = cfgs["trainer"]
    tiger_cfg = cfgs["tiger"]
    run_dir = out_dir / "finetune_sasrec"
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}/sasrec-rerun"

    # Precomputed logZ = log(sum of full-catalog SASRec rewards per context).
    # Recomputed on every run so it always matches the specified reward
    # parameters (--reward-transform / --reward-temperature); other rewards'
    # cache entries are preserved. --reuse-logz-cache opts back into caching.
    reward_key = sasrec_reward_key(args.reward_transform, reward_temperature)
    logz_payload = load_or_compute_logz(
        path_from_repo(args.logz_cache_dir) / f"{dataset}.pt",
        recompute=not args.reuse_logz_cache,
        dm=dm,
        reward_names=[reward_key],
        reward_models={reward_key: base_reward},
        item_tokens=item_tokens,
        row_to_item=row_to_item,
        item_to_row=item_to_row,
        device=device,
        batch_size=512,
        example_infos=[],
    )
    log_z = logz_payload["rewards"][reward_key]["log_z"]

    model = build_tiger_model(dm, tiger_cfg)
    load_sft_checkpoint(model, path_from_repo(args.sft_artifacts) / dataset / "checkpoints" / "best.pt")

    sampler = build_curriculum_sampler(
        sasrec=sasrec,
        model=model,
        item_to_row=item_to_row,
        easy_negative_steps=easy_steps,
        num_random_negatives=args.num_random_negatives,
        include_positive=not args.tb_exclude_positive,
    )
    recorder = SamplingRecorder(
        item_vocab_size=int(dm.item_vocab_size),
        num_items=int(dm.num_items),
        example_infos=example_infos,
        sasrec=sasrec,
        catalog_lookup=catalog_lookup,
        row_to_item=row_to_item,
        pct_batches_per_epoch=args.pct_batches_per_epoch,
    )
    method = build_gflow_method(
        reward_model=RecordingRewardModel(base_reward, recorder),
        sampler=sampler,
        log_z=log_z,
        tiger_cfg=tiger_cfg,
        gflow_weight=args.gflow_weight,
        include_positive=not args.tb_exclude_positive,
        val_topk=args.val_topk,
    )
    method.sft_method = RecordingSFTMethod(method.sft_method.cfg, recorder)

    max_epochs = int(args.max_epochs if args.max_epochs is not None else trainer_yaml.get("max_epochs", 500))
    patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else trainer_yaml.get("early_stopping_patience", 10)
    )
    monitor_k = int(args.val_topk) if args.val_topk else 1
    monitor = f"val/sft/ndcg@{monitor_k}" if args.val_topk else str(trainer_yaml.get("monitor", "val/sft/ndcg@1"))
    lr = float(args.lr if args.lr is not None else trainer_yaml.get("lr", 3e-4))
    base_logger = make_status_logger(tag)

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") == "epoch_end":
            recorder.end_epoch(int(payload.get("epoch", 0)))
        base_logger(payload)

    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=int(trainer_yaml.get("gradient_accumulation_steps", 4)),
            max_grad_norm=float(trainer_yaml.get("max_grad_norm", 1.0)),
            use_amp=bool(trainer_yaml.get("use_amp", True)) and device.type == "cuda",
            amp_dtype=str(trainer_yaml.get("amp_dtype", "float16")),
            validate_every_epochs=1,  # per-epoch validation, as requested
            log_every_steps=int(trainer_yaml.get("log_every_steps", 200)),
            monitor=monitor,
            monitor_mode="max",
            early_stopping_patience=patience,
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            val_sample_fraction=args.val_sample_fraction,
        ),
        lr=lr,
        weight_decay=float(trainer_yaml.get("weight_decay", 1e-2)),
        device=device,
        logger=logger,
    )

    resolved = {
        "dataset": dataset,
        "reward": "sasrec",
        "reward_key": reward_key,
        "mode": "instrumented_rerun",
        "gflow": method.cfg.to_dict(),
        "sampler": {
            "type": "sasrec_curriculum",
            "num_negative_samples": 3,
            "num_random_negatives": int(args.num_random_negatives),
            "include_positive": not args.tb_exclude_positive,
            "easy_negative_steps": int(easy_steps),
        },
        "reward_options": {
            "sasrec_transform": str(args.reward_transform),
            "sasrec_temperature": float(reward_temperature),
        },
        "logz": {
            "source": "precomputed_full_catalog_reward_sum",
            "mean": float(log_z.mean()),
            "std": float(log_z.std()),
        },
        "optimizer": {"lr": lr, "weight_decay": float(trainer_yaml.get("weight_decay", 1e-2))},
        "trainer": trainer.cfg.to_dict(),
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] starting instrumented fine-tuning: epochs<={max_epochs}, device={device}", flush=True)
    fit = trainer.fit(indexed_train_loader(dm, shuffle=True), dm.val_dataloader())
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    del model, method, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_meta = {
        "run_dir": str(run_dir),
        "reward": "sasrec",
        "reward_transform": str(args.reward_transform),
        "reward_temperature": float(reward_temperature),
        "tb_exclude_positive": bool(args.tb_exclude_positive),
        "num_random_negatives": int(args.num_random_negatives),
        "lr": lr,
        "monitor_k": monitor_k,
        "easy_negative_steps": int(easy_steps),
        "num_negative_samples": 3,
        "history": fit.state.history,
        "epochs_completed": int(fit.state.epoch + 1),
        "best_epoch": fit.state.best_epoch,
        "best_metric": fit.state.best_metric,
    }
    return recorder, run_dir, run_meta


# --------------------------------------------------------------------------- #
# Mode "replay": offline sampler simulation (no training)
# --------------------------------------------------------------------------- #
def pick_epochs(n_epochs: int, max_points: int, switch_epoch: float) -> list[int]:
    core = {0, 1, 2, 3, 4, max(n_epochs - 1, 0), int(switch_epoch), int(switch_epoch) + 1}
    spread = {int(round(x)) for x in np.linspace(0, max(n_epochs - 1, 0), num=max_points)}
    return sorted(e for e in core | spread if 0 <= e < max(n_epochs, 1))


@torch.no_grad()
def sample_negatives(
    sampler: ItemModelSemanticIDCandidateSampler,
    reward_model: SASRecRewardModel,
    sasrec: SASRec,
    batch: Batch,
    *,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectories = sampler.sample(batch, model=None, step=step)
    all_rewards = reward_model(batch, trajectories).float()
    neg_ids, rewards = split_negative_candidates(trajectories, all_rewards)
    valid = neg_ids.ge(0)
    percentiles = negative_percentiles(sasrec, batch, neg_ids)
    return neg_ids, rewards, percentiles, valid


@torch.no_grad()
def replay_epoch(
    sampler: ItemModelSemanticIDCandidateSampler,
    reward_model: SASRecRewardModel,
    sasrec: SASRec,
    probe_batches: list[dict[str, Any]],
    *,
    device: torch.device,
    step: int,
    item_vocab_size: int,
    num_items: int,
) -> tuple[dict[str, float], np.ndarray]:
    counts = torch.zeros(item_vocab_size, dtype=torch.long)
    rewards_parts: list[torch.Tensor] = []
    ctx_std_parts: list[torch.Tensor] = []
    pct_parts: list[torch.Tensor] = []
    for raw_batch in probe_batches:
        batch = move_batch(raw_batch, device)
        neg_ids, rewards, percentiles, valid = sample_negatives(
            sampler, reward_model, sasrec, batch, step=step
        )
        flat = neg_ids[valid].reshape(-1).cpu()
        counts.index_add_(0, flat, torch.ones_like(flat))
        rewards_parts.append(rewards[valid].cpu())
        ctx_std_parts.append(rewards.std(dim=1, unbiased=False).cpu())
        pct_parts.append(percentiles[valid].cpu())
    stats = sampling_stats(
        counts.numpy(),
        torch.cat(rewards_parts).numpy(),
        torch.cat(ctx_std_parts).numpy(),
        torch.cat(pct_parts).numpy(),
        num_items=num_items,
    )
    return stats, counts.numpy()


@torch.no_grad()
def replay_example_users(
    sampler: ItemModelSemanticIDCandidateSampler,
    reward_model: SASRecRewardModel,
    sasrec: SASRec,
    example_batch: Batch,
    *,
    step: int,
    draws: int,
    n_users: int,
) -> dict[int, dict[str, np.ndarray]]:
    ids_parts, reward_parts = [], []
    for _ in range(draws):
        neg_ids, rewards, _, valid = sample_negatives(sampler, reward_model, sasrec, example_batch, step=step)
        ids_parts.append(neg_ids.masked_fill(~valid, -1).cpu().numpy())
        reward_parts.append(rewards.cpu().numpy())
    ids = np.stack(ids_parts)  # [draws, U, neg]
    rewards = np.stack(reward_parts)
    return {
        u: {"item_ids": ids[:, u, :], "rewards": rewards[:, u, :]}
        for u in range(n_users)
    }


# --------------------------------------------------------------------------- #
# Collapse analysis (shared by both modes)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_recommendations(
    model: TigerModel,
    dm: AmazonReviewDataModule,
    sasrec: SASRec,
    *,
    device: torch.device,
    top_k: int,
    num_beams: int,
    constrain: bool,
    max_batches: int | None,
    catalog_lookup: Mapping[tuple[int, ...], int],
    row_to_item: torch.Tensor,
    item_vocab_size: int,
    tag: str,
) -> dict[str, Any]:
    model.eval()
    model.to(device)
    row_to_item_dev = row_to_item.to(device)
    counts = np.zeros(item_vocab_size, dtype=np.int64)
    n_users = 0
    invalid = 0
    hits = 0.0
    ndcg = 0.0
    pct_sum = 0.0
    pct_n = 0
    start = time.time()
    for batch_idx, raw_batch in enumerate(dm.test_dataloader(), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        batch = move_batch(raw_batch, device)
        generated = model.generate(
            batch,
            num_beams=num_beams,
            num_return_sequences=top_k,
            constrain_valid_sequences=constrain,
        )  # [B, K, L]
        bsz = int(generated.size(0))
        n_users += bsz

        target = batch["target_token_ids"].long()
        mask = batch.get("target_token_mask", torch.ones_like(target, dtype=torch.bool)).bool()
        matches = ((generated == target.unsqueeze(1)) | ~mask.unsqueeze(1)).all(dim=-1)
        hit = matches.any(dim=1)
        rank = matches.float().argmax(dim=1)
        discount = torch.zeros_like(hit, dtype=torch.float32)
        discount[hit] = 1.0 / torch.log2(rank[hit].float() + 2.0)
        hits += float(hit.float().sum())
        ndcg += float(discount.sum())

        rows = torch.full((bsz, top_k), -1, dtype=torch.long)
        gen_cpu = generated.detach().cpu().long()
        for b in range(bsz):
            for j in range(top_k):
                row = catalog_lookup.get(tuple(gen_cpu[b, j].tolist()), -1)
                rows[b, j] = row
                if row < 0:
                    invalid += 1
        rows_dev = rows.to(device)
        item_ids = torch.where(
            rows_dev.ge(0),
            row_to_item_dev[rows_dev.clamp_min(0)],
            torch.full_like(rows_dev, -1),
        )
        valid = item_ids.ge(0)
        np.add.at(counts, item_ids[valid].reshape(-1).cpu().numpy(), 1)

        full_scores = sasrec.score_items(batch).float()
        rec_scores = full_scores.gather(1, item_ids.clamp_min(0))
        pct = full_scores.unsqueeze(1).lt(rec_scores.unsqueeze(2)).float().mean(dim=2)
        pct_sum += float(pct[valid].sum())
        pct_n += int(valid.sum())

        if batch_idx % 50 == 0:
            rate = n_users / max(time.time() - start, 1e-6)
            print(f"[recs:{tag}] {n_users:,} users ({rate:,.0f} users/s)", flush=True)

    num_items = item_vocab_size - 1
    return {
        "tag": tag,
        "counts": counts,
        "n_users": n_users,
        f"HR@{top_k}": hits / max(n_users, 1),
        f"NDCG@{top_k}": ndcg / max(n_users, 1),
        f"coverage@{top_k}": float((counts > 0).sum() / max(num_items, 1)),
        "norm_entropy": normalized_entropy(counts),
        "gini": gini_coefficient(counts),
        "effective_items": effective_items(counts),
        "top10_item_share": top_share(counts, 10),
        "mean_rec_sasrec_pct": pct_sum / max(pct_n, 1),
        "invalid_rate": invalid / max(n_users * top_k, 1),
    }


@torch.no_grad()
def collect_sasrec_recommendations(
    sasrec: SASRec,
    dm: AmazonReviewDataModule,
    *,
    device: torch.device,
    top_k: int,
    max_batches: int | None,
    item_vocab_size: int,
    tag: str = "sasrec_teacher",
) -> dict[str, Any]:
    """Top-K recommendations of the SASRec teacher itself, as a reference.

    This is the distribution the reward is pulling TIGER toward - it shows how
    concentrated the teacher's own head is and lets the collapse plots compare
    the fine-tuned model against it directly.
    """
    counts = np.zeros(item_vocab_size, dtype=np.int64)
    n_users = 0
    hits = 0.0
    ndcg = 0.0
    pct_sum = 0.0
    pct_n = 0
    for batch_idx, raw_batch in enumerate(dm.test_dataloader(), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        batch = move_batch(raw_batch, device)
        scores = sasrec.score_items(batch).float()  # [B, V]
        scores[:, 0] = -float("inf")  # never recommend the pad item
        top_ids = scores.topk(top_k, dim=1).indices  # [B, K]
        n_users += int(top_ids.size(0))

        target = batch["target_item_ids"].long().unsqueeze(1)
        matches = top_ids.eq(target)
        hit = matches.any(dim=1)
        rank = matches.float().argmax(dim=1)
        discount = torch.zeros_like(hit, dtype=torch.float32)
        discount[hit] = 1.0 / torch.log2(rank[hit].float() + 2.0)
        hits += float(hit.float().sum())
        ndcg += float(discount.sum())

        np.add.at(counts, top_ids.reshape(-1).cpu().numpy(), 1)

        rec_scores = scores.gather(1, top_ids)
        pct = scores.unsqueeze(1).lt(rec_scores.unsqueeze(2)).float().mean(dim=2)
        pct_sum += float(pct.sum())
        pct_n += int(pct.numel())

    num_items = item_vocab_size - 1
    return {
        "tag": tag,
        "counts": counts,
        "n_users": n_users,
        f"HR@{top_k}": hits / max(n_users, 1),
        f"NDCG@{top_k}": ndcg / max(n_users, 1),
        f"coverage@{top_k}": float((counts > 0).sum() / max(num_items, 1)),
        "norm_entropy": normalized_entropy(counts),
        "gini": gini_coefficient(counts),
        "effective_items": effective_items(counts),
        "top10_item_share": top_share(counts, 10),
        "mean_rec_sasrec_pct": pct_sum / max(pct_n, 1),
        "invalid_rate": 0.0,
    }


@torch.no_grad()
def mean_sasrec_percentile_per_item(
    sasrec: SASRec,
    probe_batches: list[dict[str, Any]],
    *,
    device: torch.device,
    item_vocab_size: int,
    max_batches: int,
) -> np.ndarray:
    pct_sum = torch.zeros(item_vocab_size, dtype=torch.float64)
    n_users = 0
    for raw_batch in probe_batches[:max_batches]:
        batch = move_batch(raw_batch, device)
        scores = sasrec.score_items(batch).float()  # [B, V]
        order = scores.argsort(dim=1)
        ranks = torch.empty_like(order)
        arange = torch.arange(scores.size(1), device=device).expand_as(order)
        ranks.scatter_(1, order, arange)
        pct_sum += (ranks.double() / max(scores.size(1) - 1, 1)).sum(dim=0).cpu()
        n_users += int(scores.size(0))
    return (pct_sum / max(n_users, 1)).numpy()


def item_popularity(dm: AmazonReviewDataModule, item_vocab_size: int) -> np.ndarray:
    pop = np.zeros(item_vocab_size, dtype=np.int64)
    vc = dm.interactions["item_id"].value_counts()
    idx = vc.index.to_numpy()
    keep = (idx >= 0) & (idx < item_vocab_size)
    pop[idx[keep]] = vc.to_numpy()[keep]
    return pop


def item_title(dm: AmazonReviewDataModule, item_id: int, max_chars: int = 60) -> str:
    try:
        raw = dm.id_to_raw_item(int(item_id))
        meta = dm.get_item_metadata(raw)
        title = str((meta or {}).get("title", "") or "")
        return title[:max_chars]
    except Exception:
        return ""


def run_collapse_analysis(
    *,
    args: argparse.Namespace,
    cfgs: Mapping[str, Mapping[str, Any]],
    dm: AmazonReviewDataModule,
    sasrec: SASRec,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    run_dir: Path | None,
    neg_counts_hard: np.ndarray,
    probe_batches: list[dict[str, Any]],
    device: torch.device,
    out_dir: Path,
    dataset: str,
    cached_rec_results: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame | None, dict[str, float] | None, dict[str, Any] | None]:
    vocab = int(dm.item_vocab_size)
    checkpoints: dict[str, Path] = {
        "sft_baseline": path_from_repo(args.sft_artifacts) / dataset / "checkpoints" / "best.pt",
    }
    if run_dir is not None:
        for tag, name in (("finetuned_best", "best.pt"), ("finetuned_last", "last.pt")):
            path = run_dir / "checkpoints" / name
            if path.exists():
                checkpoints[tag] = path

    catalog_lookup = {tuple(int(t) for t in row): i for i, row in enumerate(item_tokens.tolist())}
    num_beams = int(cfgs["tiger"].get("eval_num_beams", 20))
    constrain = bool(cfgs["tiger"].get("constrain_valid_sequences", True))

    rec_results: dict[str, dict[str, Any]] = {}
    for tag, checkpoint_path in checkpoints.items():
        if cached_rec_results and tag in cached_rec_results:
            rec_results[tag] = cached_rec_results[tag]
            print(f"[recs:{tag}] reusing results computed earlier in this sweep", flush=True)
            continue
        model = build_tiger_model(dm, cfgs["tiger"])
        load_sft_checkpoint(model, checkpoint_path)
        rec_results[tag] = collect_recommendations(
            model, dm, sasrec,
            device=device, top_k=args.rec_topk, num_beams=num_beams, constrain=constrain,
            max_batches=args.rec_batches, catalog_lookup=catalog_lookup,
            row_to_item=row_to_item, item_vocab_size=vocab, tag=tag,
        )
        summary = {k: v for k, v in rec_results[tag].items() if k != "counts"}
        print(f"[recs:{tag}] {json.dumps(summary, sort_keys=True, default=float)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The teacher's own top-K as a reference for what the reward optimizes toward.
    if cached_rec_results and "sasrec_teacher" in cached_rec_results:
        rec_results["sasrec_teacher"] = cached_rec_results["sasrec_teacher"]
        print("[recs:sasrec_teacher] reusing results computed earlier in this sweep", flush=True)
    else:
        rec_results["sasrec_teacher"] = collect_sasrec_recommendations(
            sasrec, dm,
            device=device, top_k=args.rec_topk, max_batches=args.rec_batches, item_vocab_size=vocab,
        )
        teacher_summary = {k: v for k, v in rec_results["sasrec_teacher"].items() if k != "counts"}
        print(f"[recs:sasrec_teacher] {json.dumps(teacher_summary, sort_keys=True, default=float)}", flush=True)

    # Actual popularity (train interactions) vs model popularity (times in top-K).
    pop = item_popularity(dm, vocab).astype(np.float64)
    pop_order = np.argsort(pop)[::-1]
    for res in rec_results.values():
        counts = res["counts"].astype(np.float64)
        total = max(counts.sum(), 1.0)
        res["rec_share_top100_popular"] = float(counts[pop_order[:100]].sum() / total)
        res["rec_share_top1000_popular"] = float(counts[pop_order[:1000]].sum() / total)

    collapsed_table: pd.DataFrame | None = None
    correlations: dict[str, float] | None = None
    overlaps: dict[str, Any] | None = None
    finetuned_tag = "finetuned_last" if "finetuned_last" in rec_results else (
        "finetuned_best" if "finetuned_best" in rec_results else None
    )
    if finetuned_tag is not None:
        ft_counts = rec_results[finetuned_tag]["counts"].astype(np.float64)
        base_counts = rec_results["sft_baseline"]["counts"].astype(np.float64)
        teacher_counts = rec_results["sasrec_teacher"]["counts"].astype(np.float64)
        mean_pct = mean_sasrec_percentile_per_item(
            sasrec, probe_batches, device=device, item_vocab_size=vocab, max_batches=args.pct_batches
        )

        top_ids = np.argsort(ft_counts)[::-1][:20]
        collapsed_table = pd.DataFrame(
            {
                "item_id": top_ids,
                "raw_item_id": [dm.id_to_raw_item(int(i)) for i in top_ids],
                "title": [item_title(dm, int(i)) for i in top_ids],
                "rec_count_finetuned": ft_counts[top_ids].astype(int),
                "rec_share_finetuned": ft_counts[top_ids] / max(ft_counts.sum(), 1.0),
                "rec_share_baseline": base_counts[top_ids] / max(base_counts.sum(), 1.0),
                "rec_share_sasrec": teacher_counts[top_ids] / max(teacher_counts.sum(), 1.0),
                "sasrec_mean_pct": mean_pct[top_ids],
                "popularity": pop[top_ids].astype(int),
                "neg_sample_count": neg_counts_hard[top_ids],
            }
        )
        collapsed_table.to_csv(out_dir / "collapsed_items.csv", index=False)

        active = (ft_counts > 0) | (neg_counts_hard > 0) | (base_counts > 0) | (teacher_counts > 0)
        correlations = {
            "rec count (finetuned) vs times sampled as hard negative":
                spearman_corr(ft_counts[active], neg_counts_hard[active].astype(np.float64)),
            "rec count (finetuned) vs mean SASRec percentile":
                spearman_corr(ft_counts[active], mean_pct[active]),
            "rec count (finetuned) vs rec count (sasrec teacher)":
                spearman_corr(ft_counts[active], teacher_counts[active]),
            "rec count (finetuned) vs train popularity":
                spearman_corr(ft_counts[active], pop[active]),
            "rec count (sft baseline) vs train popularity":
                spearman_corr(base_counts[active], pop[active]),
        }

        def top_set(values: np.ndarray, n: int = 50) -> set[int]:
            return set(int(i) for i in np.argsort(values)[::-1][:n])

        ft_top = top_set(ft_counts)
        overlaps = {
            "top recommended (finetuned) ∩ top sampled hard negatives": len(ft_top & top_set(neg_counts_hard.astype(np.float64))),
            "top recommended (finetuned) ∩ top SASRec-scored items": len(ft_top & top_set(mean_pct)),
            "top recommended (finetuned) ∩ top recommended (sasrec teacher)": len(ft_top & top_set(teacher_counts)),
            "top recommended (finetuned) ∩ top popular items": len(ft_top & top_set(pop)),
            "top recommended (finetuned) ∩ top recommended (baseline)": len(ft_top & top_set(base_counts)),
        }

        plot_collapse(
            rec_results, neg_counts_hard, top_ids, args.rec_topk, pop,
            out_dir / f"recommendation_collapse_{dataset}.png", dataset,
        )

    metrics_frame = pd.DataFrame(
        [{k: v for k, v in res.items() if k != "counts"} for res in rec_results.values()]
    )
    metrics_frame.to_csv(out_dir / "model_metrics.csv", index=False)
    return rec_results, collapsed_table, correlations, overlaps


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _shade_easy_phase(ax, switch_epoch: float, x_max: float) -> None:
    if switch_epoch <= 0:
        return
    ax.axvspan(0, min(switch_epoch, x_max), color=EASY_SHADE, alpha=0.10, zorder=0)
    ax.text(
        0.02, 0.02, "easy phase", transform=ax.get_yaxis_transform(),
        fontsize=7, color=PHASE_COLORS["easy"], va="bottom",
    )


def plot_sampling_diversity(
    frame: pd.DataFrame, switch_epoch: float, out_path: Path, dataset: str, source_label: str
) -> None:
    x = frame["epoch"].to_numpy() + 1
    x_max = float(x.max())
    panels = [
        ("item diversity of negatives", [
            ("unique_ratio", "unique / sampled"),
            ("catalog_coverage", "catalog coverage"),
        ], (0, 1.05)),
        ("concentration of negative items", [
            ("norm_entropy", "normalized entropy"),
            ("top1pct_share", "share on top-1% items"),
            ("gini", "Gini"),
        ], (0, 1.05)),
        ("negative reward level / spread", [
            ("reward_mean", "reward mean"),
            ("reward_std", "reward std"),
            ("ctx_reward_std_mean", "within-context std"),
        ], None),
        ("negative hardness (SASRec percentile)", [
            ("pct_mean", "mean percentile"),
            ("pct_median", "median percentile"),
        ], (0, 1.05)),
    ]
    series_colors = ["#2a78d6", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    fig.patch.set_facecolor(CHROME["surface"])
    for ax, (title, series, ylim) in zip(axes.ravel(), panels):
        _style_axis(ax)
        _shade_easy_phase(ax, switch_epoch, x_max)
        for (key, label), color in zip(series, series_colors):
            ax.plot(x, frame[key].to_numpy(), color=color, linewidth=2.0, marker="o", markersize=3, label=label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
        ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])
    fig.suptitle(
        f"Negative sampling per fine-tuning epoch ({source_label}) — {dataset}",
        fontsize=12, color=CHROME["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_example_users(
    example_infos: list[dict[str, Any]],
    per_epoch_user_data: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    epochs: list[int],
    switch_epoch: float,
    out_path: Path,
    dataset: str,
    source_label: str,
) -> None:
    n_users = len(example_infos)
    if n_users == 0 or not epochs:
        return
    fig, axes = plt.subplots(n_users, 2, figsize=(12.5, 3.0 * n_users), squeeze=False)
    fig.patch.set_facecolor(CHROME["surface"])
    rng = np.random.default_rng(0)
    for u, info in enumerate(example_infos):
        # ---- panel A: negative rewards per epoch --------------------------
        ax = axes[u][0]
        _style_axis(ax)
        medians, med_x = [], []
        n_sampled = []
        for e in epochs:
            data = per_epoch_user_data.get(e, {}).get(u)
            if data is None or data["item_ids"].size == 0:
                n_sampled.append(0)
                continue
            ids = data["item_ids"].reshape(-1)
            rewards = data["rewards"].reshape(-1)[ids >= 0]
            n_sampled.append(rewards.size)
            if rewards.size == 0:
                continue
            phase = "easy" if (e + 0.5) < switch_epoch else "hard"
            jitter = rng.uniform(-0.28, 0.28, size=rewards.size)
            ax.scatter(
                np.full(rewards.size, e + 1) + jitter,
                np.clip(rewards, 1e-8, None),
                s=5, alpha=0.35, color=PHASE_COLORS[phase], edgecolors="none",
            )
            med_x.append(e + 1)
            medians.append(np.median(rewards))
        if med_x:
            ax.plot(med_x, medians, color=CHROME["ink"], linewidth=1.6)
        ax.set_yscale("log")
        _shade_easy_phase(ax, switch_epoch, max(epochs) + 1)
        avg_n = np.mean([n for n in n_sampled if n > 0]) if any(n_sampled) else 0
        ax.set_ylabel(
            f"user {info['user_id']}\n(hist={info['history_len']})\nnegative reward",
            fontsize=8, color=CHROME["ink2"],
        )
        if u == 0:
            ax.set_title(
                "SASRec reward of the sampled negatives "
                "(green=easy phase, red=hard phase; line=median)",
                fontsize=9, color=CHROME["ink"],
            )
        ax.text(0.99, 0.02, f"~{avg_n:.0f} negatives/epoch", transform=ax.transAxes,
                fontsize=7, color=CHROME["muted"], ha="right")
        if u == n_users - 1:
            ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])

        # ---- panel B: item diversity of the same user's negatives ----------
        ax = axes[u][1]
        _style_axis(ax)
        xs, unique_ratio, new_ratio = [], [], []
        seen: set[int] = set()
        for e in epochs:
            data = per_epoch_user_data.get(e, {}).get(u)
            if data is None or data["item_ids"].size == 0:
                continue
            ids = data["item_ids"].reshape(-1)
            ids = ids[ids >= 0]
            if ids.size == 0:
                continue
            uniq = set(int(i) for i in ids)
            xs.append(e + 1)
            unique_ratio.append(len(uniq) / ids.size)
            new_ratio.append(len(uniq - seen) / max(len(uniq), 1))
            seen |= uniq
        ax.plot(xs, unique_ratio, color="#2a78d6", linewidth=2.0, marker="o", markersize=3,
                label="unique / sampled (within epoch)")
        ax.plot(xs, new_ratio, color="#eda100", linewidth=2.0, marker="o", markersize=3,
                label="share of uniques never sampled before")
        ax.set_ylim(0, 1.05)
        _shade_easy_phase(ax, switch_epoch, max(epochs) + 1)
        if u == 0:
            ax.set_title("item diversity of the same user's negatives", fontsize=9, color=CHROME["ink"])
            ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])
        if u == n_users - 1:
            ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
    fig.suptitle(
        f"Negative sampling for example users ({source_label}) — {dataset}",
        fontsize=12, color=CHROME["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_collapse_dynamics(
    frame: pd.DataFrame, switch_epoch: float, out_path: Path, dataset: str,
    ndcg_label: str = "val NDCG@1",
) -> None:
    """Per-epoch collapse of the top-1 validation recommendations."""
    if frame.empty:
        return
    x = frame["epoch"].to_numpy() + 1
    panels = [
        ("val_ndcg", ndcg_label, "#1baf7a"),
        ("val_top1_coverage", "catalog coverage of val top-1 recs", "#2a78d6"),
        ("val_top1_top10_share", "share of val top-1 recs on 10 items", "#e34948"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.4))
    fig.patch.set_facecolor(CHROME["surface"])
    for ax, (key, title, color) in zip(axes, panels):
        _style_axis(ax)
        _shade_easy_phase(ax, switch_epoch, float(x.max()))
        if key in frame.columns:
            ax.plot(x, frame[key].to_numpy(), color=color, linewidth=2.0, marker="o", markersize=3)
        ax.set_title(title, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
    fig.suptitle(
        f"Collapse dynamics during fine-tuning (validation top-1) — {dataset}",
        fontsize=12, color=CHROME["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_val_trajectory(
    history: list[dict[str, Any]], switch_epoch: float, out_path: Path, dataset: str
) -> None:
    xs, ndcgs, coverages = [], [], []
    for record in history:
        val = record.get("val", {})
        if "val/sft/generation_ndcg@1" in val:
            xs.append(int(record.get("epoch", 0)) + 1)
            ndcgs.append(float(val["val/sft/generation_ndcg@1"]))
            coverages.append(float(val.get("val/sft/generation_coverage@1", float("nan"))))
    if not xs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    fig.patch.set_facecolor(CHROME["surface"])
    for ax, values, title in zip(axes, (ndcgs, coverages), ("val NDCG@1", "val coverage@1")):
        _style_axis(ax)
        _shade_easy_phase(ax, switch_epoch, max(xs))
        ax.plot(xs, values, color="#1baf7a", linewidth=2.0, marker="o", markersize=3)
        ax.set_title(title, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
    fig.suptitle(
        f"Validation trajectory of the sasrec fine-tune — {dataset}", fontsize=12, color=CHROME["ink"]
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_collapse(
    rec_results: dict[str, dict[str, Any]],
    neg_counts: np.ndarray,
    top_collapsed_ids: np.ndarray,
    top_k: int,
    popularity: np.ndarray,
    out_path: Path,
    dataset: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.0))
    fig.patch.set_facecolor(CHROME["surface"])

    # Panel 1: sorted recommendation-share curves (MODEL popularity ranking:
    # each model's items sorted by how often that model recommends them, so
    # rank 1 can be a different item per curve).
    ax = axes[0]
    _style_axis(ax)
    for tag, res in rec_results.items():
        counts = np.sort(res["counts"])[::-1]
        counts = counts[counts > 0]
        share = counts / counts.sum()
        ax.plot(
            np.arange(1, share.size + 1), share,
            color=MODEL_COLORS[tag], linewidth=2.0, label=tag,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"item share of top-{top_k} recommendations (sorted)", fontsize=10, color=CHROME["ink"])
    ax.set_xlabel("item rank (model popularity: by times in top-10)", fontsize=8, color=CHROME["ink2"])
    ax.set_ylabel("recommendation share", fontsize=8, color=CHROME["ink2"])
    ax.legend(fontsize=8, frameon=False, labelcolor=CHROME["ink2"])

    # Panel 2: recommendation share vs ACTUAL popularity (train interactions).
    # One shared x-axis for all models: items sorted once by their train
    # interaction count, so position x is the same item on every curve.
    ax = axes[1]
    _style_axis(ax)
    order = np.argsort(popularity)[::-1]
    ranks = np.arange(1, order.size + 1)
    pop_sorted = popularity[order].astype(np.float64)
    if pop_sorted.sum() > 0:
        ax.plot(
            ranks, np.cumsum(pop_sorted) / pop_sorted.sum(),
            color=CHROME["ink2"], linewidth=1.6, linestyle="--",
            label="train interactions (actual popularity)",
        )
    for tag, res in rec_results.items():
        counts = res["counts"][order].astype(np.float64)
        total = counts.sum()
        if total <= 0:
            continue
        ax.plot(
            ranks, np.cumsum(counts) / total,
            color=MODEL_COLORS[tag], linewidth=2.0, label=tag,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_title("cumulative rec share vs actual popularity", fontsize=10, color=CHROME["ink"])
    ax.set_xlabel("item rank (actual popularity: by train interactions)", fontsize=8, color=CHROME["ink2"])
    ax.set_ylabel("cumulative recommendation share", fontsize=8, color=CHROME["ink2"])
    ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])

    # Panel 3: grouped bars of concentration metrics.
    ax = axes[2]
    _style_axis(ax)
    metrics = [(f"coverage@{top_k}", "coverage"), ("norm_entropy", "norm. entropy"), ("top10_item_share", "top-10 items' share")]
    tags = list(rec_results.keys())
    width = 0.8 / len(tags)
    for t_idx, tag in enumerate(tags):
        values = [rec_results[tag][key] for key, _ in metrics]
        positions = np.arange(len(metrics)) + (t_idx - (len(tags) - 1) / 2) * width
        bars = ax.bar(positions, values, width=width * 0.92, color=MODEL_COLORS[tag], label=tag)
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, value, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7, color=CHROME["ink2"],
            )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("recommendation concentration", fontsize=10, color=CHROME["ink"])
    ax.legend(fontsize=8, frameon=False, labelcolor=CHROME["ink2"])

    # Panel 4: negatives-sampled vs recommended (per item).
    ax = axes[3]
    _style_axis(ax)
    last = rec_results.get("finetuned_last") or rec_results.get("finetuned_best")
    rec_counts = last["counts"]
    mask = (neg_counts > 0) | (rec_counts > 0)
    rho = spearman_corr(neg_counts[mask].astype(np.float64), rec_counts[mask].astype(np.float64))
    ax.scatter(
        neg_counts[mask] + 1, rec_counts[mask] + 1,
        s=6, alpha=0.3, color="#2a78d6", edgecolors="none",
    )
    highlight = top_collapsed_ids[top_collapsed_ids < neg_counts.size]
    ax.scatter(
        neg_counts[highlight] + 1, rec_counts[highlight] + 1,
        s=26, color="#e34948", edgecolors=CHROME["surface"], linewidths=0.5,
        label="top-20 recommended",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"sampled as negative vs recommended ({last['tag']})", fontsize=10, color=CHROME["ink"])
    ax.set_xlabel("times sampled as negative + 1 (hard phase)", fontsize=8, color=CHROME["ink2"])
    ax.set_ylabel(f"times in top-{top_k} recs + 1", fontsize=8, color=CHROME["ink2"])
    ax.text(0.03, 0.95, f"Spearman rho = {rho:.3f}", transform=ax.transAxes, fontsize=8,
            color=CHROME["ink"], va="top")
    ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])

    fig.suptitle(f"Recommendation collapse after sasrec fine-tuning — {dataset}", fontsize=12, color=CHROME["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_temperature_frontier(
    rows: list[dict[str, Any]],
    baseline: Mapping[str, Any] | None,
    teacher: Mapping[str, Any] | None,
    top_k: int,
    out_path: Path,
    dataset: str,
) -> None:
    """Accuracy-diversity frontier traced by the reward temperature."""
    cov_key = f"coverage@{top_k}"
    ndcg_key = f"NDCG@{top_k}"
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    fig.patch.set_facecolor(CHROME["surface"])
    _style_axis(ax)
    for tag in ("finetuned_best", "finetuned_last"):
        points = sorted((r for r in rows if r["model"] == tag), key=lambda r: r["temperature"])
        if not points:
            continue
        ax.plot(
            [p[cov_key] for p in points],
            [p[ndcg_key] for p in points],
            color=MODEL_COLORS[tag], linewidth=2.0, marker="o", markersize=5, label=tag,
        )
        for p in points:
            ax.annotate(
                f"T={p['temperature']:g}", (p[cov_key], p[ndcg_key]),
                textcoords="offset points", xytext=(6, 4), fontsize=7, color=CHROME["ink2"],
            )
    for tag, res in (("sft_baseline", baseline), ("sasrec_teacher", teacher)):
        if res is not None:
            ax.scatter(
                [res[cov_key]], [res[ndcg_key]],
                color=MODEL_COLORS[tag], s=70, marker="D",
                edgecolors=CHROME["surface"], linewidths=0.8, label=tag, zorder=5,
            )
    ax.set_xlabel(f"coverage@{top_k} (test)", fontsize=9, color=CHROME["ink2"])
    ax.set_ylabel(f"NDCG@{top_k} (test)", fontsize=9, color=CHROME["ink2"])
    ax.set_title(
        f"Reward-temperature frontier: ranking vs diversity — {dataset}",
        fontsize=11, color=CHROME["ink"],
    )
    ax.legend(fontsize=8, frameon=False, labelcolor=CHROME["ink2"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def write_sweep_summary(
    out_dir: Path,
    rows: list[dict[str, Any]],
    shared_recs: Mapping[str, Mapping[str, Any]],
    top_k: int,
    dataset: str,
    monitor_k: int,
) -> None:
    lines = [
        f"# Reward-temperature sweep — {dataset}",
        "",
        f"| T | model | best epoch | best val ndcg@{monitor_k} | HR@{top_k} | NDCG@{top_k} | "
        f"coverage@{top_k} | norm. entropy | effective items | top-10 share |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (r["temperature"], r["model"])):
        best_val = row.get("best_val_ndcg")
        lines.append(
            f"| {row['temperature']:g} | {row['model']} | {row.get('best_epoch', '')} | "
            + (f"{best_val:.4f}" if isinstance(best_val, float) else "")
            + f" | {row[f'HR@{top_k}']:.4f} | {row[f'NDCG@{top_k}']:.4f} | {row[f'coverage@{top_k}']:.4f} | "
            f"{row['norm_entropy']:.3f} | {row['effective_items']:.0f} | {row['top10_item_share']:.3f} |"
        )
    for tag in ("sft_baseline", "sasrec_teacher"):
        res = shared_recs.get(tag)
        if res is not None:
            lines.append(
                f"| — | {tag} |  |  | {res[f'HR@{top_k}']:.4f} | {res[f'NDCG@{top_k}']:.4f} | "
                f"{res[f'coverage@{top_k}']:.4f} | {res['norm_entropy']:.3f} | "
                f"{res['effective_items']:.0f} | {res['top10_item_share']:.3f} |"
            )
    lines.append("")
    (out_dir / "temperature_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved {out_dir / 'temperature_sweep.md'}", flush=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(
    out_dir: Path,
    *,
    dataset: str,
    mode: str,
    run_meta: Mapping[str, Any],
    switch_epoch: float,
    sampling_frame: pd.DataFrame | None,
    rec_results: Mapping[str, Mapping[str, Any]] | None,
    collapsed_table: pd.DataFrame | None,
    correlations: Mapping[str, float] | None,
    overlaps: Mapping[str, Any] | None,
    top_k: int,
) -> None:
    lines = [
        f"# SASRec-reward fine-tuning diagnosis — {dataset}",
        "",
        f"- mode: **{mode}** "
        + (f"(instrumented re-run of the rewards_ablation pipeline, per-epoch validation on "
           f"ndcg@{run_meta.get('monitor_k', 1)} over the full val set)"
           if mode == "train" else "(offline sampler replay against an existing run)"),
        f"- fine-tuned run: `{run_meta.get('run_dir', 'n/a')}`",
        f"- epochs completed: {run_meta.get('epochs_completed', 'n/a')}, "
        f"best epoch: {run_meta.get('best_epoch', 'n/a')}",
        f"- curriculum: easy negatives for the first {run_meta.get('easy_negative_steps', 'n/a')} steps "
        f"(≈ first {switch_epoch:.2f} epochs), hard (high-SASRec-score) negatives afterwards",
    ]
    if mode == "train" and run_meta.get("reward_transform") is not None:
        num_random = int(run_meta.get("num_random_negatives", 0))
        lines.append(
            f"- config: reward transform=**{run_meta['reward_transform']}** "
            f"(T={run_meta.get('reward_temperature', 1.0):g}), "
            f"TB positive included={not run_meta.get('tb_exclude_positive', False)}, "
            f"negatives=3 ({3 - num_random} curriculum + {num_random} random), "
            f"lr={run_meta.get('lr', 'trainer.yaml')}, monitor=ndcg@{run_meta.get('monitor_k', 1)}"
        )
    lines.append("")
    if sampling_frame is not None and not sampling_frame.empty:
        has_val = "val_top1_coverage" in sampling_frame.columns
        header = (
            "| epoch | phase | unique/sampled | catalog coverage | norm. entropy | top-1% share | "
            "reward mean | within-ctx std | hardness pct |"
        )
        divider = "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        if has_val:
            header += " val ndcg | val top-1 coverage | val top-10 share |"
            divider += " ---: | ---: | ---: |"
        lines += ["## Negative-sampling diversity per epoch (actual training draws)" if mode == "train"
                  else "## Negative-sampling diversity per epoch (replayed)", "", header, divider]
        for _, row in sampling_frame.iterrows():
            line = (
                f"| {int(row['epoch']) + 1} | {row['phase']} | {row['unique_ratio']:.3f} | "
                f"{row['catalog_coverage']:.3f} | {row['norm_entropy']:.3f} | {row['top1pct_share']:.3f} | "
                f"{row['reward_mean']:.3f} | {row['ctx_reward_std_mean']:.3f} | {row['pct_mean']:.3f} |"
            )
            if has_val:
                def fmt(key: str, spec: str) -> str:
                    value = row.get(key)
                    return format(value, spec) if pd.notna(value) else ""
                line += (
                    f" {fmt('val_ndcg', '.4f')} | {fmt('val_top1_coverage', '.4f')} | "
                    f"{fmt('val_top1_top10_share', '.3f')} |"
                )
            lines.append(line)
        lines.append("")
    if rec_results:
        has_pop = any("rec_share_top100_popular" in res for res in rec_results.values())
        header = (
            f"| model | HR@{top_k} | NDCG@{top_k} | coverage@{top_k} | norm. entropy | "
            "effective items | top-10 items' share | mean SASRec pct of recs |"
        )
        divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        if has_pop:
            header += " rec share on top-100 popular | on top-1000 popular |"
            divider += " ---: | ---: |"
        lines += ["## Recommendation concentration (test set)", "", header, divider]
        for tag, res in rec_results.items():
            line = (
                f"| {tag} | {res[f'HR@{top_k}']:.4f} | {res[f'NDCG@{top_k}']:.4f} | "
                f"{res[f'coverage@{top_k}']:.4f} | {res['norm_entropy']:.3f} | "
                f"{res['effective_items']:.0f} | {res['top10_item_share']:.3f} | {res['mean_rec_sasrec_pct']:.3f} |"
            )
            if has_pop:
                line += (
                    f" {res.get('rec_share_top100_popular', float('nan')):.3f} | "
                    f"{res.get('rec_share_top1000_popular', float('nan')):.3f} |"
                )
            lines.append(line)
        lines.append("")
    if collapsed_table is not None and not collapsed_table.empty:
        has_teacher = "rec_share_sasrec" in collapsed_table.columns
        lines += ["## Items the fine-tuned recommendations collapse into", ""]
        header = "| # | item | title | rec share (ft) | rec share (sft) |"
        divider = "| ---: | --- | --- | ---: | ---: |"
        if has_teacher:
            header += " rec share (sasrec) |"
            divider += " ---: |"
        header += " SASRec pct | popularity | sampled as neg |"
        divider += " ---: | ---: | ---: |"
        lines += [header, divider]
        for i, (_, row) in enumerate(collapsed_table.head(15).iterrows(), start=1):
            line = (
                f"| {i} | {row['raw_item_id']} | {row['title']} | {row['rec_share_finetuned']:.3%} | "
                f"{row['rec_share_baseline']:.3%} |"
            )
            if has_teacher:
                line += f" {row['rec_share_sasrec']:.3%} |"
            line += (
                f" {row['sasrec_mean_pct']:.3f} | {int(row['popularity'])} | {int(row['neg_sample_count'])} |"
            )
            lines.append(line)
        lines.append("")
    if correlations:
        lines += ["## Correlations (per item, Spearman)", ""]
        for name, value in correlations.items():
            lines.append(f"- {name}: **{value:.3f}**")
        lines.append("")
    if overlaps:
        lines += ["## Top-50 set overlaps", ""]
        for name, value in overlaps.items():
            lines.append(f"- {name}: **{value}/50**")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved {out_dir / 'report.md'}", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Diagnose negative sampling and recommendation collapse for the sasrec reward")
    p.add_argument("--mode", choices=["train", "replay"], default="train",
                   help="train: instrumented re-run of the fine-tuning; replay: offline sampler simulation")
    p.add_argument("--dataset", choices=["beauty", "instruments"], default="beauty")
    p.add_argument("--config-dir", default="configs/exp1-standard_params")
    p.add_argument("--sft-artifacts", default="artifacts/tiger_sft_amazon")
    p.add_argument("--semantic-dir", default=None)
    p.add_argument("--sasrec-checkpoint-dir", default="checkpoints/exp2-tuned-params/sasrec")
    p.add_argument("--output-dir", default="experiments/sasrec_reward_analysis")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-example-users", type=int, default=4)
    p.add_argument("--easy-negative-steps", type=int, default=10000,
                   help="Default: run config (replay) or 2000 (train)")
    # train mode
    p.add_argument("--max-epochs", type=int, default=None, help="Default: trainer.yaml (early stopping applies)")
    p.add_argument("--early-stopping-patience", type=int, default=None, help="Epochs; default: trainer.yaml")
    p.add_argument("--gflow-weight", type=float, default=0.1)
    p.add_argument("--improved", action="store_true", default=True,
                   help="Preset of the collapse/ranking fixes: exp reward (T=1, teacher-softmax distillation), "
                        "TB excludes the positive, 2 hard + 1 random negatives, lr=3e-5. "
                        "Explicit flags below override individual preset values.")
    p.add_argument("--reuse-logz-cache", action="store_true",
                   help="Reuse a cached logZ entry for the current reward transform/temperature; "
                        "by default logZ is recomputed every run to match the specified reward parameters")
    p.add_argument("--reward-transform", choices=list(TB_SAFE_SASREC_TRANSFORMS), default=None,
                   help="SASRec score -> reward transform (default: softplus; improved: exp)")
    p.add_argument("--reward-temperature", type=float, nargs="+", default=None,
                   help="Temperature(s) for the reward transform (default 1.0). Several values run a "
                        "sweep in train mode: one full fine-tune per temperature plus an "
                        "NDCG-vs-coverage frontier plot and summary across temperatures")
    p.add_argument("--tb-exclude-positive", action=argparse.BooleanOptionalAction, default=None,
                   help="TB candidates = negatives only; SFT alone owns the target")
    p.add_argument("--num-random-negatives", type=int, default=None,
                   help="Of the 3 negatives, keep this many uniform-random (default 0; improved: 1)")
    p.add_argument("--lr", type=float, default=None,
                   help="Fine-tuning learning rate (default: trainer.yaml 3e-4; improved: 3e-5)")
    p.add_argument("--val-topk", type=int, default=None,
                   help="Validate/monitor ndcg@K with K beams (default: ndcg@5, beam 5)")
    p.add_argument("--val-sample-fraction", type=float, default=None,
                   help="Subsample the val set per validation (default: the entire val set)")
    p.add_argument("--logz-cache-dir", default="artifacts/rewards_ablation_logz")
    p.add_argument("--pct-batches-per-epoch", type=int, default=8,
                   help="Train batches per epoch for negative-hardness percentiles")
    p.add_argument("--max-train-batches", type=int, default=None, help="Debug cap")
    p.add_argument("--max-eval-batches", type=int, default=None, help="Debug cap for per-epoch validation")
    # replay mode
    p.add_argument("--run-dir", default=None,
                   help="replay: fine-tuned run dir (default: latest experiments/rewards_ablation/*/<dataset>/sasrec)")
    p.add_argument("--epochs", type=int, default=None, help="replay: epochs to simulate (default: run's count)")
    p.add_argument("--max-analyzed-epochs", type=int, default=20)
    p.add_argument("--sample-batches", type=int, default=24, help="replay: probe train batches per epoch")
    p.add_argument("--example-draws", type=int, default=20, help="replay: sampler draws per example user per epoch")
    # collapse analysis
    p.add_argument("--rec-topk", type=int, default=10)
    p.add_argument("--rec-batches", type=int, default=None, help="Cap test batches for generation (default: all)")
    p.add_argument("--pct-batches", type=int, default=8, help="Probe batches for per-item SASRec percentiles")
    p.add_argument("--skip-collapse", action="store_true")
    return p.parse_args()


def resolve_experiment_options(args: argparse.Namespace) -> None:
    """Fill unset experiment knobs from the legacy or --improved preset.

    Both presets monitor ndcg@5 (beam 5) on the ENTIRE validation set.
    """
    presets = {
        False: dict(reward_transform="softplus", reward_temperature=[1.0],
                    tb_exclude_positive=False, num_random_negatives=0, lr=None, val_topk=5),
        True: dict(reward_transform="exp", reward_temperature=[1.0],
                   tb_exclude_positive=True, num_random_negatives=1, lr=3e-5, val_topk=5),
    }
    for key, value in presets[bool(args.improved)].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def main() -> int:
    args = parse_args()
    resolve_experiment_options(args)
    cfgs = load_exp_configs(path_from_repo(args.config_dir))
    seed = int(args.seed if args.seed is not None else cfgs["trainer"].get("seed", 43))
    set_seed(seed)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = path_from_repo(args.output_dir) / f"{stamp}_{args.dataset}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "cli_args.json", vars(args))
    print(f"[main] output dir: {out_dir}", flush=True)

    semantic_dir = path_from_repo(
        args.semantic_dir if args.semantic_dir is not None else cfgs["semantic"].get("semantic_dir")
    )
    dm = build_datamodule(
        args.dataset,
        cfgs["dataset"],
        semantic_id_path=semantic_dir / args.dataset / "semantic_ids.json",
        seed=seed,
        num_workers=0,
    )
    item_tokens = semantic_item_tokens(dm)
    row_to_item = semantic_row_to_item_id(dm)
    item_to_row = item_id_to_semantic_row(dm)
    num_items = int(dm.num_items)
    vocab = int(dm.item_vocab_size)
    catalog_lookup = {tuple(int(t) for t in row): i for i, row in enumerate(item_tokens.tolist())}

    sasrec = load_sasrec_teacher(
        find_sasrec_checkpoint(path_from_repo(args.sasrec_checkpoint_dir) / args.dataset), dm
    )
    sasrec.to(device)
    sasrec_alignment_check(sasrec, dm, device)

    temperatures = [float(t) for t in args.reward_temperature]

    def make_base_reward(temperature: float) -> tuple[str, torch.nn.Module]:
        key = sasrec_reward_key(args.reward_transform, temperature)
        reward = build_reward_model(
            key,
            sasrec=sasrec,
            row_to_item=row_to_item,
            sasrec_transform=args.reward_transform,
            sasrec_temperature=temperature,
        ).to(device)
        return key, reward

    print(
        f"[main] experiment options: transform={args.reward_transform}, "
        f"temperatures={temperatures}, tb_exclude_positive={args.tb_exclude_positive}, "
        f"num_random_negatives={args.num_random_negatives}, lr={args.lr}, "
        f"val_topk={args.val_topk}"
        + (" (--improved preset)" if args.improved else ""),
        flush=True,
    )

    example_infos = select_example_indices(dm, args.num_example_users)
    print(f"[main] example users: {example_infos}", flush=True)

    steps_per_epoch = int(np.ceil(len(dm.train_dataset) / dm.cfg.batch_size))
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(args.max_train_batches))

    sampling_frame: pd.DataFrame | None = None
    neg_counts_hard = np.zeros(vocab, dtype=np.int64)
    run_dir: Path | None = None
    run_meta: dict[str, Any] = {}

    if args.mode == "train":
        easy_steps = int(args.easy_negative_steps if args.easy_negative_steps is not None else 2000)
        switch_epoch = easy_steps / steps_per_epoch
        print(f"[main] steps/epoch={steps_per_epoch} -> curriculum switch at epoch {switch_epoch:.2f}", flush=True)
        probe_batches = build_probe_batches(dm, args.pct_batches, seed)

        sweep = len(temperatures) > 1
        shared_recs: dict[str, dict[str, Any]] = {}
        sweep_rows: list[dict[str, Any]] = []
        for temperature in temperatures:
            temp_dir = out_dir / f"T{temperature:g}" if sweep else out_dir
            temp_dir.mkdir(parents=True, exist_ok=True)
            reward_key, base_reward = make_base_reward(temperature)
            print(f"\n[main] === temperature {temperature:g} (reward={reward_key}) ===", flush=True)

            recorder, run_dir, run_meta = run_instrumented_finetune(
                args=args, cfgs=cfgs, dm=dm, dataset=args.dataset,
                sasrec=sasrec, base_reward=base_reward, reward_temperature=temperature,
                item_tokens=item_tokens, row_to_item=row_to_item, item_to_row=item_to_row,
                example_infos=example_infos, catalog_lookup=catalog_lookup,
                device=device, out_dir=temp_dir, easy_steps=easy_steps,
            )

            sampling_frame = pd.DataFrame(recorder.epoch_rows)
            if recorder.val_rows:
                sampling_frame = sampling_frame.merge(pd.DataFrame(recorder.val_rows), on="epoch", how="left")
            monitor_k = int(run_meta.get("monitor_k", 1))
            ndcg_key = f"val/sft/generation_ndcg@{monitor_k}"
            ndcg_by_epoch = {
                int(record.get("epoch", -1)): float(record["val"][ndcg_key])
                for record in run_meta["history"]
                if ndcg_key in record.get("val", {})
            }
            sampling_frame["val_ndcg"] = sampling_frame["epoch"].map(ndcg_by_epoch)
            sampling_frame.to_csv(temp_dir / "sampling_by_epoch.csv", index=False)

            epochs = sorted(recorder.per_epoch_user_data)
            source_label = "actual training draws"
            plot_sampling_diversity(
                sampling_frame, switch_epoch, temp_dir / f"sampling_diversity_{args.dataset}.png",
                args.dataset, source_label,
            )
            plot_example_users(
                example_infos, recorder.per_epoch_user_data, epochs, switch_epoch,
                temp_dir / f"example_users_sampling_{args.dataset}.png", args.dataset, source_label,
            )
            plot_collapse_dynamics(
                sampling_frame, switch_epoch, temp_dir / f"collapse_dynamics_{args.dataset}.png", args.dataset,
                ndcg_label=f"val NDCG@{monitor_k}",
            )

            rec_results: dict[str, dict[str, Any]] = {}
            collapsed_table = correlations = overlaps = None
            if not args.skip_collapse:
                rec_results, collapsed_table, correlations, overlaps = run_collapse_analysis(
                    args=args, cfgs=cfgs, dm=dm, sasrec=sasrec,
                    item_tokens=item_tokens, row_to_item=row_to_item,
                    run_dir=run_dir, neg_counts_hard=recorder.hard_neg_counts,
                    probe_batches=probe_batches, device=device, out_dir=temp_dir, dataset=args.dataset,
                    cached_rec_results=shared_recs,
                )
                # Baseline/teacher generations do not depend on the temperature;
                # compute once and reuse across the sweep.
                for tag in ("sft_baseline", "sasrec_teacher"):
                    if tag in rec_results:
                        shared_recs.setdefault(tag, rec_results[tag])
                for tag in ("finetuned_best", "finetuned_last"):
                    if tag not in rec_results:
                        continue
                    res = rec_results[tag]
                    sweep_rows.append(
                        {
                            "temperature": temperature,
                            "model": tag,
                            "best_epoch": run_meta.get("best_epoch"),
                            "best_val_ndcg": run_meta.get("best_metric"),
                            f"HR@{args.rec_topk}": res[f"HR@{args.rec_topk}"],
                            f"NDCG@{args.rec_topk}": res[f"NDCG@{args.rec_topk}"],
                            f"coverage@{args.rec_topk}": res[f"coverage@{args.rec_topk}"],
                            "norm_entropy": res["norm_entropy"],
                            "effective_items": res["effective_items"],
                            "top10_item_share": res["top10_item_share"],
                            "run_dir": str(temp_dir),
                        }
                    )

            write_report(
                temp_dir,
                dataset=args.dataset,
                mode=args.mode,
                run_meta=run_meta,
                switch_epoch=switch_epoch,
                sampling_frame=sampling_frame,
                rec_results=rec_results or None,
                collapsed_table=collapsed_table,
                correlations=correlations,
                overlaps=overlaps,
                top_k=args.rec_topk,
            )

        if sweep and sweep_rows:
            pd.DataFrame(sweep_rows).to_csv(out_dir / "temperature_sweep.csv", index=False)
            monitor_k = int(args.val_topk) if args.val_topk else 1
            write_sweep_summary(out_dir, sweep_rows, shared_recs, args.rec_topk, args.dataset, monitor_k)
            plot_temperature_frontier(
                sweep_rows,
                shared_recs.get("sft_baseline"),
                shared_recs.get("sasrec_teacher"),
                args.rec_topk,
                out_dir / f"temperature_frontier_{args.dataset}.png",
                args.dataset,
            )
        print(f"\n[main] done. Output: {out_dir}", flush=True)
        return 0

    else:  # replay
        if len(temperatures) > 1:
            print("[warn] replay mode uses only the first temperature", flush=True)
        _, base_reward = make_base_reward(temperatures[0])
        run_dir = discover_run_dir(args.dataset, args.run_dir)
        if run_dir is None:
            print("[warn] no fine-tuned sasrec run found; collapse analysis limited to the SFT baseline", flush=True)
        else:
            run_meta = load_run_metadata(run_dir)
            print(f"[main] analyzing run: {run_dir}", flush=True)
            if run_meta.get("reward") not in (None, "sasrec"):
                print(f"[warn] run reward is {run_meta.get('reward')!r}, expected 'sasrec'", flush=True)

        easy_steps = int(
            args.easy_negative_steps
            if args.easy_negative_steps is not None
            else run_meta.get("easy_negative_steps", 2000)
        )
        n_epochs = int(args.epochs if args.epochs is not None else run_meta.get("epochs_completed", 10))
        switch_epoch = easy_steps / steps_per_epoch
        print(f"[main] steps/epoch={steps_per_epoch} -> curriculum switch at epoch {switch_epoch:.2f}", flush=True)

        probe_batches = build_probe_batches(dm, max(args.sample_batches, args.pct_batches), seed)
        collator = dm._make_collator()
        example_batch = move_batch(
            collator([dm.train_dataset[info["example_index"]] for info in example_infos]), device
        )
        sampler = build_replay_sampler(
            sasrec, item_tokens, item_to_row,
            num_negatives=int(run_meta.get("num_negative_samples", 3)),
            easy_negative_steps=easy_steps,
            num_random_negatives=int(run_meta.get("num_random_negatives", 0)),
            include_positive=bool(run_meta.get("include_positive", True)),
        )
        epochs = pick_epochs(n_epochs, args.max_analyzed_epochs, switch_epoch)
        print(f"[sampling] analyzing epochs: {[e + 1 for e in epochs]}", flush=True)

        rows = []
        per_epoch_user_data: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        for e in epochs:
            step = int((e + 0.5) * steps_per_epoch)
            stats, counts = replay_epoch(
                sampler, base_reward, sasrec, probe_batches[: args.sample_batches],
                device=device, step=step, item_vocab_size=vocab, num_items=num_items,
            )
            phase = "easy" if (e + 0.5) < switch_epoch else "hard"
            if phase == "hard":
                neg_counts_hard += counts
            rows.append({"epoch": e, "phase": phase, **stats})
            per_epoch_user_data[e] = replay_example_users(
                sampler, base_reward, sasrec, example_batch,
                step=step, draws=args.example_draws, n_users=len(example_infos),
            )
            print(
                f"[sampling] epoch {e + 1} ({phase}): unique_ratio={stats['unique_ratio']:.3f} "
                f"coverage={stats['catalog_coverage']:.3f} top1%={stats['top1pct_share']:.3f} "
                f"reward_mean={stats['reward_mean']:.3f} pct={stats['pct_mean']:.3f}",
                flush=True,
            )

        sampling_frame = pd.DataFrame(rows)
        sampling_frame.to_csv(out_dir / "sampling_by_epoch.csv", index=False)
        source_label = "sampler replay over probe contexts"
        plot_sampling_diversity(
            sampling_frame, switch_epoch, out_dir / f"sampling_diversity_{args.dataset}.png",
            args.dataset, source_label,
        )
        plot_example_users(
            example_infos, per_epoch_user_data, epochs, switch_epoch,
            out_dir / f"example_users_sampling_{args.dataset}.png", args.dataset, source_label,
        )
        if run_meta.get("history"):
            plot_val_trajectory(
                run_meta["history"], switch_epoch, out_dir / f"val_trajectory_{args.dataset}.png", args.dataset
            )

    # ------------------------------------------------------------------ #
    # Collapse analysis on checkpoints (both modes)
    # ------------------------------------------------------------------ #
    rec_results: dict[str, dict[str, Any]] = {}
    collapsed_table = correlations = overlaps = None
    if not args.skip_collapse:
        rec_results, collapsed_table, correlations, overlaps = run_collapse_analysis(
            args=args, cfgs=cfgs, dm=dm, sasrec=sasrec,
            item_tokens=item_tokens, row_to_item=row_to_item,
            run_dir=run_dir, neg_counts_hard=neg_counts_hard,
            probe_batches=probe_batches, device=device, out_dir=out_dir, dataset=args.dataset,
        )

    write_report(
        out_dir,
        dataset=args.dataset,
        mode=args.mode,
        run_meta=run_meta,
        switch_epoch=switch_epoch,
        sampling_frame=sampling_frame,
        rec_results=rec_results or None,
        collapsed_table=collapsed_table,
        correlations=correlations,
        overlaps=overlaps,
        top_k=args.rec_topk,
    )
    print(f"[main] done. Report: {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
