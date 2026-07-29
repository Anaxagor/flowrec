#!/usr/bin/env python
"""Reward ablation for GFlowNet (trajectory-balance) fine-tuning of TIGER.

For each Amazon dataset (beauty / instruments) and each reward model
(amazon_log, sasrec, similarity) this script:

1. Loads the SFT-trained TIGER checkpoint from
   ``artifacts/tiger_sft_amazon/<dataset>/checkpoints/best.pt``.
2. Reuses the cached semantic IDs from
   ``artifacts/tiger_sft_amazon/semantic_ids/<dataset>/semantic_ids.json``.
3. Fine-tunes with SFT loss + trajectory-balance loss; optimizer / trainer /
   model / data parameters come from the YAML files in
   ``configs/exp1-standard_params/``.
4. logZ is NOT learned: for every training example (user context) it is
   precomputed as ``log(sum of rewards over the full item catalog)`` and looked
   up by example index during training.
5. Negatives are drawn with the SASRec-signal curriculum sampler
   (easy negatives for the first ``--easy-negative-steps`` steps, then hard
   negatives), always 3 negatives per context. The tuned SASRec teacher comes
   from ``checkpoints/exp2-tuned-params/sasrec/<dataset>``.
6. Saves reward-distribution plots (full-catalog reward histograms and sorted
   top-k reward curves) for a few example users, per-reward training curves,
   and a cross-reward summary table.

Run (from the repo root, torch env):
    python scripts/rewards_ablation.py --dataset all
Smoke test:
    python scripts/rewards_ablation.py --dataset beauty --rewards similarity \
        --max-epochs 1 --max-train-batches 8 --max-eval-batches 3 --skip-baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowrec.candidates.item_model import (
    ItemModelSemanticCandidateConfig,
    ItemModelSemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.models.tiger import TigerModel
from flowrec.rewards.amazon_log import (
    AmazonLogRewardConfig,
    AmazonLogRewardModel,
    compute_item_mean_ratings,
)
from flowrec.rewards.mix import MixedRewardModel, MultiplicativeRewardModel
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
from flowrec.rewards.similarity import (
    SimilarityRewardConfig,
    SimilarityRewardModel,
    build_item_embedding_table,
    load_rqvae_codebooks,
)
from flowrec.trajectories.semantic import SemanticTrajectoryBatch
from flowrec.training import Trainer, TrainerConfig


# Raw interaction files, matching what the SFT checkpoints were trained on
# (scripts/tiger_for_small_hard.py).
AMAZON_DATASETS = {
    "beauty": {
        "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        "item_key": "asin",
    },
    "instruments": {
        "reviews_path": "data/raw/dataframe_format/reviews_instruments_2018.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_instruments_2018.parquet",
        "item_key": "asin",
    },
}

# Reward keys, distilled from the analysis in examples/amazon_reward_analysis.ipynb:
# individual rewards keep only their KEY mode, plus the multiplicative
# combinations. Reward temperature is fixed at 1 everywhere (not ablated).
REWARD_CHOICES = (
    "sasrec",              # softplus, T=1 - dense collaborative prior
    "amazon_log_ic",       # item_centered logged-evidence modulator (control)
    "sim_exact",           # positionwise token match (legacy similarity baseline)
    "sim_hier",            # hierarchical prefix-gated similarity, equal weights
    "sim_embed",           # content-embedding cosine similarity
    "sasrec_x_ic",         # sasrec x item_centered
    "sasrec_x_hier",       # sasrec x hierarchical similarity
    "sasrec_x_ic_x_hier",  # sasrec x item_centered x hierarchical similarity
    # additive combinations: UNNORMALIZED weighted sums (normalizers="none").
    # The batch_max-normalized sum used by reward.name="mix" in run_tiger is
    # candidate-set dependent and therefore incompatible with this script's
    # precomputed full-catalog logZ; the raw sum of pointwise components is a
    # fixed terminal reward and TB-safe.
    "sasrec_plus_ic",
    "sasrec_plus_hier",
    "sasrec_plus_ic_plus_hier",
    # legacy single-reward names, kept runnable for comparison with old runs
    "amazon_log",
    "similarity",
)
DEFAULT_REWARDS = REWARD_CHOICES[:11]

# Fixed categorical colors per reward (dataviz reference palette, light mode).
REWARD_COLORS = {
    "sasrec": "#2a78d6",              # blue
    "amazon_log_ic": "#008300",       # green
    "sim_exact": "#e87ba4",           # magenta
    "sim_hier": "#eda100",            # yellow
    "sim_embed": "#1baf7a",           # aqua
    "sasrec_x_ic": "#eb6834",         # orange
    "sasrec_x_hier": "#4a3aa7",       # violet
    "sasrec_x_ic_x_hier": "#e34948",  # red
    "sasrec_plus_ic": "#d95926",
    "sasrec_plus_hier": "#9085e9",
    "sasrec_plus_ic_plus_hier": "#e66767",
    "amazon_log": "#006300",
    "similarity": "#e87ba4",
}
USER_COLORS = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
CHROME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def path_from_repo(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_exp_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "trainer": load_yaml(config_dir / "trainer_tiger.yaml"),
        "tiger": load_yaml(config_dir / "tiger_model.yaml"),
        "dataset": load_yaml(config_dir / "dataset_amazon.yaml"),
        "semantic": load_yaml(config_dir / "semantic_ids.yaml"),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_datamodule(
    dataset_name: str,
    dataset_cfg: Mapping[str, Any],
    *,
    semantic_id_path: Path,
    seed: int,
    num_workers: int | None,
) -> AmazonReviewDataModule:
    paths = AMAZON_DATASETS[dataset_name]
    workers = int(dataset_cfg.get("num_workers", 0)) if num_workers is None else int(num_workers)
    dm = AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=path_from_repo(paths["reviews_path"]),
            metadata_path=path_from_repo(paths["metadata_path"]),
            # Filtering interactions to the semantic-ID catalog BEFORE k-core /
            # ID factorization reproduces the item catalog (and the
            # first-occurrence item IDs) the SFT TIGER and SASRec checkpoints
            # were trained on.
            semantic_id_path=semantic_id_path,
            drop_items_missing_semantic_id=True,
            item_key=paths["item_key"],
            split_strategy=str(dataset_cfg.get("split_strategy", "leave_one_out")),
            train_all_prefixes=bool(dataset_cfg.get("train_all_prefixes", True)),
            min_user_interactions=int(dataset_cfg.get("min_user_interactions", 5)),
            min_item_interactions=int(dataset_cfg.get("min_item_interactions", 5)),
            max_history_len=int(dataset_cfg.get("max_history_len", 20)),
            deduplicate_consecutive=bool(dataset_cfg.get("deduplicate_consecutive", False)),
            filter_cold_items_after_split=bool(dataset_cfg.get("filter_cold_items_after_split", False)),
            batch_size=int(dataset_cfg.get("batch_size", 128)),
            eval_batch_size=int(dataset_cfg.get("eval_batch_size", 64)),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            offset_semantic_ids_by_one=bool(dataset_cfg.get("offset_semantic_ids_by_one", True)),
            seed=seed,
        )
    )
    dm.setup()
    return dm


class IndexedDataset(Dataset):
    """Wrap a SequenceDataset so every record carries its dataset index.

    The index is what links a training example to its precomputed logZ entry.
    """

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


def indexed_train_loader(
    dm: AmazonReviewDataModule, *, shuffle: bool, batch_size: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed) if seed is not None else dm.cfg.seed)
    return DataLoader(
        IndexedDataset(dm.train_dataset),
        batch_size=int(batch_size or dm.cfg.batch_size),
        shuffle=shuffle,
        num_workers=dm.cfg.num_workers,
        pin_memory=dm.cfg.pin_memory,
        persistent_workers=dm.cfg.persistent_workers and dm.cfg.num_workers > 0,
        drop_last=False,
        collate_fn=IndexedCollator(dm._make_collator()),
        generator=generator if shuffle else None,
    )


def semantic_item_tokens(dm: AmazonReviewDataModule) -> torch.Tensor:
    """[num_rows, sid_len] token table, rows sorted by raw item id (matches TigerModel)."""
    if not dm.semantic_ids:
        raise ValueError("Datamodule has no semantic IDs attached.")
    ordered = sorted(dm.semantic_ids)
    return torch.tensor([dm.semantic_ids[item] for item in ordered], dtype=torch.long)


def semantic_row_to_item_id(dm: AmazonReviewDataModule) -> torch.Tensor:
    """Map semantic table row -> internal atomic item id (-1 when unmapped)."""
    return torch.tensor(
        [int(dm.item2id.get(raw, -1)) for raw in sorted(dm.semantic_ids or {})],
        dtype=torch.long,
    )


def item_id_to_semantic_row(dm: AmazonReviewDataModule) -> torch.Tensor:
    out = torch.full((int(dm.item_vocab_size),), -1, dtype=torch.long)
    for row, raw in enumerate(sorted(dm.semantic_ids or {})):
        item_id = dm.item2id.get(raw)
        if item_id is not None and 0 <= int(item_id) < out.numel():
            out[int(item_id)] = int(row)
    return out


def restrict_mapping_to_scorer_vocab(mapping: torch.Tensor, scorer: SASRec) -> torch.Tensor:
    limit = int(scorer.item_vocab_size)
    if limit >= mapping.numel():
        return mapping
    restricted = mapping.clone()
    restricted[max(limit, 0):] = -1
    return restricted


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_tiger_model(dm: AmazonReviewDataModule, tiger_cfg: Mapping[str, Any]) -> TigerModel:
    semantic_id_mapping = {
        item: dm.semantic_ids[item]
        for item in dm.id2item[1:]
        if dm.semantic_ids is not None and item in dm.semantic_ids
    }
    ks = [int(k) for k in tiger_cfg.get("ks", [5, 10, 20])]
    model = TigerModel(
        codebook_size=int(tiger_cfg.get("codebook_size", 256)),
        sem_id_len=int(dm.sid_len),
        user_ids_count=int(tiger_cfg.get("user_ids_count", 2000)),
        semantic_id_mapping=semantic_id_mapping,
        embedding_dim=int(tiger_cfg.get("embedding_dim", 128)),
        num_heads=int(tiger_cfg.get("num_heads", 6)),
        num_encoder_layers=int(tiger_cfg.get("num_encoder_layers", 4)),
        num_decoder_layers=int(tiger_cfg.get("num_decoder_layers", 4)),
        dim_feedforward=int(tiger_cfg.get("dim_feedforward", 512)),
        d_kv=int(tiger_cfg.get("d_kv", 32)),
        dropout=float(tiger_cfg.get("dropout", 0.1)),
        num_beams=int(tiger_cfg.get("eval_num_beams", 20)),
        num_return_sequences=max(ks),
    )
    if bool(tiger_cfg.get("gradient_checkpointing", True)):
        model.model.gradient_checkpointing_enable()
    return model


def load_sft_checkpoint(model: TigerModel, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    saved_cfg = payload.get("model_config", {}) if isinstance(payload, Mapping) else {}
    current_cfg = model.config_dict()
    skip_keys = {"num_beams", "num_return_sequences"}  # generation-time knobs
    mismatches = {
        key: (saved_cfg[key], current_cfg[key])
        for key in saved_cfg
        if key in current_cfg and key not in skip_keys and saved_cfg[key] != current_cfg[key]
    }
    if mismatches:
        raise ValueError(
            f"TIGER architecture mismatch vs checkpoint {checkpoint_path}: {mismatches}. "
            "Check the tiger_model.yaml / semantic ID configuration."
        )
    model.load_state_dict(state_dict, strict=True)
    print(f"[init] loaded SFT TIGER weights from {checkpoint_path}", flush=True)


def find_sasrec_checkpoint(sasrec_dir: Path) -> Path:
    candidates = sorted(sasrec_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No SASRec checkpoint (*.pt) found in {sasrec_dir}")
    if len(candidates) > 1:
        print(f"[warn] multiple SASRec checkpoints in {sasrec_dir}; using {candidates[0].name}", flush=True)
    return candidates[0]


def load_sasrec_teacher(checkpoint_path: Path, dm: AmazonReviewDataModule) -> SASRec:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_config" not in payload:
        raise ValueError(f"SASRec checkpoint {checkpoint_path} lacks model_config.")
    model = SASRec(SASRecConfig(**payload["model_config"]))
    model.pad_item_id = 0
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if int(model.item_vocab_size) != int(dm.item_vocab_size):
        raise ValueError(
            f"SASRec item vocab ({model.item_vocab_size}) does not match the datamodule "
            f"({dm.item_vocab_size}). The teacher checkpoint was trained on a different "
            "item catalog; rewards/sampling would be misaligned."
        )
    print(
        f"[init] loaded SASRec teacher from {checkpoint_path} "
        f"(d_model={model.d_model}, layers={model.num_layers}, vocab={model.item_vocab_size})",
        flush=True,
    )
    return model


@torch.no_grad()
def sasrec_alignment_check(sasrec: SASRec, dm: AmazonReviewDataModule, device: torch.device) -> None:
    """Sanity-check that internal item IDs line up with the SASRec checkpoint.

    Item IDs are assigned by first occurrence, so any drift in the interaction
    set would silently scramble the teacher's embedding rows. A well-trained
    teacher must rank the held-out target far better than random.
    """
    batch = move_batch(next(iter(dm.val_dataloader())), device)
    scores = sasrec.score_items(batch).float()
    target = batch["target_item_ids"].long().clamp_max(scores.size(1) - 1)
    target_scores = scores.gather(1, target.unsqueeze(1))
    ranks = scores.gt(target_scores).sum(dim=1).float() + 1.0
    median_rank = float(ranks.median())
    vocab = float(scores.size(1))
    print(
        f"[check] SASRec alignment: median target rank {median_rank:,.0f} of {vocab:,.0f} "
        f"(random would be ~{vocab / 2:,.0f})",
        flush=True,
    )
    if median_rank > 0.25 * vocab:
        raise ValueError(
            "SASRec teacher ranks held-out targets no better than random - the internal "
            "item-ID mapping very likely does not match the checkpoint's training catalog."
        )


# SASRec-score transforms usable with trajectory balance + precomputed logZ.
# minmax_batch / rank_percentile depend on the candidate set and would make the
# live reward inconsistent with the full-catalog logZ, so they are rejected.
TB_SAFE_SASREC_TRANSFORMS = ("exp", "sigmoid", "identity", "softplus")


def reward_base_name(name: str) -> str:
    """Map cache keys such as 'sasrec_exp_T0.5' back to the base computation."""
    if name.startswith(("sasrec_x", "sasrec_plus")):
        return name  # reward combinations are their own computation
    return "sasrec" if name.startswith("sasrec") else name


def sasrec_reward_key(transform: str, temperature: float) -> str:
    """Cache/run key for the sasrec reward; default transform keeps the legacy name."""
    if transform == "softplus" and abs(float(temperature) - 1.0) < 1e-12:
        return "sasrec"
    return f"sasrec_{transform}_T{float(temperature):g}"


def build_reward_model(
    name: str,
    *,
    dm: AmazonReviewDataModule,
    sasrec: SASRec,
    row_to_item: torch.Tensor,
    rqvae_model_path: Path,
    embeddings_path: Path,
    sasrec_transform: str = "softplus",
    sasrec_temperature: float = 1.0,
) -> torch.nn.Module:
    def _sasrec_model() -> torch.nn.Module:
        if sasrec_transform not in TB_SAFE_SASREC_TRANSFORMS:
            raise ValueError(
                f"sasrec reward transform {sasrec_transform!r} is candidate-set dependent and "
                f"inconsistent with precomputed logZ. Use one of {TB_SAFE_SASREC_TRANSFORMS}."
            )
        return SASRecRewardModel(
            sasrec,
            SASRecRewardConfig(
                transform=sasrec_transform,
                temperature=float(sasrec_temperature),
            ),
            item_id_by_semantic_row=row_to_item,
        )

    def _item_centered_model() -> torch.nn.Module:
        item_means, global_mean = compute_item_mean_ratings(
            torch.tensor(dm.interactions["item_id"].to_numpy()),
            torch.tensor(dm.interactions["rating"].to_numpy(dtype="float32")),
            num_items=int(dm.item_vocab_size),
            shrinkage=10.0,
        )
        return AmazonLogRewardModel(
            AmazonLogRewardConfig(reward_mode="item_centered"),
            item_mean_rating=item_means,
            global_mean_rating=global_mean,
            item_id_by_semantic_row=row_to_item,
        )

    def _hier_model() -> torch.nn.Module:
        # Equal level weights: under prefix gating, deep matches are rare and
        # strong evidence (notebook section 5.5).
        return SimilarityRewardModel(
            SimilarityRewardConfig(mode="hierarchical"),
            codebooks=load_rqvae_codebooks(rqvae_model_path),
        )

    def _embed_model() -> torch.nn.Module:
        return SimilarityRewardModel(
            SimilarityRewardConfig(mode="embedding_cosine"),
            item_embeddings=build_item_embedding_table(
                embeddings_path, dm.item2id, int(dm.item_vocab_size)
            ),
            item_id_by_semantic_row=row_to_item,
        )

    base = reward_base_name(name)
    if base == "amazon_log":
        return AmazonLogRewardModel(AmazonLogRewardConfig(), item_id_by_semantic_row=row_to_item)
    if base == "amazon_log_ic":
        return _item_centered_model()
    if base == "sasrec":
        return _sasrec_model()
    if base in {"similarity", "sim_exact"}:
        return SimilarityRewardModel(SimilarityRewardConfig())
    if base == "sim_hier":
        return _hier_model()
    if base == "sim_embed":
        return _embed_model()
    if base == "sasrec_x_ic":
        return MultiplicativeRewardModel([_sasrec_model(), _item_centered_model()])
    if base == "sasrec_x_hier":
        return MultiplicativeRewardModel([_sasrec_model(), _hier_model()])
    if base == "sasrec_x_ic_x_hier":
        return MultiplicativeRewardModel(
            [_sasrec_model(), _item_centered_model(), _hier_model()]
        )

    def _plain_sum(components: list[torch.nn.Module]) -> torch.nn.Module:
        model = MixedRewardModel(components, normalizers=["none"] * len(components))
        assert not model.uses_candidate_set_normalization, (
            "sum combos must stay pointwise to be consistent with precomputed logZ"
        )
        return model

    if base == "sasrec_plus_ic":
        return _plain_sum([_sasrec_model(), _item_centered_model()])
    if base == "sasrec_plus_hier":
        return _plain_sum([_sasrec_model(), _hier_model()])
    if base == "sasrec_plus_ic_plus_hier":
        return _plain_sum([_sasrec_model(), _item_centered_model(), _hier_model()])
    raise ValueError(f"Unknown reward {name!r}. Expected one of {REWARD_CHOICES}.")


@torch.no_grad()
def live_catalog_rewards(
    reward_model: torch.nn.Module,
    batch: Batch,
    *,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    chunk_items: int = 4096,
) -> torch.Tensor:
    """Full-catalog rewards by running the LIVE reward module on chunked
    trajectories - guarantees the precomputed logZ is consistent with the
    per-candidate rewards inside the TB loss, for any reward composition."""
    device = item_tokens.device
    num_rows, sid_len = item_tokens.shape
    bsz = int(batch["target_item_ids"].size(0))
    target_items = batch["target_item_ids"].to(device).long().reshape(bsz, 1)
    mapping = row_to_item.to(device).long()
    parts = []
    for start in range(0, num_rows, chunk_items):
        stop = min(start + chunk_items, num_rows)
        n = stop - start
        cand_items = mapping[start:stop].unsqueeze(0).expand(bsz, n)
        traj = SemanticTrajectoryBatch(
            candidate_token_ids=item_tokens[start:stop].unsqueeze(0).expand(bsz, n, sid_len),
            candidate_token_mask=torch.ones((bsz, n, sid_len), dtype=torch.bool, device=device),
            is_positive=cand_items.eq(target_items),
            extra={"candidate_item_ids": cand_items},
        )
        parts.append(reward_model(batch, traj).float())
    return torch.cat(parts, dim=1)


# --------------------------------------------------------------------------- #
# Full-catalog rewards and logZ precompute
# --------------------------------------------------------------------------- #
@torch.no_grad()
def full_catalog_rewards(
    name: str,
    reward_model: torch.nn.Module,
    batch: Batch,
    *,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    chunk_items: int = 8192,
) -> torch.Tensor:
    """Rewards over the whole semantic catalog, [B, num_rows].

    Each branch mirrors the corresponding live reward module exactly, so the
    precomputed logZ = log(sum of rewards) is consistent with the per-candidate
    rewards used inside the TB loss.
    """
    device = item_tokens.device
    num_rows, sid_len = item_tokens.shape
    name = reward_base_name(name)

    if name == "sasrec":
        eps = float(reward_model.cfg.reward_epsilon)
        reward_model.sasrec_model.eval()
        raw = reward_model.sasrec_model.score_items(batch).float()  # [B, item_vocab]
        mapping = row_to_item.to(device).long()
        valid = mapping.ge(0) & mapping.lt(raw.size(1))
        safe = mapping.clamp_min(0).clamp_max(max(raw.size(1) - 1, 0))
        scores = raw[:, safe]
        rewards = reward_model.transform_scores(scores)
        if (~valid).any():
            rewards[:, ~valid] = eps
        return rewards.clamp_min(eps)

    if name == "amazon_log":
        cfg = reward_model.cfg
        eps = float(cfg.reward_epsilon)
        bsz = int(batch["target_item_ids"].size(0))

        rewards = torch.full((bsz, num_rows), float(cfg.unobserved_reward), device=device)

        # History-observed items: aggregate base interaction values per catalog row.
        hist_rows = batch["history_semantic_row_ids"].to(device).long()  # [B, H]
        hist_mask = batch["history_mask"].to(device).bool() & hist_rows.ge(0) & hist_rows.lt(num_rows)
        hist_ratings = batch["history_ratings"].to(device).float()
        hist_verified = batch.get("history_verified_purchase")
        base = reward_model._base_interaction_value(
            rating=hist_ratings,
            verified=hist_verified.to(device) if torch.is_tensor(hist_verified) else None,
        )  # [B, H]
        base = base * hist_mask.float()
        safe_rows = hist_rows.clamp_min(0)
        sums = torch.zeros((bsz, num_rows), device=device).scatter_add_(1, safe_rows, base)
        counts = torch.zeros((bsz, num_rows), device=device).scatter_add_(1, safe_rows, hist_mask.float())
        if cfg.history_aggregation == "mean":
            aggregated = sums / counts.clamp_min(1.0)
        elif cfg.history_aggregation == "sum":
            aggregated = sums
        else:  # max: fall back to per-slot maximum via scatter_reduce
            aggregated = torch.zeros((bsz, num_rows), device=device).scatter_reduce_(
                1, safe_rows, base, reduce="amax", include_self=True
            )
        observed = counts.gt(0)
        hist_reward = reward_model._apply_transform(aggregated)
        rewards = torch.where(observed, hist_reward, rewards)

        # Target (positive) override.
        target_items = batch["target_item_ids"].to(device).long()
        mapping = item_to_row.to(device)
        valid_t = target_items.ge(0) & target_items.lt(mapping.numel())
        target_rows = mapping[target_items.clamp_min(0).clamp_max(mapping.numel() - 1)]
        valid_t = valid_t & target_rows.ge(0) & target_rows.lt(num_rows)
        target_ratings = batch["target_ratings"].to(device).float().reshape(bsz)
        target_verified = batch.get("target_verified_purchase")
        target_base = reward_model._base_interaction_value(
            rating=target_ratings,
            verified=target_verified.to(device).reshape(bsz) if torch.is_tensor(target_verified) else None,
        )
        target_reward = reward_model._apply_transform(target_base)
        b_idx = torch.arange(bsz, device=device)[valid_t]
        rewards[b_idx, target_rows[valid_t]] = target_reward[valid_t]

        rewards = rewards * float(cfg.interaction_weight)
        return rewards.clamp_min(eps)

    # Everything else (similarity modes, item_centered, multiplicative combos)
    # is evaluated with the live module itself - exact consistency for free.
    return live_catalog_rewards(
        reward_model, batch, item_tokens=item_tokens, row_to_item=row_to_item,
        chunk_items=min(chunk_items, 4096),
    )


def select_example_indices(dm: AmazonReviewDataModule, num_users: int) -> list[dict[str, Any]]:
    """Pick example users spread across history lengths; last train prefix per user."""
    df = dm.train_dataset.examples
    hist_len = df["history_item_ids"].str.len()
    per_user = (
        df.assign(_hist_len=hist_len)
        .sort_values(["user_id", "target_position"])
        .groupby("user_id", sort=True)
        .tail(1)
    )
    per_user = per_user.sort_values("_hist_len", ascending=False)
    n = len(per_user)
    if n == 0:
        return []
    picks = sorted({int(round(q * (n - 1))) for q in np.linspace(0.0, 0.75, num=max(num_users, 1))})
    out = []
    for pos in picks[:num_users]:
        row = per_user.iloc[pos]
        out.append(
            {
                "example_index": int(row.name),
                "user_id": int(row["user_id"]),
                "raw_user_id": str(row["raw_user_id"]),
                "history_len": int(row["_hist_len"]),
            }
        )
    return out


@torch.no_grad()
def precompute_logz(
    dm: AmazonReviewDataModule,
    reward_names: list[str],
    reward_models: Mapping[str, torch.nn.Module],
    *,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    device: torch.device,
    batch_size: int,
    example_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    """One pass over the train split computing, per reward, the full-catalog
    reward sum of every training example. logZ = log(reward_sum)."""
    loader = indexed_train_loader(dm, shuffle=False, batch_size=batch_size)
    n_examples = len(dm.train_dataset)
    example_rows = {info["example_index"]: {} for info in example_infos}

    item_tokens = item_tokens.to(device)
    row_to_item = row_to_item.to(device)
    item_to_row = item_to_row.to(device)
    for model in reward_models.values():
        model.to(device)

    stats = {
        name: {
            "reward_sum": torch.empty(n_examples),
            "target_reward": torch.empty(n_examples),
            "target_rank": torch.empty(n_examples, dtype=torch.long),
            "entropy": torch.empty(n_examples),
        }
        for name in reward_names
    }

    start_time = time.time()
    processed = 0
    for batch_idx, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        idx = batch["example_indices"].cpu()

        target_items = batch["target_item_ids"].long()
        safe_t = target_items.clamp_min(0).clamp_max(item_to_row.numel() - 1)
        target_rows = item_to_row[safe_t]
        valid_t = target_items.ge(0) & target_items.lt(item_to_row.numel()) & target_rows.ge(0)

        for name in reward_names:
            rewards = full_catalog_rewards(
                name,
                reward_models[name],
                batch,
                item_tokens=item_tokens,
                row_to_item=row_to_item,
                item_to_row=item_to_row,
            ).float()
            reward_sum = rewards.sum(dim=1).clamp_min(1e-12)
            log_r = rewards.clamp_min(1e-12).log()
            entropy = reward_sum.log() - (rewards * log_r).sum(dim=1) / reward_sum

            gathered = torch.gather(rewards, 1, target_rows.clamp_min(0).unsqueeze(1)).squeeze(1)
            target_reward = torch.where(valid_t, gathered, torch.zeros_like(gathered))
            target_rank = rewards.gt(target_reward.unsqueeze(1)).sum(dim=1).long() + 1
            target_rank = torch.where(valid_t, target_rank, torch.full_like(target_rank, -1))

            stats[name]["reward_sum"][idx] = reward_sum.cpu()
            stats[name]["target_reward"][idx] = target_reward.cpu()
            stats[name]["target_rank"][idx] = target_rank.cpu()
            stats[name]["entropy"][idx] = entropy.cpu()

            for row_pos, example_index in enumerate(idx.tolist()):
                if example_index in example_rows:
                    example_rows[example_index][name] = rewards[row_pos].cpu()

        processed += int(idx.numel())
        if batch_idx % 20 == 0:
            rate = processed / max(time.time() - start_time, 1e-6)
            print(
                f"[logz] {processed:,}/{n_examples:,} examples ({rate:,.0f} ex/s)",
                flush=True,
            )

    payload: dict[str, Any] = {
        "num_examples": n_examples,
        "rewards": {},
        "example_infos": example_infos,
        "example_rows": example_rows,
    }
    for name in reward_names:
        reward_sum = stats[name]["reward_sum"]
        payload["rewards"][name] = {
            "reward_sum": reward_sum,
            "log_z": reward_sum.clamp_min(1e-12).log(),
            "target_reward": stats[name]["target_reward"],
            "target_rank": stats[name]["target_rank"],
            "entropy": stats[name]["entropy"],
        }
    return payload


def load_or_compute_logz(
    cache_path: Path,
    *,
    recompute: bool,
    dm: AmazonReviewDataModule,
    reward_names: list[str],
    reward_models: Mapping[str, torch.nn.Module],
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    device: torch.device,
    batch_size: int,
    example_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    wanted_examples = {info["example_index"] for info in example_infos}
    existing: dict[str, Any] | None = None
    compute_names = list(reward_names)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if int(payload.get("num_examples", -1)) == len(dm.train_dataset):
            existing = payload
            cached_rows = payload.get("example_rows", {})
            examples_ok = wanted_examples.issubset(set(cached_rows.keys()))
            missing = [name for name in reward_names if name not in payload.get("rewards", {})]
            if examples_ok:
                # The wanted example rows must also carry every requested
                # reward (a merged cache may hold rows from older runs).
                missing = sorted(
                    set(missing)
                    | {
                        name
                        for name in reward_names
                        if any(name not in cached_rows.get(idx, {}) for idx in wanted_examples)
                    }
                )
            if recompute:
                # Recompute the requested rewards from scratch; entries for
                # other rewards in the shared cache file are kept and merged.
                compute_names = list(reward_names)
                print(
                    f"[logz] cache {cache_path}: recomputing {compute_names} "
                    f"(other cached rewards kept: "
                    f"{sorted(set(payload.get('rewards', {})) - set(compute_names))})",
                    flush=True,
                )
            elif examples_ok and not missing:
                print(f"[logz] using cached logZ from {cache_path}", flush=True)
                return payload
            else:
                # Example rows are stored per reward; if new example users are
                # requested, recompute every requested reward so their rows exist.
                compute_names = missing if examples_ok else list(reward_names)
                print(
                    f"[logz] cache {cache_path}: computing {compute_names} "
                    f"(cached rewards kept: {sorted(payload.get('rewards', {}).keys())})",
                    flush=True,
                )
        else:
            print(f"[logz] cache {cache_path} has a different train split; recomputing", flush=True)

    new_payload = precompute_logz(
        dm,
        compute_names,
        {name: reward_models[name] for name in compute_names},
        item_tokens=item_tokens,
        row_to_item=row_to_item,
        item_to_row=item_to_row,
        device=device,
        batch_size=batch_size,
        example_infos=example_infos,
    )
    if existing is not None:
        existing["rewards"].update(new_payload["rewards"])
        example_rows = existing.setdefault("example_rows", {})
        for idx, rows in new_payload["example_rows"].items():
            example_rows.setdefault(idx, {}).update(rows)
        merged_infos = {
            int(info["example_index"]): info
            for info in list(existing.get("example_infos", [])) + list(new_payload["example_infos"])
        }
        existing["example_infos"] = list(merged_infos.values())
        payload = existing
    else:
        payload = new_payload
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[logz] saved logZ cache to {cache_path}", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# Method pieces: precomputed logZ + monitor-friendly SFT metrics
# --------------------------------------------------------------------------- #
class PrecomputedLogZ(torch.nn.Module):
    """LogZ lookup table indexed by ``batch['example_indices']``.

    logZ(context) = log(sum over the full catalog of R(item | context)),
    precomputed offline. It carries no trainable parameters, so the TB loss
    only shapes the policy (TIGER token probabilities). Validation batches have
    no train example indices; they fall back to the mean logZ, which only
    affects diagnostic metrics.
    """

    def __init__(self, log_z_values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("log_z_values", log_z_values.float())
        self.register_buffer("log_z_mean", log_z_values.float().mean())

    def forward(self, batch: Batch) -> torch.Tensor:
        indices = batch.get("example_indices")
        if torch.is_tensor(indices):
            idx = indices.long()
            if int(idx.min()) < 0 or int(idx.max()) >= self.log_z_values.numel():
                raise IndexError(
                    f"example_indices out of range for precomputed logZ table "
                    f"({self.log_z_values.numel()} entries)."
                )
            return self.log_z_values[idx]
        user_ids = batch.get("user_ids")
        if torch.is_tensor(user_ids):
            return self.log_z_mean.expand(user_ids.size(0))
        return self.log_z_mean


class AliasedSFTMethod(SFTMethod):
    """SFT with generation_ndcg@K aliased to ndcg@K so the trainer can monitor
    ``val/sft/ndcg@1`` (same trick as TigerSFTMethod in tiger_for_small_hard)."""

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


def build_gflow_method(
    *,
    reward_model: torch.nn.Module,
    sampler: ItemModelSemanticIDCandidateSampler,
    log_z: torch.Tensor,
    tiger_cfg: Mapping[str, Any],
    gflow_weight: float,
    include_positive: bool = True,
    val_topk: int | None = None,
) -> GFlowMethod:
    # val_topk sets the validation-generation depth (beam K, monitor ndcg@K).
    # Default keeps the legacy setup: beam 1, ndcg@1.
    val_k = int(val_topk) if val_topk else None
    gen_beams = val_k if val_k else int(tiger_cfg.get("val_num_beams", 1))
    gen_topk = (val_k,) if val_k else (1,)
    constrain = bool(tiger_cfg.get("constrain_valid_sequences", True))
    gflow_cfg = GFlowConfig(
        objective="tb",
        sft_weight=1.0,
        gflow_weight=float(gflow_weight),
        logz_context="global",  # placeholder; replaced by PrecomputedLogZ below
        num_negative_samples=3,
        include_positive=bool(include_positive),
        avoid_positive=True,
        reward_epsilon=1e-6,
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        evaluate_generation=True,
        generation_topk=gen_topk,
        generation_num_beams=gen_beams,
        generation_num_return_sequences=gen_beams,
        generation_constrain_valid_sequences=constrain,
    )
    sft_cfg = SFTConfig(
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        compute_item_metrics=False,
        evaluate_generation=True,
        generation_topk=gen_topk,
        generation_num_beams=gen_beams,
        generation_num_return_sequences=gen_beams,
        generation_constrain_valid_sequences=constrain,
    )
    method = GFlowMethod(
        gflow_cfg,
        candidate_sampler=sampler,
        reward_model=reward_model,
        sft_method=AliasedSFTMethod(sft_cfg),
    )
    # Fixed, precomputed logZ (requirement: logZ = sum of all rewards per context).
    method.log_z_model = PrecomputedLogZ(log_z)
    return method


def build_curriculum_sampler(
    *,
    sasrec: SASRec,
    model: TigerModel,
    item_to_row: torch.Tensor,
    easy_negative_steps: int,
    num_random_negatives: int = 0,
    include_positive: bool = True,
) -> ItemModelSemanticIDCandidateSampler:
    return ItemModelSemanticIDCandidateSampler(
        sasrec,
        item_semantic_ids=model.item_semantic_ids.detach().cpu(),
        item_id_to_semantic_row=restrict_mapping_to_scorer_vocab(item_to_row, sasrec),
        cfg=ItemModelSemanticCandidateConfig(
            num_negative_samples=3,
            include_positive=bool(include_positive),
            avoid_positive=True,
            easy_negative_steps=int(easy_negative_steps),
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


# --------------------------------------------------------------------------- #
# Evaluation (beam-search HR/NDCG/Coverage, mirrors tiger_for_small_hard)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_generation(
    model: TigerModel,
    dm: AmazonReviewDataModule,
    *,
    ks: tuple[int, ...],
    num_beams: int,
    constrain_valid_sequences: bool,
    max_batches: int | None = None,
    log_every: int = 50,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    max_k = max(ks)
    catalog = {tuple(map(int, row)) for row in model.item_semantic_ids.detach().cpu().tolist()}

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    coverage = {k: set() for k in ks}
    n = 0

    for batch_idx, batch in enumerate(dm.test_dataloader(), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[eval] processed {batch_idx} test batches", flush=True)
        batch = move_batch(batch, device)
        generated = model.generate(
            batch,
            num_beams=num_beams,
            num_return_sequences=max_k,
            constrain_valid_sequences=constrain_valid_sequences,
        )
        target = batch["target_token_ids"].long()
        mask = batch.get("target_token_mask", torch.ones_like(target, dtype=torch.bool)).bool()

        matches = (generated == target.unsqueeze(1)) | ~mask.unsqueeze(1)
        matches = matches.all(dim=-1)
        n += int(target.size(0))

        generated_cpu = generated.detach().cpu().long()
        for k in ks:
            m = matches[:, :k]
            hit = m.any(dim=1)
            rank = m.float().argmax(dim=1)
            discount = torch.zeros_like(hit, dtype=torch.float32)
            discount[hit] = 1.0 / torch.log2(rank[hit].float() + 2.0)
            hits[k] += float(hit.float().sum().item())
            ndcgs[k] += float(discount.sum().item())
            for row in generated_cpu[:, :k, :].reshape(-1, generated_cpu.size(-1)).tolist():
                key = tuple(map(int, row))
                if key in catalog:
                    coverage[k].add(key)

    out: dict[str, float] = {"num_test_examples": float(n)}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / max(n, 1)
        out[f"NDCG@{k}"] = ndcgs[k] / max(n, 1)
        out[f"Coverage@{k}"] = len(coverage[k]) / max(len(catalog), 1)
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style_axis(ax) -> None:
    ax.set_facecolor(CHROME["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHROME["axis"])
    ax.grid(True, color=CHROME["grid"], linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=CHROME["muted"], labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(CHROME["ink2"])


def plot_reward_distributions(
    logz_payload: Mapping[str, Any],
    reward_names: list[str],
    out_path: Path,
    dataset_name: str,
) -> None:
    example_rows = logz_payload["example_rows"]
    # A merged cache can contain example users from earlier runs whose rows
    # lack newer rewards - plot only users that carry every requested reward.
    infos = [
        info for info in logz_payload["example_infos"]
        if all(name in example_rows.get(info["example_index"], {}) for name in reward_names)
    ]
    if not infos:
        return
    n_users, n_rewards = len(infos), len(reward_names)
    fig, axes = plt.subplots(
        n_users,
        n_rewards,
        figsize=(4.2 * n_rewards, 2.9 * n_users),
        squeeze=False,
    )
    fig.patch.set_facecolor(CHROME["surface"])
    for u, info in enumerate(infos):
        idx = info["example_index"]
        for r, name in enumerate(reward_names):
            ax = axes[u][r]
            _style_axis(ax)
            rewards = example_rows[idx][name].float().numpy()
            log_rewards = np.log10(np.clip(rewards, 1e-12, None))
            ax.hist(
                log_rewards,
                bins=60,
                color=REWARD_COLORS.get(reward_base_name(name), CHROME["ink2"]),
                edgecolor=CHROME["surface"],
                linewidth=0.3,
            )
            ax.set_yscale("log")
            stats = logz_payload["rewards"][name]
            target_reward = float(stats["target_reward"][idx])
            log_z = float(stats["log_z"][idx])
            eff_items = float(np.exp(stats["entropy"][idx]))
            rank = int(stats["target_rank"][idx])
            if target_reward > 0:
                ax.axvline(
                    np.log10(max(target_reward, 1e-12)),
                    color=CHROME["ink"],
                    linestyle="--",
                    linewidth=1.2,
                )
                ax.text(
                    np.log10(max(target_reward, 1e-12)),
                    0.97,
                    " target",
                    transform=ax.get_xaxis_transform(),
                    color=CHROME["ink"],
                    fontsize=7,
                    va="top",
                )
            ax.text(
                0.02,
                0.95,
                f"logZ={log_z:.2f}\neff. items={eff_items:,.0f}\ntarget rank={rank:,}",
                transform=ax.transAxes,
                fontsize=7,
                color=CHROME["ink2"],
                va="top",
            )
            if u == 0:
                ax.set_title(name, fontsize=10, color=CHROME["ink"])
            if r == 0:
                ax.set_ylabel(
                    f"user {info['user_id']}\n(hist={info['history_len']})\nitem count",
                    fontsize=8,
                    color=CHROME["ink2"],
                )
            if u == n_users - 1:
                ax.set_xlabel("reward (log10)", fontsize=8, color=CHROME["ink2"])
    fig.suptitle(
        f"Full-catalog reward distributions per user context — {dataset_name}",
        fontsize=12,
        color=CHROME["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_topk_reward_curves(
    logz_payload: Mapping[str, Any],
    reward_names: list[str],
    out_path: Path,
    dataset_name: str,
    top_k: int = 300,
) -> None:
    example_rows = logz_payload["example_rows"]
    # A merged cache can contain example users from earlier runs whose rows
    # lack newer rewards - plot only users that carry every requested reward.
    infos = [
        info for info in logz_payload["example_infos"]
        if all(name in example_rows.get(info["example_index"], {}) for name in reward_names)
    ]
    if not infos:
        return
    fig, axes = plt.subplots(1, len(reward_names), figsize=(4.6 * len(reward_names), 3.4), squeeze=False)
    fig.patch.set_facecolor(CHROME["surface"])
    for r, name in enumerate(reward_names):
        ax = axes[0][r]
        _style_axis(ax)
        for u, info in enumerate(infos):
            rewards = example_rows[info["example_index"]][name].float().numpy()
            top = np.sort(rewards)[::-1][:top_k]
            ax.plot(
                np.arange(1, top.size + 1),
                np.clip(top, 1e-12, None),
                color=USER_COLORS[u % len(USER_COLORS)],
                linewidth=2.0,
                label=f"user {info['user_id']} (hist={info['history_len']})",
            )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("item rank", fontsize=8, color=CHROME["ink2"])
        if r == 0:
            ax.set_ylabel("reward", fontsize=8, color=CHROME["ink2"])
            ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])
    fig.suptitle(
        f"Sorted top-{top_k} catalog rewards per user context — {dataset_name}",
        fontsize=12,
        color=CHROME["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def plot_training_curves(
    histories: Mapping[str, list[dict[str, Any]]],
    out_path: Path,
    dataset_name: str,
) -> None:
    panels = (
        ("train/loss", "train total loss", False),
        ("train/gflow/tb_loss", "train TB loss", True),
        ("val/sft/generation_ndcg@1", "val NDCG@1 (beam=1)", False),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False)
    fig.patch.set_facecolor(CHROME["surface"])
    for p, (key, title, log_scale) in enumerate(panels):
        ax = axes[0][p]
        _style_axis(ax)
        for name, history in histories.items():
            xs, ys = [], []
            for record in history:
                for section in ("train", "val"):
                    block = record.get(section, {})
                    if key in block:
                        xs.append(int(record.get("epoch", 0)) + 1)
                        ys.append(float(block[key]))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=REWARD_COLORS.get(reward_base_name(name), CHROME["ink2"]),
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                    label=name,
                )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
        if p == 0:
            ax.legend(fontsize=8, frameon=False, labelcolor=CHROME["ink2"])
    fig.suptitle(f"GFlow fine-tuning by reward — {dataset_name}", fontsize=12, color=CHROME["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def make_status_logger(tag: str):
    keys = (
        "train/loss",
        "train/sft/loss",
        "train/gflow/tb_loss",
        "train/gflow/log_z_mean",
        "train/gflow/reward_mean",
        "val/loss",
        "val/sft/ndcg@1",
        "val/sft/ndcg@5",
        "val/sft/ndcg@10",
        "val/sft/generation_coverage@1",
    )

    def logger(payload: dict[str, Any]) -> None:
        event = payload.get("type")
        if event not in {"epoch_end", "val_end", "test_end", "non_finite_loss_skipped"}:
            return
        parts = [f"{k}={payload[k]:.5g}" for k in keys if isinstance(payload.get(k), (int, float))]
        train_block = payload.get("train", {})
        val_block = payload.get("val", {})
        for block in (train_block, val_block):
            if isinstance(block, Mapping):
                parts += [f"{k}={block[k]:.5g}" for k in keys if isinstance(block.get(k), (int, float))]
        epoch = payload.get("epoch")
        header = f"[{tag}] {event}" + (f" epoch={int(epoch) + 1}" if isinstance(epoch, int) else "")
        print(header + (" | " + " | ".join(parts) if parts else ""), flush=True)

    return logger


def run_single(
    *,
    dataset_name: str,
    reward_name: str,
    train_seed: int,
    dm: AmazonReviewDataModule,
    sasrec: SASRec,
    reward_model: torch.nn.Module,
    log_z: torch.Tensor,
    item_to_row: torch.Tensor,
    cfgs: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    trainer_cfg_yaml = cfgs["trainer"]
    tiger_cfg = cfgs["tiger"]
    tag = f"{dataset_name}/{reward_name}/seed{train_seed}"

    model = build_tiger_model(dm, tiger_cfg)
    load_sft_checkpoint(model, sft_checkpoint_path(args, dataset_name))

    sampler = build_curriculum_sampler(
        sasrec=sasrec,
        model=model,
        item_to_row=item_to_row,
        easy_negative_steps=args.easy_negative_steps,
        num_random_negatives=args.num_random_negatives,
        include_positive=not args.tb_exclude_positive,
    )
    method = build_gflow_method(
        reward_model=reward_model,
        sampler=sampler,
        log_z=log_z,
        tiger_cfg=tiger_cfg,
        gflow_weight=args.gflow_weight,
        include_positive=not args.tb_exclude_positive,
        val_topk=args.val_topk,
    )

    device = resolve_device(args, trainer_cfg_yaml)
    max_epochs = int(args.max_epochs if args.max_epochs is not None else trainer_cfg_yaml.get("max_epochs", 500))
    validate_every = int(
        args.validate_every_epochs
        if args.validate_every_epochs is not None
        else trainer_cfg_yaml.get("validate_every_epochs", 5)
    )
    monitor = (
        f"val/sft/ndcg@{int(args.val_topk)}"
        if args.val_topk
        else str(trainer_cfg_yaml.get("monitor", "val/sft/ndcg@1"))
    )
    lr = float(args.lr if args.lr is not None else trainer_cfg_yaml.get("lr", 3e-4))
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=int(trainer_cfg_yaml.get("gradient_accumulation_steps", 4)),
            max_grad_norm=float(trainer_cfg_yaml.get("max_grad_norm", 1.0)),
            use_amp=bool(trainer_cfg_yaml.get("use_amp", True)) and device.type == "cuda",
            amp_dtype=str(trainer_cfg_yaml.get("amp_dtype", "float16")),
            validate_every_epochs=validate_every,
            log_every_steps=int(trainer_cfg_yaml.get("log_every_steps", 200)),
            monitor=monitor,
            monitor_mode=str(trainer_cfg_yaml.get("monitor_mode", "max")),
            early_stopping_patience=trainer_cfg_yaml.get("early_stopping_patience", 10),
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            val_sample_fraction=args.val_sample_fraction,
        ),
        lr=lr,
        weight_decay=float(trainer_cfg_yaml.get("weight_decay", 1e-2)),
        device=device,
        logger=make_status_logger(tag),
    )

    resolved = {
        "dataset": dataset_name,
        "reward": reward_name,
        "seed": int(train_seed),
        "gflow": method.cfg.to_dict(),
        "sampler": {
            "type": "sasrec_curriculum",
            "num_negative_samples": 3,
            "num_random_negatives": int(args.num_random_negatives),
            "include_positive": not args.tb_exclude_positive,
            "easy_negative_steps": int(args.easy_negative_steps),
        },
        "reward_options": {
            "sasrec_transform": str(args.reward_transform),
            "sasrec_temperature": float(args.reward_temperature),
        },
        "logz": {
            "source": "precomputed_full_catalog_reward_sum",
            "mean": float(log_z.mean()),
            "std": float(log_z.std()),
            "min": float(log_z.min()),
            "max": float(log_z.max()),
        },
        "trainer": trainer.cfg.to_dict(),
        "optimizer": {
            "lr": lr,
            "weight_decay": float(trainer_cfg_yaml.get("weight_decay", 1e-2)),
        },
        "sft_checkpoint": str(sft_checkpoint_path(args, dataset_name)),
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] starting fine-tuning: epochs={max_epochs}, device={device}", flush=True)
    fit = trainer.fit(indexed_train_loader(dm, shuffle=True, seed=train_seed), dm.val_dataloader())
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    checkpoint_for_test = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_test:
        trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)

    ks = tuple(int(k) for k in tiger_cfg.get("ks", [5, 10, 20]))
    test_metrics = evaluate_generation(
        model,
        dm,
        ks=ks,
        num_beams=int(tiger_cfg.get("eval_num_beams", 20)),
        constrain_valid_sequences=bool(tiger_cfg.get("constrain_valid_sequences", True)),
        max_batches=args.max_eval_batches,
        log_every=int(trainer_cfg_yaml.get("eval_log_every_batches", 50)),
    )
    write_json(run_dir / "test_metrics.json", {"dataset": dataset_name, "reward": reward_name, **test_metrics})
    print(f"[{tag}] test metrics: {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    return {
        "dataset": dataset_name,
        "reward": reward_name,
        "seed": int(train_seed),
        "status": "completed",
        "best_epoch": fit.state.best_epoch,
        "best_metric": fit.state.best_metric,
        "epochs_completed": fit.state.epoch + 1,
        "steps_completed": fit.state.global_step,
        "run_dir": str(run_dir),
        "history": fit.state.history,
        **test_metrics,
    }


def sft_checkpoint_path(args: argparse.Namespace, dataset_name: str) -> Path:
    return path_from_repo(args.sft_artifacts) / dataset_name / "checkpoints" / "best.pt"


def resolve_device(args: argparse.Namespace, trainer_cfg: Mapping[str, Any]) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    wanted = str(trainer_cfg.get("device", "cuda"))
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(wanted)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
SUMMARY_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "Coverage@5", "Coverage@10", "Coverage@20")


def write_summary(out_dir: Path, records: list[dict[str, Any]]) -> None:
    fields = ["dataset", "reward", "seed", "status", "best_epoch", "best_metric", "epochs_completed", "steps_completed"]
    fields += list(SUMMARY_METRICS) + ["run_dir"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# GFlow reward ablation (fixed logZ = log sum of catalog rewards, sasrec curriculum, 3 negatives)",
        "",
        "| dataset | reward | seed | status | best epoch | " + " | ".join(SUMMARY_METRICS) + " |",
        "| --- | --- | ---: | --- | ---: | " + " | ".join("---:" for _ in SUMMARY_METRICS) + " |",
    ]
    for record in records:
        cells = [
            str(record.get("dataset", "")),
            str(record.get("reward", "")),
            str(record.get("seed", "")),
            str(record.get("status", "")),
            str(record.get("best_epoch", "")),
        ]
        cells += [
            f"{record[m]:.5f}" if isinstance(record.get(m), float) else ""
            for m in SUMMARY_METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # Cross-seed aggregation: mean +/- std per (dataset, reward) over completed runs.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "completed":
            groups.setdefault((str(record.get("dataset")), str(record.get("reward"))), []).append(record)
    agg_rows = []
    for (ds, reward), group in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": ds, "reward": reward, "n_seeds": len(group)}
        for m in SUMMARY_METRICS:
            vals = [float(r[m]) for r in group if isinstance(r.get(m), (int, float))]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        agg_rows.append(row)
    if agg_rows:
        agg_fields = ["dataset", "reward", "n_seeds"]
        for m in SUMMARY_METRICS:
            agg_fields += [m, f"{m}_std"]
        with open(out_dir / "summary_aggregated.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(agg_rows)
        lines += [
            "",
            "## Aggregated over seeds (mean ± std, completed runs only)",
            "",
            "| dataset | reward | seeds | " + " | ".join(SUMMARY_METRICS) + " |",
            "| --- | --- | ---: | " + " | ".join("---:" for _ in SUMMARY_METRICS) + " |",
        ]
        for row in agg_rows:
            cells = [row["dataset"], row["reward"], str(row["n_seeds"])]
            cells += [
                f"{row[m]:.5f} ± {row[f'{m}_std']:.5f}" if isinstance(row.get(m), float) else ""
                for m in SUMMARY_METRICS
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GFlow TB reward ablation for TIGER (amazon_log / sasrec / similarity)")
    p.add_argument("--dataset", choices=["beauty", "instruments", "all"], default="all")
    p.add_argument("--rewards", nargs="+", choices=list(REWARD_CHOICES), default=list(DEFAULT_REWARDS))
    p.add_argument("--config-dir", default="configs/exp1-standard_params")
    p.add_argument("--sft-artifacts", default="artifacts/tiger_sft_amazon")
    p.add_argument("--semantic-dir", default=None, help="Defaults to semantic_ids.yaml semantic_dir")
    p.add_argument("--sasrec-checkpoint-dir", default="checkpoints/exp2-tuned-params/sasrec")
    p.add_argument("--output-dir", default="experiments/rewards_ablation")
    p.add_argument("--logz-cache-dir", default="artifacts/rewards_ablation_logz")
    p.add_argument("--recompute-logz", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None, help="Defaults to trainer.yaml seed")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Training seeds; each reward is fine-tuned once per seed "
                        "(data split / logZ stay fixed). Overrides --seed.")
    p.add_argument("--embeddings-dir", default="artifacts/tiger_sft_amazon/item_embeddings",
                   help="Directory with <dataset>_content_embeddings.pt (sim_embed reward)")
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--validate-every-epochs", type=int, default=5)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--easy-negative-steps", type=int, default=10000,
                   help="Curriculum: easy (low-SASRec-score) negatives for this many steps, then hard")
    p.add_argument("--gflow-weight", type=float, default=0.1)
    p.add_argument("--reward-transform", choices=list(TB_SAFE_SASREC_TRANSFORMS), default="softplus",
                   help="SASRec score -> reward transform. 'exp' with a temperature = teacher-softmax distillation")
    p.add_argument("--reward-temperature", type=float, default=1.0,
                   help="Temperature for the SASRec reward transform (scores are divided by it)")
    p.add_argument("--tb-exclude-positive", action="store_true",
                   help="TB candidates are negatives only; SFT alone owns the target (avoids TB/SFT conflict)")
    p.add_argument("--num-random-negatives", type=int, default=3,
                   help="Of the 3 negatives, keep this many uniform-random (rest follow the curriculum)")
    p.add_argument("--lr", type=float, default=None,
                   help="Fine-tuning learning rate (default: trainer.yaml; try 3e-5 for gentle fine-tuning)")
    p.add_argument("--val-topk", type=int, default=None,
                   help="Validate/monitor ndcg@K with K beams (default: trainer.yaml monitor, beam 1)")
    p.add_argument("--val-sample-fraction", type=float, default=None,
                   help="Randomly subsample this fraction of the val set per validation")
    p.add_argument("--num-example-users", type=int, default=4)
    p.add_argument("--logz-batch-size", type=int, default=512)
    p.add_argument("--skip-training", action="store_true", help="Only precompute logZ and produce plots")
    p.add_argument("--skip-baseline", action="store_true", help="Skip test evaluation of the raw SFT checkpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_dir = path_from_repo(args.config_dir)
    cfgs = load_exp_configs(config_dir)
    seed = int(args.seed if args.seed is not None else cfgs["trainer"].get("seed", 43))
    train_seeds = [int(s) for s in (args.seeds or [seed])]
    set_seed(seed)

    semantic_dir = path_from_repo(
        args.semantic_dir if args.semantic_dir is not None else cfgs["semantic"].get("semantic_dir")
    )
    device = resolve_device(args, cfgs["trainer"])

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # SLURM array tasks can start within the same second - suffix the run dir
    # with the job/task id (or pid) so concurrent invocations never collide.
    import os
    if os.environ.get("SLURM_ARRAY_JOB_ID") and os.environ.get("SLURM_ARRAY_TASK_ID"):
        uniq = f"{os.environ['SLURM_ARRAY_JOB_ID']}-{os.environ['SLURM_ARRAY_TASK_ID']}"
    else:
        uniq = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    out_root = path_from_repo(args.output_dir) / f"{stamp}_{uniq}_rewards_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "cli_args.json", vars(args))
    print(f"[main] output dir: {out_root}", flush=True)

    datasets = ["beauty", "instruments"] if args.dataset == "all" else [args.dataset]
    records: list[dict[str, Any]] = []

    for dataset_name in datasets:
        print(f"\n=== {dataset_name} ===", flush=True)
        dm = build_datamodule(
            dataset_name,
            cfgs["dataset"],
            semantic_id_path=semantic_dir / dataset_name / "semantic_ids.json",
            seed=seed,
            num_workers=args.num_workers,
        )
        print(
            f"[{dataset_name}] users={dm.num_users:,} items={dm.num_items:,} "
            f"train={len(dm.train_dataset):,} val={len(dm.val_dataset):,} "
            f"test={len(dm.test_dataset):,} sid_len={dm.sid_len}",
            flush=True,
        )

        item_tokens = semantic_item_tokens(dm)
        row_to_item = semantic_row_to_item_id(dm)
        item_to_row = item_id_to_semantic_row(dm)

        sasrec = load_sasrec_teacher(
            find_sasrec_checkpoint(path_from_repo(args.sasrec_checkpoint_dir) / dataset_name), dm
        )
        sasrec.to(device)
        sasrec_alignment_check(sasrec, dm, device)

        sasrec_key = sasrec_reward_key(args.reward_transform, args.reward_temperature)
        reward_keys = [sasrec_key if name == "sasrec" else name for name in args.rewards]
        rqvae_model_path = semantic_dir / dataset_name / "model.pt"
        embeddings_path = path_from_repo(args.embeddings_dir) / f"{dataset_name}_content_embeddings.pt"
        reward_models = {
            key: build_reward_model(
                key,
                dm=dm,
                sasrec=sasrec,
                row_to_item=row_to_item,
                rqvae_model_path=rqvae_model_path,
                embeddings_path=embeddings_path,
                sasrec_transform=args.reward_transform,
                sasrec_temperature=args.reward_temperature,
            )
            for key in reward_keys
        }

        example_infos = select_example_indices(dm, args.num_example_users)
        print(f"[{dataset_name}] example users for plots: {example_infos}", flush=True)

        logz_payload = load_or_compute_logz(
            path_from_repo(args.logz_cache_dir) / f"{dataset_name}.pt",
            recompute=args.recompute_logz,
            dm=dm,
            reward_names=reward_keys,
            reward_models=reward_models,
            item_tokens=item_tokens,
            row_to_item=row_to_item,
            item_to_row=item_to_row,
            device=device,
            batch_size=args.logz_batch_size,
            example_infos=example_infos,
        )

        plots_dir = out_root / dataset_name / "plots"
        plot_reward_distributions(
            logz_payload, reward_keys, plots_dir / f"reward_distributions_{dataset_name}.png", dataset_name
        )
        plot_topk_reward_curves(
            logz_payload, reward_keys, plots_dir / f"reward_topk_{dataset_name}.png", dataset_name
        )
        example_stats = {
            str(info["example_index"]): {
                "user_id": info["user_id"],
                "history_len": info["history_len"],
                **{
                    name: {
                        "log_z": float(logz_payload["rewards"][name]["log_z"][info["example_index"]]),
                        "reward_sum": float(logz_payload["rewards"][name]["reward_sum"][info["example_index"]]),
                        "target_reward": float(logz_payload["rewards"][name]["target_reward"][info["example_index"]]),
                        "target_rank": int(logz_payload["rewards"][name]["target_rank"][info["example_index"]]),
                        "effective_items": float(np.exp(logz_payload["rewards"][name]["entropy"][info["example_index"]])),
                    }
                    for name in reward_keys
                    if info["example_index"] in logz_payload["example_rows"]
                },
            }
            for info in example_infos
        }
        write_json(out_root / dataset_name / "example_users.json", example_stats)

        for name in reward_keys:
            block = logz_payload["rewards"][name]
            print(
                f"[{dataset_name}] logZ[{name}]: mean={float(block['log_z'].mean()):.3f} "
                f"std={float(block['log_z'].std()):.3f} "
                f"min={float(block['log_z'].min()):.3f} max={float(block['log_z'].max()):.3f}",
                flush=True,
            )

        if args.skip_training:
            continue

        if not args.skip_baseline:
            baseline_model = build_tiger_model(dm, cfgs["tiger"])
            load_sft_checkpoint(baseline_model, sft_checkpoint_path(args, dataset_name))
            baseline_model.to(device)
            ks = tuple(int(k) for k in cfgs["tiger"].get("ks", [5, 10, 20]))
            baseline_metrics = evaluate_generation(
                baseline_model,
                dm,
                ks=ks,
                num_beams=int(cfgs["tiger"].get("eval_num_beams", 20)),
                constrain_valid_sequences=bool(cfgs["tiger"].get("constrain_valid_sequences", True)),
                max_batches=args.max_eval_batches,
            )
            del baseline_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            records.append(
                {"dataset": dataset_name, "reward": "sft_baseline", "status": "completed", **baseline_metrics}
            )
            write_summary(out_root, records)
            print(f"[{dataset_name}] SFT baseline: {json.dumps(baseline_metrics, sort_keys=True)}", flush=True)

        histories: dict[str, list[dict[str, Any]]] = {}
        for reward_name in reward_keys:
            for train_seed in train_seeds:
                run_dir = out_root / dataset_name / reward_name / f"seed_{train_seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                set_seed(train_seed)
                try:
                    record = run_single(
                        dataset_name=dataset_name,
                        reward_name=reward_name,
                        train_seed=train_seed,
                        dm=dm,
                        sasrec=sasrec,
                        reward_model=reward_models[reward_name],
                        log_z=logz_payload["rewards"][reward_name]["log_z"],
                        item_to_row=item_to_row,
                        cfgs=cfgs,
                        args=args,
                        run_dir=run_dir,
                    )
                except Exception as exc:  # keep the sweep alive; report at the end
                    print(f"[{dataset_name}/{reward_name}/seed{train_seed}] FAILED: {exc!r}", flush=True)
                    record = {
                        "dataset": dataset_name,
                        "reward": reward_name,
                        "seed": train_seed,
                        "status": f"failed: {exc}",
                    }
                history = record.pop("history", [])
                if train_seed == train_seeds[0]:
                    histories[reward_name] = history
                records.append(record)
                write_summary(out_root, records)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if histories:
            plot_training_curves(histories, plots_dir / f"training_curves_{dataset_name}.png", dataset_name)

    write_summary(out_root, records)
    print(f"\n[main] done. Summary: {out_root / 'summary.md'}", flush=True)
    return 0 if all(r.get("status") == "completed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
