#!/usr/bin/env python
"""Exp4: reward ablation for GFlowNet (trajectory-balance) fine-tuning of TIGER.

.. warning::
    NOT RUNNABLE against the current GFlowMethod yet. The reward stack this
    script builds (interaction/positive rewards, level-offset similarity,
    power compositions) is current, but the training wiring still targets the
    pre-"simplify gflow training pipeline" (#43) method API: it passes a live
    ``reward_model`` and a ``PrecomputedLogZ`` to GFlowMethod, which now only
    consumes a precomputed reward cache (``reward_cache_path``). Port the
    GFlow wiring (build the cache via scripts/precompute_reward_cache.py and
    drop the live-reward path) before launching exp4 runs.

Datasets: Amazon Beauty (2014) and Amazon Beauty 2023. For each dataset the
script fine-tunes the exp2 SFT TIGER checkpoint under every requested
(reward composition, reward degree) pair and evaluates on the test split.

Reward compositions (18 total):
  * individual rewards:            sasrec, rating, sim_embed, sim_hier
  * pairwise combinations:         sum ("_plus_") and product ("_x_") of
                                   {sasrec, rating} x {rating, sim_embed, sim_hier}
  * three-reward combinations:     sasrec+rating+sim and sasrec*rating*sim
  where:
  * rating    - per-item shrunk mean-rating multiplier (item-quality prior)
    computed from the FULL raw reviews frame. Under the positive/auxiliary
    datamodule contract the positive stream carries no rating payload (and on
    all-verified dumps such as beauty 2014 the auxiliary store is empty), so
    this is the rating signal that exists identically on every dataset;
  * sim_embed - cosine similarity over the content ("semantic") item
    embeddings from datasets/embeddings/;
  * sim_hier  - hierarchical prefix-gated similarity over the RQ-VAE
    codebook geometry (datasets/semantic_ids/.../model.pt).

Reward degree: the final composed reward is raised to R**degree
(default grid: 1, 0.25, 4) - the GFlowNet reward-sharpness knob.

Method (fixed across all runs):
  * trajectory-balance loss on top of SFT (sft_weight=1, gflow_weight=0.25);
  * logZ is NOT learned - it is computed analytically per training context as
    log(sum of R**degree over the full item catalog) and looked up by example
    index during training;
  * 3 negatives per context from the SASRec-signal curriculum sampler
    (easy negatives for the first --easy-negative-steps steps, then hard).

Inputs (defaults; see parse_args):
  * preprocessing configs:  configs/exp4-reward_ablation/
  * TIGER SFT checkpoints:  checkpoints/exp2/loo_split/initial/tiger/<ds>/best.pt
  * SASRec teachers:        checkpoints/exp2/loo_split/initial/sasrec/<ds>/best.pt
  * semantic IDs + RQ-VAE:  datasets/semantic_ids/exp2-tuned-params/<ds>/
  * content embeddings:     datasets/embeddings/<ds>_content_embeddings.pt

Outputs: per-run training history, checkpoints and test metrics under
--output-dir; test-metric summaries (csv/md/jsonl) mirrored into
--artifacts-dir.

Run (from the repo root, torch env):
    python scripts/exp4-reward-ablations/rewards_ablation_amazon.py --dataset all
Smoke test:
    python scripts/exp4-reward-ablations/rewards_ablation_amazon.py \
        --dataset beauty --rewards sasrec sasrec_x_ic --reward-degrees 1 4 \
        --max-epochs 1 --max-train-batches 4 --max-eval-batches 2 \
        --logz-max-batches 4 --skip-baseline --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowrec.candidates.sampler import (
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.tensor_batches import TensorizedSequenceCollator
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.models.tiger import TigerModel
from flowrec.data.interaction_store import build_positive_history_store
from flowrec.rewards.interaction import (
    InteractionRewardConfig,
    InteractionRewardModel,
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


# Datasets in this experiment and the exp4 preprocessing YAML each comes from.
DATASET_CONFIG_FILES = {
    "beauty": "dataset_amazon.yaml",
    "beauty_2023": "dataset_amazon_2023.yaml",
}

# --------------------------------------------------------------------------- #
# Reward grid: individuals, pairwise sum/product, triple sum/product
# --------------------------------------------------------------------------- #
SIM_VARIANTS = ("sim_embed", "sim_hier")
COMPONENT_CHOICES = ("sasrec", "rating", *SIM_VARIANTS)


def _build_reward_specs() -> dict[str, tuple[str, tuple[str, ...]]]:
    """name -> (op, component keys); op in {single, sum, product}."""
    specs: dict[str, tuple[str, tuple[str, ...]]] = {
        comp: ("single", (comp,)) for comp in COMPONENT_CHOICES
    }
    pair_sets = [("sasrec", "rating")] + [
        (base, sim) for base in ("sasrec", "rating") for sim in SIM_VARIANTS
    ]
    triple_sets = [("sasrec", "rating", sim) for sim in SIM_VARIANTS]
    for comps in [*pair_sets, *triple_sets]:
        specs["_plus_".join(comps)] = ("sum", tuple(comps))
        specs["_x_".join(comps)] = ("product", tuple(comps))
    return specs


REWARD_SPECS = _build_reward_specs()
REWARD_CHOICES = tuple(REWARD_SPECS)
DEFAULT_DEGREES = (1.0, 0.25, 4.0)


def run_key(reward_name: str, degree: float) -> str:
    return f"{reward_name}__deg{degree:g}"


# Fixed categorical palette (dataviz reference palette, light mode); rewards
# cycle through it in grid order.
PALETTE = (
    "#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
    "#4a3aa7", "#e34948", "#d95926", "#9085e9", "#e66767", "#006300",
)
REWARD_COLORS = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(REWARD_CHOICES)}
USER_COLORS = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
CHROME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def reward_color(name: str) -> str:
    return REWARD_COLORS.get(name.split("__deg")[0], CHROME["ink2"])


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


def load_exp_configs(config_dir: Path) -> dict[str, Any]:
    return {
        "trainer": load_yaml(config_dir / "trainer_tiger.yaml"),
        "tiger": load_yaml(config_dir / "tiger_model.yaml"),
        "semantic": load_yaml(config_dir / "semantic_ids.yaml"),
        "datasets": {
            name: load_yaml(config_dir / file_name)
            for name, file_name in DATASET_CONFIG_FILES.items()
        },
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
def build_data_config(
    dataset_name: str,
    dataset_yaml: Mapping[str, Any],
    *,
    seed: int,
    semantic_dir: Path,
    user_hash_mod: int,
    args: argparse.Namespace,
) -> tuple[AmazonReviewDataConfig, int]:
    """Merge shared + per-dataset + tiger-block YAML values into the datamodule
    config (mirrors scripts/exp1-standard-params/run_tiger_sft_all.py, so the
    reproduced item catalog matches the exp2 SFT checkpoints).

    Returns the config plus the runner-only test_eval_batch_size (final
    20-beam evaluation).
    """
    if dataset_name not in dataset_yaml.get("datasets", {}):
        raise KeyError(
            f"{DATASET_CONFIG_FILES[dataset_name]} defines no datasets.{dataset_name} entry."
        )
    entry = dict(dataset_yaml["datasets"][dataset_name])
    entry.pop("sasrec", None)
    tiger_values = dict(dataset_yaml.get("tiger", {}))
    tiger_values.update(dict(entry.pop("tiger", {}) or {}))
    test_eval_batch_size = int(tiger_values.pop("test_eval_batch_size", 64))

    field_names = {field.name for field in fields(AmazonReviewDataConfig)}
    shared = {key: value for key, value in dataset_yaml.items() if key in field_names}
    unknown = sorted(set(entry) - field_names)
    if unknown:
        raise KeyError(f"Unknown AmazonReviewDataConfig dataset keys: {unknown}")
    unknown_tiger = sorted(set(tiger_values) - field_names)
    if unknown_tiger:
        raise KeyError(f"Unknown AmazonReviewDataConfig tiger-block keys: {unknown_tiger}")

    values = {**shared, **entry, **tiger_values}
    values["seed"] = int(seed)
    values["user_hash_mod"] = int(user_hash_mod)
    values["semantic_id_path"] = semantic_dir / dataset_name / "semantic_ids.json"
    if args.num_workers is not None:
        values["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        values["batch_size"] = int(args.batch_size)
    if args.eval_batch_size is not None:
        values["eval_batch_size"] = int(args.eval_batch_size)
    if args.test_eval_batch_size is not None:
        test_eval_batch_size = int(args.test_eval_batch_size)
    values["pin_memory"] = bool(values.get("pin_memory", True)) and torch.cuda.is_available()
    values["persistent_workers"] = (
        bool(values.get("persistent_workers", False)) and int(values.get("num_workers", 0)) > 0
    )
    for key in ("reviews_path", "metadata_path"):
        if key in values and values[key] is not None:
            values[key] = path_from_repo(values[key])
    return AmazonReviewDataConfig(**values), test_eval_batch_size


def build_datamodule(cfg: AmazonReviewDataConfig) -> AmazonReviewDataModule:
    dm = AmazonReviewDataModule(cfg)
    dm.setup()
    return dm


class IndexedDataset(Dataset):
    """Fallback wrapper carrying the dataset index into each record (the index
    links a training example to its precomputed logZ entry). The tensorized
    collator emits ``example_indices`` natively; this covers the rest."""

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
    dm: AmazonReviewDataModule,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    dataset, collate_fn = dm._loader_parts(dm.train_dataset, dm._make_collator())
    if not isinstance(collate_fn, TensorizedSequenceCollator):
        dataset = IndexedDataset(dm.train_dataset)
        collate_fn = IndexedCollator(dm._make_collator())
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed) if seed is not None else int(dm.cfg.seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size or dm.cfg.batch_size),
        shuffle=shuffle,
        num_workers=dm.cfg.num_workers,
        pin_memory=dm.cfg.pin_memory,
        persistent_workers=dm.cfg.persistent_workers and dm.cfg.num_workers > 0,
        drop_last=False,
        collate_fn=collate_fn,
        generator=generator,
    )


@contextmanager
def eval_batch_size(dm: AmazonReviewDataModule, batch_size: int) -> Iterator[None]:
    """Temporarily swap the eval loader batch size (final 20-beam test pass)."""
    original = dm.cfg
    dm.cfg = replace(dm.cfg, eval_batch_size=int(batch_size))
    try:
        yield
    finally:
        dm.cfg = original


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
def sft_checkpoint_path(args: argparse.Namespace, dataset_name: str) -> Path:
    return path_from_repo(args.sft_checkpoint_dir) / dataset_name / "best.pt"


def rqvae_checkpoint_path(semantic_dir: Path, dataset_name: str) -> Path:
    dataset_dir = semantic_dir / dataset_name
    nested = dataset_dir / "rqvae" / "model.pt"
    return nested if nested.is_file() else dataset_dir / "model.pt"


def saved_model_config(checkpoint_path: Path) -> dict[str, Any]:
    """model_config of a checkpoint, read from the .json sidecar when present."""
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".json")
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            return dict(json.load(f).get("model_config", {}))
    payload = torch.load(checkpoint_path, map_location="cpu")
    return dict(payload.get("model_config", {})) if isinstance(payload, Mapping) else {}


def build_tiger_model(
    dm: AmazonReviewDataModule,
    tiger_cfg: Mapping[str, Any],
    saved_cfg: Mapping[str, Any],
) -> TigerModel:
    semantic_id_mapping = {
        item: dm.semantic_ids[item]
        for item in dm.id2item[1:]
        if dm.semantic_ids is not None and item in dm.semantic_ids
    }
    if not semantic_id_mapping:
        raise ValueError("No catalog item has a semantic ID.")
    ks = [int(k) for k in tiger_cfg.get("ks", [5, 10, 20])]
    # user_ids_count comes from the checkpoint: the exp2 SFT models were
    # trained without a user token (user_ids_count=0).
    user_ids_count = int(saved_cfg.get("user_ids_count", tiger_cfg.get("user_ids_count", 0)))
    model = TigerModel(
        codebook_size=int(tiger_cfg.get("codebook_size", 256)),
        sem_id_len=int(dm.sid_len),
        user_ids_count=user_ids_count,
        use_user_token=bool(saved_cfg.get("use_user_token", user_ids_count > 0)),
        semantic_id_mapping=semantic_id_mapping,
        embedding_dim=int(tiger_cfg.get("embedding_dim", 128)),
        num_heads=int(tiger_cfg.get("num_heads", 6)),
        num_encoder_layers=int(tiger_cfg.get("num_encoder_layers", 2)),
        num_decoder_layers=int(tiger_cfg.get("num_decoder_layers", 2)),
        dim_feedforward=int(tiger_cfg.get("dim_feedforward", 512)),
        d_kv=int(tiger_cfg.get("d_kv", 32)),
        dropout=float(tiger_cfg.get("dropout", 0.1)),
        num_beams=int(tiger_cfg.get("eval_num_beams", 20)),
        num_return_sequences=max(ks),
    )
    if bool(tiger_cfg.get("gradient_checkpointing", False)):
        model.model.gradient_checkpointing_enable()
    return model


def build_finetune_model(
    dm: AmazonReviewDataModule,
    tiger_cfg: Mapping[str, Any],
    checkpoint_path: Path,
) -> TigerModel:
    """Fresh TIGER with the exp2 SFT weights loaded.

    The exp2 checkpoints come from a training pipeline whose TigerModel also
    carried a DB/SubTB flow head; those parameters are dropped (TB fine-tuning
    never uses them). Everything the local model owns must be present.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(f"TIGER checkpoint {checkpoint_path} lacks model_state_dict.")
    saved_cfg = dict(payload.get("model_config", {}))
    model = build_tiger_model(dm, tiger_cfg, saved_cfg)

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
            "Check tiger_model.yaml / the semantic-ID configuration."
        )

    state_dict = payload["model_state_dict"]
    own_keys = set(model.state_dict().keys())
    dropped = sorted(k for k in state_dict if k not in own_keys)
    missing = sorted(own_keys - set(state_dict))
    if missing:
        raise ValueError(
            f"TIGER checkpoint {checkpoint_path} is missing parameters: {missing[:8]}..."
        )
    model.load_state_dict({k: v for k, v in state_dict.items() if k in own_keys}, strict=True)
    extra = f" (dropped foreign keys: {dropped})" if dropped else ""
    print(f"[init] loaded SFT TIGER weights from {checkpoint_path}{extra}", flush=True)
    return model


def find_sasrec_checkpoint(sasrec_dir: Path) -> Path:
    """Read the selected primary teacher without searching historical copies."""
    checkpoint = sasrec_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing primary SASRec checkpoint: {checkpoint}")
    return checkpoint


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


# --------------------------------------------------------------------------- #
# Reward components and compositions
# --------------------------------------------------------------------------- #
# SASRec-score transforms usable with trajectory balance + precomputed logZ.
# minmax_batch / rank_percentile depend on the candidate set and would make the
# live reward inconsistent with the full-catalog logZ, so they are rejected.
TB_SAFE_SASREC_TRANSFORMS = ("exp", "sigmoid", "identity", "softplus")


class LevelOffsetSimilarityReward(SimilarityRewardModel):
    """SimilarityRewardModel for level-offset semantic IDs.

    The exp2 tokenizers emit ``token = code + level * codebook_size`` (plus the
    datamodule's +1 shift), while the hierarchical mode expects raw per-level
    codes in [0, codebook_size). Subtracting the level stride restores that
    contract; equality-based levels (exact mode, collision suffix) are
    unaffected because both sides share the same offset.
    """

    def __init__(self, cfg: SimilarityRewardConfig | None = None, *, level_token_stride: int, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self._level_token_stride = int(level_token_stride)

    def _candidate_and_target_tokens(self, batch, trajectories):
        cand, target, mask = super()._candidate_and_target_tokens(batch, trajectories)
        strides = (
            torch.arange(cand.shape[-1], device=cand.device, dtype=cand.dtype)
            * self._level_token_stride
        )
        return cand - strides, target - strides.unsqueeze(0), mask


class ItemPriorRatingReward(torch.nn.Module):
    """Per-item rating-prior multiplier (the exp4 'rating' reward).

    multiplier(i) = clip(exp(beta * (shrunk_mean_rating(i) - global_mean)
    / residual_scale), min, max), precomputed over the item vocabulary from the
    FULL raw reviews frame. 1.0-centered like the other multiplicative
    modulators; unresolved/invalid candidates stay neutral. Pointwise in the
    item, hence TB-safe with the precomputed full-catalog logZ.

    Rationale: under the positive/auxiliary datamodule contract batches carry
    no usable rating payload and the auxiliary store is empty on all-verified
    dumps (beauty 2014), so the per-item prior is the rating signal available
    identically on every dataset.
    """

    def __init__(
        self,
        item_multiplier: torch.Tensor,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
        reward_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("item_multiplier", item_multiplier.float())
        self.register_buffer(
            "item_id_by_semantic_row",
            item_id_by_semantic_row.long()
            if item_id_by_semantic_row is not None
            else torch.empty(0, dtype=torch.long),
        )
        self.reward_epsilon = float(reward_epsilon)

    def _candidate_item_ids(self, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        extra = trajectories.extra or {}
        cand = extra.get("candidate_item_ids")
        if torch.is_tensor(cand):
            return cand.long()
        rows = getattr(trajectories, "candidate_item_row_ids", None)
        if torch.is_tensor(rows) and self.item_id_by_semantic_row.numel() > 0:
            mapping = self.item_id_by_semantic_row.to(rows.device)
            safe = rows.long().clamp_min(0).clamp_max(mapping.numel() - 1)
            ids = mapping[safe]
            return torch.where(rows.ge(0), ids, torch.full_like(ids, -1))
        raise ValueError(
            "ItemPriorRatingReward needs candidate_item_ids in trajectories.extra "
            "or candidate_item_row_ids + item_id_by_semantic_row."
        )

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        ids = self._candidate_item_ids(trajectories)
        table = self.item_multiplier.to(ids.device)
        valid = ids.gt(0) & ids.lt(table.numel())
        mult = table[ids.clamp_min(0).clamp_max(table.numel() - 1)]
        mult = torch.where(valid, mult, torch.ones_like(mult))
        return mult.clamp_min(self.reward_epsilon)


class PowerRewardModel(torch.nn.Module):
    """R -> R**degree (GFlowNet reward sharpness). Elementwise and
    candidate-set independent, so it stays consistent with the precomputed
    full-catalog logZ (which is computed from the powered rewards)."""

    def __init__(self, base: torch.nn.Module, degree: float, *, reward_epsilon: float = 1e-6) -> None:
        super().__init__()
        self.base = base
        self.degree = float(degree)
        self.reward_epsilon = float(reward_epsilon)

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        rewards = self.base(batch, trajectories).float()
        if self.degree == 1.0:
            return rewards
        return rewards.clamp_min(0.0).pow(self.degree).clamp_min(self.reward_epsilon)


class RewardComponentFactory:
    """Builds (and caches) the per-dataset reward component modules."""

    def __init__(
        self,
        *,
        dm: AmazonReviewDataModule,
        sasrec: SASRec,
        row_to_item: torch.Tensor,
        rqvae_model_path: Path,
        embeddings_path: Path,
        reviews_path: Path,
        item_key: str,
        sasrec_transform: str,
        sasrec_temperature: float,
    ) -> None:
        self.dm = dm
        self.sasrec = sasrec
        self.row_to_item = row_to_item
        self.rqvae_model_path = rqvae_model_path
        self.embeddings_path = embeddings_path
        self.reviews_path = reviews_path
        self.item_key = item_key
        self.sasrec_transform = sasrec_transform
        self.sasrec_temperature = float(sasrec_temperature)
        self._cache: dict[str, torch.nn.Module] = {}

    def component(self, key: str) -> torch.nn.Module:
        if key not in self._cache:
            self._cache[key] = getattr(self, f"_build_{key}")()
        return self._cache[key]

    def _build_sasrec(self) -> torch.nn.Module:
        if self.sasrec_transform not in TB_SAFE_SASREC_TRANSFORMS:
            raise ValueError(
                f"sasrec reward transform {self.sasrec_transform!r} is candidate-set dependent "
                f"and inconsistent with precomputed logZ. Use one of {TB_SAFE_SASREC_TRANSFORMS}."
            )
        return SASRecRewardModel(
            self.sasrec,
            SASRecRewardConfig(
                transform=self.sasrec_transform,
                temperature=self.sasrec_temperature,
            ),
            item_id_by_semantic_row=self.row_to_item,
        )

    def _build_sim_embed(self) -> torch.nn.Module:
        return SimilarityRewardModel(
            SimilarityRewardConfig(mode="embedding_cosine"),
            item_embeddings=build_item_embedding_table(
                self.embeddings_path, self.dm.item2id, int(self.dm.item_vocab_size)
            ),
            item_id_by_semantic_row=self.row_to_item,
        )

    def _build_sim_hier(self) -> torch.nn.Module:
        # Equal level weights: under prefix gating, deep matches are rare and
        # strong evidence. The level stride de-offsets the exp2 token layout.
        codebooks = load_rqvae_codebooks(self.rqvae_model_path)
        return LevelOffsetSimilarityReward(
            SimilarityRewardConfig(mode="hierarchical"),
            codebooks=codebooks,
            level_token_stride=int(codebooks.shape[1]),
            item_id_by_semantic_row=self.row_to_item,
        )

    # Rating-prior constants. NOTE: the original construction code was lost in
    # the branch stash; this reimplements the formula documented on
    # ItemPriorRatingReward with explicit choices: empirical-Bayes shrinkage of
    # 10 pseudo-ratings toward the global mean, residuals scaled by their own
    # std (beta = 1), multipliers clipped to [0.5, 2].
    RATING_SHRINKAGE = 10.0
    RATING_BETA = 1.0
    RATING_CLIP = (0.5, 2.0)

    def _build_rating(self) -> torch.nn.Module:
        import pandas as pd

        frame = pd.read_parquet(self.reviews_path, columns=[self.item_key, "rating"])
        frame = frame.dropna(subset=["rating"])
        if not len(frame):
            raise ValueError(f"no ratings in {self.reviews_path}")
        grouped = frame.groupby(self.item_key)["rating"].agg(["sum", "count"])
        global_mean = float(frame["rating"].mean())
        shrunk = (grouped["sum"] + self.RATING_SHRINKAGE * global_mean) / (
            grouped["count"] + self.RATING_SHRINKAGE
        )
        residuals = shrunk - global_mean
        scale = float(residuals.std()) or 1.0
        low, high = self.RATING_CLIP
        multipliers = np.exp(self.RATING_BETA * residuals.to_numpy() / scale).clip(low, high)

        table = torch.ones(int(self.dm.item_vocab_size), dtype=torch.float32)
        item2id = self.dm.item2id
        for raw_item, value in zip(grouped.index, multipliers):
            item_id = item2id.get(str(raw_item))
            if item_id is not None:
                table[int(item_id)] = float(value)
        return ItemPriorRatingReward(table, item_id_by_semantic_row=self.row_to_item)

    def build(self, name: str) -> torch.nn.Module:
        """Compose one grid entry from cached components (single / sum / product)."""
        if name not in REWARD_SPECS:
            raise ValueError(f"Unknown reward {name!r}. Expected one of {REWARD_CHOICES}.")
        op, keys = REWARD_SPECS[name]
        components = [self.component(key) for key in keys]
        if op == "single":
            return components[0]
        if op == "product":
            return MultiplicativeRewardModel(components)
        model = MixedRewardModel(components, normalizers=["none"] * len(components))
        assert not model.uses_candidate_set_normalization, (
            "sum combos must stay pointwise to be consistent with precomputed logZ"
        )
        return model


def sasrec_reward_key(transform: str, temperature: float) -> str:
    """Cache/run key for the sasrec reward; default transform keeps the legacy name."""
    if transform == "softplus" and abs(float(temperature) - 1.0) < 1e-12:
        return "sasrec"
    return f"sasrec_{transform}_T{float(temperature):g}"


def build_reward_model(
    name: str,
    *,
    sasrec: SASRec,
    row_to_item: torch.Tensor,
    sasrec_transform: str = "softplus",
    sasrec_temperature: float = 1.0,
) -> torch.nn.Module:
    """SASRec-only reward builder kept for sasrec_reward_analysis.py.

    The exp4 grid itself composes rewards via RewardComponentFactory.build.
    """
    if not str(name).startswith("sasrec"):
        raise ValueError(f"build_reward_model only builds sasrec rewards, got {name!r}.")
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


def load_sft_checkpoint(model: TigerModel, checkpoint_path: Path) -> None:
    """Load SFT TIGER weights into an existing model (strict, arch-checked)."""
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


@torch.no_grad()
def full_catalog_rewards(
    name: str,
    reward_model: torch.nn.Module,
    batch: Batch,
    *,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    chunk_items: int = 4096,
) -> torch.Tensor:
    """Base (degree-1) rewards over the whole semantic catalog, [B, num_rows].

    The single-sasrec reward gets a vectorized fast path that mirrors
    SASRecRewardModel exactly; every other composition runs the live module
    itself, so the precomputed logZ is exactly consistent with the training
    rewards.
    """
    device = item_tokens.device

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

    # Everything else (similarity modes, interaction evidence, multiplicative
    # combos) is evaluated with the live module itself - exact consistency for
    # free. The interaction reward's before-t store lookups run on CPU numpy,
    # so full-catalog precompute pays a per-chunk lookup cost.
    return live_catalog_rewards(
        reward_model, batch, item_tokens=item_tokens, row_to_item=row_to_item,
        chunk_items=chunk_items,
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
    degrees: list[float],
    *,
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    device: torch.device,
    batch_size: int,
    example_infos: list[dict[str, Any]],
    max_batches: int | None = None,
) -> dict[str, Any]:
    """One pass over the train split. Per (reward, degree) it records, for every
    training example, the full-catalog sum of R**degree - the analytical
    normalizing constant logZ = log(sum of powered rewards)."""
    loader = indexed_train_loader(dm, shuffle=False, batch_size=batch_size)
    n_examples = len(dm.train_dataset)
    example_rows = {info["example_index"]: {} for info in example_infos}

    item_tokens = item_tokens.to(device)
    row_to_item = row_to_item.to(device)
    item_to_row = item_to_row.to(device)
    for model in reward_models.values():
        model.to(device)

    keys = [run_key(name, deg) for name in reward_names for deg in degrees]
    stats = {
        key: {
            "reward_sum": torch.empty(n_examples),
            "target_reward": torch.empty(n_examples),
            "target_rank": torch.empty(n_examples, dtype=torch.long),
            "entropy": torch.empty(n_examples),
        }
        for key in keys
    }
    computed = torch.zeros(n_examples, dtype=torch.bool)

    start_time = time.time()
    processed = 0
    for batch_idx, raw_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        idx = batch["example_indices"].cpu()

        target_items = batch["target_item_ids"].long()
        safe_t = target_items.clamp_min(0).clamp_max(item_to_row.numel() - 1)
        target_rows = item_to_row[safe_t]
        valid_t = target_items.ge(0) & target_items.lt(item_to_row.numel()) & target_rows.ge(0)

        for name in reward_names:
            base = full_catalog_rewards(
                name,
                reward_models[name],
                batch,
                item_tokens=item_tokens,
                row_to_item=row_to_item,
            ).float()
            for degree in degrees:
                rewards = base if float(degree) == 1.0 else base.pow(float(degree))
                key = run_key(name, degree)
                reward_sum = rewards.sum(dim=1).clamp_min(1e-12)
                log_r = rewards.clamp_min(1e-12).log()
                entropy = reward_sum.log() - (rewards * log_r).sum(dim=1) / reward_sum

                gathered = torch.gather(rewards, 1, target_rows.clamp_min(0).unsqueeze(1)).squeeze(1)
                target_reward = torch.where(valid_t, gathered, torch.zeros_like(gathered))
                target_rank = rewards.gt(target_reward.unsqueeze(1)).sum(dim=1).long() + 1
                target_rank = torch.where(valid_t, target_rank, torch.full_like(target_rank, -1))

                stats[key]["reward_sum"][idx] = reward_sum.cpu()
                stats[key]["target_reward"][idx] = target_reward.cpu()
                stats[key]["target_rank"][idx] = target_rank.cpu()
                stats[key]["entropy"][idx] = entropy.cpu()

            for row_pos, example_index in enumerate(idx.tolist()):
                if example_index in example_rows:
                    example_rows[example_index][name] = base[row_pos].cpu()

        computed[idx] = True
        processed += int(idx.numel())
        if batch_idx % 20 == 0:
            rate = processed / max(time.time() - start_time, 1e-6)
            print(
                f"[logz] {processed:,}/{n_examples:,} examples ({rate:,.0f} ex/s)",
                flush=True,
            )

    partial = bool((~computed).any())
    if partial:
        # Smoke runs (--logz-max-batches) leave most entries untouched; fill
        # them with the mean of the computed ones so lookups stay finite.
        done = computed.nonzero(as_tuple=True)[0]
        rest = (~computed).nonzero(as_tuple=True)[0]
        for key in keys:
            for field in ("reward_sum", "target_reward", "entropy"):
                stats[key][field][rest] = float(stats[key][field][done].mean())
            stats[key]["target_rank"][rest] = -1
        print(
            f"[logz] WARNING: only {int(done.numel()):,}/{n_examples:,} examples computed; "
            "remaining logZ entries filled with the computed mean (smoke mode).",
            flush=True,
        )

    payload: dict[str, Any] = {
        "num_examples": n_examples,
        "rewards": {},
        "example_infos": example_infos,
        "example_rows": example_rows,
        "partial": partial,
    }
    for key in keys:
        reward_sum = stats[key]["reward_sum"]
        payload["rewards"][key] = {
            "reward_sum": reward_sum,
            "log_z": reward_sum.clamp_min(1e-12).log(),
            "target_reward": stats[key]["target_reward"],
            "target_rank": stats[key]["target_rank"],
            "entropy": stats[key]["entropy"],
        }
    return payload


def load_or_compute_logz(
    cache_path: Path,
    *,
    recompute: bool,
    dm: AmazonReviewDataModule,
    reward_names: list[str],
    reward_models: Mapping[str, torch.nn.Module],
    degrees: list[float],
    item_tokens: torch.Tensor,
    row_to_item: torch.Tensor,
    item_to_row: torch.Tensor,
    device: torch.device,
    batch_size: int,
    example_infos: list[dict[str, Any]],
    max_batches: int | None = None,
) -> dict[str, Any]:
    if max_batches is not None:
        # Smoke mode computes a partial table and never touches the cache.
        return precompute_logz(
            dm, list(reward_names), reward_models, list(degrees),
            item_tokens=item_tokens, row_to_item=row_to_item, item_to_row=item_to_row,
            device=device, batch_size=batch_size, example_infos=example_infos,
            max_batches=max_batches,
        )

    wanted_examples = {info["example_index"] for info in example_infos}
    wanted_keys = {run_key(name, deg) for name in reward_names for deg in degrees}
    existing: dict[str, Any] | None = None
    compute_names = list(reward_names)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if int(payload.get("num_examples", -1)) == len(dm.train_dataset) and not payload.get("partial"):
            existing = payload
            cached_rows = payload.get("example_rows", {})
            examples_ok = wanted_examples.issubset(set(cached_rows.keys()))
            missing_keys = sorted(wanted_keys - set(payload.get("rewards", {})))
            missing = sorted({key.split("__deg")[0] for key in missing_keys})
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
                compute_names = list(reward_names)
                print(f"[logz] cache {cache_path}: recomputing {compute_names}", flush=True)
            elif examples_ok and not missing:
                print(f"[logz] using cached logZ from {cache_path}", flush=True)
                return payload
            else:
                compute_names = missing if examples_ok else list(reward_names)
                print(
                    f"[logz] cache {cache_path}: computing {compute_names} "
                    f"(cached entries kept: {sorted(payload.get('rewards', {}).keys())})",
                    flush=True,
                )
        else:
            print(f"[logz] cache {cache_path} has a different/partial train split; recomputing", flush=True)

    new_payload = precompute_logz(
        dm,
        compute_names,
        {name: reward_models[name] for name in compute_names},
        list(degrees),
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

    logZ(context) = log(sum over the full catalog of R(item | context)**degree),
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
    ``val/sft/ndcg@K``."""

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
    sampler: SemanticIDCandidateSampler,
    log_z: torch.Tensor,
    tiger_cfg: Mapping[str, Any],
    gflow_weight: float,
    num_negatives: int,
    include_positive: bool = True,
    val_topk: int = 1,
) -> GFlowMethod:
    # val_topk sets the validation-generation depth (beam K, monitor ndcg@K);
    # it is derived from the trainer YAML's monitor key unless --val-topk is given.
    val_k = max(int(val_topk), 1)
    constrain = bool(tiger_cfg.get("constrain_valid_sequences", True))
    gflow_cfg = GFlowConfig(
        objective="tb",
        sft_weight=1.0,
        gflow_weight=float(gflow_weight),
        logz_context="global",  # placeholder; replaced by PrecomputedLogZ below
        num_negative_samples=int(num_negatives),
        include_positive=bool(include_positive),
        reward_epsilon=1e-6,
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        evaluate_generation=True,
        generation_topk=(val_k,),
        generation_num_beams=val_k,
        generation_num_return_sequences=val_k,
        generation_constrain_valid_sequences=constrain,
    )
    sft_cfg = SFTConfig(
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        compute_item_metrics=False,
        evaluate_generation=True,
        generation_topk=(val_k,),
        generation_num_beams=val_k,
        generation_num_return_sequences=val_k,
        generation_constrain_valid_sequences=constrain,
    )
    method = GFlowMethod(
        gflow_cfg,
        candidate_sampler=sampler,
        reward_model=reward_model,
        sft_method=AliasedSFTMethod(sft_cfg),
    )
    # Analytical, fixed logZ (logZ = log sum of powered catalog rewards).
    method.log_z_model = PrecomputedLogZ(log_z)
    return method


def build_curriculum_sampler(
    *,
    sasrec: SASRec,
    model: TigerModel,
    item_to_row: torch.Tensor,
    curriculum_steps: int,
    include_positive: bool = True,
    exclude_history: bool = True,
) -> SemanticIDCandidateSampler:
    return SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="curriculum",
            num_negatives=3,
            include_positive=bool(include_positive),
            exclude_history=bool(exclude_history),
            curriculum_steps=int(curriculum_steps),
            temperature=1.0,
        ),
        item_semantic_ids=model.item_semantic_ids.detach().cpu(),
        item_id_to_semantic_row=restrict_mapping_to_scorer_vocab(item_to_row, sasrec),
        scorer=sasrec,
    )


# --------------------------------------------------------------------------- #
# Evaluation (beam-search HR/NDCG/Coverage on the test split)
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
    degrees: list[float],
    out_path: Path,
    dataset_name: str,
) -> None:
    example_rows = logz_payload["example_rows"]
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
                color=reward_color(name),
                edgecolor=CHROME["surface"],
                linewidth=0.3,
            )
            ax.set_yscale("log")
            annotation_lines = []
            for degree in degrees:
                stats = logz_payload["rewards"].get(run_key(name, degree))
                if stats is None:
                    continue
                annotation_lines.append(
                    f"d={degree:g}: logZ={float(stats['log_z'][idx]):.2f} "
                    f"eff={float(np.exp(stats['entropy'][idx])):,.0f}"
                )
            first_stats = logz_payload["rewards"].get(run_key(name, degrees[0]))
            if first_stats is not None:
                annotation_lines.append(f"target rank={int(first_stats['target_rank'][idx]):,}")
            target_reward = float(first_stats["target_reward"][idx]) if first_stats is not None else 0.0
            if target_reward > 0 and float(degrees[0]) == 1.0:
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
            if annotation_lines:
                ax.text(
                    0.02,
                    0.95,
                    "\n".join(annotation_lines),
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
        f"Full-catalog base-reward distributions per user context — {dataset_name}",
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
        f"Sorted top-{top_k} catalog base rewards per user context — {dataset_name}",
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
    title: str,
    val_topk: int,
) -> None:
    panels = (
        ("train/loss", "train total loss", False),
        ("train/gflow/tb_loss", "train TB loss", True),
        (f"val/sft/generation_ndcg@{val_topk}", f"val NDCG@{val_topk} (beam={val_topk})", False),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False)
    fig.patch.set_facecolor(CHROME["surface"])
    for p, (key, panel_title, log_scale) in enumerate(panels):
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
                    color=reward_color(name),
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                    label=name,
                )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(panel_title, fontsize=10, color=CHROME["ink"])
        ax.set_xlabel("epoch", fontsize=8, color=CHROME["ink2"])
        if p == 0:
            ax.legend(fontsize=7, frameon=False, labelcolor=CHROME["ink2"])
    fig.suptitle(title, fontsize=12, color=CHROME["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def make_status_logger(tag: str, val_topk: int):
    keys = (
        "train/loss",
        "train/sft/loss",
        "train/gflow/tb_loss",
        "train/gflow/log_z_mean",
        "train/gflow/reward_mean",
        "val/loss",
        f"val/sft/ndcg@{val_topk}",
        f"val/sft/generation_coverage@{val_topk}",
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


def resolve_device(args: argparse.Namespace, trainer_cfg: Mapping[str, Any]) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    wanted = str(trainer_cfg.get("device", "cuda"))
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(wanted)


def resolve_val_topk(args: argparse.Namespace, trainer_cfg: Mapping[str, Any]) -> tuple[int, str]:
    """Validation beam depth K and the monitored metric val/sft/ndcg@K.

    --val-topk wins; otherwise K is parsed out of the trainer YAML's monitor
    key (e.g. val/sft/ndcg@5 -> 5), so the YAML alone fully defines the run.
    """
    if args.val_topk:
        k = int(args.val_topk)
        return k, f"val/sft/ndcg@{k}"
    monitor = str(trainer_cfg.get("monitor", "val/sft/ndcg@5"))
    match = re.search(r"@(\d+)$", monitor)
    k = int(match.group(1)) if match else 1
    return k, monitor


def run_single(
    *,
    dataset_name: str,
    reward_name: str,
    degree: float,
    train_seed: int,
    dm: AmazonReviewDataModule,
    sasrec: SASRec,
    base_reward_model: torch.nn.Module,
    log_z: torch.Tensor,
    item_to_row: torch.Tensor,
    test_eval_batch_size: int,
    cfgs: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    trainer_cfg_yaml = cfgs["trainer"]
    tiger_cfg = cfgs["tiger"]
    key = run_key(reward_name, degree)
    tag = f"{dataset_name}/{key}/seed{train_seed}"

    model = build_finetune_model(dm, tiger_cfg, sft_checkpoint_path(args, dataset_name))
    reward_model = (
        base_reward_model
        if float(degree) == 1.0
        else PowerRewardModel(base_reward_model, float(degree))
    )

    sampler = build_curriculum_sampler(
        sasrec=sasrec,
        model=model,
        item_to_row=item_to_row,
        curriculum_steps=args.easy_negative_steps,
        include_positive=not args.tb_exclude_positive,
    )
    val_topk, monitor = resolve_val_topk(args, trainer_cfg_yaml)
    method = build_gflow_method(
        reward_model=reward_model,
        sampler=sampler,
        log_z=log_z,
        tiger_cfg=tiger_cfg,
        gflow_weight=args.gflow_weight,
        num_negatives=args.num_negatives,
        include_positive=not args.tb_exclude_positive,
        val_topk=val_topk,
    )

    device = resolve_device(args, trainer_cfg_yaml)
    max_epochs = int(args.max_epochs if args.max_epochs is not None else trainer_cfg_yaml.get("max_epochs", 500))
    validate_every = int(
        args.validate_every_epochs
        if args.validate_every_epochs is not None
        else trainer_cfg_yaml.get("validate_every_epochs", 1)
    )
    lr = float(args.lr if args.lr is not None else trainer_cfg_yaml.get("lr", 3e-5))
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=int(trainer_cfg_yaml.get("gradient_accumulation_steps", 1)),
            max_grad_norm=float(trainer_cfg_yaml.get("max_grad_norm", 2.0)),
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
        logger=make_status_logger(tag, val_topk),
    )

    op, components = REWARD_SPECS[reward_name]
    resolved = {
        "dataset": dataset_name,
        "reward": reward_name,
        "reward_op": op,
        "reward_components": list(components),
        "degree": float(degree),
        "seed": int(train_seed),
        "gflow": method.cfg.to_dict(),
        "sampler": {
            "type": "sasrec_curriculum",
            "num_negatives": 3,
            "include_positive": not args.tb_exclude_positive,
            "curriculum_steps": int(args.easy_negative_steps),
        },
        "reward_options": {
            "sasrec_transform": str(args.reward_transform),
            "sasrec_temperature": float(args.reward_temperature),
        },
        "logz": {
            "source": "analytical_full_catalog_powered_reward_sum",
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
    with eval_batch_size(dm, test_eval_batch_size):
        test_metrics = evaluate_generation(
            model,
            dm,
            ks=ks,
            num_beams=int(tiger_cfg.get("eval_num_beams", 20)),
            constrain_valid_sequences=bool(tiger_cfg.get("constrain_valid_sequences", True)),
            max_batches=args.max_eval_batches,
            log_every=int(trainer_cfg_yaml.get("eval_log_every_batches", 50)),
        )
    write_json(
        run_dir / "test_metrics.json",
        {"dataset": dataset_name, "reward": reward_name, "degree": float(degree), **test_metrics},
    )
    print(f"[{tag}] test metrics: {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    return {
        "dataset": dataset_name,
        "reward": reward_name,
        "degree": float(degree),
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


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
SUMMARY_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "Coverage@5", "Coverage@10", "Coverage@20")


def write_summary(out_dir: Path, records: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_ = [
        "dataset", "reward", "degree", "seed", "status",
        "best_epoch", "best_metric", "epochs_completed", "steps_completed",
    ]
    fields_ += list(SUMMARY_METRICS) + ["run_dir"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields_, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Exp4 GFlow TB reward ablation "
        "(analytical logZ, sasrec curriculum negatives, gflow_weight fixed)",
        "",
        "| dataset | reward | degree | seed | status | best epoch | " + " | ".join(SUMMARY_METRICS) + " |",
        "| --- | --- | ---: | ---: | --- | ---: | " + " | ".join("---:" for _ in SUMMARY_METRICS) + " |",
    ]
    for record in records:
        cells = [
            str(record.get("dataset", "")),
            str(record.get("reward", "")),
            str(record.get("degree", "")),
            str(record.get("seed", "")),
            str(record.get("status", "")),
            str(record.get("best_epoch", "")),
        ]
        cells += [
            f"{record[m]:.5f}" if isinstance(record.get(m), float) else ""
            for m in SUMMARY_METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # Cross-seed aggregation: mean +/- std per (dataset, reward, degree).
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "completed":
            group_key = (
                str(record.get("dataset")),
                str(record.get("reward")),
                str(record.get("degree")),
            )
            groups.setdefault(group_key, []).append(record)
    agg_rows = []
    for (ds, reward, degree), group in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": ds, "reward": reward, "degree": degree, "n_seeds": len(group)}
        for m in SUMMARY_METRICS:
            vals = [float(r[m]) for r in group if isinstance(r.get(m), (int, float))]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        agg_rows.append(row)
    if agg_rows:
        agg_fields = ["dataset", "reward", "degree", "n_seeds"]
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
            "| dataset | reward | degree | seeds | " + " | ".join(SUMMARY_METRICS) + " |",
            "| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in SUMMARY_METRICS) + " |",
        ]
        for row in agg_rows:
            cells = [row["dataset"], row["reward"], str(row["degree"]), str(row["n_seeds"])]
            cells += [
                f"{row[m]:.5f} ± {row[f'{m}_std']:.5f}" if isinstance(row.get(m), float) else ""
                for m in SUMMARY_METRICS
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def append_record_jsonl(out_dir: Path, record: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in record.items() if k != "history"}
    with open(out_dir / "test_metrics_records.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(slim, sort_keys=True, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Exp4 GFlow TB reward ablation for TIGER fine-tuning "
        "(individual / pairwise / triple reward combinations x reward degree)"
    )
    p.add_argument("--dataset", choices=[*DATASET_CONFIG_FILES, "all"], default="all")
    p.add_argument("--rewards", nargs="+", choices=list(REWARD_CHOICES), default=list(REWARD_CHOICES),
                   help="Reward compositions to run (default: all 18)")
    p.add_argument("--reward-degrees", nargs="+", type=float, default=list(DEFAULT_DEGREES),
                   help="Reward degree grid: final reward = R**degree (default: 1 0.25 4)")
    p.add_argument("--config-dir", default="configs/exp4-reward_ablation")
    p.add_argument("--sft-checkpoint-dir", default="checkpoints/exp2/loo_split/initial/tiger",
                   help="Directory with <dataset>/best.pt SFT TIGER checkpoints")
    p.add_argument("--sasrec-checkpoint-dir", default="checkpoints/exp2/loo_split/initial/sasrec",
                   help="Directory with <dataset>/best.pt SASRec teachers")
    p.add_argument("--semantic-dir", default=None,
                   help="Semantic-ID root (<ds>/semantic_ids.json + RQ-VAE model.pt). "
                        "Defaults to semantic_ids.yaml semantic_dir")
    p.add_argument("--embeddings-dir", default="datasets/embeddings",
                   help="Directory with <dataset>_content_embeddings.pt (sim_embed reward)")
    p.add_argument("--output-dir", default="experiments/exp4-reward_ablation")
    p.add_argument("--artifacts-dir", default=None,
                   help="Test-metric summaries are mirrored here "
                        "(default: trainer YAML artifacts_dir or artifacts/exp4-reward_ablation)")
    p.add_argument("--logz-cache-dir", default="artifacts/exp4-reward_ablation/logz_cache")
    p.add_argument("--recompute-logz", action="store_true")
    p.add_argument("--device", default=None, help="Defaults to trainer.yaml device")
    p.add_argument("--seed", type=int, default=None, help="Defaults to trainer.yaml seed")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Training seeds; each (reward, degree) is fine-tuned once per seed "
                        "(data split / logZ stay fixed). Overrides --seed.")
    p.add_argument("--max-epochs", type=int, default=None, help="Defaults to trainer.yaml max_epochs")
    p.add_argument("--validate-every-epochs", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override the dataset YAML tiger-block train batch size")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Override the in-training validation batch size")
    p.add_argument("--test-eval-batch-size", type=int, default=None,
                   help="Override the final 20-beam test evaluation batch size")
    p.add_argument("--num-negatives", type=int, default=3,
                   help="Negatives per context from the curriculum sampler (experiment spec: 3)")
    p.add_argument("--easy-negative-steps", type=int, default=10000,
                   help="Curriculum: easy (low-SASRec-score) negatives for this many steps, then hard")
    p.add_argument("--num-random-negatives", type=int, default=0,
                   help="Of the negatives, keep this many uniform-random (0 = pure curriculum)")
    p.add_argument("--gflow-weight", type=float, default=0.25,
                   help="TB loss weight next to sft_weight=1 (experiment spec: 0.25)")
    p.add_argument("--reward-transform", choices=list(TB_SAFE_SASREC_TRANSFORMS), default="softplus",
                   help="SASRec score -> reward transform (changing it requires --recompute-logz)")
    p.add_argument("--reward-temperature", type=float, default=1.0,
                   help="Temperature for the SASRec reward transform (changing it requires --recompute-logz)")
    p.add_argument("--tb-exclude-positive", action="store_true",
                   help="TB candidates are negatives only; SFT alone owns the target (avoids TB/SFT conflict)")
    p.add_argument("--lr", type=float, default=None,
                   help="Fine-tuning learning rate (default: trainer.yaml lr)")
    p.add_argument("--val-topk", type=int, default=None,
                   help="Validate/monitor ndcg@K with K beams (default: derived from trainer.yaml monitor)")
    p.add_argument("--val-sample-fraction", type=float, default=None,
                   help="Randomly subsample this fraction of the val set per validation")
    p.add_argument("--num-example-users", type=int, default=4)
    p.add_argument("--logz-batch-size", type=int, default=512)
    p.add_argument("--logz-max-batches", type=int, default=None,
                   help="Smoke mode: only precompute logZ for this many batches (cache is bypassed)")
    p.add_argument("--skip-training", action="store_true", help="Only precompute logZ and produce plots")
    p.add_argument("--skip-baseline", action="store_true", help="Skip test evaluation of the raw SFT checkpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_dir = path_from_repo(args.config_dir)
    cfgs = load_exp_configs(config_dir)
    seed = int(args.seed if args.seed is not None else cfgs["trainer"].get("seed", 42))
    train_seeds = [int(s) for s in (args.seeds or [seed])]
    set_seed(seed)

    semantic_dir = path_from_repo(
        args.semantic_dir
        if args.semantic_dir is not None
        else cfgs["semantic"].get("semantic_dir", "datasets/semantic_ids/exp2-tuned-params")
    )
    artifacts_dir = path_from_repo(
        args.artifacts_dir
        if args.artifacts_dir is not None
        else cfgs["trainer"].get("artifacts_dir", "artifacts/exp4-reward_ablation")
    )
    device = resolve_device(args, cfgs["trainer"])
    degrees = [float(d) for d in args.reward_degrees]

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Independent invocations can start within the same second.
    from uuid import uuid4
    uniq = uuid4().hex[:12]
    out_root = path_from_repo(args.output_dir) / f"{stamp}_{uniq}_rewards_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    artifacts_root = artifacts_dir / out_root.name
    write_json(out_root / "cli_args.json", vars(args))
    print(f"[main] output dir: {out_root}", flush=True)
    print(f"[main] artifacts dir: {artifacts_root}", flush=True)
    print(f"[main] rewards={list(args.rewards)} degrees={degrees}", flush=True)

    datasets = list(DATASET_CONFIG_FILES) if args.dataset == "all" else [args.dataset]
    records: list[dict[str, Any]] = []

    def flush_summaries() -> None:
        write_summary(out_root, records)
        write_summary(artifacts_root, records)

    for dataset_name in datasets:
        print(f"\n=== {dataset_name} ===", flush=True)
        sft_ckpt = sft_checkpoint_path(args, dataset_name)
        if not sft_ckpt.exists():
            print(f"[{dataset_name}] SKIPPED: SFT checkpoint {sft_ckpt} does not exist.", flush=True)
            records.append(
                {"dataset": dataset_name, "reward": "-", "degree": "", "status": f"failed: missing {sft_ckpt}"}
            )
            flush_summaries()
            continue
        saved_cfg = saved_model_config(sft_ckpt)
        user_hash_mod = max(1, int(saved_cfg.get("user_ids_count", cfgs["tiger"].get("user_ids_count", 0))))

        data_cfg, test_eval_batch_size = build_data_config(
            dataset_name,
            cfgs["datasets"][dataset_name],
            seed=seed,
            semantic_dir=semantic_dir,
            user_hash_mod=user_hash_mod,
            args=args,
        )
        dm = build_datamodule(data_cfg)
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

        factory = RewardComponentFactory(
            dm=dm,
            sasrec=sasrec,
            row_to_item=row_to_item,
            rqvae_model_path=rqvae_checkpoint_path(semantic_dir, dataset_name),
            embeddings_path=path_from_repo(args.embeddings_dir) / f"{dataset_name}_content_embeddings.pt",
            reviews_path=Path(data_cfg.reviews_path),
            item_key=str(data_cfg.item_key),
            sasrec_transform=args.reward_transform,
            sasrec_temperature=args.reward_temperature,
        )
        reward_models: dict[str, torch.nn.Module] = {}
        for name in args.rewards:
            try:
                reward_models[name] = factory.build(name)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[{dataset_name}/{name}] SKIPPED: {exc}", flush=True)
                records.append(
                    {"dataset": dataset_name, "reward": name, "degree": "", "status": f"failed: {exc}"}
                )
        flush_summaries()
        reward_names = list(reward_models)
        if not reward_names:
            continue

        example_infos = select_example_indices(dm, args.num_example_users)
        print(f"[{dataset_name}] example users for plots: {example_infos}", flush=True)

        logz_payload = load_or_compute_logz(
            path_from_repo(args.logz_cache_dir) / f"{dataset_name}.pt",
            recompute=args.recompute_logz,
            dm=dm,
            reward_names=reward_names,
            reward_models=reward_models,
            degrees=degrees,
            item_tokens=item_tokens,
            row_to_item=row_to_item,
            item_to_row=item_to_row,
            device=device,
            batch_size=args.logz_batch_size,
            example_infos=example_infos,
            max_batches=args.logz_max_batches,
        )

        plots_dir = out_root / dataset_name / "plots"
        plot_reward_distributions(
            logz_payload, reward_names, degrees,
            plots_dir / f"reward_distributions_{dataset_name}.png", dataset_name,
        )
        plot_topk_reward_curves(
            logz_payload, reward_names, plots_dir / f"reward_topk_{dataset_name}.png", dataset_name
        )

        for name in reward_names:
            for degree in degrees:
                key = run_key(name, degree)
                block = logz_payload["rewards"][key]
                print(
                    f"[{dataset_name}] logZ[{key}]: mean={float(block['log_z'].mean()):.3f} "
                    f"std={float(block['log_z'].std()):.3f} "
                    f"min={float(block['log_z'].min()):.3f} max={float(block['log_z'].max()):.3f}",
                    flush=True,
                )

        if args.skip_training:
            continue

        if not args.skip_baseline:
            baseline_model = build_finetune_model(dm, cfgs["tiger"], sft_ckpt)
            baseline_model.to(device)
            ks = tuple(int(k) for k in cfgs["tiger"].get("ks", [5, 10, 20]))
            with eval_batch_size(dm, test_eval_batch_size):
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
            baseline_record = {
                "dataset": dataset_name, "reward": "sft_baseline", "degree": "",
                "status": "completed", **baseline_metrics,
            }
            records.append(baseline_record)
            append_record_jsonl(artifacts_root, baseline_record)
            flush_summaries()
            print(f"[{dataset_name}] SFT baseline: {json.dumps(baseline_metrics, sort_keys=True)}", flush=True)

        histories: dict[str, list[dict[str, Any]]] = {}
        for reward_name in reward_names:
            for degree in degrees:
                key = run_key(reward_name, degree)
                for train_seed in train_seeds:
                    run_dir = out_root / dataset_name / key / f"seed_{train_seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    set_seed(train_seed)
                    try:
                        record = run_single(
                            dataset_name=dataset_name,
                            reward_name=reward_name,
                            degree=degree,
                            train_seed=train_seed,
                            dm=dm,
                            sasrec=sasrec,
                            base_reward_model=reward_models[reward_name],
                            log_z=logz_payload["rewards"][key]["log_z"],
                            item_to_row=item_to_row,
                            test_eval_batch_size=test_eval_batch_size,
                            cfgs=cfgs,
                            args=args,
                            run_dir=run_dir,
                        )
                    except Exception as exc:  # keep the sweep alive; report at the end
                        print(f"[{dataset_name}/{key}/seed{train_seed}] FAILED: {exc!r}", flush=True)
                        record = {
                            "dataset": dataset_name,
                            "reward": reward_name,
                            "degree": float(degree),
                            "seed": train_seed,
                            "status": f"failed: {exc}",
                        }
                    history = record.pop("history", [])
                    if train_seed == train_seeds[0]:
                        histories[key] = history
                    records.append(record)
                    append_record_jsonl(artifacts_root, record)
                    flush_summaries()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        val_topk, _ = resolve_val_topk(args, cfgs["trainer"])
        for degree in degrees:
            suffix = f"__deg{degree:g}"
            histories_d = {
                key.split("__deg")[0]: history
                for key, history in histories.items()
                if key.endswith(suffix) and history
            }
            if histories_d:
                plot_training_curves(
                    histories_d,
                    plots_dir / f"training_curves_{dataset_name}_deg{degree:g}.png",
                    f"GFlow fine-tuning by reward — {dataset_name}, degree {degree:g}",
                    val_topk,
                )

    flush_summaries()
    print(f"\n[main] done. Summary: {out_root / 'summary.md'} (mirrored to {artifacts_root})", flush=True)
    return 0 if all(r.get("status") == "completed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
