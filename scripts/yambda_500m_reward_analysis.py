#!/usr/bin/env python
"""Yambda 500m reward analysis (script version of notebook section 7).

Two parts, mirroring examples/yambda_reward_analysis.ipynb:

1. Item-similarity measures over the trained 500m artifacts (semantic-ID
   token match by level, prefix depth, token-Gower vs audio cosine). Runs on
   any machine: only catalog SIDs + audio embeddings are materialized.
2. Per-user full-catalog distributions of the three reward types (SASRec,
   embedding-cosine similarity, interaction-log). Rebuilds the exact training
   datamodule from configs/exp1-standard_params/dataset_yambda.yaml — the
   ~500M-row pandas pass needs a full-memory machine (the training server).
   Skip it with --similarity-only on smaller machines.

All charts (PNG) and tables (CSV/JSON) are saved under --output-dir
(default artifacts/yambda_500m_reward_analysis).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

# Chart style: dataviz reference palette (validated categorical slots 1-2).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/yambda_500m_reward_analysis",
    )
    p.add_argument(
        "--sid-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/semantic_ids/exp1-standard_params/yambda_500m_listens",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints/exp1-standard_params/sasrec/yambda/yambda_500m_listens/seed_42/best.pt",
    )
    p.add_argument(
        "--dataset-config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp1-standard_params/dataset_yambda.yaml",
    )
    p.add_argument("--dataset-key", default="yambda_500m_listens")
    p.add_argument(
        "--similarity-only",
        action="store_true",
        help="Skip the per-user reward analysis (the 500m datamodule pass needs a full-memory machine).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-users", type=int, default=8, help="Sampled test users for reward distributions.")
    p.add_argument("--seed", type=int, default=42)
    # Similarity-analysis sampling sizes.
    p.add_argument("--pool-size", type=int, default=30_000)
    p.add_argument("--num-anchors", type=int, default=400)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--num-random-pairs", type=int, default=200_000)
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    sid_meta = json.load(open(args.sid_dir / "semantic_id_metadata.json"))
    sid_map = json.load(open(args.sid_dir / "semantic_ids.json"))
    sid_items = list(sid_map.keys())
    sid_tokens = np.asarray([sid_map[k] for k in sid_items], dtype=np.int64)

    ckpt_meta = json.load(open(str(args.checkpoint) + ".json"))
    vocab = int(ckpt_meta["model_config"]["item_vocab_size"])
    if len(sid_items) != vocab - 1:
        raise ValueError(
            f"SID catalog ({len(sid_items):,}) does not match checkpoint vocab ({vocab:,} - 1)."
        )

    dataset_cfg = yaml.safe_load(open(args.dataset_config))
    print(
        f"semantic IDs: {len(sid_items):,} items, sid_len={sid_tokens.shape[1]}, "
        f"levels {sid_meta['rqvae_config']['num_codebooks']}×{sid_meta['rqvae_config']['codebook_size']}"
    )
    print(f"SASRec checkpoint vocab: {vocab:,}")
    return {
        "sid_meta": sid_meta,
        "sid_items": sid_items,
        "sid_tokens": sid_tokens,
        "ckpt_model_config": ckpt_meta["model_config"],
        "vocab": vocab,
        "dataset_cfg": dataset_cfg,
    }


def load_catalog_embeddings(
    embeddings_path: Path, sid_items: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Streaming pyarrow scan: only catalog rows are materialized."""
    dataset = pa_ds.dataset(embeddings_path)
    id_type = dataset.schema.field("item_id").type
    catalog_keys = pa.array([int(x) for x in sid_items], type=id_type)
    scanner = dataset.scanner(
        columns=["item_id", "normalized_embed"],
        filter=pa_ds.field("item_id").isin(catalog_keys),
        batch_size=200_000,
    )
    ids_chunks, emb_chunks = [], []
    for record_batch in scanner.to_batches():
        if record_batch.num_rows == 0:
            continue
        ids_chunks.append(record_batch.column("item_id").to_numpy(zero_copy_only=False))
        emb_chunks.append(np.stack(record_batch.column("normalized_embed").to_pylist()))
    emb_item_ids = np.concatenate(ids_chunks)
    emb_matrix = np.concatenate(emb_chunks).astype(np.float32)
    emb_matrix /= np.clip(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-9, None)
    row_of_item = {str(item): row for row, item in enumerate(emb_item_ids)}
    print(f"embeddings loaded for {len(row_of_item):,}/{len(sid_items):,} catalog items")
    return emb_matrix, row_of_item


def similarity_analysis(args: argparse.Namespace, art: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sid_items, sid_tokens = art["sid_items"], art["sid_tokens"]
    emb_matrix, row_of_item = art["emb_matrix"], art["row_of_item"]
    codebook_size = int(art["sid_meta"]["rqvae_config"]["codebook_size"])

    covered = [item for item in sid_items if item in row_of_item]
    sid_row_of_item = {item: row for row, item in enumerate(sid_items)}
    pool_rows = rng.choice(len(covered), size=min(args.pool_size, len(covered)), replace=False)
    pool_items = [covered[i] for i in pool_rows]
    pool_emb = emb_matrix[[row_of_item[i] for i in pool_items]]
    pool_sid = sid_tokens[[sid_row_of_item[i] for i in pool_items]]

    anchor_idx = rng.choice(len(pool_items), size=args.num_anchors, replace=False)
    sims = pool_emb[anchor_idx] @ pool_emb.T
    sims[np.arange(args.num_anchors), anchor_idx] = -np.inf
    neighbor_idx = np.argpartition(-sims, args.top_k, axis=1)[:, : args.top_k]

    pairs_a = np.repeat(anchor_idx, args.top_k)
    pairs_b = neighbor_idx.reshape(-1)
    rand_a = rng.integers(0, len(pool_items), size=args.num_random_pairs)
    rand_b = rng.integers(0, len(pool_items), size=args.num_random_pairs)
    keep = rand_a != rand_b
    rand_a, rand_b = rand_a[keep], rand_b[keep]

    def level_match_rates(a, b):
        return [float((pool_sid[a, level] == pool_sid[b, level]).mean()) for level in range(3)]

    def prefix_depth(a, b):
        depth = (pool_sid[a, 0] == pool_sid[b, 0]).astype(int)
        depth += (depth == 1) & (pool_sid[a, 1] == pool_sid[b, 1])
        depth += (depth == 2) & (pool_sid[a, 2] == pool_sid[b, 2])
        return depth

    neigh_rates = level_match_rates(pairs_a, pairs_b)
    rand_rates = level_match_rates(rand_a, rand_b)
    depth_neigh = prefix_depth(pairs_a, pairs_b)

    level_table = pd.DataFrame(
        {
            "level": [1, 2, 3],
            "neighbor_match_rate": neigh_rates,
            "random_match_rate": rand_rates,
            "lift": [n / max(r, 1e-9) for n, r in zip(neigh_rates, rand_rates)],
        }
    )
    level_table.to_csv(out_dir / "token_match_by_level.csv", index=False)
    depth_counts = {int(d): int(c) for d, c in zip(*np.unique(depth_neigh, return_counts=True))}
    write_json(out_dir / "neighbor_prefix_depth.json", depth_counts)
    print(level_table.round(4).to_string(index=False))
    print("neighbor shared-prefix depth:", depth_counts)

    cos_rand = np.einsum("ij,ij->i", pool_emb[rand_a], pool_emb[rand_b])
    depth_rand = prefix_depth(rand_a, rand_b)
    codes_a = pool_sid[rand_a, :3] % codebook_size
    codes_b = pool_sid[rand_b, :3] % codebook_size
    gower_rand = 1.0 - (np.abs(codes_a - codes_b) / codebook_size).mean(axis=1)

    bins = np.quantile(cos_rand, np.linspace(0, 1, 11))
    bucket = np.clip(np.searchsorted(bins, cos_rand, side="right") - 1, 0, 9)
    rows = []
    for b in range(10):
        m = bucket == b
        rows.append(
            {
                "cos_decile": b + 1,
                "cos_mean": float(cos_rand[m].mean()),
                "level1_match_pct": 100 * float((pool_sid[rand_a[m], 0] == pool_sid[rand_b[m], 0]).mean()),
                "mean_prefix_depth": float(depth_rand[m].mean()),
                "gower_similarity": float(gower_rand[m].mean()),
                "n_pairs": int(m.sum()),
            }
        )
    decile_table = pd.DataFrame(rows).set_index("cos_decile")
    decile_table.to_csv(out_dir / "cosine_decile_table.csv")
    print(decile_table.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(decile_table["cos_mean"], decile_table["level1_match_pct"],
                 color=BLUE, marker="o", markersize=5, linewidth=2)
    axes[0].set_xlabel("audio cosine (decile mean)")
    axes[0].set_ylabel("level-1 token match, %")
    axes[0].set_title("Coarse token agreement vs audio cosine")
    axes[1].plot(decile_table["cos_mean"], decile_table["gower_similarity"],
                 color=BLUE, marker="o", markersize=5, linewidth=2)
    axes[1].set_xlabel("audio cosine (decile mean)")
    axes[1].set_ylabel("token-Gower similarity")
    axes[1].set_title("Token-index Gower vs audio cosine")
    plt.tight_layout()
    fig.savefig(out_dir / "cosine_vs_tokens.png", dpi=150)
    plt.close(fig)

    x = np.arange(3)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b1 = ax.bar(x - width / 2, [100 * r for r in neigh_rates], width,
                label=f"audio top-{args.top_k} neighbors", color=BLUE)
    b2 = ax.bar(x + width / 2, [100 * r for r in rand_rates], width,
                label="random pairs", color=ORANGE)
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=9, color=INK)
    ax.bar_label(b2, fmt="%.2f%%", padding=2, fontsize=9, color=INK)
    ax.set_xticks(x, ["level 1", "level 2", "level 3"])
    ax.set_ylabel("% of pairs with equal token")
    ax.set_title("Semantic-ID token match by level (500m catalog)")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / "token_match_by_level.png", dpi=150)
    plt.close(fig)

    return {
        "level_match": level_table.to_dict(orient="records"),
        "neighbor_prefix_depth": depth_counts,
    }


def build_datamodule(
    args: argparse.Namespace,
    art: dict[str, Any],
    *,
    interactions_path: Path,
) -> YambdaDataModule:
    d = art["dataset_cfg"]
    entry = d["datasets"][args.dataset_key]
    dm = YambdaDataModule(
        YambdaDataConfig(
            interactions_path=interactions_path,
            embeddings_path=PROJECT_ROOT / d["embeddings_path"],
            use_hf=False,
            dataset_type=d["dataset_type"],
            dataset_size=entry["dataset_size"],
            event_file=d["event_file"],
            hf_recent_months=d["hf_recent_months"],
            positive_criterion=d["positive_criterion"],
            organic_filter=d["organic_filter"],
            split_strategy=d["split_strategy"],
            min_history_len=d["min_history_len"],
            deduplicate_consecutive=d["deduplicate_consecutive"],
            filter_cold_items_after_split=d["filter_cold_items_after_split"],
            cold_item_train_source=d["cold_item_train_source"],
            min_user_interactions=d["min_user_interactions"],
            min_item_interactions=d["min_item_interactions"],
            max_history_len=d["max_history_len"],
            history_padding="left",
            batch_size=args.num_users,
            eval_batch_size=args.num_users,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            include_raw_in_batch=False,
            seed=args.seed,
        )
    )
    dm.setup()
    return dm


def reward_analysis(args: argparse.Namespace, art: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from flowrec.data.interaction_store import build_positive_history_store
    from flowrec.models.sasrec import SASRec, SASRecConfig
    from flowrec.rewards.interaction import (
        InteractionRewardConfig,
        InteractionRewardModel,
    )
    from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
    from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel
    from flowrec.trajectories.semantic import SemanticTrajectoryBatch

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    vocab = art["vocab"]

    entry = art["dataset_cfg"]["datasets"][args.dataset_key]
    # The datamodule pushes the 2-month window down to the parquet scan, so
    # only the trailing-window rows ever reach pandas.
    print("building the 500m datamodule ...")
    dm = build_datamodule(
        args, art, interactions_path=PROJECT_ROOT / entry["interactions_path"]
    )
    if dm.item_vocab_size != vocab:
        raise ValueError(
            f"catalog mismatch: datamodule={dm.item_vocab_size:,}, checkpoint={vocab:,}"
        )
    summary = dm.summary()
    write_json(out_dir / "dm_summary.json", summary)
    write_json(out_dir / "store_summary.json", summary.get("interaction_store"))
    print(f"datamodule: users={dm.num_users:,} items={dm.num_items:,}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sasrec = SASRec(SASRecConfig(**payload["model_config"]))
    sasrec.load_state_dict(payload["model_state_dict"])
    sasrec.eval().to(device)

    emb_matrix, row_of_item = art["emb_matrix"], art["row_of_item"]
    item_emb_table = torch.zeros((vocab, emb_matrix.shape[1]), dtype=torch.float32)
    for raw_item, internal in dm.item2id.items():
        row = row_of_item.get(raw_item)
        if row is not None:
            item_emb_table[internal] = torch.from_numpy(emb_matrix[row])

    reward_models = {
        "sasrec": SASRecRewardModel(sasrec, SASRecRewardConfig(transform="softplus")).to(device),
        "similarity": SimilarityRewardModel(
            SimilarityRewardConfig(mode="embedding_cosine"), item_embeddings=item_emb_table
        ).to(device),
        "interaction_log": InteractionRewardModel(
            dm.interaction_store,
            InteractionRewardConfig(dataset="yambda"),
            # Engagement scoring reads full (criterion-passing) listens
            # through the positive-history store; without it they are invisible.
            positive_store=build_positive_history_store(
                dm.interactions,
                num_users=int(dm.num_users),
                item_vocab_size=int(dm.item_vocab_size),
            ),
        ).to(device),
    }

    batch = next(iter(dm.test_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    bsz = batch["user_ids"].shape[0]
    candidates = torch.arange(1, vocab, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
    is_positive = candidates.eq(batch["target_item_ids"].unsqueeze(1))
    trajectories = SemanticTrajectoryBatch(
        candidate_token_ids=torch.zeros(bsz, candidates.shape[1], 1, dtype=torch.long, device=device),
        candidate_token_mask=torch.ones(bsz, candidates.shape[1], 1, dtype=torch.bool, device=device),
        is_positive=is_positive,
        extra={"candidate_item_ids": candidates},
    )

    reward_rows: dict[str, np.ndarray] = {}
    percentile_rows = []
    for name, model in reward_models.items():
        with torch.no_grad():
            rewards = model(batch, trajectories).float().cpu().numpy()
        reward_rows[name] = rewards
        for col in range(bsz):
            r = rewards[col]
            target_val = float(r[int(batch["target_item_ids"][col].cpu()) - 1])
            percentile_rows.append(
                {
                    "reward": name,
                    "user_id": int(batch["user_ids"][col].cpu()),
                    "min": float(r.min()),
                    "p50": float(np.median(r)),
                    "p90": float(np.quantile(r, 0.9)),
                    "p99": float(np.quantile(r, 0.99)),
                    "max": float(r.max()),
                    "target_value": target_val,
                    "target_percentile": float((r < target_val).mean() * 100),
                }
            )
        r = rewards
        print(
            f"{name}: min={r.min():.4f} p50={np.median(r):.4f} "
            f"p99={np.quantile(r, 0.99):.4f} max={r.max():.4f}"
        )
    percentile_table = pd.DataFrame(percentile_rows)
    percentile_table.to_csv(out_dir / "reward_percentiles.csv", index=False)

    n_cols = min(bsz, 6)
    fig, axes = plt.subplots(len(reward_rows), n_cols, figsize=(2.6 * n_cols, 7),
                             sharex="row", squeeze=False)
    for row_i, (name, rewards) in enumerate(reward_rows.items()):
        for col in range(n_cols):
            ax = axes[row_i][col]
            ax.hist(rewards[col], bins=60, color=BLUE, log=True)
            target_val = rewards[col][int(batch["target_item_ids"][col].cpu()) - 1]
            ax.axvline(target_val, color=ORANGE, linewidth=1.4)
            if row_i == 0:
                ax.set_title(f"user {int(batch['user_ids'][col].cpu())}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}\n(log count)")
    fig.suptitle("Full-catalog reward distributions per user — orange line = positive target")
    plt.tight_layout()
    fig.savefig(out_dir / "per_user_reward_distributions.png", dpi=150)
    plt.close(fig)

    return {
        "dm_users": dm.num_users,
        "dm_items": dm.num_items,
        "store_events": len(dm.interaction_store) if dm.interaction_store is not None else 0,
        "sampled_users": [int(u) for u in batch["user_ids"].cpu().tolist()],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    art = load_artifacts(args)
    emb_path = PROJECT_ROOT / art["dataset_cfg"]["embeddings_path"]
    art["emb_matrix"], art["row_of_item"] = load_catalog_embeddings(emb_path, art["sid_items"])

    summary: dict[str, Any] = {
        "catalog_items": len(art["sid_items"]),
        "checkpoint_vocab": art["vocab"],
        "embedding_coverage": len(art["row_of_item"]),
    }
    summary["similarity"] = similarity_analysis(args, art, out / "similarity")

    if args.similarity_only:
        print("--similarity-only: skipping the per-user reward analysis.")
    else:
        summary["rewards"] = reward_analysis(args, art, out / "rewards")

    write_json(out / "summary.json", summary)
    print(f"artifacts saved under {out}")


if __name__ == "__main__":
    main()
