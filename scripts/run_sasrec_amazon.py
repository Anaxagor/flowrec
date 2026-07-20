#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
# to enable flowrec imports normally
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import (
    AmazonReviewDataConfig,
    AmazonReviewDataModule,
)
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training import Trainer, TrainerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_hit_ndcg_coverage(
    model: torch.nn.Module,
    dm: AmazonReviewDataModule,
    *,
    ks=(5, 10, 20),
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    pad_item_id = getattr(dm.cfg, "pad_item_id", 0)
    num_items = dm.item_vocab_size
    catalog_size = max(num_items - 1, 1)

    hit_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    recommended_sets = {k: set() for k in ks}
    n_examples = 0

    for batch in dm.test_dataloader():
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch[key] = val.to(device, non_blocking=True)

        out = model(batch)
        scores = out.scores

        target_item_ids = batch.get("target_item_ids", batch.get("target_item_id"))
        if target_item_ids is None:
            raise KeyError("Batch must contain target_item_ids or target_item_id")

        targets = target_item_ids.long()
        valid = targets != pad_item_id

        if valid.sum().item() == 0:
            continue

        max_k = max(ks)
        topk = torch.topk(scores, k=max_k, dim=-1).indices

        topk_valid = topk[valid]
        targets_valid = targets[valid]

        n_examples += int(targets_valid.numel())

        for k in ks:
            kk = int(k)
            topk_k = topk_valid[:, :kk]

            matches = topk_k.eq(targets_valid.unsqueeze(1))
            hit = matches.any(dim=1)
            rank_pos = matches.float().argmax(dim=1)

            ndcg = torch.zeros_like(hit, dtype=torch.float32)

            if hit.any():
                ndcg[hit] = 1.0 / torch.log2(rank_pos[hit].float() + 2.0)

            hit_sum[k] += float(hit.float().sum().item())
            ndcg_sum[k] += float(ndcg.sum().item())

            uniq = torch.unique(topk_k).detach().cpu().tolist()
            recommended_sets[k].update(
                int(x) for x in uniq if int(x) != pad_item_id
            )

    if n_examples == 0:
        raise RuntimeError("No valid test examples found")

    metrics: dict[str, float] = {}

    for k in ks:
        metrics[f"HitRate@{k}"] = hit_sum[k] / n_examples
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / n_examples
        metrics[f"Coverage@{k}"] = len(recommended_sets[k]) / catalog_size

    metrics["num_test_examples"] = float(n_examples)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SASRec on Amazon datasets"
    )

    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beauty",
        choices=["Beauty", "Musical_Instruments"],
    )

    parser.add_argument("--max_history_len", type=int, default=20)

    # Dense-causal training (default): one row per user window, the causal
    # attention mask supervises every position in a single forward pass.
    # "prefix_target" restores the old one-example-per-prefix layout.
    parser.add_argument(
        "--train_example_layout",
        type=str,
        default="dense_causal",
        choices=["dense_causal", "prefix_target"],
    )
    parser.add_argument("--train_window_len", type=int, default=None)
    parser.add_argument("--train_window_stride", type=int, default=None)

    # Dense mode requires right padding (positions must index from the
    # sequence start); the datamodule rejects dense + left.
    parser.add_argument(
        "--history_padding",
        type=str,
        default="right",
        choices=["left", "right"],
    )

    # Only used with --train_example_layout=prefix_target.
    parser.add_argument("--train_last_n", type=int, default=20)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--min_item_interactions", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pos_drop", type=float, default=0.2)

    # Trainer
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--use_amp", action="store_true", default=True)

    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
    )

    parser.add_argument("--log_every_steps", type=int, default=200)
    parser.add_argument("--validate_every_epochs", type=int, default=2)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val/sft/ndcg@10",
    )
    parser.add_argument(
        "--monitor_mode",
        type=str,
        default="max",
        choices=["min", "max"],
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../artifacts/amazon_sasrec",
    )

    # Eval
    parser.add_argument("--k5", type=int, default=5)
    parser.add_argument("--k10", type=int, default=10)
    parser.add_argument("--k20", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    ks = (int(args.k5), int(args.k10), int(args.k20))

    # Parquet is the primary interaction format (see load_interactions_frame);
    # paths are repo-anchored so the script works from any working directory.
    if args.dataset == "Beauty":
        reviews_path = PROJECT_ROOT / "data/raw/dataframe_format/reviews_beauty_2014.parquet"
        metadata_path = PROJECT_ROOT / "data/raw/dataframe_format/meta_beauty_2014.parquet"
    else:
        reviews_path = PROJECT_ROOT / "data/raw/dataframe_format/reviews_instruments_2018.parquet"
        metadata_path = PROJECT_ROOT / "data/raw/dataframe_format/meta_instruments_2018.parquet"

    dataset_cfg = AmazonReviewDataConfig(
        #category=args.dataset,
        reviews_path=reviews_path,
        metadata_path=metadata_path,

        split_strategy="leave_one_out",

        global_history_policy="rolling",
        global_eval_target_mode="last_per_user",

        train_example_layout=args.train_example_layout,
        train_window_len=args.train_window_len,
        train_window_stride=args.train_window_stride,

        # Prefix-mode-only target selection (ignored in dense mode).
        train_all_prefixes=False,
        train_target_policy="last_n",
        train_last_n=args.train_last_n,

        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,

        max_history_len=args.max_history_len,

        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,

        num_workers=0,
        seed=args.seed,

        history_padding=args.history_padding,
    )

    dm = AmazonReviewDataModule(dataset_cfg)

    dm.prepare_data()
    dm.setup()

    model_cfg = SASRecConfig(
        item_vocab_size=dm.item_vocab_size,
        # Positional embeddings must cover dense training windows too.
        max_history_len=max(
            dataset_cfg.max_history_len, args.train_window_len or 0
        ),

        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,

        dropout=args.dropout,
        pos_drop=args.pos_drop,
    )

    model = SASRec(model_cfg)
    if args.train_example_layout not in model.supported_train_layouts:
        raise ValueError(
            f"SASRec does not support train_example_layout="
            f"{args.train_example_layout!r}; supported: {model.supported_train_layouts}."
        )

    method = SFTMethod(SFTConfig())

    trainer_cfg = TrainerConfig(
        max_epochs=args.max_epochs,

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        max_grad_norm=args.max_grad_norm,

        use_amp=bool(args.use_amp),
        amp_dtype=args.amp_dtype,

        log_every_steps=args.log_every_steps,

        validate_every_epochs=(
            args.validate_every_epochs
        ),

        monitor=args.monitor,
        monitor_mode=args.monitor_mode,

        early_stopping_patience=(
            args.early_stopping_patience
        ),

        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=trainer_cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=== Dataset summary ===")
    print(json.dumps(dm.summary(), indent=2, default=str))

    print("=== Model summary ===")
    print(json.dumps(asdict(model.cfg), indent=2))

    print("Model parameters (M):", model.num_parameters() / 1e6)

    trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
    )

    # Save checkpoint
    out_ckpt = (
        Path(args.checkpoint_dir)
        / f"final_sasrec_{args.dataset}_seed_{args.seed}.pt"
    )

    model.save_checkpoint(out_ckpt)

    # Evaluation
    metrics = evaluate_hit_ndcg_coverage(
        model,
        dm,
        ks=ks,
    )

    metrics["seed"] = args.seed
    metrics["dataset"] = args.dataset
    metrics["history_padding"] = args.history_padding
    metrics["train_example_layout"] = args.train_example_layout

    # Save metrics
    metrics_dir = Path("../artifacts/metrics/amazon_sasrec")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_csv = metrics_dir / f"{args.dataset.lower()}_metrics_seed={args.seed}.csv"

    row_df = pd.DataFrame([metrics])

    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        row_df = pd.concat([prev, row_df], ignore_index=True)

    row_df.to_csv(out_csv, index=False)

    print("=== Test metrics ===")

    for k in sorted(
        [
            k
            for k in metrics.keys()
            if k not in {"seed", "dataset", "history_padding", "train_example_layout"}
        ]
    ):
        print(f"  {k}: {metrics[k]:.6f}")

    print(f"Saved metrics to: {out_csv}")


if __name__ == "__main__":
    main()