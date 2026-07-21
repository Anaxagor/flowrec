"""Benchmark SASRec training with prefix_target vs dense_causal data layouts.

Subsamples users from an Amazon reviews parquet, then trains the same SASRec
twice on the same interactions:

  1. prefix_target + train_target_policy="all"  (old logic: one example per
     history prefix, loss only at the last position), and
  2. dense_causal (new logic: one example per user window, causal attention
     supervises every position in a single forward pass).

Both runs share the eval pipeline (prefix-schema val/test batches), so
inference numbers are expected to match; the training numbers are the point.

Usage:
    python scripts/benchmark_sasrec_layouts.py \
        --reviews-path data/raw/reviews_beauty_2014.parquet \
        --num-users 3000 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.trainer import Trainer, TrainerConfig, move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "reviews_beauty_2014.parquet",
    )
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "experiments" / "benchmark_sasrec_layouts")
    parser.add_argument("--num-users", type=int, default=3000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_subset(args: argparse.Namespace) -> Path:
    subset_path = args.work_dir / f"subset_u{args.num_users}_seed{args.seed}.parquet"
    if subset_path.exists():
        return subset_path

    frame = pd.read_parquet(args.reviews_path)
    user_col = "reviewerID" if "reviewerID" in frame.columns else "user_id"
    counts = frame[user_col].value_counts()
    eligible = counts[counts >= args.min_user_interactions].index
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, size=min(args.num_users, len(eligible)), replace=False)
    subset = frame[frame[user_col].isin(set(chosen))].reset_index(drop=True)

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(subset_path)
    print(f"Subset: {len(chosen)} users, {len(subset)} interactions -> {subset_path}")
    return subset_path


def build_datamodule(args: argparse.Namespace, subset_path: Path, layout: str) -> AmazonReviewDataModule:
    cfg = AmazonReviewDataConfig(
        reviews_path=subset_path,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=0,
        train_example_layout=layout,
        train_target_policy="all",
        train_all_prefixes=True,
        max_history_len=args.max_history_len,
        history_padding="right",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
        seed=args.seed,
    )
    data = AmazonReviewDataModule(cfg)
    data.setup()
    return data


def run_one(args: argparse.Namespace, subset_path: Path, layout: str) -> dict:
    seed_everything(args.seed)
    data = build_datamodule(args, subset_path, layout)
    stats = data.summary()

    seed_everything(args.seed)
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(data.item_vocab_size),
            max_history_len=args.max_history_len,
            d_model=64,
            nhead=2,
            num_layers=2,
            dropout=0.2,
            pos_drop=0.2,
        )
    )
    method = SFTMethod(
        SFTConfig(
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_item_metrics=True,
            compute_token_metrics=False,
            evaluate_generation=False,
        )
    )
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=args.epochs,
            validate_every_epochs=None,  # timing runs: validate explicitly below
            save_best=False,
            save_last=False,
            log_every_steps=0,
            progress_bar=False,
        ),
        device=args.device,
        lr=3e-4,
        weight_decay=0.0,
    )

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    fit_start = time.perf_counter()
    fit_result = trainer.fit(train_loader)
    total_train_time = time.perf_counter() - fit_start
    epoch_times = [rec["epoch_time_sec"] for rec in trainer.state.history]

    val_start = time.perf_counter()
    val_metrics = trainer.evaluate(val_loader, stage="val")
    val_time = time.perf_counter() - val_start

    test_start = time.perf_counter()
    test_metrics = trainer.evaluate(test_loader, stage="test")
    test_time = time.perf_counter() - test_start

    # Pure model inference over the test split: forward passes only, no
    # metric computation. Identical prefix-schema batches for both layouts.
    model.eval()
    n_examples = 0
    forward_start = time.perf_counter()
    with torch.no_grad():
        for raw_batch in test_loader:
            batch = move_to_device(raw_batch, trainer.device)
            model(batch)
            n_examples += int(batch["target_item_ids"].size(0))
    forward_time = time.perf_counter() - forward_start

    return {
        "layout": layout,
        "num_train_examples": stats["num_train_examples"],
        "num_val_examples": stats["num_val_examples"],
        "num_test_examples": stats["num_test_examples"],
        "num_users": stats["num_users"],
        "num_items": stats["num_items"],
        "steps_per_epoch": len(train_loader),
        "epoch_time_sec": epoch_times,
        "mean_epoch_time_sec": float(np.mean(epoch_times)),
        "total_train_time_sec": total_train_time,
        "final_train_loss": fit_result.train_metrics.get("train/loss")
        or fit_result.train_metrics.get("train/sft/loss"),
        "val_eval_time_sec": val_time,
        "test_eval_time_sec": test_time,
        "test_forward_only_time_sec": forward_time,
        "test_forward_examples_per_sec": n_examples / forward_time if forward_time else None,
        "val_metrics": {k: v for k, v in val_metrics.items() if "@" in k or k.endswith("loss")},
        "test_metrics": {k: v for k, v in test_metrics.items() if "@" in k or k.endswith("loss")},
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    subset_path = prepare_subset(args)

    results = {}
    for layout in ("prefix_target", "dense_causal"):
        print(f"\n=== layout={layout} ===")
        results[layout] = run_one(args, subset_path, layout)
        r = results[layout]
        print(
            f"train examples: {r['num_train_examples']}, steps/epoch: {r['steps_per_epoch']}, "
            f"mean epoch: {r['mean_epoch_time_sec']:.2f}s, total train: {r['total_train_time_sec']:.2f}s"
        )
        print(
            f"val eval: {r['val_eval_time_sec']:.2f}s, test eval: {r['test_eval_time_sec']:.2f}s, "
            f"forward-only: {r['test_forward_only_time_sec']:.2f}s"
        )

    p, d = results["prefix_target"], results["dense_causal"]
    summary = {
        "config": {
            "num_users": args.num_users,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_history_len": args.max_history_len,
            "device": args.device,
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
        },
        "results": results,
        "speedup": {
            "train_examples_ratio": p["num_train_examples"] / max(d["num_train_examples"], 1),
            "epoch_time_ratio": p["mean_epoch_time_sec"] / max(d["mean_epoch_time_sec"], 1e-9),
            "total_train_time_ratio": p["total_train_time_sec"] / max(d["total_train_time_sec"], 1e-9),
            "test_eval_time_ratio": p["test_eval_time_sec"] / max(d["test_eval_time_sec"], 1e-9),
        },
    }

    out_path = args.work_dir / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved: {out_path}")
    print(json.dumps(summary["speedup"], indent=2))


if __name__ == "__main__":
    main()
