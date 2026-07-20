"""SASRec on Amazon Instruments 2018 with dense-causal training.

Training uses train_example_layout="dense_causal": one row per user window,
and the causal self-attention mask lets a single forward pass supervise every
position (label[i] = input[i+1]).  This replaces the old per-prefix layout
(train_all_prefixes / train_target_policy), which materialized one example per
target and only used the last position of each forward pass.

Evaluation is unchanged: val/test batches keep the prefix schema, so the
ranking metrics are directly comparable to earlier per-prefix runs.

Note on batch size: a dense row carries ~history_len supervised targets, so at
equal batch size the optimizer takes far fewer steps per epoch than a
per-prefix run.  The train batch size below is deliberately smaller than the
old 512 to keep the optimizer-step count comparable (see
scripts/benchmark_sasrec_layouts.py for the measured effect).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------

SEEDS = (42, 43)
KS = (5, 10, 20)

REVIEWS_PATH = PROJECT_ROOT / "data/raw/dataframe_format/reviews_instruments_2018.parquet"
METADATA_PATH = PROJECT_ROOT / "data/raw/dataframe_format/meta_instruments_2018.parquet"
ITEM_KEY = "asin"

MAX_HISTORY_LEN = 20
# Dense-causal windowing. None -> window length = MAX_HISTORY_LEN and
# non-overlapping tiling. Set TRAIN_WINDOW_STRIDE < TRAIN_WINDOW_LEN to give
# window-boundary labels more context at proportional extra compute.
TRAIN_WINDOW_LEN: int | None = None
TRAIN_WINDOW_STRIDE: int | None = None

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512

OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/sasrec_instruments_dense_causal"
PER_SEED_CSV = OUTPUT_DIR / "per_seed_test_metrics.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary_test_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary_test_metrics.json"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_dtype_for_gpu() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def to_jsonable(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int, str)):
            out[key] = value
        elif value is None:
            continue
        else:
            try:
                out[key] = float(value)
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def build_datamodule(seed: int) -> AmazonReviewDataModule:
    data_cfg = AmazonReviewDataConfig(
        reviews_path=REVIEWS_PATH,
        metadata_path=METADATA_PATH,
        item_key=ITEM_KEY,

        split_strategy="leave_one_out",

        # Filtering.
        min_user_interactions=5,
        min_item_interactions=5,
        deduplicate_consecutive=False,
        filter_cold_items_after_split=False,

        # Dense-causal training layout: every position in the train region is
        # supervised exactly once; no per-prefix example explosion and no
        # train_target_policy subsampling.
        train_example_layout="dense_causal",
        train_window_len=TRAIN_WINDOW_LEN,
        train_window_stride=TRAIN_WINDOW_STRIDE,

        # Batching. Right padding is required for dense training: position i
        # in the padded tensor must be position i in the window.
        max_history_len=MAX_HISTORY_LEN,
        history_padding="right",
        batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        include_raw_in_batch=False,

        seed=seed,
    )

    dm = AmazonReviewDataModule(data_cfg)
    dm.setup()

    print("Dataset summary:")
    print(json.dumps(dm.summary(), indent=2))
    return dm


# ---------------------------------------------------------------------
# Model / method / trainer
# ---------------------------------------------------------------------

def build_model(dm: AmazonReviewDataModule) -> SASRec:
    # Positional embeddings must cover the training window, which may exceed
    # the eval history length.
    max_positions = max(MAX_HISTORY_LEN, TRAIN_WINDOW_LEN or 0)
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(dm.item_vocab_size),
            max_history_len=max_positions,
            d_model=128,
            nhead=2,
            num_layers=3,
            dropout=0.3,
            pos_drop=0.3,
        )
    )

    layout = dm.cfg.train_example_layout
    if layout not in model.supported_train_layouts:
        raise ValueError(
            f"{model.__class__.__name__} does not support train_example_layout="
            f"{layout!r}; supported: {model.supported_train_layouts}."
        )

    print(model)
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    return model


def build_method() -> SFTMethod:
    return SFTMethod(
        SFTConfig(
            loss_weight=1.0,
            metric_prefix="sft",
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_item_metrics=True,
            item_topk=KS,
            compute_token_metrics=False,
            evaluate_generation=False,
            pad_item_id=0,
        )
    )


def make_epoch_logger():
    """Print loss and ndcg@10 for train/val/test at the end of every epoch."""

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") != "epoch_end":
            return

        parts = [f"epoch {payload.get('epoch')}"]

        train = payload.get("train") or {}
        train_loss = train.get("train/sft/loss", train.get("train/loss"))
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")

        for stage in ("val", "test"):
            metrics = payload.get(stage) or {}
            loss = metrics.get(f"{stage}/sft/loss", metrics.get(f"{stage}/loss"))
            ndcg = metrics.get(f"{stage}/sft/ndcg@10")
            if loss is not None:
                parts.append(f"{stage}_loss={loss:.4f}")
            if ndcg is not None:
                parts.append(f"{stage}_ndcg@10={ndcg:.4f}")

        print(" | ".join(parts), flush=True)

    return logger


def build_trainer(
    *,
    model: SASRec,
    method: SFTMethod,
    seed: int,
    device: str,
) -> Trainer:
    checkpoint_dir = OUTPUT_DIR / f"seed_{seed}" / "checkpoints"

    cfg = TrainerConfig(
        max_epochs=500,

        validate_every_epochs=1,
        validate_every_steps=None,

        monitor="val/sft/ndcg@10",
        monitor_mode="max",
        early_stopping_patience=10,
        min_delta=1e-5,

        gradient_accumulation_steps=1,
        max_grad_norm=1.0,

        # AMP is applied only on CUDA devices; harmless elsewhere.
        use_amp=True,
        amp_dtype=amp_dtype_for_gpu(),

        checkpoint_dir=checkpoint_dir,
        save_best=True,
        save_last=True,

        log_every_steps=100,
        progress_bar=True,
        prefix_method_metrics_with_stage=True,
    )

    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=cfg,
        device=device,
        lr=1e-3,
        weight_decay=1e-6,
        logger=make_epoch_logger(),
    )


# ---------------------------------------------------------------------
# Manual top-k metrics with coverage
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_topk(
    model: SASRec,
    dataloader,
    *,
    device: str,
    num_items: int,
    ks: tuple[int, ...] = KS,
    pad_item_id: int = 0,
    exclude_history: bool = False,
) -> dict[str, float]:
    model.eval()
    device_t = torch.device(device)

    max_k = max(ks)

    total = 0
    hit_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    coverage_mask = {k: torch.zeros(num_items, dtype=torch.bool) for k in ks}

    for raw_batch in dataloader:
        batch = {
            k: v.to(device_t, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in raw_batch.items()
        }

        output = model(batch)
        scores = output.scores

        if scores is None:
            raise RuntimeError("SASRec returned no scores.")

        scores = scores.clone()
        scores[:, pad_item_id] = -torch.inf

        if exclude_history:
            history = batch["history_item_ids"].long()
            history_mask = batch["history_mask"].bool()
            seen = history.masked_fill(~history_mask, pad_item_id)

            scores.scatter_(dim=1, index=seen, value=-torch.inf)
            scores[:, pad_item_id] = -torch.inf

        recs = torch.topk(scores, k=max_k, dim=1).indices.cpu()
        target = batch["target_item_ids"].detach().cpu().long()

        valid = target != pad_item_id
        valid &= target >= 0
        valid &= target < num_items

        if valid.sum() == 0:
            continue

        target = target[valid]
        recs = recs[valid]

        bsz = target.size(0)
        total += bsz

        for k in ks:
            topk = recs[:, :k]

            matches = topk.eq(target.unsqueeze(1))
            hit = matches.any(dim=1)

            hit_sum[k] += float(hit.float().sum().item())

            rank = matches.float().argmax(dim=1)

            ndcg = torch.zeros(bsz, dtype=torch.float32)
            if hit.any():
                ndcg[hit] = 1.0 / torch.log2(rank[hit].float() + 2.0)

            ndcg_sum[k] += float(ndcg.sum().item())

            valid_recs = topk[
                (topk != pad_item_id)
                & (topk >= 0)
                & (topk < num_items)
            ]

            if valid_recs.numel() > 0:
                coverage_mask[k][valid_recs.unique()] = True

    if total == 0:
        raise RuntimeError("No valid test examples found.")

    catalog_size = max(num_items - 1, 1)

    metrics: dict[str, float] = {
        "num_test_examples": float(total),
        "num_items": float(catalog_size),
    }

    for k in ks:
        metrics[f"hit_rate@{k}"] = hit_sum[k] / total
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / total
        metrics[f"coverage@{k}"] = float(coverage_mask[k][1:].float().mean().item())

    return metrics


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        f"hit_rate@{k}" for k in KS
    ] + [
        f"ndcg@{k}" for k in KS
    ] + [
        f"coverage@{k}" for k in KS
    ] + [
        "num_test_examples",
        "num_items",
    ]

    summary: dict[str, Any] = {
        "dataset": "instruments",
        "model": "sasrec",
        "train_example_layout": "dense_causal",
        "seeds": ",".join(str(seed) for seed in SEEDS),
    }

    for metric in metric_keys:
        values = [
            float(row[metric])
            for row in rows
            if metric in row and row[metric] is not None
        ]

        if not values:
            continue

        arr = np.asarray(values, dtype=np.float64)

        summary[f"{metric}_mean"] = float(arr.mean())
        summary[f"{metric}_std"] = float(arr.std(ddof=0))
        summary[f"{metric}_min"] = float(arr.min())
        summary[f"{metric}_max"] = float(arr.max())
        summary[f"{metric}_range"] = float(arr.max() - arr.min())

    return [summary]


# ---------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------

def run_one_seed(seed: int, device: str) -> dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"Running SASRec instruments experiment (dense_causal) | seed={seed}")
    print(f"{'=' * 80}")

    set_seed(seed)
    dm = build_datamodule(seed)

    model = build_model(dm)
    method = build_method()
    trainer = build_trainer(
        model=model,
        method=method,
        seed=seed,
        device=device,
    )

    fit_result = trainer.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        test_loader=dm.test_dataloader(),
    )

    print("Fit result:")
    print(f"  best_checkpoint: {fit_result.best_checkpoint_path}")
    print(f"  last_checkpoint: {fit_result.last_checkpoint_path}")
    print("  val_metrics:")
    print(json.dumps(to_jsonable(fit_result.val_metrics), indent=2))

    if fit_result.best_checkpoint_path is not None:
        print(f"Loading best checkpoint: {fit_result.best_checkpoint_path}")
        trainer.load_checkpoint(
            fit_result.best_checkpoint_path,
            load_optimizer=False,
            load_scheduler=False,
            strict=True,
        )

    trainer_test_metrics = to_jsonable(trainer.test(dm.test_dataloader()))

    topk_metrics = evaluate_topk(
        model,
        dm.test_dataloader(),
        device=device,
        num_items=int(dm.item_vocab_size),
        ks=KS,
        pad_item_id=0,
        exclude_history=False,
    )

    row: dict[str, Any] = {
        "dataset": "instruments",
        "model": "sasrec",
        "train_example_layout": "dense_causal",
        "seed": seed,
        **topk_metrics,
    }

    # Keep trainer metrics too, but prefix them to avoid name collisions.
    for key, value in trainer_test_metrics.items():
        row[f"trainer/{key}"] = value

    seed_output_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)

    print("Test metrics:")
    print(json.dumps(row, indent=2))

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"AMP dtype: {amp_dtype_for_gpu()}")

    per_seed_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        row = run_one_seed(seed, device=device)
        per_seed_rows.append(row)

        # Save after every seed so partial results are not lost.
        write_csv(PER_SEED_CSV, per_seed_rows)

    summary_rows = summarize_rows(per_seed_rows)

    write_csv(SUMMARY_CSV, summary_rows)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_seed": per_seed_rows,
                "summary": summary_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"\nSaved per-seed metrics to: {PER_SEED_CSV}")
    print(f"Saved summary metrics to:  {SUMMARY_CSV}")
    print(f"Saved summary JSON to:     {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
