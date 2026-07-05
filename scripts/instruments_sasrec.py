from __future__ import annotations

import csv
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.training.trainer import Trainer, TrainerConfig

try:
    from flowrec.models.sasrec import SASRecModel as SASRec
except ImportError:
    from flowrec.models.sasrec import SASRec

from flowrec.models.sasrec import SASRecConfig


# ---------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------

SEEDS = (42, 43)
KS = (5, 10, 20)

REVIEWS_PATH = PROJECT_ROOT / "data/raw/2018/Musical_Instruments.json.gz"
METADATA_PATH = PROJECT_ROOT / "data/raw/2018/meta_Musical_Instruments.json.gz"
ITEM_KEY = "asin"

OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/sasrec_instruments_gpu_3seeds"
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


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This script is configured for GPU training."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return "cuda"


def amp_dtype_for_gpu() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def make_config(cls, kwargs: dict[str, Any], *, required: tuple[str, ...] = ()):
    if not is_dataclass(cls):
        return cls(**kwargs)

    supported = {f.name for f in fields(cls)}
    missing_required = [name for name in required if name not in supported]

    if missing_required:
        raise RuntimeError(
            f"{cls.__name__} does not support required fields: {missing_required}."
        )

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    ignored = sorted(set(kwargs) - supported)

    if ignored:
        print(f"[Config warning] Ignoring unsupported fields for {cls.__name__}: {ignored}")

    return cls(**filtered)


def get_item_vocab_size(dm: AmazonReviewDataModule) -> int:
    if hasattr(dm, "item_vocab_size"):
        return int(dm.item_vocab_size)

    if hasattr(dm, "vocab_item_size"):
        return int(dm.vocab_item_size)

    if hasattr(dm, "item2id"):
        return max(dm.item2id.values()) + 1

    raise AttributeError("Could not infer item vocabulary size.")


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
    data_cfg = make_config(
        AmazonReviewDataConfig,
        {
            "reviews_path": REVIEWS_PATH,
            "metadata_path": METADATA_PATH,
            "item_key": ITEM_KEY,

            # Global time split.
            "split_strategy": "leave_one_out",
           

            # Filtering.
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "max_history_len": 20,
            "deduplicate_consecutive": False,
            "filter_cold_items_after_split":False,

            # Training target sampling.
            "train_all_prefixes": True,


            # GPU-friendly dataloader.
            "batch_size": 512,
            "eval_batch_size": 512,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,

            "history_padding": "left",
            "include_raw_in_batch": False,
            "seed": seed,
        },
        required=("train_target_policy", "train_last_n"),
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
    cfg = make_config(
        SASRecConfig,
        {
            "item_vocab_size": get_item_vocab_size(dm),
            "max_history_len": dm.cfg.max_history_len,

            # Strong GPU setting for instruments.
            "d_model": 128,
            "nhead": 2,
            "num_layers": 3,

            "dropout": 0.3,
            "pos_drop": 0.3,

            "initializer_range": 0.02,
            "layer_norm_eps": 1e-8,

            # If supported by your SASRec implementation.
            "exclude_history_in_generate": False,
        },
    )

    model = SASRec(cfg)

    print(model)
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    return model


def build_method() -> SFTMethod:
    cfg = make_config(
        SFTConfig,
        {
            "loss_weight": 1.0,
            "metric_prefix": "sft",

            "compute_train_metrics": False,
            "compute_eval_metrics": True,
            "compute_item_metrics": True,
            "item_topk": KS,

            "compute_token_metrics": False,
            "evaluate_generation": False,
            "use_sample_weights":True,

            "pad_item_id": 0,
        },
    )

    return SFTMethod(cfg)


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

    cfg = make_config(
        TrainerConfig,
        {
            "max_epochs": 500,

            "validate_every_epochs": 1,
            "validate_every_steps": None,

            "monitor": "val/sft/ndcg@10",
            "monitor_mode": "max",
            "early_stopping_patience": 10,
            "min_delta": 1e-5,

            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,

            "use_amp": True,
            "amp_dtype": "bfloat16",

            "checkpoint_dir": checkpoint_dir,
            "save_best": True,
            "save_last": True,

            "log_every_steps": 100,
            "progress_bar": True,
            "prefix_method_metrics_with_stage": True,
        },
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
    print(f"Running SASRec instruments experiment | seed={seed}")
    print(f"{'=' * 80}")

    set_seed(seed)
    print('---------')
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
        num_items=get_item_vocab_size(dm),
        ks=KS,
        pad_item_id=0,
        exclude_history=False,
    )

    row: dict[str, Any] = {
        "dataset": "instruments",
        "model": "sasrec",
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

    device = "cpu"
    print(f"Using device: {device}")
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