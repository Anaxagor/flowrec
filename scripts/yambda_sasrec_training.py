#!/usr/bin/env python
"""Train SASRec with the SFT (next-item cross-entropy) loss on Yambda 50M.

This is the atomic-item counterpart to ``yambda_tiger_learning.py``: it reuses
*the exact same* ``YambdaDataModule`` preprocessing (leave-one-out, 50M listens,
played-ratio >= 50%, k-core >= 5, ...), but trains a classic SASRec sequence
model directly on atomic item IDs instead of RQ-VAE semantic IDs.

Because SASRec ranks over the full item vocabulary, there is no tokenizer step:
the collator already emits ``history_item_ids`` / ``history_mask`` /
``target_item_ids`` for every example, and ``SFTMethod`` turns the model's item
scores into a next-item cross-entropy loss and Recall/NDCG@K ranking metrics.

Model / training hyperparameters (from the task):
  * SASRec: maxlen=128, hidden_units=128, num_blocks=3, num_heads=2, dropout=0.3
  * patience=10, max_epochs=500
  * validate + monitor every epoch on val/NDCG@10.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# IMPORTANT: import pyarrow's native extensions before torch.  The HF ``datasets``
# loader used in ``YambdaDataModule.setup()`` imports ``pyarrow.dataset``, and on
# this Windows/conda stack importing it *after* torch is a hard native access
# violation (SIGSEGV, exit 139).  Pre-loading it here forces the safe init order
# (the TIGER script sidesteps the same issue via its top-level ``pyarrow.parquet``
# import).  Imported purely for the side effect.
import pyarrow.dataset  # noqa: F401
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.yambda_sasrec_training")


# --------------------------------------------------------------------------- #
# Experiment specification (from the task).                                    #
# --------------------------------------------------------------------------- #

# SASRec model hyperparameters (model_params.*).
MODEL: dict[str, Any] = {
    "maxlen": 128,          # positional-embedding capacity (>= data max_history_len)
    "hidden_units": 128,    # SASRecConfig.d_model
    "num_blocks": 3,        # SASRecConfig.num_layers (self-attention blocks)
    "num_heads": 2,         # SASRecConfig.nhead
    "dropout_rate": 0.3,    # SASRecConfig.dropout / pos_drop
}
PAD_ITEM_ID = 0

# Item-ranking evaluation: full-vocab Recall/NDCG@K.  ``10`` must be present so
# the monitored metric below exists.
METRICS_AT_K = (5, 10, 20)          # validation.metrics_at_k
# The task monitors "val/NDCG@10".  SFTMethod emits item-ranking metrics as
# sft/ndcg@K and the trainer stage-prefixes them, so the concrete key is:
MONITOR = "val/sft/ndcg@10"
MONITOR_MODE = "max"

# Training (training.*).
LR = 1e-3                   # Adam LR (standard SASRec default; override via CLI)
MAX_EPOCHS = 500            # max_epochs
VALIDATE_EVERY_EPOCHS = 1   # "monitor the model every epoch"
EARLY_STOPPING_PATIENCE = 10  # patience

# Dataloader batch sizes: mirror the TIGER GenRec run.
TRAIN_BATCH_SIZE = 2048
VALIDATION_BATCH_SIZE = 512


# --------------------------------------------------------------------------- #
# Data, model, method, trainer.                                                #
# --------------------------------------------------------------------------- #

def build_data_config(args: argparse.Namespace) -> YambdaDataConfig:
    """The Yambda 50M preprocessing, identical to ``yambda_tiger_learning.py``.

    ``keep_raw_interactions`` is off by default (it keeps millions of unused
    interaction dicts at 50M scale).  No ``user_hash_mod`` is set: SASRec does
    not have a user-token table, so hashed user IDs are never read.
    """
    return YambdaDataConfig(
        use_hf=True,
        dataset_type="flat",
        dataset_size="50m",
        event_file="listens",
        min_played_ratio_pct=50,
        split_strategy="leave_one_out",
        min_user_interactions=5,
        min_item_interactions=5,
        max_history_len=50,
        keep_raw_interactions=bool(args.keep_raw_interactions),
        global_eval_target_mode="last_per_user",
        train_all_prefixes=False,
        train_target_policy="last_n",
        train_last_n=128,
        deduplicate_consecutive=True,
        filter_cold_items_after_split=True,
        hf_recent_months=2,
        batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.validation_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        # SASRec reads history_item_ids/history_mask (always emitted); the raw
        # item strings are not needed, so keep them out of the batch.
        include_raw_in_batch=False,
        pad_item_id=PAD_ITEM_ID,
        seed=int(args.seed),
    )


def build_model(data: YambdaDataModule) -> SASRec:
    """Build SASRec sized to the Yambda item catalog.

    ``max_history_len`` uses the requested ``maxlen`` (128) as the positional
    capacity.  The datamodule produces length-``data.cfg.max_history_len`` (50)
    histories, which is well within this capacity.
    """
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(data.item_vocab_size),
            max_history_len=int(MODEL["maxlen"]),
            d_model=int(MODEL["hidden_units"]),
            nhead=int(MODEL["num_heads"]),
            num_layers=int(MODEL["num_blocks"]),
            dropout=float(MODEL["dropout_rate"]),
            pos_drop=float(MODEL["dropout_rate"]),
        )
    )
    model.pad_item_id = PAD_ITEM_ID
    return model


def build_method() -> SFTMethod:
    return SFTMethod(
        SFTConfig(
            # SASRec returns [B, num_items] item scores: rank with Recall/NDCG@K.
            compute_item_metrics=True,
            compute_token_metrics=False,
            evaluate_generation=False,
            item_topk=METRICS_AT_K,
            pad_item_id=PAD_ITEM_ID,
        )
    )


def build_trainer_config(args: argparse.Namespace, checkpoint_dir: Path) -> TrainerConfig:
    return TrainerConfig(
        max_epochs=MAX_EPOCHS,
        gradient_accumulation_steps=1,
        max_grad_norm=args.max_grad_norm,
        use_amp=bool(args.use_amp),
        amp_dtype=str(args.amp_dtype),
        validate_every_epochs=VALIDATE_EVERY_EPOCHS,
        run_validation_at_start=False,
        log_every_steps=int(args.log_every_steps),
        save_best=True,
        save_last=True,
        monitor=MONITOR,
        monitor_mode=MONITOR_MODE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=checkpoint_dir,
    )


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    run_dir = create_run_dir(resolve(args.output_dir))
    setup_logging(run_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Device: %s", device)

    data = YambdaDataModule(build_data_config(args))
    LOGGER.info("Loading Yambda 50M and building leave-one-out splits...")
    data.setup()
    LOGGER.info("Data summary: %s", json.dumps(data.summary(), sort_keys=True, default=str))

    model = build_model(data)
    method = build_method()
    LOGGER.info(
        "SASRec: item_vocab_size=%s total_params=%s trainable=%s maxlen=%s hidden=%s blocks=%s heads=%s",
        f"{data.item_vocab_size:,}",
        f"{model.num_parameters():,}",
        f"{model.num_parameters(trainable_only=True):,}",
        MODEL["maxlen"], MODEL["hidden_units"], MODEL["num_blocks"], MODEL["num_heads"],
    )

    trainer_cfg = build_trainer_config(args, run_dir / "checkpoints")
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=trainer_cfg,
        device=device,
        lr=LR,
        weight_decay=float(args.weight_decay),
        logger=make_status_logger(),
    )

    write_json(
        run_dir / "resolved_config.json",
        {
            "data": asdict(data.cfg),
            "data_summary": data.summary(),
            "model": model.config_dict(),
            "method": method.cfg.to_dict(),
            "trainer": trainer_cfg.to_dict(),
            "optimizer": {"name": "adamw", "lr": LR, "weight_decay": float(args.weight_decay)},
        },
    )

    LOGGER.info("Starting training: max_epochs=%s validate_every=%s monitor=%s patience=%s",
                MAX_EPOCHS, VALIDATE_EVERY_EPOCHS, MONITOR, EARLY_STOPPING_PATIENCE)
    fit = trainer.fit(data.train_dataloader(), data.val_dataloader())

    # Snapshot the full training state now: ``load_checkpoint`` below restores
    # the trainer state saved inside the checkpoint (i.e. the best epoch), which
    # would otherwise truncate the epoch count and history we report.
    training_history = list(trainer.state.history)
    epochs_completed = int(trainer.state.epoch + 1)
    steps_completed = int(trainer.state.global_step)
    best_metric = trainer.state.best_metric
    best_epoch = trainer.state.best_epoch

    # Reload the model from the best epoch and evaluate it on BOTH the
    # validation and test splits (Recall/NDCG/coverage@5,10,20).
    best_val_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    checkpoint = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint is not None:
        LOGGER.info("Reloading best-epoch checkpoint for final evaluation: %s", checkpoint)
        trainer.load_checkpoint(checkpoint, load_optimizer=False, load_scheduler=False)
        LOGGER.info("Evaluating best model on the validation split...")
        best_val_metrics = trainer.evaluate(data.val_dataloader(), stage="val")
        LOGGER.info("Evaluating best model on the test split...")
        test_metrics = trainer.test(data.test_dataloader())

    summary = {
        "run_dir": str(run_dir),
        "epochs_completed": epochs_completed,
        "steps_completed": steps_completed,
        "monitor": MONITOR,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        # Metrics of the best-epoch model, re-evaluated on each split.
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        # For reference, the validation metrics from the final training epoch.
        "last_epoch_validation_metrics": fit.val_metrics,
        "best_checkpoint": fit.best_checkpoint_path,
        "last_checkpoint": fit.last_checkpoint_path,
    }
    write_json(run_dir / "final_summary.json", summary)
    write_json(run_dir / "training_history.json", training_history)
    LOGGER.info("Done. Best %s=%s at epoch %s.", MONITOR, best_metric, best_epoch)
    LOGGER.info("Best-model validation metrics: %s", json.dumps(best_val_metrics, sort_keys=True, default=str))
    LOGGER.info("Best-model test metrics: %s", json.dumps(test_metrics, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# CLI + small helpers.                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SASRec with SFT on Yambda 50M.")
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default: cuda if available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 is the safest default on Windows).")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/yambda_sasrec_training"))
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--keep-raw-interactions", action="store_true",
                   help="Keep raw interaction records (memory-heavy at 50M; unused by SASRec).")
    p.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--validation-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--use-amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    p.add_argument("--amp-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    return p.parse_args()


def make_status_logger():
    keys = [
        "train/loss", "train/sft/loss",
        "val/loss", MONITOR, "val/sft/recall@10", "val/sft/coverage@10",
        "test/loss", MONITOR.replace("val/", "test/"),
        "test/sft/recall@10", "test/sft/coverage@10",
    ]

    def _fmt(payload: dict[str, Any]) -> str:
        parts = []
        for key in keys:
            if key in payload:
                value = payload[key]
                parts.append(f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}")
        return " | ".join(parts)

    def logger(payload: dict[str, Any]) -> None:
        event = payload.get("type")
        if event == "train_step":
            LOGGER.info("step %s (epoch %s) | %s",
                        payload.get("global_step"), int(payload.get("epoch", 0)) + 1, _fmt(payload))
        elif event in {"val_end", "test_end"}:
            LOGGER.info("%s | %s", event, _fmt(payload))
        elif event == "epoch_end":
            best = " | best checkpoint saved" if payload.get("best_checkpoint") else ""
            LOGGER.info("epoch %s done in %.2f min%s",
                        int(payload.get("epoch", 0)) + 1, float(payload.get("epoch_time_min", 0.0)), best)

    return logger


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def create_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
