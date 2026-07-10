from __future__ import annotations

import faulthandler
import json
import random
import sys
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

# datasets must be imported before torch: the reverse order segfaults in this
# environment (torch 2.9.1+cu126 / datasets 4.8.5 DLL conflict on Windows).
import datasets  # noqa: F401

import numpy as np
import torch
import yaml

faulthandler.enable()


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------

SEED = 42
KS = (5, 10, 20)

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "exp1-standard params" / "sasrec_model.yaml"

RUN_ROOT = PROJECT_ROOT / "experiments" / "yambda_sasrec_training"

DATASET_PARAMS: dict[str, Any] = {
    "dataset_type": "flat",
    "dataset_size": "50m",
    "event_file": "listens",
    "max_history_len": 50,
    "min_user_interactions": 5,
    "min_item_interactions": 5,
    "split_strategy": "leave_one_out",
    "min_history_len": 1,
    "train_all_prefixes": False,
    "train_target_policy": "last_n",
    "train_last_n": 128,
    "deduplicate_consecutive": True,
    "organic_filter": "any",
    "min_played_ratio_pct": 50,
    "offset_semantic_ids_by_one": False,
    "drop_items_missing_semantic_id": True,
}


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
    return "cpu"


def make_config(cls, kwargs: dict[str, Any]):
    if not is_dataclass(cls):
        return cls(**kwargs)

    supported = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    ignored = sorted(set(kwargs) - supported)

    if ignored:
        print(f"[Config warning] Ignoring unsupported fields for {cls.__name__}: {ignored}")

    return cls(**filtered)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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

def build_datamodule(seed: int) -> YambdaDataModule:
    data_cfg = make_config(
        YambdaDataConfig,
        {
            **DATASET_PARAMS,

            # Load Yambda from Hugging Face (no local parquet path).
            "use_hf": True,

            # Dataloading.
            "batch_size": 2048,
            "eval_batch_size": 512,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
            "include_raw_in_batch": False,

            "seed": seed,
        },
    )

    dm = YambdaDataModule(data_cfg)
    dm.setup()

    print("Dataset summary:")
    print(json.dumps(dm.summary(), indent=2))

    return dm


# ---------------------------------------------------------------------
# Model / method / trainer
# ---------------------------------------------------------------------

def build_model(dm: YambdaDataModule) -> SASRec:
    model_params = load_yaml(MODEL_CONFIG_PATH)

    cfg = make_config(
        SASRecConfig,
        {
            **model_params,
            # Derived from the datamodule, as documented in the YAML config.
            "item_vocab_size": dm.item_vocab_size,
            "max_history_len": dm.cfg.max_history_len,
        },
    )

    model = SASRec(cfg)

    print(f"SASRec config: {asdict(cfg)}")
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

            "pad_item_id": 0,
        },
    )

    return SFTMethod(cfg)


def make_epoch_logger():
    """Print loss and ndcg@10 for train/val at the end of every epoch."""

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") != "epoch_end":
            return

        parts = [f"epoch {payload.get('epoch')}"]

        train = payload.get("train") or {}
        train_loss = train.get("train/sft/loss", train.get("train/loss"))
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")

        val = payload.get("val") or {}
        val_loss = val.get("val/sft/loss", val.get("val/loss"))
        val_ndcg = val.get("val/sft/ndcg@10")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_ndcg is not None:
            parts.append(f"val_ndcg@10={val_ndcg:.4f}")

        epoch_time = payload.get("epoch_time_min")
        if epoch_time is not None:
            parts.append(f"time={epoch_time:.2f}min")

        print(" | ".join(parts), flush=True)

    return logger


def build_trainer(
    *,
    model: SASRec,
    method: SFTMethod,
    checkpoint_dir: Path,
    device: str,
) -> Trainer:
    cfg = make_config(
        TrainerConfig,
        {
            "max_epochs": 500,

            "validate_every_epochs": 1,
            "validate_every_steps": None,

            "monitor": "val/sft/ndcg@10",
            "monitor_mode": "max",
            "early_stopping_patience": 10,
            "min_delta": 0.0,

            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,

            "use_amp": False,

            "checkpoint_dir": checkpoint_dir,
            "save_best": True,
            "save_last": True,

            "log_every_steps": 50,
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
        weight_decay=0.0,
        logger=make_epoch_logger(),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    run_dir = RUN_ROOT / f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    set_seed(SEED)

    print("Loading Yambda and building splits...")
    dm = build_datamodule(SEED)

    model = build_model(dm)
    method = build_method()
    trainer = build_trainer(
        model=model,
        method=method,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": asdict(dm.cfg),
                "data_summary": dm.summary(),
                "model": asdict(model.cfg),
                "method": method.cfg.to_dict(),
                "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
                "trainer": trainer.cfg.to_dict(),
                "model_config_path": str(MODEL_CONFIG_PATH),
                "seed": SEED,
            },
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )

    print(
        f"Starting training: max_epochs={trainer.cfg.max_epochs} "
        f"monitor={trainer.cfg.monitor} patience={trainer.cfg.early_stopping_patience}"
    )

    fit_result = trainer.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
    )

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(fit_result.state.history, f, indent=2, default=str)

    print("Training finished:")
    print(f"  epochs_completed: {fit_result.state.epoch + 1}")
    print(f"  best_epoch:       {fit_result.state.best_epoch}")
    print(f"  best_metric:      {fit_result.state.best_metric}")
    print(f"  best_checkpoint:  {fit_result.best_checkpoint_path}")
    print(f"  last_checkpoint:  {fit_result.last_checkpoint_path}")

    if fit_result.best_checkpoint_path is not None:
        print(f"Loading best checkpoint: {fit_result.best_checkpoint_path}")
        trainer.load_checkpoint(
            fit_result.best_checkpoint_path,
            load_optimizer=False,
            load_scheduler=False,
            strict=True,
        )

    val_metrics = to_jsonable(trainer.evaluate(dm.val_dataloader(), stage="val"))
    test_metrics = to_jsonable(trainer.test(dm.test_dataloader()))

    print("\nValidation metrics (best checkpoint):")
    print(json.dumps(val_metrics, indent=2, sort_keys=True))
    print("\nTest metrics (best checkpoint):")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))

    summary = {
        "run_dir": str(run_dir),
        "monitor": trainer.cfg.monitor,
        "best_epoch": fit_result.state.best_epoch,
        "best_metric": fit_result.state.best_metric,
        "epochs_completed": fit_result.state.epoch + 1,
        "steps_completed": fit_result.state.global_step,
        "best_checkpoint": fit_result.best_checkpoint_path,
        "last_checkpoint": fit_result.last_checkpoint_path,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(run_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\nSaved final summary to: {run_dir / 'final_summary.json'}")


if __name__ == "__main__":
    main()
