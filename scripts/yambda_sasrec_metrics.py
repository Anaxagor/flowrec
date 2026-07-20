"""Evaluate a trained yambda_sasrec run: HR / NDCG / coverage @5,10,20 on val and test.

Usage:
    python scripts/yambda_sasrec_metrics.py [run_dir]

Defaults to the run directory in DEFAULT_RUN_DIR. The datamodule is rebuilt
from the run's resolved_config.json so the splits match training exactly, and
the model is restored from checkpoints/best.pt.
"""

from __future__ import annotations

import faulthandler
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

# datasets must be imported before torch: the reverse order segfaults in this
# environment (torch 2.9.1+cu126 / datasets 4.8.5 DLL conflict on Windows).
import datasets  # noqa: F401

import torch

faulthandler.enable()


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training.trainer import move_to_device


# Dense-causal runs land under yambda_sasrec_dense_causal; pass a run dir as
# argv[1] to evaluate a specific run (old per-prefix runs still work — the
# datamodule is rebuilt from the run's own resolved_config.json either way).
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "experiments" / "yambda_sasrec_training" / "run_2026-07-07_20-24-38"
)

KS = (5, 10, 20)
PAD_ITEM_ID = 0


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def make_config(cls, kwargs: dict[str, Any]):
    if not is_dataclass(cls):
        return cls(**kwargs)

    supported = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    ignored = sorted(set(kwargs) - supported)

    if ignored:
        print(f"[Config warning] Ignoring unsupported fields for {cls.__name__}: {ignored}")

    return cls(**filtered)


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def build_datamodule(run_dir: Path) -> YambdaDataModule:
    with open(run_dir / "resolved_config.json", "r", encoding="utf-8") as f:
        resolved = json.load(f)

    data_cfg = make_config(YambdaDataConfig, resolved["data"])
    dm = YambdaDataModule(data_cfg)
    dm.setup()

    print("Dataset summary:")
    print(json.dumps(dm.summary(), indent=2))

    return dm


def load_model(checkpoint_path: Path, device: str) -> SASRec:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = make_config(SASRecConfig, payload["model_config"])
    model = SASRec(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    state = payload.get("trainer_state", {})
    print(
        f"Loaded checkpoint: {checkpoint_path}\n"
        f"  best_epoch:  {state.get('best_epoch')}\n"
        f"  best_metric: {state.get('best_metric')}"
    )

    return model


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model: SASRec,
    loader,
    *,
    device: str,
    num_items: int,
    ks: tuple[int, ...] = KS,
) -> dict[str, float]:
    """Compute HR@K, NDCG@K and catalog coverage@K over a dataloader.

    HR/NDCG use raw full-catalog scores (pad column included), matching the
    training-time item_ranking_metrics, so hr@k here equals train-time
    recall@k under leave-one-out. Coverage@k is the fraction of the real item
    catalog (pad excluded) that appears in at least one user's top-k list.
    """

    max_k = max(ks)
    num_valid = 0
    hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    seen_items: dict[int, set[int]] = {k: set() for k in ks}

    for raw_batch in loader:
        batch = move_to_device(raw_batch, torch.device(device))

        scores = model(batch).scores  # [B, V]
        targets = batch["target_item_ids"].to(scores.device).long()  # [B]

        topk = torch.topk(scores, k=min(max_k, scores.size(1)), dim=-1).indices  # [B, max_k]

        valid = (targets != PAD_ITEM_ID) & (targets >= 0) & (targets < scores.size(1))
        num_valid += int(valid.sum().item())

        matches = topk.eq(targets.unsqueeze(1))  # [B, max_k]
        has_hit = matches.any(dim=1)
        ranks = matches.float().argmax(dim=1)  # 0-based, meaningful only where has_hit

        for k in ks:
            hit_k = valid & has_hit & (ranks < k)
            hits[k] += int(hit_k.sum().item())
            if hit_k.any():
                ndcg_sums[k] += float(
                    (1.0 / torch.log2(ranks[hit_k].float() + 2.0)).sum().item()
                )

            unique_items = topk[:, :k].unique()
            unique_items = unique_items[unique_items != PAD_ITEM_ID]
            seen_items[k].update(unique_items.cpu().tolist())

    denom = max(num_valid, 1)
    metrics: dict[str, float] = {"num_examples": num_valid}
    for k in ks:
        metrics[f"hr@{k}"] = hits[k] / denom
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / denom
        metrics[f"coverage@{k}"] = len(seen_items[k]) / max(num_items, 1)

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    device = pick_device()
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    print("Loading Yambda and rebuilding splits from resolved_config.json...")
    dm = build_datamodule(run_dir)

    model = load_model(checkpoint_path, device)

    if model.item_vocab_size != dm.item_vocab_size:
        raise ValueError(
            f"Vocab mismatch: checkpoint item_vocab_size={model.item_vocab_size} "
            f"vs datamodule item_vocab_size={dm.item_vocab_size}. "
            "The rebuilt splits do not match the training run."
        )

    # Real catalog size excludes the pad item.
    num_items = dm.item_vocab_size - 1

    results: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "ks": list(KS),
        "num_items": num_items,
    }

    for split, loader in (
        ("val", dm.val_dataloader()),
        ("test", dm.test_dataloader()),
    ):
        print(f"\nEvaluating {split}...")
        metrics = evaluate_split(
            model,
            loader,
            device=device,
            num_items=num_items,
            ks=KS,
        )
        results[split] = metrics
        print(f"{split} metrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))

    out_path = run_dir / "metrics_hr_ndcg_coverage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
