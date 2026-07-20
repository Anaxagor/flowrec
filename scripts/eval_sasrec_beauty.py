#!/usr/bin/env python
"""Evaluate the best Optuna SASRec checkpoint on Amazon Beauty (val + test).

Loads ``artifacts/optuna/sasrec_beauty/best_trial_best_epoch_checkpoint.pt`` and
reports NDCG / Recall @ {5, 10, 20} on the validation and test splits, using the
exact same Trainer/Method evaluation path that produced the training metrics.

The model architecture is reconstructed from the ``model_config`` stored inside
the checkpoint, and the datamodule is rebuilt from the ``fixed_data_config`` that
the Optuna study used (see artifacts/optuna/sasrec_beauty/best_sasrec_optuna.json).
As a safety check the script asserts that the rebuilt item vocabulary matches the
checkpoint before evaluating -- a mismatch means the data files below do not
reproduce the dataset the checkpoint was trained on.

Example:
    python scripts/eval_sasrec_beauty.py
    python scripts/eval_sasrec_beauty.py --device cuda --eval-batch-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.training import Trainer, TrainerConfig


def path_from_repo(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/optuna/sasrec_beauty/best_trial_best_epoch_checkpoint.pt",
        help="Path to the SASRec checkpoint (.pt).",
    )
    parser.add_argument(
        "--reviews-path",
        default="data/raw/dataframe_format/reviews_beauty_2014.parquet",
        help=(
            "Amazon Beauty 2014 reviews file used to rebuild the dataset. "
            "Defaults to the parquet the current tuning_sasrec.py study trains "
            "on; for a checkpoint trained from a legacy json.gz dump, pass "
            "that file instead (the vocab check below catches mismatches)."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="data/raw/dataframe_format/meta_beauty_2014.parquet",
        help="Amazon Beauty 2014 metadata file (set to 'none' to skip).",
    )
    parser.add_argument("--item-key", default="asin")
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/optuna/sasrec_beauty/eval_metrics.json",
        help="Where to write the resulting metrics (set to 'none' to skip).",
    )
    return parser.parse_args()


def build_datamodule(args: argparse.Namespace) -> AmazonReviewDataModule:
    metadata_path = None
    if args.metadata_path and args.metadata_path.lower() != "none":
        metadata_path = path_from_repo(args.metadata_path)
    dm = AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=path_from_repo(args.reviews_path),
            metadata_path=metadata_path,
            item_key=args.item_key,
            # Matches artifacts/optuna/sasrec_beauty/best_sasrec_optuna.json
            # -> "fixed_data_config".
            split_strategy="leave_one_out",
            train_all_prefixes=True,
            deduplicate_consecutive=False,
            filter_cold_items_after_split=False,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            max_history_len=args.max_history_len,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            seed=args.seed,
        )
    )
    dm.setup()
    return dm


def build_model_from_checkpoint(checkpoint_path: Path) -> tuple[SASRec, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(payload["model_config"])
    model = SASRec(
        SASRecConfig(
            item_vocab_size=int(model_config["item_vocab_size"]),
            max_history_len=int(model_config["max_history_len"]),
            d_model=int(model_config["d_model"]),
            nhead=int(model_config["nhead"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            pos_drop=float(model_config["pos_drop"]),
        )
    )
    model.pad_item_id = 0
    return model, model_config


class CoverageSFTMethod(SFTMethod):
    """SFT method that also accumulates catalog coverage@K during evaluation.

    The trainer's built-in coverage accumulator only handles semantic-ID
    generation models (TIGER). For a full-softmax ranker like SASRec we track,
    per K, the set of distinct items that appear in any user's top-K list, and
    report ``coverage@K = |unique recommended items| / |catalog|``. This runs in
    the same forward pass as recall/ndcg, so it adds no extra model evaluation.
    """

    def __init__(self, cfg: SFTConfig) -> None:
        super().__init__(cfg)
        self.reset_coverage()

    def reset_coverage(self) -> None:
        self._coverage: dict[int, set[int]] = {int(k): set() for k in self.cfg.item_topk}

    def _compute_generic_metrics(
        self,
        model,
        batch,
        model_output,
        *,
        stage: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = super()._compute_generic_metrics(
            model, batch, model_output, stage=stage, extra=extra
        )
        if (
            stage != "train"
            and model_output is not None
            and model_output.scores is not None
            and model_output.scores.ndim == 2
            and self._coverage
        ):
            scores = model_output.scores
            max_k = min(max(self._coverage), scores.size(1))
            topk = torch.topk(scores, k=max_k, dim=-1).indices
            for k in self._coverage:
                kk = min(int(k), max_k)
                uniq = torch.unique(topk[:, :kk])
                uniq = uniq[uniq != self.cfg.pad_item_id]
                self._coverage[k].update(int(x) for x in uniq.cpu().tolist())
        return metrics

    def coverage_metrics(self, catalog_size: int, *, stage: str) -> dict[str, float]:
        denom = float(catalog_size)
        prefix = self.cfg.metric_prefix.rstrip("/") or "sft"
        return {
            f"{stage}/{prefix}/coverage@{k}": (len(seen) / denom if denom > 0 else 0.0)
            for k, seen in sorted(self._coverage.items())
        }


def build_method(ks: list[int]) -> CoverageSFTMethod:
    return CoverageSFTMethod(
        SFTConfig(
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_item_metrics=True,
            compute_token_metrics=False,
            evaluate_generation=False,
            item_topk=tuple(ks),
            pad_item_id=0,
        )
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_path = path_from_repo(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    print(f"[data] building datamodule from {args.reviews_path}", flush=True)
    dm = build_datamodule(args)
    print(
        f"[data] users={dm.num_users:,} items={dm.num_items:,} "
        f"vocab={dm.item_vocab_size:,} "
        f"train={len(dm.train_dataset):,} val={len(dm.val_dataset):,} test={len(dm.test_dataset):,}",
        flush=True,
    )

    print(f"[model] loading checkpoint {checkpoint_path}", flush=True)
    model, model_config = build_model_from_checkpoint(checkpoint_path)

    expected_vocab = int(model_config["item_vocab_size"])
    if dm.item_vocab_size != expected_vocab:
        raise ValueError(
            "Rebuilt item vocabulary does not match the checkpoint: "
            f"datamodule vocab={dm.item_vocab_size} vs checkpoint={expected_vocab}. "
            "The reviews/metadata files do not reproduce the training dataset; "
            "point --reviews-path / --metadata-path at the correct files."
        )
    print(f"[model] config={json.dumps(model_config)}", flush=True)

    method = build_method(args.ks)
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(),
        device=args.device,
        lr=0.0,
        weight_decay=0.0,
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False, load_scheduler=False)
    print(
        f"[model] loaded (best_epoch={trainer.state.best_epoch}, "
        f"best_metric={trainer.state.best_metric})",
        flush=True,
    )

    catalog_size = dm.num_items

    print("[eval] evaluating validation split ...", flush=True)
    method.reset_coverage()
    val_metrics = trainer.evaluate(dm.val_dataloader(), stage="val")
    val_metrics.update(method.coverage_metrics(catalog_size, stage="val"))

    print("[eval] evaluating test split ...", flush=True)
    method.reset_coverage()
    test_metrics = trainer.evaluate(dm.test_dataloader(), stage="test")
    test_metrics.update(method.coverage_metrics(catalog_size, stage="test"))

    result = {
        "checkpoint": str(checkpoint_path),
        "device": args.device,
        "model_config": model_config,
        "data": {
            "reviews_path": args.reviews_path,
            "metadata_path": args.metadata_path,
            "num_users": dm.num_users,
            "num_items": dm.num_items,
            "num_val_examples": len(dm.val_dataset),
            "num_test_examples": len(dm.test_dataset),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    print("\n===== SASRec Beauty metrics =====", flush=True)
    print_metric_table(val_metrics, test_metrics, args.ks)

    if args.output_json and args.output_json.lower() != "none":
        output_json = path_from_repo(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"\n[out] metrics written to {output_json}", flush=True)


def print_metric_table(
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    ks: list[int],
) -> None:
    header = f"{'metric':<14}{'val':>12}{'test':>12}"
    print(header)
    print("-" * len(header))
    for metric in ("ndcg", "recall", "coverage"):
        for k in ks:
            val_key = f"val/sft/{metric}@{k}"
            test_key = f"test/sft/{metric}@{k}"
            val = val_metrics.get(val_key)
            test = test_metrics.get(test_key)
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
            test_str = f"{test:.4f}" if isinstance(test, (int, float)) else "n/a"
            print(f"{metric.upper() + '@' + str(k):<14}{val_str:>12}{test_str:>12}")


if __name__ == "__main__":
    main()
