#!/usr/bin/env python
"""Evaluate the best TIGER-SFT checkpoint per Amazon dataset on validation and test.

For each dataset this script:
  1. rebuilds the AmazonReviewDataModule with the same config used for training
     (scripts/tiger_for_small_hard.py), so the item set / semantic-ID alignment
     matches the saved checkpoint;
  2. loads the semantic IDs from artifacts/tiger_sft_amazon/<dataset>/../semantic_ids;
  3. loads the best checkpoint (best.pt) for the dataset;
  4. computes NDCG / hit-rate (recall) / coverage @5,10,20 on the validation and
     test sets via the same Trainer.evaluate path used during training;
  5. prints the metrics to the terminal.

Despite the "beam1" name, generation uses 20 beams / 20 return sequences so the
top-20 rankings are genuine (a single beam only yields one candidate, which would
collapse @5==@10==@20). Override with --num-beams if desired.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.tiger import TigerModel
from flowrec.training import Trainer, TrainerConfig


# Same dataset paths as scripts/tiger_for_small_hard.py (the training script that
# produced the checkpoints in artifacts/tiger_sft_amazon).
AMAZON_DATASETS = {
    "beauty": {
        "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        "item_key": "asin",
    },
    "instruments": {
        "reviews_path": "data/raw/dataframe_format/reviews_instruments_2018.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_instruments_2018.parquet",
        "item_key": "asin",
    },
}


class TigerSFTMethod(SFTMethod):
    """SFT method with TIGER generation metrics aliased to ndcg/recall.

    Mirrors the wrapper in scripts/tiger_for_small_hard.py so evaluation goes
    through the exact metric-computation path used during training/validation.
    """

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
            gen_recall = f"{prefix}/generation_recall@{k}"
            gen_ndcg = f"{prefix}/generation_ndcg@{k}"
            if gen_recall in metrics:
                metrics[f"{prefix}/recall@{k}"] = metrics[gen_recall]
            if gen_ndcg in metrics:
                metrics[f"{prefix}/ndcg@{k}"] = metrics[gen_ndcg]
        return MethodOutput(
            loss=out.loss,
            metrics=metrics,
            model_output=out.model_output,
            extra=out.extra,
        )


def path_from_repo(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def status(message: str, args: argparse.Namespace) -> None:
    if not args.quiet_status:
        print(message, flush=True)


def build_datamodule(dataset_name: str, args: argparse.Namespace) -> AmazonReviewDataModule:
    cfg = AMAZON_DATASETS[dataset_name]
    dm = AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=path_from_repo(cfg["reviews_path"]),
            metadata_path=path_from_repo(cfg["metadata_path"]),
            item_key=cfg["item_key"],
            split_strategy="leave_one_out",
            train_all_prefixes=True,
            min_user_interactions=5,
            min_item_interactions=5,
            max_history_len=20,
            deduplicate_consecutive=False,
            filter_cold_items_after_split=False,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            offset_semantic_ids_by_one=True,
            seed=args.seed,
        )
    )
    dm.setup()
    return dm


def load_dataset_semantic_ids(dataset_name: str, args: argparse.Namespace) -> dict[str, list[int]]:
    """Load raw (pre-offset) semantic IDs directly from the saved JSON.

    The JSON is a single object mapping raw item id -> list of codebook tokens,
    e.g. {"B00004TMFE": [161, 236, 14, 64], ...}. We read it with json.load
    rather than the datamodule's line-oriented loader so a single-line JSON
    document parses correctly.
    """
    sid_path = path_from_repo(args.semantic_dir) / dataset_name / "semantic_ids.json"
    if not sid_path.is_file():
        raise FileNotFoundError(f"Semantic ID file does not exist: {sid_path}")
    with open(sid_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object of item->tokens in {sid_path}, got {type(raw)}")
    semantic_ids: dict[str, list[int]] = {}
    for item_id, tokens in raw.items():
        if isinstance(tokens, dict):
            tokens = tokens.get("tokens", tokens.get("semantic_id", tokens.get("sid")))
        if not isinstance(tokens, list):
            raise ValueError(f"Semantic ID for {item_id!r} must be a list, got {type(tokens)}")
        semantic_ids[str(item_id)] = [int(x) for x in tokens]
    status(f"[{dataset_name}] semantic IDs: {sid_path} ({len(semantic_ids):,} items)", args)
    return semantic_ids


def semantic_id_mapping_for_model(dm: AmazonReviewDataModule) -> dict[str, list[int]]:
    if dm.semantic_ids is None:
        raise ValueError("Semantic IDs must be attached before building TIGER.")
    return {
        item: dm.semantic_ids[item]
        for item in dm.id2item[1:]
        if item in dm.semantic_ids
    }


def build_model(dm: AmazonReviewDataModule, args: argparse.Namespace) -> TigerModel:
    return TigerModel(
        codebook_size=args.codebook_size,
        sem_id_len=int(dm.sid_len),
        user_ids_count=args.user_ids_count,
        semantic_id_mapping=semantic_id_mapping_for_model(dm),
        embedding_dim=args.tiger_dim,
        num_heads=args.tiger_heads,
        num_encoder_layers=args.tiger_layers,
        num_decoder_layers=args.tiger_layers,
        dim_feedforward=args.tiger_ff,
        d_kv=args.tiger_d_kv,
        dropout=args.tiger_dropout,
        num_beams=args.num_beams,
        num_return_sequences=max(args.ks),
    )


def build_method(args: argparse.Namespace) -> TigerSFTMethod:
    return TigerSFTMethod(
        SFTConfig(
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_token_metrics=True,
            compute_item_metrics=False,
            evaluate_generation=True,
            generation_topk=tuple(args.ks),
            generation_num_beams=args.num_beams,
            generation_num_return_sequences=max(args.ks),
            generation_constrain_valid_sequences=args.constrain_valid_sequences,
        )
    )


def select_checkpoint(dataset_name: str, args: argparse.Namespace) -> Path:
    checkpoint_path = path_from_repo(args.checkpoint_root) / dataset_name / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def display_metrics(stage: str, metrics: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    """Extract NDCG / hit-rate / coverage @k from the raw Trainer metric dict.

    Trainer prefixes metrics as ``{stage}/sft/generation_<name>@{k}``.
    """
    out: dict[str, float] = {}
    for k in args.ks:
        recall_key = f"{stage}/sft/generation_recall@{k}"
        ndcg_key = f"{stage}/sft/generation_ndcg@{k}"
        coverage_key = f"{stage}/sft/generation_coverage@{k}"
        if ndcg_key in metrics:
            out[f"ndcg@{k}"] = float(metrics[ndcg_key])
        if recall_key in metrics:
            out[f"hit_rate@{k}"] = float(metrics[recall_key])
        if coverage_key in metrics:
            out[f"coverage@{k}"] = float(metrics[coverage_key])
    return out


def print_metrics_table(dataset_name: str, stage: str, metrics: dict[str, float], args: argparse.Namespace) -> None:
    print(f"\n[{dataset_name}] {stage} metrics (num_beams={args.num_beams}):", flush=True)
    header = f"{'metric':<10}" + "".join(f"@{k:<10}" for k in args.ks)
    print(header, flush=True)
    for name in ("ndcg", "hit_rate", "coverage"):
        row = f"{name:<10}"
        for k in args.ks:
            value = metrics.get(f"{name}@{k}")
            row += f"{value:<11.6f}" if value is not None else f"{'-':<11}"
        print(row, flush=True)


def run_dataset(dataset_name: str, args: argparse.Namespace) -> dict[str, dict[str, float] | str]:
    status(f"\n=== {dataset_name} ===", args)
    status(f"[{dataset_name}] loading data", args)
    dm = build_datamodule(dataset_name, args)

    semantic_ids = load_dataset_semantic_ids(dataset_name, args)
    dm.add_semantic_ids(semantic_ids=semantic_ids, offset_by_one=dm.cfg.offset_semantic_ids_by_one)
    status(
        f"[{dataset_name}] data: users={dm.num_users:,}, items={dm.num_items:,}, "
        f"val={len(dm.val_dataset):,}, test={len(dm.test_dataset):,}, sid_len={dm.sid_len}",
        args,
    )

    checkpoint_path = select_checkpoint(dataset_name, args)
    status(f"[{dataset_name}] checkpoint: {checkpoint_path}", args)

    model = build_model(dm, args)
    method = build_method(args)
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(max_eval_batches=args.max_eval_batches),
        device=args.device,
        lr=0.0,
        weight_decay=0.0,
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False, load_scheduler=False)

    status(f"[{dataset_name}] evaluating validation set", args)
    val_raw = trainer.evaluate(dm.val_dataloader(), stage="val")
    val_metrics = display_metrics("val", val_raw, args)

    status(f"[{dataset_name}] evaluating test set", args)
    test_raw = trainer.evaluate(dm.test_dataloader(), stage="test")
    test_metrics = display_metrics("test", test_raw, args)

    print_metrics_table(dataset_name, "validation", val_metrics, args)
    print_metrics_table(dataset_name, "test", test_metrics, args)

    return {
        "dataset": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "validation": val_metrics,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Evaluate the best TIGER-SFT checkpoint per dataset on validation and test."
    )
    parser.add_argument("--dataset", choices=["beauty", "instruments", "all"], default="all")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-root", default="artifacts/tiger_sft_amazon")
    parser.add_argument("--semantic-dir", default="artifacts/tiger_sft_amazon/semantic_ids")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quiet-status", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=None)

    # Model architecture (must match the saved checkpoint config).
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--tiger-dim", type=int, default=128)
    parser.add_argument("--tiger-heads", type=int, default=6)
    parser.add_argument("--tiger-layers", type=int, default=4)
    parser.add_argument("--tiger-ff", type=int, default=512)
    parser.add_argument("--tiger-d-kv", type=int, default=32)
    parser.add_argument("--tiger-dropout", type=float, default=0.1)
    parser.add_argument("--user-ids-count", type=int, default=2000)

    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--constrain-valid-sequences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    datasets = ["beauty", "instruments"] if args.dataset == "all" else [args.dataset]
    results = [run_dataset(dataset_name, args) for dataset_name in datasets]

    print("\n=== Summary ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
