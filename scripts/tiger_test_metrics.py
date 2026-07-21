#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
from flowrec.data.preprocessors import load_semantic_ids
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.tiger import TigerModel
from flowrec.training import Trainer, TrainerConfig


AMAZON_DATASETS = {
    "beauty": {
        "reviews_path": "data/raw/reviews_Beauty_14.json.gz",
        "metadata_path": "data/raw/meta_Beauty_14.json.gz",
        "item_key": "asin",
    },
    "instruments": {
        "reviews_path": "data/raw/Musical_Instruments.json.gz",
        "metadata_path": "data/raw/meta_Musical_Instruments.json.gz",
        "item_key": "asin",
    },
    "beauty_tiger_repo":
    {"reviews_path": "data/raw/reviews_inter_converted_for_amazonreview.jsonl.gz",
     "metadata_path": None,
        "item_key": "asin",}
}


class TigerSFTMethod(SFTMethod):
    """SFT method with TIGER generation metrics aliased to ndcg/recall.

    This is the same wrapper used by scripts/tiger_for_small_hard.py, so
    Trainer.test() uses the same metric calculation path as validation.
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
            deduplicate_consecutive=True,
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
    sid_path = path_from_repo(args.semantic_dir) / dataset_name / "semantic_ids.json"
    if not sid_path.is_file():
        raise FileNotFoundError(f"Semantic ID file does not exist: {sid_path}")
    semantic_ids, _ = load_semantic_ids(sid_path, offset_by_one=False)
    status(f"[{dataset_name}] semantic IDs: {sid_path}", args)
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
        num_beams=args.eval_num_beams,
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
            generation_num_beams=args.val_num_beams,
            generation_num_return_sequences=max(args.ks),
            generation_constrain_valid_sequences=args.constrain_valid_sequences,
        )
    )


def select_checkpoint(dataset_name: str, args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        checkpoint_path = path_from_repo(args.checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best.pt"
    elif args.checkpoint_dir is not None:
        checkpoint_path = path_from_repo(args.checkpoint_dir) / "best.pt"
    else:
        checkpoint_path = path_from_repo(args.checkpoint_root) / dataset_name / "checkpoints" / "best.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    return checkpoint_path


def add_display_aliases(metrics: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = dict(metrics)
    for k in args.ks:
        recall_key = f"test/sft/generation_recall@{k}"
        ndcg_key = f"test/sft/generation_ndcg@{k}"
        coverage_key = f"test/sft/generation_coverage@{k}"
        if recall_key in metrics:
            out[f"HR@{k}"] = metrics[recall_key]
            out[f"Recall@{k}"] = metrics[recall_key]
        if ndcg_key in metrics:
            out[f"NDCG@{k}"] = metrics[ndcg_key]
        if coverage_key in metrics:
            out[f"Coverage@{k}"] = metrics[coverage_key]
    return out


def run_dataset(dataset_name: str, args: argparse.Namespace) -> dict[str, float | str]:
    status(f"[{dataset_name}] loading data", args)
    dm = build_datamodule(dataset_name, args)

    semantic_ids = load_dataset_semantic_ids(dataset_name, args)
    dm.add_semantic_ids(semantic_ids=semantic_ids, offset_by_one=dm.cfg.offset_semantic_ids_by_one)
    status(
        f"[{dataset_name}] data: users={dm.num_users:,}, items={dm.num_items:,}, "
        f"test={len(dm.test_dataset):,}, sid_len={dm.sid_len}",
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

    metrics = add_display_aliases(trainer.test(dm.test_dataloader()), args)
    result: dict[str, float | str] = {
        "dataset": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        **metrics,
    }

    out_dir = path_from_repo(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = (
        path_from_repo(args.output_json)
        if args.output_json is not None
        else out_dir / "test_metrics_validation_code.json"
    )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    status(f"[{dataset_name}] metrics saved to {output_json}", args)
    status(json.dumps(result, indent=2, sort_keys=True), args)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Evaluate TIGER test metrics using the same Trainer/Method code as validation."
    )
    parser.add_argument("--dataset", choices=["beauty", "instruments", "all"], default="beauty_tiger_repo")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="artifacts/tiger_sft_amazon")
    parser.add_argument("--semantic-dir", default="artifacts/tiger_sft_amazon/semantic_ids")
    parser.add_argument("--checkpoint-root", default="artifacts/tiger_sft_amazon")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quiet-status", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--tiger-dim", type=int, default=128)
    parser.add_argument("--tiger-heads", type=int, default=6)
    parser.add_argument("--tiger-layers", type=int, default=4)
    parser.add_argument("--tiger-ff", type=int, default=512)
    parser.add_argument("--tiger-d-kv", type=int, default=32)
    parser.add_argument("--tiger-dropout", type=float, default=0.1)
    parser.add_argument("--user-ids-count", type=int, default=2000)
    parser.add_argument("--eval-num-beams", type=int, default=100)
    parser.add_argument("--val-num-beams", type=int, default=50)
    parser.add_argument("--constrain-valid-sequences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    datasets = ["beauty", "instruments"] if args.dataset == "all" else [args.dataset]
    rows = [run_dataset(dataset_name, args) for dataset_name in datasets]

    if len(rows) > 1:
        out = path_from_repo(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "test_metrics_validation_code_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
        status(f"Summary saved to {summary_path}", args)


if __name__ == "__main__":
    main()
