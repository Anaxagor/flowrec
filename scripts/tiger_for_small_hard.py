#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.preprocessors import  load_semantic_ids
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.item_representation.content_embeddings import (
    AmazonItemTextBuilder,
    ContentEmbeddingBuilder,
    HFTextEmbedder,
)
from flowrec.item_representation.base import ItemEmbeddingStore, SemanticIDStore
from flowrec.item_representation.semantic_id.rqvae import RQVAEConfig, RQVAETrainConfig
from flowrec.item_representation.semantic_id.rqvae_tokenizer import (
    RQVAEItemTokenizer,
    RQVAEItemTokenizerConfig,
)
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
}


class TigerSFTMethod(SFTMethod):
    """SFT method with TIGER validation-generation metrics aliased to ndcg/recall.

    SFTMethod naturally emits generation_ndcg@K for TIGER.  This wrapper also
    exposes ndcg@K and recall@K so Trainer can monitor val/sft/ndcg@10.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def path_from_repo(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_dims(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def status(message: str, args: argparse.Namespace) -> None:
    if not getattr(args, "quiet_status", False):
        print(message, flush=True)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def metric_value(payload: Mapping[str, Any], name: str) -> Any | None:
    if name in payload:
        return payload[name]
    for section in ("train", "val", "test"):
        block = payload.get(section)
        if isinstance(block, Mapping) and name in block:
            return block[name]
    return None


def format_metric(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_selected_metrics(payload: Mapping[str, Any], names: list[str]) -> str:
    parts: list[str] = []
    for name in names:
        value = metric_value(payload, name)
        if value is not None:
            parts.append(f"{name}={format_metric(value)}")
    return " | ".join(parts)


def make_status_logger(args: argparse.Namespace, dataset_name: str):
    def logger(payload: dict[str, Any]) -> None:
        if args.quiet_status or not args.show_train_status:
            return

        event = payload.get("type")
        prefix = f"[{dataset_name}]"

        if event == "train_step":
            metrics = format_selected_metrics(payload, args.status_metrics)
            print(
                f"{prefix} train step={payload.get('global_step')} "
                f"opt_step={payload.get('optimizer_step')} "
                f"epoch={int(payload.get('epoch', 0)) + 1}/{args.max_epochs}"
                + (f" | {metrics}" if metrics else ""),
                flush=True,
            )

        elif event == "val_end":
            metrics = format_selected_metrics(payload, args.status_metrics)
            print(
                f"{prefix} validation step={payload.get('global_step')}"
                + (f" | {metrics}" if metrics else ""),
                flush=True,
            )

        elif event == "test_end":
            metrics = format_selected_metrics(payload, args.status_metrics)
            print(
                f"{prefix} test step={payload.get('global_step')}"
                + (f" | {metrics}" if metrics else ""),
                flush=True,
            )

        elif event == "epoch_end":
            epoch = int(payload.get("epoch", 0)) + 1
            epoch_min = float(payload.get("epoch_time_min", 0.0))
            metrics = format_selected_metrics(payload, args.status_metrics)
            ckpt = " | best checkpoint saved" if payload.get("best_checkpoint") else ""
            print(
                f"{prefix} epoch {epoch}/{args.max_epochs} done "
                f"in {epoch_min:.2f} min"
                + (f" | {metrics}" if metrics else "")
                + ckpt,
                flush=True,
            )

    return logger


def print_rqvae_training_log(dataset_name: str, tokenizer: RQVAEItemTokenizer, args: argparse.Namespace) -> None:
    if args.quiet_status or not args.print_rqvae_log:
        return
    state = tokenizer.train_state
    if state is None or not state.train_log:
        return

    print(f"[{dataset_name}] RQ-VAE training summary:", flush=True)
    every = max(int(args.rqvae_log_every_epochs), 1)
    last_epoch = int(state.train_log[-1].get("epoch", len(state.train_log)))
    for row in state.train_log:
        epoch = int(row.get("epoch", 0))
        if epoch == 1 or epoch == last_epoch or epoch % every == 0:
            keys = [
                "train_loss",
                "train_recon_loss",
                "train_vq_loss",
                "val_loss",
                "val_recon_loss",
                "val_vq_loss",
            ]
            msg = " | ".join(
                f"{key}={format_metric(row[key])}" for key in keys if key in row
            )
            print(f"  epoch {epoch}/{last_epoch} | {msg}", flush=True)
    print(
        f"[{dataset_name}] RQ-VAE best epoch={state.best_epoch}, "
        f"best_val_loss={format_metric(state.best_val_loss)}",
        flush=True,
    )


def train_semantic_ids(
    dataset_name: str,
    args: argparse.Namespace,
    dm: AmazonReviewDataModule,
) -> dict[str, list[int]]:
    cfg = AMAZON_DATASETS[dataset_name]
    out_dir = path_from_repo(args.semantic_dir) / dataset_name
    sid_path = out_dir / "semantic_ids.json"
    sid_metadata_path = out_dir / "semantic_id_metadata.json"
    allowed_items = set(dm.item_ids_in_data())

    if sid_path.exists() and not args.retrain_semantic_ids:
        loaded_semantic_ids, _ = load_semantic_ids(sid_path, offset_by_one=False)
        cached_items = set(loaded_semantic_ids.keys())
        reasons = []
        if cached_items != allowed_items:
            reasons.append(
                f"item set differs (cached={len(cached_items):,}, allowed={len(allowed_items):,})"
            )

        if sid_metadata_path.exists():
            with open(sid_metadata_path, "r", encoding="utf-8") as f:
                sid_metadata = json.load(f)
            rqvae_metadata = sid_metadata.get("rqvae_config", {})
            expected_metadata = {
                "level_offset_tokens": args.level_offset_tokens,
                "normalize": args.semantic_normalize,
                "codebook_init": args.rqvae_codebook_init,
                "num_codebooks": args.num_codebooks,
                "codebook_size": args.codebook_size,
            }
            actual_metadata = {
                "level_offset_tokens": sid_metadata.get("level_offset_tokens"),
                "normalize": sid_metadata.get("normalize"),
                "codebook_init": sid_metadata.get("codebook_init"),
                "num_codebooks": rqvae_metadata.get("num_codebooks"),
                "codebook_size": rqvae_metadata.get("codebook_size"),
            }
            for key, expected in expected_metadata.items():
                if actual_metadata.get(key) != expected:
                    reasons.append(
                        f"{key} differs (cached={actual_metadata.get(key)!r}, requested={expected!r})"
                    )
        else:
            reasons.append("semantic ID metadata is missing")

        if not reasons:
            status(f"[{dataset_name}] semantic IDs: using cached file {sid_path}", args)
            return loaded_semantic_ids

        status(f"[{dataset_name}] semantic IDs: cached file mismatch: {', '.join(reasons)}; retraining", args)

    embedding_dir = path_from_repo(args.embedding_dir)
    embedding_path = embedding_dir / f"{dataset_name}_content_embeddings.pt"
    if embedding_path.exists() and not args.rebuild_embeddings:
        embeddings = ItemEmbeddingStore.load(embedding_path)
        embedding_items = set(embeddings.item_ids)
        if embedding_items == allowed_items:
            status(f"[{dataset_name}] embeddings: using cached file {embedding_path}", args)
        else:
            status(
                f"[{dataset_name}] embeddings: cached file item set does not match "
                f"current datamodule allowed_items "
                f"(cached={len(embedding_items):,}, allowed={len(allowed_items):,}); rebuilding",
                args,
            )
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        status(f"[{dataset_name}] semantic IDs: building item text embeddings with {args.hf_model_name}", args)


        embedder = HFTextEmbedder(
            model_name=args.hf_model_name,
            device=args.device,
            max_length=args.hf_max_length,
            normalize=args.hf_normalize,
            pooling="masked_mean",
            show_progress=True,
        )
        embeddings = ContentEmbeddingBuilder(
            text_builder=AmazonItemTextBuilder(
                include_title=True,
                include_categories=True,
                include_features=True,
                include_description=True,
                include_details=True,
                include_price=False,
                max_chars=args.text_max_chars,
            ),
            embedder=embedder,
            item_key=args.embedding_item_key,
        ).build_from_metadata(
            path_from_repo(cfg["metadata_path"]),
            batch_size=args.embedding_batch_size,
            allowed_item_ids=list(allowed_items),
        ).store
        embedding_items = set(embeddings.item_ids)
        if embedding_items != allowed_items:
            missing = allowed_items - embedding_items
            extra = embedding_items - allowed_items
            raise ValueError(
                f"[{dataset_name}] embeddings do not match datamodule allowed_items: "
                f"missing={len(missing):,}, extra={len(extra):,}."
            )
        embeddings.save(embedding_path)
        status(f"[{dataset_name}] embeddings saved to {embedding_path}", args)

    status(
        f"[{dataset_name}] embeddings: items={len(embeddings.item_ids):,}, "
        f"dim={int(embeddings.embeddings.shape[1])}",
        args,
    )
    status(
        f"[{dataset_name}] semantic IDs: training RQ-VAE "
        f"epochs={args.rqvae_epochs}, codebooks={args.num_codebooks}, "
        f"codebook_size={args.codebook_size}, init={args.rqvae_codebook_init}, "
        f"normalize={args.semantic_normalize}, collision={args.collision_policy}, "
        f"level_offset_tokens={args.level_offset_tokens}",
        args,
    )

    tokenizer = RQVAEItemTokenizer(
        RQVAEItemTokenizerConfig(
            rqvae=RQVAEConfig(
                input_dim=int(embeddings.embeddings.shape[1]),
                latent_dim=args.rqvae_latent_dim,
                hidden_dims=parse_dims(args.rqvae_hidden_dims),
                num_codebooks=args.num_codebooks,
                codebook_size=args.codebook_size,
                commitment_weight=args.commitment_weight,
                dropout=args.rqvae_dropout,
                use_batch_norm=args.rqvae_use_batch_norm,
            ),
            train=RQVAETrainConfig(
                epochs=args.rqvae_epochs,
                batch_size=args.rqvae_batch_size,
                learning_rate=args.rqvae_lr,
                weight_decay=args.rqvae_weight_decay,
                val_ratio=args.rqvae_val_ratio,
                device=args.device,
                seed=args.seed,
                gradient_clip_norm=args.rqvae_gradient_clip_norm,
                codebook_init=args.rqvae_codebook_init,
            ),
            normalize=args.semantic_normalize,
            collision_policy=args.collision_policy,
            level_offset_tokens=args.level_offset_tokens,
            collision_seed=args.seed,
            encode_batch_size=args.rqvae_encode_batch_size,
        )
    )
    tokenizer.fit(embeddings.embeddings, item_ids=embeddings.item_ids)
    print_rqvae_training_log(dataset_name, tokenizer, args)
    tokenizer.save(out_dir)
    if tokenizer.semantic_id_store is None:
        raise RuntimeError("RQ-VAE tokenizer did not produce a semantic ID store.")

    semantic_id_mapping = tokenizer.semantic_id_store.to_plain_dict()
    if tokenizer.collision_stats is not None:
        stats = tokenizer.collision_stats.to_dict()
        status(
            f"[{dataset_name}] semantic IDs: collisions before={stats['num_collisions_before']:,}, "
            f"after={stats['num_collisions_after']:,}, "
            f"sid_path={sid_path}",
            args,
        )
    else:
        status(f"[{dataset_name}] semantic IDs saved to {sid_path}", args)
    return semantic_id_mapping


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


def build_model(dm: AmazonReviewDataModule, args: argparse.Namespace) -> TigerModel:
    semantic_id_mapping = {
        item: dm.semantic_ids[item]
        for item in dm.id2item[1:]
        if dm.semantic_ids is not None and item in dm.semantic_ids
    }
    model = TigerModel(
        codebook_size=args.codebook_size,
        sem_id_len=dm.sid_len,
        user_ids_count=args.user_ids_count,
        semantic_id_mapping=semantic_id_mapping,
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
    model.model.gradient_checkpointing_enable()
    return model


def build_method(args: argparse.Namespace) -> TigerSFTMethod:
    return TigerSFTMethod(
        SFTConfig(
            compute_train_metrics=True,
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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_generation(model: TigerModel, dm: AmazonReviewDataModule, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    ks = tuple(args.ks)
    max_k = max(ks)
    catalog = {tuple(map(int, row)) for row in model.item_semantic_ids.detach().cpu().tolist()}

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    coverage = {k: set() for k in ks}
    n = 0

    for batch_idx, batch in enumerate(dm.test_dataloader(), start=1):
        if not args.quiet_status and args.eval_log_every_batches > 0 and batch_idx % args.eval_log_every_batches == 0:
            print(f"[eval] processed {batch_idx} test batches", flush=True)
        batch = move_batch(batch, device)
        generated = model.generate(
            batch,
            num_beams=args.eval_num_beams,
            num_return_sequences=max_k,
            constrain_valid_sequences=args.constrain_valid_sequences,
        )
        target = batch["target_token_ids"].long()
        mask = batch.get("target_token_mask", torch.ones_like(target, dtype=torch.bool)).bool()

        matches = (generated == target.unsqueeze(1)) | ~mask.unsqueeze(1)
        matches = matches.all(dim=-1)
        n += int(target.size(0))

        generated_cpu = generated.detach().cpu().long()
        for k in ks:
            m = matches[:, :k]
            hit = m.any(dim=1)
            rank = m.float().argmax(dim=1)
            discount = torch.zeros_like(hit, dtype=torch.float32)
            discount[hit] = 1.0 / torch.log2(rank[hit].float() + 2.0)
            hits[k] += float(hit.float().sum().item())
            ndcgs[k] += float(discount.sum().item())

            for row in generated_cpu[:, :k, :].reshape(-1, generated_cpu.size(-1)).tolist():
                row = tuple(map(int, row))
                if row in catalog:
                    coverage[k].add(row)

    out: dict[str, float] = {"num_test_examples": float(n)}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / max(n, 1)
        out[f"NDCG@{k}"] = ndcgs[k] / max(n, 1)
        out[f"Coverage@{k}"] = len(coverage[k]) / max(len(catalog), 1)
    return out


def run_dataset(dataset_name: str, args: argparse.Namespace) -> dict[str, float | str]:

    status(f"[{dataset_name}] data: building AmazonReviewDataModule", args)
    dm = build_datamodule(dataset_name, args)
    status(
        f"[{dataset_name}] data: users={dm.num_users:,}, items={dm.num_items:,}, "
        f"train={len(dm.train_dataset):,}, val={len(dm.val_dataset):,}, "
        f"test={len(dm.test_dataset):,}, sid_len={dm.sid_len}",
        args,
    )


    status(f"\n=== {dataset_name} ===", args)
    semantic_id_mapping = train_semantic_ids(dataset_name, args, dm)
    dm.add_semantic_ids(semantic_ids=semantic_id_mapping, offset_by_one=dm.cfg.offset_semantic_ids_by_one)
    status(
        f"[{dataset_name}] data: semantic IDs attached with add_semantic_ids; "
        f"mapped_items={len(dm.semantic_ids or {}):,}, sid_len={dm.sid_len}",
        args,
    )

    
    model = build_model(dm, args)
    total_params, trainable_params = count_parameters(model)
    status(
        f"[{dataset_name}] model: TIGER params={total_params:,}, "
        f"trainable={trainable_params:,}",
        args,
    )
    method = build_method(args)

    status(
        f"[{dataset_name}] training: epochs={args.max_epochs}, "
        f"monitor=val/sft/ndcg@10, batch_size={args.batch_size}, "
        f"eval_batch_size={args.eval_batch_size}, grad_accum={args.grad_accum}, "
        f"constrain_valid_sequences={args.constrain_valid_sequences}",
        args,
    )

    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=args.max_epochs,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=1.0,
            use_amp=torch.cuda.is_available(),
            amp_dtype="float16",
            validate_every_epochs=args.validate_every_epochs,
            log_every_steps=args.log_every_steps,
            monitor="val/sft/ndcg@10",
            monitor_mode="max",
            early_stopping_patience=40,
            checkpoint_dir=path_from_repo(args.output_dir) / dataset_name / "checkpoints",
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        logger=make_status_logger(args, dataset_name),
    )

    fit = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        test_loader=dm.test_dataloader(),
    )
    status(
        f"[{dataset_name}] training finished: epochs_ran={fit.state.epoch + 1}, "
        f"global_steps={fit.state.global_step}, optimizer_steps={fit.state.optimizer_step}, "
        f"best_epoch={fit.state.best_epoch}, best_metric={format_metric(fit.state.best_metric)}",
        args,
    )
    if fit.best_checkpoint_path:
        trainer.load_checkpoint(fit.best_checkpoint_path, load_optimizer=False, load_scheduler=False)

    out_dir = path_from_repo(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(fit.state.history, f, indent=2)

    status(f"[{dataset_name}] test: generating recommendations for HR/NDCG/Coverage", args)
    metrics = evaluate_generation(model, dm, args)
    result = {"dataset": dataset_name, **metrics}
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    status(f"[{dataset_name}] test metrics:\n" + json.dumps(result, indent=2), args)
    return result


def main() -> None:
    p = argparse.ArgumentParser("TIGER SFT on Amazon datasets")
    p.add_argument("--dataset", choices=["beauty", "instruments", "all"], default="all")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="artifacts/tiger_sft_amazon")
    p.add_argument("--semantic-dir", default="artifacts/tiger_sft_amazon/semantic_ids")
    p.add_argument("--retrain-semantic-ids", action="store_true")
    p.add_argument("--embedding-dir", default="artifacts/tiger_sft_amazon/item_embeddings")
    p.add_argument("--rebuild-embeddings", action="store_true")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--log-every-steps", type=int, default=200)
    p.add_argument("--validate-every-epochs", type=int, default=2)
    p.add_argument("--show-train-status", dest="show_train_status", action="store_true", default=True)
    p.add_argument("--no-show-train-status", dest="show_train_status", action="store_false")
    p.add_argument("--quiet-status", action="store_true")
    p.add_argument(
        "--status-metrics",
        default=(
            "train/loss,train/sft/loss,train/sft/token_accuracy,"
            "val/loss,val/sft/ndcg@10,val/sft/generation_ndcg@10,val/sft/generation_coverage@10,"
            "test/loss,test/sft/ndcg@10,test/sft/generation_ndcg@10,test/sft/generation_coverage@10"
        ),
        help="Comma-separated metric names to print during training, validation, and test evaluation.",
    )
    p.add_argument("--eval-log-every-batches", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    p.add_argument("--hf-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--hf-max-length", type=int, default=256)
    p.add_argument("--embedding-batch-size", type=int, default=512)
    p.add_argument("--embedding-item-key", default="parent_asin")
    p.add_argument("--hf-normalize", action="store_true")
    p.add_argument("--text-max-chars", type=int, default=4096)

    p.add_argument("--rqvae-epochs", type=int, default=200)
    p.add_argument("--rqvae-log-every-epochs", type=int, default=5)
    p.add_argument("--print-rqvae-log", dest="print_rqvae_log", action="store_true", default=True)
    p.add_argument("--no-rqvae-log", dest="print_rqvae_log", action="store_false")
    p.add_argument("--rqvae-batch-size", type=int, default=1024)
    p.add_argument("--rqvae-lr", type=float, default=1e-3)
    p.add_argument("--rqvae-weight-decay", type=float, default=0.0)
    p.add_argument("--rqvae-val-ratio", type=float, default=0.05)
    p.add_argument("--rqvae-latent-dim", type=int, default=32)
    p.add_argument("--rqvae-hidden-dims", default="512,256,128")
    p.add_argument("--rqvae-encode-batch-size", type=int, default=4096)
    p.add_argument("--rqvae-gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--rqvae-codebook-init", choices=["data", "kmeans"], default="kmeans")
    p.add_argument("--rqvae-dropout", type=float, default=0.0)
    p.add_argument("--rqvae-use-batch-norm", action="store_true")
    p.add_argument("--num-codebooks", type=int, default=4)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--commitment-weight", type=float, default=0.25)
    p.add_argument("--semantic-normalize", choices=["none", "l2", "standard"], default="none")
    p.add_argument("--collision-policy", choices=["none", "append", "random_append"], default="random_append")
    p.add_argument("--level-offset-tokens", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--tiger-dim", type=int, default=128)
    p.add_argument("--tiger-heads", type=int, default=6)
    p.add_argument("--tiger-layers", type=int, default=4)
    p.add_argument("--tiger-ff", type=int, default=512)
    p.add_argument("--tiger-d-kv", type=int, default=32)
    p.add_argument("--tiger-dropout", type=float, default=0.1)
    p.add_argument("--user-ids-count", type=int, default=2000)
    p.add_argument("--val-num-beams", type=int, default=20)
    p.add_argument("--eval-num-beams", type=int, default=20)
    p.add_argument("--constrain-valid-sequences", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = p.parse_args()
    args.status_metrics = [m.strip() for m in args.status_metrics.split(",") if m.strip()]

    set_seed(args.seed)
    datasets = ["beauty", "instruments"] if args.dataset == "all" else [args.dataset]
    rows = [run_dataset(name, args) for name in datasets]

    out = path_from_repo(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    status(f"Summary saved to {out / 'summary.csv'}", args)


if __name__ == "__main__":
    main()