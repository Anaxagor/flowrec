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
import yaml

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


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def as_dims(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


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
    cfg = args.dataset_registry[dataset_name]
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
            pooling=args.pooling,
            show_progress=True,
        )
        text_builder_cfg = dict(args.text_builder)
        embeddings = ContentEmbeddingBuilder(
            text_builder=AmazonItemTextBuilder(
                include_title=bool(text_builder_cfg.get("include_title", True)),
                include_categories=bool(text_builder_cfg.get("include_categories", True)),
                include_features=bool(text_builder_cfg.get("include_features", True)),
                include_description=bool(text_builder_cfg.get("include_description", True)),
                include_details=bool(text_builder_cfg.get("include_details", True)),
                include_price=bool(text_builder_cfg.get("include_price", False)),
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
                hidden_dims=as_dims(args.rqvae_hidden_dims),
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
            collision_token_offset=args.collision_token_offset,
            collision_suffix_size=args.collision_suffix_size,
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
    entry = args.dataset_registry[dataset_name]
    data = args.data_cfg
    num_workers = int(data.get("num_workers", 0))
    dm = AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=path_from_repo(entry["reviews_path"]),
            metadata_path=path_from_repo(entry["metadata_path"]),
            item_key=entry.get("item_key", "asin"),
            # Positive/auxiliary partition: matching rows form the positive
            # training stream; the complement is exposed as
            # dm.interaction_store for rewards / negative sampling.
            positive_criterion=entry.get(
                "positive_criterion", data.get("positive_criterion")
            ),
            split_strategy=data.get("split_strategy", "leave_one_out"),
            train_all_prefixes=bool(data.get("train_all_prefixes", True)),
            min_user_interactions=int(data.get("min_user_interactions", 5)),
            min_item_interactions=int(data.get("min_item_interactions", 5)),
            max_history_len=int(data.get("max_history_len", 20)),
            deduplicate_consecutive=bool(data.get("deduplicate_consecutive", False)),
            filter_cold_items_after_split=bool(
                data.get("filter_cold_items_after_split", False)
            ),
            batch_size=int(data.get("batch_size", 128)),
            eval_batch_size=int(data.get("eval_batch_size", 64)),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            offset_semantic_ids_by_one=bool(data.get("offset_semantic_ids_by_one", True)),
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
        num_encoder_layers=args.tiger_encoder_layers,
        num_decoder_layers=args.tiger_decoder_layers,
        dim_feedforward=args.tiger_ff,
        d_kv=args.tiger_d_kv,
        dropout=args.tiger_dropout,
        num_beams=args.eval_num_beams,
        num_return_sequences=max(args.ks),
    )
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
    return model


def build_method(args: argparse.Namespace) -> TigerSFTMethod:
    sft = dict(args.sft_cfg)
    return TigerSFTMethod(
        SFTConfig(
            compute_train_metrics=bool(sft.get("compute_train_metrics", True)),
            compute_eval_metrics=bool(sft.get("compute_eval_metrics", True)),
            compute_token_metrics=bool(sft.get("compute_token_metrics", True)),
            compute_item_metrics=bool(sft.get("compute_item_metrics", False)),
            evaluate_generation=bool(sft.get("evaluate_generation", True)),
            generation_topk=tuple(sft.get("generation_topk", (1,))),
            generation_num_beams=args.val_num_beams,
            generation_num_return_sequences=args.val_num_beams,
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

    if dm.interaction_store is not None and len(dm.interaction_store):
        store_dir = path_from_repo(args.output_dir) / dataset_name / "interaction_store"
        dm.interaction_store.save(store_dir)
        status(
            f"[{dataset_name}] data: auxiliary interaction store "
            f"({len(dm.interaction_store):,} events: {dm.interaction_store.counts_by_type()}) "
            f"saved to {store_dir}",
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
        f"monitor={args.monitor}, batch_size={dm.cfg.batch_size}, "
        f"eval_batch_size={dm.cfg.eval_batch_size}, grad_accum={args.grad_accum}, "
        f"constrain_valid_sequences={args.constrain_valid_sequences}",
        args,
    )

    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=args.max_epochs,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp and torch.cuda.is_available(),
            amp_dtype=args.amp_dtype,
            validate_every_epochs=args.validate_every_epochs,
            log_every_steps=args.log_every_steps,
            monitor=args.monitor,
            monitor_mode=args.monitor_mode,
            early_stopping_patience=args.early_stopping_patience,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "TIGER SFT on Amazon datasets. Training parameters come from the "
        "config directory (dataset_amazon.yaml, semantic_ids.yaml, "
        "tiger_model.yaml, trainer.yaml); CLI flags override selection and "
        "runtime behavior only."
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory with dataset_amazon.yaml, semantic_ids.yaml, tiger_model.yaml and trainer_tiger.yaml.",
    )
    p.add_argument(
        "--dataset",
        choices=["beauty", "instruments", "all"],
        default=None,
        help="Dataset selection; defaults to the `dataset` key in dataset_amazon.yaml.",
    )
    p.add_argument("--device", default=None, help="Override trainer.yaml device.")
    p.add_argument("--seed", type=int, default=None, help="Override trainer.yaml seed.")
    p.add_argument("--max-epochs", type=int, default=None, help="Override trainer.yaml max_epochs.")
    p.add_argument(
        "--retrain-semantic-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override semantic_ids.yaml retrain_semantic_ids.",
    )
    p.add_argument(
        "--rebuild-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override semantic_ids.yaml rebuild_embeddings.",
    )

    # Runtime status output (not experiment parameters).
    p.add_argument("--show-train-status", dest="show_train_status", action="store_true", default=True)
    p.add_argument("--no-show-train-status", dest="show_train_status", action="store_false")
    p.add_argument("--quiet-status", action="store_true")
    p.add_argument(
        "--status-metrics",
        default=(
            "train/loss,train/sft/loss,train/sft/token_accuracy,"
            "val/loss,val/sft/ndcg@1,val/sft/generation_ndcg@1,val/sft/generation_coverage@1,"
            "test/loss,test/sft/ndcg@10,test/sft/generation_ndcg@10,test/sft/generation_coverage@10"
        ),
        help="Comma-separated metric names to print during training, validation, and test evaluation.",
    )
    p.add_argument("--print-rqvae-log", dest="print_rqvae_log", action="store_true", default=True)
    p.add_argument("--no-rqvae-log", dest="print_rqvae_log", action="store_false")
    p.add_argument("--rqvae-log-every-epochs", type=int, default=5)

    args = p.parse_args()
    args.status_metrics = [m.strip() for m in args.status_metrics.split(",") if m.strip()]
    return args


def resolve_settings(args: argparse.Namespace) -> None:
    """Load the config directory and materialize settings onto ``args``.

    CLI values win where explicitly provided; everything else comes from the
    YAML files, so the config directory is the single source of truth for
    experiment parameters.
    """
    config_dir = path_from_repo(args.config_dir)
    data_cfg = load_yaml(config_dir / "dataset_amazon.yaml")
    semantic = load_yaml(config_dir / "semantic_ids.yaml")
    tiger = load_yaml(config_dir / "tiger_model.yaml")
    trainer = load_yaml(config_dir / "trainer_tiger.yaml")

    # -- dataset / datamodule
    args.data_cfg = data_cfg
    args.dataset_registry = data_cfg["datasets"]
    selection = args.dataset or str(data_cfg.get("dataset", "all"))
    args.dataset_names = (
        list(args.dataset_registry) if selection == "all" else [selection]
    )
    for name in args.dataset_names:
        if name not in args.dataset_registry:
            raise KeyError(
                f"Dataset {name!r} not found in dataset_amazon.yaml "
                f"({list(args.dataset_registry)})."
            )

    # -- semantic IDs / embeddings
    args.semantic_dir = semantic["semantic_dir"]
    args.embedding_dir = semantic["embedding_dir"]
    if args.retrain_semantic_ids is None:
        args.retrain_semantic_ids = bool(semantic.get("retrain_semantic_ids", False))
    if args.rebuild_embeddings is None:
        args.rebuild_embeddings = bool(semantic.get("rebuild_embeddings", False))
    args.hf_model_name = semantic["hf_model_name"]
    args.hf_max_length = int(semantic.get("hf_max_length", 256))
    args.hf_normalize = bool(semantic.get("hf_normalize", False))
    args.embedding_batch_size = int(semantic.get("embedding_batch_size", 512))
    args.embedding_item_key = semantic.get("embedding_item_key", "parent_asin")
    args.pooling = semantic.get("pooling", "masked_mean")
    args.text_max_chars = int(semantic.get("text_max_chars", 4096))
    args.text_builder = dict(semantic.get("text_builder", {}))
    args.rqvae_latent_dim = int(semantic["rqvae_latent_dim"])
    args.rqvae_hidden_dims = semantic["rqvae_hidden_dims"]
    args.num_codebooks = int(semantic["num_codebooks"])
    args.codebook_size = int(semantic["codebook_size"])
    args.commitment_weight = float(semantic.get("commitment_weight", 0.25))
    args.rqvae_dropout = float(semantic.get("rqvae_dropout", 0.0))
    args.rqvae_use_batch_norm = bool(semantic.get("rqvae_use_batch_norm", False))
    args.rqvae_epochs = int(semantic["rqvae_epochs"])
    args.rqvae_batch_size = int(semantic.get("rqvae_batch_size", 1024))
    args.rqvae_lr = float(semantic.get("rqvae_lr", 1e-3))
    args.rqvae_weight_decay = float(semantic.get("rqvae_weight_decay", 0.0))
    args.rqvae_val_ratio = float(semantic.get("rqvae_val_ratio", 0.05))
    args.rqvae_gradient_clip_norm = float(semantic.get("rqvae_gradient_clip_norm", 1.0))
    args.rqvae_codebook_init = semantic.get("rqvae_codebook_init", "kmeans")
    args.rqvae_encode_batch_size = int(semantic.get("rqvae_encode_batch_size", 4096))
    args.semantic_normalize = semantic.get("semantic_normalize", "none")
    args.collision_policy = semantic.get("collision_policy", "append")
    args.level_offset_tokens = bool(semantic.get("level_offset_tokens", False))
    args.collision_token_offset = semantic.get("collision_token_offset")
    args.collision_suffix_size = semantic.get("collision_suffix_size")

    # -- TIGER model / generation
    tiger_codebook_size = int(tiger.get("codebook_size", args.codebook_size))
    if tiger_codebook_size != args.codebook_size:
        raise ValueError(
            f"codebook_size mismatch: semantic_ids.yaml={args.codebook_size}, "
            f"tiger_model.yaml={tiger_codebook_size}."
        )
    args.tiger_dim = int(tiger["embedding_dim"])
    args.tiger_heads = int(tiger["num_heads"])
    args.tiger_encoder_layers = int(tiger["num_encoder_layers"])
    args.tiger_decoder_layers = int(tiger["num_decoder_layers"])
    args.tiger_ff = int(tiger["dim_feedforward"])
    args.tiger_d_kv = int(tiger["d_kv"])
    args.tiger_dropout = float(tiger.get("dropout", 0.1))
    args.user_ids_count = int(tiger.get("user_ids_count", 2000))
    args.gradient_checkpointing = bool(tiger.get("gradient_checkpointing", True))
    args.eval_num_beams = int(tiger.get("eval_num_beams", 20))
    args.val_num_beams = int(tiger.get("val_num_beams", 1))
    args.constrain_valid_sequences = bool(tiger.get("constrain_valid_sequences", True))
    args.ks = [int(k) for k in tiger.get("ks", [5, 10, 20])]
    args.sft_cfg = dict(tiger.get("sft", {}))

    # -- trainer / optimizer / runtime
    args.lr = float(trainer.get("lr", 3e-4))
    args.weight_decay = float(trainer.get("weight_decay", 1e-2))
    if args.max_epochs is None:
        args.max_epochs = int(trainer.get("max_epochs", 500))
    args.grad_accum = int(trainer.get("gradient_accumulation_steps", 4))
    args.max_grad_norm = float(trainer.get("max_grad_norm", 1.0))
    args.validate_every_epochs = int(trainer.get("validate_every_epochs", 1))
    args.log_every_steps = int(trainer.get("log_every_steps", 200))
    args.monitor = trainer.get("monitor", "val/sft/ndcg@1")
    args.monitor_mode = trainer.get("monitor_mode", "max")
    args.early_stopping_patience = int(trainer.get("early_stopping_patience", 10))
    args.use_amp = bool(trainer.get("use_amp", True))
    args.amp_dtype = trainer.get("amp_dtype", "float16")
    if args.seed is None:
        args.seed = int(trainer.get("seed", 43))
    if args.device is None:
        configured = trainer.get("device", "cuda")
        args.device = configured if torch.cuda.is_available() or configured == "cpu" else "cpu"
    args.output_dir = trainer.get("output_dir", "artifacts/tiger_sft_amazon")
    args.eval_log_every_batches = int(trainer.get("eval_log_every_batches", 50))


def main() -> None:
    args = parse_args()
    resolve_settings(args)
    set_seed(args.seed)

    out = path_from_repo(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = {
        key: value
        for key, value in sorted(vars(args).items())
        if isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }
    resolved["config_dir"] = str(path_from_repo(args.config_dir))
    with open(out / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, default=str)

    rows = [run_dataset(name, args) for name in args.dataset_names]

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    status(f"Summary saved to {out / 'summary.csv'}", args)


if __name__ == "__main__":
    main()