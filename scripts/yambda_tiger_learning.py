#!/usr/bin/env python
"""Train TIGER with the SFT loss on the Yambda 50M dataset.

Pipeline
--------
1. Build the ``YambdaDataModule`` (leave-one-out) and its item catalog.
2. Stream the Yambda audio embeddings (``normalized_embed``) for the catalog
   items out of the ~14 GB ``embeddings.parquet`` and train an RQ-VAE tokenizer
   (``num_codebooks=3`` / ``codebook_size=256``) into ``semantic_ids.json``.
3. Attach the semantic IDs and train ``TigerModel`` with ``SFTMethod`` (token
   cross-entropy), monitoring beam-search NDCG@1 on the validation split.

Notes on scale (why this is not a naive port of the notebook):
  * ``embeddings.parquet`` holds ~7.7M tracks; reading it whole and exploding it
    into Python objects OOMs.  We stream it with pyarrow, keeping only the
    ``item_id`` + ``normalized_embed`` columns and only the catalog items.
  * Training RQ-VAE over all 7.7M tracks would also OOM the GPU during codebook
    init (an ``[N, codebook_size]`` distance matrix).  Restricting to the
    catalog keeps it small; items outside the catalog are never used anyway.

Both the embedding store and the tokenizer are cached under ``artifacts/`` and
reused on subsequent runs.  Run with ``--help`` for the operational knobs.
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
import pyarrow.parquet as pq
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.item_representation.base import ItemEmbeddingStore
from flowrec.item_representation.semantic_id import (
    RQVAEConfig,
    RQVAEItemTokenizer,
    RQVAEItemTokenizerConfig,
    RQVAETrainConfig,
)
from flowrec.models.tiger import TigerModel
from flowrec.training import Trainer, TrainerConfig


LOGGER = logging.getLogger("flowrec.yambda_tiger_learning")


# --------------------------------------------------------------------------- #
# Experiment specification (from the task).                                    #
# --------------------------------------------------------------------------- #

# Semantic IDs: RQ-VAE over the Yambda audio embeddings.
NUM_CODEBOOKS = 3          # dataset.num_codebooks
CODEBOOK_SIZE = 256        # model.codebook_size

# Model hyperparameters (model.*).  ``num_positions`` from the spec is
# intentionally unused: TigerModel is a T5 encoder-decoder with relative
# position bias, so there is no fixed max-position embedding table to size.
MODEL: dict[str, Any] = {
    "embedding_dim": 128,
    "codebook_size": CODEBOOK_SIZE,
    "user_ids_count": 2000,
    "num_heads": 2,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 512,
    "d_kv": 64,
    "dropout": 0.2,
    "activation": "relu",
    "layer_norm_eps": 1e-6,
    "initializer_range": 0.02,
}

# In-loop validation generation (validation.*): a single greedy beam returning
# one sequence, scored with NDCG@1.
VAL_NUM_BEAMS = 1          # validation.num_beams
VAL_TOP_K = 1             # validation.val_top_k (sequences returned per user)
METRICS_AT_K = (1,)       # validation.metrics_at_k
# validation.monitor "val_NDCG@1" maps to this codebase's metric name: SFTMethod
# emits sft/ndcg@K (an alias of sft/generation_ndcg@K) and the trainer
# stage-prefixes it to val/sft/ndcg@K.
MONITOR = "val/sft/ndcg@1"
MONITOR_MODE = "max"

# TIGER decoding constraints during evaluation.  With a single greedy beam,
# constraining decoding to catalog-valid semantic-ID prefixes keeps NDCG@1
# meaningful (an unconstrained greedy decode can emit a non-catalog sequence).
GENERATION_CONSTRAIN_VALID_SEQUENCES = True
GENERATION_EXCLUDE_SEEN_ITEMS = False

# Training (training.*).
LR = 1e-3                  # lr
MAX_EPOCHS = 300           # max_epochs
EVAL_ON_VAL_EVERY = 8      # eval_on_val_every -> validate_every_epochs
EARLY_STOPPING_PATIENCE = 30

# Dataloader (dataloader.*).  The preprocessing snippet lists batch_size=256, but
# the training hyperparameters give the GenRec batch sizes below, which take
# precedence for the actual DataLoaders.
TRAIN_BATCH_SIZE = 2048          # dataloader.train_batch_size
VALIDATION_BATCH_SIZE = 512      # dataloader.validation_batch_size


# --------------------------------------------------------------------------- #
# Semantic IDs (RQ-VAE over the audio embeddings).                             #
# --------------------------------------------------------------------------- #

def stream_yambda_audio_embeddings(
    parquet_path: Path,
    *,
    allowed_items: set[str] | None,
    embedding_column: str = "normalized_embed",
    fallback_column: str = "embed",
    item_id_column: str = "item_id",
    max_items: int | None = None,
    batch_rows: int = 65536,
) -> tuple[list[str], torch.Tensor, int]:
    """Stream item embeddings out of the Yambda parquet, filtered to ``allowed_items``.

    The parquet stores each track's audio embedding as a list<double>; reading it
    with pandas + ``to_dict`` explodes into tens of GB of Python objects.  This
    reader projects only ``item_id`` + the embedding column and materialises a
    single contiguous float32 matrix, so peak memory is one row-batch plus the
    kept vectors.
    """
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    emb_col = embedding_column if embedding_column in names else fallback_column
    if emb_col not in names or item_id_column not in names:
        raise ValueError(
            f"Parquet {parquet_path} is missing {item_id_column!r}/{emb_col!r}; has {names}."
        )

    allowed_arr: np.ndarray | None = None
    if allowed_items is not None:
        allowed_arr = np.fromiter((int(x) for x in allowed_items), dtype=np.int64, count=len(allowed_items))
        allowed_arr.sort()

    id_chunks: list[np.ndarray] = []
    mat_chunks: list[np.ndarray] = []
    dim: int | None = None
    total = 0
    for record_batch in pf.iter_batches(batch_size=batch_rows, columns=[item_id_column, emb_col]):
        id_idx = record_batch.schema.get_field_index(item_id_column)
        emb_idx = record_batch.schema.get_field_index(emb_col)
        ids = record_batch.column(id_idx).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if allowed_arr is not None:
            keep = np.isin(ids, allowed_arr)
            if not keep.any():
                continue
        else:
            keep = np.ones(len(ids), dtype=bool)

        emb = record_batch.column(emb_idx)
        n = len(emb)
        flat = emb.flatten().to_numpy(zero_copy_only=False)
        if dim is None:
            dim = int(flat.shape[0] // n)
        if flat.shape[0] == n * dim:
            mat = flat.reshape(n, dim).astype(np.float32, copy=False)
        else:  # ragged / null rows: fall back to a slower per-row conversion.
            mat = np.asarray([np.asarray(x, dtype=np.float32) for x in emb.to_pylist()], dtype=np.float32)

        id_chunks.append(ids[keep])
        mat_chunks.append(mat[keep])
        total += int(keep.sum())
        if max_items is not None and total >= max_items:
            break

    if not id_chunks:
        raise ValueError(
            "No audio embeddings matched the requested items. Check that the "
            "embeddings parquet covers the interaction item IDs."
        )

    all_ids = np.concatenate(id_chunks)
    all_mat = np.concatenate(mat_chunks, axis=0)
    uniq_ids, first_idx = np.unique(all_ids, return_index=True)  # dedup, keep first
    embeddings = torch.from_numpy(np.ascontiguousarray(all_mat[first_idx]))
    item_ids = [str(int(x)) for x in uniq_ids]
    if max_items is not None and len(item_ids) > max_items:
        item_ids = item_ids[:max_items]
        embeddings = embeddings[:max_items].contiguous()
    return item_ids, embeddings, int(dim or embeddings.shape[1])


def build_embedding_store(args: argparse.Namespace, allowed_items: set[str]) -> ItemEmbeddingStore:
    """Load the cached audio embedding store, or stream + build it for the catalog."""
    store_path = resolve(args.embedding_store_path)
    if store_path.exists() and not args.rebuild_embeddings:
        LOGGER.info("Reusing cached audio embedding store: %s", store_path)
        return ItemEmbeddingStore.load(store_path)

    embeddings_path = resolve(args.embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Yambda audio embeddings not found: {embeddings_path}. Provide "
            "--embeddings-path or a cached --embedding-store-path."
        )
    LOGGER.info(
        "Streaming audio embeddings for %s catalog items from %s ...",
        f"{len(allowed_items):,}", embeddings_path,
    )
    item_ids, embeddings, dim = stream_yambda_audio_embeddings(
        embeddings_path,
        allowed_items=allowed_items,
        embedding_column="normalized_embed",
        max_items=args.max_embedding_items,
    )
    store = ItemEmbeddingStore(
        item_ids=item_ids,
        embeddings=embeddings,
        metadata={
            "source": str(embeddings_path),
            "dataset": "yambda",
            "embedding_column": "normalized_embed",
            "embedding_dim": dim,
            "num_items": len(item_ids),
        },
    )
    store.save(store_path)
    LOGGER.info("Saved audio embedding store (%s items, dim=%s) to %s",
                f"{len(item_ids):,}", dim, store_path)
    return store


def train_and_save_tokenizer(
    args: argparse.Namespace,
    store: ItemEmbeddingStore,
    tokenizer_dir: Path,
) -> None:
    LOGGER.info(
        "Training RQ-VAE: items=%s dim=%s codebooks=%s codebook_size=%s device=%s epochs=%s",
        f"{store.num_items:,}", store.embedding_dim, NUM_CODEBOOKS, CODEBOOK_SIZE,
        args.rqvae_device, args.rqvae_epochs,
    )
    # 0 disables full-dataset dead-code resets, which is usually better at Yambda scale.
    dead_code_reset = (
        None if int(args.rqvae_dead_code_reset_interval) <= 0
        else int(args.rqvae_dead_code_reset_interval)
    )
    tokenizer = RQVAEItemTokenizer(
        RQVAEItemTokenizerConfig(
            rqvae=RQVAEConfig(
                input_dim=store.embedding_dim,
                latent_dim=int(args.rqvae_latent_dim),
                num_codebooks=NUM_CODEBOOKS,
                codebook_size=CODEBOOK_SIZE,
            ),
            train=RQVAETrainConfig(
                epochs=int(args.rqvae_epochs),
                batch_size=int(args.rqvae_batch_size),
                learning_rate=float(args.rqvae_lr),
                device=args.rqvae_device,
                seed=int(args.seed),
                dead_code_reset_interval=dead_code_reset,
            ),
            normalize="l2",
            collision_policy="append",
        )
    )
    tokenizer.fit(store.embeddings, store.item_ids)
    tokenizer.save(tokenizer_dir)
    if tokenizer.semantic_id_store is not None:
        LOGGER.info(
            "Semantic IDs: length=%s unique=%s collisions=%s",
            tokenizer.semantic_id_store.sid_length,
            tokenizer.semantic_id_store.unique_code_count,
            tokenizer.semantic_id_store.collision_count,
        )


def attach_semantic_ids(args: argparse.Namespace, data: YambdaDataModule) -> Path:
    """Ensure ``semantic_ids.json`` exists for the catalog and attach it to ``data``.

    The RQ-VAE is trained only over the catalog items (``data.id2item``), so the
    embedding store and codebook init stay small.  With semantic IDs attached
    post-setup, catalog items that lack an audio embedding (should be none for
    Yambda) simply act as padding; coverage is logged below.
    """
    tokenizer_dir = resolve(args.tokenizer_dir)
    semantic_id_path = tokenizer_dir / "semantic_ids.json"

    if not semantic_id_path.exists() or args.rebuild_semantic_ids:
        catalog = [str(item) for item in data.id2item[1:]]  # skip the <PAD> row
        if not catalog:
            raise ValueError("Empty item catalog after datamodule.setup().")
        store = build_embedding_store(args, allowed_items=set(catalog))
        train_and_save_tokenizer(args, store, tokenizer_dir)
    else:
        LOGGER.info("Reusing cached semantic IDs: %s", semantic_id_path)

    # The datamodule's semantic_id_path loader parses the file as JSONL and does
    # not round-trip the single-object semantic_ids.json the tokenizer writes, so
    # load the mapping directly and hand the dict to add_semantic_ids, which
    # applies the offset per cfg.offset_semantic_ids_by_one.
    with open(semantic_id_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"Unexpected semantic_ids.json content: {type(mapping)!r}")
    data.add_semantic_ids({str(item): tokens for item, tokens in mapping.items()})
    covered = len(data.semantic_ids or {})
    total = data.num_items
    LOGGER.info(
        "Semantic ID coverage: %s/%s catalog items (%.1f%%), sid_len=%s",
        f"{covered:,}", f"{total:,}", 100.0 * covered / max(total, 1), data.sid_len,
    )
    if covered < total:
        LOGGER.warning(
            "%s catalog items have no audio embedding / semantic ID; they will act "
            "as padding in histories and targets.", f"{total - covered:,}",
        )
    return semantic_id_path


# --------------------------------------------------------------------------- #
# Data, model, method, trainer.                                                #
# --------------------------------------------------------------------------- #

def build_data_config(args: argparse.Namespace) -> YambdaDataConfig:
    """The requested Yambda preprocessing (semantic IDs are attached post-setup).

    The preprocessing block is reproduced from the task.  ``keep_raw_interactions``
    is off by default: it keeps a full copy of every retained interaction record
    (millions of dicts at 50M scale) that SFT never reads, and is the main memory
    risk during ``setup()``.  Re-enable with ``--keep-raw-interactions`` if needed.
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
        # dataloader.train_batch_size / validation_batch_size (supersede the 256
        # in the preprocessing snippet).
        batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.validation_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        # Raw item strings are not needed for SFT: the collator provides
        # history_semantic_row_ids directly.  Skipping them lightens the batch.
        include_raw_in_batch=False,
        # Hash user IDs into the model's user-token table.
        user_hash_mod=int(MODEL["user_ids_count"]),
        seed=int(args.seed),
    )


def build_model(data: YambdaDataModule) -> TigerModel:
    if not data.semantic_ids or not data.sid_len:
        raise ValueError("TIGER requires semantic IDs on the datamodule.")
    return TigerModel(
        codebook_size=CODEBOOK_SIZE,
        sem_id_len=int(data.sid_len),
        user_ids_count=int(MODEL["user_ids_count"]),
        semantic_id_mapping=data.semantic_ids,
        embedding_dim=int(MODEL["embedding_dim"]),
        num_heads=int(MODEL["num_heads"]),
        num_encoder_layers=int(MODEL["num_encoder_layers"]),
        num_decoder_layers=int(MODEL["num_decoder_layers"]),
        dim_feedforward=int(MODEL["dim_feedforward"]),
        dropout=float(MODEL["dropout"]),
        num_beams=VAL_NUM_BEAMS,
        num_return_sequences=VAL_TOP_K,
        d_kv=int(MODEL["d_kv"]),
        layer_norm_eps=float(MODEL["layer_norm_eps"]),
        activation=str(MODEL["activation"]),
        initializer_range=float(MODEL["initializer_range"]),
    )


def build_method() -> SFTMethod:
    return SFTMethod(
        SFTConfig(
            # TIGER returns token logits, not item scores, so item-ranking
            # metrics do not apply.
            compute_item_metrics=False,
            evaluate_generation=True,
            generation_topk=METRICS_AT_K,
            generation_num_beams=VAL_NUM_BEAMS,
            generation_num_return_sequences=VAL_TOP_K,
            generation_constrain_valid_sequences=GENERATION_CONSTRAIN_VALID_SEQUENCES,
            generation_exclude_seen_items=GENERATION_EXCLUDE_SEEN_ITEMS,
        )
    )


def build_trainer_config(args: argparse.Namespace, checkpoint_dir: Path) -> TrainerConfig:
    return TrainerConfig(
        max_epochs=MAX_EPOCHS,
        gradient_accumulation_steps=1,
        max_grad_norm=args.max_grad_norm,
        use_amp=bool(args.use_amp),
        amp_dtype=str(args.amp_dtype),
        validate_every_epochs=EVAL_ON_VAL_EVERY,
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
    LOGGER.info("Device: %s | RQ-VAE device: %s", device, args.rqvae_device)

    data = YambdaDataModule(build_data_config(args))
    LOGGER.info("Loading Yambda 50M and building leave-one-out splits...")
    data.setup()
    LOGGER.info("Data summary: %s", json.dumps(data.summary(), sort_keys=True, default=str))

    semantic_id_path = attach_semantic_ids(args, data)
    if not data.semantic_ids or not data.sid_len:
        raise ValueError(
            "No semantic IDs matched the Yambda catalog. Check that the audio "
            "embeddings cover the interaction items."
        )

    model = build_model(data)
    method = build_method()
    LOGGER.info(
        "TIGER: total_params=%s trainable=%s sid_len=%s semantic_vocab=%s",
        f"{model.num_parameters():,}",
        f"{model.num_parameters(trainable_only=True):,}",
        data.sid_len,
        model.config_dict()["semantic_vocab_size"],
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
            "semantic_id_path": str(semantic_id_path),
        },
    )

    LOGGER.info("Starting training: max_epochs=%s validate_every=%s monitor=%s",
                MAX_EPOCHS, EVAL_ON_VAL_EVERY, MONITOR)
    fit = trainer.fit(data.train_dataloader(), data.val_dataloader())

    # Final test with the best (or last) checkpoint.
    test_metrics: dict[str, float] = {}
    checkpoint = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint is not None:
        LOGGER.info("Evaluating on the test split with checkpoint: %s", checkpoint)
        trainer.load_checkpoint(checkpoint, load_optimizer=False, load_scheduler=False)
        test_metrics = trainer.test(data.test_dataloader())

    summary = {
        "run_dir": str(run_dir),
        "epochs_completed": int(trainer.state.epoch + 1),
        "steps_completed": int(trainer.state.global_step),
        "monitor": MONITOR,
        "best_metric": trainer.state.best_metric,
        "best_epoch": trainer.state.best_epoch,
        "validation_metrics": fit.val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": fit.best_checkpoint_path,
        "last_checkpoint": fit.last_checkpoint_path,
    }
    write_json(run_dir / "final_summary.json", summary)
    write_json(run_dir / "training_history.json", trainer.state.history)
    LOGGER.info("Done. Best %s=%s at epoch %s.", MONITOR, trainer.state.best_metric, trainer.state.best_epoch)
    LOGGER.info("Test metrics: %s", json.dumps(test_metrics, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# CLI + small helpers.                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TIGER with SFT on Yambda 50M.")
    # Runtime.
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default: cuda if available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 is the safest default on Windows).")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/yambda_tiger_learning"))
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--keep-raw-interactions", action="store_true",
                   help="Keep raw interaction records (memory-heavy at 50M; unused by SFT).")
    # Dataloader batch sizes (GenRec defaults).
    p.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--validation-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    # Optimization extras not pinned by the spec.
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--use-amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    p.add_argument("--amp-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    # Embeddings / tokenizer (cache paths + rebuild switches).
    p.add_argument("--embeddings-path", type=Path,
                   default=Path("data/raw/yambda/embeddings.parquet"))
    p.add_argument("--embedding-store-path", type=Path,
                   default=Path("artifacts/item_embeddings/yambda_audio_embeddings.pt"))
    p.add_argument("--tokenizer-dir", type=Path,
                   default=Path("artifacts/tokenizers/yambda_rqvae"))
    p.add_argument("--rebuild-embeddings", action="store_true",
                   help="Rebuild the audio embedding store even if cached.")
    p.add_argument("--rebuild-semantic-ids", action="store_true",
                   help="Retrain the RQ-VAE tokenizer even if semantic_ids.json exists.")
    p.add_argument("--max-embedding-items", type=int, default=None,
                   help="Cap the number of catalog items embedded (debugging).")
    p.add_argument("--rqvae-epochs", type=int, default=50)
    p.add_argument("--rqvae-batch-size", type=int, default=4096)
    p.add_argument("--rqvae-lr", type=float, default=1e-3)
    p.add_argument("--rqvae-latent-dim", type=int, default=32)
    p.add_argument("--rqvae-device", default=None,
                   help="Device for RQ-VAE training. Default: cuda if available.")
    p.add_argument("--rqvae-dead-code-reset-interval", type=int, default=0,
                   help="0 disables dead-code resets (recommended at Yambda scale).")
    args = p.parse_args()
    if args.rqvae_device is None:
        args.rqvae_device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def make_status_logger():
    keys = [
        "train/loss", "train/sft/loss", "train/sft/token_accuracy",
        "val/loss", MONITOR, "val/sft/generation_ndcg@1", "val/sft/generation_coverage@1",
        "test/loss", MONITOR.replace("val/", "test/"),
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
