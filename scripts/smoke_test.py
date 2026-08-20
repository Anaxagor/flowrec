from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.version_info < (3, 10) or "slots" not in inspect.signature(dataclasses.dataclass).parameters:
    raise SystemExit(
        "FlowRec requires Python >= 3.10 with stdlib dataclasses supporting "
        "`@dataclass(slots=True)`. Current Python: "
        f"{sys.version.split()[0]}, dataclasses: {getattr(dataclasses, '__file__', 'stdlib')}. "
        "Use a Python 3.10+ venv. If you are already on Python 3.10+, uninstall the "
        "old dataclasses backport from the venv with `pip uninstall dataclasses`."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.candidates.sampler import (
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.preference.config import DPOConfig
from flowrec.ft_methods.preference.method import DPOMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.tiger import TigerModel
from flowrec.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast FlowRec smoke test on synthetic data.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--work-dir", type=Path, default=None, help="Keep artifacts in this directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--skip-gflow", action="store_true")
    parser.add_argument("--skip-trainer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.num_threads)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="flowrec_smoke_") as tmp:
            run_smoke(Path(tmp), device=device, args=args)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_smoke(args.work_dir, device=device, args=args)

    print(f"OK: FlowRec smoke test passed in {time.time() - started:.1f}s on {device}.")


def run_smoke(work_dir: Path, *, device: torch.device, args: argparse.Namespace) -> None:
    reviews_path = work_dir / "reviews.jsonl"
    metadata_path = work_dir / "metadata.jsonl"
    semantic_ids_path = work_dir / "semantic_ids.json"

    write_synthetic_data(reviews_path, metadata_path, semantic_ids_path)
    data = build_data_module(reviews_path, metadata_path, semantic_ids_path, seed=int(args.seed))
    data.setup()
    summary = data.summary()
    assert summary["num_train_examples"] > 0, summary
    assert summary["num_val_examples"] > 0, summary
    assert summary["num_test_examples"] > 0, summary
    print("data:", json.dumps(summary, sort_keys=True))

    train_batch = move_tensors(next(iter(data.train_dataloader())), device)
    val_batch = move_tensors(next(iter(data.val_dataloader())), device)

    model = build_tiny_tiger(data).to(device)
    print("model:", json.dumps(model.config_dict(), sort_keys=True))

    smoke_sft_forward_backward(model, train_batch)
    smoke_generation(model, val_batch)

    if not args.skip_dpo:
        smoke_dpo(model, train_batch)

    if not args.skip_gflow:
        smoke_gflow(model, train_batch, work_dir)

    if not args.skip_trainer:
        smoke_trainer(data, device=device, work_dir=work_dir, seed=int(args.seed))


def write_synthetic_data(
    reviews_path: Path,
    metadata_path: Path,
    semantic_ids_path: Path,
) -> None:
    item_ids = [f"I{i:03d}" for i in range(10)]
    users = {
        "U000": item_ids[0:6],
        "U001": item_ids[2:8],
        "U002": item_ids[4:10],
        "U003": [item_ids[i] for i in [1, 3, 5, 7, 9, 0]],
    }

    rows: list[dict[str, Any]] = []
    timestamp = 1_700_000_000
    for user_id, history in users.items():
        for pos, item_id in enumerate(history):
            rows.append(
                {
                    "user_id": user_id,
                    "asin": item_id,
                    "rating": float(1 + ((pos + len(user_id)) % 5)),
                    "timestamp": timestamp,
                }
            )
            timestamp += 60

    with reviews_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with metadata_path.open("w", encoding="utf-8") as f:
        for item_id in item_ids:
            f.write(
                json.dumps(
                    {
                        "asin": item_id,
                        "title": f"Synthetic item {item_id}",
                        "description": f"Small smoke-test item {item_id}.",
                        "categories": [["Smoke", "FlowRec"]],
                    }
                )
                + "\n"
            )

    codebook_size = 8
    semantic_ids = {
        item_id: [
            idx % codebook_size,
            codebook_size + ((idx // codebook_size) % codebook_size),
            2 * codebook_size + idx,
        ]
        for idx, item_id in enumerate(item_ids)
    }
    with semantic_ids_path.open("w", encoding="utf-8") as f:
        json.dump(semantic_ids, f)


def build_data_module(
    reviews_path: Path,
    metadata_path: Path,
    semantic_ids_path: Path,
    *,
    seed: int,
) -> AmazonReviewDataModule:
    return AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=reviews_path,
            metadata_path=metadata_path,
            semantic_id_path=semantic_ids_path,
            item_key="asin",
            min_user_interactions=5,
            min_item_interactions=1,
            split_strategy="leave_one_out",
            min_history_len=1,
            train_all_prefixes=True,
            max_history_len=5,
            batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            include_raw_in_batch=True,
            offset_semantic_ids_by_one=False,
            drop_items_missing_semantic_id=True,
            user_hash_mod=32,
            seed=seed,
        )
    )


def build_tiny_tiger(data: AmazonReviewDataModule) -> TigerModel:
    if data.semantic_ids is None or data.sid_len is None:
        raise RuntimeError("Synthetic smoke data should have semantic IDs.")
    return TigerModel(
        codebook_size=8,
        sem_id_len=int(data.sid_len),
        user_ids_count=32,
        semantic_id_mapping=data.semantic_ids,
        embedding_dim=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        num_beams=2,
        num_return_sequences=2,
        d_kv=16,
    )


def smoke_sft_forward_backward(model: TigerModel, batch: dict[str, Any]) -> None:
    method = SFTMethod(
        SFTConfig(
            compute_train_metrics=True,
            compute_eval_metrics=True,
            compute_token_metrics=True,
            evaluate_generation=False,
        )
    )
    model.train()
    method.train()
    out = method.compute_loss(model, batch, stage="train", step=0)
    assert torch.isfinite(out.loss), out.detached_metrics()
    out.loss.backward()
    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_norm > 0.0
    model.zero_grad(set_to_none=True)
    print("sft:", json.dumps(out.detached_metrics(), sort_keys=True))


@torch.no_grad()
def smoke_generation(model: TigerModel, batch: dict[str, Any]) -> None:
    model.eval()
    generated = model.generate(
        batch,
        num_beams=2,
        num_return_sequences=2,
        constrain_valid_sequences=True,
        exclude_seen_items=True,
    )
    expected = (batch["user_ids"].size(0), 2, model.config_dict()["sem_id_len"])
    assert tuple(generated.shape) == expected, (tuple(generated.shape), expected)
    assert torch.isfinite(generated.float()).all()

    method = SFTMethod(
        SFTConfig(
            compute_eval_metrics=True,
            compute_token_metrics=True,
            evaluate_generation=True,
            generation_topk=(1, 2),
            generation_num_beams=2,
            generation_num_return_sequences=2,
            generation_constrain_valid_sequences=True,
            generation_exclude_seen_items=True,
        )
    )
    out = method.compute_loss(model, batch, stage="val", step=0)
    metrics = out.detached_metrics()
    assert "sft/generation_recall@1" in metrics
    print("generation:", json.dumps({"shape": list(generated.shape), **metrics}, sort_keys=True))


def smoke_gflow(model: TigerModel, batch: dict[str, Any], work_dir: Path) -> None:
    cache_dir = work_dir / "gflow_reward_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    example_indices = batch["example_indices"].detach().cpu().long()
    num_examples = int(example_indices.max().item()) + 1
    item_tokens = model.item_semantic_ids.detach().cpu().long()
    full_rewards = torch.ones((num_examples, item_tokens.size(0)), dtype=torch.float32)
    target_tokens = batch["target_token_ids"].detach().cpu().long()
    target_rows = torch.zeros(num_examples, dtype=torch.long)
    for batch_row, example_idx in enumerate(example_indices.tolist()):
        matches = item_tokens.eq(target_tokens[batch_row]).all(dim=1).nonzero(as_tuple=True)[0]
        target_rows[example_idx] = matches[0] if matches.numel() else 0
    torch.save(
        {
            "example_start": 0,
            "num_examples": num_examples,
            "target_semantic_row_ids": target_rows,
            "full_rewards": full_rewards,
            "reward_sum": full_rewards.sum(dim=1, keepdim=True),
        },
        cache_dir / "shard_00000.pt",
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_examples": num_examples,
                "shards": [
                    {"path": "shard_00000.pt", "example_start": 0, "num_examples": num_examples}
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = GFlowConfig(
        objective="tb",
        sft_weight=1.0,
        gflow_weight=0.01,
        reward_cache_path=str(cache_dir),
    )
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="random",
            num_negatives=1,
            include_positive=True,
            exclude_history=False,
        ),
        item_semantic_ids=model.item_semantic_ids.detach().cpu(),
    )
    method = GFlowMethod(
        cfg,
        candidate_sampler=sampler,
        sft_method=SFTMethod(
            SFTConfig(
                compute_train_metrics=True,
                compute_eval_metrics=True,
                compute_token_metrics=True,
                compute_item_metrics=False,
                evaluate_generation=False,
            )
        ),
    ).to(model.device)
    model.train()
    method.train()
    out = method.compute_loss(model, batch, stage="train", step=0)
    assert torch.isfinite(out.loss), out.detached_metrics()
    optimizer = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(method.parameters()) if p.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    optimizer.step()
    metrics = out.detached_metrics()
    assert "gflow/gflow_loss" in metrics
    assert "gflow/log_z_mean" in metrics
    print("gflow:", json.dumps(metrics, sort_keys=True))


def smoke_dpo(model: TigerModel, batch: dict[str, Any]) -> None:
    cfg = DPOConfig(
        beta=0.2,
        negative_sampler="random",
        reference_model_strategy="frozen_init",
        loss_reduction="mean",
        sft_weight=0.0,
        evaluate_generation=False,
    )
    method = DPOMethod(cfg, model=model).to(model.device)
    model.train()
    method.train()

    out = method.compute_loss(model, batch, stage="train", step=0)
    assert torch.isfinite(out.loss), out.detached_metrics()

    model.zero_grad(set_to_none=True)
    method.zero_grad(set_to_none=True)
    out.loss.backward()

    model_grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    reference_grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in method.reference_model.parameters()
        if p.grad is not None
    )
    assert model_grad_norm > 0.0
    assert reference_grad_norm == 0.0

    metrics = out.detached_metrics()
    assert "dpo/loss" in metrics
    assert "dpo/preference_accuracy" in metrics
    print("dpo:", json.dumps(metrics, sort_keys=True))
    model.zero_grad(set_to_none=True)
    method.zero_grad(set_to_none=True)


def smoke_trainer(
    data: AmazonReviewDataModule,
    *,
    device: torch.device,
    work_dir: Path,
    seed: int,
) -> None:
    torch.manual_seed(seed + 1)
    model = build_tiny_tiger(data)
    method = SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False))
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=1,
            max_steps=1,
            max_train_batches=1,
            validate_every_epochs=None,
            validate_every_steps=None,
            checkpoint_dir=work_dir / "checkpoints",
            save_best=False,
            save_last=True,
            progress_bar=False,
            log_every_steps=1,
        ),
        device=device,
        lr=1e-4,
    )
    result = trainer.fit(data.train_dataloader(), data.val_dataloader())
    assert result.state.global_step == 1
    assert result.last_checkpoint_path is not None
    print(
        "trainer:",
        json.dumps(
            {
                "global_step": result.state.global_step,
                "last_checkpoint": result.last_checkpoint_path,
            },
            sort_keys=True,
        ),
    )


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


if __name__ == "__main__":
    main()
