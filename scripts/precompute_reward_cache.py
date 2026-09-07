#!/usr/bin/env python
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

# pyarrow.dataset's native module must load before torch on Windows;
# pandas.read_parquet imports it lazily and crashes otherwise.
import pyarrow.dataset  # noqa: F401
import torch

faulthandler.enable()
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tiger as tiger_runner
from flowrec.data.datasets import SequenceDataset
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON
from flowrec.rewards.interaction import PositiveRewardModel
from flowrec.rewards.sasrec import SASRecRewardModel
from flowrec.rewards.similarity import SimilarityRewardModel
from flowrec.rewards.svd import SVDRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Precompute full-catalogue rewards for TIGER/GFlow training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size-examples", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalization-scope", choices=["row", "batch"], default="row")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--store-power-sums", type=float, nargs="*", default=[])
    parser.add_argument(
        "--per-user-last-n",
        type=int,
        default=None,
        help="Cache only the last N examples of every user (deterministic "
             "fine-tuning subset; the datamodule and its catalog stay untouched). "
             "Training must use per_user_tail_indices with the same N.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return value
    return value


def slice_batch(batch: Mapping[str, Any], n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.size(0) >= n:
            out[key] = value[:n]
        elif isinstance(value, list) and len(value) >= n:
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def per_user_tail_indices(dataset, last_n: int) -> list[int]:
    """Deterministic per-user fine-tuning subset of a SequenceDataset.

    Returns the dataset indices of every user's last ``last_n`` examples, in
    original dataset order. This applies the train_target_policy='last_n'
    reduction on top of an already-built split, so the item catalog (which the
    post-split cold-item filter derives from the FULL train example layout)
    is untouched — a checkpoint trained on the full layout still loads.
    """
    frame = getattr(dataset, "examples", None)
    if frame is None or "user_id" not in frame.columns:
        raise KeyError(
            "per-user subsetting requires a SequenceDataset with a user_id column"
        )
    tail = frame.groupby("user_id", sort=False).tail(int(last_n))
    return [int(i) for i in tail.index]


def build_cache_loader(
    data, split: str, batch_size: int, num_workers: int,
    per_user_last_n: int | None = None,
) -> DataLoader:
    dataset = {
        "train": data.train_dataset,
        "val": data.val_dataset,
        "test": data.test_dataset,
    }[split]
    if dataset is None:
        raise RuntimeError(f"Data split {split!r} is not initialized.")
    if per_user_last_n is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, per_user_tail_indices(dataset, per_user_last_n))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=data._make_collator(),
    )


def build_reward_components(config: Mapping[str, Any], data, train_loader, config_path: Path, device: torch.device):
    method_cfg = dict(config.get("method", {}))
    reward_name = str(method_cfg.get("reward_model", "rating")).lower()
    reward_cfg = dict(method_cfg.get("reward", {}))
    svd_cfg = dict(method_cfg.get("svd", {}))
    sasrec_cfg = dict(method_cfg.get("sasrec", {}))
    seed = int(config.get("seed", 42))
    reward_epsilon = GFLOW_REWARD_EPSILON

    needs_sasrec = tiger_runner.reward_config_requires_model(reward_name, reward_cfg, "sasrec")
    needs_svd = tiger_runner.reward_config_requires_model(reward_name, reward_cfg, "svd")
    sasrec_model = (
        tiger_runner.build_sasrec_teacher_model(data, sasrec_cfg, config_path=config_path)
        if needs_sasrec
        else None
    )
    svd_model = (
        tiger_runner.build_svd_scoring_model(
            data=data,
            train_loader=train_loader,
            seed=seed,
            svd_cfg=svd_cfg,
        )
        if needs_svd
        else None
    )

    specs: list[dict[str, Any]] = []
    if reward_name in {"mix", "mixed", "mixture", "sum"}:
        default_norm = str(reward_cfg.get("normalize", reward_cfg.get("normalization", "batch_max")))
        for component in tiger_runner.reward_components(reward_cfg):
            name = tiger_runner.component_reward_name(component)
            cfg = tiger_runner.component_reward_config(component)
            model = tiger_runner.build_reward_model(
                name,
                data=data,
                train_loader=train_loader,
                config_path=config_path,
                seed=seed,
                reward_cfg=cfg,
                svd_cfg=svd_cfg,
                svd_model=svd_model,
                sasrec_model=sasrec_model,
                reward_epsilon=reward_epsilon,
                allow_default_rating=False,
            )
            specs.append(
                {
                    "name": name,
                    "model": model.to(device).eval(),
                    "weight": float(component.get("weight", 1.0)),
                    "normalizer": str(component.get("normalize", component.get("normalization", default_norm))),
                }
            )
    else:
        model = tiger_runner.build_reward_model(
            reward_name,
            data=data,
            train_loader=train_loader,
            config_path=config_path,
            seed=seed,
            reward_cfg=reward_cfg,
            svd_cfg=svd_cfg,
            svd_model=svd_model,
            sasrec_model=sasrec_model,
            reward_epsilon=reward_epsilon,
            allow_default_rating=False,
        )
        specs.append({"name": reward_name, "model": model.to(device).eval(), "weight": 1.0, "normalizer": "none"})

    for spec in specs:
        for param in spec["model"].parameters():
            param.requires_grad_(False)
    return specs, reward_epsilon


def compute_component_rewards(
    spec: Mapping[str, Any],
    batch: Mapping[str, Any],
    item_tokens: torch.Tensor,
    semantic_row_to_item_id: torch.Tensor,
    item_id_to_semantic_row: torch.Tensor,
    reward_epsilon: float,
) -> torch.Tensor:
    model = spec["model"]
    bsz = int(batch["target_item_ids"].size(0))
    num_items = int(item_tokens.size(0))
    device = item_tokens.device

    if isinstance(model, SimilarityRewardModel):
        mode = getattr(model.cfg, "mode", "exact")
        sharpness = float(getattr(model.cfg, "sharpness", 1.0))
        if mode == "exact":
            target = batch[model.cfg.target_token_key].to(device).long()
            rewards = item_tokens.unsqueeze(0).eq(target.unsqueeze(1)).float().mean(dim=-1)
            if sharpness != 1.0:
                rewards = rewards.clamp_min(0.0).pow(sharpness)
            return rewards.clamp_min(float(model.cfg.reward_epsilon))
        if mode == "embedding_cosine":
            # Full-catalog form of SimilarityRewardModel._embedding_similarity:
            # (cos(z_item, z_target) + 1) / 2 over the normalized content-embedding
            # table (indexed by internal item id), laid out by semantic row.
            table = model.item_embeddings.to(device)  # [V, D], unit rows
            vocab_size = int(table.shape[0])
            target_item_ids = batch[model.cfg.target_item_key].to(device).long()
            target_vecs = table[target_item_ids.clamp(0, vocab_size - 1)]  # [B, D]
            row_item_ids = semantic_row_to_item_id.to(device).long()
            valid_rows = row_item_ids.ge(0) & row_item_ids.lt(vocab_size)
            row_vecs = table[row_item_ids.clamp(0, vocab_size - 1)]  # [num_items, D]
            sim = (target_vecs @ row_vecs.T + 1.0) / 2.0  # [B, num_items]
            sim = sim.masked_fill(~valid_rows.unsqueeze(0), 0.0)
            if sharpness != 1.0:
                sim = sim.clamp_min(0.0).pow(sharpness)
            return sim.clamp_min(float(model.cfg.reward_epsilon))
        if mode != "hierarchical":
            raise NotImplementedError(
                "precompute_reward_cache supports similarity modes 'exact', "
                f"'embedding_cosine', and 'hierarchical'; got mode={mode!r}."
            )
        # hierarchical: fall through to the generic trajectory path below.
        # SimilarityRewardModel._hierarchical_similarity is vectorized over
        # [B, N, L] candidates, so scoring the full catalog per batch through
        # the model itself is both exact and cheap.

    if isinstance(model, PositiveRewardModel):
        rewards = torch.full(
            (bsz, num_items),
            fill_value=float(model.cfg.negative_reward),
            device=device,
            dtype=torch.float32,
        )
        target_item_ids = batch["target_item_ids"].to(device).long()
        valid = (target_item_ids >= 0) & (target_item_ids < item_id_to_semantic_row.numel())
        safe = target_item_ids.clamp_min(0).clamp_max(max(item_id_to_semantic_row.numel() - 1, 0))
        rows = item_id_to_semantic_row[safe]
        valid = valid & rows.ge(0) & rows.lt(num_items)
        if valid.any():
            rewards[valid, rows[valid]] = float(model.cfg.positive_reward)
        return rewards.clamp_min(float(model.cfg.reward_epsilon))

    if isinstance(model, SASRecRewardModel):
        model.sasrec_model.eval()
        raw_scores = model.sasrec_model.score_items(batch)
        semantic_scores = gather_semantic_scores(raw_scores, semantic_row_to_item_id)
        rewards = model.transform_scores(semantic_scores)
        return rewards.clamp_min(float(model.cfg.reward_epsilon))

    if isinstance(model, SVDRewardModel):
        raw_scores = model.svd_model.score_items(batch)
        semantic_scores = gather_semantic_scores(raw_scores, semantic_row_to_item_id)
        rewards = model.transform_scores(semantic_scores)
        return rewards.clamp_min(float(model.cfg.reward_epsilon))

    # Fallback for custom reward models. This is slower but keeps the script extensible.
    mask = torch.ones_like(item_tokens, dtype=torch.bool, device=device)
    rows = torch.arange(num_items, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    trajectories = SemanticTrajectoryBatch(
        candidate_token_ids=item_tokens.unsqueeze(0).expand(bsz, -1, -1),
        candidate_token_mask=mask.unsqueeze(0).expand(bsz, -1, -1),
        is_positive=catalog_positive_mask(batch, item_tokens),
        candidate_item_row_ids=rows,
    )
    return model(batch, trajectories).float().clamp_min(reward_epsilon)


def gather_semantic_scores(raw_scores: torch.Tensor, semantic_row_to_item_id: torch.Tensor) -> torch.Tensor:
    scores = raw_scores.float()
    device = scores.device
    mapping = semantic_row_to_item_id.to(device).long()
    valid = mapping.ge(0) & mapping.lt(scores.size(1))
    safe = mapping.clamp_min(0).clamp_max(max(scores.size(1) - 1, 0))
    semantic_scores = scores[:, safe]
    if (~valid).any():
        semantic_scores[:, ~valid] = -float("inf")
    return semantic_scores


def normalize_component(rewards: torch.Tensor, mode: str, scope: str, eps: float) -> torch.Tensor:
    mode = mode.lower()
    if mode in {"none", "identity"}:
        return rewards.clamp_min(eps)
    if scope == "batch":
        if mode in {"batch_max", "max", "row_max"}:
            return (rewards / rewards.detach().amax().clamp_min(eps)).clamp_min(eps)
        if mode in {"batch_mean", "mean", "row_mean"}:
            return (rewards / rewards.detach().mean().clamp_min(eps)).clamp_min(eps)
        if mode in {"batch_minmax", "minmax", "row_minmax"}:
            min_val = rewards.detach().amin()
            span = (rewards.detach().amax() - min_val).clamp_min(eps)
            return ((rewards - min_val) / span).clamp_min(eps)
    else:
        if mode in {"batch_max", "max", "row_max"}:
            return (rewards / rewards.detach().amax(dim=1, keepdim=True).clamp_min(eps)).clamp_min(eps)
        if mode in {"batch_mean", "mean", "row_mean"}:
            return (rewards / rewards.detach().mean(dim=1, keepdim=True).clamp_min(eps)).clamp_min(eps)
        if mode in {"batch_minmax", "minmax", "row_minmax"}:
            min_val = rewards.detach().amin(dim=1, keepdim=True)
            span = (rewards.detach().amax(dim=1, keepdim=True) - min_val).clamp_min(eps)
            return ((rewards - min_val) / span).clamp_min(eps)
    raise ValueError(f"Unknown reward normalizer {mode!r}")


def catalog_positive_mask(batch: Mapping[str, Any], item_tokens: torch.Tensor) -> torch.Tensor:
    target = batch["target_token_ids"].to(item_tokens.device).long()
    target_mask = batch.get("target_token_mask")
    if torch.is_tensor(target_mask):
        target_mask = target_mask.to(item_tokens.device).bool()
    else:
        target_mask = torch.ones_like(target, dtype=torch.bool, device=item_tokens.device)
    matches = item_tokens.unsqueeze(0).eq(target.unsqueeze(1)) | ~target_mask.unsqueeze(1)
    return matches.all(dim=-1)


def make_mixed_rewards(
    specs: list[Mapping[str, Any]],
    batch: Mapping[str, Any],
    item_tokens: torch.Tensor,
    semantic_row_to_item_id: torch.Tensor,
    item_id_to_semantic_row: torch.Tensor,
    reward_epsilon: float,
    normalization_scope: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    mixed = None
    component_means: dict[str, float] = {}
    for idx, spec in enumerate(specs):
        rewards = compute_component_rewards(
            spec,
            batch,
            item_tokens,
            semantic_row_to_item_id,
            item_id_to_semantic_row,
            reward_epsilon,
        )
        rewards = rewards.float().clamp_min(reward_epsilon)
        normalized = normalize_component(rewards, str(spec["normalizer"]), normalization_scope, reward_epsilon)
        term = normalized * float(spec["weight"])
        mixed = term if mixed is None else mixed + term
        component_means[f"component_{idx}_{spec['name']}_mean"] = float(normalized.detach().mean().cpu())
    if mixed is None:
        raise RuntimeError("No reward components were built.")
    return mixed.clamp_min(reward_epsilon), component_means


def append_batch(accum: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    accum.setdefault(key, []).append(value.detach().cpu())


def power_sum_key(power: float) -> str:
    slug = format(float(power), ".12g").replace("-", "m").replace(".", "p")
    return f"reward_sum_power_{slug}"


def flush_shard(
    accum: dict[str, list[torch.Tensor]],
    output_dir: Path,
    shard_idx: int,
    example_start: int,
    dtype: torch.dtype,
) -> tuple[Path | None, int]:
    if not accum:
        return None, 0
    payload: dict[str, Any] = {"example_start": example_start}
    count = None
    for key, values in accum.items():
        tensor = torch.cat(values, dim=0)
        if key == "full_rewards":
            tensor = tensor.to(dtype)
        payload[key] = tensor
        if count is None:
            count = int(tensor.size(0))
    payload["num_examples"] = int(count or 0)
    path = output_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(payload, path)
    accum.clear()
    return path, int(count or 0)


def main() -> None:
    args = parse_args()
    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)
    config.setdefault("dataloader", {})["train_batch_size"] = int(args.batch_size)
    config.setdefault("dataloader", {})["batch_size"] = int(args.batch_size)
    config.setdefault("dataloader", {})["validation_batch_size"] = int(args.batch_size)
    config.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    config.setdefault("dataloader", {})["persistent_workers"] = int(args.num_workers) > 0
    config.setdefault("dataloader", {})["include_raw_in_batch"] = False

    seed = int(config.get("seed", 42))
    tiger_runner.seed_everything(seed)
    device = torch.device(args.device)
    store_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("Reward cache precompute requires data.semantic_id_path.")

    train_loader_for_models = data.train_dataloader()
    specs, reward_epsilon = build_reward_components(config, data, train_loader_for_models, config_path, device)
    cache_loader = build_cache_loader(
        data, args.split, int(args.batch_size), int(args.num_workers),
        per_user_last_n=args.per_user_last_n,
    )

    model = tiger_runner.build_model(config, data, config_path=config_path)
    item_tokens = model.item_semantic_ids.to(device).long()
    num_items = int(item_tokens.size(0))
    semantic_row_to_item_id = tiger_runner.item_id_by_semantic_row(data)
    if semantic_row_to_item_id is None:
        semantic_row_to_item_id = torch.arange(num_items, dtype=torch.long)
    semantic_row_to_item_id = semantic_row_to_item_id.to(device).long()
    item_id_to_semantic_row = tiger_runner.item_id_to_semantic_row(data).to(device).long()

    output_dir = tiger_runner.resolve_path(args.output_dir, base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "normalization_scope": args.normalization_scope,
        "dtype": args.dtype,
        "store_full_rewards": True,
        "stored_power_sums": [float(x) for x in args.store_power_sums],
        "per_user_last_n": args.per_user_last_n,
        "reward_epsilon": reward_epsilon,
        "data": data.summary(),
        "data_config": jsonable(asdict(data_cfg)),
        "num_catalog_items": num_items,
        "components": [
            {"name": spec["name"], "weight": spec["weight"], "normalizer": spec["normalizer"]}
            for spec in specs
        ],
        "shards": [],
    }
    (output_dir / "manifest.prelim.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    accum: dict[str, list[torch.Tensor]] = {}
    shard_idx = 0
    next_shard_start = 0
    processed = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(cache_loader):
            if args.max_examples is not None and processed >= int(args.max_examples):
                break
            if args.max_examples is not None:
                remaining = int(args.max_examples) - processed
                if remaining < len(raw_batch["user_ids"]):
                    raw_batch = slice_batch(raw_batch, remaining)
            batch = move_to_device(raw_batch, device)
            rewards, component_means = make_mixed_rewards(
                specs,
                batch,
                item_tokens,
                semantic_row_to_item_id,
                item_id_to_semantic_row,
                reward_epsilon,
                args.normalization_scope,
            )
            cached_rewards = rewards.to(store_dtype).float().clamp_min(reward_epsilon)
            reward_sum = cached_rewards.sum(dim=1).clamp_min(reward_epsilon)
            reward_log_sum = (cached_rewards * cached_rewards.log()).sum(dim=1)
            reward_entropy = reward_sum.log() - reward_log_sum / reward_sum
            target_item_ids = batch["target_item_ids"].long()
            safe_targets = target_item_ids.clamp_min(0).clamp_max(max(item_id_to_semantic_row.numel() - 1, 0))
            target_rows = item_id_to_semantic_row[safe_targets]
            valid_targets = (target_item_ids >= 0) & (target_item_ids < item_id_to_semantic_row.numel())
            target_rows = torch.where(valid_targets, target_rows, torch.full_like(target_rows, -1))
            target_rewards = torch.gather(
                cached_rewards,
                1,
                target_rows.clamp_min(0).clamp_max(num_items - 1).unsqueeze(1),
            ).squeeze(1)
            target_rewards = torch.where(target_rows.ge(0), target_rewards, torch.zeros_like(target_rewards))
            target_ranks = cached_rewards.gt(target_rewards.unsqueeze(1)).sum(dim=1).long() + 1
            target_ranks = torch.where(target_rows.ge(0), target_ranks, torch.full_like(target_ranks, -1))

            append_batch(accum, "user_ids", batch["user_ids"].long())
            append_batch(accum, "target_item_ids", target_item_ids.cpu())
            append_batch(accum, "target_semantic_row_ids", target_rows.cpu())
            append_batch(accum, "target_reward_ranks", target_ranks.cpu())
            append_batch(accum, "reward_sum", reward_sum.unsqueeze(1))
            append_batch(accum, "reward_entropy", reward_entropy.unsqueeze(1))
            append_batch(accum, "full_rewards", cached_rewards)
            for power in args.store_power_sums:
                if float(power) <= 0.0:
                    raise ValueError("--store-power-sums values must be positive.")
                powered_sum = cached_rewards.pow(float(power))
                powered_sum = powered_sum.clamp_min(reward_epsilon).sum(dim=1)
                append_batch(accum, power_sum_key(float(power)), powered_sum.unsqueeze(1))
            append_batch(accum, "reward_effective_items", reward_entropy.exp().unsqueeze(1))
            append_batch(accum, "reward_max", cached_rewards.amax(dim=1, keepdim=True))

            bsz = int(target_item_ids.size(0))
            processed += bsz
            if processed - next_shard_start >= int(args.shard_size_examples):
                path, count = flush_shard(accum, output_dir, shard_idx, next_shard_start, store_dtype)
                if path is not None:
                    manifest["shards"].append({"path": str(path), "example_start": next_shard_start, "num_examples": count})
                    shard_idx += 1
                    next_shard_start += count
                    (output_dir / "manifest.prelim.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                json.dumps(
                    {
                        "type": "progress",
                        "batch_idx": batch_idx,
                        "processed": processed,
                        "examples_per_sec": processed / elapsed,
                        "reward_sum_mean": float(reward_sum.mean().cpu()),
                        **component_means,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    path, count = flush_shard(accum, output_dir, shard_idx, next_shard_start, store_dtype)
    if path is not None:
        manifest["shards"].append({"path": str(path), "example_start": next_shard_start, "num_examples": count})
    manifest["num_examples"] = processed
    manifest["elapsed_sec"] = time.time() - start_time
    manifest["examples_per_sec"] = processed / max(manifest["elapsed_sec"], 1e-6)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    prelim = output_dir / "manifest.prelim.json"
    if prelim.exists():
        prelim.unlink()
    print(json.dumps({"type": "done", "manifest": str(output_dir / "manifest.json"), "num_examples": processed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
