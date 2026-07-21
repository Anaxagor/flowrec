#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tiger as tiger_runner
from gflow_one_context_probe import (
    load_manifest,
    load_shard_for_example,
    move_to_device,
    resolve_project_path,
    target_in_top_from_payload,
    top_mass_from_payload,
    transform_cached_rewards,
    to_jsonable,
)
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train TIGER with TB-only on one user's prefixes and plot distribution TV."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--reward-cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--min-prefixes", type=int, default=6)
    parser.add_argument("--max-prefixes", type=int, default=8)
    parser.add_argument("--top-m", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--tb-loss-weighting", choices=["uniform", "reward"], default="uniform")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--reward-transform", choices=["power", "exp", "softplus", "rank"], default="power")
    parser.add_argument("--reward-power", type=float, default=1.0)
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--rank-alpha", type=float, default=1.0)
    parser.add_argument("--reward-score-source", choices=["inverse_softplus", "reward"], default="inverse_softplus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cache_rows(
    cache_path: Path,
    manifest: Mapping[str, Any],
    example_indices: list[int],
    top_m: int,
) -> list[dict[str, Any]]:
    rows = []
    for example_index in example_indices:
        payload, offset = load_shard_for_example(cache_path, manifest, int(example_index))
        stored_m = int(payload["top_item_row_ids"].size(1))
        if top_m > stored_m:
            raise ValueError(f"top_m={top_m} exceeds cache stored_top_m={stored_m}")
        item_ids = payload.get("top_item_ids")
        if torch.is_tensor(item_ids):
            item_ids = item_ids[offset, :top_m].long().contiguous()
        rows.append(
            {
                "example_index": int(example_index),
                "rows": payload["top_item_row_ids"][offset, :top_m].long().contiguous(),
                "item_ids": item_ids,
                "rewards": payload["top_rewards"][offset, :top_m].float().contiguous(),
                "reward_sum_full": payload["reward_sum"][offset].float().reshape(()).contiguous(),
                "top_mass": top_mass_from_payload(
                    payload,
                    top_m,
                    payload["top_rewards"].float(),
                    payload["reward_sum"].float(),
                )[offset].float(),
                "target_in_top": bool(target_in_top_from_payload(payload, top_m)[offset].item()),
                "target_item_id": int(payload["target_item_ids"][offset].item())
                if torch.is_tensor(payload.get("target_item_ids"))
                else None,
                "target_semantic_row_id": int(payload["target_semantic_row_ids"][offset].item())
                if torch.is_tensor(payload.get("target_semantic_row_ids"))
                else None,
                "target_reward_rank": int(payload["target_reward_ranks"][offset].item())
                if torch.is_tensor(payload.get("target_reward_ranks"))
                else None,
            }
        )
    return rows


def cache_quality_arrays(
    cache_path: Path,
    manifest: Mapping[str, Any],
    top_m: int,
    n_examples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(n_examples, dtype=torch.float32)
    mass = torch.zeros(n_examples, dtype=torch.float32)
    for shard in manifest.get("shards", []):
        start = int(shard["example_start"])
        n = int(shard["num_examples"])
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = cache_path / shard_path
        payload = torch.load(shard_path, map_location="cpu")
        target[start : start + n] = target_in_top_from_payload(payload, top_m)[:n].float()
        mass[start : start + n] = top_mass_from_payload(
            payload,
            top_m,
            payload["top_rewards"].float(),
            payload["reward_sum"].float(),
        )[:n].float()
    return target, mass


def choose_user_examples(
    dataset,
    cache_path: Path,
    manifest: Mapping[str, Any],
    top_m: int,
    user_id: int | None,
    min_prefixes: int,
    max_prefixes: int,
) -> tuple[int, list[int], dict[str, Any]]:
    by_user: dict[int, list[int]] = defaultdict(list)
    raw_user_by_id: dict[int, str] = {}
    for idx in range(len(dataset)):
        ex = dataset[idx]
        uid = int(ex["user_id"])
        by_user[uid].append(idx)
        raw_user_by_id.setdefault(uid, str(ex.get("raw_user_id", uid)))

    if user_id is not None:
        if int(user_id) not in by_user:
            raise ValueError(f"user_id={user_id} is not present in train dataset.")
        indices = sorted(by_user[int(user_id)])
        if max_prefixes > 0:
            indices = indices[:max_prefixes]
        return int(user_id), indices, {
            "raw_user_id": raw_user_by_id[int(user_id)],
            "selection": "requested",
            "num_available_prefixes": len(by_user[int(user_id)]),
        }

    target, mass = cache_quality_arrays(cache_path, manifest, top_m, len(dataset))
    candidates = []
    for uid, indices in by_user.items():
        n = len(indices)
        if n < min_prefixes:
            continue
        if max_prefixes > 0 and n > max_prefixes:
            continue
        idx = torch.tensor(indices, dtype=torch.long)
        candidates.append(
            (
                float(target[idx].mean().item()),
                float(mass[idx].mean().item()),
                n,
                uid,
            )
        )
    if not candidates:
        for uid, indices in by_user.items():
            n = len(indices)
            if n < min_prefixes:
                continue
            idx = torch.tensor(indices[: max(max_prefixes, 1)], dtype=torch.long)
            candidates.append(
                (
                    float(target[idx].mean().item()),
                    float(mass[idx].mean().item()),
                    min(n, max(max_prefixes, 1)),
                    uid,
                )
            )
    if not candidates:
        raise RuntimeError("Could not find a user with enough prefixes.")
    candidates.sort(reverse=True)
    _, _, _, uid = candidates[0]
    indices = sorted(by_user[int(uid)])
    if max_prefixes > 0:
        indices = indices[:max_prefixes]
    return int(uid), indices, {
        "raw_user_id": raw_user_by_id[int(uid)],
        "selection": "auto_best_target_in_top_then_top_mass",
        "num_available_prefixes": len(by_user[int(uid)]),
        "target_in_top_mean_all_selected": float(target[torch.tensor(indices)].mean().item()),
        "top_mass_mean_all_selected": float(mass[torch.tensor(indices)].mean().item()),
    }


def sample_tv(
    selected_probs: torch.Tensor,
    target_probs: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    *,
    target_outside_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, n = selected_probs.shape
    model_outside = (1.0 - selected_probs.sum(dim=1, keepdim=True)).clamp_min(0.0)
    probs = torch.cat([selected_probs.clamp_min(0.0), model_outside], dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    counts = []
    for row in probs:
        draw = torch.multinomial(row.cpu(), int(num_samples), replacement=True, generator=generator)
        counts.append(torch.bincount(draw, minlength=n + 1).float())
    freq = torch.stack(counts, dim=0).to(selected_probs.device) / max(int(num_samples), 1)
    if target_outside_mass is None:
        target_outside_mass = torch.zeros((bsz, 1), device=target_probs.device, dtype=target_probs.dtype)
    else:
        target_outside_mass = target_outside_mass.to(target_probs.device).to(target_probs.dtype).reshape(bsz, 1)
    target_ext = torch.cat([target_probs, target_outside_mass], dim=1)
    return 0.5 * (freq - target_ext).abs().sum(dim=1)


def evaluate(
    model,
    batch: Mapping[str, Any],
    candidate_tokens: torch.Tensor,
    candidate_mask: torch.Tensor,
    rewards: torch.Tensor,
    reward_sum: torch.Tensor,
    *,
    step: int,
    elapsed_sec: float,
    num_samples: int,
    seed: int,
    tb_loss_weighting: str = "uniform",
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        seq_lp = model.score_items(
            batch,
            candidate_token_ids=candidate_tokens,
            candidate_token_mask=candidate_mask,
        ).float()
        catalog_target = rewards / reward_sum.unsqueeze(1).clamp_min(1e-12)
        top_target = rewards / rewards.sum(dim=1, keepdim=True).clamp_min(1e-12)
        model_probs = seq_lp.exp()
        model_mass = model_probs.sum(dim=1)
        model_outside = (1.0 - model_mass).clamp_min(0.0)
        target_outside = (1.0 - catalog_target.sum(dim=1, keepdim=True)).clamp_min(0.0)
        outside_tv = (model_outside - target_outside).abs().squeeze(1)
        tv_catalog = 0.5 * ((model_probs - catalog_target).abs().sum(dim=1) + outside_tv)
        conditional = torch.softmax(seq_lp, dim=1)
        tv_top_conditional = 0.5 * (conditional - top_target).abs().sum(dim=1)
        kl_catalog_target_to_model = (catalog_target * (catalog_target.clamp_min(1e-12).log() - seq_lp)).sum(dim=1)
        gen = torch.Generator().manual_seed(int(seed) + int(step))
        tv_catalog_empirical = sample_tv(
            model_probs,
            catalog_target,
            num_samples,
            gen,
            target_outside_mass=target_outside,
        )
        perfect_tv = sample_tv(
            catalog_target,
            catalog_target,
            num_samples,
            gen,
            target_outside_mass=target_outside,
        )
        tb_weights = top_target if tb_loss_weighting == "reward" else None
        tb_loss, tb_metrics = trajectory_balance_loss(
            sequence_logprobs=seq_lp,
            rewards=rewards,
            log_z=reward_sum.log(),
            reward_epsilon=1e-6,
            candidate_weights=tb_weights,
        )
        return {
            "step": int(step),
            "elapsed_sec": float(elapsed_sec),
            "tb_loss": float(tb_loss.item()),
            "tb_residual_mean": float(tb_metrics["tb_residual_mean"].item()),
            "tb_residual_std": float(tb_metrics["tb_residual_std"].item()),
            "tv_catalog_mean": float(tv_catalog.mean().item()),
            "tv_catalog_std": float(tv_catalog.std(unbiased=False).item()),
            "tv_catalog_empirical_mean": float(tv_catalog_empirical.mean().item()),
            "tv_top_conditional_mean": float(tv_top_conditional.mean().item()),
            "tv_exact_mean": float(tv_catalog.mean().item()),
            "tv_exact_std": float(tv_catalog.std(unbiased=False).item()),
            "tv_conditional_mean": float(tv_top_conditional.mean().item()),
            "tv_empirical_mean": float(tv_catalog_empirical.mean().item()),
            "tv_perfect_sampler_mean": float(perfect_tv.mean().item()),
            "kl_catalog_target_to_model_mean": float(kl_catalog_target_to_model.mean().item()),
            "kl_target_to_model_mean": float(kl_catalog_target_to_model.mean().item()),
            "model_selected_mass_mean": float(model_mass.mean().item()),
            "model_selected_mass_min": float(model_mass.min().item()),
            "model_selected_mass_max": float(model_mass.max().item()),
            "catalog_target_outside_mass_mean": float(target_outside.mean().item()),
            "target_entropy_mean": float((-(top_target * top_target.clamp_min(1e-12).log()).sum(dim=1)).mean().item()),
        }


def plot_outputs(
    output_dir: Path,
    trace: list[Mapping[str, Any]],
    rewards: torch.Tensor,
    seq_lp: torch.Tensor,
    *,
    prefix_idx: int = 0,
    top_k: int = 30,
) -> None:
    xs = [max(float(row["elapsed_sec"]), 1e-3) for row in trace]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xs, [row["tv_catalog_mean"] for row in trace], label="catalog + outside bucket exact", linewidth=2)
    ax.plot(xs, [row["tv_catalog_empirical_mean"] for row in trace], label="catalog + outside bucket samples", linewidth=2)
    ax.plot(xs, [row["tv_top_conditional_mean"] for row in trace], label="top-M conditional", linewidth=1.5, alpha=0.75)
    ax.plot(xs, [row["tv_perfect_sampler_mean"] for row in trace], "--", label="perfect sampler empirical floor", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Total variation")
    ax.set_title("TB-only GFlow distribution fitting on one user's prefixes")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "tv_vs_time.png", dpi=180)
    plt.close(fig)

    target = rewards[prefix_idx] / rewards[prefix_idx].sum().clamp_min(1e-12)
    model_probs = seq_lp[prefix_idx].exp()
    empirical_cond = torch.softmax(seq_lp[prefix_idx], dim=0)
    order = torch.argsort(target, descending=True)[:top_k]
    x = torch.arange(order.numel()).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    width = 0.28
    ax.bar(x - width, target[order].detach().cpu().numpy(), width=width, label="reward target")
    ax.bar(x, model_probs[order].detach().cpu().numpy(), width=width, label="model exact prob")
    ax.bar(x + width, empirical_cond[order].detach().cpu().numpy(), width=width, label="model conditional prob")
    ax.set_xlabel("Top items by reward rank")
    ax.set_ylabel("Probability")
    ax.set_title("Final distribution for one prefix")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "final_prefix_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tiger_runner.seed_everything(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)
    cache_cfg = args.reward_cache_path or config.get("method", {}).get("reward_cache_path")
    if cache_cfg is None:
        raise ValueError("Reward cache path is required.")
    cache_path = resolve_project_path(cache_cfg, config_path=config_path)
    manifest = load_manifest(cache_path)

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.train_dataset is None:
        raise RuntimeError("Training dataset is not initialized.")
    if int(manifest.get("num_examples", -1)) != len(data.train_dataset):
        raise ValueError("Reward cache does not match train dataset length.")

    user_id, example_indices, selection_info = choose_user_examples(
        data.train_dataset,
        cache_path,
        manifest,
        int(args.top_m),
        args.user_id,
        int(args.min_prefixes),
        int(args.max_prefixes),
    )
    cache_rows = load_cache_rows(cache_path, manifest, example_indices, int(args.top_m))
    examples = [data.train_dataset[i] for i in example_indices]
    batch = move_to_device(data._make_collator()(examples), device)

    model = tiger_runner.build_model(config, data)
    checkpoint_cfg = args.checkpoint_path or config.get("training", {}).get("init_checkpoint_path")
    if checkpoint_cfg is None:
        raise ValueError("Checkpoint path is required.")
    checkpoint_path = resolve_project_path(checkpoint_cfg, config_path=config_path)
    tiger_runner.load_model_state_dict(model, checkpoint_path, strict=True)
    model.to(device)

    item_tokens = model.item_semantic_ids.to(device).long()
    row_ids = torch.stack([row["rows"] for row in cache_rows], dim=0).to(device).long()
    safe_rows = row_ids.clamp_min(0).clamp_max(item_tokens.size(0) - 1)
    candidate_tokens = item_tokens[safe_rows].contiguous()
    candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
    raw_rewards = torch.stack([row["rewards"] for row in cache_rows], dim=0).to(device).float().clamp_min(1e-6)
    rewards = transform_cached_rewards(raw_rewards, args)
    reward_sum = rewards.sum(dim=1).clamp_min(1e-6)
    cached_full_reward_sum = (
        torch.stack([row["reward_sum_full"] for row in cache_rows], dim=0).to(device).float().reshape(-1).clamp_min(1e-6)
    )
    uses_cached_reward_scale = str(args.reward_transform) == "power" and abs(float(args.reward_power) - 1.0) < 1e-12
    catalog_reward_sum = cached_full_reward_sum if uses_cached_reward_scale else reward_sum
    catalog_eval_mode = "cached_full_catalog_sum" if uses_cached_reward_scale else "top_m_only_transform_unknown_outside"

    output_dir = resolve_project_path(args.output_dir, config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "reward_cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "user_id": int(user_id),
        "example_indices": example_indices,
        "num_prefixes": len(example_indices),
        "top_m": int(args.top_m),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "tb_loss_weighting": str(args.tb_loss_weighting),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "num_samples": int(args.num_samples),
        "reward_transform": str(args.reward_transform),
        "reward_power": float(args.reward_power),
        "reward_temperature": float(args.reward_temperature),
        "rank_alpha": float(args.rank_alpha),
        "catalog_eval_mode": catalog_eval_mode,
        "selection": selection_info,
        "prefixes": [
            {
                "example_index": int(idx),
                "history_len": len(examples[pos]["history_item_ids"]),
                "target_item_id": cache_rows[pos]["target_item_id"],
                "target_reward_rank": cache_rows[pos]["target_reward_rank"],
                "target_in_top": cache_rows[pos]["target_in_top"],
                "top_mass_full_catalog": float(cache_rows[pos]["top_mass"].item()),
            }
            for pos, idx in enumerate(example_indices)
        ],
        "objective_note": "TB-only diagnostic: no SFT loss, no early stopping, analytic logZ over selected top-M support.",
    }
    write_json(output_dir / "metadata.json", metadata)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    trace_path = output_dir / "trace.jsonl"
    trace: list[dict[str, Any]] = []
    started = time.time()

    def record(step: int, grad_norm: float | None = None) -> None:
        elapsed = time.time() - started
        row = evaluate(
            model,
            batch,
            candidate_tokens,
            candidate_mask,
            rewards,
            catalog_reward_sum,
            step=step,
            elapsed_sec=elapsed,
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            tb_loss_weighting=str(args.tb_loss_weighting),
        )
        if grad_norm is not None:
            row["grad_norm"] = float(grad_norm)
        trace.append(row)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        print(
            f"step={step} elapsed={elapsed:.1f}s tv={row['tv_exact_mean']:.4f} "
            f"emp_tv={row['tv_empirical_mean']:.4f} tb={row['tb_loss']:.4f} "
            f"mass={row['model_selected_mass_mean']:.4f}",
            flush=True,
        )

    trace_path.write_text("", encoding="utf-8")
    record(0)
    for step in range(1, int(args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seq_lp = model.score_items(
            batch,
            candidate_token_ids=candidate_tokens,
            candidate_token_mask=candidate_mask,
        )
        tb_weights = rewards / reward_sum.unsqueeze(1).clamp_min(1e-12) if args.tb_loss_weighting == "reward" else None
        loss, _ = trajectory_balance_loss(
            sequence_logprobs=seq_lp,
            rewards=rewards,
            log_z=reward_sum.log(),
            reward_epsilon=1e-6,
            candidate_weights=tb_weights,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            record(step, float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm))

    model.eval()
    with torch.no_grad():
        final_seq_lp = model.score_items(
            batch,
            candidate_token_ids=candidate_tokens,
            candidate_token_mask=candidate_mask,
        ).float().detach().cpu()
    rewards_cpu = rewards.detach().cpu()

    plot_outputs(output_dir, trace, rewards_cpu, final_seq_lp)

    final = trace[-1]
    summary = {**metadata, "initial": trace[0], "final": final, "trace_path": str(trace_path)}
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "final_prefix0_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["reward_rank", "reward_prob", "model_exact_prob", "model_conditional_prob"])
        writer.writeheader()
        target0 = rewards_cpu[0] / rewards_cpu[0].sum().clamp_min(1e-12)
        exact0 = final_seq_lp[0].exp()
        cond0 = torch.softmax(final_seq_lp[0], dim=0)
        order = torch.argsort(target0, descending=True)
        for rank, idx_t in enumerate(order.tolist(), start=1):
            writer.writerow(
                {
                    "reward_rank": rank,
                    "reward_prob": float(target0[idx_t].item()),
                    "model_exact_prob": float(exact0[idx_t].item()),
                    "model_conditional_prob": float(cond0[idx_t].item()),
                }
            )

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
