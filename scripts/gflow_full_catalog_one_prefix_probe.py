#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
    resolve_project_path,
    target_in_top_from_payload,
    to_jsonable,
    top_mass_from_payload,
)
from precompute_reward_cache import build_reward_components, make_mixed_rewards, move_to_device
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Full-catalog one-prefix GFlow/TB distribution diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--reward-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-index", type=int, default=None)
    parser.add_argument("--example-search-limit", type=int, default=8192)
    parser.add_argument("--min-history-len", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--score-chunk-size", type=int, default=1024)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--tb-loss-weighting", choices=["reward", "uniform"], default="reward")
    parser.add_argument("--score-reward-transform", choices=["cache", "config", "power", "exp", "softplus", "rank"], default="cache")
    parser.add_argument("--score-power", type=float, default=None)
    parser.add_argument("--score-temperature", type=float, default=None)
    parser.add_argument("--score-rank-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cache_reward_options(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[str, float, float, float]:
    transform = str(args.score_reward_transform)
    if transform == "cache":
        transform = str(manifest.get("score_reward_transform", "config"))
    power = float(args.score_power if args.score_power is not None else manifest.get("score_power", 1.0))
    temperature = float(
        args.score_temperature if args.score_temperature is not None else manifest.get("score_temperature", 1.0)
    )
    rank_alpha = float(args.score_rank_alpha if args.score_rank_alpha is not None else manifest.get("score_rank_alpha", 1.0))
    return transform, power, temperature, rank_alpha


def choose_interesting_example(
    dataset,
    cache_path: Path,
    manifest: Mapping[str, Any],
    *,
    top_m: int,
    search_limit: int,
    min_history_len: int,
) -> tuple[int, dict[str, Any]]:
    best: tuple[float, int, dict[str, Any]] | None = None
    remaining = max(1, int(search_limit))
    for shard in manifest.get("shards", []):
        if remaining <= 0:
            break
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = cache_path / shard_path
        payload = torch.load(shard_path, map_location="cpu")
        n = min(int(shard["num_examples"]), remaining)
        rewards = payload["top_rewards"][:n, :top_m].float()
        reward_sum = payload["reward_sum"][:n].float().reshape(-1).clamp_min(1e-12)
        top_sum = rewards.sum(dim=1).clamp_min(1e-12)
        top_mass = top_mass_from_payload(payload, top_m, payload["top_rewards"].float(), payload["reward_sum"].float())[:n]
        target_in_top = target_in_top_from_payload(payload, top_m)[:n].float()
        target_rank = payload.get("target_reward_ranks")
        if torch.is_tensor(target_rank):
            target_rank = target_rank[:n].float().clamp_min(1.0)
        else:
            target_rank = torch.full((n,), 9999.0)
        effective = payload.get("reward_effective_items")
        if torch.is_tensor(effective):
            effective = effective[:n].float().reshape(-1).clamp_min(1.0)
        else:
            effective = torch.full((n,), 20.0)
        top1_full = rewards[:, 0] / reward_sum
        top1_top = rewards[:, 0] / top_sum

        for local in range(n):
            example_index = int(shard["example_start"]) + local
            history_len = len(dataset[example_index]["history_item_ids"])
            if history_len < int(min_history_len):
                continue
            mass = float(top_mass[local].item())
            eff = float(effective[local].item())
            top1 = float(top1_full[local].item())
            top1_cond = float(top1_top[local].item())
            rank = float(target_rank[local].item())
            mass_score = 1.0 - min(abs(mass - 0.9) / 0.3, 1.0)
            eff_score = 1.0 - min(abs(math.log(eff) - math.log(20.0)) / math.log(20.0), 1.0)
            top1_score = 1.0 - min(abs(top1 - 0.15) / 0.25, 1.0)
            rank_score = 1.0 / math.log(rank + 2.0)
            history_score = min(history_len / 20.0, 1.0)
            target_score = 2.0 if bool(target_in_top[local].item()) else 0.0
            score = target_score + mass_score + eff_score + top1_score + rank_score + 0.5 * history_score
            info = {
                "selection_score": score,
                "history_len": history_len,
                "target_in_top": bool(target_in_top[local].item()),
                "target_reward_rank": int(rank),
                "top_m_reward_mass": mass,
                "reward_effective_items": eff,
                "top1_full_prob": top1,
                "top1_top_m_prob": top1_cond,
            }
            if best is None or score > best[0]:
                best = (score, example_index, info)
        remaining -= n
    if best is None:
        raise RuntimeError("Could not select an interesting example from the reward cache.")
    return best[1], best[2]


def score_items_in_chunks(
    model,
    batch: Mapping[str, Any],
    item_tokens: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    mask = torch.ones_like(item_tokens, dtype=torch.bool, device=item_tokens.device)
    parts = []
    for start in range(0, int(item_tokens.size(0)), int(chunk_size)):
        end = min(start + int(chunk_size), int(item_tokens.size(0)))
        parts.append(
            model.score_items(
                batch,
                candidate_token_ids=item_tokens[start:end].unsqueeze(0),
                candidate_token_mask=mask[start:end].unsqueeze(0),
            )[0].float()
        )
    return torch.cat(parts, dim=0)


def train_tb_step(
    model,
    batch: Mapping[str, Any],
    item_tokens: torch.Tensor,
    rewards: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    chunk_size: int,
    max_grad_norm: float,
    tb_loss_weighting: str,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_z = rewards.sum().clamp_min(1e-12).log()
    log_rewards = rewards.clamp_min(1e-12).log()
    weights = rewards / rewards.sum().clamp_min(1e-12) if tb_loss_weighting == "reward" else None
    denom = float(rewards.numel()) if weights is None else 1.0
    total = 0.0
    mask = torch.ones_like(item_tokens, dtype=torch.bool, device=item_tokens.device)
    for start in range(0, int(item_tokens.size(0)), int(chunk_size)):
        end = min(start + int(chunk_size), int(item_tokens.size(0)))
        seq_lp = model.score_items(
            batch,
            candidate_token_ids=item_tokens[start:end].unsqueeze(0),
            candidate_token_mask=mask[start:end].unsqueeze(0),
        )[0].float()
        residual = log_z + seq_lp - log_rewards[start:end]
        if weights is None:
            loss = residual.square().sum() / denom
        else:
            loss = (weights[start:end] * residual.square()).sum()
        loss.backward()
        total += float(loss.detach().cpu().item())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
    optimizer.step()
    return total, float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)


def evaluate_distribution(
    model,
    batch: Mapping[str, Any],
    item_tokens: torch.Tensor,
    rewards: torch.Tensor,
    *,
    top_m: int,
    score_chunk_size: int,
    step: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        seq_lp = score_items_in_chunks(model, batch, item_tokens, chunk_size=score_chunk_size)
        target = rewards / rewards.sum().clamp_min(1e-12)
        model_probs = seq_lp.exp()
        model_mass = model_probs.sum()
        outside_mass = (1.0 - model_mass).clamp_min(0.0)
        tv_catalog = 0.5 * ((model_probs - target).abs().sum() + outside_mass)
        order = torch.argsort(rewards, descending=True)
        top_idx = order[: min(int(top_m), int(order.numel()))]
        target_top = target[top_idx] / target[top_idx].sum().clamp_min(1e-12)
        model_top = model_probs[top_idx] / model_probs[top_idx].sum().clamp_min(1e-12)
        tv_top = 0.5 * (model_top - target_top).abs().sum()
        kl = (target * (target.clamp_min(1e-12).log() - seq_lp)).sum()
        top_mass_target = target[top_idx].sum()
        top_mass_model = model_probs[top_idx].sum()
        tb_loss, tb_metrics = trajectory_balance_loss(
            sequence_logprobs=seq_lp.unsqueeze(0),
            rewards=rewards.unsqueeze(0),
            log_z=rewards.sum().clamp_min(1e-12).log().reshape(1),
            reward_epsilon=1e-6,
            candidate_weights=target.unsqueeze(0),
        )
        return {
            "step": int(step),
            "elapsed_sec": float(elapsed_sec),
            "tv_catalog_exact": float(tv_catalog.item()),
            "tv_top_m_conditional": float(tv_top.item()),
            "tv_perfect_exact_floor": 0.0,
            "kl_target_to_model": float(kl.item()),
            "tb_loss_reward_weighted": float(tb_loss.item()),
            "tb_residual_mean": float(tb_metrics["tb_residual_mean"].item()),
            "tb_residual_std": float(tb_metrics["tb_residual_std"].item()),
            "model_catalog_mass": float(model_mass.item()),
            "model_outside_mass": float(outside_mass.item()),
            "target_top_m_mass": float(top_mass_target.item()),
            "model_top_m_mass": float(top_mass_model.item()),
            "target_entropy": float((-(target * target.clamp_min(1e-12).log()).sum()).item()),
            "target_effective_items": float(torch.exp(-(target * target.clamp_min(1e-12).log()).sum()).item()),
        }


def plot_trace(output_dir: Path, trace: list[Mapping[str, Any]]) -> None:
    xs = [max(float(row["elapsed_sec"]), 1e-3) for row in trace]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(xs, [row["tv_catalog_exact"] for row in trace], marker="o", label="full catalog exact", linewidth=2.2)
    ax.plot(xs, [row["tv_top_m_conditional"] for row in trace], marker="o", label="top-256 conditional", linewidth=1.8)
    ax.plot(xs, [0.0 for _ in trace], "--", label="perfect sampler exact floor", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Total variation")
    ax.set_title("TB-only full-catalog GFlow fitting on one fixed prefix")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "tv_full_catalog_vs_top256.png", dpi=180)
    plt.close(fig)


def plot_final_distribution(
    output_dir: Path,
    rewards: torch.Tensor,
    seq_lp: torch.Tensor,
    *,
    top_m: int,
    top_k: int = 40,
) -> None:
    target = rewards / rewards.sum().clamp_min(1e-12)
    model_probs = seq_lp.exp()
    order = torch.argsort(rewards, descending=True)
    head = order[: min(top_k, int(order.numel()))]
    x = torch.arange(head.numel()).cpu().numpy()
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    ax = axes[0]
    ax.bar(x - width / 2, target[head].detach().cpu().numpy(), width=width, label="reward prob, full catalog")
    ax.bar(x + width / 2, model_probs[head].detach().cpu().numpy(), width=width, label="model exact prob")
    ax.set_title("Final item distribution for one fixed prefix")
    ax.set_xlabel("Top items by reward rank")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1]
    k = min(300, int(order.numel()))
    xs = torch.arange(1, k + 1).cpu().numpy()
    target_cum = torch.cumsum(target[order[:k]], dim=0).detach().cpu().numpy()
    model_cum = torch.cumsum(model_probs[order[:k]], dim=0).detach().cpu().numpy()
    ax.plot(xs, target_cum, label="reward cumulative mass", linewidth=2.2)
    ax.plot(xs, model_cum, label="model cumulative mass", linewidth=2.2)
    ax.axvline(min(top_m, k), linestyle="--", color="gray", alpha=0.6, label=f"top-{top_m}")
    ax.set_title("Cumulative mass by reward rank")
    ax.set_xlabel("Top-k items by reward rank")
    ax.set_ylabel("Cumulative mass")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "final_full_catalog_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tiger_runner.seed_everything(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)
    cache_path = resolve_project_path(args.reward_cache_path, config_path=config_path)
    manifest = load_manifest(cache_path)
    reward_config_path = Path(str(manifest.get("config_path", config_path)))
    if not reward_config_path.is_absolute():
        reward_config_path = resolve_project_path(reward_config_path, config_path=config_path)
    reward_config = tiger_runner.load_json(reward_config_path)
    score_transform, score_power, score_temperature, score_rank_alpha = cache_reward_options(args, manifest)

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.train_dataset is None:
        raise RuntimeError("Training dataset is not initialized.")
    if int(manifest.get("num_examples", -1)) != len(data.train_dataset):
        raise ValueError("Reward cache does not match train dataset length.")

    if args.example_index is None:
        example_index, selection_info = choose_interesting_example(
            data.train_dataset,
            cache_path,
            manifest,
            top_m=int(args.top_m),
            search_limit=int(args.example_search_limit),
            min_history_len=int(args.min_history_len),
        )
    else:
        example_index = int(args.example_index)
        selection_info = {"selection": "requested"}

    example = data.train_dataset[example_index]
    batch = move_to_device(data._make_collator()([example]), device)
    train_loader_for_models = data.train_dataloader()
    specs, reward_epsilon = build_reward_components(reward_config, data, train_loader_for_models, reward_config_path, device)

    model = tiger_runner.build_model(config, data)
    checkpoint_path = resolve_project_path(args.checkpoint_path, config_path=config_path)
    tiger_runner.load_model_state_dict(model, checkpoint_path, strict=True)
    model.to(device)

    item_tokens = model.item_semantic_ids.to(device).long()
    num_items = int(item_tokens.size(0))
    semantic_row_to_item_id = tiger_runner.item_id_by_semantic_row(data)
    if semantic_row_to_item_id is None:
        semantic_row_to_item_id = torch.arange(num_items, dtype=torch.long)
    semantic_row_to_item_id = semantic_row_to_item_id.to(device).long()
    item_id_to_semantic_row = tiger_runner.item_id_to_semantic_row(data).to(device).long()

    with torch.no_grad():
        rewards, component_means, _ = make_mixed_rewards(
            specs,
            batch,
            item_tokens,
            semantic_row_to_item_id,
            item_id_to_semantic_row,
            reward_epsilon,
            str(manifest.get("normalization_scope", "row")),
            score_reward_transform=score_transform,
            score_power=score_power,
            score_temperature=score_temperature,
            score_rank_alpha=score_rank_alpha,
        )
    rewards = rewards[0].to(device).float().clamp_min(float(reward_epsilon))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    output_dir = resolve_project_path(args.output_dir, config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(config_path),
        "reward_config_path": str(reward_config_path),
        "checkpoint_path": str(checkpoint_path),
        "reward_cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "example_index": int(example_index),
        "user_id": int(batch["user_ids"][0].detach().cpu().item()),
        "history_len": int(batch["history_mask"][0].detach().cpu().sum().item()),
        "target_item_id": int(batch["target_item_ids"][0].detach().cpu().item()),
        "num_catalog_items": num_items,
        "top_m": int(args.top_m),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "lr": float(args.lr),
        "tb_loss_weighting": str(args.tb_loss_weighting),
        "score_transform": score_transform,
        "score_power": score_power,
        "score_temperature": score_temperature,
        "score_rank_alpha": score_rank_alpha,
        "selection": selection_info,
        "component_means": component_means,
    }
    write_json(output_dir / "metadata.json", metadata)

    trace: list[dict[str, Any]] = []
    trace_path = output_dir / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    started = time.time()

    def record(step: int, train_loss: float | None = None, grad_norm: float | None = None) -> None:
        row = evaluate_distribution(
            model,
            batch,
            item_tokens,
            rewards,
            top_m=int(args.top_m),
            score_chunk_size=int(args.score_chunk_size),
            step=step,
            elapsed_sec=time.time() - started,
        )
        if train_loss is not None:
            row["train_loss"] = float(train_loss)
        if grad_norm is not None:
            row["grad_norm"] = float(grad_norm)
        trace.append(row)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        print(
            f"step={step} tv_full={row['tv_catalog_exact']:.5f} "
            f"tv_top={row['tv_top_m_conditional']:.5f} kl={row['kl_target_to_model']:.5f} "
            f"model_mass={row['model_catalog_mass']:.4f}",
            flush=True,
        )

    record(0)
    for step in range(1, int(args.steps) + 1):
        train_loss, grad_norm = train_tb_step(
            model,
            batch,
            item_tokens,
            rewards,
            optimizer,
            chunk_size=int(args.train_chunk_size),
            max_grad_norm=float(args.max_grad_norm),
            tb_loss_weighting=str(args.tb_loss_weighting),
        )
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            record(step, train_loss=train_loss, grad_norm=grad_norm)

    model.eval()
    with torch.no_grad():
        final_seq_lp = score_items_in_chunks(model, batch, item_tokens, chunk_size=int(args.score_chunk_size)).detach().cpu()
    rewards_cpu = rewards.detach().cpu()
    plot_trace(output_dir, trace)
    plot_final_distribution(output_dir, rewards_cpu, final_seq_lp, top_m=int(args.top_m))

    target = rewards_cpu / rewards_cpu.sum().clamp_min(1e-12)
    model_probs = final_seq_lp.exp()
    order = torch.argsort(rewards_cpu, descending=True)
    with (output_dir / "final_top_items.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["reward_rank", "reward", "reward_prob", "model_prob"])
        writer.writeheader()
        for rank, idx in enumerate(order[: min(500, int(order.numel()))].tolist(), start=1):
            writer.writerow(
                {
                    "reward_rank": rank,
                    "reward": float(rewards_cpu[idx].item()),
                    "reward_prob": float(target[idx].item()),
                    "model_prob": float(model_probs[idx].item()),
                }
            )

    summary = {**metadata, "initial": trace[0], "final": trace[-1], "trace_path": str(trace_path)}
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
