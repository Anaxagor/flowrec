#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tiger as tiger_runner
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "One-context GFlow/TB diagnostic: fit one fixed prefix against cached rewards."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--reward-cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-index", type=int, default=None)
    parser.add_argument("--example-search-limit", type=int, default=4096)
    parser.add_argument("--top-m", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--reward-transform", choices=["power", "exp", "softplus", "rank"], default="power")
    parser.add_argument("--reward-power", type=float, default=1.0)
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--rank-alpha", type=float, default=1.0)
    parser.add_argument("--reward-score-source", choices=["inverse_softplus", "reward"], default="inverse_softplus")
    parser.add_argument("--logz-scope", choices=["full", "top"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x).clamp_min(1e-12)))


def transform_cached_rewards(raw_rewards: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    transform = str(args.reward_transform)
    if transform == "power":
        reward_power = float(args.reward_power)
        if reward_power <= 0.0:
            raise ValueError("reward_power must be positive.")
        return raw_rewards.pow(reward_power).clamp_min(1e-6)
    if transform == "rank":
        rank_alpha = float(args.rank_alpha)
        if rank_alpha <= 0.0:
            raise ValueError("rank_alpha must be positive.")
        ranks = torch.arange(1, raw_rewards.numel() + 1, device=raw_rewards.device, dtype=raw_rewards.dtype)
        return ranks.pow(-rank_alpha).clamp_min(1e-6)

    temperature = float(args.reward_temperature)
    if temperature <= 0.0:
        raise ValueError("reward_temperature must be positive.")
    if str(args.reward_score_source) == "inverse_softplus":
        scores = inverse_softplus(raw_rewards)
    else:
        scores = raw_rewards
    if transform == "exp":
        scaled = (scores - scores.max()) / temperature
        return torch.exp(scaled).clamp_min(1e-6)
    if transform == "softplus":
        return F.softplus(scores / temperature).clamp_min(1e-6)
    raise ValueError(f"Unsupported reward_transform: {transform}")


def resolve_project_path(path: str | Path, *, config_path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return tiger_runner.resolve_experiment_path(path, config_path=config_path)


def load_manifest(cache_path: Path) -> dict[str, Any]:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reward cache manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_shard_for_example(cache_path: Path, manifest: Mapping[str, Any], example_index: int) -> tuple[dict[str, Any], int]:
    for shard in manifest.get("shards", []):
        start = int(shard["example_start"])
        n = int(shard["num_examples"])
        if start <= example_index < start + n:
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute():
                shard_path = cache_path / shard_path
            payload = torch.load(shard_path, map_location="cpu")
            return payload, example_index - start
    raise IndexError(f"example_index={example_index} is outside reward cache range")


def top_mass_from_payload(payload: Mapping[str, Any], top_m: int, rewards: torch.Tensor, reward_sum: torch.Tensor) -> torch.Tensor:
    stored_m = int(payload["top_item_row_ids"].size(1))
    if top_m == stored_m and "top_reward_mass" in payload:
        return payload["top_reward_mass"].reshape(-1)
    key = f"top_reward_mass_{top_m}"
    if key in payload:
        return payload[key].reshape(-1)
    return rewards[:, :top_m].sum(dim=1) / reward_sum.reshape(-1).clamp_min(1e-12)


def target_in_top_from_payload(payload: Mapping[str, Any], top_m: int) -> torch.Tensor:
    stored_m = int(payload["top_item_row_ids"].size(1))
    if top_m == stored_m and "target_in_top_m" in payload:
        return payload["target_in_top_m"].reshape(-1).bool()
    key = f"target_in_top_{top_m}"
    if key in payload:
        return payload[key].reshape(-1).bool()
    target_rows = payload.get("target_semantic_row_ids")
    if target_rows is None:
        return torch.zeros(payload["top_item_row_ids"].size(0), dtype=torch.bool)
    return payload["top_item_row_ids"][:, :top_m].eq(target_rows.reshape(-1, 1)).any(dim=1)


def choose_example(cache_path: Path, manifest: Mapping[str, Any], top_m: int, search_limit: int) -> int:
    best_index: int | None = None
    best_score = -float("inf")
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
        reward_sum = payload["reward_sum"][:n].float().reshape(-1)
        mass = top_mass_from_payload(payload, top_m, payload["top_rewards"].float(), payload["reward_sum"].float())[:n]
        target = target_in_top_from_payload(payload, top_m)[:n].float()
        # Prefer examples where the target is in top-M, then high top-M reward mass.
        score = target * 10.0 + mass
        local = int(torch.argmax(score).item())
        local_score = float(score[local].item())
        if local_score > best_score:
            best_score = local_score
            best_index = int(shard["example_start"]) + local
        remaining -= n
    if best_index is None:
        raise RuntimeError("Could not select an example from reward cache.")
    return best_index


def load_cache_row(cache_path: Path, manifest: Mapping[str, Any], example_index: int, top_m: int) -> dict[str, Any]:
    payload, offset = load_shard_for_example(cache_path, manifest, example_index)
    stored_m = int(payload["top_item_row_ids"].size(1))
    if top_m > stored_m:
        raise ValueError(f"top_m={top_m} exceeds cache stored_top_m={stored_m}")
    rows = payload["top_item_row_ids"][offset, :top_m].long().contiguous()
    item_ids = payload.get("top_item_ids")
    if torch.is_tensor(item_ids):
        item_ids = item_ids[offset, :top_m].long().contiguous()
    rewards = payload["top_rewards"][offset, :top_m].float().contiguous()
    reward_sum = payload["reward_sum"][offset].float().reshape(()).contiguous()
    top_mass = top_mass_from_payload(payload, top_m, payload["top_rewards"].float(), payload["reward_sum"].float())[offset].float()
    target_in_top = bool(target_in_top_from_payload(payload, top_m)[offset].item())
    return {
        "example_index": int(example_index),
        "rows": rows,
        "item_ids": item_ids,
        "rewards": rewards,
        "reward_sum": reward_sum,
        "top_mass": top_mass,
        "target_in_top": target_in_top,
        "target_item_id": int(payload["target_item_ids"][offset].item()) if torch.is_tensor(payload.get("target_item_ids")) else None,
        "target_semantic_row_id": int(payload["target_semantic_row_ids"][offset].item()) if torch.is_tensor(payload.get("target_semantic_row_ids")) else None,
        "target_reward_rank": int(payload["target_reward_ranks"][offset].item()) if torch.is_tensor(payload.get("target_reward_ranks")) else None,
    }


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float((a * b).sum().div(denom).item())


def ranks_desc(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device)
    return ranks.float()


def top_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, int(a.numel()), int(b.numel()))
    aa = set(torch.argsort(a, descending=True)[:k].detach().cpu().tolist())
    bb = set(torch.argsort(b, descending=True)[:k].detach().cpu().tolist())
    return len(aa & bb) / max(k, 1)


def rank_of(index: int, scores: torch.Tensor) -> int:
    order = torch.argsort(scores, descending=True)
    where = (order == int(index)).nonzero(as_tuple=False)
    if where.numel() == 0:
        return -1
    return int(where[0, 0].item()) + 1


def make_top_table(
    *,
    order: torch.Tensor,
    rows: torch.Tensor,
    item_ids: torch.Tensor | None,
    rewards: torch.Tensor,
    reward_dist: torch.Tensor,
    model_dist: torch.Tensor,
    sample_freq: torch.Tensor,
    limit: int = 20,
) -> list[dict[str, Any]]:
    out = []
    for pos, idx_t in enumerate(order[:limit], start=1):
        idx = int(idx_t.item())
        out.append(
            {
                "rank": pos,
                "candidate_index": idx,
                "semantic_row_id": int(rows[idx].item()),
                "item_id": int(item_ids[idx].item()) if torch.is_tensor(item_ids) else None,
                "reward": float(rewards[idx].item()),
                "reward_prob_top": float(reward_dist[idx].item()),
                "model_prob_top": float(model_dist[idx].item()),
                "sample_freq_top": float(sample_freq[idx].item()),
            }
        )
    return out


def evaluate_state(
    *,
    model,
    batch: Mapping[str, Any],
    candidate_tokens: torch.Tensor,
    candidate_mask: torch.Tensor,
    rewards: torch.Tensor,
    reward_sum: torch.Tensor,
    rows: torch.Tensor,
    item_ids: torch.Tensor | None,
    num_samples: int,
    step: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        seq_logp = model.score_items(batch, candidate_token_ids=candidate_tokens, candidate_token_mask=candidate_mask)[0].float()
        tb_loss, tb_metrics = trajectory_balance_loss(
            sequence_logprobs=seq_logp.unsqueeze(0),
            rewards=rewards.unsqueeze(0),
            log_z=reward_sum.log().reshape(1),
            reward_epsilon=1e-6,
        )
        log_model_top = torch.log_softmax(seq_logp, dim=0)
        model_dist = log_model_top.exp()
        reward_dist = rewards / rewards.sum().clamp_min(1e-12)
        sample_idx = torch.multinomial(model_dist, int(num_samples), replacement=True)
        sample_freq = torch.bincount(sample_idx, minlength=rewards.numel()).float() / max(int(num_samples), 1)
        reward_order = torch.argsort(rewards, descending=True)
        model_order = torch.argsort(model_dist, descending=True)
        top_reward_idx = int(reward_order[0].item())
        top_model_idx = int(model_order[0].item())
        kl_reward_model = (reward_dist * (reward_dist.clamp_min(1e-12).log() - log_model_top)).sum()
        kl_model_reward = (model_dist * (log_model_top - reward_dist.clamp_min(1e-12).log())).sum()
        ce_reward_model = -(reward_dist * log_model_top).sum()
        model_top_mass = seq_logp.exp().sum()
        reward_top_mass = rewards.sum() / reward_sum.clamp_min(1e-12)
        metrics = {
            "step": int(step),
            "tb_loss": float(tb_loss.item()),
            "tb_residual_mean": float(tb_metrics["tb_residual_mean"].item()),
            "tb_residual_std": float(tb_metrics["tb_residual_std"].item()),
            "kl_reward_to_model_top": float(kl_reward_model.item()),
            "kl_model_to_reward_top": float(kl_model_reward.item()),
            "ce_reward_to_model_top": float(ce_reward_model.item()),
            "pearson_reward_logp": pearson(rewards, seq_logp),
            "spearman_reward_logp": pearson(ranks_desc(rewards), ranks_desc(seq_logp)),
            "top_overlap_at_5": top_overlap(rewards, model_dist, 5),
            "top_overlap_at_10": top_overlap(rewards, model_dist, 10),
            "reward_top_mass": float(reward_top_mass.item()),
            "model_top_mass": float(model_top_mass.item()),
            "expected_reward_model_top": float((model_dist * rewards).sum().item()),
            "expected_reward_reward_top": float((reward_dist * rewards).sum().item()),
            "expected_reward_uniform_top": float(rewards.mean().item()),
            "sample_expected_reward_top": float((sample_freq * rewards).sum().item()),
            "reward_top1_model_rank": rank_of(top_reward_idx, model_dist),
            "model_top1_reward_rank": rank_of(top_model_idx, rewards),
            "reward_top1_sample_freq": float(sample_freq[top_reward_idx].item()),
            "model_top1_sample_freq": float(sample_freq[top_model_idx].item()),
            "reward_top10_sample_freq": float(sample_freq[reward_order[:10]].sum().item()),
            "model_top10_sample_freq": float(sample_freq[model_order[:10]].sum().item()),
            "top_by_reward": make_top_table(
                order=reward_order,
                rows=rows,
                item_ids=item_ids,
                rewards=rewards,
                reward_dist=reward_dist,
                model_dist=model_dist,
                sample_freq=sample_freq,
                limit=20,
            ),
            "top_by_model": make_top_table(
                order=model_order,
                rows=rows,
                item_ids=item_ids,
                rewards=rewards,
                reward_dist=reward_dist,
                model_dist=model_dist,
                sample_freq=sample_freq,
                limit=20,
            ),
        }
        return metrics


def render_markdown(summary: Mapping[str, Any]) -> str:
    initial = summary["initial"]
    final = summary["final"]
    lines = [
        "# One-Context GFlow Probe",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- Reward cache: `{summary['reward_cache_path']}`",
        f"- Example index: `{summary['example']['example_index']}`",
        f"- Top-M: `{summary['top_m']}`",
        f"- Steps: `{summary['steps']}`",
        f"- LR: `{summary['lr']}`",
        f"- Reward transform: `{summary.get('reward_transform', 'power')}`",
        f"- Reward power: `{summary.get('reward_power', 1.0)}`",
        f"- Reward temperature: `{summary.get('reward_temperature', 1.0)}`",
        f"- Rank alpha: `{summary.get('rank_alpha', 1.0)}`",
        f"- Reward score source: `{summary.get('reward_score_source', 'inverse_softplus')}`",
        f"- LogZ scope: `{summary.get('logz_scope', 'full')}`",
        "",
        "## Main Metrics",
        "",
        "| metric | initial | final |",
        "|---|---:|---:|",
    ]
    keys = [
        "tb_loss",
        "kl_reward_to_model_top",
        "ce_reward_to_model_top",
        "pearson_reward_logp",
        "spearman_reward_logp",
        "top_overlap_at_10",
        "reward_top1_model_rank",
        "reward_top10_sample_freq",
        "expected_reward_model_top",
        "expected_reward_reward_top",
        "reward_top_mass",
        "model_top_mass",
    ]
    for key in keys:
        lines.append(f"| `{key}` | {initial.get(key)} | {final.get(key)} |")
    lines.extend(["", "## Final Top Items By Reward", ""])
    lines.append("| reward rank | row | item | reward | reward p(top) | model p(top) | sample freq |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in final["top_by_reward"][:15]:
        lines.append(
            f"| {row['rank']} | {row['semantic_row_id']} | {row['item_id']} | "
            f"{row['reward']:.6g} | {row['reward_prob_top']:.6g} | "
            f"{row['model_prob_top']:.6g} | {row['sample_freq_top']:.6g} |"
        )
    lines.extend(["", "## Final Top Items By Model", ""])
    lines.append("| model rank | row | item | reward | reward p(top) | model p(top) | sample freq |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in final["top_by_model"][:15]:
        lines.append(
            f"| {row['rank']} | {row['semantic_row_id']} | {row['item_id']} | "
            f"{row['reward']:.6g} | {row['reward_prob_top']:.6g} | "
            f"{row['model_prob_top']:.6g} | {row['sample_freq_top']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)

    cache_cfg = args.reward_cache_path or config.get("method", {}).get("reward_cache_path")
    if cache_cfg is None:
        raise ValueError("Reward cache path must be provided either by --reward-cache-path or config.method.reward_cache_path.")
    cache_path = resolve_project_path(cache_cfg, config_path=config_path)
    manifest = load_manifest(cache_path)

    if args.example_index is None:
        example_index = choose_example(cache_path, manifest, int(args.top_m), int(args.example_search_limit))
    else:
        example_index = int(args.example_index)

    tiger_runner.seed_everything(int(args.seed))
    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.train_dataset is None:
        raise RuntimeError("Training dataset is not initialized.")
    cache_examples = int(manifest.get("num_examples", -1))
    if cache_examples != len(data.train_dataset):
        raise ValueError(
            f"Reward cache num_examples={cache_examples} does not match train dataset size={len(data.train_dataset)}."
        )

    row = load_cache_row(cache_path, manifest, example_index, int(args.top_m))
    example = data.train_dataset[example_index]
    batch = data._make_collator()([example])
    batch = move_to_device(batch, device)

    model = tiger_runner.build_model(config, data)
    checkpoint_cfg = args.checkpoint_path or config.get("training", {}).get("init_checkpoint_path")
    if checkpoint_cfg is None:
        raise ValueError("Checkpoint path must be provided either by --checkpoint-path or config.training.init_checkpoint_path.")
    checkpoint_path = resolve_project_path(checkpoint_cfg, config_path=config_path)
    tiger_runner.load_model_state_dict(model, checkpoint_path, strict=True)
    model.to(device)

    item_tokens = model.item_semantic_ids.to(device).long()
    rows = row["rows"].to(device).long()
    safe_rows = rows.clamp_min(0).clamp_max(item_tokens.size(0) - 1)
    candidate_tokens = item_tokens[safe_rows].unsqueeze(0).contiguous()
    candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
    raw_rewards = row["rewards"].to(device).float().clamp_min(1e-6)
    raw_reward_sum = row["reward_sum"].to(device).float().clamp_min(1e-6)
    rewards = transform_cached_rewards(raw_rewards, args)
    if args.logz_scope == "full":
        if str(args.reward_transform) != "power" or abs(float(args.reward_power) - 1.0) > 1e-12:
            raise ValueError("logz_scope=full is only valid for the unmodified cached reward. Use --logz-scope top for transformed cached rewards.")
        reward_sum = raw_reward_sum
    else:
        reward_sum = rewards.sum().clamp_min(1e-6)
    item_ids = row["item_ids"]
    if torch.is_tensor(item_ids):
        item_ids = item_ids.long()

    output_dir = resolve_project_path(args.output_dir, config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    summary_base = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "reward_cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "top_m": int(args.top_m),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "num_samples": int(args.num_samples),
        "reward_transform": str(args.reward_transform),
        "reward_power": float(args.reward_power),
        "reward_temperature": float(args.reward_temperature),
        "rank_alpha": float(args.rank_alpha),
        "reward_score_source": str(args.reward_score_source),
        "logz_scope": str(args.logz_scope),
        "raw_reward_sum": float(raw_reward_sum.detach().cpu().item()),
        "transformed_reward_sum": float(reward_sum.detach().cpu().item()),
        "example": {
            "example_index": int(example_index),
            "user_id": int(batch["user_ids"][0].detach().cpu().item()),
            "target_item_id": row["target_item_id"],
            "target_semantic_row_id": row["target_semantic_row_id"],
            "target_reward_rank": row["target_reward_rank"],
            "target_in_top_m": row["target_in_top"],
            "history_len": int(batch["history_mask"][0].detach().cpu().sum().item()),
        },
    }
    write_json(output_dir / "metadata.json", summary_base)

    initial = evaluate_state(
        model=model,
        batch=batch,
        candidate_tokens=candidate_tokens,
        candidate_mask=candidate_mask,
        rewards=rewards,
        reward_sum=reward_sum,
        rows=row["rows"],
        item_ids=item_ids,
        num_samples=int(args.num_samples),
        step=0,
    )
    with trace_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(initial), sort_keys=True) + "\n")

    print(
        "step=0 "
        f"tb={initial['tb_loss']:.6g} kl={initial['kl_reward_to_model_top']:.6g} "
        f"rho={initial['spearman_reward_logp']:.4f} top10_sample={initial['reward_top10_sample_freq']:.4f}",
        flush=True,
    )

    started = time.time()
    final = initial
    for step in range(1, int(args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seq_logp = model.score_items(batch, candidate_token_ids=candidate_tokens, candidate_token_mask=candidate_mask)
        loss, _ = trajectory_balance_loss(
            sequence_logprobs=seq_logp,
            rewards=rewards.unsqueeze(0),
            log_z=reward_sum.log().reshape(1),
            reward_epsilon=1e-6,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            final = evaluate_state(
                model=model,
                batch=batch,
                candidate_tokens=candidate_tokens,
                candidate_mask=candidate_mask,
                rewards=rewards,
                reward_sum=reward_sum,
                rows=row["rows"],
                item_ids=item_ids,
                num_samples=int(args.num_samples),
                step=step,
            )
            final["train_grad_norm"] = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            final["elapsed_sec"] = time.time() - started
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(to_jsonable(final), sort_keys=True) + "\n")
            print(
                f"step={step} tb={final['tb_loss']:.6g} kl={final['kl_reward_to_model_top']:.6g} "
                f"rho={final['spearman_reward_logp']:.4f} "
                f"top10_sample={final['reward_top10_sample_freq']:.4f} "
                f"top1_rank={final['reward_top1_model_rank']}",
                flush=True,
            )

    summary = {**summary_base, "initial": initial, "final": final}
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
