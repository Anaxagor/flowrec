#!/usr/bin/env python
"""Precompute FACTORIZED full-catalog rewards for TIGER/GPT-2 GFlow fine-tuning.

Counterpart of scripts/precompute_reward_cache.py that stores, instead of one
dense row per example over the catalog, the small factors from which
flowrec.ft_methods.gflow.factorized_cache.FactorizedRewardCache rebuilds the
same rows exactly on the training device:

  sasrec        per-example SASRec final hidden state (float16 by default) +
                the teacher item table gathered into catalog-row order
  similarity    hierarchical: target SID tokens + catalog SIDs + codebook
                distances; embedding_cosine: target item id + the normalized
                content-embedding table; exact: target SID tokens + catalog SIDs
  interaction   sparse CSR of the rows' non-fill entries (the user's logged
                before-t evidence), scored by the reward model on the user's
                evidence items only
  positive      the two constants

Same CLI as the dense script (--config with a SINGLE-component reward, the
same --split / --batch-size / --per-user-last-n / --max-examples), same
example order, so a dense and a factorized cache built from one config are
row-aligned. --verify-batches K recomputes the dense rows of the first K
batches through the dense script's own code path and records the max
abs / rel difference against the reconstruction in manifest.json.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.dataset  # noqa: F401  (must load before torch on Windows)
import torch

faulthandler.enable()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
from precompute_reward_cache import (  # noqa: E402
    build_cache_loader,
    build_reward_components,
    compute_component_rewards,
    jsonable,
    move_to_device,
    slice_batch,
)
from flowrec.data.interaction_store import POSITIVE_EVENT_TYPE  # noqa: E402
from flowrec.ft_methods.gflow.factorized_cache import (  # noqa: E402
    CACHE_KIND,
    FACTORS_FILENAME,
    FactorizedRewardCache,
)
from flowrec.models.sasrec import SASRec  # noqa: E402
from flowrec.rewards.interaction import InteractionRewardModel, PositiveRewardModel  # noqa: E402
from flowrec.rewards.sasrec import SASRecRewardModel  # noqa: E402
from flowrec.rewards.similarity import SimilarityRewardModel  # noqa: E402
from flowrec.trajectories.semantic import SemanticTrajectoryBatch  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Precompute factorized full-catalogue rewards for GFlow training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--per-user-last-n", type=int, default=None,
                        help="Cache only the last N train examples of every user "
                             "(same subset as the dense script).")
    parser.add_argument("--per-user-random-n", type=int, default=None,
                        help="Cache only N examples of every user drawn uniformly at random from all "
                             "of the user's eligible examples (optionally limited by --per-user-pool-last-n); "
                             "exclusive with --per-user-last-n.")
    parser.add_argument("--per-user-random-seed", type=int, default=0)
    parser.add_argument("--per-user-pool-last-n", type=int, default=None,
                        help="Restrict eligible examples to each user's last N before sampling.")
    parser.add_argument("--hidden-dtype", choices=["float16", "float32"], default="float16",
                        help="Storage dtype of the per-example SASRec hidden state "
                             "(the only per-example float factor).")
    parser.add_argument("--verify-batches", type=int, default=1,
                        help="Recompute the dense rows of the first K batches through "
                             "precompute_reward_cache's code path and compare (0 = skip).")
    parser.add_argument("--chunk-rows", type=int, default=64)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# SASRec factors
# --------------------------------------------------------------------------- #
def _unwrap_sasrec(scorer: torch.nn.Module) -> tuple[SASRec, torch.nn.Module | None]:
    """(raw SASRec, IdTranslatingItemScorer wrapper or None)."""
    if isinstance(scorer, tiger_runner.IdTranslatingItemScorer):
        raw = scorer.scorer
        if not isinstance(raw, SASRec):
            raise TypeError(f"IdTranslatingItemScorer wraps {type(raw).__name__}, expected SASRec.")
        return raw, scorer
    if isinstance(scorer, SASRec):
        return scorer, None
    raise TypeError(
        f"Factorized SASRec rewards need a SASRec teacher (got {type(scorer).__name__}); "
        "its scores must be final_hidden . item_embedding."
    )


@torch.no_grad()
def sasrec_final_hidden(raw: SASRec, history_item_ids: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    """The [B, d_model] state SASRec.forward multiplies with the item table."""
    history_mask = history_mask.to(torch.bool)
    seq_out = raw._encode_history(history_item_ids.long(), history_mask)
    seq_len = history_item_ids.size(1)
    positions = torch.arange(seq_len, device=history_item_ids.device).unsqueeze(0)
    last_idx = positions.masked_fill(~history_mask, -1).max(dim=1).values
    if (last_idx < 0).any():
        raise ValueError("Found sequence with no valid history items.")
    batch_idx = torch.arange(history_item_ids.size(0), device=history_item_ids.device)
    return seq_out[batch_idx, last_idx]


def sasrec_item_table(
    raw: SASRec, wrapper, semantic_row_to_item_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher item embeddings in catalog-row order + the row validity mask,
    mirroring score_items(batch) -> gather_semantic_scores in the dense path."""
    table = raw.item_embedding.weight.detach().float().cpu()
    teacher_vocab = int(table.size(0))
    row_ids = semantic_row_to_item_id.detach().cpu().long()
    if wrapper is not None:
        translation = wrapper.id_translation.detach().cpu().long().clamp_min(0)
        score_width = int(translation.numel())
        valid = row_ids.ge(0) & row_ids.lt(score_width)
        teacher_ids = translation[row_ids.clamp(0, score_width - 1)]
    else:
        valid = row_ids.ge(0) & row_ids.lt(teacher_vocab)
        teacher_ids = row_ids.clamp(0, teacher_vocab - 1)
    return table[teacher_ids.clamp(0, teacher_vocab - 1)].contiguous(), valid


# --------------------------------------------------------------------------- #
# Interaction (sparse evidence) factors
# --------------------------------------------------------------------------- #
class EvidenceIndex:
    """Distinct items with ANY logged event (auxiliary + positive stream) per
    user; the reward model applies the strictly-before-t rule itself."""

    def __init__(self, model: InteractionRewardModel) -> None:
        self.store = model.store
        self.positive_store = model.positive_store
        self.aux_types = list(self.store.event_types()) if len(self.store) else []
        self.has_positive = self.positive_store is not None and len(self.positive_store) > 0
        self._cache: dict[int, np.ndarray] = {}

    def items(self, user_id: int) -> np.ndarray:
        cached = self._cache.get(user_id)
        if cached is not None:
            return cached
        parts = [self.store.items_for_user(t, user_id) for t in self.aux_types]
        if self.has_positive:
            parts.append(self.positive_store.items_for_user(POSITIVE_EVENT_TYPE, user_id))
        parts = [p for p in parts if p.size]
        out = np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.empty(0, dtype=np.int64)
        self._cache[user_id] = out
        return out


@torch.no_grad()
def interaction_fill_value(model: InteractionRewardModel, batch: Mapping[str, Any], sid_len: int, device) -> float:
    """Reward of an unresolvable candidate = the constant every non-evidence
    catalog row takes in the dense path."""
    bsz = int(batch["target_item_ids"].size(0))
    traj = SemanticTrajectoryBatch(
        candidate_token_ids=torch.zeros((bsz, 1, sid_len), dtype=torch.long, device=device),
        candidate_token_mask=torch.ones((bsz, 1, sid_len), dtype=torch.bool, device=device),
        is_positive=torch.zeros((bsz, 1), dtype=torch.bool, device=device),
        candidate_item_row_ids=None,
        extra={"candidate_item_ids": torch.full((bsz, 1), -1, dtype=torch.long, device=device)},
    )
    values = model(batch, traj).float().reshape(-1)
    if not bool(torch.all(values == values[0])):
        raise RuntimeError("Interaction reward is not constant for unresolvable candidates.")
    return float(values[0])


@torch.no_grad()
def interaction_sparse_batch(
    model: InteractionRewardModel,
    evidence: EvidenceIndex,
    batch: Mapping[str, Any],
    item_id_to_semantic_row: torch.Tensor,
    *,
    sid_len: int,
    fill_value: float,
    reward_epsilon: float,
    device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per example: (sorted catalog rows, values) of the entries != fill."""
    users = batch["user_ids"].detach().cpu().numpy().astype(np.int64)
    targets = batch["target_item_ids"].detach().cpu().numpy().astype(np.int64)
    bsz = int(users.shape[0])
    cand_lists = []
    for u, t in zip(users.tolist(), targets.tolist()):
        items = evidence.items(int(u))
        if t >= 0:
            items = np.union1d(items, np.asarray([t], dtype=np.int64))
        cand_lists.append(items)
    width = max((int(c.size) for c in cand_lists), default=0)
    if width == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(bsz)], [np.empty(0, dtype=np.float32) for _ in range(bsz)]
    cand = np.full((bsz, width), -1, dtype=np.int64)
    for b, items in enumerate(cand_lists):
        cand[b, : items.size] = items
    cand_t = torch.as_tensor(cand, device=device)
    is_positive = cand_t.eq(torch.as_tensor(targets, device=device).unsqueeze(1)) & cand_t.ge(0)
    traj = SemanticTrajectoryBatch(
        candidate_token_ids=torch.zeros((bsz, width, sid_len), dtype=torch.long, device=device),
        candidate_token_mask=torch.ones((bsz, width, sid_len), dtype=torch.bool, device=device),
        is_positive=is_positive,
        candidate_item_row_ids=None,
        extra={"candidate_item_ids": cand_t},
    )
    values = model(batch, traj).float().clamp_min(reward_epsilon).cpu().numpy()
    mapping = item_id_to_semantic_row.detach().cpu().numpy().astype(np.int64)
    rows_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    for b in range(bsz):
        ids = cand[b]
        valid = ids >= 0
        rows = np.full(ids.shape, -1, dtype=np.int64)
        in_range = valid & (ids < mapping.shape[0])
        rows[in_range] = mapping[ids[in_range]]
        keep = (rows >= 0) & (values[b] != np.float32(fill_value))
        order = np.argsort(rows[keep], kind="stable")
        rows_out.append(rows[keep][order])
        vals_out.append(values[b][keep][order].astype(np.float32))
    return rows_out, vals_out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    args = parse_args(argv)
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
    hidden_dtype = torch.float16 if args.hidden_dtype == "float16" else torch.float32

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("Reward factor precompute requires data.semantic_id_path.")

    train_loader_for_models = data.train_dataloader()
    specs, reward_epsilon = build_reward_components(config, data, train_loader_for_models, config_path, device)
    if len(specs) != 1:
        raise ValueError(
            "precompute_reward_factors stores ONE raw component per cache; "
            f"the config builds {len(specs)} components. Use one single-component "
            "config per cache (see gflow_finetune_gpt2_amazon.cache_config)."
        )
    spec = specs[0]
    if float(spec["weight"]) != 1.0 or str(spec["normalizer"]).lower() not in {"none", "identity"}:
        raise ValueError("Single-component factor caches store the raw component (weight 1, no normalizer).")
    rmodel = spec["model"]
    cache_loader = build_cache_loader(
        data, args.split, int(args.batch_size), int(args.num_workers),
        per_user_last_n=args.per_user_last_n,
        per_user_random_n=args.per_user_random_n,
        per_user_random_seed=args.per_user_random_seed,
        per_user_pool_last_n=args.per_user_pool_last_n,
    )

    # Catalog rows depend only on the datamodule, not on a TIGER/GPT-2 model.
    item_tokens = torch.tensor([data.semantic_ids[key] for key in sorted(data.semantic_ids)],
                               dtype=torch.long, device=device)
    num_items, sid_len = int(item_tokens.size(0)), int(item_tokens.size(1))
    semantic_row_to_item_id = tiger_runner.item_id_by_semantic_row(data)
    if semantic_row_to_item_id is None:
        semantic_row_to_item_id = torch.arange(num_items, dtype=torch.long)
    semantic_row_to_item_id = semantic_row_to_item_id.to(device).long()
    item_id_to_semantic_row = tiger_runner.item_id_to_semantic_row(data).to(device).long()

    output_dir = tiger_runner.resolve_path(args.output_dir, base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- component-specific shared tables --------------------------------------
    factors: dict[str, torch.Tensor] = {}
    component: dict[str, Any] = {"name": str(spec["name"]), "mode": None, "config": {}}
    extra_manifest: dict[str, Any] = {}
    evidence: EvidenceIndex | None = None
    fill_value: float | None = None
    sasrec_raw = None
    sasrec_wrapper = None
    if isinstance(rmodel, SASRecRewardModel):
        component["name"] = "sasrec"
        component["config"] = asdict(rmodel.cfg)
        sasrec_raw, sasrec_wrapper = _unwrap_sasrec(rmodel.sasrec_model)
        sasrec_raw.eval()
        table, row_valid = sasrec_item_table(sasrec_raw, sasrec_wrapper, semantic_row_to_item_id)
        factors["item_table"] = table
        factors["row_valid"] = row_valid
        extra_manifest["teacher_id_translation"] = sasrec_wrapper is not None
        extra_manifest["hidden_dim"] = int(table.size(1))
    elif isinstance(rmodel, SimilarityRewardModel):
        component["name"] = "similarity"
        component["mode"] = str(rmodel.cfg.mode)
        cfg = asdict(rmodel.cfg)
        if cfg.get("level_weights") is not None:
            cfg["level_weights"] = [float(w) for w in cfg["level_weights"]]
        component["config"] = cfg
        if rmodel.cfg.mode in {"hierarchical", "exact"}:
            factors["item_tokens"] = item_tokens.detach().cpu().long()
        if rmodel.cfg.mode == "hierarchical":
            factors["code_dist"] = rmodel.code_dist.detach().float().cpu()
        elif rmodel.cfg.mode == "embedding_cosine":
            factors["item_embeddings"] = rmodel.item_embeddings.detach().float().cpu()
            factors["row_item_ids"] = semantic_row_to_item_id.detach().cpu().long()
        elif rmodel.cfg.mode != "exact":
            raise NotImplementedError(f"similarity mode {rmodel.cfg.mode!r} is not factorizable here.")
    elif isinstance(rmodel, InteractionRewardModel):
        component["name"] = "interaction"
        component["config"] = jsonable(asdict(rmodel.cfg))
        evidence = EvidenceIndex(rmodel)
    elif isinstance(rmodel, PositiveRewardModel):
        component["name"] = "positive"
        component["config"] = asdict(rmodel.cfg)
    else:
        raise NotImplementedError(
            f"Reward model {type(rmodel).__name__} has no factorized form; "
            "use scripts/precompute_reward_cache.py for it."
        )

    manifest: dict[str, Any] = {
        "kind": CACHE_KIND,
        "factors_path": FACTORS_FILENAME,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "hidden_dtype": args.hidden_dtype,
        "per_user_last_n": args.per_user_last_n,
        "per_user_random_n": args.per_user_random_n,
        "per_user_random_seed": args.per_user_random_seed if args.per_user_random_n is not None else None,
        "per_user_pool_last_n": args.per_user_pool_last_n,
        "reward_epsilon": reward_epsilon,
        "data": data.summary(),
        "data_config": jsonable(asdict(data_cfg)),
        "num_catalog_items": num_items,
        "sid_len": sid_len,
        "component": component,
        "components": [{"name": spec["name"], "weight": spec["weight"], "normalizer": spec["normalizer"]}],
        **extra_manifest,
    }
    from flowrec.data.dense_prefix import dense_prefix_signature

    manifest["training_context"] = dense_prefix_signature(config.get("dataset", {}))

    # -- per-example factors -----------------------------------------------------
    user_ids: list[torch.Tensor] = []
    target_item_ids: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    target_tokens: list[torch.Tensor] = []
    target_timestamps: list[torch.Tensor] = []
    hidden: list[torch.Tensor] = []
    csr_rows: list[np.ndarray] = []
    csr_vals: list[np.ndarray] = []
    dense_reference: list[tuple[int, torch.Tensor]] = []  # (example_start, dense rows) for verification

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
            bsz = int(batch["target_item_ids"].size(0))

            tgt = batch["target_item_ids"].long()
            safe = tgt.clamp_min(0).clamp_max(max(item_id_to_semantic_row.numel() - 1, 0))
            rows = item_id_to_semantic_row[safe]
            valid = (tgt >= 0) & (tgt < item_id_to_semantic_row.numel())
            rows = torch.where(valid, rows, torch.full_like(rows, -1))
            user_ids.append(batch["user_ids"].long().cpu())
            target_item_ids.append(tgt.cpu())
            target_rows.append(rows.cpu())
            target_tokens.append(batch["target_token_ids"].long().cpu())
            if torch.is_tensor(batch.get("target_timestamps")):
                target_timestamps.append(batch["target_timestamps"].long().cpu())

            if component["name"] == "sasrec":
                history = batch["history_item_ids"].long()
                if sasrec_wrapper is not None:
                    history = sasrec_wrapper._translate(history)
                h = sasrec_final_hidden(sasrec_raw, history, batch["history_mask"])
                hidden.append(h.detach().to(hidden_dtype).cpu())
            elif component["name"] == "interaction":
                assert evidence is not None
                if fill_value is None:
                    fill_value = max(interaction_fill_value(rmodel, batch, sid_len, device), reward_epsilon)
                    manifest["fill_value"] = fill_value
                r, v = interaction_sparse_batch(
                    rmodel, evidence, batch, item_id_to_semantic_row,
                    sid_len=sid_len, fill_value=fill_value, reward_epsilon=reward_epsilon, device=device,
                )
                csr_rows.extend(r)
                csr_vals.extend(v)

            if batch_idx < int(args.verify_batches):
                dense = compute_component_rewards(
                    spec, batch, item_tokens, semantic_row_to_item_id, item_id_to_semantic_row, reward_epsilon,
                ).float().clamp_min(reward_epsilon)
                dense_reference.append((processed, dense.cpu()))

            processed += bsz
            elapsed = max(time.time() - start_time, 1e-6)
            print(json.dumps({
                "type": "progress", "batch_idx": batch_idx, "processed": processed,
                "examples_per_sec": processed / elapsed,
            }, sort_keys=True), flush=True)

    if processed == 0:
        raise RuntimeError("No examples were processed.")

    factors["user_ids"] = torch.cat(user_ids)
    factors["target_item_ids"] = torch.cat(target_item_ids)
    factors["target_semantic_row_ids"] = torch.cat(target_rows)
    factors["target_token_ids"] = torch.cat(target_tokens)
    if target_timestamps:
        factors["target_timestamps"] = torch.cat(target_timestamps)
    if component["name"] == "sasrec":
        factors["hidden"] = torch.cat(hidden).contiguous()
    elif component["name"] == "interaction":
        lengths = np.asarray([int(r.size) for r in csr_rows], dtype=np.int64)
        indptr = np.zeros(processed + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(lengths)
        factors["indptr"] = torch.as_tensor(indptr)
        factors["cols"] = torch.as_tensor(np.concatenate(csr_rows) if csr_rows else np.empty(0, dtype=np.int64)).long()
        factors["vals"] = torch.as_tensor(np.concatenate(csr_vals) if csr_vals else np.empty(0, dtype=np.float32)).float()
        manifest["num_sparse_entries"] = int(indptr[-1])
        manifest["mean_sparse_entries_per_example"] = float(indptr[-1] / max(processed, 1))

    manifest["num_examples"] = processed
    manifest["elapsed_sec"] = time.time() - start_time
    manifest["examples_per_sec"] = processed / max(manifest["elapsed_sec"], 1e-6)
    torch.save(factors, output_dir / FACTORS_FILENAME)
    manifest["factor_bytes"] = int((output_dir / FACTORS_FILENAME).stat().st_size)
    manifest["dense_row_bytes_float16"] = int(num_items * 2 * processed)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # -- verification against the dense code path --------------------------------
    if dense_reference:
        cache = FactorizedRewardCache(output_dir, power=1.0, chunk_rows=int(args.chunk_rows))
        max_abs = 0.0
        max_rel = 0.0
        n_checked = 0
        for start, dense in dense_reference:
            idx = torch.arange(start, start + int(dense.size(0)), dtype=torch.long)
            rebuilt = cache.component_rows(idx, device).cpu()
            diff = (rebuilt - dense).abs()
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / dense.abs().clamp_min(1e-12)).max()))
            n_checked += int(dense.size(0))
        manifest["verification"] = {
            "batches": len(dense_reference), "examples": n_checked,
            "max_abs_diff": max_abs, "max_rel_diff": max_rel,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"type": "verification", **manifest["verification"]}, sort_keys=True), flush=True)

    print(json.dumps({
        "type": "done", "manifest": str(output_dir / "manifest.json"), "num_examples": processed,
        "component": component["name"], "mode": component["mode"],
        "factor_bytes": manifest["factor_bytes"], "dense_row_bytes_float16": manifest["dense_row_bytes_float16"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
