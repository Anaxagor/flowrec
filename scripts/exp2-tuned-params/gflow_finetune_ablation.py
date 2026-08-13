#!/usr/bin/env python
"""GFlow fine-tuning ablation on the exp2 tuned-params assets.

Grid: {beauty, instruments} x num_negatives {1, 3, 10, 100} x objective
{tb, db, subtb} (24 runs by default).

Per run:
  - TIGER init: the exp2 SFT checkpoint
    (checkpoints/exp2-tuned_params/initial/tiger/<ds>/best.pt).
  - Candidates: the unified curriculum sampler over SASRec-teacher scores
    (checkpoints/exp2-tuned_params/initial/sasrec/<ds>/best_model.pt),
    ramping easy -> hard over --curriculum-steps, history excluded.
  - Reward: (R_sasrec * R_interaction * R_sim_hier) ** power with the power
    (default 16) baked in as MultiplicativeRewardModel component exponents,
    so the cached logZ / flows see the already-powered reward.
  - Loss: SFT + gflow_weight (default 0.1) * {TB | DB | SubTB}. The
    normalizing constant logZ = log sum_catalog R and the DB/SubTB prefix
    flows are ANALYTICAL: a full-catalog reward cache per dataset
    (scripts/precompute_reward_cache.py) supplies the exact per-example
    reward sum, and exact prefix flows for the sampled candidates are
    computed from the cache's dense top-M rewards via CatalogPrefixFlowIndex.
    Nothing about Z or F is learned.
  - Validation: constrained beam-5 NDCG@5 (val/sft/ndcg@5), <= 500 epochs.

Dataset notes:
  - beauty 2014 is all-verified, so it trains without a positive criterion:
    the auxiliary interaction store is empty and the interaction reward is a
    neutral 1.0 there (the product degrades to sasrec * sim_hier).
  - instruments trains with positive_criterion="verified_purchase == True"
    instead of use_verified_only=true. The positive stream (and therefore the
    item catalog the checkpoints were trained on) is identical; unverified
    reviews become the auxiliary store the interaction reward reads, with
    strictly-before-t lookups. A catalog mismatch would fail loudly when the
    checkpoint's item_semantic_ids buffer refuses to load.

Example:
  python scripts/exp2-tuned-params/gflow_finetune_ablation.py --device cuda
  python scripts/exp2-tuned-params/gflow_finetune_ablation.py \
      --datasets beauty --num-candidates 3 --objectives tb --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import MethodOutput  # noqa: E402
from flowrec.ft_methods.gflow.config import GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.method import GFlowMethod  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.ft_methods.sft.method import SFTMethod  # noqa: E402
from flowrec.models.base import Batch, FlowRecModel  # noqa: E402
from flowrec.models.tiger import TigerModel  # noqa: E402
from flowrec.trajectories.semantic import SemanticTrajectoryBatch  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Assets (all paths PROJECT_ROOT-relative; made absolute in the run config)
# --------------------------------------------------------------------------- #
DATASETS: dict[str, dict[str, Any]] = {
    "beauty": {
        "sft_config": "configs/exp2-tuned-params/tiger_beauty_sft.json",
        "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt",
        # The promoted best_model.pt architecture lives in the tuning summary
        # next to the checkpoint; the configs/ YAML is stale (2 layers vs 3).
        "sasrec_params": "checkpoints/exp2-tuned_params/initial/sasrec/tuning_summary.json",
        "sasrec_checkpoint": "checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_model.pt",
        "rqvae_model": "datasets/semantic_ids/exp2-tuned-params/beauty/rqvae/model.pt",
        # 2014 beauty is all-verified: a verified criterion would leave an
        # empty auxiliary store AND change nothing about the positives.
        "positive_criterion": None,
    },
    "instruments": {
        "sft_config": "configs/exp2-tuned-params/tiger_instruments_sft.json",
        "reviews_path": "data/raw/dataframe_format/reviews_instruments_2018.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_instruments_2018.parquet",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/instruments/best.pt",
        "sasrec_params": "checkpoints/exp2-tuned_params/initial/sasrec/instruments/best_trial.json",
        "sasrec_checkpoint": "checkpoints/exp2-tuned_params/initial/sasrec/instruments/best_model.pt",
        "rqvae_model": "datasets/semantic_ids/exp2-tuned-params/instruments/rqvae/model.pt",
        # Same positive stream as the checkpoint's use_verified_only=true,
        # but unverified reviews land in the auxiliary store for the
        # interaction reward.
        "positive_criterion": "verified_purchase == True",
    },
}

# exp2 semantic-ID tables are level-offset: token = code + 1 + level * 256.
LEVEL_STRIDE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "GFlow fine-tuning ablation (exp2): curriculum SASRec sampler, "
        "product-of-three reward ** power, analytical logZ/flows for TB/DB/SubTB."
    )
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--num-candidates", type=int, nargs="+", default=[1, 3, 10, 100],
                   help="Negative-sample counts for the ablation")
    p.add_argument("--objectives", nargs="+", choices=["tb", "db", "subtb"],
                   default=["tb", "db", "subtb"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Reward / loss shape.
    p.add_argument("--reward-power", type=float, default=16.0,
                   help="Exponent applied to the reward product (component weights)")
    p.add_argument("--gflow-weight", type=float, default=0.1)
    p.add_argument("--sasrec-transform", default="softplus",
                   help="SASRec score->reward transform (TB-safe: softplus/exp)")
    p.add_argument("--sasrec-temperature", type=float, default=1.0)

    # Sampler.
    p.add_argument("--curriculum-steps", type=int, default=10_000,
                   help="Steps to ramp curriculum negatives from easy to hard")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Curriculum softmax temperature over SASRec scores")
    p.add_argument("--exclude-history", action=argparse.BooleanOptionalAction, default=True)

    # Reward cache (analytical logZ + flows).
    p.add_argument("--top-m", type=int, default=1024,
                   help="Cache top-M rewards kept per example (flows use these)")
    p.add_argument("--cache-batch-size", type=int, default=512)
    p.add_argument("--force-cache", action="store_true",
                   help="Rebuild the reward cache even if a manifest exists")

    # Training.
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience in validations of val/sft/ndcg@5")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-sample-fraction", type=float, default=None)

    # Test evaluation.
    p.add_argument("--test-ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--eval-num-beams", type=int, default=20)

    # Output.
    p.add_argument("--output-root", type=Path, default=Path("experiments/exp2-gflow-finetune"))
    p.add_argument("--cache-root", type=Path, default=Path("artifacts/exp2-gflow-finetune/reward_cache"))
    p.add_argument("--promote-root", type=Path,
                   default=Path("checkpoints/exp2-tuned_params/finetuned/tiger"),
                   help="Best checkpoints are copied here; pass '' to disable")

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0,
                   help="Abort when the SFT checkpoint's teacher-forced val NLL exceeds this")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: ~2k cached examples, 1 epoch, unshuffled")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class IndexedDataset(Dataset):
    """Wrap a SequenceDataset so every record carries its dataset index.

    The index links each training example to its reward-cache row.
    """

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = dict(self.base[idx])
        record["example_index"] = int(idx)
        return record


class IndexedCollator:
    def __init__(self, base) -> None:
        self.base = base

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base(examples)
        batch["example_indices"] = torch.as_tensor(
            [int(ex["example_index"]) for ex in examples], dtype=torch.long
        )
        return batch


def indexed_train_loader(dm, *, shuffle: bool, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        IndexedDataset(dm.train_dataset),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=dm.cfg.num_workers,
        pin_memory=dm.cfg.pin_memory,
        persistent_workers=dm.cfg.persistent_workers and dm.cfg.num_workers > 0,
        drop_last=False,
        collate_fn=IndexedCollator(dm._make_collator()),
        generator=generator if shuffle else None,
    )


class AliasedSFTMethod(SFTMethod):
    """SFT with generation_ndcg@K aliased to ndcg@K so the trainer can
    monitor ``val/sft/ndcg@5``."""

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
            for src, dst in (
                (f"{prefix}/generation_recall@{k}", f"{prefix}/recall@{k}"),
                (f"{prefix}/generation_ndcg@{k}", f"{prefix}/ndcg@{k}"),
            ):
                if src in metrics:
                    metrics[dst] = metrics[src]
        return MethodOutput(loss=out.loss, metrics=metrics, model_output=out.model_output, extra=out.extra)


# --------------------------------------------------------------------------- #
# GFlow method: curriculum candidates + cache-analytical logZ / flows
# --------------------------------------------------------------------------- #
class CurriculumAnalyticalGFlow(GFlowMethod):
    """GFlow whose reward-cache mode keeps the external curriculum sampler.

    The stock reward-cache path replaces the candidate sampler with the
    cache's top-M rows. Here the candidates always come from
    ``self.candidate_sampler`` (curriculum over SASRec scores); the cache
    contributes only the analytical quantities:

      - logZ = log(full-catalog reward sum) per example, exact from the cache;
      - for DB/SubTB, ``cache_top_rows``/``cache_top_rewards`` extras, from
        which GFlowMethod computes exact prefix flows for the sampled
        candidates via CatalogPrefixFlowIndex (dense top-M, epsilon tail).

    TB rewards are evaluated with the live reward model (exact for any
    candidate). DB/SubTB rewards are gathered from the same dense top-M
    vector the flows are built from, so terminal flows and rewards agree.
    """

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        self._sampler_step = step
        return super().compute_loss(model, batch, stage=stage, step=step)

    def _precomputed_reward_cache_tb_inputs(
        self,
        model: FlowRecModel,
        batch: Batch,
    ):
        if self.candidate_sampler is None:
            raise ValueError("CurriculumAnalyticalGFlow requires an external candidate_sampler.")
        if "example_indices" not in batch:
            raise KeyError("Reward-cache lookups require batch['example_indices'].")
        eps = float(self.cfg.reward_epsilon)
        device = batch["example_indices"].device

        cache = self._get_reward_cache()
        rows, top_rewards, reward_sum, _, top_mass, target_in_top, _ = cache.select_tb(
            batch["example_indices"], device=device
        )
        log_z = reward_sum.to(device=device, dtype=torch.float32).clamp_min(eps).log()

        trajectories = self.candidate_sampler.sample(
            batch, model=model, step=getattr(self, "_sampler_step", None)
        )
        trajectories.extra["cache_top_rows"] = rows.clamp_min(0)
        trajectories.extra["cache_top_rewards"] = top_rewards.to(
            device=device, dtype=torch.float32
        ).clamp_min(eps)
        trajectories.extra["sampler_phase"] = "fixed_logz"

        if self.cfg.objective == "tb":
            rewards = self.reward_model(batch, trajectories).float().clamp_min(eps)
        else:
            # Gather rewards from the same dense vector the analytical flows
            # use, so terminal prefix flows equal log reward exactly.
            index = self._get_exact_flow_index(model)
            dense = index.dense_rewards(
                trajectories.extra["cache_top_rows"],
                trajectories.extra["cache_top_rewards"],
                reward_epsilon=eps,
            )
            cand_rows = trajectories.candidate_item_row_ids
            if cand_rows is None:
                raise ValueError("Sampler must provide candidate_item_row_ids for DB/SubTB.")
            safe = cand_rows.to(device).clamp(0, dense.size(1) - 1)
            rewards = dense.gather(1, safe)
            rewards = torch.where(
                cand_rows.to(device).ge(0), rewards, torch.full_like(rewards, eps)
            )

        metrics = {
            "reward_cache_tb_active": torch.tensor(1.0, device=device),
            "fixed_log_z_active": torch.tensor(1.0, device=device),
            "curriculum_candidates_active": torch.tensor(1.0, device=device),
            "reward_cache_tb_reward_sum_mean": reward_sum.detach().float().mean(),
            "reward_cache_tb_top_mass_mean": top_mass.detach().float().mean(),
            "reward_cache_tb_target_in_top_mean": target_in_top.detach().float().mean(),
        }
        # candidate_weights=None: TB(uniform) ignores them; DB defaults to
        # reward-proportional residual weights; SubTB(uniform) uses ones.
        return trajectories, rewards, log_z, None, metrics


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def sasrec_method_cfg(dataset: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """SASRec teacher config from the promoted-checkpoint metadata.

    Handles both shapes: ``tuning_summary.json`` ({dataset: {best_params}})
    and ``best_trial.json`` ({best_params}).
    """
    payload = load_json(abspath(spec["sasrec_params"]))
    if dataset in payload:
        payload = payload[dataset]
    params = payload.get("best_params", payload)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
    return {
        "checkpoint_path": str(abspath(spec["sasrec_checkpoint"])),
        "model": {k: params[k] for k in model_keys if k in params},
    }


def build_run_config(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    """One config drives the datamodule, the reward registry, and the cache
    precompute — a single source of truth for reward semantics."""
    spec = DATASETS[dataset]
    config = copy.deepcopy(load_json(abspath(spec["sft_config"])))

    config["experiment_name"] = f"gflow_finetune_{dataset}"
    config["seed"] = int(args.seed)
    config["data"]["reviews_path"] = str(abspath(spec["reviews_path"]))
    config["data"]["metadata_path"] = str(abspath(spec["metadata_path"]))
    config["data"]["semantic_id_path"] = str(abspath(config["data"]["semantic_id_path"]))

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_cfg.pop("use_verified_only", None)
    if spec["positive_criterion"] is not None:
        dataset_cfg["positive_criterion"] = spec["positive_criterion"]
    config["dataset"] = dataset_cfg

    dataloader = dict(config.get("dataloader", {}))
    dataloader["train_batch_size"] = int(args.batch_size)
    dataloader["batch_size"] = int(args.batch_size)
    dataloader["validation_batch_size"] = int(args.val_batch_size)
    dataloader["num_workers"] = int(args.num_workers)
    dataloader["persistent_workers"] = int(args.num_workers) > 0
    config["dataloader"] = dataloader

    power = float(args.reward_power)
    config["method"] = {
        "name": "gflow",
        "reward_model": "multiply",
        "reward_epsilon": 1e-6,
        "reward": {
            "components": [
                {
                    "name": "sasrec",
                    "weight": power,
                    "reward": {
                        "transform": str(args.sasrec_transform),
                        "temperature": float(args.sasrec_temperature),
                    },
                },
                {
                    "name": "interaction",
                    "weight": power,
                    "reward": {},
                },
                {
                    "name": "similarity",
                    "weight": power,
                    "reward": {
                        "mode": "hierarchical",
                        "level_stride": LEVEL_STRIDE,
                        "rqvae_model_path": str(abspath(spec["rqvae_model"])),
                    },
                },
            ]
        },
        "sasrec": sasrec_method_cfg(dataset, spec),
    }
    return config


# --------------------------------------------------------------------------- #
# Assets per dataset
# --------------------------------------------------------------------------- #
def load_tiger_checkpoint(model: TigerModel, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload
    if isinstance(payload, Mapping):
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
    # Cluster runs may carry a DB/SubTB flow head the local model doesn't have.
    state = {k: v for k, v in state.items() if not k.startswith("flow_head.")}
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def sanity_check_checkpoint(
    model: TigerModel, dm, device: torch.device, threshold: float
) -> float:
    """Teacher-forced val NLL of the SFT init; ~2-4 is healthy, ~15 is chance.

    This guards against silent data/checkpoint mismatches (catalog, SID
    table, ordering), which this repo has been bitten by before.
    """
    probe = SFTMethod(SFTConfig(compute_eval_metrics=False, compute_token_metrics=False,
                                evaluate_generation=False))
    model.eval()
    batch = move_batch(next(iter(dm.val_dataloader())), device)
    nll = float(probe.compute_loss(model, batch, stage="val").loss.item())
    if nll > threshold:
        raise RuntimeError(
            f"SFT checkpoint sanity check FAILED: teacher-forced val NLL {nll:.3f} "
            f"> {threshold:.1f}. The checkpoint does not match the datamodule "
            "(catalog / semantic IDs / ordering); refusing to fine-tune noise. "
            "Override with --skip-sanity-check or --sanity-nll-threshold."
        )
    return nll


def ensure_reward_cache(
    dataset: str, config_path: Path, cache_dir: Path, num_train_examples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not args.force_cache:
        manifest = load_json(manifest_path)
    else:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "precompute_reward_cache.py"),
            "--config", str(config_path),
            "--output-dir", str(cache_dir),
            "--split", "train",
            "--top-m", str(int(args.top_m)),
            "--batch-size", str(int(args.cache_batch_size)),
            "--num-workers", str(int(args.num_workers)),
            "--device", str(args.device),
        ]
        if args.smoke:
            cmd += ["--max-examples", str(_smoke_examples(args))]
        print(f"[{dataset}] precomputing reward cache: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        manifest = load_json(manifest_path)

    cached = sum(int(s.get("num_examples", 0)) for s in manifest.get("shards", []))
    expected = _smoke_examples(args) if args.smoke else num_train_examples
    if cached < min(expected, num_train_examples):
        raise RuntimeError(
            f"Reward cache at {cache_dir} covers {cached} examples but the train "
            f"split has {num_train_examples}. Rebuild it with --force-cache."
        )
    return manifest


def _smoke_examples(args: argparse.Namespace) -> int:
    return int(args.batch_size) * 8


# --------------------------------------------------------------------------- #
# Test-time beam evaluation (constrained HR/NDCG/Coverage)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_generation(
    model: TigerModel, dm, *, ks: tuple[int, ...], num_beams: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    max_k = max(ks)
    catalog = {tuple(map(int, row)) for row in model.item_semantic_ids.detach().cpu().tolist()}

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    coverage: dict[int, set] = {k: set() for k in ks}
    n = 0
    for batch_idx, batch in enumerate(dm.test_dataloader(), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        batch = move_batch(batch, device)
        generated = model.generate(
            batch,
            num_beams=num_beams,
            num_return_sequences=max_k,
            constrain_valid_sequences=True,
        )
        target = batch["target_token_ids"].long()
        mask = batch.get("target_token_mask", torch.ones_like(target, dtype=torch.bool)).bool()
        matches = ((generated == target.unsqueeze(1)) | ~mask.unsqueeze(1)).all(dim=-1)
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
                key = tuple(map(int, row))
                if key in catalog:
                    coverage[k].add(key)

    out: dict[str, float] = {"num_test_examples": float(n)}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / max(n, 1)
        out[f"NDCG@{k}"] = ndcgs[k] / max(n, 1)
        out[f"Coverage@{k}"] = len(coverage[k]) / max(len(catalog), 1)
    return out


def make_status_logger(tag: str):
    keys = (
        "train/loss", "train/sft/loss", "train/gflow/gflow_loss",
        "train/gflow/reward_mean", "val/loss", "val/sft/ndcg@5",
    )

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") not in {"epoch_end", "val_end", "test_end", "non_finite_loss_skipped"}:
            return
        parts = [f"{k}={payload[k]:.5g}" for k in keys if isinstance(payload.get(k), (int, float))]
        for block_key in ("train", "val"):
            block = payload.get(block_key, {})
            if isinstance(block, Mapping):
                parts += [f"{k}={block[k]:.5g}" for k in keys if isinstance(block.get(k), (int, float))]
        epoch = payload.get("epoch")
        header = f"[{tag}] {payload.get('type')}" + (
            f" epoch={int(epoch) + 1}" if isinstance(epoch, int) else ""
        )
        print(header + (" | " + " | ".join(parts) if parts else ""), flush=True)

    return logger


# --------------------------------------------------------------------------- #
# One run of the grid
# --------------------------------------------------------------------------- #
def run_single(
    *,
    dataset: str,
    objective: str,
    num_negatives: int,
    config: Mapping[str, Any],
    config_path: Path,
    dm,
    sasrec,
    reward_model: torch.nn.Module,
    cache_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = f"{dataset}/{objective}_k{num_negatives}"
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fresh SFT init for every run.
    model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    load_tiger_checkpoint(model, abspath(DATASETS[dataset]["tiger_checkpoint"]))
    model.to(device)

    mapping = tiger_runner.item_id_to_semantic_row(dm)
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="curriculum",
            num_negatives=int(num_negatives),
            include_positive=True,
            exclude_history=bool(args.exclude_history),
            curriculum_steps=int(args.curriculum_steps),
            temperature=float(args.temperature),
        ),
        item_semantic_ids=model.item_semantic_ids.detach(),
        item_id_to_semantic_row=tiger_runner.restrict_item_id_mapping_to_scorer_vocab(
            mapping, sasrec
        ),
        scorer=sasrec,
    )

    gen_topk = (5,)
    gflow_cfg = GFlowConfig(
        objective=objective,
        sft_weight=1.0,
        gflow_weight=float(args.gflow_weight),
        logz_context="global",  # unused: reward_cache mode fixes logZ analytically
        tb_logz_source="reward_cache",
        reward_cache_path=str(cache_dir),
        reward_cache_top_m=int(args.top_m),
        reward_cache_probability="full",
        tb_loss_weighting="uniform",
        num_negative_samples=int(num_negatives),
        include_positive=True,
        reward_epsilon=1e-6,
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        evaluate_generation=True,
        generation_topk=gen_topk,
        generation_num_beams=5,
        generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
    )
    sft_cfg = SFTConfig(
        compute_train_metrics=False,
        compute_eval_metrics=True,
        compute_token_metrics=True,
        compute_item_metrics=False,
        evaluate_generation=True,
        generation_topk=gen_topk,
        generation_num_beams=5,
        generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
    )
    method = CurriculumAnalyticalGFlow(
        gflow_cfg,
        candidate_sampler=sampler,
        reward_model=reward_model,
        sft_method=AliasedSFTMethod(sft_cfg),
    )

    max_epochs = 1 if args.smoke else int(args.max_epochs)
    max_train_batches = 8 if args.smoke else args.max_train_batches
    max_eval_batches = 2 if args.smoke else args.max_eval_batches
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs,
            gradient_accumulation_steps=int(args.grad_accum),
            max_grad_norm=0.5,
            use_amp=device.type == "cuda",
            amp_dtype="float16",
            validate_every_epochs=1 if args.smoke else int(args.validate_every),
            log_every_steps=200,
            monitor="val/sft/ndcg@5",
            monitor_mode="max",
            early_stopping_patience=int(args.patience),
            checkpoint_dir=run_dir / "checkpoints",
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            val_sample_fraction=args.val_sample_fraction,
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        logger=make_status_logger(tag),
    )

    resolved = {
        "dataset": dataset,
        "objective": objective,
        "num_negatives": int(num_negatives),
        "seed": int(args.seed),
        "gflow": method.cfg.to_dict(),
        "sampler": {
            "type": "sasrec_curriculum",
            "mode": "curriculum",
            "num_negatives": int(num_negatives),
            "include_positive": True,
            "exclude_history": bool(args.exclude_history),
            "curriculum_steps": int(args.curriculum_steps),
            "temperature": float(args.temperature),
        },
        "reward": {
            "form": "(sasrec * interaction * sim_hier) ** power",
            "power": float(args.reward_power),
            "components": config["method"]["reward"]["components"],
        },
        "analytical": {
            "log_z": "log full-catalog reward sum (reward cache)",
            "flows": "exact prefix flows from cache dense top-M" if objective != "tb" else "n/a",
            "cache_dir": str(cache_dir),
            "top_m": int(args.top_m),
        },
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "data_config_path": str(config_path),
        "sft_checkpoint": str(abspath(DATASETS[dataset]["tiger_checkpoint"])),
        "item_vocab_size": int(dm.item_vocab_size),
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] fine-tuning: epochs<={max_epochs}, device={device}", flush=True)
    started = time.time()
    fit = trainer.fit(
        indexed_train_loader(
            dm, shuffle=not args.smoke, batch_size=int(args.batch_size), seed=int(args.seed)
        ),
        dm.val_dataloader(),
    )
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    checkpoint_for_test = fit.best_checkpoint_path or fit.last_checkpoint_path
    if checkpoint_for_test:
        trainer.load_checkpoint(checkpoint_for_test, load_optimizer=False, load_scheduler=False)

    test_metrics = evaluate_generation(
        model, dm,
        ks=tuple(int(k) for k in args.test_ks),
        num_beams=int(args.eval_num_beams),
        max_batches=max_eval_batches,
    )
    write_json(run_dir / "test_metrics.json", {"dataset": dataset, "objective": objective,
                                               "num_negatives": int(num_negatives), **test_metrics})
    print(f"[{tag}] test: {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    if str(args.promote_root) and fit.best_checkpoint_path:
        promote_dir = abspath(args.promote_root) / dataset / f"gflow_{objective}_k{num_negatives}"
        promote_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit.best_checkpoint_path, promote_dir / "best.pt")
        sidecar = Path(str(fit.best_checkpoint_path) + ".json")
        if sidecar.exists():
            shutil.copy2(sidecar, promote_dir / "best.pt.json")
        write_json(promote_dir / "test_metrics.json", test_metrics)

    result = {
        "dataset": dataset,
        "objective": objective,
        "num_negatives": int(num_negatives),
        "status": "completed",
        "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1),
        "steps_completed": int(fit.state.global_step),
        "elapsed_sec": time.time() - started,
        "run_dir": str(run_dir),
        **{k: v for k, v in test_metrics.items() if k != "num_test_examples"},
    }

    del model, method, trainer, sampler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    tiger_runner.seed_everything(int(args.seed))
    device = torch.device(args.device)
    output_root = abspath(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"\n=== dataset: {dataset} ===", flush=True)
        config = build_run_config(dataset, args)
        config_path = output_root / dataset / "data_reward_config.json"
        write_json(config_path, config)

        data_cfg = tiger_runner.build_data_config(dict(config), config_path=config_path)
        dm = tiger_runner.build_data_module(data_cfg)
        dm.setup()
        store = dm.interaction_store
        print(
            f"[{dataset}] catalog={dm.item_vocab_size} items, "
            f"train={len(dm.train_dataset)} examples, "
            f"aux store={len(store) if store is not None else 0} events",
            flush=True,
        )
        if dataset == "beauty" and store is not None and len(store) == 0:
            print(
                f"[{dataset}] note: no auxiliary events (all-verified 2014 dump) — "
                "the interaction reward is a neutral 1.0 here.",
                flush=True,
            )

        # SFT-init sanity: refuse to fine-tune a checkpoint that does not
        # match this datamodule.
        probe_model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
        load_tiger_checkpoint(probe_model, abspath(DATASETS[dataset]["tiger_checkpoint"]))
        probe_model.to(device)
        if not args.skip_sanity_check:
            nll = sanity_check_checkpoint(probe_model, dm, device, float(args.sanity_nll_threshold))
            print(f"[{dataset}] SFT checkpoint sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
        del probe_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Shared per-dataset assets: SASRec teacher + the composed reward.
        sasrec = tiger_runner.build_sasrec_teacher_model(
            dm, dict(config["method"]["sasrec"]), config_path=config_path
        )
        sasrec.to(device).eval()
        reward_model = tiger_runner.build_reward_model(
            "multiply",
            data=dm,
            train_loader=None,
            config_path=config_path,
            seed=int(args.seed),
            reward_cfg=dict(config["method"]["reward"]),
            svd_cfg={},
            svd_model=None,
            sasrec_model=sasrec,
            reward_epsilon=float(config["method"]["reward_epsilon"]),
        )
        reward_model.to(device).eval()
        for module in (reward_model, sasrec):
            for param in module.parameters():
                param.requires_grad_(False)

        # Analytical constant + flows: full-catalog reward cache (train split).
        cache_dir = abspath(args.cache_root) / dataset
        manifest = ensure_reward_cache(
            dataset, config_path, cache_dir, len(dm.train_dataset), args
        )
        print(
            f"[{dataset}] reward cache ready: top_m={manifest['stored_top_m']}, "
            f"components={[c['name'] for c in manifest['components']]}",
            flush=True,
        )

        for num_negatives in args.num_candidates:
            for objective in args.objectives:
                run_dir = output_root / dataset / f"{objective}_k{num_negatives}"
                try:
                    result = run_single(
                        dataset=dataset,
                        objective=objective,
                        num_negatives=int(num_negatives),
                        config=config,
                        config_path=config_path,
                        dm=dm,
                        sasrec=sasrec,
                        reward_model=reward_model,
                        cache_dir=cache_dir,
                        run_dir=run_dir,
                        args=args,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the grid going
                    import traceback

                    print(f"[{dataset}/{objective}_k{num_negatives}] FAILED: {exc}", flush=True)
                    traceback.print_exc()
                    result = {
                        "dataset": dataset,
                        "objective": objective,
                        "num_negatives": int(num_negatives),
                        "status": "failed",
                        "error": str(exc),
                        "run_dir": str(run_dir),
                    }
                all_results.append(result)
                write_json(output_root / "summary.json", {"runs": all_results})

    print("\n=== summary ===", flush=True)
    for row in all_results:
        line = (
            f"{row['dataset']:<12} {row['objective']:<6} K={row['num_negatives']:<4} "
            f"{row['status']:<10}"
        )
        if row.get("status") == "completed":
            ndcg5 = row.get("NDCG@5")
            best = row.get("best_val_ndcg5")
            line += (
                f" best_val_ndcg@5={best:.5f}" if isinstance(best, float) else ""
            ) + (f" test_NDCG@5={ndcg5:.5f}" if isinstance(ndcg5, float) else "")
        print(line, flush=True)
    print(f"summary written to {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
