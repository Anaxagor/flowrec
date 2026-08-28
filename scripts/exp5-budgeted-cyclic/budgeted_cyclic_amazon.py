#!/usr/bin/env python
"""Budgeted cyclic GFlowNet sampling on Amazon Beauty 2014 (exp2 assets).

Follow-up to the beauty_2023 non-acyclic experiment (global flow oracle,
``flowrec/trajectories/nonacyclic.py``): there, exact correction of frozen
TIGER to the SASRec target R/Z needed thousands of DELETEs per trajectory.
This experiment asks the practical question instead: can a SMALL, hard
rollback budget B improve actual recommendation quality of a frozen TIGER
checkpoint, and at what inference cost?

Frozen models (never updated): the exp2 tuned-params TIGER SFT checkpoint and
the exp2 SASRec teacher. The cyclic policy keeps TIGER as its conditional ADD
proposal; only the per-state rollback probability d(c, s) (from the global
flow oracle, truncated / scaled / level-masked) changes.

Stages (all hyperparameters are chosen on the VALIDATION split; stage F runs
the selected configurations once on the TEST split):
  A  asset audit + frozen-model health: teacher-forced NLL gate, optional
     project-standard beam-20 TIGER metrics + full-catalog SASRec metrics,
     per-user rank_TIGER vs rank_SASRec and the user slices
     sasrec_better / tiger_better / approximately_equal (H5).
  B  small-user regression: rewards, prefix masses, global-flow-oracle
     identities, infinite-budget TV to R/Z, B=0 == append-only, alpha=1
     reward-conditional == p_R, monotone-transform rank equivalence.
  C  exact hard-budget study: P_T^(B) by dynamic programming for
     B in --budgets x reward transforms, exact costs, level-ablation rows.
  D  budgeted variants + matched-compute baselines: scaled oracle (lambda),
     no-immediate-repeat (MC), Best-of-N + SASRec, trie beam (+ SASRec
     rerank), direct reward-guided decoding (alpha), decoder temperatures,
     SASRec full-catalog Top-K; serving-style sampled evaluation.
  E  aggregation: summary CSVs, quality-compute Pareto plots, per-slice and
     per-level tables, config selection for the test run, report.md.
  F  test evaluation of the selected configurations + paired bootstrap CIs.

Per-user heavy quantities (TIGER prefix sweep, SASRec full-catalog scores)
are cached under the artifact tree and reused across stages/reruns.

Outputs (spec layout) under
``artifacts/non_acyclic_gflownet/beauty_2014/budgeted_sampling/``:
  config.yaml, metadata.json, selected_users_<split>.json,
  per_user/<split>/<uid>/{prefix_tables.npz, sasrec_scores.npz,
      exact_distributions.npz (first --save-distributions-users users)},
  summary/{model_comparison.csv, oracle_regression.csv, budget_metrics.csv,
      baseline_metrics.csv, matched_compute_comparison.csv, by_level.csv,
      oracle_by_level.csv, by_user_slice.csv, reward_transform_comparison.csv,
      serving_metrics.csv, test_metrics.csv, selected_configs.json},
  per_user/{budget_metrics_val.parquet, ...}, plots/*.png, report.md

Examples:
  # wiring check (a few users, small grids, CPU)
  python scripts/exp5-budgeted-cyclic/budgeted_cyclic_amazon.py \
      --num-users 3 --num-users-regression 2 --smoke
  # full validation study
  python scripts/exp5-budgeted-cyclic/budgeted_cyclic_amazon.py --stages ABCDE
  # test evaluation of the selected configs
  python scripts/exp5-budgeted-cyclic/budgeted_cyclic_amazon.py --stages F
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "scripts" / "exp2-tuned-params")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_tiger as tiger_runner  # noqa: E402
import sasrec_config_runner as sasrec_runner  # noqa: E402
import sft_finetune_control as sft_control  # noqa: E402  (loaders, sanity gate, beam eval)
from flowrec.evaluation.ranking import (  # noqa: E402
    aggregate_rank_metrics,
    coverage_at_k,
    metrics_from_rank,
    rank_of_target,
    topk_rows,
)
from flowrec.models.sasrec import SASRec, SASRecConfig  # noqa: E402
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel  # noqa: E402
from flowrec.trajectories.budgeted import (  # noqa: E402
    BudgetedCyclicSampler,
    best_of_n_exact_distribution,
    exact_budgeted_terminal_distribution,
    geometric_guidance_edge_prob,
    level_masked_rollback,
    scaled_rollback_prob,
    temperature_edge_prob,
    trie_beam_search,
)
from flowrec.trajectories.nonacyclic import (  # noqa: E402
    SemanticIDTrie,
    TigerPrefixTables,
    build_global_flow_oracle,
    compute_tiger_edge_distributions,
    exact_cyclic_terminal_distribution,
    exact_terminal_distribution,
    js_divergence,
    node_reach_probability,
    reward_conditionals,
    reward_mass,
    tv_distance,
)

KS = (5, 10, 20)

DATASET_SPECS: dict[str, dict[str, Any]] = {
    # beauty == Amazon Beauty 2014 (exp2 tuned-params assets)
    "beauty": {
        "sft_config": "configs/exp2-tuned-params/tiger_beauty_sft.json",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt",
        "sasrec_checkpoint": "checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_model.pt",
        # the JSON's data paths are stale; these are the local parquet exports
        # (same overrides sft_finetune_control uses)
        "data_overrides": {
            "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
            "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        },
        "artifact_name": "beauty_2014",
    },
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="beauty", choices=sorted(DATASET_SPECS))
    p.add_argument("--stages", default="ABCDE", help="subset of ABCDEF, e.g. 'AB' or 'F'")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda (cpu is fastest for this tiny T5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-users", type=int, default=500, help="validation users for stages A/C/D")
    p.add_argument("--num-users-regression", type=int, default=100, help="users for stage B identities")
    p.add_argument("--num-test-users", type=int, default=None, help="stage F users (default: --num-users)")
    p.add_argument("--min-history-len", type=int, default=3)
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    p.add_argument("--reward-temperatures", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--no-softplus", action="store_true", help="drop the project-default softplus transform")
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
    p.add_argument("--scaled-budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--decoder-temperatures", type=float, nargs="+", default=[0.7, 1.5, 2.0])
    p.add_argument("--best-of-n", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--beam-sizes", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--level-ablation-budget", type=int, default=4)
    p.add_argument("--equal-rank-threshold", type=int, default=10,
                   help="|rank_TIGER - rank_SASRec| < threshold => approximately_equal slice")
    p.add_argument("--serving-trajectories", type=int, default=500)
    p.add_argument("--serving-candidates", type=int, nargs="+", default=[20, 50],
                   help="M values for candidate-generation + SASRec reranking")
    p.add_argument("--mc-seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="seeds for stochastic samplers (no-repeat, serving)")
    p.add_argument("--prefix-batch-size", type=int, default=512)
    p.add_argument("--standard-eval-max-batches", type=int, default=2,
                   help="val batches for the project-standard TIGER beam eval in stage A")
    p.add_argument("--skip-standard-eval", action="store_true")
    p.add_argument("--skip-sasrec-standard-eval", action="store_true")
    p.add_argument("--save-distributions-users", type=int, default=25,
                   help="save exact P_T^(B) matrices for the first N users")
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--optimize-budget-oracle", action="store_true",
                   help="phase-2: differentiable budget-optimal rollback on the regression users")
    p.add_argument("--output-dir", default=None,
                   help="default: artifacts/non_acyclic_gflownet/<artifact_name>/budgeted_sampling")
    p.add_argument("--no-reuse-cache", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny grids + _smoke output dir (wiring check)")
    args = p.parse_args(argv)
    if args.smoke:
        args.num_users = min(args.num_users, 3)
        args.num_users_regression = min(args.num_users_regression, 2)
        args.budgets = [0, 1, 4]
        args.reward_temperatures = [1.0]
        args.lambda_grid = [1.0]
        args.scaled_budgets = [2]
        args.alpha_grid = [0.5, 1.0]
        args.decoder_temperatures = [1.5]
        args.best_of_n = [4]
        args.beam_sizes = [10]
        args.serving_trajectories = 100
        args.serving_candidates = [20]
        args.mc_seeds = args.mc_seeds[:1]
        args.save_distributions_users = 1
        args.bootstrap_samples = 500
    return args


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return None


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def assert_still_frozen(model: torch.nn.Module, fingerprint: str) -> None:
    assert state_dict_fingerprint(model) == fingerprint, "frozen model parameters changed!"


def state_dict_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def history_lengths(ds: Any) -> np.ndarray:
    col = ds.examples["history_item_ids"]
    for attr in ("list", "str"):
        try:
            return getattr(col, attr).len().to_numpy()
        except Exception:
            pass
    return np.asarray([len(x) for x in col])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({len(df)} rows)", flush=True)


# --------------------------------------------------------------------------- #
# Experiment context: frozen models, trie, artifact dirs
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentContext:
    args: argparse.Namespace
    spec: dict[str, Any]
    out_dir: Path
    dm: Any
    tiger: Any
    sasrec: SASRec
    trie: SemanticIDTrie
    row_to_item: np.ndarray          # semantic row -> atomic item id
    item_to_row: np.ndarray          # atomic item id -> semantic row (-1 absent)
    device: torch.device
    tiger_fingerprint: str = ""
    sasrec_fingerprint: str = ""

    @property
    def summary_dir(self) -> Path:
        return self.out_dir / "summary"

    @property
    def plots_dir(self) -> Path:
        return self.out_dir / "plots"

    def per_user_dir(self, split: str, uid: str) -> Path:
        return self.out_dir / "per_user" / split / uid


def build_run_config(spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The checkpoint's own training config with local data paths (absolute)."""
    config = copy.deepcopy(json.loads(abspath(spec["sft_config"]).read_text()))
    config["seed"] = int(seed)
    config["comet"] = {"enabled": False}
    data = dict(config.get("data", {}))
    data.update(dict(spec.get("data_overrides", {})))
    for key in ("reviews_path", "metadata_path", "interactions_path", "embeddings_path", "semantic_id_path"):
        if data.get(key):
            data[key] = str(abspath(data[key]))
    config["data"] = data
    dataloader = dict(config.get("dataloader", {}))
    dataloader.update(num_workers=0, persistent_workers=False, pin_memory=False)
    config["dataloader"] = dataloader
    return config


def load_experiment_context(args: argparse.Namespace) -> ExperimentContext:
    spec = DATASET_SPECS[args.dataset]
    out_dir = abspath(
        args.output_dir
        or f"artifacts/non_acyclic_gflownet/{spec['artifact_name']}/budgeted_sampling"
    )
    if args.smoke and args.output_dir is None:
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    device = torch.device(args.device)

    config = build_run_config(spec, args.seed)
    config_path = out_dir / "resolved_run_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    print("building datamodule (leave-one-out split identical to the checkpoints) ...", flush=True)
    data_cfg = tiger_runner.build_data_config(dict(config), config_path=config_path)
    if hasattr(data_cfg, "processed_cache_dir") and not data_cfg.processed_cache_dir:
        # fingerprinted cache (config + source file identity) -> fast reruns
        data_cfg = dataclasses.replace(
            data_cfg, processed_cache_dir=str(abspath("data/processed_cache")))
    dm = tiger_runner.build_data_module(data_cfg)
    dm.setup()
    print(f"  users={dm.num_users} catalog_items={dm.num_items} sid_len={dm.sid_len}", flush=True)

    tiger = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    sft_control.load_tiger_checkpoint(tiger, abspath(spec["tiger_checkpoint"]))
    tiger.to(device)
    freeze_model(tiger)

    sasrec_payload = torch.load(abspath(spec["sasrec_checkpoint"]), map_location="cpu")
    sasrec = SASRec(SASRecConfig(**sasrec_payload["model_config"]))
    sasrec.load_state_dict(sasrec_payload["model_state_dict"], strict=True)
    if int(sasrec.cfg.item_vocab_size) != int(dm.item_vocab_size):
        raise RuntimeError(
            f"SASRec vocab {sasrec.cfg.item_vocab_size} != datamodule {dm.item_vocab_size}"
        )
    sasrec.to(device)
    freeze_model(sasrec)

    row_to_item_t = tiger_runner.item_id_by_semantic_row(dm)
    if row_to_item_t is None:
        raise RuntimeError("datamodule has no semantic-row -> item mapping")
    row_to_item = row_to_item_t.cpu().numpy().astype(np.int64)
    item_to_row = tiger_runner.item_id_to_semantic_row(dm).cpu().numpy().astype(np.int64)

    sids = tiger.item_semantic_ids.detach().cpu().numpy().astype(np.int64)
    trie = SemanticIDTrie(sids)  # catalog rows == semantic rows (sorted(dm.semantic_ids))
    assert trie.num_items == len(row_to_item) == dm.num_items

    ctx = ExperimentContext(
        args=args, spec=spec, out_dir=out_dir, dm=dm, tiger=tiger, sasrec=sasrec,
        trie=trie, row_to_item=row_to_item, item_to_row=item_to_row, device=device,
        tiger_fingerprint=state_dict_fingerprint(tiger),
        sasrec_fingerprint=state_dict_fingerprint(sasrec),
    )
    write_metadata(ctx, config)
    return ctx


def write_metadata(ctx: ExperimentContext, config: Mapping[str, Any]) -> None:
    import transformers

    spec = ctx.spec
    meta = {
        "dataset": ctx.args.dataset,
        "artifact_name": spec["artifact_name"],
        "git_commit": git_commit(),
        "created_unix": time.time(),
        "assets": {
            "tiger_checkpoint": {
                "path": str(spec["tiger_checkpoint"]),
                "sha256": sha256_file(abspath(spec["tiger_checkpoint"])),
            },
            "sasrec_checkpoint": {
                "path": str(spec["sasrec_checkpoint"]),
                "sha256": sha256_file(abspath(spec["sasrec_checkpoint"])),
            },
            "semantic_ids": {
                "path": config["data"]["semantic_id_path"],
                "sha256": sha256_file(Path(config["data"]["semantic_id_path"])),
            },
            "reviews_path": config["data"]["reviews_path"],
            "metadata_path": config["data"]["metadata_path"],
        },
        "catalog": {
            "num_items": int(ctx.dm.num_items),
            "num_users": int(ctx.dm.num_users),
            "sid_len": int(ctx.trie.sid_len),
            "trie_nodes": int(ctx.trie.num_nodes),
            "trie_nodes_per_level": [int(x) for x in ctx.trie.num_nodes_per_level],
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "cli_args": {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(ctx.args).items()},
    }
    meta = json.loads(json.dumps(meta, default=str))  # plain types only (TorchVersion etc.)
    (ctx.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (ctx.out_dir / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


# --------------------------------------------------------------------------- #
# Per-user assets: batch, SASRec scores, TIGER prefix tables (cached)
# --------------------------------------------------------------------------- #
@dataclass
class UserContext:
    uid: str
    split: str
    dataset_index: int
    history_len: int
    target_item_id: int
    target_row: int
    raw_scores: np.ndarray  # [num_items] SASRec scores, semantic-row order, float64


def select_users(ctx: ExperimentContext, split: str, num: int) -> list[int]:
    ds = ctx.dm.val_dataset if split == "validation" else ctx.dm.test_dataset
    lens = history_lengths(ds)
    eligible = np.flatnonzero(lens >= ctx.args.min_history_len)
    if len(eligible) < num:
        raise RuntimeError(f"only {len(eligible)} users with history >= {ctx.args.min_history_len}")
    rng = np.random.default_rng(ctx.args.seed)
    return sorted(int(i) for i in rng.choice(eligible, size=num, replace=False))


def single_user_batch(ctx: ExperimentContext, split: str, index: int) -> dict[str, Any]:
    dm = ctx.dm
    ds = dm.val_dataset if split == "validation" else dm.test_dataset
    ds_t, collate = dm._loader_parts(ds, dm._make_collator())
    batch = collate([ds_t[int(index)]])
    if int(batch["user_ids"].size(0)) != 1:
        raise AssertionError("expected a single-user batch")
    return batch


def build_user_context(ctx: ExperimentContext, split: str, index: int) -> tuple[UserContext, dict]:
    batch = single_user_batch(ctx, split, index)
    uid = str(ctx.dm.id2user[int(batch["user_ids"][0])])
    target_item = int(batch["target_item_ids"][0])
    target_row = int(ctx.item_to_row[target_item])
    if target_row < 0:
        raise AssertionError(f"target item {target_item} has no semantic row (user {uid})")
    hist_len = int(batch["history_mask"][0].sum())

    udir = ctx.per_user_dir(split, uid)
    udir.mkdir(parents=True, exist_ok=True)
    cache = udir / "sasrec_scores.npz"
    key = {"sasrec_checkpoint": ctx.spec["sasrec_checkpoint"], "num_items": ctx.trie.num_items,
           "dataset_index": int(index)}
    key_file = udir / "sasrec_scores.key.json"
    if (not ctx.args.no_reuse_cache and cache.exists() and key_file.exists()
            and json.loads(key_file.read_text()) == key):
        raw_rows = np.load(cache)["raw_scores"]
    else:
        with torch.no_grad():
            moved = {k: (v.to(ctx.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            raw = ctx.sasrec.score_items(moved).float().cpu()[0]  # [item_vocab], atomic ids
        raw_rows = raw.numpy().astype(np.float64)[ctx.row_to_item]  # -> semantic-row order
        np.savez_compressed(cache, raw_scores=raw_rows)
        key_file.write_text(json.dumps(key))
    uc = UserContext(uid=uid, split=split, dataset_index=int(index), history_len=hist_len,
                     target_item_id=target_item, target_row=target_row, raw_scores=raw_rows)
    return uc, batch


def load_or_compute_prefix_tables(
    ctx: ExperimentContext, uc: UserContext, batch: dict
) -> TigerPrefixTables:
    udir = ctx.per_user_dir(uc.split, uc.uid)
    cache = udir / "prefix_tables.npz"
    key = {"tiger_checkpoint": ctx.spec["tiger_checkpoint"], "num_nodes": int(ctx.trie.num_nodes),
           "dataset_index": uc.dataset_index, "skip_single_child": True}
    key_file = udir / "prefix_tables.key.json"
    if (not ctx.args.no_reuse_cache and cache.exists() and key_file.exists()
            and json.loads(key_file.read_text()) == key):
        z = np.load(cache)
        tables = TigerPrefixTables(**{k: z[k] for k in z.files})
    else:
        t0 = time.time()
        tables = compute_tiger_edge_distributions(
            ctx.tiger, batch, ctx.trie, batch_size=ctx.args.prefix_batch_size,
            skip_single_child=True,
        )
        np.savez_compressed(cache, **dataclasses.asdict(tables))
        key_file.write_text(json.dumps(key))
        print(f"  [{uc.split}/{uc.uid}] prefix sweep {time.time() - t0:.1f}s", flush=True)
    # Constrained float32 softmax leaves ~1e-6 mass deviations; the flow algebra
    # needs exact per-node normalization in float64 (beauty_2023 convention).
    trie = ctx.trie
    internal = trie.internal_node_ids
    sums = np.add.reduceat(tables.edge_prob, trie.child_start[internal])
    assert np.abs(sums - 1.0).max() < 1e-4
    norm = np.repeat(sums, trie.child_count[internal])
    tables.edge_prob[1:] = tables.edge_prob[1:] / norm
    tables.log_edge_prob[1:] = tables.log_edge_prob[1:] - np.log(norm)
    sums = np.add.reduceat(tables.edge_prob, trie.child_start[internal])
    assert np.abs(sums - 1.0).max() < 1e-12
    return tables


# --------------------------------------------------------------------------- #
# Reward transforms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RewardVariant:
    name: str
    transform: str
    temperature: float
    shift_to_max: bool          # numerically stable b(c) = max score
    reward_epsilon: float


def reward_variants(args: argparse.Namespace) -> list[RewardVariant]:
    out = []
    if not args.no_softplus:
        out.append(RewardVariant("softplus_T1", "softplus", 1.0, False, 1e-6))
    for t in args.reward_temperatures:
        out.append(RewardVariant(f"exp_T{t:g}", "exp", float(t), True, 1e-30))
    return out


def compute_rewards(ctx: ExperimentContext, uc: UserContext, variant: RewardVariant) -> np.ndarray:
    raw = uc.raw_scores
    shifted = raw - raw.max() if variant.shift_to_max else raw
    wrapper = SASRecRewardModel(
        ctx.sasrec,
        SASRecRewardConfig(transform=variant.transform, temperature=variant.temperature,
                           score_shift=0.0, reward_epsilon=variant.reward_epsilon),
    )
    rewards = wrapper.transform_scores(torch.from_numpy(shifted).unsqueeze(0)).squeeze(0)
    rewards = rewards.double().numpy()
    assert (rewards > 0).all()
    return rewards


@dataclass
class TargetBundle:
    variant: RewardVariant
    rewards: np.ndarray
    mass: np.ndarray
    target_q: np.ndarray
    p_r: np.ndarray
    oracle: Any | None
    oracle_error: str | None
    entropy: float
    d_star: np.ndarray | None = None   # oracle d*, clipped below 1 for the DP
    n_d_star_clipped: int = 0          # leaf STOP-prob underflow count (sharp targets)

    @property
    def z(self) -> float:
        return float(self.mass[0])


def build_target_bundle(
    ctx: ExperimentContext, uc: UserContext, tables: TigerPrefixTables, variant: RewardVariant
) -> TargetBundle:
    rewards = compute_rewards(ctx, uc, variant)
    mass = reward_mass(ctx.trie, rewards)
    q = reward_conditionals(ctx.trie, mass)
    p_r = rewards / rewards.sum()
    entropy = float(-(p_r * np.log(np.maximum(p_r, 1e-300))).sum())
    oracle, err = None, None
    d_star, n_clipped = None, 0
    try:
        oracle = build_global_flow_oracle(ctx.trie, tables.edge_prob, q, mass)
        # Sharp targets can underflow leaf STOP probs so that d* rounds to
        # exactly 1.0 in float64; the DP/sampler require d < 1 strictly.
        n_clipped = int((oracle.d_star >= 1.0).sum())
        d_star = np.minimum(oracle.d_star, 1.0 - 1e-12)
    except (ValueError, AssertionError) as exc:  # support violation / numerics
        err = str(exc)
    return TargetBundle(variant=variant, rewards=rewards, mass=mass, target_q=q,
                        p_r=p_r, oracle=oracle, oracle_error=err, entropy=entropy,
                        d_star=d_star, n_d_star_clipped=n_clipped)


# --------------------------------------------------------------------------- #
# Metric rows
# --------------------------------------------------------------------------- #
def base_row(uc: UserContext, variant: str, family: str, policy: str, **params: Any) -> dict[str, Any]:
    row = {
        "user": uc.uid, "split": uc.split, "transform": variant,
        "family": family, "policy": policy,
        "budget": np.nan, "lam": np.nan, "alpha": np.nan, "tau": np.nan, "n_cand": np.nan,
        "status": "ok",
    }
    row.update(params)
    return row


def dist_metrics(
    dist: np.ndarray, uc: UserContext, bundle: TargetBundle
) -> dict[str, Any]:
    rank = rank_of_target(dist, uc.target_row)
    out = metrics_from_rank(rank, KS)
    out["tv_to_pR"] = tv_distance(dist, bundle.p_r)
    out["js_to_pR"] = js_divergence(dist, bundle.p_r)
    out["expected_raw_reward"] = float((dist * uc.raw_scores).sum())
    out["top20"] = ",".join(str(int(r)) for r in topk_rows(dist, 20))
    return out


def list_metrics(
    rows_ranked: np.ndarray, uc: UserContext, num_items: int
) -> dict[str, Any]:
    """Metrics for a ranked candidate list (beam / rerank); misses rank last."""
    pos = np.flatnonzero(rows_ranked == uc.target_row)
    rank = int(pos[0]) + 1 if len(pos) else num_items
    out = metrics_from_rank(rank, KS)
    out["candidate_recall"] = float(len(pos) > 0)
    out["n_candidates_returned"] = int(len(rows_ranked))
    out["top20"] = ",".join(str(int(r)) for r in rows_ranked[:20])
    return out


def cost_fields(*, decoder_evals: float, steps: float, deletes: float = 0.0,
                sasrec_full_passes: int = 0, **extra: Any) -> dict[str, Any]:
    out = {"decoder_evals": float(decoder_evals), "expected_steps": float(steps),
           "expected_deletes": float(deletes), "sasrec_full_passes": int(sasrec_full_passes)}
    out.update(extra)
    return out


def dp_rows_fields(res: Any, sid_len: int) -> dict[str, Any]:
    out = cost_fields(
        decoder_evals=res.expected_decoder_evals, steps=res.expected_steps,
        deletes=res.expected_deletes, sasrec_full_passes=1,
        p_any_delete=res.p_any_delete, p_full_budget=res.p_full_budget,
        same_token_reselect=res.expected_same_token_reselect,
    )
    for lvl in range(1, sid_len + 1):
        out[f"deletes_L{lvl}"] = float(res.expected_deletes_by_level[lvl])
        out[f"reselect_L{lvl}"] = float(res.expected_same_token_reselect_by_level[lvl])
    return out


# --------------------------------------------------------------------------- #
# Stage A: audit + model comparison + user slices
# --------------------------------------------------------------------------- #
def stage_a(ctx: ExperimentContext, users: list[tuple[UserContext, TigerPrefixTables]]) -> pd.DataFrame:
    args = ctx.args
    print("== Stage A: asset audit and frozen-model comparison ==", flush=True)
    nll = sft_control.sanity_check_checkpoint(ctx.tiger, ctx.dm, ctx.device, 6.0)
    print(f"  TIGER teacher-forced val NLL {nll:.3f} (healthy ~2-4)", flush=True)

    health: dict[str, Any] = {"tiger_val_nll": nll}
    if not args.skip_standard_eval:
        index = sft_control.CatalogIndex(ctx.tiger, ctx.dm)
        health["tiger_beam20_val_subset"] = sft_control.evaluate_split(
            ctx.tiger, ctx.dm.val_dataloader(), index, ks=KS, num_beams=20,
            max_batches=args.standard_eval_max_batches,
        )
        print(f"  TIGER beam-20 val subset: {health['tiger_beam20_val_subset']}", flush=True)
    if not args.skip_sasrec_standard_eval:
        health["sasrec_val_full"] = sasrec_runner.evaluate_ranking_metrics(
            ctx.sasrec, ctx.dm.val_dataloader(), device=ctx.device,
            item_popularity=sasrec_runner.training_item_popularity(ctx.dm),
            num_catalog_items=ctx.dm.num_items, ks=KS,
        )
        print(f"  SASRec full val: {health['sasrec_val_full']}", flush=True)
    (ctx.summary_dir / "model_health.json").parent.mkdir(parents=True, exist_ok=True)
    (ctx.summary_dir / "model_health.json").write_text(json.dumps(health, indent=2, default=float))

    rows = []
    for uc, tables in users:
        p_append = exact_terminal_distribution(ctx.trie, tables.edge_prob)
        rank_t = rank_of_target(p_append, uc.target_row)
        rank_s = rank_of_target(uc.raw_scores, uc.target_row)
        if abs(rank_t - rank_s) < args.equal_rank_threshold:
            slice_name = "approximately_equal"
        elif rank_s < rank_t:
            slice_name = "sasrec_better"
        else:
            slice_name = "tiger_better"
        row = {"user": uc.uid, "split": uc.split, "history_len": uc.history_len,
               "rank_tiger": rank_t, "rank_sasrec": rank_s, "user_slice": slice_name}
        row.update({f"tiger_{k}": v for k, v in metrics_from_rank(rank_t, KS).items()})
        row.update({f"sasrec_{k}": v for k, v in metrics_from_rank(rank_s, KS).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, ctx.summary_dir / f"model_comparison_{users[0][0].split}.csv")
    agg = {
        "tiger": aggregate_rank_metrics(df["rank_tiger"].to_numpy(), KS),
        "sasrec": aggregate_rank_metrics(df["rank_sasrec"].to_numpy(), KS),
        "slices": df["user_slice"].value_counts().to_dict(),
    }
    print(f"  slices: {agg['slices']}", flush=True)
    (ctx.summary_dir / f"model_comparison_{users[0][0].split}.json").write_text(
        json.dumps(agg, indent=2, default=float))
    return df


# --------------------------------------------------------------------------- #
# Stage B: regression of the global flow oracle + spec sanity checks
# --------------------------------------------------------------------------- #
def stage_b(ctx: ExperimentContext, users: list[tuple[UserContext, TigerPrefixTables]],
            variants: list[RewardVariant]) -> pd.DataFrame:
    """Verify the spec's mandatory identities per (user, transform), recording
    failures instead of crashing: near-degenerate targets (Boltzmann at sharp
    T can put ~all mass on ONE item on beauty-2014) drive subtree masses to
    the 1e-30 floor, where the *relative* flow identities and leaf STOP probs
    lose float64 meaning (the beauty_2023 notebook documented the same). The
    run aborts only if a WELL-CONDITIONED pair fails a check: reward dynamic
    range max(R)/min(R) < 1e16 (inside float64's exact-flow regime) and no d*
    clipping. Beyond that range the identity errors are still tiny in ABSOLUTE
    flow terms (clipped kappa ~1e-3 of subtree masses that are ~1e-22 of Z);
    they are recorded, not fatal."""
    print("== Stage B: small-user regression (flow identities, exactness) ==", flush=True)
    trie = ctx.trie
    rows = []
    hard_failures = []
    for uc, tables in users:
        p_append = exact_terminal_distribution(trie, tables.edge_prob)
        for variant in variants:
            bundle = build_target_bundle(ctx, uc, tables, variant)
            row = {"user": uc.uid, "split": uc.split, "transform": variant.name,
                   "target_entropy": bundle.entropy,
                   "n_d_star_clipped": bundle.n_d_star_clipped, "status": "ok"}
            if bundle.oracle is None:
                row.update(status=f"oracle_failed: {bundle.oracle_error}")
                rows.append(row)
                continue
            g = bundle.oracle
            diag = g.diagnostics
            row.update({k: float(v) for k, v in diag.items()})
            row["E_del_inf"] = float(g.kappa.sum() / bundle.z)
            row["E_n_inf"] = float(g.state_flow.sum() / bundle.z)

            failed: list[str] = []

            def check(name: str, ok: bool) -> None:
                row[f"check_{name}"] = bool(ok)
                if not ok:
                    failed.append(name)

            check("support", diag["n_support_violations"] == 0)
            check("C_min", diag["C_min"] >= 1 - 1e-9)
            # Net-flow exactness on scale-aware yardsticks: the per-edge
            # |add-kappa-M|/M blows fp noise up by Z/M on tiny subtrees (the
            # raw value stays recorded in max_rel_err_net_flow).
            check("kappa_formula", diag["max_kappa_formula_disagreement"] < 1e-9)
            check("net_flow_abs",
                  diag["max_negative_kappa_before_clip"] <= 1e-9 * bundle.z)
            check("conservation", diag["max_rel_err_conservation"] < 1e-9)

            # infinite-budget policy reproduces R/Z exactly (regression test)
            term = trie.terminal_node_ids
            if float(np.min(1.0 - g.d_star[term])) > 0.0:
                inf_dist = exact_cyclic_terminal_distribution(trie, tables.edge_prob, g.d_star)
                row["tv_infinite_to_pR"] = tv_distance(inf_dist, bundle.p_r)
                check("infinite_tv", row["tv_infinite_to_pR"] < 1e-6)
            else:
                row["tv_infinite_to_pR"] = np.nan  # leaf STOP underflow (sharp target)
                check("infinite_tv_computable", False)

            # B=0 exactly reproduces append-only constrained TIGER
            res0 = exact_budgeted_terminal_distribution(trie, tables.edge_prob, bundle.d_star, 0)
            row["tv_B0_vs_append_only"] = tv_distance(res0.terminal, p_append)
            check("B0_append_only", row["tv_B0_vs_append_only"] < 1e-12)

            # alpha=1 direct reward-conditional decoding reproduces p_R
            p1 = geometric_guidance_edge_prob(trie, tables.edge_prob, bundle.target_q, 1.0)
            row["tv_alpha1_to_pR"] = tv_distance(exact_terminal_distribution(trie, p1), bundle.p_r)
            check("alpha1_pR", row["tv_alpha1_to_pR"] < 1e-9)

            # monotone transform: Top-K of p_R == SASRec full-catalog ranking
            check("rank_equal_raw_vs_pR",
                  rank_of_target(bundle.p_r, uc.target_row)
                  == rank_of_target(uc.raw_scores, uc.target_row))

            dynamic_range = float(bundle.rewards.max() / bundle.rewards.min())
            row["reward_dynamic_range"] = dynamic_range
            row["reward_floor_clamped"] = bool(
                (bundle.rewards <= variant.reward_epsilon * (1 + 1e-9)).any())
            well_conditioned = dynamic_range < 1e16 and bundle.n_d_star_clipped == 0
            if failed:
                row["status"] = ("check_failed: " if well_conditioned
                                 else "check_failed_degenerate_target: ") + ",".join(failed)
                if well_conditioned:
                    hard_failures.append((uc.uid, variant.name, failed))
            rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, ctx.summary_dir / "oracle_regression.csv")
    n_ok = int((df["status"] == "ok").sum())
    print(f"  {n_ok}/{len(df)} (user, transform) pairs passed all identities; "
          f"failures on degenerate targets: "
          f"{int(df['status'].str.startswith('check_failed_degenerate').sum())}", flush=True)
    if hard_failures:
        raise AssertionError(
            f"flow identities failed on well-conditioned targets: {hard_failures[:5]}")
    return df


# --------------------------------------------------------------------------- #
# Stages C/D: exact rows for one user
# --------------------------------------------------------------------------- #
@dataclass
class Grids:
    budgets: list[int] = field(default_factory=list)
    lambdas: list[float] = field(default_factory=list)
    scaled_budgets: list[int] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    taus: list[float] = field(default_factory=list)
    best_of_n: list[int] = field(default_factory=list)
    beams: list[int] = field(default_factory=list)
    level_ablation_budget: int | None = None
    serving_budgets: list[int] = field(default_factory=list)


def level_ablation_variants(trie: SemanticIDTrie, oracle: Any, d_star: np.ndarray) -> dict[str, np.ndarray]:
    L = trie.sid_len
    kappa_by_level = np.zeros(L + 1)
    np.add.at(kappa_by_level, trie.level[1:], oracle.kappa[1:])
    top_level = int(np.argmax(kappa_by_level[1:]) + 1)
    return {
        "all_levels": d_star,
        "semantic_only": level_masked_rollback(trie, d_star, set(range(1, L))),
        "suffix_only": level_masked_rollback(trie, d_star, {L}),
        f"top_level_L{top_level}": level_masked_rollback(trie, d_star, {top_level}),
    }


def compute_exact_rows(
    ctx: ExperimentContext, uc: UserContext, tables: TigerPrefixTables,
    bundles: list[TargetBundle], grids: Grids,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, np.ndarray]]:
    """All exact-distribution rows for one user. Returns (metric_rows,
    transform_rows, level_rows, oracle_level_rows, saved_distributions)."""
    trie = ctx.trie
    sid_len = trie.sid_len
    rows: list[dict] = []
    level_rows: list[dict] = []
    oracle_level_rows: list[dict] = []
    transform_rows: list[dict] = []
    dists_to_save: dict[str, np.ndarray] = {}

    p_append = exact_terminal_distribution(trie, tables.edge_prob)
    reach = node_reach_probability(trie, tables.edge_prob)

    # -- transform-independent baselines --------------------------------------
    tau_dists = {}
    for tau in grids.taus:
        p_tau = temperature_edge_prob(trie, tables.log_edge_prob, tau)
        tau_dists[tau] = exact_terminal_distribution(trie, p_tau)
    beam_lists = {k: trie_beam_search(trie, tables.log_edge_prob, k) for k in grids.beams}
    bofn_dists = {n: best_of_n_exact_distribution(p_append, uc.raw_scores, n)
                  for n in grids.best_of_n}

    for bundle in bundles:
        vname = bundle.variant.name
        # reward-target stats + oracle cost (spec 4 & 16.10)
        trow = {"user": uc.uid, "split": uc.split, "transform": vname,
                "target_entropy": bundle.entropy,
                "effective_support": float(np.exp(bundle.entropy)),
                "tv_append_to_pR": tv_distance(p_append, bundle.p_r),
                "oracle_ok": bundle.oracle is not None}
        if bundle.oracle is not None:
            trow["E_del_inf"] = float(bundle.oracle.kappa.sum() / bundle.z)
            trow["E_n_inf"] = float(bundle.oracle.state_flow.sum() / bundle.z)
        transform_rows.append(trow)

        # baseline: append-only == B=0 (family 'baseline' for the tables)
        row = base_row(uc, vname, "baseline", "append_only", budget=0)
        row.update(dist_metrics(p_append, uc, bundle))
        row.update(cost_fields(decoder_evals=sid_len, steps=sid_len + 1))
        rows.append(row)

        # SASRec full-catalog Top-K (reward-model reference bound)
        row = base_row(uc, vname, "baseline", "sasrec_full")
        row.update(dist_metrics(bundle.p_r, uc, bundle))
        row.update(cost_fields(decoder_evals=0, steps=0, sasrec_full_passes=1))
        rows.append(row)

        # decoder temperatures
        for tau, dist in tau_dists.items():
            row = base_row(uc, vname, "stochastic_tiger", "temperature", tau=tau)
            row.update(dist_metrics(dist, uc, bundle))
            row.update(cost_fields(decoder_evals=sid_len, steps=sid_len + 1))
            rows.append(row)

        # Best-of-N + SASRec selection (exact order-statistics distribution)
        for n, dist in bofn_dists.items():
            row = base_row(uc, vname, "best_of_n", "best_of_n", n_cand=n)
            row.update(dist_metrics(dist, uc, bundle))
            row.update(cost_fields(decoder_evals=sid_len * n, steps=(sid_len + 1) * n,
                                   sasrec_full_passes=1))
            rows.append(row)

        # trie beam search (+ SASRec rerank)
        for k, (beam_rows_arr, beam_scores) in beam_lists.items():
            row = base_row(uc, vname, "beam", "beam", n_cand=k)
            row.update(list_metrics(beam_rows_arr, uc, trie.num_items))
            row.update(cost_fields(decoder_evals=sid_len * k, steps=sid_len + 1))
            rows.append(row)
            order = np.argsort(-uc.raw_scores[beam_rows_arr], kind="stable")
            row = base_row(uc, vname, "beam_rerank", "beam_rerank", n_cand=k)
            row.update(list_metrics(beam_rows_arr[order], uc, trie.num_items))
            row.update(cost_fields(decoder_evals=sid_len * k, steps=sid_len + 1,
                                   sasrec_full_passes=1))
            rows.append(row)

        # direct reward-guided decoding
        for alpha in grids.alphas:
            try:
                p_a = geometric_guidance_edge_prob(trie, tables.edge_prob, bundle.target_q, alpha)
            except ValueError as exc:
                rows.append({**base_row(uc, vname, "direct_guidance", "guidance", alpha=alpha),
                             "status": f"failed: {exc}"})
                continue
            row = base_row(uc, vname, "direct_guidance", "guidance", alpha=alpha)
            row.update(dist_metrics(exact_terminal_distribution(trie, p_a), uc, bundle))
            row.update(cost_fields(decoder_evals=sid_len, steps=sid_len + 1, sasrec_full_passes=1))
            rows.append(row)

        if bundle.oracle is None:
            rows.append({**base_row(uc, vname, "truncated_oracle", "truncated_oracle"),
                         "status": f"oracle_failed: {bundle.oracle_error}"})
            continue
        oracle = bundle.oracle

        # exact hard-budget truncated oracle (the core experiment)
        save_mat = []
        for budget in grids.budgets:
            res = exact_budgeted_terminal_distribution(trie, tables.edge_prob, bundle.d_star, budget)
            row = base_row(uc, vname, "truncated_oracle", "truncated_oracle", budget=budget)
            row.update(dist_metrics(res.terminal, uc, bundle))
            row.update(dp_rows_fields(res, sid_len))
            rows.append(row)
            save_mat.append(res.terminal.astype(np.float32))
        if save_mat:
            dists_to_save[f"truncated_oracle_{vname}"] = np.stack(save_mat)

        # scaled oracle d_lambda (budget-aware strength; lambda tuned on val)
        for lam in grids.lambdas:
            d_lam = scaled_rollback_prob(bundle.d_star, lam)
            for budget in grids.scaled_budgets:
                res = exact_budgeted_terminal_distribution(trie, tables.edge_prob, d_lam, budget)
                row = base_row(uc, vname, "scaled_oracle", "scaled_oracle", budget=budget, lam=lam)
                row.update(dist_metrics(res.terminal, uc, bundle))
                row.update(dp_rows_fields(res, sid_len))
                rows.append(row)

        # level-specific rollback ablation (spec 13)
        if grids.level_ablation_budget is not None:
            for lname, d_masked in level_ablation_variants(trie, oracle, bundle.d_star).items():
                res = exact_budgeted_terminal_distribution(
                    trie, tables.edge_prob, d_masked, grids.level_ablation_budget)
                row = base_row(uc, vname, "level_ablation", lname,
                               budget=grids.level_ablation_budget)
                row.update(dist_metrics(res.terminal, uc, bundle))
                row.update(dp_rows_fields(res, sid_len))
                level_rows.append(row)

        # per-level oracle anatomy (spec 13: E[N^(l)], d_flow_l, C_l, KL_l).
        # Vectorized per-node D_KL(q_R || p_theta) (mismatch_tables' Python
        # loop over ~14k internal nodes is too slow at 500 users x transforms).
        q_ = bundle.target_q
        contrib = np.where(
            q_ > 0, q_ * (np.log(np.maximum(q_, 1e-300)) - tables.log_edge_prob), 0.0)
        contrib[0] = 0.0
        kl_full = np.zeros(trie.num_nodes)
        kl_full[trie.internal_node_ids] = np.add.reduceat(
            contrib, trie.child_start[trie.internal_node_ids])
        for lvl in range(1, sid_len + 1):
            at = trie.level == lvl
            parents_at = trie.level[trie.internal_node_ids] == (lvl - 1)
            par_ids = trie.internal_node_ids[parents_at]
            w = reach[par_ids]
            oracle_level_rows.append({
                "user": uc.uid, "split": uc.split, "transform": vname, "level": lvl,
                "E_del_level_inf": float(oracle.kappa[at].sum() / bundle.z),
                "flow_weighted_d": float(oracle.kappa[at].sum()
                                         / max(oracle.state_flow[at].sum(), 1e-300)),
                "C_reach_weighted": float((oracle.C[par_ids] * w).sum() / max(w.sum(), 1e-300)),
                "kl_q_p_reach_weighted": float((kl_full[par_ids] * w).sum() / max(w.sum(), 1e-300)),
            })
    return rows, transform_rows, level_rows, oracle_level_rows, dists_to_save


# --------------------------------------------------------------------------- #
# Stage D (sampled part): no-repeat MC + serving-style evaluation
# --------------------------------------------------------------------------- #
def serving_metrics_from_samples(
    terminal_rows: np.ndarray, uc: UserContext, m_values: list[int]
) -> dict[str, Any]:
    """Pure-sampling first-K-unique + candidate-recall@M + SASRec rerank Top-K."""
    _, first_idx = np.unique(terminal_rows, return_index=True)
    unique_in_order = terminal_rows[np.sort(first_idx)]
    out: dict[str, Any] = {"n_unique_items": int(len(unique_in_order))}
    for k in KS:
        firstk = unique_in_order[:k]
        hit = uc.target_row in set(int(x) for x in firstk)
        rank = int(np.flatnonzero(firstk == uc.target_row)[0]) + 1 if hit else None
        out[f"pure_recall@{k}"] = float(hit)
        out[f"pure_ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
    for m in m_values:
        cand = unique_in_order[:m]
        out[f"candidate_recall@{m}"] = float(uc.target_row in set(int(x) for x in cand))
        order = np.argsort(-uc.raw_scores[cand], kind="stable")
        reranked = cand[order]
        pos = np.flatnonzero(reranked == uc.target_row)
        rank = int(pos[0]) + 1 if len(pos) else None
        for k in KS:
            hit = rank is not None and rank <= k
            out[f"rerank_M{m}_recall@{k}"] = float(bool(hit))
            out[f"rerank_M{m}_ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
    return out


def compute_serving_rows(
    ctx: ExperimentContext, uc: UserContext, tables: TigerPrefixTables,
    bundles: list[TargetBundle], grids: Grids,
) -> list[dict]:
    args = ctx.args
    trie = ctx.trie
    rows: list[dict] = []
    n_traj = args.serving_trajectories
    m_values = list(args.serving_candidates)

    def run(policy_name: str, family: str, edge_prob: np.ndarray, d: np.ndarray,
            budget: int, vname: str, *, no_repeat: bool = False,
            node_value: np.ndarray | None = None, **params: Any) -> None:
        sampler = BudgetedCyclicSampler(trie, edge_prob, d, budget=budget,
                                        no_immediate_repeat=no_repeat, node_value=node_value)
        for seed in args.mc_seeds:
            t0 = time.time()
            out = sampler.sample(n_traj, seed=seed)
            runtime = time.time() - t0
            row = base_row(uc, vname, family, policy_name, budget=budget, **params)
            row["mc_seed"] = seed
            row["n_trajectories"] = n_traj
            row.update(serving_metrics_from_samples(out.terminal_row, uc, m_values))
            row.update(cost_fields(
                decoder_evals=float(trie.sid_len * n_traj + out.n_deletes.sum()),
                steps=float(out.n_steps.mean()), deletes=float(out.n_deletes.mean()),
                sasrec_full_passes=1,
                same_token_reselect=float(out.n_same_token_reselect) / max(out.n_revisions, 1),
                revision_precision=(float(out.n_revisions_improved) / max(out.n_revisions, 1)
                                    if out.n_revisions_improved >= 0 else np.nan),
                mask_fallbacks=int(out.n_mask_fallbacks),
                runtime_s=runtime,
                runtime_per_trajectory_ms=1e3 * runtime / n_traj,
            ))
            rows.append(row)

    zero_d = np.zeros(trie.num_nodes)
    for pos, bundle in enumerate(bundles):
        vname = bundle.variant.name
        if pos == 0:  # transform-independent policies: sample once
            run("append_only", "baseline", tables.edge_prob, zero_d, 0, vname)
            for tau in grids.taus:
                p_tau = temperature_edge_prob(trie, tables.log_edge_prob, tau)
                run("temperature", "stochastic_tiger", p_tau, zero_d, 0, vname, tau=tau)
        if bundle.oracle is None:
            continue
        for budget in grids.serving_budgets:
            run("truncated_oracle", "truncated_oracle", tables.edge_prob,
                bundle.d_star, budget, vname, node_value=bundle.mass)
            run("no_repeat", "no_repeat", tables.edge_prob,
                bundle.d_star, budget, vname, no_repeat=True,
                node_value=bundle.mass)
    return rows


# --------------------------------------------------------------------------- #
# Aggregation, selection, plots, report (stage E)
# --------------------------------------------------------------------------- #
CONFIG_COLS = ["transform", "family", "policy", "budget", "lam", "alpha", "tau", "n_cand"]
METRIC_COLS = [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS] + [
    "mrr", "target_rank", "tv_to_pR", "js_to_pR", "expected_raw_reward",
    "decoder_evals", "expected_steps", "expected_deletes", "p_any_delete",
    "p_full_budget", "same_token_reselect",
]


def aggregate_exact(df: pd.DataFrame, extra_group: list[str] | None = None) -> pd.DataFrame:
    df = df[df["status"] == "ok"].copy()
    group = CONFIG_COLS + (extra_group or [])
    cols = [c for c in METRIC_COLS if c in df.columns] + [
        c for c in df.columns if c.startswith("deletes_L") or c.startswith("reselect_L")]
    agg = df.groupby(group, dropna=False)[cols].mean().reset_index()
    agg["n_users"] = df.groupby(group, dropna=False)["user"].nunique().values
    return agg


_NUM_CFG_COLS = ["budget", "lam", "alpha", "tau", "n_cand"]


def merge_on_config(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge on CONFIG_COLS treating NaN parameters as equal (NaN != NaN in joins)."""
    lf, rf = left.copy(), right.copy()
    for c in _NUM_CFG_COLS:
        lf[c] = lf[c].fillna(-999.0)
        rf[c] = rf[c].fillna(-999.0)
    out = lf.merge(rf, on=CONFIG_COLS, how="left")
    for c in _NUM_CFG_COLS:
        out[c] = out[c].replace(-999.0, np.nan)
    return out


def coverage_table(df: pd.DataFrame, num_items: int) -> pd.DataFrame:
    df = df[(df["status"] == "ok") & df["top20"].notna()]
    rows = []
    for key, sub in df.groupby(CONFIG_COLS, dropna=False):
        lists = [np.array([int(x) for x in s.split(",") if x != ""], dtype=np.int64)
                 for s in sub["top20"]]
        row = dict(zip(CONFIG_COLS, key))
        for k in (10, 20):
            row[f"coverage@{k}"] = coverage_at_k([l[:k] for l in lists], num_items)
        rows.append(row)
    return pd.DataFrame(rows)


def config_label(row: Mapping[str, Any]) -> str:
    bits = [str(row["policy"])]
    for c, tag in (("budget", "B"), ("lam", "lam"), ("alpha", "a"), ("tau", "tau"), ("n_cand", "N")):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            bits.append(f"{tag}{v:g}")
    return "_".join(bits)


def select_configs(agg: pd.DataFrame, metric: str = "ndcg@10") -> dict[str, dict]:
    selected = {}
    for family, sub in agg.groupby("family"):
        sub = sub.dropna(subset=[metric])
        if not len(sub):
            continue
        best = sub.loc[sub[metric].idxmax()]
        selected[family] = {c: (None if pd.isna(best[c]) else
                                (best[c] if isinstance(best[c], str) else float(best[c])))
                            for c in CONFIG_COLS}
        selected[family][metric] = float(best[metric])
        selected[family]["decoder_evals"] = float(best.get("decoder_evals", np.nan))
    return selected


def paired_bootstrap(
    deltas: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    return {"mean_delta": float(deltas.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_improved": float((means > 0).mean()), "n_users": int(n)}


def stage_e(ctx: ExperimentContext, exact_df: pd.DataFrame, level_df: pd.DataFrame,
            oracle_level_df: pd.DataFrame, transform_df: pd.DataFrame,
            serving_df: pd.DataFrame, slices_df: pd.DataFrame) -> dict[str, dict]:
    print("== Stage E: aggregation, selection, plots, report ==", flush=True)
    sdir = ctx.summary_dir

    agg = aggregate_exact(exact_df)
    cov = coverage_table(exact_df, ctx.trie.num_items)
    agg = merge_on_config(agg, cov)
    write_csv(agg, sdir / "budget_metrics.csv")
    write_csv(agg[agg["family"].isin(["baseline", "stochastic_tiger", "best_of_n",
                                      "beam", "beam_rerank", "direct_guidance"])],
              sdir / "baseline_metrics.csv")

    # per-slice aggregation (H5)
    merged = exact_df.merge(slices_df[["user", "user_slice"]], on="user", how="left")
    by_slice = aggregate_exact(merged, extra_group=["user_slice"])
    write_csv(by_slice, sdir / "by_user_slice.csv")

    if len(level_df):
        write_csv(aggregate_exact(level_df), sdir / "by_level.csv")
    if len(oracle_level_df):
        write_csv(oracle_level_df.groupby(["transform", "level"]).mean(numeric_only=True)
                  .reset_index(), sdir / "oracle_by_level.csv")
    write_csv(transform_df.groupby("transform").mean(numeric_only=True).reset_index(),
              sdir / "reward_transform_comparison.csv")
    if len(serving_df):
        num_cols = [c for c in serving_df.select_dtypes(include=[np.number]).columns
                    if c not in CONFIG_COLS]
        write_csv(serving_df.groupby(CONFIG_COLS, dropna=False)[num_cols].mean().reset_index(),
                  sdir / "serving_metrics.csv")

    # matched-compute comparison: nearest-decoder-eval quality across families
    mc = agg.dropna(subset=["decoder_evals"]).copy()
    mc["config"] = mc.apply(config_label, axis=1)
    mc = mc.sort_values("decoder_evals")
    write_csv(mc[["transform", "family", "config", "decoder_evals", "expected_steps",
                  "ndcg@10", "recall@10", "mrr", "tv_to_pR", "js_to_pR"]],
              sdir / "matched_compute_comparison.csv")

    selected = select_configs(agg)
    (sdir / "selected_configs.json").write_text(json.dumps(selected, indent=2))
    print(f"  selected per family: { {k: config_label(v) for k, v in selected.items()} }", flush=True)

    make_plots(ctx, agg, exact_df, serving_df, transform_df, oracle_level_df, slices_df)
    write_report(ctx, agg, by_slice, level_df, transform_df, serving_df, selected, slices_df)
    return selected


# --------------------------------------------------------------------------- #
# Plots (spec 16)
# --------------------------------------------------------------------------- #
FAMILY_COLORS = {
    "truncated_oracle": "#d62728", "scaled_oracle": "#ff7f0e", "no_repeat": "#8c564b",
    "baseline": "#7f7f7f", "stochastic_tiger": "#17becf", "best_of_n": "#2ca02c",
    "beam": "#9467bd", "beam_rerank": "#e377c2", "direct_guidance": "#1f77b4",
    "level_ablation": "#bcbd22",
}


def _save(ctx: ExperimentContext, fig: plt.Figure, name: str) -> None:
    ctx.plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(ctx.plots_dir / name, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  plot {name}", flush=True)


def make_plots(ctx, agg, exact_df, serving_df, transform_df, oracle_level_df, slices_df) -> None:
    plt.rcParams.update({"figure.figsize": (7, 4.5), "axes.grid": True, "grid.alpha": 0.3})

    # 1-4: quality / alignment vs mean trajectory length (Pareto views)
    for metric, fname in (("recall@10", "plot_1_recall10_vs_length.png"),
                          ("ndcg@10", "plot_2_ndcg10_vs_length.png"),
                          ("coverage@10", "plot_3_coverage10_vs_length.png")):
        sub = agg.dropna(subset=["expected_steps", metric])
        if not len(sub):
            continue
        fig, ax = plt.subplots()
        for fam, fsub in sub.groupby("family"):
            fsub = fsub.sort_values("expected_steps")
            ax.plot(fsub["expected_steps"], fsub[metric], "o-", ms=4, lw=1,
                    color=FAMILY_COLORS.get(fam, "k"), label=fam, alpha=0.8)
        ax.set_xlabel("mean trajectory length E[n]")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
        ax.set_title(f"{metric} vs trajectory length (validation)")
        _save(ctx, fig, fname)
    sub = agg.dropna(subset=["expected_steps", "tv_to_pR"])
    if len(sub):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, m in zip(axes, ("tv_to_pR", "js_to_pR")):
            for fam, fsub in sub.groupby("family"):
                fsub = fsub.sort_values("expected_steps")
                ax.plot(fsub["expected_steps"], fsub[m], "o-", ms=4, lw=1,
                        color=FAMILY_COLORS.get(fam, "k"), label=fam, alpha=0.8)
            ax.set_xlabel("mean trajectory length E[n]")
            ax.set_ylabel(m)
        axes[0].legend(fontsize=7)
        fig.suptitle("terminal divergence to p_R vs trajectory length")
        _save(ctx, fig, "plot_4_tv_js_vs_length.png")

    # 5: quality vs rollback budget
    to = agg[agg["family"] == "truncated_oracle"].dropna(subset=["budget"])
    if len(to):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for tname, tsub in to.groupby("transform"):
            tsub = tsub.sort_values("budget")
            axes[0].plot(tsub["budget"], tsub["ndcg@10"], "o-", label=tname)
            axes[1].plot(tsub["budget"], tsub["recall@10"], "o-", label=tname)
        for ax, m in zip(axes, ("NDCG@10", "Recall@10")):
            ax.set_xlabel("rollback budget B")
            ax.set_ylabel(m)
            ax.set_xscale("symlog", linthresh=1)
        axes[0].legend(fontsize=7)
        fig.suptitle("recommendation quality vs hard rollback budget")
        _save(ctx, fig, "plot_5_quality_vs_budget.png")

    # 6-7: exact deletes / same-token reselection by SID level
    to_users = exact_df[(exact_df["family"] == "truncated_oracle") & (exact_df["status"] == "ok")]
    lvl_cols = [c for c in to_users.columns if c.startswith("deletes_L")]
    if len(to_users) and lvl_cols:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        big_b = to_users["budget"].max()
        at_b = to_users[to_users["budget"] == big_b]
        means = at_b.groupby("transform")[lvl_cols].mean()
        means.T.plot.bar(ax=axes[0], legend=True)
        axes[0].set_title(f"E[DELETE] by level (B={big_b:g})")
        rsel = [c.replace("deletes", "reselect") for c in lvl_cols]
        (at_b.groupby("transform")[rsel].mean()
             .div(at_b.groupby("transform")[lvl_cols].mean().values.clip(min=1e-12))
             .T.plot.bar(ax=axes[1], legend=False))
        axes[1].set_title("same-token reselection prob per DELETE, by level")
        for ax in axes:
            ax.tick_params(axis="x", rotation=0, labelsize=8)
        _save(ctx, fig, "plot_6_7_deletes_and_reselect_by_level.png")

    # 8: candidate recall vs compute (serving)
    if len(serving_df):
        m = ctx.args.serving_candidates[0]
        col = f"candidate_recall@{m}"
        sub = serving_df[serving_df["status"] == "ok"].groupby(
            CONFIG_COLS, dropna=False)[[col, "decoder_evals"]].mean().reset_index()
        if len(sub):
            fig, ax = plt.subplots()
            for fam, fsub in sub.groupby("family"):
                fsub = fsub.sort_values("decoder_evals")
                ax.plot(fsub["decoder_evals"], fsub[col], "o-", ms=4,
                        color=FAMILY_COLORS.get(fam, "k"), label=fam)
            ax.set_xlabel("TIGER decoder evaluations (total for the sample budget)")
            ax.set_ylabel(col)
            ax.legend(fontsize=7)
            ax.set_title("candidate recall vs compute (serving-style)")
            _save(ctx, fig, "plot_8_candidate_recall_vs_compute.png")

    # 9: per-user improvement colored by slice
    to_all = exact_df[(exact_df["family"] == "truncated_oracle") & (exact_df["status"] == "ok")]
    if len(to_all):
        best_b = (to_all.groupby("budget")["ndcg@10"].mean().idxmax())
        base = exact_df[(exact_df["policy"] == "append_only")]
        b0 = base.set_index(["user", "transform"])["ndcg@10"]
        bb = to_all[to_all["budget"] == best_b].set_index(["user", "transform"])["ndcg@10"]
        both = pd.DataFrame({"base": b0, "budget": bb}).dropna().reset_index()
        both = both.merge(slices_df[["user", "user_slice"]], on="user", how="left")
        fig, ax = plt.subplots()
        colors = {"sasrec_better": "#2ca02c", "tiger_better": "#d62728",
                  "approximately_equal": "#7f7f7f"}
        for sl, ssub in both.groupby("user_slice"):
            ax.scatter(ssub["base"], ssub["budget"] - ssub["base"], s=12, alpha=0.6,
                       color=colors.get(sl, "k"), label=sl)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("append-only NDCG@10")
        ax.set_ylabel(f"Delta NDCG@10 (truncated oracle B={best_b:g})")
        ax.legend(fontsize=7)
        ax.set_title("per-user improvement by initial model advantage (H5)")
        _save(ctx, fig, "plot_9_per_user_improvement.png")

    # 10: target entropy / effective support vs correction cost
    tsub = transform_df.dropna(subset=["E_del_inf"]) if "E_del_inf" in transform_df else pd.DataFrame()
    if len(tsub):
        fig, ax = plt.subplots()
        for tname, ssub in tsub.groupby("transform"):
            ax.scatter(ssub["target_entropy"], ssub["E_del_inf"], s=14, alpha=0.6, label=tname)
        ax.set_xlabel("target entropy H(p_R) [nats]")
        ax.set_ylabel("exact E[N_DELETE] of the infinite-budget oracle")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.set_title("reward sharpness vs exact correction cost")
        _save(ctx, fig, "plot_10_entropy_vs_cost.png")


# --------------------------------------------------------------------------- #
# Report (spec 20)
# --------------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_report(ctx, agg, by_slice, level_df, transform_df, serving_df, selected, slices_df) -> None:
    to = agg[agg["family"] == "truncated_oracle"].dropna(subset=["budget"]).sort_values("budget")
    base = agg[agg["policy"] == "append_only"]
    base_ndcg = float(base["ndcg@10"].mean()) if len(base) else np.nan
    lines = [
        "# Budgeted cyclic GFlowNet sampling — Amazon Beauty 2014 (validation)",
        "",
        f"Frozen exp2 TIGER + SASRec; {agg['n_users'].max() if len(agg) else 0} evaluated users; "
        f"generated {time.strftime('%Y-%m-%d')} by scripts/exp5-budgeted-cyclic/budgeted_cyclic_amazon.py.",
        "",
        "## Does a small rollback budget improve final recommendation metrics?",
    ]
    if len(to):
        best = to.loc[to["ndcg@10"].idxmax()]
        gain = best["ndcg@10"] - base_ndcg
        lines += [
            f"Append-only NDCG@10 = {_fmt(base_ndcg)}. Best truncated-oracle budget: "
            f"B={best['budget']:g} ({best['transform']}) with NDCG@10 = {_fmt(best['ndcg@10'])} "
            f"({'+' if gain >= 0 else ''}{_fmt(gain)}), at E[n] = {_fmt(best['expected_steps'], 2)} "
            f"steps and E[deletes] = {_fmt(best['expected_deletes'], 2)}.",
            "",
            "Per-budget NDCG@10 (mean over transforms): " + ", ".join(
                f"B={b:g}: {_fmt(v)}" for b, v in to.groupby("budget")["ndcg@10"].mean().items()),
        ]
    lines += ["", "## What budget gives the best quality-compute trade-off?",
              "See plots/plot_2_ndcg10_vs_length.png (Pareto frontier) and "
              "summary/matched_compute_comparison.csv.",
              "", "## On which user slices does rollback help?"]
    if len(by_slice):
        sl = by_slice[(by_slice["family"] == "truncated_oracle")]
        for name, ssub in sl.groupby("user_slice"):
            b0 = by_slice[(by_slice["policy"] == "append_only")
                          & (by_slice["user_slice"] == name)]["ndcg@10"].mean()
            bb = ssub["ndcg@10"].max()
            lines.append(f"- {name}: append-only {_fmt(b0)} -> best budgeted {_fmt(bb)}")
    lines += ["", "## At which SID levels is the budget spent?"]
    lvl_cols = [c for c in agg.columns if c.startswith("deletes_L")]
    if len(to) and lvl_cols:
        big = to.loc[to["budget"].idxmax()]
        lines.append("At the largest budget, exact E[DELETE] by level: " + ", ".join(
            f"L{i + 1}={_fmt(big[c], 3)}" for i, c in enumerate(lvl_cols)))
    lines += ["", "## How much budget is wasted on immediate reselection?"]
    if len(to):
        lines.append(f"Expected same-token reselections per trajectory at the best budget: "
                     f"{_fmt(to.loc[to['ndcg@10'].idxmax()]['same_token_reselect'], 3)} "
                     f"(see plot_6_7 and the no_repeat rows in serving_metrics.csv).")
    lines += ["", "## Does cyclic rollback beat simpler matched-compute baselines?",
              "Selected best per family (validation NDCG@10):"]
    for fam, cfg in selected.items():
        lines.append(f"- {fam}: {config_label(cfg)} -> NDCG@10 {_fmt(cfg.get('ndcg@10'))} "
                     f"at {_fmt(cfg.get('decoder_evals'), 1)} decoder evals")
    lines += ["", "## How sensitive is the result to SASRec reward temperature?",
              "summary/reward_transform_comparison.csv:"]
    for _, r in transform_df.groupby("transform").mean(numeric_only=True).reset_index().iterrows():
        lines.append(f"- {r['transform']}: H(p_R) = {_fmt(r['target_entropy'], 2)} nats, "
                     f"effective support {_fmt(r.get('effective_support'), 1)}, "
                     f"exact E[N_DELETE]_inf = {_fmt(r.get('E_del_inf'), 1)}")
    lines += [
        "", "## Is there sufficient evidence for a learned rollback head?",
        "Decide from the gap between the truncated oracle and the scaled/level-masked variants: "
        "a positive small-budget oracle gain that truncation does not capture justifies a "
        "learned budget-aware controller (per-state d* targets are reproducible from the "
        "cached prefix tables + reward caches).",
        "", "## Interpretation constraints",
        "- Divergence to p_R improving without Recall/NDCG improving indicts the reward target, "
        "not the cyclic mechanism.",
        "- Best-of-N or direct guidance dominating at matched compute means cycles are not "
        "serving-optimal here; report as such.",
        "- All hyperparameters above were chosen on validation; test numbers (stage F) are in "
        "summary/test_metrics.csv.",
    ]
    (ctx.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {ctx.out_dir / 'report.md'}", flush=True)


# --------------------------------------------------------------------------- #
# Stage F: test evaluation of the selected configurations
# --------------------------------------------------------------------------- #
def grids_from_selected(selected: Mapping[str, Mapping[str, Any]], args) -> Grids:
    g = Grids()
    for fam, cfg in selected.items():
        if fam == "truncated_oracle" and cfg.get("budget") is not None:
            g.budgets = sorted({0, int(cfg["budget"])})
            g.serving_budgets = [int(cfg["budget"])]
        elif fam == "scaled_oracle":
            if cfg.get("lam") is not None:
                g.lambdas = [float(cfg["lam"])]
                g.scaled_budgets = [int(cfg["budget"])]
        elif fam == "direct_guidance" and cfg.get("alpha") is not None:
            g.alphas = [float(cfg["alpha"])]
        elif fam == "stochastic_tiger" and cfg.get("tau") is not None:
            g.taus = [float(cfg["tau"])]
        elif fam == "best_of_n" and cfg.get("n_cand") is not None:
            g.best_of_n = [int(cfg["n_cand"])]
        elif fam in ("beam", "beam_rerank") and cfg.get("n_cand") is not None:
            g.beams = sorted(set(g.beams) | {int(cfg["n_cand"])})
    if not g.budgets:
        g.budgets = [0] + [b for b in args.budgets if b > 0][:1]
    return g


def stage_f(ctx: ExperimentContext, variants: list[RewardVariant]) -> None:
    args = ctx.args
    print("== Stage F: test evaluation of selected configurations ==", flush=True)
    sel_path = ctx.summary_dir / "selected_configs.json"
    if not sel_path.exists():
        raise RuntimeError("run stage E first: summary/selected_configs.json missing")
    selected = json.loads(sel_path.read_text())
    grids = grids_from_selected(selected, args)
    # restrict transforms to the selected ones
    used = {cfg.get("transform") for cfg in selected.values() if cfg.get("transform")}
    variants = [v for v in variants if v.name in used] or variants

    n_users = args.num_test_users or args.num_users
    indices = select_users(ctx, "test", n_users)
    save_meta_users(ctx, "test", indices)

    exact_rows, serving_rows = [], []
    for pos, idx in enumerate(indices):
        uc, batch = build_user_context(ctx, "test", idx)
        tables = load_or_compute_prefix_tables(ctx, uc, batch)
        bundles = [build_target_bundle(ctx, uc, tables, v) for v in variants]
        rows, _, _, _, _ = compute_exact_rows(ctx, uc, tables, bundles, grids)
        exact_rows.extend(rows)
        serving_rows.extend(compute_serving_rows(ctx, uc, tables, bundles, grids))
        if (pos + 1) % 25 == 0:
            print(f"  test user {pos + 1}/{len(indices)}", flush=True)

    df = pd.DataFrame(exact_rows)
    (ctx.out_dir / "per_user").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ctx.out_dir / "per_user" / "budget_metrics_test.parquet", index=False)
    agg = aggregate_exact(df)
    agg = merge_on_config(agg, coverage_table(df, ctx.trie.num_items))
    write_csv(agg, ctx.summary_dir / "test_metrics.csv")
    if serving_rows:
        sdf = pd.DataFrame(serving_rows)
        num_cols = [c for c in sdf.select_dtypes(include=[np.number]).columns
                    if c not in CONFIG_COLS]
        write_csv(sdf.groupby(CONFIG_COLS, dropna=False)[num_cols].mean().reset_index(),
                  ctx.summary_dir / "test_serving_metrics.csv")

    # paired bootstrap vs append-only, per selected config
    ok = df[df["status"] == "ok"].copy()
    base = ok[ok["policy"] == "append_only"].set_index(["user", "transform"])
    boot_rows = []
    for key, sub in ok[ok["policy"] != "append_only"].groupby(CONFIG_COLS, dropna=False):
        joined = sub.set_index(["user", "transform"]).join(
            base[["ndcg@10", "recall@10"]], rsuffix="_base").dropna(
            subset=["ndcg@10", "ndcg@10_base"])
        if len(joined) < 5:
            continue
        for metric in ("ndcg@10", "recall@10"):
            deltas = (joined[metric] - joined[f"{metric}_base"]).to_numpy()
            row = dict(zip(CONFIG_COLS, key))
            row["metric"] = metric
            row.update(paired_bootstrap(deltas, args.bootstrap_samples, args.seed))
            boot_rows.append(row)
    if boot_rows:
        bdf = pd.DataFrame(boot_rows)
        write_csv(bdf, ctx.summary_dir / "test_bootstrap.csv")
        append_test_section_to_report(ctx, bdf)
    assert_still_frozen(ctx.tiger, ctx.tiger_fingerprint)
    assert_still_frozen(ctx.sasrec, ctx.sasrec_fingerprint)


def append_test_section_to_report(ctx: ExperimentContext, boot: pd.DataFrame) -> None:
    """Summarize the paired-bootstrap test outcome at the end of report.md."""
    report = ctx.out_dir / "report.md"
    if not report.exists():
        return
    text = report.read_text()
    marker = "\n## Test evaluation (stage F"
    if marker in text:
        text = text[: text.index(marker)]
    nd = boot[boot["metric"] == "ndcg@10"].sort_values("mean_delta", ascending=False)
    lines = [f"\n## Test evaluation (stage F, {int(nd['n_users'].max())} users, "
             "paired bootstrap vs append-only, NDCG@10)", ""]
    for _, r in nd.head(10).iterrows():
        sig = "SIGNIFICANT" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "n.s."
        lines.append(
            f"- {r['transform']} {config_label(r)}: delta = {r['mean_delta']:+.4f} "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] (p_improved {r['p_improved']:.2f}) {sig}")
    report.write_text(text + "\n".join(lines) + "\n")
    print("  appended test section to report.md", flush=True)


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def save_meta_users(ctx: ExperimentContext, split: str, indices: list[int]) -> None:
    meta = []
    for idx in indices:
        batch = single_user_batch(ctx, split, idx)
        meta.append({
            "raw_user_id": str(ctx.dm.id2user[int(batch["user_ids"][0])]),
            "dataset_index": int(idx),
            "history_len": int(batch["history_mask"][0].sum()),
            "target_item_id": int(batch["target_item_ids"][0]),
        })
    (ctx.out_dir / f"selected_users_{split}.json").write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = set(args.stages.upper())
    torch.manual_seed(args.seed)
    ctx = load_experiment_context(args)
    variants = reward_variants(args)
    print(f"reward transforms: {[v.name for v in variants]}", flush=True)

    if stages & set("ABCDE"):
        indices = select_users(ctx, "validation", args.num_users)
        save_meta_users(ctx, "validation", indices)

        # per-user assets (cached); heavy part = one TIGER prefix sweep per user
        users: list[tuple[UserContext, TigerPrefixTables]] = []
        for pos, idx in enumerate(indices):
            uc, batch = build_user_context(ctx, "validation", idx)
            tables = load_or_compute_prefix_tables(ctx, uc, batch)
            users.append((uc, tables))
            if (pos + 1) % 25 == 0:
                print(f"  prepared user {pos + 1}/{len(indices)}", flush=True)

        if "A" in stages:
            slices_df = stage_a(ctx, users)
        else:
            slices_path = ctx.summary_dir / "model_comparison_validation.csv"
            slices_df = (pd.read_csv(slices_path) if slices_path.exists() else
                         pd.DataFrame({"user": [uc.uid for uc, _ in users],
                                       "user_slice": "unknown"}))

        if "B" in stages:
            stage_b(ctx, users[: args.num_users_regression], variants)
            if args.optimize_budget_oracle:
                run_budget_optimal_probe(ctx, users[: min(5, len(users))], variants)

        exact_rows: list[dict] = []
        transform_rows: list[dict] = []
        level_rows: list[dict] = []
        oracle_level_rows: list[dict] = []
        serving_rows: list[dict] = []
        if stages & set("CD"):
            grids = Grids(
                budgets=list(args.budgets) if "C" in stages else [],
                lambdas=list(args.lambda_grid) if "D" in stages else [],
                scaled_budgets=list(args.scaled_budgets),
                alphas=list(args.alpha_grid) if "D" in stages else [],
                taus=list(args.decoder_temperatures) if "D" in stages else [],
                best_of_n=list(args.best_of_n) if "D" in stages else [],
                beams=list(args.beam_sizes) if "D" in stages else [],
                level_ablation_budget=args.level_ablation_budget if "C" in stages else None,
                serving_budgets=[b for b in args.budgets if b > 0] if "D" in stages else [],
            )
            print(f"== Stage C/D: exact + sampled evaluation over {len(users)} users ==", flush=True)
            t0 = time.time()
            for pos, (uc, tables) in enumerate(users):
                bundles = [build_target_bundle(ctx, uc, tables, v) for v in variants]
                rows, trows, lrows, orows, dists = compute_exact_rows(ctx, uc, tables, bundles, grids)
                exact_rows.extend(rows)
                transform_rows.extend(trows)
                level_rows.extend(lrows)
                oracle_level_rows.extend(orows)
                if pos < args.save_distributions_users and dists:
                    np.savez_compressed(
                        ctx.per_user_dir("validation", uc.uid) / "exact_distributions.npz",
                        budgets=np.asarray(grids.budgets), **dists)
                if "D" in stages:
                    serving_rows.extend(compute_serving_rows(ctx, uc, tables, bundles, grids))
                if (pos + 1) % 25 == 0:
                    rate = (pos + 1) / (time.time() - t0)
                    print(f"  user {pos + 1}/{len(users)} ({rate:.2f} users/s)", flush=True)

            exact_df = pd.DataFrame(exact_rows)
            pu = ctx.out_dir / "per_user"
            pu.mkdir(parents=True, exist_ok=True)
            exact_df.to_parquet(pu / "budget_metrics_val.parquet", index=False)
            pd.DataFrame(transform_rows).to_parquet(pu / "reward_transforms_val.parquet", index=False)
            if level_rows:
                pd.DataFrame(level_rows).to_parquet(pu / "level_ablation_val.parquet", index=False)
            if oracle_level_rows:
                pd.DataFrame(oracle_level_rows).to_parquet(pu / "oracle_by_level_val.parquet", index=False)
            if serving_rows:
                pd.DataFrame(serving_rows).to_parquet(pu / "serving_val.parquet", index=False)

        if "E" in stages:
            pu = ctx.out_dir / "per_user"
            exact_df = pd.DataFrame(exact_rows) if exact_rows else pd.read_parquet(
                pu / "budget_metrics_val.parquet")
            transform_df = pd.DataFrame(transform_rows) if transform_rows else pd.read_parquet(
                pu / "reward_transforms_val.parquet")
            level_df = (pd.DataFrame(level_rows) if level_rows else
                        (pd.read_parquet(pu / "level_ablation_val.parquet")
                         if (pu / "level_ablation_val.parquet").exists() else pd.DataFrame()))
            oracle_level_df = (pd.DataFrame(oracle_level_rows) if oracle_level_rows else
                               (pd.read_parquet(pu / "oracle_by_level_val.parquet")
                                if (pu / "oracle_by_level_val.parquet").exists() else pd.DataFrame()))
            serving_df = (pd.DataFrame(serving_rows) if serving_rows else
                          (pd.read_parquet(pu / "serving_val.parquet")
                           if (pu / "serving_val.parquet").exists() else pd.DataFrame()))
            stage_e(ctx, exact_df, level_df, oracle_level_df, transform_df, serving_df, slices_df)

        assert_still_frozen(ctx.tiger, ctx.tiger_fingerprint)
        assert_still_frozen(ctx.sasrec, ctx.sasrec_fingerprint)

    if "F" in stages:
        stage_f(ctx, variants)
    print("done.", flush=True)


def run_budget_optimal_probe(ctx: ExperimentContext, users, variants) -> None:
    """Phase-2 (spec 9.3): validation-only budget-optimal rollback upper bound."""
    from flowrec.trajectories.budgeted import optimize_budgeted_rollback

    rows = []
    budget = ctx.args.level_ablation_budget
    for uc, tables in users:
        for variant in variants[:1]:
            bundle = build_target_bundle(ctx, uc, tables, variant)
            if bundle.oracle is None:
                continue
            base = exact_budgeted_terminal_distribution(
                ctx.trie, tables.edge_prob, bundle.d_star, budget)
            out = optimize_budgeted_rollback(
                ctx.trie, tables.edge_prob, bundle.p_r, budget=budget,
                expected_deletes_cap=float(base.expected_deletes),
                init_rollback=np.clip(bundle.d_star, 1e-6, 1 - 1e-6),
                steps=150, lr=0.05, seed=ctx.args.seed,
            )
            opt = exact_budgeted_terminal_distribution(
                ctx.trie, tables.edge_prob, out.rollback_prob, budget)
            rows.append({
                "user": uc.uid, "transform": variant.name, "budget": budget,
                "js_truncated": js_divergence(base.terminal, bundle.p_r),
                "js_optimized": js_divergence(opt.terminal, bundle.p_r),
                "ndcg10_truncated": metrics_from_rank(
                    rank_of_target(base.terminal, uc.target_row), KS)["ndcg@10"],
                "ndcg10_optimized": metrics_from_rank(
                    rank_of_target(opt.terminal, uc.target_row), KS)["ndcg@10"],
                "E_del_truncated": base.expected_deletes,
                "E_del_optimized": opt.expected_deletes,
            })
    if rows:
        write_csv(pd.DataFrame(rows), ctx.summary_dir / "budget_optimal_probe.csv")


if __name__ == "__main__":
    main()
