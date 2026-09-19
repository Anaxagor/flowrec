#!/usr/bin/env python
"""Sanity check: factorized reward caches == dense full pre-calculation.

Builds, for a few Beauty (2014) train examples, BOTH the dense float32 cache
(scripts/precompute_reward_cache.py) and the factorized cache
(scripts/precompute_reward_factors.py) of every concrete reward component the
GPT-2 Amazon GFlow grid uses (sasrec, interaction, sim_hier, sim_embed), from
the very same single-component configs that grid writes, then compares what
GFlowMethod would see at training time:

  * per component and power in --powers: full rows, the selected candidate
    rewards (random catalog rows plus -1 = target-row substitution) and the
    full-catalog reward sum (logZ), dense _PrecomputedRewardCache vs
    FactorizedRewardCache;
  * the grid's composed rewards (multiply and mixed, unequal weights, every
    power): the script-local ComposedRewardCache over dense caches vs
    ComposedFactorizedRewardCache over factorized caches.

Both hidden-state storage dtypes are checked: float32 (exact reconstruction,
tight tolerance) and float16 (the production default; differences are the
float16 rounding of the 1-vector-per-example SASRec state).

    python scripts/verify_factorized_reward_cache.py --device cpu --num-workers 0

Exit status 1 when any comparison exceeds its tolerance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR), str(SCRIPTS_DIR / "exp2-tuned-params"),
           str(SCRIPTS_DIR / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_gpt2_amazon as g  # noqa: E402
from flowrec.ft_methods.gflow.factorized_cache import (  # noqa: E402
    ComposedFactorizedRewardCache,
    FactorizedRewardCache,
)
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache  # noqa: E402

COMPONENTS = ("sasrec", "interaction", "sim_hier", "sim_embed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Verify factorized reward caches against dense precomputation.")
    p.add_argument("--dataset", default="beauty", choices=list(g.DATASETS))
    p.add_argument("--num-examples", type=int, default=24)
    p.add_argument("--train-last-n", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--powers", type=float, nargs="+", default=[0.25, 1.0, 4.0, 16.0])
    p.add_argument("--num-candidates", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "factorized_reward_verification")
    p.add_argument("--rtol-f32", type=float, default=1e-4)
    p.add_argument("--rtol-f16", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--reuse", action="store_true", help="Skip builds whose manifest exists")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def build_caches(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    grid_args = g.parse_args([
        "--datasets", args.dataset, "--smoke", "--device", args.device,
        "--num-workers", str(args.num_workers), "--train-last-n", str(args.train_last_n),
        "--similarity-modes", "hierarchical", "embedding_cosine",
    ])
    run_config = g.build_run_config(args.dataset, grid_args, user_ids_count=1)
    config_dir = args.work_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Path]] = {}
    common = [
        "--split", "train", "--per-user-last-n", str(args.train_last_n),
        "--batch-size", str(args.cache_batch_size), "--num-workers", str(args.num_workers),
        "--device", args.device, "--max-examples", str(args.num_examples),
    ]
    for key in COMPONENTS:
        config_path = config_dir / f"cache_config_{key}.json"
        g.write_json(config_path, g.cache_config(run_config, key))
        dense_dir = args.work_dir / "dense" / key
        f32_dir = args.work_dir / "factorized_f32" / key
        f16_dir = args.work_dir / "factorized_f16" / key
        if not (args.reuse and (dense_dir / "manifest.json").exists()):
            run([sys.executable, str(SCRIPTS_DIR / "precompute_reward_cache.py"),
                 "--config", str(config_path), "--output-dir", str(dense_dir), "--dtype", "float32",
                 "--store-power-sums", *[format(p, "g") for p in args.powers], *common])
        for hidden_dtype, target in (("float32", f32_dir), ("float16", f16_dir)):
            if key != "sasrec" and hidden_dtype == "float16":
                continue  # only sasrec has a per-example float factor
            if not (args.reuse and (target / "manifest.json").exists()):
                run([sys.executable, str(SCRIPTS_DIR / "precompute_reward_factors.py"),
                     "--config", str(config_path), "--output-dir", str(target),
                     "--hidden-dtype", hidden_dtype, "--verify-batches", "1", *common])
        out[key] = {"dense": dense_dir, "f32": f32_dir, "f16": f16_dir if key == "sasrec" else f32_dir}
    return out


def compare(name: str, a: torch.Tensor, b: torch.Tensor, *, rtol: float, atol: float, rows: list[dict]) -> bool:
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-12)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0
    ok = bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))
    rows.append({"check": name, "shape": list(a.shape), "max_abs_diff": max_abs,
                 "max_rel_diff": max_rel, "rtol": rtol, "ok": ok})
    return ok


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    dirs = build_caches(args)

    dense_meta = json.loads((dirs["sasrec"]["dense"] / "manifest.json").read_text())
    num_examples = int(dense_meta["num_examples"])
    num_items = int(dense_meta["num_catalog_items"])
    print(f"[verify] {num_examples} examples x {num_items} catalog items", flush=True)

    # Random lookups the trainer would issue: shuffled example indices and
    # candidate rows, with -1 (the sampler's positive) in the first column.
    gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_examples, generator=gen)[: min(num_examples, 16)]
    cand_rows = torch.randint(0, num_items, (indices.numel(), args.num_candidates), generator=gen)
    cand_rows[:, 0] = -1

    rows: list[dict] = []
    all_ok = True
    for key in COMPONENTS:
        for variant in ("f32", "f16"):
            if variant == "f16" and key != "sasrec":
                continue
            rtol = args.rtol_f32 if variant == "f32" else args.rtol_f16
            for power in args.powers:
                dense = _PrecomputedRewardCache(dirs[key]["dense"], power=power)
                fact = FactorizedRewardCache(dirs[key][variant], power=power)
                d_sel, d_sum, d_full = dense.select_candidates(indices, cand_rows, device, include_full=True)
                f_sel, f_sum, f_full = fact.select_candidates(indices, cand_rows, device, include_full=True)
                tag = f"{key}[{variant}] power={power:g}"
                all_ok &= compare(f"{tag} full rows", f_full, d_full, rtol=rtol, atol=args.atol, rows=rows)
                all_ok &= compare(f"{tag} selected", f_sel, d_sel, rtol=rtol, atol=args.atol, rows=rows)
                all_ok &= compare(f"{tag} reward_sum (logZ)", f_sum, d_sum, rtol=rtol, atol=args.atol, rows=rows)
                all_ok &= compare(f"{tag} logZ", f_sum.clamp_min(1e-6).log(), d_sum.clamp_min(1e-6).log(),
                                  rtol=rtol, atol=1e-5, rows=rows)

    weights = {"sasrec": 1.0, "interaction": 2.0, "sim_hier": 1.0, "sim_embed": 2.0}
    for composition in ("multiply", "mixed"):
        for power in args.powers:
            for variant in ("f32", "f16"):
                rtol = args.rtol_f32 if variant == "f32" else args.rtol_f16
                dense = g.ComposedRewardCache([(k, dirs[k]["dense"], weights[k]) for k in COMPONENTS],
                                              composition=composition, power=power)
                fact = ComposedFactorizedRewardCache([(k, dirs[k][variant], weights[k]) for k in COMPONENTS],
                                                     composition=composition, power=power)
                d_sel, d_sum, d_full = dense.select_candidates(indices, cand_rows, device, include_full=True)
                f_sel, f_sum, f_full = fact.select_candidates(indices, cand_rows, device, include_full=True)
                tag = f"composed {composition}[{variant}] power={power:g}"
                all_ok &= compare(f"{tag} full rows", f_full, d_full, rtol=rtol, atol=args.atol, rows=rows)
                all_ok &= compare(f"{tag} selected", f_sel, d_sel, rtol=rtol, atol=args.atol, rows=rows)
                all_ok &= compare(f"{tag} logZ", f_sum.clamp_min(1e-6).log(), d_sum.clamp_min(1e-6).log(),
                                  rtol=rtol, atol=1e-5, rows=rows)

    sizes = {}
    for key in COMPONENTS:
        dense_bytes = sum(p.stat().st_size for p in dirs[key]["dense"].glob("shard_*.pt"))
        fact_meta = json.loads((dirs[key]["f16"] / "manifest.json").read_text())
        sizes[key] = {
            "dense_float32_bytes": dense_bytes,
            "dense_float16_bytes_equiv": int(fact_meta["dense_row_bytes_float16"]),
            "factor_bytes": int(fact_meta["factor_bytes"]),
            "builder_self_check": fact_meta.get("verification"),
            "sparse_entries_per_example": fact_meta.get("mean_sparse_entries_per_example"),
        }

    report = {
        "dataset": args.dataset, "num_examples": num_examples, "num_catalog_items": num_items,
        "powers": args.powers, "checks": rows, "sizes": sizes, "all_ok": all_ok,
        "elapsed_sec": time.time() - t0,
    }
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2))

    width = max(len(r["check"]) for r in rows)
    print(f"\n{'check'.ljust(width)}  max_abs_diff  max_rel_diff  ok")
    for r in rows:
        print(f"{r['check'].ljust(width)}  {r['max_abs_diff']:.3e}     {r['max_rel_diff']:.3e}     {'OK' if r['ok'] else 'FAIL'}")
    print("\nsizes (this subset):")
    for key, s in sizes.items():
        print(f"  {key:12s} dense f16-equivalent {s['dense_float16_bytes_equiv']/1e6:9.2f} MB  "
              f"factors {s['factor_bytes']/1e6:9.2f} MB  builder self-check {s['builder_self_check']}")
    print(f"\nALL OK: {all_ok}  (report: {args.work_dir / 'report.json'})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
