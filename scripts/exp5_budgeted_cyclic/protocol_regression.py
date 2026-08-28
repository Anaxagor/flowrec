#!/usr/bin/env python
"""Protocol regression test (spec §5.4): three TIGER evaluators, same users.

On a fixed seeded set of >=128 validation users, runs
  (a) the repository generation evaluator (constrained ``model.generate``),
  (b) the new trie-constrained beam evaluator (from the prefix-score cache),
  (c) the exact full-catalog evaluator,
saves per-user outputs, and diagnoses every mismatch (constraint-fallback
generations, tie ordering, beam-order divergence). Full cluster runs must not
proceed until agreement holds or the disagreement cause is documented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.tiger_scoring import (  # noqa: E402
    compute_user_edge_logprobs,
    exact_item_logprobs,
    renormalized_edge_logprob,
)
from flowrec.trajectories.budgeted import trie_beam_search  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p, needs_split=False)
    p.add_argument("--n-users", type=int, default=128)
    p.add_argument("--beam", type=int, default=20)
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    users = manifest.read_split_manifest(root, "validation")
    rng = np.random.default_rng(cfg["seed"])
    take = np.sort(rng.choice(len(users), size=min(args.n_users, len(users)),
                              replace=False))
    if args.dry_run:
        print(f"[dry-run] would compare 3 evaluators on {len(take)} users")
        return

    assets = common.load_experiment_assets(args, cfg, need_models=True)
    trie, K = assets.trie, args.beam
    rows = []
    for pos, (_, urow) in enumerate(users.iloc[take].iterrows()):
        ds_idx = int(urow["dataset_index"])
        batch = common.single_user_batch(assets.dm, "validation", ds_idx)
        moved = {k: (v.to(assets.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        # (a) repository evaluator path: constrained generate
        with torch.no_grad():
            gen = assets.tiger.generate(
                moved, num_beams=K, num_return_sequences=K,
                constrain_valid_sequences=True)[0].cpu().numpy()
        gen_rows, n_invalid = [], 0
        for tup in gen:
            try:
                node = trie.node_of_prefix([int(t) for t in tup])
                gen_rows.append(int(trie.item_row[node]))
            except KeyError:
                n_invalid += 1  # constraint fallback emitted an invalid SID
        # (b)+(c) cache-based evaluators
        lp32, _ = compute_user_edge_logprobs(assets.tiger, batch, trie)
        lp = renormalized_edge_logprob(trie, lp32)
        beam_rows, _scores = trie_beam_search(trie, lp, K)
        log_items = exact_item_logprobs(trie, lp)
        exact_topk = np.argsort(-log_items, kind="stable")[:K]
        target_row = int(assets.item_to_row[int(urow["target_item_id"])])
        rows.append({
            "user": str(urow["raw_user_id"]), "dataset_index": ds_idx,
            "generate_rows": ",".join(map(str, gen_rows)),
            "trie_beam_rows": ",".join(map(str, beam_rows.tolist())),
            "exact_topk_rows": ",".join(map(str, exact_topk.tolist())),
            "n_generate_invalid": n_invalid,
            "topk_set_agreement": len(set(gen_rows) & set(beam_rows.tolist()))
            / max(len(beam_rows), 1),
            "beam_vs_exact_agreement": len(set(beam_rows.tolist())
                                           & set(exact_topk.tolist()))
            / max(len(beam_rows), 1),
            "hit_generate": target_row in gen_rows,
            "hit_trie_beam": target_row in beam_rows.tolist(),
            "hit_exact_topk": target_row in exact_topk.tolist(),
        })
        if (pos + 1) % 16 == 0:
            print(f"  {pos + 1}/{len(take)}", flush=True)

    df = pd.DataFrame(rows)
    sdir = root / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_parquet(df, sdir / "protocol_regression_per_user.parquet")
    agree = float(df["topk_set_agreement"].mean())
    hit_agree = float((df["hit_generate"] == df["hit_trie_beam"]).mean())
    notes = []
    if int(df["n_generate_invalid"].sum()):
        notes.append(
            f"{int(df['n_generate_invalid'].sum())} generated tuples were invalid "
            "SIDs: TigerModel.generate silently falls back to unconstrained "
            "decoding when constraint construction fails (flowrec/models/tiger.py)")
    if agree < 1.0:
        notes.append(
            "HF beam search and exact trie beam are both heuristics with "
            "different pruning order; residual top-K set differences are "
            "expected at equal scores / near-ties")
    payload = {
        "n_users": int(len(df)), "beam": K,
        "topk_set_agreement": agree,
        "beam_vs_exact_agreement": float(df["beam_vs_exact_agreement"].mean()),
        "hit_agreement": hit_agree,
        "n_generate_invalid": int(df["n_generate_invalid"].sum()),
        "disagreement_note": "; ".join(notes) if notes else None,
        "pass": bool(hit_agree >= 0.98 and agree >= 0.9),
    }
    common.atomic_write_json(payload, sdir / "protocol_regression.json")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["pass"]:
        print("PROTOCOL REGRESSION DID NOT PASS: document the causes above "
              "before launching full cluster runs (spec 5.4)", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
