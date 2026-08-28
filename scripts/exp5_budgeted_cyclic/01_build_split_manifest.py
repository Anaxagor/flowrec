#!/usr/bin/env python
"""Stage 01: full-universe split manifests (spec §4).

Loads EVERY user of each leave-one-out split directly from the dataset (no
fixed user lists), records one row per user with eligibility and exclusion
reasons, asserts the expected user count, and verifies the leave-one-out
protocol (test context = training history + validation item) on a sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402


def build_manifest(assets, split: str, *, batch_size: int = 1024) -> pd.DataFrame:
    dm = assets.dm
    ds = dm.val_dataset if split == "validation" else dm.test_dataset
    n = len(ds.examples) if hasattr(ds, "examples") else len(ds)
    rows = []
    for chunk, batch in common.user_batches(dm, split, list(range(n)), batch_size):
        uids = batch["user_ids"].numpy()
        targets = batch["target_item_ids"].numpy()
        hist = batch["history_item_ids"].numpy()
        mask = batch["history_mask"].numpy()
        for k, ds_idx in enumerate(chunk):
            target = int(targets[k])
            hlen = int(mask[k].sum())
            in_cat = 1 <= target < dm.item_vocab_size
            has_sid = bool(in_cat and assets.item_to_row[target] >= 0)
            reasons = []
            if hlen < 1:
                reasons.append("empty_history")
            if not in_cat:
                reasons.append("target_not_in_catalog")
            if not has_sid:
                reasons.append("target_has_no_sid")
            rows.append({
                "raw_user_id": str(dm.id2user[int(uids[k])]),
                "internal_user_id": int(uids[k]),
                "dataset_index": int(ds_idx),
                "split": split,
                "history_item_ids": [int(x) for x in hist[k][mask[k].astype(bool)]],
                "history_length": hlen,
                "target_item_id": target,
                "target_in_catalog": bool(in_cat),
                "target_has_sid": has_sid,
                "eligible": not reasons,
                "exclusion_reason": ";".join(reasons) if reasons else None,
            })
        if (chunk[-1] + 1) % (batch_size * 8) < batch_size:
            print(f"  {split}: {chunk[-1] + 1}/{n}", flush=True)
    return pd.DataFrame(rows)


def check_leave_one_out(val_df: pd.DataFrame, test_df: pd.DataFrame, max_len: int) -> None:
    """Test context must equal training history + validation target (windowed)."""
    joined = val_df.merge(test_df, on="raw_user_id", suffixes=("_v", "_t"))
    sample = joined.sample(min(64, len(joined)), random_state=0)
    for _, r in sample.iterrows():
        want = (list(r["history_item_ids_v"]) + [int(r["target_item_id_v"])])[-max_len:]
        got = list(r["history_item_ids_t"])
        if got != want:
            raise AssertionError(
                f"leave-one-out violation for user {r['raw_user_id']}: test "
                f"history {got} != train-history+val-target {want}")
    print(f"  leave-one-out protocol verified on {len(sample)} users", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p, needs_split=False)
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    assets = common.load_experiment_assets(args, cfg, need_models=False)
    expected_users = cfg.get("expected", {}).get("num_users")

    frames = {}
    for split in ("validation", "test"):
        df = build_manifest(assets, split, batch_size=args.batch_size or 1024)
        n_total, n_elig = len(df), int(df["eligible"].sum())
        print(f"{split}: {n_total} users, {n_elig} eligible; exclusions: "
              f"{df.loc[~df['eligible'], 'exclusion_reason'].value_counts().to_dict()}",
              flush=True)
        if expected_users is not None and n_total != int(expected_users):
            raise SystemExit(f"{split}: {n_total} users != expected {expected_users}")
        if args.dry_run:
            continue
        manifest.write_split_manifest(root, split, df)
        frames[split] = df
    if not args.dry_run:
        max_len = int(getattr(assets.dm.cfg, "max_history_len", 20))
        check_leave_one_out(frames["validation"], frames["test"], max_len)
    print("stage 01 complete", flush=True)


if __name__ == "__main__":
    main()
