#!/usr/bin/env python
"""Stage 00: resolve assets, verify checksums and catalog/trie invariants.

Writes manifest/asset_checksums.json and manifest/run_manifest.json. Refuses
to continue when a stored checksum disagrees (never silently substitutes a
dataset). ``--check-models`` additionally loads both frozen checkpoints and
runs the teacher-forced NLL health gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest, paths, trie as trie_mod  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p, needs_split=False)
    p.add_argument("--check-models", action="store_true",
                   help="also load TIGER/SASRec and run the NLL health gate")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)

    resolver = common.resolver_from_args(args)
    resolved = resolver.resolve_all(checksum=True)
    resolver.print_resolved(resolved)
    checks_path = manifest.manifest_dir(root) / "asset_checksums.json"
    mismatches = paths.verify_known_checksums(resolved, checks_path)
    if mismatches:
        raise SystemExit(
            f"asset checksum MISMATCH vs stored manifest for {mismatches}; "
            "refusing to continue with a different dataset/checkpoint")
    if args.dry_run:
        print("[dry-run] assets resolved; exiting before datamodule build")
        return

    assets = common.load_experiment_assets(args, cfg, need_models=args.check_models)
    expected = cfg.get("expected", {})
    meta = trie_mod.validate_trie(assets.trie, expected)
    print(f"trie invariants OK: {meta}", flush=True)
    n_users = int(assets.dm.num_users)
    if expected.get("num_users") is not None and n_users != int(expected["num_users"]):
        raise SystemExit(f"num_users {n_users} != expected {expected['num_users']}")

    if args.check_models:
        import sft_finetune_control as sft_control
        nll = sft_control.sanity_check_checkpoint(
            assets.tiger, assets.dm, assets.device, 6.0)
        print(f"TIGER teacher-forced val NLL {nll:.3f} (healthy ~2-4)", flush=True)

    paths.write_checksums(resolved, checks_path)
    manifest.write_run_manifest(root, {
        "run_id": cfg["run_id"],
        "config_path": cfg.get("_config_path"),
        "expected": expected,
        "trie": meta,
        "trie_hash": trie_mod.trie_hash(assets.trie),
        "num_users": n_users,
        "assets": {k: {"logical": a.logical, "resolved": str(a.path),
                       "sha256": a.sha256, "source": a.source}
                   for k, a in resolved.items()},
    })
    print(f"stage 00 complete; manifests under {manifest.manifest_dir(root)}", flush=True)


if __name__ == "__main__":
    main()
