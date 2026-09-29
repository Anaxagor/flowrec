#!/usr/bin/env python
"""GPT-2 Yambda fine-tuning with prefixes extracted from base dense windows.

50M retains the last 128 active targets/user; 500M samples 15 of those with
seed 42, using the existing matched protocol. Both SFT and GFlow consume the
same prefixes. Catalog/validation/test are retained from the full dense split.
Rewards use factor caches in a separate directory from sliding-prefix runs.

Actions are inherited from the Amazon runner: --cache-only, --sft-only,
--index, --list, --status, --smoke, etc. --teacher-free selects the existing
teacher-free grid. --audit-data-only builds data and writes the layout audit
without loading a model or computing rewards.

Cluster launchers: RECALC_GROUPS=gpt2_yambda or teacherfree_gpt2_yambda
bash scripts/slurm/submit_base_matched_recalc.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_gpt2_amazon_fullhist as matched
import gflow_finetune_gpt2_amazon_teacherfree as teacherfree

base = matched.base
DATASETS = ("yambda_50m_listens", "yambda_500m_listens")
CACHE_ROOT = "artifacts/exp4-reward_ablation/gflow_gpt2_yambda_dense_prefix"


def prepare_argv(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--teacher-free", action="store_true")
    parser.add_argument("--audit-data-only", action="store_true")
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    defaults = teacherfree.teacherfree_defaults() if options.teacher_free else matched.full_history_defaults()
    name = "gflow_gpt2_yambda_dense_prefix" + ("_teacherfree" if options.teacher_free else "")
    return options, [*defaults, "--datasets", *DATASETS, "--prefix-from-dense",
                     "--cache-format", "factorized", "--cache-root", CACHE_ROOT,
                     "--output-root", f"experiments/exp4-{name.replace('_', '-')}",
                     "--metrics-csv", f"artifacts/exp4-reward_ablation/{name}_metrics_summary.csv",
                     *(["--audit-data-only"] if options.audit_data_only else []),
                     *remaining]


def parse_args(argv: Sequence[str] | None = None):
    options, command = prepare_argv(argv)
    args = base.parse_args(command)
    if any(dataset not in DATASETS for dataset in args.datasets):
        raise ValueError("Use the existing Amazon script for Amazon datasets")
    if args.cache_format != "factorized":
        raise ValueError("This Yambda experiment requires factorized reward caches")
    if options.teacher_free and not args.sft_only and base.needs_teacher(base.build_grid(args)):
        raise ValueError("--teacher-free cannot use a teacher sampler or SASRec rewards")
    args.teacher_free = options.teacher_free
    args.audit_data_only = options.audit_data_only
    return args


def main(argv: Sequence[str] | None = None) -> None:
    _, command = prepare_argv(argv)
    if "--help" in command or "-h" in command:
        print(__doc__)
        print("Yambda options: --teacher-free, --audit-data-only\n", flush=True)
    args = parse_args(argv)
    if args.audit_data_only:
        base.audit_data(args)
        return
    base.main(command)


if __name__ == "__main__":
    main()
