#!/usr/bin/env python
"""GPT-2 Yambda fine-tuning and reward ablations.

The default experiment is Yambda 50M matched trial 36, with three negatives,
the last 128 targets and their full prefix histories. Training starts at
128 x 16 = 2048 for both GFlow and SFT-only. Automatic OOM backoff preserves
the effective batch. Reward caches, outputs and metrics use dedicated paths.

--yambda50-base legacy selects the previous experiment described below.
An explicit Yambda 500M-only invocation also retains its previous defaults.

50M retains the last 128 active targets/user; 500M samples 15 of those with
seed 42, using the existing matched protocol. Both SFT and GFlow consume the
same prefixes. Catalog/validation/test are retained from the full dense split.
Rewards use factor caches in a separate directory from sliding-prefix runs.
Initial/final beam-20 evaluation uses batches of 64: the 50M checkpoint's
batch of 256 exhausted even an 80 GB H100. Training batches are unchanged.

Actions are inherited from the Amazon runner: --cache-only, --sft-only,
--index, --list, --status, --smoke, etc. --teacher-free selects the existing
teacher-free grid. --audit-data-only builds data and writes the layout audit
without loading a model or computing rewards.

Cluster launchers: RECALC_GROUPS=gpt2_yambda or teacherfree_gpt2_yambda
bash scripts/scheduler/submit_base_matched_recalc.sh
Matched trial 36, new factors + controls + ablations on A100/H100 80 GB:
bash scripts/scheduler/submit_gpt2_yambda50_trial36.sh
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
TRIAL36_NAME = "gflow_gpt2_yambda50_trial36_k3"
TRIAL36_CACHE_ROOT = f"artifacts/exp4-reward_ablation/{TRIAL36_NAME}"


def prepare_argv(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--teacher-free", action="store_true")
    parser.add_argument("--audit-data-only", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--yambda50-base", choices=("legacy", "trial36"))
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.datasets is None:
        options.datasets = list(DATASETS if options.yambda50_base == "legacy" else DATASETS[:1])
    if any(dataset not in DATASETS for dataset in options.datasets):
        raise ValueError("Use the existing Amazon script for Amazon datasets")
    if options.yambda50_base is None:
        options.yambda50_base = "trial36" if options.datasets == [DATASETS[0]] else "legacy"
    if options.yambda50_base == "trial36" and options.datasets != [DATASETS[0]]:
        parser.error("--yambda50-base trial36 requires only --datasets yambda_50m_listens")
    defaults = teacherfree.teacherfree_defaults() if options.teacher_free else matched.full_history_defaults()
    if options.yambda50_base == "trial36":
        name = TRIAL36_NAME + ("_teacherfree" if options.teacher_free else "")
        return options, [*defaults, "--datasets", *options.datasets,
                         "--yambda50-base", "trial36", "--num-candidates", "3",
                         "--batch-size", "128", "--grad-accum", "16",
                         "--uniform-micro-batch", "--max-candidate-batch", "512",
                         "--oom-retries", "7", "--val-batch-size", "256",
                         "--test-batch-size", "64", "--cache-format", "factorized",
                         "--reward-prefetch", "--tensorized-train-loader",
                         "--loader-prefetch-factor", "2", "--cache-root", TRIAL36_CACHE_ROOT,
                         "--output-root", f"experiments/exp4-{name.replace('_', '-')}",
                         "--metrics-csv", f"artifacts/exp4-reward_ablation/{name}_metrics_summary.csv",
                         *(["--audit-data-only"] if options.audit_data_only else []),
                         *remaining]
    name = "gflow_gpt2_yambda_dense_prefix" + ("_teacherfree" if options.teacher_free else "")
    return options, [*defaults, "--datasets", *options.datasets,
                     "--yambda50-base", "legacy", "--prefix-from-dense",
                     "--test-batch-size", "64",
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
        print("Yambda options: --yambda50-base {trial36,legacy}, --teacher-free, --audit-data-only\n", flush=True)
    args = parse_args(argv)
    if args.audit_data_only:
        base.audit_data(args)
        return
    base.main(command)


if __name__ == "__main__":
    main()
