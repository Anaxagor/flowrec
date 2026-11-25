"""Tune GPT-2 on three Amazon and two Yambda datasets with global time splits.

Reuses the exp2 GPT-2 search spaces, trial budgets (100/100/60/40/20),
baseline profiles, pruning, constrained 20-beam validation NDCG@10,
checkpoint promotion and optional three-seed reranking. Each dataset uses
its pinned global cutoffs, rolling histories and last-per-user targets.
Both Yambda sizes train on TIGER's last 128 eligible targets per user with up
to 50 preceding items and a fixed 25% validation sample. --match-tiger selects
this global Yambda data protocol explicitly. --yambda-regime full_history
reopens the original all-target search with its 10% sample.

Semantic IDs reuse datasets/semantic_ids/exp2-tuned-params/<dataset>/semantic_ids.json.
Studies, checkpoints and tuned configs are isolated under global-time.

Usage::

    python scripts/global-time/tuning_gpt2.py --device cuda
    python scripts/global-time/tuning_gpt2.py --match-tiger --time-budget-hours 15
    python scripts/global-time/tuning_gpt2.py --datasets beauty --rerank

For a fresh search on a time-limited cluster, set --time-budget-hours on each
worker. They share a fixed trial budget and reclaim interrupted trials from the
last completed epoch. Exit code 75 requests another allocation. The same trial
retains its seed, optimizer, scaler, best weights, stopping and pruning history.
RNG and loader shuffling restart on each new process. Run reranking after all
search workers finish. --fail-stale-trials is for the legacy allocation mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_global_time_gpt2_base", PROJECT_ROOT / "scripts/exp2-tuned-params/tuning_gpt2.py"
)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

DATASETS = base.DATASETS_TO_TUNE
CONFIG_DIR = base.GLOBAL_CONFIG_DIR
CHECKPOINT_ROOT = base.GLOBAL_CHECKPOINT_ROOT
SEMANTIC_IDS_DIR = base.DEFAULT_SEMANTIC_IDS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = base.build_parser(include_loo_options=False)
    parser.description = __doc__
    parser.add_argument("--match-tiger", dest="match_tiger_data", action="store_true",
                        help="Use TIGER's global-time training targets, histories, catalog and "
                             "validation sample for Yambda. Defaults to both Yambda sizes.")
    parser.add_argument("--yambda-regime", choices=("last_128", "full_history"), default="last_128",
                        help="Training data regime for both Yambda sizes. Default matches global-time TIGER.")
    parser.add_argument("--time-budget-hours", type=float,
                        help="Allocation time available to this worker including setup. "
                             "Starts a fresh resumable study or continues one using the same mode.")
    parser.set_defaults(
        config_dir=CONFIG_DIR, output_root=CHECKPOINT_ROOT,
        semantic_ids_dir=SEMANTIC_IDS_DIR, datasets=None,
    )
    args = parser.parse_args(argv)
    if args.datasets is None:
        args.datasets = list(base.MATCHED_DATASETS if args.match_tiger_data else DATASETS)
    if args.match_tiger_data:
        if args.yambda_regime == "full_history":
            parser.error("--match-tiger conflicts with --yambda-regime full_history")
        if any(name not in base.MATCHED_DATASETS for name in args.datasets):
            parser.error("--match-tiger selects only global-time Yambda 50M and 500M")
    if args.time_budget_hours is not None:
        if not math.isfinite(args.time_budget_hours) or args.time_budget_hours <= 0:
            parser.error("--time-budget-hours must be positive and finite")
        if args.fail_stale_trials or args.rerank:
            parser.error("--time-budget-hours is for global-time search workers and recovers trials automatically")
    return args


def build_context(args: argparse.Namespace, regime: base.FixedRegime) -> base.RunContext:
    ctx = base.build_context(args, regime)
    if args.config_out_dir is None and ctx.output_root.resolve() == CHECKPOINT_ROOT.resolve():
        ctx = replace(ctx, config_out_dir=CONFIG_DIR)
    hours = args.time_budget_hours
    deadline = time.monotonic() + hours * 3600 if hours is not None else None
    return replace(ctx, required_split_strategy="global_time", deadline=deadline,
                   global_yambda_regime="last_128" if args.match_tiger_data else args.yambda_regime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        base.run(args, build_context(args, base.load_regime()))
    except base.WorkerTimeLimit as exc:
        print(exc, flush=True)
        raise SystemExit(75)


if __name__ == "__main__":
    main()
