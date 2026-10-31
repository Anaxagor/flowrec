"""Tune GPT-2 on three Amazon and two Yambda datasets with global time splits.

Reuses the exp2 GPT-2 search spaces, trial budgets (100/100/60/40/20),
baseline profiles, pruning, constrained 20-beam validation NDCG@10,
checkpoint promotion and optional three-seed reranking. Each dataset uses
its pinned global cutoffs, rolling histories and last-per-user targets.

Semantic IDs reuse datasets/semantic_ids/exp2-tuned-params/<dataset>/semantic_ids.json.
Studies, checkpoints and tuned configs are isolated under global-time.

Usage::

    python scripts/global-time/tuning_gpt2.py --device cuda
    python scripts/global-time/tuning_gpt2.py --datasets beauty --rerank

Parallel workers may join a dataset's journal with --worker-id. Include
worker 0 to enqueue the baseline. As in the original tuner, trials already
running can finish beyond the budget. Run reranking once per dataset after
all search workers finish. Use --fail-stale-trials only after every worker
of that dataset has exited.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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
CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp2-tuned-params"
CHECKPOINT_ROOT = PROJECT_ROOT / "experiments/global-time/checkpoints/exp2-tuned_params/gpt2"
SEMANTIC_IDS_DIR = base.DEFAULT_SEMANTIC_IDS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        config_dir=CONFIG_DIR, output_root=CHECKPOINT_ROOT,
        semantic_ids_dir=SEMANTIC_IDS_DIR,
    )
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace, regime: base.FixedRegime) -> base.RunContext:
    ctx = base.build_context(args, regime)
    if args.config_out_dir is None and ctx.output_root.resolve() == CHECKPOINT_ROOT.resolve():
        ctx = replace(ctx, config_out_dir=CONFIG_DIR)
    return replace(ctx, required_split_strategy="global_time")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base.run(args, build_context(args, base.load_regime()))


if __name__ == "__main__":
    main()
