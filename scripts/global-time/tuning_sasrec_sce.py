"""Tune SASRec with SCE on Yambda 500M using the pinned global-time split.

Reuses the exp2 SCE tuner, including its 30-trial Yambda 500M budget,
architecture/optimizer/SCE search spaces, incumbents, pruning, checkpoint
promotion, journal resume and optional three-seed reranking. Only the data
split and experiment paths change. Validation scores the full catalog.

Usage::

    python scripts/global-time/tuning_sasrec_sce.py --worker-id 0 --device cuda
    python scripts/global-time/tuning_sasrec_sce.py --rerank --device cuda
    submit-worker scripts/scheduler/tuning_sasrec_sce_global_time_array.submit-worker

Worker 0 enqueues the exp1 baseline and, when present, the global-time
full-CE Yambda 50M winner from --tuned-config-dir, as in the original tuner.
All workers share one journal; reranking runs once after the search ends.
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
    "_global_time_sce_base", PROJECT_ROOT / "scripts/exp2-tuned-params/tuning_sasrec_sce.py"
)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

DATASETS = ("yambda_500m_listens",)
CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp2-tuned-params"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/global-time/exp2-tuned_params/sasrec_sce"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params/sasrec_sce"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        datasets=list(DATASETS), config_dir=CONFIG_DIR, tuned_config_dir=CONFIG_DIR,
        checkpoint_root=CHECKPOINT_ROOT, artifacts_dir=ARTIFACTS_DIR,
    )
    return parser.parse_args(argv)


def runtime_from_args(args: argparse.Namespace) -> base.Runtime:
    return replace(base.runtime_from_args(args), required_split_strategy="global_time")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base.run(args, runtime_from_args(args))


if __name__ == "__main__":
    main()
