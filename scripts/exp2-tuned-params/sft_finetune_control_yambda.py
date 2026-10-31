#!/usr/bin/env python
"""SFT-only fine-tuning control on the exp2 tuned Yambda TIGER checkpoints.

The Yambda counterpart of ``sft_finetune_control.py`` (same hypothesis: does
plain low-LR SFT fine-tuning already deliver the quality attributed to
GFlow / RL?). It reuses that script's machinery wholesale -- runs, seeds,
NDCG@5 monitoring (<= 500 epochs), init-baseline evaluation, artifacts CSVs,
checkpoint promotion, file-locked concurrent aggregation, --aggregate-only --
and swaps in the two exp2 Yambda checkpoints:

  yambda_50m_listens   checkpoints/exp2/loo_split/initial/tiger/yambda_50m_listens/
      best.pt -- single tuned run (sweep winner y50_sft_t031_...,
      seed 42, lr 3e-4, d_model 192 / 2 heads / d_kv 32, no user token, NO
      token offset). Data: last-2-months 50m listens with played_ratio >= 50,
      leave-one-out, last-128-targets policy (catalog 112,943 items).
  yambda_500m_listens  checkpoints/exp2/loo_split/initial/tiger/
      yambda_500m_listens/best.pt -- the OFAT tuning winner (trials/dropout-0)
      (best_trial.json: dropout 0, lr 1.85e-3, d_model 128 / 6 heads / d_kv
      32, shared user token, JSON+1 token offset) TRAINED TO CONVERGENCE at
      the tuned lr, since the 8-epoch trial snapshot was still improving
      when its budget ran out. Data: last-2-months 500m
      multi_event positives (listen & played_ratio_pct >= 50), embedding-
      filtered catalog (374,283 items), leave-one-out, last-128 targets.

Both configs are captured in configs/exp2-tuned-params/
tiger_yambda_{50m,500m}_listens_sft.json (verified against the checkpoints:
strict state-dict load and 100% item_semantic_ids/table agreement). The local
tables ship in datasets/semantic_ids/exp2-tuned-params/<ds>/.

Deltas vs. the Amazon control, all per-dataset spec defaults (CLI overrides):
  - eval-loader batch size 256 (the Amazon 64 is needlessly small here);
  - yambda_500m_listens validates on a 0.25 sample of the val split during
    training (as its tuning did) -- pass --val-sample-fraction 1.0 for full
    validation; the final best.pt evaluation always uses the FULL splits.

Outputs (identical layout to the Amazon control):
  experiments/exp2-sft-finetune-yambda/<ds>/...              run dirs + baseline
  checkpoints/exp2/loo_split/sft-finetuned/tiger/<ds>/
  artifacts/exp2-tuned-params/tiger_sft_finetune_<ds>_metrics_summary.csv
  artifacts/exp2-tuned-params/tiger_sft_finetune_<ds>_summary.json

Examples:
  python scripts/exp2-tuned-params/sft_finetune_control_yambda.py --device cuda
  python scripts/exp2-tuned-params/sft_finetune_control_yambda.py \
      --datasets yambda_50m_listens --seeds 42 --smoke --num-workers 0   # wiring check
  python scripts/exp2-tuned-params/sft_finetune_control_yambda.py --aggregate-only
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sft_finetune_control as base  # noqa: E402


YAMBDA_DATASETS: dict[str, dict[str, Any]] = {
    "yambda_50m_listens": {
        "sft_config": "configs/exp2-tuned-params/tiger_yambda_50m_listens_sft.json",
        "tiger_checkpoint": (
            "checkpoints/exp2/loo_split/initial/tiger/yambda_50m_listens/best.pt"
        ),
        # Paths already correct in the config JSON; nothing to re-point.
        "data_overrides": {},
        "val_batch_size": 256,
        "val_sample_fraction": None,
    },
    "yambda_500m_listens": {
        "sft_config": "configs/exp2-tuned-params/tiger_yambda_500m_listens_sft.json",
        # The OFAT winner (trials/dropout-0) was an 8-epoch tuning snapshot -
        # still improving when its trial budget ran out - so it was first
        # trained to convergence at the tuned lr (stage 1, run
        # sft_only_lr0.00185119/seed_42: val NDCG@5 0.0973 -> 0.1109) and the
        # converged best.pt retained as the current primary initial checkpoint.
        "tiger_checkpoint": (
            "checkpoints/exp2/loo_split/initial/tiger/yambda_500m_listens/best.pt"
        ),
        "data_overrides": {},
        "val_batch_size": 256,
        # The 500m val split is large; its tuning also validated on a 0.25
        # sample. Only the in-training monitor is sampled -- the final best.pt
        # evaluation runs on the full splits.
        "val_sample_fraction": 0.25,
    },
}


def main() -> None:
    # The shared machinery resolves DATASETS / defaults at call time, so
    # swapping the module attributes retargets everything: argparse choices,
    # config assembly, checkpoints, artifacts naming, aggregation.
    base.DATASETS = YAMBDA_DATASETS
    base.DEFAULT_OUTPUT_ROOT = Path("experiments/exp2-sft-finetune-yambda")
    base.main()


if __name__ == "__main__":
    main()
