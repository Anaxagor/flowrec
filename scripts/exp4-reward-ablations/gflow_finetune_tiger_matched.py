#!/usr/bin/env python
"""Base-matched TIGER GFlow fine-tuning on every dataset of the study.

  beauty, instruments, beauty_2023, yambda_50m_listens, yambda_500m_listens

Same grid machinery, rewards, factorized caches, calibration, resume and
evaluation as gflow_finetune_yambda500m.py (this script only fixes its
defaults and calls its main). Every default is read from
configs/exp2-tuned-params/finetune_base_matched.json - the file the TIGER
SFT-only control (sft_finetune_control_matched.py) and the GPT-2 arms
(gflow_finetune_gpt2_amazon_fullhist.py) read too, so this is the TIGER side of
ONE experiment: same data rule, schedule, grid, K, curriculum and seeds.

  data      the GFlow arm fine-tunes on EVERY training example of the base SFT
            run, like the SFT-only control (--train-last-n 0):
              Amazon   every prefix of the full user history (beauty 131,413,
                       instruments 116,989, beauty_2023 3,493,253 examples).
                       The checkpoint's captured config as the control builds
                       it, with verified-purchase expressed as
                       positive_criterion (identical catalog and examples,
                       checked on all three; the unverified reviews -
                       instruments 10,792, beauty_2023 296,645 - stay
                       available to the interaction reward).
              Yambda   50M: the checkpoint's whole last-128-targets layout
                       (993,926 examples), not the per-user last-8 subset of
                       the first sweep (69,495).
                       500M: 15 targets per user drawn uniformly at RANDOM,
                       once (seed 42), from that last-128 layout: ~1.3M of
                       ~9.9M examples (--train-random-n 15). A cost decision:
                       an epoch takes ~11 min instead of ~1.4 h. Unlike the
                       first sweep's last-8 tail (697,485), the draw keeps the
                       base run's mix of target positions, and the SFT-only
                       control uses the very same examples.
            All datasets use the factorized per-component caches (sasrec
            hidden states: num_examples x d_model x 2 bytes).
  assets    TIGER init = the control's init checkpoint; SASRec teacher
            (curriculum negatives + sasrec reward) = the exp2 tuned teacher
            the GPT-2 arm uses on Amazon; hierarchical similarity over the
            RQ-VAE behind each promoted level-offset SID table
            (<sid dir>/rqvae/model.pt on beauty / instruments).
  batches   per dataset, identical in every arm and model: beauty /
            instruments 32 x 64 (GPT-2's candidate cap at K=10), beauty_2023
            and yambda 256 x 8; validation = final-eval batch 64 / 256. The
            micro-batch must not shrink with K: the script refuses a
            --num-candidates / --max-candidate-batch pair that would scale it.
  monitor   init checkpoint scored first; one FIXED validation sample (seed 0)
            where the split is large: 25% on yambda_500m, 10% on beauty_2023 -
            the same examples as the matched control.
  schedule  validation after every epoch; fine-tuning stops once 5
            consecutive epochs bring no new best (the running best, seeded by
            the init checkpoint's score). The counter starts after the 10
            curriculum epochs (same in the control): while the negatives are
            still easy, validation sits below the init, and counting from
            epoch 0 would stop 52 of the 56 finished per-epoch-validated 50m
            runs at epoch 5 although 43 of them beat the init right after the
            ramp. Earliest stop: epoch 15.
  grid      the study-wide grid, identical cell for cell to the GPT-2 arm's:
            48 runs per dataset, seed 42, K=10, component weights 1:
              combos      sasrec | sasrec_x_interaction_x_similarity
              powers      0.25, 1, 4, 16   (exponent of the composed reward)
              objectives  tb, db, subtb
              rho         1, 0.1           (target GFlow/SFT grad-norm ratio;
                                            run dirs *_rho0p1)
  sampler   the easy->hard curriculum lasts 10 EPOCHS of the fine-tuning
            examples (--curriculum-epochs).
  --resume  runs continue from checkpoints/last.pt, so they can span several
            walltime windows by resubmission.

Cost (0.5 ms per example, recorded in the finished K=10 runs). yambda_50m:
~8-13 min per epoch. yambda_500m with 15 random targets per user: ~11 min per
epoch (the full layout would be ~1.4 h). A run lasts at least 15 epochs:
>= 2-3 h on either, i.e. >= ~6 GPU-days for the 48 yambda_500m runs.
beauty / instruments are small (131k / 117k examples, micro-batch 32);
beauty_2023 is large: 3.49M examples = 13,646 micro-steps per epoch at 256, and
its full-split beam-20 final evaluation covers 662,540 users per split. Cut a grid with --combos / --powers / --objectives /
--rho-targets (size the sbatch --array to `--list`).

Everything lives apart from the last-8 sweeps (cache rows are positions in the
fine-tuning subset, so the caches cannot be shared either):

  runs     experiments/exp4-gflow-tiger-<amazon ds>-matched/, exp4-gflow-yambda50m-matched/,
           exp4-gflow-yambda500m-matched-r15/        (r15 = 15 random targets per user)
  caches   artifacts/exp4-reward_ablation/gflow_{tiger_<amazon ds>,yambda50m}_matched/reward_factors/<component>/,
           .../gflow_yambda500m_matched_r15/reward_factors/<component>/
  results  artifacts/exp4-reward_ablation/gflow_finetune_{tiger_<amazon ds>,yambda_50m}_matched_metrics_summary.csv,
           .../gflow_finetune_yambda_500m_matched_r15_metrics_summary.csv

Every default below is an ordinary flag of the base script placed BEFORE the
command line, so any of them can still be overridden (argparse keeps the last
occurrence), and every base-script action works unchanged:

  python scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py --dataset beauty --list
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py \
      --dataset instruments --cache-only --cache-components sasrec
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py \
      --dataset beauty_2023 --index $SLURM_ARRAY_TASK_ID
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py \
      --dataset beauty --smoke --device cpu --num-workers 0 --index 24

Cluster: scripts/slurm/gflow_finetune_tiger_matched_array.sbatch
(DATASET=..., CACHE_ONLY=1 array 0-2, then the grid array 0-47).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_yambda500m as base  # noqa: E402

PROTOCOL_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))


def selected_dataset(argv: Sequence[str]) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default=None)
    dataset = pre.parse_known_args(list(argv))[0].dataset
    if dataset is None:
        raise SystemExit(f"--dataset is required: one of {load_protocol()['arms']['tiger_gflow']['datasets']}")
    return str(dataset)


def matched_defaults(dataset: str, protocol: dict[str, Any]) -> list[str]:
    """Base-script flags of one dataset (placed before the user's argv)."""
    arm = protocol["arms"]["tiger_gflow"]
    if dataset not in arm["datasets"]:
        raise SystemExit(f"--dataset must be one of {arm['datasets']} for the base-matched run, got {dataset!r}")
    block = protocol["datasets"][dataset]
    shared = protocol["shared_schedule"]
    grid = protocol["gflow_grid"]
    if float(grid["component_weights"]) != 1.0:
        raise ValueError("the factorized yambda grid composes every reward component with weight 1")
    if str(grid["similarity_mode"]) != "hierarchical":
        raise ValueError("the factorized yambda caches hold the hierarchical similarity")
    if int(shared["early_stopping_min_epochs"]) < int(shared["gflow_curriculum_epochs"]):
        raise ValueError("early_stopping_min_epochs must cover the curriculum, or a run can be "
                         "stopped before its negatives reach full difficulty")
    if int(block["batch_size"]) * int(block["grad_accum"]) != int(shared["effective_batch"]):
        raise ValueError(f"{dataset}: batch_size x grad_accum != effective_batch in {PROTOCOL_CONFIG.name}")
    if int(block["test_batch_size"]) != int(block["val_batch_size"]):
        # best.pt is evaluated through the datamodule's eval loaders, whose
        # batch size is the validation batch (same as the SFT-only control).
        raise ValueError(f"{dataset}: the TIGER arms need test_batch_size == val_batch_size")
    defaults = [
        "--train-last-n", str(int(block["train_last_n"])),
        "--train-random-n", str(int(block.get("train_random_n", 0))),
        "--train-subset-seed", str(int(block.get("train_subset_seed", 0))),
        "--batch-size", str(int(block["batch_size"])),
        "--grad-accum", str(int(block["grad_accum"])),
        "--val-batch-size", str(int(block["val_batch_size"])),
        "--num-candidates", str(int(grid["num_negatives"])),
        "--seed", str(int(shared["seed_gflow"])),
        "--combos", *[str(c) for c in grid["combos"]],
        "--powers", *[str(float(p)) for p in grid["powers"]],
        "--objectives", *[str(o) for o in grid["objectives"]],
        "--rho-targets", *[str(float(r)) for r in grid["rho_targets"]],
        "--validate-every", str(int(shared["validate_every"])),
        "--patience", str(int(shared["patience"])),
        "--min-epochs", str(int(shared["early_stopping_min_epochs"])),
        "--curriculum-epochs", str(float(shared["gflow_curriculum_epochs"])),
        "--val-sample-resample" if block["val_sample_resample"] else "--no-val-sample-resample",
        "--validate-at-start" if shared["run_validation_at_start"] else "--no-validate-at-start",
        "--resume",
        "--output-root", str(arm["output_roots"][dataset]),
        "--cache-root", str(arm["cache_roots"][dataset]),
        "--metrics-csv-name", str(arm["metrics_csv_names"][dataset]),
    ]
    if block["val_sample_fraction"] is not None:
        defaults += ["--val-sample-fraction", str(float(block["val_sample_fraction"]))]
    if dataset in arm.get("control_init_baselines", {}):
        # The matched control's evaluation of the same init checkpoint (imported
        # when present and complete, else the script evaluates the init itself).
        defaults += ["--control-init-baseline", str(arm["control_init_baselines"][dataset])]
    return defaults


def matched_argv(argv: Sequence[str] | None = None) -> list[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    return [*matched_defaults(selected_dataset(argv), load_protocol()), *argv]


def check_uniform_micro_batch(args: argparse.Namespace) -> None:
    """Every selected K must run at the protocol micro-batch: a K that shrinks
    it under --max-candidate-batch would give that run a different micro-batch
    than the SFT-only control and the other cells."""
    protocol = (int(args.batch_size), int(args.grad_accum))
    for k in args.num_candidates:
        if base.base.batch_settings_for_k(int(k), args) != protocol:
            raise SystemExit(
                f"--num-candidates {k}: batch x (K+1) = {int(args.batch_size) * (int(k) + 1)} exceeds "
                f"--max-candidate-batch {args.max_candidate_batch}, so this run would not use the "
                f"protocol micro-batch {protocol[0]} x {protocol[1]}. The base-matched comparison "
                "needs one micro-batch for every arm: lower batch_size for the whole dataset in "
                f"{PROTOCOL_CONFIG.name} (and rerun the control), or raise --max-candidate-batch."
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return base.parse_args(matched_argv(argv))


def main(argv: Sequence[str] | None = None) -> None:
    full_argv = matched_argv(argv)
    if not any(flag in full_argv for flag in ("-h", "--help")):
        check_uniform_micro_batch(base.parse_args(full_argv))
    base.main(full_argv)


if __name__ == "__main__":
    main()
