#!/usr/bin/env python
"""Base-matched (full-user-history) run of the GPT-2 fine-tuning arms on Amazon.

Same machinery, rewards, cache design, claims and evaluation as
gflow_finetune_gpt2_amazon.py (this script only fixes its defaults and calls
its main). Every default is read from
configs/exp2-tuned-params/finetune_base_matched.json - the file the TIGER
wrappers read too (sft_finetune_control_matched.py,
gflow_finetune_tiger_matched.py) - so the GPT-2 and the TIGER fine-tuning
experiments are ONE experiment on two models: nothing below is GPT-2-specific.

  data      EVERY arm - the GFlow runs and the SFT-only control - fine-tunes
            on the FULL user history: every prefix example of every user
            (beauty 131,413, instruments 116,989) = the base SFT run's
            targets, not the per-user last-8 subset the base script inherited
            from the yambda protocol (95,061 / 94,226). --train-last-n 0.
  grid      the study-wide GFlow grid, 48 runs per dataset, in the same order
            as TIGER's: {sasrec, sasrec x interaction x similarity
            (hierarchical)} x reward power {0.25, 1, 4, 16} x rho {1, 0.1} x
            {tb, db, subtb}; component weights 1, K=10, seed 42.
            (Until 2026-09-19: 11 combos incl. sim_embed at K=3 = 264 runs.)
  control   --sft-only, seeds 42 / 43 / 44 like the TIGER control
            (--seed N; run dirs <ds>/sft_only/seed_<N>).
  batches   one setting for every arm and both models on these datasets:
            micro-batch 32 x accumulation 64 = 2048 (GPT-2 scores at most 512
            candidate sequences per micro-batch on a 32 GB V100; at K=10 that
            is 32 x 11 = 352), validation batch 64, final-eval batch 64.
            --uniform-micro-batch gives the control the same 32 x 64, and the
            script refuses a K / cap pair that would shrink the micro-batch.
  schedule  validation after every epoch; stop when 5 consecutive epochs bring
            no new best (the running best, seeded by the init checkpoint's
            score: --validate-at-start); the counter starts after the 10
            curriculum epochs (--min-epochs 10), in the control too.
  sampler   --curriculum-epochs 10: the easy->hard negative-sampling ramp in
            epochs of the fine-tuning examples.
  --resume  an interrupted run continues from checkpoints/last.pt.

Everything lives apart from the last-8 sweep (reward-cache rows are positions
in the fine-tuning subset, so the caches cannot be shared either):

  runs     experiments/exp4-gflow-gpt2-amazon-fullhist/<ds>/...
  caches   artifacts/exp4-reward_ablation/gflow_gpt2_amazon_fullhist/<ds>/reward_cache/<component>/
           (sasrec, interaction, sim_hier; float16: ~3.2 GB per component for
           beauty, ~2.0 GB for instruments)
  results  artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_fullhist_metrics_summary.csv
           (variant = init | sft_only | gflow)

Every default is an ordinary flag of the base script placed BEFORE the command
line, so any of them can still be overridden (argparse keeps the last
occurrence), and every base-script action works unchanged:

  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py --list
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py --status
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py --cache-status
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py \
      --cache-only --datasets beauty --components sim_hier
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py --sft-only --seed 43
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py \
      --worker-id $SLURM_ARRAY_TASK_ID --time-budget-hours 23.5
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py \
      --smoke --device cpu --num-workers 0 --datasets beauty --index 0

Cluster: scripts/slurm/gflow_finetune_gpt2_amazon_fullhist.sbatch
(CACHE_ONLY=1 array 0-2,4-6, SFT_ONLY=1 array 0-5, then the worker array).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_gpt2_amazon as base  # noqa: E402

# Every default comes from the base-matched protocol file, the same one the
# TIGER wrappers read (sft_finetune_control_matched.py,
# gflow_finetune_tiger_matched.py): nothing here is GPT-2-specific.
PROTOCOL_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"
ARM = "gpt2_sft_only_and_gflow"
# The grid names `similarity` abstractly; these are the base script's modes.
SIMILARITY_MODES = {"hierarchical": "hierarchical", "embedding_cosine": "embedding_cosine"}


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))


def dataset_block(protocol: dict[str, Any]) -> dict[str, Any]:
    """The arm's datasets share one batch setting (the script takes one
    --batch-size for all of them)."""
    names = protocol["arms"][ARM]["datasets"]
    keys = ("train_last_n", "batch_size", "grad_accum", "val_batch_size", "test_batch_size")
    blocks = [{k: protocol["datasets"][n][k] for k in keys} for n in names]
    if any(b != blocks[0] for b in blocks):
        raise ValueError(f"{PROTOCOL_CONFIG.name}: {names} must share one batch setting")
    return blocks[0]


def full_history_defaults(protocol: dict[str, Any] | None = None) -> tuple[str, ...]:
    protocol = load_protocol() if protocol is None else protocol
    block, shared = dataset_block(protocol), protocol["shared_schedule"]
    grid, arm = protocol["gflow_grid"], protocol["arms"][ARM]
    if int(block["batch_size"]) * int(block["grad_accum"]) != int(shared["effective_batch"]):
        raise ValueError(f"batch_size x grad_accum != effective_batch in {PROTOCOL_CONFIG.name}")
    return (
        "--train-last-n", str(int(block["train_last_n"])),
        "--batch-size", str(int(block["batch_size"])),
        "--grad-accum", str(int(block["grad_accum"])),
        "--uniform-micro-batch",
        "--val-batch-size", str(int(block["val_batch_size"])),
        "--test-batch-size", str(int(block["test_batch_size"])),
        "--validate-at-start" if shared["run_validation_at_start"] else "--no-validate-at-start",
        "--validate-every", str(int(shared["validate_every"])),
        "--patience", str(int(shared["patience"])),
        "--min-epochs", str(int(shared["early_stopping_min_epochs"])),
        "--curriculum-epochs", str(float(shared["gflow_curriculum_epochs"])),
        "--resume",
        "--seed", str(int(shared["seed_gflow"])),
        "--combos", *[str(c) for c in grid["combos"]],
        "--similarity-modes", SIMILARITY_MODES[str(grid["similarity_mode"])],
        "--weight-levels", str(float(grid["component_weights"])),
        "--powers", *[str(float(p)) for p in grid["powers"]],
        "--objectives", *[str(o) for o in grid["objectives"]],
        "--rho-targets", *[str(float(r)) for r in grid["rho_targets"]],
        "--num-candidates", str(int(grid["num_negatives"])),
        "--output-root", str(arm["output_root"]),
        "--cache-root", str(arm["cache_root"]),
        "--metrics-csv", str(arm["metrics_csv"]),
    )


_PROTOCOL = load_protocol()
OUTPUT_ROOT = str(_PROTOCOL["arms"][ARM]["output_root"])
CACHE_ROOT = str(_PROTOCOL["arms"][ARM]["cache_root"])
METRICS_CSV = str(_PROTOCOL["arms"][ARM]["metrics_csv"])
CONTROL_SEEDS = tuple(int(x) for x in _PROTOCOL["shared_schedule"]["seeds_sft_only"])
FULL_HISTORY_DEFAULTS: tuple[str, ...] = full_history_defaults(_PROTOCOL)


def full_history_argv(argv: Sequence[str] | None = None) -> list[str]:
    return [*FULL_HISTORY_DEFAULTS, *(sys.argv[1:] if argv is None else argv)]


def check_uniform_micro_batch(args) -> None:
    """Every arm must run at the protocol micro-batch: a K / candidate-cap pair
    that shrinks it would make this model's runs differ from the TIGER control
    on the same dataset."""
    protocol = (int(args.batch_size), int(args.grad_accum))
    for k in (*args.num_candidates, None):
        if base.run_batch_settings(args, k) != protocol:
            raise SystemExit(
                f"--num-candidates {list(args.num_candidates)} under --max-candidate-batch "
                f"{args.max_candidate_batch} would run at {base.run_batch_settings(args, k)} instead "
                f"of the protocol micro-batch {protocol}. Lower batch_size for the dataset in "
                f"{PROTOCOL_CONFIG.name} (it then changes for every arm and model)."
            )


def parse_args(argv: Sequence[str] | None = None):
    return base.parse_args(full_history_argv(argv))


def main(argv: Sequence[str] | None = None) -> None:
    full_argv = full_history_argv(argv)
    if not any(flag in full_argv for flag in ("-h", "--help")):
        check_uniform_micro_batch(base.parse_args(full_argv))
    base.main(full_argv)


if __name__ == "__main__":
    main()
