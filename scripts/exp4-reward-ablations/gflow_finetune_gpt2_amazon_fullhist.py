#!/usr/bin/env python
"""Base-matched GPT-2 fine-tuning on Amazon and Yambda.

Same machinery, rewards, cache design, claims and evaluation as
gflow_finetune_gpt2_amazon.py (this script only fixes its defaults and calls
its main). Every default is read from
configs/exp2-tuned-params/finetune_base_matched.json - the file the TIGER
wrappers read too (sft_finetune_control_matched.py,
gflow_finetune_tiger_matched.py) - so the GPT-2 and the TIGER fine-tuning
experiments use the same data selection, schedule and effective batch.

  data      Beauty, Instruments, Beauty 2023: every active dense label, window/stride 20.
            Yambda 50M: last 128 targets per user, history window 50.
            Yambda 500M: 15 random targets from each user's last 128 (seed 42),
            dense window/stride 50. Prefixes retain their dense-window contexts.
            Training and caches use identical indices;
            subsetting preserves the GPT-2 checkpoint's catalog.
  grid      the study-wide GFlow grid, 168 runs per dataset, in the same order
            as TIGER's: all seven nonempty products of sasrec, interaction,
            and hierarchical similarity x reward power {0.25, 1, 4, 16} x rho {1, 0.1} x
            {tb, db, subtb}; component weights 1, K=10, seed 42.
            (Until 2026-09-19: 11 combos incl. sim_embed at K=3 = 264 runs.)
  control   --sft-only, seeds 42 / 43 / 44 like the TIGER control
            (--seed N; run dirs <ds>/sft_only/seed_<N>).
  batches   dataset-specific settings from the existing protocol, effective
            batch 2048. The GPT-2 candidate cap reduces large micro-batches
            and increases accumulation to preserve that effective batch.
            --uniform-micro-batch gives GFlow and its control the same batch.
  schedule  validation after every epoch; stop when 5 consecutive epochs bring
            no new best (the running best, seeded by the init checkpoint's
            score: --validate-at-start); the counter starts after the 10
            curriculum epochs (--min-epochs 10), in the control too.
  sampler   --curriculum-epochs 10: the easy->hard negative-sampling ramp in
            epochs of the fine-tuning examples.
  --resume  an interrupted run continues from checkpoints/last.pt.

Outputs and caches are separate from the earlier sliding-prefix experiments:

  runs     experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/<ds>/...
  caches   artifacts/exp4-reward_ablation/gflow_gpt2_amazon_fullhist_dense_prefix/<ds>/reward_cache/<component>/
           (sasrec, interaction, sim_hier; float16: ~3.2 GB per component for
           beauty, ~2.0 GB for instruments)
           Beauty 2023 and both Yambda datasets use reward_factors/<component>/
           instead; full example-by-catalog reward matrices are never stored.
  results  artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_fullhist_dense_prefix_metrics_summary.csv
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
      --worker-id $WORKER_ID --time-budget-hours 23.5
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py \
      --smoke --device cpu --num-workers 0 --datasets beauty --index 0

Cluster: scripts/scheduler/gflow_finetune_gpt2_amazon_fullhist.submit-worker
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
    """Legacy constants for Beauty; runtime settings are resolved per dataset."""
    # The first dataset supplies the legacy public constants. Runtime settings
    # are resolved separately for each selected dataset by base.dataset_args.
    return protocol["datasets"]["beauty"]


def full_history_defaults(protocol: dict[str, Any] | None = None) -> tuple[str, ...]:
    protocol = load_protocol() if protocol is None else protocol
    block, shared = dataset_block(protocol), protocol["shared_schedule"]
    grid, arm = protocol["gflow_grid"], protocol["arms"][ARM]
    if int(block["batch_size"]) * int(block["grad_accum"]) != int(shared["effective_batch"]):
        raise ValueError(f"batch_size x grad_accum != effective_batch in {PROTOCOL_CONFIG.name}")
    return (
        "--dataset-profile", "matched",
        "--uniform-micro-batch",
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
    for dataset in args.datasets:
        resolved = base.dataset_args(args, dataset)
        # Large GPT-2 architectures use the existing candidate cap to reduce
        # micro-batches, with accumulation preserving the configured effective
        # batch. The same capped batch is used by GFlow and its SFT control.
        if base.DATASETS[dataset].get("factorized"):
            continue
        protocol = (int(resolved.batch_size), int(resolved.grad_accum))
        for k in (*resolved.num_candidates, None):
            if base.run_batch_settings(resolved, k) != protocol:
                raise SystemExit(
                    f"--num-candidates {list(resolved.num_candidates)} under --max-candidate-batch "
                    f"{resolved.max_candidate_batch} would not use the protocol micro-batch {protocol}. "
                    f"Lower batch_size for the dataset in {PROTOCOL_CONFIG.name}."
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
