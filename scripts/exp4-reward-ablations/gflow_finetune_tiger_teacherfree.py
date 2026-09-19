#!/usr/bin/env python
"""TEACHER-FREE TIGER GFlow fine-tuning on all five datasets.

An additional experiment on top of the base-matched protocol
(configs/exp2-tuned-params/finetune_base_matched.json, block
`teacher_free_experiment`). The SASRec teacher served the GFlow arm in two
places - it scored the whole catalog every step to draw the curriculum
negatives (~30% of a step) and it was a reward - and neither exists here:

  negatives   --samplers random policy (a grid dimension)
                random  uniform over the catalog
                policy  sampled without replacement from the TIGER model being
                        fine-tuned, ALWAYS on-policy (its own distribution from
                        the first step: no curriculum, temperature 1), by
                        stochastic beam search constrained at every level to
                        the semantic IDs of valid catalog items
                        (flowrec/candidates/sampler.py)
  rewards     interaction (logs), similarity (hierarchical, RQ-VAE codebooks),
              sim_embed (cosine between item content embeddings) and every
              product of them: 7 combos

Everything else is the base-matched run, read from the same file through
gflow_finetune_tiger_matched.matched_defaults: the dataset's own train layout
(yambda_500m: the same 15 random targets per user), micro-batch x accumulation,
validation every epoch, patience 5 counted after 10 epochs, K = 10, powers
{0.25, 1, 4, 16}, objectives {tb, db, subtb}, rho {1, 0.1}, seed 42. The grid
is 2 x 7 x 4 x 2 x 3 = 336 runs per dataset (sampler-major).

Outputs and the metrics CSV are this experiment's own; the factor caches are
SHARED with the base-matched grid (same data, same rows): interaction and
similarity are reused, sim_embed is added next to them.

Usage (--dataset is required; every flag of gflow_finetune_yambda500m.py
still applies and overrides the defaults):
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_teacherfree.py --dataset beauty --list
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_teacherfree.py --dataset beauty \\
      --cache-only --cache-components sim_embed
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_teacherfree.py --dataset beauty --index 0-6
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_tiger_matched as matched  # noqa: E402

base = matched.base
EXPERIMENT_BLOCK = "teacher_free_experiment"


def experiment_block(protocol: dict[str, Any]) -> dict[str, Any]:
    block = protocol[EXPERIMENT_BLOCK]
    if float(block["component_weights"]) != 1.0:
        raise ValueError("the factorized grid composes every reward component with weight 1")
    if "teacher" in block["samplers"] or any("sasrec" in str(c).split("_x_") for c in block["combos"]):
        raise ValueError(f"{EXPERIMENT_BLOCK}: no sampler or reward may use the SASRec teacher")
    return block


def teacherfree_defaults(dataset: str, protocol: dict[str, Any]) -> list[str]:
    """The base-matched flags of the dataset, then this experiment's grid and
    locations (later flags win; the cache root stays the base-matched one)."""
    block = experiment_block(protocol)
    tiger = block["tiger"]
    return [
        *matched.matched_defaults(dataset, protocol),
        "--samplers", *[str(s) for s in block["samplers"]],
        "--combos", *[str(c) for c in block["combos"]],
        "--powers", *[str(float(p)) for p in block["powers"]],
        "--objectives", *[str(o) for o in block["objectives"]],
        "--rho-targets", *[str(float(r)) for r in block["rho_targets"]],
        "--num-candidates", str(int(block["num_negatives"])),
        "--output-root", str(tiger["output_roots"][dataset]),
        "--metrics-csv-name", str(tiger["metrics_csv_names"][dataset]),
    ]


def teacherfree_argv(argv: Sequence[str] | None = None) -> list[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    return [*teacherfree_defaults(matched.selected_dataset(argv), matched.load_protocol()), *argv]


def parse_args(argv: Sequence[str] | None = None):
    return base.parse_args(teacherfree_argv(argv))


def main(argv: Sequence[str] | None = None) -> None:
    full_argv = teacherfree_argv(argv)
    if not any(flag in full_argv for flag in ("-h", "--help")):
        args = base.parse_args(full_argv)
        matched.check_uniform_micro_batch(args)
        if base.needs_teacher(base.build_grid(args)):
            raise SystemExit("this experiment is teacher-free: drop the teacher sampler / sasrec combos "
                             "(they belong to gflow_finetune_tiger_matched.py)")
    base.main(full_argv)


if __name__ == "__main__":
    main()
