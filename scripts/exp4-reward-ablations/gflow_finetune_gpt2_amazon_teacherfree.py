#!/usr/bin/env python
"""TEACHER-FREE GPT-2 GFlow fine-tuning on Amazon and Yambda.

The GPT-2 side of the `teacher_free_experiment` block of
configs/exp2-tuned-params/finetune_base_matched.json; the TIGER side is
gflow_finetune_tiger_teacherfree.py and both read the SAME block, so the two
models run one grid:

  negatives   --samplers random policy
                random  uniform over the catalog
                policy  sampled without replacement from the GPT-2 model being
                        fine-tuned, ALWAYS on-policy (no curriculum,
                        temperature 1), by stochastic beam search constrained
                        at every level to the semantic IDs of valid catalog items
  rewards     interaction (logs), sim_hier (hierarchical similarity),
              sim_embed (cosine between item content embeddings) and every
              product of them: 7 combos (--component-combos)

No SASRec teacher is loaded. Everything else comes from the full-history
wrapper's defaults (gflow_finetune_gpt2_amazon_fullhist.full_history_defaults):
full histories for Amazon; last 128 targets for Yambda 50M; 15 sampled from
the last 128 for Yambda 500M. Effective batch 2048, validation every epoch, patience 5
counted after 10 epochs, K = 10, powers {0.25, 1, 4, 16}, objectives
{tb, db, subtb}, rho {1, 0.1}, seed 42: 336 runs per dataset.

Outputs and the metrics CSV are this experiment's own; the component caches
are SHARED with the full-history grid (interaction, sim_hier, sim_embed).

Usage (every base flag still applies and overrides the defaults):
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_teacherfree.py --list
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_teacherfree.py \\
      --cache-only --datasets beauty --components sim_embed
  python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_teacherfree.py \\
      --datasets beauty --index 0-6
Cluster: RECALC_GROUPS=teacherfree_gpt2 GPT2_DATASETS=all BLOCK_SIZE=4 \\
    bash scripts/slurm/submit_base_matched_recalc.sh
Default datasets remain beauty / instruments. Beauty 2023 and Yambda store
reward factors using the same implementation as TIGER.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_gpt2_amazon_fullhist as fullhist  # noqa: E402

base = fullhist.base
EXPERIMENT_BLOCK = "teacher_free_experiment"


def component_combos(block: dict[str, Any]) -> list[str]:
    """The block's model-agnostic combo names in this script's component keys
    (similarity -> sim_hier), canonical component order."""
    names = block["gpt2_component_names"]
    combos = []
    for combo in block["combos"]:
        components = [str(names[part]) for part in str(combo).split("_x_")]
        if "sasrec" in components:
            raise ValueError(f"{EXPERIMENT_BLOCK}: no reward may use the SASRec teacher")
        combos.append(base.combo_key(base.parse_component_combo("_x_".join(components))))
    return combos


def teacherfree_defaults(protocol: dict[str, Any] | None = None) -> tuple[str, ...]:
    protocol = fullhist.load_protocol() if protocol is None else protocol
    block = protocol[EXPERIMENT_BLOCK]
    if "teacher" in block["samplers"]:
        raise ValueError(f"{EXPERIMENT_BLOCK}: no sampler may use the SASRec teacher")
    gpt2 = block["gpt2"]
    return (
        *fullhist.full_history_defaults(protocol),
        "--samplers", *[str(s) for s in block["samplers"]],
        "--component-combos", *component_combos(block),
        "--similarity-modes", "hierarchical", "embedding_cosine",
        "--weight-levels", str(float(block["component_weights"])),
        "--powers", *[str(float(p)) for p in block["powers"]],
        "--objectives", *[str(o) for o in block["objectives"]],
        "--rho-targets", *[str(float(r)) for r in block["rho_targets"]],
        "--num-candidates", str(int(block["num_negatives"])),
        "--output-root", str(gpt2["output_root"]),
        "--metrics-csv", str(gpt2["metrics_csv"]),
    )


def teacherfree_argv(argv: Sequence[str] | None = None) -> list[str]:
    return [*teacherfree_defaults(), *(sys.argv[1:] if argv is None else argv)]


def parse_args(argv: Sequence[str] | None = None):
    return base.parse_args(teacherfree_argv(argv))


def main(argv: Sequence[str] | None = None) -> None:
    full_argv = teacherfree_argv(argv)
    if not any(flag in full_argv for flag in ("-h", "--help")):
        args = base.parse_args(full_argv)
        fullhist.check_uniform_micro_batch(args)
        if not args.sft_only and base.needs_teacher(base.build_grid(args)):
            raise SystemExit("this experiment is teacher-free: drop the teacher sampler / sasrec combos "
                             "(they belong to gflow_finetune_gpt2_amazon_fullhist.py)")
    base.main(full_argv)


if __name__ == "__main__":
    main()
