#!/usr/bin/env python3
"""Reward CE from the tuned leave-one-out TIGER checkpoint on Amazon Beauty 2023."""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ce_distill_tiger as ce
from ce_beauty2023_checkpoint_paths import initial_checkpoint


DATASET = "beauty_2023"
DEFAULT_OUTPUT_ROOT = "experiments/exp7-ce-distillation-tiger-beauty2023"
COMBOS = (*ce.COMPONENTS, ce.DEFAULT_COMBO)


def parse_args(argv=None):
    args = ce.parse_args(
        argv, description=__doc__, dataset_choices=(DATASET,), default_output_root=DEFAULT_OUTPUT_ROOT,
        parser_defaults={"powers": [.25, 1., 16.], "rho_targets": [1., .1],
                         "sampling": "curriculum_smooth_is", "num_samples": 3, "curriculum_epochs": 5.,
                         "checkpoint_every_updates": 100},
    )
    if args.estimator != "sampled" or args.sampling != "curriculum_smooth_is":
        raise SystemExit("Beauty 2023 CE requires sampled curriculum_smooth_is")
    if any(combo not in COMBOS for combo in args.combos):
        raise SystemExit("Beauty 2023 CE supports each reward separately and their three-way product")
    return args


def matched_args(args, dataset, seed, matched):
    b = ce.matched_args(args, dataset, seed, matched)
    b.num_candidates = [args.num_samples]
    # The GFlow parser has now activated all constants for this dataset. Use
    # the same resolved init/teacher files in its config, training and audits.
    base = matched.base
    root = base.PROJECT_ROOT
    init = initial_checkpoint(root, "tiger").relative_to(root)
    teacher = initial_checkpoint(root, "sasrec").relative_to(root)
    base.TIGER_CHECKPOINT = str(init)
    base.SASREC_CHECKPOINT = str(teacher)
    base.SASREC_TRIAL = str(teacher.with_name("best_trial.json"))
    return b


def result_metadata(args, entry, b, config):
    return {"model": "tiger", "source_profile": "exp2_tiger_beauty_2023_loo_initial",
            "target_selection": "all_prefixes"}


def main(argv=None):
    ce.run_experiment(parse_args(argv), matched_args_fn=matched_args,
                      result_metadata_fn=result_metadata)


if __name__ == "__main__":
    main()
