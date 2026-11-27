"""Global SFT and GFlow from the promoted TIGER init on Beauty 2014 and Instruments."""
from __future__ import annotations

from functools import partial

import ce_distill_inputs as inputs
import gflow_finetune_yambda as gflow
from gpt2_gflow_inputs import DATASETS, SAMPLERS

EXPERIMENT = "tiger-gflow-k3"


def grid(model, dataset):
    cells = gflow.cells(model, dataset)
    expanded = []
    for sampler in SAMPLERS:
        counts = {}
        if sampler == "policy_random":
            counts = {"num_policy_negatives": 3, "num_random_negatives": 3, "num_negatives": 6}
        expanded.extend(dict(cell, sampler=sampler, **counts) for cell in cells["gflow"])
    return {**cells, "gflow": expanded}


def execute(cli, cell, *, train_cell):
    return gflow.execute_global(cli, cell, train_cell=train_cell, input_module=inputs, experiment=EXPERIMENT)


def main(argv=None):
    gflow.run_cli(argv, description=__doc__, experiment=EXPERIMENT, datasets=DATASETS, models=("tiger",),
                  training_stages=("sft", "gflow"), build_grid=grid, execute_stage=execute,
                  train_cell=partial(gflow.run_cell, sampler_directory=True))


if __name__ == "__main__":
    main()
