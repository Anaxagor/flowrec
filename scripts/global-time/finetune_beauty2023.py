"""Global-time Beauty 2023 fine-tuning from the original TIGER and GPT-2 checkpoints."""
from __future__ import annotations

import ce_finetune_yambda as ce
import gflow_finetune_yambda as gflow

DATASET = "beauty_2023"
EXPERIMENT = "beauty2023-global-k3"


def grid(model, dataset):
    return {**gflow.cells(model, dataset), "ce": ce.cells(model, dataset)}


def train(session, dm, cell):
    if cell["arm"] == "ce":
        return ce.run_cell(session, dm, cell)
    return gflow.run_cell(session, dm, cell)


def execute(cli, cell, *, train_cell):
    import beauty2023_global_inputs as inputs
    # Factor and prefix identities are shared. CE and GFlow use different
    # baseline-metric formats, so keep their persistent experiment trees separate.
    cli.output_root /= "ce" if cli.stage == "ce" else "gflow-sft"
    return gflow.execute_global(cli, cell, train_cell=train_cell, input_module=inputs, experiment=EXPERIMENT)


def main(argv=None):
    gflow.run_cli(argv, description=__doc__, experiment=EXPERIMENT, datasets=(DATASET,),
                  training_stages=("sft", "gflow", "ce"), build_grid=grid, train_cell=train,
                  execute_stage=execute)


if __name__ == "__main__":
    main()
