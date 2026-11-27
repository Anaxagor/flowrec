"""Global-time Yambda SFT + CE distillation with three IID reward-sampled negatives."""
from __future__ import annotations

from itertools import product
import json

import gflow_finetune_yambda as global_yambda

EXPERIMENT = "yambda-ce-q-k3"


def cells(model, dataset):
    seed = json.loads(global_yambda.PROTOCOL.read_text())["shared_schedule"]["seed_gflow"]
    return [
        dict(model=model, dataset=dataset, arm="ce", seed=seed,
             combo="_x_".join(global_yambda.COMPONENTS), power=power, rho=rho,
             estimator="sampled", sampling="reward_negatives", num_samples=3,
             importance_correction=False)
        for power, rho in product(global_yambda.REWARD_POWERS, global_yambda.RHO_TARGETS)
    ]


def run_cell(session, dm, cell):
    from yambda_ce_training import run_cell as train
    return train(session, dm, cell)


def main(argv=None):
    global_yambda.run_cli(argv, description=__doc__, experiment=EXPERIMENT, training_stages=("ce",),
                          build_grid=lambda model, dataset: {"ce": cells(model, dataset)},
                          train_cell=run_cell)


if __name__ == "__main__":
    main()
