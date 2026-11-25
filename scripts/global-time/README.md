# Global-time experiments

These scripts run the standard and tuned model pipelines with a global temporal
split. They complement the leave-one-out pipelines in
[experiment 1](../exp1-standard-params/README.md) and
[experiment 2](../exp2-tuned-params/README.md). The data windows, catalog and
checkpoints are specific to the global-time protocol; leave-one-out checkpoints
must not be substituted automatically.

The scripts build Amazon embeddings and semantic IDs, tune GPT-2, SASRec,
SASRec with scalable cross-entropy and TIGER, and train selected SASRec models.
Configurations, data split details and execution guidance are documented in
[the global-time configuration guide](../../configs/global-time/README.md).

Primary tuned checkpoints use `checkpoints/exp2/global_split/initial/`.
Some models do not have a selected checkpoint and global-time SFT-only primary
weights are not present; consult the [checkpoint layout](../../checkpoints/README.md)
before launching a dependent run. For example, run
`python scripts/global-time/tuning_gpt2.py --help` from the repository root to
inspect the tuning options.

## CE distillation on Beauty and Instruments

Each model and dataset pair has a CPU preparation stage and six independent
GPU runs with seed 42, for 24 runs across both models and datasets. With
`workspace` set to an absolute shared scratch directory, run the following
from the repository root to prepare TIGER on Beauty and train its first index.

```bash
python scripts/global-time/ce_distill.py prepare --model tiger --dataset beauty --workspace "$workspace"
python scripts/global-time/ce_distill.py train --model tiger --dataset beauty --workspace "$workspace" --index 0
```

Each training prefix contributes its observed target to the SFT loss. For the
distillation loss, the teacher multiplies the SASRec, interaction and hierarchical
similarity rewards with unit component weights and applies the GFlow reward floor
and power. It excludes the observed target and the items in the input history,
normalizes the remaining scores, and draws ten items independently with
replacement. Their mean negative log probability is the CE term. The powers
are 0.25, 1 and 16 and the calibrated CE-to-SFT gradient ratios are 1 and 0.1.

To list the six array indices for one pair, run
`python scripts/global-time/ce_distill.py list --model tiger --dataset beauty`.
Each run saves its calibrated weight, selected checkpoint, and final validation
and test metrics under `experiments/global-time/ce-distill-q-k10/<model>/<dataset>/`.

### CE walltime estimates

The closest LOO runs used a V100 32 GB, microbatch 32, accumulation 64,
10 draws and the same powers and gradient ratios. The TIGER runs drew from
full catalog q with the same three reward factors. The available GPT-2 runs
used a smooth curriculum without IS and one or two factors. The global
experiment conditions q on eligible negatives for both models.

| Model and dataset | Comparable runs | LOO median / max, minutes | Epochs | Max across all sampled LOO variants, minutes | Global training limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| TIGER Beauty | 6 | 61.0 / 61.4 | 15 | 70.6 | 12 h |
| GPT-2 Beauty | 36 | 78.6 / 193.2 | 15–40 | 193.2 | 8 h |
| TIGER Instruments | 6 | 55.1 / 75.7 | 15–22 | 97.3 | 4 h |
| GPT-2 Instruments | 36 | 76.0 / 104.7 | 15–22 | 108.9 | 4 h |

Durations include setup inside the training job, fitting with epoch validation,
and final evaluation. The broader maxima also cover power 4 and every available
sampled K=10 reward variant from the initial checkpoints.

To estimate walltime, multiply the broader maximum by the global/LOO train
prefix count ratio (137,864/131,413 for Beauty, 127,025/116,989 for Instruments).
Scale Beauty times by the ratio of transformer-block parameter counts (7.99 for
TIGER and 1.68 for GPT-2), a conservative proxy for compute. Use a factor of
one on Instruments for both models. Add 20% and round up to the next whole
hour. The table lists the estimated training budgets. The number of epochs before
early stopping can vary. The independent CPU preparation limit is 30 minutes. Both analogous
global GFlow preparations on the cluster took 4 min 39 s.

## Yambda global-time GFlow and SFT-only

From the repository root, prepare each model and dataset in an absolute shared
scratch directory. For example, the following prepares TIGER on Yambda 500M.

```bash
python scripts/global-time/gflow_finetune_yambda.py prepare --model tiger --dataset yambda_500m_listens --workspace "$workspace"
```

Across the three model/dataset pairs there are 54 GFlow configurations, nine
SFT-only controls and 12 preparation tasks. For each pair the `sft` stage can
start after `prepare`. Before starting `gflow`, run the `factors` stage once
per component with `--index 0`, `1` and `2`. Pass the same model, dataset
and workspace to every stage.

| Dataset | Global init checkpoint | Train targets | SFT-only / GFlow walltime per cell | Training GPU |
|---|---|---:|---|---|
| Yambda 50M | TIGER | 951,675 | 4 h / 24 h | V100 32 GB |
| Yambda 50M | GPT-2 `last_128` | 951,675 | 8 h / 24 h | A100 or H100 80 GB |
| Yambda 500M | TIGER | 4,026,761 | 24 h / 48 h, subject to the partition limit | A100 or H100 80 GB |

The 48 h limit is per 500M GFlow configuration. Two completed TIGER LOO
runs on 4,162,653 sampled targets took 14.34 h on H100 (SASRec) and
26.48 h on A100 (SASRec × similarity). Both used DB, 10 negatives and 23
epochs. The global grid uses all three factors, three negatives and three
losses. The separate CPU interaction cache job has a 24 h budget. Building
the LOO sampled50 cache took about 7.9 h.

On 50M, both arms use all of each user's last 128 eligible pre-validation
targets. On 500M, start from the same last-128 pool (9,560,668 targets).
For each user with more than 50 eligible positions, sample 50 uniformly
without replacement. Keep every eligible position for shorter histories.
All SFT seeds, GFlow configurations and reward caches share the sample drawn
with seed 42. For the original 500M inputs, the expected `source_indices_sha256`
in `preparation/layout_audit.json` under `finetune_subset` is
`25a03aa05c4ec4466015b668edc77f96d3ba5901d5096024bbe44eb7ff542db1`.
Each target keeps its original prefix of up to 50 preceding
items. The 50M GPT-2 checkpoint trained with one
active final label per dense window, stride one. CPU preparation checks each
prefix's history and target against that dense layout and compares the held-out
examples. Validation and test use the original rolling, last-per-user protocol.

GFlow uses SASRec × interaction × hierarchical SID similarity, powers 0.25,
1 and 16, gradient-norm ratios 1 and 0.1, and TB, DB and SubTB. Three distinct
negatives per prefix are sampled with the SASRec score curriculum. The sampler
excludes the observed target and input history. The global teacher is SASRec
on 50M and SASRec-SCE on 500M. Model selection uses validation NDCG@5.
GFlow uses seed 42 and SFT-only uses 42, 43 and 44.
The effective batch is 2,048 in both arms. GPT-2 accumulates 16 microbatches
of 128 and TIGER accumulates 8 microbatches of 256.

The worker stages inputs in the directory passed via `--workspace`.
On 500M, prepared prefixes and reward factors live under
`$workspace/yambda-gflow-k3/tiger/yambda_500m_listens/sampled50/`. Checkpoints,
configuration and final validation/test metrics live under
`experiments/global-time/yambda-gflow-k3/tiger/yambda_500m_listens/sampled50/`.
On 50M the corresponding directories are
`$workspace/yambda-gflow-k3/<model>/yambda_50m_listens/` and
`experiments/global-time/yambda-gflow-k3/<model>/yambda_50m_listens/`.
To inspect the index-to-configuration mapping for a pair, run

```bash
python scripts/global-time/gflow_finetune_yambda.py list --model tiger --dataset yambda_500m_listens
```

Once the sampled50 data and factors are ready, run one training index per
GPU job. Repeating the same index in the same workspace and output directory
resumes an interrupted run.

```bash
python scripts/global-time/gflow_finetune_yambda.py gflow --model tiger --dataset yambda_500m_listens --workspace "$workspace" --index 0
```
