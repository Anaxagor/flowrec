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

## Fine-tuning TIGER and GPT-2 on Beauty 2014 and Instruments 2018

Both models support 108 GFlow configurations, three SFT-only seeds (42, 43
and 44) and six CE configurations per dataset. GFlow and CE use seed 42.
To list the GFlow and SFT configurations on Beauty 2014, run

```bash
python scripts/global-time/finetune_amazon_tiger.py list --dataset beauty
python scripts/global-time/gflow_finetune_gpt2.py list --dataset beauty
```

For training, pass an absolute shared directory as `--workspace` and use the
same model and dataset throughout. Run `prepare` before `sft` or `gflow`.
TIGER GFlow also requires the `factors` stage for each of indices 0, 1 and 2.
Pass `--index` to select the training configuration from the list.

Each GFlow sampler uses the product of the SASRec, interaction and hierarchical
similarity rewards, powers 0.25, 1 and 16, gradient ratios 1 and 0.1, and the
TB, DB and SubTB losses. The original four samplers are uniform random, on-policy,
SASRec curriculum and reward-weighted q. Both models also support

- `q_curriculum` (indices 72–89). Draw three negatives without replacement
  with weights proportional to q raised to α. The same ten-epoch schedule used
  for the SASRec curriculum moves α linearly from −1 to +1. The distribution
  is uniform halfway through and reaches q at the end.
- `policy_random` (indices 90–107). Draw three distinct negatives from the
  current student, then three uniformly from the remaining eligible items.
  This gives six distinct negatives per prefix. Both stages exclude the target
  and items in the input history.

Preparation uses the global train-only init for each model and the pinned time
cutoffs recorded with the checkpoints. Both models use all eligible training
targets. GPT-2 reproduces the history seen by each active label in its original
dense-causal window. TIGER uses the rolling prefixes from its initial training.
Validation scores determine which checkpoint is used for test evaluation. CE
uses the q-only protocol described below. Results and checkpoints are written
under `experiments/global-time/`, in
`tiger-gflow-k3/tiger/<dataset>/`, `gpt2-gflow-k3/<dataset>/` and
`ce-distill-q-k10/<model>/<dataset>/`.

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

To list the six configurations for one pair, run
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

Run the stages for each model and dataset in an absolute shared
scratch directory. For example, the following prepares GPT-2 on Yambda 500M.

```bash
python scripts/global-time/gflow_finetune_yambda.py prepare --model gpt2 --dataset yambda_500m_listens --workspace "$workspace"
```

Across the four model/dataset pairs there are 72 GFlow configurations, 12
SFT-only controls and 16 preparation tasks. For each pair the `sft` stage can
start after `prepare`. Before starting `gflow`, run the `factors` stage once
per component with `--index 0`, `1` and `2`. Pass the same model, dataset
and workspace to every stage.

| Dataset | Global init checkpoint | Train targets | SFT-only / GFlow walltime per cell | Training GPU |
|---|---|---:|---|---|
| Yambda 50M | TIGER | 951,675 | 4 h / 24 h | V100 32 GB |
| Yambda 50M | GPT-2 `last_128` | 951,675 | 8 h / 24 h | A100 or H100 80 GB |
| Yambda 500M | TIGER | 4,026,761 | 24 h / 48 h | A100 or H100 80 GB |
| Yambda 500M | GPT-2 `last_128/trial_17.pt` | 4,026,761 | 24 h / 72 h | A100 or H100 80 GB |

Two TIGER LOO runs on 4,162,653 sampled targets took 14.34 h on H100 (SASRec) and
26.48 h on A100 (SASRec × similarity). Both used DB, 10 negatives and 23
epochs. For GPT-2, full LOO runs with all three factors and three negatives
on 993,926 Yambda 50M targets took 12.49–16.67 h for 15 epochs. Scaling by
the number of targets gives 51–68 h on Yambda 500M. The GPT-2 500M LOO
SFT-only control took 6.67 h for 16 epochs. The global 500M checkpoint uses
a different architecture, so these timings are estimates.
The separate CPU interaction cache job has a 24 h budget. Building the LOO
sampled50 cache took about 7.9 h.

On 50M, both arms use all of each user's last 128 eligible pre-validation
targets. On 500M, start from the same last-128 pool (9,560,668 targets).
For each user with more than 50 eligible positions, sample 50 uniformly
without replacement. Keep every eligible position for shorter histories.
All SFT seeds, GFlow configurations and reward caches share the sample drawn
with seed 42. For the original 500M inputs, the expected `source_indices_sha256`
in `preparation/layout_audit.json` under `finetune_subset` is
`25a03aa05c4ec4466015b668edc77f96d3ba5901d5096024bbe44eb7ff542db1`.
Each target keeps its original prefix of up to 50 preceding
items. The GPT-2 checkpoints trained with one active final label per dense
window, stride one. The 500M student starts from
`checkpoints/exp2/global_split/initial/gpt2/yambda_500m_listens/last_128/trial_17.pt`.
Before the fine-tuning subset is drawn, CPU preparation checks each
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
`$workspace/yambda-gflow-k3/<model>/yambda_500m_listens/sampled50/`. Checkpoints,
configuration and final validation/test metrics live under
`experiments/global-time/yambda-gflow-k3/<model>/yambda_500m_listens/sampled50/`.
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

## CE distillation on Yambda global time splits

CE can reuse the prepared GFlow prefixes and reward factors for the same
student and dataset. To prepare these inputs and train the first configuration,
set `workspace` to an absolute shared scratch directory and run

```bash
python scripts/global-time/ce_finetune_yambda.py prepare --model tiger --dataset yambda_500m_listens --workspace "$workspace"
for factor in 0 1 2; do
    python scripts/global-time/ce_finetune_yambda.py factors --model tiger --dataset yambda_500m_listens --workspace "$workspace" --index "$factor"
done
python scripts/global-time/ce_finetune_yambda.py ce --model tiger --dataset yambda_500m_listens --workspace "$workspace" --index 0
```

CE starts from the same global initial checkpoint and uses the same training
prefixes as GFlow. Both models use 951,675 last-128 targets on 50M and 4,026,761
sampled50 targets on 500M. The observed target supplies the SFT loss. The CE loss uses
three independent draws with replacement from the powered product of SASRec
(SASRec-SCE on 500M), interaction and hierarchical SID similarity rewards.
The distribution is normalized over catalog items excluding the observed target
and the items in the input history. The sampled CE has no importance correction.
The grid crosses powers 0.25, 1 and 16 with calibrated CE-to-SFT gradient-norm
ratios 1 and 0.1, using seed 42. To list the six configurations for one model and dataset, run

```bash
python scripts/global-time/ce_finetune_yambda.py list --model tiger --dataset yambda_500m_listens
```

CE checkpoints and metrics are written under
`experiments/global-time/yambda-ce-q-k3/<model>/<dataset>/`, with an additional
`sampled50/` directory on 500M.

| Dataset and model | Training GPU | Estimated time budget per CE run | Observed LOO walltime with K=3 |
|---|---|---:|---|
| Yambda 50M, TIGER | V100 32 GB | 12 h | 2.48–6.64 h with three factors and a smooth curriculum |
| Yambda 50M, GPT-2 | A100/H100 80 GB | 16 h | 8.87–10.00 h for six three-factor runs sampled from q |
| Yambda 500M, TIGER | A100/H100 80 GB | 36 h | 22.63 and 23.03 h for three-factor sampled50 runs on A100 |
| Yambda 500M, GPT-2 | A100/H100 80 GB | 72 h | 15.13–41.81 h for single-factor sampled50 runs on A100 |

The LOO times include setup, fit and final evaluation on 993,926 targets (50M)
or 4,162,653 (500M). The GPT-2 500M limit allows additional time to score all
three reward factors. Global runs condition q on negatives and can stop at a
different epoch.

## Beauty 2023 global fine-tuning

Each model shares its prepared prefixes across SFT, GFlow and CE. Run
`prepare` once for the model, then run `factors` for indices 0, 1 and 2 before
GFlow or CE. SFT only needs the prepared prefixes. Pass the same absolute
`--workspace` to each stage. For example, these commands prepare the data
and train the first SFT seed for TIGER.

```bash
python scripts/global-time/finetune_beauty2023.py prepare --model tiger --workspace "$workspace"
python scripts/global-time/finetune_beauty2023.py sft --model tiger --workspace "$workspace" --index 0
```

| Method | Configurations per model |
|---|---|
| SFT-only | 3 seeds (42, 43, 44) |
| GFlow | 3 powers × 2 ratios × 3 losses (TB, DB, SubTB) |
| CE distillation | 3 powers × 2 ratios |

The students start from `checkpoints/exp2/global_split/initial/tiger/beauty_2023/trial_31.pt` and `checkpoints/exp2/global_split/initial/gpt2/beauty_2023/best.pt`. The global SASRec teacher is `sasrec/beauty_2023/best.pt` under the same initial-checkpoint root.

The verified-purchase 5-core and the two archived cutoffs yield 3,760,037 training targets, 206,547 validation targets, 158,576 test targets and 150,480 catalog items after cold filtering. TIGER uses rolling prefixes with up to 20 preceding items. GPT-2 reconstructs all active labels in the original 621,963 dense-causal windows, preserving the context from the start of each window. The scratch workspace stores prefixes and factors separately for the two models. Interaction rewards use training-period events.

Both reward-based methods use one product of SASRec, interaction and hierarchical SID similarity, with total power 0.25, 1 or 16 and initial auxiliary-to-SFT gradient-norm ratio 1 or 0.1 (seed 42). GFlow draws three distinct negatives per prefix using the SASRec curriculum. CE draws three negatives independently with replacement from the powered reward distribution `q`, excluding the target and items in the input prefix, and averages their negative log probabilities without importance weights. All three methods select by validation NDCG@5 and score the selected weights on the full held-out splits.

Checkpoints and metrics are saved under `experiments/global-time/beauty2023-global-k3/{gflow-sft,ce}/<model>/beauty_2023/`. This command prints the training configurations and their indices for one model.

```bash
python scripts/global-time/finetune_beauty2023.py list --model gpt2
```

## Continue the global TIGER Yambda 500M initial checkpoint

Set `workspace` to an absolute directory for the original last-128 train and
validation splits. Prepare the data, then continue from the `dropout-0`
checkpoint.

```bash
python scripts/global-time/continue_tiger_yambda500m.py prepare --workspace "$workspace"
python scripts/global-time/continue_tiger_yambda500m.py train --workspace "$workspace"
```

Training restores the model and AdamW state saved at the end of epoch 7 and
runs 60 additional SFT epochs at `lr=0.0018511868508356536`.
It checks validation NDCG@10 at cumulative epochs 17, 27, 37, 47, 57 and 67 using
the HPO trial's fixed 25% validation sample and 10 beams. Early stopping is disabled.

Under `checkpoints/exp2/global_split/initial/tiger/yambda_500m_listens/last_128/dropout-0_plus60/`,
`best.pt` contains the winner of the six validation checks, `last.pt` stores the latest
completed epoch, and `result.json` records the selection after training finishes.

## Global initial checkpoints on train + validation

For each model and dataset, select the completed global HPO trial with the
highest validation NDCG@10 among the eligible trials for GPT-2, TIGER or the
applicable SASRec variant. Record its configuration, seed and duration. The
number of training epochs is one plus the zero-based `best_epoch`.

Combine training and validation interactions up to the original test cutoff,
preserving the warm-item catalog, rolling histories and held-out test targets.
GPT-2 and TIGER on Yambda select the last 128 eligible targets per user in
this combined period. Train a fresh initial model for the selected number of
epochs and evaluate its terminal weights on the full test set.

`experiments/global-time/initial_train_with_validation/<model>/<dataset>/<trial>-<id>/result.json`
records the winning trial, epochs, test size and HR, NDCG, Coverage and ARP at 5, 10 and 20.
The terminal checkpoint is `checkpoints/last.pt` inside that run directory.

## Final train+validation refits on the global splits

The fine-tuning refit protocol uses the original validation-selected runs for
the following methods and datasets.

| Original global experiments | SFT-only and GFlow | CE distillation |
|---|---|---|
| Beauty and Instruments | GPT-2, all four original GFlow samplers, K=3 | TIGER and GPT-2, reward negatives, K=10 |
| Beauty 2023, Yambda 50M and Yambda 500M | TIGER and GPT-2, SASRec curriculum, K=3 | TIGER and GPT-2, reward negatives, K=3 |

Each original checkpoint has three independent refits with seeds 42, 43 and 44.
A source SFT-only seed identifies a separate original checkpoint with its own epoch
count. The complete grid contains 336 original checkpoints and 1008 refits.
Each refit starts from the same global **init** weights as its original fine-tuning
run and trains for that run's recorded `epochs_completed`, including epochs after
its best validation score. Training uses the source optimizer, batching and method
settings. Reward-based methods calibrate the auxiliary loss weight on the new
train+validation prefixes for the source target ratio.

Training uses eligible positive events strictly before the original test cutoff,
including the full validation period. The original item catalog, semantic IDs,
frozen reward teacher, and test examples are preserved. GPT-2 uses the prefix
layout corresponding to its init checkpoint's dense labels. Yambda selects the
last 128 eligible target positions per user at the new cutoff. Yambda 500M
samples up to 50 from this pool with the original subset seed 42, shared across
refit seeds. Prefixes and three reward factors are rebuilt once per model and
dataset in a shared scratch directory. SFT needs only the prefixes.
Each training run can resume from its own optimizer checkpoint.

The final epoch is evaluated on the entire original test cohort. The shared file
`experiments/global-time/refit_global_train_with_val/metrics_global_train_with_val.jsonl`
has one row per source checkpoint and refit seed. Each row records the dataset,
model, both seeds, source artifacts and configuration, refit settings, epoch count,
data audit, and test HR, NDCG, Coverage and ARP at 5, 10 and 20. Writes are atomic
across concurrent runs. Per-run results and `checkpoints/last.pt` are under
`experiments/global-time/refit_global_train_with_val/runs/<source_id>/refit_seed_<seed>/`.
