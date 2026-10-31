# TIGER: CE distillation on Yambda 50M and 500M

Runner: `scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py`.
For a paired comparison, use `--sampling reward` and
`--sampling curriculum_smooth_is --curriculum-epochs 5` with the same grid.
50M retains last128; 500M fine-tuning always uses a random target subset,
with sampled50 as default. K=3 refers to CE candidates, not training subset
size. See the [CE formulas and parameters](README.md#comparing-two-unbiased-estimates-of-ce-under-q). This separate experiment
starts from initial exp2 TIGER checkpoints and uses the same reward components
and CE calibration as the [Amazon experiment](README.md).

## Compact grid for 500M

The compact 500M grid uses four reward sets, powers `{1,0.25,16}`, rho
`{1,0.1}`, seed 42, K=3, and a five-epoch transition with q mixture 0.1.
The `reward` and `curriculum_smooth_is` modes produce 48 configurations in total.

Run from the repository root with the datasets and source assets available.
Prepare each reward component before training:

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
  --datasets yambda_500m_listens \
  --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
  --powers 1 0.25 16 \
  --rho-targets 1 0.1 \
  --sampling reward \
  --curriculum-epochs 5 \
  --cache-only \
  --cache-component sasrec \
  --device cuda
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
  --datasets yambda_500m_listens \
  --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
  --powers 1 0.25 16 \
  --rho-targets 1 0.1 \
  --sampling reward \
  --curriculum-epochs 5 \
  --cache-only \
  --cache-component interaction \
  --device cpu
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
  --datasets yambda_500m_listens \
  --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
  --powers 1 0.25 16 \
  --rho-targets 1 0.1 \
  --sampling reward \
  --curriculum-epochs 5 \
  --cache-only \
  --cache-component similarity \
  --device cpu
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
  --datasets yambda_500m_listens \
  --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
  --powers 1 0.25 16 \
  --rho-targets 1 0.1 \
  --sampling reward \
  --curriculum-epochs 5 \
  --list
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
  --datasets yambda_500m_listens \
  --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
  --powers 1 0.25 16 \
  --rho-targets 1 0.1 \
  --sampling reward \
  --curriculum-epochs 5 \
  --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

The commands above select direct q sampling. Repeat training with
`--sampling curriculum_smooth_is` for the second mode, retaining
`--curriculum-epochs 5`.

For TIGER 500M, up to 50 random targets per user are selected from the last
128; subset seed 42 is fixed across epochs, CE modes, and configurations. Each
selected target has context of up to 50 items. Set subset size using
`--train-random-n` (1–127) and seed using `--train-subset-seed`. Values of 0
and 128 or greater are disallowed to prevent fine-tuning silently switching to
full last128. These options do not affect 50M. Each sampled N, subset seed, and
CE mode uses corresponding result directories and compatible factor caches;
full last128 checkpoints and caches are not used for sampled50.

Inspect the selected grid with `--list` or `--count`. Pass the same
`--train-random-n` and `--train-subset-seed` to preparation, caches, training,
and aggregation. Reuse only factors whose recorded protocol and subset match.

Changing N or subset seed creates separate caches and runs. Resume accepts
only a checkpoint with the same protocol and subset.

Specify the compact grid explicitly to summarize it:

```bash
for sampling in reward curriculum_smooth_is; do
  python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
    --aggregate-only --datasets yambda_500m_listens \
    --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
    --powers 1 0.25 16 --rho-targets 1 0.1 --curriculum-epochs 5 --sampling "$sampling"
done
```

## Data and grid

Both datasets use the standard exp2 `prefix_target` layout:
`train_target_policy=last_n`, `train_last_n=128`, `max_history_len=50`.
This is the original pool of up to the last 128 **training targets** per
user; each target's history contains up to 50 previous positive events. 50M
uses the full pool; 500M uses a fixed random subset of up to N targets per user,
with N=50 by default. Selection is without replacement; users with fewer than N
targets retain all. Validation and test remain separate leave-one-out events
and are not shortened by training subsampling.

Positive events are `listen` with `played_ratio_pct >= 50`. As in GFlow, the
filter is expressed using `positive_criterion`: other events remain available
for interaction reward but do not become positive training targets. The exp2
configuration's limits and time window are retained. Expected source asset
sizes are 993,926 / 10,004,891 train-pool examples and 112,943 / 374,283 items
for 50M / 500M respectively. The number of selected sampled50 training examples
is saved in the manifest and result.json and depends on each user's available
targets. The full pool is built for preparation and validation, but is not
passed in full to 500M training.

By default there are **112 runs**, 56 per dataset:

| Parameter | Values |
|---|---|
| Rewards | `sasrec`, `interaction`, `similarity`, all three pairs, and the product of all three |
| Power | `0.25`, `1`, `4`, `16` |
| Rho, target CE/SFT gradient norm ratio | `1`, `0.1` |
| Seed | `42` |
| Sampling | `curriculum_smooth`, 3 samples per history |
| Smooth transition duration | 5 epochs |

`similarity` is hierarchical similarity based on RQ-VAE. `sasrec` uses the
teacher from the corresponding matched protocol: standard exp2 SASRec for 50M
and SASRec-SCE for 500M. Its transform is sigmoid with temperature 1. The power
applies to the entire product of selected rewards with exp4 epsilon floors;
the grid varies the overall power, not individual component powers.

## Smooth sampling

For each history, the normalized full-catalog reward is
`q(i|h) = R(h,i) / sum_j R(h,j)`. The catalog is sorted by descending `q` and
split in half by item count; ties preserve catalog order. The upper half has
`floor(N/2)` items, and the lower half contains the rest.

```text
p_t = (1 - alpha_t) * q_low + alpha_t * q_high
alpha_t = clip(global_microstep / transition_steps, 0, 1)
transition_steps = ceil(curriculum_epochs * batches_per_epoch)
```

Here `q_low` and `q_high` are separately normalized within their halves.
Initially only the lower half is sampled; the upper-half mass smoothly rises
to 1 over five epochs. After the transition, samples come from the upper half.
`--curriculum-epochs 0` selects the upper half immediately. Within a selected
half, probabilities are proportional to `q`.

For each history, **3 IID samples are drawn with replacement**, and their
`-log pi(i|h)` values are averaged. This is CE for the current mixture `p_t`,
without importance weights. History items and the positive target are not
excluded; the target is not separately forced into CE and is used in SFT.
Fewer samples speed up TIGER decoding, but full `q` computation and catalog
sorting are still performed.

The final loss is `L_SFT + lambda_D * CE`. `lambda_D` is calibrated once
using gradient norms on 5 batches. Batch 256 × accumulation 8, AdamW lr 3e-5,
weight decay 0.01. Validation occurs every epoch using NDCG@5 with beam 5;
500M uses a fixed 25% of validation (seed 0), while 50M uses the full split.
The initial checkpoint is eligible for best selection. Patience is 5 after the
first 10 epochs, maximum 500. Final validation/test are full: beam 20,
HR/NDCG/Coverage/ARP at K=5,10,20.

## Running

For the complete default grid, use the component preparation and training
commands above without the compact-grid `--combos`/`--powers` restrictions.
Select datasets with `--datasets`. Use `--index N` for one selected configuration.
Completed runs are skipped and compatible unfinished runs resume optimizer,
RNG, and curriculum state. `last.pt` is saved every ten optimizer updates and
at epoch end. Use identical options during aggregation.

## Resources

SASRec factor preparation uses GPU; interaction and hierarchical similarity
factors use CPU. Finish all preparation before training. At batch 256, one FP32
catalog matrix occupies about 110 / 366 MiB for 50M / 500M. Sorting, mixture,
encoder, and gradient buffers require additional memory. Subsampling follows
source-pool construction, so reducing target count does not proportionally
reduce preparation peak RAM. Measure actual host/device peaks for each setting.

## Caches and results

New caches:

```text
artifacts/exp7-ce-distillation-tiger-yambda/yambda_50m_listens/last128/reward_factors/<component>/
artifacts/exp7-ce-distillation-tiger-yambda/yambda_500m_listens/sampled50_seed42/reward_factors/<component>/
```

Compatible previous GFlow caches can be reused after protocol validation:
full last128 for 50M and only the matching sampled subset for 500M. Automatic
reuse requires ready factors for all three components and available original
config JSON; otherwise a separate exp7 directory is selected. Data config,
subset, row count, target metadata, and catalog order are checked. Full-last128
and last8 caches for 500M are incompatible; matching row count alone is
insufficient—N, subset seed, and targets must match. `--cache-root` selects a
different parent directory; power/rho/seed do not require separate caches.

Results of each run:

```text
experiments/exp7-ce-distillation-tiger-yambda/yambda_50m_listens/<combo>/<ce_mode>/pow_<p>/rho_<r>/seed_<s>/
experiments/exp7-ce-distillation-tiger-yambda/yambda_500m_listens/sampled50_seed42/<combo>/<ce_mode>/pow_<p>/rho_<r>/seed_<s>/
```

For the product of three rewards, `<combo>/` is omitted, as in TIGER Amazon.
Contents: `identity.json`, `resolved_config.json`, `data_reward_config.json`,
`calibration.json`, `training_log.jsonl`, `training_history.json`,
`checkpoints/{best,last}.pt`, `result.json`. Metrics and weights are in
`experiments/`; `artifacts/` holds reward caches.

Aggregate results after training completes:

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py --aggregate-only
```

`metrics.csv` and `summary.json` appear in the summary branch specified by
the runner; when 500M is enabled, it is further divided by subset N and seed.
Previous last128 results remain on disk and do not count as completed sampled50
runs. If the grid, target count, subset seed, CE samples, or curriculum changed,
pass the same flags during aggregation. Power/rho are selected separately for
each reward combination by full validation NDCG@5, not by test.

`--list`, `--count`, and `--aggregate-only` do not load ML packages or
datasets. `--smoke` limits train/eval batches and isolates results, but **still
builds the full original dataset**. This is not a small local check for Yambda
500M; run it only on a machine with sufficient host memory.
