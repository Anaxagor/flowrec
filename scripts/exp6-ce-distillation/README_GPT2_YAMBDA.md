# GPT-2: CE on Yambda 50M and 500M

Runner: `scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py`.

## Yambda 500M, trial 4, sampled50

The sampled50 protocol uses matched initial GPT-2 **trial 4**, selecting up to
50 targets per user from the last 128 with subset seed 42. Four reward sets,
three powers `{1,0.25,16}`, two rho values `{1,0.1}`, and two sampling modes
produce 48 configurations at seed 42. K=3, transition length five epochs,
and full-q proposal mixture 0.1 are shared across modes.

Run from the repository root with the datasets and source assets available.
Prepare each reward component before training:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --prepare-data-only \
  --device cpu \
  --num-workers 0
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --cache-only \
  --cache-component sasrec \
  --device cuda
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --cache-only \
  --cache-component interaction \
  --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --cache-only \
  --cache-component similarity \
  --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --require-prepared-data \
  --list
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --datasets yambda_500m_listens \
  --train-random-n 50 \
  --train-subset-seed 42 \
  --require-prepared-data \
  --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

Repeat training with `--sampling curriculum_smooth_is` for the second mode.
The default `reward` mode uses direct q sampling. Both use the same prepared
snapshot and compatible reward factors.

Sampled50 results, checkpoint, and caches are stored separately from sampled15:

```text
experiments/exp7-ce-distillation-gpt2-yambda/yambda_500m_listens/sampled50_seed42/
artifacts/exp7-ce-distillation-gpt2-yambda/sampled50_seed42/yambda_500m_listens/
```

Results also include `cache_build/` with configurations and preparation
audit. Artifacts contain `reward_factors/` and `source_assets/` with the teacher
ID mapping. Training starts from the original trial 4; resumption requires a
compatible checkpoint from the same sampled50 configuration. Sampled15 runs
are a separate experiment.

## General grid

For the default 50M last128 and 500M sampled15 protocols, omit
`--train-random-n 50`. Run the same preparation, cache, and training phases for
each selected dataset. Compact preparation is supported for 500M only.

## Models and data

| Dataset | Initial GPT-2 | Training targets |
|---|---|---|
| `yambda_50m_listens` | matched exp2, trial 36 | All last 128 per user, as in TIGER CE |
| `yambda_500m_listens`, default protocol | matched exp2, trial 4 | Up to 15 random targets from the last 128 per user; fixed subset seed 42 |
| `yambda_500m_listens`, sampled50 protocol | matched exp2, trial 4 | Up to 50 random targets from the last 128 per user; fixed subset seed 42 |

The selected source is trial 36 for 50M and trial 4 for 500M. For the legacy checkpoint layout, CE first checks
`checkpoints/exp2-tuned_params/gpt2_tiger_matched/<dataset>/trials/trial_<N>/best.pt`.
If that exact checkpoint is absent, it uses the migrated local path
`checkpoints/exp2/loo_split/initial/gpt2/<dataset>/last_128/best.pt`.
The source's `resolved_config.json` and saved data audit (embedded in
`resume_identity.data_audit` or the same layout's `data_audit.json`) are also
required. Metadata are never mixed between layouts, and no other trial or
full-history checkpoint is substituted. The saved trial architecture is used;
trial 4 also checks all weights for NaN/Inf. The original optimizer state is
not transferred.

SASRec similarly uses the original
`checkpoints/exp2-tuned_params/initial/<sasrec|sasrec_sce>/<dataset>/best_model.pt`
and its sibling `best_trial.json` when the checkpoint exists. Original paths
take precedence even if both layouts are present, so existing CE identities
and reward-cache provenance remain valid. This compatibility is installed by
the CE entry point; it does not modify the shared GFlow implementation or the
compact-data identity and does not require rebuilding the ready TIGER snapshot.

The dataset uses `prefix_target` and the selected checkpoint's recorded
`dense_causal_match_prefix` protocol. This already matches full prefix histories:
up to 50 previous positive events per target. Targets, contexts, catalog,
semantic IDs, and held-out splits are checked against the original trial's
audit. Here `train_prefix_from_dense=False`: the window conversion used in
older Amazon experiments does not match the selected Yambda trials.

For 500M, random targets are selected **after** building and validating the
full last128 pool. Selection is fixed across epochs, CE modes, powers, and rho;
it does not shorten target histories or alter validation/test. In the general
runner, `--train-random-n` sets the size (1..127, default 15), and
`--train-subset-seed` sets the seed (default 42). These options apply only to
500M. Use `--train-random-n 50` for sampled50; other values select separate subsets. If a user has fewer than N targets, all are retained. Sampling
training targets and sampling 3 items from the reward distribution are distinct.

Positive events and filters come from the saved trial. Additional events are
retained for interaction reward. SASRec uses the same teacher as matched GFlow;
item IDs are checked against original raw IDs before building its cache. The
teacher checkpoint, trial JSON, and `training_summary.json` are required, as is
the relevant RQ-VAE asset for hierarchical similarity. Validating the teacher
catalog may require building its dataset.

## Grid and CE

By default there are **96 runs**, 24 per dataset and mode:

| Parameter | Values |
|---|---|
| Rewards | `sasrec`, `interaction`, `similarity`, `sasrec_x_interaction_x_similarity` |
| Overall reward power | `1`, `0.25`, `16` |
| Rho | `1`, `0.1` |
| Sampling | `reward`, `curriculum_smooth_is` |
| Samples per history | 3 IID with replacement |
| Training seed | 42 |
| Smooth transition | 5 epochs |
| Full-q mixture | 0.1 |

`similarity` means hierarchical similarity. For the triple, the power is
applied to the product of all three components using exp4 epsilon floors;
component powers are not varied independently.

In `reward` mode, items are sampled directly from the full
`q(i|h) ∝ R(h,i)`. In `curriculum_smooth_is`, the catalog is sorted by
descending q and split in half by item count. Each half is normalized separately:

```text
alpha = clip(global_microstep / transition_steps, 0, 1)
c = (1 - alpha) * q_low + alpha * q_high
m = 0.9 * c + 0.1 * q
sampled CE = mean[-q(i) / m(i) * log pi(i|h)],  i ~ m
```

Initially the mass favors low rewards; it shifts to high rewards over five
epochs. The full-q mixture retains support; raw q/m weights give an unbiased
estimate of CE under q. Weights are neither clipped nor self-normalized.
History items and the positive target are not excluded from q.

The total loss is `L_SFT + lambda_CE * CE`. Rho sets the target CE/SFT
gradient norm ratio for one-time lambda calibration on five batches. Both modes
calibrate using direct q sampling. Training uses AdamW, lr `3e-5`, weight
decay `0.01`, and BF16 on GPU. Default batch × accumulation is 128 × 16 for
50M and 256 × 8 for 500M.

Validation runs each epoch and selects checkpoints by NDCG@5 with beam 5.
50M uses full validation; 500M uses a fixed 25%, seed 0. The initial checkpoint
is eligible for best selection; patience is 5 after the first 10 epochs, with a
maximum of 500 epochs. Final evaluation uses full validation/test, beam 20,
and HR/NDCG/Coverage/ARP at K=5,10,20.

## Running and resuming

`--curriculum-epochs` changes the transition duration. `--list` and `--count`
inspect the grid without loading ML packages or datasets:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py --list
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py --count
```

One invocation selects one sampling mode; the default count is 48 across both
datasets. Completed compatible runs are skipped. Unfinished runs restore
optimizer, RNG, curriculum, and progress from `last.pt`, saved every ten
optimizer updates and at epoch end. Subset size/seed form part of the identity.
`--smoke` limits training but still prepares the full source dataset.

## Prepared data and result locations

Compact 500M input data are prepared on CPU and loaded
via mmap during training; raw interactions, the full last128 pool and Python
history lists are not retained by GPU workers. The full source audit is
performed during CPU preparation and preserved with the selected row indices.
Loading a committed snapshot checks its data configuration, subset, source
audit and artifact integrity. Changed code hashes or exact package versions
only warn; compatible existing snapshots can be reused without rebuilding.
To actually change the prepared examples after a preprocessing-code change,
prepare under a fresh `--prepared-data-root`.
See [500M memory handling](README_YAMBDA.md#avoiding-repeated-yambda-500m-preprocessing).
Prepare compact data before reward caches. SASRec factor construction uses GPU;
interaction and similarity factors use CPU. Run training only after all required
components are ready. Preparation builds the full last128 pool before selecting
targets, so host RAM does not scale proportionally with sampled target count.

Reward caches and teacher ID mappings for the default subsets:

```text
artifacts/exp7-ce-distillation-gpt2-yambda/<dataset>/reward_factors/<component>/
artifacts/exp7-ce-distillation-gpt2-yambda/<dataset>/source_assets/
```

Metrics, logs, and trained weights:

```text
experiments/exp7-ce-distillation-gpt2-yambda/<dataset>/<combo>/<mode>/pow_<p>/rho_<r>/seed_42/
```

For the product of three rewards, `<combo>/` is omitted. For sampled50,
`sampled50_seed42/` is added after `<dataset>/` in the run path and before it
in the cache path, as shown above. `<mode>` is `sampled_n3` or
`curriculum_smooth_is_n3_epochs5_qmix0p1`. Contents include `result.json`,
`training_history.json`, `training_log.jsonl`, `calibration.json`, configs,
and `checkpoints/{best,last}.pt`. The baseline is saved separately in
`<dataset>/baseline/init_metrics.json`.

Summaries of completed runs:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py --aggregate-only
python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
  --aggregate-only --sampling curriculum_smooth_is
```

`metrics.csv` and `summary.json` appear in
`experiments/exp7-ce-distillation-gpt2-yambda/ablations/<mode>/`.
If the grid or sampling parameters change, pass the same flags.
For the new sampled50 grid:

```bash
for sampling in reward curriculum_smooth_is; do
  python scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py \
    --aggregate-only --datasets yambda_500m_listens --train-random-n 50 \
    --train-subset-seed 42 --sampling "$sampling"
done
```

The sampled50 summary is saved separately to avoid overwriting the sampled15
summary. The shared `report_ce_results.py` also includes the GPT-2 Yambda root.
