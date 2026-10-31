# Global-time-split configs

> **Availability:** Slurm launchers mentioned here are local cluster files and are not distributed with this repository. Their examples require a separately supplied launcher. The Python entry points are available in [`scripts/global-time/`](../../scripts/global-time/README.md); for options run `python scripts/global-time/tuning_gpt2.py --help` or `python scripts/global-time/tuning_sasrec.py --help`.

This configuration tree defines standard-parameter runs, hyperparameter
tuning, selected-model training and fine-tuning on a **global temporal
split**. Its configurations and outputs are separate from leave-one-out.

```
configs/global-time/
  exp1-standard_params/    dataset_*.yaml  *_model.yaml  trainer_*.yaml
  exp2-tuned-params/       dataset_*.yaml  finetune_base_matched.json
```

Consult the individual protocol configurations when comparing runs across
split strategies.

## Protocol

| | |
|---|---|
| Split | `split_strategy: global_time` — train = positive interactions before `global_val_timestamp`, validation = `[val, test)`, test = from `global_test_timestamp` on |
| Cutoffs | 80 % / 90 % interaction quantiles of each dataset's positive 5-core stream, **pinned as timestamps** per dataset |
| History | `global_history_policy: rolling` — a target sees every earlier event of its user, including earlier events of the same window |
| Targets | `global_eval_target_mode: last_per_user` — one target per user active in the window (its last event there) |
| Catalog | `filter_cold_items_after_split: true` on every dataset — train-observed items only; cold targets are dropped for every model |
| VK-LSVD | Its `provided` weekly split is a global temporal split |

The two trees share many preprocessing and training settings. They are not
fully identical: in particular, the leave-one-out standard GFlow grid has
seven reward compositions (168 cells per model/dataset), while the global-time
grid specifies two compositions (48 cells). Check the individual configuration
files before comparing runs.

The cutoffs are pinned instead of re-derived from `global_val_ratio` /
`global_test_ratio` so every model, representation job, teacher and reward
cache uses the same windows. `drop_items_missing_semantic_id: false` keeps
SID coverage from changing the pre-split stream: the normal cold-item filter
removes unavailable items after target selection.

### Cutoffs and measured split sizes

Measured on the local raw data (SASRec view; Beauty and
Instruments also through the TIGER prefix layout, with identical catalogs and
evaluation sets).

| Dataset | val cutoff | test cutoff | Interactions | Val / test targets | Cold items removed | Test targets lost to cold items |
|---|---|---|---|---|---|---|
| beauty | 1394150400 (2014-03-07) | 1399939200 (2014-05-13) | 198,502 | 6,073 / 4,732 | 328 of 12,101 | 1,456 of 6,188 (24 %) |
| instruments | 1488499200 (2017-03-03) | 1507334400 (2017-10-07) | 184,897 | 5,982 / 5,779 | 87 of 8,556 | 227 of 6,006 (4 %) |
| beauty_2023 | 1637083975718 ms (2021-11-16) | 1661181715044 ms (2022-08-22) | 5,480,873 | ~228k / ~201k (before the cold filter) | ~12.5k of 163,050 | not measured |
| yambda_50m_listens | 25016630 (−11.4 d) | 25520600 (−5.5 d) | 7,018,513 | 7,148 / 7,044 | 898 of 133,747 | 181 of 7,225 (3 %) |
| yambda_500m_listens | 25001165 (−11.6 d) | 25510950 (−5.7 d) | 72,531,548 | ~73.7k / ~72.9k (before the cold filter) | ~3.2k of 434,657 | not measured |

Yambda timestamps are seconds on the dataset's own clock; the offsets are
days before the last event of the `hf_recent_months: 2` window. The
beauty_2023 and yambda_500m rows come from lightweight replications of the
pre-split pipeline (filters → positive criterion → 5-core, plus the window
and consecutive dedupe on Yambda), because those datamodules do not fit a
laptop. Both replications are validated: the Yambda one reproduces the real
datamodule exactly on 50m (interactions, users, items, cutoffs), the Amazon
one reproduces the documented leave-one-out beauty_2023 numbers (662,540
users, 163,050 items, 3,493,253 train examples). Their target counts are
before the cold-item filter.

One rule, five datasets — realized interaction shares under the pinned
cutoffs (each pinned value was re-checked to equal the 80 % / 90 % quantile of
its stream):

| Dataset | Train | Val | Test | Val window | Test window |
|---|---|---|---|---|---|
| beauty | 79.94 % | 9.91 % | 10.15 % | 67 d | 71 d |
| instruments | 79.94 % | 10.04 % | 10.02 % | 218 d | 356 d |
| beauty_2023 | 80.00 % | 10.00 % | 10.00 % | 279 d | 386 d |
| yambda_50m_listens | 80.00 % | 10.00 % | 10.00 % | 5.8 d | 5.5 d |
| yambda_500m_listens | 80.00 % | 10.00 % | 10.00 % | 5.9 d | 5.7 d |

The rule fixes the share of interactions, not the wall-clock length: Amazon
review streams span 12-23 years and thin out slowly, Yambda is a dense 61-day
window. (Beauty / Instruments deviate by < 0.2 pp because their timestamps are
whole days, and a day cannot be split.)

**VK-LSVD is the one exception**: it keeps the authors' weekly split (train
weeks 0-24, validation week 25, test week 26) instead of the 80/10/10
interaction rule. Same family (global temporal, rolling histories,
last_per_user targets, train-observed catalog), different cutoff rule. Its
data is not downloaded and it is wired into exp1 only, so nothing can be
pinned yet; to bring it under the study-wide rule, set `split_strategy:
global_time` in `dataset_vk_lsvd.yaml` and pin the two quantiles of its
(synthetic, order-preserving) timestamps once the subsamples are local.

TIGER train rows under this split: beauty 137,864, instruments 127,025,
beauty_2023 3,760,037, yambda_50m 951,675, yambda_500m ~9.56M (last-128
layout).

Two things worth knowing before reading results:

* **Fewer evaluated users than leave-one-out.** Only users active in a window
  get a target: roughly a quarter of the Beauty / Instruments users per
  window, a third on beauty_2023, ~81 % on Yambda.
* **Scoring every window event is not affordable on Yambda**: `all` would mean
  ~0.7M (50m) / ~7M (500m) 20-beam decodes per split, hence `last_per_user`.

## Semantic IDs for the final training catalog

The scripts in `scripts/global-time/` derive the item set from a fully set-up
datamodule with no semantic IDs attached. They use the existing dataset
config builders and preprocessing: metadata/embedding coverage, positive
criterion, deduplication, k-core, sequence layout, temporal split, then the
post-split cold-item filter. They do not move k-core filtering to after the
cutoff or rerun it on the reduced catalog. VK-LSVD keeps its provided weekly
split. Auxiliary-only items are excluded.

To keep the existing item-to-token mapping and export only the final catalog:

```bash
python scripts/global-time/train_tokenizer_all.py \
  --datasets beauty instruments beauty_2023 yambda_50m_listens yambda_500m_listens \
  --reuse-semantic-ids-dir datasets/semantic_ids/exp1-standard_params
```

This skips content encoding and tokenizer training, preserves every retained
token (including collision suffixes), and fails if any training item lacks
an SID. It writes `semantic_ids.json`, `semantic_ids.pt`, metadata and catalog
provenance; it does not create new tokenizer weights. The source codebooks
retain their original training scope. Use this mode when matching the final
item set is sufficient, not when tokenizer fitting must exclude future items.

To fit new tokenizers on the final training catalogs instead:

```bash
# All three Amazon datasets; the pretrained text encoder is unchanged.
python scripts/global-time/build_amazon_embeddings.py --device cuda

# Every configured dataset, including both Yambda sizes and VK-LSVD subsamples.
python scripts/global-time/train_tokenizer_all.py --device cuda
```

Both scripts accept `--datasets` to run a subset (`all` is the default),
`--config-dir`, and `--semantic-config`. The default semantic config is the
original `configs/exp1-standard_params/semantic_ids.yaml`: text construction,
encoder, RQ-VAE settings, normalization and collision handling are reused
directly. Yambda and VK-LSVD load their existing audio/video embedding sources
and restrict them to the final catalog. Amazon embeddings are checked against
that catalog again before fitting, so extra vectors cannot enter training.

Default outputs:

* `datasets/embeddings/global-time/exp1-standard_params/<dataset>_content_embeddings.pt`
* `datasets/semantic_ids/global-time/exp1-standard_params/<dataset>/`
* `artifacts/global-time/exp1-standard_params/semantic_id_training_report.json`

Training saves the RQ-VAE weights, normalizer, config, training log, SID JSON/PT
and `catalog.json` with the resolved preprocessing config and split statistics.
The combined report is updated after each dataset. For parallel subset jobs,
give each job a distinct `--report-path`. `--epochs` and `--max-items` are debug
overrides; a truncated catalog is not suitable for full downstream training.
The existing SID dictionaries currently cover the five downloaded Amazon /
Yambda datasets; VK-LSVD requires its source data (and existing SIDs for reuse).

**To use separately generated global-time IDs, pass `--semantic-ids-dir` explicitly.**
The global-time GPT-2 tuner reuses `datasets/semantic_ids/exp2-tuned-params`
by default and restricts the mapping to the training catalog after splitting.
Keep one SID folder for each base checkpoint and every fine-tuning arm derived
from it. Exporting exp2's existing tokens into a separate folder is optional;
use `--reuse-semantic-ids-dir datasets/semantic_ids/exp2-tuned-params` and
`--output-dir datasets/semantic_ids/global-time/exp2-tuned-params` for that export.

The global-time configs disable pre-split SID filtering. Otherwise a dictionary
containing only training items would remove cold events before selecting each
user's last target, changing evaluation sets and potentially k-core membership.
Cold items continue to be removed by the same post-split filter used without
SIDs. Synthetic tests cover this for Amazon, Yambda and VK-LSVD.

## Output roots

Global-time writes remain separate from the selected leave-one-out checkpoints. Raw weights live under `experiments/global-time/checkpoints/`:

```
checkpoints/exp1/tiger                  ->  experiments/global-time/checkpoints/exp1-standard_params/tiger
experiments/sasrec/yambda                ->  experiments/global-time/sasrec/yambda
artifacts/exp1-standard_params           ->  artifacts/global-time/exp1-standard_params
```

Selected global-time bases live under `checkpoints/exp2/global_split/initial/<model>/<dataset>/best.pt`. GPT-2 Yambda 50M uses the `full_history/best.pt` subfolder, matching the saved all-target dense-window protocol. Its global 500M trial and the TIGER Beauty 2023 trial have no selected winner and remain archived. Leave-one-out bases use `checkpoints/exp2/loo_split/initial/`. These splits require their own weights.

Original global-time weights are kept under `artifacts/checkpoint_archive/`. The global-time fine-tuning config points its base inputs at the selected global models and its outputs at `experiments/global-time/`. No global SFT-only model is currently selected.

## Full-cross-entropy SASRec tuning

`scripts/global-time/tuning_sasrec.py` tunes Beauty 2014, Instruments 2018,
Beauty 2023 and Yambda 50M with **100 trials per dataset**, maximizing
`val/sft/ndcg@10`. It uses the full-CE search spaces, baseline
configuration, optimizer settings, gradient accumulation, epoch limits,
early stopping and median pruning. No SCE loss is used and test targets
are not evaluated during tuning.

Every dataset uses its pinned global-time cutoffs and the normal post-split
filter: `filter_cold_items_after_split: true`, `cold_item_train_source:
observed`. The tuner refuses incompatible configs. Cold targets are dropped
after target selection, cold items are removed from histories and the item
vocabulary is compacted to the training catalog. The existing positive
criterion, metadata/embedding coverage filters, deduplication, k-core and
dense training layout follow the dataset configuration. SASRec does not read semantic IDs.

The Python experiment entry points are in `scripts/global-time/`. If the
optional local launcher is supplied separately, it can be submitted from the
repository root on the cluster:

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/tuning_sasrec_global_time_array.sbatch
```

The default array is **32 tasks = 4 datasets x 8 trial workers**, each with
one GPU. There are no partition, node-type or GPU-type restrictions. The
launcher uses the `flowrec` conda environment. Supply an appropriate project
account and override Slurm resource flags at submission if needed.

| Array indices | Dataset | Workers |
|---|---|---|
| 0–7 | beauty | 0–7 |
| 8–15 | instruments | 0–7 |
| 16–23 | beauty_2023 | 0–7 |
| 24–31 | yambda_50m_listens | 0–7 |

Each task repeatedly claims trials from one shared Optuna journal. The
budget is **100 total trials, not 100 per worker**; complete, pruned and
failed trials count toward it. Trial allocation is locked, so concurrent
workers cannot overshoot. The first worker enqueues the original baseline,
regardless of its worker ID. TPE uses distinct worker seeds and
`constant_liar` for concurrent suggestions.

For four workers per dataset (16 tasks), change both settings:

```bash
sbatch --array=0-15 --export=ALL,GLOBAL_SASREC_WORKERS=4 \
  scripts/slurm/tuning_sasrec_global_time_array.sbatch
```

An optional `%N` on `--array` limits total simultaneous tasks. A direct
worker invocation is also supported:

```bash
python scripts/global-time/tuning_sasrec.py \
  --datasets beauty --worker-id 0 --device cuda
```

All workers must use the same shared filesystem with working `flock`
support. Resubmitting an interrupted array resumes the journals; an orphaned
RUNNING trial restarts from its original parameters and seed under the same
trial number. This restarts the trial's training, not its last optimizer
step. Active trials retain an exclusive lock and cannot be stolen. Completed
studies exit without rebuilding the datamodule. A changed split or training
protocol is rejected when resuming an existing journal.

Per-dataset outputs are stored under `global-time/`:

* `experiments/global-time/checkpoints/exp2-tuned_params/sasrec/<dataset>/`: journal,
  `best_model.pt`, `best_trial.json`, `epoch_val_ndcg10.jsonl`,
  `all_trials.json`, `tuning_summary.json`, plus `study_config.json` and
  `data_summary.json` with the resolved split and cold-item counts.
* `artifacts/global-time/exp2-tuned-params/sasrec/`: per-dataset best-trial
  JSON, trial CSV and tuning-summary JSON.
* `configs/global-time/exp2-tuned-params/sasrec_model_<dataset>.yaml`:
  winning architecture, optimizer and full-CE SFT settings.

Best checkpoints are promoted as trials finish; temporary trial checkpoints
are removed after publication. Shared outputs are locked and atomically
replaced, so an older worker cannot overwrite a newer winner. Failed/OOM
trials are recorded; OOM and non-finite losses prune that trial and let the
worker continue. Unexpected exceptions mark the trial failed and stop its
worker. Array jobs write per-dataset summaries, avoiding a shared summary
race across datasets.

`--config-dir`, `--checkpoint-root`, `--artifacts-dir` and
`--tuned-config-dir` allow alternate locations. `--n-trials`, `--max-epochs`,
`--train-batches-per-epoch` and `--max-eval-batches` allow small smoke runs;
use separate output roots for these because the study records its protocol.

## Final full-cross-entropy SASRec runs (three seeds)

After global-time tuning finishes, `scripts/global-time/run_sasrec_tuned_all.py`
loads its winning `sasrec_model_<dataset>.yaml` files and trains fresh models
for **seeds 42, 43 and 44** on Beauty 2014, Instruments 2018, Beauty 2023 and
Yambda 50M. All seeds use the same pinned global split and post-split
train-observed catalog filter. Other preprocessing, full-CE training,
optimizer settings, epoch cap and early stopping follow the existing exp2
runner and tuning regime. Training uses the training split; validation
NDCG@10 selects the checkpoint. Each run reloads that checkpoint before
evaluating validation and test HR, NDCG, Coverage and ARP at 5/10/20.

From the repository root:

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/run_sasrec_global_time_tuned_all_array.sbatch
```

This four-task array has no partition, node-type or GPU-type restriction.
Submit it with an appropriate project account. Tasks 0–3 own Beauty, Instruments, Beauty 2023 and
Yambda 50M respectively; each runs its three seeds sequentially. To submit
while tuning is still running, add `--dependency=afterok:<TUNING_JOB_ID>` to
the `sbatch` command. To run locally or select a subset:

```bash
python scripts/global-time/run_sasrec_tuned_all.py
python scripts/global-time/run_sasrec_tuned_all.py --datasets beauty --device cuda
```

Final-run outputs are stored under `global-time/`:

* `checkpoints/exp2/global_split/initial/sasrec/<dataset>/best.pt`: the primary seed42 model.
* `experiments/global-time/checkpoints/exp2-tuned_params/sasrec/<dataset>/seed_<seed>/best.pt`
  and a per-dataset `training_summary.json`, alongside the tuning outputs.
* `experiments/global-time/exp2-tuned-params/sasrec/<run>/`: resolved configs,
  logs, final metrics and the same auxiliary interaction stores as before.
* `artifacts/global-time/exp2-tuned-params/sasrec_<dataset>_metrics_summary.csv`:
  six rows (validation/test for each seed).
* `artifacts/global-time/exp2-tuned-params/sasrec_<dataset>_metrics_mean_std.csv`:
  mean and sample standard deviation (`ddof=1`) for each split/metric, plus
  `num_seeds`. Tables update after each seed; standard deviation is blank
  until two seeds finish. Multi-dataset invocations also write `sasrec_all_*`
  tables; single-dataset array tasks only write their own dataset's tables.

Missing tuned configs fail before training; the script does not fall back
to leave-one-out winners. `--config-dir`, `--tuned-config-dir`,
`--checkpoints-dir`, `--experiments-dir` and `--artifacts-dir` allow alternate
locations. `--seeds`, `--num-workers`, `--max-epochs` and `--max-steps` support
smoke runs; use separate outputs for those. Rerunning starts fresh training
and replaces the selected seed checkpoints and dataset tables; it does not
resume interrupted optimizer state.

## Yambda 500M SASRec + SCE tuning

`scripts/global-time/tuning_sasrec_sce.py` defaults to `yambda_500m_listens`
and directly reuses the exp2 SCE tuner. Its **30-trial budget**, architecture,
optimizer and SCE search spaces, epoch cap, early stopping, median pruning,
OOM handling, exact full-catalog validation (`val/sft/ndcg@10`) and optional
three-seed reranking follow the SCE tuner configuration. It requires pinned,
ordered global-time cutoffs, rolling histories,
`last_per_user` targets and the train-observed post-split catalog filter.

Submit four cooperative workers from the repository root:

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/tuning_sasrec_sce_global_time_array.sbatch
```

The array has one dataset and four workers (`--array=0-3`), with no partition
or GPU-type restriction. Change parallelism with `--array=0-N`; include
worker 0, which enqueues the exp1 baseline and the global-time full-CE Yambda
50M winner when its tuned config exists. Incumbent lookup uses
`configs/global-time/exp2-tuned-params/`. The original SCE journal/resume and
shared finished-trial budget logic is retained: in-flight trials may finish
beyond the planned budget. This entry point does not use the full-CE tuner's
trial-leasing and interrupted-trial recovery mechanism described above.

After every search worker finishes, rerank once:

```bash
sbatch --array=0 scripts/slurm/tuning_sasrec_sce_global_time_array.sbatch --rerank
```

Direct execution and the original SCE runtime/debug flags are also available:

```bash
python scripts/global-time/tuning_sasrec_sce.py --worker-id 0 --device cuda
python scripts/global-time/tuning_sasrec_sce.py --rerank --rerank-top-k 3 --device cuda
```

The journal study name ends in `_global_time`, for both search and reranking.
Default outputs are:

* `experiments/global-time/checkpoints/exp2-tuned_params/sasrec_sce/yambda_500m_listens/`:
  journal, best checkpoint, best-trial JSON, epoch log and trial dump.
* `artifacts/global-time/exp2-tuned-params/sasrec_sce/`: best-trial JSON,
  trial CSV and optional reranking summary.
* `configs/global-time/exp2-tuned-params/sasrec_sce_model_yambda_500m_listens.yaml`:
  winning model, optimizer and resolved SCE settings.

## Final Yambda 500M SCE test metrics (three seeds)

After the global-time SCE search and any optional reranking finish, run the
winning configuration on seeds **42, 43 and 44**:

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/run_sasrec_sce_global_time_tuned_all.sbatch
```

The launcher uses one GPU; submit it with an appropriate project account. It trains all three seeds
sequentially so one job owns the result tables. To wait for a submitted
tuning array or reranking job, add `--dependency=afterok:<JOB_ID>` to `sbatch`.
Direct execution is also supported:

```bash
python scripts/global-time/run_sasrec_sce_tuned_all.py --device cuda
```

The script reads
`configs/global-time/exp2-tuned-params/sasrec_sce_model_yambda_500m_listens.yaml`
once, initializes an independent model for each seed, and reuses the existing
SCE training regime. Each run selects its best epoch by **validation NDCG@10**
and reloads that checkpoint before calculating full-catalog validation/test
**HR, NDCG, Coverage and ARP at 5/10/20**. Test metrics never select the winning
configuration or epoch. The pinned global-time cutoffs, rolling histories,
last-per-user targets and train-observed catalog stay identical across seeds.

Outputs are updated after every completed seed:

* `checkpoints/exp2/global_split/initial/sasrec_sce/yambda_500m_listens/best.pt`: the primary seed42 model.
* `artifacts/global-time/exp2-tuned-params/sasrec_sce/sasrec_sce_yambda_500m_listens_metrics_summary.csv`:
  one row per seed and split (six rows when all three seeds finish).
* `artifacts/global-time/exp2-tuned-params/sasrec_sce/sasrec_sce_yambda_500m_listens_metrics_mean_std.csv`:
  mean, sample standard deviation and completed seed count for every metric
  and split. A partial one-seed result has an empty standard deviation.
* `experiments/global-time/checkpoints/exp2-tuned_params/sasrec_sce/yambda_500m_listens/seed_<seed>/best.pt`
  and `training_summary.json` with the per-seed results. The tuning
  `best_model.pt` stays unchanged.
* `experiments/global-time/exp2-tuned-params/sasrec_sce/`: resolved configs,
  logs and final summaries for each run.

`--config-dir`, `--tuned-config-dir` and the output directory flags support
alternate locations. `--max-epochs` / `--max-steps` allow smoke runs; use
separate output directories for them. `--datasets` can select the other
SCE-tuned datasets when their global-time winning YAMLs are available.

## TIGER tuning and full training of the winners

Two entry points cover the five exp2 datasets:

* `scripts/global-time/tuning_tiger.py`: Optuna for Beauty 2014, Instruments
  2018, Beauty 2023 and Yambda 50M, with 100/100/60/40 trials respectively.
  TPE searches learning rate, dropout, embedding width, tied encoder/decoder
  depth, weight decay and attention head width. The original configuration
  is queued first; median pruning starts after ten startup trials and ten
  warmup epochs. Each trial uses the original 500-epoch cap and patience 10.
* `scripts/global-time/tuning_tiger_yambda_500m.py`: a compact grid for Yambda
  500M. The default is ten points: the original configuration plus nine
  single-parameter alternatives from the existing TIGER grid, with **eight
  epochs per point and no early stopping**. `--design full`, `--params` and
  repeatable `--grid NAME=V1,V2` allow an explicit Cartesian grid or smaller
  selection. The original value is always included in each axis.

Both select by validation NDCG@10 using constrained ten-beam decoding.
Beauty and Instruments use all validation targets; Beauty 2023 and both
Yambda sizes use a fixed 25% validation sample, identical across trials and
epochs. Test targets never select hyperparameters. The scripts enforce the
pinned global cutoffs, rolling histories, last-per-user targets, train-observed
catalog and disabled pre-split SID filtering. Existing exp2 semantic IDs are
used by default; every training item must have an SID.

The default `--stage all` finishes the search, exports the winning config,
then **trains the winner from scratch with the existing TIGER trainer
parameters** from `configs/global-time/exp1-standard_params/trainer_tiger.yaml`:
500 epochs maximum, patience 10, full one-beam validation NDCG@1, the original
clipping/accumulation/AMP settings, and seeds 42/43/44. Tuned optimizer and
model values carry over; the short grid budget and sampled validation do not.
Early stopping remains enabled, so the final run can finish before epoch 500.
Final validation/test metrics use full splits and constrained 20-beam decoding,
reporting HR/NDCG/Coverage/ARP at 5/10/20.

Submit from the repository root:

```bash
mkdir -p slurm_logs
# Four tasks, one dataset per GPU; no GPU-type restriction.
sbatch scripts/slurm/tuning_tiger_global_time_array.sbatch
# One 500M worker; both search and full winner training use E/F cards only.
sbatch scripts/slurm/tuning_tiger_yambda_500m_global_time.sbatch
```

The 500M launcher requests `--partition=rocky --constraint="type_e|type_f"`.
Submit both launchers with an appropriate project account; they use the `flowrec` conda environment.
There is one worker per dataset: a filesystem lock prevents duplicate jobs
from training the same dataset concurrently. Each worker loads its datamodule
once and reuses it for trials and final seeds. A 23.5-hour budget stops at
epoch boundaries; resubmit the same command to resume the current trial or
final seed from `last.pt`. Interrupted Optuna trials retain their trial number
and parameters. Completed work skips data loading. Resubmission is manual;
the scripts do not submit follow-up jobs.

During search, a non-finite loss disqualifies the trial and the worker continues
with the remaining configurations. The grid records `status: non_finite_loss`
in that trial's `result.json`; Optuna prunes it with the same reason in the
journal. These trials count toward the search budget, cannot win, and are
skipped on resubmission. Earlier checkpoints and logs are retained for diagnosis.
Final winner training still stops on a non-finite loss. On resume, a crash
without a failure marker resumes the last saved epoch; if the loss diverges,
the failure is recorded and the worker continues. Completed trials in the
study directory can be reused with the same split, precision and settings.

On resume, a protocol without an explicit `train_dense_match_prefix` value is
accepted when it is equivalent to `false`, preserving TIGER's `prefix_target`
training. Other protocol mismatches report the exact fields and saved/current
values; use matching settings to reuse a study. Do not delete the study config
or checkpoints to bypass the check.

Direct use, inspection and separate stages:

```bash
python scripts/global-time/tuning_tiger.py --datasets beauty --device cuda
python scripts/global-time/tuning_tiger_yambda_500m.py --list-plan
python scripts/global-time/tuning_tiger_yambda_500m.py --stage report
sbatch scripts/slurm/tuning_tiger_yambda_500m_global_time.sbatch --stage tune
# After the complete grid, including any wall-time resubmissions:
sbatch scripts/slurm/tuning_tiger_yambda_500m_global_time.sbatch --stage train-best
```

Results live under `experiments/global-time/checkpoints/exp2-tuned_params/tiger/<dataset>/`:
`study_config.json`, `data_summary.json`, trial results/checkpoints,
`tuning_summary.json`, `best_trial.json`, and final `seed_<seed>/` runs.
The smaller-dataset studies also save `optuna_journal.log`. Winning YAMLs go
to `configs/global-time/exp2-tuned-params/tiger_model_<dataset>.yaml` only after
search completion; summaries and per-seed metrics go to
`artifacts/global-time/exp2-tuned-params/tiger/`. Changed split, training or
search settings are rejected on resume. `--seeds` can select final seeds;
`--n-trials` (Optuna), `--epochs-per-trial`, `--max-epochs` (final training),
`--train-batches-per-epoch` and `--max-eval-batches` support small smoke runs.
Use separate checkpoint, artifact and tuned-config directories for smoke runs.

## GPT-2 tuning on all five datasets

`scripts/global-time/tuning_gpt2.py` runs a separate Optuna study for each of
Beauty 2014, Instruments 2018, Beauty 2023, Yambda 50M and Yambda 500M. It
reuses the exp2 GPT-2 architecture/optimizer search spaces and trial budgets
(100, 100, 60, 40, 20), baseline profiles, median pruning and early stopping.
The objective is validation generation NDCG@10 with constrained 20-beam
decoding. Beauty and Instruments use the full validation split; Beauty 2023
uses a fixed 25% sample and both Yambda datasets use a fixed 10% sample.
The test split does not select hyperparameters.

Use the existing exp2 semantic IDs and run:

```bash
# All five datasets, sequentially on one GPU.
python scripts/global-time/tuning_gpt2.py --device cuda

# One GPU job per dataset; indices 0..4 follow the dataset order above.
mkdir -p slurm_logs
sbatch scripts/slurm/tuning_gpt2_global_time_array.sbatch

# Optional three-seed reranking, after the search workers have finished.
python scripts/global-time/tuning_gpt2.py --datasets beauty --rerank --device cuda
```

The default inputs are `configs/global-time/exp2-tuned-params` and
`datasets/semantic_ids/exp2-tuned-params`. No separate global-time SID folder
or tokenizer run is required. Use `--semantic-ids-dir` to select a different
mapping. Every training item must
have an SID. The entry point requires pinned ordered cutoffs, rolling
histories, last-per-user targets, post-split train-observed item filtering,
and no pre-split SID filtering. It records the resolved protocol and rejects
resumes or reranking with changed split/training settings. If a launch failed
on missing semantic IDs before any trial started, the SID path can be corrected
and the existing journal reused, including its queued baseline.

Outputs:

* `experiments/global-time/checkpoints/exp2-tuned_params/gpt2/<dataset>/`: Optuna journal,
  `best_model.pt`, `best_trial.json`, epoch log, `all_trials.json`,
  `study_config.json`, `data_summary.json`, and optional reranking results.
* `configs/global-time/exp2-tuned-params/gpt2_model_<dataset>.yaml`: the winning
  profile, compatible with the existing GPT-2 runner.
* A multi-dataset invocation also writes `tuning_summary.json` at the checkpoint root.

Use `--datasets` to select a subset. Parallel trial workers can join a study
with distinct `--worker-id` values; include worker 0 to enqueue the baseline.
They share the original tuner's budget behavior: trials already running can
finish beyond the planned count. Resubmission resumes the journal. After
all workers of a dataset have exited, `--fail-stale-trials` marks interrupted
RUNNING trials failed so they no longer block a baseline retry; failed trials
do not count toward this tuner's budget. Run this recovery once before
relaunching parallel workers. Reranking runs once per dataset.

For a smoke run, use `--n-trials 1 --max-epochs 1 --debug-max-batches 1
--num-workers 0 --device cpu --no-progress-bar --output-root /tmp/gpt2-global-smoke`.
Redirected runs save YAMLs under `<output-root>/configs` unless
`--config-out-dir` is given, keeping debug results separate from the tuned
configs. These batch limits still require building the selected datamodule.

## Entry points and supported configurations

The following scripts accept these global-time configurations via `--config-dir`:

```
python scripts/exp1-standard-params/run_sasrec_all.py    --config-dir configs/global-time/exp1-standard_params [--variant sce]
python scripts/exp1-standard-params/run_tiger_sft_all.py --config-dir configs/global-time/exp1-standard_params --semantic-ids-dir datasets/semantic_ids/global-time/exp1-standard_params
python scripts/exp1-standard-params/run_gpt2_sft_all.py  --config-dir configs/global-time/exp1-standard_params --semantic-ids-dir datasets/semantic_ids/global-time/exp1-standard_params
python scripts/exp1-standard-params/evaluate_sasrec_all.py --config-dir configs/global-time/exp1-standard_params
python scripts/exp1-standard-params/evaluate_tiger_sft_all.py --config-dir configs/global-time/exp1-standard_params --semantic-ids-dir datasets/semantic_ids/global-time/exp1-standard_params
python scripts/exp1-standard-params/evaluate_gpt2_sft_all.py --config-dir configs/global-time/exp1-standard_params --semantic-ids-dir datasets/semantic_ids/global-time/exp1-standard_params
```

The following scripts default to leave-one-out paths. Use the listed global-time
entry point or pass all relevant path flags:

| Script | Flags to pass |
|---|---|
| Legacy `exp2-tuned-params/tuning_sasrec_sce.py` | Use the global-time SCE entry point above, or pass `--config-dir --tuned-config-dir --checkpoint-root --artifacts-dir` |
| Legacy `exp2-tuned-params/tuning_gpt2.py` | Use the global-time GPT-2 entry point above for split enforcement and isolated study names/paths |
| Legacy `exp2-tuned-params/run_sasrec_sce_tuned_all.py` | Use the global-time final SCE runner above, or pass all config, checkpoint, experiment and artifact directory flags |
| `evaluate_sasrec_sce_tuned.py` | `--config-dir --tuned-config-dir --checkpoint-root --artifacts-dir` (use `--seeds 42 43 44` to re-score the final runs) |
| `run_gpt2_tuned_all.py` | `--config-dir --tuned-config-dir --checkpoints-dir --artifacts-dir --semantic-ids-dir` |
| `tuning_tiger_yambda_500m.py` | `--dataset-config --semantic-ids-dir --checkpoint-root --artifacts-dir --tuned-config-path` (`--model-config` / `--trainer-config` for the incumbent) |

The following scripts do not currently provide a complete global-time protocol:

| Script | Blocker |
|---|---|
| Legacy `exp2-tuned-params/tuning_sasrec.py` | Use the global-time tuner described above; the legacy script retains leave-one-out defaults |
| Legacy `exp2-tuned-params/run_sasrec_tuned_all.py` | Use the global-time final-run entry point above; the legacy entry point retains leave-one-out defaults |
| `sft_finetune_control_matched.py`, `gflow_finetune_tiger_matched.py`, `gflow_finetune_gpt2_amazon_fullhist.py`, both `*_teacherfree.py` | `PROTOCOL_CONFIG` is a constant, and the init checkpoints / SFT configs / teachers are hard-coded leave-one-out assets (`configs/exp2-tuned-params/tiger_<ds>_sft.json`, `checkpoints/exp2/loo_split/initial/...`) |
| Other local cluster launchers | not distributed with the repository; separately supplied launchers must use the correct global-time entry points and paths |

Tuned model configs (`sasrec_model_<ds>.yaml`, `sasrec_sce_model_<ds>.yaml`,
`gpt2_model_<ds>.yaml`, `tiger_model_<ds>.yaml`) are **outputs** of the
global-time tuning runs and are written into `exp2-tuned-params/` of this
tree; the leave-one-out ones are not copied because they were selected on
leave-one-out validation.

## Leakage rules for the fine-tuning stage

Everything a fine-tuning arm learns from must predate `global_val_timestamp`:

* the SASRec teacher is the **global-time** SASRec checkpoint, never a
  leave-one-out one (that one trained on events after the cutoff);
* the interaction reward reads the auxiliary store through
  `InteractionLogStore.filter_by_time(global_val_timestamp)`;
* reward caches are rebuilt under the global-time cache roots.

Before running a fine-tuning arm, verify that its interaction reward applies
`filter_by_time` and that all of its inputs use the same global-time split.
