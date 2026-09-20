# Global-time-split configs

A parallel config tree for re-running the study (exp1 standard-parameter runs,
exp2 tuning + tuned multi-seed runs, fine-tuning arms) on a **global temporal
split** instead of leave-one-out. Nothing here touches the leave-one-out
configs or results.

```
configs/global-time/
  exp1-standard_params/    dataset_*.yaml  *_model.yaml  trainer_*.yaml
  exp2-tuned-params/       dataset_*.yaml  finetune_base_matched.json
```

`tests/test_global_time_configs.py` is the parity guard: the two trees may
differ only in the split block, the pinned cutoffs and the output roots.

## Protocol

| | |
|---|---|
| Split | `split_strategy: global_time` — train = positive interactions before `global_val_timestamp`, validation = `[val, test)`, test = from `global_test_timestamp` on |
| Cutoffs | 80 % / 90 % interaction quantiles of each dataset's positive 5-core stream, **pinned as timestamps** per dataset |
| History | `global_history_policy: rolling` — a target sees every earlier event of its user, including earlier events of the same window |
| Targets | `global_eval_target_mode: last_per_user` — one target per user active in the window (its last event there) |
| Catalog | `filter_cold_items_after_split: true` on every dataset — train-observed items only; cold targets are dropped for every model |
| VK-LSVD | unchanged: its `provided` weekly split already is a global temporal split |

Everything else (positive criterion, 5-core, history length, layouts, batch
sizes, model and trainer parameters, schedules, grids) equals the
leave-one-out tree.

Why the cutoffs are pinned instead of `global_val_ratio` / `global_test_ratio`:
the datamodules drop items without semantic IDs *before* the split, so a ratio
quantile is only guaranteed to be the same for SASRec, TIGER and GPT-2 when
the semantic IDs cover the whole catalog. Pinned timestamps make the windows
identical for every model, teacher and reward cache regardless.

### Cutoffs and measured split sizes

Measured 2026-09-19 on the local raw data (SASRec view; Beauty and
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

## Semantic IDs: reuse, do not retrain

There is deliberately no `semantic_ids.yaml` in this tree. The existing
semantic IDs cover **100 % of the pre-split 5-core catalog** (checked for
Yambda 50m / 500m against both `datasets/semantic_ids/exp1-standard_params`
and `.../exp2-tuned-params`; the Amazon stores are built from that catalog by
construction), and the pre-split catalog does not depend on the split. The
`--semantic-ids-dir` defaults of the exp1 / exp2 scripts already point at
those folders, so nothing has to be passed. A base checkpoint and every arm
fine-tuned from it must read the same folder.

Retraining through this tree would be harmful on Yambda / VK-LSVD:
`train_tokenizer_all.py` tokenizes the catalog of a fully set-up datamodule,
i.e. the *train-observed* catalog. TIGER / GPT-2 would then drop the cold
items before the split and score each user's last *warm* event, while SASRec
picks the last event and drops the example when it is cold — different
evaluation sets per model. (Under leave-one-out the same mechanism was
dormant: no Yambda item was cold.)

## Output roots

Every output root gains a `global-time/` level:

```
checkpoints/exp1-standard_params/tiger   ->  checkpoints/global-time/exp1-standard_params/tiger
experiments/sasrec/yambda                ->  experiments/global-time/sasrec/yambda
artifacts/exp1-standard_params           ->  artifacts/global-time/exp1-standard_params
```

The same rule applies to every path of `finetune_base_matched.json`.

## What runs today, what needs script work

Works with the existing scripts (they take `--config-dir`):

```
python scripts/exp1-standard-params/run_sasrec_all.py    --config-dir configs/global-time/exp1-standard_params [--variant sce]
python scripts/exp1-standard-params/run_tiger_sft_all.py --config-dir configs/global-time/exp1-standard_params
python scripts/exp1-standard-params/run_gpt2_sft_all.py  --config-dir configs/global-time/exp1-standard_params
python scripts/exp1-standard-params/evaluate_{sasrec,tiger_sft,gpt2_sft}_all.py --config-dir configs/global-time/exp1-standard_params
```

Takes path flags, so it can be pointed here, but its **output defaults still
name the leave-one-out roots** — pass all of them:

| Script | Flags to pass |
|---|---|
| `tuning_sasrec_sce.py`, `run_sasrec_sce_tuned_all.py`, `evaluate_sasrec_sce_tuned.py` | `--config-dir --tuned-config-dir --checkpoint-root/--checkpoints-dir --artifacts-dir` |
| `run_gpt2_tuned_all.py` | `--config-dir --tuned-config-dir --checkpoints-dir --artifacts-dir --semantic-ids-dir` |
| `tuning_tiger_yambda_500m.py` | `--dataset-config --semantic-ids-dir --checkpoint-root --artifacts-dir --tuned-config-path` (`--model-config` / `--trainer-config` for the incumbent) |

Needs a code change before it can run on this split:

| Script | Blocker |
|---|---|
| `tuning_sasrec.py`, `run_sasrec_tuned_all.py` | `CONFIG_DIR`, checkpoint and artifact roots are module constants |
| `tuning_gpt2.py` | `CONFIG_DIR` is a constant (only `--output-root`, `--config-out-dir`, `--semantic-ids-dir` are flags) |
| TIGER tuning on Amazon | the only Optuna entry points (`scripts/run_sft_optuna_trial.py`, `run_sft_optuna_source_style_trial.py`) hard-code `split_strategy: leave_one_out` and read `configs/sweeps/...` base configs that are not in this checkout |
| `sft_finetune_control_matched.py`, `gflow_finetune_tiger_matched.py`, `gflow_finetune_gpt2_amazon_fullhist.py`, both `*_teacherfree.py` | `PROTOCOL_CONFIG` is a constant, and the init checkpoints / SFT configs / teachers are hard-coded leave-one-out assets (`configs/exp2-tuned-params/tiger_<ds>_sft.json`, `checkpoints/exp2-tuned_params/initial/...`) |
| `scripts/slurm/*` | launchers pass none of the flags above (the folder is gitignored; cluster copies need the same edits) |

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

Whether the fine-tuning scripts already apply `filter_by_time` has not been
checked yet — it belongs to the script step above.
