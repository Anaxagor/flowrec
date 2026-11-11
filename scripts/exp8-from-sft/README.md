# Fine-tuning after SFT-only

This experiment tests whether GFlow/CE gains persist after SFT-only fine-tuning.
Each arm starts from the selected `best.pt` of that control. The source weights
and new-run seed are always **42**; the new-stage learning rate is **3e-5**. An
additional SFT arm measures the effect of another training stage on the same data.

Each new SFT-only, GFlow, and CE run loads only `model_state_dict` from the source
checkpoint. AdamW is initialized anew with empty moments and zero updates; the
original SFT-only optimizer, AMP, and training state are not restored. This starts
a new training stage. Full resume from `checkpoints/last.pt` applies only to an
interrupted cell of the new experiment, after checking its source and settings.
The distinction between loading weights and restoring optimizer state is also
covered by the [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training).

An audit of all nine source `best.pt` files confirmed a **constant LR of 3e-5**:
`scheduler_state_dict=None`, `trainer_config.scheduler_step_on="none"`, and LR
in every optimizer group is `3e-5`. The original SFT-only runs used neither warmup
nor decay. All three new arms retain the same constant LR and `scheduler=None`.
AdamW parameters: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`.

Batch size, accumulation, validation settings, and early stopping come from the
source SFT-only protocol. An initial validation always precedes the new stage:
the unchanged source can remain the best checkpoint. GFlow uses a teacher
curriculum lasting **10 epochs**.

Each source has **121 runs**:

| Arm | Grid | Runs |
|---|---|---:|
| SFT | Supervised loss only | 1 |
| GFlow | 4 rewards × 3 powers × 2 rho × TB/DB/SubTB | 72 |
| CE | 4 rewards × 3 powers × 2 rho × 2 sampling methods | 48 |

Rewards: `sasrec`, `interaction`, `similarity`, and the product of all three.
Powers: **0.25, 1, 16**; rho: **0.1, 1**. The power is applied to the composite
reward once. Rho sets the target gradient-norm ratio of the auxiliary and SFT
losses; the auxiliary loss coefficient is calibrated on the source SFT-only model.
For each new GFlow/CE cell, λ is recalculated on the selected SFT-only weights
before the first optimizer update and saved in its own `calibration.json`.
Calibration is not imported from the previous experiment. Resuming an interrupted
cell retains its calculated λ; CE may also reuse its calibration if interrupted
before the first checkpoint.

GFlow uses **K=3** negative candidates. CE uses **three independent catalog
samples with replacement**: duplicates and selection of the target item are
allowed. This preserves `q = R / sum(R)`; excluding the target would change the
training objective. CE modes: `reward` (direct q sampling) and
`curriculum_smooth_is` (smooth proposal transition with q/proposal correction).
The CE curriculum transition lasts **5 epochs**, with **epsilon = 0.1** full-q
mixing in the proposal.

## Sources and protocol reproduction

The default selects **9 sources, 1089 runs**:

| source_id | Source SFT-only |
|---|---|
| `tiger_beauty` | `finetuned`, Beauty |
| `tiger_instruments` | `finetuned`, Instruments |
| `tiger_yambda50_matched` | `finetuned_matched`, full last128 |
| `tiger_yambda500_matched_r15` | `finetuned_matched_r15`, 15 targets from last128, subset seed 42 |
| `gpt2_beauty` | Amazon dense-prefix, Beauty |
| `gpt2_instruments` | Amazon dense-prefix, Instruments |
| `gpt2_beauty_2023` | Amazon dense-prefix, Beauty2023 |
| `gpt2_yambda50_trial36` | Trial 36, full last128 |
| `gpt2_yambda500_trial4_sampled50` | Trial 4, 50 targets from last128, subset seed 42 |

[sft_source_protocol.py](sft_source_protocol.py) specifies source paths and
identities. Primary weights are read from
`checkpoints/exp2/loo_split/sft-finetuned/<model>/<dataset>/best.pt`. For GPT2
Yambda50 the path includes `last_128`; for GPT2 Yambda500 it includes
`sampled_50`. Weights are used only if `provenance.json` confirms the required
source and seed and their SHA-256 matches the recorded value. References to other
seeds of the same source use the primary model; its actual seed is recorded in
paths, results, and identities, and jobs identical after substitution are merged.
Source-run metadata is read alongside primary weights; additional records come
from frozen JSON. Source identity is determined by the selected weights' path and
hash and the hashes of metadata and protocol.

Preparation uses the selected source's architecture, semantic IDs, prefix
construction, target subset, and evaluation settings. Additional protocol records
are frozen in `source_protocols.json`. Weights, source metadata, datasets,
tokenizer/RQ-VAE, and SASRec teacher must be available at project paths. A source
mismatch raises an error; no initial checkpoint is substituted. Source files are
not overwritten.

TIGER Beauty2023 and TIGER Yambda500 with the full matched protocol are excluded
from the default set. Beauty2023 has primary seed42 weights and a sidecar, but the
full original config, result, and frozen protocol were not supplied. Full matched
Yambda500 is unavailable as a current primary source; the selected r15 source
requires explicit `--source-ids tiger_yambda500_matched_r15`.

```bash
python scripts/exp8-from-sft/finetune_from_sft.py --phase prepare --list
python scripts/exp8-from-sft/finetune_from_sft.py --check-sources
python scripts/exp8-from-sft/finetune_from_sft.py --count
python scripts/exp8-from-sft/finetune_from_sft.py --list
```

`--count` and `--list` use only the Python standard library and do not load data,
weights, or Torch. Indices depend on the supplied filters.

## Running

Run from the repository root after checking the selected source assets:

```bash
python scripts/exp8-from-sft/finetune_from_sft.py --check-sources
python scripts/exp8-from-sft/finetune_from_sft.py --phase prepare --prepare-kind cpu --device cpu
python scripts/exp8-from-sft/finetune_from_sft.py --phase prepare --prepare-kind gpu --device cuda
python scripts/exp8-from-sft/finetune_from_sft.py --phase train --list
python scripts/exp8-from-sft/finetune_from_sft.py --phase train --device cuda
```

CPU preparation builds interaction and similarity factors. GPU preparation builds
SASRec factors and evaluates the source SFT model. Finish both phases before
training. Without `--run-index`, the selected runs execute sequentially. Use
`--run-index N` from `--list` to evaluate independent configurations separately.
Indices depend on filters; keep those filters unchanged across all phases.

Use `--source-ids`, `--models`, `--datasets`, and `--arms` to narrow the experiment.
For example, append `--source-ids gpt2_beauty --arms sft gflow ce` to every command
for one source. All phases must use the same `--cache-root` and `--output-root`.
`--smoke` selects separate limited runs but still builds the source data.

For TIGER Beauty/Instruments, `--reuse-tiger-gflow-caches` validates and imports
compatible existing GFlow factors instead of rebuilding them. Retain source
identity and all component manifests; incompatible or missing inputs fail.

## TIGER Beauty/Instruments controls and cache reuse

Selecting `--source-ids tiger_beauty tiger_instruments --arms sft ce` produces
98 configurations: two SFT controls and 96 CE runs. The source weights and
new-stage seed remain 42; learning rate is 3e-5. CE uses three samples, four
reward combinations, powers 0.25/1/16, rho 0.1/1, and both sampling methods.

`--reuse-tiger-gflow-caches` imports factors from
`artifacts/exp4-reward_ablation/gflow_tiger_from_sft/<dataset>/reward_factors/`.
It compares all user/target/token/timestamp rows, the SASRec item table, and
similarity codebook, and spot-checks SASRec hidden states and interaction
rewards. Source payload and manifest hashes are recorded. If an older manifest
lacks data/teacher hashes, the limitation is recorded in provenance; this is
not a complete reevaluation of every reward. Known changes to data or teacher
require a separate cache. Original factor payloads and manifests are preserved.

The separate GFlow protocol uses K=10, batch 32×64, validation every epoch,
and patience 5. These exp8 controls use batch 256×8, validation every five
epochs, patience 10, and three CE samples. Both effective batches are 2048;
validation/stopping schedules differ. The separate GFlow grid also includes
power 4, which exp8 excludes. Preserve these distinctions when comparing methods.

CPU preparation writes a verified `prepared_data/` snapshot. GPU preparation
and training reuse it, and fail if it is missing or incompatible. GPT-2/Yambda
also requires the verified SASRec teacher catalog-ID mapping. Both preparation
phases record its hash; deleting or changing it invalidates prepared inputs.
The GPU phase evaluates the source SFT checkpoint with its evaluation settings
and source hashes, reusing only compatible completed baseline records.

## Memory and diagnostics

`--history-build-chunk-size 100000` bounds intermediate history expansion while retaining
the final dataset's examples. Compact prepared snapshots and reward factors are
reused only when their inputs and identities match. Large Yambda preparation can
require substantial host RAM even when the final training subset is small.

`--traceback-interval` optionally records periodic Python stack snapshots;
`0` disables these snapshots while retaining normal stage/progress messages.
Use `--num-workers`, `--batch-size`, and `--grad-accum` to match available hardware,
keeping effective batch size and the scientific protocol consistent.

## Resuming and results

Compatible completed cells are skipped. Interrupted cells restore their own
model, optimizer, AMP, RNG, curriculum, early-stopping state, and calibration
from `checkpoints/last.pt`; they do not restart from the source SFT model.
Checkpoints are written atomically every 100 optimizer updates by default;
`--checkpoint-every-updates` changes this interval. A changed source,
data identity, or training protocol requires a separate result directory.

Results are stored under `experiments/exp8-from-sft/<source>/sft_seed_42/` and
prepared assets under `artifacts/exp8-from-sft/`. Per-configuration `result.json`
records completion; metrics collection should use these records rather than
assuming that the presence of weights implies a completed evaluation.

Collect the available results into `artifacts/exp8-from-sft/all_metrics.csv`,
then export the metrics and training histories to `all_metrics.xlsx`:

```bash
python scripts/exp8-from-sft/collect_metrics.py
python scripts/exp8-from-sft/export_metrics_workbook.py
```

The workbook exporter requires the optional `openpyxl` dependency.
