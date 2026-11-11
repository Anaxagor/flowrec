# Reported experiment metrics

This snapshot contains **Beauty**, **Musical Instruments**, and **Yambda 50M**,
the three datasets in the draft paper. The main table separates starting
checkpoints and shows three selection labels: **best NDCG@10**,
**best Coverage@10**, and **Pareto**. Original workbook annotations and execution
records are excluded; scientific values, reported uncertainty, and available
configuration parameters are retained.

## Files

| File | Contents |
| --- | --- |
| [metrics.xlsx](metrics.xlsx) | Formatted workbook with the selected results, references, candidate selections, per-run metrics, and dataset statistics |
| [selected_metrics.csv](selected_metrics.csv) | 273 main selection roles across 91 comparison groups, with starting checkpoints and source protocols kept separate |
| [reported_selections.csv](reported_selections.csv) | All 126 original selection roles from 42 reported groups, including the source's pooled-initialization selections |
| [reference_metrics.csv](reference_metrics.csv) | 41 reference or historical results, including initial models and SFT controls |
| [reported_metrics.csv](reported_metrics.csv) | All 165 source-summary rows for the three datasets, with numeric metrics, interval half-widths, source values, parameters, and selection labels |
| [candidate_metrics.csv](candidate_metrics.csv) | 1,244 candidate records: 550 original tracker records, 48 GRPO results, 422 CE results from initial models, and 224 CE results from SFT-only models |
| [candidate_selections.csv](candidate_selections.csv) | Reproducible three-criterion selection within the supplied candidate groups |
| [dataset_statistics.csv](dataset_statistics.csv) | Dataset counts reported in draft Table II |
| [provenance.json](provenance.json) | Source identifiers, checksums, field transformations, and selection scope |

`reported_metrics.csv` transcribes `GFlow+Recsys updated (15).xlsx`, sheet
`LOO setting`. The candidate inputs are:

| Source | Records | Starting stage and scope |
| --- | ---: | --- |
| `Эксперименты flowrec.xlsx`, `Наши метрики` joined to `Прогоны` | 550 | Original tracker records, including rows hidden by its filters |
| `RL-Experiments (1).xlsx`, `Base` / `SFT tuned` | 48 | 24 GRPO configurations at each starting stage; primary NDCG@10/Cov@10 columns only |
| `ce_all_results_2026-11-09_from_init.csv` | 422 | CE from tuned initial models; latest cumulative exp7 report |
| `all_metrics_based_sft-only.xlsx`, `results` | 224 | Completed CE configurations from SFT-only models; checkpoint SHA256 retained |

The exp7 snapshot contains every CE run identity from the earlier reports, with
unchanged test metrics on shared identities. Only the latest snapshot is imported.
Distinct configurations remain separate even when they select the same initial
checkpoint and consequently have equal metrics. The GRPO `Last` columns and the
SFT workbook's `history`/`events` sheets are excluded.

The original books and private experiment directories are not required to
regenerate the public tables. Source checksums and transformations are recorded
in `provenance.json`; candidate `source_references` identify source, sheet and row.
Global-time results, Beauty 2023, and Yambda 500M are outside this snapshot.

## Interpreting the main table

**The supplied selections use test metrics.** They are post-hoc descriptions
of the reported test results, not validation-selected held-out estimates.
For the added CE records, validation NDCG@5 selects the checkpoint within each
run; selection between configurations in this table still uses test metrics.
The data provider confirmed the GRPO primary columns as test metrics. Their
within-run checkpoint-selection criterion is not reported.

The main table retains the source's GFlow and CE selections, which already
separate the initial and SFT-only starting checkpoints. The source's Prefix-CE
and Q-MSE selections pool both starts. In the main table these pooled rows are
replaced by selections computed **separately for each initialization** from the
supplied candidate records, retaining separate experiment batches and
calibration protocols. These recomputed selections cover the available partial
candidate groups, not an independently verified full search grid.

The main view contains 102 retained source roles, 99 recomputed Prefix-CE/Q-MSE
roles, and 72 additional GRPO/CE roles. There are 91 comparison groups; each has
one row for each of the three selection criteria.

Historical authored GRPO/CE selections remain in the `reported_summary`
namespace. New candidate selections use `rl_experiments_primary`,
`ce_exp7_from_init`, and `ce_exp8_from_sft`. Rounded metric agreement does not
establish that a new record and a historical summary represent the same run or
the same repetition aggregate. These views must not be counted as independent
replications or pooled to recompute historical confidence intervals.

`reported_selections.csv` retains all original selection labels unchanged for
comparison, including the pooled rows. `selection_scope`, `selection_basis`,
and source identifiers distinguish the two kinds of result in the main table.
An unspecified starting checkpoint remains explicitly unspecified; it is not
assigned to an initialization by guesswork. History layouts such as
`last_8_targets` and `last_128_targets` remain distinct.

The three roles can select the same configuration. Each role remains visible,
with the same source record identifier when it comes from the same row. These
are not independent runs. References are retained once and are not relabelled
as three independently selected configurations.

HR, NDCG, and coverage are fractions, not percentages. All supplied metrics at
5, 10, and 20 are retained. An empty value means unreported, not zero; source
missing-value markers such as `—` are normalized to empty fields. According
to the source's sheet-level note (`LOO setting!A238`), values written as
`mean ± value` use a Student-t **95% confidence-interval half-width**. They are
not standard deviations. The reported nonmissing strings are preserved in `*_reported`;
numeric means and half-widths are separate columns. Per-row seed counts are
not consistently available. Explicit seed identifiers and counts are retained
in `parameters_json` where supplied; the intervals are not recomputed.

The `notes` column preserves scientific limitations. In particular, source row
145 names GRPO but has an SDPO loss label; this reference row is retained with
`method_label_conflicts_with_loss_column`. Some records use reward variants,
powers, or history lengths beyond the draft's main grid. They remain labelled
as reported rather than being silently changed to match the paper.

## Reproducing the candidate selection

Seed **42** is reported for the original tracker and the added CE records.
The 48 added GRPO records do not report seed, repetition count, uncertainty,
sampling settings or SFT loss weight; these remain empty. `Base` and `SFT tuned`
identify GRPO starting stages, not the SFT loss coefficient during GRPO.
The explicit `reported_beta` on one GRPO row is retained without assigning it
to a reward power or KL coefficient. No repetition averaging is performed.

`repeat_index` is not a seed: the four FR0023 records use different calibration
protocols and are not averaged. Exact starting-checkpoint SHA256 values are
available for the 224 added CE-from-SFT records. The original tracker and GRPO
do not provide those identities; the exp7 report identifies the initial stage
without a verified per-run checkpoint hash. Missing values remain unknown.

Candidates are compared within dataset, model, initialization, method family,
source experiment batch, calibration protocol, history layout, evaluation point,
and any supplied split/version/checkpoint identity. GFlow TB, DB, and SubTB are
configuration choices within the GFlow family. Different experiment batches
are kept separate because their complete evaluation protocols cannot be
established from these tables alone.

Within each CE source protocol, estimator and sampler are configuration choices.
The exp7 rows include 406 sampled and 16 exact item-CE runs; `ce_target` is
`curriculum_proposal` for 248 records and `q` for 174. All 224 CE-from-SFT records
use sampled item CE targeting `q`. Parameters retain target, importance
correction, sample count and calibrated coefficient where supplied. Exact item
CE is not relabelled Prefix-CE. This pooled selection is descriptive and does
not constitute a controlled comparison of estimators or targets.

Completed CE runs that retain their initial checkpoint remain candidate
configurations, with `best_status=init_kept`. Runtime is included only where
reported; one resumed exp7 exact-CE run measures its latest invocation rather
than the full run, as labelled in `runtime_scope`.

For each group:

1. **Best NDCG@10:** maximize NDCG@10, then Coverage@10.
2. **Best Coverage@10:** maximize Coverage@10, then NDCG@10.
3. **Pareto:** keep the nondominated NDCG@10–Coverage@10 frontier; min–max
   normalize each coordinate over that frontier and choose the point nearest
   to `(1, 1)` with equal weights. A constant coordinate contributes zero
   distance. Break ties by NDCG@10, then Coverage@10.

Remaining ties use the source row for a stable representative. Tie counts and
source record identifiers are retained. No requirement for three distinct
configurations overrides the selection criteria. SFT-only controls remain
single reference records.

From the repository root:

```bash
python scripts/export_public_metrics.py
```

CSV regeneration uses the Python standard library. Creating `metrics.xlsx`
also requires the optional `openpyxl` dependency. Use `--skip-xlsx` for CSV only,
or `--output-dir /tmp/flowrec-metrics` to regenerate into a separate directory.

Check the selection rules and export with:

```bash
python -m pytest tests/test_public_metrics_export.py -q
```

The [ICDE 2027 EAB supplemental-material rules](https://icde2027.github.io/cf-research-papers.html#supplemental-material)
require the artifacts needed to reproduce the reported results. This directory
provides the numerical snapshot and reproducible table construction; training
data, model weights, experiment configurations, and evaluation code are also
needed to reproduce the underlying experiments. The table alone does not
establish that the full artifact or paper meets every submission requirement.
