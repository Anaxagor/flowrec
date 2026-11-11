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
| [selected_metrics.csv](selected_metrics.csv) | 201 main selection roles across 67 comparison groups, with starting checkpoints kept separate |
| [reported_selections.csv](reported_selections.csv) | All 126 original selection roles from 42 reported groups, including the source's pooled-initialization selections |
| [reference_metrics.csv](reference_metrics.csv) | 41 reference or historical results, including initial models and SFT controls |
| [reported_metrics.csv](reported_metrics.csv) | All 165 source-summary rows for the three datasets, with numeric metrics, interval half-widths, source values, parameters, and selection labels |
| [candidate_metrics.csv](candidate_metrics.csv) | 550 individual result rows from the separate experiment tracker, including rows hidden by its filters |
| [candidate_selections.csv](candidate_selections.csv) | Reproducible three-criterion selection within the supplied candidate groups |
| [dataset_statistics.csv](dataset_statistics.csv) | Dataset counts reported in draft Table II |
| [provenance.json](provenance.json) | Source identifiers, checksums, field transformations, and selection scope |

`reported_metrics.csv` transcribes `GFlow+Recsys updated (15).xlsx`, sheet
`LOO setting`. `candidate_metrics.csv` transcribes the numerical results in
`Эксперименты flowrec.xlsx`, sheet `Наши метрики`, joined to the scientific
parameters in `Прогоны` by registry ID. The original books are not required to
regenerate the public tables. Global-time results, Beauty 2023, and Yambda 500M
are outside this snapshot.

## Interpreting the main table

**The supplied selections use test metrics.** They are post-hoc descriptions
of the reported test results, not validation-selected held-out estimates. The
draft's validation-selection description must be reconciled with the actual
experimental records before these tables are presented as validation-selected.

The main table retains the source's GFlow and CE selections, which already
separate the initial and SFT-only starting checkpoints. The source's Prefix-CE
and Q-MSE selections pool both starts. In the main table these pooled rows are
replaced by selections computed **separately for each initialization** from the
supplied candidate records, retaining separate experiment batches and
calibration protocols. These recomputed selections cover the available partial
candidate groups, not an independently verified full search grid.

The main view contains 102 retained source roles and 99 recomputed roles.
There are 67 comparison groups after retaining separate experiment protocols;
each has one row for each of the three selection criteria.

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

The separate candidate table contains only seed **42**. `repeat_index` is not
a seed: the four FR0023 records use different calibration protocols and are
not averaged. Dataset split identifiers, full starting-checkpoint identities,
and validation metrics are not supplied for this snapshot.

Candidates are compared within dataset, model, initialization, method family,
source experiment batch, and calibration protocol. GFlow TB, DB, and SubTB are
configuration choices within the GFlow family. Different experiment batches
are kept separate because their complete evaluation protocols cannot be
established from these tables alone.

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
