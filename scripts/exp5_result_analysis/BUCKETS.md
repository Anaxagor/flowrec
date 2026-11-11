# Item popularity and history length: init → SFT-only / GFlow

`finetune_buckets.py` compares saved **best checkpoints** from experiment 4 on
the same test users. No training is required. The analysis measures changes
between init and fine-tuning; it does not measure the trajectory across epochs.

## Selected configurations

The leave-one-out configurations below are selected for analysis. Their
parameters are saved in `bucket_selection_provenance.json`.

| Data / model | best NDCG | best coverage | Pareto |
|---|---|---|---|
| Beauty / GPT-2 | sasrec, pow=4, DB, rho=1 | sasrec×interaction, pow=16, TB, rho=0.1 | same checkpoint as best NDCG |
| Instruments / GPT-2 | interaction×sim_hier, pow=1, DB, rho=1 | interaction×sim_hier, pow=16, TB, rho=0.1 | sim_hier, pow=16, SubTB, rho=1 |
| Yambda 50M matched / TIGER | sasrec, pow=1, DB, rho=1 | sasrec×interaction×similarity, pow=16, SubTB, rho=1 | sasrec, pow=0.25, SubTB, rho=1 |
| Yambda 500M sampled-15 / TIGER | sasrec, pow=4, DB, rho=1 | sasrec×interaction, pow=16, SubTB, rho=1 | sasrec×similarity, pow=0.25, DB, rho=1 |

All selected GFlow rows use seed=42, 10 negatives, and unit component weights.
For Beauty, three labels correspond to two unique configurations: both labels
remain in the manifest, while the duplicate checkpoint is generated once.
There are **19 inference runs**: 4 init + 4 primary SFT-only + 11 GFlow.
`--seeds 42 43 44` selects the same primary SFT models, each counted once with
its actual seed from `provenance.json`. Variability between GFlow seeds is not
estimated, and the shared init is not counted as an independent repeat.

Paths are fixed in `bucket_manifest.py` and available through `--mode discover`:

- GPT-2: `experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/<dataset>/`.
  SFT-only comes from the same family; a separate dense control is not substituted.
- TIGER 50M GFlow: `experiments/exp4-gflow-yambda50m-matched/`;
  SFT: `experiments/exp2-sft-finetune-matched/yambda_50m_listens/`.
- TIGER 500M GFlow: `experiments/exp4-gflow-yambda500m-matched-r15/`;
  SFT: `experiments/exp2-sft-finetune-matched-r15/yambda_500m_listens/`.

Missing weights remain in the manifest: their local absence does not exclude a
configuration. Paths are relative to the repository root; old absolute paths in
captured configs are resolved on load. Catalog boundaries, semantic IDs, the
initial checkpoint, and matched/random15 protocol are checked against available
artifacts. Test results selected in the table support this descriptive analysis,
not independent configuration selection.

## Buckets and metrics

Popularity is the number of **training interactions** from the existing
`training_item_popularity`, excluding validation/test and without recounting
prefix targets. Every catalog item receives a fixed bucket:

- `unseen`: training count=0;
- `tail`: mean percentile rank among items with positive frequency ≤50%;
- `middle`: (50%, 80%];
- `head`: >80%.

Equal frequencies are not split, so group sizes differ from the nominal
50/30/20%. `catalog_buckets.csv.gz` contains every item, count, and group.

Users are grouped by full history length into `0–5`, `6–10`, `11–20`, `21–50`,
`51–100`, `101–200`, `201+` (configurable with `--history-edges`). Length comes
from `target_position`: positive events before the test target after initial
preprocessing, before model context truncation and removal of cold history
items. For LOO this includes the validation item. Actual context length and
SHA-256 of its contents are also saved. User groups and intersections of target
popularity with history length are identical for all models. Empty groups retain
`num_users=0` and NaN.

At K=5/10/20 the analysis calculates:

- **Quality by target group:** HR/Recall (one target) and NDCG at original top-K
  positions. Recommendations are not filtered by bucket or renumbered.
- **Recommendations for a user group:** full-catalog coverage, ARP, exposure
  Gini, entropy and exp(entropy), and mean recommendation overlap across users.
- **Bucket representation in recommendations:** exposure share, unique items,
  coverage with that bucket's size as denominator, and exposure concentration.
- **Paired changes:** GFlow versus init and SFT of the same seed; SFT versus
  init. New/lost hits, improved/degraded NDCG, rank changes, list overlap and
  metric deltas are reported. `accuracy_attribution.csv` records bucket
  contributions to overall ΔHR/ΔNDCG; contributions within a `cohort_type`
  sum to the overall change.
- **Individual items:** test target counts, hits, HR/NDCG, before/after
  exposures, newly covered/lost items, gains/losses and contributions to accuracy.

Coverage and exposure concentration measure **aggregate diversity**, not
within-list semantic diversity or model probability entropy. User group sizes
differ, so coverage across groups also depends on user count; model comparisons
within a group have the same sample size. Popular test targets weigh more in
micro metrics; per-item tables make this visible.

`--bootstrap-samples 500` adds paired user-level ΔHR/ΔNDCG intervals (default
0). These describe test-sample uncertainty with fixed models; for small groups
they may be uninformative. They do not measure training variability across seeds.

## Execution

Prepare datasets on CPU before inference. Inference loads compact evaluation
tensors, semantic IDs, and model weights; it does not need training optimizer
state or reward caches. A 32 GB GPU is a reasonable initial estimate for the
listed models at batch 64 and beam 20; verify actual memory use and reduce batch
size if necessary. Yambda preprocessing still builds the complete source layout
and can require substantial host RAM even for a small inference subset.

## Individual stages and local checks

```bash
python scripts/exp5_result_analysis/finetune_buckets.py --mode discover
python scripts/exp5_result_analysis/finetune_buckets.py --mode prepare --datasets beauty
python scripts/exp5_result_analysis/finetune_buckets.py --mode generate --run-index 0 --device cuda
python scripts/exp5_result_analysis/finetune_buckets.py --mode analyze --datasets beauty
```

The last command requires **all** selected Beauty recommendations; a missing
file fails analysis rather than silently producing a partial experiment. For a
CPU smoke test, create a separate manifest and directory:

```bash
OUT=artifacts/exp6_result_analysis/buckets_smoke
python scripts/exp5_result_analysis/finetune_buckets.py --mode discover --datasets beauty --output-dir "$OUT"
python scripts/exp5_result_analysis/finetune_buckets.py --mode prepare --output-dir "$OUT" --max-users 8
python scripts/exp5_result_analysis/finetune_buckets.py \
  --mode generate \
  --output-dir "$OUT" \
  --run-index 0 \
  --device cpu \
  --batch-size 4
```

All relative paths are resolved from the repository root. Do not change the
manifest while inference is active: indices refer to its fixed order. On
rerun, compatible caches are checked and reused. Changes to inputs or protocol
trigger preparation and generation again; `--force` refreshes a stage. NPZ,
prepared tensors, catalog and sidecars are checked by content hash; large source
parquets by size/mtime. FP32 and the current decoder may differ from the old
evaluation; full-test metric discrepancies are reported in `metric_validation.csv`.

## Artifacts

`artifacts/exp6_result_analysis/finetune_buckets/<dataset>/`:

- `report.html`: standalone report with embedded charts;
- `figures/*.png`, `*.svg`: NDCG and ΔNDCG by bucket, exposure and coverage,
  and change matrices for group intersections;
- `tables/summary.csv`, `cohorts.csv`, `popularity_exposure.csv`: metrics;
- `tables/comparisons.csv`, `accuracy_attribution.csv`: paired changes;
- `tables/items.csv.gz`, `item_deltas.csv.gz`, `catalog_buckets.csv.gz`: items;
- `tables/definitions.csv`, `metric_validation.csv`, `analysis_metadata.json`:
  definitions, validation and protocol;
- `prepared/`: one set of frozen evaluation inputs per dataset;
- `cache/<run_id>.npz` and `.json`: recommendations and provenance per checkpoint.

```bash
python -m pytest tests/test_bucket_manifest.py tests/test_bucket_metrics.py \
  tests/test_bucket_inference.py tests/test_finetune_buckets.py -q
```
