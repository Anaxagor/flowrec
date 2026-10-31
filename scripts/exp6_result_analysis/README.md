# Beauty: GPT-2 and TIGER recommendations before and after SFT-only

For comparisons of **init / SFT-only / selected GFlow** on Beauty, Instruments,
Yambda 50M matched, and Yambda 500M sampled-15 by item popularity and user
history length, see [BUCKETS.md](BUCKETS.md). That experiment uses separate
caches and complements the Beauty analysis.

`beauty_sft_analysis.py` regenerates recommendations from saved models for the
**same Beauty test users**, saves them, and produces a standalone HTML report
with PNG/SVG plots. It does not train models or modify the original experiments
and checkpoints.

By default, it compares four primary models: two initial checkpoints and two
checkpoints after the first SFT-only fine-tuning.

| Model | Initial checkpoint | Primary SFT-only |
|---|---|---|
| GPT-2 | `checkpoints/exp2/loo_split/initial/gpt2/beauty/best.pt` | `checkpoints/exp2/loo_split/sft-finetuned/gpt2/beauty/best.pt` |
| TIGER | `checkpoints/exp2/loo_split/initial/tiger/beauty/best.pt` | `checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty/best.pt` |

The primary SFT checkpoint is included once, using its actual seed from
`provenance.json`, regardless of `--seeds`. If the primary file is missing,
inference fails and reports the current path.

These are dense-prefix GPT-2 and exp2 SFT-only TIGER. Other full-history,
dense-control, and GFlow runs are not substituted automatically. The current
set contains one selected SFT checkpoint per model.

## Running

From the repository root, in an environment with the project dependencies,
pandas, and matplotlib:

```bash
# Check selected sources without importing PyTorch or generating recommendations.
python scripts/exp6_result_analysis/beauty_sft_analysis.py --mode discover

# Full analysis: all 22,363 test users, primary init and SFT models.
python scripts/exp6_result_analysis/beauty_sft_analysis.py --device cuda

# Local check on a common subset, using a separate output directory.
python scripts/exp6_result_analysis/beauty_sft_analysis.py \
  --device cpu --seeds 42 --max-users 128 \
  --output-dir artifacts/exp6_result_analysis/beauty_smoke

# Rebuild plots and calculations from saved recommendations; no PyTorch needed.
python scripts/exp6_result_analysis/beauty_sft_analysis.py --mode analyze
```

`--device auto` selects CUDA when available and CPU otherwise. MPS is not used:
an existing GPT-2 constrained generation check failed on that backend with an
invalid catalog prefix. CPU defaults to four compute threads (`--cpu-threads`).

Work can be split into `--mode generate` and `--mode analyze`. Full defaults are
`--num-beams 20 --top-k 20 --ks 5 10 20 --batch-size 32`. Reducing batch size
preserves the decoding protocol but changes the cache fingerprint. `--force`
regenerates recommendations. A custom `--manifest runs.json` explicitly selects
sources in the format produced by `--mode discover`.

Generation uses eval mode and FP32, constrained to valid catalog sequences;
previously viewed items are not excluded, matching the original evaluation.
The model, semantic IDs, catalogs, popularity, users, targets, and history
contents are checked for compatibility. Popularity is computed with the
project's existing function on training interactions, not prefix-dataset target
frequencies.

The `--max-users` subset is deterministic by raw user ID and `--sample-seed`;
all models receive the same users. Its coverage cannot be directly compared
with the full test. Omit `--max-users` for conclusions over all Beauty users.

## Outputs

Default directory: `artifacts/exp6_result_analysis/beauty/`.

- `report.html`: standalone report with embedded images.
- `figures/*.png`, `figures/*.svg`: seven individual visualizations for export.
- `tables/summary.csv`: HR, NDCG, coverage, ARP, exposure-frequency Gini, effective number of items, and average overlap of different users' lists.
- `tables/comparisons.csv`: paired changes, new/lost items and their exposure shares, before/after recommendation overlap, and bootstrap intervals for NDCG changes.
- `tables/accuracy_attribution.csv`: exact additive decomposition of NDCG change by targets newly appearing in aggregate coverage and targets covered before.
- `tables/users.csv.gz`: each user's metric changes for every SFT seed and K, including history length, target popularity, and new recommendations.
- `tables/items.csv.gz`: exposure counts, popularity, and `new`, `lost`, `retained`, or `unseen` status relative to the corresponding init.
- `tables/popularity.csv`: exposure shares by fixed training-popularity groups.
- `tables/cross_model.csv`: same-user list overlap, aggregate coverage Jaccard, and exposure-frequency correlation for every checkpoint pair.
- `tables/metric_validation.csv`: recomputed metrics compared with original artifacts for the full test. FP32 and the current decoder can produce differences, which are reported.
- `runs.json`, `analysis_metadata.json`: sources and calculation conditions.
- `cache/*.npz`, `cache/*.json`, `cache/catalog.csv`: raw recommendations, verifiable fingerprints, and training popularity. NPZ does not require pickle.

## Analysis

1. **Accuracy and coverage:** arrows from init to each SFT seed in NDCG/coverage coordinates; the initial point is not duplicated.
2. **New targets' contribution to accuracy:** a correct user target that never appeared in any baseline top-K is in the new group. Group contributions sum exactly to overall ΔNDCG. This is descriptive, not causal.
3. **Exposure concentration:** Lorenz curves cover the entire catalog, including items without exposure. Gini and `exp(entropy)` are also calculated for the exposure distribution.
4. **Sources of coverage growth:** exposure frequency and training popularity of newly covered items. Growth from single exposures is separated from substantial redistribution of recommendation positions.
5. **User-level changes:** groups with improved, unchanged, and degraded NDCG, old/new top-K overlap, and the share of newly covered items.
6. **Popularity:** head/middle/tail exposure shares in one common catalog. Equal frequencies are not split, so actual group sizes can differ from nominal 20/30/50%.
7. **Model similarity:** same-user list overlap and aggregate coverage overlap are measured separately.

Coverage describes aggregate reach, not semantic diversity within a list.
Exposure-frequency entropy is not model entropy. Mean overlap between different
users is computed exactly from exposure counts:
`sum_i c_i(c_i-1) / (N(N-1)K)`, without quadratic user-pair enumeration.

Paired user bootstrap estimates ΔNDCG uncertainty (`--bootstrap-samples 500`;
`0` disables it). It describes test-user variability with fixed checkpoints.
With a single selected SFT checkpoint, variability across seeds is not estimated.
Correlation between two architectures is not estimated, and no causal claims
about the SFT mechanism are made.

Validate the calculations with:

```bash
python -m pytest tests/test_beauty_sft_analysis.py -q
```
