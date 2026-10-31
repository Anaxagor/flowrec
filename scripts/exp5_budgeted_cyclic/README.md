# exp5_budgeted_cyclic — pipeline for the Beauty 2014 budgeted cyclic experiment

The pipeline evaluates the following question on the validation/test user universe:

> Can a frozen TIGER recommender be improved at inference time by a small,
> structured rollback budget, and does any cyclic method outperform simpler
> test-time baselines under a fair end-to-end compute budget?

Both models stay frozen. See `SCHEMAS.md` for output schemas.

## Layout

Reusable math lives in `flowrec/trajectories/budgeted.py` (exact scalar and
level-reserved finite-budget DPs, budget-dependent d(c,s,b), samplers — unit
tested in `tests/test_budgeted_cyclic.py`). This package adds log-space
rewards/oracle, caches, policies, statistics, and staged scripts:

```
paths.py             asset resolution: CLI > FLOWREC_DATA_ROOT/PROJECT_ROOT > repo > fail
common.py            CLI, sharding, atomic IO, cache readers, asset loading
manifest.py          run/split manifests, frozen selection manifest
trie.py              invariant validation + CSR export
rewards.py           log-space transforms, prefix masses, hybrid targets
oracle.py            log-space global flow oracle + scale-aware diagnostics
tiger_scoring.py     TIGER prefix cache, exact ranking, cache-vs-model checks
sasrec_scoring.py    SASRec full-catalog raw-score cache
finite_budget_dp.py  DP wrappers + budget-aware controller (spec 23)
policies.py          taxonomy, canonical hashing, per-user evaluation
metrics.py           full-catalog vs candidate-censored rows (spec 8)
compute_counters.py  explicit compute accounting + latency benchmark (spec 9-10)
bootstrap.py         paired bootstrap, Holm, coverage uncertainty (spec 14)
gate.py              user-adaptive policy gate (spec 25)
reports.py           aggregation, plots, report generation (spec 15/31)
00..09_*.py          pipeline stages; protocol_regression.py (spec 5.4)
configs/             experiment configs
```

## Local smoke test (minutes, CPU)

```bash
CFG=scripts/exp5_budgeted_cyclic/configs/beauty2014_smoke.yaml
python scripts/exp5_budgeted_cyclic/00_validate_assets.py    --config $CFG
python scripts/exp5_budgeted_cyclic/01_build_split_manifest.py --config $CFG
for SPLIT in validation test; do
  python scripts/exp5_budgeted_cyclic/02_precompute_sasrec_scores.py --config $CFG --split $SPLIT --user-start 0 --user-end 16
  python scripts/exp5_budgeted_cyclic/03_precompute_tiger_prefix_scores.py --config $CFG --split $SPLIT --user-start 0 --user-end 16
  python scripts/exp5_budgeted_cyclic/04_build_flow_caches.py --config $CFG --split $SPLIT --user-start 0 --user-end 16
done
python scripts/exp5_budgeted_cyclic/05_run_validation_grid.py --config $CFG --fit-controller-only
python scripts/exp5_budgeted_cyclic/05_run_validation_grid.py --config $CFG --split validation --user-start 0 --user-end 16
python scripts/exp5_budgeted_cyclic/06_select_validation_configs.py --config $CFG --allow-partial-users
python scripts/exp5_budgeted_cyclic/07_run_full_test.py --config $CFG --user-start 0 --user-end 16
python scripts/exp5_budgeted_cyclic/08_run_serving_evaluation.py --config $CFG --split test --user-start 0 --user-end 16
python scripts/exp5_budgeted_cyclic/09_aggregate_and_report.py --config $CFG --allow-partial-users --n-boot 500
python scripts/exp5_budgeted_cyclic/protocol_regression.py --config $CFG --n-users 12
```

Unit tests: `python -m pytest tests/test_budgeted_cyclic.py tests/test_exp5_pipeline.py`
(19 + 22 tests). `--allow-partial-users` exists ONLY for smoke/regression runs;
production aggregation enforces full-manifest coverage and identical user sets
across every policy.

## Full execution

Use the same Python stages as the smoke workflow with the full experiment
configuration. Partition user ranges with `--user-start` and `--user-end` and
keep each shard's range fixed on restart. The dependency order is:

`00 → 01 → {02,03} → 04 → protocol_regression → controller fit → 05 → 06 → 07 → 08 → 09`.

The full grid requires `summary/protocol_regression.json` to report `pass: true`.
Use the full user universe and omit `--allow-partial-users` for reported results.
Each stage supports `--help`; sharded stages support deterministic assignment,
`--resume`, atomic writes, and completeness checks before aggregation.
Metadata records the GPU model and library versions needed to interpret timings.

Re-running stage 06 changes the frozen selection manifest. Regenerate stage 07
predictions after re-selection; stale test predictions are rejected.

## Expected cache sizes and compute (measured on the 16-user smoke, CPU)

| artifact | size |
|---|---|
| SASRec cache | 22,363 × 12,101 float32 ≈ **1.1 GB/split** |
| TIGER cache | edge_logprob 22,363 × 32,879 + raw_valid_mass 22,363 × 20,778 float32 ≈ **4.8 GB/split** |
| flow diagnostics | ~60 MB/split |
| validation predictions (~268 configs) | ~1.5 GB |
| test + serving predictions | ~200 MB |

| stage | measured rate | full-split estimate |
|---|---|---|
| 02 SASRec | ~150 users/s (CPU) | ~3 min/split/shard-set |
| 03 TIGER sweep | ~0.43 s/user (CPU) | ~2.7 CPU-h/split; 16 parallel shards ≈ 10 min wall; GPU faster |
| 04 flow | ~0.03 s/user-target | ~2 CPU-h/split |
| 05 grid (~268 configs) | ~0.5–1 s/user | ~6 CPU-h; 16 parallel shards ≈ 25 min wall |
| 05 controller fit (512 users × 200 steps) | — | 2–6 h single job (GPU recommended) |
| 07 test (~15 configs) | ~0.1 s/user | ~40 min single; minutes sharded |
| 08 serving (5 seeds × 4 budgets) | ~0.1 s/user | ~1–2 h sharded |

**Total ≈ 15–25 GPU/CPU-hours; wall < ~2 h with the parallel shards.**
Stages 04–09 are pure cache arithmetic — CPU execution suffices; only 02/03 (and
the 08 `--benchmark` rows) benefit from a GPU.

## Cost-regime discipline (spec 9)

Every prediction row carries `requires_full_catalog_reward` and
`online_expected_tiger_calls`; per-config `ComputeCounters` (offline-cache
regime) land in `predictions/<split>/compute/`, cache-construction counters in
the cache shard metas, and genuine end-to-end latency only in
`summary/latency_benchmark.csv` (CUDA-synced, warm-up, percentiles).
`sasrec_full_catalog` is never zero-cost and cached guidance is never "4
decoder evals" — compare methods via `compute_metrics.csv`.
