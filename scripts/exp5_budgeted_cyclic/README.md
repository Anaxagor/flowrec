# exp5_budgeted_cyclic — cluster pipeline for the Beauty 2014 budgeted cyclic experiment

Corrected, modular successor of the pilot `scripts/exp5-budgeted-cyclic/budgeted_cyclic_amazon.py`.
Answers, on the **complete** validation/test user universe (22,363 users each):

> Can a frozen TIGER recommender be improved at inference time by a small,
> structured rollback budget, and does any cyclic method outperform simpler
> test-time baselines under a fair end-to-end compute budget?

Both models stay frozen. See `SCHEMAS.md` for output schemas and
`MIGRATION.md` for the old→new naming map and the list of pilot issues fixed.

## Layout

Reusable math lives in `flowrec/trajectories/budgeted.py` (exact scalar and
level-reserved finite-budget DPs, budget-dependent d(c,s,b), samplers — unit
tested in `tests/test_budgeted_cyclic.py`). This package adds log-space
rewards/oracle, caches, policies, statistics, and staged cluster scripts:

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
configs/  slurm/     experiment configs and SLURM templates
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

## Cluster execution

**Concrete launchers for the project cluster** (account `proj_1825`, partition
`rocky`, conda env `flowrec` — the house conventions of the other flowrec
sbatch scripts) live in `scripts/slurm/exp5_*.sbatch` +
`scripts/slurm/exp5_submit_pipeline.sh`. NOTE: `scripts/slurm/` is
git-ignored in this repo, so copy those files to the cluster alongside the
checkout (or regenerate from the portable templates below). From the
repository root on the cluster:

```bash
mkdir -p slurm_logs
bash scripts/slurm/exp5_submit_pipeline.sh          # whole chained pipeline
# or individual stages, e.g.:
sbatch scripts/slurm/exp5_03_tiger_array.sbatch                # both splits
sbatch --array=0-15 scripts/slurm/exp5_03_tiger_array.sbatch   # validation only
CFG=... RUN_ID=... bash scripts/slurm/exp5_submit_pipeline.sh  # overrides
```

Stages 02/03/04 fold the split into the array index
(`index = split_index * NSHARDS + shard`; validation first, then test). GPU
stages: 00, 02, 03, 05a controller fit, 08 (benchmark shard), 10; everything
else runs on CPU nodes (`type_b`) — the grid is pure cache arithmetic.

**Portable templates** (any cluster) are in this package under `slurm/` with
`<<ACCOUNT>>/<<PARTITION>>/...` placeholders and an env contract
(`FLOWREC_PROJECT_ROOT`, `EXP5_CONFIG`, `EXP5_SPLIT`, `EXP5_DEVICE`,
`EXP5_ENV_SETUP`); `slurm/submit_pipeline.sh` chains them identically.

Both submitters encode: 00 → 01 → {02,03} → 04 → protocol-regression →
controller-fit → 05 array → 06 (freeze) → 07 → 08 → 09.
**The full grid must not be trusted until `summary/protocol_regression.json`
reports `pass: true`** (spec 5.4); the stage exits non-zero otherwise.

Every sharded stage supports `--resume` (skips completed shards; atomic
temp-file+rename writes), deterministic contiguous user→shard assignment,
completeness validation before aggregation (missing/duplicate users are
errors), `--dry-run` workload estimates, and records SLURM job/array ids,
hostname, GPU model and library versions in shard metadata.

Re-running stage 06 re-freezes the selection manifest (its hash changes);
stage 07 then refuses stale test predictions — delete `predictions/test/`
and re-run 07 after any re-selection.

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
| 03 TIGER sweep | ~0.43 s/user (CPU) | ~2.7 CPU-h/split; 16-way array ≈ 10 min wall; GPU faster |
| 04 flow | ~0.03 s/user-target | ~2 CPU-h/split |
| 05 grid (~268 configs) | ~0.5–1 s/user | ~6 CPU-h; 16-way array ≈ 25 min wall |
| 05 controller fit (512 users × 200 steps) | — | 2–6 h single job (GPU recommended) |
| 07 test (~15 configs) | ~0.1 s/user | ~40 min single; minutes sharded |
| 08 serving (5 seeds × 4 budgets) | ~0.1 s/user | ~1–2 h sharded |

**Total ≈ 15–25 GPU/CPU-hours; wall < ~2 h with the suggested arrays.**
Stages 04–09 are pure cache arithmetic — CPU nodes suffice; only 02/03 (and
the 08 `--benchmark` rows) benefit from a GPU.

## Cost-regime discipline (spec 9)

Every prediction row carries `requires_full_catalog_reward` and
`online_expected_tiger_calls`; per-config `ComputeCounters` (offline-cache
regime) land in `predictions/<split>/compute/`, cache-construction counters in
the cache shard metas, and genuine end-to-end latency only in
`summary/latency_benchmark.csv` (CUDA-synced, warm-up, percentiles).
`sasrec_full_catalog` is never zero-cost and cached guidance is never "4
decoder evals" — compare methods via `compute_metrics.csv`.
