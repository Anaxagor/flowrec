# Migration from the pilot (`scripts/exp5-budgeted-cyclic/`)

## Policy / metric renames (spec §12, §14)

| pilot name | corrected name | why |
|---|---|---|
| `truncated_oracle` | `truncated_infinite_oracle` | it truncates the INFINITE-horizon oracle with a hard cap; it is not a finite-budget-optimal policy |
| `scaled_oracle` | `scaled_truncated_infinite_oracle` | same |
| `direct_guidance` | `local_prefix_guidance` (family `local_guidance`) | local interpolation of next-token conditionals |
| — (new) | `global_hybrid_target` | globally normalized terminal hybrid `(1-a)·log p0 + a·(r-b)/T`; NOT identical to local guidance |
| `no_repeat` | `no_repeat_heuristic` | serving-only Monte Carlo policy |
| family `baseline` (mixed) | `reference/tiger_exact_full_catalog`, `reference/append_only_cyclic_parameterization`, `reference/tiger_beam`, `reference/tiger_stochastic`, `reference/sasrec_full_catalog` | non-equivalent evaluators must not share a label; the paired-test reference is the append-only cyclic parameterization (== exact ranking, asserted per user) |
| `level_ablation` rows `top_level_L*` | user slices `dominant_oracle_cost_level_L*` in `user_slice_metrics.csv` | they were user subsets, not policies; true policy ablations are `rollback_only_L*` / `rollback_semantic_L1_L3` / `rollback_suffix_L4` / `rollback_all_levels` (family `level_budget_rollback`, full user set) |
| `p_improved` | `bootstrap_prob_improvement` | it is not a p-value; `p_value_two_sided` and `holm_adjusted_p` added |
| `decoder_evals` | `ComputeCounters` field set + `online_expected_tiger_calls` + `requires_full_catalog_reward` | a single scalar hid the reward-model and full-catalog costs |
| `runtime_per_trajectory_ms` | `cached_policy_runtime_s` + `latency_benchmark.csv` (`end_to_end`) | cached arithmetic was mislabeled as serving latency |
| `target_rank` (beam rows) | `candidate_hit / rank_within_candidates / censored_rank (=M+1) / full_catalog_rank=NA` | catalog-scale ranks were fabricated for candidate-limited methods |

## Pilot issues found and how each was fixed (spec §32.9)

1. **Machine-specific absolute paths** in artifacts/configs → `paths.py`
   resolver (CLI > env > repo-relative > hard fail), logical+resolved paths and
   SHA-256 checksums stored in every manifest (stage 00 refuses on mismatch).
2. **Fixed 500-user lists** → full-universe split manifests built from the
   dataset (stage 01), per-user eligibility/exclusion rows, identical-user-set
   assertion across all policies (`validate_same_users`).
3. **Mixed non-equivalent TIGER evaluations under one label** (beam subset of
   128 examples vs exact ranking) → explicit reference taxonomy + the
   three-evaluator protocol regression (`protocol_regression.py`, spec 5.4);
   `standard_eval_max_batches=2` is gone from full experiments.
4. **SASRec selected as generic "baseline"** while the paired test compared
   against append-only → family taxonomy fixed; SASRec is an external
   reward-model reference; the paired reference is
   `reference/append_only_cyclic_parameterization`, asserted numerically equal
   to the exact ranking per user (`check_b0_equals_exact`).
5. **Transform-independent methods duplicated per reward transform**
   (identical Best-of-N/beam/stochastic rows entering the test table several
   times) → `transform=None` configs stored once, canonical config hash,
   `assert_unique_prediction_sets` guard with declared identity pairs exempt,
   redundant `tau=1.0` reference removed.
6. **Catalog-scale ranks fabricated for beam candidates** → censored candidate
   schema (spec §8); `full_catalog_rank` is NA for candidate methods.
7. **`decoder_evals` unfair compute accounting** (guidance "4 evals",
   `sasrec_full` at 0 cost) → full counter set, two cost regimes, per-stage
   counter rows including cache construction.
8. **Sub-millisecond "runtime" from cached arithmetic** → benchmark utility
   with CUDA sync/warm-up/percentiles; cached vs end-to-end kept separate.
9. **Linear-space rewards underflowed at sharp temperatures** (1e-30 floors,
   degenerate identity checks, `max_rel_err_net_flow == 1` from ~0
   denominators) → full log-space rewards/masses/oracle; scale-aware error
   metrics (`max_abs_error`, denominator-floored and mass-thresholded relative
   errors, mass-weighted error); support-violation vs underflow vs saturation
   vs implementation-clipping classified and counted (d* clipping is explicit
   and recorded).
10. **`check_infinite_tv_computable` serialized as NaN/missing** → `json_safe`
    keeps booleans booleans; parquet nullable types for candidate fields.
11. **Level "ablation" rows mixed policies with user subsets** → split into
    true full-user policies and diagnostic slices (see rename table).
12. **No selection freeze** (test could rerun with edited selections) →
    hashed immutable selection manifest; stage 07 verifies and stamps every
    test shard with the hash.
13. **Single-seed stochastic rows; `p_improved` treated as significance** →
    ≥5 serving seeds, per-user seed averaging before the user bootstrap,
    two-sided bootstrap p-values, Holm correction over the predeclared primary
    family, coverage CIs by user resampling.
14. **Report over-claims** ("reward target is the bottleneck", cached
    guidance ≈ 4 evals) → report generator emits the §15 label vocabulary and
    the multi-cause interpretation constraints verbatim.
15. **Monolithic script** → staged package with sharding, atomic writes,
    resume, deterministic user assignment, scheduler arrays, dependency-chained
    submission.

## What carried over unchanged

- The exact finite-budget DP, level-reserved DP, samplers, and their 19 unit
  tests (`flowrec/trajectories/budgeted.py`) — extended with budget-dependent
  d(c,s,b), per-level quotas, and state-visitation mass.
- The frozen-asset loading recipe (exp2 config + data-path overrides), the
  leave-one-out split, SID offset conventions, and the trie construction.
- Pilot artifacts remain at `artifacts/non_acyclic_gflownet/beauty_2014/…`
  for regression comparison (stage B of the spec's protocol); new artifacts
  live under `artifacts/exp5_budgeted_cyclic/beauty2014/<run_id>/`.
