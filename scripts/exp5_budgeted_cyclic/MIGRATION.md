# Budgeted cyclic experiment: protocol and terminology

The staged pipeline evaluates frozen recommenders under an inference-time
rollback budget. See [README.md](README.md) for the pipeline and
[SCHEMAS.md](SCHEMAS.md) for output fields.

## Policies and references

- `truncated_infinite_oracle` and `scaled_truncated_infinite_oracle` impose
  a hard cap on the infinite-horizon oracle; they are not finite-budget-optimal
  policies.
- `local_prefix_guidance` interpolates next-token conditionals. The
  `global_hybrid_target` is a globally normalized terminal hybrid
  `(1-a)·log p0 + a·(r-b)/T` and is a different target.
- `no_repeat_heuristic` is a serving-only Monte Carlo policy.
- The reference families distinguish exact full-catalog TIGER ranking,
  append-only cyclic parameterization, TIGER beam and stochastic generation,
  and full-catalog SASRec. The paired-test reference is the append-only cyclic
  parameterization, validated against exact ranking for every user.
- `rollback_only_L*`, `rollback_semantic_L1_L3`, `rollback_suffix_L4` and
  `rollback_all_levels` are full-user policies in `level_budget_rollback`.
  `dominant_oracle_cost_level_L*` labels user slices in
  `user_slice_metrics.csv`; these are diagnostics, not separate policies.

## Identity, numerical behavior and results

- `paths.py` resolves inputs from CLI, environment or repository-relative
  defaults. Manifests record resolved and logical paths and SHA-256 hashes.
- Split manifests record the full user universe and exclusions. Comparisons
  validate that each policy is evaluated on the same users. Stage 07 verifies
  and records the hash of the frozen selection manifest.
- The exact finite-budget and level-reserved dynamic programs support
  budget-dependent `d(c,s,b)`, per-level quotas and state-visitation mass.
- Rewards, prefix masses and the oracle operate in log space. Numerical
  diagnostics distinguish support violations, underflow, saturation and
  implementation clipping and report scale-aware error measures.
- Transform-independent policies are stored once with `transform=None`.
  Canonical hashes and duplicate-prediction guards protect aggregation.
- Candidate-limited methods report `candidate_hit`, `rank_within_candidates`
  and `censored_rank`; `full_catalog_rank` is unavailable for those rows.
- `ComputeCounters`, `online_expected_tiger_calls` and
  `requires_full_catalog_reward` account for separate costs. Cached-policy
  runtime and end-to-end latency are reported separately.
- `bootstrap_prob_improvement` is a probability, not a p-value. Paired
  inference averages stochastic serving seeds per user before user bootstrap,
  reports two-sided p-values and Holm adjustment for the predeclared family,
  and computes coverage confidence intervals by user resampling.
- Manifests, sharding and atomic writes support deterministic assignment and
  resumption. See the pipeline README for the ordered stages and commands.
