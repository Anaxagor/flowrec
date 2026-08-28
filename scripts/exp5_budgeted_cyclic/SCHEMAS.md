# Output schemas (schema_version exp5.2)

Artifact root: `artifacts/exp5_budgeted_cyclic/beauty2014/<run_id>/`.
Every summary row carries `run_id, git_commit, split, n_users, policy_family,
canonical_config_hash, reward_transform, seed, cache_mode`.

## manifest/

- `run_manifest.json` — run id, config path, expected invariants, trie
  metadata + hash, resolved assets `{logical, resolved, sha256, source}`,
  environment (hardware and library versions).
- `asset_checksums.json` — per asset: logical path, resolved path, sha256,
  resolution source. Stage 00 refuses to run on mismatch.
- `validation_users.parquet` / `test_users.parquet` — one row per user:
  `raw_user_id, internal_user_id, dataset_index, split, history_item_ids
  (list, context window), history_length, target_item_id, target_in_catalog,
  target_has_sid, eligible, exclusion_reason`.
- `selected_configs_full_validation.json` (+ `.sha256`) — frozen selection:
  per family `{family, config_name, canonical_config_hash, canonical_config,
  reward_transform, target, validation_metric_name, validation_metric,
  selection_rule}`, plus `selection_metric, validation_artifact_hash,
  code_git_commit, selection_timestamp, _selection_manifest_sha256`.
- `controllers/controller__<target>__B<b>.npz` — `eta [B+1, L+1, 5]`, budget,
  fit user list, target name.
- `gate/gate_model.joblib`, `gate/gate_meta.json` — frozen gate
  (`model_kind, candidate_policies, feature_names, label_metric, cv_accuracy`).

## cache/

- `sasrec/<split>/shard_XXXXX.npz` — `raw_scores [n_users, 12101] float32`
  (semantic-row order). Meta: `schema_version, kind, split, shard_id,
  num_shards, dataset_indices, manifest_rows, item_order_hash,
  batch_vs_single_max_err, counters, environment`.
- `tiger/<split>/shard_XXXXX.npz` — `edge_logprob [n_users, 32879] float32`
  (child-node indexed; root = 0; renormalized to float64 on read),
  `raw_valid_mass [n_users, 20778] float32` (internal nodes). Meta adds
  `trie_hash, model_validation` (cached path-sums vs
  `score_items_valid_actions`, spec 18.3).
- `flow/<split>/shard_XXXXX.parquet` — per (user, target): `oracle_ok,
  oracle_error, reward_* stats (raw/log min/max, reward_floor,
  num_floor_clamped, num_underflow_prevented, target_entropy,
  effective_support, shift_b)`, oracle diagnostics (`n_support_violation,
  n_proposal_logp_underflow, n_saturated_rollback, max_abs_error,
  max_relative_error_with_denominator_floor,
  max_relative_error_on_edges_with_target_mass_above_threshold,
  mass_weighted_relative_error, ...`), `E_del_inf, E_n_inf, E_del_inf_L1..L4`,
  clipping record (`n_impl_clipped, max_clip_probability_effect, clip_value`).

## predictions/

`<split>/<family_dir>/<config_name>/shard_XXXXX.parquet`, one row per user:

- identity: `user, dataset_index, policy_family, config_name,
  canonical_config_hash, reward_transform, target,
  requires_full_catalog_reward, status`, grid params
  (`budget, alpha, tau, lam, beam, n_samples, allocation` — NaN/None when not
  applicable), `selection_manifest_sha256` (test only).
- `eval_kind = full_catalog`: exact `full_catalog_rank`, `recall@{5,10,20}`,
  `ndcg@{5,10,20}`, `mrr`, `top20`; candidate fields are null.
- `eval_kind = candidate` (beam / rerank): `candidate_set_size,
  candidate_hit, rank_within_candidates (null when absent),
  censored_rank = M+1 on miss`; `full_catalog_rank` is ALWAYS null; the
  recall/ndcg columns are computed from the censored rank.
- alignment (target-directed policies): `tv_to_target, js_to_target,
  kl_to_target, expected_raw_score, expected_target_logw`.
- cyclic behavior: `expected_steps, expected_deletes, p_any_delete,
  p_full_budget, same_token_reselect, deletes_L1..L4, reselect_L1..L4,
  online_expected_tiger_calls, n_d_star_impl_clipped,
  check_b0_equals_exact` (append-only reference only).

`<split>/compute/shard_XXXXX.parquet` — per config: full `ComputeCounters`
(spec 9 field list) + `cache_mode`; a `target_construction` pseudo-row carries
the shared reward/oracle build arithmetic.

`<split>/gate_features/shard_XXXXX.parquet` — per user: the spec-25 feature
set (`history_length, tiger_entropy_L*, tiger_margin_L*, raw_valid_mass_L*,
sasrec_score_entropy, tiger_sasrec_top20_overlap,
tiger_sasrec_rank_corr_top50, js_tiger_sasrec, log_C_max, log_C_q90,
predicted_oracle_cost, suffix_collision_multiplicity`).

`serving/<split>/shard_XXXXX.parquet` — per (user, method, n_trajectories,
seed): `n_unique_candidates, duplicate_rate, pure_recall@K, pure_ndcg@K,
candidate_recall@M, rerank_M{M}_recall@K, rerank_M{M}_ndcg@K,
expected_deletes, mean_steps, same_token_reselect, revision_precision,
sampled_trajectory_transitions, num_add/delete/stop_actions,
online_tiger_calls_if_uncached, cached_policy_runtime_s` (cached arithmetic —
NOT serving latency).

## summary/

- `validation_metrics.csv` / `test_metrics.csv` — per config: mean per-user
  metrics + `n_users, n_failed, coverage@{10,20}` with user-resampled CIs.
- `test_bootstrap.csv` — per (config, metric): `mean_delta, ci_lo, ci_hi,
  bootstrap_prob_improvement` (NOT a p-value), `bootstrap_prob_degradation,
  p_value_two_sided, holm_adjusted_p` (primaries only), `is_primary, n_users,
  n_boot`. Reference = append-only cyclic parameterization; paired by user.
- `compute_metrics.csv` — all counter rows (policies, target construction,
  cache builds).
- `serving_metrics.csv` — seed-averaged (per-user mean over seeds, then user
  mean; `seed_aggregation` column documents this).
- `oracle_regression.csv` — flow diagnostics aggregated by (split, target).
- `user_slice_metrics.csv` — DIAGNOSTIC user subsets
  `dominant_oracle_cost_level_L*` (spec 13: slices, not policies).
- `by_level.csv` — level-policy summaries per split.
- `latency_benchmark.csv` — `label, batch_size, caches_enabled, device,
  gpu_model, reps, median_s, p90_s, p95_s, p99_s` (true end-to-end model
  latency only).
- `protocol_regression.json` (+ per-user parquet) — three-evaluator
  agreement + documented disagreement causes + `pass`.

## report/

`report.md` / `report.json` — answers to the spec §31 questions with the §15
language constraints (validation-selected primary / sensitivity / post-hoc
diagnostic / oracle upper bound / serving-feasible labels).
