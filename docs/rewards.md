# Rewards and candidate sampling

FlowRec represents a set of semantic-item candidates with
[`SemanticTrajectoryBatch`](../flowrec/trajectories/semantic.py). Reward
models score candidates using the batch and trajectories. The components in
[`flowrec/rewards`](../flowrec/rewards) cover:

| Reward | Description | Required inputs |
| --- | --- | --- |
| Positive | Base reward for a positive target versus other candidates. | Positive-candidate mask. |
| Interaction | Logged evidence about a user and item; Amazon ratings and Yambda feedback have dataset-specific scoring. | An interaction store and, optionally, a positive-history store. Only evidence strictly before the target time is used. |
| Similarity | Similarity of a candidate to the target: exact semantic-token match, hierarchical codebook similarity, or content-embedding cosine similarity. | Semantic IDs; hierarchical mode also needs RQ-VAE codebooks, cosine mode needs item embeddings. |
| SASRec / SVD | Transformed scores from a collaborative recommender. | The matching scorer and its item catalog. |
| Mixed / multiplicative | A weighted sum or product of other rewards. | Component rewards with the same candidate shape. |

See [`interaction.py`](../flowrec/rewards/interaction.py),
[`similarity.py`](../flowrec/rewards/similarity.py),
[`sasrec.py`](../flowrec/rewards/sasrec.py),
[`svd.py`](../flowrec/rewards/svd.py) and
[`mix.py`](../flowrec/rewards/mix.py) for configuration fields and exact
transforms. Interaction scoring treats unobserved candidates according to its
configured fallback; it is not evidence of a negative interaction.

## Combining rewards

`MultiplicativeRewardModel` computes the product of component rewards raised
to their respective weights. `MixedRewardModel` computes a weighted sum and
can normalize components. These models floor rewards at an epsilon suitable
for log-space objectives. A batch- or candidate-set-dependent normalizer
changes a reward when the candidate set changes; use pointwise components and
compatible normalization when an objective relies on a precomputed
full-catalog partition value. See the composition classes for their defaults
and `uses_candidate_set_normalization` behavior.

## GFlow reward caches

[`GFlowMethod`](../flowrec/ft_methods/gflow/method.py) combines SFT with a
GFlow objective and requires `GFlowConfig.reward_cache_path`. It reads cached
terminal rewards rather than evaluating a live reward model in each step. The
cache must match the dataset split, example ordering, item catalog, semantic
IDs and reward configuration. A mismatched or missing cache should be rebuilt
for the intended run.

[`precompute_reward_cache.py`](../scripts/precompute_reward_cache.py) builds
full-catalog reward rows; [`precompute_reward_factors.py`](../scripts/precompute_reward_factors.py)
provides a factorized representation for supported components. See their
`--help` for inputs, subset controls and verification options, and the
[experiment documentation](../scripts/exp4-reward-ablations/README.md) for
configured workflows. Live rewards are also used by other methods where their
interfaces support them; their configuration is not interchangeable with a
GFlow reward cache.

## Candidate sampling

[`SemanticIDCandidateSampler`](../flowrec/candidates/sampler.py) supports:

- `random`: uniform sampling from allowed catalog items.
- `curriculum`: score candidates with an item model and move from easier to
  harder negatives during the configured schedule.
- `top_n`: select the highest-scoring allowed negatives.
- `policy`: draw without replacement from the model's constrained
  autoregressive policy; requires a compatible generative model.

Negative candidates exclude the positive target; the sampler may also include
the positive separately. With `exclude_history=True`, negatives also exclude
the history present in the batch; older interactions outside a
truncated history may still be eligible. The GFlow config's
`candidate_sampler` accepts `random`, `sasrec` or `policy`; its runner maps
those choices to a sampler and may require a SASRec checkpoint. Set the
sampler, data processing and cached rewards consistently with the target
experiment.
