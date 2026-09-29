"""Powered rewards must agree with full-catalog Z and every analytical flow.

Exercise the actual GPT-2 dense/factorized and TIGER factorized composition
entrypoints. A policy normalized from an independent brute-force tree must
have zero TB, DB and SubTB residuals for every standard ablation reward/power.
"""
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon as gpt2
import gflow_finetune_yambda500m as tiger
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON as EPS, GFlowConfig
from flowrec.ft_methods.gflow.factorized_cache import CACHE_KIND, FACTORS_FILENAME, FactorizedRewardCache
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.rewards.sasrec import SASRecRewardConfig
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

COMPONENTS = ("sasrec", "interaction", "sim_hier")
COMBOS = [c for n in range(1, 4) for c in itertools.combinations(COMPONENTS, n)]


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("power-flow-caches")
    # Unsorted catalog, shared prefixes, unique terminals.
    tokens = torch.tensor([[3, 2], [1, 1], [2, 1], [1, 2], [2, 2], [3, 1]])
    targets = torch.tensor([1, 4, 3])
    sasrec = torch.tensor([[.05, .9, .2, .4, .8, .65], [.7, .6, .3, .5, .99, .01],
                          [.1, .8, .65, .75, .4, .2]])
    interaction = torch.tensor([[1., 1.25, .6, 1., 1., .8], [1., .7, 1., 1.2, 1., 1.],
                                [.8, 1., 1., 1.25, .6, 1.]])
    similarity_cfg = SimilarityRewardConfig(mode="hierarchical")
    reward_model = SimilarityRewardModel(similarity_cfg, codebooks=torch.tensor([
        [[0., 0.], [.2, .3], [1., 1.]], [[0., 0.], [.5, .2], [1., 1.]]]))
    trajectories = SemanticTrajectoryBatch(tokens.unsqueeze(0).expand(3, -1, -1),
                                            torch.ones(3, 6, 2, dtype=torch.bool),
                                            torch.zeros(3, 6, dtype=torch.bool))
    similarity = reward_model._hierarchical_similarity({"target_token_ids": tokens[targets]}, trajectories)
    raw = {"sasrec": sasrec, "interaction": interaction, "sim_hier": similarity.clamp_min(EPS)}
    factors = {
        "sasrec": {"hidden": torch.logit(sasrec), "item_table": torch.eye(6),
                   "row_valid": torch.ones(6, dtype=torch.bool)},
        "interaction": {"indptr": torch.tensor([0, 6, 12, 18]), "cols": torch.arange(6).repeat(3),
                        "vals": interaction.flatten()},
        "sim_hier": {"target_token_ids": tokens[targets], "item_tokens": tokens,
                     "code_dist": reward_model.code_dist},
    }
    dense_dirs, factor_dirs = {}, {}
    for key in COMPONENTS:
        dense = root / "dense" / key
        dense.mkdir(parents=True)
        torch.save({"full_rewards": raw[key], "reward_sum": raw[key].sum(1),
                    "target_semantic_row_ids": targets}, dense / "shard.pt")
        # There are deliberately NO stored powered sums: composed runs must
        # compute a fresh sum after multiplying components and applying power.
        (dense / "manifest.json").write_text(json.dumps({"num_examples": 3, "stored_power_sums": [],
            "shards": [{"path": "shard.pt", "example_start": 0, "num_examples": 3}]}))
        dense_dirs[key] = dense
        factor = root / "factorized" / key
        factor.mkdir(parents=True)
        torch.save({**factors[key], "target_semantic_row_ids": targets}, factor / FACTORS_FILENAME)
        config = asdict(SASRecRewardConfig(transform="sigmoid")) if key == "sasrec" else (
            asdict(similarity_cfg) if key == "sim_hier" else {})
        (factor / "manifest.json").write_text(json.dumps({
            "kind": CACHE_KIND, "factors_path": FACTORS_FILENAME, "num_examples": 3,
            "num_catalog_items": 6, "reward_epsilon": EPS, "fill_value": 1.,
            "component": {"name": "similarity" if key == "sim_hier" else key,
                          "mode": "hierarchical" if key == "sim_hier" else None, "config": config},
        }))
        factor_dirs[key] = factor
    return raw, tokens, targets, dense_dirs, factor_dirs


def brute_force_flows(tokens, candidate_tokens, rewards):
    # Sum catalog members directly, independent of the production prefix index.
    bsz, n, depth = candidate_tokens.shape
    flows = torch.empty(bsz, n, depth + 1, dtype=torch.float64)
    for b, candidate, length in itertools.product(range(bsz), range(n), range(depth + 1)):
        match = (tokens[:, :length] == candidate_tokens[b, candidate, :length]).all(1)
        flows[b, candidate, length] = rewards[b, match].sum()
    return flows


@pytest.mark.parametrize("backend", ["gpt2_dense", "gpt2_factorized", "tiger_factorized"])
@pytest.mark.parametrize("power", [.25, 1., 4., 16.])
@pytest.mark.parametrize("combo", COMBOS, ids=lambda combo: "_x_".join(combo))
def test_reward_power_normalizer_and_all_balance_losses(caches, backend, power, combo):
    raw, tokens, targets, dense_dirs, factor_dirs = caches
    if backend == "tiger_factorized":
        names = ["similarity" if c == "sim_hier" else c for c in combo]
        component_caches = {n: FactorizedRewardCache(factor_dirs[c]) for n, c in zip(names, combo)}
        cache = tiger.composed_cache({"combo": names, "power": power}, component_caches)
    else:
        dense = backend == "gpt2_dense"
        cache = gpt2.compose_reward_cache(
            {"dataset": "beauty" if dense else "yambda_50m_listens", "components": combo,
             "weights": [1.] * len(combo), "composition": "multiply", "power": power},
            dense_dirs if dense else factor_dirs,
            SimpleNamespace(cache_format="dense" if dense else "factorized"))
    # Scalar reference: compose, floor, exponentiate, floor, THEN sum.
    reference = torch.tensor([[max(max(math.prod(float(raw[c][b, item]) for c in combo), EPS) ** power, EPS)
                               for item in range(6)] for b in range(3)], dtype=torch.float64)
    indices = torch.tensor([2, 0])
    rows = torch.tensor([[-1, 1, 4], [-1, 3, 5]])
    resolved = torch.where(rows < 0, targets[indices, None], rows)
    candidate_tokens = tokens[resolved]
    expected = reference[indices]
    expected_z = expected.sum(1)
    expected_flows = brute_force_flows(tokens, candidate_tokens, expected)
    expected_log_flows = expected_flows.log().float()
    log_policy = expected_log_flows[..., 1:] - expected_log_flows[..., :-1]
    model = SimpleNamespace(
        item_semantic_ids=tokens,
        score_items_valid_actions=lambda *a, **kw: log_policy.sum(-1),
        score_item_tokens_valid_actions=lambda *a, **kw: log_policy,
    )
    sampler = SimpleNamespace(sample=lambda *a, **kw: SemanticTrajectoryBatch(
        candidate_tokens, torch.ones_like(candidate_tokens, dtype=torch.bool), rows.lt(0), rows))
    sft = SimpleNamespace(compute_loss=lambda *a, **kw: MethodOutput(
        loss=torch.tensor(0.), model_output=SimpleNamespace(extra={
            "encoder_last_hidden_state": torch.zeros(2, 1, 1), "encoder_attention_mask": torch.ones(2, 1)})))
    batch = {"example_indices": indices}
    method_type = gpt2.GPT2GFlowMethod if backend.startswith("gpt2") else GFlowMethod
    for objective in ("tb", "db", "subtb"):
        method = method_type(GFlowConfig(objective=objective, reward_cache_path="injected", reward_cache_power=1.),
                             candidate_sampler=sampler, sft_method=sft)
        method._reward_cache = cache
        result = method.compute_loss(model, batch)
        torch.testing.assert_close(result.extra["rewards"].double(), expected.gather(1, resolved),
                                   rtol=2e-5, atol=1e-8)
        torch.testing.assert_close(result.metrics["gflow/log_z_mean"].double(), expected_z.log().mean(),
                                   rtol=2e-5, atol=2e-6)
        assert float(result.metrics["gflow/gflow_loss"]) < 1e-9
        if objective in {"db", "subtb"}:
            trajectories = result.extra["trajectories"]
            _, flows = method._candidate_token_logprobs_and_prefix_flows(
                model, batch, trajectories, result.extra["rewards"])
            torch.testing.assert_close(flows, expected_log_flows, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(flows[..., -1], result.extra["rewards"].log(), rtol=2e-5, atol=2e-6)
        else:
            assert "cache_full_rewards" not in result.extra["trajectories"].extra
