"""New GFlow objectives using the existing candidate samplers."""

import pytest
import torch

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from test_reward_distillation import Cache, CATALOG, ToyModel, batch
from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod


@pytest.mark.parametrize('objective', ['prefix_ce', 'q_distill'])
def test_local_objective_preserves_loss_components_with_existing_sampler(objective):
    model = ToyModel()
    model.item_semantic_ids = CATALOG
    inputs = {**batch(), "target_item_ids": torch.tensor([2, 3])}
    cache = Cache()
    cfg = GFlowConfig.from_dict(dict(objective=objective, candidate_sampler='random',
        num_negative_samples=1, reward_cache_path='unused', gflow_weight=.3))
    assert GFlowConfig.from_dict(cfg.to_dict()) == cfg
    sampler = SemanticIDCandidateSampler(CandidateSamplerConfig(
        mode='random', num_negatives=1, exclude_history=False), item_semantic_ids=CATALOG,
        item_id_to_semantic_row=torch.tensor([-1, 0, 1, 2]))
    method = GFlowMethod(cfg, candidate_sampler=sampler, sft_method=SFTMethod(SFTConfig(
        compute_train_metrics=False, compute_eval_metrics=False)))
    method._reward_cache = cache
    out = method.compute_loss(model, inputs)
    assert out.extra['trajectories'].num_candidates == 2
    parts = out.extra[LOSS_COMPONENTS_KEY]
    torch.testing.assert_close(out.loss, parts['sft'] + .3 * parts['gflow'])
    out.loss.backward()
    assert model.decoder.grad.abs().sum() > 0
    cache.calls.clear()
    held_out = method.compute_loss(model, inputs, stage='validation')
    assert not cache.calls
    torch.testing.assert_close(held_out.loss, held_out.extra[LOSS_COMPONENTS_KEY]['sft'])


@pytest.mark.parametrize('objective', ['prefix_ce', 'q_distill'])
def test_gpt2_local_objective_backpropagates_through_existing_adapter(objective):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from gflow_finetune_gpt2_beauty import GPT2GFlowMethod
    from test_gpt2_reward_distillation import tiny_model, batch as gpt2_batch, Cache as GPT2Cache, sft
    model = tiny_model()
    sampler = SemanticIDCandidateSampler(CandidateSamplerConfig(
        mode='random', num_negatives=3, exclude_history=False),
        item_semantic_ids=model.item_semantic_ids,
        item_id_to_semantic_row=torch.tensor([-1, 0, 1, 2, 3]))
    method = GPT2GFlowMethod(GFlowConfig(objective=objective, candidate_sampler='random',
        num_negative_samples=3, reward_cache_path='unused'), candidate_sampler=sampler, sft_method=sft())
    method._reward_cache = GPT2Cache()
    inputs = {**gpt2_batch(), "target_item_ids": torch.tensor([2, 4])}
    out = method.compute_loss(model, inputs)
    out.extra[LOSS_COMPONENTS_KEY]['gflow'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('module,dataset_flag', [
    ('gflow_finetune_yambda500m', '--dataset'),
    ('gflow_finetune_gpt2_amazon', '--datasets'),
])
def test_existing_runners_accept_new_objectives_with_existing_sampler(module, dataset_flag):
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/exp4-reward-ablations'))
    runner = importlib.import_module(module)
    args = runner.parse_args([dataset_flag, 'instruments', '--objectives', 'prefix_ce', 'q_distill',
                              '--samplers', 'teacher', '--num-candidates', '3', '--list'])
    grid = runner.build_grid(args)
    assert {entry['objective'] for entry in grid} == {'prefix_ce', 'q_distill'}
    assert all(entry['sampler'] == 'teacher' and entry['num_negatives'] == 3 for entry in grid)


from test_gflow_prefetch_equivalence import tiny_problem


@pytest.mark.parametrize('objective', ['prefix_ce', 'q_distill'])
def test_tiger_local_objective_uses_existing_sampler_and_factorized_cache(tiny_problem, objective):
    from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
    from flowrec.ft_methods.sft.config import SFTConfig
    from flowrec.ft_methods.sft.method import SFTMethod
    inputs, model, sampler, paths = tiny_problem
    method = GFlowMethod(GFlowConfig(objective=objective, reward_cache_path='unused'),
        candidate_sampler=sampler, sft_method=SFTMethod(SFTConfig(
            compute_train_metrics=False, compute_eval_metrics=False)))
    method._reward_cache = FactorizedRewardCache(paths['interaction'])
    out = method.compute_loss(model, inputs, step=0)
    out.extra[LOSS_COMPONENTS_KEY]['gflow'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
