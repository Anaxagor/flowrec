"""Full-catalog CE values, gradients, sampling, and held-out isolation."""

import pytest
import torch
import torch.nn.functional as F

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.ft_methods.gflow.prefetch import PREPARED_REWARDS_KEY
from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.models.catalog import _CatalogActionIndex


CATALOG = torch.tensor([[1, 3], [1, 4], [2, 3]])
REWARDS = torch.tensor([[1., 3., 2.], [4., 1., 5.]])


class Cache:
    def __init__(self, rewards=REWARDS):
        self.rewards = rewards
        self.calls = []

    def select_candidates(self, indices, candidate_rows, device, *, include_full=False, prepared=None):
        self.calls.append((candidate_rows.clone(), include_full, prepared))
        rows = self.rewards[indices].to(device)
        return rows.gather(1, candidate_rows), rows.sum(1), rows if include_full else None


class ToyModel(FlowRecModel):
    def __init__(self, catalog=CATALOG):
        super().__init__()
        generator = torch.Generator().manual_seed(27)
        self.encoder = torch.nn.Parameter(torch.randn(2, 3, generator=generator) * .2)
        self.decoder = torch.nn.Parameter(torch.randn(3, 6, generator=generator) * .3)
        self.prefix = torch.nn.Parameter(torch.randn(6, 6, generator=generator) * .2)
        self.action_index = _CatalogActionIndex(catalog)
        self.encode_calls = 0
        self.candidate_counts = []

    def logits(self, hidden, tokens):
        prefixes = torch.cat([torch.zeros_like(tokens[..., :1]), tokens[..., :-1]], dim=-1)
        return (hidden[:, 0] @ self.decoder)[:, None, None, :] + self.prefix[prefixes]

    def forward(self, batch):
        self.encode_calls += 1
        hidden = (batch["x"] @ self.encoder)[:, None, :]
        return RecModelOutput(
            logits=self.logits(hidden, batch["target_token_ids"][:, None, :])[:, 0],
            extra={"encoder_last_hidden_state": hidden,
                   "encoder_attention_mask": torch.ones(len(hidden), 1, dtype=torch.long)},
        )

    def score_item_tokens_valid_actions(self, batch, *, candidate_token_ids, candidate_token_mask):
        self.candidate_counts.append(candidate_token_ids.size(1))
        logits = self.logits(batch["_tiger_encoder_last_hidden_state"], candidate_token_ids)
        depth = candidate_token_ids.size(-1)
        return self.action_index.token_logprobs(
            logits.reshape(-1, depth, 6), candidate_token_ids.reshape(-1, depth),
            candidate_token_mask.reshape(-1, depth),
        ).reshape(candidate_token_ids.shape)

    def candidate_token_logits(self, batch, *, candidate_token_ids, candidate_token_mask):
        self.candidate_counts.append(candidate_token_ids.size(1))
        return self.logits(batch["_tiger_encoder_last_hidden_state"], candidate_token_ids)


def batch():
    return {
        "x": torch.tensor([[.5, 1.], [-.2, .7]]),
        "user_ids": torch.arange(2),
        "example_indices": torch.arange(2),
        "target_token_ids": CATALOG[[1, 2]],
        "target_token_mask": torch.ones(2, 2, dtype=torch.bool),
        # Seen items remain in the full teacher distribution.
        "history_token_ids": CATALOG[[0, 1]][:, None, :],
        "history_item_ids": torch.tensor([[0], [1]]),
    }


def method(cache=None, **kwargs):
    return RewardDistillationMethod(
        Cache() if cache is None else cache, CATALOG,
        SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False)),
        **kwargs,
    )


def reference(model, inputs, rewards=REWARDS, catalog=CATALOG):
    """Enumerate catalog prefixes independently of the scoring index."""
    context = inputs["x"] @ model.encoder
    log_prob_rows = []
    for example in range(len(context)):
        log_probs = []
        for item in catalog:
            log_prob = context.new_zeros(())
            for depth in range(item.numel()):
                allowed = catalog[(catalog[:, :depth] == item[:depth]).all(-1), depth].unique()
                previous = int(item[depth - 1]) if depth else 0
                logits = context[example] @ model.decoder + model.prefix[previous]
                log_prob = log_prob + logits[item[depth]] - logits[allowed].logsumexp(0)
            log_probs.append(log_prob)
        log_prob_rows.append(torch.stack(log_probs))
    log_probs = torch.stack(log_prob_rows)
    q = rewards / rewards.sum(-1, keepdim=True)
    ce = -(q * log_probs).sum(-1).mean()
    targets = inputs["target_token_ids"]
    shifted = torch.cat([torch.zeros_like(targets[:, :1]), targets[:, :-1]], dim=1)
    logits = (context @ model.decoder)[:, None, :] + model.prefix[shifted]
    sft = F.cross_entropy(logits.reshape(-1, 6), targets.flatten())
    return ce, sft, log_probs


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_exact_value_gradient_and_single_shared_encoder(chunk_size):
    model = ToyModel()
    inputs = batch()
    trainer = method(estimator="exact", catalog_chunk_size=chunk_size, distill_weight=.7)
    expected_ce, expected_sft, log_probs = reference(model, inputs)
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(2))
    expected_grads = torch.autograd.grad(expected_sft + .7 * expected_ce, tuple(model.parameters()))

    result = trainer.compute_loss(model, inputs)
    parts = result.extra[LOSS_COMPONENTS_KEY]
    torch.testing.assert_close(parts["distill"], expected_ce)
    torch.testing.assert_close(parts["sft"], expected_sft)
    result.loss.backward()
    for parameter, expected in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)
    assert model.encode_calls == 1
    assert max(model.candidate_counts) <= chunk_size
    # One branching root and one branching prefix, each recomputed in backward.
    assert len(model.candidate_counts) == 4
    assert result.metrics["distill/exact_branching_prefixes"] == 2
    assert trainer.cfg.normalization == "valid_actions"


def test_sampled_estimator_mean_and_gradient_approach_exact_ce():
    torch.manual_seed(72)
    model = ToyModel()
    trainer = method(num_samples=30000)
    exact, _, _ = reference(model, batch())
    exact_grads = torch.autograd.grad(exact, tuple(model.parameters()))
    result = trainer.compute_loss(model, batch())
    sampled = result.extra[LOSS_COMPONENTS_KEY]["distill"]
    torch.testing.assert_close(sampled, exact, atol=.01, rtol=0)
    sampled_grads = torch.autograd.grad(sampled, tuple(model.parameters()))
    for actual, expected in zip(sampled_grads, exact_grads):
        torch.testing.assert_close(actual, expected, atol=.01, rtol=0)
    rows = result.extra["distill_sampled_item_rows"]
    assert rows.shape == (2, 30000)
    assert set(rows[0].tolist()) == {0, 1, 2}
    assert (rows[0] == 0).any()  # history item, retained in teacher support


def test_sampled_draws_full_q_with_replacement_without_positive_injection(monkeypatch):
    def draw(q, num_samples, replacement):
        torch.testing.assert_close(q, REWARDS / REWARDS.sum(-1, keepdim=True))
        assert num_samples == 2 and replacement is True
        return torch.tensor([[0, 0], [0, 1]])

    monkeypatch.setattr(torch, "multinomial", draw)
    model = ToyModel()
    result = method(num_samples=2).compute_loss(model, batch())
    _, _, full_log_probs = reference(model, batch())
    expected = -torch.stack([full_log_probs[0, 0], full_log_probs[0, 0],
                             full_log_probs[1, 0], full_log_probs[1, 1]]).mean()
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], expected)


def test_teacher_detached_and_prepared_rows_forwarded():
    rewards = REWARDS.clone().requires_grad_()
    cache = Cache(rewards)
    trainer = method(cache, estimator="exact")
    inputs = batch()
    prepared = {"raw_factors": "test-marker"}
    inputs[PREPARED_REWARDS_KEY] = prepared
    result = trainer.compute_loss(ToyModel(), inputs)
    result.loss.backward()
    assert rewards.grad is None
    rows, include_full, actual_prepared = cache.calls[0]
    assert rows.shape == (2, 0) and include_full is True and actual_prepared is prepared
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    torch.testing.assert_close(result.metrics["distill/teacher_target_prob"], q[[0, 1], [1, 2]].mean())


@pytest.mark.parametrize("stage", ["val", "test"])
@pytest.mark.parametrize("sampling", ["reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"])
def test_held_out_only_calls_sft_without_cache_or_example_indices(stage, sampling):
    trainer = method(object(), sampling=sampling, curriculum_steps=10)
    model = ToyModel()
    inputs = batch()
    del inputs["example_indices"]
    expected = trainer.sft_method.compute_loss(model, inputs, stage=stage)
    actual = trainer.compute_loss(model, inputs, stage=stage)
    torch.testing.assert_close(actual.loss, expected.loss)
    assert actual.metrics.keys() == expected.metrics.keys()
    assert LOSS_COMPONENTS_KEY not in actual.extra


def test_existing_gradient_calibration_accepts_distill_component():
    model = ToyModel()
    trainer = method(estimator="exact", catalog_chunk_size=1)
    result = calibrate_component_weight(
        model, trainer, [batch(), batch()], component="distill", target_ratio=.1, num_batches=2,
    )
    trainer.distill_weight = result["weight"]
    assert trainer.distill_weight > 0
    assert trainer.cfg.to_dict()["distill_weight"] == result["weight"]
    assert result["effective_ratio_check"] == pytest.approx(.1)


@pytest.mark.parametrize("kwargs", [
    {"estimator": "candidates"}, {"num_samples": 0}, {"catalog_chunk_size": 0},
    {"distill_weight": -1}, {"distill_weight": float("nan")},
    {"sampling": "gflow"}, {"curriculum_steps": -1},
    {"estimator": "exact", "sampling": "curriculum"},
    {"estimator": "exact", "sampling": "curriculum_smooth"},
    {"estimator": "exact", "sampling": "curriculum_smooth_is"},
])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        method(**kwargs)


def test_invalid_teacher_rows_fail_instead_of_silent_renormalization():
    for rewards in (torch.zeros_like(REWARDS), -REWARDS, REWARDS * float("nan")):
        with pytest.raises(ValueError, match="reward"):
            method(Cache(rewards)).compute_loss(ToyModel(), batch())


def test_tiny_tiger_checkpointed_exact_backward_and_encoder_reuse():
    from flowrec.models.tiger import TigerModel

    model = TigerModel(
        codebook_size=3, sem_id_len=2, user_ids_count=2,
        semantic_id_mapping={str(i): sid.tolist() for i, sid in enumerate(CATALOG)},
        embedding_dim=8, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, d_kv=4, dropout=0.,
    )
    inputs = batch()
    inputs["history_token_mask"] = torch.ones_like(inputs["history_token_ids"], dtype=torch.bool)
    encoder_calls = []
    handle = model.model.encoder.register_forward_hook(lambda *args: encoder_calls.append(1))
    trainer = method(estimator="exact", catalog_chunk_size=1)
    result = trainer.compute_loss(model, inputs)
    cached = {**inputs,
              "_tiger_encoder_last_hidden_state": result.model_output.extra["encoder_last_hidden_state"],
              "_tiger_encoder_attention_mask": result.model_output.extra["encoder_attention_mask"]}
    all_tokens = CATALOG.unsqueeze(0).expand(2, -1, -1)
    full_token_lp = model.score_item_tokens_valid_actions(
        cached, candidate_token_ids=all_tokens,
        candidate_token_mask=torch.ones_like(all_tokens, dtype=torch.bool),
    )
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    full_ce = -(q * full_token_lp.sum(-1)).sum(-1).mean()
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], full_ce)
    encoder_grads = torch.autograd.grad(
        result.extra[LOSS_COMPONENTS_KEY]["distill"], tuple(model.model.encoder.parameters()),
        retain_graph=True,
    )
    reference_encoder_grads = torch.autograd.grad(
        full_ce, tuple(model.model.encoder.parameters()), retain_graph=True,
    )
    for actual, expected in zip(encoder_grads, reference_encoder_grads):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    assert any(grad.abs().sum() > 0 for grad in encoder_grads)
    result.loss.backward()
    handle.remove()
    assert len(encoder_calls) == 1
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("chunk_size", [1, 2, 20])
def test_exact_prefix_mass_aggregation_shared_edges_and_padding(chunk_size):
    catalog = torch.tensor([[1, 3, 0], [1, 3, 1], [1, 4, 0], [2, 3, 0],
                            [2, 4, 1], [2, 4, 2], [2, 4, 3]])
    rewards = torch.tensor([[1., 2., 0., 4., 5., 6., 7.], [3., 1., 9., 2., 0., 8., 1.]])
    model = ToyModel(catalog)
    inputs = {**batch(), "target_token_ids": catalog[[2, 6]],
              "target_token_mask": torch.ones(2, 3, dtype=torch.bool)}
    trainer = RewardDistillationMethod(
        Cache(rewards), catalog,
        SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False)),
        estimator="exact", catalog_chunk_size=chunk_size,
    )
    expected, sft, _ = reference(model, inputs, rewards=rewards, catalog=catalog)
    expected_grads = torch.autograd.grad(expected + sft, tuple(model.parameters()))
    result = trainer.compute_loss(model, inputs)
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], expected)
    result.loss.backward()
    for parameter, gradient in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(parameter.grad, gradient)
    assert result.metrics["distill/exact_branching_prefixes"] == 5
    assert model.encode_calls == 1


def test_reward_curriculum_halves_preserve_q_ratios_and_switch_at_boundary():
    trainer = method(sampling="curriculum", curriculum_steps=10)
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    expected_low = torch.tensor([[1 / 3, 0., 2 / 3], [.8, .2, 0.]])
    expected_high = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    for step in (None, 0, 5, 9):
        actual, phase = trainer.sampling_distribution(q, step=step)
        assert phase == trainer.curriculum_phase(step) == "low"
        torch.testing.assert_close(actual, expected_low)
    for step in (10, 11, 100):
        actual, phase = trainer.sampling_distribution(q, step=step)
        assert phase == trainer.curriculum_phase(step) == "high"
        torch.testing.assert_close(actual, expected_high)
    for step, count, mass in ((0, 2., .5), (10, 1., .5)):
        result = trainer.compute_loss(ToyModel(), batch(), step=step)
        assert result.metrics["distill/curriculum_phase_high"] == float(step >= 10)
        assert result.metrics["distill/curriculum_high_probability"] == float(step >= 10)
        assert result.metrics["distill/sampling_support_count"] == count
        assert result.metrics["distill/selected_teacher_mass"] == pytest.approx(mass)
        assert not any("importance" in key or "beta" in key for key in result.metrics)


@pytest.mark.parametrize("step", [0, 5, 9, 10, 11])
@pytest.mark.parametrize("sampling", ["curriculum", "curriculum_smooth"])
def test_curriculum_conditional_ce_sampling_expectation_and_gradient(monkeypatch, step, sampling):
    """Enumerate every possible single draw; no Monte Carlo tolerance needed."""
    trainer = method(sampling=sampling, curriculum_steps=10, num_samples=1)
    model = ToyModel()
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    proposal, _ = trainer.sampling_distribution(q, step=step)
    _, _, full_log_probs = reference(model, batch())
    expected_loss = -(proposal * full_log_probs).sum(-1).mean()
    expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    total = expected_loss.new_zeros((), dtype=torch.float64)
    # Draw pairs independently for the two contexts; summing over all 3x3
    # outcomes checks conditional-q CE, with no importance correction.
    for left in range(len(CATALOG)):
        for right in range(len(CATALOG)):
            def draw(actual_proposal, count, replacement, left=left, right=right):
                torch.testing.assert_close(actual_proposal, proposal)
                assert count == 1 and replacement
                return torch.tensor([[left], [right]])
            monkeypatch.setattr(torch, "multinomial", draw)
            loss = trainer.compute_loss(model, batch(), step=step).extra[LOSS_COMPONENTS_KEY]["distill"]
            total = total + proposal[0, left] * proposal[1, right] * loss
    torch.testing.assert_close(total.float(), expected_loss)
    actual_grads = torch.autograd.grad(total, tuple(model.parameters()))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_curriculum_ties_follow_catalog_order_and_halves_are_disjoint():
    trainer = method(sampling="curriculum", curriculum_steps=10)
    q = torch.full((2, 4), .25)
    low, _ = trainer.sampling_distribution(q, step=0)
    high, _ = trainer.sampling_distribution(q, step=10)
    torch.testing.assert_close(high, torch.tensor([[.5, .5, 0., 0.]]).expand(2, -1))
    torch.testing.assert_close(low, torch.tensor([[0., 0., .5, .5]]).expand(2, -1))
    assert not bool((low.gt(0) & high.gt(0)).any())


@pytest.mark.parametrize("sampling", ["curriculum", "curriculum_smooth"])
def test_curriculum_single_item_and_zero_warmup(sampling):
    trainer = method(sampling=sampling, curriculum_steps=10)
    q = torch.ones(2, 1)
    for step in (0, 10):
        actual, phase = trainer.sampling_distribution(q, step=step)
        assert actual is q and phase == ("low" if step == 0 else "high")
    actual, phase = trainer.sampling_distribution(q, step=5)
    assert actual is q and phase == ("mixed" if sampling == "curriculum_smooth" else "low")
    trainer = method(sampling=sampling, curriculum_steps=0)
    assert trainer.curriculum_phase(0) == trainer.curriculum_phase(None) == "high"
    assert trainer.curriculum_high_probability(0) == 1.
    trainer = method(sampling="reward", curriculum_steps=10)
    actual, phase = trainer.sampling_distribution(q, step=0)
    assert phase == "all" and actual is q


def test_curriculum_samples_only_selected_half_without_reweighting():
    torch.manual_seed(193)
    trainer = method(sampling="curriculum", curriculum_steps=10, num_samples=1000)
    model = ToyModel()
    _, _, log_probs = reference(model, batch())
    for step in (0, 10):
        result = trainer.compute_loss(model, batch(), step=step)
        rows = result.extra["distill_sampled_item_rows"]
        high_rows = torch.tensor([[1], [2]])
        if step < 10:
            assert not bool(rows.eq(high_rows).any())
        else:
            assert bool(rows.eq(high_rows).all())
        expected = -log_probs.gather(1, rows).mean()
        torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], expected)


@pytest.mark.parametrize("step,alpha,phase", [
    (None, 0., "low"), (-2, 0., "low"), (0, 0., "low"), (2, .2, "mixed"),
    (5, .5, "mixed"), (9, .9, "mixed"), (10, 1., "high"), (20, 1., "high"),
])
def test_smooth_curriculum_mixes_normalized_halves_with_scheduled_mass(step, alpha, phase):
    trainer = method(sampling="curriculum_smooth", curriculum_steps=10)
    # The high half has 90% of the original teacher mass, and the odd-sized
    # low half has three items. Neither affects the scheduled mixture weight.
    q = torch.tensor([[.7, .2, .05, .03, .02]])
    low = torch.tensor([[0., 0., .5, .3, .2]])
    high = torch.tensor([[7 / 9, 2 / 9, 0., 0., 0.]])
    actual, actual_phase = trainer.sampling_distribution(q, step=step)
    assert actual_phase == trainer.curriculum_phase(step) == phase
    assert trainer.curriculum_high_probability(step) == alpha
    torch.testing.assert_close(actual, (1 - alpha) * low + alpha * high)
    torch.testing.assert_close(actual[:, :2].sum(), torch.tensor(alpha))
    torch.testing.assert_close(actual.sum(-1), torch.ones(1))
    if step in (0, 10):
        old = method(sampling="curriculum", curriculum_steps=10)
        old_probabilities, old_phase = old.sampling_distribution(q, step=step)
        torch.testing.assert_close(actual, old_probabilities, rtol=0, atol=0)
        assert actual_phase == old_phase


def test_smooth_curriculum_stable_ties_and_odd_half_sizes():
    trainer = method(sampling="curriculum_smooth", curriculum_steps=10)
    q = torch.full((2, 5), .2)
    actual, phase = trainer.sampling_distribution(q, step=5)
    assert phase == "mixed"
    torch.testing.assert_close(
        actual, torch.tensor([[.25, .25, 1 / 6, 1 / 6, 1 / 6]]).expand(2, -1),
    )


def test_smooth_curriculum_checks_only_active_half_mass_at_endpoints():
    trainer = method(sampling="curriculum_smooth", curriculum_steps=10)
    q = torch.tensor([[1., 0., 0.]])
    actual, phase = trainer.sampling_distribution(q, step=10)
    assert phase == "high"
    torch.testing.assert_close(actual, q)
    for step in (0, 5):
        with pytest.raises(ValueError, match="active curriculum half"):
            trainer.sampling_distribution(q, step=step)


def test_smooth_curriculum_reports_mixture_mass_and_uses_unweighted_ce():
    torch.manual_seed(271)
    trainer = method(sampling="curriculum_smooth", curriculum_steps=10, num_samples=1000)
    model = ToyModel()
    result = trainer.compute_loss(model, batch(), step=4)
    _, _, log_probs = reference(model, batch())
    rows = result.extra["distill_sampled_item_rows"]
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], -log_probs.gather(1, rows).mean())
    assert result.metrics["distill/curriculum_high_probability"] == .4
    assert result.metrics["distill/curriculum_phase_high"] == 0.
    assert result.metrics["distill/selected_teacher_mass"] == 1.
    assert result.metrics["distill/sampling_support_count"] == 3.
    assert not any("importance" in key or "beta" in key for key in result.metrics)


@pytest.mark.parametrize("epsilon", [0., -.1, 1.01, float("nan"), float("inf"), -float("inf")])
def test_importance_curriculum_rejects_invalid_q_mix(epsilon):
    with pytest.raises(ValueError, match="proposal_q_mix"):
        method(sampling="curriculum_smooth_is", proposal_q_mix=epsilon)


@pytest.mark.parametrize("epsilon", [1e-50, 1e-40])
def test_importance_curriculum_rejects_epsilon_with_unrepresentable_weights(epsilon):
    # At alpha=0, q=[.1,.2,.7] would give zero high-half proposal for 1e-50;
    # 1e-40 retains that support on CPU but overflows q/proposal to infinity.
    with pytest.raises(ValueError, match="representable in float32"):
        method(sampling="curriculum_smooth_is", proposal_q_mix=epsilon)


def test_importance_curriculum_rejects_underflow_in_small_teacher_mass():
    # 1/epsilon fits float32, yet epsilon*q can still underflow for a small
    # positive teacher probability outside the active curriculum half.
    trainer = method(sampling="curriculum_smooth_is", curriculum_steps=10, proposal_q_mix=1e-30)
    q = torch.tensor([[1e-20, .4, .6]])
    with pytest.raises(ValueError, match="preserve full q support"):
        trainer.sampling_distribution(q, step=10)


def test_importance_curriculum_rejects_nonfinite_sampled_ratios(monkeypatch):
    trainer = method(sampling="curriculum_smooth_is", num_samples=1)
    # Defensive check even if a future/custom proposal bypasses the public
    # support validation. Never silently clip an invalid ratio.
    monkeypatch.setattr(trainer, "sampling_distribution", lambda q, step: (torch.zeros_like(q), "low"))
    monkeypatch.setattr(torch, "multinomial", lambda q, count, replacement: torch.zeros(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="Non-finite importance weights"):
        trainer.compute_loss(ToyModel(), batch(), step=0)


def test_importance_config_is_explicit_without_changing_legacy_serialized_configs():
    from flowrec.ft_methods.reward_distillation import RewardDistillationConfig

    keys = {"name", "estimator", "num_samples", "catalog_chunk_size", "distill_weight",
            "normalization", "sampling", "curriculum_steps", "exact_algorithm"}
    for sampling in ("reward", "curriculum", "curriculum_smooth"):
        cfg = method(sampling=sampling).cfg
        assert set(cfg.to_dict()) == keys
        assert RewardDistillationConfig(**cfg.to_dict()) == cfg
    cfg = method(sampling="curriculum_smooth_is").cfg
    assert set(cfg.to_dict()) == keys | {"proposal_q_mix"}
    assert cfg.to_dict()["proposal_q_mix"] == .1
    assert RewardDistillationConfig(**cfg.to_dict()) == cfg


@pytest.mark.parametrize("step,alpha,phase", [
    (None, 0., "low"), (-2, 0., "low"), (0, 0., "low"), (2, .2, "mixed"),
    (5, .5, "mixed"), (9, .9, "mixed"), (10, 1., "high"), (20, 1., "high"),
])
@pytest.mark.parametrize("epsilon", [.1, .7])
def test_importance_curriculum_has_full_support_and_exact_proposal(step, alpha, phase, epsilon):
    trainer = method(sampling="curriculum_smooth_is", curriculum_steps=10, proposal_q_mix=epsilon)
    # Odd catalog size, asymmetric teacher masses, and per-half q ratios.
    q = torch.tensor([[.7, .2, .05, .03, .02]], requires_grad=True)
    low = torch.tensor([[0., 0., .5, .3, .2]])
    high = torch.tensor([[7 / 9, 2 / 9, 0., 0., 0.]])
    proposal, actual_phase = trainer.sampling_distribution(q, step=step)
    expected = (1 - epsilon) * ((1 - alpha) * low + alpha * high) + epsilon * q
    torch.testing.assert_close(proposal, expected)
    assert proposal.requires_grad is False
    torch.testing.assert_close(proposal.sum(-1), torch.ones(1))
    assert actual_phase == trainer.curriculum_phase(step) == phase
    assert trainer.curriculum_high_probability(step) == alpha
    assert bool(proposal.gt(0).all())  # In particular, both endpoints.
    assert bool((q / proposal <= 1 / epsilon + 1e-5).all())
    # The actual high-half probability includes its q contribution.
    torch.testing.assert_close(proposal[:, :2].sum(), torch.tensor((1 - epsilon) * alpha + epsilon * .9))


def test_importance_curriculum_stable_ties_singleton_and_zero_duration():
    trainer = method(sampling="curriculum_smooth_is", curriculum_steps=10, proposal_q_mix=.2)
    q = torch.full((2, 5), .2)
    low, _ = trainer.sampling_distribution(q, step=0)
    high, _ = trainer.sampling_distribution(q, step=10)
    torch.testing.assert_close(low, (.8 * torch.tensor([[0., 0., 1 / 3, 1 / 3, 1 / 3]]) + .2 * q))
    torch.testing.assert_close(high, (.8 * torch.tensor([[.5, .5, 0., 0., 0.]]) + .2 * q))
    for step in (0, 5, 10):
        proposal, _ = trainer.sampling_distribution(torch.ones(2, 1), step=step)
        torch.testing.assert_close(proposal, torch.ones(2, 1))
    immediate = method(sampling="curriculum_smooth_is", curriculum_steps=0, proposal_q_mix=.2)
    for step in (None, 0, 100):
        proposal, phase = immediate.sampling_distribution(q, step=step)
        torch.testing.assert_close(proposal, high)
        assert phase == "high" and immediate.curriculum_high_probability(step) == 1.


@pytest.mark.parametrize("step", [None, 0, 5, 10, 100])
def test_importance_q_mix_one_bypasses_empty_conditional_and_detaches_teacher(step):
    trainer = method(sampling="curriculum_smooth_is", curriculum_steps=10, proposal_q_mix=1.)
    q = torch.tensor([[1., 0., 0.]], requires_grad=True)
    proposal, _ = trainer.sampling_distribution(q, step=step)
    torch.testing.assert_close(proposal, q)
    assert proposal.requires_grad is False


@pytest.mark.parametrize("step", [0, 3, 10, 20])
@pytest.mark.parametrize("epsilon", [.1, .4, 1.])
def test_importance_curriculum_expected_loss_and_all_gradients_equal_full_ce(monkeypatch, step, epsilon):
    """Enumerate all K=1 outcomes, including both former zero-support phases."""
    trainer = method(sampling="curriculum_smooth_is", curriculum_steps=10,
                     proposal_q_mix=epsilon, num_samples=1)
    model = ToyModel()
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    proposal, _ = trainer.sampling_distribution(q, step=step)
    expected_loss, _, _ = reference(model, batch())
    expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    total = expected_loss.new_zeros((), dtype=torch.float64)
    for left in range(len(CATALOG)):
        for right in range(len(CATALOG)):
            def draw(actual_proposal, count, replacement, left=left, right=right):
                torch.testing.assert_close(actual_proposal, proposal)
                assert count == 1 and replacement and not actual_proposal.requires_grad
                return torch.tensor([[left], [right]])
            monkeypatch.setattr(torch, "multinomial", draw)
            loss = trainer.compute_loss(model, batch(), step=step).extra[LOSS_COMPONENTS_KEY]["distill"]
            total = total + proposal[0, left] * proposal[1, right] * loss
    torch.testing.assert_close(total.float(), expected_loss)
    actual_grads = torch.autograd.grad(total, tuple(model.parameters()))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_importance_weights_are_raw_not_clipped_or_sample_normalized(monkeypatch):
    rewards = REWARDS.clone().requires_grad_()
    trainer = method(Cache(rewards), sampling="curriculum_smooth_is", curriculum_steps=10,
                     proposal_q_mix=.1, num_samples=3)
    q = REWARDS / REWARDS.sum(-1, keepdim=True)
    proposal, _ = trainer.sampling_distribution(q, step=0)
    # The high halves remain reachable at alpha=0 only via epsilon*q and
    # receive raw weight 10. Include repeated draws, and a low-half item.
    rows = torch.tensor([[1, 1, 0], [2, 2, 0]])
    def draw(actual_proposal, count, replacement):
        torch.testing.assert_close(actual_proposal, proposal)
        assert count == 3 and replacement and not actual_proposal.requires_grad
        return rows
    monkeypatch.setattr(torch, "multinomial", draw)
    model = ToyModel()
    result = trainer.compute_loss(model, batch(), step=0)
    _, _, log_probs = reference(model, batch())
    ratios = q.gather(1, rows) / proposal.gather(1, rows)
    torch.testing.assert_close(ratios[:, :2], torch.full((2, 2), 10.))
    expected = -(ratios * log_probs.gather(1, rows)).mean()
    actual = result.extra[LOSS_COMPONENTS_KEY]["distill"]
    torch.testing.assert_close(actual, expected)
    self_normalized = -((ratios / ratios.sum(-1, keepdim=True)) * log_probs.gather(1, rows)).sum(-1).mean()
    assert not torch.isclose(actual, self_normalized)
    metrics = result.metrics
    torch.testing.assert_close(metrics["distill/importance_weight_mean"], ratios.mean())
    torch.testing.assert_close(metrics["distill/importance_weight_max"], ratios.max())
    torch.testing.assert_close(metrics["distill/importance_ess"],
        (ratios.sum(-1).square() / ratios.square().sum(-1)).mean())
    assert metrics["distill/proposal_q_mix"] == .1
    assert metrics["distill/selected_teacher_mass"] == pytest.approx(1.)
    assert metrics["distill/sampling_support_count"] == 3.
    result.loss.backward()
    assert rewards.grad is None
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def test_importance_q_mix_one_matches_direct_reward_sampling_loss_and_gradient():
    direct = method(num_samples=7)
    corrected = method(sampling="curriculum_smooth_is", curriculum_steps=10,
                       proposal_q_mix=1., num_samples=7)
    model = ToyModel()
    torch.manual_seed(91)
    expected = direct.compute_loss(model, batch())
    expected_grads = torch.autograd.grad(expected.loss, tuple(model.parameters()))
    torch.manual_seed(91)
    actual = corrected.compute_loss(model, batch(), step=0)
    torch.testing.assert_close(actual.extra["distill_sampled_item_rows"], expected.extra["distill_sampled_item_rows"])
    torch.testing.assert_close(actual.loss, expected.loss)
    actual_grads = torch.autograd.grad(actual.loss, tuple(model.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert actual.metrics["distill/importance_weight_mean"] == 1.
    assert actual.metrics["distill/importance_weight_max"] == 1.
    assert actual.metrics["distill/importance_ess"] == 7.

