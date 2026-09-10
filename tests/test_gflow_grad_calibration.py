"""Gradient-norm calibration of a composite objective's component weight.

``flowrec.ft_methods.calibration`` is shared by GFlow (L_SFT + lambda L_GFlow)
and the RL methods (L_SFT + lambda L_GRPO, DPO + w L_SFT). The acceptance
relation:

    median_b(weight * ||g_comp,b|| / ||g_ref,b||) ~= rho_target
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput
from flowrec.ft_methods.calibration import (
    calibrate_component_weight,
    calibrate_from_loss_pairs,
    gradient_vector,
    loss_components,
    variant_loss_pairs,
)
from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _toy() -> tuple[torch.nn.Linear, list[torch.nn.Parameter], torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(8, 4)
    return model, params, x


class _ToyComposite(FlowRecMethod):
    """L = L_ref + gain * scale * L_ref with both unweighted parts reported.

    ``gain`` is a *method* parameter that only the component depends on, so
    tests can tell whether method parameters were included in the gradient.
    """

    method_name = "toy-composite"

    def __init__(self, scale: float, *, dropout: float = 0.0, report_components: bool = True) -> None:
        super().__init__()
        self.scale = float(scale)
        self.drop = nn.Dropout(dropout)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.report_components = report_components

    def compute_loss(self, model, batch, *, stage="train", step=None) -> MethodOutput:
        h = self.drop(model(batch["x"]))
        ref = h.square().mean()
        comp = self.gain * self.scale * h.square().mean()
        extra = {LOSS_COMPONENTS_KEY: {"sft": ref, "aux": comp}} if self.report_components else {}
        return MethodOutput(loss=ref + comp, extra=extra)


class TestGradientVector:
    def test_matches_manual_grad_and_leaves_grad_buffers_alone(self) -> None:
        model, params, x = _toy()
        loss = model(x).square().mean()
        vec = gradient_vector(loss, params)
        assert vec.numel() == sum(p.numel() for p in params)
        assert all(p.grad is None for p in params)  # autograd.grad, not backward

        loss2 = model(x).square().mean()
        loss2.backward()
        manual = torch.cat([p.grad.reshape(-1) for p in params])
        assert torch.allclose(vec, manual, atol=1e-6)
        for p in params:
            p.grad = None

    def test_unused_params_contribute_zeros(self) -> None:
        model, params, x = _toy()
        extra = torch.nn.Parameter(torch.randn(5))
        loss = model(x).sum()
        vec = gradient_vector(loss, params + [extra])
        assert torch.all(vec[-5:] == 0.0)

    def test_graphless_loss_gives_zeros(self) -> None:
        model, params, _ = _toy()
        vec = gradient_vector(torch.zeros(()), params)
        assert vec.numel() == sum(p.numel() for p in params)
        assert torch.all(vec == 0.0)


class TestCalibrateFromLossPairs:
    def _pairs(self, model, x, scale: float, n: int = 5):
        for _ in range(n):
            base = model(x).square().mean()
            yield base, scale * model(x).square().mean()

    def test_known_scale_recovers_target_ratio(self) -> None:
        model, params, x = _toy()
        # g_GFlow = 10 * g_SFT exactly -> r0 = 10, lambda = rho / 10.
        for rho in (0.1, 1.0):
            result = calibrate_from_loss_pairs(
                self._pairs(model, x, scale=10.0), params, target_ratio=rho
            )
            assert result["raw_grad_ratio_median"] == pytest.approx(10.0, rel=1e-5)
            assert result["fixed_gflow_weight"] == pytest.approx(rho / 10.0, rel=1e-5)
            assert result["weight"] == result["fixed_gflow_weight"]
            # The acceptance relation: effective ratio equals the target.
            effective = [result["weight"] * r for r in result["per_batch"]["ratios"]]
            effective.sort()
            median = effective[len(effective) // 2]
            assert median == pytest.approx(rho, rel=1e-5)
            assert result["effective_ratio_check"] == pytest.approx(rho, rel=1e-5)
            # Identical directions -> cosine 1 (diagnostic only).
            assert result["grad_cosine_median"] == pytest.approx(1.0, abs=1e-5)

    def test_default_keys_match_the_exp4_calibration_json(self) -> None:
        model, params, x = _toy()
        result = calibrate_from_loss_pairs(self._pairs(model, x, scale=2.0), params, target_ratio=1.0)
        assert result["reference"] == "sft" and result["component"] == "gflow"
        for key in ("sft_grad_norm_median", "gflow_grad_norm_median", "fixed_gflow_weight",
                    "raw_grad_ratio_median", "raw_grad_ratio_mean", "raw_grad_ratio_std",
                    "grad_cosine_median", "target_grad_ratio", "effective_ratio_check"):
            assert key in result
        assert set(result["per_batch"]) == {"sft_norms", "gflow_norms", "ratios", "cosines"}
        assert result["num_batches"] == 5

    def test_component_names_drive_the_named_keys(self) -> None:
        model, params, x = _toy()
        result = calibrate_from_loss_pairs(
            self._pairs(model, x, scale=4.0), params, target_ratio=1.0,
            reference_name="sft", component_name="grpo",
        )
        assert result["component"] == "grpo"
        assert result["fixed_grpo_weight"] == pytest.approx(0.25, rel=1e-5)
        assert result["grpo_grad_norm_median"] == pytest.approx(4.0 * result["sft_grad_norm_median"], rel=1e-5)
        assert "gflow_grad_norm_median" not in result
        assert set(result["per_batch"]) == {"sft_norms", "grpo_norms", "ratios", "cosines"}

    def test_pairs_sharing_one_graph_are_supported(self) -> None:
        model, params, x = _toy()

        def pairs():
            for _ in range(3):
                h = model(x)  # one forward feeds both losses
                yield h.square().mean(), 10.0 * h.square().mean()

        result = calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)
        assert result["raw_grad_ratio_median"] == pytest.approx(10.0, rel=1e-5)

    def test_median_is_robust_to_one_outlier_batch(self) -> None:
        model, params, x = _toy()

        def pairs():
            scales = [10.0, 10.0, 1000.0, 10.0, 10.0]
            for s in scales:
                yield model(x).square().mean(), s * model(x).square().mean()

        result = calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)
        assert result["raw_grad_ratio_median"] == pytest.approx(10.0, rel=1e-5)

    def test_zero_reference_gradient_fails_loudly(self) -> None:
        model, params, x = _toy()

        def pairs():
            zero = 0.0 * model(x).sum()
            yield zero, model(x).square().mean()

        with pytest.raises(ValueError, match="sft gradient norm"):
            calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)

    def test_zero_component_gradient_fails_loudly(self) -> None:
        model, params, x = _toy()

        def pairs():
            yield model(x).square().mean(), 0.0 * model(x).sum()

        with pytest.raises(ValueError, match="gflow gradient norm"):
            calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)

    def test_no_batches_fails(self) -> None:
        model, params, _ = _toy()
        with pytest.raises(ValueError, match="no batches"):
            calibrate_from_loss_pairs(iter(()), params, target_ratio=1.0)


class TestCalibrateComponentWeight:
    """The method-agnostic driver: one forward per batch, components from extra."""

    def test_recovers_target_ratio_from_a_single_pass(self) -> None:
        model, params, x = _toy()
        method = _ToyComposite(10.0)
        result = calibrate_component_weight(
            model, method, [{"x": x}] * 5, component="aux", reference="sft",
            target_ratio=0.5, params=params,
        )
        assert result["weight"] == pytest.approx(0.05, rel=1e-5)
        assert result["fixed_aux_weight"] == result["weight"]
        assert result["reference"] == "sft" and result["component"] == "aux"
        assert result["num_batches"] == 5
        assert result["grad_cosine_median"] == pytest.approx(1.0, abs=1e-5)

    def test_default_params_cover_method_parameters_like_the_trainer(self) -> None:
        model, params, x = _toy()
        method = _ToyComposite(10.0)
        model_only = calibrate_component_weight(
            model, method, [{"x": x}] * 3, component="aux", params=params
        )
        assert model_only["raw_grad_ratio_median"] == pytest.approx(10.0, rel=1e-5)
        # ``gain`` only enters the component loss, so including it (the
        # trainer optimizes model + method params) makes the ratio exceed 10.
        with_method = calibrate_component_weight(model, method, [{"x": x}] * 3, component="aux")
        assert with_method["raw_grad_ratio_median"] > 10.0

    def test_restores_modes_rng_and_grad_buffers(self) -> None:
        model, params, x = _toy()
        method = _ToyComposite(3.0, dropout=0.5)
        model.eval()
        method.eval()
        rng_before = torch.random.get_rng_state().clone()
        calibrate_component_weight(model, method, [{"x": x}] * 4, component="aux", seed=7)
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert not model.training and not method.training
        assert all(p.grad is None for p in list(model.parameters()) + list(method.parameters()))

    def test_seed_makes_dropout_sampling_reproducible(self) -> None:
        model, params, x = _toy()
        method = _ToyComposite(3.0, dropout=0.5)
        a = calibrate_component_weight(model, method, [{"x": x}] * 4, component="aux", seed=3)
        b = calibrate_component_weight(model, method, [{"x": x}] * 4, component="aux", seed=3)
        assert a["per_batch"]["sft_norms"] == b["per_batch"]["sft_norms"]
        assert a["per_batch"]["aux_norms"] == b["per_batch"]["aux_norms"]

    def test_missing_component_lists_the_available_ones(self) -> None:
        model, _, x = _toy()
        with pytest.raises(ValueError, match=r"\['gflow'\].*available: \['aux', 'sft'\]"):
            calibrate_component_weight(model, _ToyComposite(1.0), [{"x": x}], component="gflow")

    def test_method_without_components_fails_loudly(self) -> None:
        model, _, x = _toy()
        method = _ToyComposite(1.0, report_components=False)
        with pytest.raises(ValueError, match="does not expose loss components"):
            calibrate_component_weight(model, method, [{"x": x}], component="aux")
        with pytest.raises(ValueError, match="does not expose loss components"):
            loss_components(method.compute_loss(model, {"x": x}))

    def test_component_must_differ_from_reference(self) -> None:
        model, _, x = _toy()
        with pytest.raises(ValueError, match="different"):
            calibrate_component_weight(model, _ToyComposite(1.0), [{"x": x}], component="sft", reference="sft")

    def test_num_batches_caps_and_is_enforced(self) -> None:
        model, _, x = _toy()
        capped = calibrate_component_weight(
            model, _ToyComposite(2.0), [{"x": x}] * 5, component="aux", num_batches=2, device="cpu"
        )
        assert capped["num_batches"] == 2
        with pytest.raises(ValueError, match="yielded only 2"):
            calibrate_component_weight(
                model, _ToyComposite(2.0), [{"x": x}] * 2, component="aux", num_batches=5
            )


class TestVariantLossPairs:
    """The exp4 two-variant protocol: separate isolating method instances."""

    def test_reseeding_aligns_dropout_masks_across_variants(self) -> None:
        model, _, x = _toy()
        ref_variant = _ToyComposite(1.0, dropout=0.5)
        comp_variant = _ToyComposite(1.0, dropout=0.5)  # distinct Dropout module, same RNG
        model.train()
        pairs = list(variant_loss_pairs(
            model, ref_variant, comp_variant, [{"x": x}] * 6, seed=11, num_batches=4
        ))
        assert len(pairs) == 4
        # Same scale, distinct Dropout modules: identical losses iff the
        # re-seeding gave both variants the same mask.
        for loss_ref, loss_comp in pairs:
            assert torch.allclose(loss_ref, loss_comp)
        unseeded = list(variant_loss_pairs(model, ref_variant, comp_variant, [{"x": x}] * 2))
        assert any(not torch.allclose(a, b) for a, b in unseeded)

    def test_fewer_batches_than_requested_fails(self) -> None:
        model, _, x = _toy()
        with pytest.raises(ValueError, match="yielded only 1"):
            list(variant_loss_pairs(model, _ToyComposite(1.0), _ToyComposite(1.0), [{"x": x}], num_batches=3))


class TestMiniOneRecGrpoWeight:
    def test_round_trip_alias_and_validation(self) -> None:
        assert MiniOneRecConfig().grpo_weight == 1.0
        assert MiniOneRecConfig(grpo_weight=0.3).to_dict()["grpo_weight"] == 0.3
        assert MiniOneRecConfig.from_dict({"grpo_loss_weight": 0.3}).grpo_weight == 0.3
        with pytest.raises(ValueError, match="grpo_weight"):
            MiniOneRecConfig(grpo_weight=-1.0)


class TestLegacyImportPath:
    def test_exp4_script_reexports_the_shared_maths(self) -> None:
        script_dir = PROJECT_ROOT / "scripts" / "exp4-reward-ablations"
        for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
                   str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"), str(script_dir)):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        script = pytest.importorskip("gflow_finetune_yambda50m")
        assert script.calibrate_from_loss_pairs is calibrate_from_loss_pairs
        assert script.gradient_vector is gradient_vector
        assert script.variant_loss_pairs is variant_loss_pairs
