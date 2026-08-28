"""Fixed gradient-norm calibration of the GFlow weight (exp4 yambda script).

The acceptance relation:
    median_b(lambda * ||g_GFlow,b|| / ||g_SFT,b||) ~= rho_target
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "exp4-reward-ablations"
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gflow_finetune_yambda50m import (  # noqa: E402
    calibrate_from_loss_pairs,
    gradient_vector,
)


def _toy() -> tuple[torch.nn.Linear, list[torch.nn.Parameter], torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(8, 4)
    return model, params, x


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
            # The acceptance relation: effective ratio equals the target.
            effective = [
                result["fixed_gflow_weight"] * r for r in result["per_batch"]["ratios"]
            ]
            effective.sort()
            median = effective[len(effective) // 2]
            assert median == pytest.approx(rho, rel=1e-5)
            assert result["effective_ratio_check"] == pytest.approx(rho, rel=1e-5)
            # Identical directions -> cosine 1 (diagnostic only).
            assert result["grad_cosine_median"] == pytest.approx(1.0, abs=1e-5)

    def test_median_is_robust_to_one_outlier_batch(self) -> None:
        model, params, x = _toy()

        def pairs():
            scales = [10.0, 10.0, 1000.0, 10.0, 10.0]
            for s in scales:
                yield model(x).square().mean(), s * model(x).square().mean()

        result = calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)
        assert result["raw_grad_ratio_median"] == pytest.approx(10.0, rel=1e-5)

    def test_zero_sft_gradient_fails_loudly(self) -> None:
        model, params, x = _toy()

        def pairs():
            zero = 0.0 * model(x).sum()
            yield zero, model(x).square().mean()

        with pytest.raises(ValueError, match="SFT gradient norm"):
            calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)

    def test_zero_gflow_gradient_fails_loudly(self) -> None:
        model, params, x = _toy()

        def pairs():
            yield model(x).square().mean(), 0.0 * model(x).sum()

        with pytest.raises(ValueError, match="GFlow gradient norm"):
            calibrate_from_loss_pairs(pairs(), params, target_ratio=1.0)

    def test_no_batches_fails(self) -> None:
        model, params, _ = _toy()
        with pytest.raises(ValueError, match="no batches"):
            calibrate_from_loss_pairs(iter(()), params, target_ratio=1.0)
