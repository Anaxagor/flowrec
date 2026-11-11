"""Fine-tuning methods for FlowRec."""

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput
from flowrec.ft_methods.calibration import (
    calibrate_component_weight,
    calibrate_from_loss_pairs,
    gradient_vector,
    loss_components,
    variant_loss_pairs,
)


# Import built-in methods so they register themselves.
try:
    from flowrec.ft_methods.sft import SFTConfig, SFTMethod
except Exception:  # pragma: no cover - optional dependency friendliness
    SFTConfig = None  # type: ignore[assignment]
    SFTMethod = None  # type: ignore[assignment]

try:
    from flowrec.ft_methods.gflow import GFlowConfig, GFlowMethod
except Exception:  # pragma: no cover - optional dependency friendliness
    GFlowConfig = None  # type: ignore[assignment]
    GFlowMethod = None  # type: ignore[assignment]

try:
    from flowrec.ft_methods.grpo import GRPOConfig, GRPOMethod
except Exception:  # pragma: no cover - optional dependency friendliness
    GRPOConfig = None  # type: ignore[assignment]
    GRPOMethod = None  # type: ignore[assignment]

__all__ = [
    "LOSS_COMPONENTS_KEY",
    "FlowRecMethod",
    "calibrate_component_weight",
    "calibrate_from_loss_pairs",
    "gradient_vector",
    "loss_components",
    "variant_loss_pairs",
    "GFlowConfig",
    "GFlowMethod",
    "GRPOConfig",
    "GRPOMethod",
    "MethodOutput",
    "METHOD_REGISTRY",
    "MethodRegistry",
    "SFTConfig",
    "SFTMethod",
]
