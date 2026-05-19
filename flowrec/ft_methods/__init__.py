"""Fine-tuning methods for FlowRec."""

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput


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

__all__ = [
    "FlowRecMethod",
    "GFlowConfig",
    "GFlowMethod",
    "MethodOutput",
    "METHOD_REGISTRY",
    "MethodRegistry",
    "SFTConfig",
    "SFTMethod",
]
