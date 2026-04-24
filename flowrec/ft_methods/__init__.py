"""Fine-tuning methods for FlowRec."""

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput


# Import built-in methods so they register themselves.
try:
    from flowrec.ft_methods.sft import SFTConfig, SFTMethod
except Exception:  # pragma: no cover - optional dependency friendliness
    SFTConfig = None  # type: ignore[assignment]
    SFTMethod = None  # type: ignore[assignment]

__all__ = [
    "FlowRecMethod",
    "MethodOutput",
    "METHOD_REGISTRY",
    "MethodRegistry",
    "SFTConfig",
    "SFTMethod",
]
