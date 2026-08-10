"""Softmax-DPO / S-DPO fine-tuning method."""

from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.ft_methods.preference_sdpo.method import SDPOMethod

__all__ = [
    "SDPOConfig",
    "SDPOMethod",
]
