"""Preference-based fine-tuning methods."""

from flowrec.ft_methods.preference.config import DPOConfig
from flowrec.ft_methods.preference.method import DPOMethod

__all__ = [
    "DPOConfig",
    "DPOMethod",
]
