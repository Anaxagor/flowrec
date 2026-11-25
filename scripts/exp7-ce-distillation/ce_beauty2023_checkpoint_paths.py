"""Locate the promoted Beauty 2023 leave-one-out source weights for exp7."""
from pathlib import Path


def initial_checkpoint(root, family):
    """Use the canonical checkpoint, falling back to its original exp2 location."""
    original_filename = {"tiger": "best.pt", "gpt2": "best_model.pt", "sasrec": "best_model.pt"}[family]
    checkpoints = Path(root) / "checkpoints"
    canonical = checkpoints / "exp2/loo_split/initial" / family / "beauty_2023/best.pt"
    if canonical.is_file():
        return canonical
    original = checkpoints / "exp2-tuned_params/initial" / family / "beauty_2023" / original_filename
    return original if original.is_file() else canonical
