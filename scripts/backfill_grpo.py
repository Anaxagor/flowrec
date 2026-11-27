"""Fine-tune GRPO on selected reward configurations for leave-one-out and global splits."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rl_backfill import Cell, main


REWARDS = ("S", "H", "I", "S×H×I")
Y50 = "yambda_50m_listens"
Y500 = "yambda_500m_listens"
BEAUTY = "beauty_2023"

# A run counts toward reward coverage only when its reward function is known.
MISSING = {
    ("loo", "tiger", BEAUTY): ("H", "I"),
    ("loo", "gpt2", BEAUTY): ("S", "H", "I"),
    ("loo", "gpt2", Y500): REWARDS,
    ("global", "tiger", Y50): ("S", "S×H×I"),
    ("global", "gpt2", Y50): ("S×H×I",),
    ("global", "tiger", Y500): REWARDS,
    ("global", "gpt2", Y500): REWARDS,
    ("global", "tiger", BEAUTY): REWARDS,
    ("global", "gpt2", BEAUTY): REWARDS,
}


def cells(all_rewards=False):
    return [
        Cell(split, model, dataset, reward)
        for (split, model, dataset), missing in MISSING.items()
        for reward in (REWARDS if all_rewards else missing)
    ]


if __name__ == "__main__":
    main("grpo", cells)
