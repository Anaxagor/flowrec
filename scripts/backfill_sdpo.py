"""Fine-tune multi-negative softmax DPO on the four missing Base LOO and Global cells."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rl_backfill import Cell, main


def cells():
    return [
        Cell("loo", "gpt2", "yambda_50m_listens"),
        Cell("loo", "gpt2", "yambda_500m_listens"),
        Cell("global", "tiger", "beauty_2023"),
        Cell("global", "gpt2", "yambda_500m_listens"),
    ]


if __name__ == "__main__":
    main("sdpo", cells)
