"""Synthetic Yambda-shaped dataset for CPU smoke tests of the SASRec scripts.

Builds a small ``multi_event.parquet`` + ``embeddings.parquet`` pair and a
config directory whose ``dataset_yambda.yaml`` (the name the exp1/exp2
dataset groups look up) defines a single ``yambda_smoke`` dataset with tiny
batch sizes and no DataLoader workers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMOKE_DATASET = "yambda_smoke"
SMOKE_BATCH_SIZE = 32


def build_smoke_config_dir(root: Path, *, n_users: int = 400, n_items: int = 300) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(n_users):
        n = int(rng.integers(12, 60))
        t0 = 1_700_000_000 + int(rng.integers(0, 10_000))
        for i in range(n):
            rows.append({
                "uid": user, "item_id": int(rng.zipf(1.3) % n_items), "timestamp": t0 + i * 60,
                "event_type": "listen", "is_organic": int(rng.integers(0, 2)),
                "played_ratio_pct": int(rng.integers(30, 101)), "track_length_seconds": 200,
            })
    interactions = root / "multi_event.parquet"
    pd.DataFrame(rows).to_parquet(interactions)
    embeddings = root / "embeddings.parquet"
    pd.DataFrame({"item_id": list(range(n_items)),
                  "embed": [rng.random(4).tolist() for _ in range(n_items)]}).to_parquet(embeddings)

    cfg = yaml.safe_load((PROJECT_ROOT / "configs/exp2-tuned-params/dataset_yambda.yaml").read_text())
    cfg["datasets"] = {SMOKE_DATASET: {"dataset_size": "50m", "interactions_path": str(interactions)}}
    cfg["embeddings_path"] = str(embeddings)
    cfg["hf_recent_months"] = None
    cfg["batch_size"] = SMOKE_BATCH_SIZE
    cfg["eval_batch_size"] = SMOKE_BATCH_SIZE
    cfg["num_workers"] = 0
    cfg["pin_memory"] = False
    cfg["persistent_workers"] = False
    config_dir = root / "configs"
    config_dir.mkdir()
    # The runner looks the dataset up by the group's fixed YAML name.
    (config_dir / "dataset_yambda.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_dir
