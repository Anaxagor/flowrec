"""Synthetic VK-LSVD-shaped dataset for CPU tests of the datamodule and the
exp1 runners.

``build_smoke_subsample`` writes a subsample folder with the upstream layout
and schema (weekly Parquet files under ``train`` / ``validation`` / ``test``,
uint32 ids, uint8 ``timespent``, boolean feedback flags, no timestamp column)
plus an ``item_embeddings.npz`` with float16 64-d vectors.
``build_smoke_config_dir`` derives an exp1 config directory from the real
``configs/exp1-standard_params`` YAMLs: one ``vk_lsvd_smoke`` dataset, tiny
batch sizes, no DataLoader workers, and output folders under ``root``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP1_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"

SMOKE_DATASET = "vk_lsvd_smoke"
SMOKE_BATCH_SIZE = 16
TRAIN_WEEKS = (0, 1, 2)
VALIDATION_WEEK = 25
TEST_WEEK = 26
EMBEDDING_DIM = 64


def _week_frame(
    rng: np.random.Generator,
    *,
    n_users: int,
    n_items: int,
    events_per_user: tuple[int, int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for user in range(n_users):
        n = int(rng.integers(*events_per_user))
        times = np.sort(rng.random(n))
        for t in times:
            rows.append(
                {
                    "__t": float(t),
                    "user_id": user,
                    "item_id": int(rng.zipf(1.4) % n_items),
                    "place": int(rng.integers(0, 3)),
                    "platform": int(rng.integers(0, 4)),
                    "agent": int(rng.integers(0, 2)),
                    "timespent": int(rng.integers(0, 60)),
                    "like": bool(rng.random() < 0.10),
                    "dislike": bool(rng.random() < 0.03),
                    "share": bool(rng.random() < 0.02),
                    "bookmark": bool(rng.random() < 0.02),
                    "click_on_author": bool(rng.random() < 0.05),
                    "open_comments": bool(rng.random() < 0.05),
                }
            )
    # Rows within a weekly file are in increasing time order, so users
    # interleave exactly like the upstream files.
    frame = pd.DataFrame(rows).sort_values("__t", kind="stable").drop(columns="__t")
    return frame.astype(
        {
            "user_id": "uint32",
            "item_id": "uint32",
            "place": "uint8",
            "platform": "uint8",
            "agent": "uint8",
            "timespent": "uint8",
        }
    ).reset_index(drop=True)


def build_smoke_subsample(
    root: Path,
    *,
    n_users: int = 300,
    n_items: int = 500,
    events_per_user: tuple[int, int] = (3, 12),
    seed: int = 0,
    embedding_dim: int = EMBEDDING_DIM,
) -> tuple[Path, Path]:
    """Write ``<root>/subsamples/smoke/{train,validation,test}/week_XX.parquet``
    and ``<root>/item_embeddings.npz``; return (subsample_root, npz_path)."""
    rng = np.random.default_rng(seed)
    subsample = root / "subsamples" / "smoke"
    for split, weeks in (
        ("train", TRAIN_WEEKS),
        ("validation", (VALIDATION_WEEK,)),
        ("test", (TEST_WEEK,)),
    ):
        directory = subsample / split
        directory.mkdir(parents=True, exist_ok=True)
        for week in weeks:
            _week_frame(
                rng, n_users=n_users, n_items=n_items, events_per_user=events_per_user
            ).to_parquet(directory / f"week_{week:02}.parquet", index=False)

    npz_path = root / "item_embeddings.npz"
    np.savez(
        npz_path,
        item_id=np.arange(n_items, dtype=np.uint32),
        embedding=rng.standard_normal((n_items, embedding_dim)).astype(np.float16),
    )
    return subsample, npz_path


def build_smoke_config_dir(
    root: Path,
    *,
    n_users: int = 300,
    n_items: int = 500,
    seed: int = 0,
) -> Path:
    """Synthetic subsample + an exp1 config directory for ``vk_lsvd_smoke``."""
    subsample, npz_path = build_smoke_subsample(root, n_users=n_users, n_items=n_items, seed=seed)
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)

    dataset = yaml.safe_load((EXP1_CONFIG_DIR / "dataset_vk_lsvd.yaml").read_text())
    dataset["datasets"] = {SMOKE_DATASET: {"interactions_root": str(subsample)}}
    dataset["embeddings_path"] = str(npz_path)
    dataset["min_user_interactions"] = 2
    dataset["min_item_interactions"] = 1
    dataset["max_history_len"] = 10
    dataset["batch_size"] = SMOKE_BATCH_SIZE
    dataset["eval_batch_size"] = SMOKE_BATCH_SIZE
    dataset["num_workers"] = 0
    dataset["pin_memory"] = False
    dataset["persistent_workers"] = False
    for block in ("tiger", "gpt2"):
        dataset[block].update(
            {
                "batch_size": SMOKE_BATCH_SIZE,
                "eval_batch_size": 8,
                "test_eval_batch_size": 8,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            }
        )
    (config_dir / "dataset_vk_lsvd.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )

    outputs = root / "outputs"
    for name in ("trainer_sasrec_vk_lsvd.yaml", "trainer_tiger.yaml", "trainer_gpt2.yaml"):
        trainer = yaml.safe_load((EXP1_CONFIG_DIR / name).read_text())
        stem = name.removesuffix(".yaml")
        trainer["checkpoints_dir"] = str(outputs / stem / "checkpoints")
        trainer["experiments_dir"] = str(outputs / stem / "experiments")
        trainer["artifacts_dir"] = str(outputs / "artifacts")
        trainer["device"] = "cpu"
        trainer["use_amp"] = False
        trainer["progress_bar"] = False
        trainer["seeds"] = [42]
        if "variants" in trainer:
            for variant in trainer["variants"].values():
                variant["checkpoints_dir"] = str(outputs / stem / "variant_checkpoints")
                variant["experiments_dir"] = str(outputs / stem / "variant_experiments")
                variant["artifacts_dir"] = str(outputs / "variant_artifacts")
        (config_dir / name).write_text(yaml.safe_dump(trainer, sort_keys=False), encoding="utf-8")

    gpt2 = yaml.safe_load((EXP1_CONFIG_DIR / "gpt2_model.yaml").read_text())
    gpt2["datasets"] = {
        SMOKE_DATASET: {
            "model": {
                "embedding_dim": 32,
                "codebook_size": 256,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
            },
            "optimizer": {"lr": 1.0e-3},
        }
    }
    (config_dir / "gpt2_model.yaml").write_text(yaml.safe_dump(gpt2, sort_keys=False), encoding="utf-8")

    tiger = yaml.safe_load((EXP1_CONFIG_DIR / "tiger_model.yaml").read_text())
    tiger.update(
        {
            "embedding_dim": 32,
            "num_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "d_kv": 16,
        }
    )
    (config_dir / "tiger_model.yaml").write_text(yaml.safe_dump(tiger, sort_keys=False), encoding="utf-8")

    # The runners enumerate every family's dataset YAML to resolve names
    # (inputs are only validated for the selected datasets).
    for name in (
        "sasrec_model.yaml",
        "semantic_ids.yaml",
        "dataset_amazon.yaml",
        "dataset_amazon_2023.yaml",
        "dataset_yambda.yaml",
    ):
        shutil.copy(EXP1_CONFIG_DIR / name, config_dir / name)
    return config_dir
