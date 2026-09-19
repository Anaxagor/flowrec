"""Train the exp1 RQ-VAE semantic-ID tokenizer on every exp1 dataset.

Covers the two small Amazon datasets (Beauty 2014, Instruments 2018), the
large Amazon Beauty 2023 dataset, the two Yambda datasets (50M and 500M
listens), and the VK-LSVD subsamples. Every dataset uses the same standard tokenizer parameters from
``configs/exp1-standard_params/semantic_ids.yaml``; the dataset YAMLs in the
same folder define the item catalogs.

Inputs per dataset family:

- Amazon: content-embedding stores built by build_amazon_embeddings.py in
  ``datasets/embeddings/exp1-standard_params/<dataset>_content_embeddings.pt``
  (already restricted to the interaction catalog).
- Yambda: precomputed audio embeddings from the ``embeddings_path`` configured
  in ``dataset_yambda.yaml`` (``data/raw/yambda/embeddings.parquet``), filtered
  during loading to the catalog of a fully set-up YambdaDataModule so semantic
  IDs match the SASRec/TIGER training runs exactly.
- VK-LSVD: the upstream content embeddings (``item_embeddings.npz``, 64
  ordered components per video) from the ``embeddings_path`` configured in
  ``dataset_vk_lsvd.yaml``, restricted the same way to the catalog of a fully
  set-up VKLSVDDataModule.

Each dataset's trained tokenizer (model.pt, normalizer.pt, config.json,
training log) and its semantic IDs (semantic_ids.json / semantic_ids.pt) are
saved to ``datasets/semantic_ids/exp1-standard_params/<dataset>/``, and a
combined training report with collision counts is written incrementally to
``artifacts/exp1-standard_params/semantic_id_training_report.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.vk_lsvd import VKLSVDDataModule
from flowrec.data.yambda import YambdaDataModule
from flowrec.item_representation.base import ItemEmbeddingStore
from flowrec.item_representation.semantic_id import (
    RQVAEConfig,
    RQVAEItemTokenizer,
    RQVAEItemTokenizerConfig,
    RQVAETrainConfig,
)
from flowrec.item_representation.vk_lsvd import (
    load_vk_lsvd_item_embedding_store_for_datamodule,
)
from flowrec.item_representation.yambda import (
    load_yambda_audio_embedding_store_for_datamodule,
)
from scripts import sasrec_config_runner as runner


# The exp1 scripts live in a dashed folder, so sibling modules are loaded by
# file path (same pattern as tests/test_sasrec_dataset_runners.py).
_RUN_ALL_SPEC = importlib.util.spec_from_file_location(
    "run_sasrec_all", Path(__file__).with_name("run_sasrec_all.py")
)
assert _RUN_ALL_SPEC is not None and _RUN_ALL_SPEC.loader is not None
run_sasrec_all = importlib.util.module_from_spec(_RUN_ALL_SPEC)
# The dataclasses in the script resolve their module through sys.modules.
sys.modules[_RUN_ALL_SPEC.name] = run_sasrec_all
_RUN_ALL_SPEC.loader.exec_module(run_sasrec_all)


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
DEFAULT_EMBEDDINGS_DIR = (
    PROJECT_ROOT / "datasets" / "embeddings" / "exp1-standard_params"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "datasets" / "semantic_ids" / "exp1-standard_params"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "exp1-standard_params"
    / "semantic_id_training_report.json"
)
SEMANTIC_CONFIG_NAME = "semantic_ids.yaml"
AMAZON_DATA_CONFIG_NAMES = ("dataset_amazon.yaml", "dataset_amazon_2023.yaml")
YAMBDA_DATA_CONFIG_NAME = "dataset_yambda.yaml"
VK_LSVD_DATA_CONFIG_NAME = "dataset_vk_lsvd.yaml"


@dataclass(frozen=True)
class TokenizerTarget:
    """One dataset to tokenize; exactly one source field is set per family."""

    name: str
    family: str  # "amazon" | "yambda" | "vk_lsvd"
    embeddings_path: Path | None = None
    dataset_config: Mapping[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the exp1 RQ-VAE semantic-ID tokenizer on the exp1 datasets "
            "(Amazon Beauty 2014, Instruments 2018, Beauty 2023, Yambda 50M, "
            "Yambda 500M, and the VK-LSVD subsamples) and report semantic-ID "
            "collision statistics."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys (any mix of beauty, instruments, beauty_2023, "
            "yambda_50m_listens, yambda_500m_listens, vk_lsvd_ur0.01_ir0.01, "
            "vk_lsvd_ur0.01_ip0.01) or 'all'; defaults to every configured "
            "dataset."
        ),
    )
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help=(
            "Also tokenize Yambda / VK-LSVD items seen only in auxiliary "
            "events. The exp1 default matches training: the positive catalog "
            "only."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-items", type=int, default=None, help="Debug override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    semantic_config = runner.load_yaml(config_dir / SEMANTIC_CONFIG_NAME)
    embeddings_dir = runner.resolve_path(args.embeddings_dir, base=PROJECT_ROOT)
    output_dir = runner.resolve_path(args.output_dir, base=PROJECT_ROOT)
    report_path = runner.resolve_path(args.report_path, base=PROJECT_ROOT)

    targets = collect_targets(config_dir, embeddings_dir, args.datasets)
    validate_inputs(targets)

    report: dict[str, Any] = {
        "experiment": "exp1-standard_params",
        "script": "scripts/exp1-standard-params/train_tokenizer_all.py",
        "config_dir": str(config_dir),
        "semantic_config": dict(semantic_config),
        "seed": int(args.seed),
        "device": args.device,
        "include_auxiliary": bool(args.include_auxiliary),
        "epochs_override": args.epochs,
        "max_items": args.max_items,
        "started_at": utc_now(),
        "datasets": [],
    }

    for target in targets:
        started = time.perf_counter()
        store, num_catalog_items = load_embedding_store(target, args)
        print(
            f"[{target.name}] loaded {store.num_items} x {store.embedding_dim} "
            f"embeddings; training RQ-VAE tokenizer"
        )

        tokenizer = RQVAEItemTokenizer(
            build_tokenizer_config(
                semantic_config,
                input_dim=store.embedding_dim,
                seed=int(args.seed),
                device=args.device,
                epochs_override=args.epochs,
            )
        )
        tokenizer.fit(store.embeddings, store.item_ids)
        if tokenizer.semantic_id_store is None or tokenizer.collision_stats is None:
            raise RuntimeError(f"Tokenizer produced no semantic IDs for {target.name}.")

        dataset_output_dir = output_dir / target.name
        tokenizer.save(dataset_output_dir)

        entry = summarize_run(
            target,
            tokenizer,
            store=store,
            num_catalog_items=num_catalog_items,
            dataset_output_dir=dataset_output_dir,
            train_seconds=time.perf_counter() - started,
        )
        report["datasets"].append(entry)
        report["finished_at"] = utc_now()
        # Written after every dataset so a crash mid-sweep keeps finished runs.
        runner.write_json(report_path, report)
        print(
            f"[{target.name}] {entry['unique_semantic_ids']} unique semantic IDs "
            f"for {entry['num_items']} items "
            f"({entry['collision_count']} collisions before resolution) "
            f"-> {dataset_output_dir}"
        )

    print(f"Saved semantic-ID training report to {report_path}")


def collect_targets(
    config_dir: Path,
    embeddings_dir: Path,
    requested: list[str] | None,
) -> list[TokenizerTarget]:
    explicit: list[str] | None = None
    if requested is not None and "all" not in requested:
        explicit = list(dict.fromkeys(requested))

    available: list[str] = []
    targets: list[TokenizerTarget] = []

    for config_name in AMAZON_DATA_CONFIG_NAMES:
        dataset_config = runner.load_yaml(config_dir / config_name)
        for name in dict(dataset_config.get("datasets", {})):
            available.append(name)
            if explicit is not None and name not in explicit:
                continue
            targets.append(
                TokenizerTarget(
                    name=name,
                    family="amazon",
                    embeddings_path=embeddings_dir / f"{name}_content_embeddings.pt",
                )
            )

    yambda_config = runner.load_yaml(config_dir / YAMBDA_DATA_CONFIG_NAME)
    for name in dict(yambda_config.get("datasets", {})):
        available.append(name)
        if explicit is not None and name not in explicit:
            continue
        targets.append(
            TokenizerTarget(name=name, family="yambda", dataset_config=yambda_config)
        )

    vk_lsvd_config = runner.load_yaml(config_dir / VK_LSVD_DATA_CONFIG_NAME)
    for name in dict(vk_lsvd_config.get("datasets", {})):
        available.append(name)
        if explicit is not None and name not in explicit:
            continue
        targets.append(
            TokenizerTarget(name=name, family="vk_lsvd", dataset_config=vk_lsvd_config)
        )

    if explicit is not None:
        unknown = [name for name in explicit if name not in available]
        if unknown:
            raise KeyError(f"Unknown dataset(s) {unknown}; available: {available}")
    if not targets:
        raise ValueError("No datasets selected.")
    return targets


def validate_inputs(targets: list[TokenizerTarget]) -> None:
    """Fail before any training if a later dataset's inputs are absent."""

    missing = [
        target.embeddings_path
        for target in targets
        if target.family == "amazon"
        and target.embeddings_path is not None
        and not target.embeddings_path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in dict.fromkeys(missing))
        raise FileNotFoundError(
            "Missing Amazon content-embedding stores. Build them with "
            f"build_amazon_embeddings.py first:\n{formatted}"
        )

    for target in targets:
        if target.family == "yambda":
            assert target.dataset_config is not None
            run_sasrec_all.validate_yambda_inputs([target.name], target.dataset_config)
        elif target.family == "vk_lsvd":
            assert target.dataset_config is not None
            run_sasrec_all.validate_vk_lsvd_inputs([target.name], target.dataset_config)


def load_embedding_store(
    target: TokenizerTarget,
    args: argparse.Namespace,
) -> tuple[ItemEmbeddingStore, int | None]:
    """Return the store to tokenize and, for Yambda / VK-LSVD, the catalog size."""

    if target.family == "amazon":
        assert target.embeddings_path is not None
        store = ItemEmbeddingStore.load(target.embeddings_path)
        if args.max_items is not None:
            store = ItemEmbeddingStore(
                item_ids=store.item_ids[: args.max_items],
                embeddings=store.embeddings[: args.max_items],
                texts=store.texts[: args.max_items] if store.texts is not None else None,
                metadata={**store.metadata, "max_items": args.max_items},
            )
        return store, None

    assert target.dataset_config is not None
    if target.family == "vk_lsvd":
        vk_config = run_sasrec_all.build_vk_lsvd_data_config(
            target.name, target.dataset_config, seed=int(args.seed)
        )
        print(f"[{target.name}] setting up VKLSVDDataModule to derive the item catalog")
        vk_datamodule = VKLSVDDataModule(vk_config)
        vk_datamodule.setup()
        store = load_vk_lsvd_item_embedding_store_for_datamodule(
            vk_datamodule,
            include_auxiliary=bool(args.include_auxiliary),
            max_items=args.max_items,
        )
        return store, vk_datamodule.num_items

    data_config = run_sasrec_all.build_yambda_data_config(
        target.name, target.dataset_config, seed=int(args.seed)
    )
    print(f"[{target.name}] setting up YambdaDataModule to derive the item catalog")
    datamodule = YambdaDataModule(data_config)
    datamodule.setup()
    store = load_yambda_audio_embedding_store_for_datamodule(
        datamodule,
        include_auxiliary=bool(args.include_auxiliary),
        max_items=args.max_items,
    )
    return store, datamodule.num_items


def build_tokenizer_config(
    semantic_config: Mapping[str, Any],
    *,
    input_dim: int,
    seed: int,
    device: str | None,
    epochs_override: int | None,
) -> RQVAEItemTokenizerConfig:
    """Map semantic_ids.yaml keys onto the tokenizer config dataclasses."""

    gradient_clip_norm = semantic_config.get("rqvae_gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0:
        gradient_clip_norm = None
    collision_token_offset = semantic_config.get("collision_token_offset")
    collision_suffix_size = semantic_config.get("collision_suffix_size")

    return RQVAEItemTokenizerConfig(
        rqvae=RQVAEConfig(
            input_dim=int(input_dim),
            latent_dim=int(semantic_config.get("rqvae_latent_dim", 32)),
            hidden_dims=tuple(
                int(dim) for dim in semantic_config.get("rqvae_hidden_dims", (512, 256, 128))
            ),
            num_codebooks=int(semantic_config.get("num_codebooks", 3)),
            codebook_size=int(semantic_config.get("codebook_size", 256)),
            commitment_weight=float(semantic_config.get("commitment_weight", 0.25)),
            dropout=float(semantic_config.get("rqvae_dropout", 0.0)),
            use_batch_norm=bool(semantic_config.get("rqvae_use_batch_norm", False)),
        ),
        train=RQVAETrainConfig(
            epochs=int(
                epochs_override
                if epochs_override is not None
                else semantic_config.get("rqvae_epochs", 500)
            ),
            batch_size=int(semantic_config.get("rqvae_batch_size", 1024)),
            learning_rate=float(semantic_config.get("rqvae_lr", 1e-3)),
            weight_decay=float(semantic_config.get("rqvae_weight_decay", 0.0)),
            val_ratio=float(semantic_config.get("rqvae_val_ratio", 0.05)),
            seed=seed,
            gradient_clip_norm=gradient_clip_norm,
            device=device,
            codebook_init=str(semantic_config.get("rqvae_codebook_init", "kmeans")),
        ),
        normalize=str(semantic_config.get("semantic_normalize", "none")),
        collision_policy=str(semantic_config.get("collision_policy", "append")),
        level_offset_tokens=bool(semantic_config.get("level_offset_tokens", True)),
        collision_token_offset=(
            None if collision_token_offset is None else int(collision_token_offset)
        ),
        collision_suffix_size=(
            None if collision_suffix_size is None else int(collision_suffix_size)
        ),
        collision_seed=seed,
        encode_batch_size=int(semantic_config.get("rqvae_encode_batch_size", 4096)),
    )


def summarize_run(
    target: TokenizerTarget,
    tokenizer: RQVAEItemTokenizer,
    *,
    store: ItemEmbeddingStore,
    num_catalog_items: int | None,
    dataset_output_dir: Path,
    train_seconds: float,
) -> dict[str, Any]:
    sid_store = tokenizer.semantic_id_store
    stats = tokenizer.collision_stats
    assert sid_store is not None and stats is not None
    entry: dict[str, Any] = {
        "dataset": target.name,
        "family": target.family,
        "embeddings_source": str(
            target.embeddings_path
            if target.family == "amazon"
            else target.dataset_config["embeddings_path"]  # type: ignore[index]
        ),
        "num_items": store.num_items,
        "embedding_dim": store.embedding_dim,
        "output_dir": str(dataset_output_dir),
        "semantic_id_path": str(dataset_output_dir / "semantic_ids.json"),
        "semantic_id_length": sid_store.sid_length,
        "unique_semantic_ids": sid_store.unique_code_count,
        # Items whose raw RQ-VAE code tuple clashed with another item's, before
        # the collision policy appended suffix tokens.
        "collision_count": stats.num_collisions_before,
        "unresolved_collision_count": sid_store.collision_count,
        "collision_stats": stats.to_dict(),
        "best_val_loss": (
            tokenizer.train_state.best_val_loss if tokenizer.train_state else None
        ),
        "best_epoch": (
            tokenizer.train_state.best_epoch if tokenizer.train_state else None
        ),
        "train_seconds": round(train_seconds, 2),
        "rqvae": asdict(tokenizer.config.rqvae),
        "train": asdict(tokenizer.config.train),
        "normalize": tokenizer.config.normalize,
        "collision_policy": tokenizer.config.collision_policy,
    }
    if num_catalog_items is not None:
        entry["num_catalog_items"] = num_catalog_items
    return entry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
