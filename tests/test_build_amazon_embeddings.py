"""Tests for the exp1 Amazon content-embedding builder script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from flowrec.item_representation.base import ItemEmbeddingStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "exp1-standard-params" / "build_amazon_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("build_amazon_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_embeddings = importlib.util.module_from_spec(SPEC)
# The dataclasses in the script resolve their module through sys.modules.
sys.modules[SPEC.name] = build_embeddings
SPEC.loader.exec_module(build_embeddings)


def test_targets_resolve_all_three_amazon_datasets() -> None:
    targets = build_embeddings.collect_targets(
        build_embeddings.DEFAULT_CONFIG_DIR, None
    )

    by_name = {target.name: target for target in targets}
    assert list(by_name) == ["beauty", "instruments", "beauty_2023"]
    for target in targets:
        assert target.min_user_interactions == 5
        assert target.min_item_interactions == 5
        assert target.verified_only is True
        assert target.reviews_path.is_absolute()
        assert target.metadata_path.is_absolute()

    assert by_name["beauty"].reviews_item_key == "asin"
    assert by_name["beauty"].metadata_item_key == "asin"
    assert by_name["beauty_2023"].reviews_item_key == "parent_asin"
    assert by_name["beauty_2023"].metadata_item_key == "parent_asin"

    assert build_embeddings.embeddings_output_path(
        build_embeddings.DEFAULT_OUTPUT_DIR, "beauty"
    ) == (
        PROJECT_ROOT
        / "datasets/embeddings/exp1-standard_params/beauty_content_embeddings.pt"
    )


def test_unknown_dataset_selection_raises() -> None:
    with pytest.raises(KeyError, match="unknown_dataset"):
        build_embeddings.collect_targets(
            build_embeddings.DEFAULT_CONFIG_DIR, ["unknown_dataset"]
        )


def test_hashing_embedder_builds_catalog_restricted_store(tmp_path: Path) -> None:
    # Items: a and b survive the verified-only 2-core; c is unverified-only
    # and d has too few interactions.
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u1", "u2", "u3"],
            "asin": ["a", "b", "a", "b", "c", "d", "d"],
            "timestamp": range(7),
            "rating": [5.0] * 7,
            "verified_purchase": [True, True, True, True, False, True, False],
        }
    )
    metadata = pd.DataFrame(
        {
            "asin": ["a", "b", "c", "d"],
            "title": [f"Item {key}" for key in ("a", "b", "c", "d")],
            "description": [f"Description {key}" for key in ("a", "b", "c", "d")],
        }
    )
    reviews_path = tmp_path / "reviews.parquet"
    metadata_path = tmp_path / "metadata.parquet"
    reviews.to_parquet(reviews_path)
    metadata.to_parquet(metadata_path)

    target = build_embeddings.EmbeddingTarget(
        name="tiny",
        reviews_path=reviews_path,
        metadata_path=metadata_path,
        reviews_item_key="asin",
        metadata_item_key="asin",
        min_user_interactions=2,
        min_item_interactions=2,
        verified_only=True,
    )
    allowed = build_embeddings.allowed_item_ids(target)
    assert allowed == {"a", "b"}

    from flowrec.item_representation.content_embeddings import (
        AmazonItemTextBuilder,
        ContentEmbeddingBuilder,
        HashingTextEmbedder,
    )

    builder = ContentEmbeddingBuilder(
        text_builder=AmazonItemTextBuilder(),
        embedder=HashingTextEmbedder(dim=16, normalize=False),
        item_key="asin",
    )
    result = builder.build_from_metadata(
        metadata_path=metadata_path,
        batch_size=8,
        save_texts=False,
        allowed_item_ids=allowed,
    )
    output_path = build_embeddings.embeddings_output_path(tmp_path / "out", "tiny")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.store.save(output_path)

    store = ItemEmbeddingStore.load(output_path)
    assert set(store.item_ids) == {"a", "b"}
    assert store.embeddings.shape == (2, 16)
