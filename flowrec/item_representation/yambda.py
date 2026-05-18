from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from flowrec.data.yambda import iter_tabular_records

from .base import ItemEmbeddingStore


@dataclass(frozen=True, slots=True)
class YambdaAudioEmbeddingConfig:
    """Configuration for loading Yambda precomputed audio embeddings."""

    embeddings_path: str | Path | None = None
    use_hf: bool = False
    hf_dataset_name: str = "yandex/yambda"
    embedding_column: str = "normalized_embed"
    fallback_embedding_column: str = "embed"
    item_id_column: str = "item_id"
    max_items: int | None = None
    allowed_items: set[str] | None = None


class YambdaAudioEmbeddingBuilder:
    """Load Yambda track audio embeddings into ItemEmbeddingStore.

    Yambda provides an ``embeddings.parquet`` file with ``item_id``, ``embed``
    and ``normalized_embed`` columns. This builder converts that file into the
    same ``ItemEmbeddingStore`` consumed by the RQ-VAE semantic-ID tokenizer.
    """

    def __init__(self, cfg: YambdaAudioEmbeddingConfig) -> None:
        self.cfg = cfg

    def build(self) -> ItemEmbeddingStore:
        if self.cfg.use_hf:
            records: Iterable[dict[str, Any]] = _iter_hf_yambda_audio_embeddings(
                dataset_name=self.cfg.hf_dataset_name,
                max_rows=self.cfg.max_items,
            )
            source = f"hf:{self.cfg.hf_dataset_name}:embeddings.parquet"
        else:
            if self.cfg.embeddings_path is None:
                raise ValueError("embeddings_path is required when use_hf=False")
            records = iter_tabular_records(self.cfg.embeddings_path, max_rows=self.cfg.max_items)
            source = str(self.cfg.embeddings_path)

        item_ids: list[str] = []
        vectors: list[torch.Tensor] = []
        seen: set[str] = set()
        skipped_missing = 0
        skipped_allowed = 0
        duplicates = 0
        for record in records:
            raw_item_id = record.get(self.cfg.item_id_column, record.get("item_id"))
            if raw_item_id is None:
                skipped_missing += 1
                continue
            raw_item_id = str(raw_item_id)
            if self.cfg.allowed_items is not None and raw_item_id not in self.cfg.allowed_items:
                skipped_allowed += 1
                continue
            if raw_item_id in seen:
                duplicates += 1
                continue

            value = record.get(self.cfg.embedding_column)
            if value is None:
                value = record.get(self.cfg.fallback_embedding_column)
            if value is None:
                skipped_missing += 1
                continue

            vector = _embedding_value_to_tensor(value)
            if vector.ndim != 1:
                raise ValueError(f"Yambda embedding for item {raw_item_id!r} is not 1D")
            item_ids.append(raw_item_id)
            vectors.append(vector.float())
            seen.add(raw_item_id)

        if not vectors:
            raise ValueError("No Yambda audio embeddings were loaded")

        embeddings = torch.stack(vectors, dim=0)
        return ItemEmbeddingStore(
            item_ids=item_ids,
            embeddings=embeddings,
            texts=None,
            metadata={
                "source": source,
                "dataset": "yambda",
                "embedding_column": self.cfg.embedding_column,
                "fallback_embedding_column": self.cfg.fallback_embedding_column,
                "embedding_dim": int(embeddings.shape[1]),
                "num_skipped_missing": skipped_missing,
                "num_skipped_allowed": skipped_allowed,
                "num_duplicates": duplicates,
            },
        )


def load_yambda_audio_embedding_store(
    embeddings_path: str | Path,
    *,
    embedding_column: str = "normalized_embed",
    max_items: int | None = None,
    allowed_items: set[str] | None = None,
) -> ItemEmbeddingStore:
    return YambdaAudioEmbeddingBuilder(
        YambdaAudioEmbeddingConfig(
            embeddings_path=embeddings_path,
            embedding_column=embedding_column,
            max_items=max_items,
            allowed_items=allowed_items,
        )
    ).build()


def _embedding_value_to_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence):
        raise ValueError(f"Unsupported embedding value type: {type(value)}")
    return torch.tensor([float(x) for x in value], dtype=torch.float32)


def _iter_hf_yambda_audio_embeddings(
    *,
    dataset_name: str = "yandex/yambda",
    max_rows: int | None = None,
):
    try:
        from datasets import DatasetDict, IterableDatasetDict, load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading Yambda audio embeddings from Hugging Face requires datasets. "
            "Install with: pip install datasets"
        ) from exc

    data = load_dataset(dataset_name, data_dir="", data_files="embeddings.parquet")
    if isinstance(data, (DatasetDict, IterableDatasetDict)):
        ds = data["train"]
    else:  # pragma: no cover
        ds = data
    for idx, record in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        yield dict(record)
