from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from flowrec.data.vk_lsvd import iter_tabular_records

from .base import ItemEmbeddingStore


@dataclass(frozen=True, slots=True)
class VKLSVDItemEmbeddingConfig:
    """Configuration for loading VK-LSVD content embeddings.

    VK-LSVD provides ``metadata/item_embeddings.npz`` with arrays:

        item_id: uint32[num_items]
        embedding: float16[num_items, 64]

    Components are ordered by importance; use ``embedding_dim`` to keep the
    first N components for faster experiments.
    """

    embeddings_path: str | Path | None = None
    use_hf: bool = False
    hf_repo_id: str = "deepvk/VK-LSVD"
    hf_revision: str | None = None
    hf_local_dir: str | Path | None = None

    item_id_column: str = "item_id"
    embedding_column: str = "embedding"
    embedding_dim: int | None = 64
    max_items: int | None = None
    # Restrict the store to these raw item IDs (strings, or an integer array
    # for the numeric VK-LSVD IDs). Filtering happens on the numpy arrays, so
    # restricting the 19.6M-item file to a catalog never loops in Python.
    allowed_items: set[str] | np.ndarray | None = None


class VKLSVDItemEmbeddingBuilder:
    """Load VK-LSVD item content embeddings into ItemEmbeddingStore."""

    def __init__(self, cfg: VKLSVDItemEmbeddingConfig) -> None:
        self.cfg = cfg

    def build(self) -> ItemEmbeddingStore:
        path = self._resolve_path()
        if path.suffix.lower() == ".npz":
            return self._build_from_npz(path)
        return self._build_from_tabular(path)

    def _resolve_path(self) -> Path:
        if not self.cfg.use_hf:
            if self.cfg.embeddings_path is None:
                raise ValueError("embeddings_path is required when use_hf=False")
            return Path(self.cfg.embeddings_path)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Loading VK-LSVD embeddings from Hugging Face requires huggingface_hub. "
                "Install with: pip install huggingface_hub"
            ) from exc

        filename = "metadata/item_embeddings.npz"
        return Path(
            hf_hub_download(
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                filename=filename,
                revision=self.cfg.hf_revision,
                local_dir=str(self.cfg.hf_local_dir) if self.cfg.hf_local_dir is not None else None,
            )
        )

    def _build_from_npz(self, path: Path) -> ItemEmbeddingStore:
        with np.load(path) as data:
            if self.cfg.item_id_column not in data.files or self.cfg.embedding_column not in data.files:
                raise ValueError(
                    f"{path} must contain {self.cfg.item_id_column!r} and "
                    f"{self.cfg.embedding_column!r} arrays; found {data.files}"
                )
            item_ids_arr = data[self.cfg.item_id_column]
            keep = np.ones(len(item_ids_arr), dtype=bool)
            skipped_allowed = 0
            if self.cfg.allowed_items is not None:
                keep = _allowed_item_mask(item_ids_arr, self.cfg.allowed_items)
                skipped_allowed = int((~keep).sum())
            index = np.flatnonzero(keep)
            if self.cfg.max_items is not None:
                index = index[: int(self.cfg.max_items)]
            if index.size == 0:
                raise ValueError("No VK-LSVD item embeddings were loaded")

            embeddings_arr = data[self.cfg.embedding_column]
            if embeddings_arr.ndim != 2:
                raise ValueError(
                    f"VK-LSVD embeddings must be a 2D array, got shape {embeddings_arr.shape}"
                )
            selected = embeddings_arr[index]
        if self.cfg.embedding_dim is not None:
            selected = selected[:, : int(self.cfg.embedding_dim)]
        embeddings = torch.from_numpy(np.ascontiguousarray(selected, dtype=np.float32))
        item_ids = _item_ids_to_strings(item_ids_arr[index])

        return ItemEmbeddingStore(
            item_ids=item_ids,
            embeddings=embeddings,
            texts=None,
            metadata={
                "source": str(path),
                "dataset": "vk_lsvd",
                "embedding_column": self.cfg.embedding_column,
                "embedding_dim": int(embeddings.shape[1]),
                "num_source_items": int(len(item_ids_arr)),
                "num_skipped_allowed": skipped_allowed,
            },
        )

    def _build_from_tabular(self, path: Path) -> ItemEmbeddingStore:
        item_ids: list[str] = []
        vectors: list[torch.Tensor] = []
        seen: set[str] = set()
        skipped_missing = 0
        skipped_allowed = 0
        duplicates = 0
        allowed = _allowed_items_as_strings(self.cfg.allowed_items)

        for record in iter_tabular_records(path, max_rows=self.cfg.max_items):
            raw_item_id = record.get(self.cfg.item_id_column, record.get("item_id"))
            if raw_item_id is None:
                skipped_missing += 1
                continue
            item_id = str(raw_item_id)
            if allowed is not None and item_id not in allowed:
                skipped_allowed += 1
                continue
            if item_id in seen:
                duplicates += 1
                continue
            value = record.get(self.cfg.embedding_column)
            if value is None:
                skipped_missing += 1
                continue
            vector = _embedding_value_to_tensor(value)
            if self.cfg.embedding_dim is not None:
                vector = vector[: int(self.cfg.embedding_dim)]
            item_ids.append(item_id)
            vectors.append(vector.float())
            seen.add(item_id)

        if not vectors:
            raise ValueError("No VK-LSVD item embeddings were loaded")

        embeddings = torch.stack(vectors, dim=0)
        return ItemEmbeddingStore(
            item_ids=item_ids,
            embeddings=embeddings,
            texts=None,
            metadata={
                "source": str(path),
                "dataset": "vk_lsvd",
                "embedding_column": self.cfg.embedding_column,
                "embedding_dim": int(embeddings.shape[1]),
                "num_skipped_missing": skipped_missing,
                "num_skipped_allowed": skipped_allowed,
                "num_duplicates": duplicates,
            },
        )


def load_vk_lsvd_item_embedding_store(
    embeddings_path: str | Path,
    *,
    embedding_dim: int | None = 64,
    max_items: int | None = None,
    allowed_items: set[str] | np.ndarray | None = None,
) -> ItemEmbeddingStore:
    return VKLSVDItemEmbeddingBuilder(
        VKLSVDItemEmbeddingConfig(
            embeddings_path=embeddings_path,
            embedding_dim=embedding_dim,
            max_items=max_items,
            allowed_items=allowed_items,
        )
    ).build()


def load_vk_lsvd_item_embedding_store_for_datamodule(
    datamodule,
    *,
    embeddings_path: str | Path | None = None,
    include_auxiliary: bool = False,
    embedding_dim: int | None = 64,
    max_items: int | None = None,
) -> ItemEmbeddingStore:
    """Load VK-LSVD embeddings restricted to an initialized datamodule's items.

    The VK-LSVD embedding file covers all 19.6M videos; the semantic-ID
    tokenizer only needs the items that survived preprocessing (the positive
    catalog, which is also the negative-sampling universe).
    ``include_auxiliary=True`` also covers items that catalog users touched
    only in auxiliary events. Filtering happens on the numpy arrays before any
    tensor is built. ``embeddings_path`` defaults to the datamodule's own
    ``embeddings_path``.
    """
    path = embeddings_path if embeddings_path is not None else datamodule.embeddings_path
    if path is None:
        raise ValueError(
            "embeddings_path is required: pass it explicitly or configure it "
            "on the datamodule (VKLSVDDataConfig.embeddings_path)."
        )
    allowed_items = datamodule.item_ids_in_data(include_auxiliary=include_auxiliary)
    return load_vk_lsvd_item_embedding_store(
        path,
        embedding_dim=embedding_dim,
        max_items=max_items,
        allowed_items=allowed_items,
    )


def _allowed_item_mask(item_ids_arr: np.ndarray, allowed_items: set[str] | np.ndarray) -> np.ndarray:
    """Vectorized membership of the source item IDs in ``allowed_items``."""
    if isinstance(allowed_items, np.ndarray):
        return np.isin(item_ids_arr.astype(np.int64), allowed_items.astype(np.int64))
    if np.issubdtype(item_ids_arr.dtype, np.integer):
        allowed_list = list(allowed_items)
        if all(str(x).lstrip("-").isdigit() for x in allowed_list):
            allowed_ints = np.fromiter((int(x) for x in allowed_list), dtype=np.int64, count=len(allowed_list))
            return np.isin(item_ids_arr.astype(np.int64), allowed_ints)
    return np.isin(np.asarray(_item_ids_to_strings(item_ids_arr), dtype=object), list(allowed_items))


def _allowed_items_as_strings(allowed_items: set[str] | np.ndarray | None) -> set[str] | None:
    if allowed_items is None:
        return None
    if isinstance(allowed_items, np.ndarray):
        return {str(int(x)) for x in allowed_items.tolist()}
    return {str(x) for x in allowed_items}


def _item_ids_to_strings(item_ids_arr: np.ndarray) -> list[str]:
    if np.issubdtype(item_ids_arr.dtype, np.integer):
        return item_ids_arr.astype(np.int64).astype(str).tolist()
    return [str(x) for x in item_ids_arr.tolist()]


def _embedding_value_to_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().float()
    else:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if isinstance(value, str):
            import json

            value = json.loads(value)
        if not isinstance(value, Sequence):
            raise ValueError(f"Unsupported embedding value type: {type(value)}")
        tensor = torch.tensor([float(x) for x in value], dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError("VK-LSVD embedding must be a 1D vector")
    return tensor
