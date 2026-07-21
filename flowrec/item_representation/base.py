from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ItemEmbeddingStore:
    """Container for item content embeddings.

    Attributes
    ----------
    item_ids:
        Raw item identifiers, for example Amazon Reviews 2023 ``parent_asin``.
    embeddings:
        Tensor with shape ``[num_items, embedding_dim]``.
    texts:
        Optional text used to produce each embedding. Useful for debugging but can
        be omitted for large datasets.
    metadata:
        Free-form metadata describing the embedding model, item key, source file,
        text fields, normalization, and so on.
    """

    item_ids: list[str]
    embeddings: torch.Tensor
    texts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be a 2D tensor, got shape {tuple(self.embeddings.shape)}"
            )
        if len(self.item_ids) != int(self.embeddings.shape[0]):
            raise ValueError(
                "item_ids length must match embeddings first dimension: "
                f"{len(self.item_ids)} != {int(self.embeddings.shape[0])}"
            )
        if self.texts is not None and len(self.texts) != len(self.item_ids):
            raise ValueError("texts length must match item_ids length")

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def subset(self, item_ids: Iterable[str]) -> "ItemEmbeddingStore":
        """Return a new store restricted to ``item_ids`` present in this store.

        This is used to keep only the items that survive dataset preprocessing
        (for example ``dm.item_ids_in_data()``) before training a semantic-ID
        tokenizer, so semantic IDs are generated exactly for the items that
        remain in the final dataset. This store's item ordering is preserved and
        ``item_ids``/``embeddings`` stay aligned.
        """
        allowed = {str(x) for x in item_ids}
        keep_idx = [i for i, item in enumerate(self.item_ids) if item in allowed]
        if not keep_idx:
            raise ValueError(
                "ItemEmbeddingStore.subset() produced an empty store: none of the "
                "requested item_ids are present in this store."
            )
        index = torch.tensor(keep_idx, dtype=torch.long)
        return ItemEmbeddingStore(
            item_ids=[self.item_ids[i] for i in keep_idx],
            embeddings=self.embeddings.index_select(0, index),
            texts=[self.texts[i] for i in keep_idx] if self.texts is not None else None,
            metadata={
                **self.metadata,
                "num_items_before_subset": self.num_items,
                "num_items_after_subset": len(keep_idx),
            },
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "item_ids": self.item_ids,
                "embeddings": self.embeddings.detach().cpu(),
                "texts": self.texts,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ItemEmbeddingStore":
        obj = torch.load(Path(path), map_location=map_location)
        return cls(
            item_ids=[str(x) for x in obj["item_ids"]],
            embeddings=obj["embeddings"].float(),
            texts=obj.get("texts"),
            metadata=dict(obj.get("metadata", {})),
        )


@dataclass(frozen=True)
class SemanticIDStore:
    """Container for semantic item identifiers.

    ``semantic_ids`` maps raw item IDs to a list of integer tokens. The tokens are
    intentionally kept 0-based by default. The datamodule can offset them by +1
    to reserve token 0 for padding.
    """

    semantic_ids: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.semantic_ids:
            raise ValueError("semantic_ids cannot be empty")
        lengths = {len(v) for v in self.semantic_ids.values()}
        if len(lengths) != 1:
            raise ValueError(
                "All semantic IDs must have the same length in SemanticIDStore; "
                f"observed lengths: {sorted(lengths)}"
            )

    @property
    def num_items(self) -> int:
        return len(self.semantic_ids)

    @property
    def sid_length(self) -> int:
        first = next(iter(self.semantic_ids.values()))
        return len(first)

    @property
    def unique_code_count(self) -> int:
        return len({tuple(tokens) for tokens in self.semantic_ids.values()})

    @property
    def collision_count(self) -> int:
        return self.num_items - self.unique_code_count

    def to_plain_dict(self) -> dict[str, list[int]]:
        return {item_id: list(map(int, tokens)) for item_id, tokens in self.semantic_ids.items()}

    def save_json(self, path: str | Path, *, include_metadata: bool = False) -> None:
        """Save semantic IDs as JSON.

        By default this writes exactly the mapping expected by the FlowRec
        datamodule:

        ``{"ITEM_ID": [1, 2, 3], ...}``

        If ``include_metadata=True``, each value is written as
        ``{"tokens": [...], "metadata": ...}``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            payload: dict[str, Any] = {
                item_id: {"tokens": tokens} for item_id, tokens in self.to_plain_dict().items()
            }
            payload["__metadata__"] = self.metadata
        else:
            payload = self.to_plain_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def save_pt(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "semantic_ids": self.to_plain_dict(),
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SemanticIDStore":
        with open(Path(path), "r", encoding="utf-8") as f:
            raw = json.load(f)

        metadata = {}
        if "__metadata__" in raw:
            metadata = dict(raw.pop("__metadata__"))

        semantic_ids: dict[str, list[int]] = {}
        for item_id, value in raw.items():
            if isinstance(value, Mapping):
                tokens = value.get("tokens")
            else:
                tokens = value
            if not isinstance(tokens, list):
                raise ValueError(f"Invalid semantic ID value for item {item_id!r}")
            semantic_ids[str(item_id)] = [int(x) for x in tokens]

        return cls(semantic_ids=semantic_ids, metadata=metadata)


class ItemTextBuilder(ABC):
    """Build a model-consumable text string from an item metadata record."""

    @abstractmethod
    def build_text(self, record: Mapping[str, Any]) -> str:
        raise NotImplementedError


class ContentEmbedder(ABC):
    """Encode item text into dense content embeddings."""

    @abstractmethod
    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 128) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class SemanticIDTokenizer(ABC):
    """Trainable mapping from dense item embeddings to discrete semantic IDs."""

    @abstractmethod
    def fit(self, embeddings: torch.Tensor, item_ids: Sequence[str] | None = None) -> "SemanticIDTokenizer":
        raise NotImplementedError

    @abstractmethod
    def encode(self, embeddings: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, output_dir: str | Path, map_location: str | torch.device = "cpu") -> "SemanticIDTokenizer":
        raise NotImplementedError


def dataclass_to_json_dict(obj: Any) -> dict[str, Any]:
    """Convert nested dataclasses to a JSON-friendly dictionary."""
    data = asdict(obj)
    return _jsonify(data)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value
