from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from .base import ContentEmbedder, ItemEmbeddingStore, ItemTextBuilder

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True)
class AmazonItemTextBuilder(ItemTextBuilder):
    """Build item text from Amazon Reviews 2023 metadata records.

    The Amazon Reviews 2023 metadata schema includes fields such as ``title``,
    ``features``, ``description``, ``price``, ``images``, ``videos``, ``store``,
    ``categories``, ``details``, and ``parent_asin``. This builder intentionally
    ignores image/video URLs and focuses on stable text fields.
    """

    include_title: bool = True
    include_main_category: bool = True
    include_categories: bool = True
    include_store: bool = True
    include_features: bool = True
    include_description: bool = True
    include_details: bool = True
    include_price: bool = False
    max_chars: int | None = 4096
    separator: str = " [SEP] "
    category_separator: str = " > "
    category_path_separator: str = " | "
    first_category_path_only: bool = False

    def build_text(self, record: Mapping[str, Any]) -> str:
        parts: list[str] = []

        if self.include_title:
            self._append_scalar(parts, "Title", record.get("title"))

        if self.include_main_category:
            self._append_scalar(
                parts,
                "Main category",
                record.get("main_category") or record.get("main_cat"),
            )

        if self.include_categories:
            categories = record.get("categories") or record.get("category")
            category_text = _format_categories(
                categories,
                category_separator=self.category_separator,
                path_separator=self.category_path_separator,
                first_path_only=self.first_category_path_only,
            )
            if category_text:
                parts.append("Categories: " + category_text)

        if self.include_store:
            self._append_scalar(parts, "Store", record.get("store") or record.get("brand"))

        if self.include_features:
            features = record.get("features") or record.get("feature")
            if isinstance(features, list):
                clean = [str(x) for x in features if x]
                if clean:
                    parts.append("Features: " + " ".join(clean))

        if self.include_description:
            description = record.get("description")
            if isinstance(description, list):
                clean = [str(x) for x in description if x]
                if clean:
                    parts.append("Description: " + " ".join(clean))
            elif description:
                parts.append("Description: " + str(description))

        if self.include_details:
            details = record.get("details")
            if isinstance(details, Mapping):
                detail_parts = [
                    f"{k}: {v}"
                    for k, v in details.items()
                    if k and v and not _looks_like_url(str(v))
                ]
                if detail_parts:
                    parts.append("Details: " + "; ".join(detail_parts))

        if self.include_price:
            self._append_scalar(parts, "Price", record.get("price"))

        text = self.separator.join(p.strip() for p in parts if p and p.strip())
        text = " ".join(text.split())

        if self.max_chars is not None and len(text) > self.max_chars:
            text = text[: self.max_chars]

        return text

    @staticmethod
    def _append_scalar(parts: list[str], label: str, value: Any) -> None:
        if value is not None and value != "":
            parts.append(f"{label}: {value}")


@dataclass(frozen=True)
class HashingTextEmbedder(ContentEmbedder):
    """Deterministic lightweight text embedder for tests and quick baselines.

    This is not meant to replace a real language-model embedding. It is useful
    when running smoke tests, CI, or tiny experiments without downloading a model.
    """

    dim: int = 384
    use_bigrams: bool = True
    normalize: bool = True
    lowercase: bool = True
    salt: str = "flowrec-hashing-embedder-v1"

    @property
    def output_dim(self) -> int:
        return int(self.dim)

    @property
    def name(self) -> str:
        return f"hashing:{self.dim}"

    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 128) -> torch.Tensor:
        del batch_size  # hashing is CPU-local and cheap enough to process directly
        rows = [self._encode_one(text) for text in texts]
        if not rows:
            return torch.empty((0, self.dim), dtype=torch.float32)
        embeddings = torch.stack(rows, dim=0)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _encode_one(self, text: str) -> torch.Tensor:
        text = text or ""
        if self.lowercase:
            text = text.lower()
        tokens = _WORD_RE.findall(text)
        features = list(tokens)
        if self.use_bigrams and len(tokens) > 1:
            features.extend(f"{a}_{b}" for a, b in zip(tokens[:-1], tokens[1:]))

        vec = torch.zeros(self.dim, dtype=torch.float32)
        if not features:
            return vec

        for token in features:
            digest = hashlib.blake2b(
                f"{self.salt}:{token}".encode("utf-8"), digest_size=8
            ).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            idx = value % self.dim
            sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
            vec[idx] += sign

        # Term-frequency scaling helps prevent long descriptions from dominating.
        vec /= math.sqrt(len(features))
        return vec


@dataclass
class HFTextEmbedder(ContentEmbedder):
    """Hugging Face transformer mean-pooling text embedder.

    This class intentionally imports ``transformers`` lazily so FlowRec can be
    installed without the dependency when users only need precomputed embeddings.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | torch.device | None = None
    max_length: int = 256
    normalize: bool = True
    pooling: Literal["masked_mean", "padded_mean"] = "masked_mean"
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    show_progress: bool = False
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _output_dim: int | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    @property
    def output_dim(self) -> int:
        self._ensure_loaded()
        assert self._output_dim is not None
        return self._output_dim

    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 128) -> torch.Tensor:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        if self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        self._model.to(device)
        self._model.eval()

        outputs: list[torch.Tensor] = []
        starts = range(0, len(texts), batch_size)
        if self.show_progress:
            try:
                from tqdm.auto import tqdm
                starts = tqdm(starts, desc=f"Embedding items with {self.model_name}")
            except ImportError:
                pass

        with torch.inference_mode():
            for start in starts:
                batch_texts = list(texts[start : start + batch_size])
                encoded = self._tokenizer(
                    batch_texts,
                    padding="max_length" if self.pooling == "padded_mean" else True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model_out = self._model(**encoded)
                if self.pooling == "masked_mean":
                    pooled = _mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                elif self.pooling == "padded_mean":
                    pooled = _padded_mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                else:
                    raise ValueError(f"Unknown pooling mode: {self.pooling}")
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.detach().cpu())

        if not outputs:
            return torch.empty((0, self.output_dim), dtype=torch.float32)
        return torch.cat(outputs, dim=0).float()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFTextEmbedder requires transformers. Install it with: "
                "pip install transformers accelerate"
            ) from exc

        dtype = None
        if self.torch_dtype:
            dtype = getattr(torch, self.torch_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
        )
        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self._model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for model {self.model_name}")
        self._output_dim = int(hidden_size)


@dataclass(frozen=True)
class EmbeddingBuildResult:
    store: ItemEmbeddingStore
    num_records_read: int
    num_empty_texts: int
    num_duplicate_items: int
    num_filtered_items: int = 0


class ContentEmbeddingBuilder:
    """Build item content embeddings from an Amazon metadata file."""

    def __init__(
        self,
        text_builder: ItemTextBuilder | None = None,
        embedder: ContentEmbedder | None = None,
        item_key: str = "parent_asin",
        keep_first_duplicate: bool = True,
    ) -> None:
        self.text_builder = text_builder or AmazonItemTextBuilder()
        self.embedder = embedder or HashingTextEmbedder()
        self.item_key = item_key
        self.keep_first_duplicate = keep_first_duplicate

    def build_from_metadata(
        self,
        metadata_path: str | Path,
        *,
        batch_size: int = 128,
        save_texts: bool = False,
        allowed_item_ids: Iterable[str] | None = None,
    ) -> EmbeddingBuildResult:
        metadata_path = Path(metadata_path)
        allowed_items = set(map(str, allowed_item_ids)) if allowed_item_ids is not None else None
        item_ids: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()
        num_records = 0
        num_duplicates = 0
        num_empty = 0
        num_filtered = 0

        for record in iter_jsonl(metadata_path):
            num_records += 1
            raw_item_id = record.get(self.item_key) or record.get("parent_asin") or record.get("asin")
            if not raw_item_id:
                continue
            raw_item_id = str(raw_item_id)
            if allowed_items is not None and raw_item_id not in allowed_items:
                num_filtered += 1
                continue

            if raw_item_id in seen:
                num_duplicates += 1
                if self.keep_first_duplicate:
                    continue
            seen.add(raw_item_id)

            text = self.text_builder.build_text(record)
            if not text:
                num_empty += 1
                # Use the item ID as a fallback to preserve row alignment.
                text = raw_item_id

            item_ids.append(raw_item_id)
            texts.append(text)

        embeddings = self.embedder.encode_texts(texts, batch_size=batch_size)
        store = ItemEmbeddingStore(
            item_ids=item_ids,
            embeddings=embeddings,
            texts=texts if save_texts else None,
            metadata={
                "source": str(metadata_path),
                "item_key": self.item_key,
                "embedder": self.embedder.name,
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
                "text_builder": type(self.text_builder).__name__,
                "save_texts": save_texts,
                "allowed_item_count": len(allowed_items) if allowed_items is not None else None,
                "num_filtered_items": num_filtered,
            },
        )
        return EmbeddingBuildResult(
            store=store,
            num_records_read=num_records,
            num_empty_texts=num_empty,
            num_duplicate_items=num_duplicates,
            num_filtered_items=num_filtered,
        )



def _loads_json_or_python_literal(text: str, *, path: str | Path, line_no: int | None = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            loc = f"{path}:{line_no}" if line_no is not None else str(path)
            preview = text[:160].replace("\n", "\\n")
            raise ValueError(f"Invalid JSON/JSON-like record at {loc}. Preview: {preview!r}") from exc


def _looks_like_single_record(obj: Mapping[str, Any]) -> bool:
    record_keys = {
        "user_id", "reviewerID", "rating", "overall", "timestamp", "unixReviewTime",
        "asin", "parent_asin", "title", "description", "features", "categories",
        "details", "main_category", "store", "price",
    }
    return any(key in obj for key in record_keys)


def _iter_records_from_object(obj: Any, *, path: str | Path) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for idx, record in enumerate(obj):
            if not isinstance(record, dict):
                raise ValueError(f"Expected object record at {path}[{idx}], got {type(record)}")
            yield record
        return

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object or list at {path}, got {type(obj)}")

    for key in ("records", "data", "reviews", "metadata", "items"):
        value = obj.get(key)
        if isinstance(value, list):
            yield from _iter_records_from_object(value, path=path)
            return

    if _looks_like_single_record(obj):
        yield obj
        return

    for key, value in obj.items():
        if not isinstance(value, dict):
            continue
        record = dict(value)
        record.setdefault("raw_item_id", str(key))
        record.setdefault("item_id", str(key))
        if "parent_asin" not in record and "asin" not in record:
            record["parent_asin"] = str(key)
        yield record


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield records from JSONL, JSON, gzip JSONL, or gzip JSON files.

    Also supports legacy Amazon files where each line is a Python-literal dict
    using single quotes rather than strict JSON double quotes.
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    opener = gzip.open if path.suffix == ".gz" else open

    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        with opener(path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = _loads_json_or_python_literal(line, path=path, line_no=line_no)
                yield from _iter_records_from_object(obj, path=f"{path}:{line_no}")
        return

    with opener(path, "rt", encoding="utf-8") as f:
        first_line = None
        first_line_no = 0
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                first_line = line.strip()
                first_line_no = line_no
                break

        if first_line is None:
            return

        if first_line[0] == "[":
            obj = _loads_json_or_python_literal(first_line + "\n" + f.read(), path=path)
            yield from _iter_records_from_object(obj, path=path)
            return

        try:
            first_obj = _loads_json_or_python_literal(first_line, path=path, line_no=first_line_no)
            line_mode = True
        except ValueError:
            first_obj = None
            line_mode = False

        if line_mode:
            yield from _iter_records_from_object(first_obj, path=f"{path}:{first_line_no}")
            for line_no, line in enumerate(f, start=first_line_no + 1):
                line = line.strip()
                if not line:
                    continue
                obj = _loads_json_or_python_literal(line, path=path, line_no=line_no)
                yield from _iter_records_from_object(obj, path=f"{path}:{line_no}")
            return

        obj = _loads_json_or_python_literal(first_line + "\n" + f.read(), path=path)
        yield from _iter_records_from_object(obj, path=path)
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-8)
    return summed / denom


def _padded_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    return (last_hidden_state * mask).mean(dim=1)


def _format_categories(
    categories: Any,
    *,
    category_separator: str = " > ",
    path_separator: str = " | ",
    first_path_only: bool = False,
) -> str:
    if not isinstance(categories, list) or not categories:
        return ""

    if all(isinstance(x, str) for x in categories):
        return category_separator.join(str(x) for x in categories if x)

    paths: list[str] = []
    for path in categories:
        if isinstance(path, list):
            clean = [str(x) for x in path if x]
            if clean:
                paths.append(category_separator.join(clean))
                if first_path_only:
                    break
        elif path:
            paths.append(str(path))
            if first_path_only:
                break
    return path_separator.join(paths)


def _looks_like_url(value: str) -> bool:
    value = value.lower().strip()
    return value.startswith("http://") or value.startswith("https://")
