from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .collators import HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, InteractionRecord, SequenceDataset
from .preprocessors import (
    apply_k_core_filter,
    count_interactions,
    deduplicate_consecutive_items,
    iter_jsonl,
    load_semantic_ids,
    sort_user_events,
)
from .sequence_builder import (
    GlobalEvalTargetMode,
    GlobalHistoryPolicy,
    SequenceSplitConfig,
    SplitStrategy,
    TrainTargetPolicy,
    build_sequence_splits,
)

YambdaDatasetType = Literal["flat", "sequential"]
YambdaDatasetSize = Literal["50m", "500m", "5b"]
YambdaEventFile = Literal[
    "listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event"
]
OrganicFilter = Literal["any", "organic", "recommendation"]


def _is_parquet(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".parquet"


def _is_csv(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".csv", ".tsv"}


def iter_tabular_records(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate over JSON/JSONL/CSV/TSV/Parquet records.

    Yambda is distributed as Parquet on Hugging Face, but JSONL support is
    useful for tests and local debugging. Parquet support is optional and
    requires pandas plus pyarrow or fastparquet.
    """
    path = Path(path)

    if _is_parquet(path):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading Parquet requires pandas.") from exc
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - optional parquet engine
            raise ImportError(
                "Reading Parquet requires a parquet engine such as pyarrow or fastparquet. "
                "Install with: pip install pyarrow"
            ) from exc
        if max_rows is not None:
            df = df.head(max_rows)
        for record in df.to_dict(orient="records"):
            yield dict(record)
        return

    if _is_csv(path):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading CSV/TSV requires pandas.") from exc
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=max_rows)
        for record in df.to_dict(orient="records"):
            yield dict(record)
        return

    for idx, record in enumerate(iter_jsonl(path)):
        if max_rows is not None and idx >= max_rows:
            break
        yield record


def iter_hf_yambda_records(
    *,
    dataset_type: YambdaDatasetType = "flat",
    dataset_size: YambdaDatasetSize = "50m",
    event_file: YambdaEventFile = "listens",
    dataset_name: str = "yandex/yambda",
    streaming: bool = False,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load Yambda records from Hugging Face datasets.

    This follows the dataset-card usage pattern:

        load_dataset("yandex/yambda", data_dir="flat/50m", data_files="likes.parquet")
    """
    try:
        from datasets import DatasetDict, IterableDatasetDict, load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading Yambda from Hugging Face requires datasets. "
            "Install with: pip install datasets"
        ) from exc

    data = load_dataset(
        dataset_name,
        data_dir=f"{dataset_type}/{dataset_size}",
        data_files=f"{event_file}.parquet",
        streaming=streaming,
    )
    if isinstance(data, (DatasetDict, IterableDatasetDict)):
        ds = data["train"]
    else:  # pragma: no cover
        ds = data

    for idx, record in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        yield dict(record)


@dataclass(frozen=True, slots=True)
class YambdaDataConfig:
    """Configuration for Yambda sequential recommendation data.

    Supports both Yambda formats:
      - flat event rows: uid, item_id, timestamp, is_organic, ...
      - sequential rows: uid, item_ids, timestamps, is_organic, ... lists

    Use ``interactions_path`` for local files, or set ``use_hf=True`` to load
    from ``yandex/yambda``.
    """

    interactions_path: str | Path | None = None
    semantic_id_path: str | Path | None = None

    use_hf: bool = False
    hf_dataset_name: str = "yandex/yambda"
    dataset_type: YambdaDatasetType = "flat"
    dataset_size: YambdaDatasetSize = "50m"
    event_file: YambdaEventFile = "listens"
    streaming: bool = False
    max_rows: int | None = None

    # Filtering / normalization.
    min_user_interactions: int = 5
    min_item_interactions: int = 0
    deduplicate_consecutive: bool = False
    keep_raw_interactions: bool = False
    organic_filter: OrganicFilter = "any"
    min_played_ratio_pct: float | None = None
    allowed_event_types: tuple[str, ...] | None = None
    event_value_map: dict[str, float] = field(
        default_factory=lambda: {
            "listen": 1.0,
            "like": 2.0,
            "dislike": -1.0,
            "unlike": 0.0,
            "undislike": 0.0,
        }
    )
    default_rating: float = 1.0

    # Splitting.
    split_strategy: SplitStrategy = "leave_one_out"
    min_history_len: int = 1
    train_all_prefixes: bool = True

    # Training target selection for long user histories. If
    # train_target_policy == "all" and train_all_prefixes is False, the
    # effective policy is "last" for backward compatibility.
    train_target_policy: TrainTargetPolicy = "all"
    train_targets_per_user: int = 256
    train_stride: int = 10
    train_last_n: int = 512
    train_hybrid_recent_n: int = 256
    train_hybrid_random_n: int = 256
    global_val_timestamp: int | None = None
    global_test_timestamp: int | None = None
    global_val_ratio: float = 0.1
    global_test_ratio: float = 0.1
    global_history_policy: GlobalHistoryPolicy = "rolling"
    global_eval_target_mode: GlobalEvalTargetMode = "all"

    # Batching.
    max_history_len: int = 50
    history_padding: HistoryPadding = "left"
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    include_raw_in_batch: bool = True

    # IDs.
    pad_item_id: int = 0
    pad_token_id: int = 0
    offset_semantic_ids_by_one: bool = True
    drop_items_missing_semantic_id: bool = True

    seed: int = 42


class YambdaDataModule:
    """Framework-agnostic DataModule for Yambda.

    Batches match the FlowRec format used by AmazonReviewDataModule:

        history_item_ids, history_mask, target_item_ids, target_ratings, ...

    If ``semantic_id_path`` is provided, the collator also emits:

        history_token_ids, history_token_mask, target_token_ids, target_token_mask
    """

    def __init__(self, cfg: YambdaDataConfig) -> None:
        self.cfg = cfg
        self.interactions_path = Path(cfg.interactions_path) if cfg.interactions_path else None
        self.semantic_id_path = Path(cfg.semantic_id_path) if cfg.semantic_id_path else None

        self.user2id: dict[str, int] = {}
        self.id2user: list[str] = []
        self.item2id: dict[str, int] = {}
        self.id2item: list[str] = ["<PAD>"]

        self.semantic_ids: dict[str, list[int]] | None = None
        self.sid_len: int | None = None

        self.train_dataset: SequenceDataset | None = None
        self.val_dataset: SequenceDataset | None = None
        self.test_dataset: SequenceDataset | None = None
        self.stats: DatasetStats | None = None
        self.global_val_cutoff: int | None = None
        self.global_test_cutoff: int | None = None
        self._is_setup = False

    @property
    def num_users(self) -> int:
        return len(self.id2user)

    @property
    def num_items(self) -> int:
        return len(self.id2item) - 1

    @property
    def item_vocab_size(self) -> int:
        return len(self.id2item)

    def prepare_data(self) -> None:
        if not self.cfg.use_hf:
            if self.interactions_path is None:
                raise ValueError("interactions_path is required when use_hf=False")
            if not self.interactions_path.exists():
                raise FileNotFoundError(
                    f"interactions_path does not exist: {self.interactions_path}"
                )
        if self.semantic_id_path is not None and not self.semantic_id_path.exists():
            raise FileNotFoundError(f"semantic_id_path does not exist: {self.semantic_id_path}")

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return
        self.prepare_data()

        if self.semantic_id_path is not None:
            self.semantic_ids, self.sid_len = load_semantic_ids(
                self.semantic_id_path,
                offset_by_one=self.cfg.offset_semantic_ids_by_one,
            )

        allowed_items = None
        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            allowed_items = set(self.semantic_ids)

        user_events = load_yambda_events_by_user(
            interactions_path=self.interactions_path,
            use_hf=self.cfg.use_hf,
            dataset_type=self.cfg.dataset_type,
            dataset_size=self.cfg.dataset_size,
            event_file=self.cfg.event_file,
            hf_dataset_name=self.cfg.hf_dataset_name,
            streaming=self.cfg.streaming,
            max_rows=self.cfg.max_rows,
            keep_raw=self.cfg.keep_raw_interactions,
            allowed_items=allowed_items,
            organic_filter=self.cfg.organic_filter,
            min_played_ratio_pct=self.cfg.min_played_ratio_pct,
            allowed_event_types=self.cfg.allowed_event_types,
            event_value_map=self.cfg.event_value_map,
            default_rating=self.cfg.default_rating,
        )
      

        if self.cfg.deduplicate_consecutive:
            user_events = deduplicate_consecutive_items(user_events)

        user_events = apply_k_core_filter(
            user_events,
            min_user_interactions=self.cfg.min_user_interactions,
            min_item_interactions=self.cfg.min_item_interactions,
        )
        user_events = sort_user_events(user_events)
        num_interactions = count_interactions(user_events)

        split_cfg = SequenceSplitConfig(
            split_strategy=self.cfg.split_strategy,
            min_history_len=self.cfg.min_history_len,
            train_all_prefixes=self.cfg.train_all_prefixes,
            train_target_policy=self.cfg.train_target_policy,
            train_targets_per_user=self.cfg.train_targets_per_user,
            train_stride=self.cfg.train_stride,
            train_last_n=self.cfg.train_last_n,
            train_hybrid_recent_n=self.cfg.train_hybrid_recent_n,
            train_hybrid_random_n=self.cfg.train_hybrid_random_n,
            max_history_len=self.cfg.max_history_len,
            global_val_timestamp=self.cfg.global_val_timestamp,
            global_test_timestamp=self.cfg.global_test_timestamp,
            global_val_ratio=self.cfg.global_val_ratio,
            global_test_ratio=self.cfg.global_test_ratio,
            global_history_policy=self.cfg.global_history_policy,
            global_eval_target_mode=self.cfg.global_eval_target_mode,
            seed=self.cfg.seed,
        )
        splits = build_sequence_splits(
            user_events,
            get_user_id=self._get_or_add_user,
            get_item_id=self._get_or_add_item,
            cfg=split_cfg,
        )

        self.global_val_cutoff = splits.val_cutoff
        self.global_test_cutoff = splits.test_cutoff
        self.train_dataset = SequenceDataset(splits.train, split="train")
        self.val_dataset = SequenceDataset(splits.val, split="val")
        self.test_dataset = SequenceDataset(splits.test, split="test")
        self.stats = DatasetStats(
            num_users=self.num_users,
            num_items=self.num_items,
            num_train_examples=len(splits.train),
            num_val_examples=len(splits.val),
            num_test_examples=len(splits.test),
            num_interactions=num_interactions,
        )
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            collate_fn=self._make_collator(),
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return self._eval_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before test_dataloader().")
        return self._eval_dataloader(self.test_dataset)

    def _eval_dataloader(self, dataset: SequenceDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            collate_fn=self._make_collator(),
        )

    def _make_collator(self) -> SequenceBatchCollator:
        return SequenceBatchCollator(
            max_history_len=self.cfg.max_history_len,
            history_padding=self.cfg.history_padding,
            pad_item_id=self.cfg.pad_item_id,
            pad_token_id=self.cfg.pad_token_id,
            semantic_ids=self.semantic_ids,
            sid_len=self.sid_len,
            include_raw=self.cfg.include_raw_in_batch,
        )

    def _get_or_add_user(self, raw_user_id: str) -> int:
        if raw_user_id not in self.user2id:
            self.user2id[raw_user_id] = len(self.id2user)
            self.id2user.append(raw_user_id)
        return self.user2id[raw_user_id]

    def _get_or_add_item(self, raw_item_id: str) -> int:
        if raw_item_id not in self.item2id:
            self.item2id[raw_item_id] = len(self.id2item)
            self.id2item.append(raw_item_id)
        return self.item2id[raw_item_id]

    def raw_item_to_id(self, raw_item_id: str) -> int:
        return self.item2id[raw_item_id]

    def id_to_raw_item(self, item_id: int) -> str:
        return self.id2item[item_id]

    def summary(self) -> dict[str, Any]:
        if self.stats is None:
            raise RuntimeError("Call setup() before summary().")
        out = asdict(self.stats)
        out.update(
            {
                "dataset": "yambda",
                "dataset_type": self.cfg.dataset_type,
                "dataset_size": self.cfg.dataset_size,
                "event_file": self.cfg.event_file,
                "split_strategy": self.cfg.split_strategy,
                "global_val_cutoff": self.global_val_cutoff,
                "global_test_cutoff": self.global_test_cutoff,
                "sid_len": self.sid_len,
                "item_vocab_size": self.item_vocab_size,
            }
        )
        return out


def load_yambda_events_by_user(
    interactions_path: str | Path | None = None,
    *,
    use_hf: bool = False,
    dataset_type: YambdaDatasetType = "flat",
    dataset_size: YambdaDatasetSize = "50m",
    event_file: YambdaEventFile = "listens",
    hf_dataset_name: str = "yandex/yambda",
    streaming: bool = False,
    max_rows: int | None = None,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 1.0,
) -> dict[str, list[InteractionRecord]]:
    """Load Yambda interactions grouped by raw user ID."""
    if use_hf:
        records: Iterable[dict[str, Any]] = iter_hf_yambda_records(
            dataset_type=dataset_type,
            dataset_size=dataset_size,
            event_file=event_file,
            dataset_name=hf_dataset_name,
            streaming=streaming,
            max_rows=max_rows,
        )
    else:
        if interactions_path is None:
            raise ValueError("interactions_path is required when use_hf=False")
        records = iter_tabular_records(interactions_path, max_rows=max_rows)

    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
    order_idx = 0
    if dataset_type == "sequential":
        for record in records:
            events = normalize_yambda_sequential_record(
                record,
                keep_raw=keep_raw,
                order_idx_start=order_idx,
                organic_filter=organic_filter,
                min_played_ratio_pct=min_played_ratio_pct,
                allowed_event_types=allowed_event_types,
                event_value_map=event_value_map,
                default_rating=default_rating,
            )
            for event in events:
                order_idx += 1
                if allowed_items is not None and event.raw_item_id not in allowed_items:
                    continue
                grouped[event.raw_user_id].append(event)
    else:
        for record in records:
            event = normalize_yambda_flat_record(
                record,
                keep_raw=keep_raw,
                order_idx=order_idx,
                event_file=event_file,
                organic_filter=organic_filter,
                min_played_ratio_pct=min_played_ratio_pct,
                allowed_event_types=allowed_event_types,
                event_value_map=event_value_map,
                default_rating=default_rating,
            )
            events = [] if event is None else [event]

            for event in events:
                order_idx += 1
                if allowed_items is not None and event.raw_item_id not in allowed_items:
                    continue
                grouped[event.raw_user_id].append(event)

    return dict(grouped)


def normalize_yambda_flat_record(
    record: Mapping[str, Any],
    *,
    keep_raw: bool = False,
    order_idx: int = 0,
    event_file: YambdaEventFile = "listens",
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 1.0,
) -> InteractionRecord | None:
    uid = record.get("uid", record.get("user_id", record.get("raw_user_id")))
    item_id = record.get("item_id", record.get("track_id", record.get("raw_item_id")))
    timestamp = record.get("timestamp", record.get("ts", 0))
    if uid is None or item_id is None:
        return None

    event_type = str(record.get("event_type") or _event_type_from_file(event_file))
    if not _passes_event_filters(
        record,
        event_type=event_type,
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
    ):
        return None

    return InteractionRecord(
        raw_user_id=str(uid),
        raw_item_id=str(item_id),
        timestamp=int(timestamp),
        rating=_rating_from_yambda_record(
            record,
            event_type=event_type,
            event_value_map=event_value_map,
            default_rating=default_rating,
        ),
        verified_purchase=None,
        order_idx=order_idx,
        raw=dict(record) if keep_raw else {},
    )


def normalize_yambda_sequential_record(
    record: Mapping[str, Any],
    *,
    keep_raw: bool = False,
    order_idx_start: int = 0,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 1.0,
) -> list[InteractionRecord]:
    uid = record.get("uid", record.get("user_id", record.get("raw_user_id")))
    item_ids = _as_list(record.get("item_id"))
    timestamps = _as_list(record.get("timestamp"))
    if uid is None or not item_ids:
        return []
    if timestamps and len(timestamps) != len(item_ids):
        raise ValueError("Yambda sequential row has len(timestamps) != len(item_ids)")

    organic_values = _as_list(record.get("is_organic"))
    event_types = _as_list(record.get("event_type")) or _as_list(record.get("event_types"))
    played_ratios = _as_list(record.get("played_ratio_pct"))
    lengths = _as_list(record.get("track_length_seconds"))

    events: list[InteractionRecord] = []
    for idx, item_id in enumerate(item_ids):
        event_type = (
            str(event_types[idx])
            if idx < len(event_types) and event_types[idx] is not None
            else "listen"
        )
        one_record = {
            "uid": uid,
            "item_id": item_id,
            "timestamp": timestamps[idx] if idx < len(timestamps) else idx,
            "event_type": event_type,
        }
        if idx < len(organic_values):
            one_record["is_organic"] = organic_values[idx]
        if idx < len(played_ratios):
            one_record["played_ratio_pct"] = played_ratios[idx]
        if idx < len(lengths):
            one_record["track_length_seconds"] = lengths[idx]

        if not _passes_event_filters(
            one_record,
            event_type=event_type,
            organic_filter=organic_filter,
            min_played_ratio_pct=min_played_ratio_pct,
            allowed_event_types=allowed_event_types,
        ):
            continue

        events.append(
            InteractionRecord(
                raw_user_id=str(uid),
                raw_item_id=str(item_id),
                timestamp=int(one_record["timestamp"]),
                rating=_rating_from_yambda_record(
                    one_record,
                    event_type=event_type,
                    event_value_map=event_value_map,
                    default_rating=default_rating,
                ),
                verified_purchase=None,
                order_idx=order_idx_start + idx,
                raw=dict(one_record) if keep_raw else {},
            )
        )
    return events


def _event_type_from_file(event_file: str) -> str:
    return {
        "listens": "listen",
        "likes": "like",
        "dislikes": "dislike",
        "unlikes": "unlike",
        "undislikes": "undislike",
    }.get(event_file, "listen")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        out = tolist()
        return out if isinstance(out, list) else [out]
    return [value]


def _passes_event_filters(
    record: Mapping[str, Any],
    *,
    event_type: str,
    organic_filter: OrganicFilter,
    min_played_ratio_pct: float | None,
    allowed_event_types: tuple[str, ...] | None,
) -> bool:
    if allowed_event_types is not None and event_type not in set(allowed_event_types):
        return False

    if organic_filter != "any":
        raw = record.get("is_organic")
        if raw is None:
            return False
        is_organic = bool(int(raw)) if not isinstance(raw, bool) else raw
        if organic_filter == "organic" and not is_organic:
            return False
        if organic_filter == "recommendation" and is_organic:
            return False

    if min_played_ratio_pct is not None and event_type == "listen":
        value = record.get("played_ratio_pct")
        if value is None:
            return False
        try:
            if float(value) < float(min_played_ratio_pct):
                return False
        except (TypeError, ValueError):
            return False

    return True


def _rating_from_yambda_record(
    record: Mapping[str, Any],
    *,
    event_type: str,
    event_value_map: Mapping[str, float] | None,
    default_rating: float,
) -> float:
    if "rating" in record and record["rating"] is not None:
        try:
            return float(record["rating"])
        except (TypeError, ValueError):
            pass

    if event_type == "listen" and record.get("played_ratio_pct") is not None:
        try:
            # Yambda notes that played_ratio_pct can exceed 100 when users
            # rewind/replay. Cap the target value to avoid huge rewards.
            return min(float(record["played_ratio_pct"]), 100.0) / 100.0
        except (TypeError, ValueError):
            pass

    if event_value_map is not None and event_type in event_value_map:
        return float(event_value_map[event_type])
    return float(default_rating)
