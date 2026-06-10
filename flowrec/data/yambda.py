from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .collators import HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, InteractionRecord, SequenceDataset
from .preprocessors import (
    apply_k_core_filter,
    count_interactions,
    deduplicate_consecutive_items,
    load_semantic_ids,
    sort_user_events,
)
from .post_split_filtering import (
    ColdItemTrainSource,
    PostSplitColdItemFilterStats,
    disabled_post_split_filter_stats,
    filter_splits_to_train_item_catalog,
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


def _read_local_tabular_frame(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read one local Yambda table with pandas; unsupported formats raise."""
    path = Path(path)
    if _is_parquet(path):
        frame = pd.read_parquet(path)
        return frame.head(max_rows) if max_rows is not None else frame
    if _is_csv(path):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=max_rows)

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        frame = pd.read_json(path, lines=True, compression="infer")
        return frame.head(max_rows) if max_rows is not None else frame

    raise ValueError(f"Unsupported Yambda table format: {path}")


def iter_tabular_records(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield records from the same strict pandas reader used by the datamodule."""
    frame = _read_local_tabular_frame(path, max_rows=max_rows)
    yield from frame.to_dict(orient="records")


def _read_hf_yambda_frame(
    *,
    dataset_type: YambdaDatasetType = "flat",
    dataset_size: YambdaDatasetSize = "50m",
    event_file: YambdaEventFile = "listens",
    dataset_name: str = "yandex/yambda",
    streaming: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load Yambda from Hugging Face into a pandas frame.

    Streaming is intentionally rejected because the strict path is vectorized over
    a DataFrame rather than row-by-row Python iteration.
    """
    if streaming:
        raise ValueError("streaming=True is not supported by the strict vectorized Yambda loader")

    from datasets import DatasetDict, load_dataset

    data = load_dataset(
        dataset_name,
        data_dir=f"{dataset_type}/{dataset_size}",
        data_files=f"{event_file}.parquet",
        streaming=False,
    )
    ds = data["train"] if isinstance(data, DatasetDict) else data
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds.to_pandas()


def iter_hf_yambda_records(
    *,
    dataset_type: YambdaDatasetType = "flat",
    dataset_size: YambdaDatasetSize = "50m",
    event_file: YambdaEventFile = "listens",
    dataset_name: str = "yandex/yambda",
    streaming: bool = False,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Hugging Face Yambda records through the strict DataFrame loader."""
    frame = _read_hf_yambda_frame(
        dataset_type=dataset_type,
        dataset_size=dataset_size,
        event_file=event_file,
        dataset_name=dataset_name,
        streaming=streaming,
        max_rows=max_rows,
    )
    yield from frame.to_dict(orient="records")


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
    train_drop_last: bool = False

    # IDs.
    pad_item_id: int = 0
    pad_token_id: int = 0
    offset_semantic_ids_by_one: bool = True
    drop_items_missing_semantic_id: bool = True
    user_hash_mod: int | None = None

    # Post-split cold-item filtering.
    filter_cold_items_after_split: bool = True
    cold_item_train_source: ColdItemTrainSource = "observed"

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
        self.post_split_filter_stats: PostSplitColdItemFilterStats | None = None
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

        if self.cfg.filter_cold_items_after_split:
            filter_result = filter_splits_to_train_item_catalog(
                splits,
                old_id2item=self.id2item,
                train_item_source=self.cfg.cold_item_train_source,
                min_history_len=self.cfg.min_history_len,
            )
            splits = filter_result.splits
            self.item2id = filter_result.item2id
            self.id2item = filter_result.id2item
            if self.semantic_ids is not None:
                self.semantic_ids = {
                    raw_item: tokens
                    for raw_item, tokens in self.semantic_ids.items()
                    if raw_item in self.item2id
                }
            self.post_split_filter_stats = filter_result.stats
        else:
            self.post_split_filter_stats = disabled_post_split_filter_stats(
                splits=splits,
                num_items=self.num_items,
                train_item_source=self.cfg.cold_item_train_source,
                min_history_len=self.cfg.min_history_len,
            )

        self.global_val_cutoff = splits.val_cutoff
        self.global_test_cutoff = splits.test_cutoff
        self.train_dataset = SequenceDataset(splits.train, split="train")
        self.val_dataset = SequenceDataset(splits.val, split="val")
        self.test_dataset = SequenceDataset(splits.test, split="test")
        filter_stats = self.post_split_filter_stats
        self.stats = DatasetStats(
            num_users=self.num_users,
            num_items=self.num_items,
            num_train_examples=len(splits.train),
            num_val_examples=len(splits.val),
            num_test_examples=len(splits.test),
            num_interactions=num_interactions,
            post_filter_cold_items=filter_stats.enabled if filter_stats is not None else False,
            cold_item_reference=filter_stats.train_item_source if filter_stats is not None else None,
            num_items_before_cold_filter=filter_stats.num_items_before if filter_stats is not None else self.num_items,
            num_cold_items_removed=filter_stats.num_items_removed if filter_stats is not None else 0,
            num_train_examples_dropped_cold=filter_stats.dropped_train_examples if filter_stats is not None else 0,
            num_val_examples_dropped_cold=filter_stats.dropped_val_examples if filter_stats is not None else 0,
            num_test_examples_dropped_cold=filter_stats.dropped_test_examples if filter_stats is not None else 0,
            num_train_history_items_removed_cold=filter_stats.removed_train_history_items if filter_stats is not None else 0,
            num_val_history_items_removed_cold=filter_stats.removed_val_history_items if filter_stats is not None else 0,
            num_test_history_items_removed_cold=filter_stats.removed_test_history_items if filter_stats is not None else 0,
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
            drop_last=self.cfg.train_drop_last,
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
            user_hash_mod=self.cfg.user_hash_mod,
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

    def add_semantic_ids(
        self,
        semantic_ids: Mapping[str, list[int]] | str | Path,
        *,
        offset_by_one: bool | None = None,
        filter_to_catalog: bool = True,
    ) -> None:
        """Attach semantic IDs after setup has built the item catalog."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before add_semantic_ids().")

        should_offset = (
            self.cfg.offset_semantic_ids_by_one if offset_by_one is None else offset_by_one
        )
        loaded_path: Path | None = None
        if isinstance(semantic_ids, str | Path):
            loaded_semantic_ids, loaded_sid_len = load_semantic_ids(
                semantic_ids,
                offset_by_one=should_offset,
            )
            loaded_path = Path(semantic_ids)
        else:
            loaded_semantic_ids = {
                str(raw_item): [
                    int(token) + 1 if should_offset else int(token)
                    for token in tokens
                ]
                for raw_item, tokens in semantic_ids.items()
            }
            loaded_sid_len = max(
                (len(tokens) for tokens in loaded_semantic_ids.values()),
                default=0,
            )

        if filter_to_catalog:
            loaded_semantic_ids = {
                raw_item: tokens
                for raw_item, tokens in loaded_semantic_ids.items()
                if raw_item in self.item2id
            }
            loaded_sid_len = max(
                (len(tokens) for tokens in loaded_semantic_ids.values()),
                default=0,
            )

        if not loaded_sid_len:
            raise ValueError("No semantic IDs matched the initialized item catalog.")

        self.semantic_ids = loaded_semantic_ids
        self.sid_len = loaded_sid_len
        if loaded_path is not None:
            self.semantic_id_path = loaded_path

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
                "post_split_filter": self.post_split_filter_stats.to_dict()
                if self.post_split_filter_stats is not None
                else None,
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
    """Load Yambda interactions with one strict vectorized implementation."""
    if use_hf:
        frame = _read_hf_yambda_frame(
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
        frame = _read_local_tabular_frame(interactions_path, max_rows=max_rows)

    if dataset_type == "sequential":
        return _load_yambda_sequential_events_by_user_frame(
            frame,
            keep_raw=keep_raw,
            allowed_items=allowed_items,
            organic_filter=organic_filter,
            min_played_ratio_pct=min_played_ratio_pct,
            allowed_event_types=allowed_event_types,
            event_value_map=event_value_map,
            default_rating=default_rating,
        )

    return _load_yambda_flat_events_by_user_frame(
        frame,
        keep_raw=keep_raw,
        allowed_items=allowed_items,
        event_file=event_file,
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
        event_value_map=event_value_map,
        default_rating=default_rating,
    )


def _first_frame_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out
def _bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y"])


def _events_frame_to_user_events(frame: pd.DataFrame, *, keep_raw: bool) -> dict[str, list[InteractionRecord]]:
    if frame.empty:
        return {}
    raw_records = frame["raw"].tolist() if keep_raw and "raw" in frame.columns else [{}] * len(frame)
    events = [
        InteractionRecord(
            raw_user_id=str(row.raw_user_id),
            raw_item_id=str(row.raw_item_id),
            timestamp=int(row.timestamp),
            rating=float(row.rating),
            verified_purchase=None,
            order_idx=int(row.order_idx),
            raw=raw_records[pos],
        )
        for pos, row in enumerate(frame.drop(columns=["raw"], errors="ignore").itertuples(index=False))
    ]
    tmp = pd.DataFrame({"raw_user_id": frame["raw_user_id"].astype("string").to_numpy(), "event": events})
    return {
        str(raw_user_id): list(items)
        for raw_user_id, items in tmp.groupby("raw_user_id", sort=False)["event"].agg(list).items()
    }


def _load_yambda_flat_events_by_user_frame(
    frame: pd.DataFrame,
    *,
    keep_raw: bool,
    allowed_items: set[str] | None,
    event_file: YambdaEventFile,
    organic_filter: OrganicFilter,
    min_played_ratio_pct: float | None,
    allowed_event_types: tuple[str, ...] | None,
    event_value_map: Mapping[str, float] | None,
    default_rating: float,
) -> dict[str, list[InteractionRecord]]:
    if frame.empty:
        return {}

    raw_user = _first_frame_column(frame, "uid", "user_id", "raw_user_id")
    raw_item = _first_frame_column(frame, "item_id", "track_id", "raw_item_id")
    timestamp = pd.to_numeric(_first_frame_column(frame, "timestamp", "ts"), errors="coerce").fillna(0).astype("int64")
    event_type = _first_frame_column(frame, "event_type").fillna(_event_type_from_file(event_file)).astype("string")

    mask = raw_user.notna() & raw_item.notna()
    if allowed_items is not None:
        mask &= raw_item.astype("string").isin(allowed_items)
    if allowed_event_types is not None:
        mask &= event_type.isin(set(allowed_event_types))
    if organic_filter != "any":
        organic = _first_frame_column(frame, "is_organic")
        organic_bool = _bool_series(organic)
        mask &= organic.notna()
        if organic_filter == "organic":
            mask &= organic_bool
        elif organic_filter == "recommendation":
            mask &= ~organic_bool
    if min_played_ratio_pct is not None:
        played = pd.to_numeric(_first_frame_column(frame, "played_ratio_pct", "played_ratio"), errors="coerce")
        listen_mask = event_type.eq("listen")
        mask &= (~listen_mask) | (played.notna() & (played >= float(min_played_ratio_pct)))

    if not mask.any():
        return {}

    played = pd.to_numeric(_first_frame_column(frame, "played_ratio_pct", "played_ratio"), errors="coerce")
    rating_col = pd.to_numeric(_first_frame_column(frame, "rating"), errors="coerce")
    mapped_event_rating = event_type.map(dict(event_value_map or {})).astype("float64")
    listen_rating = played.clip(upper=100.0) / 100.0
    rating = rating_col.where(
        rating_col.notna(),
        listen_rating.where(event_type.eq("listen") & played.notna(), mapped_event_rating),
    ).fillna(float(default_rating))

    filtered = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "order_idx": np.arange(len(frame), dtype=np.int64)[mask.to_numpy()],
        }
    )
    if keep_raw:
        filtered["raw"] = frame.loc[mask].to_dict(orient="records")
    return _events_frame_to_user_events(filtered, keep_raw=keep_raw)


def _align_list(values: Any, n: int, *, default: Any = None) -> list[Any]:
    seq = _as_list(values)
    if len(seq) == n:
        return seq
    if len(seq) == 0:
        return list(range(n)) if default == "range" else [default] * n
    if len(seq) == 1 and n > 1:
        return seq * n
    raise ValueError(f"Yambda sequential list length mismatch: expected {n}, got {len(seq)}")


def _load_yambda_sequential_events_by_user_frame(
    frame: pd.DataFrame,
    *,
    keep_raw: bool,
    allowed_items: set[str] | None,
    organic_filter: OrganicFilter,
    min_played_ratio_pct: float | None,
    allowed_event_types: tuple[str, ...] | None,
    event_value_map: Mapping[str, float] | None,
    default_rating: float,
) -> dict[str, list[InteractionRecord]]:
    if frame.empty:
        return {}

    base = pd.DataFrame(
        {
            "raw_user_id": _first_frame_column(frame, "uid", "user_id", "raw_user_id"),
            "raw_item_id": _first_frame_column(frame, "item_ids", "item_id", "track_ids", "track_id").map(_as_list),
            "timestamp_raw": _first_frame_column(frame, "timestamps", "timestamp", "ts"),
            "event_type_raw": _first_frame_column(frame, "event_types", "event_type"),
            "organic_raw": _first_frame_column(frame, "is_organic"),
            "played_raw": _first_frame_column(frame, "played_ratio_pct", "played_ratio_pcts", "played_ratios", "played_ratio"),
            "rating_raw": _first_frame_column(frame, "rating", "ratings"),
        }
    )
    base = base.loc[base["raw_user_id"].notna() & base["raw_item_id"].str.len().gt(0)].copy()
    if base.empty:
        return {}

    seq_len = base["raw_item_id"].str.len()
    base["timestamp"] = [
        _align_list(values, int(n), default="range")
        for values, n in zip(base["timestamp_raw"], seq_len, strict=False)
    ]
    base["event_type"] = [
        _align_list(values, int(n), default="listen")
        for values, n in zip(base["event_type_raw"], seq_len, strict=False)
    ]
    base["is_organic"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["organic_raw"], seq_len, strict=False)
    ]
    base["played_ratio_pct"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["played_raw"], seq_len, strict=False)
    ]
    base["rating"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["rating_raw"], seq_len, strict=False)
    ]

    exploded = base[["raw_user_id", "raw_item_id", "timestamp", "event_type", "is_organic", "played_ratio_pct", "rating"]].explode(
        ["raw_item_id", "timestamp", "event_type", "is_organic", "played_ratio_pct", "rating"],
        ignore_index=True,
    )
    return _load_yambda_flat_events_by_user_frame(
        exploded.rename(columns={"played_ratio_pct": "played_ratio_pct"}),
        keep_raw=keep_raw,
        allowed_items=allowed_items,
        event_file="listens",
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
        event_value_map=event_value_map,
        default_rating=default_rating,
    )


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
    uid = _first_present(record, "uid", "user_id", "raw_user_id")
    item_id = _first_present(record, "item_id", "track_id", "raw_item_id")
    timestamp = _first_present(record, "timestamp", "ts")
    if uid is None or item_id is None:
        return None
    if timestamp is None:
        timestamp = 0

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
    uid = _first_present(record, "uid", "user_id", "raw_user_id")
    item_ids = _as_list(_first_present(record, "item_ids", "item_id", "track_ids", "track_id"))
    timestamps = _as_list(_first_present(record, "timestamps", "timestamp", "ts"))
    if uid is None or not item_ids:
        return []
    if timestamps and len(timestamps) != len(item_ids):
        raise ValueError("Yambda sequential row has len(timestamps) != len(item_ids)")

    organic_values = _as_list(record.get("is_organic"))
    event_types = _as_list(record.get("event_types")) or _as_list(record.get("event_type"))
    played_ratios = _as_list(
        _first_present(record, "played_ratio_pct", "played_ratio_pcts", "played_ratios")
    )
    lengths = _as_list(
        _first_present(record, "track_length_seconds", "track_lengths_seconds", "track_lengths")
    )

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


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


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
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or float(numeric) < float(min_played_ratio_pct):
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
        rating = pd.to_numeric(pd.Series([record["rating"]]), errors="coerce").iloc[0]
        if not pd.isna(rating):
            return float(rating)

    if event_type == "listen" and record.get("played_ratio_pct") is not None:
        played = pd.to_numeric(pd.Series([record["played_ratio_pct"]]), errors="coerce").iloc[0]
        if not pd.isna(played):
            return min(float(played), 100.0) / 100.0

    if event_value_map is not None and event_type in event_value_map:
        return float(event_value_map[event_type])
    return float(default_rating)
