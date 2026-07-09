from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .collators import HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, SequenceDataset, history_length_stats
from .preprocessors import (
    apply_k_core_filter_frame,
    count_interactions,
    load_semantic_ids,
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
    factorize_interaction_ids,
)

YambdaDatasetType = Literal["flat", "sequential"]
YambdaDatasetSize = Literal["50m", "500m", "5b"]
YambdaEventFile = Literal[
    "listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event"
]
OrganicFilter = Literal["any", "organic", "recommendation"]

_CANONICAL_INTERACTION_COLUMNS = [
    "raw_user_id",
    "raw_item_id",
    "timestamp",
    "rating",
    "verified_purchase",
    "order_idx",
]
_YAMBDA_TIMESTAMP_COLUMNS = ("timestamp", "ts")
_YAMBDA_SEQUENCE_TIMESTAMP_COLUMNS = ("timestamps", "timestamp", "ts")


def _suffixes(path: str | Path) -> str:
    return "".join(Path(path).suffixes).lower()


def _is_parquet(path: str | Path) -> bool:
    return _suffixes(path).endswith(".parquet")


def _is_csv(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".csv", ".tsv"}


def _is_jsonl(path: str | Path) -> bool:
    suffixes = _suffixes(path)
    return suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz")


def _empty_interaction_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in _CANONICAL_INTERACTION_COLUMNS})


def _read_local_tabular_frame(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read one local Yambda table into a pandas DataFrame.

    Parquet is the intended fast path. CSV/TSV/JSONL are kept as strict
    compatibility inputs for smoke tests and one-off conversions.
    """
    path = Path(path)
    if _is_parquet(path):
        frame = pd.read_parquet(path)
        return frame.head(max_rows) if max_rows is not None else frame
    if _is_csv(path):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=max_rows)
    if _is_jsonl(path):
        frame = pd.read_json(path, lines=True, compression="infer")
        return frame.head(max_rows) if max_rows is not None else frame
    raise ValueError(f"Unsupported Yambda table format: {path}")


def iter_tabular_records(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw records through the same strict pandas reader as the loader."""
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
    """Load Yambda from Hugging Face into a pandas DataFrame.

    Streaming is rejected intentionally: downstream normalization is vectorized
    over a DataFrame rather than Python row iteration.
    """
    if streaming:
        raise ValueError("streaming=True is not supported by the vectorized Yambda loader")

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
    hf_start_date: str | None = None,
    hf_end_date: str | None = None,
    hf_recent_months: int | None = 2,
) -> Iterator[dict[str, Any]]:
    """Yield Hugging Face Yambda records through the DataFrame loader."""
    frame = _read_hf_yambda_frame(
        dataset_type=dataset_type,
        dataset_size=dataset_size,
        event_file=event_file,
        dataset_name=dataset_name,
        streaming=streaming,
        max_rows=max_rows,
    )
    frame = filter_yambda_frame_by_date(
        frame,
        dataset_type=dataset_type,
        start_date=hf_start_date,
        end_date=hf_end_date,
        recent_months=hf_recent_months,
    )
    yield from frame.to_dict(orient="records")


@dataclass(frozen=True, slots=True)
class YambdaDataConfig:
    """Configuration for the DataFrame-based Yambda sequential pipeline.

    Local input is expected to be a Parquet file. The flat format accepts columns
    such as uid/user_id, item_id/track_id, timestamp/ts, is_organic,
    played_ratio_pct, played_ratio, event_type, and optional played/duration
    aliases. The sequential format accepts list columns such as item_ids,
    timestamps, event_types, is_organic, played_ratio_pcts, and played_ratios;
    these rows are exploded once into the canonical interaction DataFrame.

    Yambda does not contain Amazon-style ratings or purchases. The canonical
    fields are therefore derived for compatibility with the shared collator:
      - rating: percentage of the track listened to, on a 0..100 scale.
      - verified_purchase: always True for every retained interaction.
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
    hf_start_date: str | None = None
    hf_end_date: str | None = None
    hf_recent_months: int | None = 2

    # Filtering / normalization.
    min_user_interactions: int = 5
    min_item_interactions: int = 0
    deduplicate_consecutive: bool = False
    keep_raw_interactions: bool = False
    organic_filter: OrganicFilter = "any"
    min_played_ratio_pct: float | None = None
    allowed_event_types: tuple[str, ...] | None = None
    # Deprecated compatibility knobs. Yambda has no rating concept, so the
    # canonical rating is always the listened percentage, not an event weight.
    event_value_map: dict[str, float] = field(
        default_factory=lambda: {
            "listen": 1.0,
            "like": 2.0,
            "dislike": -1.0,
            "unlike": 0.0,
            "undislike": 0.0,
        }
    )
    default_rating: float = 0.0

    # Splitting.
    split_strategy: SplitStrategy = "leave_one_out"
    min_history_len: int = 1
    train_all_prefixes: bool = True
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

    The module is DataFrame-first: loading returns a canonical interaction
    DataFrame, filtering and ID assignment add columns, split construction
    returns split DataFrames, and SequenceDataset keeps the public dict batch
    format unchanged.
    """

    def __init__(self, cfg: YambdaDataConfig) -> None:
        self.cfg = cfg
        self.interactions_path = Path(cfg.interactions_path) if cfg.interactions_path else None
        self.semantic_id_path = Path(cfg.semantic_id_path) if cfg.semantic_id_path else None

        self.user2id: dict[str, int] = {}
        self.id2user: list[str] = []
        self.item2id: dict[str, int] = {}
        self.id2item: list[str] = ["<PAD>"]

        self.interactions: pd.DataFrame = pd.DataFrame()
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
                raise FileNotFoundError(f"interactions_path does not exist: {self.interactions_path}")
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

        interactions = load_yambda_interactions_frame(
            interactions_path=self.interactions_path,
            use_hf=self.cfg.use_hf,
            dataset_type=self.cfg.dataset_type,
            dataset_size=self.cfg.dataset_size,
            event_file=self.cfg.event_file,
            hf_dataset_name=self.cfg.hf_dataset_name,
            streaming=self.cfg.streaming,
            max_rows=self.cfg.max_rows,
            hf_start_date=self.cfg.hf_start_date,
            hf_end_date=self.cfg.hf_end_date,
            hf_recent_months=self.cfg.hf_recent_months,
            keep_raw=self.cfg.keep_raw_interactions,
            allowed_items=allowed_items,
            organic_filter=self.cfg.organic_filter,
            min_played_ratio_pct=self.cfg.min_played_ratio_pct,
            allowed_event_types=self.cfg.allowed_event_types,
            event_value_map=self.cfg.event_value_map,
            default_rating=self.cfg.default_rating,
        )

        interactions = apply_k_core_filter_frame(
            interactions,
            min_user_interactions=self.cfg.min_user_interactions,
            min_item_interactions=self.cfg.min_item_interactions,
            drop_consecutive_repeats=self.cfg.deduplicate_consecutive,
        )
        num_interactions = count_interactions(interactions)

        interactions, self.user2id, self.id2user, self.item2id, self.id2item = factorize_interaction_ids(
            interactions,
            pad_raw_item="<PAD>",
        )
        self.interactions = interactions

        split_cfg = _sequence_split_config_from_yambda(self.cfg)
        splits = build_sequence_splits(interactions, cfg=split_cfg)

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
            self.post_split_filter_stats = filter_result.stats
        else:
            self.post_split_filter_stats = disabled_post_split_filter_stats(
                splits=splits,
                num_items=self.num_items,
                train_item_source=self.cfg.cold_item_train_source,
                min_history_len=self.cfg.min_history_len,
            )

        if self.semantic_ids is not None:
            self.semantic_ids = {
                raw_item: tokens for raw_item, tokens in self.semantic_ids.items() if raw_item in self.item2id
            }
            self.sid_len = max((len(tokens) for tokens in self.semantic_ids.values()), default=0) or None

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
            **history_length_stats(splits.train, splits.val, splits.test),
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

    def raw_item_to_id(self, raw_item_id: str) -> int:
        return self.item2id[raw_item_id]

    def id_to_raw_item(self, item_id: int) -> str:
        return self.id2item[item_id]

    def raw_user_to_id(self, raw_user_id: str) -> int:
        return self.user2id[raw_user_id]

    def id_to_raw_user(self, user_id: int) -> str:
        return self.id2user[user_id]

    def item_ids_in_data(self) -> set[str]:
        """Raw item IDs that remain in the dataset after all preprocessing."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before item_ids_in_data().")
        return set(self.item2id.keys())

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

        should_offset = self.cfg.offset_semantic_ids_by_one if offset_by_one is None else offset_by_one
        loaded_path: Path | None = None
        if isinstance(semantic_ids, (str, Path)):
            loaded_semantic_ids, loaded_sid_len = load_semantic_ids(
                semantic_ids,
                offset_by_one=should_offset,
            )
            loaded_path = Path(semantic_ids)
        else:
            loaded_semantic_ids = {
                str(raw_item): [int(token) + 1 if should_offset else int(token) for token in tokens]
                for raw_item, tokens in semantic_ids.items()
            }
            loaded_sid_len = max((len(tokens) for tokens in loaded_semantic_ids.values()), default=0)

        if filter_to_catalog:
            loaded_semantic_ids = {
                raw_item: tokens for raw_item, tokens in loaded_semantic_ids.items() if raw_item in self.item2id
            }
            loaded_sid_len = max((len(tokens) for tokens in loaded_semantic_ids.values()), default=0)

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
                "hf_start_date": self.cfg.hf_start_date,
                "hf_end_date": self.cfg.hf_end_date,
                "hf_recent_months": self.cfg.hf_recent_months,
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


def _sequence_split_config_from_yambda(cfg: YambdaDataConfig) -> SequenceSplitConfig:
    return SequenceSplitConfig(
        split_strategy=cfg.split_strategy,
        min_history_len=cfg.min_history_len,
        train_all_prefixes=cfg.train_all_prefixes,
        train_target_policy=cfg.train_target_policy,
        train_targets_per_user=cfg.train_targets_per_user,
        train_stride=cfg.train_stride,
        train_last_n=cfg.train_last_n,
        train_hybrid_recent_n=cfg.train_hybrid_recent_n,
        train_hybrid_random_n=cfg.train_hybrid_random_n,
        max_history_len=cfg.max_history_len,
        global_val_timestamp=cfg.global_val_timestamp,
        global_test_timestamp=cfg.global_test_timestamp,
        global_val_ratio=cfg.global_val_ratio,
        global_test_ratio=cfg.global_test_ratio,
        global_history_policy=cfg.global_history_policy,
        global_eval_target_mode=cfg.global_eval_target_mode,
        seed=cfg.seed,
    )


def load_yambda_interactions_frame(
    interactions_path: str | Path | None = None,
    *,
    use_hf: bool = False,
    dataset_type: YambdaDatasetType = "flat",
    dataset_size: YambdaDatasetSize = "50m",
    event_file: YambdaEventFile = "listens",
    hf_dataset_name: str = "yandex/yambda",
    streaming: bool = False,
    max_rows: int | None = None,
    hf_start_date: str | None = None,
    hf_end_date: str | None = None,
    hf_recent_months: int | None = 2,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 0.0,
) -> pd.DataFrame:
    """Load and normalize Yambda interactions as a canonical DataFrame."""
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

    frame = filter_yambda_frame_by_date(
        frame,
        dataset_type=dataset_type,
        start_date=hf_start_date,
        end_date=hf_end_date,
        recent_months=hf_recent_months,
    )

    if dataset_type == "sequential":
        return normalize_yambda_sequential_frame(
            frame,
            keep_raw=keep_raw,
            allowed_items=allowed_items,
            organic_filter=organic_filter,
            min_played_ratio_pct=min_played_ratio_pct,
            allowed_event_types=allowed_event_types,
            event_value_map=event_value_map,
            default_rating=default_rating,
        )

    return normalize_yambda_flat_frame(
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


# Compatibility alias with the old public loader name. It now returns a DataFrame
# instead of grouped per-row dataclass objects.
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
    hf_start_date: str | None = None,
    hf_end_date: str | None = None,
    hf_recent_months: int | None = 2,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 0.0,
) -> pd.DataFrame:
    return load_yambda_interactions_frame(
        interactions_path=interactions_path,
        use_hf=use_hf,
        dataset_type=dataset_type,
        dataset_size=dataset_size,
        event_file=event_file,
        hf_dataset_name=hf_dataset_name,
        streaming=streaming,
        max_rows=max_rows,
        hf_start_date=hf_start_date,
        hf_end_date=hf_end_date,
        hf_recent_months=hf_recent_months,
        keep_raw=keep_raw,
        allowed_items=allowed_items,
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
        event_value_map=event_value_map,
        default_rating=default_rating,
    )


def filter_by_date(
    data: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Filter rows by a DD/MM/YYYY date range using Unix-second timestamps.

    The returned frame keeps the original timestamp column for downstream
    preprocessing and uses the converted datetimes as its index.
    """
    return _filter_scalar_timestamp_frame_by_date(
        data,
        start_date=start_date,
        end_date=end_date,
        recent_months=None,
        timestamp_column="timestamp",
        reset_index=False,
    )


def filter_yambda_frame_by_date(
    frame: pd.DataFrame,
    *,
    dataset_type: YambdaDatasetType,
    start_date: str | None = None,
    end_date: str | None = None,
    recent_months: int | None = 2,
) -> pd.DataFrame:
    """Filter a raw Yambda frame before normalization.

    If explicit dates are not provided, ``recent_months`` keeps the latest
    window relative to the maximum timestamp present in the loaded frame.
    """
    if frame.empty or (start_date is None and end_date is None and recent_months is None):
        return frame.reset_index(drop=True).copy()
    if recent_months is not None and recent_months < 0:
        raise ValueError("recent_months must be >= 0 or None")

    if dataset_type == "sequential":
        return _filter_yambda_sequential_frame_by_date(
            frame,
            start_date=start_date,
            end_date=end_date,
            recent_months=recent_months,
        )

    timestamp_column = _first_existing_column(frame, *_YAMBDA_TIMESTAMP_COLUMNS)
    if timestamp_column is None:
        raise ValueError("Yambda date filtering requires a timestamp or ts column.")
    return _filter_scalar_timestamp_frame_by_date(
        frame,
        start_date=start_date,
        end_date=end_date,
        recent_months=recent_months,
        timestamp_column=timestamp_column,
        reset_index=True,
    )


def _filter_scalar_timestamp_frame_by_date(
    data: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    recent_months: int | None,
    timestamp_column: str,
    reset_index: bool,
) -> pd.DataFrame:
    if timestamp_column not in data.columns:
        raise ValueError(f"Yambda date filtering requires column {timestamp_column!r}.")

    out = data.copy(deep=True)
    datetimes = _timestamp_values_to_datetime(out[timestamp_column])
    start_datetime, end_datetime = _resolve_date_bounds(
        datetimes,
        start_date=start_date,
        end_date=end_date,
        recent_months=recent_months,
    )
    mask = datetimes.notna() & datetimes.ge(start_datetime) & datetimes.le(end_datetime)
    filtered = out.loc[mask].copy()
    filtered.index = datetimes.loc[mask]
    if reset_index:
        filtered = filtered.reset_index(drop=True)
    return filtered


def _filter_yambda_sequential_frame_by_date(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    recent_months: int | None,
) -> pd.DataFrame:
    timestamp_column = _first_existing_column(frame, *_YAMBDA_SEQUENCE_TIMESTAMP_COLUMNS)
    if timestamp_column is None:
        raise ValueError("Yambda sequential date filtering requires timestamps, timestamp, or ts.")

    if not frame[timestamp_column].map(_is_sequence_value).any():
        return _filter_scalar_timestamp_frame_by_date(
            frame,
            start_date=start_date,
            end_date=end_date,
            recent_months=recent_months,
            timestamp_column=timestamp_column,
            reset_index=True,
        )

    all_timestamps = [
        value
        for values in frame[timestamp_column]
        for value in _as_list(values)
    ]
    datetimes = _timestamp_values_to_datetime(pd.Series(all_timestamps, dtype="object"))
    start_datetime, end_datetime = _resolve_date_bounds(
        datetimes,
        start_date=start_date,
        end_date=end_date,
        recent_months=recent_months,
    )

    item_column = _first_existing_column(frame, "item_ids", "item_id", "track_ids", "track_id")
    filtered_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        n = len(_as_list(row_dict[item_column])) if item_column is not None else len(_as_list(row_dict[timestamp_column]))
        if n == 0:
            continue

        timestamps = _align_list(row_dict[timestamp_column], n, default=None)
        row_datetimes = _timestamp_values_to_datetime(pd.Series(timestamps, dtype="object"))
        keep = row_datetimes.notna() & row_datetimes.ge(start_datetime) & row_datetimes.le(end_datetime)
        keep_values = keep.to_numpy(dtype=bool)
        if not bool(keep_values.any()):
            continue

        for column, value in list(row_dict.items()):
            if not _is_sequence_value(value):
                continue
            values = _as_list(value)
            if len(values) == n:
                row_dict[column] = [x for x, should_keep in zip(values, keep_values, strict=False) if should_keep]
        filtered_rows.append(row_dict)

    return pd.DataFrame(filtered_rows, columns=frame.columns).reset_index(drop=True)


def _resolve_date_bounds(
    datetimes: pd.Series,
    *,
    start_date: str | None,
    end_date: str | None,
    recent_months: int | None,
):
    valid = datetimes.dropna()
    if valid.empty:
        return pd.Timestamp.max, pd.Timestamp.min

    end_datetime = pd.to_datetime(end_date, dayfirst=True) if end_date else valid.max()
    if end_date:
        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)

    if start_date:
        start_datetime = pd.to_datetime(start_date, dayfirst=True)
    elif recent_months is not None:
        start_datetime = end_datetime - pd.DateOffset(months=int(recent_months))
    else:
        start_datetime = valid.min()
    return start_datetime, end_datetime


def _timestamp_values_to_datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(numeric, unit="s", errors="coerce")


def _first_existing_column(frame: pd.DataFrame, *columns: str) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _is_sequence_value(value: Any) -> bool:
    if np.isscalar(value):
        return False
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return True
    tolist = getattr(value, "tolist", None)
    return callable(tolist) and not isinstance(value, (str, bytes))


def _first_frame_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out


def _bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y"])


_PLAYED_PERCENT_COLUMNS = (
    "played_ratio_pct",
    "played_ratio_pcts",
    "played_pct",
    "played_percent",
    "listened_pct",
    "listened_percent",
    "listened_percentage",
    "percent_listened",
    "completion_pct",
    "completion_percent",
)
_PLAYED_RATIO_COLUMNS = (
    "played_ratio",
    "played_ratios",
    "listen_ratio",
    "listened_ratio",
    "completion_ratio",
)
_PLAYED_DURATION_COLUMNS = (
    "played_duration_ms",
    "played_ms",
    "listened_duration_ms",
    "listened_ms",
    "ms_played",
    "played_duration",
    "played_seconds",
    "seconds_played",
)
_TRACK_DURATION_COLUMNS = (
    "track_duration_ms",
    "duration_ms",
    "track_length_ms",
    "length_ms",
    "track_duration",
    "duration",
    "duration_seconds",
    "track_length",
)


def _first_numeric_frame_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    return pd.to_numeric(_first_frame_column(frame, *columns), errors="coerce")


def _ratio_to_percentage(ratio: pd.Series) -> pd.Series:
    """Convert ratio-like values to percentages while tolerating pct-like data.

    Yambda exports and local preprocessing variants may use either [0, 1]
    ratios or [0, 100] percentages. Values with absolute magnitude <= 1 are
    interpreted as fractions; larger values are treated as already-percent.
    """
    ratio = pd.to_numeric(ratio, errors="coerce")
    return ratio.where(ratio.abs().gt(1.0), ratio * 100.0)


def _listened_percentage_series(frame: pd.DataFrame) -> pd.Series:
    """Return the listened percentage on a 0..100 scale for every row."""
    pct = _first_numeric_frame_column(frame, *_PLAYED_PERCENT_COLUMNS)

    ratio = _first_numeric_frame_column(frame, *_PLAYED_RATIO_COLUMNS)
    pct = pct.where(pct.notna(), _ratio_to_percentage(ratio))

    played_duration = _first_numeric_frame_column(frame, *_PLAYED_DURATION_COLUMNS)
    track_duration = _first_numeric_frame_column(frame, *_TRACK_DURATION_COLUMNS)
    duration_pct = (played_duration / track_duration.replace(0, np.nan)) * 100.0
    pct = pct.where(pct.notna(), duration_pct)

    return pct.clip(lower=0.0, upper=100.0)


def normalize_yambda_flat_frame(
    frame: pd.DataFrame,
    *,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
    event_file: YambdaEventFile = "listens",
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 0.0,
    order_offset: int = 0,
) -> pd.DataFrame:
    """Normalize flat Yambda rows into the canonical interaction DataFrame.

    ``rating`` is the listened percentage on a 0..100 scale.  It is never read
    from a source ``rating`` column because Yambda does not define one.
    ``verified_purchase`` is a compatibility field and is always set to True for
    retained Yambda interactions.
    """
    if frame.empty:
        return _empty_interaction_frame()

    raw_user = _first_frame_column(frame, "uid", "user_id", "raw_user_id")
    raw_item = _first_frame_column(frame, "item_id", "track_id", "raw_item_id")
    timestamp = pd.to_numeric(_first_frame_column(frame, "timestamp", "ts"), errors="coerce").fillna(0).astype("int64")
    default_event_type = _event_type_from_file(event_file)
    event_type = _first_frame_column(frame, "event_type").fillna(default_event_type).astype("string")
    listened_pct = _listened_percentage_series(frame)

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
        else:
            raise ValueError("organic_filter must be 'any', 'organic', or 'recommendation'")
    if min_played_ratio_pct is not None:
        listen_mask = event_type.eq("listen")
        mask &= (~listen_mask) | (listened_pct.notna() & (listened_pct >= float(min_played_ratio_pct)))

    if not bool(mask.any()):
        return _empty_interaction_frame()

    # Compatibility-only parameters retained in the function signature.  They
    # are intentionally ignored because Yambda has no native rating concept.
    _ = event_value_map, default_rating
    rating = listened_pct.fillna(0.0)

    if "order_idx" in frame.columns:
        order_idx = pd.to_numeric(frame["order_idx"], errors="coerce").fillna(0).astype("int64").to_numpy()
    else:
        order_idx = np.arange(order_offset, order_offset + len(frame), dtype=np.int64)
    out = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").astype(str).to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").astype(str).to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "verified_purchase": np.ones(int(mask.sum()), dtype=bool),
            "order_idx": order_idx[mask.to_numpy()],
        }
    )
    if keep_raw:
        out["raw"] = frame.loc[mask].to_dict(orient="records")
    return out.reset_index(drop=True)


def _align_list(values: Any, n: int, *, default: Any = None) -> list[Any]:
    seq = _as_list(values)
    if len(seq) == n:
        return seq
    if len(seq) == 0:
        return list(range(n)) if default == "range" else [default] * n
    if len(seq) == 1 and n > 1:
        return seq * n
    raise ValueError(f"Yambda sequential list length mismatch: expected {n}, got {len(seq)}")


def normalize_yambda_sequential_frame(
    frame: pd.DataFrame,
    *,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 0.0,
) -> pd.DataFrame:
    """Explode sequential Yambda rows into canonical interaction rows.

    Ratings are not read from source rows. After explosion the shared flat
    normalizer derives them from listened percentage columns.
    """
    if frame.empty:
        return _empty_interaction_frame()

    base = pd.DataFrame(
        {
            "__row_idx": np.arange(len(frame), dtype=np.int64),
            "raw_user_id": _first_frame_column(frame, "uid", "user_id", "raw_user_id"),
            "raw_item_id": _first_frame_column(frame, "item_ids", "item_id", "track_ids", "track_id").map(_as_list),
            "timestamp_raw": _first_frame_column(frame, "timestamps", "timestamp", "ts"),
            "event_type_raw": _first_frame_column(frame, "event_types", "event_type"),
            "organic_raw": _first_frame_column(frame, "is_organic"),
            "played_pct_raw": _first_frame_column(frame, *_PLAYED_PERCENT_COLUMNS),
            "played_ratio_raw": _first_frame_column(frame, *_PLAYED_RATIO_COLUMNS),
            "played_duration_raw": _first_frame_column(frame, *_PLAYED_DURATION_COLUMNS),
            "track_duration_raw": _first_frame_column(frame, *_TRACK_DURATION_COLUMNS),
        }
    )
    base = base.loc[base["raw_user_id"].notna() & base["raw_item_id"].str.len().gt(0)].copy()
    if base.empty:
        return _empty_interaction_frame()

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
        for values, n in zip(base["played_pct_raw"], seq_len, strict=False)
    ]
    base["played_ratio"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["played_ratio_raw"], seq_len, strict=False)
    ]
    base["played_duration"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["played_duration_raw"], seq_len, strict=False)
    ]
    base["duration"] = [
        _align_list(values, int(n), default=None)
        for values, n in zip(base["track_duration_raw"], seq_len, strict=False)
    ]
    exploded = base[
        [
            "__row_idx",
            "raw_user_id",
            "raw_item_id",
            "timestamp",
            "event_type",
            "is_organic",
            "played_ratio_pct",
            "played_ratio",
            "played_duration",
            "duration",
        ]
    ].explode(
        [
            "raw_item_id",
            "timestamp",
            "event_type",
            "is_organic",
            "played_ratio_pct",
            "played_ratio",
            "played_duration",
            "duration",
        ],
        ignore_index=True,
    )
    exploded["order_idx"] = exploded.groupby("__row_idx", sort=False).cumcount().astype(np.int64)
    exploded["order_idx"] = exploded["__row_idx"].astype(np.int64) * 10**9 + exploded["order_idx"]
    return normalize_yambda_flat_frame(
        exploded,
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
    default_rating: float = 0.0,
) -> dict[str, Any] | None:
    """Normalize one flat Yambda row for tests and edge callers."""
    frame = normalize_yambda_flat_frame(
        pd.DataFrame([dict(record)]),
        keep_raw=keep_raw,
        event_file=event_file,
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
        event_value_map=event_value_map,
        default_rating=default_rating,
        order_offset=order_idx,
    )
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def normalize_yambda_sequential_record(
    record: Mapping[str, Any],
    *,
    keep_raw: bool = False,
    order_idx_start: int = 0,
    organic_filter: OrganicFilter = "any",
    min_played_ratio_pct: float | None = None,
    allowed_event_types: tuple[str, ...] | None = None,
    event_value_map: Mapping[str, float] | None = None,
    default_rating: float = 0.0,
) -> list[dict[str, Any]]:
    """Normalize one sequential Yambda row into canonical interaction dicts."""
    frame = normalize_yambda_sequential_frame(
        pd.DataFrame([dict(record)]),
        keep_raw=keep_raw,
        organic_filter=organic_filter,
        min_played_ratio_pct=min_played_ratio_pct,
        allowed_event_types=allowed_event_types,
        event_value_map=event_value_map,
        default_rating=default_rating,
    )
    if frame.empty:
        return []
    frame = frame.copy()
    frame["order_idx"] = np.arange(order_idx_start, order_idx_start + len(frame), dtype=np.int64)
    return frame.to_dict(orient="records")


def _event_type_from_file(event_file: str) -> str:
    return {
        "listens": "listen",
        "likes": "like",
        "dislikes": "dislike",
        "unlikes": "unlike",
        "undislikes": "undislike",
    }.get(event_file, "listen")


def _as_list(value: Any) -> list[Any]:
    if value is None or value is pd.NA:
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
        listened_pct = _listened_percentage_series(pd.DataFrame([dict(record)])).iloc[0]
        if pd.isna(listened_pct) or float(listened_pct) < float(min_played_ratio_pct):
            return False

    return True


def _rating_from_yambda_record(
    record: Mapping[str, Any],
    *,
    event_type: str,
    event_value_map: Mapping[str, float] | None,
    default_rating: float,
) -> float:
    """Compatibility helper: derive the canonical Yambda rating from listens."""
    _ = event_type, event_value_map, default_rating
    listened_pct = _listened_percentage_series(pd.DataFrame([dict(record)])).iloc[0]
    return 0.0 if pd.isna(listened_pct) else float(listened_pct)
