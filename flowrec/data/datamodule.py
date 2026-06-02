from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from .collators import HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, SequenceDataset
from .preprocessors import (
    ItemKey,
    apply_k_core_filter,
    collect_item_ids,
    count_interactions,
    load_metadata,
    load_reviews_by_user,
    load_semantic_ids,
    metadata_to_text,
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


@dataclass(frozen=True, slots=True)
class AmazonReviewDataConfig:
    """Config for Amazon Reviews 2023 sequential recommendation data."""

    reviews_path: str | Path
    metadata_path: str | Path | None = None
    semantic_id_path: str | Path | None = None

    item_key: ItemKey = "parent_asin"

    # Filtering / normalization.
    min_user_interactions: int = 5
    min_item_interactions: int = 0
    use_verified_only: bool = False
    keep_raw_reviews: bool = False
    deduplicate_consecutive: bool = False

    # Splitting.
    split_strategy: SplitStrategy = "leave_one_out"
    min_history_len: int = 1
    train_all_prefixes: bool = True

    # Training target selection. Prefer this over train_all_prefixes for
    # long-history datasets. If train_target_policy == "all" and
    # train_all_prefixes is False, the effective policy is "last".
    train_target_policy: TrainTargetPolicy = "all"
    train_targets_per_user: int = 256
    train_stride: int = 10
    train_last_n: int = 512
    train_hybrid_recent_n: int = 256
    train_hybrid_random_n: int = 256

    # Used with split_strategy='global_time'.
    global_val_timestamp: int | None = None
    global_test_timestamp: int | None = None
    global_val_ratio: float = 0.1
    global_test_ratio: float = 0.1
    global_history_policy: GlobalHistoryPolicy = "rolling"
    global_eval_target_mode: GlobalEvalTargetMode = "all"

    # Batching.
    max_history_len: int = 20
    history_padding: HistoryPadding = "left"
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    train_drop_last: bool = False
    include_raw_in_batch: bool = True

    # IDs.
    pad_item_id: int = 0
    pad_token_id: int = 0
    offset_semantic_ids_by_one: bool = True
    drop_items_missing_semantic_id: bool = True
    user_hash_mod: int | None = None

    # Post-split cold-item filtering.
    # This makes the item catalog train-only after the train/val/test split:
    # validation/test targets unseen in training are dropped, unseen history
    # items are removed from user sequences, and item IDs are compacted.
    filter_cold_items_after_split: bool = True
    cold_item_train_source: ColdItemTrainSource = "observed"

    # Reproducibility.
    seed: int = 42


class AmazonReviewDataModule:
    """Framework-agnostic DataModule for Amazon Reviews 2023.

    It supports two split strategies:
      - leave_one_out: per-user last item is test, second-last is validation.
      - global_time: global chronological split by timestamp.

    The module is intentionally compatible with plain PyTorch rather than tied
    to PyTorch Lightning.
    """

    def __init__(self, cfg: AmazonReviewDataConfig) -> None:
        self.cfg = cfg
        self.reviews_path = Path(cfg.reviews_path)
        self.metadata_path = Path(cfg.metadata_path) if cfg.metadata_path else None
        self.semantic_id_path = Path(cfg.semantic_id_path) if cfg.semantic_id_path else None

        self.user2id: dict[str, int] = {}
        self.id2user: list[str] = []

        # 0 is reserved for padding.
        self.item2id: dict[str, int] = {}
        self.id2item: list[str] = ["<PAD>"]

        self.metadata: dict[str, dict] = {}
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
        """Number of non-padding items."""
        return len(self.id2item) - 1

    @property
    def item_vocab_size(self) -> int:
        """Number of item IDs including padding."""
        return len(self.id2item)

    def prepare_data(self) -> None:
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"reviews_path does not exist: {self.reviews_path}")
        if self.metadata_path is not None and not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata_path does not exist: {self.metadata_path}")
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

        if self.metadata_path is not None:
            self.metadata = load_metadata(self.metadata_path, item_key=self.cfg.item_key)

        allowed_items = None
        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            allowed_items = set(self.semantic_ids)

        user_events = load_reviews_by_user(
            self.reviews_path,
            item_key=self.cfg.item_key,
            use_verified_only=self.cfg.use_verified_only,
            keep_raw=self.cfg.keep_raw_reviews,
            allowed_items=allowed_items,
        )

        user_events = apply_k_core_filter(
            user_events,
            min_user_interactions=self.cfg.min_user_interactions,
            min_item_interactions=self.cfg.min_item_interactions,
            drop_consecutive_repeats=self.cfg.deduplicate_consecutive,
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
            self.metadata = {
                raw_item: meta
                for raw_item, meta in self.metadata.items()
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

    def raw_user_to_id(self, raw_user_id: str) -> int:
        return self.user2id[raw_user_id]

    def id_to_raw_user(self, user_id: int) -> str:
        return self.id2user[user_id]

    def get_item_metadata(self, raw_item_id: str) -> dict | None:
        return self.metadata.get(raw_item_id)

    def get_item_text(self, raw_item_id: str) -> str:
        meta = self.get_item_metadata(raw_item_id)
        return metadata_to_text(meta) if meta is not None else ""

    def item_ids_in_data(self) -> set[str]:
        if not self._is_setup:
            raise RuntimeError("Call setup() before item_ids_in_data().")
        return set(self.item2id.keys())

    def summary(self) -> dict:
        if self.stats is None:
            raise RuntimeError("Call setup() before summary().")
        out = asdict(self.stats)
        out.update(
            {
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
