from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .collators import DenseCausalCollator, HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, SequenceDataset, history_length_stats
from .tensor_batches import maybe_tensorized_parts
from .interaction_store import (
    InteractionLogStore,
    build_interaction_store,
    partition_interactions,
)
from .preprocessors import (
    ItemKey,
    apply_k_core_filter_frame,
    count_interactions,
    load_interactions_frame,
    load_metadata_frame,
    load_semantic_ids,
    metadata_frame_to_dict,
    metadata_to_text,
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
    TrainExampleLayout,
    TrainTargetPolicy,
    build_sequence_splits,
    factorize_interaction_ids,
    validate_dense_layout_config,
)


@dataclass(frozen=True, slots=True)
class AmazonReviewDataConfig:
    """Config for the DataFrame-based Amazon Reviews sequential pipeline.

    reviews_path is expected to point to a parquet file with columns:
    rating, asin, user_id, timestamp, verified_purchase. JSONL/JSON remain
    supported as compatibility inputs, but parquet is the primary fast path.
    """

    reviews_path: str | Path
    metadata_path: str | Path | None = None
    semantic_id_path: str | Path | None = None

    # Interaction and metadata keys are intentionally separated because the new
    # parquet interaction schema commonly uses asin, while metadata uses
    # parent_asin.
    item_key: ItemKey = "asin"
    metadata_item_key: ItemKey = "parent_asin"
    filter_reviews_to_metadata: bool = True

    # Positive/auxiliary partition. A pandas query expression evaluated on the
    # canonical frame, e.g. "rating >= 4.0". Matching rows become the positive
    # training stream (stripped to user/item/timestamp; they alone define the
    # catalog); the complement keeps rating/verified_purchase and is exposed as
    # ``interaction_store`` for reward generation and negative sampling.
    # None keeps the historical behavior: every review is positive.
    positive_criterion: str | None = None

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

    # Training example materialization. "prefix_target" keeps the historical
    # one-row-per-(prefix, target) layout for encoder-decoder-style training.
    # "dense_causal" emits one row per user window with per-position labels for
    # decoder-only-style training (one forward pass supervises every position);
    # train_target_policy is ignored in that mode.
    train_example_layout: TrainExampleLayout = "prefix_target"
    train_window_len: int | None = None  # defaults to max_history_len
    train_window_stride: int | None = None  # defaults to train_window_len

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

    # Assemble prefix-target batches from flat tensors instead of per-example
    # Python dicts (identical batch contents; large-scale speed/RAM win). The
    # fast path only covers include_raw_in_batch=False prefix-target loaders;
    # other loaders keep the legacy collator regardless of this flag.
    tensorized_batches: bool = True

    # Post-split cold-item filtering.
    filter_cold_items_after_split: bool = True
    cold_item_train_source: ColdItemTrainSource = "observed"

    # Reproducibility.
    seed: int = 42


class AmazonReviewDataModule:
    """Framework-agnostic DataModule for Amazon Reviews 2023.

    Data preparation is DataFrame-first: interactions are normalized to a single
    DataFrame, k-core filtering and ID assignment add columns, split builders
    produce split DataFrames, and SequenceDataset only exposes rows as the old
    dictionary batch schema.
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

        self.interactions: pd.DataFrame = pd.DataFrame()
        self.metadata: pd.DataFrame = pd.DataFrame()
        self._metadata_by_item: dict[str, dict] = {}
        self.semantic_ids: dict[str, list[int]] | None = None
        self.sid_len: int | None = None
        self.interaction_store: InteractionLogStore | None = None

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

        validate_dense_layout_config(
            layout=self.cfg.train_example_layout,
            history_padding=self.cfg.history_padding,
        )
        self.prepare_data()

        if self.semantic_id_path is not None:
            self.semantic_ids, self.sid_len = load_semantic_ids(
                self.semantic_id_path,
                offset_by_one=self.cfg.offset_semantic_ids_by_one,
            )

        allowed_items: set[str] | None = None
        if self.metadata_path is not None:
            self.metadata = load_metadata_frame(self.metadata_path, item_key=self.cfg.metadata_item_key)
            self._metadata_by_item = metadata_frame_to_dict(self.metadata)
            if self.cfg.filter_reviews_to_metadata:
                allowed_items = set(self.metadata["raw_item_id"].astype(str))

        interactions = load_interactions_frame(
            self.reviews_path,
            item_key=self.cfg.item_key,
            use_verified_only=self.cfg.use_verified_only,
            keep_raw=self.cfg.keep_raw_reviews,
            allowed_items=allowed_items,
        )

        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            semantic_catalog = set(self.semantic_ids)
            interactions = interactions.loc[interactions["raw_item_id"].astype(str).isin(semantic_catalog)].reset_index(drop=True)

        interactions, auxiliary = partition_interactions(
            interactions, self.cfg.positive_criterion
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

        split_cfg = SequenceSplitConfig(
            split_strategy=self.cfg.split_strategy,
            min_history_len=self.cfg.min_history_len,
            train_all_prefixes=self.cfg.train_all_prefixes,
            train_target_policy=self.cfg.train_target_policy,
            train_example_layout=self.cfg.train_example_layout,
            train_window_len=self.cfg.train_window_len,
            train_window_stride=self.cfg.train_window_stride,
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
                raw_item: tokens
                for raw_item, tokens in self.semantic_ids.items()
                if raw_item in self.item2id
            }
            self.sid_len = max((len(tokens) for tokens in self.semantic_ids.values()), default=0) or None

        if not self.metadata.empty:
            self.metadata = self.metadata.loc[
                self.metadata["raw_item_id"].astype(str).isin(self.item2id)
            ].reset_index(drop=True)
            self._metadata_by_item = metadata_frame_to_dict(self.metadata)

        # Built after cold-item filtering so the store is keyed by the final catalog.
        self.interaction_store = build_interaction_store(
            auxiliary,
            user2id=self.user2id,
            item2id=self.item2id,
            num_users=self.num_users,
            item_vocab_size=self.item_vocab_size,
            default_event_type="review",
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

        dataset, collate_fn = self._loader_parts(
            self.train_dataset, self._make_train_collator()
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            drop_last=self.cfg.train_drop_last,
            collate_fn=collate_fn,
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
        eval_dataset, collate_fn = self._loader_parts(dataset, self._make_collator())
        return DataLoader(
            eval_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            collate_fn=collate_fn,
        )

    def _loader_parts(self, dataset: SequenceDataset, collator: Any) -> tuple[Any, Any]:
        """Swap in the vectorized dataset/collator pair when applicable."""
        parts = maybe_tensorized_parts(
            dataset,
            collator,
            id2item=self.id2item,
            enabled=bool(self.cfg.tensorized_batches),
        )
        return parts if parts is not None else (dataset, collator)

    def _make_train_collator(self) -> SequenceBatchCollator | DenseCausalCollator:
        """Training collator; evaluation always uses the prefix-target collator."""
        if self.cfg.train_example_layout == "dense_causal":
            window_len = self.cfg.train_window_len or self.cfg.max_history_len
            return DenseCausalCollator(
                max_seq_len=window_len,
                history_padding=self.cfg.history_padding,
                pad_item_id=self.cfg.pad_item_id,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids=self.semantic_ids,
                sid_len=self.sid_len,
                include_raw=self.cfg.include_raw_in_batch,
                user_hash_mod=self.cfg.user_hash_mod,
            )        
        return self._make_collator()

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

    def get_item_metadata(self, raw_item_id: str) -> dict | None:
        return self._metadata_by_item.get(raw_item_id)

    def get_item_text(self, raw_item_id: str) -> str:
        meta = self.get_item_metadata(raw_item_id)
        return metadata_to_text(meta) if meta is not None else ""

    def item_ids_in_data(self, *, include_auxiliary: bool = False) -> set[str]:
        """Raw item IDs that remain in the dataset after all preprocessing.

        By default this is the positive catalog. With ``include_auxiliary=True``
        it also covers items that catalog users touched only in auxiliary
        events — the scope for item representations that must cover sampled
        negatives as well as positives.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before item_ids_in_data().")
        ids = set(self.item2id.keys())
        if include_auxiliary and self.interaction_store is not None:
            ids |= self.interaction_store.out_of_catalog_item_ids
        return ids

    def add_semantic_ids(
        self,
        semantic_ids: Mapping[str, list[int]] | str | Path,
        *,
        offset_by_one: bool | None = None,
    ) -> None:
        """Attach semantic IDs after setup has built the item catalog."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before add_semantic_ids().")

        should_offset = self.cfg.offset_semantic_ids_by_one if offset_by_one is None else offset_by_one
        loaded_path: Path | None = None
        if isinstance(semantic_ids, (str, Path)):
            loaded_semantic_ids, _ = load_semantic_ids(
                semantic_ids,
                offset_by_one=should_offset,
            )
            loaded_path = Path(semantic_ids)
        else:
            loaded_semantic_ids = {
                str(raw_item): [int(token) + 1 if should_offset else int(token) for token in tokens]
                for raw_item, tokens in semantic_ids.items()
            }

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

    def summary(self) -> dict:
        if self.stats is None:
            raise RuntimeError("Call setup() before summary().")
        out = asdict(self.stats)
        out.update(
            {
                "split_strategy": self.cfg.split_strategy,
                "train_example_layout": self.cfg.train_example_layout,
                "global_val_cutoff": self.global_val_cutoff,
                "global_test_cutoff": self.global_test_cutoff,
                "sid_len": self.sid_len,
                "item_vocab_size": self.item_vocab_size,
                "positive_criterion": self.cfg.positive_criterion,
                "interaction_store": self.interaction_store.to_summary_dict()
                if self.interaction_store is not None
                else None,
                "post_split_filter": self.post_split_filter_stats.to_dict()
                if self.post_split_filter_stats is not None
                else None,
            }
        )
        return out
