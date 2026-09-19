from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .collators import DenseCausalCollator, HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, SequenceDataset, history_length_stats
from .interaction_store import (
    InteractionLogStore,
    build_interaction_store,
    partition_interactions,
)
from .popularity import training_item_counts_by_raw_id
from .post_split_filtering import (
    ColdItemTrainSource,
    PostSplitColdItemFilterStats,
    disabled_post_split_filter_stats,
    filter_splits_to_train_item_catalog,
    remap_interactions_to_catalog,
)
from .preprocessors import (
    apply_k_core_filter_frame,
    count_interactions,
    load_semantic_ids,
)
from .sequence_builder import (
    GlobalEvalTargetMode,
    GlobalHistoryPolicy,
    SequenceSplitConfig,
    SequenceSplits,
    SplitStrategy,
    TrainExampleLayout,
    TrainTargetPolicy,
    _build_examples_from_targets,
    _empty_dense_frame,
    _empty_sequence_frame,
    _prepare_events,
    _select_eval_rows,
    _select_train_rows,
    build_dense_train_examples,
    build_sequence_splits,
    factorize_interaction_ids,
    validate_dense_layout_config,
)
from .tensor_batches import maybe_tensorized_parts

VKSplit = Literal["train", "validation", "test"]
VKFlowSplit = Literal["train", "val", "test"]
VKSplitStrategy = Literal["provided", "leave_one_out", "global_time"]
VKRatingMode = Literal["binary", "timespent", "weighted_feedback"]

# Native VK-LSVD feedback flags (https://huggingface.co/datasets/deepvk/VK-LSVD).
# Every interaction row carries them as booleans next to ``timespent`` (watch
# time in seconds, uint8, 0-255). They are preserved on the canonical frame so
# ``positive_criterion`` can be evaluated on them and the auxiliary interaction
# store keeps them as reward payload.
VK_FEEDBACK_FLAGS: tuple[str, ...] = (
    "like",
    "dislike",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
)
VK_POSITIVE_FLAGS: tuple[str, ...] = (
    "like",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
)

# Canonical VK-LSVD interaction schema. ``rating`` is the configurable scalar
# feedback score (``rating_mode``); ``flowrec_split`` / ``flowrec_week`` come
# from the source file so the provided train/validation/test split can be
# rebuilt after filtering; the native feedback columns follow.
_CANONICAL_INTERACTION_COLUMNS = [
    "raw_user_id",
    "raw_item_id",
    "timestamp",
    "rating",
    "order_idx",
    "flowrec_split",
    "flowrec_week",
    "timespent",
    *VK_FEEDBACK_FLAGS,
]
_VK_USER_COLUMNS = ("user_id", "uid", "raw_user_id")
_VK_ITEM_COLUMNS = ("item_id", "video_id", "raw_item_id")
_VK_TIMESTAMP_COLUMNS = ("timestamp", "ts")
# Source columns the loader consumes; the upstream ``place`` / ``platform`` /
# ``agent`` context columns are only read with ``keep_raw_interactions``.
_VK_INPUT_COLUMNS = (
    *_VK_USER_COLUMNS,
    *_VK_ITEM_COLUMNS,
    *_VK_TIMESTAMP_COLUMNS,
    "timespent",
    "rating",
    *VK_FEEDBACK_FLAGS,
)
_SPLIT_RANK: dict[str, int] = {"train": 0, "val": 1, "test": 2}
_PARQUET_BATCH_ROWS = 1_000_000


def _suffixes(path: str | Path) -> str:
    return "".join(Path(path).suffixes).lower()


def _is_parquet(path: str | Path) -> bool:
    return _suffixes(path).endswith(".parquet")


def _is_csv(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".csv", ".tsv"}


def _is_jsonl(path: str | Path) -> bool:
    suffixes = _suffixes(path)
    return suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz")


def _is_npz(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".npz"


def _empty_interaction_frame() -> pd.DataFrame:
    dtypes = {
        "raw_user_id": "object",
        "raw_item_id": "object",
        "timestamp": "int64",
        "rating": "float64",
        "order_idx": "int64",
        "flowrec_split": "object",
        "flowrec_week": "int64",
        "timespent": "float64",
        **{flag: "bool" for flag in VK_FEEDBACK_FLAGS},
    }
    return pd.DataFrame({column: pd.Series(dtype=dtypes[column]) for column in _CANONICAL_INTERACTION_COLUMNS})


def _read_vk_tabular_frame(
    path: str | Path,
    *,
    max_rows: int | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one VK-LSVD table through pandas.

    Parquet is the canonical VK-LSVD storage format; only the requested
    ``columns`` are decoded and ``max_rows`` stops after the first row groups
    instead of materializing a full weekly file. CSV/TSV/JSONL are retained as
    strict compatibility inputs.
    """
    path = Path(path)
    if _is_parquet(path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = parquet_file.schema_arrow.names
        selected = (
            [column for column in columns if column in available]
            if columns is not None
            else None
        )
        if max_rows is None:
            return parquet_file.read(columns=selected).to_pandas()
        schema = parquet_file.schema_arrow
        if selected is not None:
            schema = pa.schema([schema.field(name) for name in selected])
        batches = []
        remaining = int(max_rows)
        if remaining > 0:
            for batch in parquet_file.iter_batches(
                batch_size=min(remaining, _PARQUET_BATCH_ROWS), columns=selected
            ):
                batches.append(batch.slice(0, remaining))
                remaining -= min(batch.num_rows, remaining)
                if remaining <= 0:
                    break
        return pa.Table.from_batches(batches, schema=schema).to_pandas()
    if _is_csv(path):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, nrows=max_rows)
    elif _is_jsonl(path):
        frame = pd.read_json(path, lines=True, compression="infer")
        if max_rows is not None:
            frame = frame.head(max_rows)
    else:
        raise ValueError(f"Unsupported VK-LSVD interaction table format: {path}")
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]]
    return frame


def iter_tabular_records(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw records through the same strict pandas reader as the loader."""
    frame = _read_vk_tabular_frame(path, max_rows=max_rows)
    yield from frame.to_dict(orient="records")


def _as_path_list(value: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(x) for x in value]


def _glob_split_files(root: str | Path, split: VKSplit) -> list[Path]:
    root = Path(root)
    names = [split]
    if split == "validation":
        names.append("val")
    candidates = [root / "interactions" / name for name in names] + [root / name for name in names]
    for directory in candidates:
        if directory.exists() and directory.is_dir():
            return sorted(directory.glob("week_*.parquet"))
    return []


def _week_from_path(path: str | Path, default_week: int = 0) -> int:
    stem = Path(path).stem
    suffix = stem.split("_", 1)[1] if stem.startswith("week_") and "_" in stem else ""
    return int(suffix) if suffix.isdigit() else default_week


def _flow_split(split: VKSplit) -> VKFlowSplit:
    return "val" if split == "validation" else split  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class VKLSVDDataConfig:
    """Configuration for the DataFrame-based VK-LSVD sequential pipeline.

    VK-LSVD (https://huggingface.co/datasets/deepvk/VK-LSVD) ships weekly
    Parquet files under ``interactions/{train,validation,test}/week_XX.parquet``
    (and the same layout under ``subsamples/<name>/``). Each row is one
    user-item observation with ``user_id``, ``item_id``, ``timespent`` (watch
    seconds, uint8) and the boolean feedback flags ``like``, ``dislike``,
    ``share``, ``bookmark``, ``click_on_author``, ``open_comments``; there is no
    timestamp column, rows within a file are in increasing time order. The
    loader normalizes every file into a canonical interaction DataFrame
    (raw_user_id, raw_item_id, timestamp, rating, split marker, native
    feedback columns) and all downstream processing stays DataFrame-based.
    """

    # Local loading. ``interactions_root`` is a folder holding the weekly
    # split directories (``<root>/train``, ``<root>/validation``,
    # ``<root>/test`` or the same under ``<root>/interactions/``).
    interactions_root: str | Path | None = None
    interactions_path: str | Path | Sequence[str | Path] | None = None
    train_paths: Sequence[str | Path] | None = None
    val_paths: Sequence[str | Path] | None = None
    test_paths: Sequence[str | Path] | None = None

    # Optional semantic IDs.
    semantic_id_path: str | Path | None = None
    # Optional item-embedding artifact (VK-LSVD ``item_embeddings.npz`` with
    # ``item_id`` / ``embedding`` arrays, an ItemEmbeddingStore ``.pt`` payload,
    # or a tabular file with an item_id column). When provided and
    # drop_items_missing_embedding is True, interactions whose items have no
    # embedding are dropped, mirroring the Yambda module.
    embeddings_path: str | Path | None = None

    # Hugging Face loading.
    use_hf: bool = False
    hf_repo_id: str = "deepvk/VK-LSVD"
    hf_revision: str | None = None
    hf_local_dir: str | Path | None = None
    subsample_name: str | None = None
    train_weeks: tuple[int, ...] = tuple(range(25))
    val_weeks: tuple[int, ...] = (25,)
    test_weeks: tuple[int, ...] = (26,)

    # Row limits are useful for smoke tests and small experiments (they apply
    # per file).
    max_rows: int | None = None
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None

    # Positive/auxiliary partition. A pandas query expression evaluated on the
    # canonical frame, e.g. "timespent >= 1 and not dislike" or "like or
    # share". Rows matching it become the positive training stream (stripped
    # to user/item/timestamp/split marker; they alone define the catalog); the
    # complement keeps its payload columns (timespent, feedback flags, rating)
    # and is exposed as ``interaction_store`` for reward generation and
    # negative sampling. None keeps the historical behavior: every loaded
    # interaction is positive and the store is empty. Prefer this over the
    # destructive scope filters below (min_timespent /
    # require_positive_feedback), which remove events from both streams.
    positive_criterion: str | None = None

    # Interaction normalization / filtering.
    min_user_interactions: int = 5
    min_item_interactions: int = 0
    deduplicate_consecutive: bool = False
    keep_raw_interactions: bool = False
    min_timespent: float | None = None
    require_positive_feedback: bool = False
    rating_mode: VKRatingMode = "weighted_feedback"
    default_rating: float = 1.0
    feedback_weights: dict[str, float] = field(
        default_factory=lambda: {
            "timespent": 1.0 / 255.0,
            "like": 1.0,
            "share": 1.0,
            "bookmark": 1.0,
            "click_on_author": 0.5,
            "open_comments": 0.25,
            "dislike": -1.0,
        }
    )

    # Splitting. "provided" uses VK-LSVD's train/validation/test directories
    # (the dataset's Global Temporal Split); the other strategies ignore the
    # file labels and re-split the concatenated interactions.
    split_strategy: VKSplitStrategy = "provided"
    min_history_len: int = 1

    # Training example materialization. "prefix_target" keeps the historical
    # one-row-per-(prefix, target) layout; "dense_causal" emits one row per
    # user window with per-position labels for decoder-only-style training
    # (train_target_policy is ignored in that mode). Both layouts are available
    # for every split strategy: with the provided split the dense train region
    # is exactly the train-week rows of each user.
    train_example_layout: TrainExampleLayout = "prefix_target"
    train_window_len: int | None = None  # defaults to max_history_len
    train_window_stride: int | None = None  # defaults to train_window_len

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
    # With the provided split these two also shape validation/test examples:
    # global_history_policy decides whether an eval target sees the events
    # that precede it inside its own week ("rolling") or only the events
    # before the week ("window_start"); global_eval_target_mode keeps every
    # eval-week event as a target ("all") or only each user's last one.
    global_history_policy: GlobalHistoryPolicy = "rolling"
    global_eval_target_mode: GlobalEvalTargetMode = "all"

    # Synthetic timestamp construction when files do not contain timestamp:
    # week * stride + row index, so weeks never interleave.
    week_timestamp_stride: int = 10**12

    # Batching.
    max_history_len: int = 50
    history_padding: HistoryPadding = "left"
    batch_size: int = 256
    eval_batch_size: int = 512
    test_batch_size: int | None = None
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
    drop_items_missing_embedding: bool = True
    user_hash_mod: int | None = None

    # Assemble prefix-target batches from flat tensors instead of per-example
    # Python dicts (identical batch contents; large-scale speed/RAM win). The
    # fast path only covers include_raw_in_batch=False prefix-target loaders.
    tensorized_batches: bool = True

    # Post-split cold-item filtering.
    filter_cold_items_after_split: bool = True
    cold_item_train_source: ColdItemTrainSource = "observed"

    seed: int = 42


class VKLSVDDataModule:
    """Framework-agnostic DataModule for VK-LSVD.

    The public batch format matches the Amazon and Yambda modules: interactions,
    k-core filtering, split construction, train-catalog filtering, and datasets
    are all DataFrames, so every model runner (SASRec dense-causal, TIGER
    prefix-target, GPT-2 dense-causal over semantic IDs) consumes it unchanged.
    """

    def __init__(self, cfg: VKLSVDDataConfig) -> None:
        self.cfg = cfg
        self.semantic_id_path = Path(cfg.semantic_id_path) if cfg.semantic_id_path else None
        self.embeddings_path = Path(cfg.embeddings_path) if cfg.embeddings_path else None

        self.user2id: dict[str, int] = {}
        self.id2user: list[str] = []
        self.item2id: dict[str, int] = {}
        self.id2item: list[str] = ["<PAD>"]

        self.interactions: pd.DataFrame = pd.DataFrame()
        self.semantic_ids: dict[str, list[int]] | None = None
        self.sid_len: int | None = None
        self.embedding_item_ids: np.ndarray | set[str] | None = None
        self.interaction_store: InteractionLogStore | None = None
        self.train_item_counts_by_raw_id: dict[str, int] = {}

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
        if self.semantic_id_path is not None and not self.semantic_id_path.exists():
            raise FileNotFoundError(f"semantic_id_path does not exist: {self.semantic_id_path}")
        if self.embeddings_path is not None and not self.embeddings_path.exists():
            raise FileNotFoundError(f"embeddings_path does not exist: {self.embeddings_path}")
        if self.cfg.use_hf:
            return
        missing = [path for path in vk_lsvd_local_input_files(self.cfg) if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing VK-LSVD interaction files:\n{formatted}")

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

        semantic_items: set[str] | None = None
        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            semantic_items = set(self.semantic_ids)
        if self.embeddings_path is not None and self.cfg.drop_items_missing_embedding:
            self.embedding_item_ids = load_vk_lsvd_embedding_item_ids(self.embeddings_path)
        allowed_items = _combine_allowed_items(semantic_items, self.embedding_item_ids)

        interactions = load_vk_lsvd_interactions_frame(cfg=self.cfg, allowed_items=allowed_items)

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

        if self.cfg.split_strategy == "provided":
            splits = build_provided_sequence_splits(interactions, cfg=self.cfg)
        else:
            split_cfg = _sequence_split_config_from_vk(self.cfg, split_strategy=self.cfg.split_strategy)
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
            self.interactions = remap_interactions_to_catalog(
                self.interactions, self.item2id
            )
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

        # Built after cold-item filtering so the store is keyed by the final catalog.
        self.interaction_store = build_interaction_store(
            auxiliary,
            user2id=self.user2id,
            item2id=self.item2id,
            num_users=self.num_users,
            item_vocab_size=self.item_vocab_size,
        )

        self.global_val_cutoff = splits.val_cutoff
        self.global_test_cutoff = splits.test_cutoff
        self.train_item_counts_by_raw_id = training_item_counts_by_raw_id(
            self.interactions,
            split_strategy=self.cfg.split_strategy,
            global_val_cutoff=self.global_val_cutoff,
            item2id=self.item2id,
        )
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
        dataset, collate_fn = self._loader_parts(self.train_dataset, self._make_train_collator())
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
        return self._eval_dataloader(
            self.test_dataset,
            batch_size=self.cfg.test_batch_size or self.cfg.eval_batch_size,
        )

    def _eval_dataloader(
        self,
        dataset: SequenceDataset,
        *,
        batch_size: int | None = None,
    ) -> DataLoader:
        eval_dataset, collate_fn = self._loader_parts(dataset, self._make_collator())
        return DataLoader(
            eval_dataset,
            batch_size=batch_size or self.cfg.eval_batch_size,
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

    def item_ids_in_data(self, *, include_auxiliary: bool = False) -> set[str]:
        """Raw item IDs that remain in the dataset after all preprocessing.

        By default this is the positive catalog. With ``include_auxiliary=True``
        it also covers items that catalog users touched only in auxiliary
        events (dislikes, sub-threshold watches, ...) — the scope to use when
        training item representations (e.g. the semantic-ID tokenizer) that
        must cover sampled negatives as well as positives.
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
                "dataset": "vk_lsvd",
                "split_strategy": self.cfg.split_strategy,
                "train_example_layout": self.cfg.train_example_layout,
                "subsample_name": self.cfg.subsample_name,
                "rating_mode": self.cfg.rating_mode,
                "global_val_cutoff": self.global_val_cutoff,
                "global_test_cutoff": self.global_test_cutoff,
                "sid_len": self.sid_len,
                "item_vocab_size": self.item_vocab_size,
                "num_embedding_items": len(self.embedding_item_ids)
                if self.embedding_item_ids is not None
                else None,
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


def _sequence_split_config_from_vk(
    cfg: VKLSVDDataConfig,
    *,
    split_strategy: SplitStrategy,
    global_val_timestamp: int | None = None,
    global_test_timestamp: int | None = None,
) -> SequenceSplitConfig:
    return SequenceSplitConfig(
        split_strategy=split_strategy,
        min_history_len=cfg.min_history_len,
        train_example_layout=cfg.train_example_layout,
        train_window_len=cfg.train_window_len,
        train_window_stride=cfg.train_window_stride,
        train_all_prefixes=cfg.train_all_prefixes,
        train_target_policy=cfg.train_target_policy,
        train_targets_per_user=cfg.train_targets_per_user,
        train_stride=cfg.train_stride,
        train_last_n=cfg.train_last_n,
        train_hybrid_recent_n=cfg.train_hybrid_recent_n,
        train_hybrid_random_n=cfg.train_hybrid_random_n,
        max_history_len=cfg.max_history_len,
        global_val_timestamp=cfg.global_val_timestamp if global_val_timestamp is None else global_val_timestamp,
        global_test_timestamp=cfg.global_test_timestamp if global_test_timestamp is None else global_test_timestamp,
        global_val_ratio=cfg.global_val_ratio,
        global_test_ratio=cfg.global_test_ratio,
        global_history_policy=cfg.global_history_policy,
        global_eval_target_mode=cfg.global_eval_target_mode,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_vk_lsvd_interactions_frame(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Load VK-LSVD interactions as one canonical DataFrame."""
    paths = _resolve_paths_for_loading(cfg)
    frames = [
        _load_vk_lsvd_file_frame(
            path,
            split=split,
            week=week,
            cfg=cfg,
            max_rows=max_rows,
            allowed_items=allowed_items,
        )
        for split, path, week, max_rows in paths
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_interaction_frame()
    return pd.concat(frames, ignore_index=True)


# Compatibility alias with the old public loader name. It now returns a DataFrame
# instead of grouped per-row dataclass objects.
def load_vk_lsvd_events_by_user(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    return load_vk_lsvd_interactions_frame(cfg=cfg, allowed_items=allowed_items)


def load_vk_lsvd_events_by_split(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | np.ndarray | None = None,
) -> dict[VKFlowSplit, pd.DataFrame]:
    """Load VK-LSVD's provided split files as split DataFrames."""
    split_paths = resolve_vk_lsvd_split_paths(cfg)
    max_rows_by_split = _max_rows_by_split(cfg)
    out: dict[VKFlowSplit, pd.DataFrame] = {
        "train": _empty_interaction_frame(),
        "val": _empty_interaction_frame(),
        "test": _empty_interaction_frame(),
    }
    for split, paths in split_paths.items():
        frames = [
            _load_vk_lsvd_file_frame(
                path,
                split=split,
                week=week,
                cfg=cfg,
                max_rows=max_rows_by_split[split],
                allowed_items=allowed_items,
            )
            for path, week in paths
        ]
        frames = [frame for frame in frames if not frame.empty]
        out[_flow_split(split)] = pd.concat(frames, ignore_index=True) if frames else _empty_interaction_frame()
    return out


def _max_rows_by_split(cfg: VKLSVDDataConfig) -> dict[VKSplit, int | None]:
    return {
        "train": cfg.max_train_rows if cfg.max_train_rows is not None else cfg.max_rows,
        "validation": cfg.max_val_rows if cfg.max_val_rows is not None else cfg.max_rows,
        "test": cfg.max_test_rows if cfg.max_test_rows is not None else cfg.max_rows,
    }


def _resolve_paths_for_loading(cfg: VKLSVDDataConfig) -> list[tuple[VKSplit, Path, int, int | None]]:
    if cfg.split_strategy == "provided":
        split_paths = resolve_vk_lsvd_split_paths(cfg)
        max_rows_by_split = _max_rows_by_split(cfg)
        return [
            (split, path, week, max_rows_by_split[split])
            for split, items in split_paths.items()
            for path, week in items
        ]

    if cfg.interactions_path is not None:
        return [
            ("train", path, _week_from_path(path, idx), cfg.max_rows)
            for idx, path in enumerate(_as_path_list(cfg.interactions_path))
        ]

    split_paths = resolve_vk_lsvd_split_paths(cfg)
    return [
        (split, path, week, cfg.max_rows)
        for split, items in split_paths.items()
        for path, week in items
    ]


def vk_lsvd_local_input_files(cfg: VKLSVDDataConfig) -> list[Path]:
    """Local interaction files a config depends on, for pre-flight checks.

    Raises ``ValueError`` when no source is configured and ``FileNotFoundError``
    when ``interactions_root`` exists but holds no weekly split files. Hugging
    Face configs have no local dependencies and return an empty list.
    """
    if cfg.use_hf:
        return []
    return [path for _, path, _, _ in _resolve_paths_for_loading(cfg)]


def _first_vk_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out


def _vk_bool_series(series: pd.Series) -> pd.Series:
    """Parse a feedback flag into a plain boolean Series (missing -> False).

    VK-LSVD stores the flags as booleans; integer/float exports (0/1) and
    string exports (true/false-like tokens) are accepted as well.
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y", "1.0"]).fillna(False).astype(bool)


def _allowed_item_mask(raw_item: pd.Series, allowed_items: set[str] | np.ndarray) -> pd.Series:
    """Membership of ``raw_item`` in ``allowed_items``.

    Integer arrays (the VK-LSVD embedding catalog) are matched numerically so
    filtering never materializes 19.6M Python strings; string sets (semantic
    IDs) fall back to the string comparison shared with the other modules.
    """
    if isinstance(allowed_items, np.ndarray):
        numeric = pd.to_numeric(raw_item, errors="coerce")
        keep = numeric.notna().to_numpy() & np.isin(
            numeric.fillna(-1).to_numpy(dtype=np.int64), allowed_items
        )
        return pd.Series(keep, index=raw_item.index)
    return raw_item.astype("string").isin(allowed_items).fillna(False).astype(bool)


def _combine_allowed_items(
    semantic_items: set[str] | None,
    embedding_items: np.ndarray | set[str] | None,
) -> set[str] | np.ndarray | None:
    if semantic_items is None:
        return embedding_items
    if embedding_items is None:
        return semantic_items
    if isinstance(embedding_items, np.ndarray):
        numeric = pd.to_numeric(pd.Series(sorted(semantic_items), dtype="object"), errors="coerce")
        if numeric.notna().all():
            return np.intersect1d(numeric.to_numpy(dtype=np.int64), embedding_items)
        embedding_items = {str(int(x)) for x in embedding_items.tolist()}
    return semantic_items & embedding_items


def _load_vk_lsvd_file_frame(
    path: str | Path,
    *,
    split: VKSplit,
    week: int,
    cfg: VKLSVDDataConfig,
    max_rows: int | None,
    allowed_items: set[str] | np.ndarray | None,
) -> pd.DataFrame:
    columns = None if cfg.keep_raw_interactions else _VK_INPUT_COLUMNS
    frame = _read_vk_tabular_frame(path, max_rows=max_rows, columns=columns)
    return normalize_vk_lsvd_frame(
        frame,
        split=split,
        week=week,
        cfg=cfg,
        allowed_items=allowed_items,
    )


def normalize_vk_lsvd_frame(
    frame: pd.DataFrame,
    *,
    split: VKSplit = "train",
    week: int = 0,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Normalize one VK-LSVD table into the canonical interaction DataFrame.

    Native feedback (``timespent`` and the boolean flags) is preserved next to
    the derived ``rating`` so reward models and ``positive_criterion`` can use
    it. Files without a timestamp column (the upstream layout) get synthetic
    timestamps ``week * week_timestamp_stride + row index``, preserving the
    file order, which is chronological.
    """
    if frame.empty:
        return _empty_interaction_frame()

    row_idx = np.arange(len(frame), dtype=np.int64)
    source_order = int(week) * int(cfg.week_timestamp_stride) + row_idx
    raw_user = _first_vk_column(frame, *_VK_USER_COLUMNS)
    raw_item = _first_vk_column(frame, *_VK_ITEM_COLUMNS)
    timespent = (
        pd.to_numeric(_first_vk_column(frame, "timespent"), errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )
    flags = {key: _vk_bool_series(_first_vk_column(frame, key)) for key in VK_FEEDBACK_FLAGS}

    mask = raw_user.notna() & raw_item.notna()
    if cfg.min_timespent is not None:
        mask &= timespent >= float(cfg.min_timespent)
    if allowed_items is not None:
        mask &= _allowed_item_mask(raw_item, allowed_items)
    if cfg.require_positive_feedback:
        positive = timespent.gt(0)
        for key in VK_POSITIVE_FLAGS:
            positive |= flags[key]
        mask &= positive

    if not bool(mask.any()):
        return _empty_interaction_frame()

    synthetic_ts = pd.Series(source_order, index=frame.index)
    if any(column in frame.columns for column in _VK_TIMESTAMP_COLUMNS):
        raw_timestamp = pd.to_numeric(_first_vk_column(frame, *_VK_TIMESTAMP_COLUMNS), errors="coerce")
        timestamp = raw_timestamp.where(raw_timestamp.notna(), synthetic_ts).astype("int64")
    else:
        timestamp = synthetic_ts

    if cfg.rating_mode == "binary":
        computed_rating = pd.Series(float(cfg.default_rating), index=frame.index, dtype="float64")
    elif cfg.rating_mode == "timespent":
        computed_rating = timespent
    elif cfg.rating_mode == "weighted_feedback":
        score = pd.Series(0.0, index=frame.index, dtype="float64")
        for key, weight in cfg.feedback_weights.items():
            if key == "timespent":
                score += float(weight) * timespent
            elif key in flags:
                score += float(weight) * flags[key].astype(float)
            else:
                raise ValueError(f"Unknown feedback_weights key {key!r}; expected timespent or one of {VK_FEEDBACK_FLAGS}")
        computed_rating = score.where(score.ne(0.0), float(cfg.default_rating))
    else:
        raise ValueError(f"Unknown rating_mode: {cfg.rating_mode!r}")
    if "rating" in frame.columns:
        rating_from_col = pd.to_numeric(frame["rating"], errors="coerce")
        rating = rating_from_col.where(rating_from_col.notna(), computed_rating).astype("float64")
    else:
        rating = computed_rating.astype("float64")

    flow_split = _flow_split(split)
    mask_np = mask.to_numpy(dtype=bool)
    n_kept = int(mask_np.sum())
    out = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").astype(str).to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").astype(str).to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "order_idx": source_order[mask_np],
            "flowrec_split": np.array([flow_split] * n_kept, dtype=object),
            "flowrec_week": np.full(n_kept, int(week), dtype=np.int64),
            "timespent": timespent.loc[mask].to_numpy(dtype=float),
            **{key: flags[key].loc[mask].to_numpy(dtype=bool) for key in VK_FEEDBACK_FLAGS},
        }
    )
    if cfg.keep_raw_interactions:
        raw_records = frame.loc[mask].to_dict(orient="records")
        out["raw"] = [
            {
                **record,
                "_flowrec_split": flow_split,
                "_flowrec_week": int(week),
                "_flowrec_row_idx": int(order_idx),
            }
            for record, order_idx in zip(raw_records, source_order[mask_np], strict=False)
        ]
    return out.reset_index(drop=True)


def resolve_vk_lsvd_split_paths(cfg: VKLSVDDataConfig) -> dict[VKSplit, list[tuple[Path, int]]]:
    if cfg.use_hf:
        return _download_hf_split_paths(cfg)

    train_paths = _as_path_list(cfg.train_paths)
    val_paths = _as_path_list(cfg.val_paths)
    test_paths = _as_path_list(cfg.test_paths)

    if cfg.interactions_root is not None:
        root = Path(cfg.interactions_root)
        train_paths = train_paths or _glob_split_files(root, "train")
        val_paths = val_paths or _glob_split_files(root, "validation")
        test_paths = test_paths or _glob_split_files(root, "test")
        if not (train_paths or val_paths or test_paths):
            raise FileNotFoundError(
                f"No VK-LSVD interaction files were found under interactions_root={root}: "
                "expected <root>/{train,validation,test}/week_XX.parquet "
                "(or the same under <root>/interactions/)."
            )

    if not train_paths and cfg.interactions_path is not None:
        train_paths = _as_path_list(cfg.interactions_path)

    paths = {
        "train": [(path, _week_from_path(path, idx)) for idx, path in enumerate(train_paths)],
        "validation": [(path, _week_from_path(path, 25 + idx)) for idx, path in enumerate(val_paths)],
        "test": [(path, _week_from_path(path, 26 + idx)) for idx, path in enumerate(test_paths)],
    }
    if not any(paths.values()):
        raise ValueError(
            "No VK-LSVD interaction files were configured. Provide interactions_root, "
            "interactions_path, train/val/test paths, or set use_hf=True."
        )
    return paths


def _download_hf_split_paths(cfg: VKLSVDDataConfig) -> dict[VKSplit, list[tuple[Path, int]]]:
    from huggingface_hub import hf_hub_download

    def filename(split: VKSplit, week: int) -> str:
        if cfg.subsample_name:
            return f"subsamples/{cfg.subsample_name}/{split}/week_{week:02}.parquet"
        return f"interactions/{split}/week_{week:02}.parquet"

    out: dict[VKSplit, list[tuple[Path, int]]] = {"train": [], "validation": [], "test": []}
    for split, weeks in [("train", cfg.train_weeks), ("validation", cfg.val_weeks), ("test", cfg.test_weeks)]:
        for week in weeks:
            local = hf_hub_download(
                repo_id=cfg.hf_repo_id,
                repo_type="dataset",
                filename=filename(split, week),
                revision=cfg.hf_revision,
                local_dir=str(cfg.hf_local_dir) if cfg.hf_local_dir is not None else None,
            )
            out[split].append((Path(local), week))
    return out


def load_vk_lsvd_embedding_item_ids(path: str | Path) -> np.ndarray | set[str]:
    """Raw item IDs covered by an item-embedding artifact.

    VK-LSVD ``item_embeddings.npz`` yields a sorted int64 array (only the
    ``item_id`` array is read, never the embedding matrix). ItemEmbeddingStore
    ``.pt`` payloads and tabular files go through the shared Yambda reader and
    yield a string set.
    """
    path = Path(path)
    if _is_npz(path):
        with np.load(path) as data:
            if "item_id" not in data.files:
                raise ValueError(f"{path} has no item_id array; found {data.files}")
            return np.unique(data["item_id"].astype(np.int64))
    from .yambda import load_embedding_item_ids

    return load_embedding_item_ids(path)


def iter_vk_lsvd_file(
    path: str | Path,
    *,
    split: VKSplit = "train",
    week: int = 0,
    cfg: VKLSVDDataConfig,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    frame = _load_vk_lsvd_file_frame(
        path,
        split=split,
        week=week,
        cfg=cfg,
        max_rows=max_rows,
        allowed_items=None,
    )
    yield from frame.to_dict(orient="records")


def normalize_vk_lsvd_record(
    record: Mapping[str, Any],
    *,
    split: VKSplit = "train",
    week: int = 0,
    row_idx: int = 0,
    cfg: VKLSVDDataConfig,
) -> dict[str, Any] | None:
    """Normalize one VK-LSVD row for tests and edge callers."""
    frame = normalize_vk_lsvd_frame(
        pd.DataFrame([dict(record)]),
        split=split,
        week=week,
        cfg=cfg,
        allowed_items=None,
    )
    if frame.empty:
        return None
    out = frame.iloc[0].to_dict()
    out["order_idx"] = int(row_idx)
    return out


# ---------------------------------------------------------------------------
# Provided (weekly) split
# ---------------------------------------------------------------------------


def _first_split_timestamp(interactions: pd.DataFrame, split: VKFlowSplit) -> int | None:
    if interactions.empty or "flowrec_split" not in interactions.columns:
        return None
    values = interactions.loc[interactions["flowrec_split"].eq(split), "timestamp"]
    return int(values.min()) if not values.empty else None


def _validate_provided_split_order(events: pd.DataFrame, rank: pd.Series) -> None:
    """Every user's events must go train -> val -> test in sequence order."""
    unknown = sorted(set(events.loc[rank.isna(), "flowrec_split"].astype(str)))
    if unknown:
        raise ValueError(
            f"Unknown flowrec_split labels {unknown}; expected {sorted(_SPLIT_RANK)}."
        )
    steps = rank.groupby(events["user_id"], sort=False).diff().fillna(0)
    if bool(steps.lt(0).any()):
        offenders = events.loc[steps.lt(0), "raw_user_id"].astype(str).unique()[:5].tolist()
        raise ValueError(
            "VK-LSVD provided split labels contradict the chronological order: "
            "some users have train rows after validation/test rows (e.g. users "
            f"{offenders}). Check the timestamp column or the week numbering of the files."
        )


def build_provided_sequence_splits(interactions: pd.DataFrame, *, cfg: VKLSVDDataConfig) -> SequenceSplits:
    """Build train/val/test examples from VK-LSVD provided split labels.

    The train-week rows of each user form a chronological prefix of the user's
    sequence (validated); validation and test rows follow. Train targets are
    selected with the shared train-target policies (prefix_target) or windowed
    with the shared dense-causal builder over exactly that prefix, so the
    output schema is identical to the leave-one-out/global-time splits.
    Validation/test targets are the val/test-week rows, with rolling history
    (or history cut at the week start under ``global_history_policy=
    "window_start"``) and ``global_eval_target_mode`` choosing every row or
    the last per user.
    """
    split_cfg = _sequence_split_config_from_vk(cfg, split_strategy="global_time")
    dense = split_cfg.train_example_layout == "dense_causal"
    if interactions.empty:
        return SequenceSplits(
            train=_empty_dense_frame() if dense else _empty_sequence_frame(),
            val=_empty_sequence_frame(),
            test=_empty_sequence_frame(),
        )
    if "flowrec_split" not in interactions.columns:
        raise KeyError("Provided VK split construction requires a 'flowrec_split' column")

    events = _prepare_events(interactions)
    rank = events["flowrec_split"].map(_SPLIT_RANK)
    _validate_provided_split_order(events, rank)
    rank = rank.astype("int64")
    is_train = rank.eq(0)
    is_val = rank.eq(1)
    is_test = rank.eq(2)
    user_groups = events["user_id"]
    # Train rows occupy positions [0, train_end); validation rows
    # [train_end, test_start); test rows [test_start, user_len).
    events["train_end_exclusive"] = is_train.astype("int64").groupby(user_groups, sort=False).transform("sum")
    events["test_start_pos"] = rank.lt(2).astype("int64").groupby(user_groups, sort=False).transform("sum")

    if dense:
        train = build_dense_train_examples(events, cfg=split_cfg, region=is_train)
    else:
        train_targets = _select_train_rows(
            events.loc[is_train].copy(), end_col="train_end_exclusive", cfg=split_cfg
        )
        train = _build_examples_from_targets(train_targets, events, split="train", cfg=split_cfg)

    val_targets = _select_eval_rows(events.loc[is_val].copy(), mode=cfg.global_eval_target_mode)
    if not val_targets.empty:
        val_targets["history_end_idx"] = (
            val_targets["train_end_exclusive"]
            if cfg.global_history_policy == "window_start"
            else val_targets["position"]
        ).astype("int64")
    val = _build_examples_from_targets(val_targets, events, split="val", cfg=split_cfg)

    test_targets = _select_eval_rows(events.loc[is_test].copy(), mode=cfg.global_eval_target_mode)
    if not test_targets.empty:
        test_targets["history_end_idx"] = (
            test_targets["test_start_pos"]
            if cfg.global_history_policy == "window_start"
            else test_targets["position"]
        ).astype("int64")
    test = _build_examples_from_targets(test_targets, events, split="test", cfg=split_cfg)

    return SequenceSplits(
        train=train,
        val=val,
        test=test,
        val_cutoff=_first_split_timestamp(events, "val"),
        test_cutoff=_first_split_timestamp(events, "test"),
    )


# Compatibility alias with the old public builder name. It now returns split DataFrames.
def build_provided_gts_examples(
    interactions: pd.DataFrame,
    *,
    cfg: VKLSVDDataConfig,
    **_: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = build_provided_sequence_splits(interactions, cfg=cfg)
    return splits.train, splits.val, splits.test
