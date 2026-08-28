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
from .preprocessors import (
    apply_k_core_filter_frame,
    count_interactions,
    load_semantic_ids,
    sort_interactions_frame,
)
from .post_split_filtering import (
    remap_interactions_to_catalog,
    ColdItemTrainSource,
    PostSplitColdItemFilterStats,
    disabled_post_split_filter_stats,
    filter_splits_to_train_item_catalog,
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
    build_sequence_splits,
    factorize_interaction_ids,
    validate_dense_layout_config,
)

VKSplit = Literal["train", "validation", "test"]
VKFlowSplit = Literal["train", "val", "test"]
VKSplitStrategy = Literal["provided", "leave_one_out", "global_time"]
VKRatingMode = Literal["binary", "timespent", "weighted_feedback"]

_CANONICAL_INTERACTION_COLUMNS = [
    "raw_user_id",
    "raw_item_id",
    "timestamp",
    "rating",
    "verified_purchase",
    "order_idx",
]


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


def _read_vk_tabular_frame(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read one VK-LSVD table through pandas.

    Parquet is the canonical VK-LSVD storage format. CSV/TSV/JSONL are retained
    as strict compatibility inputs.
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
    raise ValueError(f"Unsupported VK-LSVD interaction table format: {path}")


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
    candidates = [root / "interactions" / split, root / split]
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

    VK-LSVD is normally stored as Parquet files under train/validation/test
    directories. The loader normalizes every file into a canonical interaction
    DataFrame with raw_user_id, raw_item_id, timestamp, rating, and split marker
    columns, then all downstream processing remains DataFrame-based.
    """

    # Local loading.
    interactions_root: str | Path | None = None
    interactions_path: str | Path | Sequence[str | Path] | None = None
    train_paths: Sequence[str | Path] | None = None
    val_paths: Sequence[str | Path] | None = None
    test_paths: Sequence[str | Path] | None = None

    # Optional semantic IDs.
    semantic_id_path: str | Path | None = None

    # Hugging Face loading.
    use_hf: bool = False
    hf_repo_id: str = "deepvk/VK-LSVD"
    hf_revision: str | None = None
    hf_local_dir: str | Path | None = None
    subsample_name: str | None = None
    train_weeks: tuple[int, ...] = tuple(range(25))
    val_weeks: tuple[int, ...] = (25,)
    test_weeks: tuple[int, ...] = (26,)

    # Row limits are useful for smoke tests and small experiments.
    max_rows: int | None = None
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None

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

    # Splitting. "provided" uses VK-LSVD's train/validation/test directories.
    split_strategy: VKSplitStrategy = "provided"
    min_history_len: int = 1

    # Training example materialization. "prefix_target" keeps the historical
    # one-row-per-(prefix, target) layout; "dense_causal" emits one row per
    # user window with per-position labels for decoder-only-style training.
    # dense_causal supports left or right padding and is only available for the
    # leave_one_out / global_time strategies — the "provided" split defines
    # its train region by files and has no dense builder yet.
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
    global_history_policy: GlobalHistoryPolicy = "rolling"
    global_eval_target_mode: GlobalEvalTargetMode = "all"

    # Synthetic timestamp construction when files do not contain timestamp.
    week_timestamp_stride: int = 10**12

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


class VKLSVDDataModule:
    """Framework-agnostic DataModule for VK-LSVD.

    The public batch format is unchanged, but interactions, k-core filtering,
    split construction, train-catalog filtering, and datasets are all DataFrames.
    """

    def __init__(self, cfg: VKLSVDDataConfig) -> None:
        self.cfg = cfg
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
        if self.semantic_id_path is not None and not self.semantic_id_path.exists():
            raise FileNotFoundError(f"semantic_id_path does not exist: {self.semantic_id_path}")
        if self.cfg.use_hf:
            return
        if self.cfg.interactions_path is None and self.cfg.interactions_root is None:
            if not (self.cfg.train_paths or self.cfg.val_paths or self.cfg.test_paths):
                raise ValueError(
                    "Provide interactions_path, interactions_root, train/val/test paths, or set use_hf=True."
                )

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return
        validate_dense_layout_config(
            layout=self.cfg.train_example_layout,
            history_padding=self.cfg.history_padding,
        )
        if (
            self.cfg.train_example_layout == "dense_causal"
            and self.cfg.split_strategy == "provided"
        ):
            raise NotImplementedError(
                "train_example_layout='dense_causal' is not supported with the "
                "'provided' split strategy: the provided split defines its train "
                "region by files and has no dense window builder. Use "
                "split_strategy='leave_one_out' or 'global_time'."
            )
        self.prepare_data()

        if self.semantic_id_path is not None:
            self.semantic_ids, self.sid_len = load_semantic_ids(
                self.semantic_id_path,
                offset_by_one=self.cfg.offset_semantic_ids_by_one,
            )

        allowed_items = None
        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            allowed_items = set(self.semantic_ids)

        interactions = load_vk_lsvd_interactions_frame(cfg=self.cfg, allowed_items=allowed_items)
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
            collate_fn=self._make_train_collator(),
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

    def _make_train_collator(self) -> SequenceBatchCollator | DenseCausalCollator:
        """Training collator; evaluation always uses the prefix-target collator."""
        if self.cfg.train_example_layout == "dense_causal":
            window_len = self.cfg.train_window_len or self.cfg.max_history_len
            return DenseCausalCollator(
                max_seq_len=window_len,
                history_padding=self.cfg.history_padding,
                pad_item_id=self.cfg.pad_item_id,
                include_raw=self.cfg.include_raw_in_batch,
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
                "dataset": "vk_lsvd",
                "split_strategy": self.cfg.split_strategy,
                "subsample_name": self.cfg.subsample_name,
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


def load_vk_lsvd_interactions_frame(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | None = None,
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
def load_vk_lsvd_events_by_user(*, cfg: VKLSVDDataConfig, allowed_items: set[str] | None = None) -> pd.DataFrame:
    return load_vk_lsvd_interactions_frame(cfg=cfg, allowed_items=allowed_items)


def load_vk_lsvd_events_by_split(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | None = None,
) -> dict[VKFlowSplit, pd.DataFrame]:
    """Load VK-LSVD's provided split files as split DataFrames."""
    split_paths = resolve_vk_lsvd_split_paths(cfg)
    max_rows_by_split = {
        "train": cfg.max_train_rows if cfg.max_train_rows is not None else cfg.max_rows,
        "validation": cfg.max_val_rows if cfg.max_val_rows is not None else cfg.max_rows,
        "test": cfg.max_test_rows if cfg.max_test_rows is not None else cfg.max_rows,
    }
    out: dict[VKFlowSplit, pd.DataFrame] = {"train": _empty_interaction_frame(), "val": _empty_interaction_frame(), "test": _empty_interaction_frame()}
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


def _resolve_paths_for_loading(cfg: VKLSVDDataConfig) -> list[tuple[VKSplit, Path, int, int | None]]:
    if cfg.split_strategy == "provided":
        split_paths = resolve_vk_lsvd_split_paths(cfg)
        max_rows_by_split = {
            "train": cfg.max_train_rows if cfg.max_train_rows is not None else cfg.max_rows,
            "validation": cfg.max_val_rows if cfg.max_val_rows is not None else cfg.max_rows,
            "test": cfg.max_test_rows if cfg.max_test_rows is not None else cfg.max_rows,
        }
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


def _first_vk_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out


def _vk_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y"])


def _load_vk_lsvd_file_frame(
    path: str | Path,
    *,
    split: VKSplit,
    week: int,
    cfg: VKLSVDDataConfig,
    max_rows: int | None,
    allowed_items: set[str] | None,
) -> pd.DataFrame:
    frame = _read_vk_tabular_frame(path, max_rows=max_rows)
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
    allowed_items: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize one VK-LSVD table into the canonical interaction DataFrame."""
    if frame.empty:
        out = _empty_interaction_frame()
        out["flowrec_split"] = pd.Series(dtype="object")
        out["flowrec_week"] = pd.Series(dtype="int64")
        return out

    row_idx = np.arange(len(frame), dtype=np.int64)
    source_order = int(week) * int(cfg.week_timestamp_stride) + row_idx
    raw_user = _first_vk_column(frame, "user_id", "uid", "raw_user_id")
    raw_item = _first_vk_column(frame, "item_id", "video_id", "raw_item_id")
    timespent = pd.to_numeric(_first_vk_column(frame, "timespent"), errors="coerce").fillna(0.0)

    mask = raw_user.notna() & raw_item.notna()
    if cfg.min_timespent is not None:
        mask &= timespent >= float(cfg.min_timespent)
    if allowed_items is not None:
        mask &= raw_item.astype("string").isin(allowed_items)
    if cfg.require_positive_feedback:
        positive = timespent.gt(0)
        for key in ["like", "share", "bookmark", "click_on_author", "open_comments"]:
            positive |= _vk_bool_series(_first_vk_column(frame, key))
        mask &= positive

    if not bool(mask.any()):
        out = _empty_interaction_frame()
        out["flowrec_split"] = pd.Series(dtype="object")
        out["flowrec_week"] = pd.Series(dtype="int64")
        return out

    raw_timestamp = _first_vk_column(frame, "timestamp", "ts")
    timestamp = pd.to_numeric(raw_timestamp, errors="coerce")
    synthetic_ts = int(week) * int(cfg.week_timestamp_stride) + row_idx
    timestamp = timestamp.fillna(pd.Series(synthetic_ts, index=frame.index)).astype("int64")

    rating_from_col = pd.to_numeric(_first_vk_column(frame, "rating"), errors="coerce")
    if cfg.rating_mode == "binary":
        computed_rating = pd.Series(float(cfg.default_rating), index=frame.index, dtype="float64")
    elif cfg.rating_mode == "timespent":
        computed_rating = timespent.astype("float64")
    else:
        score = pd.Series(0.0, index=frame.index, dtype="float64")
        for key, weight in cfg.feedback_weights.items():
            if key == "timespent":
                score += float(weight) * timespent
            else:
                score += float(weight) * _vk_bool_series(_first_vk_column(frame, key)).astype(float)
        computed_rating = score.where(score.ne(0.0), float(cfg.default_rating))
    rating = rating_from_col.where(rating_from_col.notna(), computed_rating).astype("float64")

    flow_split = _flow_split(split)
    out = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").astype(str).to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").astype(str).to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "verified_purchase": np.array([None] * int(mask.sum()), dtype=object),
            "order_idx": source_order[mask.to_numpy()],
            "flowrec_split": np.array([flow_split] * int(mask.sum()), dtype=object),
            "flowrec_week": np.array([int(week)] * int(mask.sum()), dtype=np.int64),
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
            for record, order_idx in zip(raw_records, source_order[mask.to_numpy()], strict=False)
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

    if not train_paths and cfg.interactions_path is not None:
        train_paths = _as_path_list(cfg.interactions_path)

    paths = {
        "train": [(path, _week_from_path(path, idx)) for idx, path in enumerate(train_paths)],
        "validation": [(path, _week_from_path(path, 25 + idx)) for idx, path in enumerate(val_paths)],
        "test": [(path, _week_from_path(path, 26 + idx)) for idx, path in enumerate(test_paths)],
    }
    if not any(paths.values()):
        raise ValueError("No VK-LSVD interaction files were found.")
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


def _rating_from_vk_lsvd_record(record: Mapping[str, Any], *, cfg: VKLSVDDataConfig) -> float:
    if "rating" in record and record["rating"] is not None:
        rating = pd.to_numeric(pd.Series([record["rating"]]), errors="coerce").iloc[0]
        if not pd.isna(rating):
            return float(rating)

    if cfg.rating_mode == "binary":
        return float(cfg.default_rating)

    timespent = _safe_float(record.get("timespent"), default=0.0)
    if cfg.rating_mode == "timespent":
        return timespent

    score = 0.0
    for key, weight in cfg.feedback_weights.items():
        if key == "timespent":
            score += float(weight) * timespent
        else:
            score += float(weight) * float(_bool_value(record.get(key)))
    return float(score if score != 0.0 else cfg.default_rating)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default) if pd.isna(numeric) else float(numeric)


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(value) if pd.isna(numeric) else bool(int(numeric))


def _has_positive_feedback(record: Mapping[str, Any]) -> bool:
    if _bool_value(record.get("like")):
        return True
    if _bool_value(record.get("share")):
        return True
    if _bool_value(record.get("bookmark")):
        return True
    if _bool_value(record.get("click_on_author")):
        return True
    if _bool_value(record.get("open_comments")):
        return True
    return _safe_float(record.get("timespent"), default=0.0) > 0


def _first_split_timestamp(interactions: pd.DataFrame, split: VKFlowSplit) -> int | None:
    if interactions.empty or "flowrec_split" not in interactions.columns:
        return None
    values = interactions.loc[interactions["flowrec_split"].eq(split), "timestamp"]
    return int(values.min()) if not values.empty else None


def build_provided_sequence_splits(interactions: pd.DataFrame, *, cfg: VKLSVDDataConfig) -> SequenceSplits:
    """Build train/val/test examples from VK-LSVD provided split labels.

    Target row selection is label-based and vectorized. History construction is
    delegated to the shared DataFrame sequence builder, so the output schema is
    identical to leave-one-out/global-time splits.
    """
    if interactions.empty:
        split_cfg = _sequence_split_config_from_vk(cfg, split_strategy="global_time")
        empty = _build_examples_from_targets(interactions, interactions, split="train", cfg=split_cfg)
        return SequenceSplits(train=empty, val=empty.copy(), test=empty.copy())
    if "flowrec_split" not in interactions.columns:
        raise KeyError("Provided VK split construction requires a 'flowrec_split' column")

    events = sort_interactions_frame(interactions).copy()
    if "verified_purchase" not in events.columns:
        events["verified_purchase"] = None
    events["raw_user_id"] = events["raw_user_id"].astype(str)
    events["raw_item_id"] = events["raw_item_id"].astype(str)
    events["user_id"] = events["user_id"].astype("int64")
    events["item_id"] = events["item_id"].astype("int64")
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce").fillna(0).astype("int64")
    events["rating"] = pd.to_numeric(events["rating"], errors="coerce").fillna(0.0).astype(float)
    events["position"] = events.groupby("user_id", sort=False).cumcount().astype("int64")
    events["user_len"] = events.groupby("user_id", sort=False)["position"].transform("size").astype("int64")

    split_cfg = _sequence_split_config_from_vk(cfg, split_strategy="global_time")
    min_history = int(cfg.min_history_len)

    train_candidates = events.loc[events["flowrec_split"].eq("train") & events["position"].ge(min_history)].copy()
    train_targets = _select_provided_train_rows(train_candidates, cfg=cfg)
    train_targets["history_end_idx"] = train_targets["position"] if not train_targets.empty else pd.Series(dtype="int64")
    train = _build_examples_from_targets(train_targets, events, split="train", cfg=split_cfg)

    val_candidates = events.loc[events["flowrec_split"].eq("val") & events["position"].ge(min_history)].copy()
    val_targets = _select_provided_eval_rows(val_candidates, mode=cfg.global_eval_target_mode)
    val_targets = val_targets.copy()
    if cfg.global_history_policy == "window_start" and not val_targets.empty:
        first_val = val_candidates.groupby("user_id", sort=False)["position"].min()
        val_targets["history_end_idx"] = val_targets["user_id"].map(first_val).astype("int64")
    elif not val_targets.empty:
        val_targets["history_end_idx"] = val_targets["position"]
    val = _build_examples_from_targets(val_targets, events, split="val", cfg=split_cfg)

    test_candidates = events.loc[events["flowrec_split"].eq("test") & events["position"].ge(min_history)].copy()
    test_targets = _select_provided_eval_rows(test_candidates, mode=cfg.global_eval_target_mode)
    test_targets = test_targets.copy()
    if cfg.global_history_policy == "window_start" and not test_targets.empty:
        first_test = test_candidates.groupby("user_id", sort=False)["position"].min()
        test_targets["history_end_idx"] = test_targets["user_id"].map(first_test).astype("int64")
    elif not test_targets.empty:
        test_targets["history_end_idx"] = test_targets["position"]
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


def _effective_train_target_policy(cfg: VKLSVDDataConfig) -> TrainTargetPolicy:
    if cfg.train_target_policy == "all" and not cfg.train_all_prefixes:
        return "last"
    return cfg.train_target_policy


def _select_provided_train_rows(candidates: pd.DataFrame, *, cfg: VKLSVDDataConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["user_id", "position"], kind="stable").copy()
    policy = _effective_train_target_policy(cfg)

    if policy == "all":
        return candidates
    if policy == "last":
        rank_desc = candidates.groupby("user_id", sort=False)["position"].rank(method="first", ascending=False)
        return candidates.loc[rank_desc.eq(1)].copy()
    if policy == "last_n":
        rank_desc = candidates.groupby("user_id", sort=False)["position"].rank(method="first", ascending=False)
        return candidates.loc[rank_desc.le(int(cfg.train_last_n))].copy().sort_values(["user_id", "position"], kind="stable")
    if policy == "stride":
        order = candidates.groupby("user_id", sort=False).cumcount()
        return candidates.loc[(order % max(int(cfg.train_stride), 1)).eq(0)].copy()
    if policy == "sampled":
        return _sample_rows_per_user(candidates, n=int(cfg.train_targets_per_user), seed=int(cfg.seed))
    if policy == "hybrid":
        rank_desc = candidates.groupby("user_id", sort=False)["position"].rank(method="first", ascending=False)
        recent = candidates.loc[rank_desc.le(int(cfg.train_hybrid_recent_n))]
        older = candidates.loc[rank_desc.gt(int(cfg.train_hybrid_recent_n))]
        sampled_old = _sample_rows_per_user(older, n=int(cfg.train_hybrid_random_n), seed=int(cfg.seed))
        return pd.concat([sampled_old, recent], ignore_index=False).sort_values(["user_id", "position"], kind="stable")
    raise ValueError(f"Unknown train_target_policy: {policy}")


def _sample_rows_per_user(frame: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    rng = np.random.default_rng(seed)
    out["__rand"] = rng.random(len(out))
    out["__rank"] = out.groupby("user_id", sort=False)["__rand"].rank(method="first")
    sampled = out.loc[out["__rank"].le(int(n))].drop(columns=["__rand", "__rank"])
    return sampled.sort_values(["user_id", "position"], kind="stable")


def _select_provided_eval_rows(frame: pd.DataFrame, *, mode: GlobalEvalTargetMode) -> pd.DataFrame:
    if frame.empty:
        return frame
    if mode == "all":
        return frame
    if mode == "last_per_user":
        last_pos = frame.groupby("user_id", sort=False)["position"].transform("max")
        return frame.loc[frame["position"].eq(last_pos)].copy()
    raise ValueError(f"Unknown global_eval_target_mode: {mode}")
