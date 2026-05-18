from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .collators import HistoryPadding, SequenceBatchCollator
from .datasets import DatasetStats, InteractionRecord, SequenceDataset, SequenceExample
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
    build_sequence_splits,
    make_sequence_example,
    select_eval_indices,
)

VKSplit = Literal["train", "validation", "test"]
VKFlowSplit = Literal["train", "val", "test"]
VKSplitStrategy = Literal["provided", "leave_one_out", "global_time"]
VKRatingMode = Literal["binary", "timespent", "weighted_feedback"]


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

    VK-LSVD interaction files are Parquet. JSON/JSONL support is included for
    tests, small local samples, and debugging.
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


def _as_path_list(value: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(x) for x in value]


def _glob_split_files(root: str | Path, split: VKSplit) -> list[Path]:
    root = Path(root)
    candidates = [
        root / "interactions" / split,
        root / split,
    ]
    for directory in candidates:
        if directory.exists() and directory.is_dir():
            return sorted(directory.glob("week_*.parquet"))
    return []


def _week_from_path(path: str | Path, fallback: int = 0) -> int:
    stem = Path(path).stem
    if stem.startswith("week_"):
        try:
            return int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return fallback
    return fallback


def _flow_split(split: VKSplit) -> VKFlowSplit:
    return "val" if split == "validation" else split  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class VKLSVDDataConfig:
    """Configuration for VK-LSVD sequential recommendation data.

    VK-LSVD is stored as globally time-ordered Parquet files. The canonical
    layout is:

        interactions/train/week_00.parquet ... week_24.parquet
        interactions/validation/week_25.parquet
        interactions/test/week_26.parquet

    Subsamples use the same split/week layout under ``subsamples/<name>/``.
    """

    # Local loading.
    interactions_root: str | Path | None = None
    interactions_path: str | Path | Sequence[str | Path] | None = None
    train_paths: Sequence[str | Path] | None = None
    val_paths: Sequence[str | Path] | None = None
    test_paths: Sequence[str | Path] | None = None

    # Optional semantic IDs.
    semantic_id_path: str | Path | None = None

    # Hugging Face loading. We use hf_hub_download instead of datasets because
    # the repository is very large and does not expose a lightweight viewer.
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
    train_all_prefixes: bool = True
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

    # IDs.
    pad_item_id: int = 0
    pad_token_id: int = 0
    offset_semantic_ids_by_one: bool = True
    drop_items_missing_semantic_id: bool = True

    seed: int = 42


class VKLSVDDataModule:
    """Framework-agnostic DataModule for VK-LSVD.

    It emits the same FlowRec batch keys as AmazonReviewDataModule and
    YambdaDataModule:

        history_item_ids, history_mask, target_item_ids, target_ratings, ...

    If semantic IDs are provided, batches also contain:

        history_token_ids, history_token_mask, target_token_ids, target_token_mask
    """

    def __init__(self, cfg: VKLSVDDataConfig) -> None:
        self.cfg = cfg
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
        self.prepare_data()

        if self.semantic_id_path is not None:
            self.semantic_ids, self.sid_len = load_semantic_ids(
                self.semantic_id_path,
                offset_by_one=self.cfg.offset_semantic_ids_by_one,
            )

        allowed_items = None
        if self.semantic_ids is not None and self.cfg.drop_items_missing_semantic_id:
            allowed_items = set(self.semantic_ids)

        if self.cfg.split_strategy == "provided":
            split_events = load_vk_lsvd_events_by_split(
                cfg=self.cfg,
                allowed_items=allowed_items,
            )
            user_events = _merge_split_events(split_events)
        else:
            user_events = load_vk_lsvd_events_by_user(
                cfg=self.cfg,
                allowed_items=allowed_items,
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
        if self.cfg.split_strategy == "provided":
            train, val, test = build_provided_gts_examples(
                user_events,
                get_user_id=self._get_or_add_user,
                get_item_id=self._get_or_add_item,
                cfg=self.cfg,
            )
            self.global_val_cutoff = _first_split_timestamp(user_events, "val")
            self.global_test_cutoff = _first_split_timestamp(user_events, "test")
        else:
            split_cfg = SequenceSplitConfig(
                split_strategy=self.cfg.split_strategy,  # type: ignore[arg-type]
                min_history_len=self.cfg.min_history_len,
                train_all_prefixes=self.cfg.train_all_prefixes,
                global_val_timestamp=self.cfg.global_val_timestamp,
                global_test_timestamp=self.cfg.global_test_timestamp,
                global_val_ratio=self.cfg.global_val_ratio,
                global_test_ratio=self.cfg.global_test_ratio,
                global_history_policy=self.cfg.global_history_policy,
                global_eval_target_mode=self.cfg.global_eval_target_mode,
            )
            splits = build_sequence_splits(
                user_events,
                get_user_id=self._get_or_add_user,
                get_item_id=self._get_or_add_item,
                cfg=split_cfg,
            )
            train, val, test = splits.train, splits.val, splits.test
            self.global_val_cutoff = splits.val_cutoff
            self.global_test_cutoff = splits.test_cutoff

        self.train_dataset = SequenceDataset(train, split="train")
        self.val_dataset = SequenceDataset(val, split="val")
        self.test_dataset = SequenceDataset(test, split="test")
        self.stats = DatasetStats(
            num_users=self.num_users,
            num_items=self.num_items,
            num_train_examples=len(train),
            num_val_examples=len(val),
            num_test_examples=len(test),
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
                "dataset": "vk_lsvd",
                "split_strategy": self.cfg.split_strategy,
                "subsample_name": self.cfg.subsample_name,
                "global_val_cutoff": self.global_val_cutoff,
                "global_test_cutoff": self.global_test_cutoff,
                "sid_len": self.sid_len,
                "item_vocab_size": self.item_vocab_size,
            }
        )
        return out


def load_vk_lsvd_events_by_user(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | None = None,
) -> dict[str, list[InteractionRecord]]:
    """Load VK-LSVD interactions grouped by raw user ID.

    This path is used when FlowRec should rebuild a leave-one-out or global-time
    split from one interaction collection.
    """
    if cfg.interactions_path is not None:
        paths = [("train", p, _week_from_path(p, 0)) for p in _as_path_list(cfg.interactions_path)]
    else:
        split_paths = resolve_vk_lsvd_split_paths(cfg)
        paths = [(split, path, week) for split, items in split_paths.items() for path, week in items]

    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
    for split, path, week in paths:
        for event in iter_vk_lsvd_file(
            path,
            split=split,
            week=week,
            cfg=cfg,
            max_rows=cfg.max_rows,
        ):
            if allowed_items is not None and event.raw_item_id not in allowed_items:
                continue
            grouped[event.raw_user_id].append(event)
    return dict(grouped)


def load_vk_lsvd_events_by_split(
    *,
    cfg: VKLSVDDataConfig,
    allowed_items: set[str] | None = None,
) -> dict[VKFlowSplit, dict[str, list[InteractionRecord]]]:
    """Load VK-LSVD's provided train/validation/test global temporal split."""
    split_paths = resolve_vk_lsvd_split_paths(cfg)
    out: dict[VKFlowSplit, dict[str, list[InteractionRecord]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    max_rows_by_split = {
        "train": cfg.max_train_rows if cfg.max_train_rows is not None else cfg.max_rows,
        "validation": cfg.max_val_rows if cfg.max_val_rows is not None else cfg.max_rows,
        "test": cfg.max_test_rows if cfg.max_test_rows is not None else cfg.max_rows,
    }
    for split, paths in split_paths.items():
        flow_split = _flow_split(split)
        for path, week in paths:
            for event in iter_vk_lsvd_file(
                path,
                split=split,
                week=week,
                cfg=cfg,
                max_rows=max_rows_by_split[split],
            ):
                if allowed_items is not None and event.raw_item_id not in allowed_items:
                    continue
                out[flow_split][event.raw_user_id].append(event)
    return {split: dict(grouped) for split, grouped in out.items()}


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
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading VK-LSVD from Hugging Face requires huggingface_hub. "
            "Install with: pip install huggingface_hub"
        ) from exc

    def filename(split: VKSplit, week: int) -> str:
        if cfg.subsample_name:
            return f"subsamples/{cfg.subsample_name}/{split}/week_{week:02}.parquet"
        return f"interactions/{split}/week_{week:02}.parquet"

    out: dict[VKSplit, list[tuple[Path, int]]] = {"train": [], "validation": [], "test": []}
    for split, weeks in [
        ("train", cfg.train_weeks),
        ("validation", cfg.val_weeks),
        ("test", cfg.test_weeks),
    ]:
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
) -> Iterator[InteractionRecord]:
    for row_idx, record in enumerate(iter_tabular_records(path, max_rows=max_rows)):
        event = normalize_vk_lsvd_record(
            record,
            split=split,
            week=week,
            row_idx=row_idx,
            cfg=cfg,
        )
        if event is not None:
            yield event


def normalize_vk_lsvd_record(
    record: Mapping[str, Any],
    *,
    split: VKSplit = "train",
    week: int = 0,
    row_idx: int = 0,
    cfg: VKLSVDDataConfig,
) -> InteractionRecord | None:
    user_id = record.get("user_id", record.get("uid", record.get("raw_user_id")))
    item_id = record.get("item_id", record.get("video_id", record.get("raw_item_id")))
    if user_id is None or item_id is None:
        return None
    
    timespent = _safe_float(record.get("timespent"), default=0.0)
    if cfg.min_timespent is not None and timespent < float(cfg.min_timespent):
        return None

    if cfg.require_positive_feedback and not _has_positive_feedback(record):
        return None

    timestamp = record.get("timestamp", record.get("ts", None))
    if timestamp is None:
        timestamp = int(week) * int(cfg.week_timestamp_stride) + int(row_idx)

    raw: dict[str, Any]
    if cfg.keep_raw_interactions:
        raw = dict(record)
    else:
        raw = {}
    raw["_flowrec_split"] = _flow_split(split)
    raw["_flowrec_week"] = int(week)
    raw["_flowrec_row_idx"] = int(row_idx)

    return InteractionRecord(
        raw_user_id=str(user_id),
        raw_item_id=str(item_id),
        timestamp=int(timestamp),
        rating=_rating_from_vk_lsvd_record(record, cfg=cfg),
        verified_purchase=None,
        order_idx=int(row_idx),
        raw=raw,
    )


def _rating_from_vk_lsvd_record(record: Mapping[str, Any], *, cfg: VKLSVDDataConfig) -> float:
    if "rating" in record and record["rating"] is not None:
        try:
            return float(record["rating"])
        except (TypeError, ValueError):
            pass

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
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


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


def _merge_split_events(
    split_events: dict[VKFlowSplit, dict[str, list[InteractionRecord]]]
) -> dict[str, list[InteractionRecord]]:
    merged: dict[str, list[InteractionRecord]] = defaultdict(list)
    for grouped in split_events.values():
        for user, events in grouped.items():
            merged[user].extend(events)
    return dict(merged)


def _event_split(event: InteractionRecord) -> VKFlowSplit:
    value = event.raw.get("_flowrec_split")
    if value in {"train", "val", "test"}:
        return value  # type: ignore[return-value]
    # If split metadata is missing, infer from timestamp cutoffs is impossible
    # here. Treat as train to avoid leaking into validation/test.
    return "train"


def _first_split_timestamp(
    user_events: dict[str, list[InteractionRecord]], split: VKFlowSplit
) -> int | None:
    values = [event.timestamp for events in user_events.values() for event in events if _event_split(event) == split]
    return min(values) if values else None


from typing import Any


def build_provided_gts_examples(
    user_events: dict[str, list[InteractionRecord]],
    *,
    get_user_id,
    get_item_id,
    cfg: VKLSVDDataConfig,
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
   

    train_examples: list[SequenceExample] = []
    val_examples: list[SequenceExample] = []
    test_examples: list[SequenceExample] = []

    min_history_len = cfg.min_history_len
    min_required_len = min_history_len + 1
    train_all_prefixes = cfg.train_all_prefixes
    eval_mode = cfg.global_eval_target_mode
    rolling_history = cfg.global_history_policy == "rolling"

    event_split = _event_split
    make = make_sequence_example

    train_append = train_examples.append
    val_append = val_examples.append
    test_append = test_examples.append

    # Global item cache helps because the same item appears across many users.
    item_id_cache: dict[Any, Any] = {}

    def cached_get_item_id(raw_item_id):
        try:
            return item_id_cache[raw_item_id]
        except KeyError:
            item_id = get_item_id(raw_item_id)
            item_id_cache[raw_item_id] = item_id
            return item_id
        except TypeError:
            # Fallback for unhashable raw_item_id values.
            return get_item_id(raw_item_id)


    if getattr(cfg, "events_are_sorted", False):
        user_iter = user_events.items()
    else:
        user_iter = sort_user_events(user_events).items()

    for raw_user_id, events in user_iter:
        n_events = len(events)
        if n_events < min_required_len:
            continue

        user_id = get_user_id(raw_user_id)

        raw_items: list[Any] = [None] * n_events
        item_ids: list[Any] = [None] * n_events

        # Store compact split codes:
        #   0 = train
        #   1 = val
        #   2 = test
        #   3 = unknown / ignored
        split_codes = bytearray(n_events)

        first_val_idx = n_events
        first_test_idx = n_events
        first_non_train_idx = n_events

        split_order_is_monotonic = True
        prev_code = 0

        # Single pass:
        #   - build raw_items
        #   - build item_ids
        #   - detect split boundaries
        #   - detect whether split labels are ordered
        for i, event in enumerate(events):
            raw_item_id = event.raw_item_id
            raw_items[i] = raw_item_id
            item_ids[i] = cached_get_item_id(raw_item_id)

            split = event_split(event)

            if split == "train":
                code = 0
            elif split == "val":
                code = 1
            elif split == "test":
                code = 2
            else:
                code = 3

            split_codes[i] = code

            if code < prev_code:
                split_order_is_monotonic = False
            if code <= 2:
                prev_code = max(prev_code, code)

            if code != 0 and first_non_train_idx == n_events:
                first_non_train_idx = i

            if code == 1 and first_val_idx == n_events:
                first_val_idx = i

            if code == 2 and first_test_idx == n_events:
                first_test_idx = i

        if split_order_is_monotonic:
            _append_ordered_provided_split_examples(
                raw_user_id=raw_user_id,
                user_id=user_id,
                raw_items=raw_items,
                item_ids=item_ids,
                events=events,
                n_events=n_events,
                min_history_len=min_history_len,
                first_non_train_idx=first_non_train_idx,
                first_val_idx=first_val_idx,
                first_test_idx=first_test_idx,
                train_all_prefixes=train_all_prefixes,
                eval_mode=eval_mode,
                rolling_history=rolling_history,
                make=make,
                train_append=train_append,
                val_append=val_append,
                test_append=test_append,
            )
        else:
            _append_unordered_provided_split_examples(
                raw_user_id=raw_user_id,
                user_id=user_id,
                raw_items=raw_items,
                item_ids=item_ids,
                events=events,
                split_codes=split_codes,
                n_events=n_events,
                min_history_len=min_history_len,
                train_all_prefixes=train_all_prefixes,
                eval_mode=eval_mode,
                rolling_history=rolling_history,
                make=make,
                train_append=train_append,
                val_append=val_append,
                test_append=test_append,
            )

    return train_examples, val_examples, test_examples
def _append_ordered_provided_split_examples(
    *,
    raw_user_id,
    user_id,
    raw_items,
    item_ids,
    events,
    n_events: int,
    min_history_len: int,
    first_non_train_idx: int,
    first_val_idx: int,
    first_test_idx: int,
    train_all_prefixes: bool,
    eval_mode: str,
    rolling_history: bool,
    make,
    train_append,
    val_append,
    test_append,
) -> None:
    """
    Fast path for official provided temporal splits.

    Expected order per user:
        train ... train, val ... val, test ... test
    """

    # -------------------------
    # Train
    # -------------------------
    train_end = first_non_train_idx

    if train_all_prefixes:
        train_indices = range(min_history_len, train_end)
    else:
        last_train = train_end - 1
        train_indices = (last_train,) if last_train >= min_history_len else ()

    for target_idx in train_indices:
        train_append(
            make(
                raw_user_id=raw_user_id,
                user_id=user_id,
                raw_items=raw_items,
                item_ids=item_ids,
                events=events,
                target_idx=target_idx,
                split="train",
                history_end_idx=target_idx,
            )
        )

    # -------------------------
    # Validation
    # -------------------------
    if first_val_idx < n_events:
        val_start = max(first_val_idx, min_history_len)
        val_end = first_test_idx if first_test_idx < n_events else n_events

        val_indices = _select_eval_range_as_iterable(
            start=val_start,
            end=val_end,
            mode=eval_mode,
        )

        if rolling_history:
            for target_idx in val_indices:
                val_append(
                    make(
                        raw_user_id=raw_user_id,
                        user_id=user_id,
                        raw_items=raw_items,
                        item_ids=item_ids,
                        events=events,
                        target_idx=target_idx,
                        split="val",
                        history_end_idx=target_idx,
                    )
                )
        else:
            val_history_end_idx = first_val_idx
            if val_history_end_idx >= min_history_len:
                for target_idx in val_indices:
                    val_append(
                        make(
                            raw_user_id=raw_user_id,
                            user_id=user_id,
                            raw_items=raw_items,
                            item_ids=item_ids,
                            events=events,
                            target_idx=target_idx,
                            split="val",
                            history_end_idx=val_history_end_idx,
                        )
                    )

    # -------------------------
    # Test
    # -------------------------
    if first_test_idx < n_events:
        test_start = max(first_test_idx, min_history_len)
        test_end = n_events

        test_indices = _select_eval_range_as_iterable(
            start=test_start,
            end=test_end,
            mode=eval_mode,
        )

        if rolling_history:
            for target_idx in test_indices:
                test_append(
                    make(
                        raw_user_id=raw_user_id,
                        user_id=user_id,
                        raw_items=raw_items,
                        item_ids=item_ids,
                        events=events,
                        target_idx=target_idx,
                        split="test",
                        history_end_idx=target_idx,
                    )
                )
        else:
            test_history_end_idx = first_test_idx
            if test_history_end_idx >= min_history_len:
                for target_idx in test_indices:
                    test_append(
                        make(
                            raw_user_id=raw_user_id,
                            user_id=user_id,
                            raw_items=raw_items,
                            item_ids=item_ids,
                            events=events,
                            target_idx=target_idx,
                            split="test",
                            history_end_idx=test_history_end_idx,
                        )
                    )
def _append_unordered_provided_split_examples(
    *,
    raw_user_id,
    user_id,
    raw_items,
    item_ids,
    events,
    split_codes: bytearray,
    n_events: int,
    min_history_len: int,
    train_all_prefixes: bool,
    eval_mode: str,
    rolling_history: bool,
    make,
    train_append,
    val_append,
    test_append,
) -> None:
    """
    Slower but robust fallback for cases where provided split labels are not
    ordered within a user's event list.
    """

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    first_val_idx = n_events
    first_test_idx = n_events
    last_train_idx = -1

    for i, code in enumerate(split_codes):
        if i < min_history_len:
            continue

        if code == 0:
            if train_all_prefixes:
                train_indices.append(i)
            else:
                last_train_idx = i

        elif code == 1:
            if first_val_idx == n_events:
                first_val_idx = i
            val_indices.append(i)

        elif code == 2:
            if first_test_idx == n_events:
                first_test_idx = i
            test_indices.append(i)

    if not train_all_prefixes:
        train_indices = [last_train_idx] if last_train_idx >= min_history_len else []

    val_indices = _select_eval_candidates_as_list(val_indices, mode=eval_mode)
    test_indices = _select_eval_candidates_as_list(test_indices, mode=eval_mode)

    for target_idx in train_indices:
        train_append(
            make(
                raw_user_id=raw_user_id,
                user_id=user_id,
                raw_items=raw_items,
                item_ids=item_ids,
                events=events,
                target_idx=target_idx,
                split="train",
                history_end_idx=target_idx,
            )
        )

    if rolling_history:
        for target_idx in val_indices:
            val_append(
                make(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=target_idx,
                    split="val",
                    history_end_idx=target_idx,
                )
            )

        for target_idx in test_indices:
            test_append(
                make(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=target_idx,
                    split="test",
                    history_end_idx=target_idx,
                )
            )
    else:
        if first_val_idx >= min_history_len:
            for target_idx in val_indices:
                val_append(
                    make(
                        raw_user_id=raw_user_id,
                        user_id=user_id,
                        raw_items=raw_items,
                        item_ids=item_ids,
                        events=events,
                        target_idx=target_idx,
                        split="val",
                        history_end_idx=first_val_idx,
                    )
                )

        if first_test_idx >= min_history_len:
            for target_idx in test_indices:
                test_append(
                    make(
                        raw_user_id=raw_user_id,
                        user_id=user_id,
                        raw_items=raw_items,
                        item_ids=item_ids,
                        events=events,
                        target_idx=target_idx,
                        split="test",
                        history_end_idx=first_test_idx,
                    )
                )
def _select_eval_range_as_iterable(
    *,
    start: int,
    end: int,
    mode: str,
) -> Iterable[int]:
    if end <= start:
        return ()

    if mode == "all":
        return range(start, end)

    if mode == "last_per_user":
        return (end - 1,)

    raise ValueError(f"Unknown global_eval_target_mode: {mode}")


def _select_eval_candidates_as_list(
    candidates: list[int],
    *,
    mode: str,
) -> list[int]:
    if not candidates:
        return []

    if mode == "all":
        return candidates

    if mode == "last_per_user":
        return [candidates[-1]]

    raise ValueError(f"Unknown global_eval_target_mode: {mode}")