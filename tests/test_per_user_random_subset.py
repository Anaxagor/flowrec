"""Per-user RANDOM fine-tuning subset (precompute_reward_cache.per_user_random_indices).

Used where the full train layout is too expensive (yambda_500m: 15 of each
user's last-128 targets). Requirements: every process that builds the same data
draws the same examples (reward-cache rows = positions in the subset, and the
SFT-only control must see the GFlow arm's examples), and - unlike a last-N
tail - the draw keeps the layout's mix of target positions."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "scripts" / "exp2-tuned-params")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import precompute_reward_cache as dense  # noqa: E402
import precompute_reward_factors as factors  # noqa: E402
import sft_finetune_control as control  # noqa: E402


class _Dataset:
    def __init__(self, sizes: list[int]) -> None:
        users = np.repeat(np.arange(len(sizes)), sizes)
        self.examples = pd.DataFrame({"user_id": users, "position": np.concatenate([np.arange(s) for s in sizes])})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {"value": int(idx)}


class TestPerUserRandomIndices:
    def test_n_per_user_in_dataset_order(self) -> None:
        ds = _Dataset([128, 9, 40, 15])
        idx = dense.per_user_random_indices(ds, 15, seed=42)
        assert idx == sorted(idx) and len(set(idx)) == len(idx)
        counts = ds.examples.loc[idx, "user_id"].value_counts().to_dict()
        assert counts == {0: 15, 1: 9, 2: 15, 3: 15}  # users with fewer examples keep them all

    def test_deterministic_and_seeded(self) -> None:
        ds = _Dataset([128] * 20)
        assert dense.per_user_random_indices(ds, 15, 42) == dense.per_user_random_indices(_Dataset([128] * 20), 15, 42)
        assert dense.per_user_random_indices(ds, 15, 42) != dense.per_user_random_indices(ds, 15, 43)

    def test_keeps_the_mix_of_target_positions(self) -> None:
        """Every one of a user's 128 targets is equally likely: the selected
        positions average the middle of the window, a tail averages its end."""
        ds = _Dataset([128] * 400)
        random_positions = ds.examples.loc[dense.per_user_random_indices(ds, 15, 0), "position"]
        tail_positions = ds.examples.loc[dense.per_user_tail_indices(ds, 15), "position"]
        assert abs(random_positions.mean() - 63.5) < 3.0
        assert tail_positions.mean() == pytest.approx(120.0)
        assert random_positions.min() < 5 and random_positions.max() > 122

    def test_requires_a_positive_n_and_a_user_column(self) -> None:
        with pytest.raises(ValueError):
            dense.per_user_random_indices(_Dataset([5]), 0, 0)
        with pytest.raises(KeyError):
            dense.per_user_random_indices(object(), 3, 0)

    def test_tail_and_random_subsets_are_exclusive(self) -> None:
        ds = _Dataset([20, 20])
        assert dense.subset_indices(ds) is None
        assert dense.subset_indices(ds, per_user_last_n=3) == dense.per_user_tail_indices(ds, 3)
        assert (dense.subset_indices(ds, per_user_random_n=3, per_user_random_seed=5)
                == dense.per_user_random_indices(ds, 3, 5))
        with pytest.raises(ValueError, match="exclusive"):
            dense.subset_indices(ds, per_user_last_n=3, per_user_random_n=3)


class TestSubsetOnTheRealLayout:
    """The subset is drawn from the split the base SFT checkpoint trained on
    (leave-one-out, last-N targets, bounded history), so every sampled example
    must be a target of that window with the items right before it as history."""

    LAST_N, MAX_HIST, PER_USER = 8, 5, 3

    def _layout(self):
        from flowrec.data.sequence_builder import SequenceSplitConfig, build_sequence_splits
        from flowrec.data.datasets import SequenceDataset

        rng = np.random.default_rng(0)
        lengths = {"short": 6, "mid": 12, "long": 40}           # train windows of 3, 8 and 8 targets
        rows = [
            {"raw_user_id": user, "raw_item_id": f"i{int(rng.integers(0, 30))}", "timestamp": 1000 + t,
             "rating": 1.0, "user_id": uid + 1, "item_id": 0, "position": t}
            for uid, (user, n) in enumerate(lengths.items()) for t in range(n)
        ]
        frame = pd.DataFrame(rows)
        frame["item_id"] = frame["raw_item_id"].astype("category").cat.codes.astype("int64") + 1
        cfg = SequenceSplitConfig(split_strategy="leave_one_out", min_history_len=1, train_target_policy="last_n",
                                  train_last_n=self.LAST_N, max_history_len=self.MAX_HIST, train_all_prefixes=True)
        splits = build_sequence_splits(frame, cfg=cfg)
        sequences = {u: g.sort_values("timestamp")["raw_item_id"].tolist() for u, g in frame.groupby("raw_user_id")}
        return SequenceDataset(splits.train, split="train"), sequences

    def test_targets_come_from_the_last_n_window_with_the_preceding_items_as_history(self) -> None:
        dataset, sequences = self._layout()
        indices = dense.per_user_random_indices(dataset, self.PER_USER, seed=42)
        examples = dataset.examples.iloc[indices]
        assert examples.groupby("raw_user_id").size().to_dict() == {"short": 3, "mid": 3, "long": 3}
        for row in examples.itertuples():
            seq = sequences[row.raw_user_id]
            end = len(seq) - 2                                    # val / test targets are the last two items
            pos = int(row.target_position)
            assert max(1, end - self.LAST_N) <= pos < end         # inside the base layout's last-N window
            assert row.target_raw_item == seq[pos]
            assert list(row.history_raw_items) == seq[max(0, pos - self.MAX_HIST):pos]
        # the long user's oldest window slots still get a full history: it reaches back BEFORE the window
        long_rows = dataset.examples[dataset.examples["raw_user_id"] == "long"]
        assert long_rows["target_position"].tolist() == list(range(30, 38))
        assert all(len(h) == self.MAX_HIST for h in long_rows["history_raw_items"])

    def test_train_all_prefixes_does_not_widen_a_last_n_layout(self) -> None:
        # the 500M base config carries train_all_prefixes=true next to train_target_policy=last_n
        dataset, _ = self._layout()
        assert dataset.examples.groupby("raw_user_id").size().to_dict() == {"short": 3, "mid": 8, "long": 8}


def test_both_cache_builders_take_the_subset_options(tmp_path) -> None:
    common = ["--config", str(tmp_path / "c.json"), "--output-dir", str(tmp_path / "o"),
              "--per-user-random-n", "15", "--per-user-random-seed", "42", "--per-user-pool-last-n", "128"]
    for module in (dense, factors):
        args = module.parse_args(common)
        assert (args.per_user_random_n, args.per_user_random_seed, args.per_user_last_n) == (15, 42, None)
        assert args.per_user_pool_last_n == 128
        assert module.parse_args(common[:4]).per_user_random_n is None


def test_cache_loader_keeps_original_indices_when_sampling_from_recent_pool():
    from types import SimpleNamespace

    dataset = _Dataset([260, 10, 180])
    data = SimpleNamespace(train_dataset=dataset, val_dataset=None, test_dataset=None,
                           _make_collator=lambda: lambda rows: rows)
    spec = dict(per_user_pool_last_n=128, per_user_random_n=15, per_user_random_seed=42)
    loader = dense.build_cache_loader(data, "train", 16, 0, **spec)
    seen = [row["value"] for batch in loader for row in batch]
    assert seen == dense.subset_indices(dataset, **spec)
    assert len(seen) == 40
    assert set(seen) <= set([*range(132, 260), *range(260, 270), *range(322, 450)])


class TestControlSubsetLoader:
    """sft_finetune_control.seeded_train_loader over a subset: same loader
    settings as the datamodule's own train loader, only fewer examples."""

    @dataclass
    class _Cfg:
        seed: int = 0
        batch_size: int = 4
        num_workers: int = 0
        pin_memory: bool = False
        persistent_workers: bool = False
        train_drop_last: bool = False

    class _DM:
        def __init__(self) -> None:
            self.cfg = TestControlSubsetLoader._Cfg()
            self.train_dataset = _Dataset([10, 10])
            self.calls = 0

        def _make_train_collator(self):
            return lambda rows: {"value": torch.tensor([r["value"] for r in rows])}

        def _loader_parts(self, dataset, collator):
            return dataset, collator

        def train_dataloader(self):
            self.calls += 1
            return "full-loader"

    def test_without_indices_the_datamodule_loader_is_used(self) -> None:
        dm = self._DM()
        assert control.seeded_train_loader(dm, 43) == "full-loader" and dm.cfg.seed == 43

    def test_with_indices_only_those_examples_are_served(self) -> None:
        dm = self._DM()
        indices = [1, 4, 7, 12, 15, 19]
        loader = control.seeded_train_loader(dm, 42, indices)
        seen = sorted(int(v) for batch in loader for v in batch["value"])
        assert seen == indices and dm.calls == 0 and dm.cfg.seed == 42
        assert loader.batch_size == 4
        # the run seed changes the ORDER, never the examples
        order = lambda seed: [int(v) for b in control.seeded_train_loader(dm, seed, indices) for v in b["value"]]  # noqa: E731
        assert order(42) == order(42) and sorted(order(43)) == indices and order(42) != order(43)
