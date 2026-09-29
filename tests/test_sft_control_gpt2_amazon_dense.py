"""The GPT-2 dense-layout SFT-only control differs from the full-history
(prefix-layout) control in the training layout only."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp1-standard-params"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_gpt2_amazon_fullhist as fullhist  # noqa: E402
import sft_control_gpt2_amazon_dense as dense  # noqa: E402

PROTOCOL = json.loads((PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json").read_text())


class TestSameProtocolAsThePrefixControl:
    def test_every_schedule_and_evaluation_setting_is_the_full_history_one(self) -> None:
        d = dense.parse_args(["--datasets", "beauty"])
        f = fullhist.parse_args(["--sft-only", "--datasets", "beauty"])
        for key in ("lr", "weight_decay", "max_grad_norm", "amp_dtype", "max_epochs", "validate_every",
                    "patience", "min_epochs", "validate_at_start", "val_batch_size", "test_batch_size",
                    "eval_num_beams", "test_ks", "batch_size", "train_last_n", "resume", "seed"):
            assert getattr(d, key) == getattr(f, key), key
        assert d.sft_only is True
        assert tuple(dense.CONTROL_SEEDS) == tuple(PROTOCOL["shared_schedule"]["seeds_sft_only"])

    def test_seed_reaches_the_base_parser(self) -> None:
        """The launcher passes --seed N; it must not be swallowed as an abbreviation of --seeds."""
        a = dense.parse_args(["--datasets", "instruments", "--seed", "43"])
        assert a.seed == 43 and a.seeds is None and a.datasets == ["instruments"]
        b = dense.parse_args(["--seeds", "42", "44", "--targets-per-step", "1426"])
        assert b.seeds == [42, 44] and b.targets_per_step == 1426

    def test_outputs_never_mix_with_the_full_history_experiment(self) -> None:
        d = dense.parse_args([])
        f = fullhist.parse_args(["--sft-only"])
        assert str(d.output_root) != str(f.output_root)
        assert str(dense.base.metrics_csv_path(d)) != str(dense.base.metrics_csv_path(f))
        arm = PROTOCOL["arms"][dense.ARM]
        assert str(d.output_root) == arm["output_root"] and arm["diagnostic"] is True
        assert (PROJECT_ROOT / arm["script"]).exists() and (PROJECT_ROOT / arm["submit-worker"]).exists()
        assert dense.VARIANT != "sft_only"  # rows stay distinct when the two CSVs are concatenated

    def test_yambda_full_and_last128_have_independent_outputs(self, tmp_path):
        full = dense.parse_args(["--datasets", "yambda_50m_listens"])
        tail = dense.parse_args(["--datasets", "yambda_50m_listens", "--train-last-n", "128"])
        assert tail.train_last_n == 128 and full.train_last_n == 0
        assert str(tail.output_root) == f"{full.output_root}-last128"
        assert tail.metrics_csv != full.metrics_csv and "last128" in tail.metrics_csv.name
        assert dense.run_dir_for(tail.output_root, "yambda_50m_listens", 42) != dense.run_dir_for(
            full.output_root, "yambda_50m_listens", 42)
        custom = dense.parse_args(["--datasets", "yambda_50m_listens", "--train-last-n=128",
                                   f"--output-root={tmp_path / 'runs'}", "--metrics-csv", str(tmp_path / 'metrics.csv')])
        assert custom.output_root == tmp_path / "runs" and custom.metrics_csv == tmp_path / "metrics.csv"

    def test_init_checkpoint_is_the_full_history_experiments(self) -> None:
        """No checkpoint path of its own: the base resolver is the only source."""
        source = Path(dense.__file__).read_text()
        assert "base.tuned_checkpoint_paths(dataset)" in source
        assert "initial/gpt2" not in source.split('"""', 2)[2]  # only the docstring names the path

    def test_run_config_differs_in_the_training_layout_only(self) -> None:
        args = dense.parse_args(["--datasets", "beauty"])
        prefix = dense.base.build_run_config("beauty", args, 1)
        cfg = dense.dense_run_config("beauty", args, 1)
        assert cfg["dataset"].pop("train_example_layout") == "dense_causal"
        assert prefix["dataset"].pop("train_example_layout") == "prefix_target"
        assert cfg["dataset"] == prefix["dataset"]
        assert cfg["data"] == prefix["data"] and cfg["dataloader"] == prefix["dataloader"]
        assert cfg["model"] == prefix["model"]
        assert cfg["method"] == {"name": "sft"}


class TestDenseBatch:
    def test_beauty_step_sees_about_the_prefix_controls_targets(self) -> None:
        b = dense.dense_batch_settings(num_rows=23_607, num_targets=131_413, micro_batch=32,
                                       targets_per_step=2048)
        assert b["micro_batch_rows"] == 32
        assert abs(b["expected_targets_per_step"] - 2048) / 2048 < 0.05
        assert b["rows_per_step"] == 32 * b["grad_accum"]
        assert b["steps_per_epoch"] == int(np.ceil(23_607 / b["rows_per_step"]))

    def test_the_closer_accumulation_wins_and_overrides_work(self) -> None:
        # 4 targets per row, micro 32 -> 128 targets per micro-batch: 2048 = exactly 16
        assert dense.dense_batch_settings(num_rows=1000, num_targets=4000, micro_batch=32,
                                          targets_per_step=2048)["grad_accum"] == 16
        # the base run's own ~256-row step
        b = dense.dense_batch_settings(num_rows=23_607, num_targets=131_413, micro_batch=32,
                                       targets_per_step=1426)
        assert b["grad_accum"] == 8 and b["rows_per_step"] == 256
        assert dense.dense_batch_settings(num_rows=10, num_targets=50, micro_batch=32,
                                          targets_per_step=2048, grad_accum=3)["grad_accum"] == 3
        with pytest.raises(ValueError):
            dense.dense_batch_settings(num_rows=0, num_targets=0, micro_batch=32, targets_per_step=2048)

    def test_target_count_reads_the_label_mask(self) -> None:
        class _DS:
            examples = pd.DataFrame({
                "label_item_ids": [[5, 6, 7], [8, 9]],
                "label_mask": [[False, True, True], [True, True]],
            })
        assert dense.count_dense_targets(_DS()) == 4


class TestYambdaDenseTargets:
    @pytest.mark.parametrize("dataset", ["beauty", "instruments", "yambda_50m_listens"])
    def test_preprocessing_and_target_scope_match_the_base_gpt2_run(self, dataset):
        path = dense.base.dataset_yaml_path(dataset)
        runner = dense.base.gpt2_runner
        group = next(g for g in runner.GROUPS if g.data_config_name == path.name)
        reference, _ = runner.build_data_config(
            group, dataset, dense.base.runner.load_yaml(path), seed=42,
            semantic_ids_dir=dense.base.SEMANTIC_IDS_ROOT, user_hash_mod=1)
        args = dense.dense_dataset_args(dense.parse_args(["--datasets", dataset]), dataset)
        config = dense.dense_run_config(dataset, args, 1)
        actual = dense.base.tiger_runner.build_data_config(config, config_path=PROJECT_ROOT / "audit.json")
        allowed = {"batch_size", "eval_batch_size", "num_workers", "persistent_workers", "history_padding"}
        assert {k: v for k, v in asdict(actual).items() if k not in allowed} == {
            k: v for k, v in asdict(reference).items() if k not in allowed}
        assert actual.train_example_layout == "dense_causal"
        assert args.train_pool_last_n is None and args.train_last_n == 0 and args.train_random_n == 0

    @pytest.mark.parametrize("dataset,last_n", [("yambda_50m_listens", 0), ("yambda_50m_listens", 128),
                                                ("yambda_500m_listens", 0)])
    def test_masks_match_prefix_targets_and_preserve_dense_contexts(self, dataset, last_n):
        from flowrec.data.datasets import SequenceDataset
        from flowrec.data.dense_prefix import prefixes_from_dense_windows
        from flowrec.data.sequence_builder import SequenceSplitConfig, build_sequence_splits, factorize_interaction_ids

        events = pd.DataFrame([
            dict(raw_user_id=str(user), raw_item_id=str(i % 31), timestamp=i,
                 rating=1.0, verified_purchase=None, order_idx=i)
            for user, n in enumerate([305, 54, 9]) for i in range(n)
        ])
        events, _, _, item2id, _ = factorize_interaction_ids(events)
        splits = build_sequence_splits(events, cfg=SequenceSplitConfig(
            train_example_layout="dense_causal", max_history_len=50))
        original = splits.train.copy(deep=True)
        dm = SimpleNamespace(train_dataset=SequenceDataset(splits.train, "train"),
                             val_dataset=SequenceDataset(splits.val, "val"),
                             test_dataset=SequenceDataset(splits.test, "test"),
                             item2id=item2id.copy(), stats=SimpleNamespace())
        args = dense.dense_dataset_args(dense.parse_args(
            ["--datasets", dataset, "--train-last-n", str(last_n)]), dataset)
        prefix = SequenceDataset(prefixes_from_dense_windows(
            original, events, last_n=args.train_pool_last_n), "train")
        chosen = prefix.examples.iloc[dense.base.finetune_indices(prefix, args)]
        expected = {(r.user_id, r.target_position, tuple(r.history_item_ids), r.target_item_id)
                    for r in chosen.itertuples()}
        dense.select_dense_targets(dm, args)
        actual = set()
        for row in dm.train_dataset.examples.itertuples():
            source = original.loc[original.user_id.eq(row.user_id) &
                                  original.target_position.eq(row.target_position)].iloc[0]
            assert row.history_item_ids == source.history_item_ids
            assert row.label_item_ids == source.label_item_ids
            assert any(row.label_mask)
            for i, active in enumerate(row.label_mask):
                if active:
                    actual.add((row.user_id, row.target_position - len(row.label_mask) + i + 1,
                                tuple(row.history_item_ids[:i + 1]), row.label_item_ids[i]))
        assert actual == expected
        assert dense.count_dense_targets(dm.train_dataset) == len(expected)
        assert dm.item2id == item2id
        pd.testing.assert_frame_equal(dm.val_dataset.examples, splits.val)
        pd.testing.assert_frame_equal(dm.test_dataset.examples, splits.test)
        if dataset == "yambda_50m_listens" and last_n == 0:
            pd.testing.assert_frame_equal(dm.train_dataset.examples, original)
            assert len(expected) > 128 + 54 + 9  # older targets are supervised too
        else:
            assert dm.stats.num_train_examples == len(dm.train_dataset)
            if dataset == "yambda_50m_listens":
                assert len(expected) == 128 + (54 - 3) + (9 - 3)

    def test_dataset_defaults_do_not_leak_and_dense_microbatch_is_bounded(self):
        args = dense.parse_args(["--datasets", "beauty", "instruments", "yambda_50m_listens"])
        large = dense.dense_dataset_args(args, "yambda_50m_listens")
        assert large.train_pool_last_n is None and large.train_random_n == 0 and large.train_last_n == 0
        assert large.batch_size == 16
        small = dense.dense_dataset_args(large, "beauty")
        assert small.train_pool_last_n is None and small.batch_size == 32
        explicit = dense.parse_args(["--datasets", "yambda_50m_listens", "--batch-size", "8"])
        assert dense.dense_dataset_args(explicit, "yambda_50m_listens").batch_size == 8

    def test_old_last128_runs_cannot_be_skipped_or_resumed_as_full_base_data(self, tmp_path):
        args = dense.dense_dataset_args(dense.parse_args(["--datasets", "yambda_50m_listens"]),
                                        "yambda_50m_listens")
        config = tmp_path / "resolved_config.json"
        config.write_text(json.dumps({"data": {"finetune_per_user_pool_last_n": 128}}))
        with pytest.raises(RuntimeError, match="saved target selection differs"):
            dense.check_existing_target_selection(tmp_path, args)
        config.write_text(json.dumps({"data": {"finetune_per_user_pool_last_n": None}}))
        dense.check_existing_target_selection(tmp_path, args)


class TestPrefixControlCheck:
    def _args(self, tmp_path: Path, smoke: bool = False):
        a = dense.parse_args(["--datasets", "beauty", "--prefix-output-root", str(tmp_path)])
        a.smoke = smoke
        return a

    def _write_prefix_record(self, tmp_path: Path, *, targets: int, init: str) -> None:
        run_dir = dense.base.sft_control_run_dir(tmp_path, "beauty", 42)
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(json.dumps(
            {"init_checkpoint": init, "data": {"num_finetune_examples": targets}}))

    def test_accepts_the_same_setup_under_another_checkout_prefix(self, tmp_path: Path) -> None:
        ours = dense.base.tuned_checkpoint_paths("beauty")["checkpoint"]
        rel = ours.relative_to(PROJECT_ROOT).as_posix()
        self._write_prefix_record(tmp_path, targets=131_413, init="." + rel)
        out = dense.check_against_prefix_control("beauty", num_targets=131_413, init_checkpoint=ours,
                                                 args=self._args(tmp_path))
        assert out["checked"] is True

    def test_refuses_another_target_count_or_checkpoint(self, tmp_path: Path) -> None:
        ours = dense.base.tuned_checkpoint_paths("beauty")["checkpoint"]
        self._write_prefix_record(tmp_path, targets=95_061, init="/x/checkpoints/other/best.pt")
        with pytest.raises(RuntimeError, match="supervised targets"):
            dense.check_against_prefix_control("beauty", num_targets=131_413, init_checkpoint=ours,
                                               args=self._args(tmp_path))

    def test_missing_prefix_control_is_reported_not_fatal(self, tmp_path: Path) -> None:
        ours = dense.base.tuned_checkpoint_paths("beauty")["checkpoint"]
        out = dense.check_against_prefix_control("beauty", num_targets=1, init_checkpoint=ours,
                                                 args=self._args(tmp_path))
        assert out == {"checked": False}
