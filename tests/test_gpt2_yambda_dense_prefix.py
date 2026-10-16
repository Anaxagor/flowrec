"""Dense-window GPT-2 fine-tuning: contexts, held-out data and cache identity."""
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_yambda as experiment
from flowrec.data.dense_prefix import dense_prefix_signature, prefixes_from_dense_windows
from flowrec.data.sequence_builder import SequenceSplitConfig, build_sequence_splits, factorize_interaction_ids
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule


def events_for(lengths):
    rows = [dict(raw_user_id=str(user), raw_item_id=str(i % 61), timestamp=1000 + i,
                 rating=float(i % 5), verified_purchase=bool(i % 2), order_idx=i)
            for user, length in enumerate(lengths) for i in range(length)]
    return factorize_interaction_ids(pd.DataFrame(rows))[0]


@pytest.mark.parametrize("window,stride", [(50, None), (50, 16), (50, 1), (3, 3)])
@pytest.mark.parametrize("last_n", [None, 128, 15])
def test_prefixes_match_every_selected_active_dense_label(window, stride, last_n):
    events = events_for([5, 53, 205, 270])
    cfg = SequenceSplitConfig(train_example_layout="dense_causal", max_history_len=window,
                              train_window_stride=stride, min_history_len=2)
    dense = build_sequence_splits(events, cfg=cfg).train
    prefixes = prefixes_from_dense_windows(dense, events, last_n=last_n)
    expected = []
    for row in dense.itertuples():
        for i, active in enumerate(row.label_mask):
            if active:
                expected.append((row.user_id, row.target_position - len(row.label_mask) + i + 1,
                                 row.label_item_ids[i], tuple(row.history_item_ids[:i + 1])))
    expected.sort()
    if last_n is not None:
        expected = [entry for entry in expected
                    if entry[1] > int(events.loc[events.user_id.eq(entry[0]), "position"].max()) - 2 - last_n]
    actual = [(r.user_id, r.target_position, r.target_item_id, tuple(r.history_item_ids))
              for r in prefixes.itertuples()]
    assert actual == expected
    assert len({(r[0], r[1]) for r in actual}) == len(actual)
    for r in prefixes.itertuples():
        assert r.target_timestamp == 1000 + r.target_position
        assert r.target_rating == float(r.target_position % 5)
        assert r.target_verified_purchase == bool(r.target_position % 2)


def test_dense_and_expanded_prefixes_have_same_gpt2_loss_and_gradients():
    from flowrec.data.collators import DenseCausalCollator, SequenceBatchCollator
    from flowrec.ft_methods.sft import SFTConfig, SFTMethod
    from flowrec.models.gpt2 import GPT2RecModel

    torch.manual_seed(42)
    events = events_for([9, 14])
    dense = build_sequence_splits(events, cfg=SequenceSplitConfig(
        train_example_layout="dense_causal", max_history_len=3, train_window_stride=2)).train
    prefixes = prefixes_from_dense_windows(dense, events)
    sids = {str(i): [i // 8 + 1, i % 8 + 1] for i in range(61)}
    dense_batch = DenseCausalCollator(max_seq_len=3, semantic_ids=sids, sid_len=2,
                                     history_padding="right")(dense.to_dict("records"))
    prefix_batch = SequenceBatchCollator(max_history_len=3, semantic_ids=sids, sid_len=2,
                                         history_padding="left")(prefixes.to_dict("records"))
    model = GPT2RecModel(codebook_size=16, sem_id_len=2, semantic_id_mapping=sids,
                         embedding_dim=8, num_heads=2, num_layers=2, dim_feedforward=16,
                         max_position_embeddings=16, dropout=0.0).eval()
    method = SFTMethod(SFTConfig(compute_train_metrics=False))
    dense_loss = method.compute_loss(model, dense_batch).loss
    dense_loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    prefix_loss = method.compute_loss(model, prefix_batch).loss
    prefix_loss.backward()
    torch.testing.assert_close(dense_loss, prefix_loss)
    for name, p in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad, rtol=2e-4, atol=2e-6)


def make_datamodule(tmp_path, *, prefix):
    rows = [dict(uid=user, item_id=i % 61, timestamp=1000 + i, played_ratio_pct=100)
            for user, length in enumerate([5, 53, 205, 270]) for i in range(length)]
    # One cold held-out target; another held-out target appears only in OLD
    # training windows, proving the catalog must precede last-128 selection.
    rows[-1]["item_id"] = 999
    rows[0]["item_id"] = 888
    rows[3]["item_id"] = 888
    path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(path)
    sid_path = tmp_path / "sids.json"
    sid_path.write_text(json.dumps({str(i): [i // 32, i % 32] for i in [*range(61), 888, 999]}))
    cfg = YambdaDataConfig(
        interactions_path=path, semantic_id_path=sid_path, hf_recent_months=None,
        min_user_interactions=3, min_item_interactions=0,
        train_example_layout="prefix_target" if prefix else "dense_causal",
        train_prefix_from_dense=prefix, train_target_policy="last_n", train_last_n=128,
        max_history_len=50, num_workers=0, persistent_workers=False,
        include_raw_in_batch=False, batch_size=4,
    )
    dm = YambdaDataModule(cfg)
    dm.setup()
    return dm


def test_independent_datamodules_have_identical_full_eval_examples_and_catalog(tmp_path):
    dense = make_datamodule(tmp_path, prefix=False)
    prefix = make_datamodule(tmp_path, prefix=True)
    assert prefix.item2id == dense.item2id and prefix.semantic_ids == dense.semantic_ids
    assert "999" not in prefix.item2id and "888" in prefix.item2id
    for name in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(prefix, name).examples, getattr(dense, name).examples)
    for dense_loader, prefix_loader in [(dense.val_dataloader(), prefix.val_dataloader()),
                                         (dense.test_dataloader(), prefix.test_dataloader())]:
        for left, right in zip(dense_loader, prefix_loader):
            for key in left:
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key])
    assert prefix.train_dataset.examples.groupby("user_id").size().tolist() == [2, 50, 128, 128]
    assert "history_token_ids" in next(iter(prefix.train_dataloader()))
    assert prefix.dense_prefix_audit["validation"]["equal"]
    # The legacy sliding-prefix path still has its original contexts.
    legacy = YambdaDataModule(replace(prefix.cfg, train_prefix_from_dense=False, train_target_policy="all"))
    legacy.setup()
    pd.testing.assert_frame_equal(legacy.val_dataset.examples, dense.val_dataset.examples)
    pd.testing.assert_frame_equal(legacy.test_dataset.examples, dense.test_dataset.examples)
    assert legacy.train_dataset.examples.history_item_ids.str.len().max() == 50
    assert prefix.train_dataset.examples.history_item_ids.str.len().min() == 1
    # Last-128 and random-15 pick the same USER/TARGET pairs as before; only
    # histories change, so factor cache and training subsets must be rebuilt.
    for dataset in experiment.DATASETS:
        args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", dataset])
        left = experiment.base.finetune_indices(legacy.train_dataset, args)
        right = experiment.base.finetune_indices(prefix.train_dataset, args)
        columns = ["raw_user_id", "target_position", "target_raw_item", "target_timestamp"]
        pd.testing.assert_frame_equal(legacy.train_dataset.examples.iloc[left][columns].reset_index(drop=True),
                                      prefix.train_dataset.examples.iloc[right][columns].reset_index(drop=True))


@pytest.mark.parametrize("dataset,random_n", [("yambda_50m_listens", 0), ("yambda_500m_listens", 15)])
def test_run_and_cache_config_preserve_base_dense_windows_and_sampling(dataset, random_n):
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", dataset])
    config = experiment.base.build_run_config(dataset, args, 1)
    assert args.prefix_from_dense and args.train_pool_last_n == 128
    assert args.train_random_n == random_n
    assert config["dataset"]["train_prefix_from_dense"]
    assert config["dataset"]["train_target_policy"] == "last_n"
    assert config["dataset"]["train_last_n"] == 128
    assert config["dataset"]["train_window_len"] is None
    assert config["dataset"]["train_window_stride"] is None
    data_cfg = experiment.base.tiger_runner.build_data_config(config, config_path=ROOT / "unused.json")
    assert data_cfg.train_prefix_from_dense
    assert dense_prefix_signature(config["dataset"])["window_len"] == 50
    for key in ("interaction", "sim_hier", "sasrec"):
        assert experiment.base.cache_config(config, key)["dataset"] == config["dataset"]


def test_teacherfree_and_sft_share_data_and_caches_and_amazon_uses_dense_prefixes():
    regular = experiment.parse_args(["--yambda50-base", "legacy"])
    teacherfree = experiment.parse_args(["--yambda50-base", "legacy", "--teacher-free"])
    control = experiment.parse_args(["--yambda50-base", "legacy", "--teacher-free", "--sft-only", "--seed", "43"])
    assert regular.cache_root == teacherfree.cache_root == control.cache_root
    assert regular.output_root != teacherfree.output_root
    assert teacherfree.output_root == control.output_root
    assert len(experiment.base.build_grid(regular)) == 336
    assert len(experiment.base.build_grid(teacherfree)) == 672
    assert not experiment.base.needs_teacher(experiment.base.build_grid(teacherfree))
    amazon = experiment.matched.parse_args(["--datasets", "beauty"])
    config = experiment.base.build_run_config("beauty", amazon, 1)
    assert amazon.prefix_from_dense and config["dataset"]["train_prefix_from_dense"]
    assert amazon.cache_root != teacherfree.cache_root
    with pytest.raises(ValueError, match="Amazon"):
        experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "beauty"])


@pytest.mark.parametrize("dataset", experiment.DATASETS)
@pytest.mark.parametrize("mode", [[], ["--sft-only"], ["--teacher-free"]])
def test_yambda_beam20_batch_is_bounded_without_changing_training(dataset, mode):
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", dataset, *mode])
    protocol = experiment.matched.load_protocol()["datasets"][dataset]
    assert args.test_batch_size == 64 and args.eval_num_beams == 20
    assert args.val_batch_size == protocol["val_batch_size"]
    assert args.batch_size == protocol["batch_size"]
    assert args.grad_accum == protocol["grad_accum"]
    resolved = experiment.base.dataset_args(args, dataset)
    assert resolved.test_batch_size == 64
    custom = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", dataset, *mode, "--test-batch-size", "32"])
    assert experiment.base.dataset_args(custom, dataset).test_batch_size == 32


def test_all_target_control_uses_same_beam20_batch():
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_50m_listens", "--sft-only",
                                  "--yambda50-all-targets"])
    assert args.test_batch_size == 64 and args.train_pool_last_n is None


def test_reject_old_cache_even_when_subset_and_target_count_match(tmp_path):
    g = experiment.base
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_500m_listens", "--cache-root", str(tmp_path)])
    config = g.build_run_config(args.datasets[0], args, 1)
    path = g.component_cache_dir(args.datasets[0], "interaction", args)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({**g.finetune_subset_spec(args), "num_examples": 15}))
    with pytest.raises(RuntimeError, match="training_context"):
        g.ensure_component_cache(args.datasets[0], "interaction", config, tmp_path, 15, args)


def test_factor_precompute_rebuilds_dense_prefix_data_and_matches_training(tmp_path):
    import precompute_reward_factors
    from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache

    dm = make_datamodule(tmp_path, prefix=True)
    config = {"data": {"source": "yambda", "interactions_path": str(dm.cfg.interactions_path),
                       "semantic_id_path": str(dm.cfg.semantic_id_path)},
              "dataset": asdict(dm.cfg), "model": {"use_user_token": False},
              "method": {"name": "gflow", "reward_model": "similarity", "reward": {"mode": "exact"}}}
    path = tmp_path / "factor_config.json"
    path.write_text(json.dumps(config, default=str))
    output = tmp_path / "factors"
    precompute_reward_factors.main([
        "--config", str(path), "--output-dir", str(output), "--device", "cpu",
        "--batch-size", "8", "--num-workers", "0", "--per-user-pool-last-n", "128",
        "--per-user-random-n", "15", "--per-user-random-seed", "42", "--verify-batches", "1",
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_context"] == dense_prefix_signature(config["dataset"])
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_500m_listens"])
    indices = experiment.base.finetune_indices(dm.train_dataset, args)
    assert manifest["num_examples"] == len(indices)
    experiment.base.factor_runner.verify_cache_alignment(dm, indices, {"sim_exact": FactorizedRewardCache(output)})


@pytest.mark.parametrize("stored_context", [False, True])
def test_resume_rejects_sliding_context_checkpoint(tmp_path, stored_context):
    g = experiment.base
    args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_500m_listens"])
    checkpoint = tmp_path / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    torch.save({}, checkpoint)
    identity = {"dataset": args.datasets[0], "seed": 42}
    resolved = {**identity, "cli_args": {"prefix_from_dense": stored_context},
                "data": {"num_finetune_examples": 15, "finetune_per_user_pool_last_n": 128,
                         "finetune_per_user_last_n": None, "finetune_per_user_random_n": 15,
                         "finetune_subset_seed": 42}}
    (tmp_path / "resolved_config.json").write_text(json.dumps(resolved))
    assert g.resumable_checkpoint(tmp_path, identity, 15, args) == (checkpoint if stored_context else None)


@pytest.mark.parametrize("group,total", [("gpt2_yambda", 96), ("teacherfree_gpt2_yambda", 180)])
@pytest.mark.skipif(
    not (ROOT / "scripts/scheduler/submit_base_matched_recalc.sh").is_file(),
    reason="scheduler scripts are gitignored",
)
def test_cluster_launches_new_script_cache_dependencies_and_controls(group, total):
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "DRY_RUN": "1", "LIVE_NODES": "0", "RECALC_GROUPS": group, "BLOCK_SIZE": "4"}
    result = subprocess.run(["bash", str(ROOT / "scripts/scheduler/submit_base_matched_recalc.sh")],
                            env=env, text=True, capture_output=True, check=True)
    commands = [line for line in result.stdout.splitlines() if " submit-worker --parsable " in line]
    assert commands and all("gflow_finetune_gpt2_yambda.py" in line for line in commands)
    assert any("SFT_ONLY=1" in line for line in commands)
    assert all("--dependency=afterok:" in line for line in commands if "GRID_INDEX=1" in line)
    assert ("--teacher-free" in result.stdout) == (group == "teacherfree_gpt2_yambda")
    assert f"array tasks planned: {total}" in result.stdout


def test_yambda50_all_target_prefix_control_retains_base_windows_and_splits(tmp_path):
    tail_args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_50m_listens", "--sft-only"])
    full_args = experiment.parse_args(["--yambda50-base", "legacy", "--datasets", "yambda_50m_listens", "--sft-only", "--yambda50-all-targets"])
    base = experiment.base
    assert full_args.prefix_from_dense and full_args.train_pool_last_n is None
    assert base.dataset_args(full_args, "yambda_50m_listens").train_pool_last_n is None
    for path in ("output_root", "cache_root", "metrics_csv"):
        assert getattr(full_args, path) != getattr(tail_args, path)
    full_config = base.build_run_config("yambda_50m_listens", full_args, 1)
    assert full_config["dataset"]["train_example_layout"] == "prefix_target"
    assert full_config["dataset"]["train_target_policy"] == "all"
    assert full_config["dataset"]["train_prefix_from_dense"]
    assert dense_prefix_signature(full_config["dataset"])["last_n"] is None
    dense = make_datamodule(tmp_path, prefix=False)
    tail = make_datamodule(tmp_path, prefix=True)
    full = YambdaDataModule(replace(tail.cfg, train_target_policy=full_config["dataset"]["train_target_policy"]))
    full.setup()
    pd.testing.assert_frame_equal(full.train_dataset.examples,
                                  prefixes_from_dense_windows(dense.train_dataset.examples, dense.interactions))
    assert full.train_dataset.examples.groupby("user_id").size().tolist() == [2, 50, 202, 267]
    assert len(base.finetune_indices(full.train_dataset, full_args)) == 521
    assert len(base.finetune_indices(full.train_dataset, tail_args)) == 308
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(full, split).examples, getattr(tail, split).examples)
    assert full.item2id == tail.item2id == dense.item2id
    # An explicit custom path cannot make the full-target arm reuse a tail run.
    (tmp_path / "resolved_config.json").write_text(json.dumps({"cli_args": {"prefix_from_dense": True}}))
    with pytest.raises(RuntimeError, match="Different target selection"):
        base.check_existing_training_context(tmp_path, full_args)


@pytest.mark.parametrize("flags", [[], ["--sft-only", "--datasets", "beauty"],
                                   ["--sft-only", "--datasets", "yambda_500m_listens"],
                                   ["--sft-only", "--train-random-n", "15"],
                                   ["--sft-only", "--train-last-n", "128"]])
def test_all_target_diagnostic_rejects_wrong_dataset_or_subsampling(flags):
    with pytest.raises(SystemExit):
        experiment.base.parse_args(["--datasets", "yambda_50m_listens", "--yambda50-all-targets", *flags])
