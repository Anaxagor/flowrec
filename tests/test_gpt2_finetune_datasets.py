"""Large-dataset GPT-2 data, factor caches, and cluster dispatch contracts."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon_fullhist as matched
import gflow_finetune_gpt2_amazon_teacherfree as teacherfree
from flowrec.data.dense_prefix import dense_prefix_signature
from flowrec.ft_methods.gflow.factorized_cache import CACHE_KIND, FACTORS_FILENAME
from flowrec.models.gpt2 import GPT2RecModel

g = matched.base
NEW_DATASETS = ("beauty_2023", "yambda_50m_listens", "yambda_500m_listens")


@pytest.mark.parametrize("mode", [[], ["--sft-only"], ["--cache-only"]])
def test_beauty2023_faster_batches_keep_amazon_protocol_in_all_phases(mode):
    beauty = matched.parse_args(["--datasets", "beauty", "--num-candidates", "3", *mode])
    args = matched.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3", *mode])
    batch_keys = {"batch_size", "grad_accum", "val_batch_size", "test_batch_size"}
    for key in (*g.DATASET_ARGUMENTS, "lr", "weight_decay", "max_epochs", "max_grad_norm",
                "validate_every", "patience", "min_epochs", "curriculum_epochs",
                "validate_at_start", "eval_num_beams", "test_ks", "prefix_from_dense"):
        if key not in batch_keys:
            assert getattr(args, key) == getattr(beauty, key), key
    assert (args.batch_size, args.grad_accum, args.val_batch_size, args.test_batch_size) == (128, 16, 128, 128)
    assert args.max_candidate_batch == 512
    assert g.run_batch_settings(args, 3) == g.run_batch_settings(args, None) == (128, 16)
    assert args.val_sample_fraction is None and not args.val_sample_resample
    assert args.prefix_from_dense and args.train_pool_last_n is None
    assert g.uses_factor_cache("beauty_2023", args)
    assert len(g.build_grid(args)) == 168
    # Resolving another dataset after Beauty 2023 must restore its own defaults.
    yambda = g.dataset_args(args, "yambda_500m_listens")
    assert (yambda.batch_size, yambda.val_batch_size, yambda.val_sample_fraction) == (256, 256, 0.25)
    assert (g.dataset_args(args, "beauty").batch_size, yambda.max_candidate_batch) == (32, 512)


def test_beauty2023_k3_has_separate_runs_but_reuses_k10_factors():
    old = matched.parse_args(["--datasets", "beauty_2023", "--num-candidates", "10"])
    new = matched.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3"])
    assert g.run_batch_settings(old, 10) == (32, 64)
    assert g.run_batch_settings(new, 3) == (128, 16)
    old_grid, new_grid = g.build_grid(old), g.build_grid(new)
    assert len(old_grid) == len(new_grid) == 168
    for a, b in zip(old_grid, new_grid):
        assert a["num_negatives"] == 10 and b["num_negatives"] == 3
        assert {k: v for k, v in a.items() if k != "num_negatives"} == {
            k: v for k, v in b.items() if k != "num_negatives"}
        assert g.run_dir_for(Path(new.output_root), a, 42) != g.run_dir_for(Path(new.output_root), b, 42)
    for component in ("sasrec", "interaction", "sim_hier"):
        assert g.component_cache_dir("beauty_2023", component, old) == g.component_cache_dir(
            "beauty_2023", component, new)


def test_beauty2023_explicit_batch_cap_override_wins():
    args = matched.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3",
                               "--batch-size", "64", "--grad-accum", "32",
                               "--max-candidate-batch", "256"])
    assert g.run_batch_settings(args, 3) == (64, 32)
    assert g.dataset_args(args, "beauty").max_candidate_batch == 256


def test_training_only_fails_instead_of_recalculating_missing_factors(tmp_path, monkeypatch):
    args = matched.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3",
                               "--cache-root", str(tmp_path / "cache"), "--require-existing-caches"])
    config = g.build_run_config("beauty_2023", args, 1)
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: pytest.fail("Must not precompute factors"))
    with pytest.raises(RuntimeError, match="forbids recalculation"):
        g.ensure_component_cache("beauty_2023", "sasrec", config, tmp_path / "config", 20, args)
    with pytest.raises(SystemExit):
        matched.parse_args(["--require-existing-caches", "--force-cache"])


@pytest.mark.parametrize("dataset", ("beauty", "instruments", *NEW_DATASETS))
def test_preprocessing_matches_gpt2_sft_and_matched_schedule(dataset, tmp_path, monkeypatch):
    args = matched.parse_args(["--datasets", dataset])
    path = g.dataset_yaml_path(dataset)
    group = next(group for group in g.gpt2_runner.GROUPS if group.data_config_name == path.name)
    reference, _ = g.gpt2_runner.build_data_config(
        group, dataset, g.runner.load_yaml(path), seed=42,
        semantic_ids_dir=g.SEMANTIC_IDS_ROOT, user_hash_mod=1)
    config = g.build_run_config(dataset, args, 1)
    actual = g.tiger_runner.build_data_config(config, config_path=ROOT / "configs/audit.json")
    allowed_differences = {"train_example_layout", "history_padding", "batch_size", "eval_batch_size",
                           "num_workers", "persistent_workers", "train_prefix_from_dense",
                           "train_target_policy", "train_last_n", "metadata_ids_only"}
    ref = asdict(reference)
    assert {k: v for k, v in asdict(actual).items() if k not in allowed_differences} == {
        k: v for k, v in ref.items() if k not in allowed_differences}
    if not dataset.startswith("yambda_"):
        assert actual.metadata_ids_only and not reference.metadata_ids_only
    assert actual.train_prefix_from_dense and not reference.train_prefix_from_dense
    assert actual.train_target_policy == ("last_n" if dataset.startswith("yambda_") else "all")
    assert actual.train_last_n == (128 if dataset.startswith("yambda_") else reference.train_last_n)
    assert actual.train_window_len == reference.train_window_len is None
    assert actual.train_window_stride == reference.train_window_stride is None
    assert actual.train_example_layout == "prefix_target"
    assert g.uses_factor_cache(dataset, args) == (dataset in NEW_DATASETS)
    cache_kind = "reward_factors" if dataset in NEW_DATASETS else "reward_cache"
    assert cache_kind in g.component_cache_dir(dataset, "interaction", args).parts
    batch, accum = g.run_batch_settings(args, 10)
    assert batch * accum == 2048 and batch * 11 <= args.max_candidate_batch
    assert g.run_batch_settings(args, None) == (batch, accum)
    assert args.train_random_n == (15 if dataset == "yambda_500m_listens" else 0)
    assert args.train_pool_last_n == (128 if dataset.startswith("yambda_") else None)
    with monkeypatch.context() as paths:
        paths.setattr(g, "PROJECT_ROOT", tmp_path)
        if dataset.startswith("yambda_"):
            legacy_dir = tmp_path / g.GPT2_ROOT / dataset / "full_history"
            # Legacy preprocessing keeps the exact archived model even before
            # its files are downloaded; it never borrows the selected base.
            assert g.tuned_checkpoint_paths(dataset)["checkpoint"] == legacy_dir / "best.pt"
            assert g.tuned_checkpoint_paths(dataset)["best_trial"] == legacy_dir / "best_trial.json"
        else:
            assert g.tuned_checkpoint_paths(dataset)["checkpoint"] == tmp_path / g.GPT2_ROOT / dataset / "best.pt"
    assert g.sasrec_method_cfg(dataset)["checkpoint_path"].startswith((
        str(ROOT / "checkpoints/exp2/loo_split/initial"),
        str(ROOT / "checkpoints/exp2/loo_split/initial"),
    ))
    # The teacher-free launch and each reward component must see identical
    # preprocessing, target policies and catalogs to the SFT-only arm.
    teacherfree_config = g.build_run_config(dataset, teacherfree.parse_args(["--datasets", dataset]), 1)
    for key in ("data", "dataset"):
        assert teacherfree_config[key] == config[key]
    for component in teacherfree_config["method"]["component_keys"]:
        cached = g.cache_config(teacherfree_config, component)
        assert cached["data"] == config["data"] and cached["dataset"] == config["dataset"]


def test_mixed_dataset_settings_do_not_leak_and_cli_wins():
    args = matched.parse_args(["--datasets", "yambda_500m_listens", "beauty"])
    large = g.dataset_args(args, "yambda_500m_listens")
    small = g.dataset_args(large, "beauty")
    assert (large.train_random_n, large.val_sample_fraction) == (15, 0.25)
    assert (small.train_random_n, small.val_sample_fraction, small.batch_size) == (0, None, 32)
    assert large.train_pool_last_n == 128 and small.train_pool_last_n is None
    args = teacherfree.parse_args(["--datasets", *NEW_DATASETS, "--batch-size=16", "--grad-accum", "128",
                                  "--train-random-n", "0", "--val-sample-fraction", "0.05"])
    for dataset in NEW_DATASETS:
        resolved = g.dataset_args(args, dataset)
        assert (resolved.batch_size, resolved.grad_accum, resolved.train_random_n) == (16, 128, 0)
        assert resolved.val_sample_fraction == 0.05
    grid = g.build_grid(args)
    assert len(grid) == 3 * 336 and not g.needs_teacher(grid)


@pytest.mark.parametrize("key,mode", [("interaction", None), ("sim_hier", "hierarchical"),
                                     ("sim_embed", "embedding_cosine"), ("sim_exact", "exact")])
def test_cache_build_uses_factors_and_identical_random_subset(tmp_path, monkeypatch, key, mode):
    args = teacherfree.parse_args(["--datasets", "yambda_500m_listens", "--cache-root", str(tmp_path),
                                  "--similarity-modes", "hierarchical", "embedding_cosine", "exact"])
    config = g.build_run_config("yambda_500m_listens", args, 1)
    cache_dir = g.component_cache_dir("yambda_500m_listens", key, args)
    calls = []

    def precompute(command, **kwargs):
        calls.append(command)
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.json").write_text(json.dumps({
            "training_context": dense_prefix_signature(config["dataset"]),
            "kind": CACHE_KIND, "component": {"name": "similarity" if mode else key, "mode": mode},
            "num_examples": 20, "per_user_last_n": None,
            "per_user_random_n": 15, "per_user_random_seed": 42,
            "per_user_pool_last_n": 128,
        }))

    monkeypatch.setattr(g.subprocess, "run", precompute)
    g.ensure_component_cache("yambda_500m_listens", key, config, tmp_path, 20, args)
    assert Path(calls[0][1]).name == "precompute_reward_factors.py"
    assert calls[0][calls[0].index("--per-user-random-n") + 1] == "15"
    assert calls[0][calls[0].index("--per-user-random-seed") + 1] == "42"
    assert calls[0][calls[0].index("--per-user-pool-last-n") + 1] == "128"
    assert "--dtype" not in calls[0]
    g.ensure_component_cache("yambda_500m_listens", key, config, tmp_path, 20, args)
    assert len(calls) == 1
    args.train_subset_seed = 43
    with pytest.raises(RuntimeError, match="Rebuild|rebuild"):
        g.ensure_component_cache("yambda_500m_listens", key, config, tmp_path, 20, args)
    args.train_subset_seed = 42
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    del manifest["per_user_pool_last_n"]
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="per_user_pool_last_n"):
        g.ensure_component_cache("yambda_500m_listens", key, config, tmp_path, 20, args)


@pytest.mark.parametrize("dataset,count", [("beauty", 260), ("instruments", 260),
                                          ("beauty_2023", 260), ("yambda_50m_listens", 128),
                                          ("yambda_500m_listens", 15)])
def test_training_and_cache_subset_use_same_recent_targets(dataset, count):
    from precompute_reward_cache import subset_indices, per_user_random_indices

    class Examples:
        examples = pd.DataFrame({"user_id": ["long"] * 260 + ["short"] * 10})

        def __len__(self):
            return len(self.examples)

    data = Examples()
    args = matched.parse_args(["--datasets", dataset])
    selected = g.finetune_indices(data, args)
    cached = subset_indices(data, **g.finetune_subset_spec(args))
    assert selected == (list(range(len(data))) if cached is None else cached)
    assert len([i for i in selected if i < 260]) == count
    assert selected[-10:] == list(range(260, 270))
    if dataset.startswith("yambda_"):
        assert min(selected) >= 132
        # Direct invocation of the base runner follows the same Yambda rule.
        assert g.finetune_indices(data, g.parse_args(["--datasets", dataset])) == selected
    if dataset == "yambda_500m_listens":
        pool = SimpleNamespace(examples=data.examples.iloc[132:])
        assert selected == per_user_random_indices(pool, 15, 42)
        assert selected == g.finetune_indices(data, args)
        args.train_subset_seed = 43
        assert selected != g.finetune_indices(data, args)


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
@pytest.mark.parametrize("composition", ["multiply", "mixed"])
def test_gpt2_trains_with_factorized_rewards(tmp_path, objective, composition):
    tokens = torch.tensor([[1, 5], [1, 6], [2, 5], [2, 6]])
    targets = torch.tensor([1, 2])
    torch.save({"item_tokens": tokens, "target_token_ids": tokens[targets],
                "target_semantic_row_ids": targets}, tmp_path / FACTORS_FILENAME)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": CACHE_KIND, "num_examples": 2, "num_catalog_items": 4,
        "component": {"name": "similarity", "mode": "exact", "config": {}},
    }))
    args = g.parse_args(["--datasets", "yambda_50m_listens"])
    cache = g.compose_reward_cache({"dataset": "yambda_50m_listens", "components": ["sim_exact"],
                                   "weights": [1.0], "power": 2.0, "composition": composition},
                                  {"sim_exact": tmp_path}, args)
    assert isinstance(cache, g.ComposedFactorizedRewardCache)
    selected, total, full = cache.select_candidates(torch.arange(2), torch.tensor([[-1, 3], [-1, 0]]),
                                                   torch.device("cpu"), include_full=True)
    expected = (tokens[targets, None, :] == tokens[None, :, :]).float().mean(-1).clamp_min(1e-6)
    expected = expected.square().clamp_min(1e-6)
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(total, expected.sum(-1))
    mapping = {str(i): row for i, row in enumerate(tokens.tolist())}
    model = GPT2RecModel(codebook_size=4, sem_id_len=2, semantic_id_mapping=mapping,
                         embedding_dim=8, num_heads=2, num_layers=1, dim_feedforward=16,
                         max_position_embeddings=16, dropout=0.0)
    dm = SimpleNamespace(item_vocab_size=5, item2id={str(i): i + 1 for i in range(4)}, semantic_ids=mapping)
    sampler = g.build_sampler("random", model=model, dm=dm, sasrec=None,
                              num_negatives=1, curriculum_steps=0, args=args)
    method = g.make_method(g.GFlowConfig(objective=objective, reward_cache_path=str(tmp_path)), sampler,
                           g.SFTConfig(compute_train_metrics=False, compute_eval_metrics=False), cache)
    batch = {"history_token_ids": tokens[[0, 3]].unsqueeze(1),
             "history_token_mask": torch.ones(2, 1, 2, dtype=torch.bool),
             "history_item_ids": torch.tensor([[1], [4]]), "history_mask": torch.ones(2, 1, dtype=torch.bool),
             "target_token_ids": tokens[targets], "target_token_mask": torch.ones(2, 2, dtype=torch.bool),
             "target_item_ids": targets + 1, "example_indices": torch.arange(2)}
    output = method.compute_loss(model, batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())






def test_resume_requires_same_recent_pool_and_random_sample(tmp_path):
    args = matched.parse_args(["--datasets", "yambda_500m_listens"])
    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints/last.pt"
    torch.save({}, checkpoint)
    data = {"num_finetune_examples": 30, "finetune_per_user_pool_last_n": 128,
            "finetune_per_user_last_n": None, "finetune_per_user_random_n": 15,
            "finetune_subset_seed": 42}
    identity = {"dataset": "yambda_500m_listens"}
    (tmp_path / "resolved_config.json").write_text(json.dumps({**identity, "data": data, "cli_args": {"prefix_from_dense": True}}))
    assert g.resumable_checkpoint(tmp_path, identity, 30, args) == checkpoint
    args.train_subset_seed = 43
    assert g.resumable_checkpoint(tmp_path, identity, 30, args) is None
    args.train_subset_seed = 42
    args.train_pool_last_n = 64
    assert g.resumable_checkpoint(tmp_path, identity, 30, args) is None


def test_power16_real_grid_preserves_all_resume_identities_and_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/exp4-reward-ablations"))
    import gflow_finetune_gpt2_amazon_fullhist as fullhist

    argv = ["--datasets", "beauty_2023", "--prefix-from-dense",
            "--cache-format", "factorized", "--num-candidates", "3"]
    full_args = fullhist.parse_args(argv)
    args = fullhist.parse_args([*argv, "--powers", "16", "--rho-targets", "0.1",
                               "--combos", "sasrec", "sasrec_x_interaction_x_similarity",
                               "--require-existing-caches", "--resume"])
    full_grid = fullhist.base.build_grid(full_args)
    filtered = fullhist.base.build_grid(args)
    assert len(full_grid) == 168 and len(filtered) == 6
    assert args.resume and args.require_existing_caches
    assert args.output_root == full_args.output_root
    assert args.cache_root == full_args.cache_root
    assert args.metrics_csv == full_args.metrics_csv
    expected_combos = {frozenset(("sasrec",)), frozenset(("sasrec", "interaction", "sim_hier"))}
    assert {(frozenset(entry["components"]), entry["objective"]) for entry in filtered} == {
        (combo, objective) for combo in expected_combos for objective in ("tb", "db", "subtb")
    }
    old_indices = [i for i, entry in enumerate(full_grid)
                   if entry["power"] == 16 and entry["rho"] == 0.1
                   and frozenset(entry["components"]) in expected_combos]
    assert old_indices == [21, 22, 23, 165, 166, 167]
    assert filtered[:3] == full_grid[21:24]
    assert filtered[3:] == full_grid[165:168]
    for entry, old_index in zip(filtered, old_indices):
        old_entry = full_grid[old_index]
        assert fullhist.base.entry_identity(entry, args.seed) == fullhist.base.entry_identity(
            old_entry, full_args.seed)
        assert fullhist.base.run_dir_for(Path(args.output_root), entry, args.seed) == (
            fullhist.base.run_dir_for(Path(full_args.output_root), old_entry, full_args.seed))
