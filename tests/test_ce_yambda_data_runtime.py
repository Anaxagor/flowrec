"""Storage identity separates data semantics from CE optimizer/loading knobs."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp6-ce-distillation"))

import ce_yambda_data_runtime as runtime
from flowrec.data.yambda import YambdaDataConfig


@pytest.fixture
def inputs(tmp_path):
    events, semantics = tmp_path / "events.parquet", tmp_path / "semantic_ids.json"
    events.write_bytes(b"identity fixture")
    semantics.write_text('{"1": [1, 2]}')
    cfg = YambdaDataConfig(interactions_path=events, semantic_id_path=semantics,
        use_hf=False, include_raw_in_batch=False, train_example_layout="prefix_target",
        train_target_policy="last_n", train_last_n=128, train_prefix_from_dense=False,
        train_dense_match_prefix=False, split_strategy="leave_one_out", seed=42)
    args = SimpleNamespace(train_random_n=50, train_subset_seed=42,
        prepared_data_root=tmp_path / "prepared", smoke=False,
        output_root=tmp_path / "results", sampling="reward", seeds=[42])
    config = {"initial_model": {"trial": 4},
              "dataset": {"matched_tuning_data_audit": {"train_contexts_sha256": "source"}},
              "method": {"reward": "sasrec"}}
    return args, cfg, config


def test_same_inputs_share_preparation_across_loss_and_loader_settings(inputs):
    args, cfg, config = inputs
    expected = runtime.identity(args, "gpt2", cfg, config)
    args.sampling, args.seeds = "curriculum_smooth_is", [43]
    config["method"] = {"reward": "interaction", "rho": .1, "power": 16}
    config["seed"] = 43
    cfg = replace(cfg, seed=43, batch_size=8, eval_batch_size=7, num_workers=0,
                  pin_memory=False, persistent_workers=False)
    assert runtime.identity(args, "gpt2", cfg, config) == expected


@pytest.mark.parametrize("change", ["subset_size", "subset_seed", "source_audit", "source_trial",
                                   "preprocessing", "raw_input"])
def test_data_identity_changes_with_every_material_input(inputs, change):
    args, cfg, config = inputs
    expected = runtime.identity(args, "gpt2", cfg, config)
    if change == "subset_size":
        args.train_random_n = 15
    elif change == "subset_seed":
        args.train_subset_seed = 43
    elif change == "source_audit":
        config["dataset"]["matched_tuning_data_audit"]["train_contexts_sha256"] = "different"
    elif change == "source_trial":
        config["initial_model"]["trial"] = 5
    elif change == "preprocessing":
        cfg = replace(cfg, deduplicate_consecutive=not cfg.deduplicate_consecutive)
    else:
        Path(cfg.interactions_path).write_bytes(b"different raw inputs")
    assert runtime.identity(args, "gpt2", cfg, config) != expected


@pytest.mark.parametrize("changed", [
    {"train_target_policy": "all"}, {"train_last_n": 50},
    {"train_prefix_from_dense": True}, {"train_dense_match_prefix": True},
    {"include_raw_in_batch": True}, {"use_hf": True},
])
def test_prepared_protocol_cannot_silently_change_target_layout(inputs, changed):
    args, cfg, config = inputs
    with pytest.raises(RuntimeError):
        runtime.identity(args, "gpt2", replace(cfg, **changed), config)
