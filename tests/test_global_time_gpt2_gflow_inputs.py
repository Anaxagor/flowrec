import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/global-time"), str(ROOT / "scripts/exp4-reward-ablations")]
import gflow_finetune_gpt2_amazon_fullhist as matched
import gpt2_gflow_inputs as inputs
import gflow_finetune_gpt2 as experiment
from flowrec.data.dense_prefix import dense_prefix_signature
from flowrec.data.interaction_store import InteractionLogStore


@pytest.mark.parametrize("dataset", inputs.DATASETS)
def test_global_array_cells_and_sft_controls(dataset, tmp_path, monkeypatch):
    base = matched.base
    monkeypatch.setattr(base, "MATCHED_PROTOCOL", experiment.PROTOCOL)
    cli = SimpleNamespace(stage="gflow", dataset=dataset, device="cpu", num_workers=0, smoke=False)
    args = experiment.settings(cli, tmp_path, tmp_path)
    grid = base.build_grid(args)
    dimensions = [(e["sampler"], e["power"], e["rho"], e["objective"]) for e in grid]
    expected = product(("random", "policy", "teacher", "q"), (0.25, 1, 16), (1, 0.1),
                       ("tb", "db", "subtb"))
    assert dimensions == list(expected)
    assert args.strict_negatives and args.seed == 42 and args.curriculum_epochs == 10
    assert args.require_existing_caches and args.cache_format == "factorized"
    assert all(e["components"] == ("sasrec", "interaction", "sim_hier") and
               e["weights"] == (1, 1, 1) and e["num_negatives"] == 3 and
               e["composition"] == "multiply" for e in grid)
    dirs = {base.run_dir_for(tmp_path, e, args.seed) for e in grid}
    cli.stage = "sft"
    for cli.index, seed in enumerate((42, 43, 44)):
        control = experiment.settings(cli, tmp_path, tmp_path)
        assert control.sft_only and control.seed == seed
        assert base.run_batch_settings(control, None) == base.run_batch_settings(args, 3) == (32, 64)
        assert (control.lr, control.validate_every, control.patience, control.min_epochs) == (
            args.lr, args.validate_every, args.patience, args.min_epochs)
        dirs.add(base.sft_control_run_dir(tmp_path, dataset, seed))
    assert len(dirs) == 75


@pytest.mark.parametrize("dataset", inputs.DATASETS)
def test_staged_sources_and_split_reach_all_training_arms(dataset, tmp_path, monkeypatch):
    base = matched.base
    settings = SimpleNamespace(DATASETS=dict(base.DATASETS))
    inputs.configure_inputs(settings, tmp_path)
    for key, value in vars(settings).items():
        monkeypatch.setattr(base, key, value)
    monkeypatch.setattr(base, "MATCHED_PROTOCOL", inputs.CONFIG_DIR / "exp2-tuned-params/finetune_base_matched.json")
    args = base.dataset_args(matched.parse_args(["--datasets", dataset, "--seed", "44"]), dataset)
    config = base.build_run_config(dataset, args, 1)
    cache = base.cache_config(config, "interaction")
    staged = inputs.input_files(dataset)
    assert all(not path.is_absolute() for path in staged)
    assert base.tuned_checkpoint_paths(dataset)["checkpoint"].relative_to(tmp_path) in staged
    assert base.sasrec_asset_paths(dataset)[0].relative_to(tmp_path) in staged
    for run in (config, cache):
        data = base.tiger_runner.build_data_config(run, config_path=tmp_path / "run.json")
        assert data.seed == 42 and args.seed == 44
        assert data.split_strategy == "global_time" and not data.drop_items_missing_semantic_id
        assert data.processed_cache_dir == tmp_path / "processed"
        assert data.train_prefix_from_dense and data.cold_item_train_source == "observed"
        assert Path(data.semantic_id_path).relative_to(tmp_path) in staged
        assert Path(data.reviews_path).relative_to(tmp_path) in staged
        assert dense_prefix_signature(run["dataset"])["global_split"]["global_val_timestamp"] == data.global_val_timestamp
    assert cache["method"]["reward"]["training_window_only"]


def test_global_interaction_evidence_ends_strictly_before_validation(tmp_path):
    frame = pd.DataFrame({"user_id": [0, 0, 0], "item_id": [1, 1, 2], "timestamp": [9, 10, 15],
                          "event_type": ["review"] * 3, "rating": [3, 4, 5]})
    store = InteractionLogStore(frame, num_users=1, item_vocab_size=3)
    dm = SimpleNamespace(interaction_store=store, interactions=frame, num_users=1, item_vocab_size=3,
                         global_val_cutoff=10)
    reward = matched.base.tiger_runner.build_reward_model(
        "interaction", data=dm, train_loader=None, config_path=tmp_path / "run.json", seed=42,
        reward_cfg={"training_window_only": True}, svd_cfg={}, svd_model=None, sasrec_model=None,
        reward_epsilon=1e-6)
    assert list(reward.store.events.timestamp) == [9]
    assert list(reward.positive_store.events.timestamp) == [9]
