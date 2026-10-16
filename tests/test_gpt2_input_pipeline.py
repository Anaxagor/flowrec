"""The production GPT-2 entrypoint connects the shared indexed/factor pipeline."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon_fullhist as fullhist

g = fullhist.base


@pytest.fixture
def cache(tmp_path):
    from flowrec.ft_methods.gflow.factorized_cache import CACHE_KIND, FACTORS_FILENAME

    path = tmp_path / "factors"
    path.mkdir()
    torch.save({"target_semantic_row_ids": torch.tensor([0, 1])}, path / FACTORS_FILENAME)
    (path / "manifest.json").write_text(json.dumps({
        "kind": CACHE_KIND, "num_examples": 2, "num_catalog_items": 2,
        "component": {"name": "positive", "config": {}},
    }))
    return g.ComposedFactorizedRewardCache([("positive", path, 1.0)], composition="multiply", power=4.0)


def test_old_queued_arguments_enable_new_pipeline_defaults():
    args = fullhist.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3",
                               "--require-existing-caches", "--num-workers", "8"])
    assert args.reward_prefetch and args.tensorized_train_loader
    assert args.loader_prefetch_factor == 2
    assert g.run_batch_settings(args, 3) == (128, 16)
    assert len(g.build_grid(args)) == 168
    for argv in (["--num-workers", "-1"], ["--loader-prefetch-factor", "0"]):
        with pytest.raises(SystemExit):
            fullhist.parse_args(argv)


@pytest.mark.parametrize("cache_kind", ["factorized", "dense", "sft", "disabled"])
def test_fit_connects_only_factorized_rewards(tmp_path, monkeypatch, cache, cache_kind):
    args = fullhist.parse_args(["--datasets", "beauty_2023", "--num-candidates", "3",
                               "--loader-prefetch-factor", "3", "--num-workers", "8"])
    if cache_kind == "disabled":
        args.reward_prefetch = False
        args.tensorized_train_loader = False
    reward_cache = None if cache_kind == "sft" else object() if cache_kind == "dense" else cache
    captured = {}

    def loader(dm, subset, **kwargs):
        captured.update(kwargs)
        assert subset == [1, 0, 1]
        return "training-loader"

    def fit(train, val):
        assert train == "training-loader" and val == "validation-loader"
        return SimpleNamespace(state=SimpleNamespace(history=[]), best_checkpoint_path=None,
                               last_checkpoint_path=None)

    monkeypatch.setattr(g.gflow_ablation, "indexed_train_loader", loader)
    monkeypatch.setattr(g.gpt2_beauty, "evaluate_splits", lambda *a, **k: {"test": {}})
    g.fit_and_evaluate(SimpleNamespace(fit=fit), None,
                       SimpleNamespace(cfg=g.tiger_runner.YambdaDataConfig(),
                                       val_dataloader=lambda: "validation-loader"), [1, 0, 1],
                       run_batch=128, run_dir=tmp_path, args=args, device=torch.device("cpu"),
                       reward_cache=reward_cache)
    active = cache_kind == "factorized"
    assert captured["reward_cache"] is (cache if active else None)
    assert captured["prefetch_factor"] == 3
    assert captured["tensorized"] == (cache_kind != "disabled")
    assert captured["batch_size"] == 128 and captured["seed"] == 42
    assert g.input_pipeline_settings(args, reward_cache=reward_cache)["reward_prefetch"] == active


def test_explicit_dense_loader_is_preserved(tmp_path, monkeypatch):
    args = fullhist.parse_args([])
    def fit(train, val):
        assert train == "dense-control-loader"
        return SimpleNamespace(state=SimpleNamespace(history=[]), best_checkpoint_path=None,
                               last_checkpoint_path=None)
    monkeypatch.setattr(g.gflow_ablation, "indexed_train_loader",
                        lambda *a, **k: pytest.fail("Do not replace the dense-layout control loader"))
    monkeypatch.setattr(g.gpt2_beauty, "evaluate_splits", lambda *a, **k: {})
    g.fit_and_evaluate(SimpleNamespace(fit=fit), None,
                       SimpleNamespace(cfg=g.tiger_runner.YambdaDataConfig(),
                                       val_dataloader=lambda: "validation"), [0, 1],
                       run_batch=32, run_dir=tmp_path, args=args, device=torch.device("cpu"),
                       train_loader="dense-control-loader")


def test_calibration_prepares_same_factors_without_worker_rng_changes(tmp_path, monkeypatch, cache):
    from flowrec.ft_methods import calibration

    args = fullhist.parse_args(["--datasets", "beauty_2023", "--num-workers", "8"])
    captured = {}
    monkeypatch.setattr(g.gflow_ablation, "indexed_train_loader",
                        lambda *a, **k: captured.update(k) or "loader")
    expected = {"sft_grad_norm_median": 1., "gflow_grad_norm_median": 2.,
                "raw_grad_ratio_median": 2., "grad_cosine_median": .5,
                "target_grad_ratio": 1., "fixed_gflow_weight": .5}
    model = torch.nn.Linear(2, 2).eval()

    def sequential(actual_model, reference, component, loader, params, **kw):
        assert actual_model is model and model.training
        assert loader == "loader" and params == list(model.parameters())
        assert isinstance(reference, g.gflow_ablation.AliasedSFTMethod)
        assert isinstance(component, g.GPT2GFlowMethod)
        assert (component.cfg.sft_weight, component.cfg.gflow_weight) == (0., 1.)
        assert kw["seed"] == args.gflow_calibration_seed and kw["step"] == 0
        assert kw["num_batches"] == args.gflow_calibration_batches
        assert kw["device"] == torch.device("cpu") and kw["target_ratio"] == 1.
        assert not torch.is_autocast_enabled("cpu")
        assert all(p.dtype == torch.float32 for p in params)
        return dict(expected)

    monkeypatch.setattr(calibration, "calibrate_sequential_variants", sequential)
    result = g.calibrate_gflow_weight(
        model=model, base_gflow_cfg=g.GFlowConfig(reward_cache_path=str(tmp_path)), sampler=object(),
        sft_cfg=g.SFTConfig(), cache=cache, dm=None, subset_indices=[1, 0, 1],
        device=torch.device("cpu"), args=args, tag="test", target_ratio=1., batch_size=128)
    assert result == {**expected, "implementation": "sequential_variants_fp32"} and not model.training
    assert torch.initial_seed() == args.seed
    assert captured["num_workers"] == 0 and captured["reward_cache"] is cache
    assert captured["seed"] == args.gflow_calibration_seed and captured["tensorized"]
    assert captured["prefetch_factor"] == args.loader_prefetch_factor and captured["batch_size"] == 128


def test_performance_artifact_retains_input_pipeline_timings(tmp_path):
    payload = {"type": "train_step", "epoch": 0, "batch_idx": 199, "global_step": 200,
               "train/data_wait_sec": .01, "train/step_wall_sec": .12,
               "train/reward_prepare_sec": .02, "train/examples_per_sec": 1000.}
    logger = g.performance_logger(lambda _: None, tag="test", run_dir=tmp_path,
                                  micro_batch=128, device=torch.device("cpu"))
    logger(payload)
    saved = json.loads((tmp_path / "performance.jsonl").read_text())
    for key, value in payload.items():
        assert saved[key] == value
