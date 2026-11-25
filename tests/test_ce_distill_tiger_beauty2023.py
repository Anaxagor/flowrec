"""Beauty 2023 CE grid, init checkpoint and matched TIGER data protocol."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp7-ce-distillation/ce_distill_tiger_beauty2023.py"
sys.path.insert(0, str(SCRIPT.parent))
import ce_distill_tiger_beauty2023 as experiment


def test_default_cli_grid_uses_six_corrected_ce_cells_for_reward_product(tmp_path):
    command = subprocess.run([sys.executable, "-S", str(SCRIPT), "--count"], cwd=tmp_path,
                             capture_output=True, text=True, check=True)
    assert command.stdout.strip() == "6"
    args = experiment.parse_args([])
    grid = experiment.ce.build_grid(args)
    assert {entry["dataset"] for entry in grid} == {"beauty_2023"}
    assert {experiment.ce.combo_key(entry) for entry in grid} == {"sasrec_x_interaction_x_similarity"}
    assert args.powers == [.25, 1., 16.] and args.rho_targets == [1., .1]
    assert args.sampling == "curriculum_smooth_is" and args.num_samples == 3
    paths = {experiment.ce.run_dir_for(args, entry) for entry in grid}
    assert len(paths) == 6 and all(p.is_relative_to(ROOT / experiment.DEFAULT_OUTPUT_ROOT) for p in paths)


def test_tiger_init_tuned_and_data_match_gflow(monkeypatch):
    shared = experiment.ce
    matched, base = shared.load_runtime()
    for key in {"DATASET", *base.DATASETS["beauty_2023"]}:
        monkeypatch.setattr(base, key, getattr(base, key))
    args = experiment.parse_args([])
    b = experiment.matched_args(args, "beauty_2023", 42, matched)
    cfg = base.build_run_config(b)
    root = base.PROJECT_ROOT
    assert base.abspath(base.TIGER_CHECKPOINT) == experiment.initial_checkpoint(root, "tiger")
    assert base.abspath(base.SASREC_TRIAL) == experiment.initial_checkpoint(root, "sasrec").with_name("best_trial.json")
    assert Path(cfg["method"]["sasrec"]["checkpoint_path"]) == experiment.initial_checkpoint(root, "sasrec")
    gflow = matched.parse_args(["--dataset", "beauty_2023", "--num-workers", str(args.num_workers)])
    reference = base.build_run_config(gflow)
    for key in ("data", "dataset", "dataloader", "model"):
        assert cfg[key] == reference[key]
    assert (b.train_last_n, b.train_random_n) == (0, 0)
    assert cfg["dataset"]["split_strategy"] == "leave_one_out"
    assert cfg["dataset"]["train_all_prefixes"] and cfg["dataset"]["train_target_policy"] == "all"
    assert cfg["dataset"]["positive_criterion"] == "verified_purchase == True"
    assert not cfg["dataset"]["use_verified_only"]
    assert (b.batch_size, b.grad_accum, b.val_sample_fraction) == (256, 8, .1)
    assert b.test_ks == [5, 10, 20] and b.eval_num_beams == 20
    assert b.cache_root == gflow.cache_root


def test_tiny_init_factors_training_and_final_splits(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import torch

    shared = experiment.ce
    matched, base = shared.load_runtime()
    import precompute_reward_factors

    dataset = experiment.DATASET
    source = base.DATASETS[dataset]
    cfg = shared.read_json(base.abspath(source["SFT_CONFIG"]))
    cfg["model"].update(embedding_dim=16, num_heads=2, d_kv=8, codebook_size=8, dim_feedforward=24,
                        num_encoder_layers=1, num_decoder_layers=1, dropout=0)
    cfg["dataset"].update(min_user_interactions=3, min_item_interactions=0)
    sids = {str(i): [i // 8, 8 + i % 8, 16, 24] for i in range(32)}
    semantic_path = tmp_path / "semantic_ids.json"
    semantic_path.write_text(json.dumps(sids))
    cfg["data"]["semantic_id_path"] = str(semantic_path)
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(cfg))
    rows = [dict(user_id=f"u{u}", parent_asin=str(i % 32), timestamp=1000 + i,
                 rating=5., verified_purchase=True) for u in range(3) for i in range(36)]
    rows.append(dict(user_id="u0", parent_asin="31", timestamp=1002, rating=1., verified_purchase=False))
    reviews, metadata = tmp_path / "reviews.parquet", tmp_path / "meta.parquet"
    pd.DataFrame(rows).to_parquet(reviews)
    pd.DataFrame({"parent_asin": list(sids)}).to_parquet(metadata)
    for key in {"DATASET", *source}:
        monkeypatch.setattr(base, key, getattr(base, key))
    monkeypatch.setitem(base.DATASETS, dataset, {**source, "SFT_CONFIG": str(config_path),
                        "DATA_OVERRIDES": {"reviews_path": str(reviews), "metadata_path": str(metadata)}})
    monkeypatch.setattr(base, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(base, "abspath", lambda path: tmp_path / path)
    # Execute the production cache command in-process against the same fixtures.
    monkeypatch.setattr(base, "subprocess", SimpleNamespace(
        run=lambda command, **kwargs: precompute_reward_factors.main(command[2:])))
    flags = ["--combos", "interaction", "--powers", "1", "--rho-targets", ".1", "--device", "cpu",
             "--num-workers", "0", "--batch-size", "2", "--grad-accum", "1", "--cache-batch-size", "16",
             "--max-epochs", "1", "--min-epochs", "0", "--max-train-batches", "2",
             "--calibration-batches", "1", "--cache-root", str(tmp_path / "factors"),
             "--output-root", str(tmp_path / "results")]
    args = experiment.parse_args(flags)
    b = experiment.matched_args(args, dataset, 42, matched)
    cfg = base.build_run_config(b)
    dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(cfg, config_path=config_path))
    dm.setup()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = base.tiger_runner.build_model(cfg, dm, config_path=config_path)
        checkpoint = experiment.initial_checkpoint(tmp_path, "tiger")
        checkpoint.parent.mkdir(parents=True)
        torch.save({"model_state_dict": model.state_dict()}, checkpoint)
        experiment.main([*flags, "--cache-only", "--cache-component", "interaction"])
        experiment.main([*flags, "--index", "0"])
    finally:
        torch.set_num_threads(threads)
    result = shared.read_json(shared.run_dir_for(args, shared.build_grid(args)[0]) / "result.json")
    assert result["importance_correction"] and result["evaluation_scope"] == "full"
    assert result["model"] == "tiger" and result["status"] == "completed"
    assert set(result["splits"]) == {"validation", "test"}
    assert all("NDCG@10" in metrics for metrics in result["splits"].values())
