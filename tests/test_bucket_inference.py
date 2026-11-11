"""Frozen evaluation keeps original contexts, while GPU tasks load no datasets."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/exp6_result_analysis"))
import bucket_inference as inference


def test_history_cohorts_use_unclipped_positions_and_selection_is_order_independent():
    frame = pd.DataFrame({"raw_user_id": ["b", "a", "c"], "target_raw_item": ["x", "y", "z"],
                          "history_raw_items": [["x", "y"], ["x"], ["x", "y"]],
                          "target_position": [200, 1, 50]})
    arrays = inference.example_arrays(frame, 2)
    assert arrays["history_lengths"].tolist() == [2, 1, 2]
    assert arrays["full_history_lengths"].tolist() == [200, 1, 50]
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    selected = frame.iloc[inference.select_examples(frame, 2, 42)].raw_user_id.tolist()
    assert selected == shuffled.iloc[inference.select_examples(shuffled, 2, 42)].raw_user_id.tolist()
    with pytest.raises(ValueError, match="original target_position"):
        inference.example_arrays(frame.drop(columns="target_position"), 2)


def test_source_aware_resolution_and_positive_stream_equivalence(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("events.parquet", "ids.json", "embeddings.parquet"):
        (folder / name).touch()
    config = {"data": {"source": "yambda", "interactions_path": "data/events.parquet",
                       "semantic_id_path": "data/ids.json", "embeddings_path": "data/embeddings.parquet"},
              "dataset": {"allowed_event_types": ["listen"], "min_played_ratio_pct": 50,
                          "positive_criterion": None, "max_history_len": 50}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    resolved = inference.resolve_data_config({"data_config": str(path)}, tmp_path)
    assert resolved["data"]["embeddings_path"] == str(folder / "embeddings.parquet")
    assert resolved["dataloader"]["num_workers"] == 0
    protocol = inference.data_protocol(resolved)
    resolved["dataset"].update(allowed_event_types=None, min_played_ratio_pct=None,
                               positive_criterion="event_type == 'listen' and played_ratio_pct >= 50")
    assert inference.data_protocol(resolved) == protocol
    resolved["dataset"]["max_history_len"] = 20
    assert inference.data_protocol(resolved) != protocol
    (folder / "embeddings.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="embeddings_path"):
        inference.resolve_data_config({"data_config": str(path)}, tmp_path)


@pytest.fixture
def frozen_pipeline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"raw_user_id": ["u2", "u1"], "target_raw_item": ["a", "b"],
                          "history_raw_items": [["b", "a"], ["a"]], "target_position": [100, 1]})
    class Dataset:
        examples = frame
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples.iloc[idx].to_dict()
    def collate(examples):
        n = len(examples)
        return {"user_ids": torch.arange(n), "history_token_ids": torch.ones(n, 2, 1, dtype=torch.long),
                "history_token_mask": torch.ones(n, 2, 1, dtype=torch.bool)}
    dataset = Dataset()
    dm = SimpleNamespace(test_dataset=dataset, cfg=SimpleNamespace(max_history_len=2, train_window_len=None),
                         num_items=2, sid_len=1, item2id={"a": 1, "b": 2}, semantic_ids={"a": [1], "b": [2]},
                         test_dataloader=lambda: SimpleNamespace(dataset=dataset, collate_fn=collate), setup=lambda: None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("item_semantic_ids", torch.tensor([[1], [2]]))
        def config_dict(self):
            return {}
        def generate(self, batch, **kwargs):
            return self.item_semantic_ids[None].expand(len(batch["user_ids"]), -1, -1)
    runner = SimpleNamespace(build_data_config=lambda *args, **kwargs: dm.cfg,
                             build_data_module=lambda cfg: dm, build_model=lambda *args, **kwargs: Model())
    monkeypatch.setattr(inference, "_imports", lambda root: runner)
    monkeypatch.setattr(inference, "_code_signatures", lambda root: {})
    monkeypatch.setitem(sys.modules, "sasrec_config_runner", SimpleNamespace(
        training_item_popularity=lambda data: torch.tensor([0, 10, 5])))
    config = {"data": {"source": "yambda", "interactions_path": "events.parquet", "semantic_id_path": "ids.json"},
              "dataset": {"max_history_len": 2}, "model": {}}
    for name in ("events.parquet", "ids.json"):
        (tmp_path / name).touch()
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_config": {}, "model_state_dict": Model().state_dict()}, checkpoint)
    spec = {"run_id": "tiger_init", "model": "tiger", "dataset": "tiny", "stage": "init",
            "data_config": "config.json", "checkpoint": "best.pt"}
    return tmp_path, spec, runner, torch


def test_preparation_inference_round_trip_without_gpu_dataset_load(frozen_pipeline):
    root, spec, runner, torch = frozen_pipeline
    prepared, output_dir = root / "prepared", root / "cache"
    inference.prepare_dataset(spec, repo_root=root, output_dir=prepared)
    runner.build_data_module = lambda *args: pytest.fail("GPU inference rebuilt the original dataset")
    # The original large interaction file is no longer necessary for decoding.
    (root / "events.parquet").unlink()
    output = inference.generate_prepared_run(spec, repo_root=root, prepared_dir=prepared, output_dir=output_dir,
                                             device="cpu", batch_size=1, num_beams=2, top_k=2)
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["user_ids"].tolist() == ["u1", "u2"]
        assert arrays["full_history_lengths"].tolist() == [1, 100]
        assert arrays["recommendations"].tolist() == [["a", "b"], ["a", "b"]]
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["num_users"] == metadata["total_test_users"] == 2
    assert metadata["fingerprint_inputs"]["prepared_fingerprint"]
    assert metadata["fingerprint_inputs"]["exclude_seen_items"] is False
    assert metadata["fingerprint_inputs"]["dtype"] == "float32"
    assert pd.read_csv(output_dir / "catalog.csv").popularity.tolist() == [10, 5]
    # Corrupted shared inputs must never become a new apparently valid run.
    with (prepared / "input_user_ids.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="missing or changed"):
        inference.generate_prepared_run(spec, repo_root=root, prepared_dir=prepared, output_dir=output_dir,
                                        device="cpu", batch_size=1, num_beams=2, top_k=2)


def test_checkpoint_semantic_table_mismatch_is_rejected(frozen_pipeline):
    root, spec, _, torch = frozen_pipeline
    prepared = inference.prepare_dataset(spec, repo_root=root, output_dir=root / "prepared")
    torch.save({"model_config": {}, "model_state_dict": {"item_semantic_ids": torch.tensor([[2], [1]])}}, root / "best.pt")
    with pytest.raises(ValueError, match="semantic-ID row table"):
        inference.generate_prepared_run(spec, repo_root=root, prepared_dir=prepared, output_dir=root / "cache",
                                        device="cpu", batch_size=1, num_beams=2, top_k=2)


def test_prepare_preserves_index_dataset_and_collator_pair(frozen_pipeline):
    root, spec, runner, _ = frozen_pipeline
    dm = runner.build_data_module(None)
    ordinary = dm.test_dataloader()
    seen = []
    class IndexDataset:
        def __len__(self):
            return len(dm.test_dataset)
        def __getitem__(self, idx):
            return int(idx)
    def indexed_collator(indices):
        assert all(isinstance(idx, int) for idx in indices)
        seen.extend(indices)
        return ordinary.collate_fn([dm.test_dataset[idx] for idx in indices])
    dm.test_dataloader = lambda: SimpleNamespace(dataset=IndexDataset(), collate_fn=indexed_collator)
    inference.prepare_dataset(spec, repo_root=root, output_dir=root / "prepared")
    assert seen == [1, 0]  # Frozen rows sort by raw user ID, not source row order.


@pytest.mark.parametrize("force", [False, True])
def test_changed_preparation_replaces_catalog_before_model_load_and_keeps_old_runs(frozen_pipeline, force):
    root, spec, runner, torch = frozen_pipeline
    prepared, cache = root / "prepared", root / "cache"
    inference.prepare_dataset(spec, repo_root=root, output_dir=prepared)
    kwargs = dict(repo_root=root, prepared_dir=prepared, output_dir=cache,
                  device="cpu", batch_size=1, num_beams=2, top_k=2)
    original = inference.generate_prepared_run(spec, **kwargs)
    other = inference.generate_prepared_run({**spec, "run_id": "other"}, **kwargs)
    untouched_npz, untouched_metadata = other.read_bytes(), other.with_suffix(".json").read_bytes()
    old = json.loads(original.with_suffix(".json").read_text())

    # New original interactions legitimately change training popularity while
    # this fixture retains the same semantic-ID space and held-out examples.
    (root / "events.parquet").write_bytes(b"new training interactions")
    sys.modules["sasrec_config_runner"].training_item_popularity = lambda dm: torch.tensor([0, 20, 5])
    inference.prepare_dataset(spec, repo_root=root, output_dir=prepared)
    build_model = runner.build_model
    def assert_catalog_then_build(*args, **kw):
        assert (cache / "catalog.csv").read_bytes() == (prepared / "catalog.csv").read_bytes()
        return build_model(*args, **kw)
    runner.build_model = assert_catalog_then_build
    inference.generate_prepared_run(spec, **kwargs, force=force)
    current = json.loads(original.with_suffix(".json").read_text())
    assert current["catalog_sha256"] != old["catalog_sha256"]
    assert current["fingerprint_inputs"]["prepared_fingerprint"] != old["fingerprint_inputs"]["prepared_fingerprint"]
    assert pd.read_csv(cache / "catalog.csv").popularity.tolist() == [20, 5]
    assert other.read_bytes() == untouched_npz
    assert other.with_suffix(".json").read_bytes() == untouched_metadata
    stale = json.loads(untouched_metadata)
    assert stale["catalog_sha256"] != current["catalog_sha256"]
    assert stale["fingerprint_inputs"]["prepared_fingerprint"] != current["fingerprint_inputs"]["prepared_fingerprint"]


@pytest.mark.parametrize("absolute", [False, True])
def test_legacy_checkpoint_requires_opt_in_and_never_uses_promoted_copy(frozen_pipeline, absolute):
    root, spec, _, torch = frozen_pipeline
    prepared = inference.prepare_dataset(spec, repo_root=root, output_dir=root / "prepared")
    legacy = root / "checkpoints/exp2-tuned_params/initial/tiger/beauty/checkpoints/best.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes((root / "best.pt").read_bytes())
    promoted = root / "checkpoints/exp2/loo_split/initial/tiger/beauty/best.pt"
    promoted.parent.mkdir(parents=True)
    torch.save({"model_config": {}, "model_state_dict": {
        "item_semantic_ids": torch.tensor([[2], [1]])}}, promoted)
    path = legacy.relative_to(root).as_posix()
    if absolute:
        path = "/historical/checkout/flowrec/" + path
    legacy_spec = {**spec, "checkpoint": path}
    kwargs = dict(repo_root=root, prepared_dir=prepared, output_dir=root / "cache",
                  device="cpu", batch_size=1, num_beams=2, top_k=2)
    with pytest.raises(ValueError, match="Unsupported checkpoint path"):
        inference.generate_prepared_run(legacy_spec, **kwargs)
    legacy_spec["preserve_legacy_checkpoint_path"] = True
    output = inference.generate_prepared_run(legacy_spec, **kwargs)
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["recommendations"].tolist() == [["a", "b"], ["a", "b"]]
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["checkpoint"] == path
    assert metadata["fingerprint_inputs"]["checkpoint"]["path"] == str(legacy.resolve())
    legacy.unlink()
    with pytest.raises(FileNotFoundError, match="exp2-tuned_params"):
        inference.generate_prepared_run(legacy_spec, **kwargs)


def test_embedded_data_config_selector_resolves_nested_config_and_reports_missing(frozen_pipeline):
    root, spec, _, _ = frozen_pipeline
    config = json.loads((root / "config.json").read_text())
    envelope = {"trial": {"evaluation": config, "invalid": "not a configuration"}}
    (root / "trial.json").write_text(json.dumps(envelope))
    embedded = {**spec, "data_config": "trial.json", "data_config_selector": ["trial", "evaluation"]}
    assert inference.resolve_data_config(embedded, root) == inference.resolve_data_config(spec, root)
    with pytest.raises(ValueError, match="missing embedded evaluation configuration absent"):
        inference.resolve_data_config({**embedded, "data_config_selector": ["trial", "absent"]}, root)
    with pytest.raises(ValueError, match="expected an evaluation config with a data section"):
        inference.resolve_data_config({**embedded, "data_config_selector": ["trial", "invalid"]}, root)
    assert json.loads((root / "trial.json").read_text()) == envelope


def test_shared_preparation_uses_each_runs_architecture_and_invalidates_changed_config(frozen_pipeline):
    root, spec, runner, _ = frozen_pipeline
    config = json.loads((root / "config.json").read_text())
    config["model"]["d_kv"] = 32
    (root / "config.json").write_text(json.dumps(config))
    prepared = inference.prepare_dataset(spec, repo_root=root, output_dir=root / "prepared")
    config["model"]["d_kv"] = 64
    (root / "other_config.json").write_text(json.dumps(config))
    current = {**spec, "data_config": "other_config.json"}
    captured = []
    original_build = runner.build_model
    def capture_model_config(build, *args, **kwargs):
        captured.append(build["model"]["d_kv"])
        return original_build(build, *args, **kwargs)
    runner.build_model = capture_model_config
    kwargs = dict(repo_root=root, prepared_dir=prepared, output_dir=root / "cache",
                  device="cpu", batch_size=1, num_beams=2, top_k=2)
    output = inference.generate_prepared_run(current, **kwargs)
    first = json.loads(output.with_suffix(".json").read_text())
    config["model"]["d_kv"] = 128
    (root / "other_config.json").write_text(json.dumps(config))
    inference.generate_prepared_run(current, **kwargs)
    second = json.loads(output.with_suffix(".json").read_text())
    assert captured == [64, 128]
    assert first["fingerprint"] != second["fingerprint"]
    assert json.loads((prepared / "metadata.json").read_text())["config"]["model"]["d_kv"] == 32


@pytest.mark.parametrize("user_ids_count", [None, 1, 2000])
def test_gpt2_generation_accepts_optional_user_hash_count(frozen_pipeline, monkeypatch, user_ids_count):
    """Exercise the real builder with a tiny constructor, without transformers."""
    import ast
    import __future__

    root, spec, runner, torch = frozen_pipeline
    # Load the production function without importing the training CLI's optional
    # dependencies. A fake constructor only replaces the neural network itself.
    source = Path(__file__).resolve().parents[1] / "scripts/run_gpt2.py"
    tree = ast.parse(source.read_text())
    build_function = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "build_model")
    captured = []

    def construct(**kwargs):
        captured.append(kwargs)
        return runner.build_model()

    namespace = {"GPT2RecModel": construct}
    exec(compile(ast.Module(body=[build_function], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    monkeypatch.setitem(sys.modules, "run_gpt2", SimpleNamespace(build_model=namespace["build_model"]))
    config = json.loads((root / "config.json").read_text())
    config["model"].update(use_user_token=False, user_ids_count=user_ids_count)
    (root / "config.json").write_text(json.dumps(config))
    current = {**spec, "model": "gpt2", "run_id": "gpt2_nullable_user_count"}
    prepared = inference.prepare_dataset(current, repo_root=root, output_dir=root / "prepared")
    runner.build_data_module = lambda *args: pytest.fail("Generation must reuse prepared inputs")
    output = inference.generate_prepared_run(
        current, repo_root=root, prepared_dir=prepared, output_dir=root / "cache",
        device="cpu", batch_size=1, num_beams=2, top_k=2,
    )
    assert len(captured) == 1 and "user_ids_count" not in captured[0]
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["user_ids"].tolist() == ["u1", "u2"]
        assert arrays["recommendations"].tolist() == [["a", "b"], ["a", "b"]]
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["fingerprint_inputs"]["config"]["model"]["user_ids_count"] == user_ids_count


@pytest.mark.parametrize("model_name", ["gpt2", "tiger"])
@pytest.mark.parametrize("stage", ["sft2", "gflow", "ce"])
def test_from_sft_stages_decode_loaded_weights_for_every_test_example(
    frozen_pipeline, monkeypatch, model_name, stage,
):
    """Exercise all second-stage routes with ten ranked raw items, no raw reload."""
    root, spec, runner, torch = frozen_pipeline
    dm = runner.build_data_module(None)
    item_ids = list("abcdefghijkl")
    dm.semantic_ids = {item: [index + 1] for index, item in enumerate(item_ids)}
    dm.item2id = {item: index + 1 for index, item in enumerate(item_ids)}
    dm.num_items = len(item_ids)
    sys.modules["sasrec_config_runner"].training_item_popularity = (
        lambda data: torch.arange(13)
    )

    class RankedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("item_semantic_ids", torch.arange(1, 13).reshape(-1, 1))
            self.register_buffer("item_scores", -torch.arange(12, dtype=torch.float32))

        def config_dict(self):
            return {}

        def generate(self, batch, **kwargs):
            assert kwargs["constrain_valid_sequences"] is True
            assert kwargs["exclude_seen_items"] is False
            rows = self.item_scores.argsort(descending=True)[:kwargs["num_return_sequences"]]
            return self.item_semantic_ids[rows][None].expand(len(batch["user_ids"]), -1, -1)

    selected = RankedModel()
    # Loading this checkpoint reverses the freshly constructed model's ranking.
    selected.item_scores.copy_(torch.arange(12, dtype=torch.float32))
    torch.save({"model_state_dict": selected.state_dict(), "model_config": {}}, root / "best.pt")
    called = []

    def build_model(build, frozen_dm, **kwargs):
        called.append(model_name)
        assert frozen_dm.semantic_ids == dm.semantic_ids
        assert "checkpoint_path" not in build["model"]
        return RankedModel()

    if model_name == "gpt2":
        monkeypatch.setitem(sys.modules, "run_gpt2", SimpleNamespace(build_model=build_model))
        runner.build_model = lambda *args, **kwargs: pytest.fail("GPT-2 used TIGER's constructor")
    else:
        runner.build_model = build_model

    family = "experiments/exp8-from-sft/example/sft_seed_42"
    config_path = root / family / "data_reward_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((root / "config.json").read_bytes())
    current = {**spec, "model": model_name, "stage": stage, "source_family": family,
               "data_config": f"{family}/data_reward_config.json", "source_seed": 42,
               "init_checkpoint": "first_sft_only/checkpoints/best.pt"}
    prepared = inference.prepare_dataset(current, repo_root=root, output_dir=root / "prepared")
    runner.build_data_module = lambda *args: pytest.fail("Generation rebuilt training inputs")
    (root / "events.parquet").unlink()
    output = inference.generate_prepared_run(
        current, repo_root=root, prepared_dir=prepared, output_dir=root / "cache",
        device="cpu", batch_size=1, num_beams=20, top_k=10,
    )
    assert called == [model_name]
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["user_ids"].tolist() == ["u1", "u2"]
        assert arrays["recommendations"].tolist() == [list("lkjihgfedc")] * 2
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["stage"] == stage
    assert metadata["init_checkpoint"] == current["init_checkpoint"]
    assert metadata["num_users"] == metadata["total_test_users"] == 2
