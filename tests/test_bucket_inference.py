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
