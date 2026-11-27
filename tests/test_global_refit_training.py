"""The final global fit uses its init weights, exact complete epochs, and a single final test."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

from test_ce_distill_gpt2_beauty2023_smoke import tiny_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
import refit_inputs
import refit_sources
import refit_training as worker
from flowrec.training import TrainerConfig


def test_refit_sft_uses_all_epochs_and_recovers_final_evaluation(tmp_path, monkeypatch, tiny_inputs):
    _, _, dm, paths = tiny_inputs
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = next(s for s in refit_sources.catalog()
                  if (s["model"], s["dataset"], s["arm"]) == ("gpt2", "beauty_2023", "sft"))
    directory = tmp_path / source["source_id"]
    (directory / "checkpoints").mkdir(parents=True)
    # This selected source checkpoint cannot be loaded as model weights.
    (directory / "checkpoints/best.pt").write_bytes(b"validation-selected source")
    (directory / "result.json").write_text(json.dumps({
        "status": "completed", "dataset": "beauty_2023", "model": "gpt2", "variant": "sft_only",
        "seed": source["source_seed"], "epochs_completed": 2, "best_epoch": -1}))
    resolved = {"trainer": TrainerConfig(max_epochs=50, early_stopping_patience=1, max_train_batches=1,
                                        max_eval_batches=1, run_validation_at_start=True,
                                        gradient_accumulation_steps=2).to_dict(),
                "batching": {"batch_size": 4, "gradient_accumulation": 2, "test_batch_size": 1},
                "optimizer": {"lr": .001, "weight_decay": .01}, "checkpoint_every_updates": 0}
    (directory / "resolved_config.json").write_text(json.dumps(resolved))
    tasks, _ = refit_sources.plan_missing(tmp_path, tmp_path / "metrics.jsonl")
    assert len(tasks) == 3 and {t["seed"] for t in tasks} == {42, 43, 44}
    task = tasks[0]
    dm._ce_prepared_manifest = str(inputs / "manifest.json")
    (inputs / "manifest.json").write_text(json.dumps({"artifacts": {"tensors": {"sha256": "fixed"}}}))
    arch = json.loads(paths["sidecar"].read_text())["model_config"]
    config = {"model": arch, "refit_audit": {"source_identity": {"student": str(paths["checkpoint"])}}}

    def load_inputs(workspace, family, dataset, *, need_factors, num_workers, batch_size, eval_batch_size):
        assert (family, dataset, need_factors) == ("gpt2", "beauty_2023", False)
        dm.cfg = replace(dm.cfg, batch_size=batch_size, eval_batch_size=eval_batch_size,
                         test_batch_size=eval_batch_size, num_workers=num_workers)
        return dm, config, paths["checkpoint"], {}

    monkeypatch.setattr(refit_inputs, "load_refit_inputs", load_inputs)
    monkeypatch.setattr(dm, "val_dataloader", lambda: pytest.fail("Refit ran validation"))
    evaluate = worker.gpt.gpt2_beauty.evaluate_splits
    folder = tmp_path / "final"
    saved = folder / "checkpoints/last.pt"

    def interrupt_evaluation(*args, splits):
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        assert payload["epoch_progress"]["epoch_complete"]
        assert payload["trainer_state"]["epoch"] == 1
        assert splits == ("test",)
        raise RuntimeError("simulated interruption after training")

    monkeypatch.setattr(worker.gpt.gpt2_beauty, "evaluate_splits", interrupt_evaluation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        worker.run(task, inputs, folder, device="cpu", num_workers=0)
    assert not (folder / "checkpoints/best.pt").exists()
    assert not (folder / "result.json").exists()
    completed_at = saved.stat().st_mtime_ns
    monkeypatch.setattr(worker.gpt.gpt2_beauty, "evaluate_splits", evaluate)
    result = worker.run(task, inputs, folder, device="cpu", num_workers=0)
    assert saved.stat().st_mtime_ns == completed_at
    assert result["initial_checkpoint"] == str(paths["checkpoint"])
    assert result["epochs_completed"] == task["epochs"] == 2
    assert result["micro_steps"] == 2 * ((len(dm.train_dataset) + 3) // 4)
    assert result["expected_test_examples"] == len(dm.test_dataset)
    assert result["test_metrics"]["num_examples"] == len(dm.test_dataset)
    assert set(refit_sources.METRIC_KEYS) <= set(result["test_metrics"])
    monkeypatch.setattr(refit_inputs, "load_refit_inputs", lambda *a, **kw: pytest.fail("Trained twice"))
    assert worker.run(task, inputs, folder, device="cpu", num_workers=0)["test_metrics"] == result["test_metrics"]
