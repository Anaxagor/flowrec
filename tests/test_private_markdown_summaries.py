"""Generated Markdown retains metrics without exposing local run details."""
from __future__ import annotations

import copy

import pytest

from scripts import run_gpt2, run_tiger, sasrec_config_runner


@pytest.mark.parametrize("runner", [run_tiger, run_gpt2, sasrec_config_runner])
@pytest.mark.parametrize("known_dataset", [True, False])
def test_summary_keeps_metrics_but_not_private_run_data(tmp_path, runner, known_dataset):
    private_user = "private_researcher"
    private_job = "93847123"
    run_dir = f"/home/{private_user}/flowrec/experiments/run_{private_job}"
    dataset_name = "beauty" if known_dataset else private_user
    summary = {
        "dataset_name": dataset_name,
        "dataset": {
            "name": dataset_name,
            "reviews_path": f"/Users/{private_user}/data/reviews.json",
            "interactions_path": f"/home/{private_user}/data/events.parquet",
            "semantic_id_path": f"/home/{private_user}/semantic_ids.json",
        },
        "seed": 42,
        "epochs_completed": 3,
        "steps_completed": 125,
        "best_metric": 0.27,
        "best_checkpoint": f"{run_dir}/best.pt",
        "last_checkpoint": f"{run_dir}/last.pt",
        "experiment_dir": run_dir,
        "run_dir": run_dir,
        "validation_metrics": {
            "val/sft/ndcg@10": 0.27,
            "HR@5": 0.42,
            "NDCG@20": 0.19,
            "Coverage@5": 0.38,
            "ARP@10": 12.5,
            "job_id": int(private_job),
            f"/home/{private_user}/metric": 0.99,
            "private_value": f"/home/{private_user}",
        },
        "test_metrics": {"test/sft/recall@5": 0.31},
    }
    before = copy.deepcopy(summary)
    json_before = tmp_path / "before.json"
    json_after = tmp_path / "after.json"
    markdown = tmp_path / "final_summary.md"
    runner.write_json(json_before, summary)
    runner.write_text_summary(markdown, summary)
    runner.write_json(json_after, summary)

    text = markdown.read_text(encoding="utf-8")
    assert private_user not in text
    assert private_job not in text
    assert "/home/" not in text and "/Users/" not in text
    assert '"val/sft/ndcg@10": 0.27' in text
    assert '"HR@5": 0.42' in text
    assert '"NDCG@20": 0.19' in text
    assert '"Coverage@5": 0.38' in text
    assert '"ARP@10": 12.5' in text
    assert '"test/sft/recall@5": 0.31' in text
    assert "Epochs completed: 3" in text and "Steps completed: 125" in text
    assert "Best metric: 0.27" in text and "Seed: 42" in text
    assert "final_summary.json" in text
    assert ("Dataset: beauty" in text) is known_dataset
    assert summary == before
    assert json_before.read_bytes() == json_after.read_bytes()
