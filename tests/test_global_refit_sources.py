"""Inventory and retry behavior shared by submitters and array workers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
import refit_sources as sources


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_init_picks_complete_trial_across_partial_studies_and_uses_its_own_epoch(tmp_path):
    directory = tmp_path / "experiments/global-time/hpo/tiger_beauty2023/123"
    rows = [dict(trial="trial_0001", status="complete", best_value=.2, overrides={"lr": .01}),
            dict(trial="trial_0002", status="pruned", best_value=.9, overrides={"lr": .2})]
    _write_json(directory / "artifacts/tiger_beauty_2023_tuning_summary.json",
                {"dataset": "beauty_2023", "search_complete": False, "trials": rows})
    _write_json(tmp_path / "checkpoints/exp2/global_split/initial/tiger/beauty_2023/best_trial.json",
                dict(trial="trial_0000", status="complete", best_value=.1, best_epoch=12, overrides={"lr": .1}))
    _write_json(directory / "checkpoints/beauty_2023/seed_42/result.json", {"best_epoch": 30})
    with pytest.raises(sources.InitialSelectionUnavailable, match="best_epoch"):
        sources.select_initial(tmp_path, "tiger", "beauty_2023")
    _write_json(directory / "checkpoints/beauty_2023/trials/trial_0001/result.json",
                dict(best_checkpoint="trials/trial_0001/best.pt", status="complete", monitor="val/sft/ndcg@10",
                     best_epoch=3, epochs_completed=14, best_value=.2))
    selected = sources.select_initial(tmp_path, "tiger", "beauty_2023")
    assert (selected["trial"], selected["epochs"], selected["seed"]) == ("trial_0001", 4, 42)


def test_init_excludes_full_history_yambda_and_derives_hpo_seed(tmp_path):
    dataset = "yambda_50m_listens"
    directory = tmp_path / "checkpoints/exp2/global_split/initial/gpt2" / dataset
    data = dict(dataset=dataset, metric="val/sft/generation_ndcg@10", best_epoch=0, best_trial_number=6,
                best_value_val_generation_ndcg10=.4, best_params={"lr": .1})
    _write_json(directory / "full_history/best_trial.json", data)
    with pytest.raises(sources.InitialSelectionUnavailable):
        sources.select_initial(tmp_path, "gpt2", dataset)
    _write_json(directory / "last_128/best_trial.json", dict(data, best_value_val_generation_ndcg10=.1))
    selected = sources.select_initial(tmp_path, "gpt2", dataset)
    assert (selected["validation_ndcg10"], selected["epochs"], selected["seed"]) == (.1, 1, 48)


def test_init_refines_rounded_tuned_yaml_from_original_gpt2_trial_sidecar(tmp_path):
    dataset = "yambda_500m_listens"
    job = tmp_path / "experiments/global-time/hpo/gpt2_yambda500m/last_128/123"
    yaml = job / f"configs/gpt2_model_{dataset}.yaml"
    yaml.parent.mkdir(parents=True)
    yaml.write_text(f"# study: gpt2_{dataset}_val_generation_ndcg10_exp2_last_128_global_time\n"
                    "# selection: best single-seed trial of the search\n# best trial: 17, best epoch: 29\n"
                    '# best val/sft/generation_ndcg@10: 0.113758\n# search-space point: {"num_layers": 4}\n')
    trial = job / "checkpoints" / dataset / "trials/trial_17"
    _write_json(trial / "best.pt.json", dict(model_type="gpt2", trainer_config={
        "checkpoint_dir": str(trial), "monitor": "val/sft/generation_ndcg@10"},
        trainer_state={"best_epoch": 29, "epoch": 29, "best_metric": .1137576505}))
    s = sources.select_initial(tmp_path, "gpt2", dataset)
    assert (s["epochs"], s["validation_ndcg10"], len(s["sources"])) == (30, .1137576505, 2)


def test_init_reads_completed_live_journal_before_provisional_gpt2_promotion(tmp_path):
    optuna = pytest.importorskip("optuna")
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    directory = tmp_path / "experiments/global-time/hpo/gpt2_beauty2023/321/checkpoints/beauty_2023"
    _write_json(directory / "best_trial.json", {"best_trial_number": 999})
    storage = JournalStorage(JournalFileBackend(str(directory / "optuna_journal.log")))
    study = optuna.create_study(storage=storage, direction="maximize",
                                study_name="gpt2_beauty_2023_val_generation_ndcg10_exp2_global_time")
    trial = study.ask()
    trial.suggest_int("num_layers", 1, 2)
    trial.set_user_attr("best_epoch", 3)
    study.tell(trial, .2)
    running = study.ask()
    running.report(.9, 0)
    # A provisional export and a new, still-empty journal can coexist with this study.
    _write_json(directory.parents[1] / "artifacts/gpt2_beauty_2023_best_trial.json",
                dict(best_trial_number=1, best_epoch=0, best_params={"num_layers": 2},
                     metric="val/sft/generation_ndcg@10", best_value_val_generation_ndcg10=.9))
    new_job = directory.parents[2] / "322/checkpoints/beauty_2023"
    new_job.mkdir(parents=True)
    (new_job / "optuna_journal.log").touch()
    selected = sources.select_initial(tmp_path, "gpt2", "beauty_2023")
    assert (selected["trial"], selected["epochs"], selected["validation_ndcg10"]) == ("trial_0000", 4, .2)


def create_source(root, source, *, epochs=15):
    directory = root / source["source_id"]
    (directory / "checkpoints").mkdir(parents=True)
    (directory / "checkpoints/best.pt").write_bytes(b"fixture")
    (directory / "resolved_config.json").write_text(json.dumps({"trainer": {"patience": 5}, "optimizer": {"lr": 3e-5}}))
    result = {**source, "status": "completed", "epochs_completed": epochs,
              "seed": source["source_seed"], "best_epoch": -1, "best_checkpoint": str(root / "previous-checkout/checkpoints/best.pt")}
    if source["arm"] == "sft":
        result["variant"] = "sft_only"
    elif source["arm"] == "gflow":
        result["target_grad_ratio"] = result.pop("rho")
    (directory / "result.json").write_text(json.dumps(result))
    return directory


def test_catalog_covers_legacy_and_compact_grids_and_distinguishes_initial_seeds():
    cells = list(sources.catalog())
    assert len(cells) == len({cell["source_id"] for cell in cells}) == 336
    assert len([s for s in cells if s["arm"] == "sft"]) == 24
    assert len([s for s in cells if s["arm"] == "ce"]) == 60
    assert len([s for s in cells if s["arm"] == "gflow"]) == 252
    for dataset in ("beauty", "instruments"):
        legacy = [s for s in cells if s["dataset"] == dataset]
        assert {s["model"] for s in legacy if s["arm"] != "ce"} == {"gpt2"}
        assert {s["num_samples"] for s in legacy if s["arm"] == "ce"} == {10}
        assert {s["sampler"] for s in legacy if s["arm"] == "gflow"} == {"teacher", "policy", "q", "random"}
    for dataset in sources.COMPACT_DATASETS:
        for model in ("tiger", "gpt2"):
            configs = [s for s in cells if (s["model"], s["dataset"]) == (model, dataset)]
            assert len(configs) == 27
            assert {s["source_seed"] for s in configs if s["arm"] == "sft"} == {42, 43, 44}
            assert {s["num_samples"] for s in configs if s["arm"] == "ce"} == {3}
            assert all(("/sampled50/" in s["source_id"]) == (dataset == sources.global_grid.Y500) for s in configs)


@pytest.fixture
def tiny_catalog(tmp_path, monkeypatch):
    source = next(s for s in sources.catalog() if s["model"] == "gpt2" and s["arm"] == "gflow")
    monkeypatch.setattr(sources, "catalog", lambda: iter([source]))
    ledger = tmp_path / "test_global_train_with_val.jsonl"
    directory = create_source(tmp_path, source)
    return directory, ledger, source


def test_completed_epoch_count_and_atomic_concurrent_metric_publication(tmp_path, tiny_catalog):
    directory, ledger, source = tiny_catalog
    tasks, stats = sources.plan_missing(tmp_path, ledger)
    assert stats["planned_runs"] == 3 and stats["invalid_sources"] == []
    assert {t["seed"] for t in tasks} == {42, 43, 44}
    assert all(t["epochs"] == 15 and t["source_best_epoch"] == -1 for t in tasks)
    assert all(Path(t["source_checkpoint"]) == directory / "checkpoints/best.pt" for t in tasks)
    metrics = dict.fromkeys(sources.METRIC_KEYS, 0.15)
    metrics["num_examples"] = 101
    def commit(task):
        return sources.append_metrics(ledger, task, metrics, details={"expected_test_examples": 101})
    with pytest.raises(ValueError, match="incomplete test split"):
        sources.append_metrics(ledger, tasks[0], metrics, details={"expected_test_examples": 105})
    assert not ledger.exists()
    with pytest.raises(ValueError, match="nonfinite"):
        sources.append_metrics(ledger, tasks[0], {**metrics, "NDCG@10": float("nan")})
    with ThreadPoolExecutor(6) as pool:
        assert sum(pool.map(commit, tasks + tasks)) == 3
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["source_finetune_config"]["optimizer"]["lr"] == 3e-5
    assert rows[0]["training_split"] == "train+validation"
    assert all(sources.has_metrics(ledger, t["run_key"]) for t in tasks)
    # The completion file is consulted first. Already scored sources can be archived.
    (directory / "result.json").unlink()
    remaining, stats = sources.plan_missing(tmp_path, ledger)
    assert remaining == [] and stats["already_scored"] == 3 and stats["missing_results"] == 0


def test_partial_ledger_changed_source_and_missing_files(tmp_path, tiny_catalog):
    directory, ledger, source = tiny_catalog
    tasks, _ = sources.plan_missing(tmp_path, ledger)
    metrics = dict.fromkeys(sources.METRIC_KEYS, 0.)
    metrics["num_examples"] = 2
    assert sources.append_metrics(ledger, tasks[0], metrics)
    remaining, _ = sources.plan_missing(tmp_path, ledger)
    assert [t["seed"] for t in remaining] == [43, 44]
    (directory / "resolved_config.json").write_text('{"trainer":{"lr":0.1}}')
    with pytest.raises(ValueError, match="has changed"):
        sources.verify_source(tasks[1])
    assert sources.plan_missing(tmp_path, ledger)[1]["invalid_sources"]
    (directory / "checkpoints/best.pt").unlink()
    assert sources.plan_missing(tmp_path, ledger)[1]["missing_checkpoints"] == 1
    (directory / "checkpoints/last.pt").write_bytes(b"in-progress")
    assert sources.plan_missing(tmp_path, ledger)[0] == []
    (directory / "result.json").write_text('{"status":"failed"}')
    assert sources.plan_missing(tmp_path, ledger)[1]["unfinished_sources"] == 1
