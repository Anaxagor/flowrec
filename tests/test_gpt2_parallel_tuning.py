"""Shared journals, baseline enqueueing and winner publication across workers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from threading import Barrier

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState, create_trial
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpt2_parallel_tuning_test", ROOT / "scripts/exp2-tuned-params/tuning_gpt2.py"
)
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)
DATASET = "yambda_50m_listens"


@pytest.fixture
def ctx(tmp_path):
    args = tuning.parse_args([
        "--match-tiger", "--datasets", DATASET, "--device", "cpu",
        "--output-root", str(tmp_path / "out"), "--num-workers", "0",
        "--n-trials", "1", "--no-progress-bar",
    ])
    result = tuning.build_context(args, tuning.load_regime()).for_dataset(DATASET)
    result.dataset_dir(DATASET).mkdir(parents=True)
    return result


def distributions(ctx):
    spec = ctx.spec(DATASET)
    return {
        "embedding_dim": CategoricalDistribution(spec.embedding_dim),
        "num_heads": CategoricalDistribution(spec.num_heads),
        "num_layers": IntDistribution(*spec.num_layers),
        "ff_mult": CategoricalDistribution(spec.ff_mult),
        "dropout": FloatDistribution(*spec.dropout),
        "lr": FloatDistribution(*spec.lr, log=True),
        "weight_decay": FloatDistribution(*spec.weight_decay, log=True),
    }


def completed_trial(ctx, *, value=0.1, **overrides):
    return create_trial(
        params={**ctx.regime.baselines[DATASET], **overrides},
        distributions=distributions(ctx), value=value, user_attrs={"best_epoch": 2},
    )


def test_four_simultaneous_workers_enqueue_baseline_once(ctx):
    journal = ctx.dataset_dir(DATASET) / "journal.log"
    storage = JournalStorage(JournalFileBackend(str(journal)))
    study = optuna.create_study(storage=storage, study_name="parallel", direction="maximize")
    barrier = Barrier(4)

    def worker(_):
        # Separate storage instances mimic independently started workers.
        local = optuna.load_study(
            study_name="parallel", storage=JournalStorage(JournalFileBackend(str(journal)))
        )
        barrier.wait(timeout=10)
        return tuning.ensure_study_incumbent(local, ctx=ctx, dataset_name=DATASET)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    assert results.count(True) == 1
    trials = study.get_trials(deepcopy=False)
    assert len(trials) == 1 and trials[0].state == TrialState.WAITING
    assert tuning.trial_has_params(trials[0], ctx.regime.baselines[DATASET])


def test_nonzero_worker_queues_baseline_and_uses_remaining_shared_budget(ctx, monkeypatch):
    # Work from earlier workers already accounts for two of three planned
    # trials. A nonzero worker must only finish the remaining shared budget.
    ctx = replace(ctx, n_trials_override=3)
    study = optuna.create_study(direction="maximize")
    tuning.validate_study_protocol(study, DATASET, ctx)
    study.add_trial(completed_trial(ctx, embedding_dim=128))
    study.add_trial(create_trial(state=TrialState.PRUNED))
    monkeypatch.setattr(tuning.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(tuning, "build_datamodule", lambda *args: object())
    executed = []

    def objective_factory(**kwargs):
        def objective(trial):
            executed.append(trial.number)
            params = tuning.suggest_hparams(trial, kwargs["spec"])
            assert params == ctx.regime.baselines[DATASET]
            return 0.2
        return objective

    monkeypatch.setattr(tuning, "make_objective", objective_factory)
    summary = tuning.tune_one_dataset(ctx, DATASET, worker_id=7)
    assert executed == [2]
    assert tuning.count_finished(study) == 3
    assert summary["trials_completed"] == 3
    assert study.trials[2].system_attrs["fixed_params"] == ctx.regime.baselines[DATASET]


def test_winner_is_refreshed_after_waiting_for_publication_lock(ctx, monkeypatch):
    study = optuna.create_study(direction="maximize")
    tuning.validate_study_protocol(study, DATASET, ctx)
    study.add_trial(completed_trial(ctx))
    monkeypatch.setattr(tuning.optuna, "create_study", lambda **kwargs: study)
    original_lock = tuning.promotion_lock
    entered = []

    @contextmanager
    def competing_worker_lock(path, *, strict=False):
        with original_lock(path, strict=strict):
            entered.append(strict)
            # Protocol and baseline locks happen first. Another worker
            # completes before this process acquires its publication lock.
            if len(entered) == 3:
                study.add_trial(completed_trial(ctx, value=0.9, embedding_dim=128))
            yield

    monkeypatch.setattr(tuning, "promotion_lock", competing_worker_lock)
    summary = tuning.tune_one_dataset(ctx, DATASET, worker_id=3)
    assert summary["best_trial_number"] == 1
    assert summary["best_params"]["embedding_dim"] == 128
    assert summary["best_value"] == 0.9
    assert summary["trials_completed"] == 2
    assert entered == [True, True, True]
    doc = yaml.safe_load(Path(summary["tuned_config"]).read_text())
    assert doc["datasets"][DATASET]["model"]["embedding_dim"] == 128
    assert "# best trial: 1" in Path(summary["tuned_config"]).read_text()
    assert len(json.loads((ctx.dataset_dir(DATASET) / "all_trials.json").read_text())) == 2


def test_published_params_and_value_come_from_one_trial_snapshot(ctx, monkeypatch):
    study = optuna.create_study(direction="maximize")
    tuning.validate_study_protocol(study, DATASET, ctx)
    study.add_trial(completed_trial(ctx))
    monkeypatch.setattr(tuning.optuna, "create_study", lambda **kwargs: study)

    def unrelated_latest_value(_):
        raise AssertionError("Read best.value from the captured FrozenTrial, not study.best_value")

    monkeypatch.setattr(optuna.Study, "best_value", property(unrelated_latest_value))
    summary = tuning.tune_one_dataset(ctx, DATASET, worker_id=2)
    assert summary["best_value"] == 0.1
    assert summary["best_params"] == ctx.regime.baselines[DATASET]


def test_matched_publication_fails_when_flock_is_unavailable(ctx, monkeypatch):
    def no_lock(*args):
        raise OSError("shared filesystem lock unavailable")

    monkeypatch.setattr(tuning.fcntl, "flock", no_lock)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    with pytest.raises(OSError, match="lock unavailable"):
        tuning.promote_if_incumbent(
            ctx=ctx, dataset_name=DATASET, trial=trial,
            hparams=ctx.regime.baselines[DATASET], value=0.3, best_epoch=0,
            checkpoint_path=None, study_name=study.study_name,
        )
    assert not (ctx.dataset_dir(DATASET) / "best_trial.json").exists()
