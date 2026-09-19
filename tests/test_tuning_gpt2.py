"""Data-free tests for the exp2 GPT-2 Optuna tuning script (tuning_gpt2.py).

They cover the search-space/baseline containment against the real exp1
configs, the exp1-profile parsing guards, the tuned-YAML emitter (which must
round-trip through the exp1 runner's profile reader), and the study-resume
helpers (baseline re-enqueue and stale-trial cleanup) on a journal study.
The end-to-end CPU smoke test is documented in the script's usage section.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import optuna
import pytest
import yaml
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from flowrec.ft_methods.sft.config import SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "exp2-tuned-params" / "tuning_gpt2.py"
SPEC = importlib.util.spec_from_file_location("tuning_gpt2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


@pytest.fixture(scope="module")
def regime() -> "tuning.FixedRegime":
    return tuning.load_regime()


def _distributions(spec: "tuning.TuningSpec") -> dict[str, object]:
    return {
        "embedding_dim": CategoricalDistribution(list(spec.embedding_dim)),
        "num_heads": CategoricalDistribution(list(spec.num_heads)),
        "num_layers": IntDistribution(*spec.num_layers),
        "ff_mult": CategoricalDistribution(list(spec.ff_mult)),
        "dropout": FloatDistribution(*spec.dropout),
        "lr": FloatDistribution(*spec.lr, log=True),
        "weight_decay": FloatDistribution(*spec.weight_decay, log=True),
    }


# ---------------------------------------------------------------------------
# Search space and baselines
# ---------------------------------------------------------------------------

def test_every_space_contains_its_exp1_baseline(regime) -> None:
    tuning.validate_specs(regime)
    assert set(tuning.TUNING_SPECS) == set(tuning.DATASETS_TO_TUNE)
    assert set(regime.baselines) == set(tuning.DATASETS_TO_TUNE)
    for name, base in regime.baselines.items():
        assert set(base) == set(tuning.SEARCH_PARAMS), name


def test_budgets_are_consistent() -> None:
    for name, spec in tuning.TUNING_SPECS.items():
        assert 0 < spec.n_startup_trials < spec.n_trials, name
        assert spec.pruner_warmup_epochs < spec.max_epochs, name
        assert spec.patience < spec.max_epochs, name
    small = tuning.TUNING_SPECS["beauty"].n_trials
    assert tuning.TUNING_SPECS["instruments"].n_trials == small
    assert small > tuning.TUNING_SPECS["beauty_2023"].n_trials
    assert tuning.TUNING_SPECS["beauty_2023"].n_trials > tuning.TUNING_SPECS["yambda_50m_listens"].n_trials
    assert tuning.TUNING_SPECS["yambda_50m_listens"].n_trials > tuning.TUNING_SPECS["yambda_500m_listens"].n_trials
    # Large datasets monitor a fixed subsample; the small Amazon sets the full split.
    assert tuning.TUNING_SPECS["beauty"].val_sample_fraction is None
    assert tuning.TUNING_SPECS["yambda_50m_listens"].val_sample_fraction == 0.1


def test_validate_specs_rejects_a_space_without_the_baseline(regime) -> None:
    from dataclasses import replace

    original = tuning.TUNING_SPECS["beauty"]
    tuning.TUNING_SPECS["beauty"] = replace(original, num_layers=(1, 2))  # baseline has 3
    try:
        with pytest.raises(ValueError, match="excludes the exp1 baseline"):
            tuning.validate_specs(regime)
    finally:
        tuning.TUNING_SPECS["beauty"] = original


def test_validate_specs_rejects_heads_that_do_not_divide_the_width(regime) -> None:
    from dataclasses import replace

    original = tuning.TUNING_SPECS["beauty"]
    tuning.TUNING_SPECS["beauty"] = replace(original, num_heads=(3, 8))
    try:
        with pytest.raises(ValueError, match="does not divide"):
            tuning.validate_specs(regime)
    finally:
        tuning.TUNING_SPECS["beauty"] = original


def test_exp1_baseline_parsing_guards() -> None:
    model_config = {
        "defaults": {"model": {"user_ids_count": 1}},
        "datasets": {
            "odd_ffn": {
                "model": {"embedding_dim": 256, "dim_feedforward": 700, "num_heads": 8, "num_layers": 3, "dropout": 0.1},
                "optimizer": {"lr": 1e-3},
            },
            "split_dropout": {
                "model": {"embedding_dim": 256, "dim_feedforward": 768, "num_heads": 8, "num_layers": 3, "dropout": 0.1, "attention_dropout": 0.3},
                "optimizer": {"lr": 1e-3},
            },
            "no_lr": {
                "model": {"embedding_dim": 256, "dim_feedforward": 768, "num_heads": 8, "num_layers": 3, "dropout": 0.1},
            },
            "ok": {
                "model": {"embedding_dim": 256, "dim_feedforward": 768, "num_heads": 8, "num_layers": 3, "dropout": 0.1, "codebook_size": 512},
                "optimizer": {"lr": 1e-3, "weight_decay": 0.05},
            },
        },
    }
    with pytest.raises(ValueError, match="not a multiple"):
        tuning.exp1_baseline(model_config, "odd_ffn", weight_decay=0.01)
    with pytest.raises(ValueError, match="single dropout"):
        tuning.exp1_baseline(model_config, "split_dropout", weight_decay=0.01)
    with pytest.raises(KeyError, match="optimizer.lr"):
        tuning.exp1_baseline(model_config, "no_lr", weight_decay=0.01)
    params, codebook = tuning.exp1_baseline(model_config, "ok", weight_decay=0.01)
    assert params == {
        "embedding_dim": 256, "num_heads": 8, "num_layers": 3, "ff_mult": 3,
        "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.05,
    }
    assert codebook == 512


def test_profile_derives_the_ffn_width_and_fans_dropout_out() -> None:
    hparams = {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_mult": 4, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-2}
    profile = tuning.build_profile(hparams, tuning.TUNING_SPECS["yambda_50m_listens"], codebook_size=256)
    assert profile["model"]["dim_feedforward"] == 512
    assert {profile["model"][k] for k in ("dropout", "attention_dropout", "residual_dropout", "embedding_dropout")} == {0.25}
    assert profile["optimizer"] == {"lr": 1e-3, "weight_decay": 1e-2}
    assert profile["training"] == {"val_sample_fraction": 0.1, "val_sample_seed": 0, "val_sample_resample": False}
    assert tuning.build_profile(hparams, tuning.TUNING_SPECS["beauty"], codebook_size=256)["training"] == {}


# ---------------------------------------------------------------------------
# Tuned YAML
# ---------------------------------------------------------------------------

def test_tuned_yaml_round_trips_through_the_exp1_runner_profile_reader(regime, tmp_path: Path) -> None:
    ctx = tuning.RunContext(
        regime=regime,
        semantic_ids_dir=tuning.DEFAULT_SEMANTIC_IDS_DIR,
        output_root=tmp_path / "out",
        config_out_dir=tmp_path / "configs",
    )
    name = "yambda_50m_listens"
    hparams = {**regime.baselines[name], "lr": 7.5e-4, "weight_decay": 3e-2, "ff_mult": 4}
    path = tuning.write_tuned_config(
        ctx=ctx, dataset_name=name, hparams=hparams, best_value=0.1234,
        best_trial_number=7, best_epoch=12, trials_completed=40, trials_planned=40,
        study_name=tuning.study_name_for(name),
    )
    assert path == tmp_path / "configs" / f"gpt2_model_{name}.yaml"
    text = path.read_text()
    assert text.startswith("# GPT-2 hyperparameters for yambda_50m_listens")
    doc = yaml.safe_load(text)

    profile = tuning.run_all.model_profile(doc, name)
    assert profile["model"]["embedding_dim"] == 512
    assert profile["model"]["dim_feedforward"] == 4 * 512
    assert profile["optimizer"] == {"lr": 7.5e-4, "weight_decay": 3e-2}
    assert profile["training"]["val_sample_fraction"] == 0.1
    assert tuning.run_all.resolve_ranking_ks(doc) == (5, 10, 20)
    assert tuning.run_all.user_ids_count(doc, profile) == 1
    # The exp1 sft block is carried over verbatim and still builds an SFTConfig.
    sft = SFTConfig.from_dict(doc["sft"])
    assert sft.evaluate_generation is True
    assert doc["val_num_beams"] == regime.val_num_beams
    assert doc["tuning_regime"]["monitor"] == tuning.MONITOR
    # Scientific notation stays a float on re-read.
    assert isinstance(doc["tuning_regime"]["eps"], float)


# ---------------------------------------------------------------------------
# Study resume helpers
# ---------------------------------------------------------------------------

def _journal_study(tmp_path: Path) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(str(tmp_path / "journal.log")))
    return optuna.create_study(study_name="s", direction="maximize", storage=storage, load_if_exists=True)


def test_baseline_is_enqueued_once_and_retried_after_a_failed_attempt(regime, tmp_path: Path) -> None:
    study = _journal_study(tmp_path)
    baseline = regime.baselines["beauty"]
    spec = tuning.TUNING_SPECS["beauty"]

    assert tuning.ensure_incumbent_enqueued(study, baseline) is True
    assert tuning.ensure_incumbent_enqueued(study, baseline) is False  # WAITING copy
    trial = study.ask(fixed_distributions=_distributions(spec))  # -> RUNNING with baseline params
    assert trial.params == baseline
    assert tuning.ensure_incumbent_enqueued(study, baseline) is False  # RUNNING copy

    # A worker killed by the time limit leaves the trial RUNNING; the cleanup
    # marks it FAIL and the baseline is enqueued again (Optuna's own
    # skip_if_exists would have refused because the FAIL trial "exists").
    assert tuning.fail_stale_running_trials(study) == [trial.number]
    assert study.trials[0].state == TrialState.FAIL
    assert tuning.count_finished(study) == 0
    assert tuning.ensure_incumbent_enqueued(study, baseline) is True

    # Once a finished trial holds the baseline, nothing is enqueued anymore.
    trial2 = study.ask(fixed_distributions=_distributions(spec))
    study.tell(trial2, 0.5)
    assert tuning.count_finished(study) == 1
    assert tuning.ensure_incumbent_enqueued(study, baseline) is False
    assert tuning.fail_stale_running_trials(study) == []
    # A journal re-open (new worker process) sees the same state.
    reopened = _journal_study(tmp_path)
    assert tuning.ensure_incumbent_enqueued(reopened, baseline) is False


def test_trial_has_params_compares_values_not_types(regime, tmp_path: Path) -> None:
    study = _journal_study(tmp_path)
    baseline = regime.baselines["beauty"]
    study.enqueue_trial({**baseline, "lr": float(baseline["lr"])})
    waiting = study.trials[0]
    assert tuning.trial_has_params(waiting, baseline)
    assert not tuning.trial_has_params(waiting, {**baseline, "num_layers": baseline["num_layers"] + 1})
    assert not tuning.trial_has_params(waiting, {k: v for k, v in baseline.items() if k != "lr"})
