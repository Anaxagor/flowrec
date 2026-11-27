"""Global CE configurations must use the same audited source and targets as GFlow."""
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


GLOBAL = Path(__file__).resolve().parents[1] / "scripts/global-time"
sys.path.insert(0, str(GLOBAL))
import ce_finetune_yambda as ce
import gflow_finetune_yambda as gflow


@pytest.mark.parametrize("model,dataset", product(("tiger", "gpt2"), (gflow.Y50, gflow.Y500)))
def test_ce_array_reuses_gflow_targets_and_selects_negative_only_ce(model, dataset, tmp_path, monkeypatch):
    import yambda_gflow_inputs as inputs
    import yambda_ce_training as training

    trial = tmp_path / "teacher_trial.json"
    trial.write_text("{}")
    checkpoint = tmp_path / "global_init.pt"
    profile = SimpleNamespace(
        checkpoint=checkpoint, teacher_trial=trial, teacher_checkpoint=tmp_path / "teacher.pt",
        codebook=tmp_path / "codebook.pt", model_config={}, source_identity={"student": str(checkpoint)},
        run_config=lambda args: {"data": {}, "dataset": {"train_last_n": 128}, "model": {}},
    )
    dm = object()
    monkeypatch.setattr(inputs, "load_profile", lambda *args: profile)
    monkeypatch.setattr(gflow, "load_prepared", lambda session: dm)
    checked_factors = []
    monkeypatch.setattr(gflow, "check_factor", lambda session, dm, name: checked_factors.append(name))

    pairs = list(product((.25, 1., 16.), (1., .1)))
    assert set(pairs) == {(row["power"], row["rho"]) for row in gflow.cells(model, dataset)["gflow"]}
    output = tmp_path / "results"
    suffix = Path(model) / dataset
    if dataset == gflow.Y500:
        suffix /= "sampled50"
    seen = []

    def check_training(args, entry, matched, base, *, profile):
        seen.append((entry["power"], entry["rho"]))
        session = profile.session
        assert profile.model_name == model and profile.dataset == entry["dataset"] == dataset
        assert entry["seed"] == 42 and training.ce.shared.combo_key(entry) == "sasrec_x_interaction_x_similarity"
        assert args.sampling == "reward_negatives" and args.estimator == "sampled" and args.num_samples == 3
        sampling = training.ce.shared.sampling_metadata(args)
        assert sampling["replacement"] and sampling["importance_correction"] is False
        assert sampling["calibration_sampling"] == "reward_negatives"
        assert profile.load_data(args, base, session.args, session.config, session.config_path) is dm
        assert profile.finetune_indices(object(), session.args) is None
        assert set(profile.cache_dirs.values()) == set(session.cache_dirs().values())
        assert session.checkpoint == checkpoint
        assert session.output_root == output / suffix
        assert session.cache_root == tmp_path / gflow.EXPERIMENT / suffix
        assert session.source_identity["split"] == "global_time"
        assert session.source_identity["last_n"] == 128
        if dataset == gflow.Y500:
            subset = session.source_identity["finetune_subset"]
            assert subset["per_user_random_n"] == 50 and subset["per_user_random_seed"] == 42
        else:
            assert not session.args.train_random_n
        directory = training.ce.run_dir_for(args, entry, dataset_root=session.output_root)
        (directory / "result.json").write_text('{"status": "completed"}')

    monkeypatch.setattr(training.ce, "run_entry", check_training)
    args = ["--model", model, "--dataset", dataset, "--output-root", str(output),
            "--workspace", str(tmp_path), "--device", "cpu"]
    for index in range(len(pairs)):
        ce.main(["ce", "--index", str(index), *args])
    assert seen == pairs
    expected_factors = ["sasrec", "interaction", "sim_hier" if model == "gpt2" else "similarity"]
    assert checked_factors == expected_factors * len(pairs)
    with pytest.raises(SystemExit):
        ce.main(["ce", "--index", "6", *args])
