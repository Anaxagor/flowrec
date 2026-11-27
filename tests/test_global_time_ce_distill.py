"""Regression coverage for the global CE array-to-training contract."""
import argparse
import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("global_ce_experiment", ROOT / "scripts/global-time/ce_distill.py")
ce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ce)


def test_login_python_can_find_missing_cells(tmp_path, monkeypatch, capsys):
    monkeypatch.delattr(argparse, "BooleanOptionalAction", raising=False)
    result = tmp_path / "tiger/beauty/reward_negatives_n10/pow_1/rho_0p1/seed_42/result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"status": "completed"}')
    ce.main(["missing", "--model", "tiger", "--dataset", "beauty", "--output-root", str(tmp_path)])
    assert capsys.readouterr().out.strip() == "0,1,2,4,5"
    assert ce.shared.parse_args([]).resume
    assert not ce.shared.parse_args(["--no-resume"]).resume
    assert ce.shared.parse_args(["--no-resume", "--resume"]).resume


@pytest.mark.parametrize("model,dataset", product(("gpt2", "tiger"), ("beauty", "instruments")))
def test_array_selects_all_six_cells_and_their_training_distribution(model, dataset, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/global-time"))
    import ce_distill_inputs
    from contextlib import nullcontext

    seen = []
    monkeypatch.setattr(ce_distill_inputs, "configure_inputs", lambda *args: None)
    base = SimpleNamespace(control=SimpleNamespace(file_lock=lambda path: nullcontext()))
    monkeypatch.setattr(ce.ce, "load_runtime", lambda: (None, base))

    def run(args, entry, matched, base, profile):
        seen.append((entry["power"], entry["rho"]))
        assert entry["dataset"] == dataset and entry["seed"] == 42
        assert ce.shared.combo_key(entry) == "sasrec_x_interaction_x_similarity"
        assert args.protocol_path == ce.PROTOCOL
        assert args.sampling == "reward_negatives" and args.num_samples == 10
        assert args.estimator == "sampled" and args.output_root == tmp_path / "out" / model
        assert profile.model_name == model
        assert profile.scratch == tmp_path / ce.EXPERIMENT / model / dataset
        assert ce.shared.curriculum_steps_for(args, 128, 32) == 0
        fake_runner = SimpleNamespace(item_id_to_semantic_row=lambda dm: [1, 0])
        opts = ce.shared.distillation_sampling_args(args, None, fake_runner, 0)
        assert opts["item_id_to_semantic_row"] == [1, 0]
        meta = ce.shared.sampling_metadata(args)
        assert meta["replacement"] and not meta["importance_correction"]
        assert meta["calibration_sampling"] == "reward_negatives"
        assert "negative" in meta["ce_target"]

    monkeypatch.setattr(ce.ce, "run_entry", run)
    for index in range(6):
        ce.main(["train", "--index", str(index), "--model", model, "--dataset", dataset,
                 "--workspace", str(tmp_path), "--output-root", str(tmp_path / "out")])
    assert seen == list(product((.25, 1., 16.), (1., .1)))
