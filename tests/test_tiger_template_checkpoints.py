"""Legacy templates must request current weights before expensive preprocessing."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_tiger as runner


@pytest.mark.parametrize("name,field", [
    ("tiger_beauty_grpo.json", "model.checkpoint_path"),
    ("tiger_beauty_gflow_mix_curriculum_example.json", "method.sasrec.checkpoint_path"),
])
def test_unconfigured_template_stops_before_creating_a_run(monkeypatch, name, field):
    config = runner.PROJECT_ROOT / "configs" / name
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "create_run_dir", lambda *a, **k: pytest.fail("created a run before validating inputs"))
    monkeypatch.setattr(runner, "build_data_module", lambda *a, **k: pytest.fail("loaded data before validating inputs"))
    with pytest.raises(ValueError, match=field):
        runner.main()


@pytest.mark.parametrize("relative", [
    "checkpoints/exp1/tiger/beauty/best.pt",
    "checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty/best.pt",
    "checkpoints/exp2/global_split/initial/gpt2/yambda_50m_listens/full_history/best.pt",
])
def test_template_accepts_only_existing_current_checkpoint(tmp_path, relative):
    path = tmp_path / relative
    config = {"required_checkpoints": ["model.checkpoint_path"], "model": {"checkpoint_path": relative}}
    config_path = tmp_path / "configs/template.json"
    with pytest.raises(FileNotFoundError, match="selected checkpoint is missing"):
        runner.validate_required_checkpoint_inputs(config, config_path=config_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test weights")
    runner.validate_required_checkpoint_inputs(config, config_path=config_path)


@pytest.mark.parametrize("relative", [
    "checkpoints/exp1-standard params/beauty/best.pt",
    "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt",
    "artifacts/checkpoint_archive/2026-10-30/exp2-tuned_params/initial/tiger/beauty/best.pt",
    "experiments/old-run/checkpoints/best.pt",
])
def test_template_rejects_existing_weights_outside_current_structure(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old weights")
    config = {"required_checkpoints": ["model.checkpoint_path"], "model": {"checkpoint_path": relative}}
    with pytest.raises(ValueError, match="current checkpoints structure"):
        runner.validate_required_checkpoint_inputs(config, config_path=tmp_path / "configs/template.json")
