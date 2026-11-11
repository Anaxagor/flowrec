"""A sampled50 continuation must preserve and validate the original run."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_yambda as yambda  # noqa: E402


@pytest.fixture
def interrupted(tmp_path):
    args = yambda.parse_args([
        "--datasets", "yambda_500m_listens", "--yambda500-base", "trial4",
        "--yambda500-targets", "sampled50", "--index", "0", "--resume", "--device", "cpu",
    ])
    args = yambda.base.dataset_args(args, "yambda_500m_listens")
    identity = yambda.base.entry_identity(yambda.base.build_grid(args)[0], args.seed)
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    best_metric = 0.10931828339696122
    torch.save({"trainer_state": {"epoch": 13, "global_step": 227654,
                                   "best_epoch": 3, "best_metric": best_metric}}, ckpt_dir / "last.pt")
    torch.save({"trainer_state": {"epoch": 3, "global_step": 65044,
                                   "best_epoch": 3, "best_metric": best_metric}}, ckpt_dir / "best.pt")
    resolved = {
        **identity,
        # The real run_single overwrites identity['sampler'] with this object.
        "sampler": {"name": "teacher", "mode": "curriculum", "curriculum_steps": 162610},
        "data": {"num_finetune_examples": 4162653,
                 "finetune_per_user_pool_last_n": 128, "finetune_per_user_last_n": None,
                 "finetune_per_user_random_n": 50, "finetune_subset_seed": 42},
        "cli_args": {"prefix_from_dense": False, "yambda50_all_targets": False,
                     "yambda50_base": "legacy", "yambda500_base": "trial4",
                     "yambda500_targets": "sampled50"},
        "calibration": {"fixed_gflow_weight": 0.000372545},
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved))
    (run_dir / "calibration.json").write_text(json.dumps({"fixed_gflow_weight": 0.000372545}))
    return run_dir, identity, args


def test_real_sampler_object_resumes_fourteen_epoch_checkpoint(interrupted, capsys):
    run_dir, identity, args = interrupted
    checkpoint = yambda.base.resumable_checkpoint(run_dir, identity, 4162653, args)
    assert checkpoint == run_dir / "checkpoints/last.pt"
    yambda.base.require_safe_yambda500_resume(run_dir, identity, args)
    assert "14 epochs recorded" in capsys.readouterr().out
    state = torch.load(checkpoint, map_location="cpu")["trainer_state"]
    assert state["epoch"] + 1 == 14 and state["global_step"] == 227654


def test_replaced_best_checkpoint_stops_before_any_file_is_changed(interrupted):
    run_dir, identity, args = interrupted
    best = run_dir / "checkpoints/best.pt"
    torch.save({"trainer_state": {"epoch": 0, "global_step": 0, "best_epoch": -1,
                                  "best_metric": 0.10637251452748643}}, best)
    paths = [run_dir / "checkpoints/last.pt", best, run_dir / "resolved_config.json",
             run_dir / "calibration.json"]
    before = [p.read_bytes() for p in paths]
    with pytest.raises(RuntimeError, match="best.pt does not match"):
        yambda.base.require_safe_yambda500_resume(run_dir, identity, args)
    assert [p.read_bytes() for p in paths] == before


def test_index_worker_rejects_replaced_best_before_loading_data(interrupted, monkeypatch, tmp_path):
    run_dir, _, _ = interrupted
    best = run_dir / "checkpoints/best.pt"
    torch.save({"trainer_state": {"global_step": 0, "best_epoch": -1,
                                  "best_metric": 0.10637251452748643}}, best)
    before = {p: p.read_bytes() for p in (run_dir / "checkpoints/last.pt", best,
                                           run_dir / "resolved_config.json", run_dir / "calibration.json")}
    monkeypatch.setattr(yambda.base, "run_dir_for", lambda *a, **k: run_dir)
    monkeypatch.setattr(yambda.base, "build_dataset_context", lambda *a, **k: pytest.fail("loaded data"))
    with pytest.raises(RuntimeError, match="best.pt does not match"):
        yambda.main(["--datasets", "yambda_500m_listens", "--yambda500-base", "trial4",
                     "--yambda500-targets", "sampled50", "--index", "0", "--resume",
                     "--device", "cpu", "--output-root", str(tmp_path / "output")])
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("damage,diagnostic", [
    ("missing_calibration", "calibration.json is missing"),
    ("changed_lambda", "disagree on lambda"),
    ("wrong_sampler", "metadata does not match"),
    ("corrupt_last", "last.pt is unreadable"),
    ("missing_best", "best.pt is missing"),
])
def test_incompatible_artifacts_do_not_silently_restart(interrupted, damage, diagnostic):
    run_dir, identity, args = interrupted
    if damage == "missing_calibration":
        (run_dir / "calibration.json").unlink()
    elif damage == "changed_lambda":
        (run_dir / "calibration.json").write_text(json.dumps({"fixed_gflow_weight": 0.2}))
    elif damage == "wrong_sampler":
        p = run_dir / "resolved_config.json"
        resolved = json.loads(p.read_text()); resolved["sampler"]["name"] = "random"
        p.write_text(json.dumps(resolved))
    elif damage == "corrupt_last":
        (run_dir / "checkpoints/last.pt").write_bytes(b"truncated")
    else:
        (run_dir / "checkpoints/best.pt").unlink()
    with pytest.raises(RuntimeError, match=diagnostic):
        yambda.base.require_safe_yambda500_resume(run_dir, identity, args)


def test_missing_last_after_completed_epoch_is_not_treated_as_fresh(interrupted):
    run_dir, identity, args = interrupted
    (run_dir / "checkpoints/last.pt").unlink()
    (run_dir / "performance.jsonl").write_text('{"type":"epoch_end","epoch":13}\n')
    with pytest.raises(RuntimeError, match="completed epoch was recorded"):
        yambda.base.require_safe_yambda500_resume(run_dir, identity, args)


def test_first_attempt_without_completed_epoch_can_start(interrupted):
    run_dir, identity, args = interrupted
    (run_dir / "checkpoints/last.pt").unlink()
    torch.save({"trainer_state": {"global_step": 0, "best_epoch": -1}}, run_dir / "checkpoints/best.pt")
    (run_dir / "performance.jsonl").write_text('{"type":"train_step","epoch":0}\n')
    yambda.base.require_safe_yambda500_resume(run_dir, identity, args)


def test_missing_last_with_a_trained_best_is_not_treated_as_fresh(interrupted):
    run_dir, identity, args = interrupted
    (run_dir / "checkpoints/last.pt").unlink()
    with pytest.raises(RuntimeError, match="best.pt records completed training"):
        yambda.base.require_safe_yambda500_resume(run_dir, identity, args)


def test_real_data_subset_mismatch_stops_instead_of_restarting(interrupted):
    run_dir, identity, args = interrupted
    with pytest.raises(RuntimeError, match="checkpoint does not match the prepared data"):
        yambda.base.interrupted_attempt_args(run_dir, identity, 4162652, args)


def test_resume_restores_real_trainer_optimizer_and_best_state(interrupted):
    from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
    from flowrec.models.base import FlowRecModel, RecModelOutput
    from flowrec.training import Trainer, TrainerConfig

    class Model(FlowRecModel):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            return RecModelOutput()

    class Method(FlowRecMethod):
        def __init__(self, score):
            super().__init__()
            self.score = score

        def compute_loss(self, model, batch, *, stage="train", step=None):
            if stage == "train":
                return MethodOutput(loss=(model.weight - 1).square().sum())
            return MethodOutput(loss=torch.zeros(()), metrics={"score": self.score})

    run_dir, identity, args = interrupted
    cfg = TrainerConfig(max_epochs=2, checkpoint_dir=run_dir / "checkpoints", progress_bar=False,
                        log_every_steps=0, monitor="val/score", monitor_mode="max")

    def make(score, logger=None):
        return Trainer.from_optimizer_config(model=Model(), method=Method(score), cfg=cfg,
                                             device="cpu", lr=0.01, logger=logger)

    def interrupt(event):
        if event.get("type") == "epoch_end":
            raise KeyboardInterrupt

    first = make(0.5, logger=interrupt)
    batch = [{"x": torch.zeros(2, 1)}]
    with pytest.raises(KeyboardInterrupt):
        first.fit(batch, batch)

    # Same metadata and sampler layout as the real sampled50 GFlow run.
    yambda.base.require_safe_yambda500_resume(run_dir, identity, args)
    last = yambda.base.resumable_checkpoint(run_dir, identity, 4162653, args)
    resumed = make(0.4)
    assert yambda.base.resume_trainer(resumed, last, tag="test") == 0
    assert resumed.state.epoch == 1 and resumed.state.optimizer_step == 1
    torch.testing.assert_close(resumed.model.weight, first.model.weight)
    assert all(int(state["step"]) == 1 for state in resumed.optimizer.state.values())
    result = resumed.fit(batch, batch)
    assert result.state.epoch == 1 and result.state.optimizer_step == 2
    assert result.state.best_epoch == 0 and result.state.best_metric == 0.5
