"""Source-stage dispatch and atomic mid-epoch resume across SFT/GFlow."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("from_sft_stage_tests", ROOT / "scripts/exp8-from-sft/stage_training.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)
HELPERS = importlib.util.spec_from_file_location("source_resume_test_helpers", ROOT / "tests/test_ce_distill_resume.py")
resume = importlib.util.module_from_spec(HELPERS)
HELPERS.loader.exec_module(resume)


def test_native_entry_has_exact_three_negatives_and_unit_product_rewards():
    cell = dict(model="gpt2", dataset="beauty", combo=("sasrec", "interaction", "similarity"),
                objective="subtb", power=16, rho=.1, num_negatives=3)
    entry = stage.gflow_entry(cell)
    assert entry["components"] == ("sasrec", "interaction", "sim_hier")
    assert entry["weights"] == (1., 1., 1.) and entry["composition"] == "multiply"
    assert entry["similarity_mode"] == "hierarchical"
    assert entry["num_negatives"] == 3 and entry["sampler"] == "teacher"
    tiger = stage.gflow_entry({**cell, "model": "tiger"})
    assert tiger["combo"] == cell["combo"] and "components" not in tiger
    with pytest.raises(ValueError, match="three negatives"):
        stage.gflow_entry({**cell, "num_negatives": 10})


def test_bound_trainer_resumes_mid_epoch_without_replaying_updates(tmp_path):
    args = SimpleNamespace(seed=37, checkpoint_every_updates=1)
    BoundTrainer = stage.trainer_class(args)

    class Interrupting(BoundTrainer):
        def save_checkpoint(self, filename):
            result = super().save_checkpoint(filename)
            if self._in_epoch and self.state.optimizer_step == 2:
                raise resume.Interrupted("atomic checkpoint written")
            return result

    torch.manual_seed(9)
    initial = deepcopy(resume.Model().state_dict())
    full = resume.make_trainer(tmp_path / "full", initial, trainer_type=BoundTrainer)
    full.fit(resume.loader())
    interrupted = resume.make_trainer(tmp_path / "resumed", initial, trainer_type=Interrupting)
    with pytest.raises(resume.Interrupted):
        interrupted.fit(resume.loader())
    restored = resume.make_trainer(tmp_path / "resumed", initial, trainer_type=BoundTrainer)
    previous_epoch = stage.resume_gpt2(restored, tmp_path / "resumed/last.pt", tag="test")
    assert previous_epoch == restored.state.epoch == 0
    restored.fit(resume.loader())
    resume.assert_nested_equal(restored.model.state_dict(), full.model.state_dict())
    resume.assert_nested_equal(restored.optimizer.state_dict(), full.optimizer.state_dict())
    assert interrupted.method.seen + restored.method.seen == full.method.seen
    assert restored.state.optimizer_step == full.state.optimizer_step


def test_bound_trainer_advances_completed_epoch_once_and_eval_load_does_not_restore_progress(tmp_path):
    args = SimpleNamespace(seed=37, checkpoint_every_updates=1)
    BoundTrainer = stage.trainer_class(args)
    torch.manual_seed(3)
    initial = deepcopy(resume.Model().state_dict())
    trainer = resume.make_trainer(tmp_path, initial, trainer_type=BoundTrainer)
    trainer.state.epoch = 2
    trainer._epoch_complete = True
    path = trainer.save_checkpoint("last.pt")
    restored = resume.make_trainer(tmp_path, initial, trainer_type=BoundTrainer)
    assert stage.resume_gpt2(restored, Path(path), tag="test") == 2
    assert restored.state.epoch == 3
    restored.load_checkpoint(path, load_optimizer=False, load_scheduler=False)
    assert restored.state.epoch == 2


@pytest.fixture
def session(tmp_path):
    base = SimpleNamespace(Trainer=object(), resume_trainer=object(), DatasetContext=SimpleNamespace)
    return SimpleNamespace(base=base,
        args=SimpleNamespace(seed=42, device="cpu", smoke=False, checkpoint_every_updates=1,
                             tensorized_train_loader=True),
        source={"model": "gpt2", "dataset": "beauty"}, config={"frozen": "source data"},
        run_dir=tmp_path, output_root=tmp_path, checkpoint=tmp_path / "source.pt",
        cache_dirs=lambda: {name: tmp_path / name for name in ("sasrec", "interaction", "sim_hier")})


def test_sft_dispatch_avoids_teacher_rewards_and_preserves_source_args(session, monkeypatch):
    monkeypatch.setattr(stage, "load_teacher", lambda *_args: pytest.fail("SFT must not load SASRec"))
    captured = {}

    def run(dataset, context, run_dir, *, args, csv_path):
        captured.update(dataset=dataset, context=context, args=args)
        return {"status": "completed"}

    session.base.run_sft_control = run
    before = (session.base.Trainer, session.base.resume_trainer)
    dm = SimpleNamespace(train_dataset=list(range(10)))
    result = stage.run_training(session, dict(model="gpt2", dataset="beauty", seed=42, arm="sft"), dm, [3, 7])
    assert result["status"] == "completed"
    assert captured["context"].subset_indices == [3, 7]
    assert captured["context"].run_config is session.config
    assert captured["context"].cache_dirs == {} and captured["context"].sasrec is None
    assert captured["args"] is session.args
    assert (session.base.Trainer, session.base.resume_trainer) == before


def test_gflow_dispatch_preserves_cell_subset_caches_and_propagates_failure(session, monkeypatch):
    import flowrec.ft_methods.gflow.factorized_cache as factor_module
    teacher = object()
    monkeypatch.setattr(stage, "load_teacher", lambda *_args: teacher)
    checked = []
    monkeypatch.setattr(factor_module, "FactorizedRewardCache", lambda path, **_kwargs: path)
    session.base.factor_runner = SimpleNamespace(verify_cache_alignment=lambda dm, indices, caches, **_kwargs:
                                                checked.append((indices, set(caches))))
    before = (session.base.Trainer, session.base.resume_trainer)

    def run(**kwargs):
        assert kwargs["sasrec"] is teacher
        assert kwargs["subset_indices"] == [3, 7]
        assert kwargs["run_config"] is session.config
        assert kwargs["entry"]["components"] == ("sim_hier",)
        assert set(kwargs["cache_dirs"]) == {"sim_hier"}
        raise RuntimeError("calibration failed")

    session.base.run_single = run
    cell = dict(model="gpt2", dataset="beauty", seed=42, arm="gflow", combo=("similarity",),
                objective="db", power=1, rho=1)
    with pytest.raises(RuntimeError, match="calibration failed"):
        stage.run_training(session, cell, SimpleNamespace(), [3, 7])
    assert (session.base.Trainer, session.base.resume_trainer) == before
    assert checked == [([3, 7], {"sim_hier"})]


def test_gflow_checkpoint_without_calibration_fails_instead_of_restarting(session):
    (session.run_dir / "checkpoints").mkdir()
    (session.run_dir / "checkpoints/last.pt").write_bytes(b"saved checkpoint")
    with pytest.raises(RuntimeError, match="original calibration"):
        stage.run_training(session, dict(model="gpt2", dataset="beauty", seed=42, arm="gflow"),
                           SimpleNamespace(), [3, 7])


def test_tiger_loading_checks_catalog_values_not_only_buffer_shape():
    model = SimpleNamespace(item_semantic_ids=torch.tensor([[1, 2], [3, 4]]))
    def mismatched_load(model, _checkpoint):
        model.item_semantic_ids = model.item_semantic_ids.flip(0)
    with pytest.raises(RuntimeError, match="semantic-ID table differs"):
        stage.checked_tiger_load(mismatched_load, model, Path("source.pt"))
