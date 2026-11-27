"""Exercise a real AdamW/SFT continuation and epoch-boundary recovery on a tiny Yambda stream."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/global-time"))
import continue_tiger_yambda500m as cont
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.training import Trainer, TrainerConfig


@pytest.fixture
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_selected_weights_and_scheduled_validation_with_resume(tmp_path, monkeypatch, single_threaded_torch):
    config = json.loads((cont.INITIAL / "resolved_config.json").read_text())
    config["model"].update(embedding_dim=8, num_heads=2, d_kv=4, num_encoder_layers=1,
                           num_decoder_layers=1, dim_feedforward=16)
    interactions, semantic = tmp_path / "events.parquet", tmp_path / "ids.json"
    pd.DataFrame([dict(uid=u, item_id=(i+u) % 24, timestamp=1000+i, event_type="listen", played_ratio_pct=100)
                  for u in range(4) for i in range(32)]).to_parquet(interactions)
    semantic.write_text(json.dumps({str(i): [i // 8, i % 8] for i in range(24)}))
    cfg = YambdaDataConfig(interactions_path=interactions, semantic_id_path=semantic, dataset_size="500m",
                           hf_recent_months=None, split_strategy="global_time", global_val_timestamp=1025,
                           global_test_timestamp=1030, train_target_policy="last_n", train_last_n=128,
                           min_item_interactions=1, min_user_interactions=1, max_history_len=3,
                           batch_size=128, eval_batch_size=4, num_workers=0, persistent_workers=False)
    dm = YambdaDataModule(cfg)
    dm.setup()
    dm._ce_prepared_manifest = tmp_path / "manifest.json"
    dm._ce_prepared_manifest.write_text('{"artifacts":{"tensors":{"sha256":"test-input"}}}')
    monkeypatch.setattr(dm, "test_dataloader", lambda: pytest.fail("Unexpected test loader"))
    source = tmp_path / "trial/best.pt"
    out = tmp_path / "continuation"
    # Standard HPO checkpoint with trained moments and scaler metadata, without an epoch_progress extension.
    initial_cfg = replace(TrainerConfig.from_dict(config["trainer"]), max_epochs=7, checkpoint_dir=source.parent,
                          validate_every_epochs=None, save_best=False, save_last=False)
    initial = Trainer.from_optimizer_config(model=cont.tiger.build_model(config["model"], dm),
                    method=cont.SFTMethod.from_dict(config["method"]), cfg=initial_cfg, device="cpu",
                    **config["optimizer"])
    initial.fit(dm.train_dataloader())
    initial.state.best_epoch, initial.state.best_metric = 6, 1.0
    initial.save_checkpoint("best.pt")
    source_state = torch.load(source, weights_only=True)
    assert "epoch_progress" not in source_state

    calls = []
    scores = [.1, .5, .5, .25, .4, .3]
    evaluate = cont.ContinuationTrainer.evaluate
    def validate(self, loader, *, stage):
        calls.append(self.state.epoch + 1)
        metrics = evaluate(self, loader, stage=stage)
        assert metrics[cont.MONITOR] >= 0 and stage == "val"
        metrics[cont.MONITOR] = scores[len(calls)-1]
        return metrics
    monkeypatch.setattr(cont.ContinuationTrainer, "evaluate", validate)
    original_train = cont.ContinuationTrainer.train_one_epoch
    def train_checked(self, train_loader, **kwargs):
        if kwargs["epoch"] == 7:
            moments = self.optimizer.state_dict()["state"]
            for key, values in source_state["optimizer_state_dict"]["state"].items():
                assert torch.equal(values["exp_avg"], moments[key]["exp_avg"])
                assert torch.equal(values["exp_avg_sq"], moments[key]["exp_avg_sq"])
                assert torch.equal(values["step"], moments[key]["step"])
            assert self.optimizer.param_groups[0]["lr"] == cont.LR
        if kwargs["epoch"] == 8:
            raise InterruptedError("before first validation")
        return original_train(self, train_loader, **kwargs)
    monkeypatch.setattr(cont.ContinuationTrainer, "train_one_epoch", train_checked)
    def run():
        return cont.train(dm, source, "fixture", config, out, device="cpu")
    with pytest.raises(InterruptedError, match="first validation"):
        run()
    assert calls == [] and not (out / "best.pt").exists()
    monkeypatch.setattr(cont.ContinuationTrainer, "train_one_epoch", original_train)
    publish = cont.publish_epoch
    def interrupted_best(directory):
        raise InterruptedError("after last.pt, before publishing best")
    monkeypatch.setattr(cont, "publish_epoch", interrupted_best)
    with pytest.raises(InterruptedError, match="before publishing"):
        run()
    assert calls == [17] and not (out / "best.pt").exists()
    monkeypatch.setattr(cont, "publish_epoch", publish)
    result = run()
    assert result["total_epochs_completed"] == 67 and result["additional_epochs_completed"] == 60
    assert calls == [17, 27, 37, 47, 57, 67]
    assert [m["ndcg_at_10"] for m in result["validation"]] == scores
    assert result["selected_epoch_1based"] == 27
    assert torch.load(out / "best.pt", weights_only=True)["trainer_state"]["epoch"] == 26
    assert torch.load(out / "last.pt", weights_only=True)["trainer_state"]["epoch"] == 66
    assert run() == result


def test_source_lookup_accepts_the_original_when_canonical_weights_differ(tmp_path, monkeypatch):
    import hashlib
    wrong, right = (tmp_path / "canonical/best.pt", tmp_path / "trial/best.pt")
    for path in (wrong, right):
        path.parent.mkdir()
        path.write_bytes(path.parent.name.encode())
    right.with_name("resolved_config.json").write_bytes((cont.INITIAL / "resolved_config.json").read_bytes())
    digest = hashlib.sha256(right.read_bytes()).hexdigest()
    monkeypatch.setattr(cont, "SOURCE_DIGESTS", {digest})
    monkeypatch.setattr(cont, "SOURCES", (wrong, right))
    assert cont.source_files()[:2] == (right, digest)
    with pytest.raises(ValueError, match="No matching"):
        cont.source_files(wrong)
