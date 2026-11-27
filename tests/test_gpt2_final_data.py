"""Full matched final inputs and the last complete optimizer epoch survive a retry."""
import hashlib
import json

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import FlowRecModel
from flowrec.training.trainer import TrainerConfig
from scripts import gpt2_final_data as final


def test_compact_preserves_every_shuffled_dense_and_heldout_batch(tmp_path, monkeypatch):
    events = tmp_path / "events.parquet"
    pd.DataFrame([dict(uid=u, item_id=(i + u) % 25, timestamp=i, played_ratio_pct=100)
                  for u, length in enumerate([8, 150, 170]) for i in range(length)]).to_parquet(events)
    mapping = tmp_path / "sids.json"
    mapping.write_text(json.dumps({str(i): [i // 5, i % 5] for i in range(25)}))
    cfg = YambdaDataConfig(interactions_path=events, semantic_id_path=mapping, hf_recent_months=None,
        split_strategy="global_time", global_val_timestamp=145, global_test_timestamp=148,
        global_eval_target_mode="last_per_user", global_history_policy="rolling",
        min_user_interactions=3, min_item_interactions=0, train_target_policy="last_n", train_last_n=128,
        max_history_len=50, train_dense_match_prefix=True, train_example_layout="dense_causal",
        train_window_len=50, train_window_stride=1, include_raw_in_batch=False,
        batch_size=23, eval_batch_size=2, num_workers=0, persistent_workers=False)
    reference = YambdaDataModule(cfg)
    reference.setup()
    counts = {s: len(getattr(reference, s + "_dataset")) for s in ("train", "val", "test")}
    assert counts == {"train": 263, "val": 2, "test": 2}
    directory = tmp_path / "compact"
    final.prepare(directory, cfg, expected_counts=counts)
    monkeypatch.setattr(YambdaDataModule, "setup", lambda *_: pytest.fail("GPU reopened raw data"))
    final.prepare(directory, cfg, expected_counts=counts)
    dm = final.load(directory, cfg, expected_counts=counts)
    assert dm.item2id == reference.item2id
    assert dm.train_item_counts_by_raw_id == reference.train_item_counts_by_raw_id
    for split in counts:
        left = getattr(reference, split + "_dataloader")()
        right = getattr(dm, split + "_dataloader")()
        for original, mapped in zip(left, right, strict=True):
            assert original.keys() == mapped.keys()
            for key in original:
                torch.testing.assert_close(mapped[key], original[key])
    with pytest.raises(ValueError, match="Full matched dataset"):
        final.load(directory, cfg, expected_counts={**counts, "train": 50})


def test_duplicate_worker_fails_before_training(tmp_path):
    with final.training_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with final.training_lock(tmp_path):
                pytest.fail("second GPU worker started")


def test_winner_fingerprint(tmp_path, monkeypatch):
    export = tmp_path / "gpt2_model_yambda_500m_listens.yaml"
    export.write_text("# best trial: 17\nmodel: gpt2\n")
    record = tmp_path / "scripts/global-time/gpt2_yambda500m_last128_checkpoint.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({
        "sources": {export.name: hashlib.sha256(export.read_bytes()).hexdigest()},
        "data_summary": {"num_train_examples": 100, "num_val_examples": 20, "num_test_examples": 15},
    }))
    monkeypatch.setattr(final, "ROOT", tmp_path)
    assert final.winner_counts(export) == {"train": 100, "val": 20, "test": 15}
    export.write_text("# best trial: 17\n")
    with pytest.raises(ValueError, match="original winning trial 17"):
        final.winner_counts(export)


class SmallModel(FlowRecModel):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        raise NotImplementedError


class SmallMethod(FlowRecMethod):
    def compute_loss(self, model, batch, *, stage="train", step=None):
        return MethodOutput(loss=(model.weight - torch.as_tensor(batch["x"]) * torch.rand(())).square().mean(),
                            metrics={"score": 1.0})


def _collate(rows):
    return {"x": rows}


@pytest.mark.parametrize("workers", [0, 1])
def test_interrupted_final_has_same_weights_optimizer_steps_and_earlier_best(tmp_path, workers):
    def run(directory, *, interrupt=False, identity="same"):
        directory.mkdir(exist_ok=True)
        cfg = TrainerConfig(max_epochs=3, checkpoint_dir=str(directory), monitor="val/score",
                            monitor_mode="max", progress_bar=False, log_every_steps=0)
        torch.manual_seed(42)
        trainer = final.FinalTrainer.from_optimizer_config(model=SmallModel(), method=SmallMethod(),
            cfg=cfg, device="cpu", lr=0.01, weight_decay=0.1)
        def logger(event):
            if interrupt and event.get("type") == "epoch_end":
                raise KeyboardInterrupt
        trainer.logger = logger
        loader = DataLoader(list(range(7)), batch_size=2, collate_fn=_collate,
            shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=workers, persistent_workers=bool(workers))
        resolved = {"seed": 42, "model": trainer.model.config_dict(), "method": {}, "trainer": cfg.to_dict(),
                    "batch_plan": {"train_batch_size": 2}, "ranking_ks": [5, 10, 20],
                    "optimizer": {"lr": 0.01, "weight_decay": 0.1}}
        return trainer, final.final_fit(trainer, loader, [{"x": torch.tensor(0.5)}], resolved, {"source": identity})

    expected, _ = run(tmp_path / "reference")
    interrupted = tmp_path / "retry"
    with pytest.raises(KeyboardInterrupt):
        run(interrupted, interrupt=True)
    with pytest.raises(ValueError, match="identity changed"):
        run(interrupted, identity="different")
    resumed, fit = run(interrupted)
    assert fit.best_checkpoint_path == str(interrupted / "best.pt")
    assert fit.last_checkpoint_path == str(interrupted / "last.pt")
    assert fit.state.optimizer_step == expected.state.optimizer_step == 12
    assert fit.state.best_epoch == expected.state.best_epoch == 0
    torch.testing.assert_close(resumed.model.weight, expected.model.weight, rtol=0, atol=0)
    # A retry during full validation/test must not train another epoch.
    done, fit_again = run(interrupted)
    assert fit_again.state.optimizer_step == 12
    torch.testing.assert_close(done.model.weight, expected.model.weight, rtol=0, atol=0)
    # An orphaned later best can leave the JSON beside recovered weights stale.
    sidecar = interrupted / "best.pt.json"
    sidecar.write_text('{"trainer_state": {"best_epoch": 100}}')
    done.load_checkpoint(fit_again.best_checkpoint_path, load_optimizer=False, load_scheduler=False)
    assert json.loads(sidecar.read_text())["trainer_state"]["best_epoch"] == 0
