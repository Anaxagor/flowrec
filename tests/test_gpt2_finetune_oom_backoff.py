"""Memory fallback must change real loader batches and preserve optimizer batches."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "scripts/exp4-reward-ablations"):
    sys.path.insert(0, str(path))
import gflow_finetune_gpt2_amazon as g
import gflow_finetune_gpt2_yambda as y
from flowrec.data.yambda import YambdaDataConfig


@pytest.mark.parametrize("sft", [False, True])
def test_trial36_backoff_preserves_effective_batch(sft):
    args = y.parse_args(["--device", "cpu"])
    initial = args
    k = None if sft else 3
    for batch in (128, 64, 32, 16, 8, 4, 2, 1):
        actual, accum = g.run_batch_settings(args, k)
        assert actual == batch
        assert actual * accum == 2048
        assert args.max_candidate_batch == batch * 4
        assert args.val_batch_size == batch * 2
        assert args.test_batch_size == max(1, batch // 2)
        args = g.oom_retry_args(args, k)
    assert initial.max_candidate_batch == 512


def test_unrestricted_sft_also_shrinks_and_preserves_batch():
    args = g.parse_args(["--batch-size", "128", "--grad-accum", "16"])
    assert not args.uniform_micro_batch
    retry = g.oom_retry_args(args, None)
    assert g.run_batch_settings(retry, None) == (64, 32)


def test_validation_loader_uses_reduced_batch(tmp_path, monkeypatch):
    args = y.parse_args(["--device", "cpu", "--val-batch-size", "16"])
    observed = []
    dm = SimpleNamespace(cfg=YambdaDataConfig(eval_batch_size=256))
    dm.val_dataloader = lambda: observed.append(dm.cfg.eval_batch_size) or "validation"
    fit = SimpleNamespace(state=SimpleNamespace(history=[]), best_checkpoint_path=None,
                          last_checkpoint_path=None)
    trainer = SimpleNamespace(fit=lambda train, val: fit)
    monkeypatch.setattr(g.gpt2_beauty, "evaluate_splits", lambda *a, **kw: {})
    g.fit_and_evaluate(trainer, object(), dm, [], run_batch=8, run_dir=tmp_path,
                       args=args, device=torch.device("cpu"), train_loader="train")
    assert observed == [16]


def test_sft_retries_oom_and_discards_failed_checkpoint(tmp_path, monkeypatch):
    args = y.parse_args(["--device", "cpu"])
    ctx = g.DatasetContext("yambda_50m_listens", {}, None, [], None, {})
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "stale").write_text("failed")
    seen = []

    def train(*a, **kw):
        seen.append(g.run_batch_settings(kw["args"], None))
        if len(seen) < 3:
            raise torch.cuda.OutOfMemoryError("out of memory")
        assert not (run_dir / "checkpoints").exists()
        return {"status": "completed", "oom_retries": kw["attempt"]}

    monkeypatch.setattr(g, "run_sft_control", train)
    result = g.execute_sft_control(ctx.dataset, ctx, run_dir, args=args, csv_path=tmp_path / "m.csv")
    assert seen == [(128, 16), (64, 32), (32, 64)]
    assert result["oom_retries"] == 2


def test_sft_does_not_retry_other_failures(tmp_path, monkeypatch):
    args = y.parse_args(["--device", "cpu"])
    ctx = g.DatasetContext("yambda_50m_listens", {}, None, [], None, {})
    seen = []

    def fail(*a, **kw):
        seen.append(1)
        raise ValueError("wrong checkpoint")

    monkeypatch.setattr(g, "run_sft_control", fail)
    with pytest.raises(ValueError, match="wrong checkpoint"):
        g.execute_sft_control(ctx.dataset, ctx, tmp_path, args=args, csv_path=tmp_path / "m.csv")
    assert seen == [1]


def test_resume_restores_saved_eval_and_sft_batches(tmp_path, monkeypatch):
    args = g.parse_args(["--batch-size", "128", "--grad-accum", "16", "--resume"])
    (tmp_path / "resolved_config.json").write_text(json.dumps({"batching": {
        "batch_size": 32, "gradient_accumulation": 64, "oom_retries": 2,
        "max_candidate_batch": args.max_candidate_batch,
        "val_batch_size": 16, "test_batch_size": 8}}))
    monkeypatch.setattr(g, "resumable_checkpoint", lambda *a: tmp_path / "last.pt")
    attempt, resumed = g.interrupted_attempt_args(
        tmp_path, g.sft_control_identity("yambda_50m_listens", 42), 10, args)
    assert attempt == 2
    assert g.run_batch_settings(resumed, None) == (32, 64)
    assert resumed.val_batch_size == 16 and resumed.test_batch_size == 8
