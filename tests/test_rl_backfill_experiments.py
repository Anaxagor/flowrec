import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flowrec.models.gpt2 import GPT2RecModel
from flowrec.models.tiger import TigerModel
from scripts import backfill_grpo, backfill_sdpo, rl_backfill, rl_backfill_inputs, rl_backfill_rewards


def test_missing_cells_and_rewards():
    grpo = backfill_grpo.cells()
    assert len(grpo) == len(set(grpo)) == 28
    assert sum(cell.split == "loo" for cell in grpo) == 9
    assert len(backfill_grpo.cells(all_rewards=True)) == 36
    assert [cell.reward for cell in grpo if cell.split == "global" and cell.dataset == backfill_grpo.Y50
            and cell.model == "gpt2"] == ["S×H×I"]
    assert set(backfill_sdpo.cells()) == {
        rl_backfill.Cell("loo", "gpt2", backfill_grpo.Y50),
        rl_backfill.Cell("loo", "gpt2", backfill_grpo.Y500),
        rl_backfill.Cell("global", "gpt2", backfill_grpo.Y500),
        rl_backfill.Cell("global", "tiger", backfill_grpo.BEAUTY),
    }


@pytest.mark.parametrize(("algorithm", "index", "model_kind", "peak_epoch", "stop_epoch"),
                         [("grpo", 0, "tiger", 0, 15), ("sdpo", 1, "gpt2", 12, 17)])
def test_cli_trains_and_retries_heldout_evaluation(tmp_path, monkeypatch, algorithm, index, model_kind,
                                                   peak_epoch, stop_epoch):
    # Real models, policy sampling, training, checkpoint restore and beam evaluation.
    # Substitute the full-size inputs, keeping distinct loaders for all three splits.
    catalog = {str(i): [1 + i // 8, 5 + (i // 4) % 2, 9 + i % 4] for i in range(24)}
    table = torch.tensor(list(catalog.values()))
    batch = {
        "user_ids": torch.tensor([0, 1]), "history_token_ids": table[torch.tensor([[2, 3], [5, 6]])],
        "history_token_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "target_item_ids": torch.tensor([1, 5]), "target_token_ids": table[[0, 4]],
        "target_token_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    checkpoint = tmp_path / "source.pt"
    calls = []
    val_batches = [batch]
    best_weights = {}

    def heldout():
        calls.append("test")
        if calls.count("test") == 1:
            raise RuntimeError("simulated evaluation interruption")
        return [batch]

    def tiny_inputs(split, model, dataset, **kwargs):
        assert model == model_kind
        common = dict(semantic_id_mapping=catalog, codebook_size=4, sem_id_len=3,
                      embedding_dim=8, num_heads=2, dim_feedforward=16, dropout=0.)
        if model == "gpt2":
            policy = GPT2RecModel(**common, num_layers=1, max_position_embeddings=32)
        else:
            policy = TigerModel(**common, num_encoder_layers=1, num_decoder_layers=1,
                                d_kv=4, use_user_token=False, user_ids_count=0)
        if checkpoint.exists():
            policy.load_state_dict(torch.load(checkpoint, weights_only=True))
        else:
            torch.save(policy.state_dict(), checkpoint)
        data = SimpleNamespace(
            semantic_ids=catalog, item2id={key: int(key) + 1 for key in catalog},
            item_vocab_size=25, num_items=24, cfg=SimpleNamespace(pad_item_id=0),
            train_item_counts_by_raw_id={key: int(key) + 1 for key in catalog},
            train_dataloader=lambda: [batch], val_dataloader=lambda: val_batches, test_dataloader=heldout,
        )
        return rl_backfill_inputs.BackfillInputs(policy, data, checkpoint, {})

    class Reward(torch.nn.Module):
        def forward(self, batch, trajectories):
            calls.append("train")
            if calls.count("train") == 1:
                raise RuntimeError("simulated first training step interruption")
            return trajectories.is_positive.float()

    monkeypatch.setattr(rl_backfill_inputs, "load_inputs", tiny_inputs)
    monkeypatch.setattr(rl_backfill_rewards, "build_backfill_reward", lambda *args, **kwargs: Reward())
    evaluation = import_module(f"scripts.exp1-standard-params.run_{model_kind}_sft_all")
    beam_evaluate = evaluation.evaluate_generation_ranking_metrics

    def evaluate(model, loader, *, ks, **kwargs):
        if ks == (5,):
            epoch = calls.count("validation")  # 0 is init, 1 is after the first epoch.
            assert loader is val_batches and kwargs["num_beams"] == 5
            if epoch == 0:
                assert calls == []
                torch.testing.assert_close(model.state_dict(), torch.load(checkpoint, weights_only=True),
                                           atol=0, rtol=0)
            if epoch == peak_epoch:
                best_weights.update({k: v.clone() for k, v in model.state_dict().items()})
            calls.append("validation")
            # GRPO ties init on the first epoch, then regresses. SDPO improves on epoch 12.
            return {"NDCG@5": 1.0 if epoch == peak_epoch or (peak_epoch == 0 and epoch == 1) else 0.5}
        torch.testing.assert_close(model.state_dict(), best_weights, atol=0, rtol=0)
        return beam_evaluate(model, loader, ks=ks, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_generation_ranking_metrics", evaluate)
    grid = backfill_grpo.cells if algorithm == "grpo" else backfill_sdpo.cells
    argv = ["--index", str(index), "--output-root", str(tmp_path), "--num-workers", "0", "--device", "cpu"]
    if algorithm == "grpo":
        with pytest.raises(RuntimeError, match="simulated first training step interruption"):
            rl_backfill.main(algorithm, grid, argv)
        assert calls == ["validation", "train"]
    with pytest.raises(RuntimeError, match="simulated evaluation interruption"):
        rl_backfill.main(algorithm, grid, argv)
    assert not list(tmp_path.rglob("final_summary.json"))
    rl_backfill.main(algorithm, grid, argv)
    rl_backfill.main(algorithm, grid, argv)
    summary_path, = tmp_path.rglob("final_summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["sft_weight"] == 0 and summary["num_negatives"] == 16
    assert summary["early_stopping_min_epochs"] == 10 and summary["patience"] == 5
    assert summary["initial_validation"]["NDCG@5"] == (1.0 if peak_epoch == 0 else 0.5)
    last = torch.load(summary_path.parent / "last.pt", weights_only=True)
    assert last["initial_validation"] == summary["initial_validation"]
    assert calls.count("validation") == stop_epoch + 1
    assert summary["epochs_completed"] == len(summary["history"]) == stop_epoch
    assert summary["best_validation"]["epoch"] + 1 == peak_epoch
    best_path = Path(summary["best_checkpoint"])
    assert best_path.is_file()
    if peak_epoch == 0:
        assert best_path.name == "best_init.pt"
    assert summary["test_metrics"]["num_examples"] == 2
    for k in (5, 10, 20):
        assert 0 <= summary["test_metrics"][f"NDCG@{k}"] <= summary["test_metrics"][f"HR@{k}"] <= 1
        assert 0 < summary["test_metrics"][f"Coverage@{k}"] <= 1
        assert summary["test_metrics"][f"ARP@{k}"] > 0
    assert calls.count("test") == 2 and calls.count("train") == (stop_epoch + 1 if algorithm == "grpo" else 0)
