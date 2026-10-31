"""The 500m gflow script refuses to mix runs and baselines from different init checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_yambda500m as g  # noqa: E402


def test_all_dataset_checkpoints_use_initial_layout() -> None:
    root = Path("checkpoints/exp2/loo_split/initial")
    for spec in g.DATASETS.values():
        for key in ("TIGER_CHECKPOINT", "SASREC_CHECKPOINT", "SASREC_TRIAL"):
            assert Path(spec[key]).is_relative_to(root)
    assert g.DATASETS["yambda_500m_listens"]["SASREC_CHECKPOINT"] == str(
        root / "sasrec_sce/yambda_500m_listens/best.pt")


@pytest.mark.parametrize("saved", [{"user_ids_count": 0},
                                  {"user_ids_count": 1, "use_user_token": False}])
def test_legacy_ablation_preserves_checkpoint_user_token_setting(saved) -> None:
    import rewards_ablation_amazon as legacy

    dm = SimpleNamespace(sid_len=2, id2item=[None, "item"], semantic_ids={"item": [1, 257]})
    config = {"embedding_dim": 8, "num_heads": 2, "num_encoder_layers": 1,
              "num_decoder_layers": 1, "dim_feedforward": 16, "d_kv": 4}
    model = legacy.build_tiger_model(dm, config, saved)
    assert model.use_user_token is False
    assert model.user_ids_count == saved["user_ids_count"]


def test_legacy_ablation_resolves_promoted_and_flat_tokenizers(tmp_path) -> None:
    import rewards_ablation_amazon as legacy

    flat = tmp_path / "beauty_2023" / "model.pt"
    nested = tmp_path / "beauty" / "rqvae" / "model.pt"
    for path in (flat, nested):
        path.parent.mkdir(parents=True)
        path.touch()
    assert legacy.rqvae_checkpoint_path(tmp_path, "beauty") == nested
    assert legacy.rqvae_checkpoint_path(tmp_path, "beauty_2023") == flat


class TestCheckpointMatches:
    def test_repo_relative_suffix_ignores_host_prefixes(self) -> None:
        assert g.checkpoint_matches("/home/test_researcher/flowrec/" + g.TIGER_CHECKPOINT)
        assert g.checkpoint_matches(str(PROJECT_ROOT / g.TIGER_CHECKPOINT))
        assert g.checkpoint_matches(g.TIGER_CHECKPOINT)

    def test_other_checkpoints_do_not_match(self) -> None:
        snapshot = ("/home/test_researcher/flowrec/checkpoints/exp2-tuned_params/initial/tiger/"
                    "yambda_500m_listens/trials/dropout-0/best.pt")
        assert not g.checkpoint_matches(snapshot)
        assert not g.checkpoint_matches(None)
        assert not g.checkpoint_matches("")
        # a longer path that merely contains the name as a substring
        assert not g.checkpoint_matches("x" + g.TIGER_CHECKPOINT)


class TestStaleRuns:
    def test_run_from_another_checkpoint_is_stale(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "interaction" / "pow_1" / "db_k10" / "seed_42"
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(json.dumps(
            {"sft_checkpoint": "/home/test_researcher/flowrec/checkpoints/exp2-tuned_params/initial/tiger/"
                               "yambda_500m_listens/trials/dropout-0/best.pt"}))
        reason = g.stale_run_reason(run_dir)
        assert reason is not None and "trials/dropout-0" in reason

        moved = g.retire_stale_run(run_dir, reason, "tag")
        assert not run_dir.exists()
        assert moved.exists() and moved.name.startswith("seed_42.stale-")
        assert (moved / "resolved_config.json").exists()

    def test_current_and_fresh_runs_are_not_stale(self, tmp_path: Path) -> None:
        assert g.stale_run_reason(tmp_path / "missing") is None
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "resolved_config.json").write_text(json.dumps(
            {"sft_checkpoint": "/cluster/flowrec/" + g.TIGER_CHECKPOINT}))
        assert g.stale_run_reason(run_dir) is None
