"""The 500m gflow script refuses to mix runs and baselines from different init checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_yambda500m as g  # noqa: E402


class TestCheckpointMatches:
    def test_repo_relative_suffix_ignores_host_prefixes(self) -> None:
        assert g.checkpoint_matches("/home/ideeva/flowrec/" + g.TIGER_CHECKPOINT)
        assert g.checkpoint_matches(str(PROJECT_ROOT / g.TIGER_CHECKPOINT))
        assert g.checkpoint_matches(g.TIGER_CHECKPOINT)

    def test_other_checkpoints_do_not_match(self) -> None:
        snapshot = ("/home/ideeva/flowrec/checkpoints/exp2-tuned_params/initial/tiger/"
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
            {"sft_checkpoint": "/home/ideeva/flowrec/checkpoints/exp2-tuned_params/initial/tiger/"
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
