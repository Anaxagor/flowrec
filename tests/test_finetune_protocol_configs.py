"""The exp2 fine-tuning protocol configs stay in sync with their scripts.

configs/exp2-tuned-params/finetune_sft_only.json documents the SFT-only
control; finetune_gflow.json is CONSUMED by the gflow ablation script as its
argparse defaults. The shared block must match the control, so the two
fine-tuning arms remain directly comparable.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SFT_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_sft_only.json"
GFLOW_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_gflow.json"


def _gflow_args():
    import gflow_finetune_yambda50m as g

    old_argv = sys.argv
    sys.argv = ["x"]
    try:
        return g.parse_args()
    finally:
        sys.argv = old_argv


def _control_args():
    import sft_finetune_control as control

    old_argv = sys.argv
    sys.argv = ["x"]
    try:
        return control.parse_args()
    finally:
        sys.argv = old_argv


class TestGflowScriptConsumesItsConfig:
    def test_defaults_come_from_the_config(self) -> None:
        cfg = json.loads(GFLOW_CONFIG.read_text())
        shared, gflow = cfg["shared_with_sft_control"], cfg["gflow"]
        a = _gflow_args()
        assert a.lr == shared["lr"]
        assert a.weight_decay == shared["weight_decay"]
        assert a.batch_size == shared["batch_size"]
        assert a.grad_accum == shared["grad_accum"]
        assert a.max_epochs == shared["max_epochs"]
        assert a.validate_every == shared["validate_every"]
        assert a.patience == shared["patience"]
        assert a.max_grad_norm == shared["max_grad_norm"]
        assert a.amp_dtype == shared["amp_dtype"]
        assert a.val_batch_size == shared["val_batch_size"]
        assert a.eval_num_beams == shared["eval_num_beams"]
        assert a.test_ks == shared["test_ks"]
        assert a.seed == shared["seed"]
        assert a.validate_at_start == shared["run_validation_at_start"]
        assert a.gflow_weight_mode == gflow["gflow_weight_mode"]
        assert a.gflow_target_grad_ratio == gflow["gflow_target_grad_ratio"]
        assert a.gflow_calibration_batches == gflow["gflow_calibration_batches"]
        assert a.curriculum_steps == gflow["curriculum_steps"]
        assert a.sasrec_transform == gflow["sasrec_transform"]
        assert a.train_last_n == gflow["train_last_n"]
        assert a.max_candidate_batch == gflow["max_candidate_batch"]
        assert a.cache_dtype == gflow["cache_dtype"]


class TestSftControlMatchesItsConfig:
    def test_argparse_defaults(self) -> None:
        cfg = json.loads(SFT_CONFIG.read_text())
        a = _control_args()
        assert a.lr == [cfg["optimizer"]["lr"]]
        assert a.weight_decay == cfg["optimizer"]["weight_decay"]
        assert a.batch_size == cfg["batching"]["batch_size"]
        assert a.grad_accum == cfg["batching"]["grad_accum"]
        assert a.max_epochs == cfg["schedule"]["max_epochs"]
        assert a.validate_every == cfg["schedule"]["validate_every"]
        assert a.patience == cfg["schedule"]["patience"]
        assert a.max_grad_norm == cfg["schedule"]["max_grad_norm"]
        assert a.val_num_beams == cfg["monitoring"]["val_num_beams"]
        assert a.eval_num_beams == cfg["final_evaluation"]["eval_num_beams"]
        assert a.test_ks == cfg["final_evaluation"]["test_ks"]
        assert a.seeds == cfg["data"]["seeds"]
        assert a.validate_at_start == cfg["schedule"]["run_validation_at_start"]

    def test_hardcoded_protocol_values(self) -> None:
        import sft_finetune_control as control

        cfg = json.loads(SFT_CONFIG.read_text())
        assert control.MONITOR == cfg["monitoring"]["monitor"]
        # amp dtype is assembled inside build_run_config, not argparse.
        source = Path(control.__file__).read_text()
        assert f'"amp_dtype": "{cfg["schedule"]["amp_dtype"]}"' in source


class TestArmsShareTheProtocol:
    def test_gflow_matches_the_sft_control(self) -> None:
        """The core comparability contract between the two fine-tuning arms."""
        g = _gflow_args()
        c = _control_args()
        assert g.lr == c.lr[0]
        assert g.weight_decay == c.weight_decay
        assert g.batch_size == c.batch_size
        assert g.grad_accum == c.grad_accum
        assert g.max_epochs == c.max_epochs
        assert g.validate_every == c.validate_every
        assert g.patience == c.patience
        assert g.max_grad_norm == c.max_grad_norm
        assert g.eval_num_beams == c.eval_num_beams
        assert g.test_ks == c.test_ks
        assert g.validate_at_start == c.validate_at_start
        # amp: control hardcodes float16 in build_run_config (checked above);
        # the gflow arm must use the same.
        assert g.amp_dtype == "float16"

    def test_init_checkpoint_is_the_patience_reference(self) -> None:
        """Both arms score the init first: a run that never beats it stops after
        `patience` validations and reports the init (best_epoch -1) as its best."""
        assert _gflow_args().validate_at_start is True
        assert _control_args().validate_at_start is True

    def test_configs_agree_with_each_other(self) -> None:
        sft = json.loads(SFT_CONFIG.read_text())
        shared = json.loads(GFLOW_CONFIG.read_text())["shared_with_sft_control"]
        assert shared["lr"] == sft["optimizer"]["lr"]
        assert shared["weight_decay"] == sft["optimizer"]["weight_decay"]
        assert shared["batch_size"] == sft["batching"]["batch_size"]
        assert shared["grad_accum"] == sft["batching"]["grad_accum"]
        assert shared["max_epochs"] == sft["schedule"]["max_epochs"]
        assert shared["validate_every"] == sft["schedule"]["validate_every"]
        assert shared["patience"] == sft["schedule"]["patience"]
        assert shared["max_grad_norm"] == sft["schedule"]["max_grad_norm"]
        assert shared["amp_dtype"] == sft["schedule"]["amp_dtype"]
        assert shared["run_validation_at_start"] == sft["schedule"]["run_validation_at_start"]
        assert shared["eval_num_beams"] == sft["final_evaluation"]["eval_num_beams"]
        assert shared["test_ks"] == sft["final_evaluation"]["test_ks"]


def _gflow_500m_args():
    import gflow_finetune_yambda500m as g

    old_argv = sys.argv
    sys.argv = ["x"]
    try:
        return g.parse_args()
    finally:
        sys.argv = old_argv


class TestYambda500mScriptConsumesItsConfig:
    """The 500m variant shares the SFT-control block and takes only its
    dataset-scale knobs from the yambda_500m override block."""

    def test_registered_as_consumer(self) -> None:
        cfg = json.loads(GFLOW_CONFIG.read_text())
        assert "scripts/exp4-reward-ablations/gflow_finetune_yambda500m.py" in cfg["consumers"]
        assert set(cfg["yambda_500m"]) >= {"train_last_n", "val_sample_fraction", "hidden_dtype"}

    def test_shared_protocol_and_override_block(self) -> None:
        cfg = json.loads(GFLOW_CONFIG.read_text())
        shared, block = cfg["shared_with_sft_control"], cfg["yambda_500m"]
        a = _gflow_500m_args()
        assert a.lr == shared["lr"]
        assert a.weight_decay == shared["weight_decay"]
        assert a.batch_size == shared["batch_size"]
        assert a.grad_accum == shared["grad_accum"]
        assert a.max_epochs == shared["max_epochs"]
        assert a.validate_every == shared["validate_every"]
        assert a.patience == shared["patience"]
        assert a.max_grad_norm == shared["max_grad_norm"]
        assert a.amp_dtype == shared["amp_dtype"]
        assert a.val_batch_size == shared["val_batch_size"]
        assert a.eval_num_beams == shared["eval_num_beams"]
        assert a.test_ks == shared["test_ks"]
        assert a.train_last_n == block["train_last_n"]
        assert a.val_sample_fraction == block["val_sample_fraction"]
        assert a.hidden_dtype == block["hidden_dtype"]
        assert a.validate_at_start == shared["run_validation_at_start"]
        # the monitor sample is drawn once: a re-drawn sample made every run
        # pick the epoch with the easiest subset
        assert block["val_sample_resample"] is False
        assert a.val_sample_resample is False

    def test_curriculum_covers_the_same_examples_for_every_k(self) -> None:
        """Larger K shrinks the micro-batch; the easy->hard ramp must not
        shorten with it (it is defined in examples, not micro-steps)."""
        import gflow_finetune_yambda500m as g

        a = _gflow_500m_args()
        protocol_examples = a.curriculum_steps * a.batch_size
        for k in (3, 10, 30, 100):
            batch, _ = g.base.batch_settings_for_k(k, a)
            steps = g.curriculum_steps_for_batch(a, batch)
            if batch == a.batch_size:
                assert steps == a.curriculum_steps
            else:
                assert steps > a.curriculum_steps
                assert abs(steps * batch - protocol_examples) <= batch

    def test_matches_the_sft_control(self) -> None:
        g = _gflow_500m_args()
        c = _control_args()
        assert g.lr == c.lr[0]
        assert g.weight_decay == c.weight_decay
        assert g.batch_size == c.batch_size
        assert g.grad_accum == c.grad_accum
        assert g.max_epochs == c.max_epochs
        assert g.validate_every == c.validate_every
        assert g.patience == c.patience
        assert g.max_grad_norm == c.max_grad_norm
        assert g.eval_num_beams == c.eval_num_beams
        assert g.test_ks == c.test_ks
        assert g.amp_dtype == "float16"
        assert g.validate_at_start == c.validate_at_start

    def test_starts_from_the_sft_controls_init_checkpoint(self) -> None:
        """The imported init baseline is the control's evaluation of ITS init
        checkpoint, so the gflow runs must fine-tune the same weights (the
        converged promoted best.pt, not the 8-epoch trials/dropout-0 snapshot)."""
        import gflow_finetune_yambda500m as g
        import sft_finetune_control_yambda as yambda_control

        spec = yambda_control.YAMBDA_DATASETS["yambda_500m_listens"]
        assert g.TIGER_CHECKPOINT == spec["tiger_checkpoint"]
        assert "trials/" not in g.TIGER_CHECKPOINT

    def test_key_grid_is_24_runs(self) -> None:
        import gflow_finetune_yambda500m as g

        a = _gflow_500m_args()
        grid = g.base.build_grid(a)
        assert len(grid) == 24
        assert {e["num_negatives"] for e in grid} == {10}
        assert {e["power"] for e in grid} == {1.0, 16.0}
        assert {g.combo_key(e["combo"]) for e in grid} == set(g.DEFAULT_COMBOS)
        assert g.components_for([e["combo"] for e in grid]) == ["sasrec", "interaction", "similarity"]
