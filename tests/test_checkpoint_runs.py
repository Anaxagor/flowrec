"""Current primary models replace historical seed refs without absorbing run trees."""
import json
from pathlib import Path
import pytest

from scripts.checkpoint_runs import (
    incumbent_checkpoint_path,
    publish_primary_checkpoint,
    seed_checkpoint_path,
    training_checkpoint_dir,
)


def write(path: Path, value: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_best_seed_references_use_current_primary(tmp_path):
    root = tmp_path / "checkpoints/exp1/gpt2"
    primary = write(root / "beauty/best.pt")
    write(primary.parent / "provenance.json", b'{"seed": 42}')
    assert seed_checkpoint_path(tmp_path, root, "beauty", 42) == primary
    assert seed_checkpoint_path(tmp_path, root, "beauty", 43) == primary
    write(primary.parent / "provenance.json", b'{"seed": null}')
    assert seed_checkpoint_path(tmp_path, root, "beauty", 44) == primary


def test_explicit_last_uses_current_run_while_best_uses_current_primary(tmp_path):
    root = tmp_path / "checkpoints/exp1/sasrec"
    primary = write(root / "beauty_2023/best.pt", b"current")
    assert seed_checkpoint_path(tmp_path, root, "beauty_2023", 43) == primary
    last = write(training_checkpoint_dir(tmp_path, root, "beauty_2023", 43) / "last.pt")
    assert seed_checkpoint_path(tmp_path, root, "beauty_2023", 43, which="last") == last


def test_selected_matched_base_does_not_select_full_history_variant(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
    primary = write(root / "yambda_50m_listens/last_128/best.pt")
    write(primary.parent / "provenance.json", b'{"seed": 42, "trial": 36}')
    write(root / "yambda_50m_listens/full_history/best.pt", b"another regime")
    assert seed_checkpoint_path(tmp_path, root, "yambda_50m_listens", 42) == primary
    assert seed_checkpoint_path(tmp_path, root, "yambda_50m_listens", 43) == primary


def test_missing_primary_never_falls_back_to_raw_seed_best(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/initial/sasrec"
    directory = training_checkpoint_dir(tmp_path, root, "beauty", 43)
    assert directory == tmp_path / "experiments/checkpoints/exp2/loo_split/initial/sasrec/beauty/seed_43"
    write(directory / "best.pt", b"new run")
    missing = seed_checkpoint_path(tmp_path, root, "beauty", 43)
    assert missing == root / "beauty/best.pt"
    assert not missing.exists()
    primary = write(root / "beauty/best.pt", b"selected primary")
    assert seed_checkpoint_path(tmp_path, root, "beauty", 43) == primary


def test_custom_global_time_roots_remain_private(tmp_path):
    root = tmp_path / "experiments/global-time/checkpoints/exp2-tuned_params/sasrec"
    checkpoint = write(root / "beauty/seed_42/best.pt")
    assert training_checkpoint_dir(tmp_path, root, "beauty", 42) == checkpoint.parent
    assert seed_checkpoint_path(tmp_path, root, "beauty", 42) == checkpoint
    assert incumbent_checkpoint_path(tmp_path, root, "beauty") == root / "beauty/best_model.pt"
    publish_primary_checkpoint(tmp_path, root, "beauty", 42, checkpoint)
    assert not (tmp_path / "checkpoints").exists()


def test_global_primary_has_separate_current_seed_storage(tmp_path):
    root = tmp_path / "checkpoints/exp2/global_split/initial/sasrec"
    raw = training_checkpoint_dir(tmp_path, root, "beauty", 42)
    assert raw == tmp_path / "experiments/global-time/checkpoints/exp2-tuned_params/sasrec/beauty/seed_42"
    primary = write(root / "beauty/best.pt", b"global primary")
    assert seed_checkpoint_path(tmp_path, root, "beauty", 43) == primary
    last = training_checkpoint_dir(tmp_path, root, "beauty", 43) / "last.pt"
    assert seed_checkpoint_path(tmp_path, root, "beauty", 43, which="last") == last
    assert not last.exists()
    assert not (tmp_path / "checkpoints/exp2/loo_split").exists()


def test_gpt2_yambda_regimes_follow_the_recorded_split(tmp_path):
    loo = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
    global_root = tmp_path / "checkpoints/exp2/global_split/initial/gpt2"
    matched = write(loo / "yambda_50m_listens/last_128/best.pt", b"matched")
    full = write(global_root / "yambda_50m_listens/full_history/best.pt", b"global full history")
    assert incumbent_checkpoint_path(tmp_path, loo, "yambda_50m_listens") == matched
    assert seed_checkpoint_path(tmp_path, global_root, "yambda_50m_listens", 44) == full
    missing = seed_checkpoint_path(tmp_path, global_root, "yambda_500m_listens", 42)
    assert missing == global_root / "yambda_500m_listens/full_history/best.pt"
    assert not missing.exists()


def test_legacy_training_cannot_overwrite_selected_last128_base(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
    selected = write(root / "yambda_50m_listens/last_128/best.pt", b"selected trial")
    source = write(tmp_path / "experiments/legacy/checkpoints/best.pt", b"wrong model")
    run = source.parent.parent
    write(run / "resolved_config.json", b'{"data": {"split_strategy": "leave_one_out", "train_target_policy": "all"}}')
    with pytest.raises(ValueError, match="different GPT-2 data protocol"):
        publish_primary_checkpoint(tmp_path, root, "yambda_50m_listens", 42, source, run_dir=run)
    assert selected.read_bytes() == b"selected trial"


def test_seed42_publication_is_flat_and_replaces_stale_metadata(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
    source = write(tmp_path / "experiments/run/checkpoints/best.pt")
    write(Path(str(source) + ".json"), b'{"model_type": "gpt2"}')
    run = source.parent.parent
    write(run / "resolved_config.json", b'{"seed": 42}')
    write(root / "beauty/best_trial.json", b'{"best_trial_number": 99}')
    write(root / "beauty/data_audit.json", b'{"old": true}')
    write(root / "beauty/study_config.json", b'{"old": true}')
    write(root / "beauty/data_summary.json", b'{"old": true}')
    publish_primary_checkpoint(tmp_path, root, "beauty", 43, source, run_dir=run)
    assert not (root / "beauty/best.pt").exists()
    publish_primary_checkpoint(tmp_path, root, "beauty", 42, source, run_dir=run,
                               parameters={"dropout": .1}, summary={"seed": 42, "dataset": {"name": "beauty"}})
    primary = root / "beauty/best.pt"
    assert primary.read_bytes() == source.read_bytes()
    assert all(p.is_file() for p in primary.parent.iterdir())
    assert not (primary.parent / "data_audit.json").exists()
    assert not (primary.parent / "study_config.json").exists()
    assert not (primary.parent / "data_summary.json").exists()
    metadata = json.loads((primary.parent / "best_trial.json").read_text())
    assert "best_trial_number" not in metadata
    assert metadata["best_params"] == {"dropout": .1}
    assert json.loads((primary.parent / "training_summary.json").read_text())["runs"][0]["seed"] == 42
    assert json.loads((primary.parent / "provenance.json").read_text())["seed"] == 42
    assert incumbent_checkpoint_path(tmp_path, root, "beauty") == primary
