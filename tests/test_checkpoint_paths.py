"""Current primary paths never fall back to old checkpoint layouts."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_paths import checkpoint_path_matches, initial_asset_path, sft_promotion_directory
from gpt2_matched_finetune import trial_paths


def put(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint fixture")
    return path


@pytest.mark.parametrize("model,dataset", [
    ("tiger", "yambda_50m_listens"),
    ("tiger", "yambda_500m_listens"),
    ("sasrec", "beauty"),
    ("sasrec_sce", "yambda_500m_listens"),
])
def test_current_reader_and_identity_use_selected_checkpoint(tmp_path, model, dataset):
    relative = f"checkpoints/exp2/loo_split/initial/{model}/{dataset}/best.pt"
    assert initial_asset_path(tmp_path, relative) == tmp_path / relative
    assert not initial_asset_path(tmp_path, relative).exists()
    canonical = put(tmp_path, relative)
    assert initial_asset_path(tmp_path, relative) == canonical
    assert checkpoint_path_matches(tmp_path, "/cluster/flowrec/" + relative, relative)
    assert not checkpoint_path_matches(tmp_path, "/cluster/flowrec/checkpoints/exp2/loo_split/initial/wrong/best.pt", relative)


def test_current_metadata_and_explicit_experiment_paths_are_used_as_configured(tmp_path):
    for relative in ("checkpoints/exp2/loo_split/initial/sasrec/beauty/best.pt.json",
                     "experiments/new_run/checkpoints/last.pt"):
        assert initial_asset_path(tmp_path, relative) == tmp_path / relative


@pytest.mark.parametrize("dataset", ["yambda_50m_listens", "yambda_500m_listens"])
def test_selected_trial_uses_only_current_last128_assets(tmp_path, dataset):
    relative = f"checkpoints/exp2/loo_split/initial/gpt2/{dataset}/last_128/best.pt"
    assert trial_paths(tmp_path, dataset)["checkpoint"] == tmp_path / relative
    assert not trial_paths(tmp_path, dataset)["checkpoint"].exists()
    canonical = put(tmp_path, relative)
    paths = trial_paths(tmp_path, dataset)
    assert paths["checkpoint"] == canonical
    assert {p.parent for p in paths.values()} == {canonical.parent}
    assert paths["data_audit"].name == "data_audit.json"


def test_primary_sft_cannot_be_overwritten_by_another_seed_or_protocol(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/sft-finetuned/tiger"
    dataset = "yambda_500m_listens"
    primary = root / dataset
    primary.mkdir(parents=True)
    source = tmp_path / f"experiments/exp2-sft-finetune-matched-r15/{dataset}/sft_lr3e-05/seed_42/checkpoints/best.pt"
    (primary / "provenance.json").write_text(json.dumps({"seed":42,"source_checkpoint":str(source.relative_to(tmp_path))}))
    assert sft_promotion_directory(tmp_path, root, dataset, 3e-5, 42, source) == primary
    other = Path(str(source).replace("-matched-r15/", "-matched/"))
    assert sft_promotion_directory(tmp_path, root, dataset, 3e-5, 42, other) is None
    assert sft_promotion_directory(tmp_path, root, dataset, 3e-5, 43, source) is None
    assert sft_promotion_directory(tmp_path, root, dataset, 1e-3, 42, source) is None


def test_new_primary_sft_is_flat_and_custom_experiment_roots_keep_seed_weights(tmp_path):
    root = tmp_path / "checkpoints/exp2/loo_split/sft-finetuned/tiger"
    source = tmp_path / "experiments/run/checkpoints/best.pt"
    assert sft_promotion_directory(tmp_path, root, "beauty", 3e-5, 42, source) == root / "beauty"
    custom = tmp_path / "experiments/custom/promoted/tiger"
    assert sft_promotion_directory(tmp_path, custom, "beauty", 3e-5, 43, source) == (
        custom / "beauty/sft_only_lr3e-05/seed_43")


def test_yambda_initial_regimes_keep_legacy_separate(tmp_path):
    dataset = "yambda_500m_listens"
    base = f"checkpoints/exp2/loo_split/initial/gpt2/{dataset}"
    current = put(tmp_path, f"{base}/last_128/best.pt")
    legacy = put(tmp_path, f"{base}/full_history/best.pt")
    assert initial_asset_path(tmp_path, str(current)) == current
    assert initial_asset_path(tmp_path, str(legacy)) == legacy
    assert checkpoint_path_matches(tmp_path, f"/cluster/flowrec/{base}/last_128/best.pt", str(current))
    assert checkpoint_path_matches(tmp_path, f"/cluster/flowrec/{base}/full_history/best.pt", str(legacy))
    assert not checkpoint_path_matches(tmp_path, str(legacy), str(current))
