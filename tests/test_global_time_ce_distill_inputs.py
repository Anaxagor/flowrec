import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/global-time"))
import ce_distill_inputs as inputs
from flowrec.data.dense_prefix import dense_prefix_signature


@pytest.mark.parametrize("dataset", ["beauty", "instruments"])
@pytest.mark.parametrize("model", ["tiger", "gpt2"])
def test_each_student_stages_its_global_weights_and_native_layout(model, dataset, tmp_path):
    files = inputs.sources.input_files(dataset, student=model)
    student = inputs.checkpoint_path(model, dataset, tmp_path)
    assert student.relative_to(tmp_path) in files
    assert inputs.checkpoint_path("sasrec", dataset, tmp_path).relative_to(tmp_path) in files
    assert all(f.parts[:4] == ("checkpoints", "exp2", "global_split", "initial")
               for f in files if f.parts[0] == "checkpoints")
    # Both arms go through the same reward configuration builder, and select
    # their different training histories before the datamodule is constructed.
    base = SimpleNamespace(
        DATASETS={},
        tuned_profile=lambda *args: ({"model": {"embedding_dim": 128}}, {}),
        gpt2_runner=SimpleNamespace(user_ids_count=lambda *_: 1),
        runner=SimpleNamespace(load_yaml=lambda _: {}),
        load_json=lambda path: json.loads(path.read_text()),
        build_run_config=lambda ds, b, user_count: {
            "dataset": {"train_prefix_from_dense": b.prefix_from_dense,
                        "max_history_len": 20, "train_target_policy": "all"},
            "model": {"user_ids_count": user_count}},
    )
    inputs.configure_inputs(base, model, dataset, tmp_path)
    if model == "tiger":
        student.parent.mkdir(parents=True)
        student.with_name("resolved_config.json").write_text(json.dumps({"model": {"d_kv": 32}}))
        student.with_name("best.pt.json").write_text(json.dumps({"model_config": {"user_ids_count": 1}}))
    config = inputs.build_run_config(base, model, dataset, SimpleNamespace())
    context = dense_prefix_signature(config["dataset"])
    if model == "gpt2":
        assert context["context"] == "dense_window" and context["window_len"] == 20
    else:
        assert context is None and config["model"]["d_kv"] == 32
        assert config["model"]["user_ids_count"] == 1
