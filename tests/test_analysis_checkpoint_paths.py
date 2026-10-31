"""A fine-tuning seed must not switch the selected primary input model."""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_exp1_gpt2_beauty_shares_primary_across_finetuning_seeds(monkeypatch):
    # Execute the actual CLI selector without importing the training stack.
    tree = ast.parse((ROOT / "scripts/gflow_finetune_gpt2_beauty.py").read_text())
    names = {"GPT2_CHECKPOINT", "SASREC_CHECKPOINT", "SASREC_TRIAL"}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in names for target in node.targets))
             or (isinstance(node, ast.FunctionDef) and node.name == "parse_args")]
    namespace = {"Path": Path, "argparse": argparse,
                 "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                 "gflow_ablation": SimpleNamespace(protocol_defaults=lambda: {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "checkpoint selector", "exec"), namespace)
    for seed in (42, 44):
        monkeypatch.setattr(sys, "argv", ["finetune", "--seed", str(seed)])
        args = namespace["parse_args"]()
        assert args.seed == seed
        assert args.init_checkpoint == Path("checkpoints/exp1/gpt2/beauty/best.pt")
    monkeypatch.setattr(sys, "argv", ["finetune", "--init-checkpoint", "/custom/best.pt"])
    assert namespace["parse_args"]().init_checkpoint == Path("/custom/best.pt")


def test_old_sft_seed_paths_resolve_only_to_current_primary(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/exp6_result_analysis"))
    from inference import resolve_project_path
    expected = tmp_path / "checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty/best.pt"
    for seed in (42, 44):
        for old in (
            f"experiments/exp2-sft-finetune/beauty/sft_lr3e-05/seed_{seed}/checkpoints/best.pt",
            f"checkpoints/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05/seed_{seed}/best.pt",
            f"artifacts/checkpoint_archive/2026-10-30/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05/seed_{seed}/best.pt",
        ):
            path = tmp_path / old
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old weights")
            assert resolve_project_path(path, tmp_path) == expected
    assert not expected.exists()


def test_old_initial_path_does_not_reopen_archived_weights(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/exp6_result_analysis"))
    from inference import resolve_project_path
    old = tmp_path / "artifacts/checkpoint_archive/2026-10-30/exp2-tuned_params/initial/gpt2/beauty/best_model.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old initial weights")
    expected = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2/beauty/best.pt"
    assert resolve_project_path(old, tmp_path) == expected
    assert not expected.exists()
