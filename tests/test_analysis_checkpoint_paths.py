"""A fine-tuning seed must not switch the selected primary input model."""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


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



def test_current_sft_checkpoint_resolves_without_substitution(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/exp5_result_analysis"))
    from inference import resolve_project_path
    relative = Path("checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty/best.pt")
    assert resolve_project_path(relative, tmp_path) == tmp_path / relative
    assert resolve_project_path("/cluster/flowrec/" + str(relative), tmp_path) == tmp_path / relative


def test_analysis_rejects_checkpoint_outside_current_structure(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/exp5_result_analysis"))
    from inference import resolve_project_path
    with pytest.raises(ValueError, match="current checkpoints structure"):
        resolve_project_path("checkpoints/obsolete/tiger/beauty/best.pt", tmp_path)
