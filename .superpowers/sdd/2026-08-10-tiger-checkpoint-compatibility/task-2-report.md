# Task 2 Report: Support Trainer Checkpoints In The Generic Loader

## Status

DONE

## Worktree And Commit

- Worktree: `sft_speedup`
- Branch: `sft_speedup`
- Implementation commit: `3bc1701 feat: support trainer checkpoint payloads`
- Scope: generic loader payload selection and one focused regression test only.
- Checkpoint and artifact files were not modified.

## RED

Added `tests/test_tiger_checkpoint_compatibility.py` with the specified real
trainer-payload test and a tiny concrete `FlowRecModel` fixture.

The literal brief command was first attempted:

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py::test_generic_loader_accepts_trainer_model_state_dict
```

Exact shell output in this environment:

```text
zsh:1: command not found: pytest
```

The same test was then run through the temporary `uv` environment with the
repository on `PYTHONPATH`:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/scam-uv-cache uv run --with pytest --with torch pytest -q tests/test_tiger_checkpoint_compatibility.py::test_generic_loader_accepts_trainer_model_state_dict
```

Exact pytest result:

```text
F                                                                        [100%]
E           RuntimeError: Error(s) in loading state_dict for _TinyCheckpointModel:
E           	Missing key(s) in state_dict: "weight".
E           	Unexpected key(s) in state_dict: "model_state_dict", "trainer_state".
1 failed, 1 warning in 0.92s
```

This is the expected RED failure: the old generic loader passed the complete
trainer payload to strict `load_state_dict()`.

## Implementation

Updated `FlowRecModel.load_checkpoint()` to select state data with the required
precedence:

1. `model_state_dict`
2. `state_dict`
3. the raw mapping itself

Non-mapping payloads remain raw state dictionaries. The existing `strict`
argument is passed through unchanged, with `strict=True` still the default.

## GREEN

Focused compatibility suite:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/scam-uv-cache uv run --with pytest --with torch pytest -q tests/test_tiger_checkpoint_compatibility.py
```

Exact pytest result:

```text
.                                                                        [100%]
1 passed, 1 warning in 0.88s
```

Relevant existing TIGER SFT speedup suite:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/scam-uv-cache uv run --with pytest --with torch --with numpy --with pandas --with polars --with pyarrow --with transformers --with pyyaml pytest -q tests/test_tiger_sft_speedups.py
```

Exact pytest result:

```text
...........                                                              [100%]
11 passed, 1 warning in 37.64s
```

The warnings were environmental: PyTorch warned that NumPy was absent in the
minimal focused environment, and the existing trainer test emitted a
`torch.cuda.amp.GradScaler` deprecation warning. Neither warning caused a
failure.

## Final dlenv Verification

Per the Task 1 environment, the focused test was also run with:

```bash
python3.11 -m pytest -q tests/test_tiger_checkpoint_compatibility.py
```

Exact result:

```text
.                                                                        [100%]
1 passed, 1 warning in 1.38s
```

The same dlenv command for the existing SFT suite first encountered the
sandbox restriction below in its persistent-worker test after 10 tests passed:

```text
RuntimeError: torch_shm_manager at "torch_shm_manager": Operation not permitted
1 failed, 10 passed, 1 warning in 4.53s
```

With the required elevated permission for PyTorch shared memory, the exact
rerun was:

```bash
python3.11 -m pytest -q tests/test_tiger_sft_speedups.py
```

Exact result:

```text
...........                                                              [100%]
11 passed, 1 warning in 4.40s
```

## Changed Files

- `flowrec/models/base.py`: added trainer/portable/raw payload precedence.
- `tests/test_tiger_checkpoint_compatibility.py`: added the specified trainer
  checkpoint regression test and tiny real model fixture.
- `.superpowers/sdd/2026-08-10-tiger-checkpoint-compatibility/task-2-report.md`:
  this report.

No checkpoint, semantic-ID, embedding, or other artifact file was touched. No
later task was implemented.

## Self-Review

- `Mapping` was already imported and is used for the required payload check.
- `model_state_dict` wins when both wrapper keys are present, matching the
  required remote precedence.
- Portable `state_dict` payloads still load, and raw state dictionaries still
  pass through.
- `strict=True` remains the default and is forwarded without suppression.
- The test verifies actual serialization, loading, parameter transfer, and
  preservation of trainer metadata in the returned payload.
- `git diff --check` passed before the implementation commit.
- The final focused and relevant existing test commands passed as recorded
  above.
