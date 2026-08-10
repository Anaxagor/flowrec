# Task 5 Report: Published TIGER Checkpoint Regression

## Status

Completed as a test-only published-artifact regression. No production code or
artifacts were modified. The new regression passed on its first execution, as
expected: Tasks 2-4 already established the generic trainer-checkpoint loader,
the checkpoint-compatible TIGER architecture, and the canonical exp2 configs.

## Coverage

`test_published_checkpoint_strictly_loads_with_shifted_semantic_ids` is
parametrized for Beauty and Instruments. For each dataset it:

- reads the canonical exp2 config and its configured SID JSON;
- applies `offset_semantic_ids_by_one` exactly once;
- builds TIGER via `scripts.run_tiger.build_model()`;
- strictly loads the configured published `best.pt` through
  `FlowRecModel.load_checkpoint()`;
- checks full `item_semantic_ids` equality, `d_kv == 32`, and no user token;
- generates one deterministic, constrained result from one valid non-empty
  synthetic history and checks the level-0 token is in `1..256`.

## Verification

All tests ran with the repository root on `PYTHONPATH` in an isolated Python
3.11 environment using `requirements.txt`.

| Command | Result | Pytest time | Wall time |
| --- | --- | --- | --- |
| `env PYTHONPATH=. uv run --python 3.11 --with-requirements requirements.txt pytest -q tests/test_tiger_checkpoint_compatibility.py -k published_checkpoint` | 2 passed, 15 deselected | 3.77s | 7.1s |
| `env PYTHONPATH=. uv run --python 3.11 --with-requirements requirements.txt /usr/bin/time -p pytest -q tests/test_tiger_checkpoint_compatibility.py` | 17 passed | 3.89s | 4.78s |
| `env PYTHONPATH=. uv run --python 3.11 --with-requirements requirements.txt /usr/bin/time -p pytest -q tests/test_tiger_sft_speedups.py` | 11 passed, 1 warning | 4.43s | 5.37s |

The SFT suite warning is the existing `torch.cuda.amp.GradScaler` deprecation
in `flowrec/training/trainer.py:326`.

## Artifact Integrity

The published checkpoint binaries were read only:

```text
439006b61a4e2aa301c6cf27bbdff69515922dfd52f18dd2f364bb9b77e63fca  beauty/best.pt
e1ea9f28f4722773d0306fdf9bb18712b25b7718de8d348b25f69bb6280b79fd  instruments/best.pt
```

## Concerns

The default local Python is 3.14, which cannot resolve the pinned
`torch==2.6.0`; the preinstalled Python 3.11 environment also has an unrelated
NumPy/pandas ABI mismatch. Verification therefore used a clean, isolated
Python 3.11 environment that honors the repository's pinned requirements.
