# Task 3 Report: TIGER Checkpoint Compatibility

## RED Evidence

1. `test_tiger_config_preserves_checkpoint_critical_fields` failed with
   `KeyError: 'd_kv'`, demonstrating that the existing model did not serialize
   checkpoint-critical configuration or build the flow head.
2. `test_tiger_strict_load_patches_only_flow_head_compatibility` failed with
   missing sequential `flow_head.0/.1/.3` parameters and unexpected legacy
   `flow_head.weight` and `flow_head.bias` keys under strict loading.
3. `test_tiger_runner_forwards_checkpoint_critical_model_fields` failed because
   a config requesting `enable_flow_head=True` produced a model with the field
   set to `False`.

## GREEN Evidence

`/Users/tsessk/anaconda3/envs/dlenv/bin/python -m pytest -q \
tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py`
completed with `15 passed` in 3.36s. The suite was run outside the filesystem
sandbox because PyTorch persistent workers require `torch_shm_manager` access.

## Changed Files

- `flowrec/models/tiger.py`
- `scripts/run_tiger.py`
- `tests/test_tiger_checkpoint_compatibility.py`
- `.superpowers/sdd/2026-08-10-tiger-checkpoint-compatibility/task-3-report.md`

## Decisions

- Kept the flow head opt-in by default to preserve existing local SFT behavior.
- Implemented the exact checkpoint head: LayerNorm, 2x Linear, GELU, final
  Linear; the final layer is zeroed after base-model initialization.
- Kept strict loading for all non-flow-head keys. Only absent current
  `flow_head.*` keys are supplied from the target state, and only obsolete
  `flow_head.weight` and `flow_head.bias` keys are removed.
- Serialized `d_kv`, `layer_norm_eps`, `activation`, `initializer_range`,
  user-token configuration, and flow-head configuration. The runner forwards
  the requested checkpoint-critical fields.
- Did not add configs, checkpoint artifacts, or unrelated flow-scoring helpers.

## Self-Review

- Verified the published exp2 checkpoint flow-head parameter names and shapes:
  `flow_head.0`, `.1`, and `.3` with 128-to-256-to-1 projections.
- Verified the terminal flow-head weight and bias are zero after construction.
- Added coverage for config serialization, strict legacy-flow loading, strict
  rejection of unrelated keys, and real runner construction.
- `git diff --check` passed.

## Remaining Concerns

The passing suite reports two pre-existing deprecation/security warnings from
the generic checkpoint loader and AMP scaler; Task 3 does not modify either.

## Review Fix Round 1/5

### RED Evidence

`/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q` for the four
new regression tests failed as expected:

1. the constructor defaulted `enable_flow_head` to `False`;
2. a disabled head still left `flow_head` present as `None`;
3. copying the state dictionary lost the child module's `_metadata`; and
4. the runner defaulted the flow-head setting to `False`.

The default-enabled test also locks the exact LayerNorm, Linear, GELU, Linear
sequence, its 8-to-16-to-1 projection in the fixture, and a zeroed terminal
weight and bias.

### GREEN Evidence

- Focused compatibility suite: `8 passed` in 3.98s.
- Required compatibility plus SFT suite:
  `/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py`
  completed with `19 passed` in 4.65s.

### Corrections and Self-Review

- The constructor and runner now default `enable_flow_head` to `True`, matching
  the remote contract. This supersedes the earlier opt-in-default decision.
- `flow_head` is created only when enabled; disabled models have no such
  attribute, and `config_dict()` derives `has_flow_head` with `hasattr`.
- `load_state_dict()` preserves the incoming OrderedDict `_metadata` after its
  compatibility copy, verified through the real child-module load hook.
- No files outside the requested model, runner, compatibility test, and task
  report were changed in this review round.
