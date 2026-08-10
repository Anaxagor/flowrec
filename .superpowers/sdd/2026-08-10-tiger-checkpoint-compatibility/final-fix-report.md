# Final Fix Report

## Status

Completed the single final-review fix wave on branch `sft_speedup`. Strict
checkpoint loading remains the default. No checkpoint artifact or experiment
config was modified.

## Fixes

1. Strict flow-head compatibility now initializes the complete modern
   sequential head only when the incoming state contains no modern
   `flow_head.*` tensors. Legacy single-linear `flow_head.weight` and
   `flow_head.bias` remain removable. A partially present modern head is no
   longer repaired and therefore fails under `strict=True` with its missing
   keys. Existing coverage continues to verify unrelated strict errors and
   preservation of `OrderedDict` metadata.
2. `TigerModel(**model.config_dict())` now accepts the recognized derived
   metadata fields without forwarding duplicate special-token arguments to
   `T5Config`. Serialized `semantic_vocab_size` is reused when no SID mapping
   is supplied so vocabulary-shaped checkpoint parameters remain compatible.
   Declared T5 options still pass through, while arbitrary unknown kwargs raise
   `TypeError`.
3. The published Beauty and Instruments generation regression now checks that
   every complete generated SID row occurs in `expected_sid_tensor`, in
   addition to its existing shape and first-token checks.

## Test-First Evidence

- Partial modern flow-head test RED: three parameterized cases failed with
  `DID NOT RAISE RuntimeError` before the loader fix.
- Config round-trip test RED: failed with `T5Config() got multiple values for
  keyword argument 'pad_token_id'` before recognized metadata was consumed.
- Unknown-metadata test RED: failed with `DID NOT RAISE TypeError` before
  explicit T5 kwarg validation.
- Published SID-row membership is test-only hardening and required no
  production change.

## Verification

Focused strict/metadata tests:

```text
python3.11 -m pytest -q tests/test_tiger_checkpoint_compatibility.py -k 'strict_load or metadata'
5 passed, 15 deselected
```

Focused config tests:

```text
python3.11 -m pytest -q tests/test_tiger_checkpoint_compatibility.py -k config_dict
2 passed, 20 deselected
```

Published checkpoint regression:

```text
python3.11 -m pytest -q tests/test_tiger_checkpoint_compatibility.py -k published_checkpoint
2 passed, 19 deselected, 2 warnings
```

Requested compatibility and SFT suites:

```text
python3.11 -m pytest -q tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py
33 passed, 5 warnings
```

The sandboxed combined run first reached `32 passed` and failed only because
`torch_shm_manager` was not permitted for the persistent-worker test. The same
command passed outside the sandbox as recorded above.

`git diff --check` passed.

## Artifact Integrity

```text
5ee08b9f56851289124f681afecc5ea45fe4502f7d0e963dca10bcc05aa82dee  checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt
6f651a24891344c074607b02bfdaa553b3c081e41faf6db3275bf38fa645669d  checkpoints/exp2-tuned_params/initial/tiger/instruments/best.pt
```

These match the previously recorded baselines.

## Concerns

The five passing-suite warnings are pre-existing `torch.load(weights_only=False)`
future warnings and the AMP `GradScaler` deprecation warning. No new functional
concern remains from this fix wave.
