# Task 4 Report: Complete Exp2 TIGER Configs

## Status

DONE

## RED Evidence

The contract test was added before either config existed and run with:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py::test_exp2_tiger_config_is_complete
```

Exact result:

```text
FF                                                                       [100%]
FAILED tests/test_tiger_checkpoint_compatibility.py::test_exp2_tiger_config_is_complete[beauty-0.5]
FileNotFoundError: [Errno 2] No such file or directory: '.../configs/exp2-tuned-params/tiger_beauty.json'
FAILED tests/test_tiger_checkpoint_compatibility.py::test_exp2_tiger_config_is_complete[instruments-2.0]
FileNotFoundError: [Errno 2] No such file or directory: '.../configs/exp2-tuned-params/tiger_instruments.json'
2 failed in 3.84s
```

This is the intended RED condition: both requested configs were absent.

## Config Choices

- Both configs use the current exp2 Amazon parquet inputs, exp2 semantic-ID tables, and fingerprinted processed-cache paths. The verified-purchase rule is `positive_criterion: "verified_purchase == True"`, preserving the complementary interactions in the datamodule's `interaction_store`.
- The dataset retains the exp2 position-specific semantic-ID offset, `prefix_target` layout, `asin` item key, and independent `2048` train, `1024` validation, and `64` final-test batches.
- Architecture is the published sidecar/fast TIGER shape: four codebooks, `embedding_dim=128`, six heads, two encoder and decoder layers, `dim_feedforward=512`, `d_kv=32`, semantic-only users, and the enabled flow head.
- Validation generation uses one beam. `test_generation_num_beams=20` and the other final generation controls live in `training`, matching `scripts/run_tiger.py`; no unused `evaluation` section was added.
- Each config initializes from its repository-relative exp2 initial `best.pt`. The model has no competing checkpoint field, so setting only `training.init_checkpoint_path` to `null` starts from scratch. Beauty uses `max_grad_norm=0.5`; Instruments uses `2.0`.
- Optimizer values mirror the tuned published run names: Beauty uses `lr=0.0012`, `weight_decay=0.01`; Instruments uses `lr=0.00166`, `weight_decay=0.0`.

## GREEN Evidence

The required config-contract command was run after adding the configs:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py -k exp2_tiger_config
```

Exact result:

```text
....                                                                     [100%]
4 passed, 9 deselected in 3.94s
```

The four passing cases cover both config contracts and both real
`build_data_config()` / `build_model()` parses with a lightweight SID fixture.

The full compatibility module was then run:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py
```

Exact result:

```text
13 passed, 2 warnings in 3.69s
```

The added runner-lifecycle test invokes `scripts.run_tiger.main()` with its
real configuration flow and observes that final test evaluation loads
`fit_result.best_checkpoint_path`, switches SFT to `(evaluate_generation=True,
generation_num_beams=20, generation_num_return_sequences=20,
generation_topk=(5, 10, 20))`, and restores the one-beam validation setting.

Existing speedup coverage was also run:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_sft_speedups.py
```

The sandboxed attempt reached 10 passing tests but could not start PyTorch's
persistent worker (`torch_shm_manager: Operation not permitted`). The same
command with approved shared-memory access produced:

```text
...........                                                              [100%]
11 passed, 1 warning in 4.35s
```

The warnings are pre-existing `torch.load(weights_only=False)` future warnings
and a `torch.cuda.amp.GradScaler` deprecation warning.

## Changed Files

- `configs/exp2-tuned-params/tiger_beauty.json`
- `configs/exp2-tuned-params/tiger_instruments.json`
- `tests/test_tiger_checkpoint_compatibility.py`
- `.superpowers/sdd/2026-08-10-tiger-checkpoint-compatibility/task-4-report.md`

## Self-Review

- Verified JSON syntax for both configs with `python -m json.tool`.
- Verified `git diff --check` with no whitespace errors.
- Confirmed all required standalone top-level sections are present: `data`,
  `dataset`, `dataloader`, `model`, `method`, `training`, and `optimizer`.
- Confirmed there is no duplicate `evaluation` block and all generation fields
  consumed by the runner are in `training`.
- Confirmed no binary checkpoint, SID, or other published artifact was changed.
- Confirmed the original implementation did not alter runner behavior; the
  review fix below adds only config-anchored processed-cache path resolution.

## Review Fix Round 1

### RED Evidence

Before changing the configs or runner, the tightened review-contract tests ran:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py \
  -k 'exp2_tiger_config or init_checkpoint_is_null'
```

Exact result:

```text
FFFF.                                                                    [100%]
4 failed, 1 passed, 9 deselected in 3.23s
```

The failures showed both configs used `metadata_item_key: "parent_asin"`, and
`build_data_config()` forwarded `processed_cache_dir` as a raw relative string.
The tightened config contract also required Beauty's `4096` validation batch
and the exact published optimizer values. The null-initialization test already
passed, confirming that the existing runner correctly skips the initial loader
when `training.init_checkpoint_path` is `null`.

### Changes

- Beauty: `validation_batch_size=4096`,
  `lr=0.0011966748408121942`, `weight_decay=0.01`.
- Instruments: `validation_batch_size=1024`,
  `lr=0.0018511868508356536`, `weight_decay=0.01`.
- Both datasets now use `metadata_item_key: "asin"`.
- `scripts/run_tiger.py` resolves `data.processed_cache_dir` through the
  existing optional-path resolver and the config-derived base, making relative
  cache paths repository/config anchored instead of dependent on the process
  current working directory.
- Contract tests are parameterized with the exact batch and optimizer values,
  assert the resolved cache path, and include the focused null-init runner
  test. The existing SFT cache-forwarding assertion now expects the supported
  resolved `Path` value.

### GREEN Evidence

The tightened focused command then produced:

```text
.....                                                                    [100%]
5 passed, 9 deselected in 2.51s
```

The full compatibility suite produced:

```text
14 passed, 2 warnings in 3.22s
```

The SFT speedup suite, with the required shared-memory permission for its
persistent DataLoader worker, produced:

```text
...........                                                              [100%]
11 passed, 1 warning in 3.34s
```

## Review Fix Round 2

The original null-init regression stopped inside `build_model()`, before
`scripts.run_tiger.main()` reached its `init_checkpoint_path` conditional. It
could therefore not detect an unconditional initialization regression.

The replacement parameterizes both branches, returns a compatible fake model,
captures `load_model_state_dict()`, and stops only in `build_method()`, which
is immediately after the branch. It asserts that `null` makes no load and that
the configured published path makes one strict load with the repository-
anchored checkpoint path.

Focused branch coverage:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py::test_tiger_runner_initializes_only_when_init_checkpoint_is_configured
```

Exact result:

```text
..                                                                       [100%]
2 passed in 3.48s
```

Full compatibility coverage:

```sh
/Users/tsessk/anaconda3/envs/dlenv/bin/python3.11 -m pytest -q \
  tests/test_tiger_checkpoint_compatibility.py
```

Exact result:

```text
15 passed, 2 warnings in 2.56s
```
