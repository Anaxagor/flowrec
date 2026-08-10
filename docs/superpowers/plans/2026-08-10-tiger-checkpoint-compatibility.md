# TIGER Checkpoint Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the published exp2 Beauty and Musical Instruments TIGER checkpoints strictly loadable and reproducible for training, validation, and beam-20 testing from the `sft_speedup` branch.

**Architecture:** Synchronize `sft_speedup` with current `main`, then port only the working remote TIGER checkpoint contract. Keep `scripts/run_tiger.py` as the single experiment entry point, extend the generic model loader to understand trainer checkpoints, and publish one complete JSON config per dataset.

**Tech Stack:** Python, PyTorch, Hugging Face T5, pytest, JSON experiment configs.

## Global Constraints

- The remote FlowRec TIGER implementation is the behavioral source of truth.
- Preserve all functionality already present on current `main` and `sft_speedup`.
- Keep strict checkpoint loading as the default; never suppress shape mismatches.
- Do not modify checkpoint binaries, semantic-ID dictionaries, RQ-VAE artifacts, or embeddings.
- Use position-specific semantic IDs with `offset_semantic_ids_by_one=true`.
- Use validation beam 1 and final test beam 20.

---

### Task 1: Synchronize The Feature Branch

**Files:**
- Merge: `origin/main` into `sft_speedup`
- Preserve: `docs/superpowers/specs/2026-08-10-tiger-checkpoint-compatibility-design.md`
- Preserve: `docs/superpowers/plans/2026-08-10-tiger-checkpoint-compatibility.md`

**Interfaces:**
- Consumes: current published checkpoints and SID tables from `origin/main`
- Produces: a conflict-resolved `sft_speedup` tree containing both current main and SFT speedups

- [ ] **Step 1: Fetch and merge current main**

```bash
git fetch origin --prune
git merge --no-edit origin/main
```

- [ ] **Step 2: Inspect every conflict against both the remote implementation and existing tests**

```bash
git status --short
git diff --name-only --diff-filter=U
```

- [ ] **Step 3: Run the pre-change SFT tests**

```bash
pytest -q tests/test_tiger_sft_speedups.py
```

Expected: all existing SFT speedup tests pass before checkpoint work begins.

---

### Task 2: Support Trainer Checkpoints In The Generic Loader

**Files:**
- Modify: `flowrec/models/base.py`
- Test: `tests/test_tiger_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: payloads containing `model_state_dict`, `state_dict`, or a raw state dictionary
- Produces: `FlowRecModel.load_checkpoint(path, strict=True, map_location="cpu")`

- [ ] **Step 1: Write the failing loader test**

```python
def test_generic_loader_accepts_trainer_model_state_dict(tmp_path: Path) -> None:
    source = _TinyCheckpointModel()
    path = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": source.state_dict(), "trainer_state": {}}, path)
    target = _TinyCheckpointModel()
    target.weight.data.zero_()

    payload = target.load_checkpoint(path)

    assert target.weight.item() == pytest.approx(source.weight.item())
    assert "trainer_state" in payload
```

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py::test_generic_loader_accepts_trainer_model_state_dict
```

Expected: failure because the generic loader passes the entire trainer payload to `load_state_dict()`.

- [ ] **Step 3: Implement the same payload precedence as remote `scripts/run_tiger.py`**

```python
if isinstance(payload, Mapping):
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
else:
    state_dict = payload
```

- [ ] **Step 4: Verify GREEN and existing portable-checkpoint behavior**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py
```

---

### Task 3: Port The Exact TIGER Architecture Contract

**Files:**
- Modify: `flowrec/models/tiger.py`
- Modify: `scripts/run_tiger.py`
- Test: `tests/test_tiger_checkpoint_compatibility.py`
- Test: `tests/test_tiger_sft_speedups.py`

**Interfaces:**
- Consumes: `d_kv`, `use_user_token`, `enable_flow_head`, and `detach_flow_head_trunk`
- Produces: a model whose `config_dict()` is sufficient to reconstruct all shape-critical tensors

- [ ] **Step 1: Write failing configuration and flow-head tests**

```python
def test_tiger_config_preserves_checkpoint_critical_fields() -> None:
    model = _checkpoint_compatible_tiger()
    config = model.config_dict()
    assert config["d_kv"] == 32
    assert config["use_user_token"] is False
    assert config["enable_flow_head"] is True
    assert config["detach_flow_head_trunk"] is False
    assert "flow_head.1.weight" in model.state_dict()
```

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py::test_tiger_config_preserves_checkpoint_critical_fields
```

- [ ] **Step 3: Port the remote flow-head construction and load behavior**

Port the optional flow-head fields, module construction, initialization, and checkpoint compatibility logic from remote `flowrec/models/tiger.py`. Preserve the existing `sft_speedup` user-token and popularity behavior.

- [ ] **Step 4: Store and serialize constructor-critical values**

Store `d_kv`, `layer_norm_eps`, `activation`, and `initializer_range` on `TigerModel`; return them with user-token and flow-head settings from `config_dict()`.

- [ ] **Step 5: Ensure the runner forwards every field**

`scripts/run_tiger.py::build_model()` must forward `d_kv`, `use_user_token`, `enable_flow_head`, and `detach_flow_head_trunk` from the model config.

- [ ] **Step 6: Verify GREEN**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py
```

---

### Task 4: Add Complete Exp2 TIGER Configs

**Files:**
- Create: `configs/exp2-tuned-params/tiger_beauty.json`
- Create: `configs/exp2-tuned-params/tiger_instruments.json`
- Test: `tests/test_tiger_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: `scripts/run_tiger.py`
- Produces: standalone training/resume/validation/test configs for both published checkpoints

- [ ] **Step 1: Write failing config-contract tests**

```python
@pytest.mark.parametrize("dataset", ["beauty", "instruments"])
def test_exp2_tiger_config_is_complete(dataset: str) -> None:
    cfg = json.loads((PROJECT_ROOT / "configs/exp2-tuned-params" / f"tiger_{dataset}.json").read_text())
    assert cfg["model"] == {
        **cfg["model"],
        "embedding_dim": 128,
        "num_heads": 6,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
        "d_kv": 32,
        "use_user_token": False,
        "user_ids_count": 0,
        "enable_flow_head": True,
    }
    assert cfg["dataset"]["offset_semantic_ids_by_one"] is True
    assert cfg["method"]["generation_num_beams"] == 1
    assert cfg["evaluation"]["test_generation_num_beams"] == 20
    assert cfg["training"]["save_best"] is True
```

- [ ] **Step 2: Verify RED because the configs do not exist**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py::test_exp2_tiger_config_is_complete
```

- [ ] **Step 3: Build both configs from the actual training contract**

Use repository-relative checkpoint and SID paths. Include the Amazon parquet inputs, verified-purchase policy, position-specific SID offset, independent train/validation/test batches, complete optimizer and trainer settings, SFT generation settings, initialization checkpoint, and final beam-20 evaluation settings. Setting `training.init_checkpoint_path` to `null` must be the only change required for training from scratch.

- [ ] **Step 4: Parse configs through the real runner builders**

Test `build_data_config()` and `build_model()` with lightweight fixture data so config field names cannot drift from the runner.

- [ ] **Step 5: Verify GREEN**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py -k exp2_tiger_config
```

---

### Task 5: Regress The Published Checkpoints

**Files:**
- Test: `tests/test_tiger_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: published `best.pt` files and exp2 SID tables
- Produces: evidence that the repository artifacts load strictly and respect the position-token contract

- [ ] **Step 1: Write the published-artifact regression test**

For each dataset, load the canonical config and SID JSON, apply the configured `+1` offset, construct TIGER, and call the generic loader with `strict=True`. Assert:

```python
assert torch.equal(model.item_semantic_ids, expected_shifted_sid_tensor)
assert model.config.d_kv == 32
assert model.use_user_token is False
```

Generate from a deterministic non-empty synthetic history with valid-sequence constraints and assert:

```python
assert generated[..., 0].min() >= 1
assert generated[..., 0].max() <= 256
```

- [ ] **Step 2: Verify RED before the architecture port is complete**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py -k published_checkpoint
```

- [ ] **Step 3: Verify GREEN for both datasets**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py -k published_checkpoint
```

Expected: two strict loads and two valid first-position generation checks pass.

---

### Task 6: Final Verification And Publication

**Files:**
- Verify all modified files
- Update the existing `sft_speedup` remote branch

**Interfaces:**
- Consumes: completed implementation
- Produces: updated GitHub PR branch with reproducible checkpoint loading

- [ ] **Step 1: Run focused tests**

```bash
pytest -q tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest -q
```

- [ ] **Step 3: Verify repository state and artifact integrity**

```bash
git diff --check
git status --short
sha256sum checkpoints/exp2-tuned_params/initial/tiger/{beauty,instruments}/best.pt
```

- [ ] **Step 4: Commit implementation and push**

```bash
git add flowrec/models/base.py flowrec/models/tiger.py scripts/run_tiger.py \
  configs/exp2-tuned-params/tiger_beauty.json \
  configs/exp2-tuned-params/tiger_instruments.json \
  tests/test_tiger_checkpoint_compatibility.py tests/test_tiger_sft_speedups.py
git commit -m "fix: make exp2 TIGER checkpoints reproducible"
git push origin sft_speedup
```
