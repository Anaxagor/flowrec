# TIGER Checkpoint Compatibility Design

## Goal

Make the published Beauty and Musical Instruments TIGER SFT checkpoints load, train, resume, validate, and generate correctly from the `sft_speedup` branch using explicit, reproducible experiment configs.

## Source Of Truth

The working remote FlowRec implementation is the behavioral reference. Port only the checkpoint-relevant TIGER changes into the local branch, preserving unrelated local functionality.

The published checkpoints require this model and token contract:

- two encoder and two decoder layers;
- embedding dimension 128;
- six attention heads with `d_kv=32`;
- feed-forward dimension 512;
- no user token and `user_ids_count=0`;
- position-specific semantic tokens;
- `offset_semantic_ids_by_one=true`, producing model tokens 1 through 1024;
- optional flow head enabled for strict compatibility with the stored state dictionaries.

## Code Changes

1. Synchronize `sft_speedup` with current `origin/main` before implementation.
2. Port the remote optional-user-token and flow-head model behavior into `flowrec/models/tiger.py`.
3. Include `d_kv`, user-token, and flow-head settings in `TigerModel.config_dict()` so future checkpoints preserve constructor-critical parameters.
4. Make the generic checkpoint loader accept both portable `state_dict` payloads and trainer `model_state_dict` payloads.
5. Keep strict loading as the default. Do not silently discard mismatched tensors.
6. Ensure the TIGER runner forwards `d_kv`, `use_user_token`, and flow-head settings from experiment configs.

## Canonical Configs

Add one complete Beauty and one complete Musical Instruments TIGER experiment config under `configs/exp2-tuned-params/`. Each config contains the exact data, model, optimizer, SFT objective, dataloader, and trainer settings required to reproduce training. It also references the corresponding published checkpoint and semantic-ID table with repository-relative paths so the same config can resume or fine-tune the checkpoint, run validation, and perform final beam-20 testing.

Each config explicitly sets all architecture-critical parameters, `offset_semantic_ids_by_one=true`, best-checkpoint selection, independent validation and test batch sizes, one-beam validation, valid-sequence constraints, and beam-20 test generation. Removing the checkpoint path from the config starts SFT training from scratch without changing the model or data contract.

The configs use the existing Amazon data pipeline and do not introduce a second loader or preprocessing path.

## Validation

Add focused regression tests that:

- prove trainer checkpoints load through the generic loader;
- prove `config_dict()` retains `d_kv`, user-token, and flow-head settings;
- build training, validation, and test loaders from both canonical configs;
- strictly load both published checkpoints with their canonical configs;
- verify the checkpoint SID buffer equals the published SID table after the configured `+1` offset;
- verify generated first-position tokens are in the valid level-0 range.

The final verification runs the focused tests and the existing TIGER SFT speedup test module. No checkpoint binaries or semantic-ID artifacts are modified by this change.
