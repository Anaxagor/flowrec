# Beauty 2023 semantic IDs for TIGER

These are the exact semantic IDs used by the tuned Beauty 2023 TIGER SFT
checkpoint and the subsequent acyclic GFlowNet ablation. This mapping is
identical to `datasets/semantic_ids/exp1-standard_params/beauty_2023/`.

- 163,050 items; three RQ-VAE codebooks of size 256 with level-offset tokens.
- A fourth collision token makes every item's semantic ID unique.
- `semantic_ids.json` maps item IDs (ASINs) to four-token semantic IDs.
- The tokenizer checkpoint, configuration, metadata, and training logs are
  included for provenance and reproducibility.
- SHA-256 of `semantic_ids.json`:
  `5433d3ffe96629079ad06f46511f34e155cb0d0926cc8d19420e0f409ae843c9`.

Use this mapping with the matching TIGER checkpoint; the sibling mapping
in the parent directory must not be substituted without compatibility checks.
