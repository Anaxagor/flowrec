# Checkpoints

```text
checkpoints/
  exp1/<model>/<dataset>/best.pt
  exp2/
    loo_split/
      initial/<model>/<dataset>/best.pt
      sft-finetuned/<model>/<dataset>/best.pt
    global_split/
      initial/<model>/<dataset>/best.pt
```

`exp1` contains the primary models with standard parameters. `exp2` separates
the data protocols: `loo_split` is leave-one-out and `global_split` is a global
time split. `initial` contains the original tuned models, and `sft-finetuned`
contains their first SFT-only fine-tuning. Subsequent SFT from exp8 is excluded.

For **GPT-2 Yambda**, the dataset directory contains a data-regime directory:

| Split / stage | Dataset | Regime directory | Primary model |
| --- | --- | --- | --- |
| loo_split / initial | Yambda50M | `last_128` | trial36 |
| loo_split / initial | Yambda500M | `last_128` | trial4 |
| loo_split / initial | both Yambda datasets | `full_history` | legacy bases |
| loo_split / sft-finetuned | Yambda50M | `last_128` | first SFT from trial36, seed42 |
| loo_split / sft-finetuned | Yambda500M | `sampled_50` | first SFT from trial4, seed42 |
| global_split / initial | Yambda50M | `full_history` | selected global base |

For example: `exp2/loo_split/initial/gpt2/yambda_50m_listens/last_128/best.pt`.
The global GPT-2 Yambda50M regime follows the saved `study_config.json`
(`train_target_policy=all`); it is not last-128. GPT-2 sampled-15 weights are
not available locally.

Each leaf directory contains one primary `best.pt` and its metadata. The
source, actual seed (when known), regime, and SHA-256 are recorded in
`provenance.json`. References to individual seeds load the selected primary model.

For TIGER LOO, the selected first SFT is `matched` for Yambda50M and
`matched-r15` for Yambda500M, both with seed42 and lr=3e-05. The initial TIGER
Yambda500M base was saved after training to convergence. The first TIGER
Beauty2023 SFT contains `best.pt` and `best.pt.json`, seed42. Continuing that
Beauty2023 model in exp8 also requires the complete original protocol and run
results: they cannot be inferred from the weights alone.

`global_split` contains 13 confirmed primary models. No primary global-split
checkpoint has been selected for GPT-2 Yambda500M or TIGER Beauty2023. Global
SFT-only weights are not available locally.

Scripts load primary input weights only from this structure. If a model is
missing, they report the current path. Per-seed snapshots, `last.pt`, trials,
and training logs are stored in `experiments/`; primary model publication
preserves the structure above.

`configs/tiger_beauty_grpo.json` and
`configs/tiger_beauty_gflow_mix_curriculum_example.json` are templates: fill in
the `required_checkpoints` fields with paths to selected primary models.
`run_tiger.py` checks them before preparing data and creating a run. The
template's architecture, data settings, and semantic IDs must match the chosen
model; missing DPO/SASRec weights are not automatically substituted.

Run logs are stored in `experiments/`. Paths in saved protocols identify runs
and are not used to load weights.

The semantic-ID vocabulary must match the model's training and be stored in
`datasets/semantic_ids/`. Binary files of 100 MiB or more remain outside Git;
metadata is kept in Git.
