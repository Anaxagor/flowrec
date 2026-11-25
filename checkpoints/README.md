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

| Split / stage | Dataset | Regime directory | Primary model | Storage |
| --- | --- | --- | --- | --- |
| loo_split / initial | Yambda50M | `last_128` | trial36 | Google Drive |
| loo_split / initial | Yambda500M | `last_128` | trial4 | Repository |
| loo_split / initial | Yambda50M | `full_history` | legacy base | Not uploaded |
| loo_split / initial | Yambda500M | `full_history` | legacy base | Not uploaded |
| loo_split / sft-finetuned | Yambda50M | `last_128` | first SFT from trial36, seed42 | Google Drive |
| loo_split / sft-finetuned | Yambda500M | `sampled_50` | first SFT from trial4, seed42 | Repository |
| global_split / initial | Yambda50M | `full_history` | legacy all-target study, trial10 | Not uploaded |
| global_split / initial | Yambda50M | `last_128` | matched base, seed42 | Google Drive |
| global_split / initial | Yambda500M | `last_128` | No selected base | — |

For example: `exp2/loo_split/initial/gpt2/yambda_50m_listens/last_128/best.pt`.
The earlier global GPT-2 Yambda50M study recorded `train_target_policy=all`.
Its weights remain available through the explicit `full_history` path.
Comparisons with TIGER use `last_128` on both Yambda sizes.
GPT-2 sampled-15 weights are not available locally.

Each selected model has one primary `best.pt`; some weights are stored outside
the repository as shown below. The source, actual seed (when known), regime, and
SHA-256 are recorded in `provenance.json`. References to individual seeds load
the selected primary model.

For TIGER LOO, the selected first SFT is `matched` for Yambda50M and
`matched-r15` for Yambda500M, both with seed42 and lr=3e-05. The initial TIGER
Yambda500M base was saved after training to convergence. The first TIGER
Beauty2023 SFT contains `best.pt` and `best.pt.json`, seed42. Continuing that
Beauty2023 model in exp8 also requires the complete original protocol and run
results: they cannot be inferred from the weights alone.

`global_split` has 13 of the 15 primary models for the comparison, plus
the historical full-history GPT-2 Yambda50M model. GPT-2 Yambda500M on
`last_128` and TIGER Beauty2023 have trial checkpoints only. The matched
GPT-2 Yambda50M checkpoint exceeds the Git file-size limit. Its SHA-256,
architecture and data protocol are in the
[checkpoint record](../scripts/global-time/gpt2_yambda50m_last128_checkpoint.json).
Global SFT-only weights are not available locally.

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
`datasets/semantic_ids/`. Files of 100 MiB or more are excluded from Git;
metadata, including `provenance.json`, is kept in the repository.

## Checkpoint availability

Large weights are stored in the [Google Drive checkpoint folder](https://drive.google.com/drive/folders/1uzOTQ5CP67vl1d3M0nCAocIcM0yJXwHL?usp=sharing) when available.
The tables list the selected primary checkpoints (`best.pt`) and two global-split trial checkpoints.

- **Repository**: the weights are tracked in Git.
- **Google Drive**: a file with the corresponding path and size was found in the linked folder.
- **Not uploaded**: the weights are excluded from Git and have not been uploaded
  to the linked folder because they are not currently needed for experiments.

The Drive folder uses `exp2-tuned_params/` where the repository uses `exp2/`.
After downloading, place each file at its repository path below. Where a checkpoint
record provides a SHA-256, use it to verify the downloaded file.

### Standard parameters

Paths are relative to `exp1/`.

| Model / dataset | Storage |
| --- | --- |
| [`gpt2/beauty/best.pt`](exp1/gpt2/beauty/best.pt) | Repository |
| [`gpt2/beauty_2023/best.pt`](exp1/gpt2/beauty_2023/best.pt) | Repository |
| [`gpt2/instruments/best.pt`](exp1/gpt2/instruments/best.pt) | Repository |
| `gpt2/yambda_500m_listens/best.pt` | Not uploaded |
| `gpt2/yambda_50m_listens/best.pt` | Not uploaded |
| [`sasrec/beauty/best.pt`](exp1/sasrec/beauty/best.pt) | Repository |
| `sasrec/beauty_2023/best.pt` | Not uploaded |
| [`sasrec/instruments/best.pt`](exp1/sasrec/instruments/best.pt) | Repository |
| `sasrec/yambda_500m_listens/best.pt` | Not uploaded |
| `sasrec/yambda_50m_listens/best.pt` | Not uploaded |
| [`tiger/beauty/best.pt`](exp1/tiger/beauty/best.pt) | Repository |
| [`tiger/beauty_2023/best.pt`](exp1/tiger/beauty_2023/best.pt) | Repository |
| [`tiger/instruments/best.pt`](exp1/tiger/instruments/best.pt) | Repository |
| [`tiger/yambda_500m_listens/best.pt`](exp1/tiger/yambda_500m_listens/best.pt) | Repository |
| [`tiger/yambda_50m_listens/best.pt`](exp1/tiger/yambda_50m_listens/best.pt) | Repository |

### LOO split: tuned initial models

Paths are relative to `exp2/loo_split/initial/`.

| Model / dataset | Storage |
| --- | --- |
| [`gpt2/beauty/best.pt`](exp2/loo_split/initial/gpt2/beauty/best.pt) | Repository |
| `gpt2/beauty_2023/best.pt` | Google Drive |
| [`gpt2/instruments/best.pt`](exp2/loo_split/initial/gpt2/instruments/best.pt) | Repository |
| `gpt2/yambda_500m_listens/full_history/best.pt` | Not uploaded |
| [`gpt2/yambda_500m_listens/last_128/best.pt`](exp2/loo_split/initial/gpt2/yambda_500m_listens/last_128/best.pt) | Repository |
| `gpt2/yambda_50m_listens/full_history/best.pt` | Not uploaded |
| `gpt2/yambda_50m_listens/last_128/best.pt` | Google Drive |
| [`sasrec/beauty/best.pt`](exp2/loo_split/initial/sasrec/beauty/best.pt) | Repository |
| `sasrec/beauty_2023/best.pt` | Google Drive |
| [`sasrec/instruments/best.pt`](exp2/loo_split/initial/sasrec/instruments/best.pt) | Repository |
| `sasrec/yambda_50m_listens/best.pt` | Google Drive |
| `sasrec_sce/beauty_2023/best.pt` | Not uploaded |
| `sasrec_sce/yambda_500m_listens/best.pt` | Google Drive |
| `sasrec_sce/yambda_50m_listens/best.pt` | Not uploaded |
| [`tiger/beauty/best.pt`](exp2/loo_split/initial/tiger/beauty/best.pt) | Repository |
| [`tiger/beauty_2023/best.pt`](exp2/loo_split/initial/tiger/beauty_2023/best.pt) | Repository |
| [`tiger/instruments/best.pt`](exp2/loo_split/initial/tiger/instruments/best.pt) | Repository |
| [`tiger/yambda_500m_listens/best.pt`](exp2/loo_split/initial/tiger/yambda_500m_listens/best.pt) | Repository |
| [`tiger/yambda_50m_listens/best.pt`](exp2/loo_split/initial/tiger/yambda_50m_listens/best.pt) | Repository |

### LOO split: first SFT-only fine-tuning

Paths are relative to `exp2/loo_split/sft-finetuned/`.

| Model / dataset | Storage |
| --- | --- |
| [`gpt2/beauty/best.pt`](exp2/loo_split/sft-finetuned/gpt2/beauty/best.pt) | Repository |
| `gpt2/beauty_2023/best.pt` | Google Drive |
| [`gpt2/instruments/best.pt`](exp2/loo_split/sft-finetuned/gpt2/instruments/best.pt) | Repository |
| [`gpt2/yambda_500m_listens/sampled_50/best.pt`](exp2/loo_split/sft-finetuned/gpt2/yambda_500m_listens/sampled_50/best.pt) | Repository |
| `gpt2/yambda_50m_listens/last_128/best.pt` | Google Drive |
| [`tiger/beauty/best.pt`](exp2/loo_split/sft-finetuned/tiger/beauty/best.pt) | Repository |
| [`tiger/beauty_2023/best.pt`](exp2/loo_split/sft-finetuned/tiger/beauty_2023/best.pt) | Repository |
| [`tiger/instruments/best.pt`](exp2/loo_split/sft-finetuned/tiger/instruments/best.pt) | Repository |
| [`tiger/yambda_500m_listens/best.pt`](exp2/loo_split/sft-finetuned/tiger/yambda_500m_listens/best.pt) | Repository |
| [`tiger/yambda_50m_listens/best.pt`](exp2/loo_split/sft-finetuned/tiger/yambda_50m_listens/best.pt) | Repository |

### Global split: tuned initial models

Repository paths are relative to `exp2/global_split/initial/`. The Drive paths
shown for GPT-2 Yambda and TIGER Beauty2023 are relative to
`exp2-tuned_params/global_split/initial/` in the linked folder.

| Model / dataset | Storage |
| --- | --- |
| [`gpt2/beauty/best.pt`](exp2/global_split/initial/gpt2/beauty/best.pt) | Repository |
| [`gpt2/beauty_2023/best.pt`](exp2/global_split/initial/gpt2/beauty_2023/best.pt) | Repository |
| [`gpt2/instruments/best.pt`](exp2/global_split/initial/gpt2/instruments/best.pt) | Repository |
| `gpt2/yambda_50m_listens/full_history/best.pt` (legacy) | Not uploaded |
| `gpt2/yambda_50m_listens/last_128/best.pt` | Google Drive (`gpt2/yambda50m/best.pt`) |
| `gpt2/yambda_500m_listens/last_128/trial_17.pt` (trial) | Google Drive (`gpt2/yambda500m/trial_17.pt`) |
| [`sasrec/beauty/best.pt`](exp2/global_split/initial/sasrec/beauty/best.pt) | Repository |
| `sasrec/beauty_2023/best.pt` | Google Drive |
| [`sasrec/instruments/best.pt`](exp2/global_split/initial/sasrec/instruments/best.pt) | Repository |
| `sasrec/yambda_50m_listens/best.pt` | Google Drive |
| `sasrec_sce/yambda_500m_listens/best.pt` | Google Drive |
| `tiger/beauty/best.pt` | Google Drive |
| `tiger/beauty_2023/trial_31.pt` (trial) | Google Drive (`tiger/beauty2023/trial_31.pt`) |
| [`tiger/instruments/best.pt`](exp2/global_split/initial/tiger/instruments/best.pt) | Repository |
| [`tiger/yambda_500m_listens/best.pt`](exp2/global_split/initial/tiger/yambda_500m_listens/best.pt) | Repository |
| [`tiger/yambda_50m_listens/best.pt`](exp2/global_split/initial/tiger/yambda_50m_listens/best.pt) | Repository |
