# GPT-2 fine-tuning inputs

> **Availability:** Slurm launchers mentioned below are local cluster files and are not distributed with this repository. Their examples require separately supplied launchers. The available Python entry points are [`gflow_finetune_gpt2_amazon_fullhist.py`](gflow_finetune_gpt2_amazon_fullhist.py) and [`gflow_finetune_gpt2_yambda.py`](gflow_finetune_gpt2_yambda.py); use `--help` for options.

All exp4 GPT-2 prefix entrypoints expand the base checkpoint's dense-causal
windows. This applies to SFT-only, standard GFlow and the separate teacher-free
experiment. Preprocessing comes from the corresponding exp2 dataset config.

For a dense window with inputs `[A, B, C]` and active labels `[B, C, D]`,
the prefix examples are `[A] -> B`, `[A, B] -> C`, `[A, B, C] -> D`.
The next window starts a new context. No prefix reaches across its original
window boundary. Window length and stride default to `max_history_len`,
as in the exp2 GPT-2 training runner.

| Dataset | Window / stride | Training targets per user |
| --- | --- | --- |
| Beauty, Instruments, Beauty 2023 | 20 / 20 | All active dense labels |
| Yambda 50M | 50 / 50 | Last 128 eligible targets |
| Yambda 500M | 50 / 50 | Up to 15 sampled from the last 128, subset seed 42 |

Selection never reanchors a window. The random subset is fixed across SFT-only,
GFlow configurations and run seeds. Users with fewer eligible targets retain
all of them. The existing subset selector is also used by reward precomputation.

The full dense training split defines the catalog **before** target selection.
Validation/test retain the complete dense-reference examples: users, targets,
histories, timestamps, item IDs and semantic IDs. Each data build verifies
ordered held-out hashes and writes `dense_prefix_audit.json`; that report also
records the context settings and final training target count. This aligns
fine-tuning with GPT-2's base checkpoint, without changing TIGER's population.

Audit data without loading models or computing rewards:

```bash
python scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py \
  --datasets beauty instruments beauty_2023 yambda_50m_listens yambda_500m_listens \
  --audit-data-only
```

Use an allocation sized for the selected datasets. This audit still loads and
preprocesses the data. The Yambda convenience script supports the same action.

The matched prefix output root is
`experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix`; its shared cache root
is `artifacts/exp4-reward_ablation/gflow_gpt2_amazon_fullhist_dense_prefix`.
Teacher-free has its own `...-teacherfree-dense-prefix` output root and shares
these caches. Metrics CSVs also gain a `dense_prefix` suffix. The already
aligned Yambda convenience runner retains its existing separate directories.

Reward manifests record the dense-window context and reject caches with a
different context even when row counts and target identities match. Resume and
completion checks require the matching data protocol. Beauty 2023 and both
Yambda datasets store reward factors only.

To submit the cluster batch:

```bash
DRY_RUN=1 bash scripts/slurm/submit_finetune_batch.sh
bash scripts/slurm/submit_finetune_batch.sh
```

It recalculates caches before the dependent fine-tuning arrays. See
[the batch instructions](FINETUNE_BATCH.md) for the grid and coverage.

The dense SFT diagnostics include both full-history and last-128 Yambda 50M
variants. They train multiple targets per forward pass. Their prefixes and
targets match the prefix arm when target selection matches; batch grouping and
optimization can differ.

Validation: independent Amazon and Yambda datamodules are checked for identical
held-out frames, catalogs and collated tensors. Tests cover exact dense-label
expansion, equal GPT-2 loss/gradients with dropout disabled, processed-cache
reloads, both reward-cache formats, and Yambda target selection. An audit of Beauty and Instruments found 131,413 / 116,989 training targets and
22,363 / 22,636 examples in each held-out split, respectively.

Yambda 50M also has an all-target **prefix** SFT-only control:

```bash
python scripts/exp4-reward-ablations/gflow_finetune_gpt2_yambda.py \
  --datasets yambda_50m_listens --sft-only --yambda50-all-targets
```

This expands every base dense label while preserving the original 50-item
window boundaries, catalog and held-out examples. It appends `-all-targets`
to run/cache roots and `_all_targets` to the metrics CSV, preventing reuse of
the last-128 control. It is included in the cluster batch.
