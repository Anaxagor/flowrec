# Fine-tuning batch

> **Availability:** Slurm launchers below are local cluster files and are not distributed with this repository. Their examples require separately supplied launchers. The available Python entry points include [`gflow_finetune_tiger_matched.py`](gflow_finetune_tiger_matched.py) and [`gflow_finetune_gpt2_amazon_fullhist.py`](gflow_finetune_gpt2_amazon_fullhist.py). For options, run the TIGER entry point with `--dataset beauty --help` or the GPT-2 entry point with `--help`.

From the repository root on the cluster:

```bash
DRY_RUN=1 GFLOW_MODE=teacher bash scripts/slurm/submit_finetune_batch.sh
GFLOW_MODE=teacher bash scripts/slurm/submit_finetune_batch.sh
```

The default `GFLOW_MODE=teacher` runs nine experiment groups:

| Model | Dataset | Arm | Training examples |
| --- | --- | --- | --- |
| TIGER | Yambda 50M | SFT-only | Base prefix layout, last 128 targets |
| TIGER | Yambda 500M | SFT-only | Base prefix layout, last 128 targets |
| TIGER | Yambda 500M | SFT-only | Fixed random 15 from the last 128 |
| TIGER | Yambda 500M | GFlow ablations | Same random 15 from the last 128 |
| GPT-2 | Beauty, Instruments | GFlow ablations | All prefixes expanded from base dense windows |
| GPT-2 | Beauty, Instruments | SFT-only | Same dense-derived prefixes |
| GPT-2 | Yambda 50M | SFT-only | Dense-derived prefixes, last 128 targets |
| GPT-2 | Yambda 50M | SFT-only | Dense-derived prefixes, all eligible targets |
| GPT-2 | Yambda 50M | GFlow ablations | Dense-derived prefixes, last 128 targets |

TIGER's full base layout on Yambda means last 128 targets/user, with maximum
context 50. Its 500M sampled control and GFlow use subset seed 42, independently
of the run seed. `--train-random-n 0` selects its base-layout control, retaining
the checkpoint's data configuration and disabling only the target sample.

All listed GPT-2 runs use the [dense-derived prefix pipeline](GPT2_DENSE_PREFIX.md).
Each context starts at the original dense window boundary. Window/stride are
20/20 for Amazon and 50/50 for Yambda; all-target training still has a maximum
context of 50. Catalog and complete validation/test examples are retained from
the full dense reference before target selection. The layout audit is written
to `dense_prefix_audit.json` beside each dataset's resolved data configuration.

The standard GFlow grid comes from
`configs/exp2-tuned-params/finetune_base_matched.json:gflow_grid`:

| Dimension | Values |
| --- | --- |
| Reward | All seven nonempty products of SASRec, interaction, hierarchical similarity |
| Total reward exponent | 0.25, 1, 4, 16 |
| GFlow loss | TB, DB, SubTB |
| GFlow/SFT gradient-norm ratio rho | 1, 0.1 |

That is **168 configurations per model/dataset**, hence **672 GFlow runs**:
TIGER 500M and GPT-2 Beauty, Instruments, 50M. Every cell uses SASRec curriculum
negatives, K=10, seed 42 and component weights 1. The coefficient in
`L_SFT + lambda * L_GFlow` is calibrated once to the chosen rho; rho is not a
literal lambda. All seven SFT dataset/selection arms use seeds **42/43/44**,
for **21 SFT runs**.

Reward components/factors are cached independently of the total exponent.
At lookup, each run composes the components and applies its power, then sums
the resulting rewards over the **full catalog** for Z. DB/SubTB sum those same
powered rewards under each semantic-ID prefix. The `reward_cache_power=1`
method setting prevents applying the exponent twice; the composed cache owns
the run's exponent. Calibration uses this same cache.

The reward floor applies before and after exponentiation:
`R_p = max(max(product of components, 1e-6)^p, 1e-6)` for the standard unit
component weights. Thus at p=16, composed rewards below approximately 0.422
become 1e-6. Z and flows are consistent with this floored reward. Regression
tests in `tests/test_reward_power_flows.py` check all seven products and four
powers through both GPT-2 cache formats and the TIGER factorized path,
including the full-catalog normalizer and all three balance losses.

With `BLOCK_SIZE=4`, this is **12 arrays / 201 tasks**: 12 cache tasks,
168 GFlow tasks and 21 SFT tasks. Each GFlow task runs up to four configurations
sequentially. `MAX_PARALLEL=20` limits concurrency per array; `SEEDS`,
`BLOCK_SIZE` and `MAX_PARALLEL` can be overridden. `BLOCK_SIZE=1` would submit
705 tasks, exceeding the configured 500-submitted-job limit.

The first three arrays force recalculation of reward caches/factors. Every
fine-tuning array then waits for **all three** using `afterok`; training reuses
those caches. Yambda rewards are stored as factors. SFT-only needs no rewards
but follows the same cache-first launch order.

Beauty, Instruments and Yambda 50M explicitly allow **all GPU node types**:
`--constraint='type_a|type_b|type_c|type_e|type_f|type_h'`. CPU-only type D is
excluded, and the explicit list avoids relying on the site's default types.
Every Yambda 500M task has `--constraint='type_e|type_f|type_h'`, including
cache tasks and both controls. No node IDs or `sinfo` inventory are queried.

The type flags are generated by the optional local cluster resource planner,
which is not distributed with the repository.

The standard GFlow array names are:

- `flowrec_gpt2_amazon_grid` — Beauty and Instruments, 84 tasks.
- `flowrec_tiger_gflow_yambda_yambda_500m_listens_grid` — 500M, 42 tasks.
- `flowrec_gpt2_yambda_yambda_50m_listens_grid` — 50M, 42 tasks.

Use `squeue -r -u "$USER"` to display individual array tasks. `Dependency` is
expected until cache calculations succeed.

Controls with different target selections have separate runs, resume state,
metrics and, for TIGER, promoted checkpoints:

| Control | Run root |
| --- | --- |
| TIGER 500M base last-128 | `experiments/exp2-sft-finetune-matched` |
| TIGER 500M random-15 | `experiments/exp2-sft-finetune-matched-r15` |
| GPT-2 50M last-128 prefixes | `experiments/exp4-gflow-gpt2-yambda-dense-prefix` |
| GPT-2 50M all-target prefixes | `experiments/exp4-gflow-gpt2-yambda-dense-prefix-all-targets` |

GPT-2 all-target CSV names receive `_all_targets`; its flag is
`--sft-only --yambda50-all-targets` with `--datasets yambda_50m_listens`.
The flag applies its suffix to custom roots too and rejects additional target
subsampling. It is restricted to this SFT diagnostic (or its data audit).

The two controls can also be submitted alone:

```bash
PLACEMENT=types \
RECALC_GROUPS="tiger_control_yambda500_last128 gpt2_yambda50_all_targets" \
bash scripts/slurm/submit_base_matched_recalc.sh
```

Teacher-free remains a separate opt-in experiment:

```bash
GFLOW_MODE=teacherfree bash scripts/slurm/submit_finetune_batch.sh
```

It substitutes the separate random/policy sampler grid and submits 369 tasks
in 12 arrays with the same SFT controls and placement rules.

Completed runs are reused and interrupted runs resume in their respective
protocol directories.
