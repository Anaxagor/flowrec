# Experiment 2: tuned baselines and SFT-only controls

This experiment tunes model hyperparameters and trains the selected SASRec,
TIGER and GPT-2 configurations. Its initial checkpoints are the starting point
for comparisons with first-stage SFT-only fine-tuning and the GFlow ablations in
[experiment 4](../exp4-reward-ablations/README.md).

The default pipeline uses leave-one-out data splits. The separate
[global-time variant](../global-time/README.md) trains and evaluates on a global
temporal split. Use each split's own configurations and checkpoints: the
catalogs, evaluation examples and training settings may differ.

## Entry points

| Purpose | Scripts |
| --- | --- |
| Hyperparameter search | `tuning_gpt2.py`, `tuning_sasrec.py`, `tuning_sasrec_sce.py`, `tuning_tiger_yambda_500m.py` |
| Train selected configurations | `run_gpt2_tuned_all.py`, `run_sasrec_tuned_all.py`, `run_sasrec_sce_tuned_all.py` |
| Evaluate SASRec scalable cross-entropy | `evaluate_sasrec_sce_tuned.py` |
| First SFT-only control | `sft_finetune_control.py`, `sft_finetune_control_yambda.py`, `sft_finetune_control_matched.py` |

For example, run `python scripts/exp2-tuned-params/tuning_gpt2.py --help` from
the repository root for dataset and protocol options. GPT-2 tuning supports a
Yambda protocol matched to the TIGER data layout through `--match-tiger`;
`--describe-protocol` displays its configuration without training.

Study trials and per-run snapshots are stored under `experiments/`, with
summaries and other artifacts under `artifacts/`. Selected models are stored in
`checkpoints/exp2/<split>/initial/` and the first SFT-only models in
`checkpoints/exp2/<split>/sft-finetuned/` where available. GPT-2 Yambda paths
also identify the data regime. For selected models and missing assets, see the
[checkpoint layout](../../checkpoints/README.md).

The SFT-only control continues an initial model using supervised loss. The
matched control and the GFlow arms share the relevant comparison protocol from
`configs/exp2-tuned-params/finetune_base_matched.json`. They are distinct from
[experiment 8](../exp8-from-sft/README.md), which starts a further training stage
from an already SFT-finetuned checkpoint.
