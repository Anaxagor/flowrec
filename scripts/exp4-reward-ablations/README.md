# Experiment 4: reward ablations and GFlow fine-tuning

This experiment compares continued supervised fine-tuning with GFlowNet
fine-tuning of TIGER and GPT-2 initial models from
[experiment 2](../exp2-tuned-params/README.md). It varies reward components and
their compositions, reward powers, GFlow objectives, and gradient calibration.
Depending on the entry point, candidates are obtained with a SASRec teacher or
with teacher-free random or policy sampling. The settings and data protocol
for a given run are specified by its configuration and command-line options.

The main matched control is described by
`configs/exp2-tuned-params/finetune_base_matched.json`. In the current
leave-one-out configuration the standard teacher-based grid has seven reward
compositions and 168 cells per model/dataset; it should not be assumed identical
to the global-time grid. Results should only be compared when the split,
training targets, initialization and evaluation protocol agree.

## Entry points and protocols

| Purpose | Scripts or guide |
| --- | --- |
| TIGER matched and teacher-free runs | `gflow_finetune_tiger_matched.py`, `gflow_finetune_tiger_teacherfree.py` |
| GPT-2 Amazon matched and teacher-free runs | `gflow_finetune_gpt2_amazon_fullhist.py`, `gflow_finetune_gpt2_amazon_teacherfree.py` |
| GPT-2 Yambda | `gflow_finetune_gpt2_yambda.py`; [Yambda guide](GPT2_YAMBDA.md) |
| Dense-window expansion for GPT-2 | [Dense-prefix protocol](GPT2_DENSE_PREFIX.md) |
| Batch submission | [Fine-tuning batch guide](FINETUNE_BATCH.md) |
| Start GFlow from a first SFT-only checkpoint | `gflow_finetune_tiger_from_sft.py`; [experiment 8](../exp8-from-sft/README.md) |

Some earlier experiment-specific scripts remain for reproducing their original
protocols. In particular, `rewards_ablation_amazon.py` documents that its
training wiring targets an older GFlow API and is not runnable without porting.
For example, run
`python scripts/exp4-reward-ablations/gflow_finetune_gpt2_yambda.py --help`
from the repository root to inspect its options. Reward caches and
per-run snapshots are kept separately from the curated primary models; see the
[checkpoint layout](../../checkpoints/README.md).
