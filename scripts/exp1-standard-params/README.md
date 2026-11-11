# Experiment 1: standard-parameter baselines

This experiment trains and evaluates the standard-parameter SASRec, TIGER and
GPT-2 recommenders on the configured datasets. These runs provide the reference
for the tuned models in [experiment 2](../exp2-tuned-params/README.md).
TIGER and GPT-2 use semantic IDs produced from item representations; SASRec uses
item IDs directly. The scripts cover Amazon and Yambda datasets, subject
to the available input data and configuration for each model.

The default configuration under `configs/exp1-standard_params/` uses the
leave-one-out protocol for the Amazon and Yambda datasets.

## Workflow

Run commands from the repository root, in an environment with the project
dependencies installed:

1. Prepare content embeddings with `build_amazon_embeddings.py` where required.
   Yambda uses its configured input embeddings.
2. Train the tokenizer and export semantic IDs with `train_tokenizer_all.py`.
3. Train the models with `run_sasrec_all.py`, `run_tiger_sft_all.py`, or
   `run_gpt2_sft_all.py`. Each runner reads the corresponding dataset, model,
   and trainer YAML files.
4. Re-evaluate saved models with `evaluate_sasrec_all.py`,
   `evaluate_tiger_sft_all.py`, or `evaluate_gpt2_sft_all.py`.

For example, `python scripts/exp1-standard-params/run_tiger_sft_all.py --help`
shows the available datasets, seed selection, data paths, and other options.
SASRec also supports a scalable-cross-entropy variant via `--variant sce`.

Primary checkpoints use `checkpoints/exp1/<model>/<dataset>/best.pt`.
Per-seed training snapshots and metrics are written under `experiments/` and
`artifacts/`. See [checkpoint layout](../../checkpoints/README.md) for selection
and [experiment 2](../exp2-tuned-params/README.md) for tuned baselines.
