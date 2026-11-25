# GPT-2: CE distillation on Amazon Beauty 2023

`ce_distill_gpt2_beauty2023.py` fine-tunes the same initial GPT-2 checkpoint
as `gflow_finetune_gpt2_amazon_fullhist.py`:

```text
checkpoints/exp2/loo_split/initial/gpt2/beauty_2023/best.pt
```

This is exp2 trial 4: a four-layer model with width 512 and four attention
heads. The architecture and dropout are read from the checkpoint and
`best_trial.json`; the model files are part of the run identity.

## Grid and loss

The default grid has **six configurations per model** at seed 42. Every run
uses the product of SASRec, interaction and hierarchical similarity
(`sasrec_x_interaction_x_similarity`). It crosses powers `0.25`, `1`, `16`
of the full product with `rho` values `1` and `0.1`. The sampler is
`curriculum_smooth_is` with three CE candidates per history.

The existing `RewardDistillationMethod` implementation is used:

```text
loss = SFT + lambda_CE * mean[-q(i)/m(i) * log pi(i | history)]
i ~ m, with replacement
m = epsilon*q + (1-epsilon)*[(1-alpha)*q_low + alpha*q_high]
epsilon = 0.1
alpha = min(1, micro_step / transition_steps)
```

The catalog is sorted by reward and divided into lower and upper halves.
`q_low` and `q_high` retain the proportions of the original `q` within each half.
The transition from low to high rewards takes **5 epochs** by default
(`--curriculum-epochs`). Mixing in the full `q` maintains support across the
catalog, including at both ends of the transition. Raw `q/m` weights are used
without clipping or self-normalization, so the target CE and its gradient
remain those of the full `q`. The target and viewed items are not excluded.

Rewards, SASRec teacher, RQ-VAE geometry, and epsilon floors match GFlow.
In particular, the post-power floor is retained for `power=16`. Gradient-norm
calibration of `lambda_CE` uses direct samples from `q`, five batches, and the
specified `rho`; the coefficient is saved for resume.

## Data and training

The experiment uses `configs/exp2-tuned-params/dataset_amazon_2023.yaml`:
verified purchases, leave-one-out, and the existing filters. Training uses a
prefix layout derived from active labels in the original dense-causal windows
of length 20 and stride 20. Context boundaries, catalog, and original
validation/test rows are preserved; no additional tail/random subset is used.
Non-positive events remain available for interaction reward. The dense/prefix
audit is saved in `dense_prefix_audit.json`.

In the existing GFlow run, this setup yields 3,493,253 training examples,
163,050 items, and 662,540 examples each for validation and test. At startup,
CE checks actual counts and reward-cache compatibility rather than substituting
these values.

CE calls the same builder as GFlow. Regression tests compare input-data,
filter, and tokenization configurations and, on a small dataset, every
train/val/test row, its tokenized representation, catalog, and selected
indices. They separately check that prefix examples match active causal-window
labels, including context boundaries. The full-dataset counts are supported by
the GFlow results; CE checks compatibility again when launched.

- Batch **128 × accumulation 16 = 2048**; FP16, lr `3e-5`, weight decay `0.01`.
- Full validation after every epoch, beam 5, monitored by NDCG@5; the initial
  checkpoint is eligible for best selection. Patience 5, minimum 10 epochs,
  maximum 500, as in the matched CE/GFlow protocol.
- Final evaluation of the best checkpoint: full validation/test, beam 20,
  HR/NDCG/Coverage/ARP at K=5,10,20.
- Resume is enabled; an intermediate checkpoint is saved every 100 optimizer
  updates (`--checkpoint-every-updates`). Identity prevents changes to data,
  hyperparameters, or reward factors within an existing directory.

## Running

Run from the repository root with the datasets and source assets available.
For a standalone run on a compute node, prepare each reward component before training.

```bash
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component sasrec --device cuda
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component interaction --device cpu
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component similarity --device cpu
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --list
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

Use `--cache-root` consistently across phases to reuse compatible dense-prefix
GFlow factors. Their manifests, training contexts, selected targets, and catalog
order are checked. `--curriculum-epochs` and `--seeds` select alternative grids;
use separate result identities for changed settings. `--smoke` still prepares
the full dataset before limiting training batches.

## TIGER and GPT-2 initial checkpoints

For TIGER, use the same cache preparation and training options as in the
GPT-2 commands above. To list the configurations for each model, run

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger_beauty2023.py --list
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --list
```

Each model has its own six configurations and reward factors. Both use
`checkpoints/exp2/loo_split/initial/<model>/beauty_2023/best.pt` and the
leave-one-out SASRec teacher under the same `initial` directory. Clusters with
the original weights can use the existing `checkpoints/exp2-tuned_params/initial`
layout. The data follow the GFlow setup for each model. TIGER uses every training
prefix and evaluates a fixed 10% of validation rows for early stopping. GPT-2
uses the dense-window prefixes described above and validates on the full split.
Both evaluate the best checkpoint on the full test split.

Prepare the SASRec factors on a GPU and the interaction and similarity
components on CPU. Train each index after all three factors are ready.
A related GPT-2 CE run on Beauty 2023 took 40.1 hours with direct
sampling on an H100. Its smooth counterpart on an A100 took about 2 hours per
epoch. The estimated training budget is 72 hours per configuration.

TIGER's output directory is
`experiments/exp7-ce-distillation-tiger-beauty2023/beauty_2023/`.

## Resources and results

GFlow on A100 measured approximately 10.94 GiB peak GPU memory for this
model. This is a card-selection reference, not a CE measurement: CE also
reconstructs the full `q` and sorts the catalog. With batch 128, one FP32
128 × 163,050 matrix occupies about 80 MiB. Check actual utilization and peak
RAM/VRAM after the first run.

```text
experiments/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/
  baseline/init_metrics.json
  curriculum_smooth_is_n3_epochs5_qmix0p1/
    pow_<power>/rho_<rho>/seed_42/
      result.json
      training_history.json
      training_log.jsonl
      identity.json
      resolved_config.json
      calibration.json
      dense_prefix_audit.json
      checkpoints/{best,last}.pt
artifacts/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/reward_factors/
```

Use `--aggregate-only` for a summary of the current grid. The shared
`report_ce_results.py` includes this CE root in the final tables.
