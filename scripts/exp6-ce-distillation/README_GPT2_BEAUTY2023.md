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

By default there are **24 configurations** with seed 42:

- rewards: `sasrec`, `interaction`, `similarity` (hierarchical),
  `sasrec_x_interaction_x_similarity`;
- powers: `0.25`, `1`, `16`, applied to the full product;
- `rho`: `1`, `0.1`;
- `curriculum_smooth_is`, **3 CE candidates per history**, as with K=3 in GFlow.

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
Prepare each reward component before training:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component sasrec --device cuda
python scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component interaction --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py --cache-only --cache-component similarity --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py --list
python scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

Use `--cache-root` consistently across phases to reuse compatible dense-prefix
GFlow factors. Their manifests, training contexts, selected targets, and catalog
order are checked. `--curriculum-epochs` and `--seeds` select alternative grids;
use separate result identities for changed settings. `--smoke` still prepares
the full dataset before limiting training batches.

## Resources and results

GFlow on A100 measured approximately 10.94 GiB peak GPU memory for this
model. This is a card-selection reference, not a CE measurement: CE also
reconstructs the full `q` and sorts the catalog. With batch 128, one FP32
128 × 163,050 matrix occupies about 80 MiB. Check actual utilization and peak
RAM/VRAM after the first run.

```text
experiments/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/
  baseline/init_metrics.json
  [<single_reward>/]curriculum_smooth_is_n3_epochs5_qmix0p1/
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

The product of three rewards has no additional `<single_reward>` directory.
Use `--aggregate-only` for a summary of the current grid. The shared
`report_ce_results.py` includes this CE root in the final tables.
