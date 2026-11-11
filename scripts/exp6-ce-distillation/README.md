# TIGER: CE distillation from reward distributions

Related experiments:

- GPT-2 with prefix layout from causal windows: [README_GPT2.md](README_GPT2.md).
- TIGER on Yambda 50M/500M: last128 for 50M, sampled50 for 500M, smooth curriculum, and three CE samples: [README_YAMBDA.md](README_YAMBDA.md).
- GPT-2 on Yambda 50M/500M, trials 36/4: last128/sampled15 and a separate 500M sampled50 experiment comparing q sampling with smooth + q/m: [README_GPT2_YAMBDA.md](README_GPT2_YAMBDA.md).
- GPT-2 on Amazon Beauty 2023: 24 configurations, three samples, and smooth with q/m: [README_GPT2_BEAUTY2023.md](README_GPT2_BEAUTY2023.md).

The experiment fine-tunes exp2 TIGER on Beauty 2014 and Instruments 2018 with

\[
L = L_{SFT} + \lambda_D\,\mathbb E_x[-\sum_{i\in I}q_x(i)\log\pi_\theta(i\mid h)].
\]

By default, CE is estimated using **10 independent samples with replacement
from the full `q` distribution**. This is an unbiased estimate of the sum.
`--num-samples` changes the sample count; `--estimator exact` computes the full
catalog sum.

The reward and numerical bounds match the current matched exp4:

```text
r = max(1e-6, sasrec * interaction * hierarchical_similarity)
R = max(1e-6, r ** power)
q = R / sum_over_entire_catalog(R)
power in {0.25, 1, 4, 16}
```

Components have unit weights and the same individual epsilon floors as exp4.
`sasrec` is the sigmoid of a frozen exp2 SASRec score (temperature 1),
`interaction` uses evidence strictly before the training target timestamp, and
`hierarchical_similarity` measures closeness to the training target by RQ-VAE
codebooks. **The power applies to the full product.** At power=16 the final
floor can substantially alter the distribution; it is retained for comparison.

`q` is normalized over the full catalog, including previously viewed items.
The original `--sampling reward` mode has no support restriction, target mixing,
or top-K selection. `pi` uses local normalization over valid SID continuations,
as in the current GFlow and equation (3) of the paper. CE sums token
log-probabilities inside the SID. SFT remains ordinary token CE over the full
vocabulary. Both terms train the encoder; its output is reused for candidates.

## Protocol

The script reads `configs/exp2-tuned-params/finetune_base_matched.json` and
uses validated functions from `gflow_finetune_tiger_matched.py`:

- Initial weights: `checkpoints/exp2/loo_split/initial/tiger/<dataset>/best.pt`.
- All exp2 training prefixes: Beauty 131,413 and Instruments 116,989.
- Catalogs of 12,101 / 8,556 items, with the same splits and semantic IDs.
- Batch 32 × accumulation 64 = 2048, AdamW, lr 3e-5, weight decay 0.01.
- Validation after each epoch, monitoring NDCG@5 with beam 5; the initial
  checkpoint participates in best selection, patience is 5 after the first 10
  epochs, and the maximum is 500 epochs.
- Final evaluation of selected best: full validation/test, beam 20,
  HR/NDCG/Coverage/ARP at K=5,10,20.
- `rho in {1, 0.1}` as in exp4: once on the initial checkpoint, calculate
  `lambda_D = rho / median(||grad CE|| / ||grad SFT||)` over five FP32 batches.
  The coefficient is saved and not recalculated on resume.

The main grid is **2 datasets × 4 powers × 2 rho × seed 42 = 16 runs**.
`--seeds 42 43 44` extends it to 48. Order: dataset, power, rho, seed. In the
original `--sampling reward` mode, the first 10 epochs retain the shared
stopping protocol but CE does not use curriculum. Reward lookup is never called
on validation/test: target-dependent reward is used only for training.

## Exact and two-phase curriculum ablations

The additional grid has two parts:

- **16 exact runs** for the product of all three rewards: the same two datasets,
  four powers, two rho, and seed 42, with exact CE over the full catalog.
- **96 sampled runs**: `sasrec`, `interaction`, `similarity`, and pairs
  `sasrec_x_interaction`, `sasrec_x_similarity`, `interaction_x_similarity`,
  with the same powers/rho/seed and curriculum.

For each combination,
`R = max(1e-6, max(1e-6, product(selected_rewards)) ** power)`.
Curriculum uses **this reward distribution**, not a separate SASRec sampler.
For each training example, sort the full catalog by decreasing `q`: the first
half contains `floor(catalog_size / 2)` high-reward items and the second contains
the remaining low-reward items. Ties preserve catalog order. Halves are defined
by item count, not total probability.

1. First `--curriculum-epochs` epochs (default 10): lower half only.
2. From the next epoch onward: upper half only.

Both phases draw `--num-samples` independent samples **with replacement** from
the original `q`, renormalized within the current half:

```text
p_phase(i) = q(i) / sum(q(j) for j in current_half),  i in current_half
p_phase(i) = 0,                                     otherwise
CE_phase = mean(-log pi(sampled_item | history))
```

`--sampling curriculum` has no smooth transition or `q/p` weights. It therefore
estimates CE of the current half's conditional distribution, not unbiased full
`q` CE in each phase. `--curriculum-epochs 0` starts immediately with the upper
half. Epochs are converted to training microbatches accounting for epoch length
limits; resume preserves the switch point. Lambda calibration uses the first
phase distribution. SFT and evaluation remain unchanged.

## Smooth curriculum

`--sampling curriculum_smooth` gradually transfers sampling probability from
the lower to upper half. The catalog split and relative `q` probabilities within
each half match the two-phase mode:

```text
T = ceil(curriculum_epochs * training_microbatches_per_epoch)
alpha = clip(global_microstep / T, 0, 1)  # when T=0: alpha=1
p(i) = (1-alpha) * q_low_normalized(i) + alpha * q_high_normalized(i)
```

Initially the lower half is chosen with probability 1; halfway through, each
half has probability 0.5; after the transition, only the upper half is chosen.
Weights change after every training microbatch. `q` is first normalized within
each half and the resulting distributions are mixed, so the upper half's original
probability mass does not suppress the first phase.

Here `--curriculum-epochs` is the **duration of the linear transition** from
training start, without a preliminary hold on the lower half. For example, 5
means sampling only from the upper half by the start of epoch six. The default
is 10 epochs; 0 switches to the upper half immediately. Sampling uses replacement
and ordinary CE without `q/p`, so the target distribution changes with `alpha`.
Lambda calibration uses `alpha=0` (1 when T=0). Resume restores the global
microstep. `distill/curriculum_high_probability` logs the upper-half weight (its
batch mean in epoch-level averages).

## Running

### Comparing two unbiased estimates of CE under q

`--sampling curriculum_smooth_is` retains the full reward-distribution target
`q`. With `c_t = (1-alpha)*q_low_normalized + alpha*q_high_normalized`:

```text
m_t = (1-epsilon)*c_t + epsilon*q
i_k ~ m_t, independently with replacement
L_CE_hat = -(1/K) * sum_k (q(i_k)/m_t(i_k)) * log pi(i_k | h)
```

`--proposal-q-mix` sets `epsilon` (default **0.1**, `0 < epsilon <= 1`). Full
`q` is present at every step, including the start and end of curriculum. Weights
are neither clipped nor normalized over sampled candidates. At epsilon=0.1 they
are at most 10; epsilon=1 gives ordinary q sampling. Values so small that they
cause loss of support or float32 weight overflow are rejected explicitly, with
no hidden clipping. The expected loss and gradient before gradient clipping match
CE under q. Curriculum changes the sampling distribution and estimator variance.
`distill/curriculum_high_probability` records alpha in `c_t`, before q mixing;
mean/max weights and empirical ESS of sampled candidates are also logged.

For comparability with `--sampling reward`, lambda is calibrated using ordinary
q sampling, the same K, and calibration seed. Importance-weighted curriculum is
applied afterward. `curriculum` / `curriculum_smooth` use their own calibration
and objective.

A paired comparison runs **both modes** on the same grid: all seven nonempty
reward combinations, powers `{0.25,1,4,16}`, rho `{1,0.1}`, and seed 42.
This gives 112 configurations per mode for Beauty/Instruments. The transition
lasts five epochs; use K=10 for those datasets and K=3 for Yambda.

Run from the repository root with the datasets and source assets available.
Prepare each reward component before training:

```bash
python scripts/exp6-ce-distillation/ce_distill_tiger.py --cache-only --cache-component sasrec --device cuda
python scripts/exp6-ce-distillation/ce_distill_tiger.py --cache-only --cache-component interaction --device cpu
python scripts/exp6-ce-distillation/ce_distill_tiger.py --cache-only --cache-component similarity --device cpu
python scripts/exp6-ce-distillation/ce_distill_tiger.py --list
python scripts/exp6-ce-distillation/ce_distill_tiger.py --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

After compatible caches are ready, run each sampling mode with the same grid:

```bash
for sampling in reward curriculum_smooth_is; do
  python scripts/exp6-ce-distillation/ce_distill_tiger.py \
    --sampling "$sampling" --curriculum-epochs 5 --proposal-q-mix 0.1 \
    --combos sasrec interaction similarity sasrec_x_interaction \
      sasrec_x_similarity interaction_x_similarity sasrec_x_interaction_x_similarity
done
```

GPT-2 uses `ce_distill_gpt2.py`; Yambda uses the dedicated runners linked above.
Keep their data protocols and reward factors separate. Results distinguish
`ce_target`, `importance_correction`, `proposal_q_mix`, and
`calibration_sampling`.

For exact CE, pass `--estimator exact --catalog-chunk-size 64` and
`--checkpoint-every-updates 1`. The sum uses shared semantic-ID prefixes and
chunked gradients, retaining the full catalog and cached reward quantization.
For uncorrected curriculum ablations, use `--sampling curriculum` or
`--sampling curriculum_smooth` with the specified duration and reward sets.
Always pass matching grid options during aggregation.

Compatible caches are reused. The training runner does not build missing
reward factors. Supply the same `--cache-root` to every preparation/training
phase. Resume restores optimizer, RNG, and curriculum state from the selected
run's `last.pt`; changed protocols require a separate output root.

## Local checks and results

For a short check on a machine with dependencies, datasets, and exp2 assets:

```bash
python scripts/exp6-ce-distillation/ce_distill_tiger.py \
  --datasets beauty --cache-only --smoke --device cpu --num-workers 0
python scripts/exp6-ce-distillation/ce_distill_tiger.py \
  --datasets beauty --powers 1 --rho-targets 1 --index 0 \
  --smoke --device cpu --num-workers 0
```

Smoke limits training to 256 examples / 8 batches, evaluation to two batches,
and separates outputs/caches with the `_smoke` suffix. `--list`, `--count`,
and `--aggregate-only` read metadata only and do not import ML libraries.

The main directory is
`experiments/exp7-ce-distillation/<dataset>/sampled_n10/pow_<p>/rho_<rho>/seed_<seed>/`.
The exact sum of three rewards uses a separate `exact/` branch. Ablations use
`experiments/exp7-ce-distillation/<dataset>/<combo>/curriculum_n10_epochs10/pow_<p>/rho_<rho>/seed_<seed>/`.
Smooth mode has a separate `curriculum_smooth_n10_epochs5/` branch (for E=5).
Sample count and first-phase/transition duration form part of the branch name.
Reusable reward factors remain in `artifacts/`; training results are in
`experiments/`. Each cell saves `identity.json`, `resolved_config.json`,
`calibration.json`, `training_log.jsonl`, `training_history.json`,
`checkpoints/{best,last}.pt`, and `result.json`. Logs include CE/SFT, teacher
entropy, maximum probability, and observed-target probability. The init baseline,
cache preparation time, step count, and training/evaluation durations are saved
separately. Teacher preparation remains in the cost accounting when a ready cache
is used: cache timing records reuse.

After the selected runs complete:

```bash
python scripts/exp6-ce-distillation/ce_distill_tiger.py --aggregate-only
# If the grid/mode changed, supply the same arguments, for example --seeds 42 43 44.

# Additional modes are aggregated separately:
python scripts/exp6-ce-distillation/ce_distill_tiger.py --aggregate-only --estimator exact
python scripts/exp6-ce-distillation/ce_distill_tiger.py --aggregate-only \
  --sampling curriculum --combos sasrec interaction similarity \
  sasrec_x_interaction sasrec_x_similarity interaction_x_similarity
python scripts/exp6-ce-distillation/ce_distill_tiger.py --aggregate-only \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

This creates `sampled_n10/metrics.csv` and `sampled_n10/summary.json`. The
summary reports missing cells and selects a configuration by mean **validation
NDCG@5** across requested seeds. Test is not used for configuration selection.
Additional summaries are in `exact/` and `ablations/curriculum_n10_epochs10/`.
For ablations, the best power/rho is selected separately per reward combination
and dataset. The smooth E=5 summary is in `ablations/curriculum_smooth_n10_epochs5/`.

Resubmission resumes `last.pt`, preserving optimizer, AMP, early stopping, and
curriculum step; completed cells are skipped. Dropout and shuffle use
`seed + epoch`. `--checkpoint-every-updates N` additionally writes an atomic
checkpoint after every N optimizer updates; `0` saves at epoch end. Mid-epoch
resume skips already processed batches and restores RNG. Default N is 1 for
exact and 0 for sampled. Resumed epoch mean training metrics cover only its
remaining batches; validation always uses its normal specified data volume. If
training completed but the job was interrupted during final evaluation, only
evaluation repeats. Changing protocol/assets in an existing run directory raises
an error; use a new `--output-root` for a new protocol.

Compare against initial TIGER and matched continued SFT, as well as
three-component GFlow cells at the same power/rho. Older GFlow runs have different
sampler support and curriculum, so this comparison alone does not isolate loss
geometry. A rigorous conclusion for §5.4 also needs equal sampling support and
separate comparisons with equal update counts and compute budgets.
