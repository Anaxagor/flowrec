# GPT-2: CE distillation from rewards on Beauty and Instruments

Runner: `scripts/exp6-ce-distillation/ce_distill_gpt2.py`.
For a paired comparison, run the same grid with `--sampling reward` and
`--sampling curriculum_smooth_is --curriculum-epochs 5`.
Both modes estimate CE under q; see the [formulas, grid, and parameters](README.md#comparing-two-unbiased-estimates-of-ce-under-q).
The initial weights are the exp2 GPT-2 `best.pt` for each dataset.
The objective, reward powers, and calibration follow the [TIGER experiment](README.md):
`L = L_SFT + lambda_D * CE`, `power in {0.25, 1, 4, 16}`,
`rho in {1, 0.1}`, seed 42. The reward is the product of the selected
`sasrec`, `interaction`, and hierarchical `similarity` components. The power
is applied to the entire product, using epsilon floors from matched exp4.
`q` is normalized over the complete GPT-2 catalog; `pi` is normalized over
valid semantic ID continuations.

## Data: prefix layout from causal layout

As in `gflow_finetune_gpt2_amazon_fullhist.py`, the original exp2
`dense_causal` windows are expanded into `prefix_target` examples. Each active
target in an original window becomes a separate example with exactly the same
prefix as in the causal layout. The history restarts at each new window;
previous windows are not appended to the context.

For Beauty and Instruments, the window length and stride are 20 items. The
catalog and full validation/test splits are retained from the dense splits.
All active training targets are used without last-N truncation. The conversion
audit and its signature are saved with the configuration; caches are checked
for matching context, subset, rows, and semantic ID order. TIGER caches are not
compatible.

## CE modes

- `--estimator sampled --sampling reward`: 10 IID samples with replacement
  from the complete `q`, yielding an unbiased estimate of full CE.
- `--sampling curriculum`: sample from the lower reward half of the catalog
  for the first `--curriculum-epochs` epochs, then from the upper half.
  Probabilities within each half are proportional to `q`.
- `--sampling curriculum_smooth`: a linear mixture of normalized halves
  `p_t = (1-alpha) * q_low + alpha * q_high`, where `alpha` increases from 0
  to 1 over `--curriculum-epochs` epochs, then the upper half is used.
- `--sampling curriculum_smooth_is`: sample from
  `m_t = (1-epsilon)*p_t + epsilon*q` with weights `q/m_t`, yielding an
  unbiased estimate of CE under q throughout training. `--proposal-q-mix`
  sets epsilon, default 0.1.
- `--estimator exact`: full CE sum over the catalog, without sampling.

In `curriculum` and `curriculum_smooth`, CE is computed over the current
conditional or mixed distribution without importance weights. The catalog is
split by item count and sorted by descending reward, with catalog order used
for ties. The default curriculum lasts 10 epochs; 0 selects the upper half
immediately. The transition advances per training microbatch, and resume
preserves progress.

For CE, the GPT-2 history is computed once into a differentiable KV cache.
Candidates reuse it while retaining gradients through the history. SFT uses
its own pass. Exact sums `q` mass on shared branching SID prefixes;
deterministic continuations have zero log-probability. Chunks use gradient
checkpointing. This is the exact sum for the cached rewards, including their
float16 quantization, without catalog truncation.

The matched protocol uses batch 32 × accumulation 64, AdamW lr 3e-5, weight
decay 0.01, and validation each epoch; the initial checkpoint is eligible for
best selection. The monitored metric is NDCG@5 with beam 5; patience is 5
after the first 10 epochs, with at most 500 epochs. Final validation/test uses
beam 20 and HR/NDCG/Coverage/ARP at K=5,10,20. `lambda_D` is calibrated once
from the FP32 CE/SFT gradient norm ratio.

## Running

Run from the repository root with the datasets and source assets available.
Prepare each reward component before training:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --cache-only --cache-component sasrec --device cuda
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --cache-only --cache-component interaction --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --cache-only --cache-component similarity --device cpu
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --list
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --device cuda
```

Without `--index`, training processes the selected grid sequentially. Use `--index N` to run one configuration from `--list`; independent indices can run in separate processes after preparation completes. Keep all grid and path options identical across phases.

For the smooth six-reward ablation, use:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2.py \
  --datasets beauty instruments --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction sasrec_x_similarity interaction_x_similarity
```

For exact CE, select `--estimator exact --catalog-chunk-size 64` and
`--checkpoint-every-updates 1`. Use the same options for aggregation and resume.
Exact evaluation requires substantially more computation than sampling.

## Caches, results, and verification

New GPT-2 factors:
`artifacts/exp7-ce-distillation-gpt2/<dataset>/reward_factors/<component>/`.
The internal name of hierarchical similarity is `sim_hier`; the public CLI
uses `similarity`, as with TIGER. `--cache-root` selects another root with the
same layout.

Results:

```text
experiments/exp7-ce-distillation-gpt2/<dataset>/exact/pow_<p>/rho_<r>/seed_<s>/
experiments/exp7-ce-distillation-gpt2/<dataset>/<combo>/curriculum_smooth_n10_epochs5/pow_<p>/rho_<r>/seed_<s>/
```

Saved assets include identity/config, dense→prefix audit, calibration, logs,
history, `checkpoints/{best,last}.pt`, and `result.json`. Different models and
sampling modes have separate directories. Resubmission resumes an unfinished
run; incompatible protocols or assets cause an error rather than overwriting
results. Reward lookup is used only during training.

After completion:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --aggregate-only --estimator exact
python scripts/exp6-ce-distillation/ce_distill_gpt2.py --aggregate-only \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

The `metrics.csv` and `summary.json` summaries appear in `exact/` and
`ablations/curriculum_smooth_n10_epochs5/`. Power/rho are selected separately
for each reward combination by validation NDCG@5, without selecting on test.

A short local check on a machine with assets and dependencies:

```bash
python scripts/exp6-ce-distillation/ce_distill_gpt2.py \
  --datasets beauty --cache-only --smoke --device cpu --num-workers 0
python scripts/exp6-ce-distillation/ce_distill_gpt2.py \
  --datasets beauty --powers 1 --rho-targets 1 --sampling curriculum_smooth \
  --curriculum-epochs 0.5 --smoke --device cpu --num-workers 0
```

Smoke uses a separate output with suffix `_smoke`, caches in its `cache/`,
eight training batches, and two evaluation batches. `--list`, `--count`, and `--aggregate-only` use only the
standard library.
