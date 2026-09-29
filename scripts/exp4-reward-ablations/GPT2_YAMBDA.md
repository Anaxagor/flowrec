# GPT-2 Yambda fine-tuning from dense windows

`gflow_finetune_gpt2_yambda.py` runs GFlow and its `--sft-only` control from the
same exp2 initial GPT-2 checkpoint. It uses the existing matched schedule,
reward definitions, candidate samplers and checkpoint resolver.

Training prefixes are extracted from active labels in the checkpoint's dense
windows: length 50, stride 50, anchored at the end of each training history.
Each prefix starts at that label's original window boundary. For a window
`A B C D`, the examples are `A -> B`, `A B -> C`, and `A B C -> D`.

- Yambda 50M: retain each user's last 128 training targets.
- Yambda 500M: sample 15 targets from the last 128, with subset seed 42.
- Sampling does not reanchor windows. SFT and GFlow use the same examples.
- The catalog is built from the full dense training split **before** selecting
  the last 128 targets. Validation/test examples and item IDs are retained
  from that dense split. This matches GPT-2's base evaluation protocol; it
  does not change TIGER's separately constructed evaluation population.
- All Amazon GPT-2 prefix entrypoints use this same expansion, with window/stride
  20 and all eligible targets. See [the shared protocol](GPT2_DENSE_PREFIX.md).

Every data build checks complete validation/test example hashes against the
dense reference. Training writes `dense_prefix_audit.json` beside
`data_reward_config.json`. `--audit-data-only` runs this check without loading
the GPT-2 checkpoint or computing rewards. It still loads the real data, so
run the 500M audit on a node with the dataset's usual RAM allocation.

Reward caches store factors only. New cache and run roots prevent reuse of
rewards computed from the old sliding contexts. Cache manifests also record
the window/target-selection protocol and reject a mismatched context.

```bash
# Recalculate required caches, then run teacher-free GFlow and SFT controls.
RECALC_GROUPS=teacherfree_gpt2_yambda BLOCK_SIZE=4 MAX_PARALLEL=20 \
  bash scripts/scheduler/submit_base_matched_recalc.sh

# SASRec-based reward/sampler grid and SFT controls instead.
RECALC_GROUPS=gpt2_yambda BLOCK_SIZE=4 MAX_PARALLEL=20 \
  bash scripts/scheduler/submit_base_matched_recalc.sh
```

Both commands select 50M and 500M. Set
`GPT2_YAMBDA_DATASETS="yambda_50m_listens"` to select just 50M. Add `DRY_RUN=1`
to inspect submissions. Existing caches in the new directory are reused when
valid. Grid jobs depend on successful cache jobs. Each group also schedules
SFT-only seeds 42/43/44, which need no
reward cache. Existing cluster resource constraints and 500M node lanes apply.

Direct examples, inside an appropriate allocation:

```bash
python scripts/exp4-reward-ablations/gflow_finetune_gpt2_yambda.py \
  --teacher-free --datasets yambda_50m_listens --index 0
python scripts/exp4-reward-ablations/gflow_finetune_gpt2_yambda.py \
  --teacher-free --datasets yambda_50m_listens --sft-only --seed 43
python scripts/exp4-reward-ablations/gflow_finetune_gpt2_yambda.py \
  --datasets yambda_50m_listens --audit-data-only
```

Outputs:

- Runs: `experiments/exp4-gflow-gpt2-yambda-dense-prefix[-teacherfree]/<dataset>/`
- Shared factors: `artifacts/exp4-reward_ablation/gflow_gpt2_yambda_dense_prefix/<dataset>/reward_factors/`
- Metrics: `artifacts/exp4-reward_ablation/gflow_gpt2_yambda_dense_prefix[_teacherfree]_metrics_summary.csv`

The launchers under `scripts/scheduler` are gitignored in this repository; include
the updated `submit_base_matched_recalc.sh` when copying changes to the cluster.

Validated against local data on 2026-09-21: all validation/test raw examples
match the dense layout (8,775 per split for 50M; 87,955 per split for 500M).
Catalog and split counts match all saved exp2 GPT-2 seeds. The reproducible
audit and hashes are under `artifacts/audits/gpt2_yambda_dense_prefix_20260921/`.

The requested target-selection ablation also includes a Yambda 50M SFT-only
prefix control using **all** dense labels. Pass `--sft-only --yambda50-all-targets`
with `--datasets yambda_50m_listens`; its outputs gain `-all-targets` / `_all_targets`
suffixes. The normal 50M SFT/GFlow runs retain their last-128 target rule.
The [current batch launcher](FINETUNE_BATCH.md) submits both SFT variants and
the full last-128 GFlow ablation.
