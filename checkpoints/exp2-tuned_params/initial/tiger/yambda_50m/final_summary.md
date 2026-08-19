# TIGER Run Summary

- Dataset: /home/sziganshin/reps/flowrec/data/raw/yambda/flat/50m/listens.parquet
- Semantic IDs: /home/sziganshin/reps/flowrec/artifacts/tokenizers/yambda_rqvae_50m_recent2m_cb3_append_leveloffset_bs2048/semantic_ids.json
- Epochs completed: 141
- Steps completed: 68526
- Best metric: 0.07047408083995128
- Best checkpoint: /home/sziganshin/reps/flowrec/experiments/tiger/tiger_yambda_50m_sft_no_user_model_100t_20260803/y50_sft_t031_h192_lr3em04_bg2048x1_gc1p0_nh2_kv32_ff512_do0p2/checkpoints/best.pt
- Last checkpoint: None
- Experiment folder: /home/sziganshin/reps/flowrec/experiments/tiger/tiger_yambda_50m_sft_no_user_model_100t_20260803/y50_sft_t031_h192_lr3em04_bg2048x1_gc1p0_nh2_kv32_ff512_do0p2

## Validation Metrics
{
  "val/loss": 2.447711568321868,
  "val/sft/base_loss": 2.447711568321868,
  "val/sft/generation_coverage@5": 0.13112809116102814,
  "val/sft/generation_ndcg@5": 0.0689999788432487,
  "val/sft/generation_recall@5": 0.09885137488416984,
  "val/sft/loss": 2.447711568321868,
  "val/sft/ndcg@5": 0.0689999788432487,
  "val/sft/recall@5": 0.09885137488416984,
  "val/sft/sequence_exact_match": 0.03283443555368986,
  "val/sft/token_accuracy": 0.48175542398704596
}

## Test Metrics
{
  "test/loss": 2.5125489060133885,
  "test/sft/base_loss": 2.5125489060133885,
  "test/sft/loss": 2.5125489060133885,
  "test/sft/sequence_exact_match": 0.035784826277336125,
  "test/sft/token_accuracy": 0.475891716370561
}
