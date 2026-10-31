# Global-time experiments

These scripts run the standard and tuned model pipelines with a global temporal
split. They complement the leave-one-out pipelines in
[experiment 1](../exp1-standard-params/README.md) and
[experiment 2](../exp2-tuned-params/README.md). The data windows, catalog and
checkpoints are specific to the global-time protocol; leave-one-out checkpoints
must not be substituted automatically.

The scripts build Amazon embeddings and semantic IDs, tune GPT-2, SASRec,
SASRec with scalable cross-entropy and TIGER, and train selected SASRec models.
Configurations, data split details and execution guidance are documented in
[the global-time configuration guide](../../configs/global-time/README.md).

Primary tuned checkpoints use `checkpoints/exp2/global_split/initial/`.
Some models do not have a selected checkpoint and global-time SFT-only primary
weights are not present; consult the [checkpoint layout](../../checkpoints/README.md)
before launching a dependent run. For example, run
`python scripts/global-time/tuning_gpt2.py --help` from the repository root to
inspect the tuning options.
