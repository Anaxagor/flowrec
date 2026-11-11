# Configuration reference

FlowRec's Python classes use configuration dataclasses. The JSON consumed by
[`scripts/run_tiger.py`](../scripts/run_tiger.py) is a separate interface that
constructs those objects from sections such as `data`, `dataset`, `dataloader`,
`model`, `method`, `training` and `optimizer`. Other experiment runners use
their own YAML and command-line options; do not assume the same keys or defaults
apply to every runner. Consult the implementation and the relevant experiment
README when reproducing a run.

## Data and batches

For the TIGER runner, `data.reviews_path` selects Amazon interactions;
`data.semantic_id_path` selects a semantic-ID mapping. `data.metadata_path` is
optional. To use Yambda, set `data.source` to `yambda` and supply its interaction
and semantic-ID paths; the runner also accepts `data.use_hf` for Hugging Face
loading. For all sources, the semantic IDs, catalog and checkpoint must agree.

The `dataset` section controls normalization, filtering and examples:

| Keys | Purpose |
| --- | --- |
| `item_key`, `positive_criterion` | Interaction item column and pandas query defining the positive stream. Remaining events can supply interaction rewards. |
| `min_user_interactions`, `min_item_interactions` | K-core filtering thresholds. |
| `split_strategy` | `leave_one_out` for the Amazon and Yambda sequence builders. |
| `max_history_len`, `min_history_len`, `history_padding` | History construction and batch padding. |
| `train_example_layout` | `prefix_target` or `dense_causal`; the chosen model and runner must support the layout. |
| `train_target_policy` | Prefix-layout policy: `all`, `last`, `last_n`, `stride`, `sampled` or `hybrid`. Policy-specific settings include `train_last_n`, `train_stride` and `train_targets_per_user`. |
| `train_window_len`, `train_window_stride` | Dense-causal window size and stride. |
| `filter_cold_items_after_split`, `cold_item_train_source` | Optional filtering against the training catalog. |
| `offset_semantic_ids_by_one`, `drop_items_missing_semantic_id` | Semantic-ID handling. Match these settings to the tokenizer and model. |

The `dataloader` section includes `train_batch_size`, `validation_batch_size`,
`num_workers`, `pin_memory`, `persistent_workers`, `train_drop_last` and
`include_raw_in_batch`. The data modules disable persistent workers when
`num_workers=0`. Options and their defaults can differ between the runner and the
underlying dataclass:

- [`AmazonReviewDataConfig`](../flowrec/data/datamodule.py)
- [`YambdaDataConfig`](../flowrec/data/yambda.py)
- [`SequenceSplitConfig`](../flowrec/data/sequence_builder.py)

## Models and objectives

`model` in the TIGER runner includes `codebook_size`, `embedding_dim`,
`num_heads`, `num_encoder_layers`, `num_decoder_layers`, `dim_feedforward`,
`d_kv`, `dropout`, `user_ids_count`, `use_user_token`, `num_beams` and `top_k`.
The loaded semantic IDs determine the catalog representation; when loading a
checkpoint, use its own architecture and matching semantic IDs. See
[`TigerModel`](../flowrec/models/tiger.py),
[`GPT2RecModel`](../flowrec/models/gpt2.py) and
[`SASRecConfig`](../flowrec/models/sasrec.py) for their individual constructor
arguments. The TIGER JSON runner is not the configuration interface for all
three models.

The `method.name` supported by the TIGER runner is `sft`, `gflow` or
`mini-one-rec`. Their Python configuration classes are:

- [`SFTConfig`](../flowrec/ft_methods/sft/config.py): supervised token or item
  loss, optional generation metrics, and `item_loss` for exact or scalable
  cross-entropy. Scalable cross-entropy is a training-time approximation for
  supported dense item models; validation and test use exact cross-entropy.
- [`GFlowConfig`](../flowrec/ft_methods/gflow/config.py): `objective` (`tb`,
  `db`, `subtb`), SFT/GFlow weights, sampling parameters and required
  `reward_cache_path`. See [rewards and candidate sampling](rewards.md).
- [`DPOConfig`](../flowrec/ft_methods/preference/config.py): preference-loss
  `beta`, negative sampling and reference-model settings.
- [`MiniOneRecConfig`](../flowrec/ft_methods/mini_one_rec/config.py):
  rollout, reward and optimization settings for `mini-one-rec`.

The TIGER runner also derives several evaluation options from `training`,
including `evaluate_generation`, `generation_topk`, `generation_num_beams`
and `generation_num_return_sequences`. Check
[`build_method`](../scripts/run_tiger.py) for the mapping: it does not simply
forward every `method` key to `SFTConfig`.

## Training and optimization

[`TrainerConfig`](../flowrec/training/trainer.py) defines `max_epochs`,
`max_steps`, gradient accumulation/clipping, CUDA mixed precision, validation
cadence, early stopping, checkpoints and debug batch limits. Useful settings
include `gradient_accumulation_steps`, `max_grad_norm`, `use_amp`, `amp_dtype`,
`validate_every_epochs`, `validate_every_steps`, `run_validation_at_start`,
`monitor`, `monitor_mode`, `early_stopping_patience`, `min_delta`, `save_best`
and `save_last`. The generic trainer's default early-stopping comparison is
`best`; experiment configurations may override it.

The TIGER JSON runner reads AdamW settings such as `optimizer.lr`,
`optimizer.weight_decay`, `optimizer.betas` and `optimizer.eps`. Its
`training.init_checkpoint_path` supports warm-starting; configure the correct
weights and matching data/tokenizer first. The runner writes checkpoints under
the run directory that it creates. Selected checkpoint layout and model
provenance are documented in [checkpoints/README.md](../checkpoints/README.md).

## Semantic-ID tokenization

[`build_rqvae_semantic_ids.py`](../scripts/build_rqvae_semantic_ids.py)
constructs a tokenizer from an embedding store. Its `--help` describes model,
training, normalization and collision-resolution options. Relevant settings
include `--num-codebooks`, `--codebook-size`, `--latent-dim`, `--epochs`,
`--normalize`, `--collision-policy` and `--level-offset-tokens`.
[`RQVAEItemTokenizerConfig`](../flowrec/item_representation/semantic_id/rqvae_tokenizer.py)
and [`RQVAEConfig` / `RQVAETrainConfig`](../flowrec/item_representation/semantic_id/rqvae.py)
provide the programmatic equivalents. The tokenizer can append a collision
suffix, so the final semantic-ID length need not equal `num_codebooks`.
