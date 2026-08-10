# Checkpoints

This folder stores trained model checkpoints for the experiments in this repo. Follow the rules below when adding a new checkpoint.

## Folder layout

Checkpoints are organized by experiment, then by model, then by dataset:

```
checkpoints/
  <experiment>/                # e.g. exp1-standard_params, exp2-tuned_params
    <model>/                   # e.g. sasrec, tiger
      <dataset>/               # e.g. beauty, instruments, beauty_2023, yambda_50m_listens
        seed_<N>/              # one subfolder per training seed (if multi-seed)
          best.pt
          best.pt.json         # metadata sidecar for the checkpoint
        training_summary.json
```

Notes:

- `exp2-tuned_params` has an extra level (`initial/`, `finetuned/`) between the experiment and the model.
- Always keep the metadata files next to the weights: `best.pt.json` / `last.pt.json` sidecars, `training_summary.json`, and (for tuned runs) `best_trial.json` / `all_trials.json`. These are small JSON files and must always be committed to git, even when the weights themselves are too large for GitHub (see below).

## Adding a new checkpoint

1. Place the checkpoint under the layout above, creating the experiment/model/dataset folders as needed.
2. Include the metadata sidecars (`*.pt.json`, `training_summary.json`, etc.) produced by the training script.
3. Check the file sizes before committing (see the size rule below).
4. If the model is a **generative recommender**, also add its semantic ID dictionary (see below).

## Size rule: files over GitHub's 100 MB limit

GitHub rejects files larger than **100 MB**. If a checkpoint file exceeds (or is close to) this limit:

1. **Do not commit the weights.** Add the file path to the root [.gitignore](../.gitignore), under the existing comment `# oversized model files (>100MB, exceed GitHub's file size limit)`.
2. **Upload the file to the shared Google Drive folder** instead:

   https://drive.google.com/drive/folders/1uzOTQ5CP67vl1d3M0nCAocIcM0yJXwHL?usp=sharing

   Recreate the same relative path on Drive as in the repo (e.g. `checkpoints/exp2-tuned_params/initial/sasrec/beauty_2023/best_model.pt`), so anyone can download it and drop it into place locally.
3. **Still commit the metadata sidecars** (`*.pt.json`, `training_summary.json`, …) to git — only the oversized binary goes to Drive.

## Generative recommenders: semantic ID dictionary is mandatory

A checkpoint of a generative recommender (e.g. **TIGER**) is unusable without the semantic ID dictionary it was trained with. Whenever you add such a checkpoint, you must **also** add the matching semantic ID artifacts under [datasets/semantic_ids/](../datasets/semantic_ids/), using the same experiment/dataset naming:

```
datasets/semantic_ids/
  <experiment>/
    <dataset>/
      semantic_ids.json          # item -> semantic ID dictionary
      semantic_ids.pt
      semantic_id_metadata.json
      config.json                # tokenizer (RQ-VAE) configuration
      model.pt                   # tokenizer weights
      normalizer.pt
      training_summary.json
      training_log.jsonl
```

The checkpoint and the semantic ID dictionary must come from the **same run/configuration** — never pair a generative checkpoint with semantic IDs produced by a different tokenizer. If any of these files exceed the 100 MB limit, apply the same Drive rule as above.
