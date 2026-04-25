# FlowRec

FlowRec is a modular research framework for fine-tuning generative recommendation models with supervised, flow-based, RL-based, and preference-optimization objectives.

## Main goals

- Train and fine-tune generative recommendation models such as TIGER.
- Build semantic item IDs from item descriptions using content embeddings and RQ-VAE.
- Support standard SFT training as well as future GFlowNet, DPO, GRPO, and RL-style fine-tuning.
- Use SASRec-style models as baselines, candidate generators, or collaborative reward models.
- Provide reproducible experiments through config-driven pipelines.

## Repository structure

```text
flowrec/
├── configs/                 # Experiment, model, method, and trainer configs
├── data/                    # Raw/processed data and semantic IDs
├── artifacts/               # Tokenizers, checkpoints, metrics, predictions
├── scripts/                 # CLI scripts for preprocessing/training
├── flowrec/
│   ├── data/                # Dataset loading, preprocessing, splits, datamodules
│   ├── item_representation/ # Content embeddings and semantic ID generation
│   ├── models/              # TIGER, SASRec, and future models
│   ├── ft_methods/          # SFT, GFlowNet, DPO, RL, hybrid methods
│   └── training/            # Generic trainer and optimization loop
└── tests/                   # Unit and smoke tests
