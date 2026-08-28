#!/bin/bash
#SBATCH --job-name=exp5_05_validation_array
#SBATCH --account=<<ACCOUNT>>
#SBATCH --partition=<<PARTITION>>
#SBATCH --time=12:00:00
#SBATCH --mem=<<MEM:32G>>
#SBATCH --cpus-per-task=<<CPUS:8>>
#SBATCH --gres=<<GRES:gpu:0>>
#SBATCH --array=0-15
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# Replace <<...>> placeholders with cluster-specific values (or override via
# sbatch CLI flags). Environment contract:
#   FLOWREC_PROJECT_ROOT  repository checkout on the cluster (required)
#   EXP5_CONFIG           experiment YAML (default: beauty2014_full.yaml)
#   EXP5_SPLIT            validation | test (stages that take --split)
#   EXP5_DEVICE           cuda | cpu (cpu is viable for every stage but 02/03)
#   EXP5_ENV_SETUP        optional script to source (conda/module loads)
set -euo pipefail
[ -n "${EXP5_ENV_SETUP:-}" ] && source "$EXP5_ENV_SETUP"
cd "${FLOWREC_PROJECT_ROOT:?set FLOWREC_PROJECT_ROOT}"
CFG="${EXP5_CONFIG:-scripts/exp5_budgeted_cyclic/configs/beauty2014_full.yaml}"
ARGS=(--config "$CFG" --device "${EXP5_DEVICE:-cuda}")
ARGS+=(--split "${EXP5_SPLIT:-validation}")
ARGS+=(--shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" --resume)
python scripts/exp5_budgeted_cyclic/05_run_validation_grid.py "${ARGS[@]}"
