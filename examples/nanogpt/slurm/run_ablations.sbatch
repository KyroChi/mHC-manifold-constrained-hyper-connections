#!/bin/bash
#SBATCH --job-name=nanogpt-mhc-ablations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/weka/kyle/research/nanoGPT-mHC/examples/nanogpt/logs/ablations_%j_%a.out
#SBATCH --error=/weka/kyle/research/nanoGPT-mHC/examples/nanogpt/logs/ablations_%j_%a.err
#SBATCH --array=0-2

# mHC Ablations: baseline, HC, and mHC experiments
# Usage: sbatch slurm/run_ablations.sbatch

set -euo pipefail

# Use absolute paths
PROJECT_ROOT="/weka/kyle/research/nanoGPT-mHC"
NANOGPT_DIR="${PROJECT_ROOT}/examples/nanogpt"

# Data directory (shared location)
export FINEWEB_DATA_DIR="/weka/kyle/data/fineweb10B"

# wandb configuration
# Set WANDB_ENTITY to your team name if you have one, or leave unset to use your personal account
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="nanogpt-mhc"

# Create logs directory
mkdir -p "${NANOGPT_DIR}/logs"

# Activate virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"

# Change to nanogpt directory
cd "${NANOGPT_DIR}"

# Define configs for each array task
CONFIGS=(
    "config/train_fineweb10B_48l.py"
    "config/train_fineweb10B_hc_48l.py"
    "config/train_fineweb10B_mhc_48l.py"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_NAME=$(basename "$CONFIG" .py)

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running config: $CONFIG"
echo "Data directory: $FINEWEB_DATA_DIR"
echo "wandb entity: ${WANDB_ENTITY:-<default>}"
echo "wandb project: $WANDB_PROJECT"
echo "=============================================="

# Run training with torchrun for 4 GPUs
torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py \
    "$CONFIG" \
    wandb_project="$WANDB_PROJECT"

echo "Training complete for $CONFIG_NAME"
