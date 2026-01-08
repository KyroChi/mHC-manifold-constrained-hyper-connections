#!/bin/bash
#SBATCH --job-name=owt-124m-ablations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/weka/kyle/research/nanoGPT-mHC/examples/nanogpt/logs/owt_124m_%j_%a.out
#SBATCH --error=/weka/kyle/research/nanoGPT-mHC/examples/nanogpt/logs/owt_124m_%j_%a.err
#SBATCH --array=0-3

# OWT 124M Ablations: baseline, HC, mHC, and 90_10 dual stream
# Usage: sbatch slurm/run_owt_124m.sbatch
#
# Experiments:
#   0: Baseline
#   1: HC
#   2: mHC
#   3: Dual Stream (0.9, 0.1)
#
# Each job uses 1 node with 8 GPUs

set -euo pipefail

# Use absolute paths
PROJECT_ROOT="/weka/kyle/research/nanoGPT-mHC"
NANOGPT_DIR="${PROJECT_ROOT}/examples/nanogpt"

# Data directory (shared location)
export OWT_DATA_DIR="/weka/kyle/data/openwebtext"

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
    "config/train_owt_124m.py"                    # 0: Baseline
    "config/train_owt_124m_hc.py"                 # 1: HC
    "config/train_owt_124m_mhc.py"                # 2: mHC
    "config/train_owt_124m_dual_stream_90_10.py"  # 3: Dual Stream (0.9, 0.1)
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_NAME=$(basename "$CONFIG" .py)

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running config: $CONFIG"
echo "Data directory: $OWT_DATA_DIR"
echo "wandb entity: ${WANDB_ENTITY:-<default>}"
echo "wandb project: $WANDB_PROJECT"
echo "Nodes: 1"
echo "GPUs per node: 8"
echo "Memory: 512GB"
echo "Time limit: 24 hours"
echo "=============================================="

# Run training with torchrun for single node (8 GPUs)
# Note: gradient_accumulation_steps in config is set for 8 GPUs (5*8=40)
# train.py will divide by world_size (8), so per GPU: 5 gradaccum
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    "$CONFIG"

echo "Training complete for $CONFIG_NAME"

