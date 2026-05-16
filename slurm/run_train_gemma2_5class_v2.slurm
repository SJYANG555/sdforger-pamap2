#!/bin/bash -l
# GPU Slurm job: train Gemma 2 2B LoRA on the full v2 five-class dataset.
#SBATCH --job-name=gemma_v2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gemma2_2b_5class_v2.yaml}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[train-v2] project_root=$PROJECT_ROOT"
echo "[train-v2] config=$CONFIG_PATH"
echo "[train-v2] python=$(which python)"
nvidia-smi || true

python scripts/train_pamap2_lm.py --config "$CONFIG_PATH"
