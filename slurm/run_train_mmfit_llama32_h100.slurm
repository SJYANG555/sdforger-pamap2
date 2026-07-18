#!/bin/bash -l
# GPU Slurm job: train Llama 3.2 3B LoRA on MM-Fit five-class dataset.
#SBATCH --job-name=mmfit_llama
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/yangs9/ChatTS}"
CONFIG_PATH="${CONFIG_PATH:-config/mmfit_sdforger_llama32_3b_5class.yaml}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

echo "[train-mmfit] project_root=$PROJECT_ROOT"
echo "[train-mmfit] config=$CONFIG_PATH"
echo "[train-mmfit] python=$(which python)"
echo "[train-mmfit] hf_home=$HF_HOME"
nvidia-smi || true

python scripts/train_pamap2_lm.py --config "$CONFIG_PATH"
