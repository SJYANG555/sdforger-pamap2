#!/bin/bash -l
# GPU-ready GPT-2 baseline training for Triton.
# Edit PROJECT_ROOT / CONDA_ENV_PATH before use.
#SBATCH --job-name=gpt2_gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2.yaml}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[train] project_root=$PROJECT_ROOT"
echo "[train] config=$CONFIG_PATH"
echo "[train] python=$(which python)"
nvidia-smi || true

python scripts/train_pamap2_lm.py --config "$CONFIG_PATH"
