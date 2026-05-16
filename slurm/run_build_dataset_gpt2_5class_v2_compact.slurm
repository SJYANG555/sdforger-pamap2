#!/bin/bash -l
# CPU Slurm job: build compact-template GPT-2 v2 dataset without touching Gemma v2 artifacts.
#SBATCH --job-name=build_gpt2c_v2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2_5class_v2_compact.yaml}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[build-gpt2-compact-v2] project_root=$PROJECT_ROOT"
echo "[build-gpt2-compact-v2] config=$CONFIG_PATH"
echo "[build-gpt2-compact-v2] python=$(which python)"

python scripts/build_pamap2_sdforger_dataset.py --config "$CONFIG_PATH"
