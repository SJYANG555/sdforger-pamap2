#!/bin/bash -l
#SBATCH --job-name=pamap2_gpt2
#SBATCH --account=YOUR_TRITON_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pamap2-sdforger}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs
source ~/.bashrc
conda activate "$CONDA_ENV_NAME"

python scripts/train_pamap2_lm.py --config "$CONFIG_PATH"

