#!/bin/bash -l
# Minimal end-to-end GPT-2 smoke test for Aalto Triton.
# Edit account / partition / PROJECT_ROOT / CONDA_ENV_NAME before use.
#SBATCH --job-name=gpt2_smoke
#SBATCH --account=YOUR_TRITON_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pamap2-sdforger}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2_smoke.yaml}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

source ~/.bashrc
conda activate "$CONDA_ENV_NAME"

echo "[smoke] project_root=$PROJECT_ROOT"
echo "[smoke] config=$CONFIG_PATH"
echo "[smoke] python=$(which python)"

python scripts/run_gpt2_smoke_test.py --config "$CONFIG_PATH"

echo "[smoke] completed"

