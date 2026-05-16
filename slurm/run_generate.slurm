#!/bin/bash -l
#SBATCH --job-name=gpt2_generate
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2.yaml}"
MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/gpt2/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/generated/gpt2}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs
module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[generate] project_root=$PROJECT_ROOT"
echo "[generate] config=$CONFIG_PATH"
echo "[generate] model=$MODEL_PATH"
echo "[generate] output=$OUTPUT_DIR"
echo "[generate] python=$(which python)"
nvidia-smi || true

python scripts/generate_pamap2_synthetic.py \
  --config "$CONFIG_PATH" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"
