#!/bin/bash -l
# GPU Slurm job: train one Gemma 2-action stats-prompt epoch variant, then generate from best checkpoint.
# Required: CONFIG_PATH. Optional: RUN_NAME.
#SBATCH --job-name=gemma_stats_tg
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .yaml)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

echo "[train-generate-stats] project_root=$PROJECT_ROOT"
echo "[train-generate-stats] run_name=$RUN_NAME"
echo "[train-generate-stats] config=$CONFIG_PATH"
echo "[train-generate-stats] python=$(which python)"
nvidia-smi || true

python scripts/train_pamap2_lm.py --config "$CONFIG_PATH"

CHECKPOINT_DIR="outputs/checkpoints/${RUN_NAME}"
GENERATED_DIR="outputs/generated/${RUN_NAME}"

echo "[train-generate-stats] generating from ${CHECKPOINT_DIR}/best to ${GENERATED_DIR}"
python scripts/generate_pamap2_synthetic.py \
  --config "$CONFIG_PATH" \
  --model-path "${CHECKPOINT_DIR}/best" \
  --output-dir "$GENERATED_DIR"
