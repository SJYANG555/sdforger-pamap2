#!/bin/bash -l
# CPU Slurm job: v2 k-NN condition consistency for real, GPT-2, and Gemma.
#SBATCH --job-name=knn_cond_v2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATASET_DIR="${DATASET_DIR:-artifacts/pamap2_sdforger_dataset_v2}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[knn-v2] project_root=$PROJECT_ROOT"
echo "[knn-v2] dataset=$DATASET_DIR"
echo "[knn-v2] python=$(which python)"

python scripts/evaluate_knn_condition_consistency.py \
  --real-train-windows "$DATASET_DIR/train_windows.npy" \
  --real-train-metadata "$DATASET_DIR/train_metadata.csv" \
  --real-test-windows "$DATASET_DIR/test_windows.npy" \
  --real-test-metadata "$DATASET_DIR/test_metadata.csv" \
  --gpt2-windows outputs/generated/gpt2_5class_v2/generated_windows.npy \
  --gpt2-metadata outputs/generated/gpt2_5class_v2/generated_embeddings.csv \
  --gemma-windows outputs/generated/gemma_5class_v2/generated_windows.npy \
  --gemma-metadata outputs/generated/gemma_5class_v2/generated_embeddings.csv \
  --output-dir outputs/evaluation/knn_condition_5class_v2
