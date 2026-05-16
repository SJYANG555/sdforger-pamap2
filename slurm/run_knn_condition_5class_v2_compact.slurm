#!/bin/bash -l
# CPU Slurm job: final v2 k-NN condition consistency using compact GPT-2 v2 and Gemma v2.
#SBATCH --job-name=knn_cond_v2c
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[knn-v2-compact] project_root=$PROJECT_ROOT"
echo "[knn-v2-compact] python=$(which python)"

python scripts/evaluate_knn_condition_consistency.py \
  --real-train-windows artifacts/pamap2_sdforger_dataset_v2/train_windows.npy \
  --real-train-metadata artifacts/pamap2_sdforger_dataset_v2/train_metadata.csv \
  --real-test-windows artifacts/pamap2_sdforger_dataset_v2/test_windows.npy \
  --real-test-metadata artifacts/pamap2_sdforger_dataset_v2/test_metadata.csv \
  --gpt2-windows outputs/generated/gpt2_5class_v2_compact_ckpt1300_reparsed/generated_windows.npy \
  --gpt2-metadata outputs/generated/gpt2_5class_v2_compact_ckpt1300_reparsed/generated_embeddings.csv \
  --gemma-windows outputs/generated/gemma_5class_v2/generated_windows.npy \
  --gemma-metadata outputs/generated/gemma_5class_v2/generated_embeddings.csv \
  --output-dir outputs/evaluation/knn_condition_5class_v2_compact
