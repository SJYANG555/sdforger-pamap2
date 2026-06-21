#!/bin/bash -l
# CPU Slurm job: evaluate a Gemma PAMAP2 SDForger-style experiment variant.
# Override CONFIG_PATH, DATASET_DIR, GENERATED_DIR, CHECKPOINT_DIR, and output dirs when submitting.
#SBATCH --job-name=gemma_eval_var
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gemma2_2b_2act_full.yaml}"
DATASET_DIR="${DATASET_DIR:-artifacts/pamap2_sdforger_dataset_2act_full}"
GENERATED_DIR="${GENERATED_DIR:-outputs/generated/gemma_2act_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/gemma_2act_full}"
SIMILARITY_DIR="${SIMILARITY_DIR:-outputs/evaluation/similarity/gemma_2act_full}"
UTILITY_DIR="${UTILITY_DIR:-outputs/evaluation/utility/gemma_2act_full}"
PLOTS_DIR="${PLOTS_DIR:-outputs/plots/gemma_2act_full}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[evaluate-variant] project_root=$PROJECT_ROOT"
echo "[evaluate-variant] config=$CONFIG_PATH"
echo "[evaluate-variant] dataset=$DATASET_DIR"
echo "[evaluate-variant] generated=$GENERATED_DIR"
echo "[evaluate-variant] checkpoint=$CHECKPOINT_DIR"
echo "[evaluate-variant] python=$(which python)"

python scripts/evaluate_similarity.py \
  --config "$CONFIG_PATH" \
  --real-windows "$DATASET_DIR/test_windows.npy" \
  --real-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$SIMILARITY_DIR"

python scripts/evaluate_utility.py \
  --config "$CONFIG_PATH" \
  --real-train-windows "$DATASET_DIR/train_windows.npy" \
  --real-train-metadata "$DATASET_DIR/train_metadata.csv" \
  --real-test-windows "$DATASET_DIR/test_windows.npy" \
  --real-test-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$UTILITY_DIR"

python scripts/make_plots.py \
  --config "$CONFIG_PATH" \
  --training-log-csv "$CHECKPOINT_DIR/training_log_history.csv" \
  --real-windows "$DATASET_DIR/test_windows.npy" \
  --real-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --real-embeddings "$DATASET_DIR/test_embeddings.csv" \
  --synthetic-embeddings "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$PLOTS_DIR"
