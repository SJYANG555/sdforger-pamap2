#!/bin/bash -l
# CPU Slurm job: evaluate one v2 channel-subset synthetic dataset.
# Set CONFIG_PATH, DATASET_DIR, GENERATED_DIR, SIMILARITY_DIR, UTILITY_DIR, and PLOTS_DIR via sbatch --export.
# Optionally set TRAINING_LOG_CSV when a CSV training curve is available.
#SBATCH --job-name=eval_v2_variant
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
GENERATED_DIR="${GENERATED_DIR:?GENERATED_DIR is required}"
SIMILARITY_DIR="${SIMILARITY_DIR:?SIMILARITY_DIR is required}"
UTILITY_DIR="${UTILITY_DIR:?UTILITY_DIR is required}"
PLOTS_DIR="${PLOTS_DIR:?PLOTS_DIR is required}"
TRAINING_LOG_CSV="${TRAINING_LOG_CSV:-}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[evaluate-v2-variant] project_root=$PROJECT_ROOT"
echo "[evaluate-v2-variant] config=$CONFIG_PATH"
echo "[evaluate-v2-variant] dataset=$DATASET_DIR"
echo "[evaluate-v2-variant] generated=$GENERATED_DIR"
echo "[evaluate-v2-variant] similarity=$SIMILARITY_DIR"
echo "[evaluate-v2-variant] utility=$UTILITY_DIR"
echo "[evaluate-v2-variant] plots=$PLOTS_DIR"
echo "[evaluate-v2-variant] python=$(which python)"

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

plot_args=(
  --config "$CONFIG_PATH"
  --real-windows "$DATASET_DIR/test_windows.npy"
  --real-metadata "$DATASET_DIR/test_metadata.csv"
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy"
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv"
  --real-embeddings "$DATASET_DIR/test_embeddings.csv"
  --synthetic-embeddings "$GENERATED_DIR/generated_embeddings.csv"
  --output-dir "$PLOTS_DIR"
)
if [[ -n "$TRAINING_LOG_CSV" && -f "$TRAINING_LOG_CSV" ]]; then
  plot_args+=(--training-log-csv "$TRAINING_LOG_CSV")
fi

python scripts/make_plots.py "${plot_args[@]}"
