#!/bin/bash -l
# CPU Slurm job: evaluate compact-template GPT-2 v2 synthetic windows.
#SBATCH --job-name=gpt2c_eval_v2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-config/pamap2_sdforger_gpt2_5class_v2_compact.yaml}"
DATASET_DIR="${DATASET_DIR:-artifacts/pamap2_sdforger_dataset_gpt2_compact_v2}"
GENERATED_DIR="${GENERATED_DIR:-outputs/generated/gpt2_5class_v2_compact}"
PLOTS_DIR="${PLOTS_DIR:-outputs/plots/gpt2_5class_v2_compact}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[evaluate-gpt2-compact-v2] project_root=$PROJECT_ROOT"
echo "[evaluate-gpt2-compact-v2] config=$CONFIG_PATH"
echo "[evaluate-gpt2-compact-v2] dataset=$DATASET_DIR"
echo "[evaluate-gpt2-compact-v2] generated=$GENERATED_DIR"
echo "[evaluate-gpt2-compact-v2] python=$(which python)"

python scripts/evaluate_similarity.py \
  --config "$CONFIG_PATH" \
  --real-windows "$DATASET_DIR/test_windows.npy" \
  --real-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir outputs/evaluation/similarity/gpt2_5class_v2_compact

python scripts/evaluate_utility.py \
  --config "$CONFIG_PATH" \
  --real-train-windows "$DATASET_DIR/train_windows.npy" \
  --real-train-metadata "$DATASET_DIR/train_metadata.csv" \
  --real-test-windows "$DATASET_DIR/test_windows.npy" \
  --real-test-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir outputs/evaluation/utility/gpt2_5class_v2_compact

python scripts/make_plots.py \
  --config "$CONFIG_PATH" \
  --training-log-csv outputs/checkpoints/gpt2_5class_v2_compact/training_log_history.csv \
  --real-windows "$DATASET_DIR/test_windows.npy" \
  --real-metadata "$DATASET_DIR/test_metadata.csv" \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --real-embeddings "$DATASET_DIR/test_embeddings.csv" \
  --synthetic-embeddings "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$PLOTS_DIR"
