#!/bin/bash -l
# CPU Slurm job: evaluate MM-Fit GPT-2 synthetic windows.
#SBATCH --job-name=mmfit_eval_gpt2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/yangs9/ChatTS}"
CONFIG_PATH="${CONFIG_PATH:-config/mmfit_sdforger_gpt2_5class.yaml}"
GENERATED_DIR="${GENERATED_DIR:-outputs/generated/mmfit_gpt2_5class}"
SIMILARITY_DIR="${SIMILARITY_DIR:-outputs/evaluation/similarity/mmfit_gpt2_5class}"
UTILITY_DIR="${UTILITY_DIR:-outputs/evaluation/utility/mmfit_gpt2_5class}"
PLOTS_DIR="${PLOTS_DIR:-outputs/plots/mmfit_gpt2_5class}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

python scripts/evaluate_similarity.py \
  --config "$CONFIG_PATH" \
  --real-windows artifacts/mmfit_sdforger_dataset_5class/test_windows.npy \
  --real-metadata artifacts/mmfit_sdforger_dataset_5class/test_metadata.csv \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$SIMILARITY_DIR"

python scripts/evaluate_utility.py \
  --config "$CONFIG_PATH" \
  --real-train-windows artifacts/mmfit_sdforger_dataset_5class/train_windows.npy \
  --real-train-metadata artifacts/mmfit_sdforger_dataset_5class/train_metadata.csv \
  --real-test-windows artifacts/mmfit_sdforger_dataset_5class/test_windows.npy \
  --real-test-metadata artifacts/mmfit_sdforger_dataset_5class/test_metadata.csv \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$UTILITY_DIR"

python scripts/make_plots.py \
  --config "$CONFIG_PATH" \
  --real-windows artifacts/mmfit_sdforger_dataset_5class/test_windows.npy \
  --real-metadata artifacts/mmfit_sdforger_dataset_5class/test_metadata.csv \
  --synthetic-windows "$GENERATED_DIR/generated_windows.npy" \
  --synthetic-metadata "$GENERATED_DIR/generated_embeddings.csv" \
  --real-embeddings artifacts/mmfit_sdforger_dataset_5class/test_embeddings.csv \
  --synthetic-embeddings "$GENERATED_DIR/generated_embeddings.csv" \
  --output-dir "$PLOTS_DIR"

