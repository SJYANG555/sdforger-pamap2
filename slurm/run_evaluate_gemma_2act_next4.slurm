#!/bin/bash -l
# CPU Slurm job: evaluate and plot one or more Gemma 2-action PAMAP2 variants.
# VARIANTS defaults to A/B only: hand_stats chest_gyro_stats.
#SBATCH --job-name=gemma_2act_eval
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
VARIANTS="${VARIANTS:-hand_stats chest_gyro_stats}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

resolve_variant() {
  case "$1" in
    hand_stats)
      CONFIG_PATH="config/pamap2_sdforger_gemma2_2b_2act_hand_stats.yaml"
      DATASET_DIR="artifacts/pamap2_sdforger_dataset_2act_hand_stats"
      RUN_NAME="gemma_2act_hand_stats"
      ;;
    chest_gyro_stats)
      CONFIG_PATH="config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_stats.yaml"
      DATASET_DIR="artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats"
      RUN_NAME="gemma_2act_chest_gyro_stats"
      ;;
    hand_ica8)
      CONFIG_PATH="config/pamap2_sdforger_gemma2_2b_2act_hand_ica8.yaml"
      DATASET_DIR="artifacts/pamap2_sdforger_dataset_2act_hand_ica8"
      RUN_NAME="gemma_2act_hand_ica8"
      ;;
    chest_gyro_ica8)
      CONFIG_PATH="config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_ica8.yaml"
      DATASET_DIR="artifacts/pamap2_sdforger_dataset_2act_chest_gyro_ica8"
      RUN_NAME="gemma_2act_chest_gyro_ica8"
      ;;
    *) echo "Unknown variant: $1" >&2; exit 2 ;;
  esac
}

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[eval-next4] project_root=$PROJECT_ROOT"
echo "[eval-next4] variants=$VARIANTS"
echo "[eval-next4] python=$(which python)"

for variant in $VARIANTS; do
  resolve_variant "$variant"
  CHECKPOINT_DIR="outputs/checkpoints/${RUN_NAME}"
  GENERATED_DIR="outputs/generated/${RUN_NAME}"
  SIMILARITY_DIR="outputs/evaluation/similarity/${RUN_NAME}"
  UTILITY_DIR="outputs/evaluation/utility/${RUN_NAME}"
  PLOTS_DIR="outputs/plots/${RUN_NAME}"

  echo "[eval-next4] evaluating variant=$variant run_name=$RUN_NAME"
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
done
