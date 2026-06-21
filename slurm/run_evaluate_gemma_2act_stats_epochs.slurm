#!/bin/bash -l
# CPU Slurm job: evaluate, plot, and summarize Gemma 2-action stats-prompt epoch variants.
# Override RUNS with space-separated run specs: config_path:dataset_dir:run_name.
#SBATCH --job-name=gemma_stats_eval
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
RUNS="${RUNS:-config/pamap2_sdforger_gemma2_2b_2act_hand_stats_ep10.yaml:artifacts/pamap2_sdforger_dataset_2act_hand_stats:gemma_2act_hand_stats_ep10 config/pamap2_sdforger_gemma2_2b_2act_hand_stats_ep20.yaml:artifacts/pamap2_sdforger_dataset_2act_hand_stats:gemma_2act_hand_stats_ep20 config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_stats_ep10.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats:gemma_2act_chest_gyro_stats_ep10 config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_stats_ep20.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats:gemma_2act_chest_gyro_stats_ep20 config/pamap2_sdforger_gemma2_2b_2act_chest_stats_ep5.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_stats:gemma_2act_chest_stats_ep5 config/pamap2_sdforger_gemma2_2b_2act_chest_stats_ep10.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_stats:gemma_2act_chest_stats_ep10 config/pamap2_sdforger_gemma2_2b_2act_chest_stats_ep20.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_stats:gemma_2act_chest_stats_ep20 config/pamap2_sdforger_gemma2_2b_2act_chest_acc_stats_ep5.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats:gemma_2act_chest_acc_stats_ep5 config/pamap2_sdforger_gemma2_2b_2act_chest_acc_stats_ep10.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats:gemma_2act_chest_acc_stats_ep10 config/pamap2_sdforger_gemma2_2b_2act_chest_acc_stats_ep20.yaml:artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats:gemma_2act_chest_acc_stats_ep20}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[eval-stats-epochs] project_root=$PROJECT_ROOT"
echo "[eval-stats-epochs] runs=$RUNS"
echo "[eval-stats-epochs] python=$(which python)"

for spec in $RUNS; do
  IFS=: read -r CONFIG_PATH DATASET_DIR RUN_NAME <<< "$spec"
  CHECKPOINT_DIR="outputs/checkpoints/${RUN_NAME}"
  GENERATED_DIR="outputs/generated/${RUN_NAME}"
  SIMILARITY_DIR="outputs/evaluation/similarity/${RUN_NAME}"
  UTILITY_DIR="outputs/evaluation/utility/${RUN_NAME}"
  PLOTS_DIR="outputs/plots/${RUN_NAME}"

  echo "[eval-stats-epochs] evaluating run_name=$RUN_NAME"
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

python scripts/summarize_gemma_2act_stats_epochs.py
