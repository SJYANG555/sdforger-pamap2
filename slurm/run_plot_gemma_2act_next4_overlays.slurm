#!/bin/bash -l
# CPU Slurm job: draw real-vs-synthetic all-channel overlays for next-round 2-action Gemma runs.
#SBATCH --job-name=gemma_2act_overlay
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python scripts/make_2act_channel_overlay_plots.py \
  --no-default-runs \
  --samples-per-activity 24 \
  --output-dir outputs/plots/gemma_2act_next4_channel_overlays \
  --run gemma_2act_hand_stats:artifacts/pamap2_sdforger_dataset_2act_hand_stats:outputs/generated/gemma_2act_hand_stats \
  --run gemma_2act_chest_gyro_stats:artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats:outputs/generated/gemma_2act_chest_gyro_stats \
  --run gemma_2act_hand_ica8:artifacts/pamap2_sdforger_dataset_2act_hand_ica8:outputs/generated/gemma_2act_hand_ica8 \
  --run gemma_2act_chest_gyro_ica8:artifacts/pamap2_sdforger_dataset_2act_chest_gyro_ica8:outputs/generated/gemma_2act_chest_gyro_ica8
