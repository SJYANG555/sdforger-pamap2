#!/bin/bash -l
# CPU Slurm job: build missing stats-prompt datasets for Gemma 2-action epoch experiments.
# Use CONFIGS to override the merged build list.
#SBATCH --job-name=gemma_stats_build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
CONFIGS="${CONFIGS:-config/pamap2_sdforger_gemma2_2b_2act_chest_stats_ep5.yaml config/pamap2_sdforger_gemma2_2b_2act_chest_acc_stats_ep5.yaml}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "[build-stats-epochs] project_root=$PROJECT_ROOT"
echo "[build-stats-epochs] configs=$CONFIGS"
echo "[build-stats-epochs] python=$(which python)"

for config_path in $CONFIGS; do
  echo "[build-stats-epochs] building config=$config_path"
  python scripts/build_pamap2_sdforger_dataset.py --config "$config_path"
done
