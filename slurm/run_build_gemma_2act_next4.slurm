#!/bin/bash -l
# CPU Slurm job: build one or more Gemma 2-action PAMAP2 SDForger datasets.
# VARIANTS defaults to A/B only: hand_stats chest_gyro_stats.
# Supported variants: hand_stats chest_gyro_stats hand_ica8 chest_gyro_ica8.
#SBATCH --job-name=gemma_2act_build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
VARIANTS="${VARIANTS:-hand_stats chest_gyro_stats}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

config_for_variant() {
  case "$1" in
    hand_stats) echo "config/pamap2_sdforger_gemma2_2b_2act_hand_stats.yaml" ;;
    chest_gyro_stats) echo "config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_stats.yaml" ;;
    hand_ica8) echo "config/pamap2_sdforger_gemma2_2b_2act_hand_ica8.yaml" ;;
    chest_gyro_ica8) echo "config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_ica8.yaml" ;;
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

echo "[build-next4] project_root=$PROJECT_ROOT"
echo "[build-next4] variants=$VARIANTS"
echo "[build-next4] python=$(which python)"

for variant in $VARIANTS; do
  config_path="$(config_for_variant "$variant")"
  echo "[build-next4] building variant=$variant config=$config_path"
  python scripts/build_pamap2_sdforger_dataset.py --config "$config_path"
done
