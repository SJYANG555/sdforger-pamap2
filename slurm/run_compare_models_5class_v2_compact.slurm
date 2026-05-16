#!/bin/bash -l
# CPU Slurm job: final v2 compact GPT-2 vs Gemma comparison plots.
#SBATCH --job-name=compare_v2c
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

echo "[compare-v2-compact] project_root=$PROJECT_ROOT"
echo "[compare-v2-compact] python=$(which python)"

python scripts/plot_5class_model_comparison.py \
  --gpt2-generated outputs/generated/gpt2_5class_v2_compact_ckpt1300_reparsed/generation_summary.json \
  --gemma-generated outputs/generated/gemma_5class_v2/generation_summary.json \
  --gpt2-utility outputs/evaluation/utility/gpt2_5class_v2_compact/utility_metrics.csv \
  --gemma-utility outputs/evaluation/utility/gemma_5class_v2/utility_metrics.csv \
  --gpt2-similarity outputs/evaluation/similarity/gpt2_5class_v2_compact/similarity_metrics.csv \
  --gemma-similarity outputs/evaluation/similarity/gemma_5class_v2/similarity_metrics.csv \
  --output-dir outputs/plots/model_comparison_5class_v2_compact
