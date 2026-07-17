#!/bin/bash -l
# GPU Slurm job: generate MM-Fit synthetic windows with Gemma 2 2B LoRA.
#SBATCH --job-name=mmfit_gen_gemma
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/yangs9/ChatTS}"
CONFIG_PATH="${CONFIG_PATH:-config/mmfit_sdforger_gemma2_2b_5class.yaml}"
MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/mmfit_gemma2_2b_5class/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/generated/mmfit_gemma2_2b_5class}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJECT_ROOT/.mamba-root}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

echo "[generate-mmfit] config=$CONFIG_PATH"
echo "[generate-mmfit] model=$MODEL_PATH"
echo "[generate-mmfit] output=$OUTPUT_DIR"
nvidia-smi || true

python scripts/generate_mmfit_synthetic.py \
  --config "$CONFIG_PATH" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"

