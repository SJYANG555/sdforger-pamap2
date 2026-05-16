#!/bin/bash -l
#SBATCH --job-name=gen_gpt2_smoke_gpu
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=/scratch/work/yangs9/ChatTS/outputs/logs/%x-%j.out
#SBATCH --error=/scratch/work/yangs9/ChatTS/outputs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="/scratch/work/yangs9/ChatTS"
CONDA_ENV_PATH="/scratch/work/yangs9/ChatTS/.conda-env"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs outputs/generated/gpt2_smoke

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV_PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6"

echo "[generate-gpu] project_root=$PROJECT_ROOT"
echo "[generate-gpu] python=$(which python)"
python --version
nvidia-smi || true

srun python -u scripts/generate_pamap2_synthetic.py \
  --config config/pamap2_sdforger_gpt2_smoke.yaml \
  --model-path outputs/checkpoints/gpt2_smoke/best \
  --output-dir outputs/generated/gpt2_smoke

echo "[generate-gpu] completed"
