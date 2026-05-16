#!/bin/bash -l
# Adjust account/partition/path variables for your Triton project before use.
#SBATCH --job-name=pamap2_preprocess
#SBATCH --account=YOUR_TRITON_ACCOUNT
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pamap2-sdforger}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs
source ~/.bashrc
conda activate "$CONDA_ENV_NAME"

python scripts/preprocess_pamap2.py \
  --data-dir "$DATA_DIR" \
  --output-dir artifacts/pamap2_baseline_protocol_5act \
  --activities walking running cycling sitting standing \
  --window-size 256 \
  --stride 128 \
  --save-raw-tensor

