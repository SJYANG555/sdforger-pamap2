#!/bin/bash
#SBATCH --job-name=mmfit-build
#SBATCH --output=outputs/logs/mmfit-build-%j.out
#SBATCH --error=outputs/logs/mmfit-build-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /scratch/work/yangs9/ChatTS
mkdir -p outputs/logs

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env

export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"

python scripts/build_mmfit_sdforger_dataset.py --config config/mmfit_sdforger_gpt2_5class.yaml
