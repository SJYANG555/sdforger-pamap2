#!/bin/bash
#SBATCH --job-name=gemma_sd_filt
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-17%6
#SBATCH --output=outputs/logs/gemma_sd_filt-%A_%a.out
#SBATCH --error=outputs/logs/gemma_sd_filt-%A_%a.err

set -eo pipefail

cd /scratch/work/yangs9/ChatTS

module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env

export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"

mkdir -p outputs/logs
mkdir -p "${MPLCONFIGDIR}"

python scripts/recompute_similarity_from_resolved_config.py \
  --manifest outputs/evaluation/summary/gemma_sdforger_filtered_similarity_recompute_manifest.txt \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --compute-shr \
  --shr-channel-mode mean \
  --shr-max-iter 1000 \
  --shr-max-inner-iter 5
