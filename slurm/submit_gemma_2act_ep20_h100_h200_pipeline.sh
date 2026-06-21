#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/yangs9/ChatTS}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"
GPU_PARTITIONS="${GPU_PARTITIONS:-gpu-h200-141g-m,gpu-h100-80g}"
GPU_TIME="${GPU_TIME:-08:00:00}"

submit_variant() {
  local variant="$1"
  local config_path="config/pamap2_sdforger_gemma2_2b_2act_${variant}_ep20.yaml"
  local dataset_dir="artifacts/pamap2_sdforger_dataset_2act_${variant}"
  local checkpoint_dir="outputs/checkpoints/gemma_2act_${variant}_ep20"
  local generated_dir="outputs/generated/gemma_2act_${variant}_ep20"
  local similarity_dir="outputs/evaluation/similarity/gemma_2act_${variant}_ep20"
  local utility_dir="outputs/evaluation/utility/gemma_2act_${variant}_ep20"
  local plots_dir="outputs/plots/gemma_2act_${variant}_ep20"

  local train_job
  train_job="$(sbatch --parsable \
    --partition="${GPU_PARTITIONS}" \
    --time="${GPU_TIME}" \
    --job-name="gemma_${variant}_ep20_train" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH="${config_path}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    slurm/run_train_gemma2.slurm)"

  local generate_job
  generate_job="$(sbatch --parsable \
    --dependency=afterok:"${train_job}" \
    --partition="${GPU_PARTITIONS}" \
    --time="${GPU_TIME}" \
    --job-name="gemma_${variant}_ep20_gen" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH="${config_path}",MODEL_PATH="${checkpoint_dir}/best",OUTPUT_DIR="${generated_dir}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    slurm/run_generate_gemma.slurm)"

  local evaluate_job
  evaluate_job="$(sbatch --parsable \
    --dependency=afterok:"${generate_job}" \
    --job-name="gemma_${variant}_ep20_eval" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH="${config_path}",DATASET_DIR="${dataset_dir}",GENERATED_DIR="${generated_dir}",CHECKPOINT_DIR="${checkpoint_dir}",SIMILARITY_DIR="${similarity_dir}",UTILITY_DIR="${utility_dir}",PLOTS_DIR="${plots_dir}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    slurm/run_evaluate_gemma_variant.slurm)"

  printf '%s\ttrain=%s\tgenerate=%s\tevaluate=%s\n' "${variant}" "${train_job}" "${generate_job}" "${evaluate_job}"
}

cd "${PROJECT_ROOT}"
mkdir -p outputs/logs

submit_variant hand
submit_variant chest_gyro
submit_variant chest
