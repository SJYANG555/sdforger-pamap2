#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/yangs9/ChatTS}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/work/yangs9/ChatTS/.conda-env}"

declare -A CONFIGS=(
  [hand_stats]="config/pamap2_sdforger_gemma2_2b_2act_hand_stats.yaml"
  [chest_gyro_stats]="config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_stats.yaml"
  [hand_ica8]="config/pamap2_sdforger_gemma2_2b_2act_hand_ica8.yaml"
  [chest_gyro_ica8]="config/pamap2_sdforger_gemma2_2b_2act_chest_gyro_ica8.yaml"
)

declare -A DATASETS=(
  [hand_stats]="artifacts/pamap2_sdforger_dataset_2act_hand_stats"
  [chest_gyro_stats]="artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats"
  [hand_ica8]="artifacts/pamap2_sdforger_dataset_2act_hand_ica8"
  [chest_gyro_ica8]="artifacts/pamap2_sdforger_dataset_2act_chest_gyro_ica8"
)

declare -A RUN_NAMES=(
  [hand_stats]="gemma_2act_hand_stats"
  [chest_gyro_stats]="gemma_2act_chest_gyro_stats"
  [hand_ica8]="gemma_2act_hand_ica8"
  [chest_gyro_ica8]="gemma_2act_chest_gyro_ica8"
)

submit_variant() {
  local variant="$1"
  local config_path="${CONFIGS[$variant]}"
  local run_name="${RUN_NAMES[$variant]}"
  local checkpoint_dir="outputs/checkpoints/${run_name}"
  local generated_dir="outputs/generated/${run_name}"

  local train_job
  train_job="$(sbatch --parsable \
    --dependency=afterok:"${BUILD_JOB}" \
    --job-name="${run_name}_train" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH="${config_path}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    slurm/run_train_gemma2.slurm)"

  local generate_job
  generate_job="$(sbatch --parsable \
    --dependency=afterok:"${train_job}" \
    --job-name="${run_name}_gen" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH="${config_path}",MODEL_PATH="${checkpoint_dir}/best",OUTPUT_DIR="${generated_dir}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    slurm/run_generate_gemma.slurm)"

  GENERATE_JOBS+=("${generate_job}")
  printf '%s\tbuild=%s\ttrain=%s\tgenerate=%s\n' "${run_name}" "${BUILD_JOB}" "${train_job}" "${generate_job}"
}

cd "${PROJECT_ROOT}"
mkdir -p outputs/logs

if [[ "${1:-}" == "--all" ]]; then
  variants=(hand_stats chest_gyro_stats hand_ica8 chest_gyro_ica8)
elif [[ "$#" -gt 0 ]]; then
  variants=("$@")
else
  variants=(hand_stats chest_gyro_stats)
fi

for variant in "${variants[@]}"; do
  if [[ -z "${CONFIGS[$variant]:-}" ]]; then
    echo "Unknown variant: ${variant}" >&2
    exit 2
  fi
done

variant_string="${variants[*]}"
BUILD_JOB="$(sbatch --parsable \
  --job-name="gemma_2act_next_build" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",VARIANTS="${variant_string}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
  slurm/run_build_gemma_2act_next4.slurm)"

GENERATE_JOBS=()
for variant in "${variants[@]}"; do
  submit_variant "${variant}"
done

generate_dependency="$(IFS=:; echo "${GENERATE_JOBS[*]}")"
EVALUATE_JOB="$(sbatch --parsable \
  --dependency=afterok:"${generate_dependency}" \
  --job-name="gemma_2act_next_eval" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",VARIANTS="${variant_string}",CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
  slurm/run_evaluate_gemma_2act_next4.slurm)"

printf 'combined_evaluate=%s\tdepends_on=%s\n' "${EVALUATE_JOB}" "${generate_dependency}"
