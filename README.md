# PAMAP2 SDForger Experiment Archive

This folder is the GitHub-friendly archive for the PAMAP2 + SDForger-style time-series generation experiments.

The full Triton workspace contains large generated assets under `artifacts/` and `outputs/`. Those directories are intentionally ignored by Git because they include model checkpoints, generated window tensors, raw PAMAP2 data, and other files that are too large for normal GitHub versioning. This archive keeps the lightweight experiment evidence needed for review and reproduction.

## What is included

- `results/artifacts/`: dataset manifests, split manifests, reconstruction metadata, and resolved configs.
- `results/outputs/checkpoints/`: training summaries and resolved training configs.
- `results/outputs/generated/`: generation summaries, prompt summaries, generated embedding CSVs, parser/filter debug CSVs, and resolved configs.
- `results/outputs/evaluation/`: similarity, utility, and condition-consistency metrics and summaries.
- `results/outputs/plots/`: generated figures and plot manifests.
- `results/outputs/logs/`: Slurm stdout/stderr logs from the recorded runs.

## What is not included

Large binary outputs remain on Triton and are not copied into this archive:

- model checkpoints and optimizer states under `outputs/checkpoints/*/`
- generated window tensors such as `generated_windows.npy`
- real dataset window tensors such as `train_windows.npy`, `val_windows.npy`, and `test_windows.npy`
- raw PAMAP2 data under `pamap2+physical+activity+monitoring/`
- local environments such as `.conda-env/` and `.mamba-root/`

If those assets need to be shared later, use Git LFS, a GitHub Release, or an external artifact store rather than regular Git history.

## Recorded experiment families

Training summaries are archived for:

- `gpt2`
- `gpt2_smoke`
- `gpt2_5class_v2`
- `gpt2_5class_v2_compact`
- `gpt2_5class_v2_hand`
- `gpt2_5class_v2_chest`
- `gpt2_5class_v2_hand_chest`
- `gemma`
- `gemma_5class_v2`
- `gemma_5class_v2_hand`
- `gemma_5class_v2_chest`
- `gemma_5class_v2_hand_chest`

Generation and evaluation folders also include earlier comparison runs such as `gpt2_5class`, `gemma_5class`, compact checkpoint reparsing, mock outputs, and KNN condition-consistency comparisons when available.

## Formal Experiment Metrics

The table below summarizes formal GPT-2 and Gemma experiment runs archived in `experiments/results`. Smoke tests, mock outputs, and empty-synthetic checks are excluded. Similarity values come from each run's `overall_mean` row in `similarity_metrics.csv`; utility values come from `utility_metrics.csv`.

| Experiment | Generated kept | Retention | Duplicate ratio | Overall DTW | Overall MDD | Synthetic-only acc | Real+synthetic acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt2` | 256 | 100.0% | 0.0% | 333.136 | 0.595 | 0.319 | 0.915 |
| `gemma` | 227 | 88.7% | 0.0% | 314.328 | 0.528 | 0.400 | 0.902 |
| `gpt2_5class` | 856 | 99.9% | 0.0% | 443.852 | 1.120 | 0.305 | 0.924 |
| `gemma_5class` | 636 | 74.2% | 0.0% | 344.462 | 0.751 | 0.347 | 0.915 |
| `gpt2_5class_v2` | 0 | 0.0% | - | - | - | - | - |
| `gemma_5class_v2` | 857 | 100.0% | 0.0% | 282.136 | 0.811 | 0.350 | 0.912 |
| `gpt2_5class_v2_compact_ckpt1300` | 0 | 0.0% | 0.0% | 338.688 | 0.861 | 0.488 | 0.931 |
| `gpt2_5class_v2_compact_ckpt1300_reparsed` | 835 | 97.4% | 0.0% | 338.688 | 0.861 | 0.488 | 0.931 |
| `gpt2_5class_v2_hand` | 57 | 6.7% | 0.0% | 634.814 | 1.739 | 0.677 | 0.923 |
| `gemma_5class_v2_hand` | 362 | 42.2% | 0.0% | 498.475 | 1.884 | 0.258 | 0.925 |
| `gpt2_5class_v2_chest` | 13 | 1.5% | 0.0% | 178.294 | 0.248 | 0.384 | 0.942 |
| `gemma_5class_v2_chest` | 695 | 81.1% | 0.0% | 109.989 | 0.363 | 0.138 | 0.947 |
| `gpt2_5class_v2_hand_chest` | 749 | 87.4% | 0.0% | 412.565 | 1.401 | 0.705 | 0.921 |
| `gemma_5class_v2_hand_chest` | 857 | 100.0% | 0.0% | 289.192 | 0.873 | 0.600 | 0.957 |

Notes:

- Lower `DTW` and `MDD` are better for similarity.
- Higher utility accuracy is better. `synthetic_only` trains the HAR classifier only on synthetic windows; `real_plus_synthetic` augments real training windows with synthetic windows.
- The compact GPT-2 checkpoint initially generated zero embeddings; the reparsed archive recovered 835 usable embeddings and shares the compact evaluation metrics.

### KNN Condition Consistency

| Evaluation set | Source | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| 5-class | `real_test` | 857 | 0.778 | 0.758 | 0.749 |
| 5-class | `gpt2_5class` | 856 | 0.217 | 0.118 | 0.125 |
| 5-class | `gemma_5class` | 636 | 0.354 | 0.196 | 0.240 |
| 5-class v2 compact | `real_test` | 857 | 0.778 | 0.758 | 0.749 |
| 5-class v2 compact | `gpt2_5class` | 835 | 0.660 | 0.618 | 0.634 |
| 5-class v2 compact | `gemma_5class` | 857 | 0.635 | 0.521 | 0.548 |

## Reproduction entry points

Assuming you are already on Triton:

```bash
cd /scratch/work/yangs9/ChatTS
module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env
export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"
```

Main GPT-2 baseline flow:

```bash
sbatch slurm/run_build_dataset.slurm
sbatch slurm/run_train_gpt2_gpu_ready.slurm
sbatch slurm/run_generate.slurm
sbatch slurm/run_evaluate.slurm
```

GPU training and generation scripts must include:

```bash
#SBATCH --gpus=1
```

Check status and logs with:

```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
tail -f outputs/logs/<job-log>.out
cat outputs/logs/<job-log>.err
```

## Two-Activity Gemma Results: Walking/Running

The latest experiment round focuses on a two-activity PAMAP2 setting with only `walking` and `running`.

Model and training setup:

- model: `google/gemma-2-2b`
- fine-tuning: LoRA/PEFT
- base model: frozen
- LoRA scope: last 30% layers
- window size: 256
- stride: 128
- main improvements tested:
  - using only two activities;
  - sensor/channel selection;
  - 10/20 epoch extension;
  - ICA8 embedding dimension test;
  - statistical prompt input;
  - statistical prompt with 5/10/20 epochs.

Final summary files are archived under:

```text
results/outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv
results/outputs/evaluation/summary/gemma_2act_stats_epoch_summary.csv
results/outputs/evaluation/summary/gemma_2act_13run_summary.csv
results/outputs/evaluation/summary/real_vs_real_baseline_similarity.csv
results/outputs/evaluation/summary/pamap2_results_slides.tex
```

Detailed notes are also available in:

```text
docs/pamap2_2act_experiment_summary.md
```

### Statistical Prompt

The statistical prompt adds window-level summary statistics to the text prompt before generation. The statistics are computed per selected sensor channel:

- `mean`
- `std`
- `min`
- `max`

Values are rounded to 3 decimal places. This provides amplitude and variability guidance in addition to activity and metadata conditions.

### Real-vs-Real Similarity Baseline

The real baseline was computed by splitting the real test windows into two random halves within each activity and computing real-vs-real similarity.

| Base dataset | Real ED | Real DTW |
|---|---:|---:|
| full | 81.13 | 65.95 |
| hand | 46.66 | 61.03 |
| chest | 27.80 | 31.83 |
| hand_acc | 45.35 | 100.34 |
| chest_acc | 27.61 | 56.20 |
| hand_gyro | 10.91 | 21.73 |
| chest_gyro | 3.21 | 7.47 |

### Baseline Seven-Variant Results

| Variant | Valid | Running | Walking | ED | DTW | ED/real | DTW/real | Synthetic-only acc | Real+Syn acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 83/305 | 27 | 56 | 250.19 | 374.43 | 3.08 | 5.68 | 0.390 | 0.984 |
| hand | 226/305 | 82 | 144 | 194.01 | 620.21 | 4.16 | 10.16 | 0.921 | 0.977 |
| chest | 213/305 | 102 | 111 | 23.98 | 70.21 | 0.86 | 2.21 | 0.384 | 0.964 |
| hand_acc | 181/305 | 71 | 110 | 193.85 | 1130.98 | 4.27 | 11.27 | 0.387 | 0.961 |
| chest_acc | 274/305 | 110 | 164 | 22.81 | 135.44 | 0.83 | 2.41 | 0.220 | 0.964 |
| hand_gyro | 202/305 | 94 | 108 | 12.04 | 84.79 | 1.10 | 3.90 | 0.387 | 0.689 |
| chest_gyro | 249/305 | 101 | 148 | 2.32 | 8.44 | 0.72 | 1.13 | 0.374 | 0.954 |

Key observation:

- `hand` gives the strongest downstream utility.
- `chest_gyro` gives the strongest similarity.

### Epoch Extension Without Statistical Prompt

| Variant | Epoch | Valid | Running | Walking | ED/real | DTW/real | Synthetic-only acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| hand | 5 | 226/305 | 82 | 144 | 4.16 | 10.16 | 0.921 |
| hand | 10 | 75/305 | 39 | 36 | 4.58 | 11.11 | 0.820 |
| hand | 20 | 139/305 | 52 | 87 | 4.38 | 10.75 | 0.377 |
| chest_gyro | 5 | 249/305 | 101 | 148 | 0.72 | 1.13 | 0.374 |
| chest_gyro | 10 | 294/305 | 115 | 179 | 0.45 | 0.74 | 0.370 |
| chest_gyro | 20 | 301/305 | 114 | 187 | 0.65 | 1.06 | 0.370 |
| chest | 5 | 213/305 | 102 | 111 | 0.86 | 2.21 | 0.384 |
| chest | 10 | 298/305 | 110 | 188 | 0.94 | 1.68 | 0.380 |
| chest | 20 | 74/305 | 28 | 46 | 1.03 | 2.90 | 0.626 |

More epochs alone did not produce a reliable improvement.

### Statistical Prompt + Epoch Scaling

| Variant | Epoch | Valid | Running | Walking | ED/real | DTW/real | Synthetic-only acc | Real+Syn acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hand + stats | 5 | 301/305 | 113 | 188 | 3.91 | 9.29 | 0.928 | 0.970 |
| hand + stats | 10 | 261/305 | 107 | 154 | 4.56 | 10.38 | 0.846 | 0.970 |
| hand + stats | 20 | 284/305 | 104 | 180 | 4.44 | 10.35 | 0.921 | 0.957 |
| chest_gyro + stats | 5 | 283/305 | 112 | 171 | 0.47 | 0.74 | 0.449 | 0.954 |
| chest_gyro + stats | 10 | 303/305 | 115 | 188 | 0.52 | 0.80 | 0.374 | 0.957 |
| chest_gyro + stats | 20 | 271/305 | 84 | 187 | 0.62 | 1.02 | 0.564 | 0.957 |
| chest + stats | 5 | 270/305 | 107 | 163 | 0.69 | 1.56 | 0.370 | 0.967 |
| chest + stats | 10 | 303/305 | 114 | 189 | 0.78 | 1.83 | 0.370 | 0.964 |
| chest + stats | 20 | 222/305 | 98 | 124 | 0.99 | 1.83 | 0.374 | 0.967 |
| chest_acc + stats | 5 | 280/305 | 101 | 179 | 0.86 | 2.31 | 0.370 | 0.967 |
| chest_acc + stats | 10 | 284/305 | 100 | 184 | 1.17 | 2.39 | 0.475 | 0.967 |
| chest_acc + stats | 20 | 294/305 | 104 | 190 | 0.72 | 1.84 | 0.377 | 0.967 |

Key observation:

- `hand + stats, 5 epochs` gives the best utility-oriented result.
- `chest_gyro + stats, 5 epochs` gives the best time-series similarity.
- `chest_gyro + stats, 20 epochs` improves synthetic-only utility while retaining relatively strong similarity.
- Longer training changes the trade-off but does not dominate 5 epochs.

### Plot Synchronization

All plot outputs are intentionally not synchronized. The full plot folder is large and mostly redundant.

Recommended policy:

1. Keep all plot outputs on Triton.
2. Commit only a few representative PNGs if they are needed for a report.
3. Prefer committing plotting scripts and summary CSV files so plots can be regenerated.
