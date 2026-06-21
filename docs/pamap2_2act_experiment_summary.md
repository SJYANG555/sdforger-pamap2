# PAMAP2 Two-Activity Gemma Experiments

This document summarizes the PAMAP2 walking/running synthetic time-series experiments based on Gemma 2 2B LoRA.

## GitHub Sync Scope

The repository remote is:

```text
git@github.com:SJYANG555/ChatTS.git
```

The synchronized scope should include:

- source code for the PAMAP2 SDForger-style pipeline;
- two-activity Gemma configuration files;
- Slurm scripts used for dataset build, training, generation, evaluation, plotting, and summary;
- final summary CSV files under `outputs/evaluation/summary/`;
- this documentation file.

The full generated windows, checkpoints, dataset artifacts, and all plot bundles should not be committed by default because they are large:

- `outputs/checkpoints/`: tens of GB;
- `outputs/generated/`: hundreds of MB;
- `outputs/plots/`: hundreds of MB;
- `artifacts/`: regenerated datasets.

If figures are needed for GitHub, only a few representative PNGs should be selected manually, rather than committing all plot outputs.

## Main Experiment Stages

1. **Two-activity setup**
   - Keep only `walking` and `running`.
   - This reduces the ambiguity from the original five-class PAMAP2 setting.

2. **Sensor/channel selection**
   - `full`
   - `hand`
   - `chest`
   - `hand_acc`
   - `chest_acc`
   - `hand_gyro`
   - `chest_gyro`

3. **Epoch scaling without statistical prompt**
   - Tested selected settings at 10 and 20 epochs.
   - More epochs did not consistently improve utility or validity.

4. **ICA8 embedding test**
   - Tried higher ICA dimensionality.
   - It did not provide consistent benefit, so it was not continued as the main direction.

5. **Statistical prompt**
   - Added window-level statistics to the text prompt.
   - Statistics used:
     - `mean`
     - `std`
     - `min`
     - `max`
   - Precision: 3 decimal places.
   - These statistics are computed per window and per selected sensor channel.

6. **Statistical prompt + epoch scaling**
   - Tested 5, 10, and 20 epochs for selected strong configurations:
     - `hand + stats`
     - `chest_gyro + stats`
     - `chest + stats`
     - `chest_acc + stats`

## Real-vs-Real Similarity Baseline

The real baseline was computed by splitting the real test windows into two random halves within each activity and computing similarity between the two real subsets.

| Base dataset | Real ED | Real DTW |
|---|---:|---:|
| full | 81.13 | 65.95 |
| hand | 46.66 | 61.03 |
| chest | 27.80 | 31.83 |
| hand_acc | 45.35 | 100.34 |
| chest_acc | 27.61 | 56.20 |
| hand_gyro | 10.91 | 21.73 |
| chest_gyro | 3.21 | 7.47 |

Ratios below 1.0 mean the synthetic-vs-real distance is lower than this random real-vs-real baseline.

## Baseline Seven-Variant Results

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

## Epoch Extension Without Statistical Prompt

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

## Statistical Prompt + Epoch Scaling

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

## Important Output Files

Final summary tables:

```text
outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv
outputs/evaluation/summary/gemma_2act_stats_epoch_summary.csv
outputs/evaluation/summary/gemma_2act_13run_summary.csv
outputs/evaluation/summary/real_vs_real_baseline_similarity.csv
outputs/evaluation/summary/pamap2_results_slides.tex
```

These summary files are small and should be committed to GitHub using force-add because `outputs/` is ignored by `.gitignore`.

## Recommendation on Plot Synchronization

Do not commit all generated plots. They are large and mostly redundant.

Recommended alternatives:

1. Keep all plots on Triton.
2. Commit only 4--8 representative PNGs if they are needed in the report.
3. Prefer committing the plotting scripts and summary CSV files so figures can be regenerated.

Good candidates for representative plots:

- real vs generated curves for `hand + stats, 5 epochs`;
- real vs generated curves for `chest_gyro + stats, 5 epochs`;
- one baseline comparison such as `hand` or `chest_gyro`;
- one failure/imbalance example if needed.
