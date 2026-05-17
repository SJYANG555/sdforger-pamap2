# PAMAP2 SDForger Experiment Archive

This folder is the GitHub-friendly archive for the PAMAP2 + SDForger-style time-series generation experiments.

The full Triton workspace contains large generated assets under `artifacts/` and `outputs/`. Those directories are intentionally ignored by Git because they include model checkpoints, generated window tensors, raw PAMAP2 data, and other files that are too large for normal GitHub versioning. This archive keeps the lightweight experiment evidence needed for review and reproduction.

## What is included

- `pamap2_forger/`: the PAMAP2 SDForger-style data, embedding, generation, training, metric, and plotting modules.
- `scripts/`: command-line entry points for dataset building, training, generation, evaluation, and report plotting.
- `config/`: experiment configs for GPT-2, Gemma 2 2B, Llama 3.2 3B, sensor subsets, and related comparison runs.
- `slurm/`: Triton Slurm job scripts for CPU preprocessing/evaluation and GPU training/generation.
- `experiments/results/artifacts/`: dataset manifests, split manifests, reconstruction metadata, and resolved configs.
- `experiments/results/outputs/checkpoints/`: training summaries and resolved training configs.
- `experiments/results/outputs/generated/`: generation summaries, prompt summaries, generated embedding CSVs, parser/filter debug CSVs, and resolved configs.
- `experiments/results/outputs/evaluation/`: similarity, utility, and condition-consistency metrics and summaries.
- `experiments/results/outputs/plots/`: generated figures and plot manifests.
- `experiments/results/outputs/logs/`: Slurm stdout/stderr logs from the recorded runs.

## What is not included

Large binary outputs remain on Triton and are not copied into this archive:

- model checkpoints and optimizer states under `outputs/checkpoints/*/`
- generated window tensors such as `generated_windows.npy`
- real dataset window tensors such as `train_windows.npy`, `val_windows.npy`, and `test_windows.npy`
- raw PAMAP2 data under `pamap2+physical+activity+monitoring/`
- local environments such as `.conda-env/` and `.mamba-root/`

If those assets need to be shared later, use Git LFS, a GitHub Release, or an external artifact store rather than regular Git history.

## Pipeline Overview

This repository archives the code pipeline used for the PAMAP2 + SDForger-style time-series generation experiments:

1. Build PAMAP2 windows with `scripts/build_pamap2_sdforger_dataset.py`.
   The dataset code in `pamap2_forger/dataset.py` selects the target activities, sensor channels, window size, stride, and train/validation/test split.
2. Convert each window into SDForger-style descriptors.
   `pamap2_forger/embeddings.py` extracts compact statistical embeddings and `pamap2_forger/text.py` serializes them into text prompts for causal language models.
3. Fine-tune a language model with `scripts/train_pamap2_lm.py`.
   `pamap2_forger/trainer.py` supports the GPT-2 baseline and PEFT/LoRA-style larger model runs such as Gemma 2 2B and Llama 3.2 3B.
4. Generate synthetic embedding text with `scripts/generate_pamap2_synthetic.py`.
   `pamap2_forger/generate.py` parses generated text, filters invalid candidates, and reconstructs synthetic windows through `pamap2_forger/reconstruction.py`.
5. Evaluate synthetic data quality.
   `scripts/evaluate_similarity.py` computes SDForger-style similarity metrics (`MDD`, `ACD`, `SD`, `KD`, `ED`, `DTW`), and `scripts/evaluate_utility.py` evaluates downstream HAR utility with real-only, synthetic-only, and real+synthetic RandomForest classifiers.
6. Produce report figures.
   `scripts/make_plots.py`, `scripts/plot_5class_model_comparison.py`, and `scripts/plot_report_model_comparison.py` generate training curves, utility plots, model comparisons, and similarity matrices.

The main formal comparison uses five PAMAP2 activities: cycling, running, sitting, standing, and walking. The current report-ready comparison covers the 18-channel full-body setting and the 12-channel hand+chest setting across GPT-2, Gemma 2 2B, and Llama 3.2 3B.

CPU jobs are used for dataset building, similarity evaluation, utility evaluation, and plotting. GPU jobs are used for language-model training and generation; Triton GPU Slurm scripts request GPUs with `#SBATCH --gpus=1`.

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
- `llama32_3b_5class_v2`
- `llama32_3b_5class_v2_hand_chest`

Generation and evaluation folders also include earlier comparison runs such as `gpt2_5class`, `gemma_5class`, compact checkpoint reparsing, mock outputs, and KNN condition-consistency comparisons when available.

## Formal Experiment Metrics

The table below summarizes formal GPT-2, Gemma, and Llama experiment runs archived in `experiments/results`. Smoke tests, mock outputs, and empty-synthetic checks are excluded. Similarity values come from each run's `overall_mean` row in `similarity_metrics.csv`; utility values come from `utility_metrics.csv`.

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
| `llama32_3b_5class_v2` | 844 | 98.5% | 0.0% | 300.361 | 0.842 | 0.433 | 0.929 |
| `llama32_3b_5class_v2_hand_chest` | 847 | 98.8% | 0.0% | 304.369 | 0.921 | 0.296 | 0.952 |

Notes:

- Lower `DTW` and `MDD` are better for similarity.
- Higher utility accuracy is better. `synthetic_only` trains the HAR classifier only on synthetic windows; `real_plus_synthetic` augments real training windows with synthetic windows.
- The compact GPT-2 checkpoint initially generated zero embeddings; the reparsed archive recovered 835 usable embeddings and shares the compact evaluation metrics.
- Report-ready GPT-2/Gemma/Llama comparison figures are archived under `experiments/results/outputs/plots/report_model_comparison_5class_v2/` for the 18-channel full-body and 12-channel hand+chest five-class settings.
- Llama generation used longer strong-GPU Slurm retries after the initial V100 generation jobs reached the time limit.

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
