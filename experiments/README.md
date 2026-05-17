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
- `llama32_3b_5class_v2`
- `llama32_3b_5class_v2_hand_chest`

Generation and evaluation folders also include earlier comparison runs such as `gpt2_5class`, `gemma_5class`, compact checkpoint reparsing, mock outputs, and KNN condition-consistency comparisons when available.

## Llama 3.2 comparison addendum

The Llama 3.2 3B five-class runs are archived in the same format as the earlier GPT-2 and Gemma experiments:

- 18-channel full-body: `experiments/results/outputs/{checkpoints,generated,evaluation}/.../llama32_3b_5class_v2`
- 12-channel hand+chest: `experiments/results/outputs/{checkpoints,generated,evaluation}/.../llama32_3b_5class_v2_hand_chest`
- report figures and summary tables: `experiments/results/outputs/plots/report_model_comparison_5class_v2/`

Both Llama settings use the same five activities as the prior formal runs: cycling, running, sitting, standing, and walking.

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
