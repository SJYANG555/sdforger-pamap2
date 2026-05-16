# GPT-2 Smoke Test

This smoke test is separate from the main experiment pipeline. It is intended to validate that the **full GPT-2 SDForger-style flow can run on Triton** with minimal cost.

## Added smoke-specific files

- `config/pamap2_sdforger_gpt2_smoke.yaml`
- `scripts/build_pamap2_sdforger_smoke_dataset.py`
- `scripts/run_gpt2_smoke_test.py`
- `slurm/run_full_gpt2_smoke_test.slurm`

## What the smoke test does

1. Builds a **tiny subset** from the existing `artifacts/pamap2_sdforger_dataset`
2. Trains GPT-2 for a minimal run
3. Saves a checkpoint
4. Generates a small number of synthetic embeddings/windows
5. Runs similarity evaluation
6. Runs utility evaluation
7. Generates a few basic plots

## Recommended Triton command

```bash
sbatch slurm/run_full_gpt2_smoke_test.slurm
```

Before submitting, update:

- `#SBATCH --account=YOUR_TRITON_ACCOUNT`
- `#SBATCH --partition=...`
- `PROJECT_ROOT`
- `CONDA_ENV_NAME`

## Manual commands

```bash
python scripts/build_pamap2_sdforger_smoke_dataset.py --config config/pamap2_sdforger_gpt2_smoke.yaml
python scripts/train_pamap2_lm.py --config config/pamap2_sdforger_gpt2_smoke.yaml
python scripts/generate_pamap2_synthetic.py --config config/pamap2_sdforger_gpt2_smoke.yaml --model-path outputs/checkpoints/gpt2_smoke/best --output-dir outputs/generated/gpt2_smoke
python scripts/evaluate_similarity.py --config config/pamap2_sdforger_gpt2_smoke.yaml --real-windows artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy --real-metadata artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv --synthetic-windows outputs/generated/gpt2_smoke/generated_windows.npy --synthetic-metadata outputs/generated/gpt2_smoke/generated_embeddings.csv --output-dir outputs/evaluation/similarity/gpt2_smoke
python scripts/evaluate_utility.py --config config/pamap2_sdforger_gpt2_smoke.yaml --real-train-windows artifacts/pamap2_sdforger_dataset_smoke/train_windows.npy --real-train-metadata artifacts/pamap2_sdforger_dataset_smoke/train_metadata.csv --real-test-windows artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy --real-test-metadata artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv --synthetic-windows outputs/generated/gpt2_smoke/generated_windows.npy --synthetic-metadata outputs/generated/gpt2_smoke/generated_embeddings.csv --output-dir outputs/evaluation/utility/gpt2_smoke
python scripts/make_plots.py --config config/pamap2_sdforger_gpt2_smoke.yaml --training-log-csv outputs/checkpoints/gpt2_smoke/training_log_history.csv --real-windows artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy --real-metadata artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv --synthetic-windows outputs/generated/gpt2_smoke/generated_windows.npy --synthetic-metadata outputs/generated/gpt2_smoke/generated_embeddings.csv --real-embeddings artifacts/pamap2_sdforger_dataset_smoke/test_embeddings.csv --synthetic-embeddings outputs/generated/gpt2_smoke/generated_embeddings.csv --output-dir outputs/plots/gpt2_smoke
```

## Success criteria

The smoke test is considered successful only if all of the following are true:

1. GPT-2 training starts and completes at least one minimal training loop
2. A checkpoint is saved under `outputs/checkpoints/gpt2_smoke/`
3. Generation runs from that checkpoint
4. Generated text is parsed into embeddings
5. Embeddings are decoded into windows
6. Similarity evaluation writes result files
7. Utility evaluation writes result files
8. At least 1-2 plots are created
9. The full job can be submitted and completed on Triton via Slurm

## How to judge success quickly

Check that these files exist after the run:

- `artifacts/pamap2_sdforger_dataset_smoke/smoke_manifest.json`
- `outputs/checkpoints/gpt2_smoke/best/`
- `outputs/generated/gpt2_smoke/generated_embeddings.csv`
- `outputs/generated/gpt2_smoke/generated_windows.npy`
- `outputs/evaluation/similarity/gpt2_smoke/similarity_metrics.csv`
- `outputs/evaluation/utility/gpt2_smoke/utility_metrics.csv`
- `outputs/plots/gpt2_smoke/`

## Scope limitations

- This smoke test validates **pipeline execution**, not model quality.
- It only covers **GPT-2**.
- Gemma is intentionally excluded from this smoke path.
