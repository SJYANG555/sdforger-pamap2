# Triton Deployment Guide

This project is organized for **Linux + Slurm + NVIDIA GPU** execution on Aalto Triton.

## 1. Prepare the workspace

Copy the whole repository to your Triton `$WRKDIR`, for example:

```bash
cd $WRKDIR
git clone <your-repo-url> ChatTS
cd ChatTS
```

Place PAMAP2 under:

```text
pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol
```

## 2. Create the environment

Option A:

```bash
conda env create -f environment.yml
conda activate pamap2-sdforger
```

Option B:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Gemma 2 2B PEFT training you need:

- access to `google/gemma-2-2b`
- Hugging Face authentication if the model requires it

## 3. Recommended pipeline

1. Preprocess PAMAP2 windows
2. Build SDForger-style dataset
3. Train GPT-2 baseline
4. Generate synthetic embeddings/windows
5. Evaluate similarity
6. Evaluate HAR utility
7. Switch config to Gemma 2 2B and repeat

## 4. Slurm entrypoints

Available scripts:

- `slurm/run_preprocess.slurm`
- `slurm/run_build_dataset.slurm`
- `slurm/run_train_gpt2.slurm`
- `slurm/run_train_gemma2.slurm`
- `slurm/run_generate.slurm`
- `slurm/run_evaluate.slurm`

Edit these fields before first use:

- `#SBATCH --account=YOUR_TRITON_ACCOUNT`
- `#SBATCH --partition=...`
- `PROJECT_ROOT`
- `DATA_DIR`
- `MODEL_PATH` for generation/evaluation

## 5. Manual command-line flow

```bash
python scripts/preprocess_pamap2.py --data-dir pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol --output-dir artifacts/pamap2_baseline_protocol_5act --activities walking running cycling sitting standing --window-size 256 --stride 128 --save-raw-tensor
python scripts/build_pamap2_sdforger_dataset.py --config config/pamap2_sdforger_gpt2.yaml
python scripts/train_pamap2_lm.py --config config/pamap2_sdforger_gpt2.yaml
python scripts/generate_pamap2_synthetic.py --config config/pamap2_sdforger_gpt2.yaml --model-path outputs/checkpoints/gpt2/best --output-dir outputs/generated/gpt2
python scripts/evaluate_similarity.py --config config/pamap2_sdforger_gpt2.yaml --real-windows artifacts/pamap2_sdforger_dataset/test_windows.npy --real-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv --synthetic-windows outputs/generated/gpt2/generated_windows.npy --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv --output-dir outputs/evaluation/similarity/gpt2
python scripts/evaluate_utility.py --config config/pamap2_sdforger_gpt2.yaml --real-train-windows artifacts/pamap2_sdforger_dataset/train_windows.npy --real-train-metadata artifacts/pamap2_sdforger_dataset/train_metadata.csv --real-test-windows artifacts/pamap2_sdforger_dataset/test_windows.npy --real-test-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv --synthetic-windows outputs/generated/gpt2/generated_windows.npy --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv --output-dir outputs/evaluation/utility/gpt2
```

## 6. Validation status

The **preprocessing** and **dataset build** stages were executed locally in this repository.

The following stages are coded for Triton but not executed in the current local environment because the current Python runtime is missing training dependencies such as `torch`, `accelerate`, and `peft`:

- model training
- LM-based generation
- Triton Slurm execution

