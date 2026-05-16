# Evaluation Guide

## Similarity evaluation

```bash
python scripts/evaluate_similarity.py \
  --config config/pamap2_sdforger_gpt2.yaml \
  --real-windows artifacts/pamap2_sdforger_dataset/test_windows.npy \
  --real-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv \
  --synthetic-windows outputs/generated/gpt2/generated_windows.npy \
  --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv \
  --output-dir outputs/evaluation/similarity/gpt2
```

Metrics currently implemented:

- `MDD`
- `ACD`
- `SD`
- `KD`
- `ED`
- `DTW`

## Utility evaluation

```bash
python scripts/evaluate_utility.py \
  --config config/pamap2_sdforger_gpt2.yaml \
  --real-train-windows artifacts/pamap2_sdforger_dataset/train_windows.npy \
  --real-train-metadata artifacts/pamap2_sdforger_dataset/train_metadata.csv \
  --real-test-windows artifacts/pamap2_sdforger_dataset/test_windows.npy \
  --real-test-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv \
  --synthetic-windows outputs/generated/gpt2/generated_windows.npy \
  --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv \
  --output-dir outputs/evaluation/utility/gpt2
```

Utility settings currently implemented:

- `real_only`
- `synthetic_only`
- `real_plus_synthetic`

Classifier:

- `RandomForestClassifier`

Outputs:

- metrics CSV
- confusion matrices as CSV and PNG
- summary plots
