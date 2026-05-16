# PAMAP2 SDForger Pipeline

## Alignment with official SDForger code

This implementation is intentionally aligned to the official IBM `fms-dgt` time-series builder flow:

- `README.md`: preprocessing -> embedding -> text -> LLM tuning -> generation -> decoding
- `trainer.py`: Hugging Face causal LM fine-tuning
- `utils.py`: embedding conversion and text templating
- `generate.py`: generate text, parse rows, filter, reconstruct
- `time_series.yaml`: config-driven model selection

## Local module mapping

- `pamap2_forger/dataset.py`
  Corresponds to SDForger's train-data preparation and dataset building.
- `pamap2_forger/embeddings.py`
  Corresponds to the embedding block, implemented for PAMAP2 window tensors.
- `pamap2_forger/text.py`
  Corresponds to `embeddings_to_text` / text parsing.
- `pamap2_forger/trainer.py`
  Corresponds to SDForger's Hugging Face training block, with GPT-2 and PEFT-enabled Gemma support.
- `pamap2_forger/generate.py`
  Corresponds to generation, parsing, filtering, and reconstruction.
- `pamap2_forger/metrics.py`
  Adds similarity-based and HAR utility-based evaluation for PAMAP2.
- `pamap2_forger/visualization.py`
  Generates loss curves, real-vs-synthetic plots, embedding scatter, and metric plots.

## Output locations

- `artifacts/`
  Preprocessed data and dataset-builder outputs.
- `outputs/checkpoints/`
  Fine-tuned model checkpoints.
- `outputs/generated/`
  Synthetic embeddings and decoded windows.
- `outputs/evaluation/`
  Similarity/utility CSVs and summaries.
- `outputs/plots/`
  Figures for training and evaluation.
