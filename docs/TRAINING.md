# Training Guide

## GPT-2 baseline

```bash
python scripts/train_pamap2_lm.py --config config/pamap2_sdforger_gpt2.yaml
```

Outputs:

- `outputs/checkpoints/gpt2/best`
- `outputs/checkpoints/gpt2/latest`
- `outputs/checkpoints/gpt2/training_log_history.csv`
- `outputs/checkpoints/gpt2/training_summary.json`

## Gemma 2 2B with LoRA

```bash
python scripts/train_pamap2_lm.py --config config/pamap2_sdforger_gemma2_2b.yaml
```

This path uses PEFT/LoRA via the `training.peft` config section.

## Notes

- GPT-2 is configured as the baseline and uses full fine-tuning.
- Gemma 2 2B is configured with LoRA adapters.
- QLoRA is not fully wired yet; the code is structured so that quantized loading can be added later in `pamap2_forger/trainer.py`.
