# PAMAP2 V2 Channel-Subset Comparison

## Key Takeaways

- Highest generation retention: Gemma 2 2B LoRA / Hand + chest (100.0%, 857/857).
- Highest synthetic-only HAR accuracy: GPT-2 / Hand + chest (0.7048).
- Lowest overall DTW: Gemma 2 2B LoRA / Chest (109.99).
- Best real+synthetic augmentation delta: Gemma 2 2B LoRA / Chest (+0.0023).

Blank metric cells mean no valid synthetic windows were available for evaluation.

## Summary Table

| model | subset | num_generated | retention_rate | synthetic_only_accuracy | real_plus_synthetic_accuracy | augmentation_accuracy_delta | overall_DTW | overall_ED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | Hand + chest + ankle | 0 | 0.0000 |  |  |  |  |  |
| GPT-2 | Hand | 57 | 0.0665 | 0.6768 | 0.9230 | -0.0035 | 634.8138 | 186.0952 |
| GPT-2 | Chest | 13 | 0.0152 | 0.3839 | 0.9417 | -0.0035 | 178.2938 | 78.4759 |
| GPT-2 | Hand + chest | 749 | 0.8740 | 0.7048 | 0.9207 | -0.0385 | 412.5653 | 239.3530 |
| Gemma 2 2B LoRA | Hand + chest + ankle | 857 | 1.0000 | 0.3501 | 0.9125 | -0.0023 | 282.1363 | 183.7506 |
| Gemma 2 2B LoRA | Hand | 362 | 0.4224 | 0.2579 | 0.9253 | -0.0012 | 498.4750 | 143.2314 |
| Gemma 2 2B LoRA | Chest | 695 | 0.8110 | 0.1377 | 0.9475 | 0.0023 | 109.9894 | 31.4647 |
| Gemma 2 2B LoRA | Hand + chest | 857 | 1.0000 | 0.5998 | 0.9568 | -0.0023 | 289.1921 | 158.1302 |
