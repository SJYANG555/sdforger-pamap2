# Five-Class Model Comparison for Report

All runs use the five PAMAP2 activities: cycling, running, sitting, standing, walking.

Implemented SDForger-style similarity metrics: MDD, ACD, SD, KD, ED, DTW.
Downstream HAR utility: RandomForest real-only, synthetic-only, and real+synthetic accuracy/F1.

| model | subset | num_generated | retention_rate | synthetic_only_accuracy | real_plus_synthetic_accuracy | augmentation_delta | overall_DTW | overall_ED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | 18ch full | 835 | 0.9743 | 0.4877 | 0.9312 | 0.0163 | 338.6882 | 235.5080 |
| Gemma 2 2B | 18ch full | 857 | 1.0000 | 0.3501 | 0.9125 | -0.0023 | 282.1363 | 183.7506 |
| Llama 3.2 3B | 18ch full | 844 | 0.9848 | 0.4329 | 0.9288 | 0.0140 | 300.3610 | 186.3491 |
| GPT-2 | 12ch hand+chest | 749 | 0.8740 | 0.7048 | 0.9207 | -0.0385 | 412.5653 | 239.3530 |
| Gemma 2 2B | 12ch hand+chest | 857 | 1.0000 | 0.5998 | 0.9568 | -0.0023 | 289.1921 | 158.1302 |
| Llama 3.2 3B | 12ch hand+chest | 847 | 0.9883 | 0.2964 | 0.9522 | -0.0070 | 304.3695 | 160.7831 |
