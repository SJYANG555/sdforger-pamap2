# Triton GPU Notes

The current smoke test was validated as a **CPU batch** run. GPU execution was not validated on the current account.

## What is already GPU-ready in code

- `scripts/train_pamap2_lm.py`
- `scripts/generate_pamap2_synthetic.py`
- `config/pamap2_sdforger_gpt2.yaml`
- `config/pamap2_sdforger_gemma2_2b.yaml`
- `slurm/run_train_gpt2_gpu_ready.slurm`

## What you need to change for GPU

1. Replace:
   - `YOUR_GPU_ACCOUNT`
   - `YOUR_GPU_PARTITION`
2. Set:
   - `CONDA_ENV_PATH`
3. Keep these library exports if Triton still hits the same ABI issue:
   - `export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"`
   - `export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6"`

## CPU smoke -> GPU baseline switch

Use:

- smoke config for CPU verification:
  `config/pamap2_sdforger_gpt2_smoke.yaml`
- full GPT-2 baseline config for formal training:
  `config/pamap2_sdforger_gpt2.yaml`

Then submit:

```bash
sbatch slurm/run_train_gpt2_gpu_ready.slurm
```

## Known limitation

If Triton rejects explicit GPU account/partition combinations, this is a cluster-permission issue, not a code issue. In that case you need a valid GPU allocation before the script can run.
