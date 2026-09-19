# Triton output archive and cleanup record

Date: 2026-09-19

This document records what was preserved from the PAMAP2/MM-Fit SDForger-style
experiments before removing large, reproducible runtime outputs from Aalto
Triton scratch storage.

## Archive location

- GitHub repository: `SJYANG555/sdforger-pamap2`
- Branch: `codex/sync-256-stats-filter-results`
- Last fully verified result commit before this record: `934d041`
- The Triton branch and GitHub branch were verified to have zero divergence.
- No output files were newer than commit `934d041` at the time of the audit.

The Git history preserves source code, experiment configurations, compact
dataset manifests, evaluation tables, selected plots, and the reports needed to
interpret the experiments. Large model checkpoints, optimizer states, generated
NumPy arrays, and redundant plots are intentionally not treated as permanent
GitHub artifacts.

## Storage audit

At audit time, `/scratch/work/yangs9/ChatTS/outputs` occupied about 65 GiB.

| Category | Approximate size | Preservation decision |
|---|---:|---|
| Git-tracked summaries, metrics, configurations, and selected plots | 306 MiB | Keep locally and on GitHub |
| Ignored checkpoints and optimizer/model state | 63 GiB | Delete after GitHub verification |
| Ignored generated data | 0.4 GiB | Delete; reproducible from configs and code |
| Ignored redundant plots | 0.6 GiB | Delete; selected plots and tables remain tracked |
| Ignored logs and evaluation intermediates | less than 0.1 GiB | Delete after summary verification |

The cleanup target is therefore approximately 65 GiB. The repository's Git
object database is separate from `outputs` and is not included in that saving.

## Scientific results worth retaining

The detailed narrative is in `docs/experiment_transition_inventory.md` and
`docs/pamap2_2act_experiment_summary.md`. The main conclusions are:

1. The formal PAMAP2 pipeline was validated end to end: sensor window to
   compact latent representation, serialized conditioning, language-model
   generation, reconstruction, similarity evaluation, and HAR utility.
2. In the five-activity experiments, changing GPT-2, Gemma, or Llama alone did
   not remove fidelity and controllability gaps. These runs are supporting or
   appendix evidence rather than a model ranking.
3. The two-activity walking/running experiments provide the strongest main-text
   evidence. Hand channels produced the strongest utility-oriented result,
   while chest gyroscope channels produced the strongest similarity-oriented
   result. No single sensor choice dominated every metric.
4. Adding window statistics to the prompt improved validity and some utility
   results, but the effect remained channel- and metric-dependent.
5. Longer training was not a reliable improvement: it changed the trade-off and
   sometimes reduced synthetic-only utility.
6. The ICA dimensionality sweep was non-monotonic. ICA12 was strong in selected
   chest-gyroscope/statistics settings, but was not universally optimal.
7. Shorter windows and raw-statistics versus ICA-statistics prompts changed the
   validity/similarity/utility balance. These are useful secondary ablations,
   not evidence of a universal winner.
8. Post-hoc consistency filtering provided a controllable quality/quantity
   trade-off but did not solve generation collapse.
9. The MM-Fit extension demonstrated that the pipeline transfers to a second
   dataset. Its metrics should be interpreted within its own protocol rather
   than directly ranked against PAMAP2.

## Source-of-truth result files

The following tracked files should be used instead of reconstructing conclusions
from raw Slurm logs:

- `docs/experiment_transition_inventory.md`
- `docs/pamap2_2act_experiment_summary.md`
- `outputs/evaluation/summary/gemma_2act_13run_summary.csv`
- `outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv`
- `outputs/evaluation/summary/gemma_2act_stats_epoch_summary.csv`
- `outputs/evaluation/summary/gemma_2act_ica3_6_12_18_stats_epoch_comparison.csv`
- `outputs/evaluation/summary/gemma_2act_stats_ica12_win128_summary.csv`
- `outputs/evaluation/summary/gemma_2act_stats_ica12_win128_p2575_filtered_summary.csv`
- `outputs/evaluation/summary/gemma_sdforger_full_similarity_summary.csv`
- `outputs/evaluation/summary/real_vs_real_baseline_similarity.csv`
- `outputs/evaluation_stats_filter/summary/stats_filter_evaluation_summary.csv`
- `outputs/evaluation_stats_filter_extra/summary/extra_stats_filter_evaluation_summary.csv`

The tracked experiment YAML files under `config/`, implementation under
`pamap2_forger/`, and Slurm entry points under `slurm/` preserve the main
PAMAP2 reproduction path.

At audit time, `mmfit_forger/` and several newer MM-Fit configurations, scripts,
and Slurm files were still untracked local code. They are not part of this
output-only archive commit and are not deleted by the output cleanup. They need
a separate code review and commit before the MM-Fit pipeline can be considered
fully backed up on GitHub.

## What deletion does and does not preserve

Deleting ignored runtime outputs preserves the conclusions, metrics, selected
figures, configurations, and code on GitHub. It does **not** preserve the exact
trained model/adapter weights or optimizer states. Recovering those exact model
states after deletion is impossible unless another copy exists; otherwise the
models must be retrained. Reproduced runs may differ slightly because of GPU and
software nondeterminism.

Git-tracked files under `outputs/` must remain in the Triton working tree during
cleanup. Only files reported by
`git ls-files -o -i --exclude-standard outputs` are eligible for removal.

## Recovery

The compact archive can be recovered from GitHub with:

```bash
git fetch sdforger-pamap2
git switch codex/sync-256-stats-filter-results
git pull --ff-only sdforger-pamap2 codex/sync-256-stats-filter-results
```

Large runtime outputs can only be regenerated by rebuilding datasets as needed
and rerunning the corresponding training, generation, evaluation, and plotting
jobs through Slurm.
