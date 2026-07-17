import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config
from pamap2_forger.metrics import compute_similarity_metrics
from pamap2_forger.utils import ensure_dir, write_json
from pamap2_forger.visualization import plot_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute SDForger-style similarity metrics from a resolved config."
    )
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--compute-shr", action="store_true")
    parser.add_argument("--shr-max-iter", type=int, default=None)
    parser.add_argument("--shr-max-inner-iter", type=int, default=None)
    parser.add_argument("--shr-max-samples-per-activity", type=int, default=None)
    parser.add_argument("--shr-channel-mode", choices=["mean", "per_channel"], default=None)
    parser.add_argument("--filtered-root", type=Path, default=Path("outputs/generated_stats_filter/win128_stats_ica12_p2575"))
    parser.add_argument("--use-config-output-dir", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.resolved_config is not None:
        return args.resolved_config
    if args.manifest is None or args.index is None:
        raise SystemExit("Provide either --resolved-config or both --manifest and --index.")
    lines = [line.strip() for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.index < 0 or args.index >= len(lines):
        raise SystemExit(f"Manifest index {args.index} is outside 0..{len(lines) - 1}.")
    return Path(lines[args.index])


def backup_existing(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    backup = path.with_suffix(path.suffix + ".spectral_sd_backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args)
    config = load_config(config_path)

    data_dir = Path(config.data.output_dir)
    filtered_generation_dir = args.filtered_root / config_path.parent.name
    generation_dir = filtered_generation_dir if filtered_generation_dir.exists() else Path(config.generation.output_dir)
    output_dir = ensure_dir(config.similarity.output_dir if args.use_config_output_dir else config_path.parent)

    real_windows_path = data_dir / "test_windows.npy"
    real_metadata_path = data_dir / "test_metadata.csv"
    synthetic_windows_path = generation_dir / "generated_windows.npy"
    synthetic_metadata_path = generation_dir / "generated_embeddings.csv"

    real_windows = np.load(real_windows_path)
    synthetic_windows = np.load(synthetic_windows_path)
    real_metadata = pd.read_csv(real_metadata_path)
    synthetic_metadata = pd.read_csv(synthetic_metadata_path)

    metrics = compute_similarity_metrics(
        real_windows=real_windows,
        real_metadata=real_metadata,
        synthetic_windows=synthetic_windows,
        synthetic_metadata=synthetic_metadata,
        max_lag=config.similarity.max_lag,
        max_samples_per_activity=config.similarity.max_samples_per_activity,
        dtw_window=config.similarity.dtw_window,
        seed=config.similarity.random_seed,
        mdd_bins=config.similarity.mdd_bins,
        compute_shr=args.compute_shr or config.similarity.compute_shr,
        shr_seed=config.similarity.shr_seed,
        shr_basis_count=config.similarity.shr_basis_count,
        shr_lambda=config.similarity.shr_lambda,
        shr_basis_ratio=config.similarity.shr_basis_ratio,
        shr_c=config.similarity.shr_c,
        shr_epsilon=config.similarity.shr_epsilon,
        shr_max_iter=args.shr_max_iter or config.similarity.shr_max_iter,
        shr_max_inner_iter=args.shr_max_inner_iter or config.similarity.shr_max_inner_iter,
        shr_max_samples_per_activity=(
            args.shr_max_samples_per_activity
            if args.shr_max_samples_per_activity is not None
            else config.similarity.shr_max_samples_per_activity
        ),
        shr_channel_mode=args.shr_channel_mode or config.similarity.shr_channel_mode,
    )

    metrics_path = output_dir / "similarity_metrics.csv"
    if not args.no_backup:
        backup_existing(metrics_path)
    metrics.to_csv(metrics_path, index=False)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    write_json(
        output_dir / "similarity_summary.json",
        {
            "num_rows": int(len(metrics)),
            "metrics_columns": metrics.columns.tolist(),
            "metric_impl": "sdforger_tsgbench_mdd_acd_skewness_sd_kd_ed_dtw_shr",
            "compute_shr": bool(args.compute_shr or config.similarity.compute_shr),
            "source_resolved_config": str(config_path),
        },
    )
    if not metrics.empty:
        plot_metric_bars(
            frame=metrics,
            metric_columns=["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"],
            output_path=str(output_dir / "similarity_metrics.png"),
            title="SDForger-style Similarity Metrics by Activity",
        )

    print(
        json.dumps(
            {
                "config": str(config_path),
                "output_dir": str(output_dir.resolve()),
                "rows": int(len(metrics)),
                "columns": metrics.columns.tolist(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
