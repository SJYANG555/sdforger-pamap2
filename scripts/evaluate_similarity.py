import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate similarity between real and synthetic PAMAP2 windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-windows", required=True)
    parser.add_argument("--real-metadata", required=True)
    parser.add_argument("--synthetic-windows", required=True)
    parser.add_argument("--synthetic-metadata", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compute-shr", action="store_true", help="Compute SDForger SHAP-RE/SHR.")
    parser.add_argument("--shr-max-iter", type=int, default=None)
    parser.add_argument("--shr-max-inner-iter", type=int, default=None)
    parser.add_argument("--shr-max-samples-per-activity", type=int, default=None)
    parser.add_argument("--shr-channel-mode", choices=["mean", "per_channel"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config.similarity.output_dir)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    real_windows = np.load(args.real_windows)
    synthetic_windows = np.load(args.synthetic_windows)
    real_metadata = pd.read_csv(args.real_metadata)
    synthetic_metadata = pd.read_csv(args.synthetic_metadata)

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
    metrics.to_csv(output_dir / "similarity_metrics.csv", index=False)
    write_json(
        output_dir / "similarity_summary.json",
        {
            "num_rows": int(len(metrics)),
            "metrics_columns": metrics.columns.tolist(),
        },
    )
    plot_metric_bars(
        frame=metrics,
        metric_columns=["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"],
        output_path=str(output_dir / "similarity_metrics.png"),
        title="SDForger-style Similarity Metrics by Activity",
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "rows": int(len(metrics))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
