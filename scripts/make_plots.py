import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config
from pamap2_forger.utils import ensure_dir
from pamap2_forger.visualization import (
    plot_embedding_scatter,
    plot_real_vs_synthetic_windows,
    plot_training_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make training and generation plots for PAMAP2 SDForger.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-log-csv", default=None)
    parser.add_argument("--real-windows", default=None)
    parser.add_argument("--real-metadata", default=None)
    parser.add_argument("--synthetic-windows", default=None)
    parser.add_argument("--synthetic-metadata", default=None)
    parser.add_argument("--real-embeddings", default=None)
    parser.add_argument("--synthetic-embeddings", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config.plots.output_dir)

    if args.training_log_csv:
        plot_training_curve(args.training_log_csv, str(output_dir / "training_curve.png"))

    if args.real_windows and args.real_metadata and args.synthetic_windows and args.synthetic_metadata:
        plot_real_vs_synthetic_windows(
            real_windows=np.load(args.real_windows),
            real_metadata=pd.read_csv(args.real_metadata),
            synthetic_windows=np.load(args.synthetic_windows),
            synthetic_metadata=pd.read_csv(args.synthetic_metadata),
            output_dir=str(output_dir / "windows"),
            max_examples_per_activity=config.plots.max_examples_per_activity,
        )

    if args.real_embeddings and args.synthetic_embeddings:
        plot_embedding_scatter(
            real_embeddings=pd.read_csv(args.real_embeddings),
            synthetic_embeddings=pd.read_csv(args.synthetic_embeddings),
            output_path=str(output_dir / "embedding_scatter.png"),
            max_points=config.plots.embedding_scatter_max_points,
        )


if __name__ == "__main__":
    main()
