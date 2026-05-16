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
from pamap2_forger.metrics import run_activity_classification_utility
from pamap2_forger.utils import ensure_dir, write_json
from pamap2_forger.visualization import plot_confusion_matrix, plot_metric_bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HAR utility using real and synthetic PAMAP2 windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-train-windows", required=True)
    parser.add_argument("--real-train-metadata", required=True)
    parser.add_argument("--real-test-windows", required=True)
    parser.add_argument("--real-test-metadata", required=True)
    parser.add_argument("--synthetic-windows", required=True)
    parser.add_argument("--synthetic-metadata", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config.utility.output_dir)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    real_train_windows = np.load(args.real_train_windows)
    real_test_windows = np.load(args.real_test_windows)
    synthetic_windows = np.load(args.synthetic_windows)
    real_train_metadata = pd.read_csv(args.real_train_metadata)
    real_test_metadata = pd.read_csv(args.real_test_metadata)
    synthetic_metadata = pd.read_csv(args.synthetic_metadata)

    metrics, confusion_matrices, notes = run_activity_classification_utility(
        real_train_windows=real_train_windows,
        real_train_labels=real_train_metadata["activity_id"].to_numpy(),
        real_test_windows=real_test_windows,
        real_test_labels=real_test_metadata["activity_id"].to_numpy(),
        synthetic_windows=synthetic_windows,
        synthetic_labels=synthetic_metadata["activity_id"].to_numpy(),
        n_estimators=config.utility.n_estimators,
        max_depth=config.utility.max_depth,
        random_seed=config.utility.random_seed,
    )
    metrics.to_csv(output_dir / "utility_metrics.csv", index=False)
    if not metrics.empty:
        plot_metric_bars(
            frame=metrics.rename(columns={"setting": "scenario"}),
            metric_columns=["accuracy", "macro_f1", "weighted_f1"],
            output_path=str(output_dir / "utility_metrics.png"),
            title="HAR Utility Metrics",
        )

    id_to_name = (
        real_test_metadata[["activity_id", "activity_name"]]
        .drop_duplicates()
        .sort_values("activity_id")
    )
    ordered_labels = id_to_name["activity_name"].tolist()
    for setting, matrix in confusion_matrices.items():
        pd.DataFrame(matrix, index=ordered_labels, columns=ordered_labels).to_csv(
            output_dir / f"confusion_matrix_{setting}.csv"
        )
        plot_confusion_matrix(
            matrix=matrix,
            labels=ordered_labels,
            output_path=str(output_dir / f"confusion_matrix_{setting}.png"),
            title=f"Confusion Matrix - {setting}",
        )

    write_json(
        output_dir / "utility_summary.json",
        {
            "num_metric_rows": int(len(metrics)),
            "notes": notes,
            "synthetic_window_count": int(len(synthetic_windows)),
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "rows": int(len(metrics))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
