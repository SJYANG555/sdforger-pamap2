import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SIMILARITY_METRICS = ["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"]


def metric_value(frame: pd.DataFrame, metric: str) -> float:
    if metric not in frame.columns or "activity_name" not in frame.columns:
        return np.nan
    match = frame[frame["activity_name"].astype(str) == "overall_mean"]
    if match.empty:
        return np.nan
    return float(match.iloc[0][metric])


def run_name_for_row(row: pd.Series) -> str:
    if "run_name" in row.index and pd.notna(row["run_name"]):
        return str(row["run_name"])
    return f"gemma_2act_{row['variant']}"


def refresh_file(path: Path) -> None:
    frame = pd.read_csv(path)
    for metric in SIMILARITY_METRICS:
        if metric not in frame.columns:
            insert_at = frame.columns.get_loc("DTW") + 1 if "DTW" in frame.columns else len(frame.columns)
            frame.insert(insert_at, metric, np.nan)

    for idx, row in frame.iterrows():
        run_name = run_name_for_row(row)
        metrics_path = PROJECT_ROOT / "outputs" / "evaluation" / "similarity" / run_name / "similarity_metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception:
            continue
        for metric in SIMILARITY_METRICS:
            frame.at[idx, metric] = metric_value(metrics, metric)

        if "real_baseline_ED" in frame.columns and "ED_vs_real_baseline" in frame.columns:
            real_ed = frame.at[idx, "real_baseline_ED"]
            frame.at[idx, "ED_vs_real_baseline"] = frame.at[idx, "ED"] / real_ed if pd.notna(real_ed) and real_ed else np.nan
        if "real_baseline_DTW" in frame.columns and "DTW_vs_real_baseline" in frame.columns:
            real_dtw = frame.at[idx, "real_baseline_DTW"]
            frame.at[idx, "DTW_vs_real_baseline"] = frame.at[idx, "DTW"] / real_dtw if pd.notna(real_dtw) and real_dtw else np.nan

    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Gemma summary CSV similarity columns from per-run metrics.")
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Summary CSV to update in place. Repeat to update multiple files.",
    )
    args = parser.parse_args()

    paths: List[Path] = [PROJECT_ROOT / path if not path.is_absolute() else path for path in args.summary]
    for path in paths:
        refresh_file(path)
        print(path.resolve())


if __name__ == "__main__":
    main()
