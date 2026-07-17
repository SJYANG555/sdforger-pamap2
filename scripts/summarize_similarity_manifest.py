import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SIMILARITY_METRICS = ["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"]


def read_manifest(path: Path) -> List[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_metrics(metrics_path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "status": "missing",
        "num_activity_rows": 0,
        "num_real": np.nan,
        "num_synthetic": np.nan,
        "missing_metrics": ",".join(SIMILARITY_METRICS),
        **{metric: np.nan for metric in SIMILARITY_METRICS},
    }
    if not metrics_path.exists():
        return row
    try:
        frame = pd.read_csv(metrics_path)
    except Exception as exc:
        row["status"] = "read_error"
        row["error"] = str(exc)
        return row
    if frame.empty:
        row["status"] = "empty"
        row["missing_metrics"] = ",".join(
            metric for metric in SIMILARITY_METRICS if metric not in frame.columns
        )
        return row

    missing = [metric for metric in SIMILARITY_METRICS if metric not in frame.columns]
    overall = frame[frame["activity_name"].astype(str) == "overall_mean"] if "activity_name" in frame.columns else pd.DataFrame()
    source = overall.iloc[0] if not overall.empty else frame.iloc[-1]
    for metric in SIMILARITY_METRICS:
        if metric in frame.columns:
            row[metric] = float(source[metric])
    row["status"] = "ok" if not missing else "missing_metrics"
    row["num_activity_rows"] = int(len(frame[frame["activity_name"].astype(str) != "overall_mean"])) if "activity_name" in frame.columns else int(len(frame))
    row["num_real"] = int(source["num_real"]) if "num_real" in frame.columns and pd.notna(source["num_real"]) else np.nan
    row["num_synthetic"] = int(source["num_synthetic"]) if "num_synthetic" in frame.columns and pd.notna(source["num_synthetic"]) else np.nan
    row["missing_metrics"] = ",".join(missing)
    return row


def build_summary(manifests: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for manifest in manifests:
        for index, config_path in enumerate(read_manifest(manifest)):
            metrics_path = config_path.parent / "similarity_metrics.csv"
            rows.append(
                {
                    "manifest": manifest.name,
                    "manifest_index": index,
                    "run_name": config_path.parent.name,
                    "metrics_path": str(metrics_path),
                    **summarize_metrics(metrics_path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing similarity metrics listed in manifest files.")
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Manifest containing resolved_config paths. Repeat to merge multiple manifests.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/summary/gemma_sdforger_full_similarity_summary.csv"))
    args = parser.parse_args()

    frame = build_summary([PROJECT_ROOT / path if not path.is_absolute() else path for path in args.manifest])
    output_path = PROJECT_ROOT / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
