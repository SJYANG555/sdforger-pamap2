import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.metrics import compute_similarity_metrics


RUNS: List[Dict[str, object]] = [
    {
        "setting_name": "2act_hand_stats_ica12_win128_ep5",
        "run_name": "gemma_2act_hand_stats_ica12_win128_ep5",
        "group": "hand",
        "epoch": 5,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_hand_stats_ica12_win128_ep5",
    },
    {
        "setting_name": "2act_hand_stats_ica12_win128_ep10",
        "run_name": "gemma_2act_hand_stats_ica12_win128_ep10",
        "group": "hand",
        "epoch": 10,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_hand_stats_ica12_win128_ep10",
    },
    {
        "setting_name": "2act_hand_acc_stats_ica12_win128_ep5",
        "run_name": "gemma_2act_hand_acc_stats_ica12_win128_ep5",
        "group": "hand_acc",
        "epoch": 5,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_hand_acc_stats_ica12_win128_ep5",
    },
    {
        "setting_name": "2act_hand_acc_stats_ica12_win128_ep10",
        "run_name": "gemma_2act_hand_acc_stats_ica12_win128_ep10",
        "group": "hand_acc",
        "epoch": 10,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_hand_acc_stats_ica12_win128_ep10",
    },
    {
        "setting_name": "2act_chest_gyro_stats_ica12_win128_ep5",
        "run_name": "gemma_2act_chest_gyro_stats_ica12_win128_ep5",
        "group": "chest_gyro",
        "epoch": 5,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats_ica12_win128_ep5",
    },
    {
        "setting_name": "2act_chest_gyro_stats_ica12_win128_ep10",
        "run_name": "gemma_2act_chest_gyro_stats_ica12_win128_ep10",
        "group": "chest_gyro",
        "epoch": 10,
        "dataset_dir": "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats_ica12_win128_ep10",
    },
]

FILTERS = ["stats_p25", "stats_p50", "stats_p75"]
SIMILARITY_METRICS = ["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"]


def read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def metric_value(frame: pd.DataFrame, index_column: str, index_value: str, metric: str) -> float:
    if frame.empty or metric not in frame.columns or index_column not in frame.columns:
        return float("nan")
    matches = frame[frame[index_column].astype(str) == index_value]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0][metric])


def split_real_reference(
    windows: np.ndarray,
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    left_indexes: List[int] = []
    right_indexes: List[int] = []
    for _, group in metadata.groupby(metadata["activity_name"].astype(str), sort=True):
        indexes = group.index.to_numpy()
        left_indexes.extend(indexes[::2].tolist())
        right_indexes.extend(indexes[1::2].tolist())
    left_indexes = sorted(left_indexes)
    right_indexes = sorted(right_indexes)
    return (
        windows[left_indexes],
        metadata.iloc[left_indexes].reset_index(drop=True),
        windows[right_indexes],
        metadata.iloc[right_indexes].reset_index(drop=True),
    )


def real_reference_metrics(dataset_dir: Path, max_lag: int = 32, max_samples: int = 128) -> Dict[str, float]:
    windows = np.load(dataset_dir / "test_windows.npy")
    metadata = pd.read_csv(dataset_dir / "test_metadata.csv")
    left_windows, left_metadata, right_windows, right_metadata = split_real_reference(windows, metadata)
    metrics = compute_similarity_metrics(
        real_windows=left_windows,
        real_metadata=left_metadata,
        synthetic_windows=right_windows,
        synthetic_metadata=right_metadata,
        max_lag=max_lag,
        max_samples_per_activity=max_samples,
        dtw_window=None,
        seed=42,
    )
    return {
        "ED_ref": metric_value(metrics, "activity_name", "overall_mean", "ED"),
        "DTW_ref": metric_value(metrics, "activity_name", "overall_mean", "DTW"),
    }


def build_summary(root: Path, filtered_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    baseline_cache: Dict[str, Dict[str, float]] = {}

    for run in RUNS:
        run_name = str(run["run_name"])
        dataset_dir = root / str(run["dataset_dir"])
        manifest_path = dataset_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        channels = manifest["channel_schema"]["selected_channels"]

        baseline_key = str(dataset_dir)
        if baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = real_reference_metrics(dataset_dir)
        baseline = baseline_cache[baseline_key]

        for filter_name in FILTERS:
            filtered_name = f"{run_name}_{filter_name}"
            generated_dir = filtered_root / filtered_name
            similarity_dir = root / "outputs" / "evaluation" / "similarity" / filtered_name
            utility_dir = root / "outputs" / "evaluation" / "utility" / filtered_name
            gen_summary_path = generated_dir / "generation_summary.json"
            sim_path = similarity_dir / "similarity_metrics.csv"
            util_path = utility_dir / "utility_metrics.csv"
            if not (gen_summary_path.exists() and sim_path.exists() and util_path.exists()):
                continue

            generation = read_json(gen_summary_path)
            filter_summary = generation.get("filter", {})
            similarity = pd.read_csv(sim_path)
            utility = pd.read_csv(util_path)
            sim_values = {
                metric: metric_value(similarity, "activity_name", "overall_mean", metric)
                for metric in SIMILARITY_METRICS
            }
            ed = sim_values["ED"]
            dtw = sim_values["DTW"]
            ed_ref = baseline["ED_ref"]
            dtw_ref = baseline["DTW_ref"]
            kept = int(filter_summary.get("kept", 0))
            total = int(filter_summary.get("total", 0))

            rows.append(
                {
                    "setting name": run["setting_name"],
                    "filtered setting": filtered_name,
                    "filter": filter_name,
                    "keep fraction target": generation.get("keep_fraction_by_activity"),
                    "group": run["group"],
                    "channels": ",".join(channels),
                    "num_channels": int(manifest["channel_schema"]["num_channels"]),
                    "window_size": int(manifest["channel_schema"]["window_size"]),
                    "stride": 64,
                    "ICA dim": 12,
                    "epoch": int(run["epoch"]),
                    "kept generated count": kept,
                    "source valid generated count": total,
                    "kept/source valid rate": kept / total if total else np.nan,
                    "total prompt count": 666,
                    "kept/prompt rate": kept / 666,
                    "stats_score_mean": filter_summary.get("stats_score_mean"),
                    "stats_score_median": filter_summary.get("stats_score_median"),
                    "stats_score_max": filter_summary.get("stats_score_max"),
                    "synthetic-only HAR accuracy": metric_value(utility, "setting", "synthetic_only", "accuracy"),
                    "real-only HAR accuracy": metric_value(utility, "setting", "real_only", "accuracy"),
                    "real-plus-synthetic HAR accuracy": metric_value(utility, "setting", "real_plus_synthetic", "accuracy"),
                    "MDD": sim_values["MDD"],
                    "ACD": sim_values["ACD"],
                    "SD": sim_values["SD"],
                    "KD": sim_values["KD"],
                    "ED": ed,
                    "ED_ref": ed_ref,
                    "ED/ref": ed / ed_ref if ed_ref else np.nan,
                    "DTW": dtw,
                    "DTW_ref": dtw_ref,
                    "DTW/ref": dtw / dtw_ref if dtw_ref else np.nan,
                    "SHR": sim_values["SHR"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize p25/p50/p75 stats-filtered win128 Gemma experiments.")
    parser.add_argument("--filtered-root", default="outputs/generated_stats_filter/win128_stats_ica12_p2575")
    parser.add_argument("--output-dir", default="outputs/evaluation/summary")
    parser.add_argument("--output-name", default="gemma_2act_stats_ica12_win128_p2575_filtered_summary.csv")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_summary(PROJECT_ROOT, PROJECT_ROOT / args.filtered_root)
    output_path = output_dir / args.output_name
    frame.to_csv(output_path, index=False)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
