import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUNS = [
    ("hand_stats", "5epoch", "hand", "gemma_2act_hand_stats"),
    ("hand_stats_ep10", "10epoch", "hand", "gemma_2act_hand_stats_ep10"),
    ("hand_stats_ep20", "20epoch", "hand", "gemma_2act_hand_stats_ep20"),
    ("chest_gyro_stats", "5epoch", "chest_gyro", "gemma_2act_chest_gyro_stats"),
    ("chest_gyro_stats_ep10", "10epoch", "chest_gyro", "gemma_2act_chest_gyro_stats_ep10"),
    ("chest_gyro_stats_ep20", "20epoch", "chest_gyro", "gemma_2act_chest_gyro_stats_ep20"),
    ("chest_stats_ep5", "5epoch", "chest", "gemma_2act_chest_stats_ep5"),
    ("chest_stats_ep10", "10epoch", "chest", "gemma_2act_chest_stats_ep10"),
    ("chest_stats_ep20", "20epoch", "chest", "gemma_2act_chest_stats_ep20"),
    ("chest_acc_stats_ep5", "5epoch", "chest_acc", "gemma_2act_chest_acc_stats_ep5"),
    ("chest_acc_stats_ep10", "10epoch", "chest_acc", "gemma_2act_chest_acc_stats_ep10"),
    ("chest_acc_stats_ep20", "20epoch", "chest_acc", "gemma_2act_chest_acc_stats_ep20"),
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def metric_value(frame: pd.DataFrame, index_column: str, index_value: str, metric: str) -> float:
    try:
        return float(frame.set_index(index_column).loc[index_value, metric])
    except (KeyError, ValueError):
        return np.nan


def count_generated_by_activity(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    frame = pd.read_csv(path)
    if "activity_name" in frame.columns:
        labels = frame["activity_name"]
    elif "activity" in frame.columns:
        labels = frame["activity"]
    else:
        return 0, 0
    counts = labels.astype(str).str.lower().value_counts()
    return int(counts.get("running", 0)), int(counts.get("walking", 0))


def build_summary(root: Path) -> pd.DataFrame:
    baseline_path = root / "outputs" / "evaluation" / "summary" / "real_vs_real_baseline_similarity.csv"
    baselines = pd.read_csv(baseline_path).set_index("base") if baseline_path.exists() else pd.DataFrame()
    rows = []

    for variant, epoch, base_dataset, run_name in RUNS:
        generated_dir = root / "outputs" / "generated" / run_name
        similarity_dir = root / "outputs" / "evaluation" / "similarity" / run_name
        utility_dir = root / "outputs" / "evaluation" / "utility" / run_name
        gen_summary_path = generated_dir / "generation_summary.json"
        sim_path = similarity_dir / "similarity_metrics.csv"
        util_path = utility_dir / "utility_metrics.csv"

        if not (gen_summary_path.exists() and sim_path.exists() and util_path.exists()):
            continue

        gen = read_json(gen_summary_path)
        generated_running, generated_walking = count_generated_by_activity(generated_dir / "generated_embeddings.csv")
        sim = pd.read_csv(sim_path)
        util = pd.read_csv(util_path)

        ed = metric_value(sim, "activity_name", "overall_mean", "ED")
        dtw = metric_value(sim, "activity_name", "overall_mean", "DTW")
        real_ed = float(baselines.loc[base_dataset, "real_baseline_ED"]) if base_dataset in baselines.index else np.nan
        real_dtw = float(baselines.loc[base_dataset, "real_baseline_DTW"]) if base_dataset in baselines.index else np.nan

        rows.append(
            {
                "variant": variant,
                "epoch": epoch,
                "valid_generated / 305": f"{int(gen.get('num_generated_embeddings', 0))}/305",
                "generated_running": generated_running,
                "generated_walking": generated_walking,
                "ED": ed,
                "DTW": dtw,
                "ED_vs_real_baseline": ed / real_ed if real_ed else np.nan,
                "DTW_vs_real_baseline": dtw / real_dtw if real_dtw else np.nan,
                "synthetic_only accuracy": metric_value(util, "setting", "synthetic_only", "accuracy"),
                "synthetic_only macro_f1": metric_value(util, "setting", "synthetic_only", "macro_f1"),
                "real_plus_synthetic accuracy": metric_value(util, "setting", "real_plus_synthetic", "accuracy"),
                "real_plus_synthetic macro_f1": metric_value(util, "setting", "real_plus_synthetic", "macro_f1"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Gemma 2-action stats prompt epoch experiments.")
    parser.add_argument("--output-dir", default="outputs/evaluation/summary")
    parser.add_argument("--output-name", default="gemma_2act_stats_epoch_summary.csv")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_summary(PROJECT_ROOT)
    output_path = output_dir / args.output_name
    frame.to_csv(output_path, index=False)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
