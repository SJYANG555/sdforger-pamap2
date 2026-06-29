import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.embeddings import WindowReducer
from pamap2_forger.utils import ensure_dir


DEFAULT_RUNS = [
    (
        "gemma_2act_chest_gyro_stats_ica12_ep10",
        "Chest gyro + stats + ICA12 + epoch10",
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats_ica12",
        "outputs/generated/gemma_2act_chest_gyro_stats_ica12_ep10",
    ),
    (
        "gemma_2act_hand_stats_ica18_ep10",
        "Hand + stats + ICA18 + epoch10",
        "artifacts/pamap2_sdforger_dataset_2act_hand_stats_ica18",
        "outputs/generated/gemma_2act_hand_stats_ica18_ep10",
    ),
]
LINE_COLORS = {
    ("walking", "real"): "#7b3fb2",
    ("walking", "synthetic"): "#2c7fb8",
    ("running", "real"): "#2f7d32",
    ("running", "synthetic"): "#f28e2b",
}


def load_channel_names(dataset_dir: Path) -> List[str]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["channel_schema"]["selected_channels"])


def maybe_standardize(windows: np.ndarray, reducer: WindowReducer, enabled: bool) -> np.ndarray:
    if not enabled:
        return windows.astype(np.float32)
    if reducer.channel_scaler_mean is None or reducer.channel_scaler_scale is None:
        mean = windows.mean(axis=(0, 1), dtype=np.float64).reshape(1, 1, -1)
        scale = windows.std(axis=(0, 1), dtype=np.float64).reshape(1, 1, -1)
        scale = np.where(scale == 0, 1.0, scale)
        return ((windows - mean) / scale).astype(np.float32)
    mean = reducer.channel_scaler_mean.reshape(1, 1, -1)
    scale = reducer.channel_scaler_scale.reshape(1, 1, -1)
    return ((windows - mean) / scale).astype(np.float32)


def build_real_lookup(metadata: pd.DataFrame) -> Dict[int, int]:
    return {int(window_id): int(idx) for idx, window_id in enumerate(metadata["window_id"].astype(int))}


def find_prompt_real_index(
    synthetic_row: pd.Series,
    real_metadata: pd.DataFrame,
    real_lookup: Dict[int, int],
) -> Optional[int]:
    window_id = int(synthetic_row["window_id"])
    if window_id in real_lookup:
        return real_lookup[window_id]

    activity = str(synthetic_row["activity_name"]).lower()
    matches = real_metadata.index[real_metadata["activity_name"].astype(str).str.lower() == activity].to_numpy()
    if len(matches) == 0:
        return None
    return int(matches[0])


def candidate_distances(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
) -> pd.DataFrame:
    real_lookup = build_real_lookup(real_metadata)
    rows = []
    for syn_idx, syn_row in synthetic_metadata.iterrows():
        real_idx = find_prompt_real_index(syn_row, real_metadata, real_lookup)
        if real_idx is None or int(syn_idx) >= len(synthetic_windows):
            continue
        diff = synthetic_windows[int(syn_idx)] - real_windows[real_idx]
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        mae = float(np.mean(np.abs(diff)))
        rows.append(
            {
                "synthetic_index": int(syn_idx),
                "candidate_id": int(syn_row.get("candidate_id", syn_idx)),
                "prompt_window_id": int(syn_row["window_id"]),
                "real_index": int(real_idx),
                "activity_name": str(syn_row["activity_name"]),
                "rmse_to_prompt_real": rmse,
                "mae_to_prompt_real": mae,
            }
        )
    return pd.DataFrame(rows)


def select_candidates(distances: pd.DataFrame, samples_per_activity: int) -> pd.DataFrame:
    selected = []
    for _, group in distances.groupby(distances["activity_name"].astype(str).str.lower(), sort=True):
        sort_columns = ["rmse_to_prompt_real"]
        if "stats_score" in group.columns:
            sort_columns = ["stats_score", "max_channel_stats_score", "rmse_to_prompt_real"]
        selected.append(group.sort_values(sort_columns).head(samples_per_activity))
    if not selected:
        return distances.head(0)
    return pd.concat(selected, ignore_index=True)


def attach_stats_filter_scores(distances: pd.DataFrame, generated_dir: Path) -> pd.DataFrame:
    stats_path = generated_dir / "stats_filter_selected.csv"
    if not stats_path.exists():
        return distances
    stats = pd.read_csv(stats_path)
    keep_columns = [
        column
        for column in ["candidate_id", "stats_score", "max_channel_stats_score", "mean_zerr_avg", "std_relerr_avg", "range_zerr_avg"]
        if column in stats.columns
    ]
    if "candidate_id" not in keep_columns:
        return distances
    stats = stats[keep_columns].drop_duplicates(subset=["candidate_id"])
    return distances.merge(stats, on="candidate_id", how="left")


def load_metric_snapshot(run_name: str) -> Dict[str, object]:
    snapshot: Dict[str, object] = {}
    sim_path = Path("outputs/evaluation/similarity") / run_name / "similarity_metrics.csv"
    util_path = Path("outputs/evaluation/utility") / run_name / "utility_metrics.csv"
    if sim_path.exists() and sim_path.stat().st_size > 0:
        sim = pd.read_csv(sim_path)
        overall = sim[sim["activity_name"].astype(str) == "overall_mean"]
        if not overall.empty:
            snapshot["similarity_overall_mean"] = overall.iloc[0].to_dict()
    if util_path.exists() and util_path.stat().st_size > 0:
        util = pd.read_csv(util_path)
        snapshot["utility"] = {
            str(row["setting"]): {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
            }
            for _, row in util.iterrows()
        }
    return snapshot


def plot_single_pair(
    output_path: Path,
    title: str,
    activity: str,
    channel_names: List[str],
    real_window: np.ndarray,
    synthetic_window: np.ndarray,
    real_label: str,
    synthetic_label: str,
    ylabel: str,
) -> None:
    time_index = np.arange(real_window.shape[0])
    num_channels = len(channel_names)
    fig_height = max(5.4, 2.05 * num_channels + 1.0)
    fig, axes = plt.subplots(num_channels, 1, figsize=(10.8, fig_height), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    activity_key = activity.lower()
    real_color = LINE_COLORS.get((activity_key, "real"), "#7b3fb2")
    synthetic_color = LINE_COLORS.get((activity_key, "synthetic"), "#2c7fb8")

    for channel_idx, channel_name in enumerate(channel_names):
        ax = axes_flat[channel_idx]
        ax.plot(
            time_index,
            real_window[:, channel_idx],
            color=real_color,
            linestyle="-",
            linewidth=1.55,
            alpha=0.86,
            label=real_label,
        )
        ax.plot(
            time_index,
            synthetic_window[:, channel_idx],
            color=synthetic_color,
            linestyle="--",
            linewidth=1.65,
            alpha=0.95,
            label=synthetic_label,
        )
        ax.set_title(channel_name, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="#555555", linewidth=0.7, alpha=0.25)
        ax.grid(alpha=0.18, linewidth=0.6)
        if channel_idx == num_channels - 1:
            ax.set_xlabel("time step")

    style_handles = [
        plt.Line2D([], [], color=real_color, linestyle="-", linewidth=2.0, label=real_label),
        plt.Line2D([], [], color=synthetic_color, linestyle="--", linewidth=2.0, label=synthetic_label),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(title, x=0.08, ha="left", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_run(
    run_name: str,
    display_name: str,
    dataset_dir: Path,
    generated_dir: Path,
    output_root: Path,
    samples_per_activity: int,
    standardize: bool,
) -> Dict[str, object]:
    reducer = WindowReducer.load(dataset_dir / "reducer.pkl")
    real_windows_raw = np.load(dataset_dir / "test_windows.npy")
    synthetic_windows_raw = np.load(generated_dir / "generated_windows.npy")
    real_metadata = pd.read_csv(dataset_dir / "test_metadata.csv")
    synthetic_metadata = pd.read_csv(generated_dir / "generated_embeddings.csv")
    channel_names = load_channel_names(dataset_dir)

    real_windows = maybe_standardize(real_windows_raw, reducer, standardize)
    synthetic_windows = maybe_standardize(synthetic_windows_raw, reducer, standardize)
    distances = candidate_distances(real_windows, real_metadata, synthetic_windows, synthetic_metadata)
    distances = attach_stats_filter_scores(distances, generated_dir)
    selected = select_candidates(distances, samples_per_activity=samples_per_activity)

    run_output_dir = ensure_dir(output_root / run_name)
    plot_rows = []
    ylabel = "standardized sensor value" if standardize else "sensor value"
    for order, row in selected.iterrows():
        synthetic_index = int(row["synthetic_index"])
        real_index = int(row["real_index"])
        activity = str(row["activity_name"])
        base_name = (
            f"{order:02d}_{activity}_candidate{int(row['candidate_id'])}"
            f"_prompt{int(row['prompt_window_id'])}"
        )
        png_path = run_output_dir / f"{base_name}.png"
        title = (
            f"{display_name} | {activity} | generated candidate {int(row['candidate_id'])} "
            f"vs prompt window {int(row['prompt_window_id'])}"
        )
        plot_single_pair(
            output_path=png_path,
            title=title,
            activity=activity,
            channel_names=channel_names,
            real_window=real_windows[real_index],
            synthetic_window=synthetic_windows[synthetic_index],
            real_label=f"real {activity} prompt window",
            synthetic_label=f"generated {activity} window",
            ylabel=ylabel,
        )
        record = row.to_dict()
        record["plot"] = str(png_path)
        plot_rows.append(record)

    selected_path = run_output_dir / "selected_samples.csv"
    pd.DataFrame(plot_rows).to_csv(selected_path, index=False)
    return {
        "run_name": run_name,
        "display_name": display_name,
        "dataset_dir": str(dataset_dir),
        "generated_dir": str(generated_dir),
        "output_dir": str(run_output_dir),
        "selected_samples_csv": str(selected_path),
        "num_plots": len(plot_rows),
        "standardized": standardize,
        "channels": channel_names,
        "real_shape": list(real_windows_raw.shape),
        "synthetic_shape": list(synthetic_windows_raw.shape),
        "real_counts": real_metadata["activity_name"].value_counts().to_dict(),
        "synthetic_counts": synthetic_metadata["activity_name"].value_counts().to_dict(),
        "metrics": load_metric_snapshot(run_name),
        "plots": plot_rows,
    }


def parse_run_spec(spec: str) -> Tuple[str, str, str, str]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--run must be name:display_name:dataset_dir:generated_dir"
        )
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one generated PAMAP2 window per figure with its prompt real-window counterpart."
    )
    parser.add_argument("--output-dir", default="outputs/plots/single_generated_window_comparisons")
    parser.add_argument("--samples-per-activity", type=int, default=5)
    parser.add_argument("--run", action="append", type=parse_run_spec, default=None)
    parser.add_argument("--raw", action="store_true", help="Plot raw sensor values instead of standardized values.")
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_dir))
    run_specs = args.run or DEFAULT_RUNS
    summaries = []
    for run_name, display_name, dataset_dir, generated_dir in run_specs:
        summaries.append(
            plot_run(
                run_name=run_name,
                display_name=display_name,
                dataset_dir=Path(dataset_dir),
                generated_dir=Path(generated_dir),
                output_root=output_root,
                samples_per_activity=args.samples_per_activity,
                standardize=not args.raw,
            )
        )
        print(f"wrote {run_name}")

    manifest = {"runs": summaries}
    (output_root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_rows = [
        {
            "run_name": item["run_name"],
            "num_plots": item["num_plots"],
            "selected_samples_csv": item["selected_samples_csv"],
            "output_dir": item["output_dir"],
        }
        for item in summaries
    ]
    pd.DataFrame(summary_rows).to_csv(output_root / "plot_summary.csv", index=False)
    print(json.dumps({"output_dir": str(output_root.resolve()), "num_runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
