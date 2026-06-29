import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.embeddings import WindowReducer
from pamap2_forger.utils import get_numeric_embedding_columns


RUNS = [
    (
        "hand_stats_ica18_ep10",
        "Hand + stats + ICA18 + epoch10",
        "artifacts/pamap2_sdforger_dataset_2act_hand_stats_ica18",
        "outputs/generated/gemma_2act_hand_stats_ica18_ep10",
    ),
    (
        "chest_gyro_stats_ica12_ep10",
        "Chest-gyro + stats + ICA12 + epoch10",
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats_ica12",
        "outputs/generated/gemma_2act_chest_gyro_stats_ica12_ep10",
    ),
    (
        "chest_gyro_stats_ica6_ep5",
        "Chest-gyro + stats + ICA6 + epoch5",
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats",
        "outputs/generated/gemma_2act_chest_gyro_stats",
    ),
    (
        "chest_gyro_stats_ica6_ep10",
        "Chest-gyro + stats + ICA6 + epoch10",
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats",
        "outputs/generated/gemma_2act_chest_gyro_stats_ep10",
    ),
]

ACTIVITY_ORDER = ["walking", "running"]
REAL_COLORS = {
    "walking": "#7b3fb2",  # purple, replacing the old gray trace.
    "running": "#2f7d32",
}
SYNTHETIC_COLORS = {
    "walking": "#2c7fb8",
    "running": "#f28e2b",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_channel_names(dataset_dir: Path) -> List[str]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["channel_schema"]["selected_channels"])


def standardize_real_windows(windows: np.ndarray, reducer: WindowReducer) -> np.ndarray:
    if reducer.channel_scaler_mean is None or reducer.channel_scaler_scale is None:
        return windows.astype(np.float32)
    mean = reducer.channel_scaler_mean.reshape(1, 1, -1)
    scale = reducer.channel_scaler_scale.reshape(1, 1, -1)
    return ((windows - mean) / scale).astype(np.float32)


def decode_embeddings_to_standardized_windows(embedding_frame: pd.DataFrame, reducer: WindowReducer) -> np.ndarray:
    numeric_columns = get_numeric_embedding_columns(embedding_frame)
    numeric = embedding_frame[numeric_columns].to_numpy(dtype=np.float32)
    split_points = np.cumsum([state.n_components for state in reducer.states])[:-1]
    parts = np.split(numeric, split_points, axis=1)
    reconstructed_channels = []

    for part, state in zip(parts, reducer.states):
        if state.method == "pca":
            scaled = part @ state.components + state.latent_mean
        else:
            scaled = part @ state.mixing.T + state.latent_mean
        reconstructed = scaled * state.scaler_scale + state.scaler_mean
        reconstructed_channels.append(reconstructed.astype(np.float32))

    return np.stack(reconstructed_channels, axis=-1)


def ordered_activities(*frames: pd.DataFrame) -> List[str]:
    seen = set()
    for frame in frames:
        seen.update(frame["activity_name"].dropna().astype(str).str.lower().tolist())
    activities = [activity for activity in ACTIVITY_ORDER if activity in seen]
    activities.extend(sorted(seen.difference(activities)))
    return activities


def sample_indices(metadata: pd.DataFrame, activity: str, samples: int, seed: int) -> np.ndarray:
    labels = metadata["activity_name"].astype(str).str.lower()
    candidates = metadata.index[labels == activity].to_numpy()
    if len(candidates) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=min(samples, len(candidates)), replace=False)


def draw_lines(
    ax,
    windows: np.ndarray,
    indices: Iterable[int],
    channel_idx: int,
    color: str,
    label: str,
    alpha: float,
    linewidth: float,
) -> None:
    time_index = np.arange(windows.shape[1])
    for line_idx, index in enumerate(indices):
        ax.plot(
            time_index,
            windows[int(index), :, channel_idx],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label if line_idx == 0 else None,
        )


def plot_run(
    run_name: str,
    title: str,
    dataset_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    samples_per_activity: int,
    seed: int,
    max_cols: int,
) -> Dict[str, object]:
    reducer = WindowReducer.load(dataset_dir / "reducer.pkl")
    real_windows = np.load(dataset_dir / "test_windows.npy")
    real_metadata = pd.read_csv(dataset_dir / "test_metadata.csv")
    generated_embeddings = pd.read_csv(generated_dir / "generated_embeddings.csv")

    real_norm = standardize_real_windows(real_windows, reducer)
    synthetic_norm = decode_embeddings_to_standardized_windows(generated_embeddings, reducer)
    channel_names = load_channel_names(dataset_dir)
    activities = ordered_activities(real_metadata, generated_embeddings)

    num_channels = len(channel_names)
    ncols = min(max_cols, num_channels)
    nrows = int(np.ceil(num_channels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7.0, 5.0 * ncols), max(3.8, 3.0 * nrows + 1.0)),
        sharex=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)

    for channel_idx, channel_name in enumerate(channel_names):
        ax = axes_flat[channel_idx]
        for activity_idx, activity in enumerate(activities):
            real_idx = sample_indices(
                real_metadata,
                activity,
                samples_per_activity,
                seed + activity_idx * 31 + channel_idx,
            )
            synthetic_idx = sample_indices(
                generated_embeddings,
                activity,
                samples_per_activity,
                seed + 1000 + activity_idx * 31 + channel_idx,
            )
            draw_lines(
                ax,
                real_norm,
                real_idx,
                channel_idx,
                REAL_COLORS.get(activity, "#7b3fb2"),
                f"real {activity}",
                alpha=0.24,
                linewidth=0.8,
            )
            draw_lines(
                ax,
                synthetic_norm,
                synthetic_idx,
                channel_idx,
                SYNTHETIC_COLORS.get(activity, "#2c7fb8"),
                f"synthetic {activity}",
                alpha=0.56,
                linewidth=0.95,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.28)
        ax.set_title(channel_name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
        if channel_idx % ncols == 0:
            ax.set_ylabel("standardized value")
        if channel_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("time step")

    for ax in axes_flat[num_channels:]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color=REAL_COLORS["walking"], linewidth=2.0, label="real walking"),
        plt.Line2D([], [], color=REAL_COLORS["running"], linewidth=2.0, label="real running"),
        plt.Line2D([], [], color=SYNTHETIC_COLORS["walking"], linewidth=2.0, label="synthetic walking"),
        plt.Line2D([], [], color=SYNTHETIC_COLORS["running"], linewidth=2.0, label="synthetic running"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{title}: real vs generated windows", y=0.985, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    png_path = output_dir / f"{run_name}_standardized_overlay_purple.png"
    pdf_path = output_dir / f"{run_name}_standardized_overlay_purple.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_name": run_name,
        "title": title,
        "dataset_dir": str(dataset_dir),
        "generated_dir": str(generated_dir),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "real_shape": list(real_norm.shape),
        "synthetic_shape": list(synthetic_norm.shape),
        "real_counts": real_metadata["activity_name"].value_counts().to_dict(),
        "synthetic_counts": generated_embeddings["activity_name"].value_counts().to_dict(),
        "channels": channel_names,
        "colors": {
            "real_walking": REAL_COLORS["walking"],
            "real_running": REAL_COLORS["running"],
            "synthetic_walking": SYNTHETIC_COLORS["walking"],
            "synthetic_running": SYNTHETIC_COLORS["running"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected final ICA result overlays with purple real walking traces.")
    parser.add_argument("--output-dir", default="outputs/plots/selected_ica_result_overlays")
    parser.add_argument("--samples-per-activity", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cols", type=int, default=3)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    summaries = []
    for run_name, title, dataset_dir, generated_dir in RUNS:
        summaries.append(
            plot_run(
                run_name=run_name,
                title=title,
                dataset_dir=Path(dataset_dir),
                generated_dir=Path(generated_dir),
                output_dir=output_dir,
                samples_per_activity=args.samples_per_activity,
                seed=args.seed,
                max_cols=args.max_cols,
            )
        )
        print(f"wrote {run_name}")

    (output_dir / "plot_manifest.json").write_text(json.dumps({"runs": summaries}, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "plot_summary.csv", index=False)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_plots": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
