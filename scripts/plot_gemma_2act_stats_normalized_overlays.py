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
    ("hand_stats", "5epoch", "hand", "artifacts/pamap2_sdforger_dataset_2act_hand_stats", "outputs/generated/gemma_2act_hand_stats"),
    ("hand_stats_ep10", "10epoch", "hand", "artifacts/pamap2_sdforger_dataset_2act_hand_stats", "outputs/generated/gemma_2act_hand_stats_ep10"),
    ("hand_stats_ep20", "20epoch", "hand", "artifacts/pamap2_sdforger_dataset_2act_hand_stats", "outputs/generated/gemma_2act_hand_stats_ep20"),
    ("chest_gyro_stats", "5epoch", "chest_gyro", "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats", "outputs/generated/gemma_2act_chest_gyro_stats"),
    ("chest_gyro_stats_ep10", "10epoch", "chest_gyro", "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats", "outputs/generated/gemma_2act_chest_gyro_stats_ep10"),
    ("chest_gyro_stats_ep20", "20epoch", "chest_gyro", "artifacts/pamap2_sdforger_dataset_2act_chest_gyro_stats", "outputs/generated/gemma_2act_chest_gyro_stats_ep20"),
    ("chest_stats_ep5", "5epoch", "chest", "artifacts/pamap2_sdforger_dataset_2act_chest_stats", "outputs/generated/gemma_2act_chest_stats_ep5"),
    ("chest_stats_ep10", "10epoch", "chest", "artifacts/pamap2_sdforger_dataset_2act_chest_stats", "outputs/generated/gemma_2act_chest_stats_ep10"),
    ("chest_stats_ep20", "20epoch", "chest", "artifacts/pamap2_sdforger_dataset_2act_chest_stats", "outputs/generated/gemma_2act_chest_stats_ep20"),
    ("chest_acc_stats_ep5", "5epoch", "chest_acc", "artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats", "outputs/generated/gemma_2act_chest_acc_stats_ep5"),
    ("chest_acc_stats_ep10", "10epoch", "chest_acc", "artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats", "outputs/generated/gemma_2act_chest_acc_stats_ep10"),
    ("chest_acc_stats_ep20", "20epoch", "chest_acc", "artifacts/pamap2_sdforger_dataset_2act_chest_acc_stats", "outputs/generated/gemma_2act_chest_acc_stats_ep20"),
]

REAL_COLORS = {"walking": "#7b3fb2", "running": "#4a7c59"}
SYNTHETIC_COLORS = {"walking": "#2c7fb8", "running": "#f28e2b"}
ACTIVITY_ORDER = ["walking", "running"]


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
    epoch: str,
    base_dataset: str,
    dataset_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    samples_per_activity: int,
    seed: int,
    max_cols: int,
    save_pdf: bool,
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
                REAL_COLORS.get(activity, "#777777"),
                f"real {activity}",
                alpha=0.25,
                linewidth=0.75,
            )
            draw_lines(
                ax,
                synthetic_norm,
                synthetic_idx,
                channel_idx,
                SYNTHETIC_COLORS.get(activity, "#2c7fb8"),
                f"synthetic {activity}",
                alpha=0.54,
                linewidth=0.9,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.65, alpha=0.35)
        ax.set_title(channel_name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
        if channel_idx % ncols == 0:
            ax.set_ylabel("standardized value")
        if channel_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("time step")

    for ax in axes_flat[num_channels:]:
        ax.axis("off")

    legend_items = [
        ("real walking", REAL_COLORS["walking"]),
        ("real running", REAL_COLORS["running"]),
        ("synthetic walking", SYNTHETIC_COLORS["walking"]),
        ("synthetic running", SYNTHETIC_COLORS["running"]),
    ]
    handles = [plt.Line2D([], [], color=color, linewidth=2.0, label=label) for label, color in legend_items]
    fig.legend(handles, [label for label, _ in legend_items], loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"{run_name} ({base_dataset}, {epoch}): real and generated windows in train-standardized space",
        y=0.985,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = output_dir / f"{run_name}_standardized_overlay.png"
    fig.savefig(output_path, dpi=220)
    pdf_path = None
    if save_pdf:
        pdf_path = output_dir / f"{run_name}_standardized_overlay.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_name": run_name,
        "epoch": epoch,
        "base_dataset": base_dataset,
        "dataset_dir": str(dataset_dir),
        "generated_dir": str(generated_dir),
        "plot_png": str(output_path),
        "plot_pdf": str(pdf_path) if pdf_path else None,
        "real_shape": list(real_norm.shape),
        "synthetic_shape": list(synthetic_norm.shape),
        "real_channel_mean": np.round(real_norm.mean(axis=(0, 1)), 6).tolist(),
        "synthetic_channel_mean": np.round(synthetic_norm.mean(axis=(0, 1)), 6).tolist(),
        "real_channel_std": np.round(real_norm.std(axis=(0, 1)), 6).tolist(),
        "synthetic_channel_std": np.round(synthetic_norm.std(axis=(0, 1)), 6).tolist(),
        "real_counts": real_metadata["activity_name"].value_counts().to_dict(),
        "synthetic_counts": generated_embeddings["activity_name"].value_counts().to_dict(),
        "channels": channel_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Gemma 2-action stats-prompt epoch runs without the final channel de-standardization."
    )
    parser.add_argument("--output-dir", default="outputs/plots/gemma_2act_stats_normalized_overlays")
    parser.add_argument("--samples-per-activity", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cols", type=int, default=3)
    parser.add_argument("--save-pdf", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    summaries = []
    for run_name, epoch, base_dataset, dataset_dir, generated_dir in RUNS:
        dataset_path = Path(dataset_dir)
        generated_path = Path(generated_dir)
        if not (dataset_path / "reducer.pkl").exists():
            print(f"skip {run_name}: missing reducer at {dataset_path / 'reducer.pkl'}")
            continue
        if not (generated_path / "generated_embeddings.csv").exists():
            print(f"skip {run_name}: missing generated embeddings at {generated_path / 'generated_embeddings.csv'}")
            continue
        summaries.append(
            plot_run(
                run_name=run_name,
                epoch=epoch,
                base_dataset=base_dataset,
                dataset_dir=dataset_path,
                generated_dir=generated_path,
                output_dir=output_dir,
                samples_per_activity=args.samples_per_activity,
                seed=args.seed,
                max_cols=args.max_cols,
                save_pdf=args.save_pdf,
            )
        )
        print(f"wrote {run_name}")

    manifest = {
        "output_dir": str(output_dir.resolve()),
        "space": "real windows standardized with reducer.channel_scaler; generated embeddings decoded before final channel inverse standardization",
        "runs": summaries,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "plot_summary.csv", index=False)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_plots": len(summaries)}))


if __name__ == "__main__":
    main()
