import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.utils import ensure_dir


ACTIVITY_ORDER = ["walking", "running"]
REAL_COLORS = {
    "walking": "#9a9a9a",
    "running": "#2ca02c",
}
SYNTHETIC_COLORS = {
    "walking": "#2c7fb8",
    "running": "#f28e2b",
}
DEFAULT_RUNS = {
    "gemma_2act_hand_ep10": (
        "artifacts/pamap2_sdforger_dataset_2act_hand",
        "outputs/generated/gemma_2act_hand_ep10",
    ),
    "gemma_2act_hand_ep20": (
        "artifacts/pamap2_sdforger_dataset_2act_hand",
        "outputs/generated/gemma_2act_hand_ep20",
    ),
    "gemma_2act_chest_gyro_ep10": (
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro",
        "outputs/generated/gemma_2act_chest_gyro_ep10",
    ),
    "gemma_2act_chest_gyro_ep20": (
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro",
        "outputs/generated/gemma_2act_chest_gyro_ep20",
    ),
    "gemma_2act_chest_ep10": (
        "artifacts/pamap2_sdforger_dataset_2act_chest",
        "outputs/generated/gemma_2act_chest_ep10",
    ),
    "gemma_2act_chest_ep20": (
        "artifacts/pamap2_sdforger_dataset_2act_chest",
        "outputs/generated/gemma_2act_chest_ep20",
    ),
}
FIVE_EPOCH_RUNS = {
    "gemma_2act_hand_ep5": (
        "artifacts/pamap2_sdforger_dataset_2act_hand",
        "outputs/generated/gemma_2act_hand",
    ),
    "gemma_2act_chest_gyro_ep5": (
        "artifacts/pamap2_sdforger_dataset_2act_chest_gyro",
        "outputs/generated/gemma_2act_chest_gyro",
    ),
    "gemma_2act_chest_ep5": (
        "artifacts/pamap2_sdforger_dataset_2act_chest",
        "outputs/generated/gemma_2act_chest",
    ),
}


def load_channel_names(dataset_dir: Path) -> List[str]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload["channel_schema"]["selected_channels"])


def ordered_activities(*metadata_frames: pd.DataFrame) -> List[str]:
    seen = set()
    for frame in metadata_frames:
        seen.update(frame["activity_name"].dropna().astype(str).tolist())
    activities = [activity for activity in ACTIVITY_ORDER if activity in seen]
    activities.extend(sorted(seen.difference(activities)))
    return activities


def sample_indices(metadata: pd.DataFrame, activity: str, samples: int, seed: int) -> np.ndarray:
    candidates = metadata.index[metadata["activity_name"] == activity].to_numpy()
    if len(candidates) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=min(samples, len(candidates)), replace=False)


def draw_activity_set(
    ax,
    time_index: np.ndarray,
    windows: np.ndarray,
    indices: Iterable[int],
    channel_idx: int,
    color: str,
    label: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    indices = list(indices)
    if not indices:
        return
    for line_idx, index in enumerate(indices):
        ax.plot(
            time_index,
            windows[int(index), :, channel_idx],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label if line_idx == 0 else None,
            zorder=zorder,
        )


def make_run_overlay(
    run_name: str,
    dataset_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    samples_per_activity: int,
    seed: int,
    max_cols: int,
) -> Dict[str, object]:
    real_windows = np.load(dataset_dir / "test_windows.npy")
    real_metadata = pd.read_csv(dataset_dir / "test_metadata.csv")
    synthetic_windows = np.load(generated_dir / "generated_windows.npy")
    synthetic_metadata = pd.read_csv(generated_dir / "generated_embeddings.csv")
    channel_names = load_channel_names(dataset_dir)
    activities = ordered_activities(real_metadata, synthetic_metadata)

    if real_windows.shape[2] != synthetic_windows.shape[2]:
        raise ValueError(
            f"{run_name}: channel mismatch real={real_windows.shape} synthetic={synthetic_windows.shape}"
        )

    num_channels = len(channel_names)
    ncols = min(max_cols, num_channels)
    nrows = int(math.ceil(num_channels / ncols))
    fig_width = max(6.0, 5.1 * ncols)
    fig_height = max(3.4, 3.0 * nrows + 1.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True)
    axes_array = np.asarray(axes).reshape(-1)
    time_index = np.arange(real_windows.shape[1])

    for channel_idx, channel_name in enumerate(channel_names):
        ax = axes_array[channel_idx]
        for activity_idx, activity in enumerate(activities):
            real_idx = sample_indices(
                real_metadata,
                activity,
                samples=samples_per_activity,
                seed=seed + activity_idx * 17 + channel_idx,
            )
            syn_idx = sample_indices(
                synthetic_metadata,
                activity,
                samples=samples_per_activity,
                seed=seed + 1000 + activity_idx * 17 + channel_idx,
            )
            draw_activity_set(
                ax=ax,
                time_index=time_index,
                windows=real_windows,
                indices=real_idx,
                channel_idx=channel_idx,
                color=REAL_COLORS.get(activity, "#777777"),
                label=f"real {activity}",
                alpha=0.26 if activity == "walking" else 0.32,
                linewidth=0.8,
                zorder=1 + activity_idx,
            )
            draw_activity_set(
                ax=ax,
                time_index=time_index,
                windows=synthetic_windows,
                indices=syn_idx,
                channel_idx=channel_idx,
                color=SYNTHETIC_COLORS.get(activity, "#2c7fb8"),
                label=f"synthetic {activity}",
                alpha=0.50,
                linewidth=0.9,
                zorder=5 + activity_idx,
            )
        ax.set_title(channel_name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
        if channel_idx % ncols == 0:
            ax.set_ylabel("sensor value")
        if channel_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("time step")

    for ax in axes_array[num_channels:]:
        ax.axis("off")

    legend_items = [
        ("real walking", REAL_COLORS["walking"]),
        ("real running", REAL_COLORS["running"]),
        ("synthetic walking", SYNTHETIC_COLORS["walking"]),
        ("synthetic running", SYNTHETIC_COLORS["running"]),
    ]
    handles = [plt.Line2D([], [], color=color, linewidth=2.0, label=label) for label, color in legend_items]
    fig.legend(
        handles,
        [label for label, _ in legend_items],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"{run_name}: real walking/running vs generated windows across all channels",
        y=0.985,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = output_dir / f"{run_name}_all_channels_overlay.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return {
        "run_name": run_name,
        "plot": str(output_path),
        "dataset_dir": str(dataset_dir),
        "generated_dir": str(generated_dir),
        "real_shape": list(real_windows.shape),
        "synthetic_shape": list(synthetic_windows.shape),
        "real_counts": real_metadata["activity_name"].value_counts().to_dict(),
        "synthetic_counts": synthetic_metadata["activity_name"].value_counts().to_dict(),
        "channels": channel_names,
    }


def parse_run_spec(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Run specs must look like name:dataset_dir:generated_dir"
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot two-activity real/generated overlays for every selected sensor channel."
    )
    parser.add_argument("--output-dir", default="outputs/plots/2act_channel_overlays")
    parser.add_argument("--samples-per-activity", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cols", type=int, default=3)
    parser.add_argument(
        "--include-5epoch",
        action="store_true",
        help="Also plot the original 5-epoch hand/chest/chest_gyro runs.",
    )
    parser.add_argument(
        "--no-default-runs",
        action="store_true",
        help="Do not plot the built-in ep10/ep20 runs; useful with --run.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        default=[],
        help="Custom run as name:dataset_dir:generated_dir. If omitted, ep10/ep20 runs are plotted.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    runs = {} if args.no_default_runs else dict(DEFAULT_RUNS)
    if args.include_5epoch:
        runs = {**FIVE_EPOCH_RUNS, **runs}
    for name, dataset_dir, generated_dir in args.run:
        runs[name] = (dataset_dir, generated_dir)

    manifest: Dict[str, object] = {}
    for run_name, (dataset_dir, generated_dir) in runs.items():
        manifest[run_name] = make_run_overlay(
            run_name=run_name,
            dataset_dir=Path(dataset_dir),
            generated_dir=Path(generated_dir),
            output_dir=output_dir,
            samples_per_activity=args.samples_per_activity,
            seed=args.seed,
            max_cols=args.max_cols,
        )

    manifest_path = output_dir / "channel_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_plots": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
