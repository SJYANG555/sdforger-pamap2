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

from pamap2_forger.utils import ensure_dir


ACTIVITY_ORDER = ["walking", "running", "cycling", "sitting", "standing"]
ACTIVITY_COLORS = {
    "walking": "#4e79a7",
    "running": "#e15759",
    "cycling": "#59a14f",
    "sitting": "#b07aa1",
    "standing": "#f28e2b",
}
DEFAULT_CHANNELS = ["hand_acc_16g_x", "chest_acc_16g_x", "ankle_acc_16g_x"]


def load_channel_names(dataset_manifest: Path) -> List[str]:
    payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    return payload["channel_schema"]["selected_channels"]


def resolve_channel_indices(channel_names: List[str], selected_channels: Iterable[str]) -> List[Tuple[str, int]]:
    resolved: List[Tuple[str, int]] = []
    for channel in selected_channels:
        if channel not in channel_names:
            raise ValueError(f"Unknown channel {channel!r}. Available channels: {channel_names}")
        resolved.append((channel, channel_names.index(channel)))
    return resolved


def ordered_activities(*metadata_frames: pd.DataFrame) -> List[str]:
    seen = set()
    for frame in metadata_frames:
        if "activity_name" in frame.columns:
            seen.update(frame["activity_name"].dropna().astype(str).tolist())
    activities = [activity for activity in ACTIVITY_ORDER if activity in seen]
    activities.extend(sorted(seen.difference(activities)))
    return activities


def sample_indices(metadata: pd.DataFrame, activity: str, n: int, seed: int) -> np.ndarray:
    candidates = metadata.index[metadata["activity_name"] == activity].to_numpy()
    if len(candidates) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    sample_size = min(n, len(candidates))
    return rng.choice(candidates, size=sample_size, replace=False)


def draw_overlay_panel(
    ax,
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    channel_index: int,
    activities: List[str],
    samples_per_activity: int,
    seed: int,
    show_legend: bool,
) -> None:
    time_index = np.arange(real_windows.shape[1])

    for activity_idx, activity in enumerate(activities):
        real_indices = sample_indices(real_metadata, activity, samples_per_activity, seed + activity_idx)
        for index in real_indices:
            ax.plot(
                time_index,
                real_windows[int(index), :, channel_index],
                color="#8a8a8a",
                alpha=0.45,
                linewidth=0.8,
                zorder=1,
            )

    for activity_idx, activity in enumerate(activities):
        synthetic_indices = sample_indices(synthetic_metadata, activity, samples_per_activity, seed + 100 + activity_idx)
        for line_idx, index in enumerate(synthetic_indices):
            ax.plot(
                time_index,
                synthetic_windows[int(index), :, channel_index],
                color=ACTIVITY_COLORS.get(activity, "#333333"),
                alpha=0.78,
                linewidth=1.1,
                label=activity if show_legend and line_idx == 0 else None,
                zorder=2,
            )

    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_xlabel("time step")


def draw_single_activity_panel(
    ax,
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    channel_index: int,
    activity: str,
    samples: int,
    seed: int,
    synthetic_label: str,
    show_legend: bool,
) -> None:
    time_index = np.arange(real_windows.shape[1])
    real_indices = sample_indices(real_metadata, activity, samples, seed)
    synthetic_indices = sample_indices(synthetic_metadata, activity, samples, seed + 100)

    for line_idx, index in enumerate(real_indices):
        ax.plot(
            time_index,
            real_windows[int(index), :, channel_index],
            color="#8a8a8a",
            alpha=0.48,
            linewidth=0.9,
            label="real" if show_legend and line_idx == 0 else None,
            zorder=1,
        )

    for line_idx, index in enumerate(synthetic_indices):
        ax.plot(
            time_index,
            synthetic_windows[int(index), :, channel_index],
            color=ACTIVITY_COLORS.get(activity, "#333333"),
            alpha=0.82,
            linewidth=1.15,
            label=synthetic_label if show_legend and line_idx == 0 else None,
            zorder=2,
        )

    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_xlabel("time step")


def make_activity_overlay(
    model_name: str,
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    channel_indices: List[Tuple[str, int]],
    output_dir: Path,
    samples: int,
    seed: int,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    activities = ordered_activities(real_metadata, synthetic_metadata)
    output_dir = ensure_dir(output_dir)

    for activity_idx, activity in enumerate(activities):
        fig, axes = plt.subplots(1, len(channel_indices), figsize=(5.2 * len(channel_indices), 3.7), sharex=True)
        if len(channel_indices) == 1:
            axes = [axes]
        for panel_idx, (ax, (channel_name, channel_index)) in enumerate(zip(axes, channel_indices)):
            draw_single_activity_panel(
                ax=ax,
                real_windows=real_windows,
                real_metadata=real_metadata,
                synthetic_windows=synthetic_windows,
                synthetic_metadata=synthetic_metadata,
                channel_index=channel_index,
                activity=activity,
                samples=samples,
                seed=seed + activity_idx * 10 + panel_idx,
                synthetic_label=f"{model_name} synthetic",
                show_legend=panel_idx == len(channel_indices) - 1,
            )
            ax.set_title(channel_name, loc="left", fontweight="bold")
            ax.text(
                0.02,
                0.94,
                f"Condition: activity is {activity}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                color="#333333",
            )
            if panel_idx == 0:
                ax.set_ylabel("sensor value")

        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{model_name}: {activity} real vs synthetic windows", y=0.98)
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))
        output_path = output_dir / f"{model_name.lower().replace(' ', '_').replace('-', '')}_{activity}_overlay.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        outputs[f"{model_name}_{activity}"] = output_path

    return outputs


def make_activity_model_comparison(
    model_payloads: Dict[str, Tuple[np.ndarray, pd.DataFrame]],
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    channel_indices: List[Tuple[str, int]],
    output_dir: Path,
    samples: int,
    seed: int,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    activities = ordered_activities(real_metadata, *(metadata for _, metadata in model_payloads.values()))
    output_dir = ensure_dir(output_dir)

    for activity_idx, activity in enumerate(activities):
        fig, axes = plt.subplots(
            len(model_payloads),
            len(channel_indices),
            figsize=(5.2 * len(channel_indices), 3.1 * len(model_payloads)),
            sharex=True,
        )
        if len(model_payloads) == 1:
            axes = np.array([axes])

        for row_idx, (model_name, (synthetic_windows, synthetic_metadata)) in enumerate(model_payloads.items()):
            for panel_idx, (channel_name, channel_index) in enumerate(channel_indices):
                ax = axes[row_idx, panel_idx]
                draw_single_activity_panel(
                    ax=ax,
                    real_windows=real_windows,
                    real_metadata=real_metadata,
                    synthetic_windows=synthetic_windows,
                    synthetic_metadata=synthetic_metadata,
                    channel_index=channel_index,
                    activity=activity,
                    samples=samples,
                    seed=seed + activity_idx * 10 + row_idx * 100 + panel_idx,
                    synthetic_label=f"{model_name} synthetic",
                    show_legend=row_idx == 0 and panel_idx == len(channel_indices) - 1,
                )
                if row_idx == 0:
                    ax.set_title(channel_name, loc="left", fontweight="bold")
                if panel_idx == 0:
                    ax.set_ylabel(f"{model_name}\nsensor value")

        handles, labels = axes[0, -1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{activity}: GPT-2 vs Gemma real/synthetic windows", y=0.98)
        fig.tight_layout(rect=(0, 0.11, 1, 0.94))
        output_path = output_dir / f"gpt2_vs_gemma_{activity}_overlay.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        outputs[f"gpt2_vs_gemma_{activity}"] = output_path

    return outputs


def make_model_overlay(
    model_name: str,
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    channel_indices: List[Tuple[str, int]],
    output_path: Path,
    samples_per_activity: int,
    seed: int,
) -> Path:
    activities = ordered_activities(real_metadata, synthetic_metadata)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, len(channel_indices), figsize=(5.4 * len(channel_indices), 3.7), sharex=True)
    if len(channel_indices) == 1:
        axes = [axes]

    for panel_idx, (ax, (channel_name, channel_index)) in enumerate(zip(axes, channel_indices)):
        draw_overlay_panel(
            ax=ax,
            real_windows=real_windows,
            real_metadata=real_metadata,
            synthetic_windows=synthetic_windows,
            synthetic_metadata=synthetic_metadata,
            channel_index=channel_index,
            activities=activities,
            samples_per_activity=samples_per_activity,
            seed=seed,
            show_legend=panel_idx == len(channel_indices) - 1,
        )
        ax.set_title(channel_name, loc="left", fontweight="bold")
        ax.text(
            0.02,
            0.94,
            "Condition: activity label",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#333333",
        )
        if panel_idx == 0:
            ax.set_ylabel("sensor value")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        handles.insert(0, plt.Line2D([], [], color="#8a8a8a", linewidth=1.2, label="real"))
        labels.insert(0, "real")
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)

    fig.suptitle(f"{model_name}: real gray windows vs synthetic colored windows", y=0.98)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def make_model_comparison(
    model_payloads: Dict[str, Tuple[np.ndarray, pd.DataFrame]],
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    channel_indices: List[Tuple[str, int]],
    output_path: Path,
    samples_per_activity: int,
    seed: int,
) -> Path:
    activities = ordered_activities(real_metadata, *(metadata for _, metadata in model_payloads.values()))
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(
        len(model_payloads),
        len(channel_indices),
        figsize=(5.2 * len(channel_indices), 3.1 * len(model_payloads)),
        sharex=True,
    )
    if len(model_payloads) == 1:
        axes = np.array([axes])

    for row_idx, (model_name, (synthetic_windows, synthetic_metadata)) in enumerate(model_payloads.items()):
        for col_idx, (channel_name, channel_index) in enumerate(channel_indices):
            ax = axes[row_idx, col_idx]
            draw_overlay_panel(
                ax=ax,
                real_windows=real_windows,
                real_metadata=real_metadata,
                synthetic_windows=synthetic_windows,
                synthetic_metadata=synthetic_metadata,
                channel_index=channel_index,
                activities=activities,
                samples_per_activity=samples_per_activity,
                seed=seed + row_idx * 10,
                show_legend=row_idx == 0 and col_idx == len(channel_indices) - 1,
            )
            if row_idx == 0:
                ax.set_title(channel_name, loc="left", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nsensor value")

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        handles.insert(0, plt.Line2D([], [], color="#8a8a8a", linewidth=1.2, label="real"))
        labels.insert(0, "real")
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)

    fig.suptitle("GPT-2 vs Gemma synthetic windows under activity conditions", y=0.98)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Forging-style real/synthetic overlay plots for GPT-2 and Gemma.")
    parser.add_argument("--real-windows", default="artifacts/pamap2_sdforger_dataset/test_windows.npy")
    parser.add_argument("--real-metadata", default="artifacts/pamap2_sdforger_dataset/test_metadata.csv")
    parser.add_argument("--dataset-manifest", default="artifacts/pamap2_sdforger_dataset/dataset_manifest.json")
    parser.add_argument("--gpt2-windows", default="outputs/generated/gpt2_5class/generated_windows.npy")
    parser.add_argument("--gpt2-metadata", default="outputs/generated/gpt2_5class/generated_embeddings.csv")
    parser.add_argument("--gemma-windows", default="outputs/generated/gemma_5class/generated_windows.npy")
    parser.add_argument("--gemma-metadata", default="outputs/generated/gemma_5class/generated_embeddings.csv")
    parser.add_argument("--output-dir", default="outputs/plots/final_forging_overlay")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--samples-per-activity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    channel_names = load_channel_names(Path(args.dataset_manifest))
    channel_indices = resolve_channel_indices(channel_names, args.channels)

    real_windows = np.load(args.real_windows)
    real_metadata = pd.read_csv(args.real_metadata)
    gpt2_windows = np.load(args.gpt2_windows)
    gpt2_metadata = pd.read_csv(args.gpt2_metadata)
    gemma_windows = np.load(args.gemma_windows)
    gemma_metadata = pd.read_csv(args.gemma_metadata)

    output_dir = ensure_dir(args.output_dir)
    outputs = {
        "gpt2_overlay": make_model_overlay(
            model_name="GPT-2",
            real_windows=real_windows,
            real_metadata=real_metadata,
            synthetic_windows=gpt2_windows,
            synthetic_metadata=gpt2_metadata,
            channel_indices=channel_indices,
            output_path=output_dir / "gpt2_forging_overlay_three_channels.png",
            samples_per_activity=args.samples_per_activity,
            seed=args.seed,
        ),
        "gemma_overlay": make_model_overlay(
            model_name="Gemma 2 2B LoRA",
            real_windows=real_windows,
            real_metadata=real_metadata,
            synthetic_windows=gemma_windows,
            synthetic_metadata=gemma_metadata,
            channel_indices=channel_indices,
            output_path=output_dir / "gemma_forging_overlay_three_channels.png",
            samples_per_activity=args.samples_per_activity,
            seed=args.seed,
        ),
        "gpt2_vs_gemma_overlay": make_model_comparison(
            model_payloads={
                "GPT-2": (gpt2_windows, gpt2_metadata),
                "Gemma": (gemma_windows, gemma_metadata),
            },
            real_windows=real_windows,
            real_metadata=real_metadata,
            channel_indices=channel_indices,
            output_path=output_dir / "gpt2_vs_gemma_forging_overlay_three_channels.png",
            samples_per_activity=args.samples_per_activity,
            seed=args.seed,
        ),
    }
    by_activity_dir = ensure_dir(output_dir / "by_activity")
    outputs.update(
        make_activity_overlay(
            model_name="GPT-2",
            real_windows=real_windows,
            real_metadata=real_metadata,
            synthetic_windows=gpt2_windows,
            synthetic_metadata=gpt2_metadata,
            channel_indices=channel_indices,
            output_dir=by_activity_dir / "gpt2",
            samples=args.samples_per_activity,
            seed=args.seed,
        )
    )
    outputs.update(
        make_activity_overlay(
            model_name="Gemma",
            real_windows=real_windows,
            real_metadata=real_metadata,
            synthetic_windows=gemma_windows,
            synthetic_metadata=gemma_metadata,
            channel_indices=channel_indices,
            output_dir=by_activity_dir / "gemma",
            samples=args.samples_per_activity,
            seed=args.seed,
        )
    )
    outputs.update(
        make_activity_model_comparison(
            model_payloads={
                "GPT-2": (gpt2_windows, gpt2_metadata),
                "Gemma": (gemma_windows, gemma_metadata),
            },
            real_windows=real_windows,
            real_metadata=real_metadata,
            channel_indices=channel_indices,
            output_dir=by_activity_dir / "gpt2_vs_gemma",
            samples=args.samples_per_activity,
            seed=args.seed,
        )
    )
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} overlay plots to {output_dir}")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
