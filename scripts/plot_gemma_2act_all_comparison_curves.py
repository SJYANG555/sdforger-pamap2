import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_COLORS = {
    "baseline": "#4e79a7",
    "epoch_extension_no_stats": "#f28e2b",
    "stats_prompt": "#59a14f",
}
GROUP_LABELS = {
    "baseline": "baseline",
    "epoch_extension_no_stats": "no stats epoch",
    "stats_prompt": "stats prompt",
}
DATASET_ORDER = ["full", "hand", "chest", "hand_acc", "chest_acc", "hand_gyro", "chest_gyro"]
EPOCH_ORDER = {"5epoch": 5, "10epoch": 10, "20epoch": 20}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["epoch_num"] = frame["epoch"].map(EPOCH_ORDER)
    frame["retention_rate"] = frame["valid_generated"] / frame["num_prompts"].replace(0, np.nan)
    frame["run_walk_ratio"] = frame["generated_running"] / (
        frame["generated_walking"] + frame["generated_running"]
    ).replace(0, np.nan)
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def style_axis(ax, ylabel: str | None = None, ylim: tuple[float, float] | None = None) -> None:
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_bar_ranked(frame: pd.DataFrame, value: str, title: str, ylabel: str, output_dir: Path, stem: str) -> list[str]:
    plot_frame = frame.sort_values(value, ascending=False).copy()
    labels = [f"{row.variant}\n{row.epoch}" for row in plot_frame.itertuples()]
    colors = [GROUP_COLORS.get(group, "#777777") for group in plot_frame["group"]]

    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.bar(np.arange(len(plot_frame)), plot_frame[value], color=colors, width=0.72)
    ax.set_xticks(np.arange(len(plot_frame)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    style_axis(ax, ylabel)
    handles = [
        plt.Line2D([], [], color=color, marker="s", linestyle="", markersize=8, label=GROUP_LABELS[group])
        for group, color in GROUP_COLORS.items()
        if group in set(plot_frame["group"])
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    return save_figure(fig, output_dir, stem)


def series_for_dataset(frame: pd.DataFrame, dataset: str, group: str, value: str) -> pd.DataFrame:
    subset = frame[(frame["base_dataset"] == dataset) & (frame["group"] == group)].copy()
    subset = subset.dropna(subset=["epoch_num", value])
    return subset.sort_values("epoch_num")


def plot_metric_curves(
    frame: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    datasets: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[str]:
    datasets = datasets or DATASET_ORDER
    datasets = [dataset for dataset in datasets if dataset in set(frame["base_dataset"])]
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, dataset in zip(axes_flat, datasets):
        baseline = series_for_dataset(frame, dataset, "baseline", value)
        if not baseline.empty:
            ax.scatter(
                baseline["epoch_num"],
                baseline[value],
                color=GROUP_COLORS["baseline"],
                s=54,
                marker="D",
                label="baseline",
                zorder=4,
            )
        no_stats = series_for_dataset(frame, dataset, "epoch_extension_no_stats", value)
        if not no_stats.empty:
            ax.plot(
                no_stats["epoch_num"],
                no_stats[value],
                color=GROUP_COLORS["epoch_extension_no_stats"],
                marker="o",
                linewidth=2.0,
                label="no stats epoch",
            )
        stats = series_for_dataset(frame, dataset, "stats_prompt", value)
        if not stats.empty:
            ax.plot(
                stats["epoch_num"],
                stats[value],
                color=GROUP_COLORS["stats_prompt"],
                marker="o",
                linewidth=2.0,
                label="stats prompt",
            )
        ax.set_title(dataset, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel("epoch")
        style_axis(ax, ylabel, ylim)

    for ax in axes_flat[len(datasets):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, stem)


def plot_similarity_pair(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(frame["base_dataset"])]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13.5, 3.0 * len(datasets)), squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, value in enumerate(["ED_vs_real_baseline", "DTW_vs_real_baseline"]):
            ax = axes[row_idx, col_idx]
            for group in ("baseline", "epoch_extension_no_stats", "stats_prompt"):
                subset = series_for_dataset(frame, dataset, group, value)
                if subset.empty:
                    continue
                marker = "D" if group == "baseline" else "o"
                linestyle = "" if group == "baseline" else "-"
                ax.plot(
                    subset["epoch_num"],
                    subset[value],
                    color=GROUP_COLORS[group],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.0,
                    label=GROUP_LABELS[group],
                )
            ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.65)
            ax.set_title(f"{dataset} - {value.replace('_', ' ')}", loc="left", fontsize=10, fontweight="bold")
            ax.set_xticks([5, 10, 20])
            ax.set_xlabel("epoch")
            style_axis(ax, "ratio, lower is better")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Similarity vs Real Baseline", x=0.02, y=0.998, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return save_figure(fig, output_dir, "similarity_ed_dtw_vs_real_baseline_curves")


def plot_generation_distribution(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    plot_frame = frame.copy()
    labels = [f"{row.variant}\n{row.epoch}" for row in plot_frame.itertuples()]
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(16, 6.3))
    ax.bar(x, plot_frame["generated_walking"], color="#2c7fb8", label="walking")
    ax.bar(
        x,
        plot_frame["generated_running"],
        bottom=plot_frame["generated_walking"],
        color="#f28e2b",
        label="running",
    )
    ax.axhline(305, color="#666666", linewidth=1.1, linestyle="--", alpha=0.7, label="305 prompts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_title("Generated Class Distribution", loc="left", fontsize=14, fontweight="bold")
    style_axis(ax, "valid generated windows")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    return save_figure(fig, output_dir, "generated_class_distribution_stacked")


def plot_dashboard(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    metrics = [
        ("synthetic_only_acc", "Synthetic-only accuracy", "higher is better"),
        ("real_plus_synthetic_acc", "Real + synthetic accuracy", "higher is better"),
        ("ED_vs_real_baseline", "ED vs real baseline", "lower is better"),
        ("DTW_vs_real_baseline", "DTW vs real baseline", "lower is better"),
    ]
    for ax, (value, title, subtitle) in zip(axes.ravel(), metrics):
        top = frame.sort_values(value, ascending=value.startswith(("ED", "DTW"))).head(12)
        y = np.arange(len(top))
        ax.barh(y, top[value], color=[GROUP_COLORS.get(g, "#777777") for g in top["group"]])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.variant} ({r.epoch})" for r in top.itertuples()], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{title} - top 12, {subtitle}", loc="left", fontsize=11, fontweight="bold")
        style_axis(ax)
    handles = [
        plt.Line2D([], [], color=color, marker="s", linestyle="", markersize=8, label=GROUP_LABELS[group])
        for group, color in GROUP_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Gemma 2-Action All Experiment Dashboard", x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, "all_experiment_dashboard")


def write_manifest(output_dir: Path, generated: list[str], args: argparse.Namespace) -> None:
    manifest = {
        "summary_csv": str(Path(args.summary_csv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "plots": generated,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Gemma 2-action baseline/epoch/stats comparison curves.")
    parser.add_argument(
        "--summary-csv",
        default="outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv",
    )
    parser.add_argument("--output-dir", default="outputs/plots/gemma_2act_all_comparison_curves")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    frame = load_frame(Path(args.summary_csv))
    frame.to_csv(output_dir / "plot_input_summary.csv", index=False)

    generated: list[str] = []
    generated.extend(
        plot_metric_curves(
            frame,
            "synthetic_only_acc",
            "Synthetic-only HAR Utility",
            "accuracy",
            output_dir,
            "synthetic_only_accuracy_curves",
            ylim=(0, 1.02),
        )
    )
    generated.extend(
        plot_metric_curves(
            frame,
            "synthetic_only_macro_f1",
            "Synthetic-only HAR Macro F1",
            "macro F1",
            output_dir,
            "synthetic_only_macro_f1_curves",
            ylim=(0, 1.02),
        )
    )
    generated.extend(
        plot_metric_curves(
            frame,
            "real_plus_synthetic_acc",
            "Real + Synthetic HAR Utility",
            "accuracy",
            output_dir,
            "real_plus_synthetic_accuracy_curves",
            ylim=(0.5, 1.02),
        )
    )
    generated.extend(
        plot_metric_curves(
            frame,
            "retention_rate",
            "Generation Retention",
            "valid / prompts",
            output_dir,
            "generation_retention_curves",
            ylim=(0, 1.05),
        )
    )
    generated.extend(plot_similarity_pair(frame, output_dir))
    generated.extend(plot_generation_distribution(frame, output_dir))
    generated.extend(
        plot_bar_ranked(
            frame,
            "synthetic_only_acc",
            "All Runs Ranked by Synthetic-only Accuracy",
            "accuracy",
            output_dir,
            "ranked_synthetic_only_accuracy",
        )
    )
    generated.extend(
        plot_bar_ranked(
            frame,
            "ED_vs_real_baseline",
            "All Runs Ranked by ED vs Real Baseline",
            "ED ratio, lower is better",
            output_dir,
            "ranked_ed_vs_real_baseline",
        )
    )
    generated.extend(plot_dashboard(frame, output_dir))
    write_manifest(output_dir, generated, args)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_files": len(generated)}, indent=2))


if __name__ == "__main__":
    main()
