import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUNS = [
    {
        "model": "GPT-2",
        "subset": "18ch full",
        "prefix": "gpt2_5class_v2_compact",
        "generation_prefix": "gpt2_5class_v2_compact_ckpt1300_reparsed",
    },
    {
        "model": "Gemma 2 2B",
        "subset": "18ch full",
        "prefix": "gemma_5class_v2",
    },
    {
        "model": "Llama 3.2 3B",
        "subset": "18ch full",
        "prefix": "llama32_3b_5class_v2",
    },
    {
        "model": "GPT-2",
        "subset": "12ch hand+chest",
        "prefix": "gpt2_5class_v2_hand_chest",
    },
    {
        "model": "Gemma 2 2B",
        "subset": "12ch hand+chest",
        "prefix": "gemma_5class_v2_hand_chest",
    },
    {
        "model": "Llama 3.2 3B",
        "subset": "12ch hand+chest",
        "prefix": "llama32_3b_5class_v2_hand_chest",
    },
]

METRICS = ["MDD", "ACD", "SD", "KD", "ED", "DTW", "SHR"]
ACTIVITY_ORDER = ["cycling", "running", "sitting", "standing", "walking"]
MODEL_ORDER = ["GPT-2", "Gemma 2 2B", "Llama 3.2 3B"]
SUBSET_ORDER = ["18ch full", "12ch hand+chest"]
COLORS = {
    "GPT-2": "#4C78A8",
    "Gemma 2 2B": "#F58518",
    "Llama 3.2 3B": "#54A24B",
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for run in RUNS:
        prefix = run["prefix"]
        generation_prefix = run.get("generation_prefix", prefix)
        generation_path = root / "outputs" / "generated" / generation_prefix / "generation_summary.json"
        utility_path = root / "outputs" / "evaluation" / "utility" / prefix / "utility_metrics.csv"
        similarity_path = root / "outputs" / "evaluation" / "similarity" / prefix / "similarity_metrics.csv"
        if not generation_path.exists() or not utility_path.exists() or not similarity_path.exists():
            continue

        generation = read_json(generation_path)
        utility = pd.read_csv(utility_path).set_index("setting")
        similarity = pd.read_csv(similarity_path).set_index("activity_name")
        overall = similarity.loc["overall_mean"]
        real_only = float(utility.loc["real_only", "accuracy"])
        real_plus_synthetic = float(utility.loc["real_plus_synthetic", "accuracy"])

        rows.append(
            {
                "model": run["model"],
                "subset": run["subset"],
                "prefix": prefix,
                "num_prompts": int(generation["num_prompts"]),
                "num_generated": int(generation["num_generated_embeddings"]),
                "retention_rate": float(generation["num_generated_embeddings"] / generation["num_prompts"]),
                "parse_success_rate": float(generation["num_parse_success"] / generation["num_prompts"]),
                "duplicate_ratio": float(generation["duplicate_ratio"]),
                "real_only_accuracy": real_only,
                "synthetic_only_accuracy": float(utility.loc["synthetic_only", "accuracy"]),
                "real_plus_synthetic_accuracy": real_plus_synthetic,
                "augmentation_delta": real_plus_synthetic - real_only,
                "real_only_macro_f1": float(utility.loc["real_only", "macro_f1"]),
                "synthetic_only_macro_f1": float(utility.loc["synthetic_only", "macro_f1"]),
                "real_plus_synthetic_macro_f1": float(utility.loc["real_plus_synthetic", "macro_f1"]),
                **{f"overall_{metric}": float(overall[metric]) if metric in overall.index else np.nan for metric in METRICS},
            }
        )
    return pd.DataFrame(rows)


def load_activity_similarity(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for run in RUNS:
        path = root / "outputs" / "evaluation" / "similarity" / run["prefix"] / "similarity_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["activity_name"] != "overall_mean"].copy()
        frame["model"] = run["model"]
        frame["subset"] = run["subset"]
        frame["prefix"] = run["prefix"]
        rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _annotate_bars(ax, bars, values, fmt="{:.3f}") -> None:
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.004 if y >= 0 else -0.004
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset, fmt.format(value), ha="center", va=va, fontsize=8)


def plot_metric_by_subset(frame: pd.DataFrame, value: str, ylabel: str, title: str, output_path: Path) -> None:
    x = np.arange(len(SUBSET_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for idx, model in enumerate(MODEL_ORDER):
        values = []
        for subset in SUBSET_ORDER:
            match = frame[(frame["model"] == model) & (frame["subset"] == subset)]
            values.append(float(match[value].iloc[0]) if not match.empty else np.nan)
        bars = ax.bar(x + (idx - 1) * width, values, width, label=model, color=COLORS[model])
        _annotate_bars(ax, bars, values)
    ax.set_xticks(x, SUBSET_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    if "delta" in value:
        ax.axhline(0, color="#333333", linewidth=0.8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_groups(frame: pd.DataFrame, output_path: Path) -> None:
    settings = [
        ("real_only_accuracy", "Real-only"),
        ("synthetic_only_accuracy", "Synthetic-only"),
        ("real_plus_synthetic_accuracy", "Real + synthetic"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
    for ax, subset in zip(axes, SUBSET_ORDER):
        subset_frame = frame[frame["subset"] == subset]
        x = np.arange(len(settings))
        width = 0.24
        for idx, model in enumerate(MODEL_ORDER):
            values = []
            for column, _ in settings:
                match = subset_frame[subset_frame["model"] == model]
                values.append(float(match[column].iloc[0]) if not match.empty else np.nan)
            ax.bar(x + (idx - 1) * width, values, width, label=model, color=COLORS[model])
        ax.set_xticks(x, [label for _, label in settings], rotation=15, ha="right")
        ax.set_title(subset)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("HAR Utility by Model and Training Setting")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path, cmap: str = "viridis") -> None:
    values = matrix.to_numpy(dtype=float)
    fig_width = max(7.5, 1.0 * len(matrix.columns) + 2.2)
    fig_height = max(3.6, 0.55 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_overall_similarity_matrices(frame: pd.DataFrame, out_dir: Path) -> None:
    for subset in SUBSET_ORDER:
        subset_frame = frame[frame["subset"] == subset].set_index("model")
        matrix = subset_frame.loc[[m for m in MODEL_ORDER if m in subset_frame.index], [f"overall_{m}" for m in METRICS]]
        matrix.columns = METRICS
        safe_name = subset.replace(" ", "_").replace("+", "plus").lower()
        plot_heatmap(
            matrix,
            f"Overall Similarity Metrics ({subset})",
            out_dir / f"similarity_overall_matrix_{safe_name}.png",
            cmap="magma",
        )


def plot_activity_similarity_matrices(activity_frame: pd.DataFrame, out_dir: Path) -> None:
    for subset in SUBSET_ORDER:
        subset_frame = activity_frame[activity_frame["subset"] == subset]
        safe_name = subset.replace(" ", "_").replace("+", "plus").lower()
        for metric in ["ED", "DTW"]:
            matrix = subset_frame.pivot(index="activity_name", columns="model", values=metric)
            rows = [activity for activity in ACTIVITY_ORDER if activity in matrix.index]
            cols = [model for model in MODEL_ORDER if model in matrix.columns]
            matrix = matrix.loc[rows, cols]
            plot_heatmap(
                matrix,
                f"{metric} by Activity and Model ({subset})",
                out_dir / f"similarity_activity_{metric.lower()}_{safe_name}.png",
                cmap="viridis",
            )


def write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    table = summary[
        [
            "model",
            "subset",
            "num_generated",
            "retention_rate",
            "synthetic_only_accuracy",
            "real_plus_synthetic_accuracy",
            "augmentation_delta",
            "overall_DTW",
            "overall_ED",
        ]
    ].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{value:.4f}")

    lines = [
        "# Five-Class Model Comparison for Report",
        "",
        "All runs use the five PAMAP2 activities: cycling, running, sitting, standing, walking.",
        "",
        "Implemented SDForger-style similarity metrics: MDD, ACD, SD (Skewness Difference), KD, ED, DTW, and SHR.",
        "Downstream HAR utility: RandomForest real-only, synthetic-only, and real+synthetic accuracy/F1.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready comparison plots for five-class PAMAP2 runs.")
    parser.add_argument("--output-dir", default="outputs/plots/report_model_comparison_5class_v2")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(PROJECT_ROOT)
    activity_similarity = load_activity_similarity(PROJECT_ROOT)
    summary.to_csv(out_dir / "report_model_comparison_summary.csv", index=False)
    activity_similarity.to_csv(out_dir / "report_activity_similarity_long.csv", index=False)
    write_markdown(summary, out_dir / "report_model_comparison_summary.md")

    plt.rcParams.update({"figure.dpi": 170, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    plot_accuracy_groups(summary, out_dir / "utility_accuracy_by_model_and_subset.png")
    plot_metric_by_subset(summary, "augmentation_delta", "Accuracy delta", "Real + Synthetic minus Real-only", out_dir / "augmentation_delta_by_subset.png")
    plot_metric_by_subset(summary, "synthetic_only_accuracy", "Accuracy", "Synthetic-only HAR Utility", out_dir / "synthetic_only_accuracy_by_subset.png")
    plot_metric_by_subset(summary, "retention_rate", "Retention rate", "Generation Retention", out_dir / "generation_retention_by_subset.png")
    plot_overall_similarity_matrices(summary, out_dir)
    plot_activity_similarity_matrices(activity_similarity, out_dir)

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
