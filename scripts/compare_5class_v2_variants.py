import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUNS = [
    ("gpt2", "all", "GPT-2", "Hand + chest + ankle"),
    ("gpt2", "hand", "GPT-2", "Hand"),
    ("gpt2", "chest", "GPT-2", "Chest"),
    ("gpt2", "hand_chest", "GPT-2", "Hand + chest"),
    ("gemma", "all", "Gemma 2 2B LoRA", "Hand + chest + ankle"),
    ("gemma", "hand", "Gemma 2 2B LoRA", "Hand"),
    ("gemma", "chest", "Gemma 2 2B LoRA", "Chest"),
    ("gemma", "hand_chest", "Gemma 2 2B LoRA", "Hand + chest"),
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def run_prefix(model_key: str, subset: str) -> str:
    model_prefix = "gemma" if model_key == "gemma" else "gpt2"
    if subset == "all":
        return f"{model_prefix}_5class_v2"
    return f"{model_prefix}_5class_v2_{subset}"


def empty_metrics() -> dict:
    return {
        "synthetic_only_accuracy": np.nan,
        "real_only_accuracy": np.nan,
        "real_plus_synthetic_accuracy": np.nan,
        "augmentation_accuracy_delta": np.nan,
        "synthetic_only_macro_f1": np.nan,
        "real_plus_synthetic_macro_f1": np.nan,
        "overall_MDD": np.nan,
        "overall_ACD": np.nan,
        "overall_SD": np.nan,
        "overall_KD": np.nan,
        "overall_ED": np.nan,
        "overall_DTW": np.nan,
    }


def build_summary(root: Path) -> pd.DataFrame:
    rows = []
    for model_key, subset, model_label, subset_label in RUNS:
        prefix = run_prefix(model_key, subset)
        gen = read_json(root / "outputs" / "generated" / prefix / "generation_summary.json")
        metrics = empty_metrics()
        utility_path = root / "outputs" / "evaluation" / "utility" / prefix / "utility_metrics.csv"
        similarity_path = root / "outputs" / "evaluation" / "similarity" / prefix / "similarity_metrics.csv"
        if utility_path.exists() and similarity_path.exists() and int(gen["num_generated_embeddings"]) > 0:
            utility = pd.read_csv(utility_path)
            similarity = pd.read_csv(similarity_path)
            util = utility.set_index("setting")
            sim_overall = similarity.set_index("activity_name").loc["overall_mean"]
            metrics.update(
                {
                    "synthetic_only_accuracy": float(util.loc["synthetic_only", "accuracy"]),
                    "real_only_accuracy": float(util.loc["real_only", "accuracy"]),
                    "real_plus_synthetic_accuracy": float(util.loc["real_plus_synthetic", "accuracy"]),
                    "augmentation_accuracy_delta": float(
                        util.loc["real_plus_synthetic", "accuracy"] - util.loc["real_only", "accuracy"]
                    ),
                    "synthetic_only_macro_f1": float(util.loc["synthetic_only", "macro_f1"]),
                    "real_plus_synthetic_macro_f1": float(util.loc["real_plus_synthetic", "macro_f1"]),
                    "overall_MDD": float(sim_overall["MDD"]),
                    "overall_ACD": float(sim_overall["ACD"]),
                    "overall_SD": float(sim_overall["SD"]),
                    "overall_KD": float(sim_overall["KD"]),
                    "overall_ED": float(sim_overall["ED"]),
                    "overall_DTW": float(sim_overall["DTW"]),
                }
            )

        rows.append(
            {
                "model": model_label,
                "model_key": model_key,
                "subset": subset_label,
                "subset_key": subset,
                "num_prompts": int(gen["num_prompts"]),
                "num_generated": int(gen["num_generated_embeddings"]),
                "retention_rate": float(gen["num_generated_embeddings"] / gen["num_prompts"]),
                "after_missing_filter": int(gen["filter_stats"]["after_missing_filter"]),
                "after_norm_filter": int(gen["filter_stats"]["after_norm_filter"]),
                "duplicate_ratio": gen["duplicate_ratio"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_bars(frame: pd.DataFrame, value: str, ylabel: str, title: str, output_path: Path) -> None:
    subsets = ["Hand + chest + ankle", "Hand", "Chest", "Hand + chest"]
    models = ["GPT-2", "Gemma 2 2B LoRA"]
    colors = {"GPT-2": "#4C78A8", "Gemma 2 2B LoRA": "#F58518"}
    x = np.arange(len(subsets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for idx, model in enumerate(models):
        values = [
            frame[(frame["model"] == model) & (frame["subset"] == subset)][value].iloc[0]
            for subset in subsets
        ]
        bars = ax.bar(x + (idx - 0.5) * width, values, width, label=model, color=colors[model])
        for bar, item in zip(bars, values):
            if np.isfinite(item):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{item:.3f}" if abs(item) < 10 else f"{item:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x, subsets, rotation=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(frame: pd.DataFrame, output_path: Path) -> None:
    best_retention = frame.sort_values("retention_rate", ascending=False).iloc[0]
    valid_metrics = frame.dropna(subset=["synthetic_only_accuracy", "overall_DTW", "augmentation_accuracy_delta"])
    best_synth_acc = valid_metrics.sort_values("synthetic_only_accuracy", ascending=False).iloc[0]
    best_dtw = valid_metrics.sort_values("overall_DTW", ascending=True).iloc[0]
    best_aug = valid_metrics.sort_values("augmentation_accuracy_delta", ascending=False).iloc[0]

    table_columns = [
        "model",
        "subset",
        "num_generated",
        "retention_rate",
        "synthetic_only_accuracy",
        "real_plus_synthetic_accuracy",
        "augmentation_accuracy_delta",
        "overall_DTW",
        "overall_ED",
    ]
    table_frame = frame[table_columns].copy()
    for column in table_frame.columns:
        if pd.api.types.is_float_dtype(table_frame[column]):
            table_frame[column] = table_frame[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for row in table_frame.to_dict(orient="records"):
        table_lines.append("| " + " | ".join(str(row[column]) for column in table_columns) + " |")

    lines = [
        "# PAMAP2 V2 Channel-Subset Comparison",
        "",
        "## Key Takeaways",
        "",
        f"- Highest generation retention: {best_retention['model']} / {best_retention['subset']} "
        f"({best_retention['retention_rate']:.1%}, {int(best_retention['num_generated'])}/"
        f"{int(best_retention['num_prompts'])}).",
        f"- Highest synthetic-only HAR accuracy: {best_synth_acc['model']} / {best_synth_acc['subset']} "
        f"({best_synth_acc['synthetic_only_accuracy']:.4f}).",
        f"- Lowest overall DTW: {best_dtw['model']} / {best_dtw['subset']} ({best_dtw['overall_DTW']:.2f}).",
        f"- Best real+synthetic augmentation delta: {best_aug['model']} / {best_aug['subset']} "
        f"({best_aug['augmentation_accuracy_delta']:+.4f}).",
        "",
        "Blank metric cells mean no valid synthetic windows were available for evaluation.",
        "",
        "## Summary Table",
        "",
        *table_lines,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPT-2/Gemma v2 channel-subset PAMAP2 results.")
    parser.add_argument("--output-dir", default="outputs/plots/model_comparison_5class_v2_variants")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_summary(PROJECT_ROOT)
    frame.to_csv(out_dir / "variant_comparison_summary.csv", index=False)
    write_markdown(frame, out_dir / "variant_comparison_summary.md")

    plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    plot_grouped_bars(frame, "retention_rate", "Retention rate", "Generation Retention", out_dir / "generation_retention.png")
    plot_grouped_bars(
        frame,
        "synthetic_only_accuracy",
        "Accuracy",
        "Synthetic-only HAR Utility",
        out_dir / "synthetic_only_accuracy.png",
    )
    plot_grouped_bars(
        frame,
        "augmentation_accuracy_delta",
        "Accuracy delta",
        "Real + Synthetic vs Real-only",
        out_dir / "augmentation_delta.png",
    )
    plot_grouped_bars(frame, "overall_DTW", "DTW, lower is better", "Overall Temporal Similarity", out_dir / "overall_dtw.png")

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
