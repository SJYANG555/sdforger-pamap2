import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPT-2 vs Gemma five-class model comparison.")
    parser.add_argument("--gpt2-generated", default="outputs/generated/gpt2_5class/generation_summary.json")
    parser.add_argument("--gemma-generated", default="outputs/generated/gemma_5class/generation_summary.json")
    parser.add_argument("--gpt2-utility", default="outputs/evaluation/utility/gpt2_5class/utility_metrics.csv")
    parser.add_argument("--gemma-utility", default="outputs/evaluation/utility/gemma_5class/utility_metrics.csv")
    parser.add_argument("--gpt2-similarity", default="outputs/evaluation/similarity/gpt2_5class/similarity_metrics.csv")
    parser.add_argument("--gemma-similarity", default="outputs/evaluation/similarity/gemma_5class/similarity_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/plots/model_comparison_5class")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = {
        "GPT-2": read_json(args.gpt2_generated),
        "Gemma 2 2B LoRA": read_json(args.gemma_generated),
    }
    utility = {
        "GPT-2": pd.read_csv(ROOT / args.gpt2_utility),
        "Gemma 2 2B LoRA": pd.read_csv(ROOT / args.gemma_utility),
    }
    similarity = {
        "GPT-2": pd.read_csv(ROOT / args.gpt2_similarity),
        "Gemma 2 2B LoRA": pd.read_csv(ROOT / args.gemma_similarity),
    }

    summary_rows = []
    for model, payload in gen.items():
        util = utility[model].set_index("setting")
        sim_overall = similarity[model].set_index("activity_name").loc["overall_mean"]
        summary_rows.append(
            {
                "model": model,
                "num_prompts": payload["num_prompts"],
                "num_generated_embeddings": payload["num_generated_embeddings"],
                "retention_rate": payload["num_generated_embeddings"] / payload["num_prompts"],
                "duplicate_ratio": payload["duplicate_ratio"],
                "synthetic_only_accuracy": util.loc["synthetic_only", "accuracy"],
                "real_plus_synthetic_accuracy": util.loc["real_plus_synthetic", "accuracy"],
                "overall_MDD": sim_overall["MDD"],
                "overall_ED": sim_overall["ED"],
                "overall_DTW": sim_overall["DTW"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "model_comparison_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )
    colors = {"GPT-2": "#4C78A8", "Gemma 2 2B LoRA": "#F58518"}
    models = list(gen.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("PAMAP2 SDForger-style Five-class Synthetic Generation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    prompts = [gen[m]["num_prompts"] for m in models]
    generated = [gen[m]["num_generated_embeddings"] for m in models]
    x = np.arange(len(models))
    width = 0.36
    ax.bar(x - width / 2, prompts, width, label="Prompts", color="#B8B8B8")
    bars = ax.bar(x + width / 2, generated, width, label="Valid synthetic", color=[colors[m] for m in models])
    ax.set_xticks(x, models)
    ax.set_ylabel("Count")
    ax.set_title("A. Generation yield")
    ax.legend(frameon=False)
    for bar, model in zip(bars, models):
        rate = gen[model]["num_generated_embeddings"] / gen[model]["num_prompts"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, f"{rate:.1%}", ha="center")

    ax = axes[0, 1]
    settings = ["real_only", "synthetic_only", "real_plus_synthetic"]
    labels = ["Real", "Synthetic", "Real + synthetic"]
    x = np.arange(len(settings))
    for idx, model in enumerate(models):
        frame = utility[model].set_index("setting")
        vals = [frame.loc[s, "accuracy"] for s in settings]
        ax.bar(x + (idx - 0.5) * width, vals, width, label=model, color=colors[model])
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("B. HAR utility")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    activities = ["cycling", "running", "sitting", "standing", "walking"]
    x = np.arange(len(activities))
    for idx, model in enumerate(models):
        frame = similarity[model].set_index("activity_name")
        vals = [frame.loc[a, "DTW"] for a in activities]
        ax.bar(x + (idx - 0.5) * width, vals, width, label=model, color=colors[model])
    ax.set_xticks(x, activities, rotation=20)
    ax.set_ylabel("DTW, lower is better")
    ax.set_title("C. Temporal similarity by activity")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    gains = []
    for model in models:
        frame = utility[model].set_index("setting")
        gains.append(frame.loc["real_plus_synthetic", "accuracy"] - frame.loc["real_only", "accuracy"])
    bars = ax.bar(models, gains, color=[colors[m] for m in models])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Accuracy gain vs real-only")
    ax.set_title("D. Synthetic augmentation effect")
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, gain + (0.002 if gain >= 0 else -0.006), f"{gain:+.4f}", ha="center")

    fig.savefig(out_dir / "sdforger_style_5class_model_comparison.png", bbox_inches="tight")
    fig.savefig(out_dir / "sdforger_style_5class_model_comparison.pdf", bbox_inches="tight")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
