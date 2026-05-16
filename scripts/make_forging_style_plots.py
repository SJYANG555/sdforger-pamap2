import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.utils import ensure_dir, get_numeric_embedding_columns


ACTIVITY_ORDER = ["walking", "running", "cycling", "sitting", "standing"]
ACTIVITY_COLORS = {
    "walking": "#1f77b4",
    "running": "#d62728",
    "cycling": "#2ca02c",
    "sitting": "#9467bd",
    "standing": "#ff7f0e",
}
SOURCE_COLORS = {"real": "#2f5597", "synthetic": "#d98324"}


def _read_training_history(checkpoint_dir: Path) -> pd.DataFrame:
    candidates = [
        checkpoint_dir / "training_log_history.csv",
        checkpoint_dir / "checkpoint-2760" / "trainer_state.json",
        checkpoint_dir / "latest" / "trainer_state.json",
    ]
    candidates.extend(sorted(checkpoint_dir.glob("checkpoint-*/trainer_state.json")))

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            return pd.read_csv(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = pd.DataFrame(payload.get("log_history", []))
        if not history.empty:
            return history
    return pd.DataFrame()


def _activity_list(*frames: pd.DataFrame) -> List[str]:
    seen = set()
    for frame in frames:
        if "activity_name" in frame.columns:
            seen.update(frame["activity_name"].dropna().astype(str).tolist())
    ordered = [activity for activity in ACTIVITY_ORDER if activity in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def plot_training_curve(checkpoint_dir: Path, output_path: Path) -> Optional[Path]:
    frame = _read_training_history(checkpoint_dir)
    if frame.empty:
        return None
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if "loss" in frame.columns:
        train = frame[frame["loss"].notna()]
        if not train.empty:
            ax.plot(train["step"], train["loss"], color="#2f5597", linewidth=1.8, label="train loss")

    if "eval_loss" in frame.columns:
        val = frame[frame["eval_loss"].notna()]
        if not val.empty:
            ax.plot(val["step"], val["eval_loss"], color="#d98324", linewidth=2.0, marker="o", markersize=3, label="validation loss")

    ax.set_title("GPT-2 fine-tuning curve")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Causal LM loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _combined_embedding_pca(real_embeddings: pd.DataFrame, synthetic_embeddings: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = get_numeric_embedding_columns(real_embeddings)
    if not numeric_columns:
        return pd.DataFrame()
    real = real_embeddings.copy()
    synthetic = synthetic_embeddings.copy()
    real["source"] = "real"
    synthetic["source"] = "synthetic"
    combined = pd.concat([real, synthetic], ignore_index=True)
    reduced = PCA(n_components=2, random_state=42).fit_transform(combined[numeric_columns].to_numpy())
    combined["pc1"] = reduced[:, 0]
    combined["pc2"] = reduced[:, 1]
    return combined


def plot_embedding_by_activity(real_embeddings: pd.DataFrame, synthetic_embeddings: pd.DataFrame, output_path: Path) -> Optional[Path]:
    combined = _combined_embedding_pca(real_embeddings, synthetic_embeddings)
    if combined.empty:
        return None
    activities = _activity_list(combined)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, source in zip(axes, ["real", "synthetic"]):
        source_frame = combined[combined["source"] == source]
        for activity in activities:
            subset = source_frame[source_frame["activity_name"] == activity]
            if subset.empty:
                continue
            ax.scatter(
                subset["pc1"],
                subset["pc2"],
                s=12,
                alpha=0.62,
                color=ACTIVITY_COLORS.get(activity, "#666666"),
                label=activity,
                edgecolors="none",
            )
        ax.set_title(source.capitalize())
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle("Embedding distribution by activity")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_embedding_source_overlay(real_embeddings: pd.DataFrame, synthetic_embeddings: pd.DataFrame, output_path: Path) -> Optional[Path]:
    combined = _combined_embedding_pca(real_embeddings, synthetic_embeddings)
    if combined.empty:
        return None
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    for source in ["real", "synthetic"]:
        subset = combined[combined["source"] == source]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            s=11,
            alpha=0.48,
            color=SOURCE_COLORS[source],
            label=source,
            edgecolors="none",
        )
    ax.set_title("Real vs synthetic embedding overlap")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_window_grid(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    output_path: Path,
    channel_index: int = 0,
) -> Optional[Path]:
    activities = _activity_list(real_metadata, synthetic_metadata)
    if not activities:
        return None
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(len(activities), 2, figsize=(10, 2.2 * len(activities)), sharex=True)
    if len(activities) == 1:
        axes = np.array([axes])
    for row_idx, activity in enumerate(activities):
        real_idx = real_metadata.index[real_metadata["activity_name"] == activity].to_numpy()
        syn_idx = synthetic_metadata.index[synthetic_metadata["activity_name"] == activity].to_numpy()
        if len(real_idx) == 0 or len(syn_idx) == 0:
            continue
        real_signal = real_windows[int(real_idx[0]), :, channel_index]
        syn_signal = synthetic_windows[int(syn_idx[0]), :, channel_index]
        axes[row_idx, 0].plot(real_signal, color=SOURCE_COLORS["real"], linewidth=1.1)
        axes[row_idx, 1].plot(syn_signal, color=SOURCE_COLORS["synthetic"], linewidth=1.1)
        axes[row_idx, 0].set_ylabel(activity)
        axes[row_idx, 0].grid(alpha=0.18)
        axes[row_idx, 1].grid(alpha=0.18)
    axes[0, 0].set_title("Real")
    axes[0, 1].set_title("Synthetic")
    axes[-1, 0].set_xlabel("Time index")
    axes[-1, 1].set_xlabel("Time index")
    fig.suptitle(f"Window examples, channel {channel_index}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_similarity_metrics(metrics: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if metrics.empty:
        return None
    metrics = metrics[metrics["activity_name"] != "overall_mean"].copy()
    metric_columns = ["MDD", "ACD", "SD", "KD", "ED", "DTW"]
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.ravel(), metric_columns):
        ax.bar(
            metrics["activity_name"],
            metrics[metric],
            color=[ACTIVITY_COLORS.get(activity, "#666666") for activity in metrics["activity_name"]],
        )
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Real/synthetic similarity metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_utility_metrics(metrics: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if metrics.empty:
        return None
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics))
    width = 0.24
    for offset, metric, color in [(-width, "accuracy", "#2f5597"), (0, "macro_f1", "#d98324"), (width, "weighted_f1", "#4f7f52")]:
        ax.bar(x + offset, metrics[metric], width=width, label=metric, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["setting"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("HAR utility evaluation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(utility_dir: Path, output_path: Path) -> Optional[Path]:
    paths = [
        ("Real only", utility_dir / "confusion_matrix_real_only.csv"),
        ("Synthetic only", utility_dir / "confusion_matrix_synthetic_only.csv"),
        ("Real + synthetic", utility_dir / "confusion_matrix_real_plus_synthetic.csv"),
    ]
    matrices = [(title, pd.read_csv(path, index_col=0)) for title, path in paths if path.exists()]
    if not matrices:
        return None
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.3 * len(matrices), 4.8), constrained_layout=True)
    if len(matrices) == 1:
        axes = [axes]
    vmax = max(float(matrix.to_numpy().max()) for _, matrix in matrices)
    for ax, (title, matrix) in zip(axes, matrices):
        ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix.iloc[i, j])
                color = "white" if value > vmax * 0.55 else "#222222"
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)
    fig.suptitle("HAR confusion matrices")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style visual summaries for PAMAP2 SDForger GPT-2 outputs.")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/gpt2")
    parser.add_argument("--real-windows", default="artifacts/pamap2_sdforger_dataset/test_windows.npy")
    parser.add_argument("--real-metadata", default="artifacts/pamap2_sdforger_dataset/test_metadata.csv")
    parser.add_argument("--synthetic-windows", default="outputs/generated/gpt2_5class/generated_windows.npy")
    parser.add_argument("--synthetic-metadata", default="outputs/generated/gpt2_5class/generated_embeddings.csv")
    parser.add_argument("--real-embeddings", default="artifacts/pamap2_sdforger_dataset/test_embeddings.csv")
    parser.add_argument("--synthetic-embeddings", default="outputs/generated/gpt2_5class/generated_embeddings.csv")
    parser.add_argument("--similarity-metrics", default="outputs/evaluation/similarity/gpt2_5class/similarity_metrics.csv")
    parser.add_argument("--utility-metrics", default="outputs/evaluation/utility/gpt2_5class/utility_metrics.csv")
    parser.add_argument("--utility-dir", default="outputs/evaluation/utility/gpt2_5class")
    parser.add_argument("--output-dir", default="outputs/plots/gpt2_5class/forging_style")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    generated: Dict[str, str] = {}

    training = plot_training_curve(Path(args.checkpoint_dir), output_dir / "01_training_curve.png")
    if training:
        generated["training_curve"] = str(training)

    real_embeddings = pd.read_csv(args.real_embeddings)
    synthetic_embeddings = pd.read_csv(args.synthetic_embeddings)
    for key, path in [
        ("embedding_by_activity", plot_embedding_by_activity(real_embeddings, synthetic_embeddings, output_dir / "02_embedding_pca_by_activity.png")),
        ("embedding_source_overlay", plot_embedding_source_overlay(real_embeddings, synthetic_embeddings, output_dir / "03_embedding_pca_source_overlay.png")),
    ]:
        if path:
            generated[key] = str(path)

    real_windows = np.load(args.real_windows)
    real_metadata = pd.read_csv(args.real_metadata)
    synthetic_windows = np.load(args.synthetic_windows)
    synthetic_metadata = pd.read_csv(args.synthetic_metadata)
    window_grid = plot_window_grid(real_windows, real_metadata, synthetic_windows, synthetic_metadata, output_dir / "04_window_examples_ch0.png")
    if window_grid:
        generated["window_examples"] = str(window_grid)

    similarity = plot_similarity_metrics(pd.read_csv(args.similarity_metrics), output_dir / "05_similarity_metrics.png")
    if similarity:
        generated["similarity_metrics"] = str(similarity)

    utility = plot_utility_metrics(pd.read_csv(args.utility_metrics), output_dir / "06_utility_metrics.png")
    if utility:
        generated["utility_metrics"] = str(utility)

    confusion = plot_confusion_matrices(Path(args.utility_dir), output_dir / "07_confusion_matrices.png")
    if confusion:
        generated["confusion_matrices"] = str(confusion)

    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(f"Wrote {len(generated)} plots to {output_dir}")
    for name, path in generated.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
