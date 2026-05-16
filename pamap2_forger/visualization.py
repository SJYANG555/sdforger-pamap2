import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pamap2_forger.utils import ensure_dir


def _load_training_history(log_csv_path: str) -> pd.DataFrame:
    log_path = Path(log_csv_path)
    if log_path.exists():
        return pd.read_csv(log_path)

    trainer_state_path = log_path.with_name("trainer_state.json")
    if trainer_state_path.exists():
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload.get("log_history", []))

    return pd.DataFrame()


def plot_training_curve(log_csv_path: str, output_path: str) -> None:
    frame = _load_training_history(log_csv_path)
    if frame.empty or "loss" not in frame.columns:
        return
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(8, 4))
    train_frame = frame[frame["loss"].notna()]
    if not train_frame.empty:
        plt.plot(train_frame["step"], train_frame["loss"], label="train_loss")
    if "eval_loss" in frame.columns:
        eval_frame = frame[frame["eval_loss"].notna()]
        if not eval_frame.empty:
            plt.plot(eval_frame["step"], eval_frame["eval_loss"], label="val_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_real_vs_synthetic_windows(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    output_dir: str,
    max_examples_per_activity: int = 3,
) -> None:
    output_dir = ensure_dir(output_dir)
    activities = sorted(set(real_metadata["activity_name"]).intersection(set(synthetic_metadata["activity_name"])))
    for activity in activities:
        real_subset = real_windows[real_metadata["activity_name"] == activity]
        synthetic_subset = synthetic_windows[synthetic_metadata["activity_name"] == activity]
        if len(real_subset) == 0 or len(synthetic_subset) == 0:
            continue

        for example_idx in range(min(max_examples_per_activity, len(real_subset), len(synthetic_subset))):
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(real_subset[example_idx][:, 0], label="real_ch0")
            axes[0].plot(real_subset[example_idx][:, 1], label="real_ch1")
            axes[0].set_title(f"{activity} real window {example_idx}")
            axes[0].legend()
            axes[1].plot(synthetic_subset[example_idx][:, 0], label="syn_ch0")
            axes[1].plot(synthetic_subset[example_idx][:, 1], label="syn_ch1")
            axes[1].set_title(f"{activity} synthetic window {example_idx}")
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{activity}_example_{example_idx}.png")
            plt.close(fig)


def plot_metric_bars(frame: pd.DataFrame, metric_columns, output_path: str, title: str) -> None:
    if frame.empty:
        return
    plot_frame = frame[frame["activity_name"] != "overall_mean"] if "activity_name" in frame.columns else frame
    ensure_dir(Path(output_path).parent)
    plot_frame.set_index(plot_frame.columns[0])[list(metric_columns)].plot(kind="bar", figsize=(12, 5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, labels: list, output_path: str, title: str) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_embedding_scatter(
    real_embeddings: pd.DataFrame,
    synthetic_embeddings: pd.DataFrame,
    output_path: str,
    max_points: int = 1000,
) -> None:
    numeric_columns = [col for col in real_embeddings.columns if col.startswith("value_")]
    if not numeric_columns:
        return
    real_sample = real_embeddings.sample(n=min(max_points, len(real_embeddings)), random_state=42)
    synthetic_sample = synthetic_embeddings.sample(n=min(max_points, len(synthetic_embeddings)), random_state=42)
    combined = pd.concat(
        [
            real_sample.assign(source="real"),
            synthetic_sample.assign(source="synthetic"),
        ],
        ignore_index=True,
    )
    reduced = PCA(n_components=2, random_state=42).fit_transform(combined[numeric_columns].to_numpy())
    plt.figure(figsize=(7, 6))
    for source, color in [("real", "steelblue"), ("synthetic", "darkorange")]:
        mask = combined["source"] == source
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=source, alpha=0.5, s=10, c=color)
    plt.legend()
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()
