import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.utils import ensure_dir, flatten_window_features, write_json
from pamap2_forger.visualization import plot_confusion_matrix


def evaluate_split(
    clf,
    windows_path: Path,
    metadata_path: Path,
    label: str,
    output_dir: Path,
    labels: list[int],
    label_names: list[str],
) -> dict:
    windows = np.load(windows_path)
    metadata = pd.read_csv(metadata_path)
    x = flatten_window_features(windows).to_numpy(dtype=np.float32)
    y_true = metadata["activity_id"].to_numpy()
    y_pred = clf.predict(x)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"confusion_matrix_{label}.csv"
    )
    plot_confusion_matrix(
        matrix=matrix,
        labels=label_names,
        output_path=str(output_dir / f"confusion_matrix_{label}.png"),
        title=f"k-NN condition consistency - {label}",
    )

    rows = []
    for activity_id, activity_name in zip(labels, label_names):
        mask = y_true == activity_id
        rows.append(
            {
                "source": label,
                "activity_id": int(activity_id),
                "activity_name": activity_name,
                "num_samples": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else None,
            }
        )
    per_activity = pd.DataFrame(rows)
    per_activity.to_csv(output_dir / f"per_activity_accuracy_{label}.csv", index=False)

    return {
        "source": label,
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SDForger-style text-condition consistency with a k-NN classifier trained on real data."
    )
    parser.add_argument("--real-train-windows", default="artifacts/pamap2_sdforger_dataset/train_windows.npy")
    parser.add_argument("--real-train-metadata", default="artifacts/pamap2_sdforger_dataset/train_metadata.csv")
    parser.add_argument("--real-test-windows", default="artifacts/pamap2_sdforger_dataset/test_windows.npy")
    parser.add_argument("--real-test-metadata", default="artifacts/pamap2_sdforger_dataset/test_metadata.csv")
    parser.add_argument("--gpt2-windows", default="outputs/generated/gpt2_5class/generated_windows.npy")
    parser.add_argument("--gpt2-metadata", default="outputs/generated/gpt2_5class/generated_embeddings.csv")
    parser.add_argument("--gemma-windows", default="outputs/generated/gemma_5class/generated_windows.npy")
    parser.add_argument("--gemma-metadata", default="outputs/generated/gemma_5class/generated_embeddings.csv")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/evaluation/knn_condition_5class")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    train_windows = np.load(args.real_train_windows)
    train_metadata = pd.read_csv(args.real_train_metadata)
    x_train = flatten_window_features(train_windows).to_numpy(dtype=np.float32)
    y_train = train_metadata["activity_id"].to_numpy()

    label_frame = (
        train_metadata[["activity_id", "activity_name"]]
        .drop_duplicates()
        .sort_values("activity_id")
    )
    labels = label_frame["activity_id"].astype(int).tolist()
    label_names = label_frame["activity_name"].tolist()

    clf = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=args.neighbors, weights="distance", metric="minkowski", p=2),
    )
    clf.fit(x_train, y_train)

    results = [
        evaluate_split(
            clf,
            Path(args.real_test_windows),
            Path(args.real_test_metadata),
            "real_test",
            output_dir,
            labels,
            label_names,
        ),
        evaluate_split(
            clf,
            Path(args.gpt2_windows),
            Path(args.gpt2_metadata),
            "gpt2_5class",
            output_dir,
            labels,
            label_names,
        ),
        evaluate_split(
            clf,
            Path(args.gemma_windows),
            Path(args.gemma_metadata),
            "gemma_5class",
            output_dir,
            labels,
            label_names,
        ),
    ]
    metrics = pd.DataFrame(results)
    metrics.to_csv(output_dir / "knn_condition_metrics.csv", index=False)
    write_json(
        output_dir / "knn_condition_summary.json",
        {
            "classifier": "StandardScaler + KNeighborsClassifier",
            "n_neighbors": args.neighbors,
            "train_source": "real_train",
            "train_samples": int(len(y_train)),
            "sources": results,
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "rows": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
