from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pamap2_forger.utils import flatten_window_features, pairwise_sample_indices


def dtw_distance(x: np.ndarray, y: np.ndarray, window: int = None) -> float:
    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        start = max(1, i - window)
        end = min(m, i + window)
        for j in range(start, end + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def autocorrelation_vector(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag, dtype=np.float64)
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            result.append(0.0)
        else:
            result.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(result, dtype=np.float64)


def spectral_magnitude(x: np.ndarray) -> np.ndarray:
    mag = np.abs(rfft(x))
    norm = np.linalg.norm(mag)
    return mag if norm == 0 else mag / norm


def _mean_channel_metric(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    fn,
) -> float:
    values: List[float] = []
    for channel_idx in range(real_windows.shape[2]):
        real_series = real_windows[:, :, channel_idx].reshape(-1)
        synthetic_series = synthetic_windows[:, :, channel_idx].reshape(-1)
        values.append(float(fn(real_series, synthetic_series)))
    return float(np.mean(values))


def _prototype_windows(real_windows: np.ndarray, synthetic_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return real_windows.mean(axis=0), synthetic_windows.mean(axis=0)


def compute_similarity_metrics(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    max_lag: int = 32,
    max_samples_per_activity: int = 128,
    dtw_window: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    activities = sorted(set(real_metadata["activity_name"]).intersection(set(synthetic_metadata["activity_name"])))

    for activity in activities:
        real_mask = real_metadata["activity_name"] == activity
        syn_mask = synthetic_metadata["activity_name"] == activity
        real_subset = real_windows[real_mask.to_numpy()]
        syn_subset = synthetic_windows[syn_mask.to_numpy()]
        if len(real_subset) == 0 or len(syn_subset) == 0:
            continue

        real_idx, syn_idx = pairwise_sample_indices(
            n_real=len(real_subset),
            n_syn=len(syn_subset),
            max_pairs=max_samples_per_activity,
            seed=seed,
        )
        real_sample = real_subset[real_idx]
        syn_sample = syn_subset[syn_idx]
        real_proto, syn_proto = _prototype_windows(real_sample, syn_sample)

        mdd = float(abs(real_sample.mean() - syn_sample.mean()))
        acd = float(
            np.mean(
                [
                    np.mean(
                        np.abs(
                            autocorrelation_vector(real_proto[:, channel_idx], max_lag)
                            - autocorrelation_vector(syn_proto[:, channel_idx], max_lag)
                        )
                    )
                    for channel_idx in range(real_proto.shape[1])
                ]
            )
        )
        sd = float(
            np.mean(
                [
                    np.mean(
                        np.abs(
                            spectral_magnitude(real_proto[:, channel_idx])
                            - spectral_magnitude(syn_proto[:, channel_idx])
                        )
                    )
                    for channel_idx in range(real_proto.shape[1])
                ]
            )
        )
        kd = _mean_channel_metric(
            real_sample,
            syn_sample,
            lambda r, s: abs(kurtosis(r, fisher=False) - kurtosis(s, fisher=False)),
        )
        ed = float(np.linalg.norm(real_proto.reshape(-1) - syn_proto.reshape(-1)))
        dtw = float(
            np.mean(
                [
                    dtw_distance(real_proto[:, channel_idx], syn_proto[:, channel_idx], window=dtw_window)
                    for channel_idx in range(real_proto.shape[1])
                ]
            )
        )

        rows.append(
            {
                "activity_name": activity,
                "num_real": int(len(real_sample)),
                "num_synthetic": int(len(syn_sample)),
                "MDD": mdd,
                "ACD": acd,
                "SD": sd,
                "KD": kd,
                "ED": ed,
                "DTW": dtw,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        summary = {
            "activity_name": "overall_mean",
            "num_real": int(result["num_real"].sum()),
            "num_synthetic": int(result["num_synthetic"].sum()),
            "MDD": float(result["MDD"].mean()),
            "ACD": float(result["ACD"].mean()),
            "SD": float(result["SD"].mean()),
            "KD": float(result["KD"].mean()),
            "ED": float(result["ED"].mean()),
            "DTW": float(result["DTW"].mean()),
        }
        result = pd.concat([result, pd.DataFrame([summary])], ignore_index=True)
    return result


def run_activity_classification_utility(
    real_train_windows: np.ndarray,
    real_train_labels: np.ndarray,
    real_test_windows: np.ndarray,
    real_test_labels: np.ndarray,
    synthetic_windows: np.ndarray,
    synthetic_labels: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    x_train_real = flatten_window_features(real_train_windows).to_numpy(dtype=np.float32)
    x_test = flatten_window_features(real_test_windows).to_numpy(dtype=np.float32)
    x_syn = flatten_window_features(synthetic_windows).to_numpy(dtype=np.float32) if len(synthetic_windows) > 0 else np.empty((0, x_train_real.shape[1]), dtype=np.float32)

    settings = {"real_only": (x_train_real, real_train_labels)}
    notes: List[str] = []
    if len(synthetic_windows) == 0 or len(synthetic_labels) == 0:
        notes.append("Synthetic dataset is empty. Skipped synthetic_only and real_plus_synthetic settings.")
    else:
        settings["synthetic_only"] = (x_syn, synthetic_labels)
        settings["real_plus_synthetic"] = (
            np.vstack([x_train_real, x_syn]),
            np.concatenate([real_train_labels, synthetic_labels]),
        )

    rows: List[Dict[str, object]] = []
    confusion_matrices: Dict[str, np.ndarray] = {}
    labels = sorted(np.unique(real_test_labels).tolist())

    for setting_name, (x_train, y_train) in settings.items():
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=1,
        )
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        confusion_matrices[setting_name] = confusion_matrix(real_test_labels, preds, labels=labels)
        rows.append(
            {
                "setting": setting_name,
                "accuracy": float(accuracy_score(real_test_labels, preds)),
                "macro_f1": float(f1_score(real_test_labels, preds, average="macro")),
                "weighted_f1": float(f1_score(real_test_labels, preds, average="weighted")),
            }
        )

    return pd.DataFrame(rows), confusion_matrices, notes
