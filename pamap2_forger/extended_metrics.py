from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import kurtosis, skew, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pamap2_forger.metrics import autocorrelation_vector, dtw_distance
from pamap2_forger.utils import flatten_window_features


@dataclass
class ExtendedMetricConfig:
    max_samples_per_activity: int = 128
    max_pairwise_pairs: int = 64
    max_lag: int = 32
    dtw_window: int | None = None
    duplicate_decimals: int = 4
    random_seed: int = 42
    n_estimators: int = 300
    max_depth: int | None = None


def _empty_fidelity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "comparison",
            "activity_name",
            "num_reference",
            "num_candidate",
            "mean_difference",
            "std_difference",
            "wasserstein_distance",
            "skewness_difference",
            "kurtosis_difference",
            "autocorrelation_difference",
            "spectral_difference",
            "euclidean_distance",
            "dtw_distance",
        ]
    )


def _empty_diversity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "activity_name",
            "num_samples",
            "unique_windows",
            "duplicate_count",
            "duplicate_rate",
            "mean_pairwise_euclidean",
            "mean_pairwise_dtw",
            "wpd",
        ]
    )


def _sample_indices(n: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_items:
        return np.arange(n)
    return rng.choice(n, size=max_items, replace=False)


def _paired_indices(n_left: int, n_right: int, max_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pair_count = min(n_left, n_right, max_pairs)
    if pair_count <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return (
        rng.choice(n_left, size=pair_count, replace=False),
        rng.choice(n_right, size=pair_count, replace=False),
    )


def _within_pairs(n: int, max_pairs: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if n < 2 or max_pairs <= 0:
        return []
    all_count = n * (n - 1) // 2
    if all_count <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    seen = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < max_pairs:
        i, j = rng.choice(n, size=2, replace=False)
        if i > j:
            i, j = j, i
        key = (int(i), int(j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _prototype(windows: np.ndarray) -> np.ndarray:
    return windows.mean(axis=0)


def _spectral_magnitude(x: np.ndarray) -> np.ndarray:
    magnitude = np.abs(rfft(x))
    norm = np.linalg.norm(magnitude)
    return magnitude if norm == 0 else magnitude / norm


def _dtw_path(x: np.ndarray, y: np.ndarray, window: int | None = None) -> List[Tuple[int, int]]:
    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[0, 0] = 0.0
    for i in range(1, n + 1):
        start = max(1, i - window)
        end = min(m, i + window)
        for j in range(start, end + 1):
            prev = (costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])
            step = int(np.argmin(prev))
            costs[i, j] = abs(x[i - 1] - y[j - 1]) + prev[step]
            steps[i, j] = step
    if not np.isfinite(costs[n, m]):
        return []
    i, j = n, m
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = steps[i, j]
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path


def warping_path_diversity(x: np.ndarray, y: np.ndarray, window: int | None = None) -> float:
    """Average normalized distance of the DTW path from the diagonal.

    This follows the WPD intuition from the human-motion evaluation paper:
    diversity in timing is reflected by how far the alignment path bends away
    from a straight diagonal. Multivariate windows are first reduced to their
    channel-averaged trajectory so the metric is comparable across subsets.
    """

    x_1d = np.asarray(x, dtype=np.float64).mean(axis=1)
    y_1d = np.asarray(y, dtype=np.float64).mean(axis=1)
    path = _dtw_path(x_1d, y_1d, window=window)
    if not path:
        return float("nan")
    n, m = len(x_1d), len(y_1d)
    denom = np.sqrt(n * n + m * m)
    distances = [abs(m * i - n * j) / denom for i, j in path]
    return float(np.mean(distances))


def _mean_channel_stat_difference(reference: np.ndarray, candidate: np.ndarray, fn) -> float:
    values: List[float] = []
    for channel_idx in range(reference.shape[2]):
        ref_series = reference[:, :, channel_idx].reshape(-1)
        cand_series = candidate[:, :, channel_idx].reshape(-1)
        values.append(abs(float(fn(ref_series)) - float(fn(cand_series))))
    return float(np.nanmean(values))


def _mean_channel_wasserstein(reference: np.ndarray, candidate: np.ndarray) -> float:
    values: List[float] = []
    for channel_idx in range(reference.shape[2]):
        ref_series = reference[:, :, channel_idx].reshape(-1)
        cand_series = candidate[:, :, channel_idx].reshape(-1)
        values.append(float(wasserstein_distance(ref_series, cand_series)))
    return float(np.mean(values))


def _prototype_temporal_metrics(reference: np.ndarray, candidate: np.ndarray, max_lag: int, dtw_window: int | None) -> Dict[str, float]:
    ref_proto = _prototype(reference)
    cand_proto = _prototype(candidate)
    acd_values = []
    spectral_values = []
    dtw_values = []
    for channel_idx in range(ref_proto.shape[1]):
        ref_series = ref_proto[:, channel_idx]
        cand_series = cand_proto[:, channel_idx]
        acd_values.append(
            np.mean(
                np.abs(
                    autocorrelation_vector(ref_series, max_lag)
                    - autocorrelation_vector(cand_series, max_lag)
                )
            )
        )
        spectral_values.append(
            np.mean(np.abs(_spectral_magnitude(ref_series) - _spectral_magnitude(cand_series)))
        )
        dtw_values.append(dtw_distance(ref_series, cand_series, window=dtw_window))
    return {
        "autocorrelation_difference": float(np.mean(acd_values)),
        "spectral_difference": float(np.mean(spectral_values)),
        "euclidean_distance": float(np.linalg.norm(ref_proto.reshape(-1) - cand_proto.reshape(-1))),
        "dtw_distance": float(np.mean(dtw_values)),
    }


def _activity_subsets(windows: np.ndarray, metadata: pd.DataFrame, activity: str) -> np.ndarray:
    return windows[(metadata["activity_name"] == activity).to_numpy()]


def compute_extended_fidelity(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    config: ExtendedMetricConfig,
) -> pd.DataFrame:
    if len(synthetic_windows) == 0 or synthetic_metadata.empty:
        return _empty_fidelity_frame()

    rng = np.random.default_rng(config.random_seed)
    rows: List[Dict[str, object]] = []
    activities = sorted(set(real_metadata["activity_name"]).intersection(set(synthetic_metadata["activity_name"])))

    for activity in activities:
        real_subset = _activity_subsets(real_windows, real_metadata, activity)
        syn_subset = _activity_subsets(synthetic_windows, synthetic_metadata, activity)
        if len(real_subset) == 0 or len(syn_subset) == 0:
            continue

        real_idx_a, real_idx_b = _paired_indices(
            len(real_subset), len(real_subset), config.max_samples_per_activity, rng
        )
        syn_ref_idx, syn_idx = _paired_indices(
            len(real_subset), len(syn_subset), config.max_samples_per_activity, rng
        )
        comparisons = [
            ("real_real", real_subset[real_idx_a], real_subset[real_idx_b]),
            ("real_synthetic", real_subset[syn_ref_idx], syn_subset[syn_idx]),
        ]
        for comparison, reference, candidate in comparisons:
            if len(reference) == 0 or len(candidate) == 0:
                continue
            temporal = _prototype_temporal_metrics(
                reference, candidate, max_lag=config.max_lag, dtw_window=config.dtw_window
            )
            rows.append(
                {
                    "comparison": comparison,
                    "activity_name": activity,
                    "num_reference": int(len(reference)),
                    "num_candidate": int(len(candidate)),
                    "mean_difference": _mean_channel_stat_difference(reference, candidate, np.mean),
                    "std_difference": _mean_channel_stat_difference(reference, candidate, np.std),
                    "wasserstein_distance": _mean_channel_wasserstein(reference, candidate),
                    "skewness_difference": _mean_channel_stat_difference(reference, candidate, skew),
                    "kurtosis_difference": _mean_channel_stat_difference(
                        reference, candidate, lambda x: kurtosis(x, fisher=False)
                    ),
                    **temporal,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_fidelity_frame()
    overall_rows = []
    for comparison, group in result.groupby("comparison", sort=True):
        numeric = group.select_dtypes(include=[np.number])
        row = {
            "comparison": comparison,
            "activity_name": "overall_mean",
            "num_reference": int(group["num_reference"].sum()),
            "num_candidate": int(group["num_candidate"].sum()),
        }
        for column in numeric.columns:
            if column.startswith("num_"):
                continue
            row[column] = float(group[column].mean())
        overall_rows.append(row)
    return pd.concat([result, pd.DataFrame(overall_rows)], ignore_index=True)


def compute_diversity_metrics(
    real_windows: np.ndarray,
    real_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    config: ExtendedMetricConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows: List[Dict[str, object]] = []
    sources = [("real", real_windows, real_metadata)]
    if len(synthetic_windows) > 0 and not synthetic_metadata.empty:
        sources.append(("synthetic", synthetic_windows, synthetic_metadata))

    for source_name, windows, metadata in sources:
        for activity in sorted(metadata["activity_name"].dropna().unique().tolist()):
            subset = _activity_subsets(windows, metadata, activity)
            if len(subset) == 0:
                continue
            sample = subset[_sample_indices(len(subset), config.max_samples_per_activity, rng)]
            rounded = np.round(sample.reshape(len(sample), -1), decimals=config.duplicate_decimals)
            unique_count = int(len(np.unique(rounded, axis=0))) if len(sample) else 0
            pairs = _within_pairs(len(sample), config.max_pairwise_pairs, rng)
            ed_values = []
            dtw_values = []
            wpd_values = []
            for i, j in pairs:
                left = sample[i]
                right = sample[j]
                ed_values.append(float(np.linalg.norm(left.reshape(-1) - right.reshape(-1))))
                dtw_values.append(
                    float(
                        np.mean(
                            [
                                dtw_distance(left[:, channel_idx], right[:, channel_idx], window=config.dtw_window)
                                for channel_idx in range(left.shape[1])
                            ]
                        )
                    )
                )
                wpd_values.append(warping_path_diversity(left, right, window=config.dtw_window))
            rows.append(
                {
                    "source": source_name,
                    "activity_name": activity,
                    "num_samples": int(len(sample)),
                    "unique_windows": unique_count,
                    "duplicate_count": int(len(sample) - unique_count),
                    "duplicate_rate": float(1.0 - unique_count / len(sample)) if len(sample) else float("nan"),
                    "mean_pairwise_euclidean": float(np.nanmean(ed_values)) if ed_values else float("nan"),
                    "mean_pairwise_dtw": float(np.nanmean(dtw_values)) if dtw_values else float("nan"),
                    "wpd": float(np.nanmean(wpd_values)) if wpd_values else float("nan"),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_diversity_frame()
    overall_rows = []
    for source, group in result.groupby("source", sort=True):
        overall_rows.append(
            {
                "source": source,
                "activity_name": "overall_mean",
                "num_samples": int(group["num_samples"].sum()),
                "unique_windows": int(group["unique_windows"].sum()),
                "duplicate_count": int(group["duplicate_count"].sum()),
                "duplicate_rate": float(group["duplicate_count"].sum() / group["num_samples"].sum()),
                "mean_pairwise_euclidean": float(group["mean_pairwise_euclidean"].mean()),
                "mean_pairwise_dtw": float(group["mean_pairwise_dtw"].mean()),
                "wpd": float(group["wpd"].mean()),
            }
        )
    return pd.concat([result, pd.DataFrame(overall_rows)], ignore_index=True)


def compute_condition_consistency(
    real_train_windows: np.ndarray,
    real_train_metadata: pd.DataFrame,
    real_test_windows: np.ndarray,
    real_test_metadata: pd.DataFrame,
    synthetic_windows: np.ndarray,
    synthetic_metadata: pd.DataFrame,
    config: ExtendedMetricConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    x_train = flatten_window_features(real_train_windows).to_numpy(dtype=np.float32)
    y_train = real_train_metadata["activity_id"].to_numpy()
    label_frame = (
        real_train_metadata[["activity_id", "activity_name"]]
        .drop_duplicates()
        .sort_values("activity_id")
    )
    labels = label_frame["activity_id"].astype(int).tolist()
    label_names = label_frame["activity_name"].tolist()
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)

    sources = [("real_test", real_test_windows, real_test_metadata)]
    if len(synthetic_windows) > 0 and not synthetic_metadata.empty:
        sources.append(("synthetic", synthetic_windows, synthetic_metadata))

    rows: List[Dict[str, object]] = []
    per_activity_rows: List[Dict[str, object]] = []
    matrices: Dict[str, np.ndarray] = {}
    for source, windows, metadata in sources:
        x = flatten_window_features(windows).to_numpy(dtype=np.float32)
        y_true = metadata["activity_id"].to_numpy()
        y_pred = clf.predict(x)
        matrices[source] = confusion_matrix(y_true, y_pred, labels=labels)
        rows.append(
            {
                "source": source,
                "num_samples": int(len(y_true)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            }
        )
        for activity_id, activity_name in zip(labels, label_names):
            mask = y_true == activity_id
            per_activity_rows.append(
                {
                    "source": source,
                    "activity_id": int(activity_id),
                    "activity_name": activity_name,
                    "num_samples": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows), matrices, pd.DataFrame(per_activity_rows)
