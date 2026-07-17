from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def marginal_distribution_difference(real_windows: np.ndarray, synthetic_windows: np.ndarray, n_bins: int = 50) -> float:
    """TSGBench/SDForger-style histogram marginal distribution difference."""

    values: List[float] = []
    start_idx = 1 if real_windows.shape[1] > 1 else 0
    for channel_idx in range(real_windows.shape[2]):
        for time_idx in range(start_idx, real_windows.shape[1]):
            real_values = real_windows[:, time_idx, channel_idx].astype(np.float64)
            syn_values = synthetic_windows[:, time_idx, channel_idx].astype(np.float64)
            low = float(np.min(real_values))
            high = float(np.max(real_values))
            if np.isclose(low, high):
                high = low + 1e-5
            bins = np.linspace(low, high, n_bins + 1, dtype=np.float64)
            delta = bins[1] - bins[0]
            real_counts, _ = np.histogram(real_values, bins=bins)
            syn_counts, _ = np.histogram(syn_values, bins=bins)
            real_density = real_counts.astype(np.float64) / delta / float(len(real_values))
            syn_density = syn_counts.astype(np.float64) / delta / float(len(syn_values))
            values.append(float(np.mean(np.abs(syn_density - real_density))))
    return float(np.mean(values)) if values else float("nan")


def autocorrelation_difference(real_windows: np.ndarray, synthetic_windows: np.ndarray, max_lag: int) -> float:
    """TSGBench/SDForger-style autocorrelation difference."""

    def acf(data: np.ndarray) -> np.ndarray:
        centered = data.astype(np.float64) - np.mean(data, axis=(0, 1), keepdims=True)
        variance = np.var(centered, axis=(0, 1))
        variance = np.where(np.isclose(variance, 0.0), 1.0, variance)
        lag_count = min(max_lag, centered.shape[1])
        values = []
        for lag in range(lag_count):
            if lag == 0:
                product = centered * centered
            else:
                product = centered[:, lag:, :] * centered[:, :-lag, :]
            values.append(np.mean(product, axis=(0, 1)) / variance)
        return np.stack(values, axis=0)

    diff = acf(synthetic_windows) - acf(real_windows)
    return float(np.mean(np.sqrt(np.sum(np.square(diff), axis=0))))


def _skewness(data: np.ndarray) -> np.ndarray:
    centered = data.astype(np.float64) - np.mean(data, axis=(0, 1), keepdims=True)
    sample_count = max(int(np.prod(centered.shape[:2])), 1)
    ddof = 1 if sample_count > 1 else 0
    std = np.std(centered, axis=(0, 1), ddof=ddof)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return np.mean(np.power(centered, 3), axis=(0, 1)) / np.power(std, 3)


def skewness_difference(real_windows: np.ndarray, synthetic_windows: np.ndarray) -> float:
    """Skewness Difference (SD) following the TSGBench/SDForger metric."""

    return float(np.mean(np.abs(_skewness(synthetic_windows) - _skewness(real_windows))))


def _excess_kurtosis(data: np.ndarray) -> np.ndarray:
    centered = data.astype(np.float64) - np.mean(data, axis=(0, 1), keepdims=True)
    variance_squared = np.square(np.var(centered, axis=(0, 1)))
    variance_squared = np.where(np.isclose(variance_squared, 0.0), 1.0, variance_squared)
    return np.mean(np.power(centered, 4), axis=(0, 1)) / variance_squared - 3.0


def kurtosis_difference(real_windows: np.ndarray, synthetic_windows: np.ndarray) -> float:
    """Kurtosis Difference (KD) following the TSGBench/SDForger metric."""

    return float(np.mean(np.abs(_excess_kurtosis(synthetic_windows) - _excess_kurtosis(real_windows))))


def euclidean_distance_pairs(real_windows: np.ndarray, synthetic_windows: np.ndarray) -> float:
    """Mean paired ED, averaged over channels as in the SDForger evaluator."""

    values = []
    for sample_idx in range(real_windows.shape[0]):
        channel_distances = [
            np.linalg.norm(real_windows[sample_idx, :, channel_idx] - synthetic_windows[sample_idx, :, channel_idx])
            for channel_idx in range(real_windows.shape[2])
        ]
        values.append(float(np.mean(channel_distances)))
    return float(np.mean(values)) if values else float("nan")


def dtw_distance_multivariate(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
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
            cost = np.linalg.norm(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def dynamic_time_warping_pairs(real_windows: np.ndarray, synthetic_windows: np.ndarray, window: Optional[int]) -> float:
    values = [
        dtw_distance_multivariate(real_windows[sample_idx], synthetic_windows[sample_idx], window=window)
        for sample_idx in range(real_windows.shape[0])
    ]
    return float(np.mean(values)) if values else float("nan")


def _op_shift(shapelets: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
    basis_count, basis_length = shapelets.shape
    result = np.zeros((basis_count, target_dim), dtype=np.float64)
    for idx in range(basis_count):
        offset = int(offsets[idx])
        result[idx, offset : offset + basis_length] = shapelets[idx]
    return result


def _unsup_obj_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
) -> float:
    objective = 0.0
    for sample_idx in range(data.shape[0]):
        shifted_shapelets = _op_shift(shapelets, offsets[sample_idx], data.shape[1])
        reconstruction = np.dot(coefficients[sample_idx], shifted_shapelets)
        objective += (
            0.5 * np.linalg.norm(data[sample_idx] - reconstruction) ** 2
            + lambda_ * np.linalg.norm(coefficients[sample_idx], 1)
        )
    return float(objective)


def _update_a_par_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
    max_iter: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    _n_samples, series_length = data.shape
    basis_count, basis_length = shapelets.shape
    segment_indices = np.add.outer(np.arange(series_length - basis_length + 1), np.arange(basis_length))
    objective_history: List[float] = []

    for _ in range(int(max_iter)):
        for sample_idx in range(data.shape[0]):
            sample = data[sample_idx]
            shifted_shapelets = _op_shift(shapelets, offsets[sample_idx], series_length)
            for basis_idx in rng.permutation(basis_count):
                basis = shapelets[basis_idx]
                temp_coefficients = coefficients[sample_idx].copy()
                temp_coefficients[basis_idx] = 0.0
                residue = sample - np.dot(temp_coefficients, shifted_shapelets)
                basis_norm2 = np.linalg.norm(basis) ** 2
                if np.isclose(basis_norm2, 0.0):
                    coefficients[sample_idx, basis_idx] = 0.0
                    continue

                segments = residue[segment_indices]
                dot_products = np.dot(segments, basis)
                best_segment_idx = int(np.argmax(np.abs(dot_products)))
                best_dot_product = dot_products[best_segment_idx]
                if np.abs(best_dot_product) <= lambda_:
                    coefficient_star = 0.0
                else:
                    coefficient_star = (
                        np.sign(best_dot_product) * (np.abs(best_dot_product) - lambda_) / basis_norm2
                    )
                    offsets[sample_idx, basis_idx] = best_segment_idx
                coefficients[sample_idx, basis_idx] = coefficient_star
                if coefficient_star != 0.0:
                    shifted_shapelets[basis_idx, :] = 0.0
                    offset = int(offsets[sample_idx, basis_idx])
                    shifted_shapelets[basis_idx, offset : offset + basis_length] = basis

        objective = _unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective - previous) / abs(previous) < epsilon:
                return coefficients, offsets, objective
    return coefficients, offsets, float(objective_history[-1]) if objective_history else 0.0


def _update_s_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
    c: float,
    max_iter: int,
    epsilon: float,
) -> np.ndarray:
    _n_samples, series_length = data.shape
    basis_count, basis_length = shapelets.shape
    objective_history: List[float] = []

    for _ in range(int(max_iter)):
        for basis_idx in range(basis_count):
            activation_norm2 = np.linalg.norm(coefficients[:, basis_idx]) ** 2
            if np.isclose(activation_norm2, 0.0):
                continue
            new_basis = np.zeros(basis_length, dtype=np.float64)
            for sample_idx in range(data.shape[0]):
                temp_coefficients = coefficients[sample_idx].copy()
                temp_coefficients[basis_idx] = 0.0
                shifted_shapelets = _op_shift(shapelets, offsets[sample_idx], series_length)
                residue = data[sample_idx] - np.dot(temp_coefficients, shifted_shapelets)
                offset = int(offsets[sample_idx, basis_idx])
                new_basis += coefficients[sample_idx, basis_idx] * residue[offset : offset + basis_length]

            new_norm = np.linalg.norm(new_basis)
            if np.isclose(new_norm, 0.0):
                continue
            if activation_norm2 <= new_norm / np.sqrt(c):
                new_basis = np.sqrt(c) / new_norm * new_basis
            else:
                new_basis = new_basis / activation_norm2
            shapelets[basis_idx, :] = new_basis

        objective = _unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective - previous) / abs(previous) < epsilon:
                return shapelets
    return shapelets


def _usidl(
    data: np.ndarray,
    lambda_: float,
    basis_count: int,
    basis_length: int,
    c: float,
    epsilon: float,
    max_iter: int,
    max_inner_iter: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, series_length = data.shape
    shapelets = rng.standard_normal((basis_count, basis_length))
    coefficients = rng.standard_normal((n_samples, basis_count))
    offsets = rng.integers(0, series_length - basis_length + 1, (n_samples, basis_count))
    objective_history: List[float] = []

    for _ in range(int(max_iter)):
        coefficients, offsets, _ = _update_a_par_sid(
            data, shapelets, coefficients, offsets, lambda_, max_inner_iter, epsilon, rng
        )
        shapelets = _update_s_sid(data, shapelets, coefficients, offsets, lambda_, c, max_inner_iter, epsilon)
        objective = _unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective - previous) / abs(previous) < epsilon:
                break
    return shapelets, coefficients, offsets


def _calculate_shr_sid_univariate(
    real_series: np.ndarray,
    synthetic_series: np.ndarray,
    seed: int,
    basis_count: int,
    lambda_: float,
    basis_ratio: float,
    c: float,
    epsilon: float,
    max_iter: int,
    max_inner_iter: int,
) -> float:
    rng = np.random.default_rng(seed)
    train_data = np.asarray(real_series, dtype=np.float64)
    test_data = np.asarray(synthetic_series, dtype=np.float64)
    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError("SHR expects 2D arrays shaped as samples x time.")
    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("Real and synthetic series must share the same length for SHR.")
    if train_data.shape[0] == 0 or test_data.shape[0] == 0:
        return float("nan")
    series_length = train_data.shape[1]
    basis_length = int(np.ceil(series_length * basis_ratio))
    basis_length = max(1, min(basis_length, series_length))
    shapelets, _coefficients, _offsets = _usidl(
        train_data, lambda_, basis_count, basis_length, c, epsilon, max_iter, max_inner_iter, rng
    )
    test_coefficients = rng.standard_normal((test_data.shape[0], basis_count))
    test_offsets = rng.integers(0, series_length - basis_length + 1, (test_data.shape[0], basis_count))
    test_coefficients, test_offsets, _ = _update_a_par_sid(
        test_data, shapelets, test_coefficients, test_offsets, lambda_, max_iter, epsilon, rng
    )
    reconstruction_error = _unsup_obj_sid(test_data, shapelets, test_coefficients, test_offsets, 0.0)
    return float(reconstruction_error / test_data.shape[0])


def shapelet_reconstruction_error(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    seed: int = 42,
    basis_count: int = 20,
    lambda_: float = 0.1,
    basis_ratio: float = 0.25,
    c: float = 100.0,
    epsilon: float = 1e-5,
    max_iter: int = 1000,
    max_inner_iter: int = 5,
    max_samples: Optional[int] = None,
    channel_mode: str = "mean",
) -> float:
    """SIDL-based SHAP-RE/SHR following the SDForger supplementary path.

    The public SDForger-style implementation is univariate. For multichannel
    sensor windows, `mean` reduces each window to a channel-averaged trajectory;
    `per_channel` computes a channel-wise SHR and averages over channels.
    """

    real_eval = real_windows
    syn_eval = synthetic_windows
    if max_samples is not None and len(real_eval) > max_samples:
        real_eval = real_eval[:max_samples]
        syn_eval = syn_eval[:max_samples]
    if channel_mode == "mean":
        return _calculate_shr_sid_univariate(
            real_eval.mean(axis=2),
            syn_eval.mean(axis=2),
            seed,
            basis_count,
            lambda_,
            basis_ratio,
            c,
            epsilon,
            max_iter,
            max_inner_iter,
        )
    if channel_mode == "per_channel":
        values = []
        for channel_idx in range(real_eval.shape[2]):
            values.append(
                _calculate_shr_sid_univariate(
                    real_eval[:, :, channel_idx],
                    syn_eval[:, :, channel_idx],
                    seed + channel_idx,
                    basis_count,
                    lambda_,
                    basis_ratio,
                    c,
                    epsilon,
                    max_iter,
                    max_inner_iter,
                )
            )
        return float(np.nanmean(values))
    raise ValueError(f"Unsupported SHR channel_mode: {channel_mode}")


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
    mdd_bins: int = 50,
    compute_shr: bool = False,
    shr_seed: int = 42,
    shr_basis_count: int = 20,
    shr_lambda: float = 0.1,
    shr_basis_ratio: float = 0.25,
    shr_c: float = 100.0,
    shr_epsilon: float = 1e-5,
    shr_max_iter: int = 1000,
    shr_max_inner_iter: int = 5,
    shr_max_samples_per_activity: Optional[int] = None,
    shr_channel_mode: str = "mean",
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

        mdd = marginal_distribution_difference(real_sample, syn_sample, n_bins=mdd_bins)
        acd = autocorrelation_difference(real_sample, syn_sample, max_lag=max_lag)
        sd = skewness_difference(real_sample, syn_sample)
        kd = kurtosis_difference(real_sample, syn_sample)
        ed = euclidean_distance_pairs(real_sample, syn_sample)
        dtw = dynamic_time_warping_pairs(real_sample, syn_sample, window=dtw_window)
        shr = (
            shapelet_reconstruction_error(
                real_sample,
                syn_sample,
                seed=shr_seed,
                basis_count=shr_basis_count,
                lambda_=shr_lambda,
                basis_ratio=shr_basis_ratio,
                c=shr_c,
                epsilon=shr_epsilon,
                max_iter=shr_max_iter,
                max_inner_iter=shr_max_inner_iter,
                max_samples=shr_max_samples_per_activity,
                channel_mode=shr_channel_mode,
            )
            if compute_shr
            else float("nan")
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
                "SHR": shr,
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
            "SHR": float(result["SHR"].mean()),
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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
