import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA


PAMAP2_ACTIVITY_MAP: Dict[int, str] = {
    0: "other_transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}


PAMAP2_COLUMNS: List[str] = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temperature",
    "hand_acc_16g_x",
    "hand_acc_16g_y",
    "hand_acc_16g_z",
    "hand_acc_6g_x",
    "hand_acc_6g_y",
    "hand_acc_6g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orientation_q1",
    "hand_orientation_q2",
    "hand_orientation_q3",
    "hand_orientation_q4",
    "chest_temperature",
    "chest_acc_16g_x",
    "chest_acc_16g_y",
    "chest_acc_16g_z",
    "chest_acc_6g_x",
    "chest_acc_6g_y",
    "chest_acc_6g_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orientation_q1",
    "chest_orientation_q2",
    "chest_orientation_q3",
    "chest_orientation_q4",
    "ankle_temperature",
    "ankle_acc_16g_x",
    "ankle_acc_16g_y",
    "ankle_acc_16g_z",
    "ankle_acc_6g_x",
    "ankle_acc_6g_y",
    "ankle_acc_6g_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orientation_q1",
    "ankle_orientation_q2",
    "ankle_orientation_q3",
    "ankle_orientation_q4",
]


DEFAULT_CHANNELS: List[str] = [
    "hand_acc_16g_x",
    "hand_acc_16g_y",
    "hand_acc_16g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_acc_16g_x",
    "chest_acc_16g_y",
    "chest_acc_16g_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "ankle_acc_16g_x",
    "ankle_acc_16g_y",
    "ankle_acc_16g_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess PAMAP2 into window-level tensors and LLM-ready metadata."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing PAMAP2 protocol files such as subject101.dat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pamap2_baseline"),
        help="Directory used to save tensors, csv files, and summary json.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=None,
        help="Optional subject ids to keep, e.g. 101 102 103.",
    )
    parser.add_argument(
        "--activities",
        nargs="*",
        type=str,
        default=["walking", "running", "cycling", "sitting", "standing"],
        help="Activity names or ids to keep.",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional column names to keep. Defaults to a compact IMU subset.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Window size in timesteps.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding window stride. Use window-size for non-overlapping windows.",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=3.0,
        help="Split runs when timestamp delta exceeds gap-factor times the median delta.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.8,
        help="Drop rows that have fewer than this fraction of valid selected channels before interpolation.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "zscore"],
        default="zscore",
        help="How to normalize selected channels before saving the main tensor.",
    )
    parser.add_argument(
        "--feature-reducer",
        choices=["none", "pca", "fastica"],
        default="pca",
        help="Optional compact representation for future LLM pipelines.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=8,
        help="Number of PCA / FastICA components.",
    )
    parser.add_argument(
        "--save-raw-tensor",
        action="store_true",
        help="Also save raw window tensor before normalization.",
    )
    return parser


def normalize_activity_tokens(tokens: Sequence[str]) -> List[int]:
    reverse_map = {name: idx for idx, name in PAMAP2_ACTIVITY_MAP.items()}
    resolved: List[int] = []
    for token in tokens:
        if token.isdigit():
            activity_id = int(token)
        else:
            normalized = token.strip().lower().replace(" ", "_")
            if normalized not in reverse_map:
                raise ValueError(f"Unknown activity token: {token}")
            activity_id = reverse_map[normalized]
        if activity_id not in PAMAP2_ACTIVITY_MAP:
            raise ValueError(f"Unsupported activity id: {activity_id}")
        resolved.append(activity_id)
    return resolved


def parse_subject_id(file_path: Path) -> int:
    digits = "".join(ch for ch in file_path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer subject id from file name: {file_path.name}")
    return int(digits)


def load_pamap2(data_dir: Path, subjects: Optional[Sequence[int]] = None) -> pd.DataFrame:
    file_paths = sorted(data_dir.glob("*.dat"))
    if not file_paths:
        raise FileNotFoundError(f"No .dat files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    subject_filter = set(subjects) if subjects else None

    for file_path in file_paths:
        subject_id = parse_subject_id(file_path)
        if subject_filter and subject_id not in subject_filter:
            continue

        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=PAMAP2_COLUMNS,
            engine="python",
        )
        if frame.shape[1] != len(PAMAP2_COLUMNS):
            raise ValueError(
                f"{file_path.name} has {frame.shape[1]} columns, expected {len(PAMAP2_COLUMNS)}."
            )
        frame["subject_id"] = subject_id
        frame["source_file"] = file_path.name
        frames.append(frame)

    if not frames:
        raise ValueError("No PAMAP2 files matched the requested subjects.")

    data = pd.concat(frames, ignore_index=True)
    data["activity_id"] = data["activity_id"].fillna(0).astype(int)
    data["activity_name"] = data["activity_id"].map(PAMAP2_ACTIVITY_MAP).fillna("unknown")
    return data


def select_activities(data: pd.DataFrame, activity_ids: Sequence[int]) -> pd.DataFrame:
    filtered = data[data["activity_id"].isin(activity_ids)].copy()
    filtered = filtered[filtered["activity_id"] != 0].copy()
    filtered["activity_name"] = filtered["activity_id"].map(PAMAP2_ACTIVITY_MAP)
    return filtered


def select_channels(data: pd.DataFrame, channels: Optional[Sequence[str]] = None) -> List[str]:
    chosen = list(channels) if channels else list(DEFAULT_CHANNELS)
    missing = [channel for channel in chosen if channel not in data.columns]
    if missing:
        raise ValueError(f"Requested channels are not present: {missing}")
    return chosen


def clean_data(
    data: pd.DataFrame,
    channels: Sequence[str],
    min_valid_ratio: float = 0.8,
) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned[channels] = cleaned[channels].replace([np.inf, -np.inf], np.nan)

    valid_counts = cleaned[channels].notna().sum(axis=1)
    min_valid = max(1, int(np.ceil(len(channels) * min_valid_ratio)))
    cleaned = cleaned[valid_counts >= min_valid].copy()

    cleaned = cleaned.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

    def fill_group(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values("timestamp").copy()
        ordered[channels] = ordered[channels].interpolate(
            method="linear",
            limit_direction="both",
        )
        ordered[channels] = ordered[channels].ffill().bfill()
        return ordered

    cleaned = (
        cleaned.groupby(["subject_id", "activity_id"], group_keys=False)
        .apply(fill_group)
        .reset_index(drop=True)
    )

    cleaned = cleaned.dropna(subset=list(channels)).reset_index(drop=True)
    return cleaned


def assign_run_ids(data: pd.DataFrame, gap_factor: float = 3.0) -> pd.DataFrame:
    with_runs: List[pd.DataFrame] = []

    for (subject_id, activity_id), frame in data.groupby(["subject_id", "activity_id"], sort=False):
        ordered = frame.sort_values("timestamp").copy()
        deltas = ordered["timestamp"].diff()
        positive_deltas = deltas[deltas > 0]
        median_delta = positive_deltas.median() if not positive_deltas.empty else np.nan
        threshold = (
            median_delta * gap_factor
            if pd.notna(median_delta) and median_delta > 0
            else np.inf
        )

        is_new_run = deltas.isna() | (deltas <= 0) | (deltas > threshold)
        ordered["run_id"] = is_new_run.cumsum().astype(int)
        ordered["subject_id"] = subject_id
        ordered["activity_id"] = activity_id
        with_runs.append(ordered)

    return pd.concat(with_runs, ignore_index=True)


def compute_channel_scaler(
    data: pd.DataFrame,
    channels: Sequence[str],
    normalization: str,
) -> Tuple[np.ndarray, np.ndarray]:
    means = data.loc[:, channels].mean().to_numpy(dtype=np.float32)
    if normalization == "none":
        scales = np.ones_like(means, dtype=np.float32)
    else:
        scales = data.loc[:, channels].std(ddof=0).to_numpy(dtype=np.float32)
        scales = np.where(scales < 1e-8, 1.0, scales)
    return means, scales


def segment_windows(
    data: pd.DataFrame,
    channels: Sequence[str],
    window_size: int,
    stride: int,
    normalization: str,
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray]]:
    tensors: List[np.ndarray] = []
    raw_tensors: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    window_id = 0

    for (subject_id, activity_id, run_id), frame in data.groupby(
        ["subject_id", "activity_id", "run_id"], sort=False
    ):
        ordered = frame.sort_values("timestamp").reset_index(drop=True)
        values = ordered.loc[:, channels].to_numpy(dtype=np.float32)
        timestamps = ordered["timestamp"].to_numpy(dtype=np.float64)

        if len(ordered) < window_size:
            continue

        for start in range(0, len(ordered) - window_size + 1, stride):
            end = start + window_size
            raw_window = values[start:end]
            if normalization == "zscore":
                window = (raw_window - channel_means) / channel_stds
            else:
                window = raw_window.copy()

            tensors.append(window.astype(np.float32))
            raw_tensors.append(raw_window.astype(np.float32))

            metadata_rows.append(
                {
                    "window_id": window_id,
                    "subject_id": int(subject_id),
                    "activity_id": int(activity_id),
                    "activity_name": PAMAP2_ACTIVITY_MAP[int(activity_id)],
                    "run_id": int(run_id),
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "window_length": int(window_size),
                    "start_timestamp": float(timestamps[start]),
                    "end_timestamp": float(timestamps[end - 1]),
                    "source_file": ordered.loc[start, "source_file"],
                }
            )
            window_id += 1

    if not tensors:
        raise ValueError("No windows were generated. Try smaller window-size or different activities.")

    tensor_array = np.stack(tensors, axis=0)
    raw_tensor_array = np.stack(raw_tensors, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    return tensor_array, metadata, raw_tensor_array


def compute_window_features(
    raw_tensor: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
) -> pd.DataFrame:
    feature_rows: List[Dict[str, object]] = []

    for index, window in enumerate(raw_tensor):
        row: Dict[str, object] = {
            "window_id": int(metadata.iloc[index]["window_id"]),
            "subject_id": int(metadata.iloc[index]["subject_id"]),
            "activity_id": int(metadata.iloc[index]["activity_id"]),
            "activity_name": metadata.iloc[index]["activity_name"],
        }
        for channel_idx, channel_name in enumerate(channels):
            values = window[:, channel_idx]
            row[f"{channel_name}__mean"] = float(values.mean())
            row[f"{channel_name}__std"] = float(values.std(ddof=0))
            row[f"{channel_name}__min"] = float(values.min())
            row[f"{channel_name}__max"] = float(values.max())
            row[f"{channel_name}__slope"] = float((values[-1] - values[0]) / max(len(values) - 1, 1))
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def compute_compact_embeddings(
    raw_tensor: np.ndarray,
    method: str,
    n_components: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    if method == "none":
        return None, None

    flattened = raw_tensor.reshape(raw_tensor.shape[0], -1)
    n_components = min(n_components, flattened.shape[0], flattened.shape[1])
    if n_components < 1:
        return None, None

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(flattened).astype(np.float32)
        info = {
            "method": method,
            "n_components": int(n_components),
            "explained_variance_ratio": reducer.explained_variance_ratio_.tolist(),
        }
    else:
        reducer = FastICA(n_components=n_components, random_state=42, whiten="unit-variance")
        reduced = reducer.fit_transform(flattened).astype(np.float32)
        info = {
            "method": method,
            "n_components": int(n_components),
        }
    return reduced, info


def compute_statistics(
    data: pd.DataFrame,
    raw_tensor: np.ndarray,
    metadata: pd.DataFrame,
    channels: Sequence[str],
) -> Dict[str, object]:
    activity_rows: List[Dict[str, object]] = []
    subject_rows: List[Dict[str, object]] = []

    duration_by_subject_activity = (
        data.groupby(["subject_id", "activity_id", "run_id"])["timestamp"]
        .agg(["min", "max"])
        .reset_index()
    )
    duration_by_subject_activity["duration_seconds"] = (
        duration_by_subject_activity["max"] - duration_by_subject_activity["min"]
    )

    for activity_id, frame in data.groupby("activity_id", sort=True):
        duration_seconds = float(
            duration_by_subject_activity.loc[
                duration_by_subject_activity["activity_id"] == activity_id,
                "duration_seconds",
            ].sum()
        )
        activity_rows.append(
            {
                "activity_id": int(activity_id),
                "activity_name": PAMAP2_ACTIVITY_MAP[int(activity_id)],
                "num_rows": int(len(frame)),
                "num_windows": int((metadata["activity_id"] == int(activity_id)).sum()),
                "duration_seconds": duration_seconds,
            }
        )

    for subject_id, frame in data.groupby("subject_id", sort=True):
        duration_seconds = float(
            duration_by_subject_activity.loc[
                duration_by_subject_activity["subject_id"] == subject_id,
                "duration_seconds",
            ].sum()
        )
        subject_rows.append(
            {
                "subject_id": int(subject_id),
                "num_rows": int(len(frame)),
                "num_windows": int((metadata["subject_id"] == int(subject_id)).sum()),
                "duration_seconds": duration_seconds,
            }
        )

    channel_rows: List[Dict[str, object]] = []
    for channel in channels:
        values = data[channel].to_numpy(dtype=np.float64)
        channel_rows.append(
            {
                "channel": channel,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    summary = {
        "num_rows_after_cleaning": int(len(data)),
        "num_windows": int(raw_tensor.shape[0]),
        "window_size": int(raw_tensor.shape[1]),
        "num_channels": int(raw_tensor.shape[2]),
        "activities": activity_rows,
        "subjects_detail": subject_rows,
        "channels": channel_rows,
    }
    return summary


def build_llm_training_records(
    metadata: pd.DataFrame,
    feature_frame: pd.DataFrame,
    compact_embeddings: Optional[np.ndarray],
    channels: Sequence[str],
) -> List[Dict[str, object]]:
    features_by_window = feature_frame.set_index("window_id").to_dict(orient="index")
    records: List[Dict[str, object]] = []

    for index, row in metadata.iterrows():
        window_id = int(row["window_id"])
        feature_payload = features_by_window[window_id]
        stats_summary = {
            key: value
            for key, value in feature_payload.items()
            if key.endswith("__mean") or key.endswith("__std")
        }
        record = {
            "window_id": window_id,
            "subject_id": int(row["subject_id"]),
            "activity_id": int(row["activity_id"]),
            "activity_name": row["activity_name"],
            "condition_text": (
                f"Activity: {row['activity_name']}. Subject: {int(row['subject_id'])}. "
                f"Window length: {int(row['window_length'])}. Channels: {', '.join(channels)}."
            ),
            "embedding_placeholder": compact_embeddings[index].tolist() if compact_embeddings is not None else None,
            "stats_placeholder": stats_summary,
            "input_template_raw_to_text": (
                "Condition: activity is {activity_name}. Subject: {subject_id}. "
                "Window embedding: {embedding_placeholder}. Describe or generate the sensor trajectory."
            ),
            "input_template_stats_to_text": (
                "Condition: activity is {activity_name}. Subject: {subject_id}. "
                "Summary stats: {stats_placeholder}. Generate a plausible sensor window."
            ),
            "target_placeholder": "<future_sensor_sequence_or_text_target>",
        }
        records.append(record)

    return records


def save_outputs(
    output_dir: Path,
    window_tensor: np.ndarray,
    raw_tensor: np.ndarray,
    metadata: pd.DataFrame,
    feature_frame: pd.DataFrame,
    summary: Dict[str, object],
    label_map: Dict[int, str],
    llm_records: List[Dict[str, object]],
    channels: Sequence[str],
    normalization: str,
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    compact_embeddings: Optional[np.ndarray],
    compact_info: Optional[Dict[str, object]],
    save_raw_tensor: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "window_tensor.npy", window_tensor)
    if save_raw_tensor:
        np.save(output_dir / "window_tensor_raw.npy", raw_tensor)
    metadata.to_csv(output_dir / "window_metadata.csv", index=False)
    feature_frame.to_csv(output_dir / "window_features.csv", index=False)

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as fp:
        json.dump({str(k): v for k, v in label_map.items() if k != 0}, fp, indent=2, ensure_ascii=False)

    scaler_payload = {
        "normalization": normalization,
        "channels": list(channels),
        "mean": channel_means.tolist(),
        "std": channel_stds.tolist(),
    }
    with open(output_dir / "channel_scaler.json", "w", encoding="utf-8") as fp:
        json.dump(scaler_payload, fp, indent=2, ensure_ascii=False)

    if compact_embeddings is not None:
        np.save(output_dir / "window_compact_embeddings.npy", compact_embeddings)
    if compact_info is not None:
        with open(output_dir / "compact_representation.json", "w", encoding="utf-8") as fp:
            json.dump(compact_info, fp, indent=2, ensure_ascii=False)

    with open(output_dir / "llm_training_records.jsonl", "w", encoding="utf-8") as fp:
        for record in llm_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    activity_ids = normalize_activity_tokens(args.activities)
    raw_data = load_pamap2(args.data_dir, subjects=args.subjects)
    filtered_data = select_activities(raw_data, activity_ids)
    channels = select_channels(filtered_data, args.channels)
    cleaned_data = clean_data(
        filtered_data,
        channels=channels,
        min_valid_ratio=args.min_valid_ratio,
    )
    cleaned_data = assign_run_ids(cleaned_data, gap_factor=args.gap_factor)

    channel_means, channel_stds = compute_channel_scaler(
        cleaned_data,
        channels=channels,
        normalization=args.normalization,
    )

    window_tensor, metadata, raw_tensor = segment_windows(
        cleaned_data,
        channels=channels,
        window_size=args.window_size,
        stride=args.stride,
        normalization=args.normalization,
        channel_means=channel_means,
        channel_stds=channel_stds,
    )

    feature_frame = compute_window_features(raw_tensor, metadata, channels)
    compact_embeddings, compact_info = compute_compact_embeddings(
        raw_tensor,
        method=args.feature_reducer,
        n_components=args.n_components,
    )
    summary = compute_statistics(cleaned_data, raw_tensor, metadata, channels)
    summary.update(
        {
            "subjects": sorted(cleaned_data["subject_id"].unique().tolist()),
            "selected_activity_ids": activity_ids,
            "selected_activity_names": [PAMAP2_ACTIVITY_MAP[idx] for idx in activity_ids],
            "selected_channels": list(channels),
            "normalization": args.normalization,
            "window_stride": args.stride,
            "feature_reducer": args.feature_reducer,
            "compact_representation": compact_info,
            "llm_ready_fields": [
                "condition_text",
                "activity_name",
                "subject_id",
                "embedding_placeholder",
                "stats_placeholder",
            ],
        }
    )

    llm_records = build_llm_training_records(
        metadata=metadata,
        feature_frame=feature_frame,
        compact_embeddings=compact_embeddings,
        channels=channels,
    )

    save_outputs(
        output_dir=args.output_dir,
        window_tensor=window_tensor,
        raw_tensor=raw_tensor,
        metadata=metadata,
        feature_frame=feature_frame,
        summary=summary,
        label_map=PAMAP2_ACTIVITY_MAP,
        llm_records=llm_records,
        channels=channels,
        normalization=args.normalization,
        channel_means=channel_means,
        channel_stds=channel_stds,
        compact_embeddings=compact_embeddings,
        compact_info=compact_info,
        save_raw_tensor=args.save_raw_tensor,
    )

    print(f"Saved {window_tensor.shape[0]} windows to {args.output_dir}")
    print(f"Tensor shape: {tuple(window_tensor.shape)}")
    print(f"Activities: {[PAMAP2_ACTIVITY_MAP[idx] for idx in activity_ids]}")
    print(f"Channels: {channels}")


if __name__ == "__main__":
    main()
