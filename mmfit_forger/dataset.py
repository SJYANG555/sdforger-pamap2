import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pamap2_forger.embeddings import WindowReducer
from pamap2_forger.utils import write_json


DEFAULT_SELECTED_ACTIVITIES = [
    "squats",
    "pushups",
    "lunges",
    "bicep_curls",
    "jumping_jacks",
]

DEFAULT_SENSOR_STREAMS = [
    "sw_l_acc",
    "sw_l_gyr",
    "sw_r_acc",
    "sw_r_gyr",
]

ACTIVITY_IDS = {
    "squats": 0,
    "pushups": 1,
    "lunges": 2,
    "bicep_curls": 3,
    "jumping_jacks": 4,
}


@dataclass
class MMFitBuildConfig:
    source_path: str
    output_dir: str
    selected_activities: List[str]
    sensor_streams: List[str]
    train_sessions: List[int]
    val_sessions: List[int]
    test_sessions: List[int]
    window_seconds: float = 5.0
    stride_seconds: float = 2.5
    video_fps: float = 30.0
    output_sample_rate_hz: float = 50.0
    random_seed: int = 42

    @property
    def window_size(self) -> int:
        return int(round(self.window_seconds * self.output_sample_rate_hz))

    @property
    def window_frames(self) -> int:
        return int(round(self.window_seconds * self.video_fps))

    @property
    def stride_frames(self) -> int:
        return max(1, int(round(self.stride_seconds * self.video_fps)))


@dataclass
class MMFitEmbeddingConfig:
    method: str = "fastica"
    n_components: int = 6
    variance_explained: float = 0.7
    standardization: str = "per_channel_train"
    fastica_max_iter: int = 3000
    fastica_tol: float = 0.001
    input_tokens_precision: int = 4
    reducer_artifact_name: str = "reducer.pkl"
    scaler_artifact_name: str = "channel_scaler.json"


class MMFitSource:
    def __init__(self, source_path: Union[str, Path]):
        self.source_path = Path(source_path)
        self._zip: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> "MMFitSource":
        if self.source_path.is_file():
            self._zip = zipfile.ZipFile(self.source_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zip is not None:
            self._zip.close()

    def read_text(self, session_id: int, suffix: str) -> str:
        relative = self._relative_path(session_id, suffix)
        if self._zip is not None:
            return self._zip.read(relative).decode("utf-8")
        return (self.source_path / f"w{session_id:02d}" / f"w{session_id:02d}_{suffix}").read_text(encoding="utf-8")

    def load_npy(self, session_id: int, suffix: str) -> np.ndarray:
        relative = self._relative_path(session_id, suffix)
        if self._zip is not None:
            return np.load(io.BytesIO(self._zip.read(relative)))
        return np.load(self.source_path / f"w{session_id:02d}" / f"w{session_id:02d}_{suffix}")

    @staticmethod
    def _relative_path(session_id: int, suffix: str) -> str:
        return f"mm-fit/w{session_id:02d}/w{session_id:02d}_{suffix}"


def load_labels(source: MMFitSource, session_id: int) -> pd.DataFrame:
    text = source.read_text(session_id, "labels.csv")
    frame = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=["start_frame", "end_frame", "repetition_count", "activity_name"],
    )
    frame["session_id"] = session_id
    return frame


def load_sensor_stream(source: MMFitSource, session_id: int, stream: str) -> pd.DataFrame:
    raw = source.load_npy(session_id, f"{stream}.npy")
    frame = pd.DataFrame(
        {
            "frame": raw[:, 0].astype(np.float64),
            f"{stream}_x": raw[:, 2].astype(np.float32),
            f"{stream}_y": raw[:, 3].astype(np.float32),
            f"{stream}_z": raw[:, 4].astype(np.float32),
        }
    )
    return frame.groupby("frame", as_index=False).mean().sort_values("frame")


def resample_stream(stream: pd.DataFrame, target_frames: np.ndarray, channel_columns: List[str]) -> np.ndarray:
    source_frames = stream["frame"].to_numpy(dtype=np.float64)
    values = []
    for column in channel_columns:
        values.append(np.interp(target_frames, source_frames, stream[column].to_numpy(dtype=np.float32)))
    return np.stack(values, axis=1).astype(np.float32)


def build_windows_for_session(
    source: MMFitSource,
    session_id: int,
    config: MMFitBuildConfig,
) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[str]]:
    labels = load_labels(source, session_id)
    labels = labels[labels["activity_name"].isin(config.selected_activities)].copy()
    streams = {
        stream_name: load_sensor_stream(source, session_id, stream_name)
        for stream_name in config.sensor_streams
    }
    channel_names = [
        channel
        for stream_name in config.sensor_streams
        for channel in [f"{stream_name}_x", f"{stream_name}_y", f"{stream_name}_z"]
    ]

    windows: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    for label_index, row in labels.reset_index(drop=True).iterrows():
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        if end_frame - start_frame < config.window_frames:
            continue

        for window_start in range(start_frame, end_frame - config.window_frames + 1, config.stride_frames):
            target_frames = np.linspace(
                window_start,
                window_start + config.window_frames,
                num=config.window_size,
                endpoint=False,
                dtype=np.float64,
            )
            parts = []
            for stream_name, stream_frame in streams.items():
                parts.append(
                    resample_stream(
                        stream_frame,
                        target_frames,
                        [f"{stream_name}_x", f"{stream_name}_y", f"{stream_name}_z"],
                    )
                )
            windows.append(np.concatenate(parts, axis=1).astype(np.float32))
            metadata.append(
                {
                    "session_id": int(session_id),
                    "subject_id": int(session_id),
                    "set_id": int(label_index),
                    "start_frame": int(window_start),
                    "end_frame": int(window_start + config.window_frames),
                    "label_start_frame": int(start_frame),
                    "label_end_frame": int(end_frame),
                    "repetition_count": int(row["repetition_count"]),
                    "activity_id": int(ACTIVITY_IDS.get(row["activity_name"], config.selected_activities.index(row["activity_name"]))),
                    "activity_name": str(row["activity_name"]),
                }
            )
    return windows, metadata, channel_names


def build_mmfit_window_dataset(config: MMFitBuildConfig) -> Tuple[np.ndarray, pd.DataFrame, List[str], Dict[str, object]]:
    all_sessions = sorted(set(config.train_sessions + config.val_sessions + config.test_sessions))
    windows: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    channel_names: List[str] = []

    with MMFitSource(config.source_path) as source:
        for session_id in all_sessions:
            session_windows, session_rows, session_channels = build_windows_for_session(source, session_id, config)
            if not channel_names:
                channel_names = session_channels
            windows.extend(session_windows)
            rows.extend(session_rows)

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError("No MM-Fit windows were built. Check selected activities, sessions, and window length.")
    metadata.insert(0, "window_id", np.arange(len(metadata), dtype=int))
    window_tensor = np.stack(windows, axis=0).astype(np.float32)

    summary = {
        "source_path": str(Path(config.source_path).resolve()),
        "selected_activities": config.selected_activities,
        "sensor_streams": config.sensor_streams,
        "selected_channels": channel_names,
        "window_seconds": config.window_seconds,
        "stride_seconds": config.stride_seconds,
        "video_fps": config.video_fps,
        "output_sample_rate_hz": config.output_sample_rate_hz,
        "window_size": int(config.window_size),
        "num_windows": int(len(metadata)),
        "class_counts": metadata["activity_name"].value_counts().sort_index().to_dict(),
    }
    return window_tensor, metadata, channel_names, summary


def split_metadata(metadata: pd.DataFrame, config: MMFitBuildConfig) -> Dict[str, pd.DataFrame]:
    splits = {
        "train": metadata[metadata["session_id"].isin(config.train_sessions)].copy(),
        "val": metadata[metadata["session_id"].isin(config.val_sessions)].copy(),
        "test": metadata[metadata["session_id"].isin(config.test_sessions)].copy(),
    }
    for frame in splits.values():
        frame.sort_values("window_id", inplace=True)
    return splits


def select_windows(windows: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    return windows[metadata["window_id"].to_numpy(dtype=int)]


def mmfit_text_records(
    frame: pd.DataFrame,
    channel_names: Iterable[str],
    input_tokens_precision: int,
    eos_token: str = "<|endoftext|>",
) -> List[Dict[str, object]]:
    numeric_columns = [column for column in frame.columns if column.startswith("value_")]
    channel_summary = ",".join(channel_names)
    records = []
    for row in frame.to_dict(orient="records"):
        target = "; ".join(
            f"{column} = {float(row[column]):.{input_tokens_precision}f}"
            for column in numeric_columns
        )
        text = (
            "Condition:\n"
            "dataset = MM-Fit\n"
            f"activity_name = {row['activity_name']}\n"
            f"activity_id = {int(row['activity_id'])}\n"
            f"subject_id = {int(row['subject_id'])}\n"
            f"window_id = {int(row['window_id'])}\n"
            f"sensor_channels = {channel_summary}\n"
            "task = generate a standardized MM-Fit smartwatch exercise embedding\n\n"
            f"Input:\nvalues = [blank] x {len(numeric_columns)}\n\n"
            f"Target:\n{target}; [end]{eos_token}"
        )
        records.append({"text": text, **row})
    return records


def write_text_jsonl(path: Union[str, Path], records: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_mmfit_sdforger_dataset(
    build_config: MMFitBuildConfig,
    embedding_config: MMFitEmbeddingConfig,
) -> Dict[str, object]:
    output_dir = Path(build_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, metadata, channel_names, source_summary = build_mmfit_window_dataset(build_config)
    split_frames = split_metadata(metadata, build_config)

    reducer = WindowReducer(
        method=embedding_config.method,
        n_components=embedding_config.n_components,
        variance_explained=embedding_config.variance_explained,
        standardization=embedding_config.standardization,
        fastica_max_iter=embedding_config.fastica_max_iter,
        fastica_tol=embedding_config.fastica_tol,
    )

    train_windows = select_windows(windows, split_frames["train"])
    train_embeddings = reducer.fit_transform(train_windows, channel_names, split_frames["train"])
    reducer.save(output_dir / embedding_config.reducer_artifact_name)
    reducer.save_channel_scaler(output_dir / embedding_config.scaler_artifact_name)

    split_windows = {"train": train_windows}
    split_embeddings = {"train": train_embeddings}
    for split_name in ("val", "test"):
        split_windows[split_name] = select_windows(windows, split_frames[split_name])
        split_embeddings[split_name] = reducer.transform(split_windows[split_name], split_frames[split_name])

    manifest = {
        "dataset": "MM-Fit",
        "source": source_summary,
        "output_dir": str(output_dir.resolve()),
        "split_strategy": "official_workout_session_split",
        "splits": {},
        "embedding": reducer.export_summary(),
        "embedding_artifacts": {
            "reducer_path": str((output_dir / embedding_config.reducer_artifact_name).resolve()),
            "channel_scaler_path": str((output_dir / embedding_config.scaler_artifact_name).resolve()),
        },
        "channel_schema": {
            "selected_channels": channel_names,
            "num_channels": len(channel_names),
            "window_size": int(windows.shape[1]),
        },
    }
    split_manifest = {"split_strategy": "official_workout_session_split", "splits": {}}

    for split_name in ("train", "val", "test"):
        split_frames[split_name].to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
        split_embeddings[split_name].to_csv(output_dir / f"{split_name}_embeddings.csv", index=False)
        np.save(output_dir / f"{split_name}_windows.npy", split_windows[split_name])
        write_text_jsonl(
            output_dir / f"{split_name}_text.jsonl",
            mmfit_text_records(
                split_embeddings[split_name],
                channel_names=channel_names,
                input_tokens_precision=embedding_config.input_tokens_precision,
            ),
        )
        manifest["splits"][split_name] = {
            "num_windows": int(len(split_frames[split_name])),
            "sessions": sorted(split_frames[split_name]["session_id"].unique().astype(int).tolist()),
            "activities": sorted(split_frames[split_name]["activity_name"].unique().tolist()),
            "class_counts": split_frames[split_name]["activity_name"].value_counts().sort_index().to_dict(),
        }
        split_manifest["splits"][split_name] = {
            "metadata_path": str((output_dir / f"{split_name}_metadata.csv").resolve()),
            "embeddings_path": str((output_dir / f"{split_name}_embeddings.csv").resolve()),
            "windows_path": str((output_dir / f"{split_name}_windows.npy").resolve()),
            "text_path": str((output_dir / f"{split_name}_text.jsonl").resolve()),
        }

    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(
        output_dir / "reconstruction_metadata.json",
        {
            "dataset": "MM-Fit",
            "reducer_path": str((output_dir / embedding_config.reducer_artifact_name).resolve()),
            "channel_scaler_path": str((output_dir / embedding_config.scaler_artifact_name).resolve()),
            "window_size": int(windows.shape[1]),
            "channel_names": channel_names,
            "embedding_method": embedding_config.method,
            "embedding_n_components": embedding_config.n_components,
            "embedding_standardization": embedding_config.standardization,
        },
    )
    return manifest

