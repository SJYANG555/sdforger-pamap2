import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ChannelReducerState:
    channel_name: str
    method: str
    n_components: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    components: np.ndarray
    mixing: Optional[np.ndarray]
    latent_mean: np.ndarray
    explained_variance_ratio: Optional[np.ndarray]


class WindowReducer:
    def __init__(
        self,
        method: str = "fastica",
        n_components: int = 3,
        variance_explained: float = 0.7,
        standardization: str = "legacy_timepoint",
        fastica_max_iter: int = 1000,
        fastica_tol: float = 1.0e-4,
    ):
        self.method = method.lower()
        self.n_components = n_components
        self.variance_explained = variance_explained
        self.standardization = standardization
        self.fastica_max_iter = fastica_max_iter
        self.fastica_tol = fastica_tol
        self.states: List[ChannelReducerState] = []
        self.channel_names: List[str] = []
        self.window_size: Optional[int] = None
        self.channel_scaler_mean: Optional[np.ndarray] = None
        self.channel_scaler_scale: Optional[np.ndarray] = None

    def _fit_prestandardizer(self, windows: np.ndarray) -> np.ndarray:
        if self.standardization in {"legacy_timepoint", "none", None}:
            self.channel_scaler_mean = None
            self.channel_scaler_scale = None
            return windows
        if self.standardization != "per_channel_train":
            raise ValueError(f"Unsupported embedding standardization: {self.standardization}")

        mean = windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale = windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.channel_scaler_mean = mean
        self.channel_scaler_scale = scale
        return ((windows - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)).astype(np.float32)

    def _apply_prestandardizer(self, windows: np.ndarray) -> np.ndarray:
        if self.channel_scaler_mean is None or self.channel_scaler_scale is None:
            return windows
        return (
            (windows - self.channel_scaler_mean.reshape(1, 1, -1))
            / self.channel_scaler_scale.reshape(1, 1, -1)
        ).astype(np.float32)

    def _invert_prestandardizer(self, windows: np.ndarray) -> np.ndarray:
        if self.channel_scaler_mean is None or self.channel_scaler_scale is None:
            return windows
        return (
            windows * self.channel_scaler_scale.reshape(1, 1, -1)
            + self.channel_scaler_mean.reshape(1, 1, -1)
        ).astype(np.float32)

    def fit_transform(
        self,
        windows: np.ndarray,
        channel_names: List[str],
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        num_windows, window_size, num_channels = windows.shape
        self.window_size = window_size
        self.channel_names = list(channel_names)
        windows = self._fit_prestandardizer(windows)
        embedded_parts: List[np.ndarray] = []
        self.states = []

        for channel_idx, channel_name in enumerate(channel_names):
            channel_matrix = windows[:, :, channel_idx]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(channel_matrix)
            effective_components = min(self.n_components, scaled.shape[0], scaled.shape[1])

            if self.method == "pca":
                reducer = PCA(n_components=effective_components, random_state=42)
                embedded = reducer.fit_transform(scaled)
                state = ChannelReducerState(
                    channel_name=channel_name,
                    method=self.method,
                    n_components=effective_components,
                    scaler_mean=scaler.mean_.astype(np.float32),
                    scaler_scale=scaler.scale_.astype(np.float32),
                    components=reducer.components_.astype(np.float32),
                    mixing=None,
                    latent_mean=reducer.mean_.astype(np.float32),
                    explained_variance_ratio=reducer.explained_variance_ratio_.astype(np.float32),
                )
            elif self.method == "fastica":
                reducer = FastICA(
                    n_components=effective_components,
                    random_state=42,
                    whiten="unit-variance",
                    max_iter=self.fastica_max_iter,
                    tol=self.fastica_tol,
                )
                embedded = reducer.fit_transform(scaled)
                state = ChannelReducerState(
                    channel_name=channel_name,
                    method=self.method,
                    n_components=effective_components,
                    scaler_mean=scaler.mean_.astype(np.float32),
                    scaler_scale=scaler.scale_.astype(np.float32),
                    components=reducer.components_.astype(np.float32),
                    mixing=reducer.mixing_.astype(np.float32),
                    latent_mean=reducer.mean_.astype(np.float32),
                    explained_variance_ratio=None,
                )
            else:
                raise ValueError(f"Unsupported reducer method: {self.method}")

            self.states.append(state)
            embedded_parts.append(embedded.astype(np.float32))

        combined = np.hstack(embedded_parts)
        columns = [f"value_{index}" for index in range(combined.shape[1])]
        frame = pd.DataFrame(combined, columns=columns)
        frame.insert(0, "window_id", metadata["window_id"].to_numpy())
        frame.insert(1, "subject_id", metadata["subject_id"].to_numpy())
        frame.insert(2, "activity_id", metadata["activity_id"].to_numpy())
        frame.insert(3, "activity_name", metadata["activity_name"].to_numpy())
        return frame

    def transform(self, windows: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
        windows = self._apply_prestandardizer(windows)
        embedded_parts: List[np.ndarray] = []
        for channel_idx, state in enumerate(self.states):
            channel_matrix = windows[:, :, channel_idx]
            scaled = (channel_matrix - state.scaler_mean) / state.scaler_scale
            if state.method == "pca":
                embedded = (scaled - state.latent_mean) @ state.components.T
            else:
                unmixing = state.components
                centered = scaled - state.latent_mean
                embedded = centered @ unmixing.T
            embedded_parts.append(embedded.astype(np.float32))

        combined = np.hstack(embedded_parts)
        columns = [f"value_{index}" for index in range(combined.shape[1])]
        frame = pd.DataFrame(combined, columns=columns)
        frame.insert(0, "window_id", metadata["window_id"].to_numpy())
        frame.insert(1, "subject_id", metadata["subject_id"].to_numpy())
        frame.insert(2, "activity_id", metadata["activity_id"].to_numpy())
        frame.insert(3, "activity_name", metadata["activity_name"].to_numpy())
        return frame

    def inverse_transform(self, embedding_frame: pd.DataFrame) -> np.ndarray:
        numeric = embedding_frame[[col for col in embedding_frame.columns if col.startswith("value_")]].to_numpy(
            dtype=np.float32
        )
        split_points = np.cumsum([state.n_components for state in self.states])[:-1]
        parts = np.split(numeric, split_points, axis=1)
        reconstructed_channels: List[np.ndarray] = []

        for part, state in zip(parts, self.states):
            if state.method == "pca":
                scaled = part @ state.components + state.latent_mean
            else:
                scaled = part @ state.mixing.T + state.latent_mean
            reconstructed = scaled * state.scaler_scale + state.scaler_mean
            reconstructed_channels.append(reconstructed.astype(np.float32))

        stacked = np.stack(reconstructed_channels, axis=-1)
        return self._invert_prestandardizer(stacked)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as fp:
            pickle.dump(
                {
                    "method": self.method,
                    "n_components": self.n_components,
                    "variance_explained": self.variance_explained,
                    "standardization": self.standardization,
                    "fastica_max_iter": self.fastica_max_iter,
                    "fastica_tol": self.fastica_tol,
                    "channel_names": self.channel_names,
                    "window_size": self.window_size,
                    "states": self.states,
                    "channel_scaler_mean": self.channel_scaler_mean,
                    "channel_scaler_scale": self.channel_scaler_scale,
                },
                fp,
            )

    def save_channel_scaler(self, path: Union[str, Path]) -> None:
        payload: Dict[str, object] = {
            "standardization": self.standardization,
            "channel_names": self.channel_names,
            "mean": None,
            "scale": None,
        }
        if self.channel_scaler_mean is not None and self.channel_scaler_scale is not None:
            payload["mean"] = [float(value) for value in self.channel_scaler_mean.tolist()]
            payload["scale"] = [float(value) for value in self.channel_scaler_scale.tolist()]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WindowReducer":
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        reducer = cls(
            method=payload["method"],
            n_components=payload["n_components"],
            variance_explained=payload["variance_explained"],
            standardization=payload.get("standardization", "legacy_timepoint"),
            fastica_max_iter=payload.get("fastica_max_iter", 1000),
            fastica_tol=payload.get("fastica_tol", 1.0e-4),
        )
        reducer.channel_names = payload["channel_names"]
        reducer.window_size = payload["window_size"]
        reducer.states = payload["states"]
        reducer.channel_scaler_mean = payload.get("channel_scaler_mean")
        reducer.channel_scaler_scale = payload.get("channel_scaler_scale")
        return reducer

    def export_summary(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "standardization": self.standardization,
            "fastica_max_iter": self.fastica_max_iter,
            "fastica_tol": self.fastica_tol,
            "n_components_per_channel": {state.channel_name: state.n_components for state in self.states},
            "total_embedding_values": int(sum(state.n_components for state in self.states)),
            "window_size": self.window_size,
            "channel_names": self.channel_names,
        }
