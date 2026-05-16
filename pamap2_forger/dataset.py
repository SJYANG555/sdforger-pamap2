import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from pamap2_forger.config import DataConfig, EmbeddingConfig
from pamap2_forger.embeddings import WindowReducer
from pamap2_forger.text import dataframe_to_text_records
from pamap2_forger.utils import write_json


CONTEXT_COLUMNS = ["subject_id", "activity_id", "activity_name"]


def load_preprocessed_artifacts(preprocessed_dir: Union[str, Path]) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    preprocessed_dir = Path(preprocessed_dir)
    windows = np.load(preprocessed_dir / "window_tensor_raw.npy")
    metadata = pd.read_csv(preprocessed_dir / "window_metadata.csv")
    summary = json.loads((preprocessed_dir / "summary.json").read_text(encoding="utf-8"))
    return windows, metadata, summary


def resolve_splits(metadata: pd.DataFrame, config: DataConfig) -> Dict[str, pd.DataFrame]:
    if config.split_strategy == "subject":
        unique_subjects = sorted(metadata["subject_id"].unique().tolist())
        train_subjects = config.train_subjects or unique_subjects[:-2]
        val_subjects = config.val_subjects or [unique_subjects[-2]]
        test_subjects = config.test_subjects or [unique_subjects[-1]]

        return {
            "train": metadata[metadata["subject_id"].isin(train_subjects)].copy(),
            "val": metadata[metadata["subject_id"].isin(val_subjects)].copy(),
            "test": metadata[metadata["subject_id"].isin(test_subjects)].copy(),
        }

    shuffled = metadata.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    total = len(shuffled)
    train_end = int(total * config.train_ratio)
    val_end = train_end + int(total * config.val_ratio)
    return {
        "train": shuffled.iloc[:train_end].copy(),
        "val": shuffled.iloc[train_end:val_end].copy(),
        "test": shuffled.iloc[val_end:].copy(),
    }


def select_windows(windows: np.ndarray, metadata: pd.DataFrame, subset_metadata: pd.DataFrame) -> np.ndarray:
    indexer = subset_metadata["window_id"].to_numpy(dtype=int)
    return windows[indexer]


def resolve_channel_selection(channel_names: list[str], config: DataConfig) -> Tuple[list[int], list[str]]:
    selected_channels = config.selected_channels
    selected_prefixes = config.selected_channel_prefixes
    if not selected_channels and not selected_prefixes:
        return list(range(len(channel_names))), channel_names

    selected = set(selected_channels or [])
    prefixes = tuple(selected_prefixes or [])
    indexes = [
        index
        for index, channel_name in enumerate(channel_names)
        if channel_name in selected or (prefixes and channel_name.startswith(prefixes))
    ]
    if not indexes:
        raise ValueError(
            "Channel selection matched no channels. "
            f"selected_channels={selected_channels}, selected_channel_prefixes={selected_prefixes}"
        )
    return indexes, [channel_names[index] for index in indexes]


def build_sdforger_dataset(config: DataConfig, embedding: EmbeddingConfig) -> Dict[str, object]:
    windows, metadata, preproc_summary = load_preprocessed_artifacts(config.preprocessed_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_channel_names = list(preproc_summary["selected_channels"])
    channel_indexes, channel_names = resolve_channel_selection(source_channel_names, config)
    windows = windows[:, :, channel_indexes]
    split_frames = resolve_splits(metadata, config)

    reducer = WindowReducer(
        method=embedding.method,
        n_components=embedding.n_components,
        variance_explained=embedding.variance_explained,
        standardization=embedding.standardization,
        fastica_max_iter=embedding.fastica_max_iter,
        fastica_tol=embedding.fastica_tol,
    )

    train_windows = select_windows(windows, metadata, split_frames["train"])
    train_embeddings = reducer.fit_transform(train_windows, channel_names, split_frames["train"])
    reducer.save(output_dir / embedding.reducer_artifact_name)
    reducer.save_channel_scaler(output_dir / embedding.scaler_artifact_name)

    all_embedding_frames: Dict[str, pd.DataFrame] = {"train": train_embeddings}
    split_window_tensors: Dict[str, np.ndarray] = {"train": train_windows}
    for split_name in ("val", "test"):
        split_windows = select_windows(windows, metadata, split_frames[split_name])
        split_window_tensors[split_name] = split_windows
        all_embedding_frames[split_name] = reducer.transform(split_windows, split_frames[split_name])

    manifest = {
        "source_preprocessed_dir": str(Path(config.preprocessed_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split_strategy": config.split_strategy,
        "embedding": reducer.export_summary(),
        "embedding_artifacts": {
            "reducer_path": str((output_dir / embedding.reducer_artifact_name).resolve()),
            "channel_scaler_path": str((output_dir / embedding.scaler_artifact_name).resolve()),
        },
        "channel_schema": {
            "source_selected_channels": source_channel_names,
            "selected_channels": channel_names,
            "selected_channel_prefixes": config.selected_channel_prefixes,
            "num_channels": len(channel_names),
            "window_size": int(windows.shape[1]),
        },
        "splits": {},
    }
    split_manifest = {"split_strategy": config.split_strategy, "splits": {}}

    for split_name, frame in all_embedding_frames.items():
        frame.to_csv(output_dir / f"{split_name}_embeddings.csv", index=False)
        np.save(output_dir / f"{split_name}_windows.npy", split_window_tensors[split_name])
        text_records = dataframe_to_text_records(
            frame=frame,
            eos_token="<|endoftext|>",
            permute=embedding.permute_columns,
            text_template=embedding.text_template,
            input_tokens_precision=embedding.input_tokens_precision,
        )
        with open(output_dir / f"{split_name}_text.jsonl", "w", encoding="utf-8") as fp:
            for record in text_records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        split_meta = split_frames[split_name].copy()
        split_meta.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
        manifest["splits"][split_name] = {
            "num_windows": int(len(split_meta)),
            "subjects": sorted(split_meta["subject_id"].unique().tolist()),
            "activities": sorted(split_meta["activity_name"].unique().tolist()),
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
            "reducer_path": str((output_dir / embedding.reducer_artifact_name).resolve()),
            "channel_scaler_path": str((output_dir / embedding.scaler_artifact_name).resolve()),
            "window_size": int(windows.shape[1]),
            "channel_names": channel_names,
            "embedding_method": embedding.method,
            "embedding_n_components": embedding.n_components,
            "embedding_standardization": embedding.standardization,
        },
    )

    return manifest
