import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def select_rows_by_window_id(frame: pd.DataFrame, window_ids: Iterable[int]) -> pd.DataFrame:
    id_list = list(window_ids)
    return frame[frame["window_id"].isin(id_list)].copy()


def get_numeric_embedding_columns(frame: pd.DataFrame) -> List[str]:
    return [column for column in frame.columns if column.startswith("value_")]


def build_output_layout(root: Union[str, Path]) -> Dict[str, Path]:
    root = ensure_dir(root)
    layout = {
        "root": root,
        "checkpoints": ensure_dir(root / "checkpoints"),
        "generated": ensure_dir(root / "generated"),
        "evaluation": ensure_dir(root / "evaluation"),
        "plots": ensure_dir(root / "plots"),
        "logs": ensure_dir(root / "logs"),
    }
    return layout


def summarize_by_group(frame: pd.DataFrame, group_column: str) -> Dict[str, int]:
    counts = frame[group_column].value_counts().sort_index()
    return {str(index): int(value) for index, value in counts.items()}


def save_table(path: Union[str, Path], frame: pd.DataFrame) -> None:
    ensure_dir(Path(path).parent)
    frame.to_csv(path, index=False)


def resolve_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def maybe_limit_records(records: List[Dict[str, Any]], max_items: Optional[int]) -> List[Dict[str, Any]]:
    if max_items is None:
        return records
    return records[:max_items]


def flatten_window_features(windows: np.ndarray, prefix: str = "") -> pd.DataFrame:
    num_windows, _, num_channels = windows.shape
    rows: List[Dict[str, Any]] = []
    for idx in range(num_windows):
        row: Dict[str, Any] = {}
        window = windows[idx]
        for channel_idx in range(num_channels):
            values = window[:, channel_idx]
            base = f"{prefix}ch{channel_idx}"
            row[f"{base}_mean"] = float(values.mean())
            row[f"{base}_std"] = float(values.std(ddof=0))
            row[f"{base}_min"] = float(values.min())
            row[f"{base}_max"] = float(values.max())
            row[f"{base}_energy"] = float(np.square(values).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_sample_indices(n_real: int, n_syn: int, max_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pair_count = min(n_real, n_syn, max_pairs)
    real_indices = rng.choice(n_real, size=pair_count, replace=False)
    syn_indices = rng.choice(n_syn, size=pair_count, replace=False)
    return real_indices, syn_indices
