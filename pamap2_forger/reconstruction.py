from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pamap2_forger.embeddings import WindowReducer
from pamap2_forger.utils import get_numeric_embedding_columns


def filter_generated_embeddings(
    generated_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    iqr_factor: float = 3.0,
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    numeric_columns = get_numeric_embedding_columns(reference_frame)
    frame = generated_frame.copy()
    before = len(frame)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = np.arange(len(frame))

    debug_frame = frame[["candidate_id"]].copy()

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_counts = frame[numeric_columns].isna().sum(axis=1)
    debug_frame["missing_numeric_fields"] = missing_counts.astype(int)
    debug_frame["missing_columns"] = frame[numeric_columns].isna().apply(
        lambda row: [column for column, is_missing in row.items() if bool(is_missing)],
        axis=1,
    )
    debug_frame["parsed_numeric_fields"] = len(numeric_columns) - debug_frame["missing_numeric_fields"]
    debug_frame["filter_reason"] = np.where(missing_counts > 0, "missing_fields", "passed_missing")
    frame_after_missing = frame[missing_counts == 0].copy()
    after_missing = len(frame_after_missing)

    if deduplicate:
        duplicated_mask = frame_after_missing.duplicated(subset=numeric_columns, keep="first")
        duplicated_ids = set(frame_after_missing.loc[duplicated_mask, "candidate_id"].tolist())
        debug_frame.loc[debug_frame["candidate_id"].isin(duplicated_ids), "filter_reason"] = "duplicate"
        frame_after_dedup = frame_after_missing.loc[~duplicated_mask].copy()
    else:
        frame_after_dedup = frame_after_missing.copy()
    after_dedup = len(frame_after_dedup)

    reference_norms = np.linalg.norm(reference_frame[numeric_columns].to_numpy(dtype=np.float32), axis=1)
    if len(reference_norms) == 0:
        debug_frame.loc[debug_frame["filter_reason"] == "passed_missing", "filter_reason"] = "kept"
        return frame_after_dedup.reset_index(drop=True), {
            "before_filter": before,
            "after_missing_filter": after_missing,
            "after_dedup_filter": after_dedup,
            "after_norm_filter": after_dedup,
        }, debug_frame

    q1, q3 = np.percentile(reference_norms, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr

    frame_norms = np.linalg.norm(frame_after_dedup[numeric_columns].to_numpy(dtype=np.float32), axis=1)
    mask = (frame_norms >= lower) & (frame_norms <= upper)
    kept_ids = set(frame_after_dedup.loc[mask, "candidate_id"].tolist())
    norm_rejected_ids = set(frame_after_dedup.loc[~mask, "candidate_id"].tolist())
    debug_frame.loc[debug_frame["candidate_id"].isin(norm_rejected_ids), "filter_reason"] = "norm_filter"
    debug_frame.loc[debug_frame["candidate_id"].isin(kept_ids), "filter_reason"] = "kept"

    frame_final = frame_after_dedup.loc[mask].reset_index(drop=True)
    return frame_final, {
        "before_filter": before,
        "after_missing_filter": after_missing,
        "after_dedup_filter": after_dedup,
        "after_norm_filter": len(frame_final),
    }, debug_frame


def decode_embeddings_to_windows(
    embedding_frame: pd.DataFrame,
    reducer_path: Union[str, Path],
) -> np.ndarray:
    reducer = WindowReducer.load(reducer_path)
    ordered_columns = ["window_id", "subject_id", "activity_id", "activity_name"] + get_numeric_embedding_columns(
        embedding_frame
    )
    frame = embedding_frame[ordered_columns].copy()
    return reducer.inverse_transform(frame)


def build_reconstruction_metadata(
    embedding_frame: pd.DataFrame,
    source_model_path: Union[str, Path],
    reducer_path: Union[str, Path],
    filter_stats: Dict[str, int],
) -> Dict[str, object]:
    return {
        "num_generated_embeddings": int(len(embedding_frame)),
        "source_model_path": str(source_model_path),
        "reducer_path": str(reducer_path),
        "filter_stats": filter_stats,
        "activities": sorted(embedding_frame["activity_name"].dropna().unique().tolist()),
    }
