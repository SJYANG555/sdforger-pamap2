import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.utils import ensure_dir, write_json


FILTERS = {
    "stats_loose_p70": 0.70,
    "stats_medium_p50": 0.50,
    "stats_strict_p35": 0.35,
}
STAT_NAMES = ["mean", "std", "min", "max"]


def load_channel_names(dataset_dir: Path) -> List[str]:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["channel_schema"]["selected_channels"])


def real_stats_from_windows(real_windows: np.ndarray, channel_names: List[str]) -> pd.DataFrame:
    rows = []
    for window in real_windows:
        row = {}
        for channel_idx, channel_name in enumerate(channel_names):
            values = window[:, channel_idx]
            prefix = f"stat_{channel_name}"
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_std"] = float(values.std(ddof=0))
            row[f"{prefix}_min"] = float(values.min())
            row[f"{prefix}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_prompt_stats(
    real_metadata: pd.DataFrame,
    real_windows: np.ndarray,
    channel_names: List[str],
) -> pd.DataFrame:
    expected = [f"stat_{channel}_{stat}" for channel in channel_names for stat in STAT_NAMES]
    if all(column in real_metadata.columns for column in expected):
        return real_metadata
    stats = real_stats_from_windows(real_windows, channel_names)
    return pd.concat([real_metadata.reset_index(drop=True), stats], axis=1)


def build_real_lookup(real_metadata: pd.DataFrame) -> Dict[int, int]:
    return {int(window_id): int(idx) for idx, window_id in enumerate(real_metadata["window_id"].astype(int))}


def compute_stats_scores(
    generated_windows: np.ndarray,
    generated_metadata: pd.DataFrame,
    real_metadata: pd.DataFrame,
    channel_names: List[str],
) -> pd.DataFrame:
    real_lookup = build_real_lookup(real_metadata)
    rows = []
    eps = 1.0e-6
    for generated_idx, generated_row in generated_metadata.iterrows():
        window_id = int(generated_row["window_id"])
        if window_id not in real_lookup or int(generated_idx) >= len(generated_windows):
            continue
        real_idx = real_lookup[window_id]
        target = real_metadata.iloc[real_idx]
        window = generated_windows[int(generated_idx)]

        mean_errors = []
        std_errors = []
        range_errors = []
        max_channel_scores = []
        record = {
            "generated_index": int(generated_idx),
            "candidate_id": int(generated_row.get("candidate_id", generated_idx)),
            "prompt_window_id": window_id,
            "real_index": int(real_idx),
            "activity_name": str(generated_row["activity_name"]),
        }
        for channel_idx, channel_name in enumerate(channel_names):
            values = window[:, channel_idx]
            target_mean = float(target[f"stat_{channel_name}_mean"])
            target_std = max(float(target[f"stat_{channel_name}_std"]), eps)
            target_min = float(target[f"stat_{channel_name}_min"])
            target_max = float(target[f"stat_{channel_name}_max"])

            mean_error = abs(float(values.mean()) - target_mean) / target_std
            std_error = abs(float(values.std(ddof=0)) - target_std) / target_std
            min_error = abs(float(values.min()) - target_min) / target_std
            max_error = abs(float(values.max()) - target_max) / target_std
            range_error = 0.5 * (min_error + max_error)
            channel_score = 0.4 * mean_error + 0.4 * std_error + 0.2 * range_error

            safe_channel = channel_name.replace(".", "_")
            record[f"{safe_channel}_mean_zerr"] = mean_error
            record[f"{safe_channel}_std_relerr"] = std_error
            record[f"{safe_channel}_range_zerr"] = range_error
            record[f"{safe_channel}_stats_score"] = channel_score
            mean_errors.append(mean_error)
            std_errors.append(std_error)
            range_errors.append(range_error)
            max_channel_scores.append(channel_score)

        record["mean_zerr_avg"] = float(np.mean(mean_errors))
        record["std_relerr_avg"] = float(np.mean(std_errors))
        record["range_zerr_avg"] = float(np.mean(range_errors))
        record["stats_score"] = float(
            0.4 * record["mean_zerr_avg"]
            + 0.4 * record["std_relerr_avg"]
            + 0.2 * record["range_zerr_avg"]
        )
        record["max_channel_stats_score"] = float(np.max(max_channel_scores))
        rows.append(record)
    return pd.DataFrame(rows)


def select_by_activity(scores: pd.DataFrame, keep_fraction: float) -> pd.DataFrame:
    selected = []
    for _, group in scores.groupby(scores["activity_name"].astype(str).str.lower(), sort=True):
        keep_count = max(1, int(np.ceil(len(group) * keep_fraction)))
        selected.append(group.sort_values(["stats_score", "max_channel_stats_score"]).head(keep_count))
    if not selected:
        return scores.head(0)
    return pd.concat(selected, ignore_index=True)


def summarize_filter(name: str, selected: pd.DataFrame, total: pd.DataFrame) -> Dict[str, object]:
    total_counts = total["activity_name"].value_counts().sort_index().to_dict()
    selected_counts = selected["activity_name"].value_counts().sort_index().to_dict()
    return {
        "filter_name": name,
        "kept": int(len(selected)),
        "total": int(len(total)),
        "kept_fraction": float(len(selected) / len(total)) if len(total) else 0.0,
        "total_counts": {str(k): int(v) for k, v in total_counts.items()},
        "kept_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "stats_score_mean": float(selected["stats_score"].mean()) if len(selected) else None,
        "stats_score_median": float(selected["stats_score"].median()) if len(selected) else None,
        "stats_score_max": float(selected["stats_score"].max()) if len(selected) else None,
    }


def parse_filter_spec(spec: str) -> Tuple[str, float]:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--filter must be name:fraction, for example stats_p25:0.25")
    name, raw_fraction = parts
    if not name:
        raise argparse.ArgumentTypeError("filter name must not be empty")
    try:
        fraction = float(raw_fraction)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid filter fraction: {raw_fraction}") from exc
    if not (0.0 < fraction <= 1.0):
        raise argparse.ArgumentTypeError("filter fraction must be in (0, 1]")
    return name, fraction


def apply_filters(
    dataset_dir: Path,
    generated_dir: Path,
    output_root: Path,
    run_prefix: str,
    filters: Dict[str, float],
) -> Dict[str, object]:
    output_root = ensure_dir(output_root)
    channel_names = load_channel_names(dataset_dir)
    real_windows = np.load(dataset_dir / "test_windows.npy")
    real_metadata = pd.read_csv(dataset_dir / "test_metadata.csv")
    real_metadata = ensure_prompt_stats(real_metadata, real_windows, channel_names)
    generated_windows = np.load(generated_dir / "generated_windows.npy")
    generated_metadata = pd.read_csv(generated_dir / "generated_embeddings.csv")

    scores = compute_stats_scores(
        generated_windows=generated_windows,
        generated_metadata=generated_metadata,
        real_metadata=real_metadata,
        channel_names=channel_names,
    )
    scores_path = output_root / f"{run_prefix}_stats_consistency_scores.csv"
    scores.to_csv(scores_path, index=False)

    outputs = []
    for filter_name, keep_fraction in filters.items():
        selected = select_by_activity(scores, keep_fraction)
        selected_indices = selected["generated_index"].to_numpy(dtype=int)
        filtered_metadata = generated_metadata.iloc[selected_indices].reset_index(drop=True)
        filtered_windows = generated_windows[selected_indices]

        filtered_dir = ensure_dir(output_root / f"{run_prefix}_{filter_name}")
        filtered_metadata.to_csv(filtered_dir / "generated_embeddings.csv", index=False)
        np.save(filtered_dir / "generated_windows.npy", filtered_windows)
        selected.to_csv(filtered_dir / "stats_filter_selected.csv", index=False)

        summary = {
            "source_generated_dir": str(generated_dir),
            "dataset_dir": str(dataset_dir),
            "run_prefix": run_prefix,
            "keep_fraction_by_activity": keep_fraction,
            "filter": summarize_filter(filter_name, selected, scores),
            "channels": channel_names,
        }
        write_json(filtered_dir / "generation_summary.json", summary)
        outputs.append(
            {
                "filter_name": filter_name,
                "keep_fraction": keep_fraction,
                "output_dir": str(filtered_dir),
                **summary["filter"],
            }
        )

    manifest = {
        "run_prefix": run_prefix,
        "dataset_dir": str(dataset_dir),
        "generated_dir": str(generated_dir),
        "scores_csv": str(scores_path),
        "filters": outputs,
    }
    write_json(output_root / f"{run_prefix}_stats_filter_manifest.json", manifest)
    return manifest


def parse_run_spec(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run must be prefix:dataset_dir:generated_dir")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply stats-consistency filters to generated PAMAP2 windows.")
    parser.add_argument("--output-root", default="outputs/generated_stats_filter")
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument(
        "--filter",
        action="append",
        type=parse_filter_spec,
        default=None,
        help="Stats-consistency filter as name:fraction. May be repeated. Defaults to legacy p70/p50/p35.",
    )
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_root))
    filters = dict(args.filter) if args.filter else FILTERS
    manifests = []
    for run_prefix, dataset_dir, generated_dir in args.run:
        manifest = apply_filters(
            dataset_dir=Path(dataset_dir),
            generated_dir=Path(generated_dir),
            output_root=output_root,
            run_prefix=run_prefix,
            filters=filters,
        )
        manifests.append(manifest)
        print(f"wrote {run_prefix}")
    write_json(output_root / "stats_filter_manifest.json", {"runs": manifests})
    print(json.dumps({"output_root": str(output_root.resolve()), "num_runs": len(manifests)}, indent=2))


if __name__ == "__main__":
    main()
