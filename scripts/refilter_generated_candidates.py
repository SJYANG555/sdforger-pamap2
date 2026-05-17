import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config
from pamap2_forger.reconstruction import (
    build_reconstruction_metadata,
    decode_embeddings_to_windows,
    filter_generated_embeddings,
)
from pamap2_forger.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Refilter already parsed generated PAMAP2 candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True, help="Directory containing parsed_generation_candidates.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-embeddings", default=None)
    parser.add_argument("--iqr-factor", type=float, default=3.0)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = Path(config.data.output_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = input_dir / "parsed_generation_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)

    reference_path = Path(args.reference_embeddings or dataset_dir / "train_embeddings.csv")
    reference_frame = pd.read_csv(reference_path)
    generated_frame = pd.read_csv(candidates_path)

    filtered_frame, filter_stats, filter_debug = filter_generated_embeddings(
        generated_frame=generated_frame,
        reference_frame=reference_frame,
        iqr_factor=args.iqr_factor,
        deduplicate=not args.keep_duplicates,
    )
    windows = decode_embeddings_to_windows(filtered_frame, dataset_dir / config.embedding.reducer_artifact_name)

    filtered_frame.to_csv(output_dir / "generated_embeddings.csv", index=False)
    np.save(output_dir / "generated_windows.npy", windows)
    filter_debug.to_csv(output_dir / "generation_filter_debug.csv", index=False)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    source_model_path = None
    source_summary = input_dir / "generation_summary.json"
    if source_summary.exists():
        source_model_path = json.loads(source_summary.read_text(encoding="utf-8")).get("source_model_path")

    summary = {
        **build_reconstruction_metadata(
            filtered_frame,
            source_model_path or "unknown",
            dataset_dir / config.embedding.reducer_artifact_name,
            filter_stats,
        ),
        "source_generation_dir": str(input_dir),
        "reference_embeddings": str(reference_path),
        "iqr_factor": args.iqr_factor,
        "deduplicate": not args.keep_duplicates,
        "num_parsed_candidates": int(len(generated_frame)),
    }
    write_json(output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
