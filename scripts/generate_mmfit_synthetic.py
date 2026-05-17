import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config  # noqa: E402
from pamap2_forger.generate import generate_texts, load_causal_lm, parse_generated_outputs  # noqa: E402
from pamap2_forger.reconstruction import (  # noqa: E402
    build_reconstruction_metadata,
    decode_embeddings_to_windows,
    filter_generated_embeddings,
)
from pamap2_forger.utils import get_numeric_embedding_columns, maybe_limit_records, write_json, write_jsonl  # noqa: E402


def build_mmfit_generation_prompt(row: Dict[str, Any], numeric_columns: Iterable[str], channel_names: List[str]) -> str:
    numeric_columns = list(numeric_columns)
    channel_summary = ",".join(channel_names)
    return (
        "Condition:\n"
        "dataset = MM-Fit\n"
        f"activity_name = {row['activity_name']}\n"
        f"activity_id = {int(row['activity_id'])}\n"
        f"subject_id = {int(row['subject_id'])}\n"
        f"window_id = {int(row['window_id'])}\n"
        f"sensor_channels = {channel_summary}\n"
        "task = generate a standardized MM-Fit smartwatch exercise embedding\n\n"
        f"Input:\nvalues = [blank] x {len(numeric_columns)}\n\n"
        "Target:\n"
    )


def build_mmfit_generation_prompts(
    metadata_path: Path,
    numeric_columns: List[str],
    channel_names: List[str],
    max_prompts: int | None,
) -> List[Dict[str, Any]]:
    metadata = pd.read_csv(metadata_path)
    prompts: List[Dict[str, Any]] = []
    for row in maybe_limit_records(metadata.to_dict(orient="records"), max_prompts):
        prompts.append(
            {
                "prompt": build_mmfit_generation_prompt(row, numeric_columns, channel_names),
                **row,
            }
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MM-Fit synthetic windows from a trained LM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model-path", required=True, help="Fine-tuned model or adapter path.")
    parser.add_argument("--prompt-metadata", default=None, help="Optional metadata CSV to condition generation.")
    parser.add_argument("--reference-embeddings", default=None, help="Optional embeddings CSV for norm filtering.")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated embeddings and decoded windows.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = Path(config.data.output_dir)
    output_dir = Path(args.output_dir or config.generation.output_dir)
    prompt_metadata = Path(args.prompt_metadata or dataset_dir / f"{config.generation.prompt_split}_metadata.csv")
    reference_embeddings_path = Path(args.reference_embeddings or dataset_dir / "train_embeddings.csv")
    reducer_path = dataset_dir / config.embedding.reducer_artifact_name

    reference_embeddings = pd.read_csv(reference_embeddings_path)
    numeric_columns = get_numeric_embedding_columns(reference_embeddings)
    reconstruction_metadata = json.loads((dataset_dir / "reconstruction_metadata.json").read_text(encoding="utf-8"))
    channel_names = list(reconstruction_metadata["channel_names"])

    prompts = build_mmfit_generation_prompts(
        metadata_path=prompt_metadata,
        numeric_columns=numeric_columns,
        channel_names=channel_names,
        max_prompts=config.generation.max_prompts,
    )
    model, tokenizer, device = load_causal_lm(args.model_path)
    outputs = generate_texts(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        config=config.generation,
        numeric_column_count=len(numeric_columns),
    )
    generated_frame, debug_records = parse_generated_outputs(outputs, numeric_columns)
    filtered_frame, filter_stats, filter_debug = filter_generated_embeddings(
        generated_frame=generated_frame,
        reference_frame=reference_embeddings,
        iqr_factor=config.generation.norm_filter_iqr_factor,
        deduplicate=config.generation.deduplicate,
    )
    windows = decode_embeddings_to_windows(filtered_frame, reducer_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    filtered_frame.to_csv(output_dir / "generated_embeddings.csv", index=False)
    np.save(output_dir / "generated_windows.npy", windows)
    generated_frame.to_csv(output_dir / "parsed_generation_candidates.csv", index=False)
    filter_debug.to_csv(output_dir / "generation_filter_debug.csv", index=False)

    filter_reason_map = {
        int(row["candidate_id"]): row["filter_reason"]
        for row in filter_debug.to_dict(orient="records")
    }
    for record in debug_records:
        record["filter_reason"] = filter_reason_map.get(record["candidate_id"], "unknown")
        record["parse_success"] = record.get("parsed_numeric_count", 0) > 0
    write_jsonl(output_dir / "raw_generations.jsonl", debug_records)

    duplicate_candidate_count = max(filter_stats["after_missing_filter"] - filter_stats["after_dedup_filter"], 0)
    duplicate_ratio = (
        float(duplicate_candidate_count / filter_stats["after_missing_filter"])
        if filter_stats["after_missing_filter"] > 0
        else None
    )
    write_json(
        output_dir / "generation_summary.json",
        {
            **build_reconstruction_metadata(filtered_frame, args.model_path, reducer_path, filter_stats),
            "dataset": "MM-Fit",
            "prompt_metadata_path": str(prompt_metadata.resolve()),
            "reference_embeddings_path": str(reference_embeddings_path.resolve()),
            "num_prompts": int(len(prompts)),
            "num_parsed_candidates": int(len(generated_frame)),
            "num_parse_success": int(sum(1 for record in debug_records if record.get("parse_success"))),
            "duplicate_candidate_count": int(duplicate_candidate_count),
            "duplicate_ratio": duplicate_ratio,
        },
    )
    print(json.dumps({"num_generated": int(len(filtered_frame)), "output_dir": str(output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

