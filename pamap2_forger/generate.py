from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from pamap2_forger.config import GenerationConfig
from pamap2_forger.reconstruction import (
    build_reconstruction_metadata,
    decode_embeddings_to_windows,
    filter_generated_embeddings,
)
from pamap2_forger.text import build_generation_prompt, parse_generated_text_with_debug
from pamap2_forger.utils import (
    get_numeric_embedding_columns,
    maybe_limit_records,
    resolve_device,
    write_json,
    write_jsonl,
)


def load_causal_lm(model_path: Union[str, Path]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_path = Path(model_path) / "adapter_config.json"
    torch_dtype = torch.float16 if resolve_device(prefer_cuda=True) == "cuda" else None
    model_load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_load_kwargs["torch_dtype"] = torch_dtype
    if adapter_config_path.exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs,
        )

    device = resolve_device(prefer_cuda=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def build_generation_prompts(
    metadata_path: Union[str, Path],
    numeric_columns: List[str],
    text_template: str = "fim_template_textual_encoding",
    max_prompts: int = None,
) -> List[Dict[str, Any]]:
    metadata = pd.read_csv(metadata_path)
    records = metadata.to_dict(orient="records")
    prompts: List[Dict[str, Any]] = []
    for row in maybe_limit_records(records, max_prompts):
        prompt = build_generation_prompt(row=row, numeric_columns=numeric_columns, text_template=text_template)
        prompts.append({"prompt": prompt, **row})
    return prompts


def generate_texts(
    model,
    tokenizer,
    device: str,
    prompts: List[Dict[str, Any]],
    config: GenerationConfig,
    numeric_column_count: int,
) -> List[Dict[str, Any]]:
    import torch

    outputs: List[Dict[str, Any]] = []
    target_token_budget = numeric_column_count * 8
    requested_max_new_tokens = max(config.max_new_tokens, target_token_budget)
    requested_min_new_tokens = max(config.min_new_tokens, numeric_column_count * 6)
    model_token_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    for start in range(0, len(prompts), config.batch_size):
        batch = prompts[start : start + config.batch_size]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        max_prompt_length = int(max(prompt_lengths)) if prompt_lengths else int(encoded["input_ids"].shape[1])
        effective_max_new_tokens = requested_max_new_tokens
        if model_token_limit is not None and model_token_limit < 10**6:
            effective_max_new_tokens = min(
                effective_max_new_tokens,
                max(int(model_token_limit) - max_prompt_length - 1, 1),
            )
        effective_min_new_tokens = min(requested_min_new_tokens, effective_max_new_tokens)
        generation_kwargs = {
            "max_new_tokens": effective_max_new_tokens,
            "min_new_tokens": effective_min_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.do_sample:
            generation_kwargs.update(
                {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "do_sample": True,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "do_sample": False,
                }
            )
        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=False)
        for row_idx, (prompt_meta, text) in enumerate(zip(batch, texts)):
            output_ids = generated[row_idx]
            generated_ids = output_ids.tolist()
            ended_with_eos = bool(generated_ids) and tokenizer.eos_token_id is not None and generated_ids[-1] == tokenizer.eos_token_id
            outputs.append(
                {
                    **prompt_meta,
                    "generated_text": text,
                    "effective_max_new_tokens": effective_max_new_tokens,
                    "effective_min_new_tokens": effective_min_new_tokens,
                    "do_sample": config.do_sample,
                    "prompt_token_count": int(prompt_lengths[row_idx]),
                    "generated_total_token_count": int(len(generated_ids)),
                    "generated_new_token_count": int(len(generated_ids) - prompt_lengths[row_idx]),
                    "ended_with_eos": ended_with_eos,
                }
            )
    return outputs


def parse_generated_outputs(outputs: List[Dict[str, Any]], numeric_columns: List[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    for candidate_id, item in enumerate(outputs):
        parsed, parse_debug = parse_generated_text_with_debug(
            text=item["generated_text"],
            numeric_columns=numeric_columns,
            categorical_seed={
                "candidate_id": candidate_id,
                "window_id": int(item["window_id"]),
                "subject_id": int(item["subject_id"]),
                "activity_id": int(item["activity_id"]),
                "activity_name": item["activity_name"],
            },
        )
        rows.append(parsed)
        debug_records.append(
            {
                "candidate_id": candidate_id,
                "window_id": int(item["window_id"]),
                "subject_id": int(item["subject_id"]),
                "activity_id": int(item["activity_id"]),
                "activity_name": item["activity_name"],
                "prompt": item["prompt"],
                "raw_generated_text": item["generated_text"],
                "prompt_token_count": item.get("prompt_token_count"),
                "generated_total_token_count": item.get("generated_total_token_count"),
                "generated_new_token_count": item.get("generated_new_token_count"),
                "ended_with_eos": item.get("ended_with_eos"),
                "effective_max_new_tokens": item.get("effective_max_new_tokens"),
                "effective_min_new_tokens": item.get("effective_min_new_tokens"),
                "do_sample": item.get("do_sample"),
                **parse_debug,
            }
        )
    frame = pd.DataFrame(rows)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = list(range(len(frame)))
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = None
    ordered_columns = ["candidate_id", "window_id", "subject_id", "activity_id", "activity_name"] + numeric_columns
    return frame[ordered_columns], debug_records


def generate_synthetic_dataset(
    model_path: Union[str, Path],
    reference_embeddings_path: Union[str, Path],
    prompt_metadata_path: Union[str, Path],
    reducer_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: GenerationConfig,
    text_template: str = "fim_template_textual_encoding",
) -> Dict[str, object]:
    reference_embeddings = pd.read_csv(reference_embeddings_path)
    numeric_columns = get_numeric_embedding_columns(reference_embeddings)
    prompts = build_generation_prompts(
        metadata_path=prompt_metadata_path,
        numeric_columns=numeric_columns,
        text_template=text_template,
        max_prompts=config.max_prompts,
    )
    model, tokenizer, device = load_causal_lm(model_path)
    outputs = generate_texts(model, tokenizer, device, prompts, config, numeric_column_count=len(numeric_columns))
    effective_max_new_tokens = outputs[0]["effective_max_new_tokens"] if outputs else max(config.max_new_tokens, len(numeric_columns) * 8)
    effective_min_new_tokens = outputs[0]["effective_min_new_tokens"] if outputs else max(config.min_new_tokens, len(numeric_columns) * 6)
    generated_frame, debug_records = parse_generated_outputs(outputs, numeric_columns)
    filtered_frame, filter_stats, filter_debug = filter_generated_embeddings(
        generated_frame=generated_frame,
        reference_frame=reference_embeddings,
        iqr_factor=config.norm_filter_iqr_factor,
        deduplicate=config.deduplicate,
    )
    duplicate_candidate_count = max(filter_stats["after_missing_filter"] - filter_stats["after_dedup_filter"], 0)
    unique_candidate_count = int(filter_stats["after_dedup_filter"])
    duplicate_ratio = (
        float(duplicate_candidate_count / filter_stats["after_missing_filter"])
        if filter_stats["after_missing_filter"] > 0
        else None
    )
    unique_ratio = (
        float(unique_candidate_count / filter_stats["after_missing_filter"])
        if filter_stats["after_missing_filter"] > 0
        else None
    )
    windows = decode_embeddings_to_windows(filtered_frame, reducer_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_frame.to_csv(output_dir / "generated_embeddings.csv", index=False)
    import numpy as np

    np.save(output_dir / "generated_windows.npy", windows)
    filter_reason_map = {
        int(row["candidate_id"]): row["filter_reason"]
        for row in filter_debug.to_dict(orient="records")
    }
    for record in debug_records:
        record["filter_reason"] = filter_reason_map.get(record["candidate_id"], "unknown")
        record["parse_success"] = record.get("parsed_numeric_count", 0) > 0
        if record["filter_reason"] == "missing_fields":
            debug_row = filter_debug.loc[filter_debug["candidate_id"] == record["candidate_id"]]
            if not debug_row.empty:
                record["missing_columns"] = debug_row.iloc[0]["missing_columns"]
    write_jsonl(output_dir / "raw_generations.jsonl", debug_records)
    generated_frame.to_csv(output_dir / "parsed_generation_candidates.csv", index=False)
    filter_debug.to_csv(output_dir / "generation_filter_debug.csv", index=False)
    write_json(
        output_dir / "generation_summary.json",
        {
            **build_reconstruction_metadata(filtered_frame, model_path, reducer_path, filter_stats),
            "num_prompts": len(prompts),
            "num_parsed_candidates": int(len(generated_frame)),
            "num_parse_success": int(sum(1 for record in debug_records if record.get("parse_success"))),
            "effective_max_new_tokens": int(effective_max_new_tokens),
            "effective_min_new_tokens": int(effective_min_new_tokens),
            "do_sample": bool(config.do_sample),
            "duplicate_candidate_count": int(duplicate_candidate_count),
            "unique_candidate_count": int(unique_candidate_count),
            "duplicate_ratio": duplicate_ratio,
            "unique_ratio": unique_ratio,
        },
    )
    write_json(
        output_dir / "generation_prompts_summary.json",
        {
            "prompt_metadata_path": str(prompt_metadata_path),
            "reference_embeddings_path": str(reference_embeddings_path),
            "num_prompts": len(prompts),
            "num_generated_rows_before_filter": len(generated_frame),
            "num_generated_rows_after_filter": len(filtered_frame),
            "raw_generations_path": str((output_dir / "raw_generations.jsonl").resolve()),
        },
    )
    return {
        "num_generated": int(len(filtered_frame)),
        "output_dir": str(output_dir.resolve()),
    }
