import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

CONTEXT_COLUMNS = {"window_id", "subject_id", "activity_id", "activity_name"}
NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
VALUE_NAME_PATTERN = re.compile(r"value_\d+")

MOTION_HINTS = {
    "walking": "periodic locomotion",
    "running": "high-energy periodic locomotion",
    "cycling": "cyclic lower-body motion",
    "sitting": "low-motion posture",
    "standing": "quiet upright posture",
}


def embeddings_to_text(
    row: Dict[str, Any],
    columns: List[str],
    eos_token: str,
    permute: bool = True,
    text_template: str = "fim_template_textual_encoding",
    input_tokens_precision: int = 4,
    prompt_stat_columns: Optional[List[str]] = None,
    prompt_stats_precision: int = 3,
) -> str:
    ordered_columns = list(columns)
    if permute:
        random.shuffle(ordered_columns)

    prompt_stat_columns = prompt_stat_columns or []
    categorical_columns = [col for col in ordered_columns if col in CONTEXT_COLUMNS or not _is_float_like(row[col])]
    numeric_columns = [
        col
        for col in ordered_columns
        if col not in CONTEXT_COLUMNS and col not in prompt_stat_columns and _is_float_like(row[col])
    ]

    if text_template in {"structured_v2", "compact_values_v2"}:
        stable_numeric_columns = sorted(
            [column for column in numeric_columns if column.startswith("value_")],
            key=_numeric_sort_key,
        )
        prompt = build_generation_prompt(
            row=row,
            numeric_columns=stable_numeric_columns,
            text_template=text_template,
            prompt_stat_columns=prompt_stat_columns,
            prompt_stats_precision=prompt_stats_precision,
        )
        if text_template == "compact_values_v2":
            target = ", ".join([_format_value(row[col], input_tokens_precision) for col in stable_numeric_columns])
            return f"{prompt}{target} [end]{eos_token}"
        target = "; ".join([f"{col} = {_format_value(row[col], input_tokens_precision)}" for col in stable_numeric_columns])
        return f"{prompt}{target}; [end]{eos_token}"

    if text_template == "base_template":
        body = ", ".join(
            [f"{col} is {_format_value(row[col], input_tokens_precision)}" for col in ordered_columns]
        )
        return body + eos_token

    if text_template == "fim_template":
        text_input = ", ".join([f"{col} is [blank]" for col in ordered_columns])
        text_target = " ".join(
            [f"{_format_value(row[col], input_tokens_precision)} [answer]" for col in ordered_columns]
        )
        return f"Input: {text_input} [sep] Target: {text_target}{eos_token}"

    condition = ", ".join(
        [f"{col} is {row[col]}" for col in categorical_columns]
    )
    blank_input = ", ".join([f"{col} is [blank]" for col in numeric_columns])
    target = " ".join(
        [f"{_format_value(row[col], input_tokens_precision)} [answer]" for col in numeric_columns]
    )
    return (
        f"Condition: {condition} [sep] Input: {blank_input} [sep] Target: {target}{eos_token}"
    )


def dataframe_to_text_records(
    frame: pd.DataFrame,
    eos_token: str,
    permute: bool,
    text_template: str,
    input_tokens_precision: int,
    prompt_stat_columns: Optional[List[str]] = None,
    prompt_stats_precision: int = 3,
) -> List[Dict[str, Any]]:
    columns = frame.columns.tolist()
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        text = embeddings_to_text(
            row=row,
            columns=columns,
            eos_token=eos_token,
            permute=permute,
            text_template=text_template,
            input_tokens_precision=input_tokens_precision,
            prompt_stat_columns=prompt_stat_columns,
            prompt_stats_precision=prompt_stats_precision,
        )
        records.append({"text": text, **row})
    return records


def build_generation_prompt(
    row: Dict[str, Any],
    numeric_columns: Iterable[str],
    text_template: str = "fim_template_textual_encoding",
    prompt_stat_columns: Optional[Iterable[str]] = None,
    prompt_stats_precision: int = 3,
) -> str:
    numeric_columns = list(numeric_columns)
    prompt_stat_columns = list(prompt_stat_columns or [])
    if text_template in {"structured_v2", "compact_values_v2"}:
        activity_name = str(row["activity_name"])
        motion_hint = MOTION_HINTS.get(activity_name, "human motion")
        stats_summary = _format_stats_summary(row, prompt_stat_columns, prompt_stats_precision)
        if text_template == "compact_values_v2":
            stats_segment = f"stats={stats_summary}; " if stats_summary else ""
            return (
                f"Condition: activity={activity_name}; "
                f"activity_id={int(row['activity_id'])}; "
                f"subject_id={int(row['subject_id'])}; "
                f"window_id={int(row['window_id'])}; "
                f"{stats_segment}"
                f"hint={motion_hint}; "
                f"task=PAMAP2_embedding_{len(numeric_columns)}_values\n"
                "Target values:\n"
            )
        stats_block = f"window_stats = {stats_summary}\n" if stats_summary else ""
        return (
            "Condition:\n"
            f"activity_name = {activity_name}\n"
            f"activity_id = {int(row['activity_id'])}\n"
            f"subject_id = {int(row['subject_id'])}\n"
            f"window_id = {int(row['window_id'])}\n"
            f"{stats_block}"
            "task = generate a standardized PAMAP2 18-channel 256-step motion embedding\n"
            f"motion_hint = {motion_hint}\n\n"
            f"Input:\nvalues = [blank] x {len(numeric_columns)}\n\n"
            "Target:\n"
        )

    return (
        f"Condition: activity_name is {row['activity_name']}, "
        f"activity_id is {int(row['activity_id'])}, "
        f"window_id is {int(row['window_id'])}, "
        f"subject_id is {int(row['subject_id'])} [sep] Input: "
        + ", ".join([f"{column} is [blank]" for column in numeric_columns])
        + " [sep] Target: "
    )


def extract_target_tail(text: str) -> Tuple[str, str]:
    normalized = text.replace("\n", " ").replace("\r", " ")
    match = None
    for candidate in re.finditer(r"(?:\[sep\]\s*)?Target(?:\s+values)?\s*:", normalized, flags=re.IGNORECASE):
        match = candidate
    if match is None:
        return normalized, "target_marker_missing"
    return normalized[match.end() :], "ok"


def parse_generated_text_with_debug(
    text: str,
    numeric_columns: Iterable[str],
    categorical_seed: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    result = dict(categorical_seed)
    numeric_columns = list(numeric_columns)
    tail, marker_status = extract_target_tail(text)

    named_values: Dict[str, float] = {}
    for match in re.finditer(
        r"\b(value_\d+)\s*(?:=|is)\s*("
        r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)",
        tail,
        flags=re.IGNORECASE,
    ):
        try:
            named_values[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    if named_values:
        for column in numeric_columns:
            if column in named_values:
                result[column] = named_values[column]
        parsed_count = sum(1 for column in numeric_columns if column in result)
        missing_columns = [column for column in numeric_columns if column not in result]
        clean_tail = VALUE_NAME_PATTERN.sub("VALUE", tail)
        stripped_tail = clean_tail.rstrip()
        debug = {
            "parse_status": "ok" if parsed_count > 0 else "empty_numeric_parse",
            "marker_status": marker_status,
            "parse_mode": "named_values",
            "expected_numeric_count": len(numeric_columns),
            "parsed_numeric_count": parsed_count,
            "missing_numeric_count": max(len(numeric_columns) - parsed_count, 0),
            "missing_columns": missing_columns,
            "extracted_token_count": len(named_values),
            "last_extracted_token": None,
            "tail_truncation_hint": bool(stripped_tail) and stripped_tail[-1] in {"-", ".", "[", ","},
            "tail_last_char": stripped_tail[-1] if stripped_tail else "",
            "target_tail_preview": clean_tail[:500],
        }
        return result, debug

    # Avoid capturing indices from tokens like value_17.
    clean_tail = VALUE_NAME_PATTERN.sub("VALUE", tail)
    clean_tail = clean_tail.replace("<|endoftext|>", " ").replace("</s>", " ")
    clean_tail = clean_tail.replace("[answer]", " ")
    extracted_tokens = NUMERIC_PATTERN.findall(clean_tail)
    values: List[float] = []
    for token in extracted_tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue

    parsed_count = min(len(values), len(numeric_columns))
    for index, column in enumerate(numeric_columns):
        if index < len(values):
            result[column] = values[index]
    missing_columns = numeric_columns[parsed_count:]
    stripped_tail = clean_tail.rstrip()
    truncation_hint = bool(stripped_tail) and stripped_tail[-1] in {"-", ".", "[", ","}
    debug = {
        "parse_status": "ok" if parsed_count > 0 else "empty_numeric_parse",
        "marker_status": marker_status,
        "parse_mode": "sequential_numbers",
        "expected_numeric_count": len(numeric_columns),
        "parsed_numeric_count": parsed_count,
        "missing_numeric_count": max(len(numeric_columns) - parsed_count, 0),
        "missing_columns": missing_columns,
        "extracted_token_count": len(extracted_tokens),
        "last_extracted_token": extracted_tokens[-1] if extracted_tokens else None,
        "tail_truncation_hint": truncation_hint,
        "tail_last_char": stripped_tail[-1] if stripped_tail else "",
        "target_tail_preview": clean_tail[:500],
    }
    return result, debug


def parse_generated_text(
    text: str,
    numeric_columns: Iterable[str],
    categorical_seed: Dict[str, Any],
) -> Dict[str, Any]:
    parsed, _ = parse_generated_text_with_debug(text, numeric_columns, categorical_seed)
    return parsed


def _is_float_like(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _format_value(value: Any, precision: int) -> str:
    if _is_float_like(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def _format_stats_summary(row: Dict[str, Any], columns: Iterable[str], precision: int) -> str:
    parts: List[str] = []
    for column in columns:
        if column not in row or not _is_float_like(row[column]):
            continue
        parts.append(f"{column}={_format_value(row[column], precision)}")
    return "; ".join(parts)


def _numeric_sort_key(column: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", column)
    if match:
        return int(match.group(1)), column
    return 10**9, column
