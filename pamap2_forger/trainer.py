import json
import os
import inspect
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from pamap2_forger.config import TrainingConfig
from pamap2_forger.utils import ensure_dir, read_jsonl, set_seed, write_json


def load_text_dataset(path: Union[str, Path]):
    from datasets import Dataset

    return Dataset.from_list(read_jsonl(path))


def disable_deepspeed_runtime() -> None:
    # Triton training here should use standard Transformers/Accelerate + PEFT.
    # Explicitly disable any inherited DeepSpeed runtime flags from the environment.
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ.pop("ACCELERATE_DEEPSPEED_CONFIG_FILE", None)
    os.environ.pop("DEEPSPEED_CONFIG_FILE", None)


def patch_accelerate_extract_model_from_parallel() -> None:
    """
    Accelerate versions in this environment may unconditionally import DeepSpeed
    inside extract_model_from_parallel(), even when DeepSpeed is not configured.
    Replace that helper with a minimal unwrap implementation that only handles
    standard torch wrappers used in our single-GPU training path.
    """
    try:
        import torch
        import accelerate.accelerator as accelerator_module
    except ImportError:
        return

    utils_other_module = None
    try:
        import accelerate.utils.other as utils_other_module
    except ImportError:
        utils_other_module = None

    wrappers = []
    for name in ("DataParallel", "DistributedDataParallel"):
        cls = getattr(torch.nn.parallel, name, None)
        if cls is not None:
            wrappers.append(cls)

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        wrappers.append(FullyShardedDataParallel)
    except Exception:
        pass

    wrapper_types = tuple(wrappers)

    def safe_extract_model_from_parallel(
        model,
        keep_fp32_wrapper: bool = True,
        keep_torch_compile: bool = True,
        recursive: bool = False,
    ):
        unwrapped = model
        while wrapper_types and isinstance(unwrapped, wrapper_types):
            unwrapped = unwrapped.module
        if not keep_torch_compile and hasattr(unwrapped, "_orig_mod"):
            return unwrapped._orig_mod
        return unwrapped

    accelerator_module.extract_model_from_parallel = safe_extract_model_from_parallel
    if utils_other_module is not None:
        utils_other_module.extract_model_from_parallel = safe_extract_model_from_parallel


def resolve_torch_dtype(use_fp16: bool, use_bf16: bool):
    import torch

    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return None


def resolve_tokenizer_and_model(config: TrainingConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token

    torch_dtype = resolve_torch_dtype(config.use_fp16, config.use_bf16)
    model_load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_load_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id_or_path,
        **model_load_kwargs,
    )
    model.resize_token_embeddings(len(tokenizer))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.peft.enabled:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_target_modules = config.peft.target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        peft_cfg = LoraConfig(
            task_type=getattr(TaskType, config.peft.task_type),
            r=config.peft.r,
            lora_alpha=config.peft.alpha,
            lora_dropout=config.peft.dropout,
            bias=config.peft.bias,
            target_modules=peft_target_modules,
        )
        model = get_peft_model(model, peft_cfg)
        if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def train_language_model(
    train_jsonl: Union[str, Path],
    val_jsonl: Union[str, Path],
    config: TrainingConfig,
) -> str:
    disable_deepspeed_runtime()
    patch_accelerate_extract_model_from_parallel()
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")

    model, tokenizer = resolve_tokenizer_and_model(config)
    train_dataset = load_text_dataset(train_jsonl)
    val_dataset = load_text_dataset(val_jsonl)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=config.max_seq_length,
        )

    train_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_batch, batched=True, remove_columns=val_dataset.column_names)

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "logging_dir": str(logs_dir),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": config.use_fp16,
        "bf16": config.use_bf16,
        "dataloader_num_workers": config.dataloader_num_workers,
        "report_to": config.report_to,
        "seed": config.seed,
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"
    if "deepspeed" in training_args_signature.parameters:
        training_args_kwargs["deepspeed"] = None

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=CausalLMCollator(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()

    latest_dir = ensure_dir(output_dir / "latest")
    best_dir = ensure_dir(output_dir / "best")
    trainer.save_model(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir))

    if config.peft.enabled:
        trainer.model.save_pretrained(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
    else:
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

    state_path = output_dir / "trainer_state.json"
    log_history_path = output_dir / "training_log_history.csv"
    if state_path.exists():
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        pd.DataFrame(state_payload.get("log_history", [])).to_csv(log_history_path, index=False)

    write_json(
        output_dir / "training_summary.json",
        {
            "model_id_or_path": config.model_id_or_path,
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "peft_enabled": config.peft.enabled,
            "best_checkpoint": str(best_dir.resolve()),
            "latest_checkpoint": str(latest_dir.resolve()),
        },
    )
    return str(best_dir.resolve())
