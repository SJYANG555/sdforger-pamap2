from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    preprocessed_dir: str
    output_dir: str
    selected_channels: Optional[List[str]] = None
    selected_channel_prefixes: Optional[List[str]] = None
    split_strategy: str = "subject"
    train_subjects: Optional[List[int]] = None
    val_subjects: Optional[List[int]] = None
    test_subjects: Optional[List[int]] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class EmbeddingConfig:
    method: str = "fastica"
    n_components: int = 3
    variance_explained: float = 0.7
    standardization: str = "legacy_timepoint"
    fastica_max_iter: int = 1000
    fastica_tol: float = 1.0e-4
    input_tokens_precision: int = 4
    permute_columns: bool = True
    text_template: str = "fim_template_textual_encoding"
    include_context_columns: bool = True
    include_window_stats_prompt: bool = False
    window_stats_source: str = "raw_window"
    window_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    window_stats_precision: int = 3
    reducer_artifact_name: str = "reducer.pkl"
    scaler_artifact_name: str = "channel_scaler.json"


@dataclass
class PeftConfig:
    enabled: bool = False
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    model_id_or_path: str = "openai-community/gpt2"
    output_dir: str = "outputs/checkpoints/gpt2"
    learning_rate: float = 8e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    max_seq_length: int = 1024
    use_fp16: bool = False
    use_bf16: bool = False
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 5
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    peft: PeftConfig = field(default_factory=PeftConfig)


@dataclass
class GenerationConfig:
    prompt_split: str = "test"
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    num_return_sequences: int = 1
    temperature: float = 1.1
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    batch_size: int = 8
    max_prompts: Optional[int] = None
    auto_token_budget: bool = True
    prompt_stats_source: str = "metadata"
    prompt_stats_seed: int = 42
    norm_filter_iqr_factor: float = 3.0
    deduplicate: bool = True
    output_dir: str = "outputs/generated/default"


@dataclass
class SimilarityConfig:
    max_lag: int = 32
    max_samples_per_activity: int = 128
    dtw_window: Optional[int] = None
    random_seed: int = 42
    output_dir: str = "outputs/evaluation/similarity"
    mdd_bins: int = 50
    compute_shr: bool = False
    shr_seed: int = 42
    shr_basis_count: int = 20
    shr_lambda: float = 0.1
    shr_basis_ratio: float = 0.25
    shr_c: float = 100.0
    shr_epsilon: float = 1.0e-5
    shr_max_iter: int = 1000
    shr_max_inner_iter: int = 5
    shr_max_samples_per_activity: Optional[int] = None
    shr_channel_mode: str = "mean"


@dataclass
class UtilityConfig:
    classifier: str = "random_forest"
    random_seed: int = 42
    output_dir: str = "outputs/evaluation/utility"
    n_estimators: int = 300
    max_depth: Optional[int] = None


@dataclass
class PlotConfig:
    output_dir: str = "outputs/plots"
    max_examples_per_activity: int = 3
    embedding_scatter_max_points: int = 1000


@dataclass
class PipelineConfig:
    data: DataConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_config(path: Union[str, Path]) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    return PipelineConfig(
        data=DataConfig(**raw["data"]),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        training=TrainingConfig(
            **{
                **raw.get("training", {}),
                "peft": PeftConfig(**raw.get("training", {}).get("peft", {})),
            }
        ),
        generation=GenerationConfig(**raw.get("generation", {})),
        similarity=SimilarityConfig(**raw.get("similarity", {})),
        utility=UtilityConfig(**raw.get("utility", {})),
        plots=PlotConfig(**raw.get("plots", {})),
    )


def save_resolved_config(config: PipelineConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(config.to_dict(), fp, sort_keys=False)
