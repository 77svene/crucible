"""Unified Configuration System with Schema Validation for crucible.

Replaces ad-hoc YAML/JSON configs with strongly-typed, versioned configuration schema using Pydantic.
Provides auto-completion, validation, and migration tools to prevent silent errors and simplify complex multi-model setups.

This module defines Pydantic models for all training parameters with nested sections for model, data, and training.
It generates JSON Schema for IDE support and provides CLI tools to validate and migrate old configs.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.schema import schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Version information
CONFIG_VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = "1.0.0"


class ModelFramework(str, Enum):
    """Supported model frameworks."""
    HUGGINGFACE = "huggingface"
    MEGATRON = "megatron"
    VLLM = "vllm"
    LLAMACPP = "llama.cpp"


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    NONE = "none"
    BITSANDBYTES_4BIT = "bitsandbytes_4bit"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"


class AttentionImplementation(str, Enum):
    """Supported attention implementations."""
    AUTO = "auto"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    XFORMERS = "xformers"


class ModelConfig(BaseModel):
    """Model configuration section."""
    model_name_or_path: str = Field(
        ...,
        description="Path to pretrained model or model identifier from huggingface.co/models"
    )
    model_revision: Optional[str] = Field(
        None,
        description="The specific model version to use (branch name, tag name, or commit id)"
    )
    model_framework: ModelFramework = Field(
        ModelFramework.HUGGINGFACE,
        description="Framework to use for model loading and inference"
    )
    tokenizer_name_or_path: Optional[str] = Field(
        None,
        description="Path to tokenizer or tokenizer identifier. If None, uses model_name_or_path"
    )
    tokenizer_revision: Optional[str] = Field(
        None,
        description="The specific tokenizer version to use"
    )
    torch_dtype: str = Field(
        "auto",
        description="PyTorch dtype for model weights (auto, float16, bfloat16, float32)"
    )
    quantization: QuantizationMethod = Field(
        QuantizationMethod.NONE,
        description="Quantization method to use for model loading"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional quantization configuration parameters"
    )
    attention_implementation: AttentionImplementation = Field(
        AttentionImplementation.AUTO,
        description="Attention implementation to use"
    )
    use_flash_attention_2: bool = Field(
        False,
        description="Whether to use Flash Attention 2 (deprecated, use attention_implementation instead)"
    )
    device_map: Optional[Union[str, Dict[str, Any]]] = Field(
        "auto",
        description="Device map for model placement"
    )
    rope_scaling: Optional[Dict[str, Any]] = Field(
        None,
        description="RoPE scaling configuration"
    )
    use_cache: bool = Field(
        True,
        description="Whether to use KV cache during inference"
    )
    trust_remote_code: bool = Field(
        False,
        description="Whether to trust remote code from Hugging Face Hub"
    )
    use_auth_token: Optional[Union[bool, str]] = Field(
        None,
        description="Hugging Face authentication token"
    )

    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"


class DatasetConfig(BaseModel):
    """Dataset configuration section."""
    dataset_name: Optional[str] = Field(
        None,
        description="Name of the dataset from Hugging Face Hub"
    )
    dataset_config_name: Optional[str] = Field(
        None,
        description="Configuration name of the dataset"
    )
    dataset_split: Optional[str] = Field(
        "train",
        description="Dataset split to use"
    )
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = Field(
        None,
        description="Path to data files"
    )
    data_dir: Optional[str] = Field(
        None,
        description="Directory containing dataset files"
    )
    dataset_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional dataset loading arguments"
    )
    streaming: bool = Field(
        False,
        description="Whether to stream the dataset"
    )
    max_samples: Optional[int] = Field(
        None,
        description="Maximum number of samples to use from dataset"
    )
    validation_split_percentage: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of dataset to use for validation"
    )
    preprocessing_num_workers: Optional[int] = Field(
        None,
        description="Number of workers for dataset preprocessing"
    )
    overwrite_cache: bool = Field(
        False,
        description="Whether to overwrite cached datasets"
    )
    column_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of dataset column names to expected names"
    )
    prompt_template: Optional[str] = Field(
        None,
        description="Template for formatting prompts"
    )
    response_template: Optional[str] = Field(
        None,
        description="Template for formatting responses"
    )
    chat_template: Optional[str] = Field(
        None,
        description="Chat template for conversational datasets"
    )

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training configuration section."""
    output_dir: str = Field(
        ...,
        description="Directory to save model checkpoints and logs"
    )
    overwrite_output_dir: bool = Field(
        False,
        description="Whether to overwrite output directory if exists"
    )
    num_train_epochs: float = Field(
        3.0,
        gt=0.0,
        description="Total number of training epochs"
    )
    max_steps: Optional[int] = Field(
        None,
        description="Maximum number of training steps. Overrides num_train_epochs if set"
    )
    per_device_train_batch_size: int = Field(
        8,
        gt=0,
        description="Batch size per GPU/TPU core for training"
    )
    per_device_eval_batch_size: int = Field(
        8,
        gt=0,
        description="Batch size per GPU/TPU core for evaluation"
    )
    gradient_accumulation_steps: int = Field(
        1,
        gt=0,
        description="Number of updates steps to accumulate before backward pass"
    )
    learning_rate: float = Field(
        5e-5,
        gt=0.0,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        0.0,
        ge=0.0,
        description="Weight decay to apply"
    )
    adam_beta1: float = Field(
        0.9,
        ge=0.0,
        lt=1.0,
        description="Beta1 for Adam optimizer"
    )
    adam_beta2: float = Field(
        0.999,
        ge=0.0,
        lt=1.0,
        description="Beta2 for Adam optimizer"
    )
    adam_epsilon: float = Field(
        1e-8,
        gt=0.0,
        description="Epsilon for Adam optimizer"
    )
    max_grad_norm: float = Field(
        1.0,
        gt=0.0,
        description="Maximum gradient norm for gradient clipping"
    )
    learning_rate_scheduler_type: str = Field(
        "linear",
        description="Learning rate scheduler type"
    )
    warmup_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of training for linear warmup"
    )
    warmup_steps: int = Field(
        0,
        ge=0,
        description="Number of warmup steps. Overrides warmup_ratio if set"
    )
    logging_dir: Optional[str] = Field(
        None,
        description="Directory for TensorBoard logs"
    )
    logging_strategy: str = Field(
        "steps",
        description="Logging strategy (steps, epoch, no)"
    )
    logging_steps: int = Field(
        500,
        gt=0,
        description="Log every X updates steps"
    )
    save_strategy: str = Field(
        "steps",
        description="Save strategy (steps, epoch, no)"
    )
    save_steps: int = Field(
        500,
        gt=0,
        description="Save checkpoint every X updates steps"
    )
    save_total_limit: Optional[int] = Field(
        None,
        description="Maximum number of checkpoints to keep"
    )
    evaluation_strategy: str = Field(
        "no",
        description="Evaluation strategy (steps, epoch, no)"
    )
    eval_steps: Optional[int] = Field(
        None,
        description="Run evaluation every X steps"
    )
    eval_delay: Optional[float] = Field(
        None,
        description="Number of epochs or steps to wait before first evaluation"
    )
    load_best_model_at_end: bool = Field(
        False,
        description="Whether to load the best model at the end of training"
    )
    metric_for_best_model: Optional[str] = Field(
        None,
        description="Metric to use for best model selection"
    )
    greater_is_better: Optional[bool] = Field(
        None,
        description="Whether greater metric value is better"
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )
    fp16: bool = Field(
        False,
        description="Whether to use FP16 training"
    )
    bf16: bool = Field(
        False,
        description="Whether to use BF16 training"
    )
    tf32: Optional[bool] = Field(
        None,
        description="Whether to use TF32 precision"
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Whether to use gradient checkpointing"
    )
    deepspeed: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="DeepSpeed configuration"
    )
    optim: str = Field(
        "adamw_hf",
        description="Optimizer to use"
    )
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional scheduler arguments"
    )
    report_to: Union[str, List[str]] = Field(
        "none",
        description="List of integrations to report results to"
    )
    resume_from_checkpoint: Optional[str] = Field(
        None,
        description="Path to checkpoint to resume training from"
    )
    neftune_noise_alpha: Optional[float] = Field(
        None,
        description="NEFTune noise alpha parameter"
    )

    @validator("evaluation_strategy", "logging_strategy", "save_strategy")
    def validate_strategy(cls, v):
        """Validate strategy values."""
        allowed = {"steps", "epoch", "no"}
        if v not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}, got {v}")
        return v

    @validator("fp16", "bf16")
    def validate_precision(cls, v, values):
        """Validate precision settings."""
        if v and values.get("fp16" if "fp16" in values else "bf16"):
            raise ValueError("Cannot use both FP16 and BF16 training")
        return v

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


class LoraConfig(BaseModel):
    """LoRA configuration section."""
    use_lora: bool = Field(
        False,
        description="Whether to use LoRA"
    )
    lora_r: int = Field(
        8,
        gt=0,
        description="LoRA attention dimension"
    )
    lora_alpha: float = Field(
        16.0,
        gt=0.0,
        description="LoRA alpha parameter"
    )
    lora_dropout: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="LoRA dropout probability"
    )
    lora_target_modules: Optional[List[str]] = Field(
        None,
        description="List of module names to apply LoRA to"
    )
    lora_bias: str = Field(
        "none",
        description="LoRA bias type (none, all, lora_only)"
    )
    lora_task_type: str = Field(
        "CAUSAL_LM",
        description="Task type for LoRA"
    )
    modules_to_save: Optional[List[str]] = Field(
        None,
        description="List of modules apart from LoRA layers to be set as trainable and saved"
    )

    @validator("lora_bias")
    def validate_lora_bias(cls, v):
        """Validate LoRA bias setting."""
        allowed = {"none", "all", "lora_only"}
        if v not in allowed:
            raise ValueError(f"lora_bias must be one of {allowed}, got {v}")
        return v

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


class QuantizationConfig(BaseModel):
    """Quantization configuration section."""
    load_in_4bit: bool = Field(
        False,
        description="Whether to load model in 4-bit precision"
    )
    load_in_8bit: bool = Field(
        False,
        description="Whether to load model in 8-bit precision"
    )
    bnb_4bit_compute_dtype: str = Field(
        "float16",
        description="Compute dtype for 4-bit quantization"
    )
    bnb_4bit_quant_type: str = Field(
        "nf4",
        description="Quantization type for 4-bit quantization"
    )
    bnb_4bit_use_double_quant: bool = Field(
        False,
        description="Whether to use double quantization for 4-bit"
    )
    bnb_4bit_quant_storage: Optional[str] = Field(
        None,
        description="Storage dtype for 4-bit quantization"
    )

    @root_validator
    def validate_quantization(cls, values):
        """Validate quantization settings."""
        load_in_4bit = values.get("load_in_4bit", False)
        load_in_8bit = values.get("load_in_8bit", False)

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")

        return values

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


class GenerationConfig(BaseModel):
    """Generation configuration section."""
    max_new_tokens: int = Field(
        256,
        gt=0,
        description="Maximum number of new tokens to generate"
    )
    do_sample: bool = Field(
        True,
        description="Whether to use sampling"
    )
    temperature: float = Field(
        0.7,
        gt=0.0,
        description="Sampling temperature"
    )
    top_k: int = Field(
        50,
        ge=0,
        description="Top-k sampling parameter"
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    repetition_penalty: float = Field(
        1.0,
        gt=0.0,
        description="Repetition penalty parameter"
    )
    length_penalty: float = Field(
        1.0,
        description="Length penalty for beam search"
    )
    num_beams: int = Field(
        1,
        gt=0,
        description="Number of beams for beam search"
    )
    early_stopping: bool = Field(
        False,
        description="Whether to stop beam search when num_beams sentences are finished"
    )

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


class crucibleConfig(BaseModel):
    """Main configuration schema for crucible."""
    config_version: str = Field(
        CONFIG_VERSION,
        description="Configuration schema version"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp when config was created/modified"
    )
    model: ModelConfig = Field(
        ...,
        description="Model configuration"
    )
    data: DatasetConfig = Field(
        ...,
        description="Dataset configuration"
    )
    training: TrainingConfig = Field(
        ...,
        description="Training configuration"
    )
    lora: LoraConfig = Field(
        LoraConfig(),
        description="LoRA configuration"
    )
    quantization: QuantizationConfig = Field(
        QuantizationConfig(),
        description="Quantization configuration"
    )
    generation: GenerationConfig = Field(
        GenerationConfig(),
        description="Generation configuration"
    )
    custom: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom configuration parameters"
    )

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        """Set timestamp if not provided."""
        return v or datetime.utcnow().isoformat()

    @root_validator
    def validate_config_consistency(cls, values):
        """Validate configuration consistency across sections."""
        model_config = values.get("model")
        training_config = values.get("training")
        quant_config = values.get("quantization")

        # Validate quantization settings
        if model_config and quant_config:
            if model_config.quantization != QuantizationMethod.NONE:
                if quant_config.load_in_4bit or quant_config.load_in_8bit:
                    raise ValueError(
                        "Conflicting quantization settings: model.quantization and quantization.load_in_*"
                    )

        # Validate precision settings
        if training_config:
            if training_config.fp16 and training_config.bf16:
                raise ValueError("Cannot use both FP16 and BF16 training")

            # Validate logging directory
            if training_config.output_dir and not training_config.logging_dir:
                values["training"].logging_dir = str(
                    Path(training_config.output_dir) / "logs"
                )

        return values

    class Config:
        """Pydantic config."""
        title = "crucible Configuration"
        description = "Configuration schema for crucible training pipeline"
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ConfigValidator:
    """Configuration validator and migration tool."""

    def __init__(self):
        """Initialize the validator."""
        self.schema = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema from Pydantic models."""
        models = [crucibleConfig]
        schema_dict = schema(models, title="crucible Configuration Schema")
        return schema_dict

    def save_schema(self, output_path: Union[str, Path]) -> None:
        """Save JSON Schema to file.

        Args:
            output_path: Path to save the schema file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)

        logger.info(f"Schema saved to {output_path}")

    def validate_config(self, config_data: Dict[str, Any]) -> crucibleConfig:
        """Validate configuration data against schema.

        Args:
            config_data: Configuration dictionary to validate

        Returns:
            Validated configuration object

        Raises:
            ValueError: If validation fails
        """
        try:
            config = crucibleConfig(**config_data)
            logger.info("Configuration validation successful")
            return config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def load_and_validate(self, config_path: Union[str, Path]) -> crucibleConfig:
        """Load and validate configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Validated configuration object
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return self.validate_config(config_data)

    def migrate_config(
        self,
        old_config_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Migrate old configuration to new schema.

        Args:
            old_config_path: Path to old configuration file
            output_path: Path to save migrated config (optional)
            dry_run: If True, only show migration without saving

        Returns:
            Migrated configuration dictionary
        """
        old_config_path = Path(old_config_path)

        # Load old config
        with open(old_config_path, "r", encoding="utf-8") as f:
            if old_config_path.suffix in [".yaml", ".yml"]:
                old_config = yaml.safe_load(f)
            elif old_config_path.suffix == ".json":
                old_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {old_config_path.suffix}")

        # Perform migration
        migrated = self._perform_migration(old_config)

        # Add metadata
        migrated["config_version"] = CONFIG_VERSION
        migrated["timestamp"] = datetime.utcnow().isoformat()

        # Validate migrated config
        try:
            self.validate_config(migrated)
            logger.info("Migration successful: config validates against new schema")
        except Exception as e:
            logger.warning(f"Migration produced invalid config: {e}")
            logger.warning("Manual adjustment may be required")

        # Save if requested
        if not dry_run and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                if output_path.suffix in [".yaml", ".yml"]:
                    yaml.dump(migrated, f, default_flow_style=False, allow_unicode=True)
                elif output_path.suffix == ".json":
                    json.dump(migrated, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_path.suffix}")

            logger.info(f"Migrated config saved to {output_path}")

        return migrated

    def _perform_migration(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual migration of configuration structure.

        Args:
            old_config: Old configuration dictionary

        Returns:
            Migrated configuration dictionary
        """
        migrated = {}

        # Map old top-level keys to new structure
        key_mapping = {
            # Model section
            "model_name_or_path": "model.model_name_or_path",
            "model_revision": "model.model_revision",
            "tokenizer_name": "model.tokenizer_name_or_path",
            "tokenizer_name_or_path": "model.tokenizer_name_or_path",
            "tokenizer_revision": "model.tokenizer_revision",
            "torch_dtype": "model.torch_dtype",
            "use_flash_attention_2": "model.use_flash_attention_2",
            "device_map": "model.device_map",
            "rope_scaling": "model.rope_scaling",
            "trust_remote_code": "model.trust_remote_code",
            "use_auth_token": "model.use_auth_token",

            # Data section
            "dataset_name": "data.dataset_name",
            "dataset_config_name": "data.dataset_config_name",
            "dataset_split": "data.dataset_split",
            "data_files": "data.data_files",
            "data_dir": "data.data_dir",
            "streaming": "data.streaming",
            "max_samples": "data.max_samples",
            "validation_split_percentage": "data.validation_split_percentage",
            "preprocessing_num_workers": "data.preprocessing_num_workers",
            "overwrite_cache": "data.overwrite_cache",
            "column_mapping": "data.column_mapping",
            "prompt_template": "data.prompt_template",
            "response_template": "data.response_template",
            "chat_template": "data.chat_template",

            # Training section
            "output_dir": "training.output_dir",
            "overwrite_output_dir": "training.overwrite_output_dir",
            "num_train_epochs": "training.num_train_epochs",
            "max_steps": "training.max_steps",
            "per_device_train_batch_size": "training.per_device_train_batch_size",
            "per_device_eval_batch_size": "training.per_device_eval_batch_size",
            "gradient_accumulation_steps": "training.gradient_accumulation_steps",
            "learning_rate": "training.learning_rate",
            "weight_decay": "training.weight_decay",
            "adam_beta1": "training.adam_beta1",
            "adam_beta2": "training.adam_beta2",
            "adam_epsilon": "training.adam_epsilon",
            "max_grad_norm": "training.max_grad_norm",
            "learning_rate_scheduler_type": "training.learning_rate_scheduler_type",
            "warmup_ratio": "training.warmup_ratio",
            "warmup_steps": "training.warmup_steps",
            "logging_dir": "training.logging_dir",
            "logging_strategy": "training.logging_strategy",
            "logging_steps": "training.logging_steps",
            "save_strategy": "training.save_strategy",
            "save_steps": "training.save_steps",
            "save_total_limit": "training.save_total_limit",
            "evaluation_strategy": "training.evaluation_strategy",
            "eval_steps": "training.eval_steps",
            "eval_delay": "training.eval_delay",
            "load_best_model_at_end": "training.load_best_model_at_end",
            "metric_for_best_model": "training.metric_for_best_model",
            "greater_is_better": "training.greater_is_better",
            "seed": "training.seed",
            "fp16": "training.fp16",
            "bf16": "training.bf16",
            "tf32": "training.tf32",
            "gradient_checkpointing": "training.gradient_checkpointing",
            "deepspeed": "training.deepspeed",
            "optim": "training.optim",
            "lr_scheduler_kwargs": "training.lr_scheduler_kwargs",
            "report_to": "training.report_to",
            "resume_from_checkpoint": "training.resume_from_checkpoint",
            "neftune_noise_alpha": "training.neftune_noise_alpha",

            # LoRA section
            "use_lora": "lora.use_lora",
            "lora_r": "lora.lora_r",
            "lora_alpha": "lora.lora_alpha",
            "lora_dropout": "lora.lora_dropout",
            "lora_target_modules": "lora.lora_target_modules",
            "lora_bias": "lora.lora_bias",
            "lora_task_type": "lora.lora_task_type",
            "modules_to_save": "lora.modules_to_save",

            # Quantization section
            "load_in_4bit": "quantization.load_in_4bit",
            "load_in_8bit": "quantization.load_in_8bit",
            "bnb_4bit_compute_dtype": "quantization.bnb_4bit_compute_dtype",
            "bnb_4bit_quant_type": "quantization.bnb_4bit_quant_type",
            "bnb_4bit_use_double_quant": "quantization.bnb_4bit_use_double_quant",
            "bnb_4bit_quant_storage": "quantization.bnb_4bit_quant_storage",

            # Generation section
            "max_new_tokens": "generation.max_new_tokens",
            "do_sample": "generation.do_sample",
            "temperature": "generation.temperature",
            "top_k": "generation.top_k",
            "top_p": "generation.top_p",
            "repetition_penalty": "generation.repetition_penalty",
            "length_penalty": "generation.length_penalty",
            "num_beams": "generation.num_beams",
            "early_stopping": "generation.early_stopping",
        }

        # Apply key mapping
        for old_key, new_path in key_mapping.items():
            if old_key in old_config:
                self._set_nested_value(migrated, new_path, old_config[old_key])

        # Handle special cases
        self._handle_special_cases(old_config, migrated)

        # Store unmapped keys in custom section
        unmapped_keys = set(old_config.keys()) - set(key_mapping.keys())
        if unmapped_keys:
            migrated["custom"] = {}
            for key in unmapped_keys:
                migrated["custom"][key] = old_config[key]
                logger.warning(f"Unmapped config key '{key}' moved to custom section")

        # Ensure required sections exist
        if "model" not in migrated:
            migrated["model"] = {"model_name_or_path": "unknown"}
            logger.warning("Model section missing, using defaults")
        if "data" not in migrated:
            migrated["data"] = {}
            logger.warning("Data section missing, using defaults")
        if "training" not in migrated:
            migrated["training"] = {"output_dir": "./output"}
            logger.warning("Training section missing, using defaults")

        return migrated

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in configuration dictionary.

        Args:
            config: Configuration dictionary
            path: Dot-separated path to value
            value: Value to set
        """
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _handle_special_cases(
        self, old_config: Dict[str, Any], migrated: Dict[str, Any]
    ) -> None:
        """Handle special migration cases.

        Args:
            old_config: Old configuration dictionary
            migrated: Migrated configuration dictionary
        """
        # Handle model quantization
        if "quantization" in old_config and isinstance(old_config["quantization"], str):
            quant_method = old_config["quantization"]
            if quant_method in ["bitsandbytes_4bit", "bitsandbytes_8bit"]:
                if "model" not in migrated:
                    migrated["model"] = {}
                migrated["model"]["quantization"] = quant_method

        # Handle use_flash_attention_2 deprecation
        if old_config.get("use_flash_attention_2"):
            if "model" not in migrated:
                migrated["model"] = {}
            migrated["model"]["attention_implementation"] = "flash_attention_2"

        # Handle dataset path variations
        if "dataset_path" in old_config and "data" in migrated:
            migrated["data"]["data_files"] = old_config["dataset_path"]

        # Handle train_file and validation_file
        if "train_file" in old_config or "validation_file" in old_config:
            if "data" not in migrated:
                migrated["data"] = {}
            data_files = {}
            if "train_file" in old_config:
                data_files["train"] = old_config["train_file"]
            if "validation_file" in old_config:
                data_files["validation"] = old_config["validation_file"]
            migrated["data"]["data_files"] = data_files

    def diff_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two configurations and show differences.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Dictionary with differences
        """
        differences = {}

        def compare_dicts(d1: Dict, d2: Dict, path: str = "") -> None:
            """Recursively compare dictionaries."""
            all_keys = set(d1.keys()) | set(d2.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key

                if key not in d1:
                    differences[current_path] = {
                        "type": "added",
                        "value": d2[key],
                    }
                elif key not in d2:
                    differences[current_path] = {
                        "type": "removed",
                        "value": d1[key],
                    }
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        "type": "changed",
                        "old_value": d1[key],
                        "new_value": d2[key],
                    }

        compare_dicts(config1, config2)
        return differences


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="crucible Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python scripts/config_migrate.py validate config.yaml

  # Migrate old configuration to new format
  python scripts/config_migrate.py migrate old_config.yaml --output new_config.yaml

  # Generate JSON Schema for IDE support
  python scripts/config_migrate.py generate-schema --output schema.json

  # Compare two configuration files
  python scripts/config_migrate.py diff config1.yaml config2.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )
    validate_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation information",
    )

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate old configuration")
    migrate_parser.add_argument(
        "old_config",
        type=str,
        help="Path to old configuration file",
    )
    migrate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for migrated configuration",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration without saving",
    )

    # Generate schema command
    schema_parser = subparsers.add_parser(
        "generate-schema", help="Generate JSON Schema"
    )
    schema_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="schema.json",
        help="Output path for JSON Schema (default: schema.json)",
    )

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare configurations")
    diff_parser.add_argument(
        "config1",
        type=str,
        help="Path to first configuration file",
    )
    diff_parser.add_argument(
        "config2",
        type=str,
        help="Path to second configuration file",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    validator = ConfigValidator()

    try:
        if args.command == "validate":
            config = validator.load_and_validate(args.config_file)
            print("✓ Configuration is valid")
            if args.verbose:
                print("\nConfiguration summary:")
                print(f"  Model: {config.model.model_name_or_path}")
                print(f"  Output: {config.training.output_dir}")
                print(f"  LoRA: {'enabled' if config.lora.use_lora else 'disabled'}")
                print(f"  Quantization: {config.model.quantization}")

        elif args.command == "migrate":
            output_path = args.output
            if not output_path and not args.dry_run:
                # Generate default output path
                old_path = Path(args.old_config)
                output_path = old_path.parent / f"{old_path.stem}_migrated{old_path.suffix}"

            migrated = validator.migrate_config(
                args.old_config,
                output_path=output_path,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                print("Migration preview:")
                print(yaml.dump(migrated, default_flow_style=False, allow_unicode=True))
            else:
                print(f"✓ Configuration migrated to {output_path}")

        elif args.command == "generate-schema":
            validator.save_schema(args.output)
            print(f"✓ JSON Schema saved to {args.output}")

        elif args.command == "diff":
            # Load both configs
            with open(args.config1, "r", encoding="utf-8") as f:
                if args.config1.endswith((".yaml", ".yml")):
                    config1 = yaml.safe_load(f)
                else:
                    config1 = json.load(f)

            with open(args.config2, "r", encoding="utf-8") as f:
                if args.config2.endswith((".yaml", ".yml")):
                    config2 = yaml.safe_load(f)
                else:
                    config2 = json.load(f)

            # Compare
            differences = validator.diff_configs(config1, config2)

            if not differences:
                print("✓ Configurations are identical")
                return

            # Output differences
            if args.format == "json":
                print(json.dumps(differences, indent=2, ensure_ascii=False))
            else:
                print(f"Found {len(differences)} differences:")
                print("-" * 80)
                for path, diff in differences.items():
                    if diff["type"] == "added":
                        print(f"+ {path}: {diff['value']}")
                    elif diff["type"] == "removed":
                        print(f"- {path}: {diff['value']}")
                    elif diff["type"] == "changed":
                        print(f"M {path}:")
                        print(f"  Old: {diff['old_value']}")
                        print(f"  New: {diff['new_value']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()