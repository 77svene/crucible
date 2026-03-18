# crucible/config/schema.py
"""
Unified Configuration System with Schema Validation for crucible.

This module replaces ad-hoc YAML/JSON configurations with a strongly-typed,
versioned configuration schema using Pydantic. It provides auto-completion,
validation, and migration tools to prevent silent errors and simplify complex
multi-model setups.

Key Features:
- Nested Pydantic models for model, data, training, and other configurations
- JSON Schema generation for IDE auto-completion support
- Configuration validation with detailed error messages
- Migration tools for converting legacy configurations
- Version tracking for backward compatibility
- CLI integration for validation and migration

Example usage:
    from crucible.config.schema import crucibleConfig, validate_config
    
    # Load and validate configuration
    config = crucibleConfig.from_file("config.yaml")
    
    # Or validate existing dictionary
    validate_config(your_config_dict)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar

import yaml
from pydantic import BaseModel, Field, validator, root_validator, ValidationError
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)

# Type variable for generic config loading
T = TypeVar('T', bound='crucibleConfig')


class ConfigVersion(str, Enum):
    """Configuration schema version for backward compatibility."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"  # Current version
    
    @classmethod
    def latest(cls) -> 'ConfigVersion':
        return cls.V2_0


class ModelFramework(str, Enum):
    """Supported model frameworks."""
    HUGGINGFACE = "huggingface"
    MEGATRON = "megatron"
    VLLM = "vllm"
    LLAMACPP = "llama.cpp"


class TrainingMethod(str, Enum):
    """Supported training methods."""
    SFT = "sft"
    RLHF = "rlhf"
    DPO = "dpo"
    ORPO = "orpo"
    PRETRAINING = "pretraining"
    LORA = "lora"
    QLORA = "qlora"
    FREEZE = "freeze"


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    BITSANDBYTES_4BIT = "bitsandbytes_4bit"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"


class AttentionImplementation(str, Enum):
    """Supported attention implementations."""
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION = "flash_attention"
    SDPA = "sdpa"
    XFORMERS = "xformers"
    DEFAULT = "default"


class Optimizer(str, Enum):
    """Supported optimizers."""
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAMW_BNB = "adamw_bnb"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_HF = "adamw_hf"
    ADAFACTOR = "adafactor"


class LRScheduler(str, Enum):
    """Supported learning rate schedulers."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"


class LoggerType(str, Enum):
    """Supported logging backends."""
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    CLEARML = "clearml"
    COMET = "comet"
    NEPTUNE = "neptune"


class BaseModelConfig(BaseModel):
    """Base configuration model with common utilities."""
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"  # Forbid extra fields to catch typos
        use_enum_values = True
        json_schema_extra = {"title": "crucible Configuration"}
        
    def to_dict(self, exclude_none: bool = True, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding None values."""
        return self.model_dump(exclude_none=exclude_none, **kwargs)
    
    def to_yaml(self, exclude_none: bool = True, **kwargs) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(exclude_none=exclude_none), 
                        default_flow_style=False, allow_unicode=True, **kwargs)
    
    def to_json(self, exclude_none: bool = True, indent: int = 2, **kwargs) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(exclude_none=exclude_none), 
                         indent=indent, ensure_ascii=False, **kwargs)
    
    def save_yaml(self, path: Union[str, Path], exclude_none: bool = True) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml(exclude_none=exclude_none))
    
    def save_json(self, path: Union[str, Path], exclude_none: bool = True, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(exclude_none=exclude_none, indent=indent))
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from file (auto-detect format by extension)."""
        path = Path(path)
        if path.suffix.lower() in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        elif path.suffix.lower() == '.json':
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")


class ModelConfig(BaseModelConfig):
    """Model configuration."""
    model_name_or_path: str = Field(
        ...,
        description="Path to pretrained model or model identifier from huggingface.co/models",
        examples=["meta-llama/Llama-2-7b-hf", "Qwen/Qwen-7B-Chat", "./models/llama-7b"]
    )
    tokenizer_path: Optional[str] = Field(
        None,
        description="Path to tokenizer. If None, uses model_name_or_path",
        examples=["meta-llama/Llama-2-7b-hf", "./tokenizers/llama"]
    )
    adapter_name_or_path: Optional[str] = Field(
        None,
        description="Path to pretrained adapter or adapter identifier",
        examples=["./adapters/lora-llama-7b"]
    )
    cache_dir: Optional[str] = Field(
        None,
        description="Directory to store downloaded models and datasets",
        examples=["~/.cache/huggingface", "./cache"]
    )
    use_fast_tokenizer: bool = Field(
        True,
        description="Whether to use fast tokenizer (if available)"
    )
    padding_side: str = Field(
        "right",
        description="Tokenizer padding side (left or right)"
    )
    model_revision: str = Field(
        "main",
        description="Specific model version to use (branch name, tag name, or commit id)"
    )
    torch_dtype: str = Field(
        "float16",
        description="PyTorch dtype for model loading",
        examples=["float16", "bfloat16", "float32", "auto"]
    )
    attn_implementation: Optional[AttentionImplementation] = Field(
        None,
        description="Attention implementation to use"
    )
    use_auth_token: Optional[str] = Field(
        None,
        description="Hugging Face token for private models"
    )
    model_max_length: Optional[int] = Field(
        None,
        description="Maximum sequence length for the model",
        ge=1
    )
    model_framework: ModelFramework = Field(
        ModelFramework.HUGGINGFACE,
        description="Framework to use for model loading"
    )
    
    @validator('padding_side')
    def validate_padding_side(cls, v):
        if v not in ['left', 'right']:
            raise ValueError("padding_side must be 'left' or 'right'")
        return v
    
    @validator('torch_dtype')
    def validate_torch_dtype(cls, v):
        valid_dtypes = ['float16', 'bfloat16', 'float32', 'auto']
        if v not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        return v


class QuantizationConfig(BaseModelConfig):
    """Quantization configuration."""
    method: Optional[QuantizationMethod] = Field(
        None,
        description="Quantization method to use"
    )
    bits: int = Field(
        4,
        description="Number of bits for quantization",
        ge=2, le=8
    )
    group_size: int = Field(
        128,
        description="Group size for quantization",
        ge=1
    )
    damp_percent: float = Field(
        0.01,
        description="Dampening percentage for quantization",
        ge=0.0, le=1.0
    )
    desc_act: bool = Field(
        False,
        description="Whether to use descending activation order"
    )
    sym: bool = Field(
        True,
        description="Whether to use symmetric quantization"
    )
    true_sequential: bool = Field(
        True,
        description="Whether to use true sequential quantization"
    )
    dataset: Optional[str] = Field(
        None,
        description="Dataset for calibration (for GPTQ/AWQ)"
    )
    
    @validator('method')
    def validate_method_bits(cls, v, values):
        if v is not None and 'bits' in values:
            bits = values['bits']
            if v == QuantizationMethod.BITSANDBYTES_8BIT and bits != 8:
                raise ValueError("bitsandbytes_8bit requires bits=8")
            if v == QuantizationMethod.BITSANDBYTES_4BIT and bits != 4:
                raise ValueError("bitsandbytes_4bit requires bits=4")
        return v


class DataConfig(BaseModelConfig):
    """Data configuration."""
    dataset: Union[str, List[str]] = Field(
        ...,
        description="Dataset name(s) or path(s) for training",
        examples=["alpaca", "vicuna", "./data/train.json", ["dataset1", "dataset2"]]
    )
    dataset_dir: str = Field(
        "data",
        description="Directory containing dataset files"
    )
    val_dataset: Optional[Union[str, List[str]]] = Field(
        None,
        description="Dataset name(s) or path(s) for validation"
    )
    split: str = Field(
        "train",
        description="Dataset split to use"
    )
    data_seed: Optional[int] = Field(
        None,
        description="Random seed for dataset shuffling"
    )
    preprocessing_num_workers: Optional[int] = Field(
        None,
        description="Number of processes for data preprocessing",
        ge=1
    )
    overwrite_cache: bool = Field(
        False,
        description="Whether to overwrite the cached datasets"
    )
    template: Optional[str] = Field(
        None,
        description="Prompt template for formatting",
        examples=["alpaca", "vicuna", "chatglm3", "llama2", "llama3"]
    )
    cutoff_len: int = Field(
        1024,
        description="Maximum length after tokenization",
        ge=1
    )
    reserved_label_len: int = Field(
        1,
        description="Number of labels to reserve at the beginning",
        ge=0
    )
    train_on_prompt: bool = Field(
        False,
        description="Whether to train on user input (prompt)"
    )
    streaming: bool = Field(
        False,
        description="Whether to stream datasets"
    )
    buffer_size: int = Field(
        16384,
        description="Buffer size for streaming datasets",
        ge=1
    )
    mix_strategy: str = Field(
        "concat",
        description="Strategy for mixing multiple datasets",
        examples=["concat", "interleave_under", "interleave_over"]
    )
    overwrite_processed_data: bool = Field(
        False,
        description="Whether to overwrite processed dataset files"
    )
    
    @validator('mix_strategy')
    def validate_mix_strategy(cls, v):
        valid_strategies = ['concat', 'interleave_under', 'interleave_over']
        if v not in valid_strategies:
            raise ValueError(f"mix_strategy must be one of {valid_strategies}")
        return v


class TrainingConfig(BaseModelConfig):
    """Training configuration."""
    output_dir: str = Field(
        ...,
        description="Directory to save model checkpoints and logs",
        examples=["./output/llama-7b-sft", "./results"]
    )
    overwrite_output_dir: bool = Field(
        False,
        description="Whether to overwrite output directory if it exists"
    )
    num_train_epochs: float = Field(
        3.0,
        description="Number of training epochs",
        gt=0
    )
    max_steps: Optional[int] = Field(
        -1,
        description="Maximum number of training steps. Overrides num_train_epochs if > 0",
        ge=-1
    )
    per_device_train_batch_size: int = Field(
        4,
        description="Batch size per GPU/CPU for training",
        ge=1
    )
    per_device_eval_batch_size: int = Field(
        4,
        description="Batch size per GPU/CPU for evaluation",
        ge=1
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Number of steps for gradient accumulation",
        ge=1
    )
    learning_rate: float = Field(
        5e-5,
        description="Initial learning rate",
        gt=0
    )
    weight_decay: float = Field(
        0.0,
        description="Weight decay for regularization",
        ge=0
    )
    adam_beta1: float = Field(
        0.9,
        description="Beta1 for Adam optimizer",
        ge=0, le=1
    )
    adam_beta2: float = Field(
        0.999,
        description="Beta2 for Adam optimizer",
        ge=0, le=1
    )
    adam_epsilon: float = Field(
        1e-8,
        description="Epsilon for Adam optimizer",
        gt=0
    )
    max_grad_norm: float = Field(
        1.0,
        description="Maximum gradient norm for clipping",
        gt=0
    )
    lr_scheduler_type: LRScheduler = Field(
        LRScheduler.COSINE,
        description="Learning rate scheduler type"
    )
    warmup_ratio: float = Field(
        0.0,
        description="Proportion of training for linear warmup",
        ge=0, le=1
    )
    warmup_steps: int = Field(
        0,
        description="Number of warmup steps",
        ge=0
    )
    optim: Optimizer = Field(
        Optimizer.ADAMW_TORCH,
        description="Optimizer to use"
    )
    bf16: bool = Field(
        False,
        description="Whether to use bfloat16 mixed precision"
    )
    fp16: bool = Field(
        False,
        description="Whether to use float16 mixed precision"
    )
    tf32: Optional[bool] = Field(
        None,
        description="Whether to use TF32 precision (Ampere+ GPUs)"
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Whether to use gradient checkpointing to save memory"
    )
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional arguments for gradient checkpointing"
    )
    deepspeed: Optional[str] = Field(
        None,
        description="Path to DeepSpeed config file or string config",
        examples=["./ds_config.json", "ds_z2_config.json"]
    )
    local_rank: int = Field(
        -1,
        description="Local rank for distributed training"
    )
    ddp_backend: Optional[str] = Field(
        None,
        description="Distributed backend (nccl, gloo, mpi)"
    )
    ddp_find_unused_parameters: Optional[bool] = Field(
        None,
        description="Find unused parameters in DDP"
    )
    ddp_bucket_cap_mb: Optional[int] = Field(
        None,
        description="DDP bucket capacity in MB"
    )
    ddp_broadcast_buffers: Optional[bool] = Field(
        None,
        description="Broadcast buffers in DDP"
    )
    report_to: List[LoggerType] = Field(
        [],
        description="List of logging backends to report to"
    )
    logging_dir: Optional[str] = Field(
        None,
        description="Directory for TensorBoard logs"
    )
    logging_steps: int = Field(
        10,
        description="Log every X steps",
        ge=1
    )
    logging_first_step: bool = Field(
        False,
        description="Log first step"
    )
    logging_nan_inf_filter: bool = Field(
        True,
        description="Filter nan and inf losses from logging"
    )
    save_strategy: str = Field(
        "steps",
        description="Checkpoint saving strategy",
        examples=["steps", "epoch", "no"]
    )
    save_steps: int = Field(
        500,
        description="Save checkpoint every X steps",
        ge=1
    )
    save_total_limit: Optional[int] = Field(
        None,
        description="Maximum number of checkpoints to keep",
        ge=1
    )
    save_safetensors: bool = Field(
        True,
        description="Save checkpoints in safetensors format"
    )
    save_on_each_node: bool = Field(
        False,
        description="Save checkpoints on each node in distributed training"
    )
    save_only_model: bool = Field(
        False,
        description="Save only model weights (not optimizer/scheduler)"
    )
    evaluation_strategy: str = Field(
        "no",
        description="Evaluation strategy",
        examples=["steps", "epoch", "no"]
    )
    eval_steps: Optional[int] = Field(
        None,
        description="Evaluate every X steps",
        ge=1
    )
    eval_delay: Optional[float] = Field(
        None,
        description="Number of epochs/steps to wait before first evaluation",
        ge=0
    )
    eval_accumulation_steps: Optional[int] = Field(
        None,
        description="Number of steps for evaluation accumulation",
        ge=1
    )
    load_best_model_at_end: bool = Field(
        False,
        description="Load best model at end of training"
    )
    metric_for_best_model: Optional[str] = Field(
        None,
        description="Metric to use for best model selection"
    )
    greater_is_better: Optional[bool] = Field(
        None,
        description="Whether higher metric is better"
    )
    resume_from_checkpoint: Optional[str] = Field(
        None,
        description="Path to checkpoint to resume training from"
    )
    
    @validator('save_strategy')
    def validate_save_strategy(cls, v):
        valid_strategies = ['steps', 'epoch', 'no']
        if v not in valid_strategies:
            raise ValueError(f"save_strategy must be one of {valid_strategies}")
        return v
    
    @validator('evaluation_strategy')
    def validate_evaluation_strategy(cls, v):
        valid_strategies = ['steps', 'epoch', 'no']
        if v not in valid_strategies:
            raise ValueError(f"evaluation_strategy must be one of {valid_strategies}")
        return v
    
    @root_validator
    def validate_mixed_precision(cls, values):
        bf16 = values.get('bf16', False)
        fp16 = values.get('fp16', False)
        if bf16 and fp16:
            raise ValueError("Cannot use both bf16 and fp16 simultaneously")
        return values
    
    @root_validator
    def validate_eval_steps(cls, values):
        eval_strategy = values.get('evaluation_strategy')
        eval_steps = values.get('eval_steps')
        if eval_strategy == 'steps' and eval_steps is None:
            raise ValueError("eval_steps must be specified when evaluation_strategy is 'steps'")
        return values


class AdapterConfig(BaseModelConfig):
    """Adapter configuration (LoRA, QLoRA, etc.)."""
    method: Optional[TrainingMethod] = Field(
        None,
        description="Adapter training method"
    )
    r: int = Field(
        8,
        description="LoRA attention dimension",
        ge=1
    )
    lora_alpha: int = Field(
        16,
        description="LoRA alpha for scaling",
        ge=1
    )
    lora_dropout: float = Field(
        0.05,
        description="Dropout probability for LoRA layers",
        ge=0, le=1
    )
    lora_target_modules: Optional[List[str]] = Field(
        None,
        description="List of module names to apply LoRA to",
        examples=[["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]]
    )
    lora_bias: str = Field(
        "none",
        description="Bias type for LoRA",
        examples=["none", "all", "lora_only"]
    )
    modules_to_save: Optional[List[str]] = Field(
        None,
        description="List of modules to unfreeze and train",
        examples=[["embed_tokens", "lm_head"]]
    )
    init_lora_weights: Union[bool, str] = Field(
        True,
        description="How to initialize LoRA weights",
        examples=[True, False, "gaussian", "pissa", "loftq"]
    )
    
    @validator('lora_bias')
    def validate_lora_bias(cls, v):
        valid_biases = ['none', 'all', 'lora_only']
        if v not in valid_biases:
            raise ValueError(f"lora_bias must be one of {valid_biases}")
        return v


class RLHFConfig(BaseModelConfig):
    """RLHF configuration."""
    reward_model: Optional[str] = Field(
        None,
        description="Path to reward model for RLHF",
        examples=["./reward_model", "OpenAssistant/reward-model-deberta-v3-large-v2"]
    )
    reward_model_adapters: Optional[List[str]] = Field(
        None,
        description="Adapters to apply to reward model"
    )
    reward_model_quantization: Optional[QuantizationConfig] = Field(
        None,
        description="Quantization config for reward model"
    )
    ppo_epochs: int = Field(
        4,
        description="Number of PPO epochs per batch",
        ge=1
    )
    whiten_rewards: bool = Field(
        False,
        description="Whether to whiten rewards"
)
    kl_penalty: str = Field(
        "kl",
        description="KL penalty type",
        examples=["kl", "abs", "mse"]
    )
    cliprange: float = Field(
        0.2,
        description="PPO clip range",
        ge=0
    )
    cliprange_value: float = Field(
        0.2,
        description="PPO value function clip range",
        ge=0
    )
    gamma: float = Field(
        1.0,
        description="Discount factor for MDP",
        ge=0, le=1
    )
    lam: float = Field(
        0.95,
        description="Lambda for GAE",
        ge=0, le=1
    )
    vf_coef: float = Field(
        0.1,
        description="Value function coefficient",
        ge=0
    )
    adap_kl_ctrl: bool = Field(
        True,
        description="Use adaptive KL control"
    )
    init_kl_coef: float = Field(
        0.2,
        description="Initial KL coefficient",
        ge=0
    )
    target: Optional[float] = Field(
        6.0,
        description="Target KL value for adaptive control"
    )
    horizon: int = Field(
        10000,
        description="Horizon for adaptive KL control",
        ge=1
    )


class DPOConfig(BaseModelConfig):
    """DPO configuration."""
    beta: float = Field(
        0.1,
        description="Beta parameter for DPO loss",
        ge=0
    )
    label_smoothing: float = Field(
        0.0,
        description="Label smoothing for DPO",
        ge=0, le=0.5
    )
    loss_type: str = Field(
        "sigmoid",
        description="Loss type for DPO",
        examples=["sigmoid", "hinge", "ipo", "kto_pair"]
    )
    pref_loss: str = Field(
        "sigmoid",
        description="Preference loss type",
        examples=["sigmoid", "hinge", "ipo", "kto_pair"]
    )
    label_pad_token_id: int = Field(
        -100,
        description="Pad token ID for labels"
    )
    padding_value: int = Field(
        0,
        description="Padding value for inputs"
    )
    truncation_mode: str = Field(
        "keep_end",
        description="Truncation mode for sequences",
        examples=["keep_end", "keep_start"]
    )
    max_length: Optional[int] = Field(
        None,
        description="Maximum length for DPO",
        ge=1
    )
    max_prompt_length: Optional[int] = Field(
        None,
        description="Maximum prompt length for DPO",
        ge=1
    )
    max_target_length: Optional[int] = Field(
        None,
        description="Maximum target length for DPO",
        ge=1
    )
    
    @validator('loss_type')
    def validate_loss_type(cls, v):
        valid_types = ['sigmoid', 'hinge', 'ipo', 'kto_pair']
        if v not in valid_types:
            raise ValueError(f"loss_type must be one of {valid_types}")
        return v


class ORPOConfig(BaseModelConfig):
    """ORPO configuration."""
    beta: float = Field(
        0.1,
        description="Beta parameter for ORPO loss",
        ge=0
    )
    label_smoothing: float = Field(
        0.0,
        description="Label smoothing for ORPO",
        ge=0, le=0.5
    )
    loss_type: str = Field(
        "sigmoid",
        description="Loss type for ORPO",
        examples=["sigmoid", "hinge", "ipo", "kto_pair"]
    )
    pref_loss: str = Field(
        "sigmoid",
        description="Preference loss type",
        examples=["sigmoid", "hinge", "ipo", "kto_pair"]
    )
    label_pad_token_id: int = Field(
        -100,
        description="Pad token ID for labels"
    )
    padding_value: int = Field(
        0,
        description="Padding value for inputs"
    )
    truncation_mode: str = Field(
        "keep_end",
        description="Truncation mode for sequences",
        examples=["keep_end", "keep_start"]
    )
    max_length: Optional[int] = Field(
        None,
        description="Maximum length for ORPO",
        ge=1
    )
    max_prompt_length: Optional[int] = Field(
        None,
        description="Maximum prompt length for ORPO",
        ge=1
    )
    max_target_length: Optional[int] = Field(
        None,
        description="Maximum target length for ORPO",
        ge=1
    )


class GenerationConfig(BaseModelConfig):
    """Generation/inference configuration."""
    max_new_tokens: int = Field(
        256,
        description="Maximum number of new tokens to generate",
        ge=1
    )
    do_sample: bool = Field(
        True,
        description="Whether to use sampling"
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature",
        gt=0
    )
    top_p: float = Field(
        0.9,
        description="Top-p (nucleus) sampling",
        ge=0, le=1
    )
    top_k: int = Field(
        50,
        description="Top-k sampling",
        ge=0
    )
    repetition_penalty: float = Field(
        1.0,
        description="Repetition penalty",
        ge=0
    )
    length_penalty: float = Field(
        1.0,
        description="Length penalty for beam search",
        ge=0
    )
    no_repeat_ngram_size: int = Field(
        0,
        description="No repeat ngram size",
        ge=0
    )
    num_beams: int = Field(
        1,
        description="Number of beams for beam search",
        ge=1
    )
    early_stopping: bool = Field(
        False,
        description="Early stopping for beam search"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for generation"
    )


class EvaluationConfig(BaseModelConfig):
    """Evaluation configuration."""
    tasks: List[str] = Field(
        [],
        description="Evaluation tasks",
        examples=[["mmlu", "hellaswag", "arc"], ["truthfulqa", "gsm8k"]]
    )
    num_fewshot: Optional[int] = Field(
        None,
        description="Number of few-shot examples",
        ge=0
    )
    batch_size: Optional[Union[int, str]] = Field(
        None,
        description="Batch size for evaluation",
        examples=[8, "auto"]
    )
    max_samples: Optional[int] = Field(
        None,
        description="Maximum number of samples for evaluation",
        ge=1
    )
    bootstrap_iters: int = Field(
        100000,
        description="Number of bootstrap iterations for stderr estimation",
        ge=0
    )
    description_dict: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary of task descriptions"
    )


class crucibleConfig(BaseModelConfig):
    """
    Main configuration class for crucible.
    
    This is the top-level configuration that includes all sub-configurations.
    It provides validation, serialization, and migration capabilities.
    """
    version: ConfigVersion = Field(
        ConfigVersion.latest(),
        description="Configuration schema version"
    )
    model: ModelConfig = Field(
        ...,
        description="Model configuration"
    )
    quantization: Optional[QuantizationConfig] = Field(
        None,
        description="Quantization configuration"
    )
    data: DataConfig = Field(
        ...,
        description="Data configuration"
    )
    training: TrainingConfig = Field(
        ...,
        description="Training configuration"
    )
    adapter: Optional[AdapterConfig] = Field(
        None,
        description="Adapter configuration (LoRA, QLoRA, etc.)"
    )
    rlhf: Optional[RLHFConfig] = Field(
        None,
        description="RLHF configuration"
    )
    dpo: Optional[DPOConfig] = Field(
        None,
        description="DPO configuration"
    )
    orpo: Optional[ORPOConfig] = Field(
        None,
        description="ORPO configuration"
    )
    generation: Optional[GenerationConfig] = Field(
        None,
        description="Generation/inference configuration"
    )
    evaluation: Optional[EvaluationConfig] = Field(
        None,
        description="Evaluation configuration"
    )
    training_method: TrainingMethod = Field(
        TrainingMethod.SFT,
        description="Training method to use"
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )
    device: Optional[str] = Field(
        None,
        description="Device to use (cuda, cpu, cuda:0, etc.)"
    )
    n_gpu: Optional[int] = Field(
        None,
        description="Number of GPUs to use",
        ge=1
    )
    world_size: int = Field(
        1,
        description="Total number of processes for distributed training",
        ge=1
    )
    local_rank: int = Field(
        -1,
        description="Local rank for distributed training"
    )
    master_port: Optional[int] = Field(
        None,
        description="Master port for distributed training",
        ge=1024, le=65535
    )
    include_tokens_per_second: bool = Field(
        False,
        description="Whether to include tokens per second in metrics"
    )
    include_num_input_tokens_seen: bool = Field(
        False,
        description="Whether to include number of input tokens seen"
    )
    neftune_noise_alpha: Optional[float] = Field(
        None,
        description="NEFTune noise alpha for embeddings",
        ge=0
    )
    use_dora: bool = Field(
        False,
        description="Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation)"
    )
    use_rslora: bool = Field(
        False,
        description="Whether to use RSLoRA (Rank-Stabilized LoRA)"
    )
    use_unsloth: bool = Field(
        False,
        description="Whether to use Unsloth optimizations"
    )
    disable_gradient_checkpointing: bool = Field(
        False,
        description="Disable gradient checkpointing (for Unsloth)"
    )
    
    @root_validator
    def validate_training_method_config(cls, values):
        """Validate that required configs are present for the training method."""
        method = values.get('training_method')
        
        if method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            if values.get('adapter') is None:
                raise ValueError(f"Adapter configuration required for {method}")
        
        if method == TrainingMethod.RLHF:
            if values.get('rlhf') is None:
                raise ValueError("RLHF configuration required for RLHF training")
        
        if method == TrainingMethod.DPO:
            if values.get('dpo') is None:
                raise ValueError("DPO configuration required for DPO training")
        
        if method == TrainingMethod.ORPO:
            if values.get('orpo') is None:
                raise ValueError("ORPO configuration required for ORPO training")
        
        return values
    
    @classmethod
    def generate_json_schema(cls, indent: int = 2) -> str:
        """Generate JSON Schema for IDE support and validation."""
        schema = cls.model_json_schema()
        return json.dumps(schema, indent=indent, ensure_ascii=False)
    
    @classmethod
    def save_json_schema(cls, path: Union[str, Path], indent: int = 2) -> None:
        """Save JSON Schema to file for IDE integration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = cls.generate_json_schema(indent=indent)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(schema)
        logger.info(f"JSON Schema saved to {path}")
    
    def migrate_from_legacy(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate legacy configuration to current schema.
        
        This method handles backward compatibility with older configuration formats.
        """
        migrated = {}
        
        # Handle flat structure (common in legacy configs)
        if 'model_name_or_path' in legacy_config:
            migrated['model'] = {
                'model_name_or_path': legacy_config['model_name_or_path']
            }
            if 'tokenizer_path' in legacy_config:
                migrated['model']['tokenizer_path'] = legacy_config['tokenizer_path']
            if 'cache_dir' in legacy_config:
                migrated['model']['cache_dir'] = legacy_config['cache_dir']
        
        # Handle data configuration
        data_keys = ['dataset', 'dataset_dir', 'val_dataset', 'template', 'cutoff_len']
        if any(k in legacy_config for k in data_keys):
            migrated['data'] = {}
            for key in data_keys:
                if key in legacy_config:
                    migrated['data'][key] = legacy_config[key]
        
        # Handle training configuration
        training_keys = [
            'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
            'learning_rate', 'bf16', 'fp16', 'gradient_checkpointing'
        ]
        if any(k in legacy_config for k in training_keys):
            migrated['training'] = {}
            for key in training_keys:
                if key in legacy_config:
                    migrated['training'][key] = legacy_config[key]
        
        # Handle adapter configuration
        adapter_keys = ['lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules']
        if any(k in legacy_config for k in adapter_keys):
            migrated['adapter'] = {}
            adapter_mapping = {
                'lora_r': 'r',
                'lora_alpha': 'lora_alpha',
                'lora_dropout': 'lora_dropout',
                'lora_target_modules': 'lora_target_modules'
            }
            for old_key, new_key in adapter_mapping.items():
                if old_key in legacy_config:
                    migrated['adapter'][new_key] = legacy_config[old_key]
        
        # Handle training method
        if 'finetuning_type' in legacy_config:
            method_map = {
                'lora': TrainingMethod.LORA,
                'qlora': TrainingMethod.QLORA,
                'full': TrainingMethod.SFT,
                'freeze': TrainingMethod.FREEZE,
                'sft': TrainingMethod.SFT,
                'rlhf': TrainingMethod.RLHF,
                'dpo': TrainingMethod.DPO,
                'orpo': TrainingMethod.ORPO
            }
            legacy_method = legacy_config['finetuning_type'].lower()
            migrated['training_method'] = method_map.get(legacy_method, TrainingMethod.SFT)
        
        # Copy other top-level fields
        top_level_fields = ['seed', 'device', 'n_gpu', 'world_size', 'local_rank']
        for field in top_level_fields:
            if field in legacy_config:
                migrated[field] = legacy_config[field]
        
        # Add version
        migrated['version'] = ConfigVersion.latest().value
        
        return migrated
    
    @classmethod
    def from_legacy_dict(cls: Type[T], legacy_config: Dict[str, Any]) -> T:
        """Create configuration from legacy dictionary format."""
        # First migrate
        temp_instance = cls.__new__(cls)
        migrated = temp_instance.migrate_from_legacy(legacy_config)
        
        # Then validate
        return cls.from_dict(migrated)


def validate_config(config_dict: Dict[str, Any], strict: bool = True) -> crucibleConfig:
    """
    Validate configuration dictionary against schema.
    
    Args:
        config_dict: Configuration dictionary to validate
        strict: If True, raise ValidationError on validation failure.
                If False, return best-effort validation with warnings.
    
    Returns:
        Validated crucibleConfig instance
    
    Raises:
        ValidationError: If strict=True and validation fails
        ValueError: If configuration cannot be processed
    """
    try:
        # Try direct validation
        return crucibleConfig.from_dict(config_dict)
    except ValidationError as e:
        if strict:
            raise
        
        # Non-strict mode: try to migrate and validate
        logger.warning(f"Configuration validation failed, attempting migration: {e}")
        try:
            return crucibleConfig.from_legacy_dict(config_dict)
        except Exception as migration_error:
            logger.error(f"Migration also failed: {migration_error}")
            raise ValueError(f"Configuration validation failed: {e}")


def load_and_validate_config(
    config_path: Union[str, Path],
    strict: bool = True,
    overrides: Optional[Dict[str, Any]] = None
) -> crucibleConfig:
    """
    Load configuration from file and validate.
    
    Args:
        config_path: Path to configuration file
        strict: If True, raise ValidationError on validation failure
        overrides: Optional dictionary of configuration overrides
    
    Returns:
        Validated crucibleConfig instance
    """
    config_path = Path(config_path)
    
    # Load configuration
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Apply overrides
    if overrides:
        config_dict = deep_merge(config_dict, overrides)
    
    # Validate
    return validate_config(config_dict, strict=strict)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override values taking precedence."""
    import copy
    merged = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def generate_schema_file(output_dir: Union[str, Path] = "./schema") -> None:
    """
    Generate JSON Schema file for IDE integration.
    
    This creates a JSON Schema file that can be used with IDEs like VS Code
    for auto-completion and validation of YAML/JSON configuration files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema_path = output_dir / "crucible-config-schema.json"
    crucibleConfig.save_json_schema(schema_path)
    
    # Also create a sample configuration
    sample_config = crucibleConfig(
        model=ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            torch_dtype="bfloat16"
        ),
        data=DataConfig(
            dataset="alpaca",
            template="llama2",
            cutoff_len=2048
        ),
        training=TrainingConfig(
            output_dir="./output/llama-2-7b-sft",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True
        ),
        adapter=AdapterConfig(
            method=TrainingMethod.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        training_method=TrainingMethod.LORA,
        seed=42
    )
    
    sample_path = output_dir / "sample-config.yaml"
    sample_config.save_yaml(sample_path)
    
    logger.info(f"Schema generated at: {schema_path}")
    logger.info(f"Sample config generated at: {sample_path}")
    logger.info("To use with VS Code, add this to your YAML file:")
    logger.info(f"# yaml-language-server: $schema={schema_path.absolute()}")


# CLI integration functions (to be called from main CLI)
def cli_validate(config_path: str, strict: bool = True) -> None:
    """CLI function to validate configuration."""
    try:
        config = load_and_validate_config(config_path, strict=strict)
        print(f"✓ Configuration is valid")
        print(f"  Model: {config.model.model_name_or_path}")
        print(f"  Training method: {config.training_method}")
        print(f"  Output: {config.training.output_dir}")
    except Exception as e:
        print(f"✗ Configuration validation failed:")
        print(f"  {e}")
        sys.exit(1)


def cli_migrate(input_path: str, output_path: str) -> None:
    """CLI function to migrate legacy configuration."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                legacy_config = yaml.safe_load(f)
            else:
                legacy_config = json.load(f)
        
        config = crucibleConfig.from_legacy_dict(legacy_config)
        
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            config.save_yaml(output_path)
        else:
            config.save_json(output_path)
        
        print(f"✓ Configuration migrated successfully")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Version: {config.version}")
    except Exception as e:
        print(f"✗ Migration failed:")
        print(f"  {e}")
        sys.exit(1)


def cli_generate_schema(output_dir: str = "./schema") -> None:
    """CLI function to generate schema files."""
    try:
        generate_schema_file(output_dir)
        print(f"✓ Schema files generated in: {output_dir}")
    except Exception as e:
        print(f"✗ Schema generation failed:")
        print(f"  {e}")
        sys.exit(1)


# Export public API
__all__ = [
    # Main config class
    'crucibleConfig',
    
    # Sub-config classes
    'ModelConfig',
    'QuantizationConfig',
    'DataConfig',
    'TrainingConfig',
    'AdapterConfig',
    'RLHFConfig',
    'DPOConfig',
    'ORPOConfig',
    'GenerationConfig',
    'EvaluationConfig',
    
    # Enums
    'ConfigVersion',
    'ModelFramework',
    'TrainingMethod',
    'QuantizationMethod',
    'AttentionImplementation',
    'Optimizer',
    'LRScheduler',
    'LoggerType',
    
    # Utility functions
    'validate_config',
    'load_and_validate_config',
    'deep_merge',
    'generate_schema_file',
    
    # CLI functions
    'cli_validate',
    'cli_migrate',
    'cli_generate_schema',
]