"""
crucible Kernel Manager - Automatic hardware-optimized attention and quantization
"""

import os
import sys
import logging
import platform
import importlib
import warnings
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Environment variable overrides
LLAMAFACTORY_ATTENTION_BACKEND = os.getenv("LLAMAFACTORY_ATTENTION_BACKEND", "auto")
LLAMAFACTORY_QUANTIZATION_BACKEND = os.getenv("LLAMAFACTORY_QUANTIZATION_BACKEND", "auto")
LLAMAFACTORY_DISABLE_FLASH_ATTENTION = os.getenv("LLAMAFACTORY_DISABLE_FLASH_ATTENTION", "0") == "1"
LLAMAFACTORY_DISABLE_BITSANDBYTES = os.getenv("LLAMAFACTORY_DISABLE_BITSANDBYTES", "0") == "1"


class AttentionBackend(Enum):
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION = "flash_attention"
    XFORMERS = "xformers"
    SDPA = "sdpa"  # PyTorch Scaled Dot Product Attention
    EAGER = "eager"


class QuantizationBackend(Enum):
    BITSANDBYTES_4BIT = "bitsandbytes_4bit"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    NONE = "none"


@dataclass
class HardwareProfile:
    gpu_name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    is_ampere_or_newer: bool
    is_hopper: bool
    cuda_version: str
    pytorch_version: str
    os_info: str


class KernelManager:
    """
    Manages automatic selection of optimized attention and quantization kernels
    based on available hardware and installed packages.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._hardware_profile = self._profile_hardware()
            self._available_backends = self._detect_available_backends()
            self._selected_attention = self._select_optimal_attention()
            self._selected_quantization = self._select_optimal_quantization()
            self._initialized = True
            self._log_selections()
    
    def _profile_hardware(self) -> HardwareProfile:
        """Profile the current hardware configuration."""
        gpu_name = "CPU"
        compute_capability = (0, 0)
        total_memory_gb = 0.0
        is_ampere_or_newer = False
        is_hopper = False
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Check architecture generations
            major, minor = compute_capability
            is_ampere_or_newer = major >= 8  # Ampere (8.0) and newer
            is_hopper = major >= 9  # Hopper (9.0) and newer
        
        return HardwareProfile(
            gpu_name=gpu_name,
            compute_capability=compute_capability,
            total_memory_gb=total_memory_gb,
            is_ampere_or_newer=is_ampere_or_newer,
            is_hopper=is_hopper,
            cuda_version=torch.version.cuda or "N/A",
            pytorch_version=torch.__version__,
            os_info=platform.platform()
        )
    
    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which backends are available in the current environment."""
        available = {
            "flash_attention_3": False,
            "flash_attention_2": False,
            "flash_attention": False,
            "xformers": False,
            "bitsandbytes_4bit": False,
            "bitsandbytes_8bit": False,
            "gptq": False,
            "awq": False,
            "aqlm": False,
        }
        
        # Check Flash Attention variants
        if not LLAMAFACTORY_DISABLE_FLASH_ATTENTION:
            try:
                import flash_attn
                version = getattr(flash_attn, "__version__", "0.0.0")
                major_version = int(version.split(".")[0])
                
                if major_version >= 3:
                    available["flash_attention_3"] = True
                elif major_version >= 2:
                    available["flash_attention_2"] = True
                else:
                    available["flash_attention"] = True
                    
                logger.debug(f"Flash Attention {version} detected")
            except ImportError:
                pass
        
        # Check xformers
        try:
            import xformers
            available["xformers"] = True
            logger.debug(f"xFormers detected")
        except ImportError:
            pass
        
        # Check bitsandbytes
        if not LLAMAFACTORY_DISABLE_BITSANDBYTES:
            try:
                import bitsandbytes as bnb
                available["bitsandbytes_4bit"] = True
                available["bitsandbytes_8bit"] = True
                logger.debug(f"bitsandbytes {bnb.__version__} detected")
            except ImportError:
                pass
        
        # Check other quantization backends
        try:
            import auto_gptq
            available["gptq"] = True
        except ImportError:
            pass
        
        try:
            import awq
            available["awq"] = True
        except ImportError:
            pass
        
        try:
            import aqlm
            available["aqlm"] = True
        except ImportError:
            pass
        
        return available
    
    def _select_optimal_attention(self) -> AttentionBackend:
        """Select the optimal attention backend based on hardware and availability."""
        # Check for environment override
        if LLAMAFACTORY_ATTENTION_BACKEND != "auto":
            try:
                return AttentionBackend(LLAMAFACTORY_ATTENTION_BACKEND)
            except ValueError:
                logger.warning(f"Invalid attention backend: {LLAMAFACTORY_ATTENTION_BACKEND}. Using auto selection.")
        
        profile = self._hardware_profile
        available = self._available_backends
        
        # Prefer Flash Attention 3 on Hopper GPUs (H100, etc.)
        if profile.is_hopper and available["flash_attention_3"]:
            return AttentionBackend.FLASH_ATTENTION_3
        
        # Prefer Flash Attention 2/3 on Ampere+ GPUs (A100, RTX 30/40 series)
        if profile.is_ampere_or_newer:
            if available["flash_attention_3"]:
                return AttentionBackend.FLASH_ATTENTION_3
            if available["flash_attention_2"]:
                return AttentionBackend.FLASH_ATTENTION_2
        
        # Fall back to Flash Attention 1 for older GPUs with enough compute capability
        if profile.compute_capability >= (7, 0) and available["flash_attention"]:
            return AttentionBackend.FLASH_ATTENTION
        
        # Use xformers if available (works on wider range of GPUs)
        if available["xformers"]:
            return AttentionBackend.XFORMERS
        
        # PyTorch SDPA (available in PyTorch 2.0+)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return AttentionBackend.SDPA
        
        # Final fallback
        return AttentionBackend.EAGER
    
    def _select_optimal_quantization(self) -> QuantizationBackend:
        """Select the optimal quantization backend based on hardware and availability."""
        # Check for environment override
        if LLAMAFACTORY_QUANTIZATION_BACKEND != "auto":
            try:
                return QuantizationBackend(LLAMAFACTORY_QUANTIZATION_BACKEND)
            except ValueError:
                logger.warning(f"Invalid quantization backend: {LLAMAFACTORY_QUANTIZATION_BACKEND}. Using auto selection.")
        
        profile = self._hardware_profile
        available = self._available_backends
        
        # Prefer 4-bit quantization for memory efficiency on consumer GPUs
        if profile.total_memory_gb < 24:  # Less than 24GB VRAM
            if available["bitsandbytes_4bit"]:
                return QuantizationBackend.BITSANDBYTES_4BIT
            if available["awq"]:
                return QuantizationBackend.AWQ
            if available["gptq"]:
                return QuantizationBackend.GPTQ
        
        # For larger GPUs, consider 8-bit or other methods
        if available["bitsandbytes_8bit"]:
            return QuantizationBackend.BITSANDBYTES_8BIT
        
        # Check for newer quantization methods
        if available["aqlm"]:
            return QuantizationBackend.AQLM
        
        return QuantizationBackend.NONE
    
    def _log_selections(self):
        """Log the selected backends and hardware profile."""
        profile = self._hardware_profile
        logger.info(f"Hardware Profile: {profile.gpu_name} (CC {profile.compute_capability[0]}.{profile.compute_capability[1]}, {profile.total_memory_gb:.1f}GB)")
        logger.info(f"Selected Attention: {self._selected_attention.value}")
        logger.info(f"Selected Quantization: {self._selected_quantization.value}")
        
        # Warn if suboptimal configuration
        if self._selected_attention == AttentionBackend.EAGER:
            logger.warning("Using eager attention. Install flash-attn or xformers for better performance.")
        
        if self._selected_quantization == QuantizationBackend.NONE:
            logger.info("No quantization selected. For memory savings, install bitsandbytes: pip install bitsandbytes")
    
    @property
    def hardware_profile(self) -> HardwareProfile:
        return self._hardware_profile
    
    @property
    def attention_backend(self) -> AttentionBackend:
        return self._selected_attention
    
    @property
    def quantization_backend(self) -> QuantizationBackend:
        return self._selected_quantization
    
    def get_attention_implementation(self) -> str:
        """Get the attention implementation string for model configuration."""
        backend = self._selected_attention
        
        if backend == AttentionBackend.FLASH_ATTENTION_3:
            return "flash_attention_3"
        elif backend == AttentionBackend.FLASH_ATTENTION_2:
            return "flash_attention_2"
        elif backend == AttentionBackend.FLASH_ATTENTION:
            return "flash_attention"
        elif backend == AttentionBackend.XFORMERS:
            return "xformers"
        elif backend == AttentionBackend.SDPA:
            return "sdpa"
        else:
            return "eager"
    
    def apply_quantization_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantization configuration to model config."""
        backend = self._selected_quantization
        config = model_config.copy()
        
        if backend == QuantizationBackend.BITSANDBYTES_4BIT:
            config.update({
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                }
            })
        elif backend == QuantizationBackend.BITSANDBYTES_8BIT:
            config.update({
                "quantization_config": {
                    "load_in_8bit": True,
                    "llm_int8_threshold": 6.0,
                }
            })
        elif backend == QuantizationBackend.GPTQ:
            config.update({
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                }
            })
        elif backend == QuantizationBackend.AWQ:
            config.update({
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                }
            })
        
        return config
    
    def get_memory_reduction_factor(self) -> float:
        """Estimate memory reduction factor based on selected quantization."""
        backend = self._selected_quantization
        
        if backend == QuantizationBackend.BITSANDBYTES_4BIT:
            return 0.25  # ~75% reduction
        elif backend == QuantizationBackend.BITSANDBYTES_8BIT:
            return 0.5   # ~50% reduction
        elif backend in [QuantizationBackend.GPTQ, QuantizationBackend.AWQ]:
            return 0.25  # ~75% reduction
        else:
            return 1.0   # No reduction
    
    def get_speedup_factor(self) -> float:
        """Estimate speedup factor based on selected attention backend."""
        backend = self._selected_attention
        profile = self._hardware_profile
        
        # Base speedups (conservative estimates)
        if backend == AttentionBackend.FLASH_ATTENTION_3:
            return 3.0 if profile.is_hopper else 2.5
        elif backend == AttentionBackend.FLASH_ATTENTION_2:
            return 2.5 if profile.is_ampere_or_newer else 2.0
        elif backend == AttentionBackend.FLASH_ATTENTION:
            return 2.0
        elif backend == AttentionBackend.XFORMERS:
            return 1.8
        elif backend == AttentionBackend.SDPA:
            return 1.5
        else:
            return 1.0


# Global kernel manager instance
kernel_manager = KernelManager()


# Convenience functions for external use
def get_attention_backend() -> AttentionBackend:
    """Get the selected attention backend."""
    return kernel_manager.attention_backend


def get_quantization_backend() -> QuantizationBackend:
    """Get the selected quantization backend."""
    return kernel_manager.quantization_backend


def get_attention_implementation() -> str:
    """Get attention implementation string for model config."""
    return kernel_manager.get_attention_implementation()


def apply_quantization_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply quantization configuration to model config."""
    return kernel_manager.apply_quantization_config(model_config)


def get_memory_reduction_factor() -> float:
    """Get estimated memory reduction factor."""
    return kernel_manager.get_memory_reduction_factor()


def get_speedup_factor() -> float:
    """Get estimated speedup factor."""
    return kernel_manager.get_speedup_factor()


def get_hardware_profile() -> HardwareProfile:
    """Get hardware profile information."""
    return kernel_manager.hardware_profile


def print_benchmark_info():
    """Print benchmark information about selected configurations."""
    profile = kernel_manager.hardware_profile
    attention = kernel_manager.attention_backend.value
    quantization = kernel_manager.quantization_backend.value
    speedup = kernel_manager.get_speedup_factor()
    memory_reduction = kernel_manager.get_memory_reduction_factor()
    
    print("\n" + "="*60)
    print("LLAMAFACTORY KERNEL MANAGER BENCHMARK INFO")
    print("="*60)
    print(f"GPU: {profile.gpu_name}")
    print(f"Compute Capability: {profile.compute_capability[0]}.{profile.compute_capability[1]}")
    print(f"Total Memory: {profile.total_memory_gb:.1f} GB")
    print(f"CUDA Version: {profile.cuda_version}")
    print(f"PyTorch Version: {profile.pytorch_version}")
    print(f"OS: {profile.os_info}")
    print("-"*60)
    print(f"Selected Attention Backend: {attention}")
    print(f"Selected Quantization Backend: {quantization}")
    print(f"Estimated Speedup: {speedup:.1f}x")
    print(f"Estimated Memory Reduction: {(1-memory_reduction)*100:.0f}%")
    print("="*60 + "\n")


# Monkey-patch for transformers integration
def patch_transformers_for_auto_kernel():
    """
    Patch transformers library to use our automatic kernel selection.
    Call this before loading any models.
    """
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
        
        original_from_pretrained = AutoConfig.from_pretrained
        
        @classmethod
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            config = original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            
            # Apply attention backend
            if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
                config._attn_implementation = get_attention_implementation()
            
            # Apply quantization config if not already set
            if not hasattr(config, "quantization_config") and get_quantization_backend() != QuantizationBackend.NONE:
                config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
                config_dict = apply_quantization_config(config_dict)
                config = type(config)(**config_dict)
            
            return config
        
        AutoConfig.from_pretrained = patched_from_pretrained
        logger.info("Patched transformers AutoConfig for automatic kernel selection")
        
    except ImportError:
        logger.warning("Transformers not installed, skipping monkey-patch")
    except Exception as e:
        logger.warning(f"Failed to patch transformers: {e}")


# Auto-patch if environment variable is set
if os.getenv("LLAMAFACTORY_AUTO_PATCH_TRANSFORMERS", "0") == "1":
    patch_transformers_for_auto_kernel()


# Example usage and self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_benchmark_info()
    
    # Example of how to use in model loading
    print("\nExample usage:")
    print("from crucible.quantization.auto import get_attention_implementation, apply_quantization_config")
    print("")
    print("# For Hugging Face models:")
    print("config = AutoConfig.from_pretrained('model_name', attn_implementation=get_attention_implementation())")
    print("")
    print("# Or with quantization:")
    print("model = AutoModelForCausalLM.from_pretrained(")
    print("    'model_name',")
    print("    attn_implementation=get_attention_implementation(),")
    print("    **apply_quantization_config({})")
    print(")")