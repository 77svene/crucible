"""
Copyright 2024 The crucible Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

One-Click Model Conversion & Deployment Backend for crucible.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import existing conversion utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.convert_ckpt.llamafy_qwen import convert_qwen_to_llama
from scripts.convert_ckpt.llamafy_baichuan2 import convert_baichuan2_to_llama

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    model_path: str
    output_dir: str
    output_format: str  # "gguf", "awq", "gptq"
    quantization_bits: int = 4
    quantization_method: str = "gptq"  # For GPTQ/AWQ
    dataset: Optional[str] = None
    max_seq_length: int = 2048
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
    hub_token: Optional[str] = None
    device_map: str = "auto"
    trust_remote_code: bool = True
    architecture_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    backend: str  # "vllm", "tgi", "local"
    model_path: str
    port: int = 8000
    host: str = "0.0.0.0"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    dtype: str = "auto"
    api_key: Optional[str] = None
    num_shard: int = 1
    max_input_length: int = 2048
    max_total_tokens: int = 4096
    max_batch_prefill_tokens: int = 4096
    docker_image: str = "ghcr.io/huggingface/text-generation-inference:latest"
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from model benchmarking."""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    tokens_per_second: float
    requests_per_second: float
    total_requests: int
    total_time: float
    input_length: int
    output_length: int
    batch_size: int


class BaseModelConverter(ABC):
    """Abstract base class for model converters."""
    
    @abstractmethod
    def convert(self, config: ConversionConfig) -> str:
        """Convert model and return path to converted model."""
        pass
    
    @abstractmethod
    def supports_architecture(self, architecture: str) -> bool:
        """Check if converter supports given architecture."""
        pass


class GGUFConverter(BaseModelConverter):
    """Converter for GGUF format using llama.cpp."""
    
    def __init__(self):
        self.llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")
    
    def supports_architecture(self, architecture: str) -> bool:
        """GGUF supports most architectures through conversion."""
        return True
    
    def convert(self, config: ConversionConfig) -> str:
        """Convert model to GGUF format."""
        output_path = Path(config.output_dir) / "model.gguf"
        
        # First, ensure model is in Llama-compatible format
        model_config = AutoConfig.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code
        )
        
        architecture = getattr(model_config, "architectures", ["unknown"])[0]
        
        # Convert to Llama format if needed
        temp_dir = None
        model_path = config.model_path
        
        if "qwen" in architecture.lower():
            logger.info("Converting Qwen model to Llama format...")
            temp_dir = tempfile.mkdtemp()
            convert_qwen_to_llama(config.model_path, temp_dir)
            model_path = temp_dir
        elif "baichuan" in architecture.lower():
            logger.info("Converting Baichuan2 model to Llama format...")
            temp_dir = tempfile.mkdtemp()
            convert_baichuan2_to_llama(config.model_path, temp_dir)
            model_path = temp_dir
        
        try:
            # Convert to GGUF using llama.cpp
            convert_script = Path(self.llama_cpp_path) / "convert.py"
            
            if not convert_script.exists():
                raise FileNotFoundError(
                    f"llama.cpp convert.py not found at {convert_script}. "
                    "Please set LLAMA_CPP_PATH environment variable."
                )
            
            cmd = [
                sys.executable,
                str(convert_script),
                model_path,
                "--outfile", str(output_path),
                "--outtype", "f16"  # Default to f16, quantization happens later
            ]
            
            # Add quantization if specified
            if config.output_format == "gguf" and config.quantization_bits:
                quant_type = self._get_quant_type(config.quantization_bits)
                cmd.extend(["--quantize", quant_type])
            
            logger.info(f"Running GGUF conversion: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
            
            logger.info(f"GGUF conversion successful: {output_path}")
            return str(output_path)
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
    
    def _get_quant_type(self, bits: int) -> str:
        """Get llama.cpp quantization type based on bits."""
        quant_map = {
            2: "q2_k",
            3: "q3_k_m",
            4: "q4_0",
            5: "q5_k_m",
            6: "q6_k",
            8: "q8_0"
        }
        return quant_map.get(bits, "q4_0")


class AWQConverter(BaseModelConverter):
    """Converter for AWQ quantization."""
    
    def supports_architecture(self, architecture: str) -> bool:
        """AWQ supports specific architectures."""
        supported = ["llama", "mistral", "qwen", "baichuan", "gpt_neox", "falcon"]
        arch_lower = architecture.lower()
        return any(supported_arch in arch_lower for supported_arch in supported)
    
    def convert(self, config: ConversionConfig) -> str:
        """Convert model to AWQ format."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("autoawq package is required for AWQ conversion. Install with: pip install autoawq")
        
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model for AWQ quantization: {config.model_path}")
        
        # Load model
        model = AutoAWQForCausalLM.from_pretrained(
            config.model_path,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code
        )
        
        # Quantization config
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": config.quantization_bits,
            "version": "GEMM"
        }
        
        # Quantize
        logger.info("Starting AWQ quantization...")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=config.dataset or "pileval",
            split="train[:100]",
            text_column="text"
        )
        
        # Save
        logger.info(f"Saving AWQ model to {output_path}")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        return str(output_path)


class GPTQConverter(BaseModelConverter):
    """Converter for GPTQ quantization."""
    
    def supports_architecture(self, architecture: str) -> bool:
        """GPTQ supports most architectures."""
        return True
    
    def convert(self, config: ConversionConfig) -> str:
        """Convert model to GPTQ format."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError("auto-gptq package is required for GPTQ conversion. Install with: pip install auto-gptq")
        
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model for GPTQ quantization: {config.model_path}")
        
        # Quantization config
        quantize_config = BaseQuantizeConfig(
            bits=config.quantization_bits,
            group_size=128,
            damp_percent=0.01,
            desc_act=True,
            sym=True,
            true_sequential=True
        )
        
        # Load model
        model = AutoGPTQForCausalLM.from_pretrained(
            config.model_path,
            quantize_config=quantize_config,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code
        )
        
        # Prepare calibration dataset
        if config.dataset:
            logger.info(f"Using dataset for calibration: {config.dataset}")
            from datasets import load_dataset
            dataset = load_dataset(config.dataset, split="train[:128]")
            examples = [tokenizer(example["text"]) for example in dataset]
        else:
            # Use default calibration
            logger.info("Using default calibration dataset")
            examples = [
                tokenizer("The quick brown fox jumps over the lazy dog."),
                tokenizer("Machine learning is transforming the world."),
                tokenizer("Large language models can generate human-like text.")
            ]
        
        # Quantize
        logger.info("Starting GPTQ quantization...")
        model.quantize(examples)
        
        # Save
        logger.info(f"Saving GPTQ model to {output_path}")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        return str(output_path)


class ModelConverterFactory:
    """Factory for creating model converters."""
    
    _converters = {
        "gguf": GGUFConverter,
        "awq": AWQConverter,
        "gptq": GPTQConverter
    }
    
    @classmethod
    def get_converter(cls, format_type: str) -> BaseModelConverter:
        """Get converter for specified format."""
        converter_class = cls._converters.get(format_type.lower())
        if not converter_class:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {list(cls._converters.keys())}")
        return converter_class()


class ModelDeployer(ABC):
    """Abstract base class for model deployers."""
    
    @abstractmethod
    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy model and return endpoint URL."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop deployment."""
        pass


class VLLMDeployer(ModelDeployer):
    """Deployer for vLLM backend."""
    
    def __init__(self):
        self.process = None
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy model using vLLM."""
        try:
            import vllm
        except ImportError:
            raise ImportError("vllm package is required for vLLM deployment. Install with: pip install vllm")
        
        # Prepare vLLM arguments
        args = {
            "model": config.model_path,
            "host": config.host,
            "port": config.port,
            "gpu-memory-utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "trust-remote-code": True
        }
        
        if config.max_model_len:
            args["max-model-len"] = config.max_model_len
        
        if config.quantization:
            args["quantization"] = config.quantization
        
        # Add extra arguments
        args.update(config.extra_args)
        
        # Build command
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"]
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        
        # Start server
        import subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        import time
        import requests
        
        endpoint = f"http://{config.host}:{config.port}/health"
        max_retries = 30
        
        for i in range(max_retries):
            try:
                response = requests.get(endpoint, timeout=1)
                if response.status_code == 200:
                    logger.info(f"vLLM server started successfully at http://{config.host}:{config.port}")
                    return f"http://{config.host}:{config.port}"
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"vLLM server failed to start: {stderr}")
        
        raise TimeoutError("vLLM server failed to start within timeout")
    
    def stop(self):
        """Stop vLLM server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("vLLM server stopped")


class TGIDeployer(ModelDeployer):
    """Deployer for Text Generation Inference backend."""
    
    def __init__(self):
        self.container_id = None
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy model using TGI in Docker."""
        import subprocess
        
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is required for TGI deployment")
        
        # Prepare Docker command
        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            "-p", f"{config.port}:80",
            "-v", f"{config.model_path}:/data",
            "--shm-size", "1g",
            config.docker_image,
            "--model-id", "/data",
            "--max-input-length", str(config.max_input_length),
            "--max-total-tokens", str(config.max_total_tokens),
            "--max-batch-prefill-tokens", str(config.max_batch_prefill_tokens)
        ]
        
        if config.num_shard > 1:
            cmd.extend(["--num-shard", str(config.num_shard)])
        
        if config.dtype != "auto":
            cmd.extend(["--dtype", config.dtype])
        
        if config.quantization:
            cmd.extend(["--quantize", config.quantization])
        
        # Add extra arguments
        for key, value in config.extra_args.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Starting TGI container: {' '.join(cmd)}")
        
        # Start container
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start TGI container: {result.stderr}")
        
        self.container_id = result.stdout.strip()
        
        # Wait for container to be ready
        import time
        import requests
        
        endpoint = f"http://localhost:{config.port}/health"
        max_retries = 60
        
        for i in range(max_retries):
            try:
                response = requests.get(endpoint, timeout=1)
                if response.status_code == 200:
                    logger.info(f"TGI server started successfully at http://localhost:{config.port}")
                    return f"http://localhost:{config.port}"
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        # If we get here, container failed to start
        self.stop()
        raise TimeoutError("TGI container failed to start within timeout")
    
    def stop(self):
        """Stop TGI container."""
        if self.container_id:
            import subprocess
            subprocess.run(["docker", "stop", self.container_id], capture_output=True)
            subprocess.run(["docker", "rm", self.container_id], capture_output=True)
            logger.info(f"TGI container {self.container_id} stopped")


class LocalDeployer(ModelDeployer):
    """Deployer for local inference without server."""
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Load model for local inference."""
        logger.info(f"Loading model locally: {config.model_path}")
        
        # This is a placeholder - in practice, you might want to return
        # a handle to the loaded model or a local API wrapper
        return f"local://{config.model_path}"
    
    def stop(self):
        """Stop local deployment (no-op for local)."""
        pass


class DeployerFactory:
    """Factory for creating model deployers."""
    
    _deployers = {
        "vllm": VLLMDeployer,
        "tgi": TGIDeployer,
        "local": LocalDeployer
    }
    
    @classmethod
    def get_deployer(cls, backend: str) -> ModelDeployer:
        """Get deployer for specified backend."""
        deployer_class = cls._deployers.get(backend.lower())
        if not deployer_class:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {list(cls._deployers.keys())}")
        return deployer_class()


class ModelBenchmarker:
    """Benchmarker for model inference."""
    
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def benchmark(
        self,
        prompt: str = "The quick brown fox jumps over the lazy dog.",
        max_tokens: int = 100,
        num_requests: int = 100,
        concurrent_requests: int = 10,
        warmup_requests: int = 10
    ) -> BenchmarkResult:
        """Run benchmark against the deployed model."""
        import asyncio
        import aiohttp
        import numpy as np
        from dataclasses import dataclass
        
        logger.info(f"Starting benchmark with {num_requests} requests, {concurrent_requests} concurrent")
        
        async def send_request(session, prompt, max_tokens):
            """Send a single request and measure latency."""
            start_time = time.time()
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            try:
                async with session.post(
                    f"{self.endpoint}/v1/completions",
                    json=payload,
                    headers=self.headers
                ) as response:
                    result = await response.json()
                    latency = time.time() - start_time
                    
                    # Extract token count from response
                    tokens_generated = len(result.get("choices", [{}])[0].get("text", "").split())
                    
                    return {
                        "latency": latency,
                        "tokens": tokens_generated,
                        "success": response.status == 200
                    }
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return {
                    "latency": time.time() - start_time,
                    "tokens": 0,
                    "success": False
                }
        
        async def run_benchmark():
            """Run the benchmark."""
            latencies = []
            total_tokens = 0
            successful_requests = 0
            
            # Warmup
            logger.info(f"Running {warmup_requests} warmup requests...")
            async with aiohttp.ClientSession() as session:
                warmup_tasks = [
                    send_request(session, prompt, max_tokens)
                    for _ in range(warmup_requests)
                ]
                await asyncio.gather(*warmup_tasks)
            
            # Actual benchmark
            logger.info("Running benchmark...")
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                # Create semaphore for concurrency control
                semaphore = asyncio.Semaphore(concurrent_requests)
                
                async def bounded_request():
                    async with semaphore:
                        return await send_request(session, prompt, max_tokens)
                
                tasks = [bounded_request() for _ in range(num_requests)]
                results = await asyncio.gather(*tasks)
                
                for result in results:
                    if result["success"]:
                        latencies.append(result["latency"])
                        total_tokens += result["tokens"]
                        successful_requests += 1
            
            total_time = time.time() - start_time
            
            if not latencies:
                raise RuntimeError("All benchmark requests failed")
            
            # Calculate statistics
            latencies = np.array(latencies)
            
            return BenchmarkResult(
                latency_p50=float(np.percentile(latencies, 50)),
                latency_p95=float(np.percentile(latencies, 95)),
                latency_p99=float(np.percentile(latencies, 99)),
                throughput=successful_requests / total_time,
                tokens_per_second=total_tokens / total_time,
                requests_per_second=successful_requests / total_time,
                total_requests=successful_requests,
                total_time=total_time,
                input_length=len(prompt.split()),
                output_length=max_tokens,
                batch_size=concurrent_requests
            )
        
        # Run async benchmark
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(run_benchmark())
        finally:
            loop.close()
    
    def save_results(self, result: BenchmarkResult, output_path: str):
        """Save benchmark results to file."""
        import json
        
        output_data = {
            "latency_p50_ms": result.latency_p50 * 1000,
            "latency_p95_ms": result.latency_p95 * 1000,
            "latency_p99_ms": result.latency_p99 * 1000,
            "throughput_requests_per_second": result.throughput,
            "tokens_per_second": result.tokens_per_second,
            "requests_per_second": result.requests_per_second,
            "total_requests": result.total_requests,
            "total_time_seconds": result.total_time,
            "input_length_tokens": result.input_length,
            "output_length_tokens": result.output_length,
            "batch_size": result.batch_size,
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")


class OneClickDeployer:
    """Main class for one-click conversion and deployment."""
    
    def __init__(self):
        self.converter_factory = ModelConverterFactory()
        self.deployer_factory = DeployerFactory()
    
    def convert_and_deploy(
        self,
        conversion_config: ConversionConfig,
        deployment_config: DeploymentConfig,
        run_benchmark: bool = True,
        benchmark_requests: int = 100,
        benchmark_concurrent: int = 10
    ) -> Dict[str, Any]:
        """Convert model and deploy it in one click."""
        results = {
            "conversion": {},
            "deployment": {},
            "benchmark": {}
        }
        
        # Step 1: Convert model
        logger.info(f"Converting model to {conversion_config.output_format} format...")
        
        converter = self.converter_factory.get_converter(conversion_config.output_format)
        
        # Check architecture support
        model_config = AutoConfig.from_pretrained(
            conversion_config.model_path,
            trust_remote_code=conversion_config.trust_remote_code
        )
        architecture = getattr(model_config, "architectures", ["unknown"])[0]
        
        if not converter.supports_architecture(architecture):
            logger.warning(
                f"Converter {conversion_config.output_format} may not fully support architecture {architecture}. "
                "Conversion will proceed but may fail."
            )
        
        converted_path = converter.convert(conversion_config)
        results["conversion"]["output_path"] = converted_path
        results["conversion"]["format"] = conversion_config.output_format
        
        # Step 2: Push to hub if requested
        if conversion_config.push_to_hub:
            logger.info(f"Pushing model to Hugging Face Hub: {conversion_config.hub_repo_id}")
            self._push_to_hub(
                converted_path,
                conversion_config.hub_repo_id,
                conversion_config.hub_token
            )
            results["conversion"]["hub_url"] = f"https://huggingface.co/{conversion_config.hub_repo_id}"
        
        # Step 3: Deploy model
        logger.info(f"Deploying model using {deployment_config.backend} backend...")
        
        # Update deployment config with converted model path
        deployment_config.model_path = converted_path
        
        deployer = self.deployer_factory.get_deployer(deployment_config.backend)
        
        try:
            endpoint = deployer.deploy(deployment_config)
            results["deployment"]["endpoint"] = endpoint
            results["deployment"]["backend"] = deployment_config.backend
            
            # Step 4: Run benchmark if requested
            if run_benchmark:
                logger.info("Running benchmark...")
                
                benchmarker = ModelBenchmarker(endpoint, deployment_config.api_key)
                benchmark_result = benchmarker.benchmark(
                    num_requests=benchmark_requests,
                    concurrent_requests=benchmark_concurrent
                )
                
                results["benchmark"] = {
                    "latency_p50_ms": benchmark_result.latency_p50 * 1000,
                    "latency_p95_ms": benchmark_result.latency_p95 * 1000,
                    "latency_p99_ms": benchmark_result.latency_p99 * 1000,
                    "throughput_rps": benchmark_result.throughput,
                    "tokens_per_second": benchmark_result.tokens_per_second,
                    "requests_per_second": benchmark_result.requests_per_second
                }
                
                # Save benchmark results
                benchmark_path = Path(conversion_config.output_dir) / "benchmark_results.json"
                benchmarker.save_results(benchmark_result, str(benchmark_path))
                results["benchmark"]["results_file"] = str(benchmark_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployer.stop()
            raise
    
    def _push_to_hub(self, model_path: str, repo_id: str, token: Optional[str] = None):
        """Push model to Hugging Face Hub."""
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, token=token, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create repo: {e}")
        
        # Upload folder
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload converted model from crucible"
        )
        
        logger.info(f"Model successfully pushed to https://huggingface.co/{repo_id}")


def main():
    """Main CLI entry point for one-click deployment."""
    parser = argparse.ArgumentParser(
        description="crucible One-Click Model Conversion & Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to GGUF and deploy with vLLM
  python -m crucible.deploy.backends \\
    --model_path ./my-finetuned-model \\
    --output_format gguf \\
    --quantization_bits 4 \\
    --backend vllm \\
    --push_to_hub \\
    --hub_repo_id my-username/my-quantized-model

  # Convert to AWQ and deploy with TGI
  python -m crucible.deploy.backends \\
    --model_path ./my-finetuned-model \\
    --output_format awq \\
    --quantization_bits 4 \\
    --backend tgi \\
    --port 8080

  # Convert to GPTQ and run locally
  python -m crucible.deploy.backends \\
    --model_path ./my-finetuned-model \\
    --output_format gptq \\
    --quantization_bits 8 \\
    --backend local
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to convert and deploy"
    )
    
    # Conversion arguments
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["gguf", "awq", "gptq"],
        required=True,
        help="Output format for model conversion"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./converted_model",
        help="Output directory for converted model"
    )
    
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Number of bits for quantization"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset for calibration during quantization"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for conversion"
    )
    
    # Hub arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID"
    )
    
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face Hub token"
    )
    
    # Deployment arguments
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "tgi", "local"],
        required=True,
        help="Deployment backend"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for deployment server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for deployment server"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length for vLLM"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for authentication"
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--run_benchmark",
        action="store_true",
        default=True,
        help="Run benchmark after deployment"
    )
    
    parser.add_argument(
        "--no_benchmark",
        action="store_false",
        dest="run_benchmark",
        help="Skip benchmarking"
    )
    
    parser.add_argument(
        "--benchmark_requests",
        type=int,
        default=100,
        help="Number of requests for benchmark"
    )
    
    parser.add_argument(
        "--benchmark_concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests for benchmark"
    )
    
    # Other arguments
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading models"
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create configs
    conversion_config = ConversionConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        output_format=args.output_format,
        quantization_bits=args.quantization_bits,
        dataset=args.dataset,
        max_seq_length=args.max_seq_length,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_token=args.hub_token,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code
    )
    
    deployment_config = DeploymentConfig(
        backend=args.backend,
        model_path=args.model_path,  # Will be updated after conversion
        port=args.port,
        host=args.host,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        api_key=args.api_key
    )
    
    # Run one-click deployment
    deployer = OneClickDeployer()
    
    try:
        results = deployer.convert_and_deploy(
            conversion_config=conversion_config,
            deployment_config=deployment_config,
            run_benchmark=args.run_benchmark,
            benchmark_requests=args.benchmark_requests,
            benchmark_concurrent=args.benchmark_concurrent
        )
        
        # Print results
        print("\n" + "="*60)
        print("ONE-CLICK DEPLOYMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print(f"\nModel converted to: {results['conversion']['output_path']}")
        print(f"Format: {results['conversion']['format']}")
        
        if "hub_url" in results["conversion"]:
            print(f"Hub URL: {results['conversion']['hub_url']}")
        
        print(f"\nDeployment endpoint: {results['deployment']['endpoint']}")
        print(f"Backend: {results['deployment']['backend']}")
        
        if results["benchmark"]:
            print(f"\nBenchmark Results:")
            print(f"  Latency (P50): {results['benchmark']['latency_p50_ms']:.2f} ms")
            print(f"  Latency (P95): {results['benchmark']['latency_p95_ms']:.2f} ms")
            print(f"  Latency (P99): {results['benchmark']['latency_p99_ms']:.2f} ms")
            print(f"  Throughput: {results['benchmark']['throughput_rps']:.2f} requests/sec")
            print(f"  Tokens/sec: {results['benchmark']['tokens_per_second']:.2f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"One-click deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()