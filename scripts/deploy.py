#!/usr/bin/env python3
"""
SOVEREIGN Model Deployment CLI - One-Click Conversion & Deployment
Extends crucible with unified model conversion, quantization, and deployment pipeline.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConversionFormat(Enum):
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    HUGGINGFACE = "huggingface"

class DeploymentBackend(Enum):
    VLLM = "vllm"
    TGI = "tgi"
    LLAMACPP = "llamacpp"

@dataclass
class DeploymentConfig:
    """Configuration for model deployment pipeline."""
    model_path: str
    output_dir: str = "./deployed_models"
    conversion_format: ConversionFormat = ConversionFormat.GGUF
    deployment_backend: DeploymentBackend = DeploymentBackend.VLLM
    quantization_bits: int = 4
    group_size: int = 128
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
    hub_token: Optional[str] = None
    benchmark: bool = True
    benchmark_samples: int = 100
    max_seq_len: int = 2048
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    dtype: str = "float16"
    trust_remote_code: bool = True
    extra_args: Dict[str, str] = field(default_factory=dict)

class ModelConverter:
    """Unified model conversion handler."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.api = HfApi()
        
    def detect_model_architecture(self, model_path: str) -> str:
        """Detect model architecture from config.json."""
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        architectures = config.get("architectures", [])
        if not architectures:
            raise ValueError("No architecture found in config.json")
        
        return architectures[0]
    
    def convert_to_gguf(self, model_path: str, output_path: str) -> str:
        """Convert model to GGUF format using llama.cpp conversion tools."""
        logger.info(f"Converting {model_path} to GGUF format")
        
        # Check for llama.cpp conversion script
        convert_script = Path("llama.cpp/convert.py")
        if not convert_script.exists():
            # Try to find in system path
            try:
                subprocess.run(["python", "-m", "llama_cpp.convert"], 
                             check=True, capture_output=True)
                convert_script = Path("-m llama_cpp.convert")
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("llama.cpp conversion tools not found. Install with: pip install llama-cpp-python")
        
        # Prepare conversion command
        cmd = [
            "python", str(convert_script),
            model_path,
            "--outfile", str(output_path),
            "--outtype", self.config.dtype,
        ]
        
        # Add quantization if specified
        if self.config.quantization_bits == 4:
            cmd.extend(["--quantize"])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"GGUF conversion successful: {result.stdout}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"GGUF conversion failed: {e.stderr}")
            raise
    
    def convert_to_awq(self, model_path: str, output_path: str) -> str:
        """Convert model to AWQ format."""
        logger.info(f"Converting {model_path} to AWQ format")
        
        try:
            from awq import AutoAWQForCausalLM
            
            # Load and quantize model
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                **self.config.extra_args
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Quantize
            quant_config = {
                "zero_point": True,
                "q_group_size": self.config.group_size,
                "w_bit": self.config.quantization_bits,
                "version": "GEMM"
            }
            
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                export_compatible=True
            )
            
            # Save
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"AWQ conversion successful: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("AutoAWQ not installed. Install with: pip install autoawq")
    
    def convert_to_gptq(self, model_path: str, output_path: str) -> str:
        """Convert model to GPTQ format."""
        logger.info(f"Converting {model_path} to GPTQ format")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            # Prepare quantization config
            quantize_config = BaseQuantizeConfig(
                bits=self.config.quantization_bits,
                group_size=self.config.group_size,
                desc_act=True,
            )
            
            # Load model
            model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                quantize_config=quantize_config,
                trust_remote_code=self.config.trust_remote_code,
                **self.config.extra_args
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Prepare calibration dataset (using a subset of wikitext)
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            examples = [tokenizer(example["text"]) for example in dataset.select(range(100))]
            
            # Quantize
            model.quantize(examples)
            
            # Save
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"GPTQ conversion successful: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("AutoGPTQ not installed. Install with: pip install auto-gptq")
    
    def convert(self, model_path: str) -> str:
        """Main conversion method that routes to appropriate converter."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename based on format
        format_suffix = self.config.conversion_format.value
        output_path = output_dir / f"{Path(model_path).name}-{format_suffix}"
        
        converters = {
            ConversionFormat.GGUF: self.convert_to_gguf,
            ConversionFormat.AWQ: self.convert_to_awq,
            ConversionFormat.GPTQ: self.convert_to_gptq,
            ConversionFormat.HUGGINGFACE: lambda x, y: x  # No conversion needed
        }
        
        converter = converters.get(self.config.conversion_format)
        if not converter:
            raise ValueError(f"Unsupported conversion format: {self.config.conversion_format}")
        
        converted_path = converter(model_path, str(output_path))
        
        # Push to hub if requested
        if self.config.push_to_hub and self.config.hub_repo_id:
            self.push_to_hub(converted_path)
        
        return converted_path
    
    def push_to_hub(self, model_path: str):
        """Push converted model to HuggingFace Hub."""
        if not self.config.hub_repo_id:
            raise ValueError("hub_repo_id must be specified for push_to_hub")
        
        logger.info(f"Pushing model to HuggingFace Hub: {self.config.hub_repo_id}")
        
        try:
            # Create or get repo
            repo_url = create_repo(
                repo_id=self.config.hub_repo_id,
                token=self.config.hub_token,
                exist_ok=True,
                private=False
            )
            
            # Upload folder
            api = HfApi(token=self.config.hub_token)
            api.upload_folder(
                folder_path=model_path,
                repo_id=self.config.hub_repo_id,
                repo_type="model",
                commit_message=f"SOVEREIGN: Upload {self.config.conversion_format.value} quantized model"
            )
            
            logger.info(f"Successfully pushed to: {repo_url}")
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

class ModelDeployer:
    """Handles deployment to various backends."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def deploy_vllm(self, model_path: str) -> subprocess.Popen:
        """Deploy model using vLLM."""
        logger.info(f"Deploying {model_path} with vLLM")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(self.config.port),
            "--dtype", self.config.dtype,
            "--max-model-len", str(self.config.max_seq_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]
        
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        # Add any extra arguments
        for key, value in self.config.extra_args.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(10)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"vLLM server failed to start: {stderr}")
        
        logger.info(f"vLLM server started on port {self.config.port}")
        return process
    
    def deploy_tgi(self, model_path: str) -> subprocess.Popen:
        """Deploy model using Text Generation Inference."""
        logger.info(f"Deploying {model_path} with TGI")
        
        # Check if running in Docker or locally
        try:
            # Try to run with Docker first
            cmd = [
                "docker", "run", "-d",
                "--gpus", "all",
                "-p", f"{self.config.port}:80",
                "-v", f"{model_path}:/data",
                "ghcr.io/huggingface/text-generation-inference:latest",
                "--model-id", "/data",
                "--max-input-length", str(self.config.max_seq_len),
                "--max-total-tokens", str(self.config.max_seq_len * 2),
                "--max-batch-total-tokens", str(self.config.max_seq_len * 4),
            ]
            
            if self.config.trust_remote_code:
                cmd.append("--trust-remote-code")
            
            logger.info(f"Starting TGI server with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(15)  # Docker containers take longer to start
            
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fall back to local installation
            logger.warning("Docker not available, trying local TGI installation")
            
            cmd = [
                "text-generation-launcher",
                "--model-id", model_path,
                "--port", str(self.config.port),
                "--max-input-length", str(self.config.max_seq_len),
                "--max-total-tokens", str(self.config.max_seq_len * 2),
            ]
            
            if self.config.trust_remote_code:
                cmd.append("--trust-remote-code")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(10)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"TGI server failed to start: {stderr}")
        
        logger.info(f"TGI server started on port {self.config.port}")
        return process
    
    def deploy_llamacpp(self, model_path: str) -> subprocess.Popen:
        """Deploy model using llama.cpp server."""
        logger.info(f"Deploying {model_path} with llama.cpp")
        
        # Check if model is GGUF format
        if not model_path.endswith(".gguf"):
            raise ValueError("llama.cpp backend requires GGUF format")
        
        cmd = [
            "python", "-m", "llama_cpp.server",
            "--model", model_path,
            "--port", str(self.config.port),
            "--n_ctx", str(self.config.max_seq_len),
        ]
        
        if self.config.quantization_bits == 4:
            cmd.append("--n_gpu_layers", "100")  # Offload all layers to GPU
        
        logger.info(f"Starting llama.cpp server with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(5)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"llama.cpp server failed to start: {stderr}")
        
        logger.info(f"llama.cpp server started on port {self.config.port}")
        return process
    
    def deploy(self, model_path: str) -> subprocess.Popen:
        """Main deployment method that routes to appropriate backend."""
        deployers = {
            DeploymentBackend.VLLM: self.deploy_vllm,
            DeploymentBackend.TGI: self.deploy_tgi,
            DeploymentBackend.LLAMACPP: self.deploy_llamacpp,
        }
        
        deployer = deployers.get(self.config.deployment_backend)
        if not deployer:
            raise ValueError(f"Unsupported deployment backend: {self.config.deployment_backend}")
        
        return deployer(model_path)

class ModelBenchmarker:
    """Automatic benchmarking for latency and throughput."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def benchmark_vllm(self, model_path: str) -> Dict[str, float]:
        """Benchmark vLLM deployment."""
        logger.info("Benchmarking vLLM deployment")
        
        try:
            from vllm import LLM, SamplingParams
            import numpy as np
            
            # Initialize model
            llm = LLM(
                model=model_path,
                dtype=self.config.dtype,
                max_model_len=self.config.max_seq_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Prepare test prompts
            prompts = [
                "The meaning of life is",
                "In a galaxy far far away,",
                "The quick brown fox jumps over",
                "Artificial intelligence will",
            ] * (self.config.benchmark_samples // 4)
            
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=128
            )
            
            # Warmup
            _ = llm.generate(prompts[:4], sampling_params)
            
            # Benchmark
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_tokens / total_time
            latency_per_token = total_time / total_tokens
            
            return {
                "throughput_tokens_per_sec": throughput,
                "latency_per_token_sec": latency_per_token,
                "total_time_sec": total_time,
                "total_tokens": total_tokens,
                "num_requests": len(prompts),
            }
            
        except Exception as e:
            logger.error(f"vLLM benchmarking failed: {e}")
            return {"error": str(e)}
    
    def benchmark_tgi(self, model_path: str) -> Dict[str, float]:
        """Benchmark TGI deployment."""
        logger.info("Benchmarking TGI deployment")
        
        import requests
        import numpy as np
        
        # TGI endpoint
        url = f"http://localhost:{self.config.port}/generate"
        
        # Prepare test prompts
        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "The quick brown fox jumps over",
            "Artificial intelligence will",
        ] * (self.config.benchmark_samples // 4)
        
        # Warmup
        _ = requests.post(url, json={"inputs": prompts[0], "parameters": {"max_new_tokens": 10}})
        
        # Benchmark
        latencies = []
        token_counts = []
        
        for prompt in prompts:
            start_time = time.time()
            response = requests.post(
                url,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 128,
                        "temperature": 0.8,
                        "top_p": 0.95
                    }
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                latencies.append(end_time - start_time)
                # Estimate token count (TGI doesn't always return token count)
                token_counts.append(len(result.get("generated_text", "").split()))
            else:
                logger.warning(f"Request failed: {response.status_code}")
        
        if not latencies:
            return {"error": "No successful requests"}
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "throughput_tokens_per_sec": throughput,
            "avg_latency_sec": avg_latency,
            "p95_latency_sec": p95_latency,
            "total_time_sec": total_time,
            "total_tokens": total_tokens,
            "num_requests": len(latencies),
        }
    
    def benchmark_llamacpp(self, model_path: str) -> Dict[str, float]:
        """Benchmark llama.cpp deployment."""
        logger.info("Benchmarking llama.cpp deployment")
        
        import requests
        import numpy as np
        
        # llama.cpp endpoint
        url = f"http://localhost:{self.config.port}/completion"
        
        # Prepare test prompts
        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "The quick brown fox jumps over",
            "Artificial intelligence will",
        ] * (self.config.benchmark_samples // 4)
        
        # Warmup
        _ = requests.post(url, json={"prompt": prompts[0], "n_predict": 10})
        
        # Benchmark
        latencies = []
        token_counts = []
        
        for prompt in prompts:
            start_time = time.time()
            response = requests.post(
                url,
                json={
                    "prompt": prompt,
                    "n_predict": 128,
                    "temperature": 0.8,
                    "top_p": 0.95
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                latencies.append(end_time - start_time)
                token_counts.append(result.get("tokens_predicted", 0))
            else:
                logger.warning(f"Request failed: {response.status_code}")
        
        if not latencies:
            return {"error": "No successful requests"}
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "throughput_tokens_per_sec": throughput,
            "avg_latency_sec": avg_latency,
            "p95_latency_sec": p95_latency,
            "total_time_sec": total_time,
            "total_tokens": total_tokens,
            "num_requests": len(latencies),
        }
    
    def benchmark(self, model_path: str) -> Dict[str, float]:
        """Main benchmarking method that routes to appropriate backend."""
        benchmarkers = {
            DeploymentBackend.VLLM: self.benchmark_vllm,
            DeploymentBackend.TGI: self.benchmark_tgi,
            DeploymentBackend.LLAMACPP: self.benchmark_llamacpp,
        }
        
        benchmarker = benchmarkers.get(self.config.deployment_backend)
        if not benchmarker:
            raise ValueError(f"Unsupported deployment backend: {self.config.deployment_backend}")
        
        return benchmarker(model_path)

class SovereignDeploymentPipeline:
    """Main deployment pipeline orchestrator."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.converter = ModelConverter(config)
        self.deployer = ModelDeployer(config)
        self.benchmarker = ModelBenchmarker(config)
    
    def run(self):
        """Execute the full deployment pipeline."""
        logger.info("Starting SOVEREIGN deployment pipeline")
        logger.info(f"Configuration: {self.config}")
        
        results = {
            "config": self.config.__dict__,
            "conversion": {},
            "deployment": {},
            "benchmark": {},
        }
        
        try:
            # Step 1: Convert model
            logger.info("Step 1: Model conversion")
            converted_path = self.converter.convert(self.config.model_path)
            results["conversion"]["status"] = "success"
            results["conversion"]["output_path"] = converted_path
            
            # Step 2: Deploy model
            logger.info("Step 2: Model deployment")
            server_process = self.deployer.deploy(converted_path)
            results["deployment"]["status"] = "success"
            results["deployment"]["backend"] = self.config.deployment_backend.value
            results["deployment"]["port"] = self.config.port
            
            # Step 3: Benchmark (if enabled)
            if self.config.benchmark:
                logger.info("Step 3: Benchmarking")
                benchmark_results = self.benchmarker.benchmark(converted_path)
                results["benchmark"] = benchmark_results
                results["benchmark"]["status"] = "success"
                
                # Log benchmark results
                logger.info("Benchmark Results:")
                for key, value in benchmark_results.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Save results
            results_path = Path(self.config.output_dir) / "deployment_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Deployment completed successfully. Results saved to {results_path}")
            
            # Keep server running if not in benchmark-only mode
            if not self.config.benchmark:
                logger.info("Server is running. Press Ctrl+C to stop.")
                try:
                    server_process.wait()
                except KeyboardInterrupt:
                    logger.info("Shutting down server...")
                    server_process.terminate()
                    server_process.wait()
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            results["error"] = str(e)
            
            # Save error results
            results_path = Path(self.config.output_dir) / "deployment_error.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOVEREIGN One-Click Model Deployment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to GGUF and deploy with vLLM
  python scripts/deploy.py --model_path ./fine-tuned-model --format gguf --backend vllm
  
  # Convert to AWQ and push to HuggingFace Hub
  python scripts/deploy.py --model_path ./fine-tuned-model --format awq --push_to_hub --hub_repo_id username/model-awq
  
  # Deploy with TGI and run benchmark
  python scripts/deploy.py --model_path ./fine-tuned-model --format huggingface --backend tgi --benchmark
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint"
    )
    
    # Conversion options
    parser.add_argument(
        "--format",
        type=str,
        choices=["gguf", "awq", "gptq", "huggingface"],
        default="gguf",
        help="Output format for conversion (default: gguf)"
    )
    
    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[2, 3, 4, 8, 16],
        default=4,
        help="Quantization bits (default: 4)"
    )
    
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)"
    )
    
    # Deployment options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "tgi", "llamacpp"],
        default="vllm",
        help="Deployment backend (default: vllm)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for deployment server (default: 8000)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)"
    )
    
    # HuggingFace Hub options
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        help="HuggingFace Hub repository ID (e.g., username/model-name)"
    )
    
    parser.add_argument(
        "--hub_token",
        type=str,
        help="HuggingFace Hub token (or set HF_TOKEN environment variable)"
    )
    
    # Benchmark options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run benchmark after deployment (default: True)"
    )
    
    parser.add_argument(
        "--no_benchmark",
        action="store_false",
        dest="benchmark",
        help="Disable benchmarking"
    )
    
    parser.add_argument(
        "--benchmark_samples",
        type=int,
        default=100,
        help="Number of benchmark samples (default: 100)"
    )
    
    # Model options
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model data type (default: float16)"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code in model repositories (default: True)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./deployed_models",
        help="Output directory for converted models and results (default: ./deployed_models)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub_repo_id is required when --push_to_hub is set")
    
    if args.backend == "llamacpp" and args.format != "gguf":
        parser.error("llamacpp backend requires GGUF format")
    
    return args

def main():
    """Main entry point for the deployment CLI."""
    args = parse_arguments()
    
    # Get HuggingFace token from environment if not provided
    hub_token = args.hub_token or os.environ.get("HF_TOKEN")
    
    # Create configuration
    config = DeploymentConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        conversion_format=ConversionFormat(args.format),
        deployment_backend=DeploymentBackend(args.backend),
        quantization_bits=args.quantization_bits,
        group_size=args.group_size,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_token=hub_token,
        benchmark=args.benchmark,
        benchmark_samples=args.benchmark_samples,
        max_seq_len=args.max_seq_len,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Run deployment pipeline
    pipeline = SovereignDeploymentPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()