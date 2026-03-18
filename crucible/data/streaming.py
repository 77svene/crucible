"""
Streaming data pipeline for crucible with memory-mapped, cloud-native dataset loading.
Replaces traditional dataset loading with zero-copy streaming from S3/GCS.
"""

import os
import io
import json
import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Union, Iterator, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pj
from pyarrow import fs

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming data pipeline."""
    prefetch_buffer_size: int = 1000
    num_workers: int = 4
    batch_size: int = 32
    max_sequence_length: int = 2048
    tokenization_workers: int = 4
    cache_dir: Optional[str] = None
    cloud_credentials: Optional[Dict[str, str]] = None
    shuffle_buffer_size: int = 10000
    seed: int = 42


class CloudFileSystem:
    """Unified interface for cloud storage systems (S3, GCS, local)."""
    
    def __init__(self, 
                 scheme: str = "file",
                 credentials: Optional[Dict[str, str]] = None):
        self.scheme = scheme
        self.credentials = credentials or {}
        self._fs = self._initialize_filesystem()
    
    def _initialize_filesystem(self) -> fs.FileSystem:
        """Initialize appropriate filesystem based on scheme."""
        if self.scheme in ("s3", "s3a"):
            from pyarrow.fs import S3FileSystem
            return S3FileSystem(
                access_key=self.credentials.get("aws_access_key_id"),
                secret_key=self.credentials.get("aws_secret_access_key"),
                region=self.credentials.get("region_name", "us-east-1")
            )
        elif self.scheme in ("gs", "gcs"):
            from pyarrow.fs import GcsFileSystem
            return GcsFileSystem()
        else:
            return fs.LocalFileSystem()
    
    def open_input_stream(self, path: str) -> pa.NativeFile:
        """Open a file for reading."""
        return self._fs.open_input_stream(path)
    
    def get_file_info(self, path: str) -> fs.FileInfo:
        """Get file metadata."""
        return self._fs.get_file_info(path)
    
    def list_files(self, path: str, pattern: str = "*") -> List[fs.FileInfo]:
        """List files matching pattern."""
        selector = fs.FileSelector(path, allow_not_found=True)
        files = self._fs.get_file_info(selector)
        return [f for f in files if f.is_file and f.path.endswith(pattern)]


class ShardedDatasetIterator:
    """Iterator for sharded datasets with automatic shard rotation."""
    
    def __init__(self,
                 file_paths: List[str],
                 cloud_fs: CloudFileSystem,
                 shuffle: bool = False,
                 seed: int = 42):
        self.file_paths = file_paths
        self.cloud_fs = cloud_fs
        self.shuffle = shuffle
        self.seed = seed
        self._current_shard_idx = 0
        self._rng = torch.Generator().manual_seed(seed)
        
        if shuffle:
            self._shuffle_shards()
    
    def _shuffle_shards(self) -> None:
        """Shuffle shard order."""
        indices = torch.randperm(len(self.file_paths), generator=self._rng).tolist()
        self.file_paths = [self.file_paths[i] for i in indices]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all shards."""
        while True:
            if self._current_shard_idx >= len(self.file_paths):
                self._current_shard_idx = 0
                if self.shuffle:
                    self._shuffle_shards()
            
            shard_path = self.file_paths[self._current_shard_idx]
            try:
                yield from self._read_shard(shard_path)
            except Exception as e:
                logger.error(f"Error reading shard {shard_path}: {e}")
                # Skip corrupted shard
                self._current_shard_idx += 1
                continue
            
            self._current_shard_idx += 1
    
    def _read_shard(self, path: str) -> Iterator[Dict[str, Any]]:
        """Read a single shard file."""
        file_info = self.cloud_fs.get_file_info(path)
        
        if file_info.path.endswith('.parquet'):
            yield from self._read_parquet_shard(path)
        elif file_info.path.endswith('.jsonl') or file_info.path.endswith('.json'):
            yield from self._read_jsonl_shard(path)
        else:
            logger.warning(f"Unsupported file format: {path}")
    
    def _read_parquet_shard(self, path: str) -> Iterator[Dict[str, Any]]:
        """Read Parquet shard with memory mapping."""
        with self.cloud_fs.open_input_stream(path) as f:
            # Use memory mapping for large files
            table = pq.read_table(
                pa.BufferReader(f.read()),
                memory_map=True,
                pre_buffer=True
            )
            
            for batch in table.to_batches():
                for i in range(len(batch)):
                    row = {col: batch[col][i].as_py() for col in batch.column_names}
                    yield row
    
    def _read_jsonl_shard(self, path: str) -> Iterator[Dict[str, Any]]:
        """Read JSONL shard with streaming."""
        with self.cloud_fs.open_input_stream(path) as f:
            for line in f:
                line = line.decode('utf-8').strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line in {path}")
                        continue


class TokenizationPipeline:
    """On-the-fly tokenization pipeline with parallel processing."""
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048,
                 num_workers: int = 4,
                 text_column: str = "text",
                 instruction_template: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers
        self.text_column = text_column
        self.instruction_template = instruction_template
        
        # Configure tokenizer
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example."""
        text = example.get(self.text_column, "")
        
        # Handle instruction format
        if self.instruction_template and "instruction" in example:
            text = self._format_instruction(example)
        
        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels for causal LM
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _format_instruction(self, example: Dict[str, Any]) -> str:
        """Format instruction-based examples."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    def tokenize_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of examples."""
        texts = []
        for example in examples:
            text = example.get(self.text_column, "")
            if self.instruction_template and "instruction" in example:
                text = self._format_instruction(example)
            texts.append(text)
        
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encoded["input_ids"].clone()
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels
        }


class PrefetchingQueue:
    """Thread-safe prefetching queue with dynamic batching."""
    
    def __init__(self,
                 maxsize: int = 1000,
                 batch_size: int = 32,
                 collate_fn: Optional[Callable] = None):
        self.queue = queue.Queue(maxsize=maxsize)
        self.batch_size = batch_size
        self.collate_fn = collate_fn or self._default_collate
        self._stop_event = threading.Event()
        self._buffer = []
    
    def put(self, item: Dict[str, Any]) -> None:
        """Add item to buffer and batch if ready."""
        if self._stop_event.is_set():
            return
        
        self._buffer.append(item)
        
        if len(self._buffer) >= self.batch_size:
            batch = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
            
            try:
                collated = self.collate_fn(batch)
                self.queue.put(collated, timeout=1.0)
            except queue.Full:
                logger.warning("Prefetch queue full, dropping batch")
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch from the queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        """Stop the prefetching queue."""
        self._stop_event.set()
        # Flush remaining buffer
        if self._buffer:
            try:
                collated = self.collate_fn(self._buffer)
                self.queue.put(collated, timeout=0.1)
            except:
                pass
    
    @staticmethod
    def _default_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Default collation function."""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        
        return collated


class StreamingDataset(IterableDataset):
    """
    Memory-mapped streaming dataset for cloud-native training.
    Supports sharded JSONL/Parquet from S3/GCS with zero-copy loading.
    """
    
    def __init__(self,
                 data_paths: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 config: Optional[StreamingConfig] = None,
                 split: str = "train",
                 text_column: str = "text",
                 instruction_template: Optional[str] = None,
                 transform: Optional[Callable] = None):
        
        self.config = config or StreamingConfig()
        self.tokenizer = tokenizer
        self.split = split
        self.text_column = text_column
        self.instruction_template = instruction_template
        self.transform = transform
        
        # Parse data paths and detect scheme
        self.data_paths = self._parse_data_paths(data_paths)
        self.cloud_scheme = self._detect_cloud_scheme()
        
        # Initialize components
        self.cloud_fs = CloudFileSystem(
            scheme=self.cloud_scheme,
            credentials=self.config.cloud_credentials
        )
        
        self.tokenization_pipeline = TokenizationPipeline(
            tokenizer=tokenizer,
            max_length=self.config.max_sequence_length,
            num_workers=self.config.tokenization_workers,
            text_column=text_column,
            instruction_template=instruction_template
        )
        
        # Discover shards
        self.shards = self._discover_shards()
        logger.info(f"Discovered {len(self.shards)} shards for split '{split}'")
        
        # Initialize prefetching queue
        self.prefetch_queue = PrefetchingQueue(
            maxsize=self.config.prefetch_buffer_size,
            batch_size=self.config.batch_size
        )
        
        # Statistics
        self.stats = {
            "examples_processed": 0,
            "batches_prefetched": 0,
            "shards_processed": 0,
            "bytes_loaded": 0
        }
    
    def _parse_data_paths(self, paths: Union[str, List[str]]) -> List[str]:
        """Parse data paths from various formats."""
        if isinstance(paths, str):
            if paths.startswith("s3://") or paths.startswith("gs://"):
                return [paths]
            else:
                # Local path or glob
                path = Path(paths)
                if path.is_dir():
                    return list(str(p) for p in path.glob("**/*"))
                else:
                    return [str(path)]
        else:
            return paths
    
    def _detect_cloud_scheme(self) -> str:
        """Detect cloud storage scheme from data paths."""
        for path in self.data_paths:
            if path.startswith("s3://"):
                return "s3"
            elif path.startswith("gs://"):
                return "gs"
        return "file"
    
    def _discover_shards(self) -> List[str]:
        """Discover all shard files."""
        shards = []
        
        for path in self.data_paths:
            if path.startswith(("s3://", "gs://")):
                # Cloud path - list files
                bucket, key = self._parse_cloud_path(path)
                if key.endswith('/') or '*' in key:
                    # Directory or glob pattern
                    pattern = key.split('/')[-1] if '*' in key else "*"
                    dir_path = '/'.join(key.split('/')[:-1]) if '/' in key else ""
                    full_path = f"{bucket}/{dir_path}" if dir_path else bucket
                    
                    try:
                        files = self.cloud_fs.list_files(full_path, pattern)
                        shards.extend(f"{bucket}/{f.path}" for f in files)
                    except Exception as e:
                        logger.error(f"Error listing cloud files: {e}")
                else:
                    # Single file
                    shards.append(path)
            else:
                # Local path
                path_obj = Path(path)
                if path_obj.is_dir():
                    # Directory - find all supported files
                    for pattern in ["*.jsonl", "*.json", "*.parquet"]:
                        shards.extend(str(p) for p in path_obj.rglob(pattern))
                elif path_obj.exists():
                    shards.append(str(path_obj))
        
        # Filter by split if using HuggingFace dataset format
        if self.split and self.split != "train":
            shards = [s for s in shards if self.split in s]
        
        return shards
    
    def _parse_cloud_path(self, path: str) -> tuple:
        """Parse cloud path into bucket and key."""
        if path.startswith("s3://"):
            path = path[5:]
        elif path.startswith("gs://"):
            path = path[5:]
        
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        return bucket, key
    
    def _create_iterator(self, worker_id: int = 0, num_workers: int = 1) -> Iterator:
        """Create iterator for this worker."""
        # Shard distribution across workers
        worker_shards = self.shards
        if num_workers > 1:
            worker_shards = self.shards[worker_id::num_workers]
        
        if not worker_shards:
            logger.warning(f"No shards for worker {worker_id}")
            return iter([])
        
        # Create shard iterator
        shard_iterator = ShardedDatasetIterator(
            file_paths=worker_shards,
            cloud_fs=self.cloud_fs,
            shuffle=(self.split == "train"),
            seed=self.config.seed
        )
        
        # Apply tokenization and transforms
        def process_example(example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            # Apply custom transform first
            if self.transform:
                example = self.transform(example)
            
            # Tokenize
            tokenized = self.tokenization_pipeline.tokenize_example(example)
            self.stats["examples_processed"] += 1
            
            return tokenized
        
        return map(process_example, shard_iterator)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Multi-worker dataloader
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single process
            worker_id = 0
            num_workers = 1
        
        # Start prefetching thread
        prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(worker_id, num_workers),
            daemon=True
        )
        prefetch_thread.start()
        
        # Yield from prefetch queue
        try:
            while True:
                batch = self.prefetch_queue.get(timeout=30.0)
                if batch is None:
                    break
                self.stats["batches_prefetched"] += 1
                yield batch
        finally:
            self.prefetch_queue.stop()
            prefetch_thread.join(timeout=5.0)
    
    def _prefetch_worker(self, worker_id: int, num_workers: int) -> None:
        """Worker thread for prefetching data."""
        try:
            iterator = self._create_iterator(worker_id, num_workers)
            
            for example in iterator:
                self.prefetch_queue.put(example)
                time.sleep(0.001)  # Small delay to prevent CPU saturation
        
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
        finally:
            self.prefetch_queue.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            **self.stats,
            "total_shards": len(self.shards),
            "config": {
                "prefetch_buffer_size": self.config.prefetch_buffer_size,
                "batch_size": self.config.batch_size,
                "max_sequence_length": self.config.max_sequence_length
            }
        }


class StreamingDataLoader:
    """
    High-performance streaming dataloader with cloud-native support.
    Wraps PyTorch DataLoader with streaming-specific optimizations.
    """
    
    def __init__(self,
                 dataset: StreamingDataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 prefetch_factor: int = 2):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> Optional[int]:
        """Length is unknown for streaming datasets."""
        return None
    
    @classmethod
    def from_config(cls,
                    data_paths: Union[str, List[str]],
                    tokenizer: PreTrainedTokenizer,
                    config: Optional[StreamingConfig] = None,
                    **kwargs) -> 'StreamingDataLoader':
        """Create dataloader from configuration."""
        config = config or StreamingConfig()
        
        dataset = StreamingDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            config=config,
            **kwargs
        )
        
        return cls(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )


def create_streaming_dataloader(
    data_paths: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 2048,
    num_workers: int = 4,
    prefetch_buffer: int = 1000,
    split: str = "train",
    text_column: str = "text",
    instruction_template: Optional[str] = None,
    cloud_credentials: Optional[Dict[str, str]] = None,
    **kwargs
) -> StreamingDataLoader:
    """
    Factory function to create streaming dataloader.
    
    Args:
        data_paths: Paths to data files (local or cloud)
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        prefetch_buffer: Size of prefetch buffer
        split: Dataset split (train/validation/test)
        text_column: Column containing text data
        instruction_template: Template for instruction tuning
        cloud_credentials: Credentials for cloud storage
        
    Returns:
        StreamingDataLoader instance
    """
    config = StreamingConfig(
        prefetch_buffer_size=prefetch_buffer,
        num_workers=num_workers,
        batch_size=batch_size,
        max_sequence_length=max_length,
        cloud_credentials=cloud_credentials
    )
    
    return StreamingDataLoader.from_config(
        data_paths=data_paths,
        tokenizer=tokenizer,
        config=config,
        split=split,
        text_column=text_column,
        instruction_template=instruction_template,
        **kwargs
    )


# Integration with existing crucible components
def patch_existing_dataloader():
    """Monkey-patch existing dataloader functions to use streaming."""
    try:
        from crucible.data import loader as existing_loader
        
        # Store original function
        original_create = getattr(existing_loader, 'create_dataloader', None)
        
        if original_create:
            def streaming_create(*args, **kwargs):
                # Check if streaming is requested
                if kwargs.pop('streaming', False):
                    return create_streaming_dataloader(*args, **kwargs)
                else:
                    return original_create(*args, **kwargs)
            
            # Replace with streaming-aware version
            existing_loader.create_dataloader = streaming_create
            logger.info("Patched existing dataloader with streaming support")
    
    except ImportError:
        logger.debug("Could not patch existing dataloader - module not found")


# Auto-patch on import
patch_existing_dataloader()


# Export public API
__all__ = [
    'StreamingConfig',
    'StreamingDataset',
    'StreamingDataLoader',
    'create_streaming_dataloader',
    'CloudFileSystem',
    'TokenizationPipeline',
    'PrefetchingQueue'
]