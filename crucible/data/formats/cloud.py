"""
crucible/data/formats/cloud.py

Zero-Copy Data Loading & Streaming Pipeline for crucible.
Supports sharded JSONL/Parquet datasets from S3, GCS, and local storage with memory-mapped streaming.
"""

import os
import json
import logging
import threading
import queue
from typing import Dict, List, Optional, Union, Iterator, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pj
from pyarrow import fs

try:
    import fsspec
    from fsspec.implementations.arrow import ArrowFSWrapper
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False

try:
    from datasets import Dataset, IterableDataset, Features, Value, Sequence
    from datasets.features import Features
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

import numpy as np
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for the streaming data pipeline."""
    prefetch_buffer_size: int = 1000
    num_workers: int = 4
    batch_size: int = 1000
    shuffle_buffer_size: int = 10000
    seed: int = 42
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: Optional[str] = None
    use_mmap: bool = True
    streaming: bool = True
    drop_last: bool = False
    collate_fn: Optional[Callable] = None


class CloudFileSystem:
    """Unified cloud filesystem interface for S3, GCS, and local storage."""
    
    def __init__(self, 
                 cloud_type: str = "local",
                 endpoint_url: Optional[str] = None,
                 region: Optional[str] = None,
                 **kwargs):
        self.cloud_type = cloud_type.lower()
        self.endpoint_url = endpoint_url
        self.region = region
        self.kwargs = kwargs
        
        if self.cloud_type == "s3":
            self.fs = fs.S3FileSystem(
                endpoint_override=endpoint_url,
                region=region,
                **kwargs
            )
        elif self.cloud_type == "gcs":
            self.fs = fs.GcsFileSystem(**kwargs)
        elif self.cloud_type == "local":
            self.fs = fs.LocalFileSystem(**kwargs)
        else:
            raise ValueError(f"Unsupported cloud type: {cloud_type}")
    
    def get_file_info(self, path: str) -> List[fs.FileInfo]:
        """Get file information for a path."""
        return self.fs.get_file_info(path)
    
    def open_input_file(self, path: str, **kwargs) -> pa.NativeFile:
        """Open a file for reading."""
        return self.fs.open_input_file(path, **kwargs)
    
    def open_input_stream(self, path: str, **kwargs) -> pa.NativeFile:
        """Open a file for streaming."""
        return self.fs.open_input_stream(path, **kwargs)
    
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory, optionally filtered by pattern."""
        if self.cloud_type == "local":
            path_obj = Path(path)
            if path_obj.is_dir():
                files = [str(f) for f in path_obj.rglob(pattern or "*") if f.is_file()]
            else:
                files = [str(path_obj)]
        else:
            selector = fs.FileSelector(path, allow_not_found=True)
            file_infos = self.fs.get_file_info(selector)
            files = [info.path for info in file_infos if info.is_file]
            if pattern:
                import fnmatch
                files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        
        return sorted(files)


class ShardedDatasetReader:
    """Reads sharded datasets from cloud storage with zero-copy operations."""
    
    def __init__(self,
                 file_paths: List[str],
                 cloud_fs: CloudFileSystem,
                 format: str = "auto",
                 streaming: bool = True,
                 use_mmap: bool = True):
        self.file_paths = file_paths
        self.cloud_fs = cloud_fs
        self.format = format
        self.streaming = streaming
        self.use_mmap = use_mmap
        self._current_shard = 0
        self._shard_iterators = {}
        
        # Detect format if auto
        if self.format == "auto":
            self.format = self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect dataset format from file extensions."""
        if not self.file_paths:
            raise ValueError("No file paths provided")
        
        first_file = self.file_paths[0]
        if first_file.endswith('.parquet'):
            return 'parquet'
        elif first_file.endswith('.jsonl') or first_file.endswith('.json'):
            return 'jsonl'
        else:
            # Try to read first few bytes to detect format
            try:
                with self.cloud_fs.open_input_stream(first_file) as f:
                    header = f.read(100)
                    if header.startswith(b'{') or header.startswith(b'['):
                        return 'jsonl'
                    elif header.startswith(b'PAR1'):
                        return 'parquet'
            except:
                pass
        
        raise ValueError(f"Cannot detect format for file: {first_file}")
    
    def read_shard(self, shard_idx: int) -> pa.Table:
        """Read a single shard as a PyArrow Table."""
        if shard_idx >= len(self.file_paths):
            raise IndexError(f"Shard index {shard_idx} out of range")
        
        file_path = self.file_paths[shard_idx]
        
        try:
            if self.format == 'parquet':
                if self.use_mmap and self.cloud_fs.cloud_type == "local":
                    # Use memory mapping for local files
                    return pq.read_table(file_path, memory_map=True)
                else:
                    with self.cloud_fs.open_input_file(file_path) as f:
                        return pq.read_table(f)
            elif self.format == 'jsonl':
                with self.cloud_fs.open_input_stream(file_path) as f:
                    # PyArrow's JSON reader expects a file-like object
                    # We'll read line by line and parse
                    return self._read_jsonl_stream(f)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
        except Exception as e:
            logger.error(f"Failed to read shard {shard_idx}: {e}")
            raise
    
    def _read_jsonl_stream(self, file_obj) -> pa.Table:
        """Read JSONL file as a PyArrow Table."""
        import io
        
        # Read all lines and parse as JSON
        lines = []
        with io.TextIOWrapper(file_obj, encoding='utf-8') as text_file:
            for line in text_file:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
        
        if not lines:
            # Return empty table with inferred schema
            return pa.table({})
        
        # Convert to PyArrow Table
        return pa.Table.from_pylist(lines)
    
    def stream_shards(self) -> Iterator[Dict[str, Any]]:
        """Stream records from all shards."""
        for shard_idx in range(len(self.file_paths)):
            try:
                table = self.read_shard(shard_idx)
                
                # Convert to Python dicts and yield
                for batch in table.to_batches():
                    for record in batch.to_pydict():
                        yield record
                        
            except Exception as e:
                logger.error(f"Error streaming shard {shard_idx}: {e}")
                continue
    
    def __iter__(self):
        """Make the reader iterable."""
        return self.stream_shards()


class PrefetchingQueue:
    """Thread-safe prefetching queue for overlapping I/O with compute."""
    
    def __init__(self, 
                 maxsize: int = 1000,
                 num_workers: int = 4,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._stop_event = threading.Event()
        self._workers = []
        self._futures = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def start(self, data_iterator: Iterator, transform_fn: Optional[Callable] = None):
        """Start prefetching from the data iterator."""
        self._stop_event.clear()
        
        def prefetch_worker():
            while not self._stop_event.is_set():
                try:
                    # Get next item with timeout to allow checking stop event
                    record = next(data_iterator)
                    
                    # Apply transformation if provided
                    if transform_fn:
                        record = transform_fn(record)
                    
                    # Put in queue with timeout
                    self.queue.put(record, timeout=1.0)
                    
                except StopIteration:
                    # End of iterator
                    break
                except queue.Full:
                    # Queue is full, wait a bit
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error in prefetch worker: {e}")
                    time.sleep(self.retry_delay)
        
        # Start prefetch workers
        for _ in range(self.num_workers):
            future = self._executor.submit(prefetch_worker)
            self._futures.append(future)
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get an item from the queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            # Check if workers are still running
            if all(f.done() for f in self._futures):
                raise StopIteration("No more data")
            raise
    
    def stop(self):
        """Stop all prefetch workers."""
        self._stop_event.set()
        
        # Wait for workers to finish
        for future in self._futures:
            try:
                future.result(timeout=5.0)
            except:
                pass
        
        self._executor.shutdown(wait=False)
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class StreamingDataset:
    """Memory-mapped streaming dataset with lazy loading and preprocessing."""
    
    def __init__(self,
                 file_paths: List[str],
                 cloud_fs: CloudFileSystem,
                 config: StreamingConfig,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 tokenize_fn: Optional[Callable] = None,
                 max_length: Optional[int] = None,
                 text_column: str = "text",
                 label_column: Optional[str] = None):
        self.file_paths = file_paths
        self.cloud_fs = cloud_fs
        self.config = config
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Initialize shard reader
        self.reader = ShardedDatasetReader(
            file_paths=file_paths,
            cloud_fs=cloud_fs,
            streaming=config.streaming,
            use_mmap=config.use_mmap
        )
        
        # Prefetching queue
        self.prefetch_queue = None
        if config.prefetch_buffer_size > 0:
            self.prefetch_queue = PrefetchingQueue(
                maxsize=config.prefetch_buffer_size,
                num_workers=config.num_workers,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
        
        # Shuffle buffer
        self.shuffle_buffer = []
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.rng = np.random.RandomState(config.seed)
        
        # Statistics
        self._stats = {
            "records_read": 0,
            "bytes_read": 0,
            "shards_processed": 0,
            "start_time": time.time()
        }
    
    def _tokenize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single record."""
        if not self.tokenizer:
            return record
        
        # Get text from record
        text = record.get(self.text_column, "")
        if not text:
            return record
        
        # Use custom tokenize function if provided
        if self.tokenize_fn:
            return self.tokenize_fn(record, self.tokenizer, self.max_length)
        
        # Default tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length" if self.max_length else False,
            return_tensors="np"
        )
        
        # Create output record
        output_record = {
            "input_ids": encoding["input_ids"].squeeze().tolist(),
            "attention_mask": encoding["attention_mask"].squeeze().tolist(),
        }
        
        # Add labels if present
        if self.label_column and self.label_column in record:
            output_record["labels"] = record[self.label_column]
        
        # Add other fields
        for key, value in record.items():
            if key not in [self.text_column, self.label_column]:
                output_record[key] = value
        
        return output_record
    
    def _apply_shuffle_buffer(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply shuffle buffer to records."""
        if self.shuffle_buffer_size <= 0:
            return record
        
        # Add to buffer
        self.shuffle_buffer.append(record)
        
        # If buffer is full, shuffle and yield
        if len(self.shuffle_buffer) >= self.shuffle_buffer_size:
            self.rng.shuffle(self.shuffle_buffer)
            return self.shuffle_buffer.pop(0)
        
        return None
    
    def _flush_shuffle_buffer(self) -> Iterator[Dict[str, Any]]:
        """Flush remaining items from shuffle buffer."""
        if self.shuffle_buffer:
            self.rng.shuffle(self.shuffle_buffer)
            while self.shuffle_buffer:
                yield self.shuffle_buffer.pop(0)
    
    def _create_data_iterator(self) -> Iterator[Dict[str, Any]]:
        """Create the main data iterator with all transformations."""
        # Start with shard streaming
        data_iter = self.reader.stream_shards()
        
        # Apply tokenization
        if self.tokenizer:
            data_iter = (self._tokenize_record(record) for record in data_iter)
        
        # Apply shuffle buffer
        if self.shuffle_buffer_size > 0:
            shuffled_iter = (self._apply_shuffle_buffer(record) for record in data_iter)
            data_iter = (record for record in shuffled_iter if record is not None)
        
        return data_iter
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset."""
        # Create data iterator
        data_iter = self._create_data_iterator()
        
        # Use prefetching if enabled
        if self.prefetch_queue:
            self.prefetch_queue.start(
                data_iter,
                transform_fn=None  # Transformations already applied
            )
            
            try:
                while True:
                    try:
                        record = self.prefetch_queue.get(timeout=1.0)
                        self._stats["records_read"] += 1
                        yield record
                    except queue.Empty:
                        # Check if prefetch workers are done
                        if all(f.done() for f in self.prefetch_queue._futures):
                            break
                        continue
            finally:
                self.prefetch_queue.stop()
        else:
            # No prefetching
            for record in data_iter:
                self._stats["records_read"] += 1
                yield record
        
        # Flush shuffle buffer
        for record in self._flush_shuffle_buffer():
            self._stats["records_read"] += 1
            yield record
    
    def __len__(self) -> int:
        """Get approximate length (may not be accurate for streaming)."""
        # For streaming datasets, we don't know the exact length
        # Return -1 to indicate unknown length
        return -1
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        current_time = time.time()
        elapsed = current_time - self._stats["start_time"]
        
        return {
            **self._stats,
            "elapsed_seconds": elapsed,
            "records_per_second": self._stats["records_read"] / max(elapsed, 1e-6)
        }


class CloudDatasetFactory:
    """Factory for creating streaming datasets from cloud storage."""
    
    @staticmethod
    def create_from_patterns(patterns: List[str],
                           cloud_type: str = "local",
                           config: Optional[StreamingConfig] = None,
                           **cloud_kwargs) -> StreamingDataset:
        """
        Create a streaming dataset from file patterns.
        
        Args:
            patterns: List of file patterns (e.g., ["s3://bucket/data/*.jsonl"])
            cloud_type: Type of cloud storage ("s3", "gcs", "local")
            config: Streaming configuration
            **cloud_kwargs: Additional arguments for cloud filesystem
        
        Returns:
            StreamingDataset instance
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library is required for streaming. Install with: pip install datasets")
        
        config = config or StreamingConfig()
        
        # Initialize cloud filesystem
        cloud_fs = CloudFileSystem(cloud_type, **cloud_kwargs)
        
        # Expand patterns to file paths
        file_paths = []
        for pattern in patterns:
            if cloud_type == "local":
                # Use glob for local files
                path = Path(pattern)
                if path.is_file():
                    file_paths.append(str(path))
                elif path.is_dir():
                    file_paths.extend([str(f) for f in path.rglob("*") if f.is_file()])
                else:
                    # Glob pattern
                    import glob
                    file_paths.extend(glob.glob(pattern, recursive=True))
            else:
                # For cloud, list files with pattern matching
                base_path = pattern.rsplit('*', 1)[0] if '*' in pattern else pattern
                files = cloud_fs.list_files(base_path)
                
                # Filter by pattern if needed
                if '*' in pattern:
                    import fnmatch
                    file_paths.extend([f for f in files if fnmatch.fnmatch(f, pattern)])
                else:
                    file_paths.extend(files)
        
        if not file_paths:
            raise ValueError(f"No files found for patterns: {patterns}")
        
        logger.info(f"Found {len(file_paths)} files for streaming")
        
        return StreamingDataset(
            file_paths=file_paths,
            cloud_fs=cloud_fs,
            config=config
        )
    
    @staticmethod
    def create_from_hf_dataset(dataset_name: str,
                              split: str = "train",
                              config: Optional[StreamingConfig] = None,
                              **kwargs) -> StreamingDataset:
        """
        Create a streaming dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            split: Dataset split to use
            config: Streaming configuration
            **kwargs: Additional arguments for load_dataset
        
        Returns:
            StreamingDataset instance
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        from datasets import load_dataset
        
        config = config or StreamingConfig()
        
        # Load dataset with streaming
        hf_dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            **kwargs
        )
        
        # Convert to our StreamingDataset format
        # This is a simplified version - in practice, we'd need to handle
        # the HuggingFace IterableDataset more carefully
        class HuggingFaceStreamingDataset(StreamingDataset):
            def __init__(self, hf_dataset, config):
                self.hf_dataset = hf_dataset
                self.config = config
                self._stats = {
                    "records_read": 0,
                    "start_time": time.time()
                }
            
            def __iter__(self):
                for record in self.hf_dataset:
                    self._stats["records_read"] += 1
                    yield record
            
            @property
            def stats(self):
                current_time = time.time()
                elapsed = current_time - self._stats["start_time"]
                return {
                    **self._stats,
                    "elapsed_seconds": elapsed,
                    "records_per_second": self._stats["records_read"] / max(elapsed, 1e-6)
                }
        
        return HuggingFaceStreamingDataset(hf_dataset, config)


# Integration with existing crucible data loading
def load_cloud_dataset(file_paths: List[str],
                      cloud_type: str = "local",
                      format: str = "auto",
                      streaming: bool = True,
                      prefetch_buffer: int = 1000,
                      num_workers: int = 4,
                      tokenizer: Optional[PreTrainedTokenizer] = None,
                      max_length: Optional[int] = None,
                      text_column: str = "text",
                      label_column: Optional[str] = None,
                      **cloud_kwargs) -> StreamingDataset:
    """
    Main entry point for loading cloud datasets.
    
    Args:
        file_paths: List of file paths or patterns
        cloud_type: Type of cloud storage ("s3", "gcs", "local")
        format: Dataset format ("jsonl", "parquet", "auto")
        streaming: Enable streaming mode
        prefetch_buffer: Size of prefetch buffer
        num_workers: Number of prefetch workers
        tokenizer: Optional tokenizer for on-the-fly tokenization
        max_length: Maximum sequence length for tokenization
        text_column: Name of text column in dataset
        label_column: Name of label column in dataset
        **cloud_kwargs: Additional cloud storage arguments
    
    Returns:
        StreamingDataset instance
    """
    config = StreamingConfig(
        prefetch_buffer_size=prefetch_buffer,
        num_workers=num_workers,
        streaming=streaming
    )
    
    # Create cloud filesystem
    cloud_fs = CloudFileSystem(cloud_type, **cloud_kwargs)
    
    # Expand file patterns
    expanded_paths = []
    for pattern in file_paths:
        if '*' in pattern or '?' in pattern:
            # Glob pattern
            base_dir = pattern.rsplit('/', 1)[0] if '/' in pattern else '.'
            files = cloud_fs.list_files(base_dir)
            
            import fnmatch
            matched = [f for f in files if fnmatch.fnmatch(f, pattern)]
            expanded_paths.extend(matched)
        else:
            # Single file or directory
            file_info = cloud_fs.get_file_info(pattern)
            if file_info and file_info[0].is_file:
                expanded_paths.append(pattern)
    
    if not expanded_paths:
        raise ValueError(f"No files found for patterns: {file_paths}")
    
    logger.info(f"Loading {len(expanded_paths)} files from {cloud_type}")
    
    return StreamingDataset(
        file_paths=expanded_paths,
        cloud_fs=cloud_fs,
        config=config,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column
    )


# Utility functions for distributed training
def shard_dataset_across_workers(dataset: StreamingDataset,
                               world_size: int,
                               rank: int) -> StreamingDataset:
    """
    Shard a dataset across distributed workers.
    
    Args:
        dataset: Original dataset
        world_size: Total number of workers
        rank: Current worker rank
    
    Returns:
        Sharded dataset for current worker
    """
    if world_size <= 1:
        return dataset
    
    # Calculate shard indices for this worker
    total_files = len(dataset.file_paths)
    files_per_worker = total_files // world_size
    remainder = total_files % world_size
    
    # Distribute remainder among first workers
    start_idx = rank * files_per_worker + min(rank, remainder)
    end_idx = start_idx + files_per_worker + (1 if rank < remainder else 0)
    
    # Get files for this worker
    worker_files = dataset.file_paths[start_idx:end_idx]
    
    if not worker_files:
        logger.warning(f"Worker {rank} has no files to process")
    
    # Create new dataset with sharded files
    return StreamingDataset(
        file_paths=worker_files,
        cloud_fs=dataset.cloud_fs,
        config=dataset.config,
        tokenizer=dataset.tokenizer,
        tokenize_fn=dataset.tokenize_fn,
        max_length=dataset.max_length,
        text_column=dataset.text_column,
        label_column=dataset.label_column
    )


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Load local JSONL files
    config = StreamingConfig(
        prefetch_buffer_size=500,
        num_workers=2,
        streaming=True
    )
    
    # Create a simple tokenizer for testing
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    
    tokenizer = DummyTokenizer()
    
    # Load dataset
    try:
        dataset = load_cloud_dataset(
            file_paths=["data/train/*.jsonl"],
            cloud_type="local",
            streaming=True,
            tokenizer=tokenizer,
            max_length=512
        )
        
        # Iterate over dataset
        for i, record in enumerate(dataset):
            if i >= 5:  # Just show first 5 records
                break
            print(f"Record {i}: {record}")
        
        print(f"Dataset stats: {dataset.stats}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    # Example 2: Distributed sharding
    print("\nDistributed sharding example:")
    dataset = load_cloud_dataset(
        file_paths=["data/train/*.jsonl"],
        cloud_type="local"
    )
    
    # Shard for 4 workers, worker 0
    sharded = shard_dataset_across_workers(dataset, world_size=4, rank=0)
    print(f"Worker 0 has {len(sharded.file_paths)} files")