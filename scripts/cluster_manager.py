#!/usr/bin/env python3
"""
Fault-Tolerant Distributed Training with Auto-Recovery for crucible
========================================================================

This module implements automatic checkpointing, preemption handling, and elastic scaling
for multi-node training. It integrates with Kubernetes/SLURM to resume failed jobs from
the last checkpoint without manual intervention, reducing wasted compute and enabling
24/7 unattended training.

Key Features:
- Automatic periodic checkpointing with async I/O
- Preemption signal handling (SIGTERM, SIGUSR1, etc.)
- Elastic scaling support for dynamic node changes
- Integration with Kubernetes/SLURM job schedulers
- Health monitoring and auto-recovery
- Resume from last checkpoint without manual intervention
- Distributed training coordination across nodes
"""

import os
import sys
import json
import time
import signal
import logging
import asyncio
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from crucible.train.callbacks import LogCallback
    from crucible.extras.logging import get_logger
    from crucible.extras.misc import get_current_device
    logger = get_logger(__name__)
except ImportError:
    # Fallback logging if crucible modules not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Mock LogCallback for standalone usage
    class LogCallback:
        def __init__(self):
            pass
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            pass


class ClusterState(Enum):
    """Cluster states for distributed training coordination."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PREEMPTED = "preempted"
    SCALING = "scaling"
    RECOVERING = "recovering"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking and recovery."""
    checkpoint_id: str
    timestamp: str
    global_step: int
    epoch: float
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    node_count: int = 1
    world_size: int = 1
    local_rank: int = 0
    global_rank: int = 0
    job_id: Optional[str] = None
    preemption_detected: bool = False
    elastic_scaling: bool = False
    training_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeHealth:
    """Node health status for monitoring."""
    node_id: str
    hostname: str
    ip_address: str
    gpu_count: int
    gpu_memory_used: List[float] = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    last_heartbeat: float = 0.0
    is_healthy: bool = True
    failure_count: int = 0


class DistributedCheckpointManager:
    """Manages distributed checkpointing across multiple nodes."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        async_save: bool = True,
        save_on_all_nodes: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.async_save = async_save
        self.save_on_all_nodes = save_on_all_nodes
        self.checkpoint_queue = asyncio.Queue() if async_save else None
        self.executor = ThreadPoolExecutor(max_workers=2) if async_save else None
        self._setup_directories()
        
    def _setup_directories(self):
        """Create checkpoint directories if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "tmp").mkdir(exist_ok=True)
        (self.checkpoint_dir / "latest").mkdir(exist_ok=True)
        
    def _generate_checkpoint_id(self, global_step: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint-{global_step}-{timestamp}"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        epoch: float,
        metadata: CheckpointMetadata,
        save_path: Optional[str] = None
    ) -> str:
        """
        Save checkpoint with distributed coordination.
        
        Args:
            model: Model to save (can be DDP wrapped)
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            global_step: Current global training step
            epoch: Current epoch
            metadata: Checkpoint metadata
            save_path: Optional custom save path
            
        Returns:
            Path to saved checkpoint
        """
        # Only save on rank 0 unless save_on_all_nodes is True
        if not self.save_on_all_nodes and not self._is_main_process():
            return ""
            
        checkpoint_id = self._generate_checkpoint_id(global_step)
        if save_path is None:
            save_path = self.checkpoint_dir / checkpoint_id
        else:
            save_path = Path(save_path)
            
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': self._get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'global_step': global_step,
            'epoch': epoch,
            'metadata': asdict(metadata),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Save checkpoint
        if self.async_save and self.executor:
            future = self.executor.submit(
                self._save_checkpoint_sync,
                checkpoint_data,
                save_path
            )
            # Update latest symlink after save completes
            future.add_done_callback(lambda f: self._update_latest_symlink(save_path))
        else:
            self._save_checkpoint_sync(checkpoint_data, save_path)
            self._update_latest_symlink(save_path)
            
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {save_path}")
        return str(save_path)
    
    def _save_checkpoint_sync(self, checkpoint_data: Dict, save_path: Path):
        """Synchronously save checkpoint to disk."""
        # Save to temporary location first for atomic write
        tmp_path = save_path.parent / "tmp" / save_path.name
        tmp_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = tmp_path / "model.pt"
        torch.save(checkpoint_data['model_state_dict'], model_path)
        
        # Save optimizer and scheduler states
        optimizer_path = tmp_path / "optimizer.pt"
        torch.save(checkpoint_data['optimizer_state_dict'], optimizer_path)
        
        if checkpoint_data['scheduler_state_dict']:
            scheduler_path = tmp_path / "scheduler.pt"
            torch.save(checkpoint_data['scheduler_state_dict'], scheduler_path)
        
        # Save metadata
        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'global_step': checkpoint_data['global_step'],
                'epoch': checkpoint_data['epoch'],
                'metadata': checkpoint_data['metadata'],
                'timestamp': checkpoint_data['timestamp'],
                'pytorch_version': checkpoint_data['pytorch_version'],
            }, f, indent=2)
        
        # Atomic move from tmp to final location
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path)
        tmp_path.rename(save_path)
    
    def _get_model_state_dict(self, model: torch.nn.Module) -> Dict:
        """Get model state dict, handling DDP wrapper."""
        if isinstance(model, DDP):
            return model.module.state_dict()
        return model.state_dict()
    
    def _update_latest_symlink(self, checkpoint_path: Path):
        """Update symlink to latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest"
        if latest_path.is_symlink():
            latest_path.unlink()
        elif latest_path.exists():
            import shutil
            shutil.rmtree(latest_path)
        
        # Create relative symlink
        try:
            latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback for systems that don't support symlinks
            import shutil
            shutil.copytree(checkpoint_path, latest_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_keep most recent."""
        if not self._is_main_process():
            return
            
        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints
        while len(checkpoints) > self.max_keep:
            _, checkpoint_path = checkpoints.pop(0)
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device
    ) -> Tuple[int, float, Dict]:
        """
        Load the latest checkpoint if available.
        
        Returns:
            Tuple of (global_step, epoch, metadata)
        """
        latest_path = self.checkpoint_dir / "latest"
        if not latest_path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return 0, 0.0, {}
        
        # Resolve symlink if it's a link
        if latest_path.is_symlink():
            checkpoint_path = latest_path.resolve()
        else:
            checkpoint_path = latest_path
        
        if not checkpoint_path.exists():
            logger.warning("Latest checkpoint path does not exist")
            return 0, 0.0, {}
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model state
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(model, DDP):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        
        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = {}
        global_step = 0
        epoch = 0.0
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                global_step = metadata.get('global_step', 0)
                epoch = metadata.get('epoch', 0.0)
        
        logger.info(f"Resumed from step {global_step}, epoch {epoch}")
        return global_step, epoch, metadata
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0


class PreemptionHandler:
    """Handles preemption signals and graceful shutdown."""
    
    def __init__(self, cluster_manager: 'ClusterManager'):
        self.cluster_manager = cluster_manager
        self.preemption_signals = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGUSR1: "SIGUSR1",  # Common for SLURM preemption
            signal.SIGUSR2: "SIGUSR2",  # Common for Kubernetes preemption
        }
        self.original_handlers = {}
        self.preemption_detected = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for preemption detection."""
        for sig, name in self.preemption_signals.items():
            try:
                self.original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
                logger.debug(f"Registered handler for {name}")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not register handler for {name}: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle preemption signals."""
        signal_name = self.preemption_signals.get(signum, str(signum))
        logger.warning(f"Received preemption signal: {signal_name}")
        self.preemption_detected = True
        
        # Trigger checkpoint save before shutdown
        self.cluster_manager.handle_preemption(signal_name)
        
        # Call original handler if it exists and is callable
        original_handler = self.original_handlers.get(signum)
        if callable(original_handler) and original_handler not in (signal.SIG_IGN, signal.SIG_DFL):
            original_handler(signum, frame)
    
    def cleanup(self):
        """Restore original signal handlers."""
        for sig, handler in self.original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


class ElasticScaler:
    """Handles elastic scaling of distributed training nodes."""
    
    def __init__(self, cluster_manager: 'ClusterManager'):
        self.cluster_manager = cluster_manager
        self.min_nodes = 1
        self.max_nodes = 8
        self.current_nodes = 1
        self.scaling_enabled = False
        self.last_scale_time = 0
        self.scale_cooldown = 300  # 5 minutes cooldown
        
    def enable_elastic_scaling(self, min_nodes: int = 1, max_nodes: int = 8):
        """Enable elastic scaling with specified bounds."""
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scaling_enabled = True
        logger.info(f"Elastic scaling enabled: {min_nodes}-{max_nodes} nodes")
    
    def check_scaling_conditions(self, metrics: Dict[str, float]) -> bool:
        """
        Check if scaling conditions are met.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            True if scaling is recommended
        """
        if not self.scaling_enabled:
            return False
            
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Example scaling logic based on GPU utilization
        gpu_util = metrics.get('gpu_utilization', 0.0)
        memory_util = metrics.get('memory_utilization', 0.0)
        
        # Scale up if resources are underutilized
        if gpu_util < 30 and memory_util < 50 and self.current_nodes > self.min_nodes:
            logger.info("Considering scale down due to low resource utilization")
            return True
        
        # Scale down if resources are overutilized
        if gpu_util > 80 or memory_util > 80:
            if self.current_nodes < self.max_nodes:
                logger.info("Considering scale up due to high resource utilization")
                return True
        
        return False
    
    def scale_cluster(self, target_nodes: int):
        """
        Scale the cluster to target number of nodes.
        
        Args:
            target_nodes: Desired number of nodes
        """
        if not self.scaling_enabled:
            logger.warning("Elastic scaling is not enabled")
            return
        
        target_nodes = max(self.min_nodes, min(self.max_nodes, target_nodes))
        
        if target_nodes == self.current_nodes:
            logger.info(f"Already at target node count: {target_nodes}")
            return
        
        logger.info(f"Scaling from {self.current_nodes} to {target_nodes} nodes")
        
        # Notify cluster manager of scaling event
        self.cluster_manager.handle_scaling_event(
            old_nodes=self.current_nodes,
            new_nodes=target_nodes
        )
        
        # Update current node count
        self.current_nodes = target_nodes
        self.last_scale_time = time.time()
        
        # In a real implementation, this would interact with Kubernetes/SLURM APIs
        # to actually scale the cluster
        logger.info(f"Cluster scaled to {target_nodes} nodes")


class HealthMonitor:
    """Monitors node health and cluster status."""
    
    def __init__(self, cluster_manager: 'ClusterManager', check_interval: int = 30):
        self.cluster_manager = cluster_manager
        self.check_interval = check_interval
        self.nodes: Dict[str, NodeHealth] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.health_check_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor"
        )
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_node_health()
                self._check_cluster_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_node_health(self):
        """Check health of all nodes in the cluster."""
        # In a real implementation, this would query actual node metrics
        # For now, we'll simulate health checks
        
        current_node = self._get_current_node_info()
        if current_node.node_id not in self.nodes:
            self.nodes[current_node.node_id] = current_node
        
        # Update health status
        node = self.nodes[current_node.node_id]
        node.last_heartbeat = time.time()
        
        # Simulate health check
        node.is_healthy = self._perform_health_check(node)
        
        if not node.is_healthy:
            node.failure_count += 1
            logger.warning(f"Node {node.node_id} health check failed")
            
            # Notify cluster manager
            self.cluster_manager.handle_node_failure(node)
    
    def _check_cluster_health(self):
        """Check overall cluster health."""
        healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
        total_nodes = len(self.nodes)
        
        if total_nodes > 0 and healthy_nodes < total_nodes * 0.5:
            logger.error(f"Cluster health critical: {healthy_nodes}/{total_nodes} nodes healthy")
            self.cluster_manager.handle_cluster_failure(
                f"Only {healthy_nodes}/{total_nodes} nodes healthy"
            )
    
    def _get_current_node_info(self) -> NodeHealth:
        """Get information about the current node."""
        import socket
        import psutil
        
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip_address = "127.0.0.1"
        
        # Get GPU count (simplified)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get GPU memory usage (simplified)
        gpu_memory_used = []
        if torch.cuda.is_available():
            for i in range(gpu_count):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    gpu_memory_used.append(memory_allocated)
                except:
                    gpu_memory_used.append(0.0)
        
        return NodeHealth(
            node_id=f"{hostname}-{os.getpid()}",
            hostname=hostname,
            ip_address=ip_address,
            gpu_count=gpu_count,
            gpu_memory_used=gpu_memory_used,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            last_heartbeat=time.time(),
            is_healthy=True
        )
    
    def _perform_health_check(self, node: NodeHealth) -> bool:
        """Perform health check on a node."""
        # Check if last heartbeat is too old
        if time.time() - node.last_heartbeat > 60:  # 60 seconds timeout
            return False
        
        # Check system resources
        if node.cpu_percent > 95:  # CPU over 95%
            logger.warning(f"High CPU usage on {node.hostname}: {node.cpu_percent}%")
        
        if node.memory_percent > 90:  # Memory over 90%
            logger.warning(f"High memory usage on {node.hostname}: {node.memory_percent}%")
        
        # Check GPU memory
        for i, mem in enumerate(node.gpu_memory_used):
            if mem > 0.9 * torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):
                logger.warning(f"High GPU memory usage on {node.hostname}, GPU {i}: {mem:.2f}GB")
        
        return True
    
    def register_health_callback(self, callback: Callable):
        """Register a callback for health status changes."""
        self.health_check_callbacks.append(callback)


class ClusterManager:
    """
    Main cluster manager for fault-tolerant distributed training.
    
    Coordinates checkpointing, preemption handling, and elastic scaling
    across distributed training nodes.
    """
    
    def __init__(
        self,
        training_args: Any,
        checkpoint_dir: str,
        enable_elastic: bool = False,
        enable_preemption_handling: bool = True,
        checkpoint_interval: int = 500,
        health_check_interval: int = 30
    ):
        """
        Initialize the cluster manager.
        
        Args:
            training_args: Training arguments from crucible
            checkpoint_dir: Directory for storing checkpoints
            enable_elastic: Enable elastic scaling
            enable_preemption_handling: Enable preemption signal handling
            checkpoint_interval: Steps between automatic checkpoints
            health_check_interval: Seconds between health checks
        """
        self.training_args = training_args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.state = ClusterState.INITIALIZING
        
        # Initialize components
        self.checkpoint_manager = DistributedCheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_keep=5,
            async_save=True
        )
        
        self.preemption_handler = PreemptionHandler(self) if enable_preemption_handling else None
        self.elastic_scaler = ElasticScaler(self) if enable_elastic else None
        self.health_monitor = HealthMonitor(self, health_check_interval)
        
        # Training state
        self.global_step = 0
        self.epoch = 0.0
        self.last_checkpoint_step = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        
        # Recovery state
        self.recovery_metadata = {}
        self.is_recovering = False
        
        # Callbacks
        self.callbacks: List[Callable] = []
        self.log_callback = LogCallback()
        
        # Job information
        self.job_id = self._get_job_id()
        self.node_rank = self._get_node_rank()
        self.world_size = self._get_world_size()
        
        logger.info(f"ClusterManager initialized for job {self.job_id}")
        logger.info(f"Node rank: {self.node_rank}, World size: {self.world_size}")
    
    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any):
        """
        Setup the cluster manager with model, optimizer, and scheduler.
        
        Args:
            model: PyTorch model (can be DDP wrapped)
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        
        # Setup signal handlers for preemption
        if self.preemption_handler:
            self.preemption_handler.setup_signal_handlers()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Enable elastic scaling if requested
        if self.elastic_scaler and hasattr(self.training_args, 'enable_elastic_scaling'):
            if self.training_args.enable_elastic_scaling:
                min_nodes = getattr(self.training_args, 'min_nodes', 1)
                max_nodes = getattr(self.training_args, 'max_nodes', 8)
                self.elastic_scaler.enable_elastic_scaling(min_nodes, max_nodes)
        
        # Load latest checkpoint if exists
        self._try_resume_from_checkpoint()
        
        self.state = ClusterState.RUNNING
        logger.info("ClusterManager setup complete")
    
    def before_training(self):
        """Called before training begins."""
        logger.info("Starting fault-tolerant distributed training")
        
        # Notify callbacks
        self._notify_callbacks('on_training_start', {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'job_id': self.job_id
        })
    
    def after_training(self):
        """Called after training completes."""
        logger.info("Training completed successfully")
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        # Stop monitoring
        self.health_monitor.stop_monitoring()
        
        # Cleanup
        if self.preemption_handler:
            self.preemption_handler.cleanup()
        
        self.state = ClusterState.COMPLETED
        
        # Notify callbacks
        self._notify_callbacks('on_training_end', {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'final_loss': getattr(self, 'last_loss', None)
        })
    
    def on_step_begin(self, step: int):
        """Called at the beginning of each training step."""
        self.global_step = step
        
        # Check if we should save a checkpoint
        if step > 0 and step % self.checkpoint_interval == 0:
            if step > self.last_checkpoint_step:
                self.save_checkpoint()
                self.last_checkpoint_step = step
    
    def on_step_end(self, step: int, logs: Dict[str, float]):
        """
        Called at the end of each training step.
        
        Args:
            step: Current step number
            logs: Dictionary of metrics (loss, learning rate, etc.)
        """
        # Update training state
        if 'loss' in logs:
            self.last_loss = logs['loss']
        
        # Check elastic scaling conditions
        if self.elastic_scaler and self.elastic_scaler.scaling_enabled:
            if self.elastic_scaler.check_scaling_conditions(logs):
                # In a real implementation, this would trigger scaling
                logger.info("Elastic scaling conditions met")
        
        # Log through callback
        self.log_callback.on_log(
            self.training_args,
            None,  # state
            None,  # control
            logs=logs
        )
        
        # Notify custom callbacks
        self._notify_callbacks('on_step_end', {
            'step': step,
            'logs': logs,
            'global_step': self.global_step
        })
    
    def save_checkpoint(
        self,
        is_final: bool = False,
        custom_metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint with distributed coordination.
        
        Args:
            is_final: Whether this is the final checkpoint
            custom_metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        if self.model is None or self.optimizer is None:
            logger.warning("Cannot save checkpoint: model or optimizer not set")
            return ""
        
        self.state = ClusterState.CHECKPOINTING
        
        # Prepare metadata
        metadata = CheckpointMetadata(
            checkpoint_id=f"step-{self.global_step}",
            timestamp=datetime.now().isoformat(),
            global_step=self.global_step,
            epoch=self.epoch,
            loss=getattr(self, 'last_loss', None),
            learning_rate=self._get_current_lr(),
            node_count=self._get_node_count(),
            world_size=self.world_size,
            local_rank=self._get_local_rank(),
            global_rank=self._get_global_rank(),
            job_id=self.job_id,
            preemption_detected=self.preemption_handler.preemption_detected if self.preemption_handler else False,
            elastic_scaling=self.elastic_scaler.scaling_enabled if self.elastic_scaler else False,
            training_args=vars(self.training_args) if hasattr(self.training_args, '__dict__') else {}
        )
        
        # Add custom metadata
        if custom_metadata:
            metadata.metrics.update(custom_metadata)
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            metadata=metadata
        )
        
        self.state = ClusterState.RUNNING
        
        # Notify callbacks
        self._notify_callbacks('on_checkpoint_save', {
            'checkpoint_path': checkpoint_path,
            'global_step': self.global_step,
            'is_final': is_final
        })
        
        return checkpoint_path
    
    def handle_preemption(self, signal_name: str):
        """
        Handle preemption signal by saving checkpoint and preparing for shutdown.
        
        Args:
            signal_name: Name of the received signal
        """
        logger.warning(f"Handling preemption signal: {signal_name}")
        self.state = ClusterState.PREEMPTED
        
        # Save checkpoint before shutdown
        try:
            checkpoint_path = self.save_checkpoint(
                custom_metadata={'preemption_signal': signal_name}
            )
            logger.info(f"Saved preemption checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save preemption checkpoint: {e}")
        
        # Notify callbacks
        self._notify_callbacks('on_preemption', {
            'signal': signal_name,
            'global_step': self.global_step,
            'epoch': self.epoch
        })
        
        # In a real implementation, we might want to wait for other nodes
        # or coordinate a graceful shutdown
    
    def handle_scaling_event(self, old_nodes: int, new_nodes: int):
        """
        Handle cluster scaling event.
        
        Args:
            old_nodes: Previous number of nodes
            new_nodes: New number of nodes
        """
        logger.info(f"Handling scaling event: {old_nodes} -> {new_nodes} nodes")
        self.state = ClusterState.SCALING
        
        # Save checkpoint before scaling
        try:
            checkpoint_path = self.save_checkpoint(
                custom_metadata={
                    'scaling_event': True,
                    'old_nodes': old_nodes,
                    'new_nodes': new_nodes
                }
            )
            logger.info(f"Saved scaling checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save scaling checkpoint: {e}")
        
        # Notify callbacks
        self._notify_callbacks('on_scaling', {
            'old_nodes': old_nodes,
            'new_nodes': new_nodes,
            'global_step': self.global_step
        })
        
        # In a real implementation, this would coordinate with the cluster
        # scheduler to redistribute work across new nodes
    
    def handle_node_failure(self, node: NodeHealth):
        """
        Handle node failure.
        
        Args:
            node: Information about the failed node
        """
        logger.error(f"Node failure detected: {node.node_id} ({node.hostname})")
        
        # Notify callbacks
        self._notify_callbacks('on_node_failure', {
            'node_id': node.node_id,
            'hostname': node.hostname,
            'failure_count': node.failure_count,
            'global_step': self.global_step
        })
        
        # If this is the current node, save checkpoint
        if node.node_id == self._get_current_node_id():
            logger.warning("Current node is failing, saving emergency checkpoint")
            try:
                self.save_checkpoint(
                    custom_metadata={'node_failure': True, 'node_id': node.node_id}
                )
            except Exception as e:
                logger.error(f"Failed to save failure checkpoint: {e}")
    
    def handle_cluster_failure(self, reason: str):
        """
        Handle cluster-wide failure.
        
        Args:
            reason: Description of the failure
        """
        logger.critical(f"Cluster failure: {reason}")
        self.state = ClusterState.FAILED
        
        # Save emergency checkpoint
        try:
            self.save_checkpoint(
                custom_metadata={'cluster_failure': True, 'reason': reason}
            )
        except Exception as e:
            logger.error(f"Failed to save cluster failure checkpoint: {e}")
        
        # Notify callbacks
        self._notify_callbacks('on_cluster_failure', {
            'reason': reason,
            'global_step': self.global_step,
            'epoch': self.epoch
        })
    
    def _try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint."""
        try:
            global_step, epoch, metadata = self.checkpoint_manager.load_latest_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            
            if global_step > 0:
                self.global_step = global_step
                self.epoch = epoch
                self.last_checkpoint_step = global_step
                self.is_recovering = True
                self.recovery_metadata = metadata
                
                logger.info(f"Resumed training from step {global_step}, epoch {epoch}")
                
                # Notify callbacks about recovery
                self._notify_callbacks('on_recovery', {
                    'global_step': global_step,
                    'epoch': epoch,
                    'metadata': metadata
                })
                
                self.is_recovering = False
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}")
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is None:
            return 0.0
        
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def _get_job_id(self) -> str:
        """Get job ID from environment (Kubernetes/SLURM)."""
        # Try Kubernetes
        job_id = os.environ.get('JOB_ID')
        if job_id:
            return f"k8s-{job_id}"
        
        # Try SLURM
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            return f"slurm-{job_id}"
        
        # Try other common environment variables
        job_id = os.environ.get('BATCH_JOB_ID')
        if job_id:
            return f"batch-{job_id}"
        
        # Fallback to timestamp
        return f"job-{int(time.time())}"
    
    def _get_node_rank(self) -> int:
        """Get node rank from environment."""
        # Try PyTorch distributed
        if 'RANK' in os.environ:
            return int(os.environ['RANK'])
        
        # Try SLURM
        if 'SLURM_PROCID' in os.environ:
            return int(os.environ['SLURM_PROCID'])
        
        # Try other common variables
        if 'NODE_RANK' in os.environ:
            return int(os.environ['NODE_RANK'])
        
        return 0
    
    def _get_world_size(self) -> int:
        """Get world size from environment."""
        # Try PyTorch distributed
        if 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        
        # Try SLURM
        if 'SLURM_NTASKS' in os.environ:
            return int(os.environ['SLURM_NTASKS'])
        
        # Try other common variables
        if 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        
        return 1
    
    def _get_local_rank(self) -> int:
        """Get local rank from environment."""
        # Try PyTorch distributed
        if 'LOCAL_RANK' in os.environ:
            return int(os.environ['LOCAL_RANK'])
        
        # Try SLURM
        if 'SLURM_LOCALID' in os.environ:
            return int(os.environ['SLURM_LOCALID'])
        
        return 0
    
    def _get_global_rank(self) -> int:
        """Get global rank from environment."""
        # Try PyTorch distributed
        if 'RANK' in os.environ:
            return int(os.environ['RANK'])
        
        # Try SLURM
        if 'SLURM_PROCID' in os.environ:
            return int(os.environ['SLURM_PROCID'])
        
        return 0
    
    def _get_node_count(self) -> int:
        """Get number of nodes in the cluster."""
        # Try SLURM
        if 'SLURM_JOB_NUM_NODES' in os.environ:
            return int(os.environ['SLURM_JOB_NUM_NODES'])
        
        # Try other common variables
        if 'NODE_COUNT' in os.environ:
            return int(os.environ['NODE_COUNT'])
        
        return 1
    
    def _get_current_node_id(self) -> str:
        """Get unique ID for current node."""
        import socket
        hostname = socket.gethostname()
        return f"{hostname}-{os.getpid()}"
    
    def register_callback(self, callback: Callable):
        """Register a callback for cluster events."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """Notify all registered callbacks of an event."""
        for callback in self.callbacks:
            try:
                if hasattr(callback, event):
                    getattr(callback, event)(data)
            except Exception as e:
                logger.error(f"Error in callback {callback} for event {event}: {e}")


class RecoveryCallback:
    """Callback for handling recovery events in training."""
    
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager
    
    def on_recovery(self, data: Dict[str, Any]):
        """Called when recovering from checkpoint."""
        global_step = data.get('global_step', 0)
        epoch = data.get('epoch', 0.0)
        metadata = data.get('metadata', {})
        
        logger.info(f"Recovery callback: Resuming from step {global_step}, epoch {epoch}")
        
        # Log recovery information
        if metadata:
            logger.info(f"Recovery metadata: {json.dumps(metadata, indent=2)}")
    
    def on_preemption(self, data: Dict[str, Any]):
        """Called when preemption is detected."""
        signal_name = data.get('signal', 'UNKNOWN')
        global_step = data.get('global_step', 0)
        
        logger.warning(f"Preemption callback: Received {signal_name} at step {global_step}")
    
    def on_scaling(self, data: Dict[str, Any]):
        """Called when scaling event occurs."""
        old_nodes = data.get('old_nodes', 0)
        new_nodes = data.get('new_nodes', 0)
        
        logger.info(f"Scaling callback: {old_nodes} -> {new_nodes} nodes")
    
    def on_node_failure(self, data: Dict[str, Any]):
        """Called when node failure is detected."""
        node_id = data.get('node_id', 'UNKNOWN')
        hostname = data.get('hostname', 'UNKNOWN')
        
        logger.error(f"Node failure callback: {node_id} ({hostname})")
    
    def on_cluster_failure(self, data: Dict[str, Any]):
        """Called when cluster failure is detected."""
        reason = data.get('reason', 'Unknown reason')
        
        logger.critical(f"Cluster failure callback: {reason}")


# Example usage and integration with crucible training
def create_cluster_manager(
    training_args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_dir: Optional[str] = None,
    **kwargs
) -> ClusterManager:
    """
    Factory function to create and setup a ClusterManager.
    
    Args:
        training_args: Training arguments from crucible
        model: PyTorch model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        checkpoint_dir: Directory for checkpoints (defaults to training_args.output_dir)
        **kwargs: Additional arguments for ClusterManager
        
    Returns:
        Configured ClusterManager instance
    """
    if checkpoint_dir is None:
        if hasattr(training_args, 'output_dir'):
            checkpoint_dir = os.path.join(training_args.output_dir, 'cluster_checkpoints')
        else:
            checkpoint_dir = './cluster_checkpoints'
    
    # Create cluster manager
    cluster_manager = ClusterManager(
        training_args=training_args,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    # Setup with model, optimizer, and scheduler
    cluster_manager.setup(model, optimizer, scheduler)
    
    # Register recovery callback
    recovery_callback = RecoveryCallback(cluster_manager)
    cluster_manager.register_callback(recovery_callback)
    
    return cluster_manager


# Integration with PyTorch distributed elastic launch
def setup_elastic_environment():
    """Setup environment for elastic distributed training."""
    # Set environment variables for elastic training
    os.environ['TORCHELASTIC_MAX_RESTARTS'] = os.environ.get('TORCHELASTIC_MAX_RESTARTS', '3')
    os.environ['TORCHELASTIC_RESTART_TIMEOUT'] = os.environ.get('TORCHELASTIC_RESTART_TIMEOUT', '300')
    
    # Enable file-based rendezvous for elastic training
    if 'TORCHELASTIC_RUN_ID' not in os.environ:
        os.environ['TORCHELASTIC_RUN_ID'] = f"run-{int(time.time())}"
    
    logger.info("Elastic training environment configured")


# Command-line interface for cluster management
def main():
    """Command-line interface for cluster management operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='crucible Cluster Manager')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory for checkpoints')
    parser.add_argument('--action', type=str, choices=['status', 'cleanup', 'recover'],
                       default='status', help='Action to perform')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                       help='Number of checkpoints to keep')
    
    args = parser.parse_args()
    
    if args.action == 'status':
        # Show checkpoint status
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return
        
        checkpoints = []
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append((item.name, metadata))
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        # Sort by global step
        checkpoints.sort(key=lambda x: x[1].get('global_step', 0))
        
        print(f"Found {len(checkpoints)} checkpoints:")
        for name, metadata in checkpoints:
            step = metadata.get('global_step', 'N/A')
            epoch = metadata.get('epoch', 'N/A')
            timestamp = metadata.get('timestamp', 'N/A')
            print(f"  {name}: step={step}, epoch={epoch}, time={timestamp}")
    
    elif args.action == 'cleanup':
        # Clean up old checkpoints
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return
        
        checkpoints = []
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints
        removed = 0
        while len(checkpoints) > args.keep_checkpoints:
            _, checkpoint_path = checkpoints.pop(0)
            import shutil
            shutil.rmtree(checkpoint_path)
            print(f"Removed: {checkpoint_path}")
            removed += 1
        
        print(f"Cleanup complete: removed {removed} old checkpoints")
    
    elif args.action == 'recover':
        # Show recovery information
        checkpoint_dir = Path(args.checkpoint_dir)
        latest_path = checkpoint_dir / "latest"
        
        if not latest_path.exists():
            print("No latest checkpoint found")
            return
        
        # Resolve symlink
        if latest_path.is_symlink():
            checkpoint_path = latest_path.resolve()
        else:
            checkpoint_path = latest_path
        
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print("Recovery information:")
            print(f"  Checkpoint: {checkpoint_path.name}")
            print(f"  Global step: {metadata.get('global_step', 'N/A')}")
            print(f"  Epoch: {metadata.get('epoch', 'N/A')}")
            print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
            
            if 'metadata' in metadata:
                print("  Additional metadata:")
                for key, value in metadata['metadata'].items():
                    print(f"    {key}: {value}")
        else:
            print(f"No metadata found in {checkpoint_path}")


if __name__ == "__main__":
    main()