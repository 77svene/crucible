"""
Fault-Tolerant Distributed Training with Auto-Recovery for crucible

This module implements automatic checkpointing, preemption handling, and elastic scaling
for multi-node distributed training. It integrates with Kubernetes/SLURM to resume
failed jobs from the last checkpoint without manual intervention.
"""

import os
import sys
import time
import signal
import logging
import threading
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.cuda.amp import GradScaler

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ..utils.logging import get_logger
from ..extras.constants import TRAINER_STATE_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, SCALER_NAME
from ..extras.misc import get_current_device, save_config, load_config

logger = get_logger(__name__)


class TrainingStatus(Enum):
    """Training job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"
    RECOVERING = "recovering"


@dataclass
class ResilienceConfig:
    """Configuration for fault-tolerant training."""
    
    # Checkpointing settings
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3
    save_on_each_node: bool = False
    async_checkpointing: bool = True
    checkpoint_timeout: int = 3600  # seconds
    
    # Recovery settings
    auto_resume: bool = True
    max_restarts: int = 5
    restart_delay: int = 30  # seconds
    resume_from_checkpoint: Optional[str] = None
    
    # Elastic scaling
    min_nodes: int = 1
    max_nodes: int = 8
    elastic_timeout: int = 600  # seconds
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    node_health_check: bool = True
    heartbeat_timeout: int = 120  # seconds
    
    # Preemption handling
    preemption_signals: List[str] = field(default_factory=lambda: ["SIGTERM", "SIGINT", "SIGUSR1"])
    graceful_shutdown_timeout: int = 300  # seconds
    
    # Storage settings
    persistent_storage: str = "/shared/checkpoints"
    storage_type: str = "nfs"  # nfs, gcs, s3, azure_blob
    
    # Logging
    log_recovery_events: bool = True
    recovery_log_file: str = "recovery.log"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_total_limit < 1:
            raise ValueError("save_total_limit must be at least 1")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes cannot be greater than max_nodes")
        
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.persistent_storage).mkdir(parents=True, exist_ok=True)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking."""
    checkpoint_id: str
    timestamp: str
    step: int
    epoch: float
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    global_step: int = 0
    world_size: int = 1
    node_rank: int = 0
    local_rank: int = 0
    hostname: str = ""
    pid: int = 0
    training_status: TrainingStatus = TrainingStatus.RUNNING
    checksum: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['training_status'] = self.training_status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        if 'training_status' in data:
            data['training_status'] = TrainingStatus(data['training_status'])
        return cls(**data)


class ResilienceCallback(TrainerCallback):
    """
    Callback for integrating resilience features with Hugging Face Trainer.
    
    Extends the existing LogCallback to handle recovery events, checkpointing,
    and elastic scaling during training.
    """
    
    def __init__(self, resilience_config: ResilienceConfig):
        self.config = resilience_config
        self.resilience_manager = ResilienceManager(resilience_config)
        self.last_checkpoint_time = time.time()
        self.preemption_detected = False
        self.recovery_events = []
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for preemption detection."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"Received signal {signal_name}, initiating graceful shutdown")
            self.preemption_detected = True
            self.resilience_manager.handle_preemption(signal_name)
        
        for signal_name in self.config.preemption_signals:
            try:
                signal_num = getattr(signal, signal_name)
                signal.signal(signal_num, signal_handler)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Could not setup handler for signal {signal_name}: {e}")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize resilience manager after trainer initialization."""
        self.resilience_manager.initialize(
            model=kwargs.get('model'),
            optimizer=kwargs.get('optimizer'),
            lr_scheduler=kwargs.get('lr_scheduler'),
            scaler=kwargs.get('scaler'),
            training_args=args
        )
        
        # Log recovery event
        self._log_recovery_event("training_initialized", {
            "world_size": state.world_size,
            "local_rank": args.local_rank,
            "device": str(get_current_device())
        })
        
        return control
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle training start and potential recovery."""
        if self.config.auto_resume:
            resume_path = self.resilience_manager.find_recovery_checkpoint()
            if resume_path:
                logger.info(f"Found recovery checkpoint at {resume_path}")
                self._log_recovery_event("auto_recovery_started", {"checkpoint_path": resume_path})
        
        # Check for existing checkpoints
        last_checkpoint = get_last_checkpoint(self.config.checkpoint_dir)
        if last_checkpoint and not args.overwrite_output_dir:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}")
            self._log_recovery_event("resuming_from_checkpoint", {"path": last_checkpoint})
        
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle periodic checkpointing and health checks."""
        current_time = time.time()
        
        # Periodic checkpointing
        if state.global_step % self.config.save_steps == 0 and state.global_step > 0:
            if current_time - self.last_checkpoint_time >= self.config.checkpoint_timeout:
                logger.info(f"Saving checkpoint at step {state.global_step}")
                self._save_checkpoint_with_resilience(args, state, kwargs)
                self.last_checkpoint_time = current_time
        
        # Health monitoring
        if self.config.node_health_check and state.global_step % (self.config.health_check_interval // args.logging_steps) == 0:
            self.resilience_manager.check_node_health()
        
        # Check for preemption
        if self.preemption_detected:
            logger.warning("Preemption detected, saving emergency checkpoint")
            self._save_emergency_checkpoint(args, state, kwargs)
            control.should_training_stop = True
        
        return control
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Enhanced save with resilience features."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            checkpoint_id=f"ckpt-{state.global_step}-{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            step=state.global_step,
            epoch=state.epoch,
            loss=state.log_history[-1].get('loss') if state.log_history else None,
            learning_rate=state.log_history[-1].get('learning_rate') if state.log_history else None,
            global_step=state.global_step,
            world_size=state.world_size,
            node_rank=int(os.environ.get("NODE_RANK", 0)),
            local_rank=args.local_rank,
            hostname=self.resilience_manager.get_hostname(),
            pid=os.getpid(),
            training_status=TrainingStatus.RUNNING
        )
        
        # Save metadata
        metadata_path = os.path.join(checkpoint_path, "resilience_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Copy to persistent storage if configured
        if self.config.persistent_storage:
            persistent_path = os.path.join(
                self.config.persistent_storage,
                f"checkpoint-{state.global_step}"
            )
            self.resilience_manager.copy_checkpoint_to_persistent(checkpoint_path, persistent_path)
        
        # Log save event
        self._log_recovery_event("checkpoint_saved", {
            "step": state.global_step,
            "path": checkpoint_path,
            "metadata": metadata.to_dict()
        })
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle training completion and cleanup."""
        # Save final checkpoint
        if state.is_world_process_zero:
            final_metadata = CheckpointMetadata(
                checkpoint_id=f"final-{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                step=state.global_step,
                epoch=state.epoch,
                training_status=TrainingStatus.COMPLETED,
                extra_info={"final_loss": state.log_history[-1].get('loss') if state.log_history else None}
            )
            
            final_path = os.path.join(args.output_dir, "final_checkpoint")
            Path(final_path).mkdir(exist_ok=True)
            
            with open(os.path.join(final_path, "resilience_metadata.json"), 'w') as f:
                json.dump(final_metadata.to_dict(), f, indent=2)
        
        # Log completion
        self._log_recovery_event("training_completed", {
            "total_steps": state.global_step,
            "final_epoch": state.epoch,
            "best_metric": state.best_metric
        })
        
        # Save recovery log
        self._save_recovery_log(args.output_dir)
        
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Log training metrics with recovery context."""
        # Add resilience metrics to logs
        if self.config.log_recovery_events:
            logs["resilience/checkpoints_saved"] = self.resilience_manager.checkpoints_saved
            logs["resilience/restarts"] = self.resilience_manager.restart_count
            logs["resilience/preemptions_handled"] = self.resilience_manager.preemptions_handled
        
        return control
    
    def _save_checkpoint_with_resilience(self, args: TrainingArguments, state: TrainerState, kwargs: Dict[str, Any]):
        """Save checkpoint with resilience features."""
        if self.config.async_checkpointing:
            # Run checkpointing in background thread
            thread = threading.Thread(
                target=self._async_checkpoint_save,
                args=(args, state, kwargs),
                daemon=True
            )
            thread.start()
        else:
            # Synchronous checkpointing
            control = self.on_save(args, state, TrainerControl(), **kwargs)
            # Trigger trainer save
            if 'trainer' in kwargs:
                kwargs['trainer'].save_state()
    
    def _async_checkpoint_save(self, args: TrainingArguments, state: TrainerState, kwargs: Dict[str, Any]):
        """Asynchronous checkpoint saving."""
        try:
            # Create temporary directory for atomic save
            temp_dir = tempfile.mkdtemp(dir=args.output_dir)
            
            # Save to temporary directory
            if 'trainer' in kwargs:
                # Use trainer's save method if available
                trainer = kwargs['trainer']
                trainer.save_state(output_dir=temp_dir)
                
                # Save model
                if hasattr(trainer, '_save'):
                    trainer._save(output_dir=temp_dir)
            else:
                # Fallback to manual saving
                self._manual_save_checkpoint(temp_dir, state, kwargs)
            
            # Atomic move to final location
            final_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.move(temp_dir, final_path)
            
            # Update metadata
            self.on_save(args, state, TrainerControl(), **kwargs)
            
            logger.info(f"Asynchronous checkpoint saved to {final_path}")
            
        except Exception as e:
            logger.error(f"Async checkpoint save failed: {e}")
            # Cleanup temporary directory
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _manual_save_checkpoint(self, output_dir: str, state: TrainerState, kwargs: Dict[str, Any]):
        """Manual checkpoint saving when trainer methods are not available."""
        # Save model state
        model = kwargs.get('model')
        if model is not None:
            if isinstance(model, (DDP, FSDP)):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save(model_state, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save optimizer state
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        
        # Save scheduler state
        scheduler = kwargs.get('lr_scheduler')
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        
        # Save scaler state
        scaler = kwargs.get('scaler')
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        
        # Save training arguments
        training_args = kwargs.get('training_args')
        if training_args is not None:
            save_config(training_args, os.path.join(output_dir, "training_args.bin"))
    
    def _save_emergency_checkpoint(self, args: TrainingArguments, state: TrainerState, kwargs: Dict[str, Any]):
        """Save emergency checkpoint when preemption is detected."""
        emergency_path = os.path.join(args.output_dir, f"emergency-{state.global_step}-{int(time.time())}")
        Path(emergency_path).mkdir(parents=True, exist_ok=True)
        
        # Save minimal state for recovery
        emergency_state = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "preemption_detected": True
        }
        
        with open(os.path.join(emergency_path, "emergency_state.json"), 'w') as f:
            json.dump(emergency_state, f)
        
        # Save model and optimizer if possible
        try:
            self._manual_save_checkpoint(emergency_path, state, kwargs)
        except Exception as e:
            logger.error(f"Failed to save model/optimizer in emergency checkpoint: {e}")
        
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
    
    def _log_recovery_event(self, event_type: str, data: Dict[str, Any]):
        """Log recovery event with timestamp."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "hostname": self.resilience_manager.get_hostname(),
            "pid": os.getpid()
        }
        self.recovery_events.append(event)
        
        if self.config.log_recovery_events:
            logger.info(f"Recovery event: {event_type} - {json.dumps(data)}")
    
    def _save_recovery_log(self, output_dir: str):
        """Save recovery events log."""
        if not self.recovery_events:
            return
        
        log_path = os.path.join(output_dir, self.config.recovery_log_file)
        try:
            with open(log_path, 'w') as f:
                json.dump(self.recovery_events, f, indent=2)
            logger.info(f"Recovery log saved to {log_path}")
        except Exception as e:
            logger.error(f"Failed to save recovery log: {e}")


class ResilienceManager:
    """
    Core manager for fault-tolerant distributed training.
    
    Handles checkpointing, recovery, elastic scaling, and integration with
    cluster schedulers (Kubernetes/SLURM).
    """
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.checkpoints_saved = 0
        self.restart_count = 0
        self.preemptions_handled = 0
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.node_health = {}
        self.recovery_lock = threading.Lock()
        
        # Distributed training state
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.node_rank = 0
        
        # Training components
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.training_args = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup resilience-specific logging."""
        self.logger = logging.getLogger("resilience")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler for resilience logs
        if not self.logger.handlers:
            handler = logging.FileHandler("resilience.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def initialize(self, model=None, optimizer=None, lr_scheduler=None, scaler=None, training_args=None):
        """Initialize resilience manager with training components."""
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.training_args = training_args
        
        # Get distributed training info
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.node_rank = int(os.environ.get("NODE_RANK", 0))
        
        self.is_initialized = True
        self.logger.info(f"Resilience manager initialized: world_size={self.world_size}, "
                        f"rank={self.rank}, node_rank={self.node_rank}")
    
    def get_hostname(self) -> str:
        """Get current hostname."""
        import socket
        return socket.gethostname()
    
    def check_node_health(self) -> bool:
        """Check health of current node."""
        try:
            # Basic health checks
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "hostname": self.get_hostname(),
                "pid": os.getpid(),
                "memory_usage": self._get_memory_usage(),
                "gpu_memory": self._get_gpu_memory(),
                "uptime": time.time() - self.start_time
            }
            
            # Check GPU health
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.get_device_properties(i)
                    except Exception as e:
                        health_status[f"gpu_{i}_error"] = str(e)
            
            self.node_health[self.rank] = health_status
            self.last_heartbeat = time.time()
            
            # Check for stale nodes
            self._check_stale_nodes()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent
        }
    
    def _get_gpu_memory(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory usage."""
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu_memory[i] = {
                        "allocated_gb": memory_allocated,
                        "reserved_gb": memory_reserved
                    }
                except Exception as e:
                    gpu_memory[i] = {"error": str(e)}
        return gpu_memory
    
    def _check_stale_nodes(self):
        """Check for nodes that haven't sent heartbeat."""
        current_time = time.time()
        stale_nodes = []
        
        for rank, health in self.node_health.items():
            last_update = datetime.fromisoformat(health["timestamp"]).timestamp()
            if current_time - last_update > self.config.heartbeat_timeout:
                stale_nodes.append(rank)
        
        if stale_nodes:
            self.logger.warning(f"Stale nodes detected: {stale_nodes}")
            self._handle_stale_nodes(stale_nodes)
    
    def _handle_stale_nodes(self, stale_nodes: List[int]):
        """Handle stale nodes by triggering recovery."""
        if self.rank == 0:  # Only master node handles recovery
            self.logger.info(f"Initiating recovery for stale nodes: {stale_nodes}")
            self.trigger_recovery("stale_nodes_detected", {"stale_nodes": stale_nodes})
    
    def handle_preemption(self, signal_name: str):
        """Handle preemption signal from cluster scheduler."""
        self.preemptions_handled += 1
        self.logger.warning(f"Handling preemption signal: {signal_name}")
        
        # Update training status
        if self.is_initialized:
            self._update_training_status(TrainingStatus.PREEMPTED)
        
        # Save emergency checkpoint
        self.save_emergency_checkpoint()
        
        # Notify cluster scheduler if possible
        self._notify_scheduler("preemption_handled")
    
    def save_checkpoint(self, step: int, epoch: float, loss: Optional[float] = None):
        """Save checkpoint with resilience features."""
        if not self.is_initialized:
            self.logger.warning("Resilience manager not initialized, skipping checkpoint")
            return
        
        with self.recovery_lock:
            try:
                # Generate checkpoint ID
                checkpoint_id = f"ckpt-{step}-{int(time.time())}"
                checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_id)
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                
                # Save model state using distributed checkpointing if available
                if hasattr(torch.distributed, 'checkpoint') and self.world_size > 1:
                    self._save_distributed_checkpoint(checkpoint_dir, step)
                else:
                    self._save_single_checkpoint(checkpoint_dir, step)
                
                # Save metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.now().isoformat(),
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    world_size=self.world_size,
                    rank=self.rank,
                    node_rank=self.node_rank,
                    hostname=self.get_hostname(),
                    pid=os.getpid(),
                    training_status=TrainingStatus.RUNNING
                )
                
                metadata_path = os.path.join(checkpoint_dir, "resilience_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                # Calculate and save checksum
                checksum = self._calculate_checkpoint_checksum(checkpoint_dir)
                metadata.checksum = checksum
                
                # Copy to persistent storage
                if self.config.persistent_storage:
                    self._copy_to_persistent_storage(checkpoint_dir, checkpoint_id)
                
                # Manage checkpoint rotation
                self._rotate_checkpoints()
                
                self.checkpoints_saved += 1
                self.logger.info(f"Checkpoint saved: {checkpoint_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {e}")
                raise
    
    def _save_distributed_checkpoint(self, checkpoint_dir: str, step: int):
        """Save checkpoint using PyTorch distributed checkpointing."""
        try:
            # Get state dictionaries
            model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
            
            # Save using FileSystemWriter for distributed storage
            storage_writer = FileSystemWriter(checkpoint_dir)
            
            # Save model state
            state_dict = {
                "model": model_state,
                "step": step,
                "epoch": self.training_args.num_train_epochs if self.training_args else 0
            }
            
            # Use torch.save for now (distributed checkpointing can be added later)
            torch.save(state_dict, os.path.join(checkpoint_dir, "model_state.pt"))
            
            # Save optimizer state
            if optimizer_state:
                torch.save(optimizer_state, os.path.join(checkpoint_dir, "optimizer_state.pt"))
            
            # Save scheduler state
            if self.lr_scheduler:
                torch.save(self.lr_scheduler.state_dict(), 
                          os.path.join(checkpoint_dir, "scheduler_state.pt"))
            
            # Save scaler state
            if self.scaler:
                torch.save(self.scaler.state_dict(), 
                          os.path.join(checkpoint_dir, "scaler_state.pt"))
            
        except Exception as e:
            self.logger.error(f"Distributed checkpoint save failed: {e}")
            # Fallback to single checkpoint
            self._save_single_checkpoint(checkpoint_dir, step)
    
    def _save_single_checkpoint(self, checkpoint_dir: str, step: int):
        """Save checkpoint for single process or as fallback."""
        # Save model
        if self.model is not None:
            if isinstance(self.model, (DDP, FSDP)):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            torch.save(model_state, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save optimizer
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), 
                      os.path.join(checkpoint_dir, OPTIMIZER_NAME))
        
        # Save scheduler
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), 
                      os.path.join(checkpoint_dir, SCHEDULER_NAME))
        
        # Save scaler
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), 
                      os.path.join(checkpoint_dir, SCALER_NAME))
        
        # Save training arguments
        if self.training_args is not None:
            save_config(self.training_args, 
                       os.path.join(checkpoint_dir, "training_args.bin"))
    
    def save_emergency_checkpoint(self):
        """Save emergency checkpoint when preemption is detected."""
        emergency_dir = os.path.join(
            self.config.checkpoint_dir,
            f"emergency-{self.rank}-{int(time.time())}"
        )
        Path(emergency_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save minimal state for recovery
            emergency_state = {
                "rank": self.rank,
                "timestamp": datetime.now().isoformat(),
                "hostname": self.get_hostname(),
                "pid": os.getpid(),
                "preemption": True
            }
            
            with open(os.path.join(emergency_dir, "emergency_state.json"), 'w') as f:
                json.dump(emergency_state, f)
            
            # Save model and optimizer if possible
            self._save_single_checkpoint(emergency_dir, 0)
            
            self.logger.info(f"Emergency checkpoint saved to {emergency_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def find_recovery_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for recovery."""
        if not self.config.auto_resume:
            return None
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        # Find all checkpoint directories
        checkpoint_dirs = []
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoint_dirs.append((step, item))
                except (IndexError, ValueError):
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Sort by step number and get latest
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_step, latest_dir = checkpoint_dirs[0]
        
        # Verify checkpoint integrity
        if self._verify_checkpoint_integrity(str(latest_dir)):
            return str(latest_dir)
        
        # Try previous checkpoints if latest is corrupted
        for step, dir_path in checkpoint_dirs[1:]:
            if self._verify_checkpoint_integrity(str(dir_path)):
                self.logger.warning(f"Latest checkpoint corrupted, using step {step}")
                return str(dir_path)
        
        return None
    
    def _verify_checkpoint_integrity(self, checkpoint_dir: str) -> bool:
        """Verify checkpoint integrity using checksum."""
        metadata_path = os.path.join(checkpoint_dir, "resilience_metadata.json")
        if not os.path.exists(metadata_path):
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # If checksum exists, verify it
            if 'checksum' in metadata and metadata['checksum']:
                calculated_checksum = self._calculate_checkpoint_checksum(checkpoint_dir)
                return calculated_checksum == metadata['checksum']
            
            # Basic verification: check essential files exist
            essential_files = ["pytorch_model.bin", OPTIMIZER_NAME, SCHEDULER_NAME]
            for file in essential_files:
                if not os.path.exists(os.path.join(checkpoint_dir, file)):
                    # Check for distributed checkpoint files
                    if not os.path.exists(os.path.join(checkpoint_dir, "model_state.pt")):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Checkpoint verification failed: {e}")
            return False
    
    def _calculate_checkpoint_checksum(self, checkpoint_dir: str) -> str:
        """Calculate checksum for checkpoint directory."""
        hasher = hashlib.sha256()
        
        # Hash all files in checkpoint directory
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in sorted(files):  # Sort for consistent ordering
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        # Read in chunks for large files
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                except Exception as e:
                    self.logger.warning(f"Could not hash file {file_path}: {e}")
        
        return hasher.hexdigest()
    
    def _rotate_checkpoints(self):
        """Rotate checkpoints to maintain save_total_limit."""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        # Get all checkpoint directories with metadata
        checkpoints = []
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                metadata_path = item / "resilience_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        step = metadata.get('step', 0)
                        checkpoints.append((step, item))
                    except Exception:
                        continue
        
        # Sort by step number (oldest first)
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove oldest checkpoints if exceeding limit
        while len(checkpoints) > self.config.save_total_limit:
            step, checkpoint_path = checkpoints.pop(0)
            try:
                shutil.rmtree(checkpoint_path)
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def _copy_to_persistent_storage(self, source_dir: str, checkpoint_id: str):
        """Copy checkpoint to persistent storage."""
        if not self.config.persistent_storage:
            return
        
        try:
            dest_dir = os.path.join(self.config.persistent_storage, checkpoint_id)
            
            # Use rsync for efficient copying if available
            import subprocess
            try:
                subprocess.run(
                    ["rsync", "-avz", "--progress", source_dir + "/", dest_dir + "/"],
                    check=True,
                    capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to shutil
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)
            
            self.logger.info(f"Checkpoint copied to persistent storage: {dest_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to copy to persistent storage: {e}")
    
    def _update_training_status(self, status: TrainingStatus):
        """Update training status in metadata."""
        # This would update the status in checkpoint metadata
        # Implementation depends on specific requirements
        pass
    
    def _notify_scheduler(self, event: str):
        """Notify cluster scheduler about events."""
        # Implementation depends on scheduler (Kubernetes/SLURM)
        # This could involve updating job annotations, sending signals, etc.
        self.logger.info(f"Scheduler notification: {event}")
    
    def trigger_recovery(self, reason: str, context: Dict[str, Any] = None):
        """Trigger recovery process."""
        self.logger.info(f"Triggering recovery: {reason}")
        
        # Increment restart count
        self.restart_count += 1
        
        if self.restart_count > self.config.max_restarts:
            self.logger.error(f"Maximum restarts ({self.config.max_restarts}) exceeded")
            return False
        
        # Find recovery checkpoint
        recovery_checkpoint = self.find_recovery_checkpoint()
        if not recovery_checkpoint:
            self.logger.error("No recovery checkpoint found")
            return False
        
        # Wait before restart
        if self.config.restart_delay > 0:
            self.logger.info(f"Waiting {self.config.restart_delay} seconds before restart")
            time.sleep(self.config.restart_delay)
        
        # Signal for restart (implementation depends on deployment)
        self.logger.info(f"Recovery triggered, restarting from {recovery_checkpoint}")
        return True
    
    def get_elastic_config(self) -> LaunchConfig:
        """Get configuration for elastic launch."""
        return LaunchConfig(
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            nproc_per_node=self._get_nproc_per_node(),
            run_id=self._get_run_id(),
            rdzv_backend="c10d",
            rdzv_endpoint=self._get_rendezvous_endpoint(),
            rdzv_configs={"timeout": self.config.elastic_timeout},
            max_restarts=self.config.max_restarts,
            monitor_interval=5,
            start_method="spawn"
        )
    
    def _get_nproc_per_node(self) -> int:
        """Get number of processes per node."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1
    
    def _get_run_id(self) -> str:
        """Get unique run ID for elastic training."""
        # Generate or retrieve run ID from environment
        run_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id:
            run_id = f"run-{int(time.time())}-{self.get_hostname()}"
            os.environ["TORCHELASTIC_RUN_ID"] = run_id
        return run_id
    
    def _get_rendezvous_endpoint(self) -> str:
        """Get rendezvous endpoint for elastic training."""
        # Try to get from environment
        endpoint = os.environ.get("TORCHELASTIC_RDZV_ENDPOINT")
        if endpoint:
            return endpoint
        
        # Default to localhost for single node
        return "localhost:29400"


def create_resilience_callback(config: ResilienceConfig) -> ResilienceCallback:
    """Factory function to create resilience callback."""
    return ResilienceCallback(config)


def setup_elastic_training(
    train_func: Callable,
    resilience_config: ResilienceConfig,
    *args, **kwargs
):
    """
    Setup elastic training with fault tolerance.
    
    Args:
        train_func: Training function to execute
        resilience_config: Configuration for resilience
        *args, **kwargs: Arguments to pass to train_func
    """
    # Get elastic configuration
    manager = ResilienceManager(resilience_config)
    elastic_config = manager.get_elastic_config()
    
    # Wrap training function with error handling
    @record
    def wrapped_train_func(local_rank, *args, **kwargs):
        try:
            # Set local rank for distributed training
            os.environ["LOCAL_RANK"] = str(local_rank)
            
            # Execute training function
            return train_func(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Training failed on local_rank {local_rank}: {e}")
            
            # Trigger recovery if configured
            if resilience_config.auto_resume:
                manager.trigger_recovery("training_failure", {"error": str(e)})
            
            raise
    
    # Launch with elastic
    with elastic_launch(config=elastic_config, entrypoint=wrapped_train_func) as launcher:
        return launcher(*args, **kwargs)


def resume_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Resume training from checkpoint with resilience features.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        scaler: GradScaler for mixed precision
        strict: Whether to strictly enforce state dict keys
    
    Returns:
        Dictionary with recovery information
    """
    recovery_info = {
        "checkpoint_path": checkpoint_path,
        "resumed": False,
        "step": 0,
        "epoch": 0.0,
        "loss": None
    }
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
        return recovery_info
    
    try:
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, "resilience_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            recovery_info.update({
                "step": metadata.get("step", 0),
                "epoch": metadata.get("epoch", 0.0),
                "loss": metadata.get("loss"),
                "metadata": metadata
            })
        
        # Load model state
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Handle DDP/FSDP wrapped models
            if isinstance(model, (DDP, FSDP)):
                model.module.load_state_dict(state_dict, strict=strict)
            else:
                model.load_state_dict(state_dict, strict=strict)
            
            logger.info(f"Model state loaded from {model_path}")
        
        # Load optimizer state
        if optimizer is not None:
            optimizer_path = os.path.join(checkpoint_path, OPTIMIZER_NAME)
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                logger.info("Optimizer state loaded")
        
        # Load scheduler state
        if scheduler is not None:
            scheduler_path = os.path.join(checkpoint_path, SCHEDULER_NAME)
            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path))
                logger.info("Scheduler state loaded")
        
        # Load scaler state
        if scaler is not None:
            scaler_path = os.path.join(checkpoint_path, SCALER_NAME)
            if os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path))
                logger.info("Scaler state loaded")
        
        recovery_info["resumed"] = True
        logger.info(f"Successfully resumed from checkpoint: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        recovery_info["error"] = str(e)
    
    return recovery_info


# Integration with existing crucible training
def integrate_with_trainer(trainer, resilience_config: ResilienceConfig):
    """
    Integrate resilience features with existing Hugging Face Trainer.
    
    Args:
        trainer: Hugging Face Trainer instance
        resilience_config: Resilience configuration
    """
    # Add resilience callback
    resilience_callback = create_resilience_callback(resilience_config)
    trainer.add_callback(resilience_callback)
    
    # Store resilience manager in trainer for access
    trainer.resilience_manager = resilience_callback.resilience_manager
    
    # Patch trainer methods for resilience
    original_train = trainer.train
    
    def resilient_train(*args, **kwargs):
        """Training with resilience features."""
        # Check for recovery checkpoint
        if resilience_config.auto_resume:
            recovery_checkpoint = trainer.resilience_manager.find_recovery_checkpoint()
            if recovery_checkpoint:
                logger.info(f"Found recovery checkpoint: {recovery_checkpoint}")
                
                # Load recovery information
                recovery_info = resume_from_checkpoint(
                    recovery_checkpoint,
                    trainer.model,
                    trainer.optimizer,
                    trainer.lr_scheduler,
                    getattr(trainer, 'scaler', None)
                )
                
                # Update trainer state if needed
                if recovery_info["resumed"] and hasattr(trainer, 'state'):
                    trainer.state.global_step = recovery_info["step"]
                    trainer.state.epoch = recovery_info["epoch"]
        
        # Execute training with original method
        return original_train(*args, **kwargs)
    
    # Replace train method
    trainer.train = resilient_train
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = ResilienceConfig(
        checkpoint_dir="./checkpoints",
        save_steps=100,
        save_total_limit=3,
        auto_resume=True,
        max_restarts=3,
        persistent_storage="/mnt/shared/checkpoints"
    )
    
    # Create resilience callback
    callback = create_resilience_callback(config)
    
    print("Resilience module loaded successfully")
    print(f"Configuration: {config}")