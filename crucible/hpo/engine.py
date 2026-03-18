"""
Intelligent Hyperparameter Optimization (HPO) Engine for crucible
Implements multi-fidelity Bayesian optimization with early stopping for efficient hyperparameter tuning.
"""

import os
import json
import logging
import asyncio
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time
import numpy as np
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.storages import RDBStorage
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..train import TrainingArguments, Trainer
from ..data import get_dataset, load_tokenizer
from ..model import load_model
from ..extras.logging import get_logger
from ..extras.callbacks import CallbackHandler, ProgressCallback
from ..train import compute_metrics

logger = get_logger(__name__)


class HPOBackend(Enum):
    """Supported HPO backends."""
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"
    AUTO = "auto"


class OptimizationDirection(Enum):
    """Optimization direction for metrics."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    # Search space configuration
    learning_rate_range: Tuple[float, float] = (1e-6, 1e-3)
    batch_size_options: List[int] = None
    lora_rank_options: List[int] = None
    lora_alpha_range: Tuple[float, float] = (8.0, 64.0)
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    warmup_ratio_range: Tuple[float, float] = (0.0, 0.2)
    weight_decay_range: Tuple[float, float] = (0.0, 0.1)
    
    # HPO configuration
    n_trials: int = 50
    n_startup_trials: int = 10
    multivariate: bool = True
    group: bool = True
    seed: int = 42
    
    # Early stopping configuration
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01
    
    # Resource configuration
    max_concurrent_trials: int = 4
    resources_per_trial: Dict[str, float] = None
    
    # Metric configuration
    metric: str = "eval_loss"
    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    
    # Backend configuration
    backend: HPOBackend = HPOBackend.AUTO
    storage_url: Optional[str] = None
    
    # Proxy training configuration
    max_steps: int = 1000
    eval_steps: int = 100
    proxy_dataset_ratio: float = 0.1
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [1, 2, 4, 8, 16, 32, 64]
        if self.lora_rank_options is None:
            self.lora_rank_options = [4, 8, 16, 32, 64]
        if self.resources_per_trial is None:
            self.resources_per_trial = {"cpu": 1, "gpu": 0.5}


@dataclass
class HPOTrial:
    """Represents a single HPO trial."""
    trial_id: str
    params: Dict[str, Any]
    metrics: Dict[str, List[float]]
    best_metric: Optional[float] = None
    status: str = "running"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to dictionary."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


class HPOEngine:
    """
    Intelligent Hyperparameter Optimization Engine for crucible.
    
    Implements multi-fidelity Bayesian optimization with early stopping to find
    optimal hyperparameter configurations 5-10x faster than manual tuning.
    """
    
    def __init__(self, hpo_config: HPOConfig, training_args: TrainingArguments):
        """
        Initialize HPO engine.
        
        Args:
            hpo_config: HPO configuration
            training_args: Base training arguments
        """
        if not OPTUNA_AVAILABLE and not RAY_AVAILABLE:
            raise ImportError(
                "HPO requires either optuna or ray[tune] to be installed. "
                "Install with: pip install 'crucible[hpo]' or pip install optuna ray[tune]"
            )
        
        self.config = hpo_config
        self.training_args = training_args
        self.trials: Dict[str, HPOTrial] = {}
        self.best_trial: Optional[HPOTrial] = None
        self.study = None
        self.backend = self._select_backend()
        
        # Setup storage
        self.storage_path = self._setup_storage()
        
        # Initialize callbacks
        self.callbacks = []
        
        logger.info(f"Initialized HPO Engine with backend: {self.backend.value}")
        logger.info(f"Search space: LR={self.config.learning_rate_range}, "
                   f"Batch sizes={self.config.batch_size_options}, "
                   f"LoRA ranks={self.config.lora_rank_options}")
    
    def _select_backend(self) -> HPOBackend:
        """Select appropriate HPO backend."""
        if self.config.backend != HPOBackend.AUTO:
            return self.config.backend
        
        if OPTUNA_AVAILABLE:
            return HPOBackend.OPTUNA
        elif RAY_AVAILABLE:
            return HPOBackend.RAY_TUNE
        else:
            raise RuntimeError("No HPO backend available")
    
    def _setup_storage(self) -> Optional[str]:
        """Setup storage for HPO study."""
        if self.config.storage_url:
            return self.config.storage_url
        
        # Create temporary directory for storage
        temp_dir = tempfile.mkdtemp(prefix="crucible_hpo_")
        storage_path = os.path.join(temp_dir, "hpo_study.db")
        
        if self.backend == HPOBackend.OPTUNA:
            return f"sqlite:///{storage_path}"
        else:
            return temp_dir
    
    def _create_search_space(self) -> Dict[str, Any]:
        """Create hyperparameter search space."""
        if self.backend == HPOBackend.OPTUNA:
            return self._create_optuna_search_space()
        else:
            return self._create_ray_search_space()
    
    def _create_optuna_search_space(self) -> Dict[str, Any]:
        """Create search space for Optuna."""
        return {
            "learning_rate": optuna.distributions.LogUniformDistribution(
                *self.config.learning_rate_range
            ),
            "per_device_train_batch_size": optuna.distributions.CategoricalDistribution(
                self.config.batch_size_options
            ),
            "lora_rank": optuna.distributions.CategoricalDistribution(
                self.config.lora_rank_options
            ),
            "lora_alpha": optuna.distributions.UniformDistribution(
                *self.config.lora_alpha_range
            ),
            "dropout": optuna.distributions.UniformDistribution(
                *self.config.dropout_range
            ),
            "warmup_ratio": optuna.distributions.UniformDistribution(
                *self.config.warmup_ratio_range
            ),
            "weight_decay": optuna.distributions.UniformDistribution(
                *self.config.weight_decay_range
            ),
        }
    
    def _create_ray_search_space(self) -> Dict[str, Any]:
        """Create search space for Ray Tune."""
        return {
            "learning_rate": tune.loguniform(*self.config.learning_rate_range),
            "per_device_train_batch_size": tune.choice(self.config.batch_size_options),
            "lora_rank": tune.choice(self.config.lora_rank_options),
            "lora_alpha": tune.uniform(*self.config.lora_alpha_range),
            "dropout": tune.uniform(*self.config.dropout_range),
            "warmup_ratio": tune.uniform(*self.config.warmup_ratio_range),
            "weight_decay": tune.uniform(*self.config.weight_decay_range),
        }
    
    def _create_objective_function(self) -> Callable:
        """Create objective function for optimization."""
        if self.backend == HPOBackend.OPTUNA:
            return self._optuna_objective
        else:
            return self._ray_objective
    
    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to optimize
        """
        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.config.learning_rate_range, log=True
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", self.config.batch_size_options
            ),
            "lora_rank": trial.suggest_categorical(
                "lora_rank", self.config.lora_rank_options
            ),
            "lora_alpha": trial.suggest_float(
                "lora_alpha", *self.config.lora_alpha_range
            ),
            "dropout": trial.suggest_float(
                "dropout", *self.config.dropout_range
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *self.config.warmup_ratio_range
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *self.config.weight_decay_range
            ),
        }
        
        # Create trial
        trial_id = f"optuna_{trial.number}"
        hpo_trial = HPOTrial(
            trial_id=trial_id,
            params=params,
            metrics={},
            start_time=datetime.now()
        )
        self.trials[trial_id] = hpo_trial
        
        try:
            # Run training with sampled parameters
            metric_value = self._run_training_trial(params, trial_id, trial)
            
            # Update trial
            hpo_trial.best_metric = metric_value
            hpo_trial.status = "completed"
            hpo_trial.end_time = datetime.now()
            
            # Update best trial
            if (self.best_trial is None or 
                (self.config.direction == OptimizationDirection.MINIMIZE and 
                 metric_value < self.best_trial.best_metric) or
                (self.config.direction == OptimizationDirection.MAXIMIZE and 
                 metric_value > self.best_trial.best_metric)):
                self.best_trial = hpo_trial
            
            return metric_value
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            hpo_trial.status = "failed"
            hpo_trial.end_time = datetime.now()
            raise optuna.TrialPruned()
    
    def _ray_objective(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Objective function for Ray Tune.
        
        Args:
            params: Sampled hyperparameters
            
        Returns:
            Dictionary with metrics
        """
        trial_id = f"ray_{int(time.time() * 1000)}"
        hpo_trial = HPOTrial(
            trial_id=trial_id,
            params=params,
            metrics={},
            start_time=datetime.now()
        )
        self.trials[trial_id] = hpo_trial
        
        try:
            # Run training
            metric_value = self._run_training_trial(params, trial_id)
            
            # Update trial
            hpo_trial.best_metric = metric_value
            hpo_trial.status = "completed"
            hpo_trial.end_time = datetime.now()
            
            return {self.config.metric: metric_value}
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            hpo_trial.status = "failed"
            hpo_trial.end_time = datetime.now()
            raise
    
    def _run_training_trial(
        self, 
        params: Dict[str, Any], 
        trial_id: str,
        optuna_trial: Optional[optuna.Trial] = None
    ) -> float:
        """
        Run a single training trial with given hyperparameters.
        
        Args:
            params: Hyperparameters for the trial
            trial_id: Unique trial identifier
            optuna_trial: Optional Optuna trial for pruning
            
        Returns:
            Final metric value
        """
        logger.info(f"Starting trial {trial_id} with params: {params}")
        
        # Create trial-specific output directory
        trial_output_dir = os.path.join(
            self.training_args.output_dir,
            "hpo_trials",
            trial_id
        )
        os.makedirs(trial_output_dir, exist_ok=True)
        
        # Create modified training arguments
        trial_args = TrainingArguments(
            output_dir=trial_output_dir,
            overwrite_output_dir=True,
            **{**vars(self.training_args), **params}
        )
        
        # Apply proxy training configuration
        trial_args.max_steps = min(self.config.max_steps, trial_args.max_steps)
        trial_args.eval_steps = self.config.eval_steps
        trial_args.save_steps = trial_args.eval_steps * 2
        trial_args.logging_steps = trial_args.eval_steps // 2
        trial_args.load_best_model_at_end = True
        trial_args.metric_for_best_model = self.config.metric
        trial_args.greater_is_better = (self.config.direction == OptimizationDirection.MAXIMIZE)
        
        # Load dataset with proxy ratio
        tokenizer = load_tokenizer(trial_args)
        train_dataset, eval_dataset = get_dataset(
            trial_args, tokenizer, proxy_ratio=self.config.proxy_dataset_ratio
        )
        
        # Load model
        model = load_model(trial_args, tokenizer)
        
        # Setup callbacks for pruning
        callbacks = []
        if optuna_trial and self.backend == HPOBackend.OPTUNA:
            callbacks.append(OptunaPruningCallback(optuna_trial, self.config.metric))
        
        # Add progress callback
        callbacks.append(ProgressCallback())
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=trial_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # Train
        train_result = trainer.train()
        
        # Get best metric
        if hasattr(trainer.state, 'best_metric'):
            best_metric = trainer.state.best_metric
        else:
            # Evaluate to get final metric
            eval_results = trainer.evaluate()
            best_metric = eval_results.get(self.config.metric, float('inf'))
        
        # Save trial metadata
        trial_metadata = {
            "trial_id": trial_id,
            "params": params,
            "best_metric": best_metric,
            "train_results": train_result.metrics,
            "output_dir": trial_output_dir
        }
        
        metadata_path = os.path.join(trial_output_dir, "trial_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(trial_metadata, f, indent=2)
        
        # Update trial with checkpoint path
        if trial_id in self.trials:
            self.trials[trial_id].checkpoint_path = trial_output_dir
        
        logger.info(f"Trial {trial_id} completed with {self.config.metric}: {best_metric}")
        
        return best_metric
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting HPO with {self.config.n_trials} trials")
        
        if self.backend == HPOBackend.OPTUNA:
            return self._optimize_with_optuna()
        else:
            return self._optimize_with_ray()
    
    def _optimize_with_optuna(self) -> Dict[str, Any]:
        """Run optimization with Optuna backend."""
        # Create study
        storage = RDBStorage(self.config.storage_url) if self.config.storage_url else None
        
        sampler = TPESampler(
            seed=self.config.seed,
            multivariate=self.config.multivariate,
            group=self.config.group,
            n_startup_trials=self.config.n_startup_trials
        )
        
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.config.max_steps,
            reduction_factor=3
        )
        
        self.study = optuna.create_study(
            study_name=f"crucible_hpo_{int(time.time())}",
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.direction.value,
            load_if_exists=True
        )
        
        # Optimize
        objective = self._create_objective_function()
        
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.max_concurrent_trials,
            show_progress_bar=True
        )
        
        # Get results
        results = self._collect_results()
        
        logger.info(f"Optimization completed. Best trial: {self.best_trial.trial_id}")
        logger.info(f"Best {self.config.metric}: {self.best_trial.best_metric}")
        logger.info(f"Best params: {self.best_trial.params}")
        
        return results
    
    def _optimize_with_ray(self) -> Dict[str, Any]:
        """Run optimization with Ray Tune backend."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is not available")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create search algorithm
        search_alg = OptunaSearch(
            metric=self.config.metric,
            mode=self.config.direction.value
        )
        
        # Create scheduler
        scheduler = ASHAScheduler(
            max_t=self.config.max_steps,
            grace_period=min(100, self.config.max_steps // 10),
            reduction_factor=3
        )
        
        # Define trainable function
        def trainable(config):
            trial_id = f"ray_{ray.get_runtime_context().get_trial_id()}"
            metric_value = self._run_training_trial(config, trial_id)
            tune.report(**{self.config.metric: metric_value})
        
        # Run optimization
        analysis = tune.run(
            trainable,
            config=self._create_ray_search_space(),
            num_samples=self.config.n_trials,
            search_alg=search_alg,
            scheduler=scheduler,
            resources_per_trial=self.config.resources_per_trial,
            local_dir=self.storage_path,
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(
            metric=self.config.metric,
            mode=self.config.direction.value
        )
        
        # Convert to our format
        results = {
            "best_trial_id": best_trial.trial_id,
            "best_params": best_trial.config,
            "best_metric": best_trial.last_result[self.config.metric],
            "all_trials": []
        }
        
        for trial in analysis.trials:
            results["all_trials"].append({
                "trial_id": trial.trial_id,
                "params": trial.config,
                "metric": trial.last_result.get(self.config.metric),
                "status": trial.status
            })
        
        return results
    
    def _collect_results(self) -> Dict[str, Any]:
        """Collect optimization results."""
        results = {
            "best_trial_id": self.best_trial.trial_id if self.best_trial else None,
            "best_params": self.best_trial.params if self.best_trial else None,
            "best_metric": self.best_trial.best_metric if self.best_trial else None,
            "n_trials": len(self.trials),
            "completed_trials": sum(1 for t in self.trials.values() if t.status == "completed"),
            "failed_trials": sum(1 for t in self.trials.values() if t.status == "failed"),
            "all_trials": [trial.to_dict() for trial in self.trials.values()]
        }
        
        # Save results
        results_path = os.path.join(
            self.training_args.output_dir,
            "hpo_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best checkpoint from optimization.
        
        Returns:
            Path to best checkpoint or None
        """
        if self.best_trial and self.best_trial.checkpoint_path:
            # Find best checkpoint in trial directory
            checkpoint_dir = Path(self.best_trial.checkpoint_path)
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
                return str(checkpoints[-1])
        
        return None
    
    def create_report(self) -> str:
        """
        Create optimization report.
        
        Returns:
            Formatted report string
        """
        if not self.best_trial:
            return "No optimization results available."
        
        report = [
            "=" * 60,
            "HYPERPARAMETER OPTIMIZATION REPORT",
            "=" * 60,
            f"Backend: {self.backend.value}",
            f"Total trials: {len(self.trials)}",
            f"Completed trials: {sum(1 for t in self.trials.values() if t.status == 'completed')}",
            f"Best trial: {self.best_trial.trial_id}",
            f"Best {self.config.metric}: {self.best_trial.best_metric:.6f}",
            "",
            "Best hyperparameters:",
        ]
        
        for param, value in self.best_trial.params.items():
            if isinstance(value, float):
                report.append(f"  {param}: {value:.6f}")
            else:
                report.append(f"  {param}: {value}")
        
        report.extend([
            "",
            "Optimization history:",
        ])
        
        # Sort trials by metric
        sorted_trials = sorted(
            [t for t in self.trials.values() if t.status == "completed"],
            key=lambda t: t.best_metric,
            reverse=(self.config.direction == OptimizationDirection.MAXIMIZE)
        )
        
        for i, trial in enumerate(sorted_trials[:10], 1):
            report.append(f"  {i}. Trial {trial.trial_id}: {self.config.metric}={trial.best_metric:.6f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class OptunaPruningCallback:
    """Callback for Optuna trial pruning."""
    
    def __init__(self, trial: optuna.Trial, metric: str):
        self.trial = trial
        self.metric = metric
        self.step = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics and self.metric in metrics:
            self.step += 1
            self.trial.report(metrics[self.metric], self.step)
            
            if self.trial.should_prune():
                raise optuna.TrialPruned()


class HPOCLIHandler:
    """Handler for HPO CLI integration."""
    
    @staticmethod
    def add_hpo_args(parser):
        """Add HPO arguments to argument parser."""
        hpo_group = parser.add_argument_group("Hyperparameter Optimization")
        
        hpo_group.add_argument(
            "--hpo",
            action="store_true",
            help="Enable hyperparameter optimization"
        )
        
        hpo_group.add_argument(
            "--hpo_backend",
            type=str,
            default="auto",
            choices=["optuna", "ray_tune", "auto"],
            help="HPO backend to use"
        )
        
        hpo_group.add_argument(
            "--hpo_n_trials",
            type=int,
            default=50,
            help="Number of HPO trials"
        )
        
        hpo_group.add_argument(
            "--hpo_metric",
            type=str,
            default="eval_loss",
            help="Metric to optimize"
        )
        
        hpo_group.add_argument(
            "--hpo_direction",
            type=str,
            default="minimize",
            choices=["minimize", "maximize"],
            help="Optimization direction"
        )
        
        hpo_group.add_argument(
            "--hpo_max_steps",
            type=int,
            default=1000,
            help="Maximum steps per trial"
        )
        
        hpo_group.add_argument(
            "--hpo_proxy_ratio",
            type=float,
            default=0.1,
            help="Ratio of dataset to use for proxy training"
        )
        
        hpo_group.add_argument(
            "--hpo_storage",
            type=str,
            default=None,
            help="Storage URL for HPO study"
        )
        
        return parser
    
    @staticmethod
    def run_hpo(args, training_args):
        """Run HPO from CLI arguments."""
        # Create HPO config
        hpo_config = HPOConfig(
            backend=HPOBackend(args.hpo_backend),
            n_trials=args.hpo_n_trials,
            metric=args.hpo_metric,
            direction=OptimizationDirection(args.hpo_direction),
            max_steps=args.hpo_max_steps,
            proxy_dataset_ratio=args.hpo_proxy_ratio,
            storage_url=args.hpo_storage
        )
        
        # Create and run HPO engine
        engine = HPOEngine(hpo_config, training_args)
        results = engine.optimize()
        
        # Print report
        print(engine.create_report())
        
        # Get best checkpoint
        best_checkpoint = engine.get_best_checkpoint()
        if best_checkpoint:
            print(f"\nBest checkpoint saved to: {best_checkpoint}")
            
            # Update training args with best params
            if results["best_params"]:
                for param, value in results["best_params"].items():
                    if hasattr(training_args, param):
                        setattr(training_args, param, value)
        
        return results, training_args


# Integration with existing training script
def run_training_with_hpo(args, training_args):
    """
    Run training with optional HPO.
    
    Args:
        args: Parsed command line arguments
        training_args: Training arguments
        
    Returns:
        Updated training args and HPO results if applicable
    """
    if hasattr(args, 'hpo') and args.hpo:
        logger.info("Starting hyperparameter optimization...")
        hpo_results, training_args = HPOCLIHandler.run_hpo(args, training_args)
        return training_args, hpo_results
    else:
        return training_args, None


# Utility functions for integration
def suggest_hyperparameters(trial, config: HPOConfig) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.
    
    Args:
        trial: Optuna trial or similar object
        config: HPO configuration
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", *config.learning_rate_range, log=True
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", config.batch_size_options
        ),
    }
    
    if config.lora_rank_options:
        params["lora_rank"] = trial.suggest_categorical(
            "lora_rank", config.lora_rank_options
        )
        params["lora_alpha"] = trial.suggest_float(
            "lora_alpha", *config.lora_alpha_range
        )
    
    params["dropout"] = trial.suggest_float(
        "dropout", *config.dropout_range
    )
    params["warmup_ratio"] = trial.suggest_float(
        "warmup_ratio", *config.warmup_ratio_range
    )
    params["weight_decay"] = trial.suggest_float(
        "weight_decay", *config.weight_decay_range
    )
    
    return params


def create_hpo_report(results: Dict[str, Any]) -> str:
    """
    Create formatted HPO report.
    
    Args:
        results: HPO results dictionary
        
    Returns:
        Formatted report string
    """
    if not results or "best_params" not in results:
        return "No HPO results available."
    
    report = [
        "=" * 60,
        "HYPERPARAMETER OPTIMIZATION RESULTS",
        "=" * 60,
        f"Best metric: {results.get('best_metric', 'N/A')}",
        f"Best trial: {results.get('best_trial_id', 'N/A')}",
        "",
        "Best parameters:",
    ]
    
    for param, value in results.get("best_params", {}).items():
        if isinstance(value, float):
            report.append(f"  {param}: {value:.6f}")
        else:
            report.append(f"  {param}: {value}")
    
    report.append("=" * 60)
    
    return "\n".join(report)


# Export main classes and functions
__all__ = [
    "HPOEngine",
    "HPOConfig",
    "HPOBackend",
    "OptimizationDirection",
    "HPOTrial",
    "HPOCLIHandler",
    "run_training_with_hpo",
    "suggest_hyperparameters",
    "create_hpo_report",
]