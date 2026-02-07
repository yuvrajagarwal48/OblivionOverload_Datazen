"""
Model Lifecycle Management for FinSim-MAPPO.
Provides versioned checkpointing, metadata tracking, and model registry.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging


logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a saved model checkpoint."""
    # Identification
    model_id: str
    version: str
    checkpoint_path: str
    
    # Training info
    training_steps: int
    episodes_completed: int
    total_timesteps: int
    
    # Performance metrics
    mean_reward: float
    best_reward: float
    default_rate: float
    
    # Configuration
    config_hash: str
    env_config: Dict[str, Any]
    agent_config: Dict[str, Any]
    
    # Timestamps
    created_at: str
    training_duration_seconds: float
    
    # Validation
    validation_scenario: str = "normal"
    validation_reward: float = 0.0
    validation_seeds: List[int] = None
    
    # Lineage
    parent_checkpoint: Optional[str] = None
    curriculum_stage: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        return cls(**data)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    path: Path
    metadata: ModelMetadata
    file_size_mb: float
    is_valid: bool = True
    error_message: str = ""


class ModelRegistry:
    """
    Registry for tracking and managing model versions.
    """
    
    def __init__(self, registry_dir: str = "outputs/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.models: Dict[str, Dict[str, ModelMetadata]] = {}
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.models[model_id] = {}
                    for version, meta in versions.items():
                        self.models[model_id][version] = ModelMetadata.from_dict(meta)
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {}
        for model_id, versions in self.models.items():
            data[model_id] = {}
            for version, meta in versions.items():
                data[model_id][version] = meta.to_dict()
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model version."""
        if metadata.model_id not in self.models:
            self.models[metadata.model_id] = {}
        
        self.models[metadata.model_id][metadata.version] = metadata
        self._save_registry()
        
        logger.info(f"Registered model {metadata.model_id} version {metadata.version}")
    
    def get_model(self, model_id: str, version: str = "latest") -> Optional[ModelMetadata]:
        """Get model metadata by ID and version."""
        if model_id not in self.models:
            return None
        
        if version == "latest":
            versions = list(self.models[model_id].keys())
            if not versions:
                return None
            version = max(versions)
        
        return self.models[model_id].get(version)
    
    def get_best_model(self, model_id: str, metric: str = "mean_reward") -> Optional[ModelMetadata]:
        """Get the best performing version of a model."""
        if model_id not in self.models:
            return None
        
        versions = list(self.models[model_id].values())
        if not versions:
            return None
        
        return max(versions, key=lambda m: getattr(m, metric, 0))
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and their versions."""
        return {model_id: list(versions.keys()) 
                for model_id, versions in self.models.items()}
    
    def delete_version(self, model_id: str, version: str, 
                       delete_files: bool = False) -> bool:
        """Delete a model version from registry."""
        if model_id not in self.models or version not in self.models[model_id]:
            return False
        
        metadata = self.models[model_id][version]
        
        if delete_files:
            checkpoint_path = Path(metadata.checkpoint_path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        
        del self.models[model_id][version]
        
        if not self.models[model_id]:
            del self.models[model_id]
        
        self._save_registry()
        return True


class ModelCheckpointer:
    """
    Handles saving and loading model checkpoints with metadata.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "outputs/checkpoints",
                 registry: ModelRegistry = None,
                 keep_last_n: int = 5,
                 keep_best_n: int = 3):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = registry or ModelRegistry()
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        
        self.checkpoints: Dict[str, List[CheckpointInfo]] = {}
    
    def _generate_version(self) -> str:
        """Generate version string from timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _compute_config_hash(self, config: Dict) -> str:
        """Compute hash of configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_checkpoint(self,
                        model_id: str,
                        agents: Dict[int, Any],
                        optimizer_states: Dict[int, Any],
                        training_state: Dict[str, Any],
                        env_config: Dict[str, Any],
                        agent_config: Dict[str, Any],
                        metrics: Dict[str, float],
                        version: str = None) -> CheckpointInfo:
        """
        Save a model checkpoint with full metadata.
        """
        version = version or self._generate_version()
        
        # Create checkpoint directory
        model_dir = self.checkpoint_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = model_dir / f"checkpoint_v{version}.pt"
        metadata_path = model_dir / f"metadata_v{version}.json"
        
        # Prepare state dict
        state_dict = {
            'agents': {},
            'optimizers': optimizer_states,
            'training_state': training_state
        }
        
        for agent_id, agent in agents.items():
            if hasattr(agent, 'state_dict'):
                state_dict['agents'][agent_id] = agent.state_dict()
            elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                state_dict['agents'][agent_id] = {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict()
                }
        
        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            checkpoint_path=str(checkpoint_path),
            training_steps=training_state.get('global_step', 0),
            episodes_completed=training_state.get('episode', 0),
            total_timesteps=training_state.get('total_timesteps', 0),
            mean_reward=metrics.get('mean_reward', 0.0),
            best_reward=metrics.get('best_reward', 0.0),
            default_rate=metrics.get('default_rate', 0.0),
            config_hash=self._compute_config_hash({**env_config, **agent_config}),
            env_config=env_config,
            agent_config=agent_config,
            created_at=datetime.now().isoformat(),
            training_duration_seconds=training_state.get('training_duration', 0),
            curriculum_stage=training_state.get('curriculum_stage', 0),
            parent_checkpoint=training_state.get('parent_checkpoint')
        )
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Register
        self.registry.register_model(metadata)
        
        # Track checkpoint
        file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
        checkpoint_info = CheckpointInfo(
            path=checkpoint_path,
            metadata=metadata,
            file_size_mb=file_size
        )
        
        if model_id not in self.checkpoints:
            self.checkpoints[model_id] = []
        self.checkpoints[model_id].append(checkpoint_info)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(model_id)
        
        logger.info(f"Saved checkpoint: {checkpoint_path} ({file_size:.2f} MB)")
        
        return checkpoint_info
    
    def load_checkpoint(self, 
                        model_id: str,
                        version: str = "latest",
                        device: str = "cpu") -> Tuple[Dict, ModelMetadata]:
        """
        Load a model checkpoint.
        Returns (state_dict, metadata).
        """
        metadata = self.registry.get_model(model_id, version)
        
        if metadata is None:
            raise ValueError(f"Model {model_id} version {version} not found in registry")
        
        checkpoint_path = Path(metadata.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return state_dict, metadata
    
    def _cleanup_old_checkpoints(self, model_id: str) -> None:
        """Clean up old checkpoints, keeping best and recent."""
        if model_id not in self.checkpoints:
            return
        
        checkpoints = self.checkpoints[model_id]
        
        if len(checkpoints) <= max(self.keep_last_n, self.keep_best_n):
            return
        
        # Sort by performance (best first)
        by_performance = sorted(
            checkpoints, 
            key=lambda c: c.metadata.mean_reward, 
            reverse=True
        )
        best_versions = {c.metadata.version for c in by_performance[:self.keep_best_n]}
        
        # Sort by time (most recent first)
        by_time = sorted(
            checkpoints,
            key=lambda c: c.metadata.created_at,
            reverse=True
        )
        recent_versions = {c.metadata.version for c in by_time[:self.keep_last_n]}
        
        # Keep union of best and recent
        keep_versions = best_versions | recent_versions
        
        # Delete others
        for checkpoint in checkpoints:
            if checkpoint.metadata.version not in keep_versions:
                try:
                    if checkpoint.path.exists():
                        checkpoint.path.unlink()
                        logger.info(f"Deleted old checkpoint: {checkpoint.path}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint: {e}")
        
        self.checkpoints[model_id] = [
            c for c in checkpoints 
            if c.metadata.version in keep_versions
        ]
    
    def get_latest_checkpoint(self, model_id: str) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint for a model."""
        if model_id not in self.checkpoints:
            return None
        
        return max(self.checkpoints[model_id], 
                   key=lambda c: c.metadata.created_at,
                   default=None)
    
    def get_best_checkpoint(self, model_id: str) -> Optional[CheckpointInfo]:
        """Get the best performing checkpoint for a model."""
        if model_id not in self.checkpoints:
            return None
        
        return max(self.checkpoints[model_id],
                   key=lambda c: c.metadata.mean_reward,
                   default=None)
    
    def export_model(self, 
                     model_id: str, 
                     version: str,
                     export_dir: str) -> Path:
        """Export a model with all necessary files for deployment."""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        metadata = self.registry.get_model(model_id, version)
        if metadata is None:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        # Copy checkpoint
        checkpoint_path = Path(metadata.checkpoint_path)
        exported_checkpoint = export_path / f"{model_id}_v{version}.pt"
        shutil.copy(checkpoint_path, exported_checkpoint)
        
        # Save configs
        config_path = export_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'env_config': metadata.env_config,
                'agent_config': metadata.agent_config,
                'metadata': metadata.to_dict()
            }, f, indent=2, default=str)
        
        logger.info(f"Exported model to {export_path}")
        
        return export_path


class TrainingResumeManager:
    """
    Manages training resumption from checkpoints.
    """
    
    def __init__(self, checkpointer: ModelCheckpointer):
        self.checkpointer = checkpointer
    
    def can_resume(self, model_id: str) -> bool:
        """Check if training can be resumed for a model."""
        return self.checkpointer.get_latest_checkpoint(model_id) is not None
    
    def resume_training(self, 
                        model_id: str,
                        agents: Dict[int, Any],
                        trainer: Any,
                        device: str = "cpu") -> Dict[str, Any]:
        """
        Resume training from the latest checkpoint.
        Returns the restored training state.
        """
        checkpoint_info = self.checkpointer.get_latest_checkpoint(model_id)
        
        if checkpoint_info is None:
            raise ValueError(f"No checkpoint found for model {model_id}")
        
        state_dict, metadata = self.checkpointer.load_checkpoint(
            model_id, 
            checkpoint_info.metadata.version,
            device
        )
        
        # Restore agent states
        for agent_id, agent_state in state_dict['agents'].items():
            agent_id = int(agent_id)
            if agent_id in agents:
                agent = agents[agent_id]
                if hasattr(agent, 'load_state_dict'):
                    agent.load_state_dict(agent_state)
                elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                    agent.actor.load_state_dict(agent_state['actor'])
                    agent.critic.load_state_dict(agent_state['critic'])
        
        # Restore training state
        training_state = state_dict['training_state']
        training_state['parent_checkpoint'] = checkpoint_info.metadata.version
        
        # Restore optimizer states if trainer supports it
        if hasattr(trainer, 'load_optimizer_state') and 'optimizers' in state_dict:
            trainer.load_optimizer_state(state_dict['optimizers'])
        
        logger.info(f"Resumed training from checkpoint v{checkpoint_info.metadata.version}")
        logger.info(f"  Episodes: {training_state.get('episode', 0)}")
        logger.info(f"  Global step: {training_state.get('global_step', 0)}")
        
        return training_state
