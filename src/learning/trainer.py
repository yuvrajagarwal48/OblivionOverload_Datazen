"""
MAPPO Trainer: Training loop for Multi-Agent PPO.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from datetime import datetime

from ..environment import FinancialEnvironment, EnvConfig
from ..agents import MAPPOAgent, MAPPOConfig
from ..agents.mappo_agent import CriticNetwork
from .ppo import PPOUpdater, compute_gae, normalize_advantages


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    num_banks: int = 30
    episode_length: int = 100
    
    # Training
    num_episodes: int = 1000
    parallel_envs: int = 1
    checkpoint_interval: int = 50
    
    # MAPPO
    actor_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_decay_steps: int = 50000
    max_grad_norm: float = 0.5
    batch_size: int = 64
    update_epochs: int = 4
    
    # Curriculum
    curriculum_enabled: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stabilization
    agent_freeze_fraction: float = 0.2
    freeze_interval: int = 10
    warmup_episodes: int = 50
    
    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"
    
    # Device
    device: str = "cpu"
    
    # Seed
    seed: int = 42


class CurriculumScheduler:
    """Manages curriculum learning stages."""
    
    def __init__(self, stages: List[Dict[str, Any]]):
        """
        Initialize curriculum scheduler.
        
        Args:
            stages: List of stage configurations
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
    
    @property
    def current_stage(self) -> Dict[str, Any]:
        if not self.stages:
            return {
                'name': 'default',
                'shock_probability': 0.1,
                'shock_magnitude': 0.1,
                'duration_episodes': float('inf')
            }
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]
    
    def step(self) -> bool:
        """
        Advance by one episode.
        
        Returns:
            True if stage changed
        """
        self.episodes_in_stage += 1
        
        if self.stages and self.current_stage_idx < len(self.stages):
            if self.episodes_in_stage >= self.current_stage.get('duration_episodes', float('inf')):
                self.current_stage_idx += 1
                self.episodes_in_stage = 0
                return True
        
        return False
    
    def get_shock_params(self) -> Tuple[float, float]:
        """Get current shock parameters."""
        stage = self.current_stage
        return stage.get('shock_probability', 0.1), stage.get('shock_magnitude', 0.1)


class TrainingLogger:
    """Logs training metrics."""
    
    def __init__(self, log_dir: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.default_rates: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
    
    def log_episode(self,
                    episode: int,
                    rewards: Dict[int, float],
                    length: int,
                    default_rate: float,
                    policy_loss: float = 0.0,
                    value_loss: float = 0.0,
                    entropy: float = 0.0) -> None:
        """Log episode metrics."""
        mean_reward = np.mean(list(rewards.values()))
        
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(length)
        self.default_rates.append(default_rate)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get statistics over recent episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_defaults = self.default_rates[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_default_rate': np.mean(recent_defaults),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards)
        }
    
    def save(self, filename: str = "training_log.json") -> None:
        """Save logs to file."""
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'default_rates': self.default_rates,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies
        }
        
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(log_data, f, indent=2)


class MAPPOTrainer:
    """
    Trainer for Multi-Agent PPO with Centralized Training, Decentralized Execution.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Initialize environment
        env_config = EnvConfig(
            num_banks=config.num_banks,
            episode_length=config.episode_length,
            seed=config.seed
        )
        self.env = FinancialEnvironment(env_config)
        
        # Initialize agents
        self.agents: Dict[int, MAPPOAgent] = {}
        self._init_agents()
        
        # Initialize centralized critic
        global_state_dim = 12  # From network.get_global_state()
        self.critic = CriticNetwork(
            global_state_dim=global_state_dim,
            hidden_dims=config.critic_hidden_dims
        ).to(self.device)
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        # Set critic for all agents
        for agent in self.agents.values():
            agent.set_critic(self.critic)
        
        # PPO updater
        self.ppo_updater = PPOUpdater(
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef_start,
            max_grad_norm=config.max_grad_norm,
            device=config.device
        )
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(config.curriculum_stages)
        
        # Logger
        self.logger = TrainingLogger(config.log_dir)
        
        # Training state
        self.total_steps = 0
        self.current_episode = 0
        self.frozen_agents: List[int] = []
    
    def _init_agents(self) -> None:
        """Initialize all agents."""
        for agent_id in range(self.config.num_banks):
            agent_config = MAPPOConfig(
                agent_id=agent_id,
                observation_dim=FinancialEnvironment.OBS_DIM,
                action_dim=FinancialEnvironment.ACTION_DIM,
                hidden_dims=self.config.actor_hidden_dims,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_epsilon=self.config.clip_epsilon,
                entropy_coef=self.config.entropy_coef_start,
                seed=self.config.seed + agent_id
            )
            
            agent = MAPPOAgent(
                config=agent_config,
                global_state_dim=12,
                device=self.config.device
            )
            
            self.agents[agent_id] = agent
    
    def _get_entropy_coef(self) -> float:
        """Get current entropy coefficient with decay."""
        progress = min(1.0, self.total_steps / self.config.entropy_decay_steps)
        return (self.config.entropy_coef_start + 
                progress * (self.config.entropy_coef_end - self.config.entropy_coef_start))
    
    def _update_frozen_agents(self) -> None:
        """Update which agents are frozen for stability."""
        if self.current_episode % self.config.freeze_interval != 0:
            return
        
        num_frozen = int(self.config.num_banks * self.config.agent_freeze_fraction)
        self.frozen_agents = list(np.random.choice(
            self.config.num_banks, 
            size=num_frozen, 
            replace=False
        ))
    
    def collect_rollout(self) -> Tuple[Dict[int, float], int, float]:
        """
        Collect one episode of experience.
        
        Returns:
            Tuple of (episode_rewards, episode_length, default_rate)
        """
        observations, global_state = self.env.reset()
        
        episode_rewards = {i: 0.0 for i in range(self.config.num_banks)}
        
        for step in range(self.config.episode_length):
            # Apply curriculum shock
            shock_prob, shock_mag = self.curriculum.get_shock_params()
            if np.random.random() < shock_prob:
                self.env.apply_scenario_shock(
                    price_shock=-shock_mag * np.random.random(),
                    volatility_shock=shock_mag * 0.5 * np.random.random()
                )
            
            # Collect actions
            actions = {}
            log_probs = {}
            
            for agent_id, agent in self.agents.items():
                obs = observations[agent_id]
                action, log_prob = agent.get_action_with_log_prob(obs)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
            
            # Step environment
            result = self.env.step(actions)
            
            # Store transitions
            for agent_id, agent in self.agents.items():
                if agent_id not in self.frozen_agents:
                    agent.store_transition(
                        observation=observations[agent_id],
                        action=actions[agent_id],
                        reward=result.rewards[agent_id],
                        next_observation=result.observations[agent_id],
                        done=result.dones[agent_id],
                        log_prob=log_probs[agent_id],
                        global_state=global_state
                    )
                
                episode_rewards[agent_id] += result.rewards[agent_id]
            
            # Update state
            observations = result.observations
            global_state = result.global_state
            self.total_steps += 1
            
            # Check if done
            if all(result.dones.values()):
                break
        
        # Calculate default rate
        stats = self.env.network.get_network_stats()
        default_rate = stats.num_defaulted / self.config.num_banks
        
        return episode_rewards, step + 1, default_rate
    
    def update_agents(self) -> Dict[str, float]:
        """
        Update all agents and critic.
        
        Returns:
            Dictionary of loss metrics
        """
        entropy_coef = self._get_entropy_coef()
        
        # Collect global states and returns for critic update
        all_global_states = []
        all_returns = []
        
        # First pass: compute advantages for each agent
        for agent_id, agent in self.agents.items():
            if agent_id in self.frozen_agents:
                continue
            
            if len(agent.buffer) == 0:
                continue
            
            agent.compute_returns_and_advantages(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )
            
            all_global_states.extend(agent.buffer.global_states)
            all_returns.extend(agent.buffer.returns.tolist())
        
        # Update critic
        value_loss_total = 0.0
        if all_global_states:
            global_states_tensor = torch.FloatTensor(np.array(all_global_states)).to(self.device)
            returns_tensor = torch.FloatTensor(all_returns).to(self.device)
            
            critic_metrics = self.ppo_updater.update_critic(
                critic=self.critic,
                optimizer=self.critic_optimizer,
                global_states=global_states_tensor,
                returns=returns_tensor,
                num_epochs=self.config.update_epochs,
                batch_size=self.config.batch_size
            )
            value_loss_total = critic_metrics.get('value_loss', 0.0)
        
        # Update each agent's actor
        policy_loss_total = 0.0
        entropy_total = 0.0
        num_updated = 0
        
        for agent_id, agent in self.agents.items():
            if agent_id in self.frozen_agents:
                agent.buffer.clear()
                continue
            
            if len(agent.buffer) == 0:
                continue
            
            metrics = agent.update(
                clip_epsilon=self.config.clip_epsilon,
                entropy_coef=entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
                num_epochs=self.config.update_epochs,
                batch_size=self.config.batch_size
            )
            
            policy_loss_total += metrics.get('policy_loss', 0.0)
            entropy_total += metrics.get('entropy', 0.0)
            num_updated += 1
        
        if num_updated > 0:
            policy_loss_total /= num_updated
            entropy_total /= num_updated
        
        return {
            'policy_loss': policy_loss_total,
            'value_loss': value_loss_total,
            'entropy': entropy_total,
            'entropy_coef': entropy_coef
        }
    
    def train(self, progress_bar: bool = True) -> None:
        """
        Run the full training loop.
        
        Args:
            progress_bar: Whether to show progress bar
        """
        print(f"Starting training for {self.config.num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Number of agents: {self.config.num_banks}")
        
        iterator = range(self.config.num_episodes)
        if progress_bar:
            iterator = tqdm(iterator, desc="Training")
        
        for episode in iterator:
            self.current_episode = episode
            
            # Update frozen agents
            self._update_frozen_agents()
            
            # Collect rollout
            rewards, length, default_rate = self.collect_rollout()
            
            # Update agents
            metrics = self.update_agents()
            
            # Advance curriculum
            self.curriculum.step()
            
            # Log
            self.logger.log_episode(
                episode=episode,
                rewards=rewards,
                length=length,
                default_rate=default_rate,
                policy_loss=metrics.get('policy_loss', 0.0),
                value_loss=metrics.get('value_loss', 0.0),
                entropy=metrics.get('entropy', 0.0)
            )
            
            # Update progress bar
            if progress_bar and episode % 10 == 0:
                stats = self.logger.get_recent_stats(100)
                iterator.set_postfix({
                    'reward': f"{stats.get('mean_reward', 0):.2f}",
                    'defaults': f"{stats.get('mean_default_rate', 0):.2%}",
                    'stage': self.curriculum.current_stage.get('name', 'default')
                })
            
            # Checkpoint
            if episode > 0 and episode % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{episode}")
        
        # Final save
        self.save_checkpoint("final")
        self.logger.save()
        
        print("Training completed!")
    
    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        save_dir = os.path.join(self.config.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save critic
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, os.path.join(save_dir, 'critic.pt'))
        
        # Save actors
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(save_dir, f'actor_{agent_id}.pt'))
        
        # Save config
        config_dict = {
            'training_config': self.config.__dict__,
            'total_steps': self.total_steps,
            'current_episode': self.current_episode,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, name: str) -> None:
        """Load training checkpoint."""
        load_dir = os.path.join(self.config.save_dir, name)
        
        # Load critic
        critic_path = os.path.join(load_dir, 'critic.pt')
        if os.path.exists(critic_path):
            checkpoint = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load actors
        for agent_id, agent in self.agents.items():
            actor_path = os.path.join(load_dir, f'actor_{agent_id}.pt')
            if os.path.exists(actor_path):
                agent.load(actor_path)
        
        # Load config
        config_path = os.path.join(load_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.total_steps = config_dict.get('total_steps', 0)
                self.current_episode = config_dict.get('current_episode', 0)
        
        print(f"Loaded checkpoint: {name}")
    
    def evaluate(self, 
                 num_episodes: int = 10,
                 deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policies
            
        Returns:
            Evaluation metrics
        """
        # Set agents to eval mode
        for agent in self.agents.values():
            agent.eval()
        
        all_rewards = []
        all_defaults = []
        all_lengths = []
        
        for _ in range(num_episodes):
            observations, global_state = self.env.reset()
            episode_rewards = {i: 0.0 for i in range(self.config.num_banks)}
            
            for step in range(self.config.episode_length):
                actions = {}
                for agent_id, agent in self.agents.items():
                    actions[agent_id] = agent.select_action(
                        observations[agent_id], 
                        deterministic=deterministic
                    )
                
                result = self.env.step(actions)
                
                for agent_id in range(self.config.num_banks):
                    episode_rewards[agent_id] += result.rewards[agent_id]
                
                observations = result.observations
                
                if all(result.dones.values()):
                    break
            
            stats = self.env.network.get_network_stats()
            all_rewards.append(np.mean(list(episode_rewards.values())))
            all_defaults.append(stats.num_defaulted / self.config.num_banks)
            all_lengths.append(step + 1)
        
        # Set back to train mode
        for agent in self.agents.values():
            agent.train()
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_default_rate': np.mean(all_defaults),
            'mean_length': np.mean(all_lengths)
        }
