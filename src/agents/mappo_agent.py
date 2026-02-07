"""
MAPPO Agent: Multi-Agent PPO with Centralized Training, Decentralized Execution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentConfig


@dataclass
class MAPPOConfig(AgentConfig):
    """Configuration for MAPPO agent."""
    hidden_dims: List[int] = None
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class ActorNetwork(nn.Module):
    """
    Actor network for decentralized policy.
    
    Outputs mean and log_std for Gaussian policy.
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        """
        Initialize actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action_mean, action_log_std)
        """
        features = self.shared(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def get_action(self, 
                   obs: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to return deterministic action
            
        Returns:
            Tuple of (action, log_prob, entropy)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action[:, 0])
            entropy = torch.zeros_like(action[:, 0])
        else:
            # Sample from Gaussian
            dist = Normal(mean, std)
            x = dist.rsample()  # Reparameterization trick
            action = torch.tanh(x)
            
            # Log probability with Tanh squashing correction
            log_prob = dist.log_prob(x)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            
            # Entropy
            entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_actions(self, 
                         obs: torch.Tensor, 
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Inverse tanh to get pre-squash action
        x = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized critic network.
    
    Takes global state as input for value estimation.
    """
    
    def __init__(self,
                 global_state_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        """
        Initialize critic network.
        
        Args:
            global_state_dim: Global state dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = global_state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Last layer with smaller initialization
        last_layer = self.network[-1]
        nn.init.orthogonal_(last_layer.weight, gain=1.0)
        nn.init.zeros_(last_layer.bias)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            global_state: Global state tensor
            
        Returns:
            Value estimate
        """
        return self.network(global_state).squeeze(-1)


class MAPPOAgent(BaseAgent):
    """
    MAPPO Agent with Centralized Training, Decentralized Execution.
    
    Each agent has its own actor network for decentralized execution,
    while sharing a centralized critic during training.
    """
    
    def __init__(self,
                 config: MAPPOConfig,
                 global_state_dim: int = 12,
                 device: str = 'cpu'):
        """
        Initialize MAPPO agent.
        
        Args:
            config: MAPPO configuration
            global_state_dim: Dimension of global state for critic
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(config)
        
        self.mappo_config = config
        self.global_state_dim = global_state_dim
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(
            obs_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims
        ).to(self.device)
        
        # Critic is shared but we keep a reference
        self.critic: Optional[CriticNetwork] = None
        
        # Optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Training state
        self.training = True
        self.entropy_coef = config.entropy_coef
    
    def set_critic(self, critic: CriticNetwork) -> None:
        """Set the shared critic network."""
        self.critic = critic
    
    def select_action(self,
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action array
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor.get_action(obs_tensor, deterministic)
            
            action_np = action.cpu().numpy().squeeze(0)
            
            # Apply action masking
            action_np = self.apply_action_mask(action_np, observation)
            
            return action_np
    
    def get_action_with_log_prob(self,
                                  observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Get action and its log probability.
        
        Args:
            observation: Observation array
            
        Returns:
            Tuple of (action, log_prob)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor.get_action(obs_tensor, deterministic=False)
            
            return action.cpu().numpy().squeeze(0), log_prob.cpu().item()
    
    def store_transition(self,
                         observation: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_observation: np.ndarray,
                         done: bool,
                         log_prob: float,
                         global_state: np.ndarray) -> None:
        """Store a transition in the buffer."""
        self.buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            log_prob=log_prob,
            global_state=global_state
        )
    
    def compute_returns_and_advantages(self,
                                       gamma: float = 0.99,
                                       gae_lambda: float = 0.95) -> None:
        """Compute returns and GAE advantages."""
        if self.critic is None:
            raise ValueError("Critic not set. Call set_critic() first.")
        
        with torch.no_grad():
            global_states = torch.FloatTensor(
                np.array(self.buffer.global_states)
            ).to(self.device)
            
            values = self.critic(global_states).cpu().numpy()
            
            # Compute GAE
            rewards = np.array(self.buffer.rewards)
            dones = np.array(self.buffer.dones)
            
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
            
            self.buffer.advantages = advantages
            self.buffer.returns = returns
    
    def update(self,
               clip_epsilon: float = 0.2,
               entropy_coef: float = 0.01,
               max_grad_norm: float = 0.5,
               num_epochs: int = 4,
               batch_size: int = 64) -> Dict[str, float]:
        """
        Update actor using PPO.
        
        Args:
            clip_epsilon: PPO clip parameter
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            num_epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of loss metrics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Convert buffer to tensors
        observations = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for _ in range(num_epochs):
            # Create random batches
            indices = torch.randperm(len(self.buffer))
            
            for start in range(0, len(self.buffer), batch_size):
                end = min(start + batch_size, len(self.buffer))
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + entropy_coef * entropy_loss
                
                # Update
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.actor_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1)
        }
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'config': self.mappo_config
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False
        self.actor.eval()
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.training = True
        self.actor.train()


class RolloutBuffer:
    """Buffer for storing rollout experiences."""
    
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.global_states: List[np.ndarray] = []
        
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
    
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_observation: np.ndarray,
            done: bool,
            log_prob: float,
            global_state: np.ndarray) -> None:
        """Add a transition to the buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.global_states.append(global_state)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.log_probs = []
        self.global_states = []
        self.advantages = None
        self.returns = None
    
    def __len__(self) -> int:
        return len(self.observations)
