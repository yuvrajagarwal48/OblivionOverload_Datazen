"""
Utility functions for FinSim-MAPPO
"""

import os
import random
import yaml
import numpy as np
import torch
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_observation(obs: np.ndarray, 
                          mean: Optional[np.ndarray] = None,
                          std: Optional[np.ndarray] = None,
                          clip_range: float = 10.0) -> np.ndarray:
    """
    Normalize observations to [-1, 1] range.
    
    Args:
        obs: Raw observation array
        mean: Running mean for normalization
        std: Running standard deviation
        clip_range: Range to clip normalized values
    
    Returns:
        Normalized observation
    """
    if mean is not None and std is not None:
        obs = (obs - mean) / (std + 1e-8)
    obs = np.clip(obs, -clip_range, clip_range)
    return obs


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominators."""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def exponential_decay(initial_value: float, 
                      final_value: float, 
                      current_step: int, 
                      decay_steps: int) -> float:
    """Calculate exponentially decayed value."""
    if current_step >= decay_steps:
        return final_value
    decay_rate = (final_value / initial_value) ** (1.0 / decay_steps)
    return initial_value * (decay_rate ** current_step)


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                next_value: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor of rewards
        values: Tensor of value estimates
        dones: Tensor of done flags
        next_value: Value estimate for the next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = []
    gae = 0
    
    values_extended = torch.cat([values, next_value.unsqueeze(0)])
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages)
    returns = advantages + values
    
    return advantages, returns


def soft_update(target_network: torch.nn.Module, 
                source_network: torch.nn.Module, 
                tau: float = 0.005) -> None:
    """Soft update target network parameters."""
    for target_param, source_param in zip(target_network.parameters(), 
                                          source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


class RunningMeanStd:
    """Tracks running mean and standard deviation."""
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: np.ndarray, 
                             batch_var: np.ndarray, 
                             batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)
