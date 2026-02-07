"""
PPO Updater: Proximal Policy Optimization update logic.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPOBatch:
    """Batch of data for PPO update."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    global_states: torch.Tensor


class PPOUpdater:
    """
    Handles PPO updates for both actors and critic.
    """
    
    def __init__(self,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = 0.02,
                 device: str = 'cpu'):
        """
        Initialize PPO updater.
        
        Args:
            clip_epsilon: PPO clip parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            target_kl: Target KL divergence for early stopping
            device: Device to use
        """
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)
    
    def compute_policy_loss(self,
                            new_log_probs: torch.Tensor,
                            old_log_probs: torch.Tensor,
                            advantages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute clipped PPO policy loss.
        
        Args:
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            
        Returns:
            Tuple of (policy_loss, approx_kl)
        """
        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Approximate KL divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()
        
        return policy_loss, approx_kl
    
    def compute_value_loss(self,
                           values: torch.Tensor,
                           returns: torch.Tensor,
                           old_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value function loss with optional clipping.
        
        Args:
            values: Value predictions
            returns: Target returns
            old_values: Old value predictions for clipping
            
        Returns:
            Value loss
        """
        if old_values is not None:
            # Clipped value loss
            value_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()
        
        return value_loss
    
    def update_critic(self,
                      critic: nn.Module,
                      optimizer: torch.optim.Optimizer,
                      global_states: torch.Tensor,
                      returns: torch.Tensor,
                      num_epochs: int = 4,
                      batch_size: int = 64) -> Dict[str, float]:
        """
        Update centralized critic.
        
        Args:
            critic: Critic network
            optimizer: Critic optimizer
            global_states: Global state tensors
            returns: Target returns
            num_epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of loss metrics
        """
        total_value_loss = 0.0
        num_updates = 0
        
        # Get old values for clipping
        with torch.no_grad():
            old_values = critic(global_states)
        
        for _ in range(num_epochs):
            indices = torch.randperm(len(global_states))
            
            for start in range(0, len(global_states), batch_size):
                end = min(start + batch_size, len(global_states))
                batch_idx = indices[start:end]
                
                batch_states = global_states[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]
                
                # Forward pass
                values = critic(batch_states)
                
                # Compute loss
                value_loss = self.compute_value_loss(values, batch_returns, batch_old_values)
                
                # Update
                optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_value_loss += value_loss.item()
                num_updates += 1
        
        return {
            'value_loss': total_value_loss / max(num_updates, 1)
        }


def compute_gae(rewards: np.ndarray,
                values: np.ndarray,
                dones: np.ndarray,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    
    gae = 0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to have zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
