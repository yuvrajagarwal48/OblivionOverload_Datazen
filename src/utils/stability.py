"""
Numerical Stability Safeguards for FinSim-MAPPO.
Provides bounds checking, NaN detection, convergence limits, and safety guards.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
import warnings
import logging


logger = logging.getLogger(__name__)


@dataclass 
class StabilityConfig:
    """Configuration for numerical stability controls."""
    # Value bounds
    min_value: float = -1e8
    max_value: float = 1e8
    min_positive: float = 1e-8  # For denominators
    
    # Gradient controls
    max_grad_norm: float = 0.5
    grad_clip_value: float = 10.0
    
    # Loss bounds
    max_loss: float = 1e6
    min_loss: float = -1e6
    
    # Convergence thresholds
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    # NaN handling
    nan_replacement: float = 0.0
    inf_replacement: float = 1e6
    
    # Observation/action clipping
    obs_clip: float = 10.0
    action_clip: float = 1.0
    reward_clip: float = 100.0
    
    # Financial constraints
    min_capital: float = 0.0
    max_leverage: float = 30.0
    min_liquidity_ratio: float = 0.01
    
    # Enable flags
    enable_nan_detection: bool = True
    enable_gradient_clipping: bool = True
    enable_value_clipping: bool = True
    raise_on_nan: bool = False  # If True, raise exception; else replace


class NumericalStabilityGuard:
    """
    Provides numerical stability safeguards for neural network training
    and financial simulation computations.
    """
    
    def __init__(self, config: StabilityConfig = None):
        self.config = config or StabilityConfig()
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0
        self.warnings_issued = 0
    
    def safe_divide(self, numerator: Union[float, np.ndarray, torch.Tensor],
                    denominator: Union[float, np.ndarray, torch.Tensor],
                    default: float = 0.0) -> Union[float, np.ndarray, torch.Tensor]:
        """Safe division that handles zero denominators."""
        if isinstance(numerator, torch.Tensor):
            # PyTorch version
            safe_denom = torch.where(
                torch.abs(denominator) < self.config.min_positive,
                torch.ones_like(denominator) * self.config.min_positive,
                denominator
            )
            result = numerator / safe_denom
            return torch.where(
                torch.abs(denominator) < self.config.min_positive,
                torch.full_like(result, default),
                result
            )
        elif isinstance(numerator, np.ndarray):
            # NumPy version
            safe_denom = np.where(
                np.abs(denominator) < self.config.min_positive,
                np.ones_like(denominator) * self.config.min_positive,
                denominator
            )
            result = numerator / safe_denom
            return np.where(
                np.abs(denominator) < self.config.min_positive,
                np.full_like(result, default),
                result
            )
        else:
            # Scalar version
            if abs(denominator) < self.config.min_positive:
                return default
            return numerator / denominator
    
    def check_and_fix_tensor(self, tensor: torch.Tensor, 
                              name: str = "tensor") -> Tuple[torch.Tensor, bool]:
        """
        Check tensor for NaN/Inf values and fix them.
        Returns (fixed_tensor, had_issues).
        """
        if not self.config.enable_nan_detection:
            return tensor, False
        
        had_issues = False
        
        # Check for NaN
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            self.nan_count += nan_mask.sum().item()
            had_issues = True
            
            if self.config.raise_on_nan:
                raise ValueError(f"NaN detected in {name}")
            
            logger.warning(f"NaN detected in {name}, replacing with {self.config.nan_replacement}")
            tensor = torch.where(nan_mask, 
                                 torch.full_like(tensor, self.config.nan_replacement), 
                                 tensor)
        
        # Check for Inf
        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            self.inf_count += inf_mask.sum().item()
            had_issues = True
            
            if self.config.raise_on_nan:
                raise ValueError(f"Inf detected in {name}")
            
            logger.warning(f"Inf detected in {name}, replacing with bounded value")
            tensor = torch.where(
                inf_mask & (tensor > 0),
                torch.full_like(tensor, self.config.inf_replacement),
                tensor
            )
            tensor = torch.where(
                inf_mask & (tensor < 0),
                torch.full_like(tensor, -self.config.inf_replacement),
                tensor
            )
        
        return tensor, had_issues
    
    def check_and_fix_array(self, array: np.ndarray,
                            name: str = "array") -> Tuple[np.ndarray, bool]:
        """
        Check numpy array for NaN/Inf values and fix them.
        Returns (fixed_array, had_issues).
        """
        if not self.config.enable_nan_detection:
            return array, False
        
        had_issues = False
        result = array.copy()
        
        # Check for NaN
        nan_mask = np.isnan(result)
        if nan_mask.any():
            self.nan_count += nan_mask.sum()
            had_issues = True
            
            if self.config.raise_on_nan:
                raise ValueError(f"NaN detected in {name}")
            
            result[nan_mask] = self.config.nan_replacement
        
        # Check for Inf
        inf_mask = np.isinf(result)
        if inf_mask.any():
            self.inf_count += inf_mask.sum()
            had_issues = True
            
            if self.config.raise_on_nan:
                raise ValueError(f"Inf detected in {name}")
            
            result[inf_mask & (result > 0)] = self.config.inf_replacement
            result[inf_mask & (result < 0)] = -self.config.inf_replacement
        
        return result, had_issues
    
    def clip_gradients(self, model: torch.nn.Module, 
                        max_norm: float = None) -> float:
        """Clip gradients and return the total norm before clipping."""
        if not self.config.enable_gradient_clipping:
            return 0.0
        
        max_norm = max_norm or self.config.max_grad_norm
        
        # Compute and clip
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm
        )
        
        if total_norm > max_norm:
            self.clip_count += 1
        
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
    
    def clip_observations(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Clip observations to valid range."""
        if not self.config.enable_value_clipping:
            return obs
        
        if isinstance(obs, torch.Tensor):
            return torch.clamp(obs, -self.config.obs_clip, self.config.obs_clip)
        return np.clip(obs, -self.config.obs_clip, self.config.obs_clip)
    
    def clip_actions(self, actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Clip actions to valid range."""
        if not self.config.enable_value_clipping:
            return actions
        
        if isinstance(actions, torch.Tensor):
            return torch.clamp(actions, -self.config.action_clip, self.config.action_clip)
        return np.clip(actions, -self.config.action_clip, self.config.action_clip)
    
    def clip_rewards(self, rewards: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Clip rewards to valid range."""
        if not self.config.enable_value_clipping:
            return rewards
        
        if isinstance(rewards, torch.Tensor):
            return torch.clamp(rewards, -self.config.reward_clip, self.config.reward_clip)
        elif isinstance(rewards, np.ndarray):
            return np.clip(rewards, -self.config.reward_clip, self.config.reward_clip)
        else:
            return max(-self.config.reward_clip, min(self.config.reward_clip, rewards))
    
    def validate_financial_state(self, 
                                  capital: float, 
                                  assets: float,
                                  liabilities: float,
                                  name: str = "entity") -> Dict[str, Any]:
        """
        Validate financial state and return any violations.
        """
        violations = {}
        
        # Check capital
        if capital < self.config.min_capital:
            violations['capital'] = {
                'value': capital,
                'min_required': self.config.min_capital,
                'message': f"{name} has insufficient capital: {capital}"
            }
        
        # Check leverage
        if liabilities > 0:
            leverage = assets / max(capital, self.config.min_positive)
            if leverage > self.config.max_leverage:
                violations['leverage'] = {
                    'value': leverage,
                    'max_allowed': self.config.max_leverage,
                    'message': f"{name} exceeds max leverage: {leverage:.2f}"
                }
        
        # Check for negative values
        if assets < 0:
            violations['assets'] = {
                'value': assets,
                'message': f"{name} has negative assets: {assets}"
            }
        
        return violations
    
    def safe_log(self, x: Union[float, np.ndarray, torch.Tensor],
                  epsilon: float = None) -> Union[float, np.ndarray, torch.Tensor]:
        """Safe logarithm that handles zero and negative values."""
        epsilon = epsilon or self.config.min_positive
        
        if isinstance(x, torch.Tensor):
            return torch.log(torch.clamp(x, min=epsilon))
        elif isinstance(x, np.ndarray):
            return np.log(np.clip(x, epsilon, None))
        else:
            return np.log(max(x, epsilon))
    
    def safe_exp(self, x: Union[float, np.ndarray, torch.Tensor],
                  max_val: float = 30.0) -> Union[float, np.ndarray, torch.Tensor]:
        """Safe exponential that prevents overflow."""
        if isinstance(x, torch.Tensor):
            return torch.exp(torch.clamp(x, max=max_val))
        elif isinstance(x, np.ndarray):
            return np.exp(np.clip(x, None, max_val))
        else:
            return np.exp(min(x, max_val))
    
    def safe_softmax(self, logits: torch.Tensor, dim: int = -1,
                      temperature: float = 1.0) -> torch.Tensor:
        """Numerically stable softmax with temperature."""
        # Subtract max for stability
        logits = logits / max(temperature, self.config.min_positive)
        logits = logits - logits.max(dim=dim, keepdim=True)[0]
        
        exp_logits = torch.exp(logits)
        probs = exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + self.config.min_positive)
        
        return probs
    
    def get_statistics(self) -> Dict[str, int]:
        """Get accumulated stability statistics."""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'clip_count': self.clip_count,
            'warnings_issued': self.warnings_issued
        }
    
    def reset_statistics(self) -> None:
        """Reset accumulated statistics."""
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0
        self.warnings_issued = 0


class ConvergenceChecker:
    """
    Checks for convergence in iterative algorithms (e.g., Eisenberg-Noe clearing).
    """
    
    def __init__(self, 
                 tolerance: float = 1e-6,
                 max_iterations: int = 1000,
                 patience: int = 10):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.patience = patience
        
        self.iteration = 0
        self.previous_value = None
        self.stagnant_count = 0
        self.history = []
    
    def check(self, current_value: Union[float, np.ndarray]) -> Tuple[bool, str]:
        """
        Check if algorithm has converged.
        Returns (converged, reason).
        """
        self.iteration += 1
        
        # Compute scalar value if array
        if isinstance(current_value, np.ndarray):
            current_scalar = np.sum(np.abs(current_value))
        else:
            current_scalar = abs(current_value)
        
        self.history.append(current_scalar)
        
        # Check max iterations
        if self.iteration >= self.max_iterations:
            return True, "max_iterations"
        
        # Check convergence
        if self.previous_value is not None:
            if isinstance(current_value, np.ndarray):
                delta = np.max(np.abs(current_value - self.previous_value))
            else:
                delta = abs(current_value - self.previous_value)
            
            if delta < self.tolerance:
                return True, "converged"
            
            # Check stagnation
            if delta < self.tolerance * 100:
                self.stagnant_count += 1
                if self.stagnant_count >= self.patience:
                    return True, "stagnant"
            else:
                self.stagnant_count = 0
        
        self.previous_value = current_value.copy() if isinstance(current_value, np.ndarray) else current_value
        return False, "continuing"
    
    def reset(self) -> None:
        """Reset checker state."""
        self.iteration = 0
        self.previous_value = None
        self.stagnant_count = 0
        self.history = []


class TrainingStabilizer:
    """
    Provides training stabilization mechanisms for MAPPO.
    """
    
    def __init__(self, 
                 guard: NumericalStabilityGuard = None,
                 warmup_steps: int = 1000,
                 lr_decay_factor: float = 0.99,
                 loss_spike_threshold: float = 10.0):
        self.guard = guard or NumericalStabilityGuard()
        self.warmup_steps = warmup_steps
        self.lr_decay_factor = lr_decay_factor
        self.loss_spike_threshold = loss_spike_threshold
        
        self.step_count = 0
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.lr_multiplier = 0.0
    
    def get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for warmup."""
        if self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps
        return 1.0
    
    def should_skip_update(self, loss: float) -> bool:
        """
        Check if update should be skipped due to loss spike.
        """
        if len(self.loss_history) < 10:
            return False
        
        mean_loss = np.mean(self.loss_history[-10:])
        if abs(loss) > mean_loss * self.loss_spike_threshold:
            logger.warning(f"Loss spike detected: {loss:.4f} vs mean {mean_loss:.4f}")
            return True
        
        return False
    
    def record_step(self, loss: float, grad_norm: float) -> None:
        """Record training step metrics."""
        self.step_count += 1
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        # Keep history bounded
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history = self.grad_norm_history[-500:]
    
    def process_loss(self, loss: torch.Tensor, name: str = "loss") -> torch.Tensor:
        """Process loss with stability checks."""
        # Check for NaN/Inf
        loss, had_issues = self.guard.check_and_fix_tensor(loss, name)
        
        if had_issues:
            logger.warning(f"Loss {name} had numerical issues, returning zero loss")
            return torch.zeros_like(loss)
        
        # Clip loss
        loss = torch.clamp(loss, 
                           self.guard.config.min_loss, 
                           self.guard.config.max_loss)
        
        return loss
    
    def process_value_estimate(self, values: torch.Tensor) -> torch.Tensor:
        """Process value estimates with stability."""
        values, _ = self.guard.check_and_fix_tensor(values, "values")
        return torch.clamp(values, 
                           self.guard.config.min_value / 1000, 
                           self.guard.config.max_value / 1000)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get training stability diagnostics."""
        diagnostics = {
            'step_count': self.step_count,
            'lr_multiplier': self.get_lr_multiplier(),
            'stability_stats': self.guard.get_statistics()
        }
        
        if self.loss_history:
            diagnostics['recent_loss_mean'] = np.mean(self.loss_history[-100:])
            diagnostics['recent_loss_std'] = np.std(self.loss_history[-100:])
        
        if self.grad_norm_history:
            diagnostics['recent_grad_mean'] = np.mean(self.grad_norm_history[-100:])
            diagnostics['recent_grad_max'] = max(self.grad_norm_history[-100:])
        
        return diagnostics


# Global guard instance for convenience
_default_guard: Optional[NumericalStabilityGuard] = None


def get_stability_guard() -> NumericalStabilityGuard:
    """Get or create the default stability guard."""
    global _default_guard
    if _default_guard is None:
        _default_guard = NumericalStabilityGuard()
    return _default_guard


def set_stability_config(config: StabilityConfig) -> None:
    """Set the global stability configuration."""
    global _default_guard
    _default_guard = NumericalStabilityGuard(config)
