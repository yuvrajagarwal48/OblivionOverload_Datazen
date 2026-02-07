"""
Baseline Agents: Non-learning agents for comparison.
"""

import numpy as np
from typing import Optional

from .base_agent import BaseAgent, AgentConfig


class RandomAgent(BaseAgent):
    """
    Random agent that samples actions uniformly.
    Used as a baseline for comparison.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Select random action."""
        action = self._rng.uniform(-1, 1, size=self.action_dim)
        return action.astype(np.float32)


class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent that follows simple heuristics.
    
    Rules:
    - Lend if capital ratio > threshold
    - Hoard if capital ratio is low
    - Sell if capital ratio is critically low
    - Borrow if cash is low
    """
    
    def __init__(self, 
                 config: AgentConfig,
                 cr_threshold: float = 0.15,
                 cr_critical: float = 0.08,
                 cash_threshold: float = 0.1):
        """
        Initialize rule-based agent.
        
        Args:
            config: Agent configuration
            cr_threshold: Capital ratio threshold for lending
            cr_critical: Critical capital ratio (triggers asset sales)
            cash_threshold: Low cash threshold
        """
        super().__init__(config)
        self.cr_threshold = cr_threshold
        self.cr_critical = cr_critical
        self.cash_threshold = cash_threshold
    
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Select action based on rules."""
        # Parse observation
        cash_norm = observation[0]
        capital_ratio = observation[2]
        is_stressed = observation[6] > 0.5
        market_volatility = observation[12] if len(observation) > 12 else 0.02
        
        # Initialize action
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Lending decision
        if capital_ratio > self.cr_threshold and not is_stressed:
            # Good shape - can lend
            action[0] = 0.5  # Moderate lending
        else:
            action[0] = -1.0  # No lending
        
        # Hoarding decision
        if capital_ratio < self.cr_threshold or market_volatility > 0.05:
            action[1] = 0.5  # Hoard more
        else:
            action[1] = -0.5  # Less hoarding
        
        # Selling decision
        if capital_ratio < self.cr_critical:
            action[2] = 0.3  # Sell some assets
        elif capital_ratio > self.cr_threshold * 1.5:
            action[2] = -0.5  # Hold assets
        else:
            action[2] = 0.0  # Neutral
        
        # Borrowing decision
        if cash_norm < self.cash_threshold:
            action[3] = 0.5  # Request borrowing
        else:
            action[3] = -1.0  # No borrowing needed
        
        # Add small noise if not deterministic
        if not deterministic:
            noise = self._rng.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action


class MyopicAgent(BaseAgent):
    """
    Myopic agent that maximizes immediate profit.
    
    Focuses on short-term gains without considering systemic effects.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._last_equity = None
    
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Select action to maximize immediate profit."""
        # Parse observation
        cash_norm = observation[0]
        equity_norm = observation[1]
        capital_ratio = observation[2]
        interest_rate = observation[11] if len(observation) > 11 else 0.05
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Track equity change
        if self._last_equity is not None:
            equity_change = equity_norm - self._last_equity
        else:
            equity_change = 0.0
        self._last_equity = equity_norm
        
        # Lending: if interest rates are high, lend more
        if interest_rate > 0.06 and capital_ratio > 0.1:
            action[0] = 0.8  # Aggressive lending
        elif interest_rate > 0.04:
            action[0] = 0.3  # Moderate lending
        else:
            action[0] = -0.5  # Low returns, don't lend
        
        # Hoarding: minimize based on expected returns
        action[1] = -0.5  # Myopic - deploy capital
        
        # Selling: sell if prices seem high
        price_norm = observation[10] if len(observation) > 10 else 1.0
        if price_norm > 1.1:
            action[2] = 0.5  # Sell at high prices
        elif price_norm < 0.9:
            action[2] = -0.5  # Hold during low prices
        else:
            action[2] = 0.0
        
        # Borrowing: borrow if rates are low
        if interest_rate < 0.04 and capital_ratio > 0.12:
            action[3] = 0.5  # Cheap borrowing for leverage
        else:
            action[3] = -0.5
        
        # Add noise if not deterministic
        if not deterministic:
            noise = self._rng.normal(0, 0.15, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action.astype(np.float32)
    
    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._last_equity = None


class GreedyAgent(BaseAgent):
    """
    Greedy agent that maximizes short-term cash position.
    
    Hoards cash aggressively and only lends at very favorable rates.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Select action to maximize cash position."""
        # Parse observation
        cash_norm = observation[0]
        capital_ratio = observation[2]
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Very conservative lending (only when very flush with cash)
        if cash_norm > 0.5 and capital_ratio > 0.2:
            action[0] = 0.3  # Small lending
        else:
            action[0] = -1.0  # No lending
        
        # Maximum hoarding
        action[1] = 1.0  # Hoard everything
        
        # Sell assets to build cash
        if capital_ratio > 0.15:
            action[2] = 0.2  # Gradual asset sales
        else:
            action[2] = 0.0  # Hold if capital is low
        
        # Never borrow
        action[3] = -1.0
        
        # Add small noise if not deterministic
        if not deterministic:
            noise = self._rng.normal(0, 0.05, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action.astype(np.float32)


class ConservativeAgent(BaseAgent):
    """
    Conservative agent that prioritizes stability and survival.
    
    Maintains high capital ratios and avoids risky positions.
    """
    
    def __init__(self, 
                 config: AgentConfig,
                 target_cr: float = 0.15):
        super().__init__(config)
        self.target_cr = target_cr
    
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Select conservative action."""
        capital_ratio = observation[2]
        is_stressed = observation[6] > 0.5
        neighbor_default_rate = observation[8] if len(observation) > 8 else 0.0
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Reduce exposure when neighbors are defaulting
        risk_multiplier = 1.0 - neighbor_default_rate
        
        # Lending: very selective
        if capital_ratio > self.target_cr and not is_stressed:
            action[0] = 0.2 * risk_multiplier
        else:
            action[0] = -0.8
        
        # Hoarding: maintain buffer
        if capital_ratio < self.target_cr:
            action[1] = 0.8  # Build reserves
        else:
            action[1] = 0.3  # Maintain buffer
        
        # Selling: only if needed for capital
        if capital_ratio < 0.1:
            action[2] = 0.3
        else:
            action[2] = -0.5  # Hold assets
        
        # Borrowing: minimal
        if capital_ratio < 0.08:
            action[3] = 0.3  # Emergency borrowing
        else:
            action[3] = -0.5
        
        if not deterministic:
            noise = self._rng.normal(0, 0.08, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action.astype(np.float32)


def create_baseline_agent(agent_type: str, config: AgentConfig) -> BaseAgent:
    """
    Factory function to create baseline agents.
    
    Args:
        agent_type: Type of agent ('random', 'rule_based', 'myopic', 'greedy', 'conservative')
        config: Agent configuration
        
    Returns:
        Instantiated agent
    """
    agents = {
        'random': RandomAgent,
        'rule_based': RuleBasedAgent,
        'myopic': MyopicAgent,
        'greedy': GreedyAgent,
        'conservative': ConservativeAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    return agents[agent_type](config)
