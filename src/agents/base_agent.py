"""
Base Agent: Abstract base class for all agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: int
    observation_dim: int = 16
    action_dim: int = 4
    seed: Optional[int] = None


class BaseAgent(ABC):
    """
    Abstract base class for financial agents.
    
    All agents must implement:
    - select_action: Choose action given observation
    - reset: Reset agent state
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        
        self._rng = np.random.default_rng(config.seed)
        
        # Episode statistics
        self.episode_reward: float = 0.0
        self.episode_length: int = 0
        self.total_episodes: int = 0
    
    @abstractmethod
    def select_action(self, 
                      observation: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """
        Select an action given an observation.
        
        Args:
            observation: Current observation vector
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action vector of shape (action_dim,)
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        if self.episode_length > 0:
            self.total_episodes += 1
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def update_stats(self, reward: float) -> None:
        """Update episode statistics."""
        self.episode_reward += reward
        self.episode_length += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'total_episodes': self.total_episodes
        }
    
    def apply_action_mask(self, 
                          action: np.ndarray, 
                          observation: np.ndarray) -> np.ndarray:
        """
        Apply action masking based on current state.
        
        Args:
            action: Raw action vector
            observation: Current observation
            
        Returns:
            Masked action vector
        """
        # Extract relevant info from observation
        cash_normalized = observation[0]  # Normalized cash
        capital_ratio = observation[2]
        is_stressed = observation[6] > 0.5
        
        masked_action = action.copy()
        
        # If very low cash, can't lend
        if cash_normalized < 0.05:
            masked_action[0] = -1.0  # No lending
        
        # If capital ratio is low, reduce selling to avoid fire sales
        if capital_ratio < 0.1 and not is_stressed:
            masked_action[2] = min(masked_action[2], 0.0)
        
        return masked_action
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id})"


class MultiAgentController:
    """
    Controller for managing multiple agents.
    """
    
    def __init__(self):
        """Initialize the controller."""
        self.agents: Dict[int, BaseAgent] = {}
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the controller."""
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: int) -> None:
        """Remove an agent from the controller."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def select_actions(self, 
                       observations: Dict[int, np.ndarray],
                       deterministic: bool = False) -> Dict[int, np.ndarray]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dict mapping agent_id -> observation
            deterministic: Whether to use deterministic policies
            
        Returns:
            Dict mapping agent_id -> action
        """
        actions = {}
        for agent_id, obs in observations.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].select_action(obs, deterministic)
        return actions
    
    def reset_all(self) -> None:
        """Reset all agents."""
        for agent in self.agents.values():
            agent.reset()
    
    def update_stats(self, rewards: Dict[int, float]) -> None:
        """Update statistics for all agents."""
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                self.agents[agent_id].update_stats(reward)
    
    def get_all_stats(self) -> List[Dict[str, float]]:
        """Get statistics for all agents."""
        return [agent.get_stats() for agent in self.agents.values()]
    
    def __len__(self) -> int:
        return len(self.agents)
    
    def __iter__(self):
        return iter(self.agents.values())
