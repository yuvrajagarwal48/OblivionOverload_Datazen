"""
Decision Support: Lookahead simulation and action recommendations.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..environment import FinancialEnvironment
from ..agents import BaseAgent, MAPPOAgent


class ActionType(Enum):
    """Types of recommended actions."""
    LEND = "lend"
    HOARD = "hoard"
    SELL = "sell"
    BORROW = "borrow"
    HOLD = "hold"


@dataclass
class Recommendation:
    """Action recommendation for a bank."""
    bank_id: int
    action_type: ActionType
    action_vector: np.ndarray
    expected_profit: float
    default_probability: float
    confidence_score: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bank_id': self.bank_id,
            'action_type': self.action_type.value,
            'action_vector': self.action_vector.tolist(),
            'expected_profit': self.expected_profit,
            'default_probability': self.default_probability,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives
        }


@dataclass
class SimulationOutcome:
    """Outcome of a lookahead simulation."""
    action: np.ndarray
    total_reward: float
    final_equity: float
    defaulted: bool
    system_defaults: int
    risk_score: float


class LookaheadSimulator:
    """
    Simulates future outcomes for different action choices.
    """
    
    def __init__(self,
                 env: FinancialEnvironment,
                 horizon: int = 10,
                 num_simulations: int = 20,
                 seed: Optional[int] = None):
        """
        Initialize lookahead simulator.
        
        Args:
            env: Reference environment (will be cloned)
            horizon: Number of steps to look ahead
            num_simulations: Number of simulations per action
            seed: Random seed
        """
        self.reference_env = env
        self.horizon = horizon
        self.num_simulations = num_simulations
        self._rng = np.random.default_rng(seed)
    
    def simulate_action(self,
                        bank_id: int,
                        action: np.ndarray,
                        current_state: Dict[int, np.ndarray],
                        other_agents: Dict[int, BaseAgent]) -> SimulationOutcome:
        """
        Simulate the outcome of a specific action.
        
        Args:
            bank_id: ID of the bank taking the action
            action: Action vector to evaluate
            current_state: Current observations for all agents
            other_agents: Other agents' policies
            
        Returns:
            Simulation outcome
        """
        total_reward = 0.0
        total_defaults = 0
        num_runs = 0
        final_equities = []
        bank_defaulted_count = 0
        
        for _ in range(self.num_simulations):
            # Clone environment
            env = copy.deepcopy(self.reference_env)
            
            observations = current_state.copy()
            bank_reward = 0.0
            
            for step in range(self.horizon):
                # Build actions
                actions = {}
                
                for agent_id, agent in other_agents.items():
                    if agent_id == bank_id:
                        actions[agent_id] = action if step == 0 else agent.select_action(
                            observations[agent_id], deterministic=True
                        )
                    else:
                        actions[agent_id] = agent.select_action(
                            observations[agent_id], deterministic=True
                        )
                
                # Step
                result = env.step(actions)
                
                bank_reward += result.rewards[bank_id]
                observations = result.observations
                
                if all(result.dones.values()):
                    break
            
            # Record outcomes
            bank = env.network.banks[bank_id]
            final_equities.append(bank.balance_sheet.equity)
            
            if bank.status.value == 'defaulted':
                bank_defaulted_count += 1
            
            total_reward += bank_reward
            total_defaults += env.network.get_network_stats().num_defaulted
            num_runs += 1
        
        avg_reward = total_reward / max(num_runs, 1)
        avg_equity = np.mean(final_equities)
        default_prob = bank_defaulted_count / max(num_runs, 1)
        avg_system_defaults = total_defaults / max(num_runs, 1)
        
        # Risk score (higher is worse)
        risk_score = 0.3 * default_prob + 0.3 * (avg_system_defaults / len(other_agents)) + 0.4 * max(0, -avg_reward / 100)
        
        return SimulationOutcome(
            action=action,
            total_reward=avg_reward,
            final_equity=avg_equity,
            defaulted=default_prob > 0.5,
            system_defaults=int(avg_system_defaults),
            risk_score=risk_score
        )
    
    def evaluate_action_space(self,
                              bank_id: int,
                              current_state: Dict[int, np.ndarray],
                              other_agents: Dict[int, BaseAgent],
                              num_candidates: int = 10) -> List[SimulationOutcome]:
        """
        Evaluate multiple candidate actions.
        
        Args:
            bank_id: Bank to evaluate
            current_state: Current observations
            other_agents: Other agents' policies
            num_candidates: Number of action candidates
            
        Returns:
            List of simulation outcomes sorted by reward
        """
        outcomes = []
        
        # Generate candidate actions
        candidates = self._generate_candidate_actions(num_candidates)
        
        for action in candidates:
            outcome = self.simulate_action(bank_id, action, current_state, other_agents)
            outcomes.append(outcome)
        
        # Sort by reward (descending)
        outcomes.sort(key=lambda x: x.total_reward, reverse=True)
        
        return outcomes
    
    def _generate_candidate_actions(self, num_candidates: int) -> List[np.ndarray]:
        """Generate diverse candidate actions."""
        actions = []
        
        # Predefined strategies
        strategies = [
            np.array([0.8, -0.5, -0.5, -0.5]),   # Aggressive lending
            np.array([-0.8, 0.8, -0.5, -0.5]),   # Conservative hoarding
            np.array([-0.5, -0.5, 0.5, -0.5]),   # Asset selling
            np.array([-0.5, -0.5, -0.5, 0.5]),   # Borrowing
            np.array([0.0, 0.0, 0.0, 0.0]),      # Neutral
            np.array([0.3, 0.3, -0.3, -0.3]),    # Balanced growth
            np.array([-0.3, 0.5, 0.0, 0.0]),     # Defensive
        ]
        
        actions.extend(strategies)
        
        # Random variations
        while len(actions) < num_candidates:
            base = strategies[self._rng.integers(0, len(strategies))]
            noise = self._rng.uniform(-0.3, 0.3, size=4)
            action = np.clip(base + noise, -1, 1)
            actions.append(action)
        
        return actions[:num_candidates]


class DecisionSupport:
    """
    Provides actionable recommendations for financial institutions.
    """
    
    def __init__(self,
                 env: FinancialEnvironment,
                 agents: Dict[int, BaseAgent],
                 horizon: int = 10,
                 num_simulations: int = 20):
        """
        Initialize decision support system.
        
        Args:
            env: Financial environment
            agents: All agents
            horizon: Lookahead horizon
            num_simulations: Simulations per action
        """
        self.env = env
        self.agents = agents
        self.lookahead = LookaheadSimulator(env, horizon, num_simulations)
    
    def get_recommendation(self,
                           bank_id: int,
                           observations: Dict[int, np.ndarray]) -> Recommendation:
        """
        Get action recommendation for a specific bank.
        
        Args:
            bank_id: Bank to get recommendation for
            observations: Current observations
            
        Returns:
            Action recommendation
        """
        # Evaluate action space
        outcomes = self.lookahead.evaluate_action_space(
            bank_id, observations, self.agents, num_candidates=15
        )
        
        if not outcomes:
            # Default recommendation
            return self._create_default_recommendation(bank_id)
        
        # Best outcome
        best = outcomes[0]
        
        # Determine action type
        action_type = self._classify_action(best.action)
        
        # Calculate confidence
        if len(outcomes) > 1:
            reward_spread = outcomes[0].total_reward - outcomes[-1].total_reward
            confidence = min(1.0, reward_spread / 50) if reward_spread > 0 else 0.5
        else:
            confidence = 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(bank_id, best, observations)
        
        # Get alternatives
        alternatives = []
        for i, outcome in enumerate(outcomes[1:4], 1):
            alternatives.append({
                'rank': i + 1,
                'action_type': self._classify_action(outcome.action).value,
                'expected_profit': outcome.total_reward,
                'default_probability': 1.0 if outcome.defaulted else outcome.risk_score
            })
        
        return Recommendation(
            bank_id=bank_id,
            action_type=action_type,
            action_vector=best.action,
            expected_profit=best.total_reward,
            default_probability=1.0 if best.defaulted else best.risk_score,
            confidence_score=confidence,
            reasoning=reasoning,
            alternatives=alternatives
        )
    
    def get_all_recommendations(self,
                                 observations: Dict[int, np.ndarray]) -> Dict[int, Recommendation]:
        """Get recommendations for all banks."""
        recommendations = {}
        for bank_id in self.agents.keys():
            recommendations[bank_id] = self.get_recommendation(bank_id, observations)
        return recommendations
    
    def evaluate_what_if(self,
                         bank_id: int,
                         action: np.ndarray,
                         observations: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate a specific what-if scenario.
        
        Args:
            bank_id: Bank taking the action
            action: Proposed action
            observations: Current observations
            
        Returns:
            Scenario evaluation results
        """
        outcome = self.lookahead.simulate_action(
            bank_id, action, observations, self.agents
        )
        
        return {
            'expected_profit': outcome.total_reward,
            'final_equity': outcome.final_equity,
            'default_probability': 1.0 if outcome.defaulted else outcome.risk_score,
            'system_defaults': outcome.system_defaults,
            'risk_score': outcome.risk_score,
            'recommendation': 'proceed' if outcome.total_reward > 0 and not outcome.defaulted else 'reconsider'
        }
    
    def _classify_action(self, action: np.ndarray) -> ActionType:
        """Classify an action vector into an action type."""
        # Find dominant component
        abs_action = np.abs(action)
        dominant_idx = np.argmax(abs_action)
        
        if abs_action[dominant_idx] < 0.2:
            return ActionType.HOLD
        
        if dominant_idx == 0:
            return ActionType.LEND if action[0] > 0 else ActionType.HOLD
        elif dominant_idx == 1:
            return ActionType.HOARD
        elif dominant_idx == 2:
            return ActionType.SELL
        else:
            return ActionType.BORROW
    
    def _generate_reasoning(self,
                            bank_id: int,
                            outcome: SimulationOutcome,
                            observations: Dict[int, np.ndarray]) -> str:
        """Generate human-readable reasoning for the recommendation."""
        obs = observations[bank_id]
        capital_ratio = obs[2]
        is_stressed = obs[6] > 0.5
        
        action_type = self._classify_action(outcome.action)
        
        reasons = []
        
        if action_type == ActionType.LEND:
            reasons.append("Lending is recommended as you have excess liquidity")
            if capital_ratio > 0.15:
                reasons.append("Your strong capital position supports increased exposure")
        elif action_type == ActionType.HOARD:
            reasons.append("Hoarding cash is advised given market conditions")
            if is_stressed:
                reasons.append("This helps maintain stability during stress")
        elif action_type == ActionType.SELL:
            reasons.append("Asset sales are recommended to improve liquidity")
            if capital_ratio < 0.1:
                reasons.append("This will help restore your capital ratio")
        elif action_type == ActionType.BORROW:
            reasons.append("Borrowing is suggested to meet liquidity needs")
        else:
            reasons.append("Maintaining current position is optimal")
        
        if outcome.total_reward > 0:
            reasons.append(f"Expected profit: {outcome.total_reward:.2f}")
        
        if outcome.risk_score > 0.5:
            reasons.append("Note: This action carries elevated risk")
        
        return ". ".join(reasons) + "."
    
    def _create_default_recommendation(self, bank_id: int) -> Recommendation:
        """Create a default conservative recommendation."""
        return Recommendation(
            bank_id=bank_id,
            action_type=ActionType.HOLD,
            action_vector=np.zeros(4),
            expected_profit=0.0,
            default_probability=0.0,
            confidence_score=0.3,
            reasoning="Default conservative recommendation due to insufficient data.",
            alternatives=[]
        )
    
    def generate_report(self,
                        observations: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Generate comprehensive decision support report.
        
        Args:
            observations: Current observations
            
        Returns:
            Complete decision support report
        """
        recommendations = self.get_all_recommendations(observations)
        
        # Aggregate statistics
        action_distribution = {}
        avg_confidence = 0.0
        high_risk_banks = []
        
        for bank_id, rec in recommendations.items():
            action_type = rec.action_type.value
            action_distribution[action_type] = action_distribution.get(action_type, 0) + 1
            avg_confidence += rec.confidence_score
            
            if rec.default_probability > 0.3:
                high_risk_banks.append(bank_id)
        
        avg_confidence /= max(len(recommendations), 1)
        
        return {
            'timestamp': None,  # Will be set by caller
            'num_banks': len(recommendations),
            'recommendations': {bid: rec.to_dict() for bid, rec in recommendations.items()},
            'action_distribution': action_distribution,
            'average_confidence': avg_confidence,
            'high_risk_banks': high_risk_banks,
            'summary': self._generate_summary(recommendations, action_distribution)
        }
    
    def _generate_summary(self,
                          recommendations: Dict[int, Recommendation],
                          action_distribution: Dict[str, int]) -> str:
        """Generate a text summary of recommendations."""
        total = len(recommendations)
        
        parts = [f"Analysis of {total} financial institutions:"]
        
        for action_type, count in action_distribution.items():
            pct = count / total * 100
            parts.append(f"  - {action_type}: {count} banks ({pct:.1f}%)")
        
        high_risk = sum(1 for r in recommendations.values() if r.default_probability > 0.3)
        if high_risk > 0:
            parts.append(f"WARNING: {high_risk} banks have elevated default risk")
        
        return "\n".join(parts)
