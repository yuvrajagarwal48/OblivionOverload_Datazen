"""
Belief Construction Layer for FinSim-MAPPO.
Each bank maintains subjective beliefs about counterparties.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json

from .memory import PrivateMemory, InteractionType


class BeliefConfidence(str, Enum):
    """Confidence levels for beliefs."""
    NONE = "none"           # No data
    VERY_LOW = "very_low"   # 1-2 interactions
    LOW = "low"             # 3-5 interactions
    MEDIUM = "medium"       # 6-15 interactions
    HIGH = "high"           # 16-30 interactions
    VERY_HIGH = "very_high" # 30+ interactions


@dataclass
class CounterpartyBelief:
    """
    Bank's subjective belief about a single counterparty.
    This may be inaccurate - that's intentional.
    """
    counterparty_id: int
    
    # Core probability estimates
    estimated_pd: float = 0.05           # Probability of default (0-1)
    estimated_delay_prob: float = 0.1    # Probability of payment delay
    estimated_lgd: float = 0.45          # Loss given default
    
    # Expected outcomes
    expected_profit_rate: float = 0.0    # Expected profit per unit
    expected_margin_stress: float = 0.0  # Expected margin pressure
    expected_volatility: float = 0.1     # Expected behavior volatility
    
    # Quality assessment
    reliability_score: float = 0.5       # Overall reliability (0-1)
    trust_score: float = 0.5             # Trust level (0-1)
    
    # Confidence in these beliefs
    confidence: BeliefConfidence = BeliefConfidence.NONE
    confidence_score: float = 0.0        # Numeric confidence (0-1)
    sample_size: int = 0                 # Number of observations
    
    # Temporal tracking
    last_updated: int = 0
    belief_age: int = 0                  # Timesteps since last update
    
    # Uncertainty bounds (for risk-aware decisions)
    pd_lower: float = 0.01
    pd_upper: float = 0.20
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['confidence'] = self.confidence.value
        return result
    
    def get_risk_adjusted_pd(self, risk_aversion: float = 1.0) -> float:
        """
        Get risk-adjusted PD based on confidence.
        Lower confidence -> higher uncertainty -> use upper bound more.
        """
        # Weight toward upper bound when uncertain
        uncertainty_weight = 1.0 - self.confidence_score
        adjusted = (self.estimated_pd * self.confidence_score + 
                   self.pd_upper * uncertainty_weight * risk_aversion)
        return min(adjusted, 0.99)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert belief to feature vector for observation."""
        return np.array([
            self.estimated_pd,
            self.estimated_delay_prob,
            self.estimated_lgd,
            self.expected_profit_rate,
            self.expected_margin_stress,
            self.reliability_score,
            self.trust_score,
            self.confidence_score,
            min(self.sample_size / 30, 1.0),  # Normalized sample size
            np.exp(-0.1 * self.belief_age)    # Recency factor
        ], dtype=np.float32)


@dataclass
class MarketPrior:
    """
    Prior beliefs from market-level signals.
    Used when no private history exists.
    """
    average_pd: float = 0.03
    average_delay_prob: float = 0.15
    average_lgd: float = 0.45
    market_stress: float = 0.0
    ccp_stress_signal: float = 0.0
    exchange_congestion: float = 0.0
    
    def to_belief(self, counterparty_id: int) -> CounterpartyBelief:
        """Create uninformed prior belief."""
        return CounterpartyBelief(
            counterparty_id=counterparty_id,
            estimated_pd=self.average_pd * (1 + self.market_stress),
            estimated_delay_prob=self.average_delay_prob,
            estimated_lgd=self.average_lgd,
            reliability_score=0.5,
            trust_score=0.5,
            confidence=BeliefConfidence.NONE,
            confidence_score=0.0,
            pd_lower=self.average_pd * 0.5,
            pd_upper=self.average_pd * 3.0
        )


class BeliefUpdateRule:
    """
    Defines how beliefs are updated from new evidence.
    """
    
    def __init__(self,
                 learning_rate: float = 0.3,
                 decay_rate: float = 0.05,
                 min_confidence: float = 0.05,
                 max_confidence: float = 0.95):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
    
    def update_from_outcome(self,
                            belief: CounterpartyBelief,
                            default_occurred: bool,
                            delay_occurred: bool,
                            profit_loss: float,
                            amount: float) -> CounterpartyBelief:
        """
        Update belief based on observed outcome.
        Uses Bayesian-like weighted update.
        """
        # Adaptive learning rate based on confidence
        # Learn faster when uncertain, slower when confident
        effective_lr = self.learning_rate * (1.0 - 0.5 * belief.confidence_score)
        
        # Update PD
        observed_default = 1.0 if default_occurred else 0.0
        belief.estimated_pd = (
            (1 - effective_lr) * belief.estimated_pd + 
            effective_lr * observed_default
        )
        
        # Update delay probability
        observed_delay = 1.0 if delay_occurred else 0.0
        belief.estimated_delay_prob = (
            (1 - effective_lr) * belief.estimated_delay_prob +
            effective_lr * observed_delay
        )
        
        # Update expected profit rate
        if amount > 0:
            observed_profit_rate = profit_loss / amount
            belief.expected_profit_rate = (
                (1 - effective_lr) * belief.expected_profit_rate +
                effective_lr * observed_profit_rate
            )
        
        # Update reliability score (inverse of problems)
        problem_occurred = default_occurred or delay_occurred
        observed_reliability = 0.0 if problem_occurred else 1.0
        belief.reliability_score = (
            (1 - effective_lr) * belief.reliability_score +
            effective_lr * observed_reliability
        )
        
        # Update trust (slower than reliability)
        trust_lr = effective_lr * 0.5
        if not default_occurred:
            belief.trust_score = min(belief.trust_score + trust_lr * 0.1, 1.0)
        else:
            belief.trust_score = max(belief.trust_score - trust_lr * 0.5, 0.0)
        
        # Increment sample size
        belief.sample_size += 1
        
        # Update confidence
        belief.confidence_score = self._compute_confidence(belief.sample_size)
        belief.confidence = self._confidence_to_level(belief.confidence_score)
        
        # Update uncertainty bounds
        self._update_uncertainty_bounds(belief)
        
        belief.last_updated = 0  # Will be set by caller
        belief.belief_age = 0
        
        return belief
    
    def decay_belief(self, belief: CounterpartyBelief, timesteps: int = 1) -> CounterpartyBelief:
        """
        Apply time decay to belief confidence.
        Old beliefs become less reliable.
        """
        belief.belief_age += timesteps
        
        # Confidence decays toward prior
        decay_factor = np.exp(-self.decay_rate * timesteps)
        belief.confidence_score = max(
            belief.confidence_score * decay_factor,
            self.min_confidence
        )
        belief.confidence = self._confidence_to_level(belief.confidence_score)
        
        # Uncertainty bounds widen
        belief.pd_upper = min(belief.pd_upper * (1 + 0.01 * timesteps), 0.5)
        belief.pd_lower = max(belief.pd_lower * (1 - 0.01 * timesteps), 0.001)
        
        return belief
    
    def _compute_confidence(self, sample_size: int) -> float:
        """Compute confidence from sample size."""
        # Saturating function
        conf = 1.0 - np.exp(-0.1 * sample_size)
        return np.clip(conf, self.min_confidence, self.max_confidence)
    
    def _confidence_to_level(self, score: float) -> BeliefConfidence:
        """Convert numeric confidence to level."""
        if score < 0.1:
            return BeliefConfidence.NONE
        elif score < 0.25:
            return BeliefConfidence.VERY_LOW
        elif score < 0.4:
            return BeliefConfidence.LOW
        elif score < 0.6:
            return BeliefConfidence.MEDIUM
        elif score < 0.8:
            return BeliefConfidence.HIGH
        else:
            return BeliefConfidence.VERY_HIGH
    
    def _update_uncertainty_bounds(self, belief: CounterpartyBelief) -> None:
        """Update uncertainty bounds based on confidence."""
        # Bounds tighten with more confidence
        uncertainty = 1.0 - belief.confidence_score
        
        base_range = 0.15 * uncertainty + 0.02  # Minimum range of 2%
        
        belief.pd_lower = max(belief.estimated_pd - base_range, 0.001)
        belief.pd_upper = min(belief.estimated_pd + base_range * 2, 0.5)


class BeliefSystem:
    """
    Complete belief system for a single bank.
    Manages beliefs about all counterparties.
    """
    
    def __init__(self,
                 bank_id: int,
                 private_memory: PrivateMemory,
                 update_rule: Optional[BeliefUpdateRule] = None,
                 market_prior: Optional[MarketPrior] = None):
        self.bank_id = bank_id
        self.memory = private_memory
        self.update_rule = update_rule or BeliefUpdateRule()
        self.market_prior = market_prior or MarketPrior()
        
        # Beliefs about each counterparty
        self.beliefs: Dict[int, CounterpartyBelief] = {}
        
        # Belief history for tracking evolution
        self.belief_history: Dict[int, List[Tuple[int, CounterpartyBelief]]] = {}
        
        # Current timestep
        self.current_timestep = 0
    
    def get_belief(self, counterparty_id: int) -> CounterpartyBelief:
        """
        Get current belief about a counterparty.
        Creates prior belief if none exists.
        """
        if counterparty_id not in self.beliefs:
            self.beliefs[counterparty_id] = self._initialize_belief(counterparty_id)
        return self.beliefs[counterparty_id]
    
    def _initialize_belief(self, counterparty_id: int) -> CounterpartyBelief:
        """
        Initialize belief for a new counterparty.
        Uses memory if available, otherwise market prior.
        """
        # Check if we have memory
        memory_features = self.memory.compute_memory_features(counterparty_id)
        
        if memory_features['has_history'] > 0:
            # Build belief from memory
            return self._belief_from_memory(counterparty_id, memory_features)
        else:
            # Use market prior
            return self.market_prior.to_belief(counterparty_id)
    
    def _belief_from_memory(self, counterparty_id: int,
                            features: Dict[str, float]) -> CounterpartyBelief:
        """Build belief from memory features."""
        # Start with prior
        belief = self.market_prior.to_belief(counterparty_id)
        
        # Blend with observed data
        weight = features['interaction_count']  # 0 to 1
        
        belief.estimated_pd = (
            (1 - weight) * belief.estimated_pd +
            weight * features['observed_default_rate']
        )
        
        belief.estimated_delay_prob = (
            (1 - weight) * belief.estimated_delay_prob +
            weight * features['average_delay']
        )
        
        belief.reliability_score = features['quality_score']
        belief.expected_profit_rate = features['profit_ratio']
        
        # Set confidence based on history
        belief.sample_size = int(features['interaction_count'] * 20)
        belief.confidence_score = min(features['interaction_count'], 0.9)
        belief.confidence = self.update_rule._confidence_to_level(belief.confidence_score)
        
        return belief
    
    def update_from_interaction(self,
                                counterparty_id: int,
                                default_occurred: bool = False,
                                delay_occurred: bool = False,
                                profit_loss: float = 0.0,
                                amount: float = 0.0) -> CounterpartyBelief:
        """
        Update belief after an interaction outcome.
        This is the primary learning mechanism.
        """
        belief = self.get_belief(counterparty_id)
        
        # Apply update rule
        updated = self.update_rule.update_from_outcome(
            belief=belief,
            default_occurred=default_occurred,
            delay_occurred=delay_occurred,
            profit_loss=profit_loss,
            amount=amount
        )
        
        updated.last_updated = self.current_timestep
        
        # Store in beliefs
        self.beliefs[counterparty_id] = updated
        
        # Record history
        self._record_belief_history(counterparty_id, updated)
        
        return updated
    
    def _record_belief_history(self, counterparty_id: int, 
                               belief: CounterpartyBelief) -> None:
        """Record belief state for tracking evolution."""
        if counterparty_id not in self.belief_history:
            self.belief_history[counterparty_id] = []
        
        # Store snapshot
        self.belief_history[counterparty_id].append(
            (self.current_timestep, CounterpartyBelief(**asdict(belief)))
        )
        
        # Keep limited history
        if len(self.belief_history[counterparty_id]) > 100:
            self.belief_history[counterparty_id] = \
                self.belief_history[counterparty_id][-100:]
    
    def advance_timestep(self, new_timestep: int) -> None:
        """Advance time and decay beliefs."""
        timesteps_passed = new_timestep - self.current_timestep
        self.current_timestep = new_timestep
        
        # Decay all beliefs
        for cp_id, belief in self.beliefs.items():
            self.update_rule.decay_belief(belief, timesteps_passed)
    
    def update_market_prior(self,
                            average_pd: Optional[float] = None,
                            market_stress: Optional[float] = None,
                            ccp_stress: Optional[float] = None,
                            exchange_congestion: Optional[float] = None) -> None:
        """Update market-level priors."""
        if average_pd is not None:
            self.market_prior.average_pd = average_pd
        if market_stress is not None:
            self.market_prior.market_stress = market_stress
        if ccp_stress is not None:
            self.market_prior.ccp_stress_signal = ccp_stress
        if exchange_congestion is not None:
            self.market_prior.exchange_congestion = exchange_congestion
    
    def get_all_beliefs(self) -> Dict[int, CounterpartyBelief]:
        """Get beliefs about all known counterparties."""
        return self.beliefs.copy()
    
    def get_risky_counterparties(self, pd_threshold: float = 0.1) -> List[int]:
        """Get counterparties believed to be risky."""
        return [cp_id for cp_id, belief in self.beliefs.items()
                if belief.estimated_pd >= pd_threshold]
    
    def get_trusted_counterparties(self, trust_threshold: float = 0.7) -> List[int]:
        """Get counterparties with high trust."""
        return [cp_id for cp_id, belief in self.beliefs.items()
                if belief.trust_score >= trust_threshold]
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary statistics about beliefs."""
        if not self.beliefs:
            return {'num_counterparties': 0}
        
        pds = [b.estimated_pd for b in self.beliefs.values()]
        trusts = [b.trust_score for b in self.beliefs.values()]
        confidences = [b.confidence_score for b in self.beliefs.values()]
        
        return {
            'num_counterparties': len(self.beliefs),
            'average_estimated_pd': np.mean(pds),
            'max_estimated_pd': np.max(pds),
            'average_trust': np.mean(trusts),
            'average_confidence': np.mean(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.3),
            'high_risk_count': sum(1 for p in pds if p > 0.1)
        }
    
    def get_belief_vector(self, counterparty_id: int) -> np.ndarray:
        """Get belief as feature vector for observations."""
        belief = self.get_belief(counterparty_id)
        return belief.to_feature_vector()
    
    def compute_expected_loss(self, counterparty_id: int, 
                              exposure: float) -> float:
        """Compute expected loss from exposure to counterparty."""
        belief = self.get_belief(counterparty_id)
        return belief.estimated_pd * belief.estimated_lgd * exposure
    
    def should_interact(self, counterparty_id: int,
                        risk_tolerance: float = 0.1) -> bool:
        """
        Recommend whether to interact with counterparty.
        Based on risk-adjusted beliefs.
        """
        belief = self.get_belief(counterparty_id)
        risk_adjusted_pd = belief.get_risk_adjusted_pd(risk_aversion=1.5)
        
        return risk_adjusted_pd <= risk_tolerance and belief.trust_score >= 0.3
    
    def get_belief_evolution(self, counterparty_id: int) -> List[Dict[str, Any]]:
        """Get history of belief evolution for a counterparty."""
        if counterparty_id not in self.belief_history:
            return []
        
        return [
            {
                'timestep': t,
                'estimated_pd': b.estimated_pd,
                'trust_score': b.trust_score,
                'confidence_score': b.confidence_score
            }
            for t, b in self.belief_history[counterparty_id]
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize belief system."""
        return {
            'bank_id': self.bank_id,
            'current_timestep': self.current_timestep,
            'beliefs': {
                str(cp_id): belief.to_dict()
                for cp_id, belief in self.beliefs.items()
            },
            'summary': self.get_belief_summary()
        }


class BeliefManager:
    """
    Manages belief systems for all banks.
    Ensures information isolation.
    """
    
    def __init__(self, memory_manager: Any):
        self.memory_manager = memory_manager
        self.belief_systems: Dict[int, BeliefSystem] = {}
        self.market_prior = MarketPrior()
    
    def get_or_create_belief_system(self, bank_id: int) -> BeliefSystem:
        """Get or create belief system for a bank."""
        if bank_id not in self.belief_systems:
            memory = self.memory_manager.get_or_create_memory(bank_id)
            self.belief_systems[bank_id] = BeliefSystem(
                bank_id=bank_id,
                private_memory=memory,
                market_prior=self.market_prior
            )
        return self.belief_systems[bank_id]
    
    def update_market_prior(self, **kwargs) -> None:
        """Update market prior for all belief systems."""
        for key, value in kwargs.items():
            if hasattr(self.market_prior, key):
                setattr(self.market_prior, key, value)
        
        # Propagate to all belief systems
        for system in self.belief_systems.values():
            system.update_market_prior(**kwargs)
    
    def advance_all(self, new_timestep: int) -> None:
        """Advance all belief systems to new timestep."""
        for system in self.belief_systems.values():
            system.advance_timestep(new_timestep)
    
    def get_all_belief_summaries(self) -> Dict[int, Dict[str, Any]]:
        """Get belief summaries for all banks."""
        return {
            bank_id: system.get_belief_summary()
            for bank_id, system in self.belief_systems.items()
        }
