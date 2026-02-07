"""
Belief-Integrated Observations and Decision Filter for FinSim-MAPPO.
Extends observations with belief features and filters decisions through beliefs.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .beliefs import BeliefSystem, CounterpartyBelief, BeliefConfidence
from .memory import PrivateMemory


class ObservationContext(str, Enum):
    """Context for observation generation."""
    GENERAL = "general"              # General market observation
    COUNTERPARTY = "counterparty"    # Specific counterparty interaction
    EXCHANGE = "exchange"            # Exchange-specific
    CCP = "ccp"                      # CCP-specific


@dataclass
class BeliefObservation:
    """
    Extended observation that includes belief features.
    This is what the MAPPO agent actually sees.
    """
    # Base observation (existing features)
    base_observation: np.ndarray
    
    # Belief-derived features
    counterparty_belief_vector: Optional[np.ndarray] = None
    self_risk_perception: np.ndarray = field(default_factory=lambda: np.zeros(5))
    market_belief_summary: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    # Context
    context: ObservationContext = ObservationContext.GENERAL
    target_counterparty: Optional[int] = None
    
    def to_array(self) -> np.ndarray:
        """Flatten to single observation vector."""
        components = [self.base_observation]
        
        if self.counterparty_belief_vector is not None:
            components.append(self.counterparty_belief_vector)
        
        components.append(self.self_risk_perception)
        components.append(self.market_belief_summary)
        
        return np.concatenate(components)


class BeliefObservationBuilder:
    """
    Builds belief-augmented observations for agents.
    MAPPO never sees "true" risk - only perceived risk through beliefs.
    """
    
    def __init__(self, 
                 belief_system: BeliefSystem,
                 base_obs_dim: int = 20,
                 belief_dim: int = 10,
                 include_uncertainty: bool = True):
        self.belief_system = belief_system
        self.base_obs_dim = base_obs_dim
        self.belief_dim = belief_dim
        self.include_uncertainty = include_uncertainty
    
    @property
    def total_observation_dim(self) -> int:
        """Total dimension of belief-augmented observation."""
        # Base + counterparty belief + self perception + market summary
        return self.base_obs_dim + self.belief_dim + 5 + 5
    
    def build_observation(self,
                          base_obs: np.ndarray,
                          context: ObservationContext = ObservationContext.GENERAL,
                          target_counterparty: Optional[int] = None,
                          own_state: Optional[Dict[str, float]] = None) -> BeliefObservation:
        """
        Build complete belief-augmented observation.
        """
        obs = BeliefObservation(
            base_observation=base_obs,
            context=context,
            target_counterparty=target_counterparty
        )
        
        # Add counterparty belief if relevant
        if target_counterparty is not None:
            belief = self.belief_system.get_belief(target_counterparty)
            obs.counterparty_belief_vector = self._belief_to_features(belief)
        else:
            # Pad with neutral beliefs
            obs.counterparty_belief_vector = self._neutral_belief_features()
        
        # Self-risk perception
        obs.self_risk_perception = self._compute_self_perception(own_state)
        
        # Market belief summary
        obs.market_belief_summary = self._compute_market_summary()
        
        return obs
    
    def _belief_to_features(self, belief: CounterpartyBelief) -> np.ndarray:
        """Convert belief to feature vector."""
        features = belief.to_feature_vector()
        
        if self.include_uncertainty:
            # Add uncertainty features
            uncertainty = np.array([
                belief.pd_upper - belief.pd_lower,  # Uncertainty range
                1.0 - belief.confidence_score,       # Uncertainty level
            ])
            features = np.concatenate([features, uncertainty])
        
        # Pad or truncate to belief_dim
        if len(features) < self.belief_dim:
            features = np.pad(features, (0, self.belief_dim - len(features)))
        elif len(features) > self.belief_dim:
            features = features[:self.belief_dim]
        
        return features
    
    def _neutral_belief_features(self) -> np.ndarray:
        """Generate neutral belief features when no counterparty specified."""
        return np.array([
            0.05,   # Neutral PD
            0.15,   # Neutral delay prob
            0.45,   # Neutral LGD
            0.0,    # Neutral profit
            0.0,    # Neutral margin stress
            0.5,    # Neutral reliability
            0.5,    # Neutral trust
            0.0,    # No confidence
            0.0,    # No history
            0.0     # No recency
        ])[:self.belief_dim]
    
    def _compute_self_perception(self, 
                                  own_state: Optional[Dict[str, float]]) -> np.ndarray:
        """Compute bank's perception of its own risk."""
        if own_state is None:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        capital_health = np.clip(own_state.get('capital_ratio', 0.1) / 0.15, 0, 1)
        liquidity_health = np.clip(own_state.get('liquidity_ratio', 0.1) / 0.15, 0, 1)
        leverage_concern = 1.0 - np.clip(own_state.get('leverage', 10) / 30, 0, 1)
        stress_level = own_state.get('stress', 0.0)
        exposure_concern = own_state.get('exposure_concentration', 0.0)
        
        return np.array([
            capital_health,
            liquidity_health,
            leverage_concern,
            stress_level,
            exposure_concern
        ])
    
    def _compute_market_summary(self) -> np.ndarray:
        """Compute summary of beliefs about the market."""
        summary = self.belief_system.get_belief_summary()
        
        return np.array([
            summary.get('average_estimated_pd', 0.05),
            summary.get('average_trust', 0.5),
            summary.get('average_confidence', 0.5),
            summary.get('high_risk_count', 0) / max(summary.get('num_counterparties', 1), 1),
            summary.get('low_confidence_count', 0) / max(summary.get('num_counterparties', 1), 1)
        ])


@dataclass
class TransactionTerms:
    """Adjustable transaction terms."""
    interest_rate: float = 0.05
    collateral_ratio: float = 0.0
    exposure_limit: float = float('inf')
    maturity: int = 1
    
    # Confidence in offering these terms
    confidence: float = 0.5


class BeliefFilter:
    """
    Filters decisions through belief system.
    Transforms raw decisions into belief-adjusted actions.
    """
    
    def __init__(self,
                 belief_system: BeliefSystem,
                 risk_aversion: float = 1.0,
                 min_trust_threshold: float = 0.2,
                 max_pd_threshold: float = 0.3):
        self.belief_system = belief_system
        self.risk_aversion = risk_aversion
        self.min_trust = min_trust_threshold
        self.max_pd = max_pd_threshold
    
    def filter_action(self,
                      raw_action: int,
                      counterparty_id: int,
                      action_context: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Filter action through beliefs.
        May modify or block action based on beliefs.
        
        Returns:
            (filtered_action, action_metadata)
        """
        belief = self.belief_system.get_belief(counterparty_id)
        
        # Get risk-adjusted assessment
        risk_pd = belief.get_risk_adjusted_pd(self.risk_aversion)
        
        metadata = {
            'original_action': raw_action,
            'belief_pd': belief.estimated_pd,
            'risk_adjusted_pd': risk_pd,
            'trust': belief.trust_score,
            'confidence': belief.confidence_score,
            'modified': False
        }
        
        # Check if action should be blocked
        if self._should_block(belief, risk_pd):
            metadata['modified'] = True
            metadata['reason'] = 'risk_threshold_exceeded'
            return 0, metadata  # Convert to safe action
        
        # Action is allowed but may have modified terms
        metadata['suggested_terms'] = self.compute_terms(belief, action_context)
        
        return raw_action, metadata
    
    def _should_block(self, belief: CounterpartyBelief, risk_pd: float) -> bool:
        """Determine if action should be blocked."""
        # Block if trust too low
        if belief.trust_score < self.min_trust:
            return True
        
        # Block if risk too high
        if risk_pd > self.max_pd:
            return True
        
        return False
    
    def compute_terms(self, 
                      belief: CounterpartyBelief,
                      context: Dict[str, Any]) -> TransactionTerms:
        """
        Compute risk-adjusted transaction terms.
        Higher risk -> higher interest, more collateral.
        """
        base_rate = context.get('base_interest_rate', 0.05)
        base_amount = context.get('amount', 1e6)
        
        # Risk premium
        risk_premium = belief.estimated_pd * 2 + belief.estimated_lgd * 0.5
        
        # Uncertainty premium
        uncertainty_premium = (1 - belief.confidence_score) * 0.02
        
        # Trust discount
        trust_discount = belief.trust_score * 0.01
        
        adjusted_rate = base_rate + risk_premium + uncertainty_premium - trust_discount
        adjusted_rate = max(adjusted_rate, 0.01)  # Minimum rate
        
        # Collateral requirement
        if belief.estimated_pd > 0.1:
            collateral_ratio = 0.3 + belief.estimated_pd
        elif belief.estimated_pd > 0.05:
            collateral_ratio = 0.1 + belief.estimated_pd
        else:
            collateral_ratio = 0.0
        
        # Exposure limit based on confidence
        if belief.confidence_score < 0.3:
            exposure_limit = base_amount * 0.5  # Limit exposure when uncertain
        elif belief.confidence_score < 0.6:
            exposure_limit = base_amount * 0.8
        else:
            exposure_limit = base_amount * 1.2  # Allow more when confident
        
        return TransactionTerms(
            interest_rate=adjusted_rate,
            collateral_ratio=collateral_ratio,
            exposure_limit=exposure_limit,
            confidence=belief.confidence_score
        )
    
    def evaluate_opportunity(self,
                             counterparty_id: int,
                             opportunity_value: float,
                             exposure: float) -> Dict[str, Any]:
        """
        Evaluate an opportunity through belief lens.
        Returns risk-adjusted assessment.
        """
        belief = self.belief_system.get_belief(counterparty_id)
        
        # Expected loss
        expected_loss = self.belief_system.compute_expected_loss(counterparty_id, exposure)
        
        # Risk-adjusted value
        risk_pd = belief.get_risk_adjusted_pd(self.risk_aversion)
        risk_adjusted_value = opportunity_value * (1 - risk_pd) - expected_loss
        
        # Confidence-weighted value
        confidence_adjusted_value = (
            risk_adjusted_value * belief.confidence_score +
            opportunity_value * 0.3 * (1 - belief.confidence_score)  # Conservative when uncertain
        )
        
        return {
            'raw_value': opportunity_value,
            'expected_loss': expected_loss,
            'risk_adjusted_value': risk_adjusted_value,
            'confidence_adjusted_value': confidence_adjusted_value,
            'recommended': confidence_adjusted_value > 0 and belief.trust_score > self.min_trust,
            'counterparty_risk': {
                'pd': belief.estimated_pd,
                'risk_pd': risk_pd,
                'trust': belief.trust_score,
                'confidence': belief.confidence_score
            }
        }


class AdaptiveTermsEngine:
    """
    Engine for computing adaptive transaction terms.
    Interest, collateral, and exposure adapt to beliefs.
    """
    
    def __init__(self,
                 belief_system: BeliefSystem,
                 base_interest_rate: float = 0.05,
                 base_collateral: float = 0.0,
                 risk_sensitivity: float = 1.0):
        self.belief_system = belief_system
        self.base_rate = base_interest_rate
        self.base_collateral = base_collateral
        self.risk_sensitivity = risk_sensitivity
        
        # Term history for learning
        self.term_history: List[Dict[str, Any]] = []
    
    def compute_loan_terms(self,
                           counterparty_id: int,
                           amount: float,
                           duration: int = 1) -> TransactionTerms:
        """Compute terms for a loan."""
        belief = self.belief_system.get_belief(counterparty_id)
        
        # Interest rate components
        risk_free_rate = self.base_rate
        credit_spread = belief.estimated_pd * belief.estimated_lgd * 2
        uncertainty_spread = (1 - belief.confidence_score) * 0.02 * self.risk_sensitivity
        duration_spread = duration * 0.005
        
        interest_rate = risk_free_rate + credit_spread + uncertainty_spread + duration_spread
        
        # Trust-based discount
        if belief.trust_score > 0.7:
            interest_rate *= 0.9  # 10% discount for trusted counterparties
        
        # Collateral requirement
        base_collateral = self.base_collateral
        
        if belief.estimated_pd > 0.1:
            collateral_ratio = 0.5  # 50% collateral for high risk
        elif belief.estimated_pd > 0.05:
            collateral_ratio = 0.25
        elif belief.confidence_score < 0.3:
            collateral_ratio = 0.2  # Extra collateral when uncertain
        else:
            collateral_ratio = base_collateral
        
        # Exposure limit
        max_exposure = amount
        if belief.confidence_score < 0.5:
            max_exposure = amount * belief.confidence_score * 2
        
        return TransactionTerms(
            interest_rate=interest_rate,
            collateral_ratio=collateral_ratio,
            exposure_limit=max_exposure,
            maturity=duration,
            confidence=belief.confidence_score
        )
    
    def compute_trading_terms(self,
                              counterparty_id: int,
                              notional: float) -> Dict[str, float]:
        """Compute terms for trading."""
        belief = self.belief_system.get_belief(counterparty_id)
        
        # Margin requirement
        base_margin = 0.03
        risk_addon = belief.estimated_pd * 0.1
        uncertainty_addon = (1 - belief.confidence_score) * 0.02
        
        margin = base_margin + risk_addon + uncertainty_addon
        
        # Position limit
        if belief.trust_score < 0.5:
            position_limit = notional * 0.5
        else:
            position_limit = notional * (0.5 + belief.trust_score * 0.5)
        
        return {
            'margin_requirement': margin,
            'position_limit': position_limit,
            'haircut': belief.estimated_lgd * 0.5,
            'netting_allowed': belief.trust_score > 0.6
        }
    
    def record_term_outcome(self,
                            counterparty_id: int,
                            terms: TransactionTerms,
                            outcome: Dict[str, Any]) -> None:
        """Record outcome for term learning."""
        self.term_history.append({
            'counterparty_id': counterparty_id,
            'terms': terms,
            'outcome': outcome,
            'timestamp': len(self.term_history)
        })
        
        # Keep limited history
        if len(self.term_history) > 1000:
            self.term_history = self.term_history[-1000:]


class BeliefIntegratedObserver:
    """
    Complete observer that integrates beliefs into agent observations.
    This is the main interface for MAPPO agents.
    """
    
    def __init__(self,
                 bank_id: int,
                 belief_system: BeliefSystem,
                 base_obs_dim: int = 20):
        self.bank_id = bank_id
        self.belief_system = belief_system
        self.obs_builder = BeliefObservationBuilder(
            belief_system=belief_system,
            base_obs_dim=base_obs_dim
        )
        self.belief_filter = BeliefFilter(belief_system=belief_system)
        self.terms_engine = AdaptiveTermsEngine(belief_system=belief_system)
    
    def get_observation(self,
                        base_obs: np.ndarray,
                        target_counterparty: Optional[int] = None,
                        own_state: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Get belief-augmented observation for MAPPO.
        This is what the policy network sees.
        """
        context = (ObservationContext.COUNTERPARTY 
                  if target_counterparty else ObservationContext.GENERAL)
        
        belief_obs = self.obs_builder.build_observation(
            base_obs=base_obs,
            context=context,
            target_counterparty=target_counterparty,
            own_state=own_state
        )
        
        return belief_obs.to_array()
    
    def filter_action(self,
                      action: int,
                      counterparty_id: int,
                      context: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Filter action through beliefs."""
        return self.belief_filter.filter_action(action, counterparty_id, context)
    
    def get_terms(self,
                  counterparty_id: int,
                  amount: float,
                  transaction_type: str = 'loan') -> TransactionTerms:
        """Get belief-adjusted transaction terms."""
        if transaction_type == 'loan':
            return self.terms_engine.compute_loan_terms(counterparty_id, amount)
        else:
            trading_terms = self.terms_engine.compute_trading_terms(counterparty_id, amount)
            return TransactionTerms(
                interest_rate=0.0,
                collateral_ratio=trading_terms['margin_requirement'],
                exposure_limit=trading_terms['position_limit']
            )
    
    @property
    def observation_dim(self) -> int:
        """Total observation dimension."""
        return self.obs_builder.total_observation_dim
