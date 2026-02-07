"""
Belief-Conditioned Counterfactual Engine for FinSim-MAPPO.
Runs counterfactual simulations using beliefs, not objective reality.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

from .beliefs import BeliefSystem, CounterpartyBelief
from .memory import PrivateMemory


@dataclass
class BeliefPrediction:
    """A prediction made based on beliefs."""
    prediction_id: str
    bank_id: int
    counterparty_id: int
    timestep: int
    
    # Predicted outcomes
    predicted_pd: float
    predicted_delay_prob: float
    predicted_profit_rate: float
    predicted_default: bool
    
    # Belief state at prediction time
    belief_confidence: float
    belief_sample_size: int
    
    # Actual outcomes (filled in when known)
    actual_default: Optional[bool] = None
    actual_delay: Optional[bool] = None
    actual_profit_rate: Optional[float] = None
    
    # Accuracy metrics
    pd_error: Optional[float] = None
    calibration_contribution: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def compute_accuracy(self) -> Optional[float]:
        """Compute accuracy once actual outcomes are known."""
        if self.actual_default is None:
            return None
        
        # Binary prediction accuracy
        predicted_default = self.predicted_pd > 0.5
        correct = predicted_default == self.actual_default
        
        return 1.0 if correct else 0.0
    
    def compute_calibration_error(self) -> Optional[float]:
        """Compute calibration error (predicted prob vs actual outcome)."""
        if self.actual_default is None:
            return None
        
        actual = 1.0 if self.actual_default else 0.0
        return abs(self.predicted_pd - actual)


@dataclass
class BeliefCounterfactualResult:
    """Result from belief-conditioned counterfactual analysis."""
    scenario_id: str
    bank_id: int
    timestep: int
    
    # What the bank believed would happen
    believed_outcome: Dict[str, float]
    
    # What actually happened (in simulation)
    actual_outcome: Dict[str, float]
    
    # Belief accuracy
    belief_accuracy: Dict[str, float]
    
    # Risk assessment
    perceived_risk: float
    actual_risk: float
    risk_perception_error: float
    
    # Decision quality
    decision_optimal: bool
    expected_utility: float
    actual_utility: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeliefSimulator:
    """
    Simulates outcomes based on a bank's beliefs.
    Used for counterfactual what-if analysis from the bank's perspective.
    """
    
    def __init__(self, belief_system: BeliefSystem):
        self.belief_system = belief_system
    
    def simulate_transaction_outcome(self,
                                      counterparty_id: int,
                                      amount: float,
                                      num_samples: int = 100) -> Dict[str, Any]:
        """
        Simulate transaction outcomes using bank's beliefs.
        Returns distribution of outcomes from bank's perspective.
        """
        belief = self.belief_system.get_belief(counterparty_id)
        
        # Sample defaults
        defaults = np.random.binomial(1, belief.estimated_pd, num_samples)
        
        # Sample delays (conditional on no default)
        delays = np.random.binomial(1, belief.estimated_delay_prob, num_samples)
        delays = delays * (1 - defaults)  # No delay if defaulted
        
        # Compute outcomes
        outcomes = []
        for i in range(num_samples):
            if defaults[i]:
                # Default outcome
                loss = amount * belief.estimated_lgd
                outcomes.append({
                    'default': True,
                    'delay': False,
                    'profit': -loss,
                    'recovery': amount * (1 - belief.estimated_lgd)
                })
            elif delays[i]:
                # Delayed payment
                delay_cost = amount * 0.01  # 1% delay cost
                outcomes.append({
                    'default': False,
                    'delay': True,
                    'profit': amount * belief.expected_profit_rate - delay_cost,
                    'recovery': amount
                })
            else:
                # Normal outcome
                outcomes.append({
                    'default': False,
                    'delay': False,
                    'profit': amount * belief.expected_profit_rate,
                    'recovery': 0
                })
        
        # Aggregate statistics
        return {
            'expected_default_rate': np.mean(defaults),
            'expected_delay_rate': np.mean([o['delay'] for o in outcomes]),
            'expected_profit': np.mean([o['profit'] for o in outcomes]),
            'profit_std': np.std([o['profit'] for o in outcomes]),
            'expected_loss': np.mean([max(-o['profit'], 0) for o in outcomes]),
            'var_95': np.percentile([o['profit'] for o in outcomes], 5),
            'var_99': np.percentile([o['profit'] for o in outcomes], 1),
            'belief_confidence': belief.confidence_score,
            'samples': num_samples
        }
    
    def simulate_portfolio_outcome(self,
                                   portfolio: Dict[int, float],
                                   num_samples: int = 100,
                                   correlation: float = 0.3) -> Dict[str, Any]:
        """
        Simulate portfolio outcomes with correlation.
        Portfolio: {counterparty_id: exposure}
        """
        counterparty_ids = list(portfolio.keys())
        n_counterparties = len(counterparty_ids)
        
        if n_counterparties == 0:
            return {'expected_loss': 0, 'portfolio_var': 0}
        
        # Get beliefs for all counterparties
        pds = np.array([self.belief_system.get_belief(cp).estimated_pd 
                       for cp in counterparty_ids])
        lgds = np.array([self.belief_system.get_belief(cp).estimated_lgd 
                        for cp in counterparty_ids])
        exposures = np.array([portfolio[cp] for cp in counterparty_ids])
        
        # Generate correlated defaults using Gaussian copula
        losses = []
        for _ in range(num_samples):
            # Correlated normals
            cov = np.full((n_counterparties, n_counterparties), correlation)
            np.fill_diagonal(cov, 1.0)
            z = np.random.multivariate_normal(np.zeros(n_counterparties), cov)
            
            # Transform to uniform
            from scipy.stats import norm
            u = norm.cdf(z)
            
            # Default if u < pd
            defaults = (u < pds).astype(float)
            
            # Compute loss
            loss = np.sum(defaults * lgds * exposures)
            losses.append(loss)
        
        return {
            'expected_loss': np.mean(losses),
            'loss_std': np.std(losses),
            'var_95': np.percentile(losses, 95),
            'var_99': np.percentile(losses, 99),
            'max_loss': np.max(losses),
            'loss_given_any_default': np.mean([l for l in losses if l > 0]) if any(l > 0 for l in losses) else 0,
            'no_loss_probability': sum(1 for l in losses if l == 0) / num_samples
        }


class BeliefVsRealityTracker:
    """
    Tracks belief predictions against actual outcomes.
    Enables meta-learning and calibration.
    """
    
    def __init__(self, bank_id: int):
        self.bank_id = bank_id
        self.predictions: List[BeliefPrediction] = []
        self._prediction_counter = 0
    
    def record_prediction(self,
                          counterparty_id: int,
                          belief: CounterpartyBelief,
                          timestep: int) -> str:
        """Record a prediction for later evaluation."""
        self._prediction_counter += 1
        pred_id = f"pred_{self.bank_id}_{timestep}_{self._prediction_counter}"
        
        prediction = BeliefPrediction(
            prediction_id=pred_id,
            bank_id=self.bank_id,
            counterparty_id=counterparty_id,
            timestep=timestep,
            predicted_pd=belief.estimated_pd,
            predicted_delay_prob=belief.estimated_delay_prob,
            predicted_profit_rate=belief.expected_profit_rate,
            predicted_default=belief.estimated_pd > 0.5,
            belief_confidence=belief.confidence_score,
            belief_sample_size=belief.sample_size
        )
        
        self.predictions.append(prediction)
        return pred_id
    
    def record_outcome(self,
                       prediction_id: str,
                       actual_default: bool,
                       actual_delay: bool = False,
                       actual_profit_rate: float = 0.0) -> Optional[BeliefPrediction]:
        """Record actual outcome for a prediction."""
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.actual_default = actual_default
                pred.actual_delay = actual_delay
                pred.actual_profit_rate = actual_profit_rate
                pred.pd_error = pred.compute_calibration_error()
                return pred
        return None
    
    def get_calibration_metrics(self) -> Dict[str, float]:
        """Compute belief calibration metrics."""
        completed = [p for p in self.predictions if p.actual_default is not None]
        
        if not completed:
            return {
                'calibration_error': 0.0,
                'accuracy': 0.0,
                'sample_count': 0,
                'overconfidence_rate': 0.0
            }
        
        # Mean calibration error
        calibration_errors = [p.compute_calibration_error() for p in completed]
        mean_calibration = np.mean(calibration_errors)
        
        # Binary accuracy
        accuracies = [p.compute_accuracy() for p in completed]
        mean_accuracy = np.mean(accuracies)
        
        # Overconfidence: predicted low risk but default occurred
        overconfident = sum(1 for p in completed 
                          if p.predicted_pd < 0.1 and p.actual_default)
        overconfidence_rate = overconfident / len(completed)
        
        # Underconfidence: predicted high risk but no default
        underconfident = sum(1 for p in completed
                            if p.predicted_pd > 0.5 and not p.actual_default)
        underconfidence_rate = underconfident / len(completed)
        
        # Brier score
        brier = np.mean([(p.predicted_pd - (1.0 if p.actual_default else 0.0))**2 
                        for p in completed])
        
        return {
            'calibration_error': mean_calibration,
            'accuracy': mean_accuracy,
            'brier_score': brier,
            'sample_count': len(completed),
            'overconfidence_rate': overconfidence_rate,
            'underconfidence_rate': underconfidence_rate
        }
    
    def get_calibration_by_confidence(self) -> Dict[str, Dict[str, float]]:
        """Get calibration metrics bucketed by confidence level."""
        completed = [p for p in self.predictions if p.actual_default is not None]
        
        buckets = {
            'low': [p for p in completed if p.belief_confidence < 0.3],
            'medium': [p for p in completed if 0.3 <= p.belief_confidence < 0.7],
            'high': [p for p in completed if p.belief_confidence >= 0.7]
        }
        
        results = {}
        for bucket_name, preds in buckets.items():
            if preds:
                results[bucket_name] = {
                    'count': len(preds),
                    'accuracy': np.mean([p.compute_accuracy() for p in preds]),
                    'calibration_error': np.mean([p.compute_calibration_error() for p in preds])
                }
            else:
                results[bucket_name] = {'count': 0, 'accuracy': 0, 'calibration_error': 0}
        
        return results


class BeliefCounterfactualEngine:
    """
    Runs counterfactual analyses conditioned on beliefs.
    Banks make decisions based on their beliefs, not reality.
    """
    
    def __init__(self,
                 belief_system: BeliefSystem,
                 private_memory: PrivateMemory):
        self.belief_system = belief_system
        self.memory = private_memory
        self.simulator = BeliefSimulator(belief_system)
        self.tracker = BeliefVsRealityTracker(belief_system.bank_id)
    
    def analyze_decision(self,
                         counterparty_id: int,
                         action_type: str,
                         amount: float,
                         actual_outcome: Optional[Dict[str, Any]] = None) -> BeliefCounterfactualResult:
        """
        Analyze a decision from belief perspective.
        Compare what bank believed would happen vs what actually happened.
        """
        belief = self.belief_system.get_belief(counterparty_id)
        
        # What the bank believed would happen
        believed_outcome = self.simulator.simulate_transaction_outcome(
            counterparty_id, amount, num_samples=100
        )
        
        # Record prediction
        pred_id = self.tracker.record_prediction(
            counterparty_id, belief, self.belief_system.current_timestep
        )
        
        # If we have actual outcome, compare
        if actual_outcome:
            self.tracker.record_outcome(
                pred_id,
                actual_default=actual_outcome.get('default', False),
                actual_delay=actual_outcome.get('delay', False),
                actual_profit_rate=actual_outcome.get('profit_rate', 0)
            )
            
            # Compute accuracy
            belief_accuracy = {
                'pd_error': abs(belief.estimated_pd - 
                              (1.0 if actual_outcome.get('default') else 0.0)),
                'profit_error': abs(believed_outcome['expected_profit'] - 
                                   actual_outcome.get('profit', 0)),
            }
            
            actual_risk = 1.0 if actual_outcome.get('default') else 0.0
        else:
            belief_accuracy = {'pd_error': None, 'profit_error': None}
            actual_risk = belief.estimated_pd  # Use belief as stand-in
            actual_outcome = believed_outcome
        
        # Risk perception error
        perceived_risk = belief.estimated_pd
        risk_error = abs(perceived_risk - actual_risk)
        
        # Decision optimality (would a rational agent have made same decision?)
        if action_type == 'lend':
            # Lending is optimal if expected profit > expected loss
            expected_profit = believed_outcome['expected_profit']
            decision_optimal = expected_profit > 0
        else:
            decision_optimal = True  # Default assumption
        
        return BeliefCounterfactualResult(
            scenario_id=f"bf_cf_{self.belief_system.bank_id}_{counterparty_id}",
            bank_id=self.belief_system.bank_id,
            timestep=self.belief_system.current_timestep,
            believed_outcome=believed_outcome,
            actual_outcome=actual_outcome,
            belief_accuracy=belief_accuracy,
            perceived_risk=perceived_risk,
            actual_risk=actual_risk,
            risk_perception_error=risk_error,
            decision_optimal=decision_optimal,
            expected_utility=believed_outcome['expected_profit'],
            actual_utility=actual_outcome.get('profit', believed_outcome['expected_profit'])
        )
    
    def compare_belief_strategies(self,
                                  counterparties: List[int],
                                  amount_per_counterparty: float) -> Dict[str, Any]:
        """
        Compare strategies: belief-based vs equal-weight vs risk-free.
        """
        # Belief-based allocation (more to trusted, less to risky)
        belief_weights = {}
        for cp_id in counterparties:
            belief = self.belief_system.get_belief(cp_id)
            # Weight inversely to risk, proportional to trust
            weight = (1 - belief.estimated_pd) * belief.trust_score
            belief_weights[cp_id] = weight
        
        # Normalize
        total_weight = sum(belief_weights.values())
        if total_weight > 0:
            belief_weights = {k: v / total_weight for k, v in belief_weights.items()}
        
        # Equal weight allocation
        equal_weights = {cp_id: 1.0 / len(counterparties) for cp_id in counterparties}
        
        # Simulate portfolios
        total_amount = amount_per_counterparty * len(counterparties)
        
        belief_portfolio = {cp_id: total_amount * w for cp_id, w in belief_weights.items()}
        equal_portfolio = {cp_id: total_amount * w for cp_id, w in equal_weights.items()}
        
        belief_result = self.simulator.simulate_portfolio_outcome(belief_portfolio)
        equal_result = self.simulator.simulate_portfolio_outcome(equal_portfolio)
        
        return {
            'belief_based': {
                'weights': belief_weights,
                'expected_loss': belief_result['expected_loss'],
                'var_95': belief_result['var_95']
            },
            'equal_weight': {
                'weights': equal_weights,
                'expected_loss': equal_result['expected_loss'],
                'var_95': equal_result['var_95']
            },
            'belief_advantage': equal_result['expected_loss'] - belief_result['expected_loss']
        }
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get comprehensive calibration report."""
        metrics = self.tracker.get_calibration_metrics()
        by_confidence = self.tracker.get_calibration_by_confidence()
        
        return {
            'overall': metrics,
            'by_confidence': by_confidence,
            'recommendations': self._calibration_recommendations(metrics)
        }
    
    def _calibration_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on calibration."""
        recommendations = []
        
        if metrics['overconfidence_rate'] > 0.2:
            recommendations.append(
                "High overconfidence detected. Consider using wider uncertainty bounds."
            )
        
        if metrics['underconfidence_rate'] > 0.3:
            recommendations.append(
                "High underconfidence detected. Beliefs may be too pessimistic."
            )
        
        if metrics['calibration_error'] > 0.2:
            recommendations.append(
                "Poor calibration. Review belief update learning rate."
            )
        
        if metrics['brier_score'] > 0.25:
            recommendations.append(
                "High Brier score indicates poor probability predictions."
            )
        
        if not recommendations:
            recommendations.append("Belief calibration appears reasonable.")
        
        return recommendations
