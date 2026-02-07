"""
Belief Evaluation, Calibration, and Safety Bounds for FinSim-MAPPO.
Monitors belief accuracy and prevents pathological belief states.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class CalibrationLevel(Enum):
    """Calibration quality levels."""
    EXCELLENT = "excellent"     # Brier score < 0.05
    GOOD = "good"              # Brier score < 0.1
    MODERATE = "moderate"      # Brier score < 0.15
    POOR = "poor"              # Brier score < 0.25
    VERY_POOR = "very_poor"    # Brier score >= 0.25


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a bank's belief system."""
    bank_id: int
    
    # Brier score (lower is better, 0 = perfect)
    brier_score: float = 0.0
    
    # Calibration by bucket
    calibration_error: float = 0.0  # Expected calibration error
    bucket_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bias metrics
    overconfidence_rate: float = 0.0  # Fraction of overconfident predictions
    underconfidence_rate: float = 0.0
    
    # Accuracy
    accuracy: float = 0.0  # Binary prediction accuracy
    auc: float = 0.5  # Area under ROC curve
    
    # Sample info
    num_predictions: int = 0
    num_defaults: int = 0
    
    @property
    def calibration_level(self) -> CalibrationLevel:
        if self.brier_score < 0.05:
            return CalibrationLevel.EXCELLENT
        elif self.brier_score < 0.1:
            return CalibrationLevel.GOOD
        elif self.brier_score < 0.15:
            return CalibrationLevel.MODERATE
        elif self.brier_score < 0.25:
            return CalibrationLevel.POOR
        return CalibrationLevel.VERY_POOR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bank_id': self.bank_id,
            'brier_score': self.brier_score,
            'calibration_error': self.calibration_error,
            'overconfidence_rate': self.overconfidence_rate,
            'underconfidence_rate': self.underconfidence_rate,
            'accuracy': self.accuracy,
            'auc': self.auc,
            'calibration_level': self.calibration_level.value,
            'num_predictions': self.num_predictions,
            'num_defaults': self.num_defaults
        }


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""
    bank_id: int
    counterparty_id: int
    timestep: int
    
    # Prediction
    predicted_pd: float
    confidence: float
    
    # Outcome (filled in later)
    actual_default: Optional[bool] = None
    outcome_timestep: Optional[int] = None


class BeliefCalibrator:
    """
    Tracks and evaluates belief calibration across all banks.
    """
    
    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets
        
        # Predictions: bank_id -> List[PredictionRecord]
        self.predictions: Dict[int, List[PredictionRecord]] = defaultdict(list)
        
        # Resolved predictions for analysis
        self.resolved: Dict[int, List[PredictionRecord]] = defaultdict(list)
    
    def record_prediction(self, bank_id: int, counterparty_id: int,
                          predicted_pd: float, confidence: float, timestep: int) -> None:
        """Record a prediction."""
        self.predictions[bank_id].append(PredictionRecord(
            bank_id=bank_id,
            counterparty_id=counterparty_id,
            timestep=timestep,
            predicted_pd=predicted_pd,
            confidence=confidence
        ))
    
    def record_outcome(self, counterparty_id: int, defaulted: bool, 
                       timestep: int) -> None:
        """Record an outcome for all predictions about a counterparty."""
        for bank_id in self.predictions:
            for pred in self.predictions[bank_id]:
                if pred.counterparty_id == counterparty_id and pred.actual_default is None:
                    pred.actual_default = defaulted
                    pred.outcome_timestep = timestep
                    self.resolved[bank_id].append(pred)
            
            # Remove resolved from pending
            self.predictions[bank_id] = [
                p for p in self.predictions[bank_id]
                if p.actual_default is None
            ]
    
    def compute_calibration(self, bank_id: int) -> CalibrationMetrics:
        """Compute calibration metrics for a bank."""
        resolved = self.resolved.get(bank_id, [])
        
        if len(resolved) < 10:
            return CalibrationMetrics(
                bank_id=bank_id,
                num_predictions=len(resolved)
            )
        
        # Extract arrays
        predicted = np.array([p.predicted_pd for p in resolved])
        actual = np.array([1.0 if p.actual_default else 0.0 for p in resolved])
        confidence = np.array([p.confidence for p in resolved])
        
        # Brier score
        brier = np.mean((predicted - actual) ** 2)
        
        # Calibration by bucket
        bucket_stats = {}
        bucket_errors = []
        bucket_sizes = []
        
        bucket_edges = np.linspace(0, 1, self.num_buckets + 1)
        
        for i in range(self.num_buckets):
            low, high = bucket_edges[i], bucket_edges[i + 1]
            mask = (predicted >= low) & (predicted < high)
            
            if mask.sum() > 0:
                bucket_pred = predicted[mask].mean()
                bucket_actual = actual[mask].mean()
                bucket_count = mask.sum()
                
                bucket_stats[f"{low:.1f}-{high:.1f}"] = {
                    'mean_predicted': bucket_pred,
                    'mean_actual': bucket_actual,
                    'count': int(bucket_count),
                    'error': abs(bucket_pred - bucket_actual)
                }
                
                bucket_errors.append(abs(bucket_pred - bucket_actual) * bucket_count)
                bucket_sizes.append(bucket_count)
        
        # Expected calibration error
        ece = sum(bucket_errors) / sum(bucket_sizes) if bucket_sizes else 0.0
        
        # Overconfidence: predicted > actual frequently
        threshold = 0.5  # Predict default if PD > 0.5
        binary_pred = (predicted >= threshold).astype(float)
        overconfident = np.mean((binary_pred > actual))
        underconfident = np.mean((binary_pred < actual))
        
        # Accuracy
        accuracy = np.mean(binary_pred == actual)
        
        # Approximate AUC using simple ranking
        try:
            from scipy.stats import rankdata
            n_pos = actual.sum()
            n_neg = len(actual) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = rankdata(predicted)
                pos_ranks = ranks[actual == 1].sum()
                auc = (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        return CalibrationMetrics(
            bank_id=bank_id,
            brier_score=brier,
            calibration_error=ece,
            bucket_stats=bucket_stats,
            overconfidence_rate=overconfident,
            underconfidence_rate=underconfident,
            accuracy=accuracy,
            auc=auc,
            num_predictions=len(resolved),
            num_defaults=int(actual.sum())
        )
    
    def get_all_calibrations(self) -> Dict[int, CalibrationMetrics]:
        """Get calibration metrics for all banks."""
        return {
            bank_id: self.compute_calibration(bank_id)
            for bank_id in self.resolved
        }
    
    def get_system_calibration(self) -> Dict[str, float]:
        """Get system-wide calibration metrics."""
        all_resolved = []
        for bank_id in self.resolved:
            all_resolved.extend(self.resolved[bank_id])
        
        if len(all_resolved) < 10:
            return {'brier_score': 0.0, 'num_predictions': len(all_resolved)}
        
        predicted = np.array([p.predicted_pd for p in all_resolved])
        actual = np.array([1.0 if p.actual_default else 0.0 for p in all_resolved])
        
        return {
            'brier_score': float(np.mean((predicted - actual) ** 2)),
            'mean_predicted_pd': float(predicted.mean()),
            'actual_default_rate': float(actual.mean()),
            'num_predictions': len(all_resolved)
        }


@dataclass
class SafetyBounds:
    """Safety bounds for belief values."""
    min_pd: float = 0.001  # Never believe anyone is risk-free
    max_pd: float = 0.999  # Never be absolutely certain of default
    
    min_trust: float = 0.01  # Never completely distrust
    max_trust: float = 0.99  # Never completely trust
    
    min_confidence: float = 0.0
    max_confidence: float = 0.95  # Never be absolutely certain
    
    max_pd_change_per_step: float = 0.1  # Prevent wild swings
    max_trust_change_per_step: float = 0.15
    
    # Anti-herding bounds
    min_pd_variance: float = 0.001  # Beliefs must have some variance
    max_correlation: float = 0.95   # Prevent perfect correlation
    
    def apply_to_pd(self, pd: float) -> float:
        """Apply safety bounds to PD."""
        return np.clip(pd, self.min_pd, self.max_pd)
    
    def apply_to_trust(self, trust: float) -> float:
        """Apply safety bounds to trust."""
        return np.clip(trust, self.min_trust, self.max_trust)
    
    def apply_to_confidence(self, confidence: float) -> float:
        """Apply safety bounds to confidence."""
        return np.clip(confidence, self.min_confidence, self.max_confidence)
    
    def limit_change(self, old_value: float, new_value: float, 
                     max_change: float) -> float:
        """Limit change in value."""
        change = new_value - old_value
        if abs(change) > max_change:
            return old_value + np.sign(change) * max_change
        return new_value


class BeliefSafetyMonitor:
    """
    Monitors beliefs for pathological states and applies corrections.
    """
    
    def __init__(self, bounds: Optional[SafetyBounds] = None):
        self.bounds = bounds or SafetyBounds()
        
        # Track violations
        self.violations: List[Dict[str, Any]] = []
        
        # Previous values for change limiting
        self.previous_pd: Dict[Tuple[int, int], float] = {}
        self.previous_trust: Dict[Tuple[int, int], float] = {}
    
    def check_and_correct_belief(self, bank_id: int, counterparty_id: int,
                                  pd: float, trust: float, confidence: float,
                                  timestep: int) -> Tuple[float, float, float, List[str]]:
        """
        Check belief values and apply corrections if needed.
        Returns corrected values and list of violations.
        """
        violations = []
        key = (bank_id, counterparty_id)
        
        # Apply bounds
        corrected_pd = self.bounds.apply_to_pd(pd)
        corrected_trust = self.bounds.apply_to_trust(trust)
        corrected_confidence = self.bounds.apply_to_confidence(confidence)
        
        if corrected_pd != pd:
            violations.append(f"PD bounded: {pd:.4f} -> {corrected_pd:.4f}")
        if corrected_trust != trust:
            violations.append(f"Trust bounded: {trust:.4f} -> {corrected_trust:.4f}")
        if corrected_confidence != confidence:
            violations.append(f"Confidence bounded: {confidence:.4f} -> {corrected_confidence:.4f}")
        
        # Limit changes
        if key in self.previous_pd:
            old_pd = self.previous_pd[key]
            limited_pd = self.bounds.limit_change(
                old_pd, corrected_pd, self.bounds.max_pd_change_per_step
            )
            if limited_pd != corrected_pd:
                violations.append(
                    f"PD change limited: {old_pd:.4f} -> {limited_pd:.4f} (requested {corrected_pd:.4f})"
                )
                corrected_pd = limited_pd
        
        if key in self.previous_trust:
            old_trust = self.previous_trust[key]
            limited_trust = self.bounds.limit_change(
                old_trust, corrected_trust, self.bounds.max_trust_change_per_step
            )
            if limited_trust != corrected_trust:
                violations.append(
                    f"Trust change limited: {old_trust:.4f} -> {limited_trust:.4f}"
                )
                corrected_trust = limited_trust
        
        # Update previous values
        self.previous_pd[key] = corrected_pd
        self.previous_trust[key] = corrected_trust
        
        # Log violations
        if violations:
            self.violations.append({
                'timestep': timestep,
                'bank_id': bank_id,
                'counterparty_id': counterparty_id,
                'violations': violations
            })
        
        return corrected_pd, corrected_trust, corrected_confidence, violations
    
    def check_herding(self, beliefs: Dict[int, Dict[int, float]], 
                       timestep: int) -> Dict[str, Any]:
        """
        Check for herding behavior (all banks believing the same thing).
        beliefs: bank_id -> counterparty_id -> PD
        """
        counterparty_beliefs = defaultdict(list)
        
        for bank_id, cp_beliefs in beliefs.items():
            for cp_id, pd in cp_beliefs.items():
                counterparty_beliefs[cp_id].append(pd)
        
        low_variance_counterparties = []
        high_correlation = False
        
        for cp_id, pds in counterparty_beliefs.items():
            if len(pds) >= 3:
                variance = np.var(pds)
                if variance < self.bounds.min_pd_variance:
                    low_variance_counterparties.append({
                        'counterparty_id': cp_id,
                        'variance': variance,
                        'mean_pd': np.mean(pds)
                    })
        
        # Check correlation between bank belief vectors
        bank_vectors = []
        bank_ids = []
        
        for bank_id in beliefs:
            if len(beliefs[bank_id]) >= 3:
                vec = [beliefs[bank_id].get(cp, 0.1) 
                       for cp in sorted(beliefs[bank_id].keys())]
                bank_vectors.append(vec)
                bank_ids.append(bank_id)
        
        if len(bank_vectors) >= 2:
            # Compute pairwise correlations
            max_corr = 0.0
            for i in range(len(bank_vectors)):
                for j in range(i + 1, len(bank_vectors)):
                    if len(bank_vectors[i]) == len(bank_vectors[j]):
                        corr = np.corrcoef(bank_vectors[i], bank_vectors[j])[0, 1]
                        max_corr = max(max_corr, abs(corr) if not np.isnan(corr) else 0)
            
            high_correlation = max_corr > self.bounds.max_correlation
        
        return {
            'timestep': timestep,
            'herding_detected': len(low_variance_counterparties) > 0 or high_correlation,
            'low_variance_counterparties': low_variance_counterparties,
            'high_correlation': high_correlation,
            'max_correlation': max_corr if 'max_corr' in dir() else 0.0
        }
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of safety violations."""
        if not self.violations:
            return {'total_violations': 0}
        
        violation_types = defaultdict(int)
        for v in self.violations:
            for vtype in v['violations']:
                if 'PD bounded' in vtype:
                    violation_types['pd_bounded'] += 1
                elif 'Trust bounded' in vtype:
                    violation_types['trust_bounded'] += 1
                elif 'PD change limited' in vtype:
                    violation_types['pd_change_limited'] += 1
                elif 'Trust change limited' in vtype:
                    violation_types['trust_change_limited'] += 1
        
        return {
            'total_violations': len(self.violations),
            'violation_types': dict(violation_types),
            'recent_violations': self.violations[-10:]
        }


@dataclass
class StabilityMetrics:
    """Trust stability metrics for the system."""
    
    # Trust volatility (standard deviation of trust changes)
    trust_volatility: float = 0.0
    
    # Trust reciprocity (do banks trust each other mutually?)
    trust_reciprocity: float = 0.0
    
    # Trust network density
    trust_density: float = 0.0
    
    # Number of trust breakdowns (sudden drops)
    trust_breakdowns: int = 0
    
    # System fragility index
    fragility_index: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trust_volatility': self.trust_volatility,
            'trust_reciprocity': self.trust_reciprocity,
            'trust_density': self.trust_density,
            'trust_breakdowns': self.trust_breakdowns,
            'fragility_index': self.fragility_index
        }


class TrustStabilityTracker:
    """
    Tracks trust stability over time.
    """
    
    def __init__(self, breakdown_threshold: float = 0.2):
        self.breakdown_threshold = breakdown_threshold
        
        # Trust history: (bank_id, counterparty_id) -> List[(timestep, trust)]
        self.trust_history: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
        
        # Breakdown events
        self.breakdowns: List[Dict[str, Any]] = []
    
    def record_trust(self, bank_id: int, counterparty_id: int,
                     trust: float, timestep: int) -> None:
        """Record trust value."""
        key = (bank_id, counterparty_id)
        
        # Check for breakdown
        if self.trust_history[key]:
            last_trust = self.trust_history[key][-1][1]
            if last_trust - trust > self.breakdown_threshold:
                self.breakdowns.append({
                    'timestep': timestep,
                    'bank_id': bank_id,
                    'counterparty_id': counterparty_id,
                    'trust_before': last_trust,
                    'trust_after': trust,
                    'drop': last_trust - trust
                })
        
        self.trust_history[key].append((timestep, trust))
        
        # Keep limited history
        if len(self.trust_history[key]) > 100:
            self.trust_history[key] = self.trust_history[key][-100:]
    
    def compute_stability_metrics(self) -> StabilityMetrics:
        """Compute trust stability metrics."""
        # Trust volatility
        volatilities = []
        for key, history in self.trust_history.items():
            if len(history) >= 5:
                trusts = [t for _, t in history]
                changes = np.diff(trusts)
                if len(changes) > 0:
                    volatilities.append(np.std(changes))
        
        trust_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # Trust reciprocity
        reciprocities = []
        all_pairs = set()
        for (b1, b2), history in self.trust_history.items():
            if history:
                all_pairs.add((b1, b2))
        
        for b1, b2 in all_pairs:
            if (b2, b1) in all_pairs:
                t1 = self.trust_history[(b1, b2)][-1][1] if self.trust_history[(b1, b2)] else 0.5
                t2 = self.trust_history[(b2, b1)][-1][1] if self.trust_history[(b2, b1)] else 0.5
                reciprocities.append(1 - abs(t1 - t2))
        
        trust_reciprocity = np.mean(reciprocities) if reciprocities else 0.5
        
        # Trust density (fraction of pairs with high trust)
        high_trust_pairs = sum(
            1 for h in self.trust_history.values()
            if h and h[-1][1] > 0.6
        )
        total_pairs = len(self.trust_history)
        trust_density = high_trust_pairs / max(total_pairs, 1)
        
        # Fragility index
        fragility = trust_volatility * (1 - trust_density) * (len(self.breakdowns) + 1) / 10
        fragility = min(fragility, 1.0)
        
        return StabilityMetrics(
            trust_volatility=trust_volatility,
            trust_reciprocity=trust_reciprocity,
            trust_density=trust_density,
            trust_breakdowns=len(self.breakdowns),
            fragility_index=fragility
        )
    
    def get_recent_breakdowns(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent trust breakdowns."""
        return self.breakdowns[-n:]


class BeliefEvaluationEngine:
    """
    Main engine for belief evaluation and safety.
    """
    
    def __init__(self, 
                 safety_bounds: Optional[SafetyBounds] = None,
                 breakdown_threshold: float = 0.2):
        self.calibrator = BeliefCalibrator()
        self.safety_monitor = BeliefSafetyMonitor(safety_bounds)
        self.stability_tracker = TrustStabilityTracker(breakdown_threshold)
    
    def record_prediction(self, bank_id: int, counterparty_id: int,
                          predicted_pd: float, confidence: float, timestep: int) -> None:
        """Record a PD prediction."""
        self.calibrator.record_prediction(
            bank_id, counterparty_id, predicted_pd, confidence, timestep
        )
    
    def record_outcome(self, counterparty_id: int, defaulted: bool, timestep: int) -> None:
        """Record a default/non-default outcome."""
        self.calibrator.record_outcome(counterparty_id, defaulted, timestep)
    
    def record_trust(self, bank_id: int, counterparty_id: int,
                     trust: float, timestep: int) -> None:
        """Record trust for stability tracking."""
        self.stability_tracker.record_trust(bank_id, counterparty_id, trust, timestep)
    
    def apply_safety_bounds(self, bank_id: int, counterparty_id: int,
                            pd: float, trust: float, confidence: float,
                            timestep: int) -> Tuple[float, float, float]:
        """Apply safety bounds to belief values."""
        corrected_pd, corrected_trust, corrected_conf, _ = \
            self.safety_monitor.check_and_correct_belief(
                bank_id, counterparty_id, pd, trust, confidence, timestep
            )
        return corrected_pd, corrected_trust, corrected_conf
    
    def check_system_health(self, beliefs: Dict[int, Dict[int, float]],
                            timestep: int) -> Dict[str, Any]:
        """Check overall system health."""
        # Calibration
        system_calibration = self.calibrator.get_system_calibration()
        
        # Herding
        herding = self.safety_monitor.check_herding(beliefs, timestep)
        
        # Stability
        stability = self.stability_tracker.compute_stability_metrics()
        
        # Safety violations
        violations = self.safety_monitor.get_violation_summary()
        
        # Overall health score
        health_score = 1.0
        
        if system_calibration.get('brier_score', 0) > 0.2:
            health_score -= 0.2
        if herding.get('herding_detected', False):
            health_score -= 0.3
        if stability.fragility_index > 0.5:
            health_score -= 0.2
        if violations.get('total_violations', 0) > 50:
            health_score -= 0.1
        
        health_score = max(0, health_score)
        
        return {
            'timestep': timestep,
            'health_score': health_score,
            'calibration': system_calibration,
            'herding': herding,
            'stability': stability.to_dict(),
            'safety_violations': violations
        }
    
    def get_bank_report(self, bank_id: int) -> Dict[str, Any]:
        """Get comprehensive evaluation report for a bank."""
        calibration = self.calibrator.compute_calibration(bank_id)
        
        return {
            'bank_id': bank_id,
            'calibration': calibration.to_dict(),
            'calibration_level': calibration.calibration_level.value
        }
    
    def get_system_report(self, beliefs: Dict[int, Dict[int, float]],
                          timestep: int) -> Dict[str, Any]:
        """Get comprehensive system evaluation report."""
        bank_reports = {}
        for bank_id in beliefs:
            bank_reports[bank_id] = self.get_bank_report(bank_id)
        
        return {
            'system_health': self.check_system_health(beliefs, timestep),
            'bank_reports': bank_reports,
            'recent_breakdowns': self.stability_tracker.get_recent_breakdowns()
        }
