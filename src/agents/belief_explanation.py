"""
Belief Explanation and UI Data Export for FinSim-MAPPO.
Provides interpretable explanations and UI-ready data for beliefs.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import json

from .beliefs import BeliefSystem, CounterpartyBelief, BeliefConfidence
from .reputation import TrustDynamicsEngine, ReputationScore


@dataclass
class BeliefExplanation:
    """Explanation for a belief state."""
    bank_id: int
    counterparty_id: int
    timestep: int
    
    # Current belief state
    estimated_pd: float
    trust_score: float
    confidence: float
    
    # Explanation
    primary_reason: str
    contributing_factors: List[str]
    
    # Evidence
    key_observations: List[Dict[str, Any]]
    sample_size: int
    
    # Recommendation
    recommendation: str
    risk_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_narrative(self) -> str:
        """Generate natural language explanation."""
        narrative = f"Bank {self.bank_id}'s assessment of Bank {self.counterparty_id}: "
        
        if self.confidence < 0.3:
            narrative += "Limited information available. "
        
        if self.estimated_pd < 0.05:
            narrative += f"Considered low risk (PD: {self.estimated_pd:.1%}). "
        elif self.estimated_pd < 0.15:
            narrative += f"Moderate risk (PD: {self.estimated_pd:.1%}). "
        else:
            narrative += f"High risk (PD: {self.estimated_pd:.1%}). "
        
        if self.trust_score > 0.7:
            narrative += "High trust based on positive history. "
        elif self.trust_score < 0.3:
            narrative += "Low trust - past issues detected. "
        
        narrative += self.primary_reason
        
        return narrative


@dataclass
class BeliefEvolutionTrace:
    """Trace of belief evolution over time."""
    bank_id: int
    counterparty_id: int
    
    timesteps: List[int] = field(default_factory=list)
    pd_values: List[float] = field(default_factory=list)
    trust_values: List[float] = field(default_factory=list)
    confidence_values: List[float] = field(default_factory=list)
    
    # Events that caused changes
    change_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_point(self, timestep: int, belief: CounterpartyBelief,
                  event: Optional[str] = None) -> None:
        self.timesteps.append(timestep)
        self.pd_values.append(belief.estimated_pd)
        self.trust_values.append(belief.trust_score)
        self.confidence_values.append(belief.confidence_score)
        
        if event:
            self.change_events.append({
                'timestep': timestep,
                'event': event,
                'pd': belief.estimated_pd,
                'trust': belief.trust_score
            })
    
    def get_trend(self) -> Dict[str, str]:
        """Get trends for each metric."""
        def compute_trend(values: List[float]) -> str:
            if len(values) < 3:
                return "stable"
            recent = np.mean(values[-3:])
            older = np.mean(values[:-3])
            if recent > older + 0.05:
                return "increasing"
            elif recent < older - 0.05:
                return "decreasing"
            return "stable"
        
        return {
            'pd_trend': compute_trend(self.pd_values),
            'trust_trend': compute_trend(self.trust_values),
            'confidence_trend': compute_trend(self.confidence_values)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bank_id': self.bank_id,
            'counterparty_id': self.counterparty_id,
            'timesteps': self.timesteps,
            'pd_values': self.pd_values,
            'trust_values': self.trust_values,
            'confidence_values': self.confidence_values,
            'change_events': self.change_events,
            'trends': self.get_trend()
        }


class BeliefLogger:
    """
    Logs belief states for analysis and explanation.
    """
    
    def __init__(self):
        # Belief snapshots: bank_id -> counterparty_id -> [(timestep, belief)]
        self.belief_snapshots: Dict[int, Dict[int, List[Tuple[int, CounterpartyBelief]]]] = \
            defaultdict(lambda: defaultdict(list))
        
        # Evolution traces
        self.traces: Dict[Tuple[int, int], BeliefEvolutionTrace] = {}
        
        # Decision logs
        self.decision_logs: List[Dict[str, Any]] = []
    
    def log_belief(self, bank_id: int, counterparty_id: int,
                   belief: CounterpartyBelief, timestep: int,
                   event: Optional[str] = None) -> None:
        """Log a belief state."""
        # Store snapshot
        self.belief_snapshots[bank_id][counterparty_id].append(
            (timestep, CounterpartyBelief(**asdict(belief)))  # Deep copy
        )
        
        # Keep limited history
        if len(self.belief_snapshots[bank_id][counterparty_id]) > 100:
            self.belief_snapshots[bank_id][counterparty_id] = \
                self.belief_snapshots[bank_id][counterparty_id][-100:]
        
        # Update trace
        key = (bank_id, counterparty_id)
        if key not in self.traces:
            self.traces[key] = BeliefEvolutionTrace(
                bank_id=bank_id,
                counterparty_id=counterparty_id
            )
        self.traces[key].add_point(timestep, belief, event)
    
    def log_decision(self, bank_id: int, counterparty_id: int,
                     action: str, belief: CounterpartyBelief,
                     outcome: Optional[Dict[str, Any]], timestep: int) -> None:
        """Log a decision made based on beliefs."""
        self.decision_logs.append({
            'timestep': timestep,
            'bank_id': bank_id,
            'counterparty_id': counterparty_id,
            'action': action,
            'belief_pd': belief.estimated_pd,
            'belief_trust': belief.trust_score,
            'belief_confidence': belief.confidence_score,
            'outcome': outcome
        })
        
        if len(self.decision_logs) > 1000:
            self.decision_logs = self.decision_logs[-1000:]
    
    def get_belief_history(self, bank_id: int, 
                           counterparty_id: int) -> List[Tuple[int, CounterpartyBelief]]:
        """Get belief history for a pair."""
        return self.belief_snapshots.get(bank_id, {}).get(counterparty_id, [])
    
    def get_evolution_trace(self, bank_id: int, 
                            counterparty_id: int) -> Optional[BeliefEvolutionTrace]:
        """Get evolution trace for a pair."""
        return self.traces.get((bank_id, counterparty_id))
    
    def get_decisions_for_bank(self, bank_id: int) -> List[Dict[str, Any]]:
        """Get all decisions made by a bank."""
        return [d for d in self.decision_logs if d['bank_id'] == bank_id]


class BeliefExplainer:
    """
    Generates explanations for belief states.
    """
    
    def __init__(self, belief_logger: BeliefLogger):
        self.logger = belief_logger
    
    def explain_belief(self, bank_id: int, counterparty_id: int,
                       belief: CounterpartyBelief, timestep: int) -> BeliefExplanation:
        """Generate explanation for current belief."""
        # Determine primary reason
        primary_reason, factors = self._analyze_belief_formation(
            bank_id, counterparty_id, belief
        )
        
        # Get key observations
        observations = self._get_key_observations(bank_id, counterparty_id)
        
        # Determine recommendation
        recommendation = self._generate_recommendation(belief)
        
        # Risk level
        if belief.estimated_pd < 0.03:
            risk_level = "low"
        elif belief.estimated_pd < 0.1:
            risk_level = "moderate"
        elif belief.estimated_pd < 0.2:
            risk_level = "elevated"
        else:
            risk_level = "high"
        
        return BeliefExplanation(
            bank_id=bank_id,
            counterparty_id=counterparty_id,
            timestep=timestep,
            estimated_pd=belief.estimated_pd,
            trust_score=belief.trust_score,
            confidence=belief.confidence_score,
            primary_reason=primary_reason,
            contributing_factors=factors,
            key_observations=observations,
            sample_size=belief.sample_size,
            recommendation=recommendation,
            risk_level=risk_level
        )
    
    def _analyze_belief_formation(self, bank_id: int, counterparty_id: int,
                                   belief: CounterpartyBelief) -> Tuple[str, List[str]]:
        """Analyze how belief was formed."""
        factors = []
        
        if belief.confidence_score < 0.2:
            primary = "Based on market priors due to limited history."
            factors.append("No significant interaction history")
        elif belief.sample_size < 5:
            primary = "Based on limited interactions."
            factors.append(f"Only {belief.sample_size} observations")
        else:
            # Analyze from history
            trace = self.logger.get_evolution_trace(bank_id, counterparty_id)
            
            if trace and trace.change_events:
                last_event = trace.change_events[-1]
                primary = f"Recent event: {last_event.get('event', 'interaction')}."
            else:
                primary = "Based on accumulated interaction history."
            
            # Analyze trends
            if trace:
                trends = trace.get_trend()
                if trends['pd_trend'] == 'increasing':
                    factors.append("Risk perception has been increasing")
                elif trends['pd_trend'] == 'decreasing':
                    factors.append("Risk perception has been improving")
                
                if trends['trust_trend'] == 'increasing':
                    factors.append("Trust has been building")
                elif trends['trust_trend'] == 'decreasing':
                    factors.append("Trust has been eroding")
        
        # Analyze current state
        if belief.estimated_pd > 0.15:
            factors.append("History suggests elevated default risk")
        if belief.trust_score < 0.3:
            factors.append("Past behavior has damaged trust")
        if belief.trust_score > 0.7:
            factors.append("Strong track record of reliability")
        
        return primary, factors
    
    def _get_key_observations(self, bank_id: int, 
                               counterparty_id: int) -> List[Dict[str, Any]]:
        """Get key observations that shaped belief."""
        decisions = [d for d in self.logger.decision_logs
                    if d['bank_id'] == bank_id and d['counterparty_id'] == counterparty_id]
        
        # Get most recent significant decisions
        significant = [d for d in decisions if d.get('outcome')]
        return significant[-5:]  # Last 5
    
    def _generate_recommendation(self, belief: CounterpartyBelief) -> str:
        """Generate recommendation based on belief."""
        if belief.confidence_score < 0.2:
            return "Insufficient data for recommendation. Proceed with caution."
        
        if belief.estimated_pd > 0.2 or belief.trust_score < 0.2:
            return "Avoid significant exposure. High risk detected."
        
        if belief.estimated_pd > 0.1 or belief.trust_score < 0.4:
            return "Limit exposure. Require additional collateral."
        
        if belief.estimated_pd < 0.03 and belief.trust_score > 0.7:
            return "Low risk. Standard terms appropriate."
        
        return "Moderate terms recommended. Monitor ongoing."


class BeliefUIExporter:
    """
    Exports belief data for UI consumption.
    """
    
    def __init__(self, 
                 belief_managers: Dict[int, BeliefSystem],
                 trust_engine: Optional[TrustDynamicsEngine] = None,
                 belief_logger: Optional[BeliefLogger] = None):
        self.belief_managers = belief_managers
        self.trust_engine = trust_engine
        self.logger = belief_logger or BeliefLogger()
        self.explainer = BeliefExplainer(self.logger)
    
    def get_bank_belief_summary(self, bank_id: int, timestep: int) -> Dict[str, Any]:
        """Get belief summary for a single bank."""
        if bank_id not in self.belief_managers:
            return {'bank_id': bank_id, 'beliefs': {}}
        
        belief_system = self.belief_managers[bank_id]
        beliefs = belief_system.get_all_beliefs()
        
        # Top risky counterparties
        risky = sorted(beliefs.items(), key=lambda x: x[1].estimated_pd, reverse=True)[:5]
        
        # Most trusted
        trusted = sorted(beliefs.items(), key=lambda x: x[1].trust_score, reverse=True)[:5]
        
        # Least confident
        uncertain = sorted(beliefs.items(), key=lambda x: x[1].confidence_score)[:5]
        
        summary = belief_system.get_belief_summary()
        
        return {
            'bank_id': bank_id,
            'timestep': timestep,
            'summary': summary,
            'top_risky': [
                {
                    'counterparty_id': cp_id,
                    'pd': b.estimated_pd,
                    'trust': b.trust_score
                } for cp_id, b in risky
            ],
            'most_trusted': [
                {
                    'counterparty_id': cp_id,
                    'trust': b.trust_score,
                    'pd': b.estimated_pd
                } for cp_id, b in trusted
            ],
            'most_uncertain': [
                {
                    'counterparty_id': cp_id,
                    'confidence': b.confidence_score,
                    'sample_size': b.sample_size
                } for cp_id, b in uncertain
            ]
        }
    
    def get_pairwise_belief_detail(self, bank_id: int, counterparty_id: int,
                                    timestep: int) -> Dict[str, Any]:
        """Get detailed belief info for a specific pair."""
        if bank_id not in self.belief_managers:
            return {'error': 'Bank not found'}
        
        belief_system = self.belief_managers[bank_id]
        belief = belief_system.get_belief(counterparty_id)
        
        # Get explanation
        explanation = self.explainer.explain_belief(
            bank_id, counterparty_id, belief, timestep
        )
        
        # Get evolution trace
        trace = self.logger.get_evolution_trace(bank_id, counterparty_id)
        
        return {
            'bank_id': bank_id,
            'counterparty_id': counterparty_id,
            'belief': belief.to_dict(),
            'explanation': explanation.to_dict(),
            'narrative': explanation.to_narrative(),
            'evolution': trace.to_dict() if trace else None
        }
    
    def get_belief_heatmap_data(self, timestep: int) -> Dict[str, Any]:
        """Get belief data as heatmap."""
        all_bank_ids = sorted(self.belief_managers.keys())
        
        # PD matrix
        pd_matrix = []
        trust_matrix = []
        
        for bank_id in all_bank_ids:
            pd_row = []
            trust_row = []
            belief_system = self.belief_managers[bank_id]
            
            for cp_id in all_bank_ids:
                if bank_id == cp_id:
                    pd_row.append(0)
                    trust_row.append(1)
                else:
                    belief = belief_system.get_belief(cp_id)
                    pd_row.append(belief.estimated_pd)
                    trust_row.append(belief.trust_score)
            
            pd_matrix.append(pd_row)
            trust_matrix.append(trust_row)
        
        return {
            'entity_ids': all_bank_ids,
            'pd_matrix': pd_matrix,
            'trust_matrix': trust_matrix,
            'timestep': timestep
        }
    
    def get_trust_network_data(self, timestep: int) -> Dict[str, Any]:
        """Get trust network for visualization."""
        if self.trust_engine:
            return self.trust_engine.get_trust_network_data()
        
        # Build from belief systems
        nodes = []
        edges = []
        
        for bank_id in self.belief_managers:
            belief_system = self.belief_managers[bank_id]
            summary = belief_system.get_belief_summary()
            
            nodes.append({
                'id': bank_id,
                'num_beliefs': summary['num_counterparties'],
                'avg_trust_given': summary.get('average_trust', 0.5)
            })
            
            for cp_id, belief in belief_system.get_all_beliefs().items():
                edges.append({
                    'from': bank_id,
                    'to': cp_id,
                    'trust': belief.trust_score,
                    'pd': belief.estimated_pd,
                    'confidence': belief.confidence_score
                })
        
        return {'nodes': nodes, 'edges': edges, 'timestep': timestep}
    
    def get_belief_time_series(self, bank_id: int, counterparty_id: int) -> Dict[str, Any]:
        """Get time series data for belief evolution."""
        trace = self.logger.get_evolution_trace(bank_id, counterparty_id)
        
        if not trace:
            return {'error': 'No history available'}
        
        return {
            'bank_id': bank_id,
            'counterparty_id': counterparty_id,
            'time_series': {
                'timesteps': trace.timesteps,
                'pd': trace.pd_values,
                'trust': trace.trust_values,
                'confidence': trace.confidence_values
            },
            'events': trace.change_events,
            'trends': trace.get_trend()
        }
    
    def export_all_beliefs(self, timestep: int) -> Dict[str, Any]:
        """Export all belief data."""
        return {
            'timestep': timestep,
            'bank_summaries': {
                bank_id: self.get_bank_belief_summary(bank_id, timestep)
                for bank_id in self.belief_managers
            },
            'heatmap': self.get_belief_heatmap_data(timestep),
            'trust_network': self.get_trust_network_data(timestep),
            'system_trust_index': self.trust_engine.get_public_signals()['system_trust_index'] 
                                  if self.trust_engine else 0.5
        }
    
    def export_to_json(self, filepath: str, timestep: int) -> None:
        """Export all beliefs to JSON file."""
        data = self.export_all_beliefs(timestep)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
