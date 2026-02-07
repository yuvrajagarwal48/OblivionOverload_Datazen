"""
Explanation and Attribution Layer for FinSim-MAPPO.
Provides interpretable explanations for simulation outcomes.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json


class EventType(str, Enum):
    """Types of events that can be explained."""
    DEFAULT = "default"
    MARGIN_CALL = "margin_call"
    FIRE_SALE = "fire_sale"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CONTAGION = "contagion"
    CLEARING_FAILURE = "clearing_failure"
    CAPITAL_BREACH = "capital_breach"
    INTERVENTION = "intervention"
    RECOVERY = "recovery"


class CauseType(str, Enum):
    """Types of causal factors."""
    DIRECT_EXPOSURE = "direct_exposure"
    NETWORK_CONTAGION = "network_contagion"
    MARKET_SHOCK = "market_shock"
    LIQUIDITY_DRAIN = "liquidity_drain"
    COUNTERPARTY_DEFAULT = "counterparty_default"
    MARGIN_SHORTFALL = "margin_shortfall"
    FIRE_SALE_SPIRAL = "fire_sale_spiral"
    CONFIDENCE_LOSS = "confidence_loss"
    REGULATORY_ACTION = "regulatory_action"


@dataclass
class CausalFactor:
    """A single causal factor in an event."""
    cause_type: CauseType
    source_entity: Optional[int] = None
    contribution: float = 0.0  # 0-1, how much this caused the event
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['cause_type'] = self.cause_type.value
        return result


@dataclass
class ContagionPath:
    """A path through which stress propagated."""
    source_entity: int
    path: List[int]  # Sequence of entity IDs
    channel: str  # e.g., "interbank_lending", "ccp_clearing", "fire_sale"
    loss_at_each_hop: List[float]
    amplification_factor: float = 1.0
    timestamp_start: int = 0
    timestamp_end: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def length(self) -> int:
        return len(self.path)
    
    @property
    def total_loss(self) -> float:
        return sum(self.loss_at_each_hop)


@dataclass
class EventExplanation:
    """Complete explanation for a simulation event."""
    event_id: str
    event_type: EventType
    timestamp: int
    affected_entity: int
    
    # Causal analysis
    primary_cause: CausalFactor
    contributing_factors: List[CausalFactor] = field(default_factory=list)
    
    # Contagion paths (if applicable)
    contagion_paths: List[ContagionPath] = field(default_factory=list)
    
    # Counterfactual analysis
    preventable: bool = False
    prevention_actions: List[str] = field(default_factory=list)
    
    # Severity assessment
    severity_score: float = 0.0  # 0-1
    systemic_impact: float = 0.0  # 0-1
    
    # Natural language explanation
    narrative: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'affected_entity': self.affected_entity,
            'primary_cause': self.primary_cause.to_dict(),
            'contributing_factors': [f.to_dict() for f in self.contributing_factors],
            'contagion_paths': [p.to_dict() for p in self.contagion_paths],
            'preventable': self.preventable,
            'prevention_actions': self.prevention_actions,
            'severity_score': self.severity_score,
            'systemic_impact': self.systemic_impact,
            'narrative': self.narrative
        }
        return result


@dataclass
class AttributionResult:
    """Attribution of outcomes to specific factors."""
    entity_id: int
    outcome_metric: str  # e.g., "capital_loss", "default_probability"
    outcome_value: float
    
    # Factor attributions (sum to ~1.0)
    factor_attributions: Dict[str, float] = field(default_factory=dict)
    
    # Top contributing entities
    entity_contributions: Dict[int, float] = field(default_factory=dict)
    
    # Time attribution
    time_attributions: Dict[int, float] = field(default_factory=dict)  # timestep -> contribution
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContagionTracker:
    """Tracks stress propagation through the network."""
    
    def __init__(self):
        self.active_paths: Dict[int, List[ContagionPath]] = {}  # source -> paths
        self.completed_paths: List[ContagionPath] = []
        self.current_timestep = 0
    
    def start_path(self, source_entity: int, channel: str, 
                   initial_loss: float, timestamp: int) -> None:
        """Start tracking a new contagion path."""
        path = ContagionPath(
            source_entity=source_entity,
            path=[source_entity],
            channel=channel,
            loss_at_each_hop=[initial_loss],
            timestamp_start=timestamp
        )
        
        if source_entity not in self.active_paths:
            self.active_paths[source_entity] = []
        self.active_paths[source_entity].append(path)
        self.current_timestep = timestamp
    
    def extend_path(self, source_entity: int, next_entity: int,
                    loss: float, amplification: float = 1.0) -> None:
        """Extend an active contagion path."""
        if source_entity not in self.active_paths:
            return
        
        for path in self.active_paths[source_entity]:
            if path.path[-1] == source_entity or next_entity in path.path[-2:]:
                # Create new branch
                new_path = ContagionPath(
                    source_entity=path.source_entity,
                    path=path.path + [next_entity],
                    channel=path.channel,
                    loss_at_each_hop=path.loss_at_each_hop + [loss],
                    amplification_factor=path.amplification_factor * amplification,
                    timestamp_start=path.timestamp_start
                )
                self.active_paths.setdefault(next_entity, []).append(new_path)
    
    def complete_paths(self, timestamp: int) -> List[ContagionPath]:
        """Mark paths as complete and return them."""
        completed = []
        for source, paths in self.active_paths.items():
            for path in paths:
                path.timestamp_end = timestamp
                completed.append(path)
        
        self.completed_paths.extend(completed)
        self.active_paths.clear()
        return completed
    
    def get_paths_to_entity(self, entity_id: int) -> List[ContagionPath]:
        """Get all contagion paths that reached a specific entity."""
        return [p for p in self.completed_paths if entity_id in p.path]
    
    def get_most_damaging_path(self) -> Optional[ContagionPath]:
        """Get the path with highest total loss."""
        if not self.completed_paths:
            return None
        return max(self.completed_paths, key=lambda p: p.total_loss)


class CausalityAnalyzer:
    """Analyzes causal relationships between events."""
    
    def __init__(self):
        self.event_timeline: List[Tuple[int, str, int, Dict[str, Any]]] = []  # (timestep, event_type, entity, data)
        self.exposure_graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.state_snapshots: Dict[int, Dict[int, Dict[str, float]]] = {}  # timestep -> entity -> metrics
    
    def record_event(self, timestep: int, event_type: str, 
                     entity_id: int, data: Dict[str, Any]) -> None:
        """Record an event for later analysis."""
        self.event_timeline.append((timestep, event_type, entity_id, data))
    
    def record_exposure(self, from_entity: int, to_entity: int, amount: float) -> None:
        """Record an exposure relationship."""
        self.exposure_graph[from_entity][to_entity] = amount
    
    def record_state(self, timestep: int, entity_id: int, metrics: Dict[str, float]) -> None:
        """Record entity state for counterfactual analysis."""
        if timestep not in self.state_snapshots:
            self.state_snapshots[timestep] = {}
        self.state_snapshots[timestep][entity_id] = metrics.copy()
    
    def analyze_default(self, entity_id: int, timestep: int) -> EventExplanation:
        """Analyze causes of a default event."""
        factors = []
        
        # Look for preceding events
        recent_events = [e for e in self.event_timeline 
                        if e[0] <= timestep and e[0] >= timestep - 5]
        
        # Check for counterparty defaults
        counterparty_defaults = [e for e in recent_events 
                                if e[1] == 'default' and e[2] != entity_id]
        
        if counterparty_defaults:
            for cd in counterparty_defaults:
                exposure = self.exposure_graph.get(entity_id, {}).get(cd[2], 0)
                if exposure > 0:
                    factors.append(CausalFactor(
                        cause_type=CauseType.COUNTERPARTY_DEFAULT,
                        source_entity=cd[2],
                        contribution=min(exposure / 1e6, 0.5),
                        description=f"Counterparty {cd[2]} defaulted with {exposure:,.0f} exposure"
                    ))
        
        # Check for margin calls
        margin_calls = [e for e in recent_events 
                       if e[1] == 'margin_call' and e[2] == entity_id]
        if margin_calls:
            factors.append(CausalFactor(
                cause_type=CauseType.MARGIN_SHORTFALL,
                contribution=0.3,
                description=f"{len(margin_calls)} margin calls in preceding timesteps"
            ))
        
        # Check state deterioration
        if timestep in self.state_snapshots and entity_id in self.state_snapshots[timestep]:
            current = self.state_snapshots[timestep][entity_id]
            prev_timestep = max(t for t in self.state_snapshots if t < timestep) if any(t < timestep for t in self.state_snapshots) else timestep
            
            if prev_timestep in self.state_snapshots and entity_id in self.state_snapshots[prev_timestep]:
                prev = self.state_snapshots[prev_timestep][entity_id]
                
                capital_drop = prev.get('capital_ratio', 0.1) - current.get('capital_ratio', 0)
                if capital_drop > 0.02:
                    factors.append(CausalFactor(
                        cause_type=CauseType.LIQUIDITY_DRAIN,
                        contribution=capital_drop * 5,
                        description=f"Capital ratio dropped by {capital_drop:.1%}"
                    ))
        
        # Determine primary cause
        if factors:
            factors.sort(key=lambda f: f.contribution, reverse=True)
            primary = factors[0]
            contributing = factors[1:4]
        else:
            primary = CausalFactor(
                cause_type=CauseType.MARKET_SHOCK,
                contribution=0.5,
                description="General market stress"
            )
            contributing = []
        
        # Assess severity
        severity = 0.5 + 0.5 * len(factors) / max(len(factors), 1)
        
        # Generate narrative
        narrative = self._generate_default_narrative(entity_id, primary, contributing, timestep)
        
        return EventExplanation(
            event_id=f"default_{entity_id}_{timestep}",
            event_type=EventType.DEFAULT,
            timestamp=timestep,
            affected_entity=entity_id,
            primary_cause=primary,
            contributing_factors=contributing,
            severity_score=min(severity, 1.0),
            systemic_impact=self._estimate_systemic_impact(entity_id),
            narrative=narrative,
            preventable=len(contributing) > 0,
            prevention_actions=self._suggest_prevention(primary, contributing)
        )
    
    def _generate_default_narrative(self, entity_id: int, 
                                     primary: CausalFactor,
                                     contributing: List[CausalFactor],
                                     timestep: int) -> str:
        """Generate natural language explanation."""
        narrative = f"Bank {entity_id} defaulted at timestep {timestep}. "
        
        cause_descriptions = {
            CauseType.COUNTERPARTY_DEFAULT: f"The primary cause was a counterparty default (Bank {primary.source_entity}), which created direct losses.",
            CauseType.MARGIN_SHORTFALL: "The primary cause was inability to meet margin requirements, depleting available liquidity.",
            CauseType.LIQUIDITY_DRAIN: "The primary cause was a severe liquidity drain, leaving insufficient cash to meet obligations.",
            CauseType.NETWORK_CONTAGION: "The primary cause was contagion spreading through the interbank network.",
            CauseType.FIRE_SALE_SPIRAL: "The primary cause was a fire sale spiral that depressed asset values.",
            CauseType.MARKET_SHOCK: "The primary cause was a general market shock affecting asset valuations."
        }
        
        narrative += cause_descriptions.get(primary.cause_type, "Multiple factors contributed to the failure. ")
        
        if contributing:
            narrative += " Contributing factors included: "
            contrib_strs = [f"{f.cause_type.value.replace('_', ' ')} ({f.contribution:.0%})" 
                           for f in contributing[:3]]
            narrative += ", ".join(contrib_strs) + "."
        
        return narrative
    
    def _estimate_systemic_impact(self, entity_id: int) -> float:
        """Estimate systemic impact of an entity's failure."""
        # Based on exposure graph
        outgoing = sum(self.exposure_graph.get(entity_id, {}).values())
        incoming = sum(exps.get(entity_id, 0) for exps in self.exposure_graph.values())
        
        total_exposures = sum(sum(exps.values()) for exps in self.exposure_graph.values())
        
        if total_exposures > 0:
            return min((outgoing + incoming) / total_exposures * 2, 1.0)
        return 0.3
    
    def _suggest_prevention(self, primary: CausalFactor, 
                           contributing: List[CausalFactor]) -> List[str]:
        """Suggest actions that could have prevented the event."""
        suggestions = []
        
        if primary.cause_type == CauseType.COUNTERPARTY_DEFAULT:
            suggestions.append(f"Reduce exposure to Bank {primary.source_entity}")
            suggestions.append("Diversify counterparty portfolio")
        
        if primary.cause_type == CauseType.MARGIN_SHORTFALL:
            suggestions.append("Maintain higher collateral buffer")
            suggestions.append("Reduce derivative positions")
        
        if primary.cause_type == CauseType.LIQUIDITY_DRAIN:
            suggestions.append("Hold larger cash reserves")
            suggestions.append("Establish committed credit lines")
        
        if any(f.cause_type == CauseType.FIRE_SALE_SPIRAL for f in [primary] + contributing):
            suggestions.append("Reduce illiquid asset holdings")
            suggestions.append("Improve funding duration matching")
        
        return suggestions[:4]


class ExplanationLayer:
    """
    Main explanation and attribution layer.
    Provides interpretable explanations for all simulation events.
    """
    
    def __init__(self):
        self.contagion_tracker = ContagionTracker()
        self.causality_analyzer = CausalityAnalyzer()
        self.explanations: List[EventExplanation] = []
        self.attributions: Dict[int, List[AttributionResult]] = defaultdict(list)
    
    def record_event(self, timestep: int, event_type: str,
                     entity_id: int, data: Dict[str, Any] = None) -> None:
        """Record an event for later explanation."""
        self.causality_analyzer.record_event(timestep, event_type, entity_id, data or {})
        
        # Auto-track contagion paths
        if event_type == 'default':
            self.contagion_tracker.start_path(
                entity_id, 'counterparty_default', 
                data.get('loss', 0) if data else 0, 
                timestep
            )
    
    def record_exposure(self, from_entity: int, to_entity: int, amount: float) -> None:
        """Record exposure for causality analysis."""
        self.causality_analyzer.record_exposure(from_entity, to_entity, amount)
    
    def record_state(self, timestep: int, entity_id: int, 
                     metrics: Dict[str, float]) -> None:
        """Record entity state snapshot."""
        self.causality_analyzer.record_state(timestep, entity_id, metrics)
    
    def record_contagion(self, source: int, target: int, 
                         loss: float, amplification: float = 1.0) -> None:
        """Record contagion propagation."""
        self.contagion_tracker.extend_path(source, target, loss, amplification)
    
    def explain_event(self, event_type: EventType, 
                      entity_id: int, timestep: int) -> EventExplanation:
        """Generate explanation for a specific event."""
        if event_type == EventType.DEFAULT:
            explanation = self.causality_analyzer.analyze_default(entity_id, timestep)
        else:
            # Generic explanation for other event types
            explanation = EventExplanation(
                event_id=f"{event_type.value}_{entity_id}_{timestep}",
                event_type=event_type,
                timestamp=timestep,
                affected_entity=entity_id,
                primary_cause=CausalFactor(
                    cause_type=CauseType.MARKET_SHOCK,
                    contribution=0.5
                ),
                narrative=f"{event_type.value.replace('_', ' ').title()} occurred for entity {entity_id}"
            )
        
        # Attach contagion paths
        explanation.contagion_paths = self.contagion_tracker.get_paths_to_entity(entity_id)
        
        self.explanations.append(explanation)
        return explanation
    
    def compute_attribution(self, entity_id: int, 
                            outcome_metric: str,
                            outcome_value: float,
                            potential_factors: Dict[str, float]) -> AttributionResult:
        """
        Compute attribution of an outcome to various factors.
        Uses Shapley-like decomposition.
        """
        # Normalize factors
        total_factor = sum(abs(v) for v in potential_factors.values())
        if total_factor > 0:
            attributions = {k: abs(v) / total_factor for k, v in potential_factors.items()}
        else:
            attributions = {k: 1.0 / len(potential_factors) for k in potential_factors}
        
        result = AttributionResult(
            entity_id=entity_id,
            outcome_metric=outcome_metric,
            outcome_value=outcome_value,
            factor_attributions=attributions
        )
        
        self.attributions[entity_id].append(result)
        return result
    
    def explain_cascade(self, start_entity: int) -> Dict[str, Any]:
        """Generate comprehensive cascade explanation."""
        paths = self.contagion_tracker.get_paths_to_entity(start_entity)
        
        if not paths:
            # Check if start_entity is a source
            paths = [p for p in self.contagion_tracker.completed_paths 
                    if p.source_entity == start_entity]
        
        cascade_info = {
            'origin': start_entity,
            'total_paths': len(paths),
            'total_entities_affected': len(set(e for p in paths for e in p.path)),
            'total_loss': sum(p.total_loss for p in paths),
            'max_depth': max((p.length for p in paths), default=0),
            'channels': list(set(p.channel for p in paths)),
            'amplification_range': (
                min((p.amplification_factor for p in paths), default=1),
                max((p.amplification_factor for p in paths), default=1)
            ),
            'paths': [p.to_dict() for p in paths[:10]]  # Top 10 paths
        }
        
        return cascade_info
    
    def generate_summary_narrative(self, timesteps: Optional[Tuple[int, int]] = None) -> str:
        """Generate overall narrative summary."""
        if timesteps:
            events = [e for e in self.explanations 
                     if timesteps[0] <= e.timestamp <= timesteps[1]]
        else:
            events = self.explanations
        
        if not events:
            return "No significant events occurred during the simulation."
        
        # Count by type
        type_counts = defaultdict(int)
        for e in events:
            type_counts[e.event_type.value] += 1
        
        # Find cascade paths
        cascade_paths = self.contagion_tracker.completed_paths
        
        narrative = f"During this period, {len(events)} significant events occurred. "
        
        if type_counts['default'] > 0:
            narrative += f"{type_counts['default']} banks defaulted. "
        
        if type_counts['margin_call'] > 0:
            narrative += f"There were {type_counts['margin_call']} margin calls. "
        
        if cascade_paths:
            max_cascade = max(cascade_paths, key=lambda p: p.length)
            narrative += f"The longest contagion chain involved {max_cascade.length} entities "
            narrative += f"with total losses of {max_cascade.total_loss:,.0f}. "
        
        # Severity assessment
        avg_severity = np.mean([e.severity_score for e in events]) if events else 0
        if avg_severity > 0.7:
            narrative += "Overall severity was HIGH, indicating systemic stress. "
        elif avg_severity > 0.4:
            narrative += "Overall severity was MODERATE. "
        else:
            narrative += "Overall severity was LOW with limited contagion. "
        
        return narrative
    
    def get_all_explanations(self) -> List[Dict[str, Any]]:
        """Get all explanations as dictionaries."""
        return [e.to_dict() for e in self.explanations]
    
    def export_to_json(self, filepath: str) -> None:
        """Export all explanations to JSON."""
        data = {
            'explanations': self.get_all_explanations(),
            'cascade_paths': [p.to_dict() for p in self.contagion_tracker.completed_paths],
            'summary': self.generate_summary_narrative()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.contagion_tracker = ContagionTracker()
        self.causality_analyzer = CausalityAnalyzer()
        self.explanations.clear()
        self.attributions.clear()
