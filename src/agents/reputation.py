"""
Reputation and Trust Dynamics for FinSim-MAPPO.
Emergent reputation signals from aggregated beliefs.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from enum import Enum


class ReputationTier(str, Enum):
    """Reputation tier classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"
    UNKNOWN = "unknown"


@dataclass
class ReputationScore:
    """Public reputation score for an entity."""
    entity_id: int
    
    # Core reputation metrics
    reliability_score: float = 0.5      # Aggregated reliability perception
    trustworthiness: float = 0.5        # Aggregated trust
    risk_perception: float = 0.1        # How risky others perceive this entity
    
    # Network-level metrics
    centrality_reputation: float = 0.5  # Reputation weighted by network position
    peer_confidence: float = 0.5        # How confident peers are in their beliefs
    
    # Tier classification
    tier: ReputationTier = ReputationTier.UNKNOWN
    
    # Metadata
    sample_size: int = 0                # Number of belief sources
    last_updated: int = 0
    volatility: float = 0.0             # How much reputation fluctuates
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['tier'] = self.tier.value
        return result
    
    @staticmethod
    def compute_tier(reliability: float, risk: float) -> ReputationTier:
        """Compute tier from metrics."""
        if reliability > 0.8 and risk < 0.05:
            return ReputationTier.EXCELLENT
        elif reliability > 0.6 and risk < 0.1:
            return ReputationTier.GOOD
        elif reliability > 0.4 and risk < 0.2:
            return ReputationTier.AVERAGE
        elif reliability > 0.2:
            return ReputationTier.POOR
        else:
            return ReputationTier.VERY_POOR


@dataclass
class TrustRelationship:
    """Bilateral trust relationship."""
    from_entity: int
    to_entity: int
    
    trust_score: float = 0.5
    confidence: float = 0.0
    interaction_count: int = 0
    last_interaction: int = -1
    
    # Trust history
    trust_history: List[Tuple[int, float]] = field(default_factory=list)
    
    def update_trust(self, new_trust: float, timestep: int) -> None:
        """Update trust score."""
        self.trust_score = new_trust
        self.last_interaction = timestep
        self.interaction_count += 1
        
        # Record history
        self.trust_history.append((timestep, new_trust))
        if len(self.trust_history) > 50:
            self.trust_history = self.trust_history[-50:]
    
    def get_trust_trend(self) -> str:
        """Get recent trust trend."""
        if len(self.trust_history) < 3:
            return "stable"
        
        recent = [t for _, t in self.trust_history[-5:]]
        older = [t for _, t in self.trust_history[-10:-5]] if len(self.trust_history) >= 10 else [0.5]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        return "stable"


class TrustNetwork:
    """
    Network of trust relationships between entities.
    """
    
    def __init__(self):
        self.relationships: Dict[Tuple[int, int], TrustRelationship] = {}
        self._entity_ids: Set[int] = set()
    
    def get_or_create_relationship(self, from_id: int, to_id: int) -> TrustRelationship:
        """Get or create trust relationship."""
        key = (from_id, to_id)
        if key not in self.relationships:
            self.relationships[key] = TrustRelationship(
                from_entity=from_id,
                to_entity=to_id
            )
            self._entity_ids.add(from_id)
            self._entity_ids.add(to_id)
        return self.relationships[key]
    
    def update_trust(self, from_id: int, to_id: int, 
                     trust: float, confidence: float, timestep: int) -> None:
        """Update trust from one entity to another."""
        rel = self.get_or_create_relationship(from_id, to_id)
        rel.update_trust(trust, timestep)
        rel.confidence = confidence
    
    def get_trust(self, from_id: int, to_id: int) -> float:
        """Get trust score from one entity to another."""
        key = (from_id, to_id)
        if key in self.relationships:
            return self.relationships[key].trust_score
        return 0.5  # Default neutral trust
    
    def get_incoming_trust(self, entity_id: int) -> Dict[int, float]:
        """Get all trust directed toward an entity."""
        incoming = {}
        for (from_id, to_id), rel in self.relationships.items():
            if to_id == entity_id:
                incoming[from_id] = rel.trust_score
        return incoming
    
    def get_outgoing_trust(self, entity_id: int) -> Dict[int, float]:
        """Get all trust from an entity."""
        outgoing = {}
        for (from_id, to_id), rel in self.relationships.items():
            if from_id == entity_id:
                outgoing[to_id] = rel.trust_score
        return outgoing
    
    def get_average_incoming_trust(self, entity_id: int) -> float:
        """Get average trust from all counterparties."""
        incoming = self.get_incoming_trust(entity_id)
        if incoming:
            return np.mean(list(incoming.values()))
        return 0.5
    
    def get_trust_matrix(self) -> Tuple[List[int], np.ndarray]:
        """Get trust as a matrix."""
        entity_ids = sorted(self._entity_ids)
        n = len(entity_ids)
        
        matrix = np.full((n, n), 0.5)  # Default neutral
        
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        
        for (from_id, to_id), rel in self.relationships.items():
            if from_id in id_to_idx and to_id in id_to_idx:
                i, j = id_to_idx[from_id], id_to_idx[to_id]
                matrix[i, j] = rel.trust_score
        
        return entity_ids, matrix
    
    def compute_pagerank_trust(self, damping: float = 0.85, 
                                iterations: int = 20) -> Dict[int, float]:
        """
        Compute PageRank-style trust centrality.
        Entities trusted by trustworthy entities have higher scores.
        """
        entity_ids, trust_matrix = self.get_trust_matrix()
        n = len(entity_ids)
        
        if n == 0:
            return {}
        
        # Normalize trust matrix (row-stochastic)
        row_sums = trust_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        P = trust_matrix / row_sums
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Power iteration
        for _ in range(iterations):
            scores = (1 - damping) / n + damping * P.T @ scores
        
        return {entity_ids[i]: scores[i] for i in range(n)}


class ReputationAggregator:
    """
    Aggregates individual beliefs into public reputation signals.
    """
    
    def __init__(self, belief_managers: Dict[int, Any]):
        """
        Args:
            belief_managers: Dict mapping bank_id to BeliefSystem
        """
        self.belief_managers = belief_managers
        self.trust_network = TrustNetwork()
        
        # Cached reputation scores
        self._reputation_cache: Dict[int, ReputationScore] = {}
        self._cache_timestep: int = -1
    
    def update_from_beliefs(self, timestep: int) -> None:
        """Update trust network from all belief systems."""
        for bank_id, belief_system in self.belief_managers.items():
            for cp_id, belief in belief_system.get_all_beliefs().items():
                self.trust_network.update_trust(
                    from_id=bank_id,
                    to_id=cp_id,
                    trust=belief.trust_score,
                    confidence=belief.confidence_score,
                    timestep=timestep
                )
    
    def compute_reputation(self, entity_id: int, timestep: int) -> ReputationScore:
        """Compute public reputation for an entity."""
        # Gather all beliefs about this entity
        incoming_beliefs = []
        
        for bank_id, belief_system in self.belief_managers.items():
            if bank_id == entity_id:
                continue  # Skip self-assessment
            
            beliefs = belief_system.get_all_beliefs()
            if entity_id in beliefs:
                b = beliefs[entity_id]
                incoming_beliefs.append({
                    'from': bank_id,
                    'trust': b.trust_score,
                    'reliability': b.reliability_score,
                    'pd': b.estimated_pd,
                    'confidence': b.confidence_score
                })
        
        if not incoming_beliefs:
            return ReputationScore(
                entity_id=entity_id,
                tier=ReputationTier.UNKNOWN,
                sample_size=0,
                last_updated=timestep
            )
        
        # Aggregate beliefs (confidence-weighted)
        total_confidence = sum(b['confidence'] for b in incoming_beliefs)
        
        if total_confidence > 0:
            weights = [b['confidence'] / total_confidence for b in incoming_beliefs]
        else:
            weights = [1.0 / len(incoming_beliefs)] * len(incoming_beliefs)
        
        reliability = sum(w * b['reliability'] for w, b in zip(weights, incoming_beliefs))
        trust = sum(w * b['trust'] for w, b in zip(weights, incoming_beliefs))
        risk = sum(w * b['pd'] for w, b in zip(weights, incoming_beliefs))
        confidence = np.mean([b['confidence'] for b in incoming_beliefs])
        
        # Compute volatility (std of beliefs)
        volatility = np.std([b['trust'] for b in incoming_beliefs])
        
        tier = ReputationScore.compute_tier(reliability, risk)
        
        return ReputationScore(
            entity_id=entity_id,
            reliability_score=reliability,
            trustworthiness=trust,
            risk_perception=risk,
            peer_confidence=confidence,
            tier=tier,
            sample_size=len(incoming_beliefs),
            last_updated=timestep,
            volatility=volatility
        )
    
    def compute_all_reputations(self, timestep: int) -> Dict[int, ReputationScore]:
        """Compute reputation for all known entities."""
        if timestep == self._cache_timestep:
            return self._reputation_cache
        
        # Get all entity IDs
        all_entities: Set[int] = set()
        for belief_system in self.belief_managers.values():
            all_entities.update(belief_system.get_all_beliefs().keys())
        all_entities.update(self.belief_managers.keys())
        
        reputations = {}
        for entity_id in all_entities:
            reputations[entity_id] = self.compute_reputation(entity_id, timestep)
        
        self._reputation_cache = reputations
        self._cache_timestep = timestep
        
        return reputations
    
    def get_system_trust_index(self, timestep: int) -> float:
        """Compute system-wide trust index."""
        reputations = self.compute_all_reputations(timestep)
        
        if not reputations:
            return 0.5
        
        # Weighted average by sample size
        total_samples = sum(r.sample_size for r in reputations.values())
        if total_samples == 0:
            return 0.5
        
        weighted_trust = sum(r.trustworthiness * r.sample_size 
                            for r in reputations.values())
        return weighted_trust / total_samples
    
    def get_reputation_distribution(self, timestep: int) -> Dict[ReputationTier, int]:
        """Get distribution of entities across reputation tiers."""
        reputations = self.compute_all_reputations(timestep)
        
        distribution = {tier: 0 for tier in ReputationTier}
        for rep in reputations.values():
            distribution[rep.tier] += 1
        
        return distribution


class PublicSignalGenerator:
    """
    Generates public signals that can be observed by all banks.
    These supplement private beliefs.
    """
    
    def __init__(self, reputation_aggregator: ReputationAggregator):
        self.aggregator = reputation_aggregator
        self.signal_history: List[Dict[str, Any]] = []
    
    def generate_signals(self, timestep: int) -> Dict[str, Any]:
        """Generate public signals for the current timestep."""
        reputations = self.aggregator.compute_all_reputations(timestep)
        
        # System trust index
        system_trust = self.aggregator.get_system_trust_index(timestep)
        
        # Reputation distribution
        distribution = self.aggregator.get_reputation_distribution(timestep)
        
        # High-risk entities (public warning)
        high_risk = [
            rep.entity_id for rep in reputations.values()
            if rep.risk_perception > 0.2 and rep.sample_size >= 3
        ]
        
        # Low-trust entities
        low_trust = [
            rep.entity_id for rep in reputations.values()
            if rep.trustworthiness < 0.3 and rep.sample_size >= 3
        ]
        
        # Volatility warning
        high_volatility = [
            rep.entity_id for rep in reputations.values()
            if rep.volatility > 0.3
        ]
        
        # PageRank trust centrality
        pagerank = self.aggregator.trust_network.compute_pagerank_trust()
        
        signals = {
            'timestep': timestep,
            'system_trust_index': system_trust,
            'reputation_distribution': {k.value: v for k, v in distribution.items()},
            'high_risk_entities': high_risk,
            'low_trust_entities': low_trust,
            'high_volatility_entities': high_volatility,
            'trust_centrality': pagerank,
            'average_peer_confidence': np.mean([r.peer_confidence for r in reputations.values()]) if reputations else 0.5
        }
        
        self.signal_history.append(signals)
        
        return signals
    
    def get_entity_public_profile(self, entity_id: int, timestep: int) -> Dict[str, Any]:
        """Get public profile for an entity."""
        reputations = self.aggregator.compute_all_reputations(timestep)
        
        if entity_id not in reputations:
            return {
                'entity_id': entity_id,
                'known': False,
                'tier': 'unknown'
            }
        
        rep = reputations[entity_id]
        pagerank = self.aggregator.trust_network.compute_pagerank_trust()
        
        return {
            'entity_id': entity_id,
            'known': True,
            'tier': rep.tier.value,
            'reliability_score': rep.reliability_score,
            'trustworthiness': rep.trustworthiness,
            'risk_perception': rep.risk_perception,
            'trust_centrality': pagerank.get(entity_id, 0.5),
            'peer_confidence': rep.peer_confidence,
            'sample_size': rep.sample_size,
            'volatility': rep.volatility
        }
    
    def get_trust_trend(self, lookback: int = 10) -> str:
        """Get system-wide trust trend."""
        if len(self.signal_history) < 2:
            return "stable"
        
        recent = self.signal_history[-lookback:]
        trust_values = [s['system_trust_index'] for s in recent]
        
        if len(trust_values) < 2:
            return "stable"
        
        # Linear regression slope
        x = np.arange(len(trust_values))
        slope = np.polyfit(x, trust_values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        return "stable"


class TrustDynamicsEngine:
    """
    Main engine for trust and reputation dynamics.
    Integrates with the belief system.
    """
    
    def __init__(self, belief_managers: Dict[int, Any]):
        self.belief_managers = belief_managers
        self.aggregator = ReputationAggregator(belief_managers)
        self.signal_generator = PublicSignalGenerator(self.aggregator)
        
        self.current_timestep = 0
    
    def update(self, timestep: int) -> Dict[str, Any]:
        """Update trust dynamics for new timestep."""
        self.current_timestep = timestep
        
        # Update trust network from beliefs
        self.aggregator.update_from_beliefs(timestep)
        
        # Generate public signals
        signals = self.signal_generator.generate_signals(timestep)
        
        return signals
    
    def get_reputation(self, entity_id: int) -> ReputationScore:
        """Get entity's current reputation."""
        return self.aggregator.compute_reputation(entity_id, self.current_timestep)
    
    def get_all_reputations(self) -> Dict[int, ReputationScore]:
        """Get all reputations."""
        return self.aggregator.compute_all_reputations(self.current_timestep)
    
    def get_trust_between(self, from_id: int, to_id: int) -> float:
        """Get trust from one entity to another."""
        return self.aggregator.trust_network.get_trust(from_id, to_id)
    
    def get_public_signals(self) -> Dict[str, Any]:
        """Get current public signals."""
        return self.signal_generator.generate_signals(self.current_timestep)
    
    def get_entity_profile(self, entity_id: int) -> Dict[str, Any]:
        """Get entity's public profile."""
        return self.signal_generator.get_entity_public_profile(
            entity_id, self.current_timestep
        )
    
    def get_trust_network_data(self) -> Dict[str, Any]:
        """Get trust network for visualization."""
        entity_ids, matrix = self.aggregator.trust_network.get_trust_matrix()
        
        nodes = []
        for eid in entity_ids:
            rep = self.get_reputation(eid)
            nodes.append({
                'id': eid,
                'tier': rep.tier.value,
                'trustworthiness': rep.trustworthiness
            })
        
        edges = []
        for (from_id, to_id), rel in self.aggregator.trust_network.relationships.items():
            edges.append({
                'from': from_id,
                'to': to_id,
                'trust': rel.trust_score,
                'confidence': rel.confidence
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'matrix': matrix.tolist() if len(matrix) > 0 else []
        }
    
    def detect_herding(self, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect belief herding (convergence of beliefs).
        Returns warning if beliefs are too synchronized.
        """
        all_beliefs_about: Dict[int, List[float]] = defaultdict(list)
        
        for bank_id, belief_system in self.belief_managers.items():
            for cp_id, belief in belief_system.get_all_beliefs().items():
                all_beliefs_about[cp_id].append(belief.estimated_pd)
        
        herding_scores = {}
        for cp_id, pds in all_beliefs_about.items():
            if len(pds) >= 3:
                # Low variance = high herding
                variance = np.var(pds)
                herding_scores[cp_id] = 1.0 - min(variance / 0.1, 1.0)
        
        avg_herding = np.mean(list(herding_scores.values())) if herding_scores else 0
        
        return {
            'herding_detected': avg_herding > 0.8,
            'average_herding': avg_herding,
            'entity_herding': herding_scores,
            'warning': "High belief synchronization detected" if avg_herding > 0.8 else None
        }
