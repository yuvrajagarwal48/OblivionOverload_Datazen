"""
Private Memory Layer for FinSim-MAPPO.
Each bank maintains its own private ledger of interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json


class InteractionType(str, Enum):
    """Types of interactions that can be recorded."""
    LOAN_GIVEN = "loan_given"
    LOAN_RECEIVED = "loan_received"
    LOAN_REPAID = "loan_repaid"
    LOAN_DEFAULTED = "loan_defaulted"
    MARGIN_CALL_MADE = "margin_call_made"
    MARGIN_CALL_RECEIVED = "margin_call_received"
    TRADE_EXECUTED = "trade_executed"
    COLLATERAL_POSTED = "collateral_posted"
    COLLATERAL_RECEIVED = "collateral_received"


@dataclass
class InteractionRecord:
    """
    A single interaction record in the private ledger.
    Contains only information observable by the recording bank.
    """
    interaction_id: str
    timestep: int
    counterparty_id: int
    interaction_type: InteractionType
    
    # Transaction details
    amount: float = 0.0
    interest_rate: float = 0.0
    collateral_amount: float = 0.0
    
    # Outcomes (filled in later when known)
    repayment_delay: int = 0  # Days/steps delayed
    default_occurred: bool = False
    profit_loss: float = 0.0
    
    # Observable context at interaction time
    observed_stress: float = 0.0  # Counterparty stress if visible
    exchange_congestion: float = 0.0
    market_volatility: float = 0.0
    ccp_margin_level: float = 1.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    outcome_recorded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['interaction_type'] = self.interaction_type.value
        return result
    
    @property
    def age(self) -> int:
        """Age in timesteps (requires current timestep context)."""
        return 0  # Computed externally
    
    def compute_quality_score(self) -> float:
        """
        Compute interaction quality from the bank's perspective.
        Higher is better (good counterparty behavior).
        """
        score = 0.5  # Neutral baseline
        
        if self.interaction_type in [InteractionType.LOAN_GIVEN, InteractionType.LOAN_RECEIVED]:
            if self.outcome_recorded:
                # Penalize defaults heavily
                if self.default_occurred:
                    score = 0.0
                else:
                    # Penalize delays
                    delay_penalty = min(self.repayment_delay / 10, 0.4)
                    score = 1.0 - delay_penalty
                    
                    # Bonus for profit
                    if self.profit_loss > 0:
                        score = min(score + 0.1, 1.0)
        
        elif self.interaction_type == InteractionType.MARGIN_CALL_RECEIVED:
            # Being margin called is stressful but not necessarily bad
            score = 0.4
        
        elif self.interaction_type == InteractionType.TRADE_EXECUTED:
            # Neutral unless we have P&L
            if self.profit_loss > 0:
                score = 0.7
            elif self.profit_loss < 0:
                score = 0.3
        
        return score


@dataclass
class CounterpartyLedger:
    """
    All interactions with a single counterparty.
    Maintained privately by each bank.
    """
    counterparty_id: int
    interactions: List[InteractionRecord] = field(default_factory=list)
    
    # Aggregated statistics (updated on each interaction)
    total_interactions: int = 0
    total_volume: float = 0.0
    default_count: int = 0
    average_delay: float = 0.0
    total_profit_loss: float = 0.0
    last_interaction_timestep: int = -1
    
    def add_interaction(self, record: InteractionRecord) -> None:
        """Add a new interaction record."""
        self.interactions.append(record)
        self.total_interactions += 1
        self.total_volume += record.amount
        self.last_interaction_timestep = record.timestep
        
        if record.outcome_recorded:
            self._update_statistics(record)
    
    def record_outcome(self, interaction_id: str, 
                       default_occurred: bool = False,
                       repayment_delay: int = 0,
                       profit_loss: float = 0.0) -> None:
        """Record outcome for a pending interaction."""
        for record in self.interactions:
            if record.interaction_id == interaction_id and not record.outcome_recorded:
                record.default_occurred = default_occurred
                record.repayment_delay = repayment_delay
                record.profit_loss = profit_loss
                record.outcome_recorded = True
                self._update_statistics(record)
                break
    
    def _update_statistics(self, record: InteractionRecord) -> None:
        """Update aggregated statistics from outcome."""
        if record.default_occurred:
            self.default_count += 1
        
        # Running average of delay
        n = sum(1 for r in self.interactions if r.outcome_recorded)
        if n > 0:
            total_delay = sum(r.repayment_delay for r in self.interactions if r.outcome_recorded)
            self.average_delay = total_delay / n
        
        self.total_profit_loss += record.profit_loss
    
    def get_recent_interactions(self, n: int = 10) -> List[InteractionRecord]:
        """Get n most recent interactions."""
        return sorted(self.interactions, key=lambda x: x.timestep, reverse=True)[:n]
    
    def compute_default_rate(self) -> float:
        """Compute observed default rate."""
        eligible = [r for r in self.interactions 
                   if r.interaction_type in [InteractionType.LOAN_GIVEN] 
                   and r.outcome_recorded]
        if not eligible:
            return 0.0
        return sum(1 for r in eligible if r.default_occurred) / len(eligible)
    
    def compute_average_quality(self, current_timestep: int, 
                                 decay_rate: float = 0.1) -> float:
        """
        Compute time-weighted average interaction quality.
        Recent interactions matter more.
        """
        if not self.interactions:
            return 0.5  # Neutral prior
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for record in self.interactions:
            age = current_timestep - record.timestep
            weight = np.exp(-decay_rate * age)  # Exponential decay
            quality = record.compute_quality_score()
            
            weighted_sum += weight * quality
            weight_total += weight
        
        if weight_total > 0:
            return weighted_sum / weight_total
        return 0.5


class MemoryRetentionPolicy:
    """
    Defines how memory decays and is pruned over time.
    """
    
    def __init__(self,
                 max_records_per_counterparty: int = 100,
                 time_decay_rate: float = 0.05,
                 min_relevance_threshold: float = 0.01,
                 importance_weights: Optional[Dict[InteractionType, float]] = None):
        self.max_records = max_records_per_counterparty
        self.decay_rate = time_decay_rate
        self.min_relevance = min_relevance_threshold
        
        # Default importance weights
        self.importance_weights = importance_weights or {
            InteractionType.LOAN_DEFAULTED: 2.0,
            InteractionType.LOAN_GIVEN: 1.0,
            InteractionType.LOAN_RECEIVED: 1.0,
            InteractionType.LOAN_REPAID: 0.8,
            InteractionType.MARGIN_CALL_MADE: 1.2,
            InteractionType.MARGIN_CALL_RECEIVED: 1.2,
            InteractionType.TRADE_EXECUTED: 0.5,
            InteractionType.COLLATERAL_POSTED: 0.6,
            InteractionType.COLLATERAL_RECEIVED: 0.6
        }
    
    def compute_relevance(self, record: InteractionRecord, 
                          current_timestep: int) -> float:
        """Compute relevance score for a record."""
        age = current_timestep - record.timestep
        time_factor = np.exp(-self.decay_rate * age)
        
        importance = self.importance_weights.get(record.interaction_type, 1.0)
        
        # Boost relevance for significant outcomes
        outcome_boost = 1.0
        if record.default_occurred:
            outcome_boost = 2.0
        elif abs(record.profit_loss) > record.amount * 0.1:
            outcome_boost = 1.5
        
        return time_factor * importance * outcome_boost
    
    def prune_ledger(self, ledger: CounterpartyLedger, 
                     current_timestep: int) -> List[InteractionRecord]:
        """
        Prune low-relevance records from ledger.
        Returns removed records for logging.
        """
        if len(ledger.interactions) <= self.max_records:
            return []
        
        # Compute relevance for all records
        scored = [(r, self.compute_relevance(r, current_timestep)) 
                  for r in ledger.interactions]
        
        # Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top records
        kept = scored[:self.max_records]
        removed = scored[self.max_records:]
        
        # Also remove anything below threshold
        kept = [(r, s) for r, s in kept if s >= self.min_relevance]
        removed.extend([(r, s) for r, s in scored if s < self.min_relevance 
                       and (r, s) not in removed])
        
        ledger.interactions = [r for r, s in kept]
        
        return [r for r, s in removed]


class PrivateMemory:
    """
    Complete private memory store for a single bank.
    Not accessible by other banks or global systems.
    """
    
    def __init__(self, 
                 bank_id: int,
                 retention_policy: Optional[MemoryRetentionPolicy] = None):
        self.bank_id = bank_id
        self.retention_policy = retention_policy or MemoryRetentionPolicy()
        
        # Private ledgers per counterparty
        self.counterparty_ledgers: Dict[int, CounterpartyLedger] = {}
        
        # Self-observations (own state history)
        self.own_state_history: List[Dict[str, float]] = []
        
        # Interaction counter for IDs
        self._interaction_counter = 0
        
        # Current timestep tracking
        self.current_timestep = 0
    
    def _generate_interaction_id(self) -> str:
        """Generate unique interaction ID."""
        self._interaction_counter += 1
        return f"{self.bank_id}_{self.current_timestep}_{self._interaction_counter}"
    
    def get_or_create_ledger(self, counterparty_id: int) -> CounterpartyLedger:
        """Get or create ledger for a counterparty."""
        if counterparty_id not in self.counterparty_ledgers:
            self.counterparty_ledgers[counterparty_id] = CounterpartyLedger(
                counterparty_id=counterparty_id
            )
        return self.counterparty_ledgers[counterparty_id]
    
    def record_interaction(self,
                           counterparty_id: int,
                           interaction_type: InteractionType,
                           amount: float = 0.0,
                           interest_rate: float = 0.0,
                           collateral_amount: float = 0.0,
                           observed_stress: float = 0.0,
                           exchange_congestion: float = 0.0,
                           market_volatility: float = 0.0,
                           ccp_margin_level: float = 1.0) -> str:
        """
        Record a new interaction in the private ledger.
        Returns interaction ID for outcome tracking.
        """
        interaction_id = self._generate_interaction_id()
        
        record = InteractionRecord(
            interaction_id=interaction_id,
            timestep=self.current_timestep,
            counterparty_id=counterparty_id,
            interaction_type=interaction_type,
            amount=amount,
            interest_rate=interest_rate,
            collateral_amount=collateral_amount,
            observed_stress=observed_stress,
            exchange_congestion=exchange_congestion,
            market_volatility=market_volatility,
            ccp_margin_level=ccp_margin_level
        )
        
        ledger = self.get_or_create_ledger(counterparty_id)
        ledger.add_interaction(record)
        
        return interaction_id
    
    def record_outcome(self,
                       counterparty_id: int,
                       interaction_id: str,
                       default_occurred: bool = False,
                       repayment_delay: int = 0,
                       profit_loss: float = 0.0) -> None:
        """Record outcome for a previous interaction."""
        if counterparty_id in self.counterparty_ledgers:
            self.counterparty_ledgers[counterparty_id].record_outcome(
                interaction_id=interaction_id,
                default_occurred=default_occurred,
                repayment_delay=repayment_delay,
                profit_loss=profit_loss
            )
    
    def record_own_state(self, state: Dict[str, float]) -> None:
        """Record own state for self-awareness."""
        state_with_time = state.copy()
        state_with_time['timestep'] = self.current_timestep
        self.own_state_history.append(state_with_time)
        
        # Keep limited history
        if len(self.own_state_history) > 100:
            self.own_state_history = self.own_state_history[-100:]
    
    def advance_timestep(self, new_timestep: int) -> None:
        """Advance to new timestep and apply retention policy."""
        self.current_timestep = new_timestep
        
        # Prune old records
        for cp_id, ledger in self.counterparty_ledgers.items():
            self.retention_policy.prune_ledger(ledger, new_timestep)
    
    def get_counterparty_history(self, counterparty_id: int) -> Optional[CounterpartyLedger]:
        """Get history with a specific counterparty."""
        return self.counterparty_ledgers.get(counterparty_id)
    
    def get_interaction_counts(self) -> Dict[int, int]:
        """Get interaction counts per counterparty."""
        return {cp_id: ledger.total_interactions 
                for cp_id, ledger in self.counterparty_ledgers.items()}
    
    def get_counterparty_quality_scores(self) -> Dict[int, float]:
        """Get quality scores for all counterparties."""
        return {cp_id: ledger.compute_average_quality(self.current_timestep, 
                                                       self.retention_policy.decay_rate)
                for cp_id, ledger in self.counterparty_ledgers.items()}
    
    def get_default_rates(self) -> Dict[int, float]:
        """Get observed default rates for all counterparties."""
        return {cp_id: ledger.compute_default_rate()
                for cp_id, ledger in self.counterparty_ledgers.items()}
    
    def has_history_with(self, counterparty_id: int) -> bool:
        """Check if any history exists with counterparty."""
        return counterparty_id in self.counterparty_ledgers
    
    def get_total_exposure_history(self, counterparty_id: int) -> float:
        """Get total historical volume with counterparty."""
        if counterparty_id in self.counterparty_ledgers:
            return self.counterparty_ledgers[counterparty_id].total_volume
        return 0.0
    
    def get_recent_interactions_all(self, n: int = 20) -> List[InteractionRecord]:
        """Get n most recent interactions across all counterparties."""
        all_records = []
        for ledger in self.counterparty_ledgers.values():
            all_records.extend(ledger.interactions)
        
        return sorted(all_records, key=lambda x: x.timestep, reverse=True)[:n]
    
    def compute_memory_features(self, counterparty_id: int) -> Dict[str, float]:
        """
        Compute features from memory for belief formation.
        Returns features derived purely from private experience.
        """
        if counterparty_id not in self.counterparty_ledgers:
            return {
                'has_history': 0.0,
                'interaction_count': 0.0,
                'observed_default_rate': 0.0,
                'average_delay': 0.0,
                'quality_score': 0.5,
                'total_volume': 0.0,
                'recency': 0.0,
                'profit_ratio': 0.0
            }
        
        ledger = self.counterparty_ledgers[counterparty_id]
        
        # Recency: how recently we interacted
        if ledger.last_interaction_timestep >= 0:
            recency = np.exp(-0.1 * (self.current_timestep - ledger.last_interaction_timestep))
        else:
            recency = 0.0
        
        # Profit ratio
        if ledger.total_volume > 0:
            profit_ratio = ledger.total_profit_loss / ledger.total_volume
        else:
            profit_ratio = 0.0
        
        return {
            'has_history': 1.0,
            'interaction_count': min(ledger.total_interactions / 20, 1.0),  # Normalized
            'observed_default_rate': ledger.compute_default_rate(),
            'average_delay': min(ledger.average_delay / 10, 1.0),  # Normalized
            'quality_score': ledger.compute_average_quality(
                self.current_timestep, self.retention_policy.decay_rate
            ),
            'total_volume': np.log1p(ledger.total_volume) / 20,  # Log-normalized
            'recency': recency,
            'profit_ratio': np.clip(profit_ratio, -1, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory for persistence."""
        return {
            'bank_id': self.bank_id,
            'current_timestep': self.current_timestep,
            'counterparty_ledgers': {
                str(cp_id): {
                    'counterparty_id': ledger.counterparty_id,
                    'interactions': [r.to_dict() for r in ledger.interactions],
                    'total_interactions': ledger.total_interactions,
                    'total_volume': ledger.total_volume,
                    'default_count': ledger.default_count,
                    'average_delay': ledger.average_delay,
                    'total_profit_loss': ledger.total_profit_loss
                }
                for cp_id, ledger in self.counterparty_ledgers.items()
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save memory to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MemoryManager:
    """
    Manages private memories for all banks.
    Ensures information isolation between banks.
    """
    
    def __init__(self, retention_policy: Optional[MemoryRetentionPolicy] = None):
        self.retention_policy = retention_policy or MemoryRetentionPolicy()
        self.bank_memories: Dict[int, PrivateMemory] = {}
    
    def get_or_create_memory(self, bank_id: int) -> PrivateMemory:
        """Get or create private memory for a bank."""
        if bank_id not in self.bank_memories:
            self.bank_memories[bank_id] = PrivateMemory(
                bank_id=bank_id,
                retention_policy=self.retention_policy
            )
        return self.bank_memories[bank_id]
    
    def advance_all(self, new_timestep: int) -> None:
        """Advance all memories to new timestep."""
        for memory in self.bank_memories.values():
            memory.advance_timestep(new_timestep)
    
    def record_bilateral_interaction(self,
                                      bank_a: int,
                                      bank_b: int,
                                      interaction_type_a: InteractionType,
                                      interaction_type_b: InteractionType,
                                      amount: float,
                                      **context) -> Tuple[str, str]:
        """
        Record a bilateral interaction from both perspectives.
        Returns interaction IDs for both sides.
        """
        memory_a = self.get_or_create_memory(bank_a)
        memory_b = self.get_or_create_memory(bank_b)
        
        id_a = memory_a.record_interaction(
            counterparty_id=bank_b,
            interaction_type=interaction_type_a,
            amount=amount,
            **context
        )
        
        id_b = memory_b.record_interaction(
            counterparty_id=bank_a,
            interaction_type=interaction_type_b,
            amount=amount,
            **context
        )
        
        return id_a, id_b
    
    def get_all_memories(self) -> Dict[int, PrivateMemory]:
        """Get all bank memories (for serialization only)."""
        return self.bank_memories
