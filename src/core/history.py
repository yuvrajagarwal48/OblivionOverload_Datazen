"""
Centralized History Repository for FinSim-MAPPO.
Authoritative source for all simulation data, events, and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import pickle
import gzip
from collections import defaultdict

from .state_capture import (
    StateCapture, BankSnapshot, ExchangeSnapshot, 
    CCPSnapshot, MarketSnapshot, FlowRecord
)


@dataclass
class ClearingOutcome:
    """Record of a clearing event."""
    timestep: int
    timestamp: str
    
    # Clearing details
    algorithm: str  # 'eisenberg_noe', 'proportional', etc.
    iterations: int
    converged: bool
    
    # Payments
    total_claims: float
    total_payments: float
    clearing_ratio: float
    
    # Shortfalls
    total_shortfall: float
    banks_with_shortfall: List[int]
    
    # Recovery
    recovery_rates: Dict[int, float]
    avg_recovery_rate: float


@dataclass
class MarginCallEvent:
    """Record of a margin call."""
    event_id: str
    timestep: int
    timestamp: str
    
    ccp_id: int
    member_id: int
    
    margin_type: str  # 'initial', 'variation', 'intraday'
    amount_required: float
    amount_posted: float
    shortfall: float
    
    deadline: int
    status: str  # 'issued', 'met', 'partially_met', 'failed'
    
    consequence: Optional[str]  # 'none', 'warning', 'position_liquidation', 'default'


@dataclass
class DefaultEvent:
    """Record of a default event."""
    event_id: str
    timestep: int
    timestamp: str
    
    entity_id: int
    entity_type: str  # 'bank', 'ccp_member'
    
    # Cause
    trigger: str  # 'capital_ratio', 'liquidity', 'margin_failure', 'cascade'
    trigger_value: float
    threshold: float
    
    # Impact
    liabilities_at_default: float
    assets_at_default: float
    loss_given_default: float
    
    # Contagion
    directly_affected: List[int]
    exposure_amounts: Dict[int, float]
    
    # Resolution
    resolution_method: str  # 'liquidation', 'waterfall', 'bailout'
    resolution_recovery: float


@dataclass
class TimestepRecord:
    """Complete record for a single timestep."""
    timestep: int
    
    # Snapshots
    bank_snapshots: Dict[int, BankSnapshot]
    exchange_snapshots: Dict[int, ExchangeSnapshot]
    ccp_snapshots: Dict[int, CCPSnapshot]
    market_snapshot: MarketSnapshot
    
    # Events
    clearing_outcomes: List[ClearingOutcome]
    margin_calls: List[MarginCallEvent]
    defaults: List[DefaultEvent]
    
    # Flows
    flows: List[FlowRecord]
    
    # Aggregate metrics
    system_metrics: Dict[str, float]
    
    # Actions and rewards (for RL)
    actions: Optional[Dict[int, np.ndarray]] = None
    rewards: Optional[Dict[int, float]] = None


class SimulationHistory:
    """
    Centralized repository for all simulation data.
    
    This is the authoritative source for:
    - UI rendering
    - Counterfactual analysis
    - Explanation generation
    - Evaluation reports
    """
    
    def __init__(self, 
                 simulation_id: str = None,
                 config: Dict[str, Any] = None):
        
        self.simulation_id = simulation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = config or {}
        self.created_at = datetime.now().isoformat()
        
        # State capture
        self.state_capture = StateCapture()
        
        # Timeline
        self.timestep_records: Dict[int, TimestepRecord] = {}
        self.current_timestep: int = 0
        
        # Event indices (for fast lookup)
        self._default_events: List[DefaultEvent] = []
        self._margin_calls: List[MarginCallEvent] = []
        self._clearing_outcomes: List[ClearingOutcome] = []
        
        # Aggregated time series (pre-computed for UI)
        self._price_series: List[float] = []
        self._default_count_series: List[int] = []
        self._stress_count_series: List[int] = []
        self._total_exposure_series: List[float] = []
        self._avg_capital_series: List[float] = []
        
        # Counters
        self._event_counter = 0
    
    def _next_event_id(self, prefix: str) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"{prefix}_{self.current_timestep:05d}_{self._event_counter:06d}"
    
    def begin_timestep(self, timestep: int) -> None:
        """Begin recording for a new timestep."""
        self.current_timestep = timestep
    
    def record_state(self,
                     banks: Dict[int, Any],
                     exchanges: List[Any],
                     ccps: List[Any],
                     market: Any,
                     network: Any) -> Dict[str, Any]:
        """Record complete system state for current timestep."""
        return self.state_capture.capture_all(
            banks, exchanges, ccps, market, network, self.current_timestep
        )
    
    def record_clearing(self,
                        algorithm: str,
                        iterations: int,
                        converged: bool,
                        total_claims: float,
                        total_payments: float,
                        shortfalls: Dict[int, float],
                        recovery_rates: Dict[int, float]) -> ClearingOutcome:
        """Record clearing mechanism outcome."""
        outcome = ClearingOutcome(
            timestep=self.current_timestep,
            timestamp=datetime.now().isoformat(),
            algorithm=algorithm,
            iterations=iterations,
            converged=converged,
            total_claims=total_claims,
            total_payments=total_payments,
            clearing_ratio=total_payments / max(total_claims, 1e-8),
            total_shortfall=sum(shortfalls.values()),
            banks_with_shortfall=[bid for bid, s in shortfalls.items() if s > 0],
            recovery_rates=recovery_rates,
            avg_recovery_rate=np.mean(list(recovery_rates.values())) if recovery_rates else 1.0
        )
        
        self._clearing_outcomes.append(outcome)
        return outcome
    
    def record_margin_call(self,
                           ccp_id: int,
                           member_id: int,
                           margin_type: str,
                           amount_required: float,
                           amount_posted: float,
                           deadline: int) -> MarginCallEvent:
        """Record margin call event."""
        shortfall = max(0, amount_required - amount_posted)
        status = 'met' if shortfall == 0 else ('partially_met' if amount_posted > 0 else 'issued')
        
        event = MarginCallEvent(
            event_id=self._next_event_id("MC"),
            timestep=self.current_timestep,
            timestamp=datetime.now().isoformat(),
            ccp_id=ccp_id,
            member_id=member_id,
            margin_type=margin_type,
            amount_required=amount_required,
            amount_posted=amount_posted,
            shortfall=shortfall,
            deadline=deadline,
            status=status,
            consequence=None
        )
        
        self._margin_calls.append(event)
        return event
    
    def record_default(self,
                       entity_id: int,
                       entity_type: str,
                       trigger: str,
                       trigger_value: float,
                       threshold: float,
                       liabilities: float,
                       assets: float,
                       affected_entities: Dict[int, float]) -> DefaultEvent:
        """Record default event."""
        lgd = max(0, liabilities - assets)
        
        event = DefaultEvent(
            event_id=self._next_event_id("DEF"),
            timestep=self.current_timestep,
            timestamp=datetime.now().isoformat(),
            entity_id=entity_id,
            entity_type=entity_type,
            trigger=trigger,
            trigger_value=trigger_value,
            threshold=threshold,
            liabilities_at_default=liabilities,
            assets_at_default=assets,
            loss_given_default=lgd,
            directly_affected=list(affected_entities.keys()),
            exposure_amounts=affected_entities,
            resolution_method='liquidation',
            resolution_recovery=0.0
        )
        
        self._default_events.append(event)
        return event
    
    def record_flow(self, **kwargs) -> FlowRecord:
        """Record a flow between entities."""
        kwargs['timestep'] = self.current_timestep
        return self.state_capture.record_flow(**kwargs)
    
    def end_timestep(self,
                     actions: Dict[int, np.ndarray] = None,
                     rewards: Dict[int, float] = None,
                     system_metrics: Dict[str, float] = None) -> TimestepRecord:
        """Complete recording for current timestep."""
        t = self.current_timestep
        
        # Get snapshots for this timestep
        bank_snapshots = {
            bid: snaps[-1] for bid, snaps in self.state_capture.bank_snapshots.items()
            if snaps and snaps[-1].timestep == t
        }
        exchange_snapshots = {
            eid: snaps[-1] for eid, snaps in self.state_capture.exchange_snapshots.items()
            if snaps and snaps[-1].timestep == t
        }
        ccp_snapshots = {
            cid: snaps[-1] for cid, snaps in self.state_capture.ccp_snapshots.items()
            if snaps and snaps[-1].timestep == t
        }
        market_snapshot = self.state_capture.market_snapshots[-1] if self.state_capture.market_snapshots else None
        
        # Get events for this timestep
        clearing = [c for c in self._clearing_outcomes if c.timestep == t]
        margins = [m for m in self._margin_calls if m.timestep == t]
        defaults = [d for d in self._default_events if d.timestep == t]
        flows = [f for f in self.state_capture.flow_records if f.timestep == t]
        
        # Create record
        record = TimestepRecord(
            timestep=t,
            bank_snapshots=bank_snapshots,
            exchange_snapshots=exchange_snapshots,
            ccp_snapshots=ccp_snapshots,
            market_snapshot=market_snapshot,
            clearing_outcomes=clearing,
            margin_calls=margins,
            defaults=defaults,
            flows=flows,
            system_metrics=system_metrics or {},
            actions=actions,
            rewards=rewards
        )
        
        self.timestep_records[t] = record
        
        # Update pre-aggregated series
        if market_snapshot:
            self._price_series.append(market_snapshot.asset_price)
        
        if bank_snapshots:
            defaulted = sum(1 for b in bank_snapshots.values() if b.solvency_status == 'defaulted')
            stressed = sum(1 for b in bank_snapshots.values() if b.solvency_status == 'stressed')
            exposure = sum(b.total_exposure for b in bank_snapshots.values())
            avg_cap = np.mean([b.capital_ratio for b in bank_snapshots.values()])
            
            self._default_count_series.append(defaulted)
            self._stress_count_series.append(stressed)
            self._total_exposure_series.append(exposure)
            self._avg_capital_series.append(avg_cap)
        
        return record
    
    def get_timestep(self, t: int) -> Optional[TimestepRecord]:
        """Get record for specific timestep."""
        return self.timestep_records.get(t)
    
    def get_entity_history(self, 
                           entity_type: str, 
                           entity_id: int) -> List[Any]:
        """Get full history for an entity."""
        if entity_type == 'bank':
            return self.state_capture.bank_snapshots.get(entity_id, [])
        elif entity_type == 'exchange':
            return self.state_capture.exchange_snapshots.get(entity_id, [])
        elif entity_type == 'ccp':
            return self.state_capture.ccp_snapshots.get(entity_id, [])
        return []
    
    def get_timeseries(self, series_name: str) -> List[float]:
        """Get pre-aggregated time series."""
        series_map = {
            'price': self._price_series,
            'defaults': self._default_count_series,
            'stressed': self._stress_count_series,
            'exposure': self._total_exposure_series,
            'capital_ratio': self._avg_capital_series
        }
        return series_map.get(series_name, [])
    
    def get_default_cascade(self, start_event_id: str) -> List[DefaultEvent]:
        """Trace cascade of defaults from initial event."""
        cascade = []
        current_defaults = {start_event_id}
        processed = set()
        
        while current_defaults - processed:
            for event_id in list(current_defaults - processed):
                event = next((d for d in self._default_events if d.event_id == event_id), None)
                if event:
                    cascade.append(event)
                    processed.add(event_id)
                    
                    # Find subsequent defaults triggered by this one
                    for d in self._default_events:
                        if d.trigger == 'cascade' and d.timestep > event.timestep:
                            current_defaults.add(d.event_id)
        
        return cascade
    
    def get_plot_data(self) -> Dict[str, Any]:
        """Get pre-aggregated data for plotting (minimizes frontend processing)."""
        return {
            'timesteps': list(range(len(self._price_series))),
            'price': self._price_series,
            'defaults': self._default_count_series,
            'stressed': self._stress_count_series,
            'exposure': self._total_exposure_series,
            'capital_ratio': self._avg_capital_series,
            'total_defaults': sum(self._default_count_series) if self._default_count_series else 0,
            'max_stressed': max(self._stress_count_series) if self._stress_count_series else 0,
            'price_min': min(self._price_series) if self._price_series else 0,
            'price_max': max(self._price_series) if self._price_series else 1
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary statistics."""
        return {
            'simulation_id': self.simulation_id,
            'created_at': self.created_at,
            'total_timesteps': len(self.timestep_records),
            'total_defaults': len(self._default_events),
            'total_margin_calls': len(self._margin_calls),
            'total_flows': len(self.state_capture.flow_records),
            'final_price': self._price_series[-1] if self._price_series else 1.0,
            'peak_defaults': max(self._default_count_series) if self._default_count_series else 0,
            'peak_stressed': max(self._stress_count_series) if self._stress_count_series else 0
        }
    
    def save(self, filepath: str, compress: bool = True) -> None:
        """Save history to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'simulation_id': self.simulation_id,
            'config': self.config,
            'created_at': self.created_at,
            'timestep_records': self.timestep_records,
            'default_events': self._default_events,
            'margin_calls': self._margin_calls,
            'clearing_outcomes': self._clearing_outcomes,
            'price_series': self._price_series,
            'default_series': self._default_count_series,
            'stress_series': self._stress_count_series,
            'exposure_series': self._total_exposure_series,
            'capital_series': self._avg_capital_series
        }
        
        if compress:
            with gzip.open(path.with_suffix('.pkl.gz'), 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationHistory':
        """Load history from file."""
        path = Path(filepath)
        
        if path.suffix == '.gz' or path.with_suffix('.pkl.gz').exists():
            with gzip.open(path.with_suffix('.pkl.gz'), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path.with_suffix('.pkl'), 'rb') as f:
                data = pickle.load(f)
        
        history = cls(
            simulation_id=data['simulation_id'],
            config=data['config']
        )
        history.created_at = data['created_at']
        history.timestep_records = data['timestep_records']
        history._default_events = data['default_events']
        history._margin_calls = data['margin_calls']
        history._clearing_outcomes = data['clearing_outcomes']
        history._price_series = data['price_series']
        history._default_count_series = data['default_series']
        history._stress_count_series = data['stress_series']
        history._total_exposure_series = data['exposure_series']
        history._avg_capital_series = data['capital_series']
        
        return history
    
    def export_json(self, filepath: str = None) -> str:
        """Export to JSON format for UI consumption."""
        data = {
            'metadata': {
                'simulation_id': self.simulation_id,
                'config': self.config,
                'created_at': self.created_at,
                'summary': self.get_summary()
            },
            'timeseries': self.get_plot_data(),
            'events': {
                'defaults': [asdict(e) for e in self._default_events],
                'margin_calls': [asdict(e) for e in self._margin_calls]
            }
        }
        
        json_str = json.dumps(data, indent=2, default=str)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
