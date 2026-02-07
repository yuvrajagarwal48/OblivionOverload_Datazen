"""
State Capture System for UI-Ready Metrics.
Provides structured snapshots of all institutional entities at each timestep.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import copy


class EntityType(Enum):
    """Types of entities in the financial system."""
    BANK = "bank"
    EXCHANGE = "exchange"
    CCP = "ccp"
    MARKET = "market"


class SolvencyStatus(Enum):
    """Solvency status indicators."""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    CRITICAL = "critical"
    DEFAULTED = "defaulted"


@dataclass
class BankSnapshot:
    """Complete snapshot of a bank's state at a timestep."""
    # Identification
    bank_id: int
    timestep: int
    timestamp: str
    
    # Cash and Liquidity
    cash_position: float
    liquidity_ratio: float
    liquidity_buffer: float
    
    # Capital Structure
    equity: float
    capital_ratio: float
    leverage_ratio: float
    
    # Balance Sheet
    total_assets: float
    total_liabilities: float
    
    # Lending Activity
    outstanding_lending: float
    outstanding_borrowing: float
    net_interbank_position: float
    
    # Margin and Collateral
    posted_margins: float
    collateral_held: float
    margin_coverage_ratio: float
    
    # Exposure and Risk
    total_exposure: float
    exposure_to_defaults: float
    largest_single_exposure: float
    concentration_index: float
    
    # Status
    solvency_status: str
    stress_level: float  # 0-1 normalized
    days_in_stress: int
    
    # Network Position
    degree_centrality: float
    betweenness_centrality: float
    tier: int
    
    # Visualization Metadata
    viz_size: float  # For node sizing
    viz_color_intensity: float  # For color mapping
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_bank(cls, bank: Any, timestep: int, network: Any = None) -> 'BankSnapshot':
        """Create snapshot from Bank object."""
        bs = bank.balance_sheet
        
        # Calculate metrics
        liquidity_ratio = bs.cash / max(bs.total_liabilities, 1e-8)
        leverage = bs.total_assets / max(bs.equity, 1e-8)
        
        # Get exposure info
        total_exposure = sum(bank.exposures.values()) if hasattr(bank, 'exposures') else 0
        largest_exposure = max(bank.exposures.values()) if hasattr(bank, 'exposures') and bank.exposures else 0
        
        # Concentration (HHI of exposures)
        if hasattr(bank, 'exposures') and bank.exposures and total_exposure > 0:
            shares = [e / total_exposure for e in bank.exposures.values()]
            concentration = sum(s**2 for s in shares)
        else:
            concentration = 0
        
        # Network centrality
        degree = 0
        betweenness = 0
        if network and hasattr(network, 'graph'):
            try:
                import networkx as nx
                degree = nx.degree_centrality(network.graph).get(bank.bank_id, 0)
                betweenness = nx.betweenness_centrality(network.graph).get(bank.bank_id, 0)
            except:
                pass
        
        # Stress level (normalized)
        stress = 0.0
        if bank.capital_ratio < 0.04:
            stress = 1.0
        elif bank.capital_ratio < 0.08:
            stress = 0.5 + (0.08 - bank.capital_ratio) / 0.08
        elif bank.capital_ratio < 0.12:
            stress = (0.12 - bank.capital_ratio) / 0.08
        
        return cls(
            bank_id=bank.bank_id,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            cash_position=bs.cash,
            liquidity_ratio=liquidity_ratio,
            liquidity_buffer=max(0, bs.cash - bs.total_liabilities * 0.1),
            equity=bs.equity,
            capital_ratio=bank.capital_ratio,
            leverage_ratio=leverage,
            total_assets=bs.total_assets,
            total_liabilities=bs.total_liabilities,
            outstanding_lending=bs.interbank_assets,
            outstanding_borrowing=bs.interbank_liabilities,
            net_interbank_position=bs.interbank_assets - bs.interbank_liabilities,
            posted_margins=getattr(bank, 'posted_margins', 0),
            collateral_held=getattr(bank, 'collateral_held', 0),
            margin_coverage_ratio=getattr(bank, 'margin_coverage', 1.0),
            total_exposure=total_exposure,
            exposure_to_defaults=getattr(bank, 'exposure_to_defaults', 0),
            largest_single_exposure=largest_exposure,
            concentration_index=concentration,
            solvency_status=bank.status.value if hasattr(bank.status, 'value') else str(bank.status),
            stress_level=stress,
            days_in_stress=getattr(bank, 'days_in_stress', 0),
            degree_centrality=degree,
            betweenness_centrality=betweenness,
            tier=bank.tier,
            viz_size=np.log1p(bs.total_assets) / 10,
            viz_color_intensity=stress
        )


@dataclass
class ExchangeSnapshot:
    """Complete snapshot of an exchange's state."""
    exchange_id: int
    timestep: int
    timestamp: str
    
    # Volume and Activity
    transaction_volume: float
    transaction_count: int
    average_transaction_size: float
    
    # Capacity
    max_throughput: float
    capacity_utilization: float
    
    # Congestion
    congestion_level: float
    order_backlog: int
    queue_depth: float
    
    # Timing
    average_settlement_delay: float
    max_settlement_delay: float
    pending_settlements: int
    
    # Fees
    total_fees_collected: float
    effective_fee_rate: float
    
    # Market Quality
    market_volatility: float
    bid_ask_spread: float
    
    # Status
    is_stressed: bool
    circuit_breaker_active: bool
    
    # Visualization
    viz_size: float
    viz_color_intensity: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_exchange(cls, exchange: Any, timestep: int) -> 'ExchangeSnapshot':
        """Create snapshot from Exchange object."""
        state = exchange.get_state() if hasattr(exchange, 'get_state') else None
        
        volume = getattr(exchange, 'total_volume', 0)
        count = getattr(exchange, 'transaction_count', 0)
        
        return cls(
            exchange_id=exchange.exchange_id,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            transaction_volume=volume,
            transaction_count=count,
            average_transaction_size=volume / max(count, 1),
            max_throughput=exchange.config.max_throughput if hasattr(exchange, 'config') else 1000,
            capacity_utilization=getattr(exchange, 'capacity_utilization', 0),
            congestion_level=exchange.congestion,
            order_backlog=len(getattr(exchange, 'order_book', [])),
            queue_depth=getattr(exchange, 'queue_depth', 0),
            average_settlement_delay=getattr(exchange, 'avg_delay', 0),
            max_settlement_delay=getattr(exchange, 'max_delay', 0),
            pending_settlements=len(getattr(exchange, 'pending_settlements', [])),
            total_fees_collected=getattr(exchange, 'total_fees', 0),
            effective_fee_rate=exchange.get_fee_rate() if hasattr(exchange, 'get_fee_rate') else 0.001,
            market_volatility=getattr(exchange, 'volatility', 0.02),
            bid_ask_spread=getattr(exchange, 'spread', 0.001),
            is_stressed=exchange.congestion > 0.8,
            circuit_breaker_active=getattr(exchange, 'circuit_breaker', False),
            viz_size=np.log1p(volume) / 10,
            viz_color_intensity=exchange.congestion
        )


@dataclass
class CCPSnapshot:
    """Complete snapshot of a CCP's state."""
    ccp_id: int
    timestep: int
    timestamp: str
    
    # Margin Pool
    total_margin_pool: float
    initial_margin_total: float
    variation_margin_total: float
    
    # Default Resources
    default_fund_balance: float
    ccp_capital_buffer: float
    total_prefunded_resources: float
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    netting_efficiency: float
    
    # Coverage
    stress_coverage_ratio: float
    cover_1_ratio: float  # Can cover largest member default
    cover_2_ratio: float  # Can cover two largest defaults
    
    # Member Status
    total_members: int
    distressed_members: int
    members_on_margin_call: int
    
    # Waterfall Status
    waterfall_layer: int  # 0=normal, 1-4=stress layers
    losses_absorbed: float
    
    # Activity
    positions_cleared: int
    margin_calls_issued: int
    
    # Visualization
    viz_size: float
    viz_color_intensity: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_ccp(cls, ccp: Any, timestep: int) -> 'CCPSnapshot':
        """Create snapshot from CentralCounterparty object."""
        # Calculate totals from margin accounts
        initial_margin = sum(
            ma.initial_margin for ma in ccp.margin_accounts.values()
        ) if hasattr(ccp, 'margin_accounts') else 0
        
        variation_margin = sum(
            ma.variation_margin for ma in ccp.margin_accounts.values()
        ) if hasattr(ccp, 'margin_accounts') else 0
        
        collateral = sum(
            ma.collateral_posted for ma in ccp.margin_accounts.values()
        ) if hasattr(ccp, 'margin_accounts') else 0
        
        default_fund = sum(
            ma.default_fund_contribution for ma in ccp.margin_accounts.values()
        ) if hasattr(ccp, 'margin_accounts') else 0
        
        # Distressed members
        distressed = sum(
            1 for ma in ccp.margin_accounts.values()
            if ma.collateral_posted < ma.initial_margin
        ) if hasattr(ccp, 'margin_accounts') else 0
        
        # Coverage ratios (simplified)
        total_resources = collateral + default_fund + ccp.capital
        gross_exp = getattr(ccp, 'gross_exposure', 0)
        net_exp = getattr(ccp, 'net_exposure', 0)
        
        stress_coverage = total_resources / max(gross_exp * 0.3, 1)  # 30% stress scenario
        
        return cls(
            ccp_id=ccp.ccp_id,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            total_margin_pool=collateral,
            initial_margin_total=initial_margin,
            variation_margin_total=variation_margin,
            default_fund_balance=default_fund,
            ccp_capital_buffer=ccp.capital,
            total_prefunded_resources=total_resources,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            netting_efficiency=1 - (net_exp / max(gross_exp, 1)),
            stress_coverage_ratio=min(stress_coverage, 3.0),
            cover_1_ratio=getattr(ccp, 'cover_1', 1.0),
            cover_2_ratio=getattr(ccp, 'cover_2', 0.8),
            total_members=len(ccp.margin_accounts) if hasattr(ccp, 'margin_accounts') else 0,
            distressed_members=distressed,
            members_on_margin_call=getattr(ccp, 'margin_calls_pending', 0),
            waterfall_layer=getattr(ccp, 'current_waterfall_layer', 0),
            losses_absorbed=getattr(ccp, 'total_losses_absorbed', 0),
            positions_cleared=getattr(ccp, 'positions_cleared', 0),
            margin_calls_issued=getattr(ccp, 'margin_calls_issued', 0),
            viz_size=np.log1p(total_resources) / 12,
            viz_color_intensity=distressed / max(len(ccp.margin_accounts), 1) if hasattr(ccp, 'margin_accounts') else 0
        )


@dataclass
class FlowRecord:
    """Record of a flow between entities."""
    flow_id: str
    timestep: int
    timestamp: str
    
    # Parties
    source_id: int
    source_type: str
    destination_id: int
    destination_type: str
    
    # Flow Details
    flow_type: str  # 'order', 'settlement', 'margin', 'guarantee', 'lending'
    volume: float
    
    # Costs
    fee: float
    spread_cost: float
    total_cost: float
    
    # Timing
    delay: int
    initiated_at: int
    completed_at: Optional[int]
    
    # Status
    status: str  # 'pending', 'completed', 'failed', 'cancelled'
    failure_reason: Optional[str]
    
    # Counterparty Chain
    intermediaries: List[str]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['intermediaries'] = list(self.intermediaries)
        return result


@dataclass
class MarketSnapshot:
    """Snapshot of market state."""
    timestep: int
    timestamp: str
    
    asset_price: float
    price_change: float
    price_change_pct: float
    
    volatility: float
    realized_volatility: float
    implied_volatility: float
    
    liquidity_index: float
    bid_ask_spread: float
    market_depth: float
    
    volume: float
    turnover: float
    
    condition: str
    stress_indicator: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_market(cls, market: Any, timestep: int, prev_price: float = 1.0) -> 'MarketSnapshot':
        """Create snapshot from Market object."""
        state = market.get_state() if hasattr(market, 'get_state') else None
        
        price = market.current_price if hasattr(market, 'current_price') else 1.0
        price_change = price - prev_price
        
        return cls(
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            asset_price=price,
            price_change=price_change,
            price_change_pct=price_change / max(prev_price, 0.01),
            volatility=market.volatility if hasattr(market, 'volatility') else 0.02,
            realized_volatility=getattr(market, 'realized_vol', 0.02),
            implied_volatility=getattr(market, 'implied_vol', 0.02),
            liquidity_index=state.liquidity_index if state else 1.0,
            bid_ask_spread=getattr(market, 'spread', 0.001),
            market_depth=getattr(market, 'depth', 1.0),
            volume=getattr(market, 'volume', 0),
            turnover=getattr(market, 'turnover', 0),
            condition=state.condition.value if state and hasattr(state.condition, 'value') else 'normal',
            stress_indicator=1.0 if state and state.condition.value != 'normal' else 0.0
        )


class StateCapture:
    """
    Main class for capturing and managing system state snapshots.
    """
    
    def __init__(self):
        self.bank_snapshots: Dict[int, List[BankSnapshot]] = {}
        self.exchange_snapshots: Dict[int, List[ExchangeSnapshot]] = {}
        self.ccp_snapshots: Dict[int, List[CCPSnapshot]] = {}
        self.market_snapshots: List[MarketSnapshot] = []
        self.flow_records: List[FlowRecord] = []
        
        self._flow_counter = 0
        self._prev_market_price = 1.0
    
    def capture_bank(self, bank: Any, timestep: int, network: Any = None) -> BankSnapshot:
        """Capture bank state snapshot."""
        snapshot = BankSnapshot.from_bank(bank, timestep, network)
        
        if bank.bank_id not in self.bank_snapshots:
            self.bank_snapshots[bank.bank_id] = []
        self.bank_snapshots[bank.bank_id].append(snapshot)
        
        return snapshot
    
    def capture_exchange(self, exchange: Any, timestep: int) -> ExchangeSnapshot:
        """Capture exchange state snapshot."""
        snapshot = ExchangeSnapshot.from_exchange(exchange, timestep)
        
        if exchange.exchange_id not in self.exchange_snapshots:
            self.exchange_snapshots[exchange.exchange_id] = []
        self.exchange_snapshots[exchange.exchange_id].append(snapshot)
        
        return snapshot
    
    def capture_ccp(self, ccp: Any, timestep: int) -> CCPSnapshot:
        """Capture CCP state snapshot."""
        snapshot = CCPSnapshot.from_ccp(ccp, timestep)
        
        if ccp.ccp_id not in self.ccp_snapshots:
            self.ccp_snapshots[ccp.ccp_id] = []
        self.ccp_snapshots[ccp.ccp_id].append(snapshot)
        
        return snapshot
    
    def capture_market(self, market: Any, timestep: int) -> MarketSnapshot:
        """Capture market state snapshot."""
        snapshot = MarketSnapshot.from_market(market, timestep, self._prev_market_price)
        self.market_snapshots.append(snapshot)
        self._prev_market_price = snapshot.asset_price
        return snapshot
    
    def record_flow(self,
                    source_id: int,
                    source_type: str,
                    dest_id: int,
                    dest_type: str,
                    flow_type: str,
                    volume: float,
                    timestep: int,
                    fee: float = 0,
                    delay: int = 0,
                    status: str = 'completed',
                    intermediaries: List[str] = None) -> FlowRecord:
        """Record a flow between entities."""
        self._flow_counter += 1
        
        record = FlowRecord(
            flow_id=f"F{timestep:05d}_{self._flow_counter:06d}",
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            source_id=source_id,
            source_type=source_type,
            destination_id=dest_id,
            destination_type=dest_type,
            flow_type=flow_type,
            volume=volume,
            fee=fee,
            spread_cost=0,
            total_cost=fee,
            delay=delay,
            initiated_at=timestep,
            completed_at=timestep + delay if status == 'completed' else None,
            status=status,
            failure_reason=None,
            intermediaries=intermediaries or []
        )
        
        self.flow_records.append(record)
        return record
    
    def capture_all(self,
                    banks: Dict[int, Any],
                    exchanges: List[Any],
                    ccps: List[Any],
                    market: Any,
                    network: Any,
                    timestep: int) -> Dict[str, Any]:
        """Capture complete system state."""
        result = {
            'timestep': timestep,
            'banks': {},
            'exchanges': {},
            'ccps': {},
            'market': None
        }
        
        for bank_id, bank in banks.items():
            result['banks'][bank_id] = self.capture_bank(bank, timestep, network)
        
        for exchange in exchanges:
            result['exchanges'][exchange.exchange_id] = self.capture_exchange(exchange, timestep)
        
        for ccp in ccps:
            result['ccps'][ccp.ccp_id] = self.capture_ccp(ccp, timestep)
        
        result['market'] = self.capture_market(market, timestep)
        
        return result
    
    def get_bank_timeseries(self, bank_id: int, field: str) -> List[Tuple[int, float]]:
        """Get time series of a bank field."""
        if bank_id not in self.bank_snapshots:
            return []
        
        return [(s.timestep, getattr(s, field, 0)) for s in self.bank_snapshots[bank_id]]
    
    def get_system_summary(self, timestep: int) -> Dict[str, Any]:
        """Get aggregated system summary at timestep."""
        banks_at_t = [
            snapshots[-1] for snapshots in self.bank_snapshots.values()
            if snapshots and snapshots[-1].timestep == timestep
        ]
        
        if not banks_at_t:
            return {}
        
        return {
            'num_banks': len(banks_at_t),
            'total_assets': sum(b.total_assets for b in banks_at_t),
            'total_equity': sum(b.equity for b in banks_at_t),
            'avg_capital_ratio': np.mean([b.capital_ratio for b in banks_at_t]),
            'num_stressed': sum(1 for b in banks_at_t if b.solvency_status == 'stressed'),
            'num_defaulted': sum(1 for b in banks_at_t if b.solvency_status == 'defaulted'),
            'total_exposure': sum(b.total_exposure for b in banks_at_t),
            'avg_liquidity': np.mean([b.liquidity_ratio for b in banks_at_t])
        }
    
    def export_to_json(self) -> str:
        """Export all captured data to JSON."""
        data = {
            'banks': {
                bid: [s.to_dict() for s in snapshots]
                for bid, snapshots in self.bank_snapshots.items()
            },
            'exchanges': {
                eid: [s.to_dict() for s in snapshots]
                for eid, snapshots in self.exchange_snapshots.items()
            },
            'ccps': {
                cid: [s.to_dict() for s in snapshots]
                for cid, snapshots in self.ccp_snapshots.items()
            },
            'market': [s.to_dict() for s in self.market_snapshots],
            'flows': [f.to_dict() for f in self.flow_records]
        }
        return json.dumps(data, indent=2, default=str)
    
    def clear(self):
        """Clear all captured data."""
        self.bank_snapshots.clear()
        self.exchange_snapshots.clear()
        self.ccp_snapshots.clear()
        self.market_snapshots.clear()
        self.flow_records.clear()
        self._flow_counter = 0
        self._prev_market_price = 1.0
