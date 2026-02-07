"""
Improved Data Aggregator for Gen AI Analysis
Works with real SimulationState history data instead of placeholders
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class AggregatedMetrics:
    """Complete aggregated metrics from simulation"""

    # Time series
    capital_ratio_history: List[float] = field(default_factory=list)
    default_count_history: List[int] = field(default_factory=list)
    stress_count_history: List[int] = field(default_factory=list)
    asset_price_history: List[float] = field(default_factory=list)
    liquidity_history: List[float] = field(default_factory=list)

    # Aggregates
    total_defaults: int = 0
    total_stressed_max: int = 0
    total_contagion_events: int = 0
    max_drawdown: float = 0.0
    final_asset_price: float = 1.0

    # Network metrics
    top_10_debtrank: Dict[int, float] = field(default_factory=dict)
    average_debtrank: float = 0.0

    # Risk metrics
    systemic_risk_index: float = 0.0
    cascade_potential: int = 0

    # Event logs
    default_events: List[Dict] = field(default_factory=list)
    margin_calls: List[Dict] = field(default_factory=list)
    shock_events: List[Dict] = field(default_factory=list)

    # Per-bank metrics
    bank_outcomes: Dict[int, Dict] = field(default_factory=dict)

    # Market metrics
    price_volatility: float = 0.0
    price_trend: float = 0.0
    liquidity_stress_events: int = 0

    # Network topology
    network_density: float = 0.0
    avg_clustering: float = 0.0
    avg_capital_ratio: float = 0.0

    # Infrastructure
    total_transactions: int = 0
    total_exchange_fees: float = 0.0
    total_margin_collected: float = 0.0

    # Simulation metadata
    num_banks: int = 0
    num_steps: int = 0
    scenario_name: str = "unknown"
    timestamp: str = ""


class DataAggregatorLive:
    """
    Aggregates simulation data from the live SimulationState.
    Uses actual step_history records instead of placeholder data.
    """

    def __init__(self, simulation_state):
        """
        Args:
            simulation_state: The SimulationState instance from api/routes/state.py
        """
        self.state = simulation_state
        self.env = simulation_state.env
        self.history = simulation_state.step_history

    def aggregate(self) -> AggregatedMetrics:
        """Compile all metrics from simulation history"""

        if not self.history:
            return self._empty_metrics()

        # Extract time series from step_history
        prices = [h.market_price for h in self.history]
        capital_ratios = [h.avg_capital_ratio for h in self.history]
        default_counts = [h.total_defaults for h in self.history]
        stress_counts = [h.num_stressed for h in self.history]
        liquidity_vals = [h.liquidity_index for h in self.history]

        # Compute risk from real data
        network = self.env.network
        debtrank = self._compute_debtrank_real()
        default_events = self._extract_default_events()
        cascade_depth = self._compute_cascade_potential()

        # Per-bank analysis
        bank_outcomes = self._per_bank_analysis()

        # Infrastructure stats
        infra = self._compute_infrastructure_stats()

        # Network topology
        network_stats = network.get_network_stats()

        # Systemic risk (normalized 0-1)
        num_banks = len(network.banks)
        peak_defaults = max(default_counts) if default_counts else 0
        peak_stressed = max(stress_counts) if stress_counts else 0
        avg_liquidity = np.mean(liquidity_vals) if liquidity_vals else 1.0

        default_rate = peak_defaults / max(num_banks, 1)
        stress_rate = peak_stressed / max(num_banks, 1)
        liquidity_risk = sum(1 for l in liquidity_vals if l < 0.3) / max(len(liquidity_vals), 1)

        systemic_risk = 0.4 * default_rate + 0.3 * stress_rate + 0.3 * liquidity_risk

        return AggregatedMetrics(
            capital_ratio_history=capital_ratios,
            default_count_history=default_counts,
            stress_count_history=stress_counts,
            asset_price_history=prices,
            liquidity_history=liquidity_vals,

            total_defaults=peak_defaults,
            total_stressed_max=peak_stressed,
            total_contagion_events=len(default_events),
            max_drawdown=self._calculate_max_drawdown(prices),
            final_asset_price=prices[-1] if prices else 1.0,

            top_10_debtrank=dict(sorted(
                debtrank.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            average_debtrank=float(np.mean(list(debtrank.values()))) if debtrank else 0.0,

            systemic_risk_index=float(systemic_risk),
            cascade_potential=cascade_depth,

            default_events=default_events,
            margin_calls=self._extract_margin_calls(),
            shock_events=self._extract_shocks(prices),

            bank_outcomes=bank_outcomes,

            price_volatility=float(np.std(np.diff(prices))) if len(prices) > 1 else 0.0,
            price_trend=float((prices[-1] - prices[0]) / prices[0]) if prices and prices[0] != 0 else 0.0,
            liquidity_stress_events=sum(1 for l in liquidity_vals if l < 0.3),

            network_density=getattr(network_stats, 'density', getattr(network_stats, 'network_density', 0.0)),
            avg_clustering=getattr(network_stats, 'clustering_coefficient', getattr(network_stats, 'avg_clustering', 0.0)),
            avg_capital_ratio=capital_ratios[-1] if capital_ratios else 0.0,

            total_transactions=infra.get("total_transactions", 0),
            total_exchange_fees=infra.get("total_fees", 0.0),
            total_margin_collected=infra.get("total_margin", 0.0),

            num_banks=num_banks,
            num_steps=len(self.history),
            scenario_name=self.state.config.get("scenario", "unknown") if self.state.config else "unknown",
            timestamp=datetime.now().isoformat()
        )

    def _compute_debtrank_real(self) -> Dict[int, float]:
        """Compute DebtRank using real exposure matrix"""
        network = self.env.network
        debtrank = {}

        try:
            from src.analytics.risk_metrics import DebtRankCalculator
            bank_ids = list(network.banks.keys())
            n = len(bank_ids)
            id_to_idx = {bid: i for i, bid in enumerate(bank_ids)}

            exposure_matrix = np.zeros((n, n))
            equity_vector = np.zeros(n)

            for bank_id, bank in network.banks.items():
                i = id_to_idx[bank_id]
                equity_vector[i] = max(bank.balance_sheet.equity, 0.01)
                for creditor_id, amount in bank.balance_sheet.interbank_liabilities.items():
                    if creditor_id in id_to_idx:
                        j = id_to_idx[creditor_id]
                        exposure_matrix[j, i] = amount

            calculator = DebtRankCalculator(recovery_rate=0.0)
            _, individual_dr = calculator.calculate(exposure_matrix, equity_vector)

            debtrank = {bank_ids[i]: float(individual_dr[i]) for i in range(n)}
        except Exception:
            # Fallback: capital-ratio proxy
            for bank_id, bank in network.banks.items():
                debtrank[bank_id] = round(0.5 / (bank.capital_ratio + 0.01), 4)

        return debtrank

    def _extract_default_events(self) -> List[Dict]:
        """Extract default events from step history"""
        events = []
        seen_defaults = set()

        for h in self.history:
            for d in h.defaults_this_step:
                if d not in seen_defaults:
                    seen_defaults.add(d)
                    bank = self.env.network.banks.get(d)
                    events.append({
                        "bank_id": d,
                        "timestep": h.timestep,
                        "trigger": "capital_ratio_below_threshold",
                        "equity_at_default": round(bank.balance_sheet.equity, 2) if bank else 0,
                        "creditors_affected": len(bank.balance_sheet.interbank_liabilities) if bank else 0,
                        "tier": bank.tier if bank else 0
                    })

        # Also check current state for any defaults not captured in step events
        for bank_id, bank in self.env.network.banks.items():
            if bank.status.value == "defaulted" and bank_id not in seen_defaults:
                events.append({
                    "bank_id": bank_id,
                    "timestep": self.env.current_step,
                    "trigger": "capital_ratio_below_threshold",
                    "equity_at_default": round(bank.balance_sheet.equity, 2),
                    "creditors_affected": len(bank.balance_sheet.interbank_liabilities),
                    "tier": bank.tier
                })

        return events

    def _extract_margin_calls(self) -> List[Dict]:
        """Extract margin call events"""
        calls = []
        for bank_id, bank in self.env.network.banks.items():
            if bank.capital_ratio < 0.08 and bank.status.value != "defaulted":
                calls.append({
                    "bank_id": bank_id,
                    "capital_ratio": round(bank.capital_ratio, 4),
                    "shortfall": round(bank.balance_sheet.total_assets * (0.08 - bank.capital_ratio), 2),
                    "reason": "capital_ratio_breach"
                })
        return calls

    def _extract_shocks(self, prices: List[float]) -> List[Dict]:
        """Detect price shocks from history"""
        shocks = []
        for i in range(1, len(prices)):
            change = (prices[i] - prices[i - 1]) / max(prices[i - 1], 0.001)
            if abs(change) > 0.03:  # 3% move
                shocks.append({
                    "timestep": i,
                    "type": "price_shock",
                    "magnitude": round(change, 4),
                    "direction": "up" if change > 0 else "down"
                })
        return shocks

    def _compute_cascade_potential(self) -> int:
        """Compute max cascade depth from network"""
        defaulted = [
            bid for bid, b in self.env.network.banks.items()
            if b.status.value == "defaulted"
        ]
        if not defaulted:
            return 0

        # Count banks that would be affected in one round
        affected = set(defaulted)
        for bid in defaulted:
            bank = self.env.network.banks[bid]
            for creditor_id in bank.balance_sheet.interbank_liabilities:
                if creditor_id in self.env.network.banks:
                    creditor = self.env.network.banks[creditor_id]
                    exposure = bank.balance_sheet.interbank_liabilities.get(creditor_id, 0)
                    if exposure > creditor.balance_sheet.equity * 0.3:
                        affected.add(creditor_id)

        return len(affected)

    def _per_bank_analysis(self) -> Dict[int, Dict]:
        """Per-bank outcome metrics"""
        outcomes = {}
        debtrank = self._compute_debtrank_real()

        for bank_id, bank in self.env.network.banks.items():
            outcomes[bank_id] = {
                "tier": bank.tier,
                "final_status": bank.status.value,
                "final_equity": round(bank.balance_sheet.equity, 2),
                "final_capital_ratio": round(bank.capital_ratio, 4),
                "cash": round(bank.balance_sheet.cash, 2),
                "total_assets": round(bank.balance_sheet.total_assets, 2),
                "total_liabilities": round(bank.balance_sheet.total_liabilities, 2),
                "interbank_assets": round(sum(bank.balance_sheet.interbank_assets.values()), 2),
                "interbank_liabilities": round(sum(bank.balance_sheet.interbank_liabilities.values()), 2),
                "debtrank": round(debtrank.get(bank_id, 0), 4),
                "defaulted": bank.status.value == "defaulted",
                "is_solvent": bank.is_solvent,
                "is_liquid": bank.is_liquid,
            }

        return outcomes

    def _compute_infrastructure_stats(self) -> Dict[str, Any]:
        """Aggregate infrastructure stats from history"""
        total_tx = 0
        total_fees = 0.0
        total_margin = 0.0

        # From step history events if captured
        for h in self.history:
            for evt in h.events:
                if evt.get("type") == "transaction":
                    total_tx += 1
                    total_fees += evt.get("fee", 0)

        # From CCP state
        if self.state.ccps:
            for ccp in self.state.ccps:
                try:
                    ccp_state = ccp.get_state()
                    total_margin += ccp_state.total_initial_margin + ccp_state.total_variation_margin
                except Exception:
                    pass

        return {
            "total_transactions": total_tx,
            "total_fees": round(total_fees, 2),
            "total_margin": round(total_margin, 2),
        }

    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Maximum price decline from peak"""
        if not prices or len(prices) < 2:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return round(max_dd, 4)

    def _empty_metrics(self) -> AggregatedMetrics:
        """Return empty metrics when no history"""
        return AggregatedMetrics(
            num_banks=len(self.env.network.banks) if self.env else 0,
            timestamp=datetime.now().isoformat()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        metrics = self.aggregate()
        return asdict(metrics)

    def export_to_json(self, filepath: str) -> str:
        """Export aggregated metrics to JSON"""
        data = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
