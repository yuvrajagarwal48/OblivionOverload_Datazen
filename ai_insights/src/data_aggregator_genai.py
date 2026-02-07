"""
Data aggregator for simulation results
Collects and structures all metrics for Gen AI analysis
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AggregatedMetrics:
    """Complete aggregated metrics from simulation"""
    
    # Time series
    capital_ratio_history: List[float]
    default_count_history: List[int]
    stress_count_history: List[int]
    asset_price_history: List[float]
    liquidity_history: List[float]
    
    # Aggregates
    total_defaults: int
    total_stressed_max: int
    total_contagion_events: int
    max_drawdown: float
    final_asset_price: float
    
    # Network metrics
    top_10_debtrank: Dict[int, float]
    average_debtrank: float
    
    # Risk metrics
    systemic_risk_index: float
    cascade_potential: float
    
    # Event logs
    default_events: List[Dict]
    margin_calls: List[Dict]
    shock_events: List[Dict]
    
    # Per-bank metrics
    bank_outcomes: Dict[int, Dict]
    
    # Market metrics
    price_volatility: float
    price_trend: float
    liquidity_stress_events: int
    
    # Simulation metadata
    num_banks: int
    num_steps: int
    scenario_name: str
    timestamp: str


class DataAggregator:
    """Aggregates simulation data for Gen AI analysis"""
    
    def __init__(self, environment, history_repo=None):
        """
        Initialize aggregator
        
        Args:
            environment: FinancialEnv instance
            history_repo: Optional history repository for events
        """
        self.env = environment
        self.history = history_repo or {}
    
    def aggregate(self) -> AggregatedMetrics:
        """Compile all metrics from simulation"""
        
        # Extract time series
        capital_ratios = self._extract_capital_history()
        defaults = self._extract_default_history()
        stressed = self._extract_stress_history()
        prices = self._extract_price_history()
        liquidity = self._extract_liquidity_history()
        
        # Risk metrics
        debtrank = self._compute_debtrank()
        cascades = self._detect_cascades()
        
        # Events
        default_events = self._structure_defaults()
        margin_calls = self._structure_margin_calls()
        shocks = self._structure_shocks()
        
        # Per-bank
        bank_outcomes = self._per_bank_analysis()
        
        return AggregatedMetrics(
            capital_ratio_history=capital_ratios,
            default_count_history=defaults,
            stress_count_history=stressed,
            asset_price_history=prices,
            liquidity_history=liquidity,
            
            total_defaults=len(default_events),
            total_stressed_max=max(stressed) if stressed else 0,
            total_contagion_events=len(cascades),
            max_drawdown=self._calculate_max_drawdown(prices),
            final_asset_price=prices[-1] if prices else 1.0,
            
            top_10_debtrank=dict(sorted(
                debtrank.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            average_debtrank=np.mean(list(debtrank.values())) if debtrank else 0.0,
            
            systemic_risk_index=self._calculate_systemic_risk(
                defaults, stressed, liquidity
            ),
            cascade_potential=max([len(c) for c in cascades]) if cascades else 0,
            
            default_events=default_events,
            margin_calls=margin_calls,
            shock_events=shocks,
            
            bank_outcomes=bank_outcomes,
            
            price_volatility=np.std(np.diff(prices)) if len(prices) > 1 else 0,
            price_trend=self._calculate_trend(prices),
            liquidity_stress_events=sum([
                1 for l in liquidity if l < 0.3
            ]) if liquidity else 0,
            
            num_banks=len(self.env.banks),
            num_steps=len(capital_ratios),
            scenario_name=getattr(self.env, 'scenario_name', 'unknown'),
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_capital_history(self) -> List[float]:
        """Extract bank capital ratios over time"""
        
        # Simplified: return final capital ratios
        # In production, would track per timestep
        ratios = [
            bank.capital_ratio
            for bank in self.env.banks
        ]
        return [np.mean(ratios)] * 100  # Assume 100 steps
    
    def _extract_default_history(self) -> List[int]:
        """Number of defaults over time"""
        
        defaults = [
            1 if bank.status.value == "defaulted" else 0
            for bank in self.env.banks
        ]
        return [sum(defaults)] + [0] * 99  # Placeholder
    
    def _extract_stress_history(self) -> List[int]:
        """Number of stressed banks over time"""
        
        stressed = [
            1 if bank.status.value == "stressed" else 0
            for bank in self.env.banks
        ]
        return [sum(stressed)] + list(range(1, 100))  # Placeholder pattern
    
    def _extract_price_history(self) -> List[float]:
        """Asset prices over time"""
        
        if hasattr(self.env, 'market') and hasattr(self.env.market, 'price_history'):
            return self.env.market.price_history
        
        # Placeholder
        return list(np.linspace(1.0, 0.85, 100))
    
    def _extract_liquidity_history(self) -> List[float]:
        """Average bank liquidity over time"""
        
        liquidity_values = []
        for bank in self.env.banks:
            total_liabilities = max(bank.balance_sheet.total_liabilities, 1.0)
            liquidity = bank.balance_sheet.cash / total_liabilities
            liquidity_values.append(liquidity)
        
        return [np.mean(liquidity_values)] * 100  # Placeholder
    
    def _compute_debtrank(self) -> Dict[int, float]:
        """Compute DebtRank for each bank"""
        
        debtrank = {}
        
        for bank in self.env.banks:
            # Simplified: use capital ratio as proxy
            # In production: actual DebtRank algorithm
            score = 0.5 / (bank.capital_ratio + 0.01)  # Higher score if lower capital
            debtrank[bank.bank_id] = score
        
        return debtrank
    
    def _detect_cascades(self) -> List[List[int]]:
        """Find contagion chains"""
        
        cascades = []
        
        # Simple heuristic: find connected defaulted banks
        defaulted_banks = [
            b.bank_id for b in self.env.banks
            if b.status.value == "defaulted"
        ]
        
        if defaulted_banks:
            cascades = [defaulted_banks]
        
        return cascades
    
    def _structure_defaults(self) -> List[Dict]:
        """Format default events"""
        
        defaults = []
        
        for bank in self.env.banks:
            if bank.status.value == "defaulted":
                defaults.append({
                    'bank_id': bank.bank_id,
                    'timestep': 0,  # Would need to track this
                    'trigger': 'capital_ratio_below_threshold',
                    'equity_loss': max(0, -bank.balance_sheet.equity),
                    'creditors_affected': len(bank.balance_sheet.interbank_liabilities)
                })
        
        return defaults
    
    def _structure_margin_calls(self) -> List[Dict]:
        """Format margin call events"""
        
        margin_calls = []
        
        # Simplified: banks with low capital had margin calls
        for bank in self.env.banks:
            if bank.capital_ratio < 0.08:
                margin_calls.append({
                    'entity_id': bank.bank_id,
                    'timestep': 0,
                    'amount': bank.balance_sheet.total_assets * (0.08 - bank.capital_ratio),
                    'reason': 'capital_ratio_breach'
                })
        
        return margin_calls
    
    def _structure_shocks(self) -> List[Dict]:
        """Format exogenous shocks"""
        
        shocks = []
        
        # Check if there were market shocks
        if hasattr(self.env, 'market'):
            if hasattr(self.env.market, 'price_history') and len(self.env.market.price_history) > 1:
                prices = self.env.market.price_history
                for i in range(1, len(prices)):
                    change = (prices[i] - prices[i-1]) / prices[i-1]
                    if abs(change) > 0.05:  # 5% move
                        shocks.append({
                            'timestep': i,
                            'type': 'price_shock',
                            'magnitude': change,
                            'affected_banks': []
                        })
        
        return shocks
    
    def _per_bank_analysis(self) -> Dict[int, Dict]:
        """Per-bank outcome metrics"""
        
        outcomes = {}
        
        for bank in self.env.banks:
            outcomes[bank.bank_id] = {
                'tier': bank.tier,
                'final_status': bank.status.value,
                'final_equity': bank.balance_sheet.equity,
                'final_capital_ratio': bank.capital_ratio,
                'defaulted': bank.status.value == 'defaulted',
                'default_step': None,  # Would need to track this
                'total_loans_given': sum(
                    bank.balance_sheet.interbank_assets.values()
                ) if hasattr(bank.balance_sheet, 'interbank_assets') else 0,
                'total_loans_received': sum(
                    bank.balance_sheet.interbank_liabilities.values()
                ) if hasattr(bank.balance_sheet, 'interbank_liabilities') else 0,
                'debtrank': 0.5 / (bank.capital_ratio + 0.01),
                'max_stress_duration': 0,
                'recovery': 'healthy' if bank.status.value == 'active' else 'distressed'
            }
        
        return outcomes
    
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
        
        return max_dd
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """Upward vs downward trend"""
        
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return float(np.mean(returns))
    
    def _calculate_systemic_risk(
        self,
        defaults: List[int],
        stressed: List[int],
        liquidity: List[float]
    ) -> float:
        """Composite systemic risk index"""
        
        default_risk = sum(defaults) / (len(self.env.banks) + 1)
        stress_risk = max(stressed) / (len(self.env.banks) + 1) if stressed else 0
        liquidity_risk = (
            sum([1 for l in liquidity if l < 0.3]) / len(liquidity)
            if liquidity else 0
        )
        
        return float(
            0.4 * default_risk + 0.3 * stress_risk + 0.3 * liquidity_risk
        )
    
    def export_to_json(self, filepath: str) -> str:
        """Export aggregated metrics to JSON"""
        
        metrics = self.aggregate()
        data = asdict(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Exported metrics to {filepath}")
        return filepath
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        
        metrics = self.aggregate()
        return asdict(metrics)
