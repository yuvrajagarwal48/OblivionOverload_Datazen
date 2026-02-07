"""
UI-Oriented Data Packaging for FinSim-MAPPO.
Aggregates simulation data for frontend consumption.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import gzip


class AlertLevel(str, Enum):
    """Alert severity levels for UI."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """UI alert notification."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    entity_id: Optional[int] = None
    timestamp: int = 0
    dismissed: bool = False
    action_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['level'] = self.level.value
        return result


@dataclass
class EntityCard:
    """Summary card for a single entity (bank, exchange, CCP)."""
    entity_id: int
    entity_type: str  # "bank", "exchange", "ccp"
    name: str
    
    # Health indicators (0-100 scale for UI)
    health_score: int = 100
    capital_health: int = 100
    liquidity_health: int = 100
    exposure_health: int = 100
    
    # Key metrics (formatted for display)
    capital_ratio: str = "10.0%"
    liquidity_ratio: str = "15.0%"
    total_assets: str = "1.0B"
    total_liabilities: str = "0.9B"
    
    # Risk indicators
    risk_rating: str = "A"
    pd_display: str = "0.1%"
    systemic_importance: str = "Low"
    
    # Status
    status: str = "normal"  # normal, stressed, critical, defaulted
    active_violations: int = 0
    
    # Trend indicators
    capital_trend: str = "stable"  # up, down, stable
    risk_trend: str = "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeSeriesData:
    """Time series data package for charts."""
    name: str
    description: str
    timestamps: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    unit: str = ""
    chart_type: str = "line"  # line, bar, area
    color: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def add_point(self, timestamp: int, value: float) -> None:
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_latest(self) -> Optional[float]:
        return self.values[-1] if self.values else None
    
    def get_change(self, periods: int = 1) -> Optional[float]:
        if len(self.values) < periods + 1:
            return None
        return self.values[-1] - self.values[-(periods + 1)]


@dataclass
class NetworkVisualization:
    """Network data for graph visualization."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_node(self, node_id: int, label: str, node_type: str,
                 health: int = 100, size: float = 1.0,
                 position: Optional[Tuple[float, float]] = None,
                 **attributes) -> None:
        node = {
            'id': node_id,
            'label': label,
            'type': node_type,
            'health': health,
            'size': size,
            **attributes
        }
        if position:
            node['x'], node['y'] = position
        self.nodes.append(node)
    
    def add_edge(self, source: int, target: int, weight: float,
                 edge_type: str = "exposure", **attributes) -> None:
        self.edges.append({
            'source': source,
            'target': target,
            'weight': weight,
            'type': edge_type,
            **attributes
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeatmapData:
    """Heatmap data for correlation/exposure matrices."""
    name: str
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 1.0
    color_scale: str = "RdYlGn"  # diverging color scale
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationSummary:
    """High-level simulation summary for dashboard."""
    # Time info
    current_timestep: int = 0
    total_timesteps: int = 100
    simulation_date: str = ""
    
    # System health (0-100)
    overall_health: int = 100
    stability_index: int = 100
    
    # Entity counts
    total_banks: int = 0
    healthy_banks: int = 0
    stressed_banks: int = 0
    defaulted_banks: int = 0
    
    # Key aggregates
    total_system_assets: str = "100B"
    total_system_capital: str = "10B"
    system_capital_ratio: str = "10.0%"
    
    # Risk metrics
    average_pd: str = "0.5%"
    total_expected_loss: str = "50M"
    systemic_risk_score: int = 25
    
    # Event counts
    defaults_today: int = 0
    margin_calls_today: int = 0
    interventions_today: int = 0
    
    # Compliance
    compliance_rate: str = "95%"
    active_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class ScenarioComparison:
    """Comparison data between scenarios."""
    scenario_names: List[str] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)  # metric_name -> values per scenario
    
    def add_scenario(self, name: str, metric_values: Dict[str, float]) -> None:
        if name not in self.scenario_names:
            self.scenario_names.append(name)
        
        for metric, value in metric_values.items():
            if metric not in self.metrics:
                self.metrics[metric] = []
            self.metrics[metric].append(value)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UIDataPackager:
    """
    Main UI data packaging class.
    Aggregates simulation data for frontend consumption.
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.entity_cards: Dict[int, EntityCard] = {}
        self.time_series: Dict[str, TimeSeriesData] = {}
        self.network_data: NetworkVisualization = NetworkVisualization()
        self.heatmaps: Dict[str, HeatmapData] = {}
        self.summary: SimulationSummary = SimulationSummary()
        
        self._alert_counter = 0
    
    # === Alert Management ===
    
    def add_alert(self, level: AlertLevel, title: str, message: str,
                  entity_id: Optional[int] = None, timestamp: int = 0,
                  action_required: bool = False) -> str:
        """Add an alert notification."""
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}"
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            entity_id=entity_id,
            timestamp=timestamp,
            action_required=action_required
        )
        
        self.alerts.append(alert)
        return alert_id
    
    def dismiss_alert(self, alert_id: str) -> None:
        """Dismiss an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.dismissed = True
                break
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all undismissed alerts."""
        return [a.to_dict() for a in self.alerts if not a.dismissed]
    
    # === Entity Cards ===
    
    def update_entity_card(self, entity: Any, entity_type: str,
                           risk_output: Optional[Any] = None,
                           violations: Optional[List[Any]] = None) -> EntityCard:
        """Create or update entity card from entity state."""
        entity_id = getattr(entity, 'bank_id', getattr(entity, 'exchange_id', getattr(entity, 'ccp_id', 0)))
        
        card = EntityCard(
            entity_id=entity_id,
            entity_type=entity_type,
            name=f"{entity_type.capitalize()} {entity_id}"
        )
        
        if entity_type == "bank" and hasattr(entity, 'balance_sheet'):
            bs = entity.balance_sheet
            
            # Calculate health scores (0-100)
            capital_ratio = entity.capital_ratio if hasattr(entity, 'capital_ratio') else 0.1
            liquidity_ratio = bs.cash / max(bs.total_liabilities, 1e-8)
            
            card.capital_health = min(int(capital_ratio / 0.15 * 100), 100)
            card.liquidity_health = min(int(liquidity_ratio / 0.15 * 100), 100)
            card.exposure_health = 100  # TODO: compute from exposures
            card.health_score = (card.capital_health + card.liquidity_health + card.exposure_health) // 3
            
            # Format metrics
            card.capital_ratio = f"{capital_ratio:.1%}"
            card.liquidity_ratio = f"{liquidity_ratio:.1%}"
            card.total_assets = self._format_currency(bs.total_assets)
            card.total_liabilities = self._format_currency(bs.total_liabilities)
            
            # Determine status
            if capital_ratio < 0.02:
                card.status = "defaulted" if hasattr(entity, 'is_defaulted') and entity.is_defaulted else "critical"
            elif capital_ratio < 0.06:
                card.status = "stressed"
            else:
                card.status = "normal"
        
        # Risk metrics from credit risk layer
        if risk_output:
            card.risk_rating = risk_output.rating.value if hasattr(risk_output, 'rating') else "N/A"
            card.pd_display = f"{risk_output.probability_of_default:.2%}" if hasattr(risk_output, 'probability_of_default') else "N/A"
            
            si = risk_output.systemic_importance if hasattr(risk_output, 'systemic_importance') else 0
            card.systemic_importance = "High" if si > 0.7 else "Medium" if si > 0.3 else "Low"
        
        # Violations
        if violations:
            card.active_violations = len(violations)
        
        self.entity_cards[entity_id] = card
        return card
    
    def get_all_entity_cards(self) -> List[Dict[str, Any]]:
        """Get all entity cards."""
        return [c.to_dict() for c in self.entity_cards.values()]
    
    # === Time Series ===
    
    def create_time_series(self, name: str, description: str = "",
                           unit: str = "", chart_type: str = "line",
                           color: Optional[str] = None) -> TimeSeriesData:
        """Create a new time series."""
        ts = TimeSeriesData(
            name=name,
            description=description,
            unit=unit,
            chart_type=chart_type,
            color=color
        )
        self.time_series[name] = ts
        return ts
    
    def update_time_series(self, name: str, timestamp: int, value: float) -> None:
        """Add point to time series."""
        if name in self.time_series:
            self.time_series[name].add_point(timestamp, value)
    
    def get_time_series(self, name: str) -> Optional[Dict[str, Any]]:
        """Get time series by name."""
        if name in self.time_series:
            return self.time_series[name].to_dict()
        return None
    
    def get_all_time_series(self) -> Dict[str, Dict[str, Any]]:
        """Get all time series."""
        return {k: v.to_dict() for k, v in self.time_series.items()}
    
    # === Network Visualization ===
    
    def build_network_from_env(self, banks: Dict[int, Any],
                                exposures: Optional[Dict[Tuple[int, int], float]] = None,
                                risk_outputs: Optional[Dict[int, Any]] = None) -> NetworkVisualization:
        """Build network visualization from environment state."""
        self.network_data = NetworkVisualization()
        
        for bank_id, bank in banks.items():
            health = 100
            if hasattr(bank, 'capital_ratio'):
                health = min(int(bank.capital_ratio / 0.15 * 100), 100)
            
            node_type = "normal"
            if hasattr(bank, 'is_defaulted') and bank.is_defaulted:
                node_type = "defaulted"
                health = 0
            elif health < 30:
                node_type = "critical"
            elif health < 60:
                node_type = "stressed"
            
            size = 1.0
            if risk_outputs and bank_id in risk_outputs:
                size = 1 + risk_outputs[bank_id].systemic_importance * 2
            elif hasattr(bank, 'balance_sheet'):
                # Size by assets
                size = np.log10(max(bank.balance_sheet.total_assets, 1e6)) / 10
            
            self.network_data.add_node(
                node_id=bank_id,
                label=f"Bank {bank_id}",
                node_type=node_type,
                health=health,
                size=size
            )
        
        # Add edges from exposures
        if exposures:
            for (source, target), amount in exposures.items():
                if source in banks and target in banks:
                    self.network_data.add_edge(
                        source=source,
                        target=target,
                        weight=amount,
                        edge_type="exposure"
                    )
        
        return self.network_data
    
    # === Heatmaps ===
    
    def create_exposure_heatmap(self, banks: Dict[int, Any],
                                 exposures: Dict[Tuple[int, int], float]) -> HeatmapData:
        """Create exposure matrix heatmap."""
        bank_ids = sorted(banks.keys())
        labels = [f"Bank {bid}" for bid in bank_ids]
        
        n = len(bank_ids)
        values = [[0.0] * n for _ in range(n)]
        
        max_exposure = 1.0
        for (src, tgt), amount in exposures.items():
            if src in banks and tgt in banks:
                i = bank_ids.index(src)
                j = bank_ids.index(tgt)
                values[i][j] = amount
                max_exposure = max(max_exposure, amount)
        
        heatmap = HeatmapData(
            name="exposure_matrix",
            row_labels=labels,
            col_labels=labels,
            values=values,
            max_value=max_exposure
        )
        
        self.heatmaps["exposure_matrix"] = heatmap
        return heatmap
    
    # === Simulation Summary ===
    
    def update_summary(self, env: Any, timestep: int,
                       risk_outputs: Optional[Dict[int, Any]] = None,
                       compliance_summary: Optional[Dict[str, Any]] = None) -> SimulationSummary:
        """Update simulation summary from environment state."""
        self.summary.current_timestep = timestep
        self.summary.simulation_date = datetime.now().isoformat()
        
        if hasattr(env, 'banks'):
            banks = env.banks
            self.summary.total_banks = len(banks)
            
            defaulted = sum(1 for b in banks.values() 
                           if hasattr(b, 'is_defaulted') and b.is_defaulted)
            stressed = sum(1 for b in banks.values() 
                          if hasattr(b, 'capital_ratio') and 0.02 < b.capital_ratio < 0.06)
            healthy = self.summary.total_banks - defaulted - stressed
            
            self.summary.defaulted_banks = defaulted
            self.summary.stressed_banks = stressed
            self.summary.healthy_banks = healthy
            
            # Aggregate assets and capital
            total_assets = sum(b.balance_sheet.total_assets for b in banks.values() 
                              if hasattr(b, 'balance_sheet'))
            total_capital = sum(b.balance_sheet.equity for b in banks.values() 
                               if hasattr(b, 'balance_sheet'))
            
            self.summary.total_system_assets = self._format_currency(total_assets)
            self.summary.total_system_capital = self._format_currency(total_capital)
            
            if total_assets > 0:
                self.summary.system_capital_ratio = f"{total_capital / total_assets:.1%}"
            
            # Overall health
            if self.summary.total_banks > 0:
                health_pct = healthy / self.summary.total_banks
                self.summary.overall_health = int(health_pct * 100)
                
                # Stability index (penalize defaults heavily)
                default_penalty = defaulted * 20
                stress_penalty = stressed * 5
                self.summary.stability_index = max(0, 100 - default_penalty - stress_penalty)
        
        # Risk metrics
        if risk_outputs:
            pds = [r.probability_of_default for r in risk_outputs.values()]
            self.summary.average_pd = f"{np.mean(pds):.2%}" if pds else "N/A"
            
            el = sum(r.expected_loss for r in risk_outputs.values())
            self.summary.total_expected_loss = self._format_currency(el)
            
            # Systemic risk score
            systemic = sum(r.systemic_importance ** 2 for r in risk_outputs.values())
            self.summary.systemic_risk_score = min(int(systemic * 100), 100)
        
        # Compliance
        if compliance_summary:
            active_v = compliance_summary.get('active_violations', 0)
            total_v = compliance_summary.get('total_violations', 1)
            rate = 1 - active_v / max(total_v, 1)
            self.summary.compliance_rate = f"{rate:.0%}"
            self.summary.active_violations = active_v
        
        return self.summary
    
    # === Export Methods ===
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export all data for dashboard consumption."""
        return {
            'summary': self.summary.to_dict(),
            'alerts': self.get_active_alerts(),
            'entity_cards': self.get_all_entity_cards(),
            'time_series': self.get_all_time_series(),
            'network': self.network_data.to_dict(),
            'heatmaps': {k: v.to_dict() for k, v in self.heatmaps.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    def export_to_json(self, filepath: str, compressed: bool = False) -> None:
        """Export dashboard data to JSON file."""
        data = self.export_dashboard_data()
        
        if compressed:
            with gzip.open(filepath + '.gz', 'wt', encoding='utf-8') as f:
                json.dump(data, f)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    # === Utility Methods ===
    
    def _format_currency(self, value: float) -> str:
        """Format large numbers for display."""
        if abs(value) >= 1e12:
            return f"{value / 1e12:.1f}T"
        elif abs(value) >= 1e9:
            return f"{value / 1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value / 1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value / 1e3:.1f}K"
        else:
            return f"{value:.0f}"
    
    def clear(self) -> None:
        """Clear all packaged data."""
        self.alerts.clear()
        self.entity_cards.clear()
        self.time_series.clear()
        self.network_data = NetworkVisualization()
        self.heatmaps.clear()
        self.summary = SimulationSummary()
        self._alert_counter = 0
