"""
Pydantic Models for API Request/Response.
Centralized data models for the FinSim-MAPPO API.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SimulationStatus(str, Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class BankStatus(str, Enum):
    ACTIVE = "active"
    STRESSED = "stressed"
    DEFAULTED = "defaulted"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SimulationConfig(BaseModel):
    """Configuration for initializing a simulation."""
    num_banks: int = Field(default=30, ge=5, le=100, description="Number of banks")
    episode_length: int = Field(default=100, ge=10, le=1000, description="Total timesteps")
    scenario: str = Field(default="normal", description="Scenario name")
    num_exchanges: int = Field(default=2, ge=1, le=5, description="Number of exchanges")
    num_ccps: int = Field(default=1, ge=1, le=3, description="Number of CCPs")
    seed: Optional[int] = Field(default=None, description="Random seed")
    bank_ids: Optional[List[int]] = Field(default=None, description="Real bank IDs from registry to use")
    bank_names: Optional[List[str]] = Field(default=None, description="Real bank names from registry to use")
    synthetic_count: Optional[int] = Field(default=None, ge=0, le=50, description="Number of synthetic banks to add")
    synthetic_stress: Optional[str] = Field(default="normal", description="Stress level for synthetic banks: normal, weak, distressed")


class StepConfig(BaseModel):
    """Configuration for stepping simulation."""
    num_steps: int = Field(default=1, ge=1, le=100, description="Steps to execute")
    capture_state: bool = Field(default=True, description="Record full state")


class ActionRequest(BaseModel):
    """Manual action specification for banks."""
    actions: Dict[int, List[float]] = Field(..., description="Bank ID to action vector")


class ShockRequest(BaseModel):
    """External shock application."""
    price_shock: float = Field(default=0.0, ge=-1.0, le=1.0)
    volatility_shock: float = Field(default=0.0, ge=0.0, le=1.0)
    liquidity_shock: float = Field(default=0.0, ge=0.0, le=1.0)
    bank_shocks: Optional[Dict[int, float]] = None


class WhatIfTransactionRequest(BaseModel):
    """Counterfactual analysis request."""
    transaction_type: str = Field(..., description="loan_approval, margin_increase, etc.")
    initiator_id: int = Field(..., description="Source bank ID")
    counterparty_id: Optional[int] = Field(None, description="Target bank ID")
    amount: float = Field(..., ge=0, description="Transaction amount")
    interest_rate: float = Field(default=0.05, ge=0, le=1)
    duration: int = Field(default=10, ge=1, le=100)
    collateral: float = Field(default=0.0, ge=0)
    horizon: int = Field(default=10, ge=1, le=50, description="Forward simulation steps")
    num_simulations: int = Field(default=20, ge=1, le=100, description="Monte Carlo runs")


class HistoryQuery(BaseModel):
    """Query parameters for history endpoints."""
    start_step: int = Field(default=0, ge=0)
    end_step: Optional[int] = None
    fields: Optional[List[str]] = None
    bank_ids: Optional[List[int]] = None


# ============================================================================
# RESPONSE MODELS - SIMULATION
# ============================================================================

class SimulationStatusResponse(BaseModel):
    """Current simulation status."""
    status: SimulationStatus
    current_step: int
    total_steps: int
    is_done: bool
    num_banks: int
    num_exchanges: int
    num_ccps: int
    scenario: str
    started_at: Optional[str] = None


class StepResult(BaseModel):
    """Result of simulation step(s)."""
    steps_completed: int
    current_step: int
    is_done: bool
    
    # Per-bank rewards
    rewards: Dict[int, float]
    
    # System metrics
    system_metrics: "SystemMetrics"
    
    # Market state
    market_state: "MarketState"
    
    # Network stats
    network_stats: "NetworkStats"
    
    # Clearing results (if any)
    clearing_results: Optional["ClearingResults"] = None
    
    # Events this step
    events: Optional[List["SimulationEvent"]] = None


class SystemMetrics(BaseModel):
    """System-wide metrics."""
    total_defaults: int
    default_rate: float
    num_stressed: int
    num_healthy: int
    total_exposure: float
    avg_capital_ratio: float
    avg_liquidity_ratio: float
    systemic_risk_index: float


class NetworkStats(BaseModel):
    """Network topology statistics."""
    num_banks: int
    num_edges: int
    network_density: float
    avg_degree: float
    largest_component_size: int
    clustering_coefficient: float


class ClearingResults(BaseModel):
    """Clearing mechanism results."""
    algorithm: str
    iterations: int
    converged: bool
    total_claims: float
    total_payments: float
    clearing_ratio: float
    total_shortfall: float
    defaults_this_step: List[int]
    recovery_rates: Dict[int, float]


class SimulationEvent(BaseModel):
    """Event that occurred during simulation."""
    event_type: str  # 'default', 'margin_call', 'intervention'
    timestep: int
    entity_id: int
    details: Dict[str, Any]


# ============================================================================
# RESPONSE MODELS - MARKET
# ============================================================================

class MarketState(BaseModel):
    """Current market conditions."""
    asset_price: float
    price_change: float
    price_change_pct: float
    interest_rate: float
    volatility: float
    realized_volatility: float
    liquidity_index: float
    bid_ask_spread: float
    condition: str  # 'normal', 'volatile', 'stressed', 'crisis'
    stress_indicator: float


class MarketHistory(BaseModel):
    """Market time series data."""
    timesteps: List[int]
    prices: List[float]
    returns: List[float]
    volatility: List[float]
    liquidity: List[float]
    volume: List[float]


# ============================================================================
# RESPONSE MODELS - BANKS
# ============================================================================

class BankSummary(BaseModel):
    """Brief bank summary for list views."""
    bank_id: int
    tier: int
    status: BankStatus
    equity: float
    capital_ratio: float
    cash: float
    risk_score: float


class BankDetails(BaseModel):
    """Complete bank details."""
    bank_id: int
    tier: int
    status: BankStatus
    
    # Balance Sheet
    balance_sheet: "BalanceSheet"
    
    # Ratios
    capital_ratio: float
    liquidity_ratio: float
    leverage_ratio: float
    
    # Risk Metrics
    probability_of_default: float
    expected_loss: float
    risk_rating: str
    systemic_importance: float
    
    # Network Position
    network_position: "NetworkPosition"
    
    # Margins (if CCP enabled)
    margin_status: Optional["MarginStatus"] = None


class BalanceSheet(BaseModel):
    """Bank balance sheet."""
    cash: float
    illiquid_assets: float
    interbank_assets: Dict[int, float]
    interbank_liabilities: Dict[int, float]
    external_liabilities: float
    total_assets: float
    total_liabilities: float
    equity: float


class NetworkPosition(BaseModel):
    """Bank's position in the network."""
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    neighbors: List[int]
    creditors: List[int]
    debtors: List[int]
    largest_exposure: float
    concentration_index: float


class MarginStatus(BaseModel):
    """Bank's margin status at CCPs."""
    ccp_id: int
    initial_margin: float
    variation_margin: float
    total_margin: float
    margin_calls_pending: float
    margin_adequacy: float


class BankHistoryPoint(BaseModel):
    """Single timestep of bank history."""
    timestep: int
    cash: float
    equity: float
    capital_ratio: float
    status: str
    exposure: float


class BankHistory(BaseModel):
    """Bank state over time."""
    bank_id: int
    timesteps: List[int]
    data: List[BankHistoryPoint]


class TransactionRecord(BaseModel):
    """Record of a transaction."""
    transaction_id: str
    timestep: int
    transaction_type: str
    counterparty_id: int
    direction: str  # 'inflow' or 'outflow'
    amount: float
    fee: float
    status: str
    intermediaries: List[str]


class BankTransactionHistory(BaseModel):
    """Bank's transaction history."""
    bank_id: int
    transactions: List[TransactionRecord]
    summary: "TransactionSummary"


class TransactionSummary(BaseModel):
    """Summary of transaction activity."""
    total_transactions: int
    total_inflows: float
    total_outflows: float
    net_flow: float
    avg_transaction_size: float
    counterparty_count: int


# ============================================================================
# RESPONSE MODELS - ANALYTICS
# ============================================================================

class SystemicRiskMetrics(BaseModel):
    """Systemic risk analysis."""
    # DebtRank
    aggregate_debt_rank: float
    individual_debt_ranks: Dict[int, float]
    
    # Contagion
    cascade_depth: float
    cascade_potential: float
    critical_banks: List[int]
    
    # Network
    network_density: float
    clustering_coefficient: float
    concentration_index: float
    avg_path_length: float
    
    # System Health
    liquidity_index: float
    stress_index: float
    systemic_risk_index: float


class DebtRankResult(BaseModel):
    """DebtRank calculation result."""
    aggregate_debt_rank: float
    individual_ranks: Dict[int, float]
    systemically_important_banks: List[int]
    vulnerable_banks: List[int]


class ContagionAnalysis(BaseModel):
    """Contagion simulation results."""
    shocked_bank: int
    shock_magnitude: float
    
    # Cascade details
    cascade_depth: int
    total_defaults: int
    total_losses: float
    
    # Affected banks by round
    cascade_rounds: List[List[int]]
    losses_per_round: List[float]
    
    # Recovery
    system_recovery_rate: float


class CreditRiskMetrics(BaseModel):
    """Credit risk metrics for a bank or system."""
    entity_id: Optional[int] = None
    
    probability_of_default: float
    loss_given_default: float
    exposure_at_default: float
    expected_loss: float
    
    # Portfolio metrics (system-wide only)
    value_at_risk: Optional[float] = None
    expected_shortfall: Optional[float] = None
    credit_concentration: Optional[float] = None
    
    # Rating
    rating: str
    rating_outlook: str


class NetworkAnalytics(BaseModel):
    """Network structure analysis."""
    # Topology
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    max_degree: int
    
    # Components
    num_components: int
    largest_component_size: int
    
    # Centrality distribution
    centrality_stats: Dict[str, Dict[str, float]]  # type -> {min, max, mean, std}
    
    # Core-periphery
    core_banks: List[int]
    periphery_banks: List[int]
    
    # Clustering
    global_clustering: float
    avg_local_clustering: float


# ============================================================================
# RESPONSE MODELS - INFRASTRUCTURE (CCP/Exchange)
# ============================================================================

class CCPStatus(BaseModel):
    """CCP/Clearing house status."""
    ccp_id: int
    status: str  # 'normal', 'stressed', 'critical', 'failed'
    
    # Margins
    total_initial_margin: float
    total_variation_margin: float
    total_margin_pool: float
    
    # Default resources
    default_fund_size: float
    ccp_capital: float
    total_prefunded_resources: float
    
    # Coverage
    stress_coverage_ratio: float
    cover_1_ratio: float
    cover_2_ratio: float
    
    # Members
    num_members: int
    num_distressed: int
    members_on_margin_call: int
    
    # Activity
    pending_settlements: int
    margin_calls_issued: int
    stress_level: float


class ExchangeStatus(BaseModel):
    """Exchange status."""
    exchange_id: int
    
    # Volume
    transaction_volume: float
    transaction_count: int
    avg_transaction_size: float
    
    # Capacity
    max_throughput: float
    capacity_utilization: float
    congestion_level: float
    
    # Timing
    avg_settlement_delay: float
    pending_settlements: int
    
    # Fees
    total_fees_collected: float
    effective_fee_rate: float
    
    # Status
    is_stressed: bool
    circuit_breaker_active: bool


class InfrastructureOverview(BaseModel):
    """Complete infrastructure status."""
    exchanges: List[ExchangeStatus]
    ccps: List[CCPStatus]
    
    # Aggregates
    total_margin_pool: float
    total_default_resources: float
    system_congestion: float
    total_pending_settlements: int


# ============================================================================
# RESPONSE MODELS - WHAT-IF
# ============================================================================

class CounterfactualResult(BaseModel):
    """Result of what-if analysis."""
    analysis_id: str
    timestamp: str
    
    # Transaction details
    transaction_type: str
    initiator_id: int
    counterparty_id: Optional[int]
    amount: float
    
    # Baseline outcome (without transaction)
    baseline: "ScenarioOutcome"
    
    # Counterfactual outcome (with transaction)
    counterfactual: "ScenarioOutcome"
    
    # Deltas
    deltas: "OutcomeDeltas"
    
    # Risk assessment
    risk_assessment: "TransactionRiskAssessment"
    
    # Recommendation
    recommendation: str
    confidence: float
    reasoning: List[str]


class ScenarioOutcome(BaseModel):
    """Outcome of a scenario simulation."""
    initiator_equity: float
    initiator_capital_ratio: float
    initiator_defaults: bool
    
    counterparty_equity: Optional[float] = None
    counterparty_capital_ratio: Optional[float] = None
    counterparty_defaults: Optional[bool] = None
    
    system_defaults: int
    system_stressed: int
    total_exposure: float
    system_equity: float


class OutcomeDeltas(BaseModel):
    """Difference between baseline and counterfactual."""
    equity_change: float
    capital_ratio_change: float
    defaults_change: int
    exposure_change: float
    system_equity_change: float
    
    # Interpretation
    is_beneficial: bool
    primary_effect: str


class TransactionRiskAssessment(BaseModel):
    """Risk metrics for proposed transaction."""
    initiator_pd: float
    counterparty_pd: float
    expected_credit_loss: float
    
    # Systemic impact
    system_pd_increase: float
    cascade_probability: float
    contagion_depth: int
    
    # Liquidity impact
    liquidity_drain: float
    margin_call_probability: float
    
    # Overall score
    overall_risk_score: float  # 0-100


# ============================================================================
# RESPONSE MODELS - GRAPHS/VISUALIZATION
# ============================================================================

class TimeSeriesData(BaseModel):
    """Generic time series for charting."""
    name: str
    timestamps: List[int]
    values: List[float]
    unit: str = ""
    chart_type: str = "line"


class MultiSeriesData(BaseModel):
    """Multiple time series for comparison charts."""
    series: List[TimeSeriesData]
    title: str
    x_label: str = "Timestep"
    y_label: str = ""


class NetworkVisualization(BaseModel):
    """Network graph data for visualization."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str = "force"  # 'force', 'circular', 'hierarchical'


class HeatmapData(BaseModel):
    """Heatmap data (e.g., exposure matrix)."""
    name: str
    row_labels: List[str]
    col_labels: List[str]
    values: List[List[float]]
    min_value: float
    max_value: float
    color_scale: str = "RdYlGn"


class DashboardData(BaseModel):
    """Complete dashboard data package."""
    # Summary cards
    summary: "DashboardSummary"
    
    # Alerts
    alerts: List["Alert"]
    
    # Charts
    price_chart: TimeSeriesData
    defaults_chart: TimeSeriesData
    capital_chart: TimeSeriesData
    stress_chart: TimeSeriesData
    
    # Tables
    bank_table: List[BankSummary]
    
    # Network
    network_graph: NetworkVisualization


class DashboardSummary(BaseModel):
    """Summary metrics for dashboard header."""
    current_step: int
    total_steps: int
    
    overall_health: int  # 0-100
    stability_score: int  # 0-100
    
    total_banks: int
    healthy_banks: int
    stressed_banks: int
    defaulted_banks: int
    
    total_assets: str  # Formatted (e.g., "150.5B")
    total_equity: str
    avg_capital_ratio: str
    
    market_price: float
    market_condition: str


class Alert(BaseModel):
    """Alert/notification for UI."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    entity_id: Optional[int] = None
    timestep: int
    action_required: bool = False


# ============================================================================
# UPDATE FORWARD REFS
# ============================================================================

StepResult.model_rebuild()
BankDetails.model_rebuild()
CounterfactualResult.model_rebuild()
DashboardData.model_rebuild()
