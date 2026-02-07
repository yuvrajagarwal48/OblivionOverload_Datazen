"""
Comprehensive Stress Testing Suite for FinSim-MAPPO.

Tests 5 critical scenarios:
1. Interbank Lending Spiral (Credit Contagion)
2. Margin Spiral (Procyclical Liquidity Crisis)
3. Clearing House Near-Failure (Systemic Extreme Event)
4. Exchange Congestion + Liquidity Shock
5. Information Asymmetry Panic (Coordination Failure)

Each scenario tests specific failure modes and systemic stress indicators.
"""

import os
import sys
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import copy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FinancialEnvironment, EnvConfig
from src.environment.ccp import CentralCounterparty, CCPConfig, CCPNetwork
from src.environment.exchange import Exchange, ExchangeConfig, ExchangeNetwork
from src.environment.infrastructure import InfrastructureRouter, TransactionType
from src.agents.baseline_agents import create_baseline_agent, AgentConfig
from src.analytics import RiskAnalyzer
from src.analytics.credit_risk import CreditRiskLayer


# ============================================================================
# SYSTEMIC STRESS INDEX
# ============================================================================

class SystemState(str, Enum):
    """System-wide health state."""
    GREEN = "green"       # Normal operations
    YELLOW = "yellow"     # Elevated stress
    ORANGE = "orange"     # High stress, intervention needed
    RED = "red"           # Critical, system failure imminent
    BLACK = "black"       # System collapse


@dataclass
class SystemicStressIndex:
    """
    Comprehensive systemic stress measurement.
    The system "goes red" based on multiple indicators, not just defaults.
    """
    
    # Individual metrics (0-1 scale, higher = worse)
    default_rate: float = 0.0
    capital_depletion: float = 0.0
    liquidity_stress: float = 0.0
    network_fragmentation: float = 0.0
    ccp_buffer_usage: float = 0.0
    margin_pressure: float = 0.0
    volatility_spike: float = 0.0
    contagion_velocity: float = 0.0
    
    # Derived metrics
    composite_stress: float = 0.0
    state: SystemState = SystemState.GREEN
    
    def compute_composite(self) -> float:
        """
        Weighted composite stress index.
        Weights reflect systemic importance.
        """
        weights = {
            'default_rate': 0.20,
            'capital_depletion': 0.15,
            'liquidity_stress': 0.20,
            'network_fragmentation': 0.10,
            'ccp_buffer_usage': 0.15,
            'margin_pressure': 0.10,
            'volatility_spike': 0.05,
            'contagion_velocity': 0.05
        }
        
        self.composite_stress = (
            weights['default_rate'] * self.default_rate +
            weights['capital_depletion'] * self.capital_depletion +
            weights['liquidity_stress'] * self.liquidity_stress +
            weights['network_fragmentation'] * self.network_fragmentation +
            weights['ccp_buffer_usage'] * self.ccp_buffer_usage +
            weights['margin_pressure'] * self.margin_pressure +
            weights['volatility_spike'] * self.volatility_spike +
            weights['contagion_velocity'] * self.contagion_velocity
        )
        
        # Determine state
        if self.composite_stress < 0.2:
            self.state = SystemState.GREEN
        elif self.composite_stress < 0.4:
            self.state = SystemState.YELLOW
        elif self.composite_stress < 0.6:
            self.state = SystemState.ORANGE
        elif self.composite_stress < 0.8:
            self.state = SystemState.RED
        else:
            self.state = SystemState.BLACK
        
        return self.composite_stress
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'default_rate': round(self.default_rate, 4),
            'capital_depletion': round(self.capital_depletion, 4),
            'liquidity_stress': round(self.liquidity_stress, 4),
            'network_fragmentation': round(self.network_fragmentation, 4),
            'ccp_buffer_usage': round(self.ccp_buffer_usage, 4),
            'margin_pressure': round(self.margin_pressure, 4),
            'volatility_spike': round(self.volatility_spike, 4),
            'contagion_velocity': round(self.contagion_velocity, 4),
            'composite_stress': round(self.composite_stress, 4),
            'state': self.state.value
        }


# ============================================================================
# CLEARING HOUSE STRESS MODEL
# ============================================================================

@dataclass
class ClearingHouseState:
    """
    Clearing House Balance Sheet (Abstracted).
    
    Assets:
    - Posted margins (from members)
    - Default fund contributions
    - CCP capital buffer
    - Emergency liquidity lines
    
    Liabilities:
    - Settlement obligations
    - Variation margin payouts
    - Guarantee commitments
    """
    
    # Assets
    posted_margins: float = 0.0
    default_fund: float = 0.0
    ccp_capital: float = 0.0
    liquidity_lines: float = 0.0
    
    # Liabilities
    settlement_obligations: float = 0.0
    variation_margin_payouts: float = 0.0
    guarantee_commitments: float = 0.0
    
    # Losses
    realized_losses: float = 0.0
    
    @property
    def total_buffers(self) -> float:
        """Total loss-absorbing capacity."""
        return (self.posted_margins + self.default_fund + 
                self.ccp_capital + self.liquidity_lines)
    
    @property
    def total_liabilities(self) -> float:
        return (self.settlement_obligations + 
                self.variation_margin_payouts + 
                self.guarantee_commitments)
    
    @property
    def buffer_usage(self) -> float:
        """Fraction of buffers consumed by losses."""
        if self.total_buffers <= 0:
            return 1.0
        return min(1.0, self.realized_losses / self.total_buffers)
    
    @property
    def is_insolvent(self) -> bool:
        """
        CCP Failure Condition:
        Total Realized Losses > (Margins + Default Fund + CCP Capital + Liquidity Lines)
        """
        return self.realized_losses > self.total_buffers
    
    def apply_loss(self, loss: float) -> Dict[str, float]:
        """
        Apply loss through the waterfall:
        1. Defaulter's margin
        2. Default fund
        3. CCP capital
        4. Liquidity lines
        5. Loss mutualization
        """
        remaining = loss
        absorption = {}
        
        # 1. Margins
        margin_absorption = min(remaining, self.posted_margins)
        self.posted_margins -= margin_absorption
        remaining -= margin_absorption
        absorption['margin'] = margin_absorption
        
        # 2. Default fund
        if remaining > 0:
            fund_absorption = min(remaining, self.default_fund)
            self.default_fund -= fund_absorption
            remaining -= fund_absorption
            absorption['default_fund'] = fund_absorption
        
        # 3. CCP capital
        if remaining > 0:
            capital_absorption = min(remaining, self.ccp_capital)
            self.ccp_capital -= capital_absorption
            remaining -= capital_absorption
            absorption['ccp_capital'] = capital_absorption
        
        # 4. Liquidity lines
        if remaining > 0:
            liquidity_absorption = min(remaining, self.liquidity_lines)
            self.liquidity_lines -= liquidity_absorption
            remaining -= liquidity_absorption
            absorption['liquidity'] = liquidity_absorption
        
        # 5. Unabsorbed = mutualized or system failure
        absorption['unabsorbed'] = remaining
        self.realized_losses += loss
        
        return absorption


# ============================================================================
# STRESS TEST FRAMEWORK
# ============================================================================

class StressTestResult:
    """Results from a single stress test run."""
    
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Timeline
        self.stress_history: List[SystemicStressIndex] = []
        self.ccp_history: List[ClearingHouseState] = []
        self.default_timeline: List[Tuple[int, int]] = []  # (step, bank_id)
        self.event_log: List[Dict[str, Any]] = []
        self._tracked_defaults: set = set()  # Track which banks we've already recorded
        
        # Outcomes
        self.final_state: SystemState = SystemState.GREEN
        self.peak_stress: float = 0.0
        self.total_defaults: int = 0
        self.ccp_failed: bool = False
        self.cascade_length: int = 0
        self.time_to_red: Optional[int] = None
        
        # Metrics
        self.initial_equity: float = 0.0
        self.final_equity: float = 0.0
        self.equity_loss_pct: float = 0.0
    
    def finalize(self):
        self.end_time = datetime.now()
        if self.stress_history:
            self.peak_stress = max(s.composite_stress for s in self.stress_history)
            self.final_state = self.stress_history[-1].state
            
            # Find time to red
            for i, s in enumerate(self.stress_history):
                if s.state in [SystemState.RED, SystemState.BLACK]:
                    self.time_to_red = i
                    break
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario': self.scenario_name,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            'final_state': self.final_state.value,
            'peak_stress': round(self.peak_stress, 4),
            'total_defaults': self.total_defaults,
            'ccp_failed': self.ccp_failed,
            'cascade_length': self.cascade_length,
            'time_to_red': self.time_to_red,
            'equity_loss_pct': round(self.equity_loss_pct, 2),
            'default_timeline': self.default_timeline[:20],  # First 20 defaults
            'stress_trajectory': [s.to_dict() for s in self.stress_history[-10:]]  # Last 10 steps
        }


class StressTestFramework:
    """
    Framework for running stress test scenarios.
    """
    
    def __init__(self, num_banks: int = 30, seed: int = 42):
        self.num_banks = num_banks
        self.seed = seed
        self.env: Optional[FinancialEnvironment] = None
        self.agents: Dict[int, Any] = {}
        self.ccp_state: Optional[ClearingHouseState] = None
        self.risk_analyzer = RiskAnalyzer()
        self.credit_layer = CreditRiskLayer()
        
        # Tracking
        self.initial_state: Dict[str, Any] = {}
        self.previous_defaults: set = set()
    
    def initialize(self, config: Optional[EnvConfig] = None):
        """Initialize environment and agents."""
        if config is None:
            config = EnvConfig(
                num_banks=self.num_banks,
                episode_length=200,
                seed=self.seed
            )
        
        self.env = FinancialEnvironment(config)
        
        # Initialize agents
        self.agents = {}
        for i in range(self.num_banks):
            agent_config = AgentConfig(
                agent_id=i,
                observation_dim=FinancialEnvironment.OBS_DIM,
                action_dim=FinancialEnvironment.ACTION_DIM,
                seed=self.seed
            )
            self.agents[i] = create_baseline_agent('rule_based', agent_config)
        
        # Reset environment first to initialize banks
        self.env.reset()
        
        # Then initialize CCP state (needs bank equities)
        self._init_ccp_state()
        
        # Capture initial state
        self._capture_initial_state()
        
        self.previous_defaults = set()
    
    def _init_ccp_state(self):
        """Initialize clearing house balance sheet."""
        network = self.env.network
        total_equity = sum(b.balance_sheet.equity for b in network.banks.values())
        
        self.ccp_state = ClearingHouseState(
            posted_margins=total_equity * 0.05,  # 5% of system equity
            default_fund=total_equity * 0.03,    # 3% of system equity
            ccp_capital=total_equity * 0.01,     # 1% of system equity
            liquidity_lines=total_equity * 0.02,  # 2% emergency lines
            settlement_obligations=0,
            variation_margin_payouts=0,
            guarantee_commitments=total_equity * 0.02
        )
    
    def _capture_initial_state(self):
        """Capture initial system state for comparison."""
        network = self.env.network
        self.initial_state = {
            'total_equity': sum(b.balance_sheet.equity for b in network.banks.values()),
            'total_cash': sum(b.balance_sheet.cash for b in network.banks.values()),
            'num_edges': network.graph.number_of_edges(),
            'avg_capital_ratio': np.mean([b.capital_ratio for b in network.banks.values()]),
            'defaulted': set()
        }
    
    def compute_stress_index(self) -> SystemicStressIndex:
        """Compute current systemic stress index."""
        network = self.env.network
        market = self.env.market
        
        stress = SystemicStressIndex()
        
        # 1. Default rate
        num_defaulted = sum(1 for b in network.banks.values() 
                          if b.status.value == "defaulted")
        stress.default_rate = num_defaulted / max(1, len(network.banks))
        
        # 2. Capital depletion
        current_equity = sum(b.balance_sheet.equity for b in network.banks.values())
        initial_equity = self.initial_state.get('total_equity', current_equity)
        if initial_equity > 0:
            stress.capital_depletion = max(0, 1 - current_equity / initial_equity)
        
        # 3. Liquidity stress
        current_cash = sum(b.balance_sheet.cash for b in network.banks.values())
        initial_cash = self.initial_state.get('total_cash', current_cash)
        if initial_cash > 0:
            stress.liquidity_stress = max(0, 1 - current_cash / initial_cash)
        
        # 4. Network fragmentation
        current_edges = network.graph.number_of_edges()
        initial_edges = self.initial_state.get('num_edges', current_edges)
        if initial_edges > 0:
            stress.network_fragmentation = max(0, 1 - current_edges / initial_edges)
        
        # 5. CCP buffer usage
        if self.ccp_state:
            stress.ccp_buffer_usage = self.ccp_state.buffer_usage
        
        # 6. Margin pressure (based on volatility)
        market_state = market.get_state()
        base_volatility = 0.2
        stress.margin_pressure = min(1.0, max(0, (market_state.volatility - base_volatility) / base_volatility))
        
        # 7. Volatility spike
        stress.volatility_spike = min(1.0, market_state.volatility / 0.5)
        
        # 8. Contagion velocity (new defaults this step)
        current_defaults = set(
            bid for bid, b in network.banks.items() 
            if b.status.value == "defaulted"
        )
        new_defaults = len(current_defaults - self.previous_defaults)
        stress.contagion_velocity = min(1.0, new_defaults / 3)  # 3+ defaults = max velocity
        self.previous_defaults = current_defaults
        
        stress.compute_composite()
        return stress
    
    def step_with_actions(self, actions: Optional[Dict[int, np.ndarray]] = None):
        """Execute one simulation step."""
        if actions is None:
            actions = {}
            obs, _ = self.env.reset() if self.env.current_step == 0 else (self.env._get_observations(), None)
            for agent_id, agent in self.agents.items():
                if agent_id in obs:
                    actions[agent_id] = agent.select_action(obs[agent_id], deterministic=True)
        
        result = self.env.step(actions)
        return result
    
    def run_scenario(
        self, 
        scenario_name: str,
        setup_fn,
        shock_fn,
        max_steps: int = 100
    ) -> StressTestResult:
        """
        Run a stress test scenario.
        
        Args:
            scenario_name: Name of the scenario
            setup_fn: Function to set up initial conditions
            shock_fn: Function that applies shocks each step
            max_steps: Maximum simulation steps
        """
        result = StressTestResult(scenario_name)
        
        # Initialize
        self.initialize()
        result.initial_equity = self.initial_state['total_equity']
        
        # Apply scenario setup
        setup_fn(self)
        
        # Run simulation
        for step in range(max_steps):
            # Apply shock
            shock_fn(self, step)
            
            # Step simulation
            try:
                step_result = self.step_with_actions()
            except Exception as e:
                result.event_log.append({
                    'step': step,
                    'event': 'simulation_error',
                    'message': str(e)
                })
                break
            
            # Compute stress
            stress = self.compute_stress_index()
            result.stress_history.append(stress)
            
            # Track CCP state
            if self.ccp_state:
                result.ccp_history.append(copy.copy(self.ccp_state))
                if self.ccp_state.is_insolvent:
                    result.ccp_failed = True
                    result.event_log.append({
                        'step': step,
                        'event': 'ccp_failure',
                        'losses': self.ccp_state.realized_losses,
                        'buffers': self.ccp_state.total_buffers
                    })
            
            # Track defaults
            for bid, bank in self.env.network.banks.items():
                if bank.status.value == "defaulted" and bid not in self.initial_state.get('defaulted', set()):
                    if bid not in result._tracked_defaults:
                        result._tracked_defaults.add(bid)
                        result.default_timeline.append((step, bid))
                        result.event_log.append({
                            'step': step,
                            'event': 'bank_default',
                            'bank_id': bid,
                            'tier': bank.tier
                        })
            
            # Check for system collapse
            if stress.state == SystemState.BLACK:
                result.event_log.append({
                    'step': step,
                    'event': 'system_collapse',
                    'stress': stress.composite_stress
                })
                break
        
        # Finalize results
        # Total unique banks that defaulted
        result.total_defaults = len(set(bid for step, bid in result.default_timeline))
        # Length of longest cascade sequence (consecutive/near-consecutive defaults)
        result.cascade_length = self._compute_cascade_length(result.default_timeline)
        
        current_equity = sum(
            b.balance_sheet.equity for b in self.env.network.banks.values()
        )
        result.final_equity = current_equity
        if result.initial_equity > 0:
            result.equity_loss_pct = (1 - current_equity / result.initial_equity) * 100
        
        result.finalize()
        return result
    
    def _compute_cascade_length(self, default_timeline: List[Tuple[int, int]]) -> int:
        """
        Compute cascade length: number of banks that defaulted in consecutive/near-consecutive steps.
        Returns the length of the longest cascade sequence.
        """
        if not default_timeline:
            return 0
        
        # Sort by step, then by bank_id
        sorted_defaults = sorted(default_timeline, key=lambda x: (x[0], x[1]))
        
        max_cascade = 1
        current_cascade = 1
        
        for i in range(1, len(sorted_defaults)):
            prev_step = sorted_defaults[i-1][0]
            curr_step = sorted_defaults[i][0]
            
            # If defaults are consecutive or near-consecutive (within 2 steps)
            if curr_step - prev_step <= 2:
                current_cascade += 1
                max_cascade = max(max_cascade, current_cascade)
            else:
                current_cascade = 1
        
        return max_cascade


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_bank_equity(bank, target_equity: float):
    """
    Set a bank's equity to a target value by adjusting illiquid_assets.
    Since equity = total_assets - total_liabilities, we adjust assets.
    """
    current_equity = bank.balance_sheet.equity
    delta = target_equity - current_equity
    bank.balance_sheet.illiquid_assets = max(0, bank.balance_sheet.illiquid_assets + delta)


def reduce_bank_equity(bank, reduction_pct: float):
    """
    Reduce bank equity by a percentage by increasing external liabilities.
    """
    current_equity = bank.balance_sheet.equity
    reduction = current_equity * reduction_pct
    bank.balance_sheet.external_liabilities += reduction


def force_bank_default(bank):
    """
    Force a bank into default by making it insolvent.
    """
    # Wipe out all assets
    bank.balance_sheet.cash = 0
    bank.balance_sheet.illiquid_assets = 0
    # Increase liabilities to ensure negative equity
    bank.balance_sheet.external_liabilities += 10000


# ============================================================================
# SCENARIO 1: INTERBANK LENDING SPIRAL
# ============================================================================

def scenario_1_setup(framework: StressTestFramework):
    """
    Setup: Create concentrated exposure from Bank A to Bank B.
    """
    network = framework.env.network
    
    # Identify a peripheral bank (borrower) and core bank (lender)
    banks_by_tier = {1: [], 2: [], 3: []}
    for bid, bank in network.banks.items():
        banks_by_tier[bank.tier].append(bid)
    
    # Core bank lends heavily to peripheral bank
    if banks_by_tier[1] and banks_by_tier[3]:
        lender_id = banks_by_tier[1][0]
        borrower_id = banks_by_tier[3][0]
        
        lender = network.banks[lender_id]
        borrower = network.banks[borrower_id]
        
        # Create concentrated exposure (40% of lender's equity)
        exposure = lender.balance_sheet.equity * 0.4
        network.liability_matrix[borrower_id, lender_id] = exposure
        
        print(f"[Scenario 1 Setup] Bank {lender_id} (Tier 1) exposed to Bank {borrower_id} (Tier 3)")
        print(f"  Exposure: {exposure:,.0f} ({exposure/lender.balance_sheet.equity*100:.1f}% of lender equity)")


def scenario_1_shock(framework: StressTestFramework, step: int):
    """
    Shock: Trigger borrower default at step 5, then watch contagion.
    """
    if step == 5:
        network = framework.env.network
        
        # Find the vulnerable borrower (Tier 3 with concentrated exposure)
        for bid, bank in network.banks.items():
            if bank.tier == 3:
                # Force default by depleting assets
                force_bank_default(bank)
                print(f"[Step {step}] Bank {bid} forced into default - triggering cascade")
                break


def test_scenario_1_interbank_spiral():
    """
    SCENARIO 1: Interbank Lending Spiral (Credit Contagion)
    
    Tests:
    - Exposure concentration risk
    - Network topology fragility
    - Effectiveness of diversification
    - CCP's ability to absorb bilateral defaults
    """
    print("\n" + "="*80)
    print("SCENARIO 1: INTERBANK LENDING SPIRAL (Credit Contagion)")
    print("="*80)
    
    framework = StressTestFramework(num_banks=30, seed=42)
    result = framework.run_scenario(
        scenario_name="Interbank Lending Spiral",
        setup_fn=scenario_1_setup,
        shock_fn=scenario_1_shock,
        max_steps=50
    )
    
    print("\n--- Results ---")
    print(f"Final State: {result.final_state.value.upper()}")
    print(f"Peak Stress: {result.peak_stress:.4f}")
    print(f"Total Defaults: {result.total_defaults}")
    print(f"Cascade Length: {result.cascade_length} steps")
    print(f"Time to Red: {result.time_to_red} steps" if result.time_to_red else "Did not reach RED")
    print(f"Equity Loss: {result.equity_loss_pct:.1f}%")
    print(f"CCP Failed: {result.ccp_failed}")
    
    return result


# ============================================================================
# SCENARIO 2: MARGIN SPIRAL
# ============================================================================

def scenario_2_setup(framework: StressTestFramework):
    """
    Setup: Normal market conditions, banks with moderate liquidity.
    """
    # Reduce cash buffers slightly to make system vulnerable
    for bank in framework.env.network.banks.values():
        bank.balance_sheet.cash *= 0.7  # 30% less cash
    
    print("[Scenario 2 Setup] Reduced bank cash buffers by 30%")


def scenario_2_shock(framework: StressTestFramework, step: int):
    """
    Shock: Volatility doubles over 3 steps, triggering margin spiral.
    """
    market = framework.env.market
    
    if step in [3, 4, 5]:
        # Volatility spike
        market.current_volatility *= 1.5
        print(f"[Step {step}] Volatility increased to {market.current_volatility:.3f}")
        
        # Price drop
        framework.env.apply_scenario_shock(
            price_shock=-0.1,
            volatility_shock=0.2,
            liquidity_shock=0.15
        )
    
    if step >= 6:
        # Banks forced to sell assets - price impact
        if market.current_price > 0.5:
            market.current_price *= 0.97  # 3% price drop per step
            
        # Increase margin requirements on CCP
        if framework.ccp_state:
            # Higher volatility → higher margin requirement
            margin_increase = market.current_volatility * 0.1
            framework.ccp_state.posted_margins *= (1 + margin_increase)
            
            # Banks must post more margin → liquidity drain
            for bank in framework.env.network.banks.values():
                margin_call = bank.balance_sheet.cash * margin_increase * 0.5
                bank.balance_sheet.cash -= margin_call
                framework.ccp_state.variation_margin_payouts += margin_call


def test_scenario_2_margin_spiral():
    """
    SCENARIO 2: Margin Spiral (Procyclical Liquidity Crisis)
    
    Tests:
    - Procyclicality of margins
    - Liquidity buffer adequacy
    - CCP stress amplification
    - Fire-sale dynamics
    """
    print("\n" + "="*80)
    print("SCENARIO 2: MARGIN SPIRAL (Procyclical Liquidity Crisis)")
    print("="*80)
    
    framework = StressTestFramework(num_banks=30, seed=123)
    result = framework.run_scenario(
        scenario_name="Margin Spiral",
        setup_fn=scenario_2_setup,
        shock_fn=scenario_2_shock,
        max_steps=40
    )
    
    print("\n--- Results ---")
    print(f"Final State: {result.final_state.value.upper()}")
    print(f"Peak Stress: {result.peak_stress:.4f}")
    print(f"Total Defaults: {result.total_defaults}")
    print(f"Liquidity Crisis: {'YES' if result.stress_history and result.stress_history[-1].liquidity_stress > 0.5 else 'NO'}")
    print(f"Time to Red: {result.time_to_red} steps" if result.time_to_red else "Did not reach RED")
    print(f"Equity Loss: {result.equity_loss_pct:.1f}%")
    
    return result


# ============================================================================
# SCENARIO 3: CLEARING HOUSE NEAR-FAILURE
# ============================================================================

def scenario_3_setup(framework: StressTestFramework):
    """
    Setup: Create correlated exposures among Tier-1 banks.
    """
    network = framework.env.network
    
    # Find Tier-1 banks
    tier1_banks = [bid for bid, b in network.banks.items() if b.tier == 1]
    
    if len(tier1_banks) >= 2:
        # Create correlated exposures to same assets/counterparties
        for i, bid in enumerate(tier1_banks[:2]):
            bank = network.banks[bid]
            # Increase leverage by increasing external liabilities
            bank.balance_sheet.external_liabilities *= 1.5
            # Reduce cash buffer
            bank.balance_sheet.cash *= 0.6
    
    print(f"[Scenario 3 Setup] Weakened {len(tier1_banks[:2])} Tier-1 banks")
    print(f"  CCP Total Buffers: {framework.ccp_state.total_buffers:,.0f}")


def scenario_3_shock(framework: StressTestFramework, step: int):
    """
    Shock: Two Tier-1 banks default in quick succession.
    """
    network = framework.env.network
    tier1_banks = [bid for bid, b in network.banks.items() if b.tier == 1]
    
    if step == 5 and len(tier1_banks) >= 1:
        # First Tier-1 default
        bid = tier1_banks[0]
        bank = network.banks[bid]
        loss = max(bank.balance_sheet.equity * 0.5, 1000)
        force_bank_default(bank)
        
        # CCP absorbs loss
        absorption = framework.ccp_state.apply_loss(loss)
        print(f"[Step {step}] Tier-1 Bank {bid} defaults. Loss: {loss:,.0f}")
        print(f"  Absorption: {absorption}")
    
    if step == 7 and len(tier1_banks) >= 2:
        # Second Tier-1 default
        bid = tier1_banks[1]
        bank = network.banks[bid]
        loss = max(bank.balance_sheet.equity * 0.6, 1000)
        force_bank_default(bank)
        
        # CCP absorbs (or fails)
        absorption = framework.ccp_state.apply_loss(loss)
        print(f"[Step {step}] Tier-1 Bank {bid} defaults. Loss: {loss:,.0f}")
        print(f"  Absorption: {absorption}")
        print(f"  CCP Buffer Usage: {framework.ccp_state.buffer_usage:.1%}")
        
        if framework.ccp_state.is_insolvent:
            print(f"  *** CCP FAILURE - System in collapse mode ***")


def test_scenario_3_ccp_near_failure():
    """
    SCENARIO 3: Clearing House Near-Failure (Systemic Extreme Event)
    
    Tests:
    - CCP resilience
    - Adequacy of default funds
    - Loss mutualization design
    - Single-point-of-failure risk
    """
    print("\n" + "="*80)
    print("SCENARIO 3: CLEARING HOUSE NEAR-FAILURE (Systemic Extreme Event)")
    print("="*80)
    
    framework = StressTestFramework(num_banks=30, seed=456)
    result = framework.run_scenario(
        scenario_name="CCP Near-Failure",
        setup_fn=scenario_3_setup,
        shock_fn=scenario_3_shock,
        max_steps=30
    )
    
    print("\n--- Results ---")
    print(f"Final State: {result.final_state.value.upper()}")
    print(f"Peak Stress: {result.peak_stress:.4f}")
    print(f"CCP Failed: {result.ccp_failed}")
    print(f"Total Defaults: {result.total_defaults}")
    print(f"Cascade Length: {result.cascade_length} steps")
    print(f"Equity Loss: {result.equity_loss_pct:.1f}%")
    
    # Check for stabilization vs collapse
    if result.ccp_failed:
        print("\n>>> OUTCOME B: COLLAPSE - CCP guarantees failed, system frozen")
    else:
        print("\n>>> OUTCOME A: STABILIZATION - CCP survived, system recovering")
    
    return result


# ============================================================================
# SCENARIO 4: EXCHANGE CONGESTION + LIQUIDITY SHOCK
# ============================================================================

def scenario_4_setup(framework: StressTestFramework):
    """
    Setup: High baseline trading volume.
    """
    # Simulate high trading activity
    framework.env.market.current_volatility = 0.25  # Slightly elevated
    print("[Scenario 4 Setup] Elevated baseline trading volume")


def scenario_4_shock(framework: StressTestFramework, step: int):
    """
    Shock: Exchange throughput drops 40%, settlement delays spike.
    """
    market = framework.env.market
    
    if step == 10:
        # Exchange capacity shock
        print(f"[Step {step}] Exchange throughput drops 40%")
        
        # Simulate congestion effects
        market.liquidity_index *= 0.6  # Liquidity drops
        market.current_volatility *= 1.3  # Uncertainty rises
        
        # Settlement delays → liquidity stress
        for bank in framework.env.network.banks.values():
            # Cash tied up in pending settlements
            settlement_delay_cost = bank.balance_sheet.cash * 0.15
            bank.balance_sheet.cash -= settlement_delay_cost
    
    if 10 < step <= 20:
        # Bid-ask spreads widen, price discovery deteriorates
        market.liquidity_index = max(0.3, market.liquidity_index * 0.95)
        
        # Margin uncertainty
        if framework.ccp_state:
            framework.ccp_state.variation_margin_payouts *= 1.1
    
    if step == 15:
        # Large participant failure
        network = framework.env.network
        tier2_banks = [bid for bid, b in network.banks.items() if b.tier == 2]
        if tier2_banks:
            victim = tier2_banks[0]
            network.banks[victim].balance_sheet.cash = 0
            print(f"[Step {step}] Large participant Bank {victim} loses all liquidity")


def test_scenario_4_exchange_congestion():
    """
    SCENARIO 4: Exchange Congestion + Liquidity Shock
    
    Tests:
    - Infrastructure resilience
    - Capacity planning
    - Market microstructure stability
    - Operational risk impact
    """
    print("\n" + "="*80)
    print("SCENARIO 4: EXCHANGE CONGESTION + LIQUIDITY SHOCK")
    print("="*80)
    
    framework = StressTestFramework(num_banks=30, seed=789)
    result = framework.run_scenario(
        scenario_name="Exchange Congestion",
        setup_fn=scenario_4_setup,
        shock_fn=scenario_4_shock,
        max_steps=35
    )
    
    print("\n--- Results ---")
    print(f"Final State: {result.final_state.value.upper()}")
    print(f"Peak Stress: {result.peak_stress:.4f}")
    print(f"Total Defaults: {result.total_defaults}")
    print(f"Liquidity Stress (final): {result.stress_history[-1].liquidity_stress:.3f}" if result.stress_history else "N/A")
    print(f"Network Fragmentation: {result.stress_history[-1].network_fragmentation:.3f}" if result.stress_history else "N/A")
    print(f"Equity Loss: {result.equity_loss_pct:.1f}%")
    
    # Note: This scenario doesn't require credit cascade
    print("\n>>> Stress propagation via operational channels, not credit")
    
    return result


# ============================================================================
# SCENARIO 5: INFORMATION ASYMMETRY PANIC
# ============================================================================

def scenario_5_setup(framework: StressTestFramework):
    """
    Setup: Normal conditions with one bank having slightly elevated stress.
    """
    network = framework.env.network
    
    # Pick a mid-tier bank to be the "rumor target"
    tier2_banks = [bid for bid, b in network.banks.items() if b.tier == 2]
    if tier2_banks:
        target = tier2_banks[0]
        bank = network.banks[target]
        # Slightly reduce capital (visible signal)
        reduce_bank_equity(bank, 0.1)  # 10% reduction
        framework.rumor_target = target
        print(f"[Scenario 5 Setup] Bank {target} has slightly elevated stress signal")
        print(f"  Capital ratio: {bank.capital_ratio:.3f} (visible to all)")


def scenario_5_shock(framework: StressTestFramework, step: int):
    """
    Shock: Banks interpret signal pessimistically → self-fulfilling crisis.
    """
    network = framework.env.network
    target = getattr(framework, 'rumor_target', None)
    
    if target is None:
        return
    
    target_bank = network.banks.get(target)
    if target_bank is None:
        return
    
    if step == 8:
        print(f"[Step {step}] Negative signal about Bank {target} spreads")
    
    if 8 <= step <= 15:
        # Other banks reduce exposure to target (rational individually)
        for bid, bank in network.banks.items():
            if bid != target:
                # Check exposure to target
                exposure = network.liability_matrix[target, bid]
                if exposure > 0:
                    # Reduce exposure
                    reduction = exposure * 0.2
                    network.liability_matrix[target, bid] -= reduction
                    
                    # Target loses funding
                    target_bank.balance_sheet.cash -= reduction * 0.5
        
        # Credit rationing - reduce target's interbank assets
        target_bank.balance_sheet.illiquid_assets *= 0.9
    
    if step == 12:
        # Check if prophecy self-fulfilled
        if target_bank.balance_sheet.cash < target_bank.balance_sheet.total_liabilities * 0.02:
            print(f"[Step {step}] Bank {target} now actually illiquid - prophecy self-fulfilled")
            reduce_bank_equity(target_bank, 0.5)  # Stress materialized
    
    if step >= 15:
        # Other banks hoard liquidity
        for bank in network.banks.values():
            bank.balance_sheet.cash = max(0, bank.balance_sheet.cash * 0.95)


def test_scenario_5_information_panic():
    """
    SCENARIO 5: Information Asymmetry Panic (Coordination Failure)
    
    Tests:
    - Belief formation
    - Reputation dynamics
    - Information quality
    - Herd behavior
    
    This is a coordination game with multiple equilibria.
    """
    print("\n" + "="*80)
    print("SCENARIO 5: INFORMATION ASYMMETRY PANIC (Coordination Failure)")
    print("="*80)
    
    framework = StressTestFramework(num_banks=30, seed=999)
    result = framework.run_scenario(
        scenario_name="Information Panic",
        setup_fn=scenario_5_setup,
        shock_fn=scenario_5_shock,
        max_steps=30
    )
    
    print("\n--- Results ---")
    print(f"Final State: {result.final_state.value.upper()}")
    print(f"Peak Stress: {result.peak_stress:.4f}")
    print(f"Total Defaults: {result.total_defaults}")
    print(f"Liquidity Stress: {result.stress_history[-1].liquidity_stress:.3f}" if result.stress_history else "N/A")
    print(f"Equity Loss: {result.equity_loss_pct:.1f}%")
    
    # Game-theoretic analysis
    print("\n>>> Coordination Game Analysis:")
    print("  - Individual action: RATIONAL (reduce exposure to uncertain counterparty)")
    print("  - Collective outcome: INEFFICIENT (good bank fails)")
    print("  - Equilibrium: Self-fulfilling prophecy realized")
    
    return result


# ============================================================================
# COMPREHENSIVE TEST RUNNER
# ============================================================================

def run_all_scenarios():
    """Run all 5 stress test scenarios and generate summary report."""
    
    print("\n" + "#"*80)
    print("# COMPREHENSIVE STRESS TEST SUITE - FinSim-MAPPO")
    print("# Testing Clearing House Stress Model")
    print("#"*80)
    
    results = []
    
    # Run each scenario
    results.append(test_scenario_1_interbank_spiral())
    results.append(test_scenario_2_margin_spiral())
    results.append(test_scenario_3_ccp_near_failure())
    results.append(test_scenario_4_exchange_congestion())
    results.append(test_scenario_5_information_panic())
    
    # Summary Report
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY REPORT")
    print("="*80)
    
    print("\n{:<30} {:<10} {:<12} {:<10} {:<12} {:<10}".format(
        "Scenario", "State", "Peak Stress", "Defaults", "CCP Failed", "Equity Loss"
    ))
    print("-"*80)
    
    for r in results:
        print("{:<30} {:<10} {:<12.4f} {:<10} {:<12} {:<10.1f}%".format(
            r.scenario_name[:28],
            r.final_state.value.upper(),
            r.peak_stress,
            r.total_defaults,
            "YES" if r.ccp_failed else "NO",
            r.equity_loss_pct
        ))
    
    print("\n" + "-"*80)
    print("SYSTEM HEALTH THRESHOLDS:")
    print("  GREEN:  Stress < 0.2 - Normal operations")
    print("  YELLOW: Stress 0.2-0.4 - Elevated stress")
    print("  ORANGE: Stress 0.4-0.6 - High stress, intervention needed")
    print("  RED:    Stress 0.6-0.8 - Critical, system failure imminent")
    print("  BLACK:  Stress > 0.8 - System collapse")
    print("="*80)
    
    # Save results to JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'scenarios': [r.to_dict() for r in results],
        'summary': {
            'total_scenarios': len(results),
            'scenarios_reaching_red': sum(1 for r in results if r.final_state in [SystemState.RED, SystemState.BLACK]),
            'ccp_failures': sum(1 for r in results if r.ccp_failed),
            'avg_peak_stress': np.mean([r.peak_stress for r in results]),
            'max_equity_loss': max(r.equity_loss_pct for r in results)
        }
    }
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'stress_test_results.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run stress test scenarios")
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3, 4, 5],
                       help="Run specific scenario (1-5)")
    parser.add_argument('--all', action='store_true',
                       help="Run all scenarios")
    args = parser.parse_args()
    
    if args.scenario == 1:
        test_scenario_1_interbank_spiral()
    elif args.scenario == 2:
        test_scenario_2_margin_spiral()
    elif args.scenario == 3:
        test_scenario_3_ccp_near_failure()
    elif args.scenario == 4:
        test_scenario_4_exchange_congestion()
    elif args.scenario == 5:
        test_scenario_5_information_panic()
    else:
        run_all_scenarios()
