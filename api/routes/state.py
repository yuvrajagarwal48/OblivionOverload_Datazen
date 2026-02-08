"""
Simulation State Manager.
Global state management for the FinSim-MAPPO API.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environment import FinancialEnvironment, EnvConfig
from src.environment import Exchange, ExchangeNetwork, ExchangeConfig
from src.environment import CentralCounterparty, CCPNetwork, CCPConfig
from src.environment import InfrastructureRouter, TransactionType
from src.agents import BaseAgent
from src.agents.baseline_agents import create_baseline_agent, AgentConfig
from src.scenarios import ScenarioEngine
from src.analytics import RiskAnalyzer
from src.analytics.credit_risk import CreditRiskLayer, RiskFeatures
from src.decision_support import DecisionSupport
from src.decision_support.counterfactual import (
    CounterfactualEngine, HypotheticalTransaction, 
    TransactionType as CFTransactionType
)
from src.core.state_capture import StateCapture
from src.core.history import SimulationHistory


@dataclass
class StepHistory:
    """History entry for a single timestep."""
    timestep: int
    timestamp: str
    
    # System metrics
    market_price: float
    interest_rate: float
    volatility: float
    liquidity_index: float
    
    # Network metrics
    total_defaults: int
    default_rate: float
    num_stressed: int
    total_exposure: float
    avg_capital_ratio: float
    network_density: float
    
    # Clearing results
    clearing_converged: bool = True
    total_shortfall: float = 0.0
    defaults_this_step: List[int] = field(default_factory=list)
    recovery_rates: Dict[int, float] = field(default_factory=dict)
    
    # Actions and rewards
    actions: Dict[int, List[float]] = field(default_factory=dict)
    rewards: Dict[int, float] = field(default_factory=dict)
    
    # Bank snapshots (condensed)
    bank_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class TransactionLog:
    """Log of transactions for a bank."""
    bank_id: int
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, tx: Dict[str, Any]):
        self.transactions.append(tx)
    
    def get_all(self) -> List[Dict[str, Any]]:
        return self.transactions
    
    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return self.transactions[-n:]


class SimulationState:
    """
    Manages complete simulation state including:
    - Environment and agents
    - Infrastructure (exchanges, CCPs)
    - History and state capture
    - Analytics engines
    """
    
    def __init__(self):
        # Core simulation
        self.env: Optional[FinancialEnvironment] = None
        self.agents: Dict[int, BaseAgent] = {}
        self.scenario_engine: Optional[ScenarioEngine] = None
        
        # Infrastructure
        self.exchange_network: Optional[ExchangeNetwork] = None
        self.ccp_network: Optional[CCPNetwork] = None
        self.router: Optional[InfrastructureRouter] = None
        self.exchanges: List[Any] = []
        self.ccps: List[Any] = []
        
        # Analytics
        self.risk_analyzer: Optional[RiskAnalyzer] = None
        self.credit_risk_layer: Optional[CreditRiskLayer] = None
        self.decision_support: Optional[DecisionSupport] = None
        self.counterfactual_engine: Optional[CounterfactualEngine] = None
        
        # State capture
        self.state_capture: Optional[StateCapture] = None
        self.history: Optional[SimulationHistory] = None
        
        # Current state
        self.current_observations: Dict[int, np.ndarray] = {}
        self.global_state: Optional[np.ndarray] = None
        
        # History
        self.step_history: List[StepHistory] = []
        self.transaction_logs: Dict[int, TransactionLog] = {}
        
        # Config
        self.config: Optional[Dict[str, Any]] = None
        self.started_at: Optional[str] = None
        self.is_running: bool = False
    
    def initialize(self, 
                   num_banks: int = 30,
                   episode_length: int = 100,
                   scenario: str = "normal",
                   num_exchanges: int = 2,
                   num_ccps: int = 1,
                   seed: Optional[int] = None,
                   real_bank_configs: Optional[List[Dict]] = None):
        """Initialize complete simulation."""
        
        # If real bank configs provided, override num_banks
        if real_bank_configs:
            num_banks = len(real_bank_configs)
        
        self.config = {
            "num_banks": num_banks,
            "episode_length": episode_length,
            "scenario": scenario,
            "num_exchanges": num_exchanges,
            "num_ccps": num_ccps,
            "seed": seed,
            "use_real_banks": real_bank_configs is not None
        }
        self.started_at = datetime.now().isoformat()
        
        # Create environment
        env_config = EnvConfig(
            num_banks=num_banks,
            episode_length=episode_length,
            seed=seed,
            real_bank_configs=real_bank_configs
        )
        self.env = FinancialEnvironment(env_config)
        
        # Create exchanges
        self.exchange_network = ExchangeNetwork(num_exchanges=num_exchanges, seed=seed)
        self.exchanges = list(self.exchange_network.exchanges.values())
        
        # Create CCPs
        self.ccp_network = CCPNetwork(num_ccps=num_ccps, seed=seed)
        self.ccps = list(self.ccp_network.ccps.values())
        
        # Register banks as CCP members
        for ccp in self.ccps:
            for bank_id in range(num_banks):
                ccp.add_member(bank_id)
        
        # Create router
        self.router = InfrastructureRouter(
            exchange_network=self.exchange_network,
            ccp_network=self.ccp_network,
            seed=seed
        )
        for bank_id in range(num_banks):
            self.router.register_bank(bank_id)
        
        # Create agents
        self.agents = {}
        agent_types = ['rule_based', 'myopic', 'conservative', 'greedy']
        for i in range(num_banks):
            agent_config = AgentConfig(
                agent_id=i,
                observation_dim=FinancialEnvironment.OBS_DIM,
                action_dim=FinancialEnvironment.ACTION_DIM,
                seed=seed
            )
            self.agents[i] = create_baseline_agent(
                agent_types[i % len(agent_types)], 
                agent_config
            )
        
        # Initialize analytics
        self.risk_analyzer = RiskAnalyzer()
        self.credit_risk_layer = CreditRiskLayer(use_neural=False)
        
        # Initialize scenario engine
        self.scenario_engine = ScenarioEngine(seed=seed)
        self.scenario_engine.set_scenario(scenario)
        
        # Initialize state capture
        self.state_capture = StateCapture()
        self.history = SimulationHistory(config=self.config)
        
        # Initialize transaction logs
        self.transaction_logs = {i: TransactionLog(bank_id=i) for i in range(num_banks)}
        
        # Reset environment
        self.current_observations, self.global_state = self.env.reset()
        
        # Initialize decision support
        self.decision_support = DecisionSupport(
            self.env, self.agents, horizon=10, num_simulations=10
        )
        
        # Clear history
        self.step_history = []
        
        # Capture initial state
        self._capture_state(is_initial=True)
        
        self.is_running = True
    
    def _capture_state(self, is_initial: bool = False):
        """Capture current state to history."""
        if not self.env:
            return
        
        network = self.env.network
        market_state = self.env.market.get_state()
        network_stats = network.get_network_stats()
        
        # Build bank states
        bank_states = {}
        for bank_id, bank in network.banks.items():
            bank_states[bank_id] = {
                "cash": bank.balance_sheet.cash,
                "equity": bank.balance_sheet.equity,
                "capital_ratio": bank.capital_ratio,
                "status": bank.status.value,
                "total_assets": bank.balance_sheet.total_assets,
                "total_liabilities": bank.balance_sheet.total_liabilities,
            }
            if hasattr(bank, 'name') and bank.name:
                bank_states[bank_id]["name"] = bank.name
            if hasattr(bank, 'metadata') and bank.metadata:
                bank_states[bank_id]["metadata"] = bank.metadata
        
        # Create history entry
        entry = StepHistory(
            timestep=self.env.current_step,
            timestamp=datetime.now().isoformat(),
            market_price=market_state.asset_price,
            interest_rate=market_state.interest_rate,
            volatility=market_state.volatility,
            liquidity_index=market_state.liquidity_index,
            total_defaults=network_stats.num_defaulted,
            default_rate=network_stats.num_defaulted / max(network_stats.num_banks, 1),
            num_stressed=network_stats.num_stressed,
            total_exposure=network_stats.total_exposure,
            avg_capital_ratio=network_stats.avg_capital_ratio,
            network_density=network_stats.density,
            bank_states=bank_states
        )
        
        self.step_history.append(entry)
        
        # Full state capture
        if self.state_capture:
            self.state_capture.capture_all(
                banks=network.banks,
                exchanges=self.exchanges,
                ccps=self.ccps,
                market=self.env.market,
                network=network,
                timestep=self.env.current_step
            )
    
    def step(self, 
             actions: Optional[Dict[int, np.ndarray]] = None,
             capture_state: bool = True) -> Dict[str, Any]:
        """Execute one simulation step."""
        if not self.is_initialized():
            raise RuntimeError("Simulation not initialized")
        
        # Get actions from agents if not provided
        if actions is None:
            actions = {}
            for agent_id, agent in self.agents.items():
                obs = self.current_observations[agent_id]
                actions[agent_id] = agent.select_action(obs, deterministic=True)
        
        # Apply scenario shocks
        shocks = self.scenario_engine.generate_shocks(
            self.env.current_step,
            self.env.config.episode_length,
            self.env.num_agents
        )
        
        if shocks['market']:
            self.env.apply_scenario_shock(**shocks['market'], bank_shocks=shocks['banks'])
        
        # Execute step
        result = self.env.step(actions)
        
        # Update current state
        self.current_observations = result.observations
        self.global_state = result.global_state
        
        # Update agent stats
        for agent_id, reward in result.rewards.items():
            self.agents[agent_id].update_stats(reward)
        
        # Process infrastructure
        infrastructure_stats = self._process_infrastructure(actions)
        
        # Capture state
        if capture_state:
            self._capture_state()
            
            # Update last history entry with actions and rewards
            if self.step_history:
                self.step_history[-1].actions = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in actions.items()
                }
                self.step_history[-1].rewards = {
                    k: float(v) for k, v in result.rewards.items()
                }
        
        # Record transactions
        self._record_transactions(actions, result)
        
        return {
            "step": self.env.current_step,
            "rewards": {int(k): float(v) for k, v in result.rewards.items()},
            "done": any(result.dones.values()),
            "network_stats": result.network_stats.to_dict(),
            "market_state": result.market_state.to_dict(),
            "infrastructure": infrastructure_stats
        }
    
    def _process_infrastructure(self, actions: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Process actions through infrastructure layer."""
        stats = {
            "transactions_processed": 0,
            "exchange_fees": 0.0,
            "margin_collected": 0.0,
            "avg_delay": 0.0
        }
        
        if not self.router:
            return stats
        
        delays = []
        
        for bank_id, action in actions.items():
            if len(action) >= 4:
                lend_ratio = action[0]
                
                if lend_ratio > 0.1:  # Meaningful lending
                    bank = self.env.network.banks[bank_id]
                    amount = bank.balance_sheet.cash * lend_ratio * 0.1
                    
                    if amount > 100:
                        # Route through infrastructure
                        market_price = self.env.market.price if hasattr(self.env.market, 'price') else 100.0
                        routing = self.router.route_transaction(
                            source_bank=bank_id,
                            transaction_type=TransactionType.INTERBANK_LEND,
                            amount=amount,
                            price=market_price
                        )
                        
                        if routing:
                            stats["transactions_processed"] += 1
                            stats["exchange_fees"] += routing.get("fee", 0)
                            delays.append(routing.get("delay", 0))
                            
                            # Collect margin at CCP
                            for ccp in self.ccps:
                                if bank_id in ccp.members:
                                    margin = ccp.calculate_margin(bank_id, amount)
                                    stats["margin_collected"] += margin
        
        if delays:
            stats["avg_delay"] = sum(delays) / len(delays)
        
        return stats
    
    def _record_transactions(self, actions: Dict[int, np.ndarray], result: Any):
        """Record transactions to bank logs."""
        for bank_id, action in actions.items():
            if len(action) >= 4:
                # Record lending activity
                if action[0] > 0.1:
                    tx = {
                        "timestep": self.env.current_step,
                        "type": "lending",
                        "direction": "outflow",
                        "amount": float(action[0] * 1000),  # Approximation
                        "counterparty": -1,  # Unknown
                        "status": "completed"
                    }
                    self.transaction_logs[bank_id].add(tx)
                
                # Record borrowing activity
                if action[3] > 0.1:
                    tx = {
                        "timestep": self.env.current_step,
                        "type": "borrowing",
                        "direction": "inflow",
                        "amount": float(action[3] * 1000),
                        "counterparty": -1,
                        "status": "completed"
                    }
                    self.transaction_logs[bank_id].add(tx)
    
    def run_steps(self, num_steps: int) -> Dict[str, Any]:
        """Run multiple simulation steps."""
        total_rewards = {i: 0.0 for i in range(self.env.num_agents)}
        steps_completed = 0
        
        for _ in range(num_steps):
            result = self.step()
            
            for k, v in result["rewards"].items():
                total_rewards[k] += v
            
            steps_completed += 1
            
            if result["done"]:
                break
        
        return {
            "steps_completed": steps_completed,
            "current_step": self.env.current_step,
            "is_done": result.get("done", False),
            "total_rewards": total_rewards,
            "final_network_stats": result.get("network_stats", {}),
            "final_market_state": result.get("market_state", {})
        }
    
    def is_initialized(self) -> bool:
        return self.env is not None
    
    def get_current_step(self) -> int:
        return self.env.current_step if self.env else 0
    
    def get_total_steps(self) -> int:
        return self.env.config.episode_length if self.env else 0
    
    def is_done(self) -> bool:
        if not self.env:
            return False
        return self.env.current_step >= self.env.config.episode_length
    
    def reset(self):
        """Reset simulation to initial state."""
        if not self.env:
            return
        
        self.current_observations, self.global_state = self.env.reset()
        
        for agent in self.agents.values():
            agent.reset()
        
        self.step_history = []
        self.transaction_logs = {i: TransactionLog(bank_id=i) for i in range(self.env.num_agents)}
        
        if self.state_capture:
            self.state_capture.clear()
        
        self._capture_state(is_initial=True)
    
    def get_history(self, 
                    start_step: int = 0, 
                    end_step: Optional[int] = None,
                    fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get simulation history."""
        if end_step is None:
            end_step = len(self.step_history)
        
        history = self.step_history[start_step:end_step]
        
        if fields:
            return [
                {f: getattr(h, f, None) for f in fields}
                for h in history
            ]
        
        return [
            {
                "timestep": h.timestep,
                "market_price": h.market_price,
                "interest_rate": h.interest_rate,
                "volatility": h.volatility,
                "liquidity_index": h.liquidity_index,
                "total_defaults": h.total_defaults,
                "default_rate": h.default_rate,
                "num_stressed": h.num_stressed,
                "total_exposure": h.total_exposure,
                "avg_capital_ratio": h.avg_capital_ratio,
                "network_density": h.network_density,
                "rewards": h.rewards
            }
            for h in history
        ]
    
    def get_bank_history(self, bank_id: int) -> List[Dict[str, Any]]:
        """Get history for a specific bank."""
        history = []
        
        for h in self.step_history:
            if bank_id in h.bank_states:
                entry = {
                    "timestep": h.timestep,
                    **h.bank_states[bank_id]
                }
                history.append(entry)
        
        return history
    
    def get_bank_transactions(self, bank_id: int) -> List[Dict[str, Any]]:
        """Get transaction history for a bank."""
        if bank_id in self.transaction_logs:
            return self.transaction_logs[bank_id].get_all()
        return []
    
    def calculate_credit_risk(self, bank_id: int) -> Dict[str, Any]:
        """Calculate credit risk metrics for a bank."""
        if not self.env or bank_id not in self.env.network.banks:
            return {}
        
        bank = self.env.network.banks[bank_id]
        
        # Extract features
        features = RiskFeatures.from_bank(
            bank, 
            self.env.network, 
            self.env.market,
            self.ccps[0] if self.ccps else None
        )
        
        # Predict
        if self.credit_risk_layer:
            output = self.credit_risk_layer.predict(
                bank,
                self.env.network,
                self.env.market,
                self.ccps[0] if self.ccps else None
            )
            return output.to_dict()
        
        # Fallback heuristic
        return {
            "probability_of_default": self._heuristic_pd(bank),
            "loss_given_default": 0.45,
            "exposure_at_default": bank.balance_sheet.total_liabilities,
            "rating": "BBB"
        }
    
    def _heuristic_pd(self, bank) -> float:
        """Simple heuristic PD calculation."""
        if bank.capital_ratio < 0.04:
            base_pd = 0.50
        elif bank.capital_ratio < 0.08:
            base_pd = 0.20
        elif bank.capital_ratio < 0.12:
            base_pd = 0.05
        else:
            base_pd = 0.01
        
        return min(base_pd, 0.95)


# Global state instance
simulation_state = SimulationState()
