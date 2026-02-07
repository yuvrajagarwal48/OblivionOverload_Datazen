"""
Financial Environment: Main simulation environment integrating all components.
Provides Gym-like interface for multi-agent reinforcement learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy

from .network import FinancialNetwork, NetworkStats
from .bank import Bank, BankStatus
from .clearing import ClearingMechanism, ClearingResult
from .market import Market, MarketState


@dataclass
class EnvConfig:
    """Environment configuration."""
    # Network
    num_banks: int = 30
    edges_per_node: int = 2
    core_fraction: float = 0.2
    
    # Bank initialization
    initial_cash_range: Tuple[float, float] = (100, 500)
    initial_assets_range: Tuple[float, float] = (200, 1000)
    initial_ext_liab_range: Tuple[float, float] = (50, 300)
    min_capital_ratio: float = 0.08
    
    # Market
    initial_asset_price: float = 1.0
    price_impact_kappa: float = 0.001
    min_price_floor: float = 0.1
    max_sell_limit: float = 0.3
    base_interest_rate: float = 0.05
    base_volatility: float = 0.02
    
    # Clearing
    clearing_max_iterations: int = 100
    clearing_threshold: float = 1e-6
    
    # Simulation
    episode_length: int = 100
    
    # Reward weights
    profit_weight: float = 1.0
    liquidity_weight: float = 0.1
    default_penalty: float = 100.0
    system_risk_weight: float = 50.0
    
    # Seed
    seed: Optional[int] = None


@dataclass
class StepResult:
    """Result of a single environment step."""
    observations: Dict[int, np.ndarray]
    rewards: Dict[int, float]
    dones: Dict[int, bool]
    infos: Dict[int, Dict[str, Any]]
    global_state: np.ndarray
    clearing_result: Optional[ClearingResult]
    market_state: MarketState
    network_stats: NetworkStats


class FinancialEnvironment:
    """
    Multi-agent financial simulation environment.
    
    Each bank is an agent that can:
    - Lend money to other banks
    - Hoard cash
    - Sell assets
    - Request borrowing
    
    The environment simulates market dynamics, clearing, and contagion.
    """
    
    # Observation space dimension per agent
    OBS_DIM = 16
    # Action space dimension per agent
    ACTION_DIM = 4
    
    def __init__(self, config: Optional[EnvConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or EnvConfig()
        
        # Initialize components
        self.network = FinancialNetwork(
            num_banks=self.config.num_banks,
            edges_per_node=self.config.edges_per_node,
            core_fraction=self.config.core_fraction,
            seed=self.config.seed
        )
        
        self.market = Market(
            initial_price=self.config.initial_asset_price,
            price_impact_kappa=self.config.price_impact_kappa,
            min_price_floor=self.config.min_price_floor,
            max_sell_limit=self.config.max_sell_limit,
            base_interest_rate=self.config.base_interest_rate,
            base_volatility=self.config.base_volatility,
            seed=self.config.seed
        )
        
        self.clearing = ClearingMechanism(
            max_iterations=self.config.clearing_max_iterations,
            convergence_threshold=self.config.clearing_threshold
        )
        
        self._rng = np.random.default_rng(self.config.seed)
        
        # State tracking
        self.current_step = 0
        self.episode_rewards: Dict[int, float] = {}
        self._last_clearing_result: Optional[ClearingResult] = None
        
        # For observation normalization
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_std: Optional[np.ndarray] = None
    
    @property
    def num_agents(self) -> int:
        return self.config.num_banks
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for reproducibility
            
        Returns:
            Tuple of (observations dict, global state)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.config.seed = seed
        
        # Regenerate network
        self.network.generate_network(
            initial_cash_range=self.config.initial_cash_range,
            initial_assets_range=self.config.initial_assets_range,
            initial_ext_liab_range=self.config.initial_ext_liab_range,
            min_capital_ratio=self.config.min_capital_ratio
        )
        
        # Reset market
        self.market.reset()
        
        # Reset tracking
        self.current_step = 0
        self.episode_rewards = {i: 0.0 for i in range(self.num_agents)}
        self._last_clearing_result = None
        
        # Get initial observations
        observations = self._get_observations()
        global_state = self._get_global_state()
        
        return observations, global_state
    
    def step(self, actions: Dict[int, np.ndarray]) -> StepResult:
        """
        Execute one step of the simulation.
        
        Args:
            actions: Dict mapping agent_id -> action vector
                    Action vector: [lend_ratio, hoard_ratio, sell_ratio, borrow_request]
                    
        Returns:
            StepResult containing observations, rewards, dones, infos
        """
        self.current_step += 1
        
        # Store pre-step state for reward calculation
        pre_step_equities = {
            i: bank.balance_sheet.equity 
            for i, bank in self.network.banks.items()
        }
        pre_step_defaults = sum(
            1 for b in self.network.banks.values() 
            if b.status == BankStatus.DEFAULTED
        )
        
        # Execute actions
        self._execute_actions(actions)
        
        # Update market prices
        price_change = self.market.step()
        
        # Apply price change to all bank assets
        for bank in self.network.banks.values():
            bank.apply_asset_price_shock(price_change)
        
        # Run clearing mechanism
        clearing_result = self._run_clearing()
        self._last_clearing_result = clearing_result
        
        # Apply clearing results (defaults)
        self._apply_clearing_results(clearing_result)
        
        # Trigger margin calls for stressed banks
        self._trigger_margin_calls()
        
        # Update bank statuses
        for bank in self.network.banks.values():
            bank.check_and_update_status()
        
        # Calculate rewards
        rewards = self._calculate_rewards(
            pre_step_equities, 
            pre_step_defaults,
            clearing_result
        )
        
        # Update episode rewards
        for i, r in rewards.items():
            self.episode_rewards[i] += r
        
        # Check if done
        done = self.current_step >= self.config.episode_length
        
        # Check for early termination (too many defaults)
        num_defaults = sum(
            1 for b in self.network.banks.values() 
            if b.status == BankStatus.DEFAULTED
        )
        if num_defaults > self.num_agents * 0.5:
            done = True
        
        dones = {i: done for i in range(self.num_agents)}
        
        # Get observations
        observations = self._get_observations()
        global_state = self._get_global_state()
        
        # Build info
        infos = self._build_infos(clearing_result)
        
        return StepResult(
            observations=observations,
            rewards=rewards,
            dones=dones,
            infos=infos,
            global_state=global_state,
            clearing_result=clearing_result,
            market_state=self.market.get_state(),
            network_stats=self.network.get_network_stats()
        )
    
    def _execute_actions(self, actions: Dict[int, np.ndarray]) -> None:
        """Execute agent actions."""
        # Collect lending and borrowing requests
        lending_offers: Dict[int, float] = {}
        borrowing_requests: Dict[int, float] = {}
        
        for agent_id, action in actions.items():
            bank = self.network.banks[agent_id]
            
            if bank.status == BankStatus.DEFAULTED:
                continue
            
            # Parse action (apply softmax for ratios)
            action = np.clip(action, -1, 1)
            ratios = self._softmax(action[:3])
            lend_ratio, hoard_ratio, sell_ratio = ratios
            borrow_request = (action[3] + 1) / 2 * bank.balance_sheet.total_assets * 0.1
            
            # Execute sell orders
            if sell_ratio > 0.1:
                sell_amount = bank.balance_sheet.illiquid_assets * sell_ratio * 0.5
                if sell_amount > 0:
                    actual_sold, cash = self.market.execute_sell(agent_id, sell_amount)
                    if actual_sold > 0:
                        bank.balance_sheet.illiquid_assets -= actual_sold
                        bank.balance_sheet.cash += cash
            
            # Record lending offers
            if lend_ratio > 0.1:
                lending_offers[agent_id] = bank.excess_cash * lend_ratio
            
            # Record borrowing requests
            if borrow_request > 0:
                borrowing_requests[agent_id] = borrow_request
        
        # Match lenders and borrowers
        self._match_lending(lending_offers, borrowing_requests)
    
    def _match_lending(self, 
                       lending_offers: Dict[int, float],
                       borrowing_requests: Dict[int, float]) -> None:
        """Match lenders with borrowers based on network connections."""
        for borrower_id, request_amount in borrowing_requests.items():
            if request_amount <= 0:
                continue
            
            # Find potential lenders (neighbors or any bank)
            neighbors = self.network.get_neighbors(borrower_id)
            potential_lenders = [
                lid for lid in neighbors 
                if lid in lending_offers and lending_offers[lid] > 0
            ]
            
            if not potential_lenders:
                # Try any bank with available funds
                potential_lenders = [
                    lid for lid in lending_offers 
                    if lending_offers[lid] > 0 and lid != borrower_id
                ]
            
            if not potential_lenders:
                continue
            
            # Distribute borrowing across lenders
            remaining_request = request_amount
            for lender_id in potential_lenders:
                if remaining_request <= 0:
                    break
                
                available = lending_offers[lender_id]
                loan_amount = min(remaining_request, available)
                
                if loan_amount > 10:  # Minimum loan threshold
                    success = self.network.execute_lending(
                        lender_id, borrower_id, loan_amount,
                        self.market.current_interest_rate
                    )
                    if success:
                        lending_offers[lender_id] -= loan_amount
                        remaining_request -= loan_amount
    
    def _run_clearing(self) -> ClearingResult:
        """Run the clearing mechanism."""
        # Build inputs for clearing
        external_assets = np.array([
            bank.balance_sheet.cash + bank.balance_sheet.illiquid_assets * self.market.current_price
            for bank in self.network.banks.values()
        ])
        
        external_liabilities = np.array([
            bank.balance_sheet.external_liabilities
            for bank in self.network.banks.values()
        ])
        
        # Run clearing
        result = self.clearing.clear(
            self.network.liability_matrix,
            external_assets,
            external_liabilities
        )
        
        return result
    
    def _apply_clearing_results(self, result: ClearingResult) -> None:
        """Apply clearing results to banks."""
        for bank_id in result.default_set:
            bank = self.network.banks[bank_id]
            if bank.status != BankStatus.DEFAULTED:
                bank.status = BankStatus.DEFAULTED
                
                # Write down interbank claims on this bank
                recovery_rate = result.recovery_rates[bank_id]
                for creditor_id, amount in bank.balance_sheet.interbank_liabilities.items():
                    creditor = self.network.banks.get(creditor_id)
                    if creditor:
                        loss = amount * (1 - recovery_rate)
                        creditor.reduce_interbank_asset(bank_id, loss)
    
    def _trigger_margin_calls(self) -> None:
        """Trigger margin calls for banks below capital ratio."""
        for bank in self.network.banks.values():
            if bank.status != BankStatus.DEFAULTED:
                sell_volume = bank.trigger_margin_call(
                    self.market.current_price,
                    self.config.max_sell_limit
                )
                if sell_volume > 0:
                    self.market._step_sell_volume += sell_volume
    
    def _calculate_rewards(self,
                           pre_step_equities: Dict[int, float],
                           pre_step_defaults: int,
                           clearing_result: ClearingResult) -> Dict[int, float]:
        """Calculate rewards for each agent."""
        rewards = {}
        
        # System-level metrics
        current_defaults = sum(
            1 for b in self.network.banks.values() 
            if b.status == BankStatus.DEFAULTED
        )
        new_defaults = current_defaults - pre_step_defaults
        default_rate = current_defaults / self.num_agents
        
        for agent_id, bank in self.network.banks.items():
            if bank.status == BankStatus.DEFAULTED:
                # Large penalty for default
                rewards[agent_id] = -self.config.default_penalty
                continue
            
            # Profit component (equity change)
            equity_change = bank.balance_sheet.equity - pre_step_equities[agent_id]
            profit_reward = equity_change * self.config.profit_weight / 100
            
            # Liquidity provision reward
            lending_volume = sum(bank.balance_sheet.interbank_assets.values())
            liquidity_reward = np.log1p(lending_volume) * self.config.liquidity_weight
            
            # System risk penalty
            system_risk_penalty = default_rate * self.config.system_risk_weight
            
            # Capital ratio bonus
            cr_bonus = 0.0
            if bank.capital_ratio > self.config.min_capital_ratio * 1.5:
                cr_bonus = 1.0
            elif bank.capital_ratio < self.config.min_capital_ratio:
                cr_bonus = -5.0
            
            # Total reward
            rewards[agent_id] = (
                profit_reward + 
                liquidity_reward - 
                system_risk_penalty + 
                cr_bonus
            )
            
            # Clip rewards for stability
            rewards[agent_id] = np.clip(rewards[agent_id], -50, 50)
        
        return rewards
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        market_obs = self.market.get_observation()
        
        # Get neighbor default rates
        neighbor_default_rates = {}
        for bank_id in range(self.num_agents):
            neighbors = self.network.get_neighbors(bank_id)
            if neighbors:
                num_defaulted = sum(
                    1 for nid in neighbors 
                    if self.network.banks[nid].status == BankStatus.DEFAULTED
                )
                neighbor_default_rates[bank_id] = num_defaulted / len(neighbors)
            else:
                neighbor_default_rates[bank_id] = 0.0
        
        for agent_id, bank in self.network.banks.items():
            bank_obs = bank.get_observation()
            
            # Combine into observation vector
            obs = np.array([
                bank_obs['cash'] / 1000,  # Normalized
                bank_obs['equity'] / 1000,
                bank_obs['capital_ratio'],
                bank_obs['total_owed'] / 1000,
                bank_obs['total_owing'] / 1000,
                bank_obs['illiquid_assets'] / 1000,
                bank_obs['is_stressed'],
                bank_obs['default_count'],
                neighbor_default_rates[agent_id],
                float(bank.tier == 1),  # Is core bank
                market_obs[0],  # Normalized price
                market_obs[1],  # Interest rate
                market_obs[2],  # Volatility
                market_obs[3],  # Liquidity index
                market_obs[4],  # Recent return
                market_obs[5],  # 5-step return
            ], dtype=np.float32)
            
            # Clip for stability
            obs = np.clip(obs, -10, 10)
            
            observations[agent_id] = obs
        
        return observations
    
    def _get_global_state(self) -> np.ndarray:
        """Get global state for centralized critic."""
        return self.network.get_global_state()
    
    def _build_infos(self, clearing_result: ClearingResult) -> Dict[int, Dict[str, Any]]:
        """Build info dict for each agent."""
        infos = {}
        for agent_id, bank in self.network.banks.items():
            infos[agent_id] = {
                'equity': bank.balance_sheet.equity,
                'capital_ratio': bank.capital_ratio,
                'status': bank.status.value,
                'tier': bank.tier,
                'in_default_set': agent_id in clearing_result.default_set
            }
        return infos
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax of array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def apply_scenario_shock(self,
                              price_shock: float = 0.0,
                              volatility_shock: float = 0.0,
                              liquidity_shock: float = 0.0,
                              bank_shocks: Optional[Dict[int, float]] = None) -> None:
        """
        Apply a scenario shock to the environment.
        
        Args:
            price_shock: Market price shock
            volatility_shock: Volatility increase
            liquidity_shock: Liquidity reduction
            bank_shocks: Individual bank shocks (bank_id -> cash loss)
        """
        # Market shock
        self.market.apply_shock(price_shock, volatility_shock, liquidity_shock)
        
        # Individual bank shocks
        if bank_shocks:
            for bank_id, loss in bank_shocks.items():
                if bank_id in self.network.banks:
                    bank = self.network.banks[bank_id]
                    bank.balance_sheet.cash = max(0, bank.balance_sheet.cash - loss)
    
    def clone(self) -> 'FinancialEnvironment':
        """Create a deep copy of the environment."""
        return copy.deepcopy(self)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for serialization."""
        return {
            'current_step': self.current_step,
            'network': self.network.to_dict(),
            'market': self.market.get_state().to_dict(),
            'episode_rewards': dict(self.episode_rewards)
        }
