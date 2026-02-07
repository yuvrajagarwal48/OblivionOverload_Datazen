"""
Market: Asset pricing, liquidity dynamics, and fire sale mechanics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MarketCondition(Enum):
    """Current market condition."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESSED = "stressed"
    CRISIS = "crisis"


@dataclass
class MarketState:
    """Current state of the market."""
    asset_price: float
    interest_rate: float
    volatility: float
    liquidity_index: float
    condition: MarketCondition
    total_sell_volume: float
    total_buy_volume: float
    
    def to_dict(self) -> dict:
        return {
            'asset_price': self.asset_price,
            'interest_rate': self.interest_rate,
            'volatility': self.volatility,
            'liquidity_index': self.liquidity_index,
            'condition': self.condition.value,
            'total_sell_volume': self.total_sell_volume,
            'total_buy_volume': self.total_buy_volume
        }


@dataclass
class Transaction:
    """Record of a market transaction."""
    timestamp: int
    bank_id: int
    action: str  # 'buy' or 'sell'
    volume: float
    price: float
    value: float


class Market:
    """
    Financial market simulation with price impact and fire sale dynamics.
    
    Implements exponential price impact model:
    P_{t+1} = P_t * exp(-kappa * V_sell)
    """
    
    def __init__(self,
                 initial_price: float = 1.0,
                 price_impact_kappa: float = 0.001,
                 min_price_floor: float = 0.1,
                 max_sell_limit: float = 0.3,
                 base_interest_rate: float = 0.05,
                 base_volatility: float = 0.02,
                 seed: Optional[int] = None):
        """
        Initialize the market.
        
        Args:
            initial_price: Starting asset price
            price_impact_kappa: Price impact parameter (kappa)
            min_price_floor: Minimum price to prevent collapse
            max_sell_limit: Maximum fraction of assets sellable per step
            base_interest_rate: Base interbank interest rate
            base_volatility: Base market volatility
            seed: Random seed
        """
        self.initial_price = initial_price
        self.price_impact_kappa = price_impact_kappa
        self.min_price_floor = min_price_floor
        self.max_sell_limit = max_sell_limit
        self.base_interest_rate = base_interest_rate
        self.base_volatility = base_volatility
        
        self._rng = np.random.default_rng(seed)
        
        # Current state
        self.current_price = initial_price
        self.current_interest_rate = base_interest_rate
        self.current_volatility = base_volatility
        self.liquidity_index = 1.0  # 1.0 = normal liquidity
        
        # History tracking
        self.price_history: List[float] = [initial_price]
        self.volatility_history: List[float] = [base_volatility]
        self.transactions: List[Transaction] = []
        
        # Per-step accumulators
        self._step_sell_volume = 0.0
        self._step_buy_volume = 0.0
        self._current_step = 0
    
    @property
    def condition(self) -> MarketCondition:
        """Determine current market condition based on volatility and price."""
        price_drop = 1.0 - (self.current_price / self.initial_price)
        
        if price_drop > 0.3 or self.current_volatility > 0.15:
            return MarketCondition.CRISIS
        elif price_drop > 0.15 or self.current_volatility > 0.08:
            return MarketCondition.STRESSED
        elif price_drop > 0.05 or self.current_volatility > 0.04:
            return MarketCondition.VOLATILE
        return MarketCondition.NORMAL
    
    def get_state(self) -> MarketState:
        """Get current market state."""
        return MarketState(
            asset_price=self.current_price,
            interest_rate=self.current_interest_rate,
            volatility=self.current_volatility,
            liquidity_index=self.liquidity_index,
            condition=self.condition,
            total_sell_volume=self._step_sell_volume,
            total_buy_volume=self._step_buy_volume
        )
    
    def execute_sell(self, bank_id: int, volume: float) -> Tuple[float, float]:
        """
        Execute a sell order with price impact.
        
        Args:
            bank_id: ID of selling bank
            volume: Volume of assets to sell
            
        Returns:
            Tuple of (actual_volume_sold, cash_received)
        """
        # Apply sell limit
        actual_volume = min(volume, volume * self.max_sell_limit)
        
        if actual_volume <= 0:
            return 0.0, 0.0
        
        # Calculate price impact BEFORE the sale
        price_before = self.current_price
        
        # Execute at current price
        cash_received = actual_volume * price_before
        
        # Accumulate sell volume for end-of-step price update
        self._step_sell_volume += actual_volume
        
        # Record transaction
        self.transactions.append(Transaction(
            timestamp=self._current_step,
            bank_id=bank_id,
            action='sell',
            volume=actual_volume,
            price=price_before,
            value=cash_received
        ))
        
        return actual_volume, cash_received
    
    def execute_buy(self, bank_id: int, cash_amount: float) -> Tuple[float, float]:
        """
        Execute a buy order.
        
        Args:
            bank_id: ID of buying bank
            cash_amount: Cash to spend on assets
            
        Returns:
            Tuple of (assets_bought, cash_spent)
        """
        if cash_amount <= 0 or self.current_price <= 0:
            return 0.0, 0.0
        
        assets_bought = cash_amount / self.current_price
        
        # Accumulate buy volume (reduces net selling pressure)
        self._step_buy_volume += assets_bought
        
        # Record transaction
        self.transactions.append(Transaction(
            timestamp=self._current_step,
            bank_id=bank_id,
            action='buy',
            volume=assets_bought,
            price=self.current_price,
            value=cash_amount
        ))
        
        return assets_bought, cash_amount
    
    def step(self) -> float:
        """
        Advance market by one timestep and update prices.
        
        Returns:
            Price change as a fraction
        """
        old_price = self.current_price
        
        # Net selling pressure
        net_sell_volume = self._step_sell_volume - self._step_buy_volume
        
        # Price impact from net selling
        if net_sell_volume > 0:
            price_impact = np.exp(-self.price_impact_kappa * net_sell_volume)
            self.current_price *= price_impact
        elif net_sell_volume < 0:
            # Buying pressure increases price (with smaller effect)
            price_boost = np.exp(self.price_impact_kappa * abs(net_sell_volume) * 0.5)
            self.current_price *= price_boost
        
        # Add random noise
        noise = self._rng.normal(0, self.current_volatility)
        self.current_price *= (1 + noise)
        
        # Apply price floor
        self.current_price = max(self.min_price_floor, self.current_price)
        
        # Update volatility based on price change
        price_change = abs(self.current_price - old_price) / old_price
        self.current_volatility = 0.9 * self.current_volatility + 0.1 * price_change
        self.current_volatility = np.clip(self.current_volatility, 0.001, 0.5)
        
        # Update liquidity index based on trading volume
        total_volume = self._step_sell_volume + self._step_buy_volume
        if total_volume > 0:
            self.liquidity_index = min(1.0, self.liquidity_index + 0.1)
        else:
            self.liquidity_index = max(0.1, self.liquidity_index - 0.05)
        
        # Update interest rate based on market condition
        self._update_interest_rate()
        
        # Record history
        self.price_history.append(self.current_price)
        self.volatility_history.append(self.current_volatility)
        
        # Reset step accumulators
        self._step_sell_volume = 0.0
        self._step_buy_volume = 0.0
        self._current_step += 1
        
        return (self.current_price - old_price) / old_price
    
    def _update_interest_rate(self) -> None:
        """Update interest rate based on market conditions."""
        condition = self.condition
        
        if condition == MarketCondition.CRISIS:
            # Rates spike during crisis
            target_rate = self.base_interest_rate * 3.0
        elif condition == MarketCondition.STRESSED:
            target_rate = self.base_interest_rate * 2.0
        elif condition == MarketCondition.VOLATILE:
            target_rate = self.base_interest_rate * 1.5
        else:
            target_rate = self.base_interest_rate
        
        # Gradual adjustment
        self.current_interest_rate = 0.9 * self.current_interest_rate + 0.1 * target_rate
    
    def apply_shock(self, 
                    price_shock: float = 0.0,
                    volatility_shock: float = 0.0,
                    liquidity_shock: float = 0.0) -> None:
        """
        Apply an external shock to the market.
        
        Args:
            price_shock: Fractional price change (negative = drop)
            volatility_shock: Additional volatility
            liquidity_shock: Reduction in liquidity (0 to 1)
        """
        # Price shock
        self.current_price *= (1 + price_shock)
        self.current_price = max(self.min_price_floor, self.current_price)
        
        # Volatility shock
        self.current_volatility += volatility_shock
        self.current_volatility = np.clip(self.current_volatility, 0.001, 0.5)
        
        # Liquidity shock
        self.liquidity_index *= (1 - liquidity_shock)
        self.liquidity_index = max(0.1, self.liquidity_index)
    
    def get_price_return(self, lookback: int = 1) -> float:
        """Get price return over lookback period."""
        if len(self.price_history) < lookback + 1:
            return 0.0
        return (self.price_history[-1] - self.price_history[-1 - lookback]) / self.price_history[-1 - lookback]
    
    def get_realized_volatility(self, lookback: int = 10) -> float:
        """Calculate realized volatility over lookback period."""
        if len(self.price_history) < lookback + 1:
            return self.current_volatility
        
        prices = self.price_history[-lookback - 1:]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def reset(self) -> None:
        """Reset market to initial state."""
        self.current_price = self.initial_price
        self.current_interest_rate = self.base_interest_rate
        self.current_volatility = self.base_volatility
        self.liquidity_index = 1.0
        
        self.price_history = [self.initial_price]
        self.volatility_history = [self.base_volatility]
        self.transactions = []
        
        self._step_sell_volume = 0.0
        self._step_buy_volume = 0.0
        self._current_step = 0
    
    def get_observation(self) -> np.ndarray:
        """Get market observation vector."""
        return np.array([
            self.current_price / self.initial_price,  # Normalized price
            self.current_interest_rate,
            self.current_volatility,
            self.liquidity_index,
            self.get_price_return(1),
            self.get_price_return(5) if len(self.price_history) > 5 else 0.0,
        ], dtype=np.float32)
