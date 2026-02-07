"""
Exchange: Trade routing layer for financial infrastructure.
Exchanges mediate all trading activity between banks and clearing houses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class ExchangeStatus(Enum):
    """Exchange operational status."""
    NORMAL = "normal"
    CONGESTED = "congested"
    HALTED = "halted"


@dataclass
class Order:
    """Trade order submitted to exchange."""
    order_id: int
    bank_id: int
    order_type: str  # 'buy', 'sell', 'lend', 'borrow'
    asset_type: str  # 'equity', 'bond', 'interbank'
    quantity: float
    price: float
    timestamp: int
    priority: int = 0
    filled: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    settlement_delay: int = 0
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'bank_id': self.bank_id,
            'order_type': self.order_type,
            'asset_type': self.asset_type,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp,
            'filled': self.filled,
            'fill_price': self.fill_price,
            'settlement_delay': self.settlement_delay
        }


@dataclass
class ExchangeState:
    """Current state of the exchange."""
    exchange_id: int
    status: ExchangeStatus
    congestion_level: float  # 0 to 1
    current_volume: float
    capacity: float
    backlog_size: int
    avg_settlement_delay: float
    effective_fee_rate: float
    volatility_indicator: float
    connected_banks: List[int]
    connected_ccps: List[int]
    
    def to_dict(self) -> dict:
        return {
            'exchange_id': self.exchange_id,
            'status': self.status.value,
            'congestion_level': self.congestion_level,
            'current_volume': self.current_volume,
            'capacity': self.capacity,
            'backlog_size': self.backlog_size,
            'avg_settlement_delay': self.avg_settlement_delay,
            'effective_fee_rate': self.effective_fee_rate,
            'volatility_indicator': self.volatility_indicator
        }


@dataclass
class ExchangeConfig:
    """Configuration for an exchange."""
    exchange_id: int
    name: str = "Exchange"
    base_capacity: float = 10000.0  # Max orders per timestep
    base_fee_rate: float = 0.001  # 0.1% base fee
    congestion_fee_multiplier: float = 3.0  # Fee multiplier at max congestion
    max_settlement_delay: int = 5  # Max delay in timesteps
    halt_threshold: float = 0.95  # Congestion level to halt trading
    volatility_window: int = 20  # Window for volatility calculation


class Exchange:
    """
    Exchange node that mediates trading between banks and clearing houses.
    
    Responsibilities:
    - Receive trade and asset sale orders from banks
    - Enforce throughput limits
    - Apply transaction fees
    - Track congestion and backlog
    - Route cleared trades to clearing houses
    - Broadcast market signals back to banks
    """
    
    def __init__(self, config: ExchangeConfig, seed: Optional[int] = None):
        self.config = config
        self.exchange_id = config.exchange_id
        self._rng = np.random.default_rng(seed)
        
        # Order management
        self._order_counter = 0
        self.order_book: Dict[int, Order] = {}
        self.pending_orders: deque = deque()
        self.backlog: List[Order] = []
        
        # Connected entities
        self.connected_banks: List[int] = []
        self.connected_ccps: List[int] = []
        
        # State tracking
        self.current_step = 0
        self.current_volume = 0.0
        self.congestion_level = 0.0
        self.status = ExchangeStatus.NORMAL
        
        # Historical data for volatility
        self.price_history: deque = deque(maxlen=config.volatility_window)
        self.volume_history: deque = deque(maxlen=config.volatility_window)
        
        # Metrics
        self.total_fees_collected = 0.0
        self.total_orders_processed = 0
        self.total_orders_rejected = 0
        self.settlement_delays: List[int] = []
    
    def connect_bank(self, bank_id: int) -> None:
        """Register a bank with this exchange."""
        if bank_id not in self.connected_banks:
            self.connected_banks.append(bank_id)
    
    def connect_ccp(self, ccp_id: int) -> None:
        """Register a clearing house with this exchange."""
        if ccp_id not in self.connected_ccps:
            self.connected_ccps.append(ccp_id)
    
    def submit_order(self, bank_id: int, order_type: str, asset_type: str,
                     quantity: float, price: float, priority: int = 0) -> Optional[Order]:
        """
        Submit an order to the exchange.
        
        Returns the order if accepted, None if rejected.
        """
        # Check if exchange is halted
        if self.status == ExchangeStatus.HALTED:
            self.total_orders_rejected += 1
            return None
        
        # Validate order
        if quantity <= 0 or price <= 0:
            return None
        
        # Bound values for numerical stability
        quantity = np.clip(quantity, 0.01, 1e9)
        price = np.clip(price, 0.001, 1e6)
        
        # Create order
        self._order_counter += 1
        order = Order(
            order_id=self._order_counter,
            bank_id=bank_id,
            order_type=order_type,
            asset_type=asset_type,
            quantity=quantity,
            price=price,
            timestamp=self.current_step,
            priority=priority
        )
        
        self.order_book[order.order_id] = order
        self.pending_orders.append(order)
        
        return order
    
    def process_orders(self, market_price: float) -> Tuple[List[Order], float]:
        """
        Process all pending orders for this timestep.
        
        Returns:
            - List of filled orders
            - Total fees collected
        """
        filled_orders = []
        fees_collected = 0.0
        
        # Calculate current congestion
        total_volume = sum(o.quantity for o in self.pending_orders)
        self.current_volume = total_volume
        self.congestion_level = min(1.0, total_volume / self.config.base_capacity)
        
        # Update status based on congestion
        if self.congestion_level >= self.config.halt_threshold:
            self.status = ExchangeStatus.HALTED
        elif self.congestion_level >= 0.7:
            self.status = ExchangeStatus.CONGESTED
        else:
            self.status = ExchangeStatus.NORMAL
        
        # Calculate effective fee rate based on congestion
        fee_multiplier = 1.0 + (self.config.congestion_fee_multiplier - 1.0) * self.congestion_level
        effective_fee_rate = self.config.base_fee_rate * fee_multiplier
        
        # Calculate settlement delay based on congestion
        base_delay = int(self.congestion_level * self.config.max_settlement_delay)
        
        # Process orders up to capacity
        processed_volume = 0.0
        orders_to_process = []
        
        # Sort by priority (higher first), then by timestamp (earlier first)
        sorted_orders = sorted(
            list(self.pending_orders),
            key=lambda o: (-o.priority, o.timestamp)
        )
        
        for order in sorted_orders:
            if processed_volume + order.quantity <= self.config.base_capacity:
                orders_to_process.append(order)
                processed_volume += order.quantity
            else:
                # Add to backlog
                self.backlog.append(order)
        
        # Fill orders
        for order in orders_to_process:
            # Apply price impact for large orders
            order_impact = 0.001 * (order.quantity / self.config.base_capacity)
            
            if order.order_type == 'sell':
                fill_price = market_price * (1 - order_impact)
            else:
                fill_price = market_price * (1 + order_impact)
            
            # Bound fill price
            fill_price = np.clip(fill_price, 0.001, 1e6)
            
            order.filled = True
            order.fill_price = fill_price
            order.fill_quantity = order.quantity
            order.settlement_delay = base_delay + self._rng.integers(0, 2)
            
            # Calculate fee
            fee = order.quantity * fill_price * effective_fee_rate
            fees_collected += fee
            
            filled_orders.append(order)
            self.settlement_delays.append(order.settlement_delay)
        
        # Update tracking
        self.total_fees_collected += fees_collected
        self.total_orders_processed += len(filled_orders)
        
        # Record for volatility calculation
        self.price_history.append(market_price)
        self.volume_history.append(processed_volume)
        
        # Clear processed orders from pending
        self.pending_orders.clear()
        
        # Move some backlog orders back to pending for next step
        if self.backlog:
            # Process oldest backlog orders first
            num_to_restore = min(len(self.backlog), 
                                 int(self.config.base_capacity * 0.3))
            for _ in range(num_to_restore):
                if self.backlog:
                    self.pending_orders.append(self.backlog.pop(0))
        
        return filled_orders, fees_collected
    
    def get_volatility_indicator(self) -> float:
        """Calculate volatility indicator from price history."""
        if len(self.price_history) < 2:
            return 0.02  # Default volatility
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Handle edge cases
        if len(returns) == 0:
            return 0.02
        
        volatility = np.std(returns) if len(returns) > 1 else 0.02
        return float(np.clip(volatility, 0.001, 1.0))
    
    def get_average_settlement_delay(self) -> float:
        """Get average settlement delay from recent orders."""
        if not self.settlement_delays:
            return 0.0
        recent_delays = self.settlement_delays[-100:]  # Last 100 orders
        return np.mean(recent_delays)
    
    def get_effective_fee_rate(self) -> float:
        """Get current effective fee rate including congestion premium."""
        fee_multiplier = 1.0 + (self.config.congestion_fee_multiplier - 1.0) * self.congestion_level
        return self.config.base_fee_rate * fee_multiplier
    
    def get_state(self) -> ExchangeState:
        """Get current exchange state."""
        return ExchangeState(
            exchange_id=self.exchange_id,
            status=self.status,
            congestion_level=self.congestion_level,
            current_volume=self.current_volume,
            capacity=self.config.base_capacity,
            backlog_size=len(self.backlog),
            avg_settlement_delay=self.get_average_settlement_delay(),
            effective_fee_rate=self.get_effective_fee_rate(),
            volatility_indicator=self.get_volatility_indicator(),
            connected_banks=self.connected_banks.copy(),
            connected_ccps=self.connected_ccps.copy()
        )
    
    def step(self) -> None:
        """Advance exchange by one timestep."""
        self.current_step += 1
        
        # Decay congestion slightly if status was halted
        if self.status == ExchangeStatus.HALTED:
            self.congestion_level *= 0.9
            if self.congestion_level < self.config.halt_threshold:
                self.status = ExchangeStatus.CONGESTED
    
    def reset(self) -> None:
        """Reset exchange to initial state."""
        self._order_counter = 0
        self.order_book.clear()
        self.pending_orders.clear()
        self.backlog.clear()
        
        self.current_step = 0
        self.current_volume = 0.0
        self.congestion_level = 0.0
        self.status = ExchangeStatus.NORMAL
        
        self.price_history.clear()
        self.volume_history.clear()
        
        self.total_fees_collected = 0.0
        self.total_orders_processed = 0
        self.total_orders_rejected = 0
        self.settlement_delays.clear()
    
    def broadcast_market_signal(self) -> Dict[str, float]:
        """Broadcast market signals to connected banks."""
        return {
            'congestion_level': self.congestion_level,
            'effective_fee_rate': self.get_effective_fee_rate(),
            'volatility_indicator': self.get_volatility_indicator(),
            'avg_settlement_delay': self.get_average_settlement_delay(),
            'status': 1.0 if self.status == ExchangeStatus.NORMAL else (
                0.5 if self.status == ExchangeStatus.CONGESTED else 0.0
            )
        }


class ExchangeNetwork:
    """
    Manages multiple exchanges in the financial infrastructure.
    """
    
    def __init__(self, num_exchanges: int = 2, seed: Optional[int] = None):
        self.num_exchanges = num_exchanges
        self._rng = np.random.default_rng(seed)
        
        self.exchanges: Dict[int, Exchange] = {}
        
        # Create exchanges with different characteristics
        for i in range(num_exchanges):
            config = ExchangeConfig(
                exchange_id=i,
                name=f"Exchange_{i}",
                base_capacity=10000.0 * (1 + 0.5 * i),  # Varying capacities
                base_fee_rate=0.001 * (1 + 0.2 * i)  # Varying fee rates
            )
            self.exchanges[i] = Exchange(config, seed=seed + i if seed else None)
    
    def assign_banks_to_exchanges(self, bank_ids: List[int]) -> Dict[int, int]:
        """
        Assign banks to exchanges (banks can route to any, but have primary).
        
        Returns mapping of bank_id -> primary_exchange_id
        """
        assignments = {}
        for bank_id in bank_ids:
            primary_exchange = bank_id % self.num_exchanges
            assignments[bank_id] = primary_exchange
            
            # Connect to primary exchange
            self.exchanges[primary_exchange].connect_bank(bank_id)
            
            # Also connect to secondary exchange for redundancy
            secondary = (primary_exchange + 1) % self.num_exchanges
            self.exchanges[secondary].connect_bank(bank_id)
        
        return assignments
    
    def get_best_exchange_for_order(self, quantity: float) -> int:
        """Find the exchange with lowest effective cost for an order."""
        best_exchange = 0
        best_cost = float('inf')
        
        for ex_id, exchange in self.exchanges.items():
            if exchange.status == ExchangeStatus.HALTED:
                continue
            
            # Cost = fee + delay penalty
            fee_cost = quantity * exchange.get_effective_fee_rate()
            delay_cost = exchange.get_average_settlement_delay() * 0.01 * quantity
            total_cost = fee_cost + delay_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_exchange = ex_id
        
        return best_exchange
    
    def get_aggregated_state(self) -> Dict[str, float]:
        """Get aggregated state across all exchanges."""
        if not self.exchanges:
            return {}
        
        congestion_levels = [ex.congestion_level for ex in self.exchanges.values()]
        fee_rates = [ex.get_effective_fee_rate() for ex in self.exchanges.values()]
        delays = [ex.get_average_settlement_delay() for ex in self.exchanges.values()]
        volatilities = [ex.get_volatility_indicator() for ex in self.exchanges.values()]
        
        return {
            'avg_congestion': np.mean(congestion_levels),
            'max_congestion': np.max(congestion_levels),
            'avg_fee_rate': np.mean(fee_rates),
            'avg_settlement_delay': np.mean(delays),
            'avg_volatility': np.mean(volatilities),
            'num_halted': sum(1 for ex in self.exchanges.values() 
                             if ex.status == ExchangeStatus.HALTED)
        }
    
    def step(self) -> None:
        """Advance all exchanges by one timestep."""
        for exchange in self.exchanges.values():
            exchange.step()
    
    def reset(self) -> None:
        """Reset all exchanges."""
        for exchange in self.exchanges.values():
            exchange.reset()
