"""
Infrastructure Router: Manages transaction routing through exchanges and CCPs.
Replaces direct bank-to-bank settlement with infrastructure-mediated flows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .exchange import Exchange, ExchangeNetwork, Order, ExchangeStatus
from .ccp import CentralCounterparty, CCPNetwork, CCPStatus


class TransactionType(Enum):
    """Types of transactions in the financial system."""
    ASSET_TRADE = "asset_trade"
    INTERBANK_LEND = "interbank_lend"
    INTERBANK_BORROW = "interbank_borrow"
    MARGIN_PAYMENT = "margin_payment"
    SETTLEMENT = "settlement"


@dataclass
class Transaction:
    """A transaction flowing through the infrastructure."""
    transaction_id: int
    transaction_type: TransactionType
    source_bank: int
    target_bank: Optional[int]  # None for market trades
    amount: float
    price: float
    timestamp: int
    
    # Routing info
    exchange_id: Optional[int] = None
    ccp_id: Optional[int] = None
    
    # Status
    submitted: bool = False
    exchange_processed: bool = False
    ccp_processed: bool = False
    settled: bool = False
    settlement_delay: int = 0
    
    # Fees and costs
    exchange_fee: float = 0.0
    ccp_margin_required: float = 0.0
    total_cost: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'transaction_id': self.transaction_id,
            'type': self.transaction_type.value,
            'source_bank': self.source_bank,
            'target_bank': self.target_bank,
            'amount': self.amount,
            'price': self.price,
            'settled': self.settled,
            'settlement_delay': self.settlement_delay,
            'total_cost': self.total_cost
        }


@dataclass
class RoutingDecision:
    """Decision on how to route a transaction."""
    transaction_id: int
    recommended_exchange: int
    recommended_ccp: int
    estimated_fee: float
    estimated_delay: float
    feasible: bool
    rejection_reason: Optional[str] = None


class InfrastructureRouter:
    """
    Routes all financial transactions through exchanges and CCPs.
    
    Enforces the rule: Banks → Exchanges → Clearing Houses → Banks
    No direct bank-to-bank settlement is allowed.
    """
    
    def __init__(self, 
                 exchange_network: ExchangeNetwork,
                 ccp_network: CCPNetwork,
                 seed: Optional[int] = None):
        self.exchange_network = exchange_network
        self.ccp_network = ccp_network
        self._rng = np.random.default_rng(seed)
        
        self._transaction_counter = 0
        self.pending_transactions: Dict[int, Transaction] = {}
        self.completed_transactions: List[Transaction] = []
        self.failed_transactions: List[Transaction] = []
        
        # Bank to infrastructure mappings
        self.bank_primary_exchange: Dict[int, int] = {}
        self.bank_ccp_memberships: Dict[int, List[int]] = {}
        
        # Current step
        self.current_step = 0
        
        # Metrics
        self.total_routed_volume = 0.0
        self.total_fees_paid = 0.0
        self.total_settlements = 0
        self.failed_routings = 0
    
    def register_bank(self, bank_id: int) -> None:
        """Register a bank with the infrastructure."""
        # Assign to exchanges
        assignments = self.exchange_network.assign_banks_to_exchanges([bank_id])
        self.bank_primary_exchange[bank_id] = assignments[bank_id]
        
        # Register with CCPs
        self.bank_ccp_memberships[bank_id] = []
        for ccp_id, ccp in self.ccp_network.ccps.items():
            ccp.add_member(bank_id)
            self.bank_ccp_memberships[bank_id].append(ccp_id)
    
    def route_transaction(self, 
                          source_bank: int,
                          transaction_type: TransactionType,
                          amount: float,
                          price: float,
                          target_bank: Optional[int] = None) -> RoutingDecision:
        """
        Determine optimal routing for a transaction.
        
        Returns a RoutingDecision with recommended infrastructure path.
        """
        self._transaction_counter += 1
        tx_id = self._transaction_counter
        
        # Check if source bank is registered
        if source_bank not in self.bank_primary_exchange:
            return RoutingDecision(
                transaction_id=tx_id,
                recommended_exchange=-1,
                recommended_ccp=-1,
                estimated_fee=0,
                estimated_delay=float('inf'),
                feasible=False,
                rejection_reason="Source bank not registered"
            )
        
        # Find best exchange
        best_exchange = self.exchange_network.get_best_exchange_for_order(amount)
        exchange = self.exchange_network.exchanges[best_exchange]
        
        # Check exchange status
        if exchange.status == ExchangeStatus.HALTED:
            # Try alternate exchange
            for ex_id, ex in self.exchange_network.exchanges.items():
                if ex.status != ExchangeStatus.HALTED:
                    best_exchange = ex_id
                    exchange = ex
                    break
            else:
                return RoutingDecision(
                    transaction_id=tx_id,
                    recommended_exchange=-1,
                    recommended_ccp=-1,
                    estimated_fee=0,
                    estimated_delay=float('inf'),
                    feasible=False,
                    rejection_reason="All exchanges halted"
                )
        
        # Find CCP
        best_ccp = 0  # Default to first CCP
        if self.bank_ccp_memberships.get(source_bank):
            best_ccp = self.bank_ccp_memberships[source_bank][0]
        
        ccp = self.ccp_network.ccps.get(best_ccp)
        if ccp and ccp.status == CCPStatus.FAILED:
            return RoutingDecision(
                transaction_id=tx_id,
                recommended_exchange=best_exchange,
                recommended_ccp=-1,
                estimated_fee=exchange.get_effective_fee_rate() * amount * price,
                estimated_delay=exchange.get_average_settlement_delay(),
                feasible=False,
                rejection_reason="CCP failed"
            )
        
        # Calculate estimated costs
        exchange_fee = exchange.get_effective_fee_rate() * amount * price
        settlement_delay = exchange.get_average_settlement_delay()
        
        # Add CCP margin requirement
        margin_cost = 0.0
        if ccp:
            margin_cost = amount * price * ccp.config.initial_margin_rate
        
        return RoutingDecision(
            transaction_id=tx_id,
            recommended_exchange=best_exchange,
            recommended_ccp=best_ccp,
            estimated_fee=exchange_fee,
            estimated_delay=settlement_delay,
            feasible=True
        )
    
    def submit_transaction(self,
                           source_bank: int,
                           transaction_type: TransactionType,
                           amount: float,
                           price: float,
                           target_bank: Optional[int] = None) -> Optional[Transaction]:
        """
        Submit a transaction to be routed through infrastructure.
        
        All transactions must flow: Bank → Exchange → CCP → Bank
        """
        # Get routing decision
        routing = self.route_transaction(source_bank, transaction_type, amount, price, target_bank)
        
        if not routing.feasible:
            self.failed_routings += 1
            return None
        
        # Create transaction
        transaction = Transaction(
            transaction_id=routing.transaction_id,
            transaction_type=transaction_type,
            source_bank=source_bank,
            target_bank=target_bank,
            amount=amount,
            price=price,
            timestamp=self.current_step,
            exchange_id=routing.recommended_exchange,
            ccp_id=routing.recommended_ccp,
            submitted=True
        )
        
        # Submit to exchange
        exchange = self.exchange_network.exchanges[routing.recommended_exchange]
        order_type = self._transaction_type_to_order_type(transaction_type)
        
        order = exchange.submit_order(
            bank_id=source_bank,
            order_type=order_type,
            asset_type='interbank' if target_bank is not None else 'equity',
            quantity=amount,
            price=price
        )
        
        if order is None:
            self.failed_routings += 1
            self.failed_transactions.append(transaction)
            return None
        
        # Store transaction
        self.pending_transactions[transaction.transaction_id] = transaction
        self.total_routed_volume += amount * price
        
        return transaction
    
    def _transaction_type_to_order_type(self, tx_type: TransactionType) -> str:
        """Convert transaction type to order type."""
        mapping = {
            TransactionType.ASSET_TRADE: 'sell',
            TransactionType.INTERBANK_LEND: 'lend',
            TransactionType.INTERBANK_BORROW: 'borrow',
            TransactionType.MARGIN_PAYMENT: 'buy',
            TransactionType.SETTLEMENT: 'sell'
        }
        return mapping.get(tx_type, 'sell')
    
    def process_step(self, market_price: float) -> Dict[str, Any]:
        """
        Process all infrastructure for one timestep.
        
        Returns summary of processing results.
        """
        results = {
            'filled_orders': [],
            'margin_calls': {},
            'settlements_completed': 0,
            'fees_collected': 0.0,
            'failed_settlements': 0
        }
        
        # Process exchanges
        for ex_id, exchange in self.exchange_network.exchanges.items():
            filled, fees = exchange.process_orders(market_price)
            results['filled_orders'].extend(filled)
            results['fees_collected'] += fees
            self.total_fees_paid += fees
            
            # Update transaction status
            for order in filled:
                for tx_id, tx in self.pending_transactions.items():
                    if tx.source_bank == order.bank_id and not tx.exchange_processed:
                        tx.exchange_processed = True
                        tx.exchange_fee = order.fill_price * order.fill_quantity * exchange.get_effective_fee_rate()
                        tx.settlement_delay = order.settlement_delay
                        break
        
        # Process CCPs
        for ccp_id, ccp in self.ccp_network.ccps.items():
            margin_calls = ccp.process_margin_calls(market_price)
            for member_id, call_amount in margin_calls.items():
                if member_id not in results['margin_calls']:
                    results['margin_calls'][member_id] = 0.0
                results['margin_calls'][member_id] += call_amount
        
        # Complete settlements
        settled_ids = []
        for tx_id, tx in self.pending_transactions.items():
            if tx.exchange_processed and tx.settlement_delay <= 0:
                # Route through CCP
                if tx.ccp_id is not None and tx.ccp_id in self.ccp_network.ccps:
                    ccp = self.ccp_network.ccps[tx.ccp_id]
                    
                    if tx.target_bank is not None:
                        success = ccp.process_settlement(tx.source_bank, tx.target_bank, tx.amount * tx.price)
                        if success:
                            tx.ccp_processed = True
                            tx.settled = True
                            tx.total_cost = tx.exchange_fee
                            settled_ids.append(tx_id)
                            results['settlements_completed'] += 1
                            self.total_settlements += 1
                        else:
                            results['failed_settlements'] += 1
                    else:
                        # Market trade, no bilateral settlement needed
                        tx.ccp_processed = True
                        tx.settled = True
                        tx.total_cost = tx.exchange_fee
                        settled_ids.append(tx_id)
                        results['settlements_completed'] += 1
                        self.total_settlements += 1
            elif tx.exchange_processed:
                tx.settlement_delay -= 1
        
        # Move settled transactions
        for tx_id in settled_ids:
            tx = self.pending_transactions.pop(tx_id)
            self.completed_transactions.append(tx)
        
        # Advance infrastructure
        self.exchange_network.step()
        self.ccp_network.step()
        self.current_step += 1
        
        return results
    
    def get_routing_info_for_bank(self, bank_id: int) -> Dict[str, Any]:
        """Get routing information for a bank's action selection."""
        info = {
            'primary_exchange': self.bank_primary_exchange.get(bank_id, 0),
            'exchange_congestion': {},
            'exchange_fees': {},
            'exchange_delays': {},
            'ccp_margin_status': {},
            'ccp_stress': {}
        }
        
        # Exchange info
        for ex_id, exchange in self.exchange_network.exchanges.items():
            signal = exchange.broadcast_market_signal()
            info['exchange_congestion'][ex_id] = signal['congestion_level']
            info['exchange_fees'][ex_id] = signal['effective_fee_rate']
            info['exchange_delays'][ex_id] = signal['avg_settlement_delay']
        
        # CCP info
        for ccp_id, ccp in self.ccp_network.ccps.items():
            margin_status = ccp.get_member_margin_status(bank_id)
            if margin_status:
                info['ccp_margin_status'][ccp_id] = margin_status
            info['ccp_stress'][ccp_id] = ccp.stress_level
        
        return info
    
    def get_infrastructure_observation(self, bank_id: int) -> np.ndarray:
        """
        Get infrastructure state as observation vector for agent.
        
        Returns 8-dimensional vector:
        - [0] Average exchange congestion
        - [1] Average effective fee rate (normalized)
        - [2] Average settlement delay (normalized)
        - [3] Best exchange status (1=normal, 0.5=congested, 0=halted)
        - [4] CCP stress level
        - [5] Margin adequacy (for this bank)
        - [6] Pending margin calls (normalized)
        - [7] Infrastructure availability (0-1)
        """
        obs = np.zeros(8)
        
        # Exchange metrics
        ex_states = self.exchange_network.get_aggregated_state()
        obs[0] = ex_states.get('avg_congestion', 0.0)
        obs[1] = min(1.0, ex_states.get('avg_fee_rate', 0.001) / 0.01)  # Normalize to 1% max
        obs[2] = min(1.0, ex_states.get('avg_settlement_delay', 0.0) / 5.0)  # Normalize to 5 max
        
        # Best exchange status
        best_status = 1.0
        for ex in self.exchange_network.exchanges.values():
            if ex.status == ExchangeStatus.NORMAL:
                best_status = 1.0
                break
            elif ex.status == ExchangeStatus.CONGESTED:
                best_status = max(0.5, best_status)
        obs[3] = best_status
        
        # CCP metrics
        ccp_states = self.ccp_network.get_aggregated_state()
        obs[4] = ccp_states.get('avg_stress_level', 0.0)
        
        # Bank-specific CCP info
        margin_adequacy = 1.0
        pending_calls = 0.0
        for ccp_id, ccp in self.ccp_network.ccps.items():
            status = ccp.get_member_margin_status(bank_id)
            if status:
                margin_adequacy = min(margin_adequacy, status['margin_adequacy'])
                pending_calls += status['margin_calls_pending']
        
        obs[5] = min(1.0, margin_adequacy)
        obs[6] = min(1.0, pending_calls / 1000.0)  # Normalize
        
        # Overall availability
        num_available_ex = sum(1 for ex in self.exchange_network.exchanges.values() 
                               if ex.status != ExchangeStatus.HALTED)
        num_available_ccp = sum(1 for ccp in self.ccp_network.ccps.values() 
                                if ccp.status != CCPStatus.FAILED)
        
        obs[7] = (num_available_ex / max(1, self.exchange_network.num_exchanges) + 
                  num_available_ccp / max(1, self.ccp_network.num_ccps)) / 2
        
        return obs
    
    def reset(self) -> None:
        """Reset router state."""
        self._transaction_counter = 0
        self.pending_transactions.clear()
        self.completed_transactions.clear()
        self.failed_transactions.clear()
        
        self.current_step = 0
        self.total_routed_volume = 0.0
        self.total_fees_paid = 0.0
        self.total_settlements = 0
        self.failed_routings = 0
        
        self.exchange_network.reset()
        self.ccp_network.reset()
