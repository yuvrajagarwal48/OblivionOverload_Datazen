"""
Bank Entity: Represents a financial institution with balance sheet management.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class BankStatus(Enum):
    """Bank operational status."""
    ACTIVE = "active"
    STRESSED = "stressed"
    DEFAULTED = "defaulted"


@dataclass
class BalanceSheet:
    """Balance sheet representation for a bank."""
    cash: float = 0.0
    illiquid_assets: float = 0.0
    interbank_assets: Dict[int, float] = field(default_factory=dict)  # bank_id -> amount owed TO this bank
    interbank_liabilities: Dict[int, float] = field(default_factory=dict)  # bank_id -> amount owed BY this bank
    external_liabilities: float = 0.0
    
    @property
    def total_interbank_assets(self) -> float:
        """Total amount owed to this bank by other banks."""
        return sum(self.interbank_assets.values())
    
    @property
    def total_interbank_liabilities(self) -> float:
        """Total amount this bank owes to other banks."""
        return sum(self.interbank_liabilities.values())
    
    @property
    def total_assets(self) -> float:
        """Total assets on the balance sheet."""
        return self.cash + self.illiquid_assets + self.total_interbank_assets
    
    @property
    def total_liabilities(self) -> float:
        """Total liabilities on the balance sheet."""
        return self.total_interbank_liabilities + self.external_liabilities
    
    @property
    def equity(self) -> float:
        """Net equity (assets - liabilities)."""
        return self.total_assets - self.total_liabilities
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'cash': self.cash,
            'illiquid_assets': self.illiquid_assets,
            'interbank_assets': dict(self.interbank_assets),
            'interbank_liabilities': dict(self.interbank_liabilities),
            'external_liabilities': self.external_liabilities,
            'total_assets': self.total_assets,
            'total_liabilities': self.total_liabilities,
            'equity': self.equity
        }


class Bank:
    """
    Represents a financial institution in the network.
    
    Attributes:
        bank_id: Unique identifier
        tier: Bank tier (1 = core, 2 = peripheral)
        balance_sheet: Current balance sheet
        status: Operational status
    """
    
    def __init__(self, 
                 bank_id: int,
                 tier: int = 2,
                 initial_cash: float = 100.0,
                 initial_assets: float = 500.0,
                 initial_external_liabilities: float = 100.0,
                 min_capital_ratio: float = 0.08):
        """
        Initialize a bank.
        
        Args:
            bank_id: Unique bank identifier
            tier: Bank tier (1=core, 2=peripheral)
            initial_cash: Starting cash
            initial_assets: Starting illiquid assets
            initial_external_liabilities: External debt
            min_capital_ratio: Minimum required capital ratio
        """
        self.bank_id = bank_id
        self.tier = tier
        self.min_capital_ratio = min_capital_ratio
        
        self.balance_sheet = BalanceSheet(
            cash=initial_cash,
            illiquid_assets=initial_assets,
            external_liabilities=initial_external_liabilities
        )
        
        self.status = BankStatus.ACTIVE
        self._default_count = 0
        self._payment_history: list = []
        self._received_history: list = []
        
    @property
    def capital_ratio(self) -> float:
        """Calculate capital ratio (equity / risk-weighted assets)."""
        # Simplified: use total assets as risk-weighted assets
        total_assets = self.balance_sheet.total_assets
        if total_assets <= 0:
            return 0.0
        return max(0.0, self.balance_sheet.equity / total_assets)
    
    @property
    def is_solvent(self) -> bool:
        """Check if bank is solvent (positive equity)."""
        return self.balance_sheet.equity > 0
    
    @property
    def is_liquid(self) -> bool:
        """Check if bank has sufficient liquidity."""
        # Must have enough cash to cover immediate obligations
        immediate_obligations = self.balance_sheet.total_interbank_liabilities * 0.1
        return self.balance_sheet.cash >= immediate_obligations
    
    @property
    def excess_cash(self) -> float:
        """Calculate excess cash available for lending."""
        # Reserve requirement: keep some cash for obligations
        reserve_requirement = self.balance_sheet.total_interbank_liabilities * 0.2
        buffer = self.balance_sheet.illiquid_assets * 0.05
        required_cash = reserve_requirement + buffer
        return max(0.0, self.balance_sheet.cash - required_cash)
    
    def add_interbank_asset(self, creditor_id: int, amount: float) -> None:
        """Record an interbank asset (loan made to another bank)."""
        if creditor_id in self.balance_sheet.interbank_assets:
            self.balance_sheet.interbank_assets[creditor_id] += amount
        else:
            self.balance_sheet.interbank_assets[creditor_id] = amount
    
    def add_interbank_liability(self, debtor_id: int, amount: float) -> None:
        """Record an interbank liability (loan received from another bank)."""
        if debtor_id in self.balance_sheet.interbank_liabilities:
            self.balance_sheet.interbank_liabilities[debtor_id] += amount
        else:
            self.balance_sheet.interbank_liabilities[debtor_id] = amount
    
    def reduce_interbank_asset(self, creditor_id: int, amount: float) -> float:
        """Reduce interbank asset (partial or full repayment received)."""
        if creditor_id not in self.balance_sheet.interbank_assets:
            return 0.0
        current = self.balance_sheet.interbank_assets[creditor_id]
        reduction = min(current, amount)
        self.balance_sheet.interbank_assets[creditor_id] -= reduction
        if self.balance_sheet.interbank_assets[creditor_id] <= 0:
            del self.balance_sheet.interbank_assets[creditor_id]
        return reduction
    
    def reduce_interbank_liability(self, debtor_id: int, amount: float) -> float:
        """Reduce interbank liability (payment made to creditor)."""
        if debtor_id not in self.balance_sheet.interbank_liabilities:
            return 0.0
        current = self.balance_sheet.interbank_liabilities[debtor_id]
        reduction = min(current, amount)
        self.balance_sheet.interbank_liabilities[debtor_id] -= reduction
        if self.balance_sheet.interbank_liabilities[debtor_id] <= 0:
            del self.balance_sheet.interbank_liabilities[debtor_id]
        return reduction
    
    def make_payment(self, amount: float) -> float:
        """
        Make a payment from cash reserves.
        
        Args:
            amount: Requested payment amount
            
        Returns:
            Actual amount paid (limited by available cash)
        """
        actual_payment = min(amount, self.balance_sheet.cash)
        self.balance_sheet.cash -= actual_payment
        self._payment_history.append(actual_payment)
        return actual_payment
    
    def receive_payment(self, amount: float) -> None:
        """Receive a payment into cash reserves."""
        self.balance_sheet.cash += amount
        self._received_history.append(amount)
    
    def sell_assets(self, fraction: float, price: float) -> float:
        """
        Sell a fraction of illiquid assets.
        
        Args:
            fraction: Fraction of assets to sell [0, 1]
            price: Current market price
            
        Returns:
            Cash received from sale
        """
        fraction = np.clip(fraction, 0.0, 1.0)
        assets_to_sell = self.balance_sheet.illiquid_assets * fraction
        cash_received = assets_to_sell * price
        self.balance_sheet.illiquid_assets -= assets_to_sell
        self.balance_sheet.cash += cash_received
        return cash_received
    
    def apply_asset_price_shock(self, price_change: float) -> None:
        """
        Apply a price shock to illiquid assets.
        
        Args:
            price_change: Fractional change in price (negative = loss)
        """
        # Mark-to-market adjustment
        value_change = self.balance_sheet.illiquid_assets * price_change
        self.balance_sheet.illiquid_assets += value_change
        self.balance_sheet.illiquid_assets = max(0.0, self.balance_sheet.illiquid_assets)
    
    def check_and_update_status(self) -> BankStatus:
        """Check solvency and update status."""
        if not self.is_solvent:
            self.status = BankStatus.DEFAULTED
            self._default_count += 1
        elif self.capital_ratio < self.min_capital_ratio:
            self.status = BankStatus.STRESSED
        else:
            self.status = BankStatus.ACTIVE
        return self.status
    
    def trigger_margin_call(self, price: float, max_sell_limit: float = 0.3) -> float:
        """
        Trigger forced asset sales if capital ratio is below minimum.
        
        Args:
            price: Current asset price
            max_sell_limit: Maximum fraction of assets to sell
            
        Returns:
            Volume of assets sold
        """
        if self.capital_ratio >= self.min_capital_ratio:
            return 0.0
        
        # Calculate how much we need to sell to meet capital ratio
        target_equity = self.balance_sheet.total_assets * self.min_capital_ratio
        equity_shortfall = target_equity - self.balance_sheet.equity
        
        if equity_shortfall <= 0:
            return 0.0
        
        # Sell assets to raise cash (limited by max_sell_limit)
        max_sellable = self.balance_sheet.illiquid_assets * max_sell_limit
        sell_volume = min(equity_shortfall / max(price, 0.01), max_sellable)
        
        if sell_volume > 0:
            self.sell_assets(sell_volume / max(self.balance_sheet.illiquid_assets, 1.0), price)
        
        return sell_volume
    
    def reset(self, 
              initial_cash: float,
              initial_assets: float,
              initial_external_liabilities: float) -> None:
        """Reset bank to initial state."""
        self.balance_sheet = BalanceSheet(
            cash=initial_cash,
            illiquid_assets=initial_assets,
            external_liabilities=initial_external_liabilities
        )
        self.status = BankStatus.ACTIVE
        self._payment_history = []
        self._received_history = []
    
    def get_observation(self) -> Dict[str, float]:
        """Get observable state for the agent."""
        return {
            'cash': self.balance_sheet.cash,
            'equity': self.balance_sheet.equity,
            'capital_ratio': self.capital_ratio,
            'total_owed': self.balance_sheet.total_interbank_liabilities,
            'total_owing': self.balance_sheet.total_interbank_assets,
            'illiquid_assets': self.balance_sheet.illiquid_assets,
            'is_stressed': float(self.status == BankStatus.STRESSED),
            'default_count': float(self._default_count)
        }
    
    def __repr__(self) -> str:
        return (f"Bank(id={self.bank_id}, tier={self.tier}, "
                f"equity={self.balance_sheet.equity:.2f}, "
                f"CR={self.capital_ratio:.2%}, status={self.status.value})")
