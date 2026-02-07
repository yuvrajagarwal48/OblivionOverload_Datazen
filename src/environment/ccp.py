"""
Central Counterparty (CCP) / Clearing House: Risk management node for financial infrastructure.
Implements margin collection, netting, and default waterfall mechanisms.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class CCPStatus(Enum):
    """CCP operational status."""
    NORMAL = "normal"
    STRESSED = "stressed"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class MarginAccount:
    """Margin account for a clearing member."""
    member_id: int
    initial_margin: float = 0.0
    variation_margin: float = 0.0
    default_fund_contribution: float = 0.0
    margin_calls_pending: float = 0.0
    last_updated: int = 0
    
    @property
    def total_margin(self) -> float:
        return self.initial_margin + self.variation_margin
    
    def to_dict(self) -> dict:
        return {
            'member_id': self.member_id,
            'initial_margin': self.initial_margin,
            'variation_margin': self.variation_margin,
            'default_fund_contribution': self.default_fund_contribution,
            'margin_calls_pending': self.margin_calls_pending,
            'total_margin': self.total_margin
        }


@dataclass
class Position:
    """Net position of a member."""
    member_id: int
    gross_long: float = 0.0
    gross_short: float = 0.0
    net_position: float = 0.0
    mark_to_market: float = 0.0
    
    def update_net(self):
        self.net_position = self.gross_long - self.gross_short


@dataclass
class CCPState:
    """Current state of the clearing house."""
    ccp_id: int
    status: CCPStatus
    total_initial_margin: float
    total_variation_margin: float
    default_fund_size: float
    ccp_capital: float
    stress_level: float  # 0 to 1
    num_members: int
    num_defaulted_members: int
    pending_settlements: int
    margin_call_volume: float
    
    def to_dict(self) -> dict:
        return {
            'ccp_id': self.ccp_id,
            'status': self.status.value,
            'total_initial_margin': self.total_initial_margin,
            'total_variation_margin': self.total_variation_margin,
            'default_fund_size': self.default_fund_size,
            'ccp_capital': self.ccp_capital,
            'stress_level': self.stress_level,
            'num_members': self.num_members,
            'num_defaulted_members': self.num_defaulted_members,
            'pending_settlements': self.pending_settlements,
            'margin_call_volume': self.margin_call_volume
        }


@dataclass
class CCPConfig:
    """Configuration for a CCP."""
    ccp_id: int
    name: str = "ClearingHouse"
    
    # Margin parameters
    initial_margin_rate: float = 0.10  # 10% of exposure
    variation_margin_rate: float = 0.05  # 5% daily variation threshold
    margin_call_threshold: float = 0.8  # Trigger margin call at 80% utilization
    
    # Default fund
    default_fund_rate: float = 0.02  # 2% of exposure contributed to default fund
    min_default_fund: float = 1000.0
    
    # CCP capital
    initial_capital: float = 10000.0
    min_capital_ratio: float = 0.05
    
    # Volatility adjustment
    volatility_margin_multiplier: float = 2.0  # Increase margin in high vol
    stress_margin_multiplier: float = 1.5
    
    # Netting efficiency
    netting_efficiency: float = 0.7  # 70% reduction through multilateral netting


class WaterfallResult:
    """Result of default waterfall application."""
    
    def __init__(self, loss_amount: float, defaulting_member: int):
        self.loss_amount = loss_amount
        self.defaulting_member = defaulting_member
        
        # Absorption layers
        self.absorbed_by_member_margin = 0.0
        self.absorbed_by_default_fund = 0.0
        self.absorbed_by_ccp_capital = 0.0
        self.absorbed_by_mutualization = 0.0
        self.unabsorbed_loss = 0.0
        
        # Status
        self.fully_absorbed = True
        self.ccp_survived = True
        self.mutualization_triggered = False
    
    def to_dict(self) -> dict:
        return {
            'loss_amount': self.loss_amount,
            'defaulting_member': self.defaulting_member,
            'absorbed_by_member_margin': self.absorbed_by_member_margin,
            'absorbed_by_default_fund': self.absorbed_by_default_fund,
            'absorbed_by_ccp_capital': self.absorbed_by_ccp_capital,
            'absorbed_by_mutualization': self.absorbed_by_mutualization,
            'unabsorbed_loss': self.unabsorbed_loss,
            'fully_absorbed': self.fully_absorbed,
            'ccp_survived': self.ccp_survived,
            'mutualization_triggered': self.mutualization_triggered
        }


class CentralCounterparty:
    """
    Central Counterparty (CCP) / Clearing House.
    
    Responsibilities:
    - Collect initial and variation margins
    - Net all member positions (multilateral netting)
    - Replace bilateral settlement with CCP settlement
    - Apply default waterfall when members default
    - Adjust margins based on volatility
    """
    
    def __init__(self, config: CCPConfig, seed: Optional[int] = None):
        self.config = config
        self.ccp_id = config.ccp_id
        self._rng = np.random.default_rng(seed)
        
        # Capital and funds
        self.ccp_capital = config.initial_capital
        self.default_fund = config.min_default_fund
        
        # Member management
        self.members: List[int] = []
        self.margin_accounts: Dict[int, MarginAccount] = {}
        self.positions: Dict[int, Position] = {}
        self.defaulted_members: List[int] = []
        
        # Settlement queue
        self.pending_settlements: List[Dict] = []
        
        # State tracking
        self.current_step = 0
        self.status = CCPStatus.NORMAL
        self.stress_level = 0.0
        self.current_volatility = 0.02
        
        # Metrics
        self.total_margin_calls = 0.0
        self.total_settlements_processed = 0
        self.waterfall_events: List[WaterfallResult] = []
    
    def add_member(self, member_id: int, initial_exposure: float = 0.0) -> None:
        """Register a new clearing member."""
        if member_id not in self.members:
            self.members.append(member_id)
            
            # Create margin account
            initial_margin = initial_exposure * self.config.initial_margin_rate
            default_fund_contrib = max(
                initial_exposure * self.config.default_fund_rate,
                self.config.min_default_fund / len(self.members) if self.members else 100
            )
            
            self.margin_accounts[member_id] = MarginAccount(
                member_id=member_id,
                initial_margin=initial_margin,
                default_fund_contribution=default_fund_contrib
            )
            
            # Create position tracker
            self.positions[member_id] = Position(member_id=member_id)
            
            # Add to default fund
            self.default_fund += default_fund_contrib
    
    def update_position(self, member_id: int, trade_type: str, 
                        quantity: float, price: float) -> None:
        """Update member position after a trade."""
        if member_id not in self.positions:
            return
        
        position = self.positions[member_id]
        notional = quantity * price
        
        if trade_type in ['buy', 'lend']:
            position.gross_long += notional
        elif trade_type in ['sell', 'borrow']:
            position.gross_short += notional
        
        position.update_net()
    
    def calculate_margin_requirements(self, member_id: int, 
                                       current_price: float) -> Tuple[float, float]:
        """
        Calculate margin requirements for a member.
        
        Returns: (initial_margin_required, variation_margin_required)
        """
        if member_id not in self.positions:
            return 0.0, 0.0
        
        position = self.positions[member_id]
        
        # Base exposure
        gross_exposure = position.gross_long + position.gross_short
        net_exposure = abs(position.net_position)
        
        # Apply netting efficiency
        effective_exposure = net_exposure + (1 - self.config.netting_efficiency) * (gross_exposure - net_exposure)
        
        # Volatility adjustment
        vol_multiplier = 1.0 + (self.current_volatility / 0.02 - 1.0) * (self.config.volatility_margin_multiplier - 1.0)
        vol_multiplier = np.clip(vol_multiplier, 1.0, self.config.volatility_margin_multiplier)
        
        # Stress adjustment
        stress_multiplier = 1.0 + self.stress_level * (self.config.stress_margin_multiplier - 1.0)
        
        # Calculate requirements
        total_multiplier = vol_multiplier * stress_multiplier
        initial_margin_required = effective_exposure * self.config.initial_margin_rate * total_multiplier
        
        # Variation margin based on mark-to-market
        mtm_change = position.net_position * (current_price - 1.0)  # Simplified MTM
        variation_margin_required = abs(mtm_change) * self.config.variation_margin_rate
        
        return initial_margin_required, variation_margin_required
    
    def process_margin_calls(self, current_price: float) -> Dict[int, float]:
        """
        Process margin calls for all members.
        
        Returns: Dict mapping member_id to margin call amount
        """
        margin_calls = {}
        
        for member_id in self.members:
            if member_id in self.defaulted_members:
                continue
            
            account = self.margin_accounts[member_id]
            im_required, vm_required = self.calculate_margin_requirements(member_id, current_price)
            
            total_required = im_required + vm_required
            current_margin = account.total_margin
            
            # Check if margin call needed
            if current_margin < total_required * self.config.margin_call_threshold:
                call_amount = total_required - current_margin
                call_amount = max(0, call_amount)
                
                if call_amount > 0:
                    margin_calls[member_id] = call_amount
                    account.margin_calls_pending = call_amount
                    self.total_margin_calls += call_amount
        
        return margin_calls
    
    def receive_margin(self, member_id: int, amount: float, margin_type: str = 'initial') -> bool:
        """Receive margin payment from a member."""
        if member_id not in self.margin_accounts:
            return False
        
        account = self.margin_accounts[member_id]
        
        # Bound for numerical stability
        amount = np.clip(amount, 0, 1e9)
        
        if margin_type == 'initial':
            account.initial_margin += amount
        elif margin_type == 'variation':
            account.variation_margin += amount
        elif margin_type == 'default_fund':
            account.default_fund_contribution += amount
            self.default_fund += amount
        
        # Clear pending margin calls
        account.margin_calls_pending = max(0, account.margin_calls_pending - amount)
        account.last_updated = self.current_step
        
        return True
    
    def multilateral_netting(self) -> Dict[int, float]:
        """
        Perform multilateral netting of all positions.
        
        Returns: Dict mapping member_id to net settlement amount
        """
        net_settlements = {}
        
        # Calculate net position for each member
        for member_id in self.members:
            if member_id in self.defaulted_members:
                continue
            
            position = self.positions[member_id]
            
            # Net amount = net position (positive = receive, negative = pay)
            # Apply netting efficiency
            net_settlements[member_id] = position.net_position * self.config.netting_efficiency
        
        # Normalize to ensure conservation (sum to zero)
        total = sum(net_settlements.values())
        if abs(total) > 1e-6 and len(net_settlements) > 0:
            adjustment = total / len(net_settlements)
            for member_id in net_settlements:
                net_settlements[member_id] -= adjustment
        
        return net_settlements
    
    def apply_default_waterfall(self, defaulting_member: int, 
                                 loss_amount: float) -> WaterfallResult:
        """
        Apply the default waterfall mechanism.
        
        Waterfall order:
        1. Defaulting member's margin
        2. Defaulting member's default fund contribution
        3. CCP capital (skin in the game)
        4. Surviving members' default fund contributions (mutualization)
        """
        result = WaterfallResult(loss_amount, defaulting_member)
        remaining_loss = loss_amount
        
        # Bound for stability
        remaining_loss = np.clip(remaining_loss, 0, 1e12)
        
        # Layer 1: Defaulting member's margin
        if defaulting_member in self.margin_accounts:
            account = self.margin_accounts[defaulting_member]
            available = account.total_margin
            absorbed = min(remaining_loss, available)
            
            result.absorbed_by_member_margin = absorbed
            remaining_loss -= absorbed
            account.initial_margin = max(0, account.initial_margin - absorbed)
            account.variation_margin = max(0, account.variation_margin - (absorbed - account.initial_margin) if absorbed > account.initial_margin else 0)
        
        # Layer 2: Default fund (member's contribution first, then shared)
        if remaining_loss > 0:
            # First, member's own contribution
            if defaulting_member in self.margin_accounts:
                account = self.margin_accounts[defaulting_member]
                member_contrib = account.default_fund_contribution
                absorbed = min(remaining_loss, member_contrib)
                
                result.absorbed_by_default_fund += absorbed
                remaining_loss -= absorbed
                self.default_fund -= absorbed
                account.default_fund_contribution = 0
        
        # Layer 3: CCP capital (skin in the game)
        if remaining_loss > 0:
            # CCP contributes up to a portion of its capital
            ccp_contribution = min(remaining_loss, self.ccp_capital * 0.25)
            
            result.absorbed_by_ccp_capital = ccp_contribution
            remaining_loss -= ccp_contribution
            self.ccp_capital -= ccp_contribution
        
        # Layer 4: Shared default fund
        if remaining_loss > 0 and self.default_fund > 0:
            absorbed = min(remaining_loss, self.default_fund)
            
            result.absorbed_by_default_fund += absorbed
            remaining_loss -= absorbed
            self.default_fund -= absorbed
        
        # Layer 5: Mutualization across surviving members
        if remaining_loss > 0:
            result.mutualization_triggered = True
            surviving_members = [m for m in self.members 
                               if m != defaulting_member and m not in self.defaulted_members]
            
            if surviving_members:
                per_member_share = remaining_loss / len(surviving_members)
                total_mutualized = 0.0
                
                for member_id in surviving_members:
                    account = self.margin_accounts[member_id]
                    # Take from their margin
                    available = account.total_margin * 0.5  # Max 50% of margin
                    contribution = min(per_member_share, available)
                    
                    account.initial_margin -= contribution
                    total_mutualized += contribution
                
                result.absorbed_by_mutualization = total_mutualized
                remaining_loss -= total_mutualized
        
        # Check if fully absorbed
        result.unabsorbed_loss = max(0, remaining_loss)
        result.fully_absorbed = remaining_loss <= 1e-6
        result.ccp_survived = self.ccp_capital > self.config.initial_capital * self.config.min_capital_ratio
        
        # Mark member as defaulted
        if defaulting_member not in self.defaulted_members:
            self.defaulted_members.append(defaulting_member)
        
        # Update stress level
        self.stress_level = min(1.0, self.stress_level + 0.2)
        
        # Store event
        self.waterfall_events.append(result)
        
        # Update CCP status
        self._update_status()
        
        return result
    
    def _update_status(self) -> None:
        """Update CCP status based on current state."""
        capital_ratio = self.ccp_capital / self.config.initial_capital
        default_ratio = len(self.defaulted_members) / max(1, len(self.members))
        
        if capital_ratio < self.config.min_capital_ratio or default_ratio > 0.5:
            self.status = CCPStatus.FAILED
        elif capital_ratio < 0.3 or self.stress_level > 0.8:
            self.status = CCPStatus.CRITICAL
        elif capital_ratio < 0.5 or self.stress_level > 0.5:
            self.status = CCPStatus.STRESSED
        else:
            self.status = CCPStatus.NORMAL
    
    def process_settlement(self, from_member: int, to_member: int, 
                           amount: float) -> bool:
        """
        Process a settlement between members through the CCP.
        All settlements go through CCP, replacing bilateral.
        """
        if from_member in self.defaulted_members:
            return False
        
        # Check margin adequacy
        if from_member in self.margin_accounts:
            account = self.margin_accounts[from_member]
            if account.total_margin < amount * 0.1:  # 10% margin coverage
                # Margin call
                account.margin_calls_pending += amount * 0.1 - account.total_margin
                return False
        
        # Process settlement
        self.pending_settlements.append({
            'from': from_member,
            'to': to_member,
            'amount': amount,
            'timestamp': self.current_step
        })
        
        self.total_settlements_processed += 1
        return True
    
    def get_state(self) -> CCPState:
        """Get current CCP state."""
        total_im = sum(a.initial_margin for a in self.margin_accounts.values())
        total_vm = sum(a.variation_margin for a in self.margin_accounts.values())
        margin_calls = sum(a.margin_calls_pending for a in self.margin_accounts.values())
        
        return CCPState(
            ccp_id=self.ccp_id,
            status=self.status,
            total_initial_margin=total_im,
            total_variation_margin=total_vm,
            default_fund_size=self.default_fund,
            ccp_capital=self.ccp_capital,
            stress_level=self.stress_level,
            num_members=len(self.members),
            num_defaulted_members=len(self.defaulted_members),
            pending_settlements=len(self.pending_settlements),
            margin_call_volume=margin_calls
        )
    
    def update_volatility(self, new_volatility: float) -> None:
        """Update current market volatility for margin calculations."""
        self.current_volatility = np.clip(new_volatility, 0.001, 1.0)
    
    def step(self) -> None:
        """Advance CCP by one timestep."""
        self.current_step += 1
        
        # Decay stress level slowly
        self.stress_level = max(0, self.stress_level - 0.01)
        
        # Clear old settlements
        self.pending_settlements = [
            s for s in self.pending_settlements 
            if self.current_step - s['timestamp'] < 5
        ]
        
        self._update_status()
    
    def reset(self) -> None:
        """Reset CCP to initial state."""
        self.ccp_capital = self.config.initial_capital
        self.default_fund = self.config.min_default_fund
        
        self.members.clear()
        self.margin_accounts.clear()
        self.positions.clear()
        self.defaulted_members.clear()
        self.pending_settlements.clear()
        
        self.current_step = 0
        self.status = CCPStatus.NORMAL
        self.stress_level = 0.0
        self.current_volatility = 0.02
        
        self.total_margin_calls = 0.0
        self.total_settlements_processed = 0
        self.waterfall_events.clear()
    
    def get_member_margin_status(self, member_id: int) -> Optional[Dict]:
        """Get margin status for a specific member."""
        if member_id not in self.margin_accounts:
            return None
        
        account = self.margin_accounts[member_id]
        im_req, vm_req = self.calculate_margin_requirements(member_id, 1.0)
        
        return {
            'member_id': member_id,
            'initial_margin': account.initial_margin,
            'variation_margin': account.variation_margin,
            'total_margin': account.total_margin,
            'initial_margin_required': im_req,
            'variation_margin_required': vm_req,
            'margin_calls_pending': account.margin_calls_pending,
            'margin_adequacy': account.total_margin / max(im_req + vm_req, 1e-6)
        }


class CCPNetwork:
    """
    Manages multiple CCPs for different asset classes or markets.
    """
    
    def __init__(self, num_ccps: int = 1, seed: Optional[int] = None):
        self.num_ccps = num_ccps
        self._rng = np.random.default_rng(seed)
        
        self.ccps: Dict[int, CentralCounterparty] = {}
        
        for i in range(num_ccps):
            config = CCPConfig(
                ccp_id=i,
                name=f"CCP_{i}",
                initial_capital=10000.0 * (1 + 0.5 * i)
            )
            self.ccps[i] = CentralCounterparty(config, seed=seed + i if seed else None)
    
    def get_aggregated_state(self) -> Dict[str, float]:
        """Get aggregated state across all CCPs."""
        if not self.ccps:
            return {}
        
        states = [ccp.get_state() for ccp in self.ccps.values()]
        
        return {
            'total_margin': sum(s.total_initial_margin + s.total_variation_margin for s in states),
            'total_default_fund': sum(s.default_fund_size for s in states),
            'total_ccp_capital': sum(s.ccp_capital for s in states),
            'avg_stress_level': np.mean([s.stress_level for s in states]),
            'max_stress_level': max(s.stress_level for s in states),
            'num_failed_ccps': sum(1 for s in states if s.status == CCPStatus.FAILED),
            'total_pending_settlements': sum(s.pending_settlements for s in states),
            'total_margin_calls': sum(s.margin_call_volume for s in states)
        }
    
    def step(self) -> None:
        """Advance all CCPs."""
        for ccp in self.ccps.values():
            ccp.step()
    
    def reset(self) -> None:
        """Reset all CCPs."""
        for ccp in self.ccps.values():
            ccp.reset()
