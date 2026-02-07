"""
Regulatory Policy Layer for FinSim-MAPPO.
Centralized policy definitions with enforcement hooks.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json


class PolicyRegime(str, Enum):
    """Predefined regulatory regimes."""
    MINIMAL = "minimal"          # Light regulation
    BASEL_II = "basel_ii"        # Basel II requirements
    BASEL_III = "basel_iii"      # Basel III requirements
    BASEL_IV = "basel_iv"        # Basel IV (more stringent)
    CRISIS_MODE = "crisis_mode"  # Emergency enhanced requirements
    CUSTOM = "custom"            # User-defined


class ViolationSeverity(str, Enum):
    """Severity of policy violations."""
    WARNING = "warning"          # Minor breach, monitoring needed
    MINOR = "minor"              # Small breach, corrective action required
    MODERATE = "moderate"        # Significant breach, restrictions applied
    SEVERE = "severe"            # Major breach, intervention triggered
    CRITICAL = "critical"        # Critical breach, resolution action


@dataclass
class PolicyViolation:
    """Record of a policy violation."""
    violation_id: str
    entity_id: int
    policy_name: str
    parameter_name: str
    required_value: float
    actual_value: float
    severity: ViolationSeverity
    timestamp: int
    description: str = ""
    remediation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['severity'] = self.severity.value
        return result


@dataclass
class PolicyParameters:
    """
    Complete set of regulatory policy parameters.
    Based on Basel III/IV framework with extensions.
    """
    # Capital Requirements
    minimum_capital_ratio: float = 0.08  # Tier 1 capital / RWA
    conservation_buffer: float = 0.025   # Capital conservation buffer
    countercyclical_buffer: float = 0.0  # Counter-cyclical buffer (0-2.5%)
    systemic_buffer: float = 0.0         # G-SIB/D-SIB buffer
    
    @property
    def total_capital_requirement(self) -> float:
        return (self.minimum_capital_ratio + self.conservation_buffer + 
                self.countercyclical_buffer + self.systemic_buffer)
    
    # Leverage Requirements
    minimum_leverage_ratio: float = 0.03  # Tier 1 / Total Exposure
    maximum_leverage_multiple: float = 33.3  # 1 / 0.03
    
    # Liquidity Requirements
    minimum_lcr: float = 1.0              # Liquidity Coverage Ratio
    minimum_nsfr: float = 1.0             # Net Stable Funding Ratio
    liquidity_buffer_ratio: float = 0.1  # Minimum cash / short-term liabilities
    
    # Exposure Limits
    large_exposure_limit: float = 0.25   # Max exposure to single counterparty / capital
    connected_exposure_limit: float = 0.15  # Max to connected parties
    interbank_limit_ratio: float = 0.5   # Max interbank / total assets
    
    # CCP/Margin Requirements
    minimum_initial_margin: float = 0.03  # 3% initial margin
    minimum_variation_margin_freq: int = 1  # Daily VM calls
    margin_call_threshold: float = 0.8   # Call at 80% of IM
    
    # Stress Testing Requirements
    stress_capital_addon: float = 0.0    # Additional capital for stress scenarios
    stress_liquidity_addon: float = 0.0  # Additional liquidity for stress
    
    # Recovery & Resolution
    resolution_trigger_ratio: float = 0.02  # Below this, resolution triggered
    early_intervention_ratio: float = 0.04  # Below this, early intervention
    
    # Operational Limits
    max_daily_settlement: float = 1e12   # Maximum daily settlement value
    max_intraday_credit: float = 1e10    # Maximum intraday credit extension
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_regime(cls, regime: PolicyRegime) -> 'PolicyParameters':
        """Create parameters for a predefined regime."""
        if regime == PolicyRegime.MINIMAL:
            return cls(
                minimum_capital_ratio=0.04,
                conservation_buffer=0.0,
                minimum_leverage_ratio=0.02,
                minimum_lcr=0.5,
                large_exposure_limit=0.5
            )
        
        elif regime == PolicyRegime.BASEL_II:
            return cls(
                minimum_capital_ratio=0.08,
                conservation_buffer=0.0,
                minimum_leverage_ratio=0.02,
                minimum_lcr=0.5,
                minimum_nsfr=0.5
            )
        
        elif regime == PolicyRegime.BASEL_III:
            return cls(
                minimum_capital_ratio=0.08,
                conservation_buffer=0.025,
                minimum_leverage_ratio=0.03,
                minimum_lcr=1.0,
                minimum_nsfr=1.0,
                large_exposure_limit=0.25
            )
        
        elif regime == PolicyRegime.BASEL_IV:
            return cls(
                minimum_capital_ratio=0.10,
                conservation_buffer=0.025,
                countercyclical_buffer=0.01,
                minimum_leverage_ratio=0.035,
                minimum_lcr=1.1,
                minimum_nsfr=1.0,
                large_exposure_limit=0.20
            )
        
        elif regime == PolicyRegime.CRISIS_MODE:
            return cls(
                minimum_capital_ratio=0.12,
                conservation_buffer=0.04,
                countercyclical_buffer=0.025,
                minimum_leverage_ratio=0.05,
                minimum_lcr=1.5,
                minimum_nsfr=1.2,
                large_exposure_limit=0.15,
                stress_capital_addon=0.02,
                stress_liquidity_addon=0.05
            )
        
        else:  # CUSTOM
            return cls()


class PolicyRule(ABC):
    """Abstract base class for policy rules."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = True
    
    @abstractmethod
    def check(self, entity: Any, parameters: PolicyParameters) -> Optional[PolicyViolation]:
        """Check if entity complies with this rule."""
        pass
    
    @abstractmethod
    def enforce(self, entity: Any, violation: PolicyViolation) -> Dict[str, Any]:
        """Enforce remedy for a violation."""
        pass


class CapitalRatioRule(PolicyRule):
    """Check and enforce capital ratio requirements."""
    
    def __init__(self):
        super().__init__(
            "capital_ratio",
            "Minimum regulatory capital ratio requirement"
        )
    
    def check(self, entity: Any, parameters: PolicyParameters) -> Optional[PolicyViolation]:
        if not hasattr(entity, 'capital_ratio'):
            return None
        
        required = parameters.total_capital_requirement
        actual = entity.capital_ratio
        
        if actual < required:
            shortfall = required - actual
            
            if actual < parameters.resolution_trigger_ratio:
                severity = ViolationSeverity.CRITICAL
            elif actual < parameters.early_intervention_ratio:
                severity = ViolationSeverity.SEVERE
            elif shortfall > 0.03:
                severity = ViolationSeverity.MODERATE
            else:
                severity = ViolationSeverity.MINOR
            
            return PolicyViolation(
                violation_id=f"cap_{entity.bank_id}_{datetime.now().isoformat()}",
                entity_id=entity.bank_id,
                policy_name=self.name,
                parameter_name="capital_ratio",
                required_value=required,
                actual_value=actual,
                severity=severity,
                timestamp=getattr(entity, 'current_timestep', 0),
                description=f"Capital ratio {actual:.2%} below requirement {required:.2%}",
                remediation_actions=[
                    "Reduce risk-weighted assets",
                    "Raise additional capital",
                    "Suspend dividend payments"
                ]
            )
        
        return None
    
    def enforce(self, entity: Any, violation: PolicyViolation) -> Dict[str, Any]:
        """Apply enforcement actions."""
        actions = {}
        
        if violation.severity == ViolationSeverity.CRITICAL:
            actions['activity_restrictions'] = ['all_lending_suspended', 'trading_suspended']
            actions['intervention'] = 'resolution_triggered'
            
        elif violation.severity == ViolationSeverity.SEVERE:
            actions['activity_restrictions'] = ['new_lending_suspended', 'dividend_blocked']
            actions['required_capital_plan'] = True
            
        elif violation.severity == ViolationSeverity.MODERATE:
            actions['activity_restrictions'] = ['dividend_restricted']
            actions['enhanced_monitoring'] = True
            
        else:  # MINOR
            actions['notification_required'] = True
        
        return actions


class LeverageRatioRule(PolicyRule):
    """Check leverage ratio requirements."""
    
    def __init__(self):
        super().__init__(
            "leverage_ratio",
            "Maximum leverage (minimum leverage ratio) requirement"
        )
    
    def check(self, entity: Any, parameters: PolicyParameters) -> Optional[PolicyViolation]:
        if not hasattr(entity, 'balance_sheet'):
            return None
        
        bs = entity.balance_sheet
        leverage_ratio = bs.equity / max(bs.total_assets, 1e-8)
        
        if leverage_ratio < parameters.minimum_leverage_ratio:
            return PolicyViolation(
                violation_id=f"lev_{entity.bank_id}_{datetime.now().isoformat()}",
                entity_id=entity.bank_id,
                policy_name=self.name,
                parameter_name="leverage_ratio",
                required_value=parameters.minimum_leverage_ratio,
                actual_value=leverage_ratio,
                severity=ViolationSeverity.MODERATE,
                timestamp=getattr(entity, 'current_timestep', 0),
                description=f"Leverage ratio {leverage_ratio:.2%} below minimum {parameters.minimum_leverage_ratio:.2%}",
                remediation_actions=[
                    "Reduce total assets",
                    "Deleverage positions",
                    "Raise equity capital"
                ]
            )
        
        return None
    
    def enforce(self, entity: Any, violation: PolicyViolation) -> Dict[str, Any]:
        return {'asset_growth_restricted': True}


class LiquidityRule(PolicyRule):
    """Check liquidity requirements."""
    
    def __init__(self):
        super().__init__(
            "liquidity_buffer",
            "Minimum liquidity buffer requirement"
        )
    
    def check(self, entity: Any, parameters: PolicyParameters) -> Optional[PolicyViolation]:
        if not hasattr(entity, 'balance_sheet'):
            return None
        
        bs = entity.balance_sheet
        liquidity_ratio = bs.cash / max(bs.total_liabilities, 1e-8)
        
        if liquidity_ratio < parameters.liquidity_buffer_ratio:
            severity = ViolationSeverity.SEVERE if liquidity_ratio < 0.02 else ViolationSeverity.MODERATE
            
            return PolicyViolation(
                violation_id=f"liq_{entity.bank_id}_{datetime.now().isoformat()}",
                entity_id=entity.bank_id,
                policy_name=self.name,
                parameter_name="liquidity_buffer_ratio",
                required_value=parameters.liquidity_buffer_ratio,
                actual_value=liquidity_ratio,
                severity=severity,
                timestamp=getattr(entity, 'current_timestep', 0),
                description=f"Liquidity ratio {liquidity_ratio:.2%} below minimum {parameters.liquidity_buffer_ratio:.2%}",
                remediation_actions=[
                    "Increase cash holdings",
                    "Reduce short-term obligations",
                    "Access central bank facilities"
                ]
            )
        
        return None
    
    def enforce(self, entity: Any, violation: PolicyViolation) -> Dict[str, Any]:
        return {'lending_restricted': True, 'liquidity_monitoring': True}


class ExposureConcentrationRule(PolicyRule):
    """Check exposure concentration limits."""
    
    def __init__(self):
        super().__init__(
            "exposure_concentration",
            "Large exposure concentration limits"
        )
    
    def check(self, entity: Any, parameters: PolicyParameters) -> Optional[PolicyViolation]:
        if not hasattr(entity, 'exposures') or not entity.exposures:
            return None
        
        bs = entity.balance_sheet
        capital = bs.equity
        
        for counterparty_id, exposure in entity.exposures.items():
            exposure_ratio = exposure / max(capital, 1e-8)
            
            if exposure_ratio > parameters.large_exposure_limit:
                return PolicyViolation(
                    violation_id=f"exp_{entity.bank_id}_{counterparty_id}_{datetime.now().isoformat()}",
                    entity_id=entity.bank_id,
                    policy_name=self.name,
                    parameter_name="large_exposure_limit",
                    required_value=parameters.large_exposure_limit,
                    actual_value=exposure_ratio,
                    severity=ViolationSeverity.MODERATE,
                    timestamp=getattr(entity, 'current_timestep', 0),
                    description=f"Exposure to counterparty {counterparty_id} is {exposure_ratio:.1%} of capital (limit: {parameters.large_exposure_limit:.0%})",
                    remediation_actions=[
                        f"Reduce exposure to counterparty {counterparty_id}",
                        "Diversify counterparty portfolio",
                        "Obtain additional capital"
                    ]
                )
        
        return None
    
    def enforce(self, entity: Any, violation: PolicyViolation) -> Dict[str, Any]:
        return {'new_exposure_blocked': True, 'exposure_reduction_required': True}


class RegulatoryPolicyLayer:
    """
    Main regulatory policy layer.
    Manages policy parameters and enforcement.
    """
    
    def __init__(self, regime: PolicyRegime = PolicyRegime.BASEL_III):
        self.regime = regime
        self.parameters = PolicyParameters.from_regime(regime)
        
        # Initialize rules
        self.rules: List[PolicyRule] = [
            CapitalRatioRule(),
            LeverageRatioRule(),
            LiquidityRule(),
            ExposureConcentrationRule()
        ]
        
        # Violation tracking
        self.violations: List[PolicyViolation] = []
        self.violation_history: Dict[int, List[PolicyViolation]] = {}  # entity_id -> violations
        
        # Enforcement state
        self.entity_restrictions: Dict[int, Dict[str, Any]] = {}
        
        # Callbacks for enforcement
        self.enforcement_callbacks: List[Callable[[int, Dict[str, Any]], None]] = []
    
    def set_regime(self, regime: PolicyRegime) -> None:
        """Change regulatory regime."""
        self.regime = regime
        self.parameters = PolicyParameters.from_regime(regime)
    
    def update_parameter(self, param_name: str, value: float) -> None:
        """Update a specific policy parameter."""
        if hasattr(self.parameters, param_name):
            setattr(self.parameters, param_name, value)
            self.regime = PolicyRegime.CUSTOM
    
    def add_rule(self, rule: PolicyRule) -> None:
        """Add a custom policy rule."""
        self.rules.append(rule)
    
    def check_compliance(self, entity: Any) -> List[PolicyViolation]:
        """Check entity compliance with all enabled rules."""
        violations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            violation = rule.check(entity, self.parameters)
            if violation:
                violations.append(violation)
                self._record_violation(violation)
        
        return violations
    
    def check_all_entities(self, entities: Dict[int, Any]) -> Dict[int, List[PolicyViolation]]:
        """Check compliance for all entities."""
        results = {}
        for entity_id, entity in entities.items():
            results[entity_id] = self.check_compliance(entity)
        return results
    
    def _record_violation(self, violation: PolicyViolation) -> None:
        """Record a violation."""
        self.violations.append(violation)
        
        if violation.entity_id not in self.violation_history:
            self.violation_history[violation.entity_id] = []
        self.violation_history[violation.entity_id].append(violation)
    
    def enforce_violations(self, violations: List[PolicyViolation]) -> Dict[int, Dict[str, Any]]:
        """Enforce remedies for all violations."""
        enforcement_actions = {}
        
        for violation in violations:
            entity_id = violation.entity_id
            
            # Find matching rule
            for rule in self.rules:
                if rule.name == violation.policy_name:
                    actions = rule.enforce(None, violation)  # Entity not needed for actions
                    
                    if entity_id not in enforcement_actions:
                        enforcement_actions[entity_id] = {}
                    enforcement_actions[entity_id].update(actions)
                    break
        
        # Update entity restrictions
        for entity_id, actions in enforcement_actions.items():
            if entity_id not in self.entity_restrictions:
                self.entity_restrictions[entity_id] = {}
            self.entity_restrictions[entity_id].update(actions)
        
        # Trigger callbacks
        for entity_id, actions in enforcement_actions.items():
            for callback in self.enforcement_callbacks:
                callback(entity_id, actions)
        
        return enforcement_actions
    
    def is_action_allowed(self, entity_id: int, action_type: str) -> bool:
        """Check if an action is allowed given current restrictions."""
        if entity_id not in self.entity_restrictions:
            return True
        
        restrictions = self.entity_restrictions[entity_id]
        
        # Map action types to restrictions
        action_restriction_map = {
            'lend': ['all_lending_suspended', 'new_lending_suspended', 'lending_restricted'],
            'borrow': [],
            'trade': ['trading_suspended'],
            'dividend': ['dividend_blocked', 'dividend_restricted'],
            'increase_exposure': ['new_exposure_blocked', 'asset_growth_restricted']
        }
        
        blocked_by = action_restriction_map.get(action_type, [])
        
        for restriction in blocked_by:
            if restrictions.get(restriction, False):
                return False
        
        return True
    
    def get_entity_restrictions(self, entity_id: int) -> Dict[str, Any]:
        """Get all restrictions for an entity."""
        return self.entity_restrictions.get(entity_id, {})
    
    def clear_restrictions(self, entity_id: int) -> None:
        """Clear all restrictions for an entity."""
        if entity_id in self.entity_restrictions:
            del self.entity_restrictions[entity_id]
    
    def register_enforcement_callback(self, 
                                       callback: Callable[[int, Dict[str, Any]], None]) -> None:
        """Register callback for enforcement actions."""
        self.enforcement_callbacks.append(callback)
    
    def compute_capital_requirement(self, entity: Any) -> float:
        """Compute total capital requirement for an entity."""
        base_requirement = self.parameters.total_capital_requirement
        
        # Add systemic buffer for large entities
        systemic_addon = 0
        if hasattr(entity, 'balance_sheet'):
            if entity.balance_sheet.total_assets > 1e10:
                systemic_addon = 0.01  # 1% G-SIB buffer
            if entity.balance_sheet.total_assets > 5e10:
                systemic_addon = 0.02
        
        # Add stress addon
        stress_addon = self.parameters.stress_capital_addon
        
        return base_requirement + systemic_addon + stress_addon
    
    def compute_penalty_reward_adjustment(self, 
                                          entity_id: int,
                                          base_reward: float) -> float:
        """
        Compute reward adjustment based on policy violations.
        For integration with MARL training.
        """
        if entity_id not in self.violation_history:
            return base_reward
        
        recent_violations = [v for v in self.violation_history[entity_id]
                            if not v.resolved]
        
        penalty = 0
        for violation in recent_violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                penalty += 50
            elif violation.severity == ViolationSeverity.SEVERE:
                penalty += 20
            elif violation.severity == ViolationSeverity.MODERATE:
                penalty += 5
            else:
                penalty += 1
        
        return base_reward - penalty
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of overall compliance status."""
        total_violations = len(self.violations)
        active_violations = [v for v in self.violations if not v.resolved]
        
        severity_counts = {s.value: 0 for s in ViolationSeverity}
        for v in active_violations:
            severity_counts[v.severity.value] += 1
        
        rule_counts = {r.name: 0 for r in self.rules}
        for v in active_violations:
            if v.policy_name in rule_counts:
                rule_counts[v.policy_name] += 1
        
        entities_with_restrictions = len(self.entity_restrictions)
        
        return {
            'regime': self.regime.value,
            'total_capital_requirement': self.parameters.total_capital_requirement,
            'total_violations': total_violations,
            'active_violations': len(active_violations),
            'severity_distribution': severity_counts,
            'violations_by_rule': rule_counts,
            'entities_restricted': entities_with_restrictions,
            'critical_count': severity_counts.get('critical', 0),
            'system_compliant': severity_counts.get('critical', 0) == 0 and 
                               severity_counts.get('severe', 0) == 0
        }
    
    def export_violations(self, filepath: str) -> None:
        """Export all violations to JSON."""
        data = {
            'regime': self.regime.value,
            'parameters': self.parameters.to_dict(),
            'violations': [v.to_dict() for v in self.violations],
            'summary': self.get_compliance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def integrate_with_environment(self, env: Any) -> None:
        """
        Integrate policy layer with environment dynamics.
        Called to inject policy checks into simulation.
        """
        # Store reference for callbacks
        self._env = env
        
        # Example enforcement callback
        def apply_restrictions(entity_id: int, actions: Dict[str, Any]) -> None:
            if hasattr(env, 'banks') and entity_id in env.banks:
                bank = env.banks[entity_id]
                
                if actions.get('all_lending_suspended'):
                    if hasattr(bank, 'lending_allowed'):
                        bank.lending_allowed = False
                
                if actions.get('trading_suspended'):
                    if hasattr(bank, 'trading_allowed'):
                        bank.trading_allowed = False
        
        self.register_enforcement_callback(apply_restrictions)
