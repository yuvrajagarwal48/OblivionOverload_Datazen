"""
Scenario Engine: Predefined scenarios and shock generation.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class ScenarioType(Enum):
    """Types of predefined scenarios."""
    NORMAL = "normal"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    ASSET_CRASH = "asset_crash"
    SYSTEMIC = "systemic"
    CUSTOM = "custom"


@dataclass
class ScenarioParams:
    """Parameters for a scenario."""
    shock_probability: float = 0.05
    shock_magnitude: float = 0.05
    liquidity_stress: float = 0.0
    price_volatility: float = 0.02
    
    # Bank-specific shocks
    targeted_banks: List[int] = field(default_factory=list)
    bank_shock_magnitude: float = 0.0
    
    # Time-varying parameters
    shock_timing: str = "random"  # "random", "early", "mid", "late", "continuous"
    shock_duration: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'shock_probability': self.shock_probability,
            'shock_magnitude': self.shock_magnitude,
            'liquidity_stress': self.liquidity_stress,
            'price_volatility': self.price_volatility,
            'targeted_banks': self.targeted_banks,
            'bank_shock_magnitude': self.bank_shock_magnitude,
            'shock_timing': self.shock_timing,
            'shock_duration': self.shock_duration
        }


@dataclass
class Scenario:
    """A complete scenario definition."""
    name: str
    scenario_type: ScenarioType
    params: ScenarioParams
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'description': self.description,
            'params': self.params.to_dict()
        }


# Predefined scenarios
PREDEFINED_SCENARIOS: Dict[str, Scenario] = {
    'normal': Scenario(
        name="Normal Market",
        scenario_type=ScenarioType.NORMAL,
        params=ScenarioParams(
            shock_probability=0.05,
            shock_magnitude=0.05,
            liquidity_stress=0.0,
            price_volatility=0.02
        ),
        description="Stable market conditions with minimal shocks"
    ),
    'liquidity_crisis': Scenario(
        name="Liquidity Crisis",
        scenario_type=ScenarioType.LIQUIDITY_CRISIS,
        params=ScenarioParams(
            shock_probability=0.3,
            shock_magnitude=0.2,
            liquidity_stress=0.5,
            price_volatility=0.05,
            shock_timing="continuous"
        ),
        description="Severe funding stress with interbank market freeze"
    ),
    'asset_crash': Scenario(
        name="Asset Price Crash",
        scenario_type=ScenarioType.ASSET_CRASH,
        params=ScenarioParams(
            shock_probability=0.4,
            shock_magnitude=0.4,
            liquidity_stress=0.2,
            price_volatility=0.15,
            shock_timing="early"
        ),
        description="Sharp decline in asset prices triggering fire sales"
    ),
    'systemic': Scenario(
        name="Systemic Crisis",
        scenario_type=ScenarioType.SYSTEMIC,
        params=ScenarioParams(
            shock_probability=0.5,
            shock_magnitude=0.5,
            liquidity_stress=0.6,
            price_volatility=0.2,
            shock_timing="continuous",
            shock_duration=5
        ),
        description="Combined liquidity and asset price stress causing cascading failures"
    )
}


class ShockGenerator:
    """Generates shocks based on scenario parameters."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize shock generator.
        
        Args:
            seed: Random seed
        """
        self._rng = np.random.default_rng(seed)
    
    def generate_market_shock(self, 
                               params: ScenarioParams,
                               current_step: int,
                               episode_length: int) -> Dict[str, float]:
        """
        Generate a market shock based on scenario parameters.
        
        Args:
            params: Scenario parameters
            current_step: Current simulation step
            episode_length: Total episode length
            
        Returns:
            Dictionary of shock values
        """
        # Check if shock should occur based on timing
        should_shock = self._check_timing(params, current_step, episode_length)
        
        if not should_shock:
            return {'price_shock': 0.0, 'volatility_shock': 0.0, 'liquidity_shock': 0.0}
        
        # Check probability
        if self._rng.random() > params.shock_probability:
            return {'price_shock': 0.0, 'volatility_shock': 0.0, 'liquidity_shock': 0.0}
        
        # Generate shock values
        price_shock = -params.shock_magnitude * self._rng.random()
        volatility_shock = params.price_volatility * self._rng.random()
        liquidity_shock = params.liquidity_stress * self._rng.random()
        
        return {
            'price_shock': price_shock,
            'volatility_shock': volatility_shock,
            'liquidity_shock': liquidity_shock
        }
    
    def generate_bank_shocks(self,
                              params: ScenarioParams,
                              num_banks: int,
                              current_step: int,
                              episode_length: int) -> Dict[int, float]:
        """
        Generate bank-specific shocks.
        
        Args:
            params: Scenario parameters
            num_banks: Total number of banks
            current_step: Current simulation step
            episode_length: Total episode length
            
        Returns:
            Dictionary mapping bank_id to shock amount
        """
        bank_shocks = {}
        
        # Check timing
        if not self._check_timing(params, current_step, episode_length):
            return bank_shocks
        
        # Targeted banks
        if params.targeted_banks:
            for bank_id in params.targeted_banks:
                if bank_id < num_banks:
                    shock = params.bank_shock_magnitude * (0.5 + 0.5 * self._rng.random())
                    bank_shocks[bank_id] = shock
        
        return bank_shocks
    
    def _check_timing(self, 
                      params: ScenarioParams,
                      current_step: int,
                      episode_length: int) -> bool:
        """Check if shock should occur based on timing."""
        progress = current_step / episode_length
        
        if params.shock_timing == "random":
            return True
        elif params.shock_timing == "early":
            return progress < 0.3
        elif params.shock_timing == "mid":
            return 0.3 <= progress < 0.7
        elif params.shock_timing == "late":
            return progress >= 0.7
        elif params.shock_timing == "continuous":
            return True
        
        return True


class ScenarioEngine:
    """
    Manages scenario selection and shock application.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize scenario engine.
        
        Args:
            seed: Random seed
        """
        self.shock_generator = ShockGenerator(seed)
        self.scenarios = PREDEFINED_SCENARIOS.copy()
        self.current_scenario: Optional[Scenario] = None
        self._rng = np.random.default_rng(seed)
    
    def add_scenario(self, scenario: Scenario) -> None:
        """Add a custom scenario."""
        self.scenarios[scenario.name.lower().replace(' ', '_')] = scenario
    
    def get_scenario(self, name: str) -> Optional[Scenario]:
        """Get a scenario by name."""
        return self.scenarios.get(name.lower())
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios."""
        return list(self.scenarios.keys())
    
    def set_scenario(self, name: str) -> bool:
        """
        Set the current scenario.
        
        Args:
            name: Scenario name
            
        Returns:
            True if scenario was found and set
        """
        scenario = self.get_scenario(name)
        if scenario:
            self.current_scenario = scenario
            return True
        return False
    
    def generate_shocks(self,
                        current_step: int,
                        episode_length: int,
                        num_banks: int) -> Dict[str, Any]:
        """
        Generate all shocks for current step.
        
        Args:
            current_step: Current simulation step
            episode_length: Total episode length
            num_banks: Number of banks
            
        Returns:
            Dictionary with market and bank shocks
        """
        if self.current_scenario is None:
            return {'market': {}, 'banks': {}}
        
        params = self.current_scenario.params
        
        market_shock = self.shock_generator.generate_market_shock(
            params, current_step, episode_length
        )
        
        bank_shocks = self.shock_generator.generate_bank_shocks(
            params, num_banks, current_step, episode_length
        )
        
        return {
            'market': market_shock,
            'banks': bank_shocks
        }
    
    def create_stress_test_scenarios(self,
                                      base_scenario: str = 'normal',
                                      shock_levels: List[float] = None) -> List[Scenario]:
        """
        Create a series of stress test scenarios with varying severity.
        
        Args:
            base_scenario: Base scenario to modify
            shock_levels: List of shock magnitudes to test
            
        Returns:
            List of stress test scenarios
        """
        if shock_levels is None:
            shock_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        base = self.get_scenario(base_scenario)
        if base is None:
            base = PREDEFINED_SCENARIOS['normal']
        
        scenarios = []
        for level in shock_levels:
            params = ScenarioParams(
                shock_probability=base.params.shock_probability * (1 + level),
                shock_magnitude=level,
                liquidity_stress=level * 0.5,
                price_volatility=base.params.price_volatility * (1 + level)
            )
            
            scenario = Scenario(
                name=f"Stress Test Level {level:.0%}",
                scenario_type=ScenarioType.CUSTOM,
                params=params,
                description=f"Stress test with {level:.0%} shock magnitude"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def create_targeted_failure_scenario(self,
                                          target_banks: List[int],
                                          shock_magnitude: float = 0.5) -> Scenario:
        """
        Create a scenario targeting specific banks.
        
        Args:
            target_banks: List of bank IDs to target
            shock_magnitude: Shock magnitude for targeted banks
            
        Returns:
            Targeted failure scenario
        """
        params = ScenarioParams(
            shock_probability=0.8,
            shock_magnitude=0.2,
            liquidity_stress=0.3,
            price_volatility=0.1,
            targeted_banks=target_banks,
            bank_shock_magnitude=shock_magnitude,
            shock_timing="early"
        )
        
        return Scenario(
            name=f"Targeted Failure ({len(target_banks)} banks)",
            scenario_type=ScenarioType.CUSTOM,
            params=params,
            description=f"Scenario targeting banks: {target_banks}"
        )
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of current scenario."""
        if self.current_scenario is None:
            return {'active': False}
        
        return {
            'active': True,
            'name': self.current_scenario.name,
            'type': self.current_scenario.scenario_type.value,
            'description': self.current_scenario.description,
            'params': self.current_scenario.params.to_dict()
        }
