"""
Core module for FinSim-MAPPO.
Contains state capture, history, and explanation components.
"""

from .state_capture import (
    BankSnapshot,
    ExchangeSnapshot,
    CCPSnapshot,
    MarketSnapshot,
    FlowRecord,
    StateCapture
)

from .history import (
    SimulationHistory,
    TimestepRecord,
    ClearingOutcome,
    MarginCallEvent,
    DefaultEvent
)

from .explanation import (
    ExplanationLayer,
    EventExplanation,
    CausalFactor,
    ContagionPath,
    AttributionResult,
    ContagionTracker,
    CausalityAnalyzer,
    EventType,
    CauseType
)

__all__ = [
    # State Capture
    'BankSnapshot',
    'ExchangeSnapshot',
    'CCPSnapshot',
    'MarketSnapshot',
    'FlowRecord',
    'StateCapture',
    
    # History
    'SimulationHistory',
    'TimestepRecord',
    'ClearingOutcome',
    'MarginCallEvent',
    'DefaultEvent',
    
    # Explanation
    'ExplanationLayer',
    'EventExplanation',
    'CausalFactor',
    'ContagionPath',
    'AttributionResult',
    'ContagionTracker',
    'CausalityAnalyzer',
    'EventType',
    'CauseType'
]
