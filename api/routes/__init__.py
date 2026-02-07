"""
API Routes Package for FinSim-MAPPO.
"""

from .simulation import router as simulation_router
from .banks import router as banks_router
from .analytics import router as analytics_router
from .market import router as market_router
from .infrastructure import router as infrastructure_router
from .whatif import router as whatif_router

__all__ = [
    'simulation_router',
    'banks_router', 
    'analytics_router',
    'market_router',
    'infrastructure_router',
    'whatif_router'
]
