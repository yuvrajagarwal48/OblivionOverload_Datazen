"""
Simulation API Routes.
Endpoints for simulation control, status, and history.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np

from .state import simulation_state
from .models import (
    SimulationConfig, StepConfig, ActionRequest, ShockRequest,
    SimulationStatusResponse, StepResult, SimulationStatus,
    SystemMetrics, NetworkStats, MarketState, ClearingResults,
    TimeSeriesData, MultiSeriesData
)

router = APIRouter(prefix="/simulation", tags=["Simulation"])


# ============================================================================
# SIMULATION CONTROL
# ============================================================================

@router.post("/init")
async def init_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """
    Initialize a new simulation with the given configuration.
    
    This sets up:
    - Financial environment with specified banks
    - Exchanges and CCPs (infrastructure layer)
    - Baseline agents for each bank
    - Analytics engines
    - State capture system
    """
    try:
        simulation_state.initialize(
            num_banks=config.num_banks,
            episode_length=config.episode_length,
            scenario=config.scenario,
            num_exchanges=config.num_exchanges,
            num_ccps=config.num_ccps,
            seed=config.seed
        )
        
        return {
            "status": "initialized",
            "config": {
                "num_banks": config.num_banks,
                "episode_length": config.episode_length,
                "scenario": config.scenario,
                "num_exchanges": config.num_exchanges,
                "num_ccps": config.num_ccps
            },
            "message": "Simulation initialized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_simulation() -> Dict[str, Any]:
    """Reset the current simulation to initial state."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    simulation_state.reset()
    
    return {
        "status": "reset",
        "current_step": 0,
        "message": "Simulation reset successfully"
    }


@router.post("/step")
async def step_simulation(
    config: Optional[StepConfig] = None,
    actions: Optional[ActionRequest] = None
) -> Dict[str, Any]:
    """
    Execute simulation step(s).
    
    - If actions provided, use those for each bank
    - Otherwise, agents select actions based on their policies
    
    Returns complete state after step(s).
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    num_steps = config.num_steps if config else 1
    capture = config.capture_state if config else True
    
    # Parse actions if provided
    action_dict = None
    if actions and actions.actions:
        action_dict = {int(k): np.array(v) for k, v in actions.actions.items()}
    
    if num_steps == 1:
        result = simulation_state.step(actions=action_dict, capture_state=capture)
        
        return {
            "steps_completed": 1,
            "current_step": result["step"],
            "is_done": result["done"],
            "rewards": result["rewards"],
            "network_stats": result["network_stats"],
            "market_state": result["market_state"],
            "infrastructure": result.get("infrastructure", {})
        }
    else:
        result = simulation_state.run_steps(num_steps)
        return result


@router.post("/run")
async def run_simulation(
    num_steps: int = Query(default=100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Run simulation for specified number of steps."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    result = simulation_state.run_steps(num_steps)
    return result


@router.post("/shock")
async def apply_shock(request: ShockRequest) -> Dict[str, Any]:
    """Apply an external shock to the system."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    bank_shocks = None
    if request.bank_shocks:
        bank_shocks = {int(k): v for k, v in request.bank_shocks.items()}
    
    simulation_state.env.apply_scenario_shock(
        price_shock=request.price_shock,
        volatility_shock=request.volatility_shock,
        liquidity_shock=request.liquidity_shock,
        bank_shocks=bank_shocks
    )
    
    return {
        "status": "shock_applied",
        "market_state": simulation_state.env.market.get_state().to_dict()
    }


# ============================================================================
# SIMULATION STATUS
# ============================================================================

@router.get("/status")
async def get_simulation_status() -> Dict[str, Any]:
    """Get current simulation status and summary."""
    if not simulation_state.is_initialized():
        return {
            "status": "not_initialized",
            "current_step": 0,
            "total_steps": 0,
            "is_done": False,
            "num_banks": 0,
            "num_exchanges": 0,
            "num_ccps": 0,
            "scenario": None
        }
    
    env = simulation_state.env
    network_stats = env.network.get_network_stats()
    market_state = env.market.get_state()
    
    return {
        "status": "running" if not simulation_state.is_done() else "completed",
        "current_step": simulation_state.get_current_step(),
        "total_steps": simulation_state.get_total_steps(),
        "is_done": simulation_state.is_done(),
        "num_banks": env.num_agents,
        "num_exchanges": len(simulation_state.exchanges),
        "num_ccps": len(simulation_state.ccps),
        "scenario": simulation_state.config.get("scenario", "normal"),
        "started_at": simulation_state.started_at,
        
        # Quick summary metrics
        "summary": {
            "healthy_banks": network_stats.num_healthy,
            "stressed_banks": network_stats.num_stressed,
            "defaulted_banks": network_stats.num_defaulted,
            "market_price": market_state.asset_price,
            "avg_capital_ratio": network_stats.avg_capital_ratio,
            "total_exposure": network_stats.total_interbank_exposure
        }
    }


@router.get("/state")
async def get_current_state() -> Dict[str, Any]:
    """Get complete current state snapshot."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    env = simulation_state.env
    network = env.network
    market_state = env.market.get_state()
    network_stats = network.get_network_stats()
    
    # Bank summaries
    banks = {}
    for bank_id, bank in network.banks.items():
        banks[bank_id] = {
            "bank_id": bank_id,
            "tier": bank.tier,
            "status": bank.status.value,
            "equity": bank.balance_sheet.equity,
            "capital_ratio": bank.capital_ratio,
            "cash": bank.balance_sheet.cash,
            "total_assets": bank.balance_sheet.total_assets,
            "total_liabilities": bank.balance_sheet.total_liabilities
        }
    
    # Exchange status
    exchanges = []
    for ex in simulation_state.exchanges:
        ex_state = ex.get_state()
        exchanges.append({
            "exchange_id": ex.exchange_id,
            "congestion_level": ex_state.congestion_level,
            "transaction_volume": ex_state.transaction_volume,
            "fees_collected": ex_state.total_fees_collected
        })
    
    # CCP status
    ccps = []
    for ccp in simulation_state.ccps:
        ccp_state = ccp.get_state()
        ccps.append({
            "ccp_id": ccp.ccp_id,
            "status": ccp_state.status.value,
            "total_margin": ccp_state.total_initial_margin + ccp_state.total_variation_margin,
            "default_fund": ccp_state.default_fund_size,
            "stress_level": ccp_state.stress_level
        })
    
    return {
        "timestep": env.current_step,
        "market": market_state.to_dict(),
        "network_stats": network_stats.to_dict(),
        "banks": banks,
        "exchanges": exchanges,
        "ccps": ccps
    }


# ============================================================================
# SIMULATION HISTORY
# ============================================================================

@router.get("/history")
async def get_simulation_history(
    start: int = Query(default=0, ge=0),
    end: Optional[int] = Query(default=None),
    fields: Optional[str] = Query(default=None, description="Comma-separated field names")
) -> Dict[str, Any]:
    """
    Get simulation history as time series data.
    
    Returns data suitable for charting: market price, defaults, stress levels, etc.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    field_list = fields.split(",") if fields else None
    history = simulation_state.get_history(start, end, field_list)
    
    return {
        "start_step": start,
        "end_step": end or len(history),
        "count": len(history),
        "data": history
    }


@router.get("/history/timeseries")
async def get_timeseries_data(
    metrics: str = Query(
        default="market_price,default_rate,avg_capital_ratio,total_exposure",
        description="Comma-separated metric names"
    )
) -> Dict[str, Any]:
    """
    Get time series data formatted for charts.
    
    Returns arrays of timestamps and values for each requested metric.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    metric_list = [m.strip() for m in metrics.split(",")]
    history = simulation_state.step_history
    
    series = {}
    for metric in metric_list:
        series[metric] = {
            "timestamps": [],
            "values": []
        }
        
        for h in history:
            series[metric]["timestamps"].append(h.timestep)
            value = getattr(h, metric, None)
            if value is not None:
                series[metric]["values"].append(float(value))
            else:
                series[metric]["values"].append(0.0)
    
    return {
        "metrics": metric_list,
        "series": series,
        "count": len(history)
    }


@router.get("/history/graphs")
async def get_graph_data() -> Dict[str, Any]:
    """
    Get pre-formatted data for frontend graphs.
    
    Returns multiple time series optimized for different chart types.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    history = simulation_state.step_history
    
    # Extract time series
    timesteps = [h.timestep for h in history]
    
    return {
        "timesteps": timesteps,
        
        "price_chart": {
            "name": "Asset Price",
            "type": "line",
            "data": [h.market_price for h in history],
            "color": "#2196F3"
        },
        
        "defaults_chart": {
            "name": "Cumulative Defaults",
            "type": "area",
            "data": [h.total_defaults for h in history],
            "color": "#F44336"
        },
        
        "stress_chart": {
            "name": "Stressed Banks",
            "type": "bar",
            "data": [h.num_stressed for h in history],
            "color": "#FF9800"
        },
        
        "capital_chart": {
            "name": "Avg Capital Ratio",
            "type": "line",
            "data": [h.avg_capital_ratio for h in history],
            "color": "#4CAF50",
            "threshold": 0.08  # Regulatory minimum
        },
        
        "exposure_chart": {
            "name": "Total Exposure",
            "type": "area",
            "data": [h.total_exposure for h in history],
            "color": "#9C27B0"
        },
        
        "liquidity_chart": {
            "name": "Liquidity Index",
            "type": "line",
            "data": [h.liquidity_index for h in history],
            "color": "#00BCD4"
        },
        
        "volatility_chart": {
            "name": "Market Volatility",
            "type": "line",
            "data": [h.volatility for h in history],
            "color": "#795548"
        }
    }


@router.get("/history/clearing")
async def get_clearing_history() -> Dict[str, Any]:
    """Get clearing mechanism results over time."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    history = simulation_state.step_history
    
    clearing_data = []
    for h in history:
        clearing_data.append({
            "timestep": h.timestep,
            "converged": h.clearing_converged,
            "total_shortfall": h.total_shortfall,
            "defaults_this_step": h.defaults_this_step,
            "recovery_rates": h.recovery_rates
        })
    
    return {
        "count": len(clearing_data),
        "data": clearing_data
    }


@router.get("/history/events")
async def get_simulation_events(
    event_type: Optional[str] = Query(default=None, description="Filter by event type")
) -> Dict[str, Any]:
    """Get simulation events (defaults, margin calls, interventions)."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    all_events = []
    for h in simulation_state.step_history:
        for event in h.events:
            if event_type is None or event.get("type") == event_type:
                all_events.append({
                    "timestep": h.timestep,
                    **event
                })
    
    return {
        "count": len(all_events),
        "events": all_events
    }
