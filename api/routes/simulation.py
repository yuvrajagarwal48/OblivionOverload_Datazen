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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.bank_registry import bank_registry

router = APIRouter(prefix="/simulation", tags=["Simulation"])


# ============================================================================
# SIMULATION CONTROL
# ============================================================================

@router.post("/init")
async def init_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """
    Initialize a new simulation with the given configuration.
    
    This sets up:
    - Financial environment with specified banks (real or random)
    - Exchanges and CCPs (infrastructure layer)
    - Baseline agents for each bank
    - Analytics engines
    - State capture system
    """
    try:
        real_bank_configs = None
        bank_info = []
        
        # Resolve real banks if requested
        if config.bank_ids or config.bank_names:
            real_banks = []
            
            if config.bank_ids:
                real_banks = bank_registry.get_banks_by_ids(config.bank_ids)
                if len(real_banks) != len(config.bank_ids):
                    found_ids = {b.bank_id for b in real_banks}
                    missing = [bid for bid in config.bank_ids if bid not in found_ids]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bank IDs not found in registry: {missing}"
                    )
            elif config.bank_names:
                real_banks = bank_registry.get_banks_by_names(config.bank_names)
                if len(real_banks) != len(config.bank_names):
                    found_names = {b.name for b in real_banks}
                    missing = [n for n in config.bank_names if n not in found_names]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bank names not found in registry: {missing}"
                    )
            
            # Add synthetic banks if requested
            if config.synthetic_count and config.synthetic_count > 0:
                synthetic = bank_registry.generate_synthetic_banks(
                    count=config.synthetic_count,
                    start_id=100,
                    stress_level=config.synthetic_stress or "normal",
                    seed=config.seed
                )
                real_banks.extend(synthetic)
            
            # Convert to config dicts
            real_bank_configs = [b.to_dict() for b in real_banks]
            bank_info = [{"id": i, "name": b.name, "tier": b.tier} 
                        for i, b in enumerate(real_banks)]
        
        simulation_state.initialize(
            num_banks=config.num_banks,
            episode_length=config.episode_length,
            scenario=config.scenario,
            num_exchanges=config.num_exchanges,
            num_ccps=config.num_ccps,
            seed=config.seed,
            real_bank_configs=real_bank_configs
        )
        
        actual_num_banks = len(real_bank_configs) if real_bank_configs else config.num_banks
        
        return {
            "status": "initialized",
            "config": {
                "num_banks": actual_num_banks,
                "episode_length": config.episode_length,
                "scenario": config.scenario,
                "num_exchanges": config.num_exchanges,
                "num_ccps": config.num_ccps,
                "use_real_banks": real_bank_configs is not None
            },
            "banks": bank_info if bank_info else None,
            "message": "Simulation initialized successfully" + 
                      (f" with {len(bank_info)} real banks" if bank_info else " with random banks")
        }
    except HTTPException:
        raise
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
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        return convert_numpy({
            "steps_completed": 1,
            "current_step": result["step"],
            "is_done": result["done"],
            "rewards": result["rewards"],
            "network_stats": result["network_stats"],
            "market_state": result["market_state"],
            "infrastructure": result.get("infrastructure", {})
        })
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
            "healthy_banks": int(network_stats.num_banks - network_stats.num_stressed - network_stats.num_defaulted),
            "stressed_banks": int(network_stats.num_stressed),
            "defaulted_banks": int(network_stats.num_defaulted),
            "market_price": float(market_state.asset_price),
            "avg_capital_ratio": float(network_stats.avg_capital_ratio),
            "total_exposure": float(network_stats.total_exposure)
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
    
    # Helper to convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Bank summaries
    banks = {}
    for bank_id, bank in network.banks.items():
        banks[bank_id] = {
            "bank_id": bank_id,
            "tier": bank.tier,
            "status": bank.status.value,
            "equity": float(bank.balance_sheet.equity),
            "capital_ratio": float(bank.capital_ratio),
            "cash": float(bank.balance_sheet.cash),
            "total_assets": float(bank.balance_sheet.total_assets),
            "total_liabilities": float(bank.balance_sheet.total_liabilities)
        }
        if hasattr(bank, 'name') and bank.name:
            banks[bank_id]["name"] = bank.name
    
    # Exchange status
    exchanges = []
    for ex in simulation_state.exchanges:
        ex_state = ex.get_state()
        exchanges.append({
            "exchange_id": ex.exchange_id,
            "congestion_level": float(ex_state.congestion_level),
            "current_volume": float(ex_state.current_volume),
            "status": ex_state.status.value,
            "backlog_size": int(ex_state.backlog_size)
        })
    
    # CCP status
    ccps = []
    for ccp in simulation_state.ccps:
        ccp_state = ccp.get_state()
        ccps.append({
            "ccp_id": ccp.ccp_id,
            "status": ccp_state.status.value,
            "total_margin": float(ccp_state.total_initial_margin + ccp_state.total_variation_margin),
            "default_fund": float(ccp_state.default_fund_size),
            "stress_level": float(ccp_state.stress_level)
        })
    
    return {
        "timestep": int(env.current_step),
        "market": convert_numpy(market_state.to_dict()),
        "network_stats": convert_numpy(network_stats.to_dict()),
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


@router.get("/history/graph-animation")
async def get_graph_animation_data(
    start: int = Query(default=0, ge=0, description="Start timestep"),
    end: Optional[int] = Query(default=None, description="End timestep (inclusive)"),
    sample_rate: int = Query(default=1, ge=1, description="Sample every Nth step")
) -> Dict[str, Any]:
    """
    Get network graph state at each timestep for frontend animation.
    
    Returns nodes and edges for each timestep, allowing the frontend to:
    - Animate network evolution over time
    - Show defaults propagating through the system
    - Visualize edge weight changes
    - Display node stress levels changing
    
    Example usage:
        GET /api/simulation/history/graph-animation?start=0&end=50&sample_rate=5
        Returns graph data for steps 0, 5, 10, 15, ..., 50
    
    Args:
        start: Starting timestep
        end: Ending timestep (None = all available steps)
        sample_rate: Sample every Nth step (1 = every step, 2 = every other step)
    
    Returns:
        {
            "timesteps": [
                {
                    "step": 0,
                    "nodes": [...],  # Bank states with vis.js properties
                    "edges": [...],  # Interbank exposures with visual properties
                    "metrics": {...} # System-wide metrics
                }
            ],
            "metadata": {...},
            "animation_config": {...}  # Recommended animation settings
        }
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if not simulation_state.history:
        raise HTTPException(status_code=400, detail="No history data available. Run simulation first.")
    
    # Determine end timestep
    max_step = simulation_state.current_step
    if end is None or end > max_step:
        end = max_step
    
    # Validate range
    if start > end:
        raise HTTPException(status_code=400, detail="Start must be <= end")
    
    # Get state capture data
    state_capture = simulation_state.history.state_capture
    
    timesteps_data = []
    sampled_steps = 0
    
    for step in range(start, end + 1, sample_rate):
        # Get bank snapshots for this timestep
        bank_snapshots = {}
        for bank_id, snapshots in state_capture.bank_snapshots.items():
            # Find snapshot for this timestep
            for snapshot in snapshots:
                if snapshot.timestep == step:
                    bank_snapshots[bank_id] = snapshot
                    break
        
        if not bank_snapshots:
            continue  # Skip if no data for this timestep
        
        # Build nodes
        nodes = []
        for bank_id, snapshot in bank_snapshots.items():
            # Status to color mapping
            color_map = {
                "healthy": "#4CAF50",    # Green
                "stressed": "#FF9800",   # Orange
                "critical": "#FF5722",   # Deep orange
                "defaulted": "#F44336"   # Red
            }
            
            # Size based on total assets (log scale)
            size = max(5, min(50, snapshot.viz_size * 10))
            
            nodes.append({
                "id": bank_id,
                "label": f"Bank {bank_id}",
                
                # Status
                "status": snapshot.solvency_status,
                "tier": snapshot.tier,
                
                # Financial metrics (for tooltips)
                "equity": round(snapshot.equity, 2),
                "capital_ratio": round(snapshot.capital_ratio, 4),
                "liquidity_ratio": round(snapshot.liquidity_ratio, 4),
                "cash": round(snapshot.cash_position, 2),
                "total_assets": round(snapshot.total_assets, 2),
                "total_liabilities": round(snapshot.total_liabilities, 2),
                
                # Lending activity
                "outstanding_lending": round(snapshot.outstanding_lending, 2),
                "outstanding_borrowing": round(snapshot.outstanding_borrowing, 2),
                "net_position": round(snapshot.net_interbank_position, 2),
                
                # Risk metrics
                "stress_level": round(snapshot.stress_level, 3),
                "exposure": round(snapshot.total_exposure, 2),
                "exposure_to_defaults": round(snapshot.exposure_to_defaults, 2),
                "concentration": round(snapshot.concentration_index, 3),
                
                # Network position
                "degree_centrality": round(snapshot.degree_centrality, 4),
                "betweenness_centrality": round(snapshot.betweenness_centrality, 4),
                
                # Visualization properties
                "size": size,
                "color": color_map.get(snapshot.solvency_status, "#9E9E9E"),
                "color_intensity": snapshot.viz_color_intensity,
                "border_width": 3 if snapshot.solvency_status == "defaulted" else (2 if snapshot.solvency_status == "stressed" else 1),
                "border_color": "#000000" if snapshot.solvency_status == "defaulted" else "#666666",
                
                # Animation properties
                "opacity": 0.3 if snapshot.solvency_status == "defaulted" else 1.0,
                "pulse": snapshot.solvency_status == "stressed"  # Frontend can animate pulsing
            })
        
        # Build edges
        edges = []
        
        # If this is the current step, we can get actual edges
        if step == simulation_state.current_step and simulation_state.env:
            network = simulation_state.env.network
            for u, v in network.graph.edges():
                exposure = network.liability_matrix[u, v]
                if exposure > 0:
                    # Get debtor and creditor status
                    debtor_snapshot = bank_snapshots.get(u)
                    creditor_snapshot = bank_snapshots.get(v)
                    
                    # Color edge based on risk
                    if debtor_snapshot and debtor_snapshot.solvency_status == "defaulted":
                        edge_color = "#F44336"  # Red - defaulted debtor
                    elif debtor_snapshot and debtor_snapshot.solvency_status == "stressed":
                        edge_color = "#FF9800"  # Orange - stressed debtor
                    else:
                        edge_color = "#2196F3"  # Blue - healthy
                    
                    edges.append({
                        "from": u,
                        "to": v,
                        "value": round(exposure, 2),
                        "label": f"${exposure:,.0f}",
                        "width": max(1, min(exposure / 100000, 10)),
                        "color": edge_color,
                        "arrows": "to",
                        "dashes": debtor_snapshot.solvency_status == "defaulted" if debtor_snapshot else False
                    })
        
        # Get system metrics for this timestep
        step_history = next((h for h in simulation_state.step_history if h.timestep == step), None)
        
        metrics = {}
        if step_history:
            metrics = {
                "total_defaults": step_history.total_defaults,
                "num_stressed": step_history.num_stressed,
                "num_active": simulation_state.num_banks - step_history.total_defaults,
                "avg_capital_ratio": round(step_history.avg_capital_ratio, 4),
                "market_price": round(step_history.market_price, 4),
                "volatility": round(step_history.volatility, 4),
                "liquidity_index": round(step_history.liquidity_index, 4),
                "total_exposure": round(step_history.total_exposure, 2)
            }
        
        timesteps_data.append({
            "step": step,
            "nodes": nodes,
            "edges": edges,
            "metrics": metrics,
            "timestamp": bank_snapshots[list(bank_snapshots.keys())[0]].timestamp if bank_snapshots else None
        })
        
        sampled_steps += 1
    
    return {
        "timesteps": timesteps_data,
        "metadata": {
            "total_steps": max_step + 1,
            "sampled_steps": sampled_steps,
            "sample_rate": sample_rate,
            "start": start,
            "end": end,
            "num_banks": simulation_state.num_banks
        },
        "animation_config": {
            "recommended_fps": 2 if sample_rate == 1 else 5,  # Frames per second
            "transition_duration_ms": 500,  # Smooth transitions
            "layout": "force-directed"  # Recommended layout algorithm
        }
    }
