"""
Market API Routes.
Endpoints for market conditions, pricing, and historical data.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np

from .state import simulation_state

router = APIRouter(prefix="/market", tags=["Market"])


# ============================================================================
# CURRENT MARKET STATE
# ============================================================================

@router.get("/state")
async def get_market_state() -> Dict[str, Any]:
    """
    Get current market conditions.
    
    Includes asset price, volatility, liquidity, and market regime.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    market_state = simulation_state.env.market.get_state()
    
    # Calculate additional metrics from history
    history = simulation_state.step_history
    
    # Price changes
    current_price = market_state.asset_price
    if len(history) >= 2:
        prev_price = history[-2].market_price
        price_change = current_price - prev_price
        price_change_pct = price_change / max(prev_price, 0.01)
    else:
        price_change = 0
        price_change_pct = 0
    
    # Returns over different horizons
    returns = {}
    for horizon in [1, 5, 10]:
        if len(history) > horizon:
            old_price = history[-(horizon+1)].market_price
            returns[f"{horizon}_step"] = round((current_price - old_price) / max(old_price, 0.01), 4)
    
    return {
        # Core state
        "asset_price": round(current_price, 4),
        "price_change": round(price_change, 4),
        "price_change_pct": round(price_change_pct, 4),
        
        "interest_rate": round(market_state.interest_rate, 4),
        "volatility": round(market_state.volatility, 4),
        "liquidity_index": round(market_state.liquidity_index, 4),
        
        # Market quality
        "bid_ask_spread": round(getattr(market_state, 'bid_ask_spread', 0.001), 4),
        "market_depth": round(getattr(market_state, 'market_depth', 1.0), 4),
        
        # Condition
        "condition": market_state.condition.value if hasattr(market_state.condition, 'value') else "normal",
        "stress_indicator": _calculate_stress_indicator(market_state),
        
        # Returns
        "returns": returns,
        
        # Trading activity
        "volume": getattr(market_state, 'volume', 0),
        "sell_pressure": _estimate_sell_pressure(simulation_state.env),
        
        # Regime
        "regime": _determine_regime(market_state)
    }


def _calculate_stress_indicator(market_state) -> float:
    """Calculate market stress level (0-1)."""
    stress = 0.0
    
    if market_state.volatility > 0.05:
        stress += 0.3
    if market_state.liquidity_index < 0.7:
        stress += 0.3
    if hasattr(market_state, 'condition') and market_state.condition.value != 'normal':
        stress += 0.4
    
    return min(stress, 1.0)


def _estimate_sell_pressure(env) -> float:
    """Estimate current sell pressure in market."""
    if not env or not env.network:
        return 0.0
    
    stressed_count = sum(1 for b in env.network.banks.values() if b.capital_ratio < 0.08)
    total_count = len(env.network.banks)
    
    return stressed_count / max(total_count, 1)


def _determine_regime(market_state) -> Dict[str, Any]:
    """Determine current market regime."""
    vol = market_state.volatility
    liq = market_state.liquidity_index
    
    if vol < 0.02 and liq > 0.9:
        regime = "calm"
        description = "Low volatility, high liquidity"
    elif vol < 0.05 and liq > 0.7:
        regime = "normal"
        description = "Normal market conditions"
    elif vol < 0.10 or liq > 0.5:
        regime = "volatile"
        description = "Elevated volatility or reduced liquidity"
    elif vol < 0.15 or liq > 0.3:
        regime = "stressed"
        description = "High stress, potential fire sale risk"
    else:
        regime = "crisis"
        description = "Extreme conditions, market dysfunction"
    
    return {
        "name": regime,
        "description": description,
        "volatility_level": "low" if vol < 0.03 else ("medium" if vol < 0.08 else "high"),
        "liquidity_level": "high" if liq > 0.8 else ("medium" if liq > 0.5 else "low")
    }


# ============================================================================
# MARKET HISTORY
# ============================================================================

@router.get("/history")
async def get_market_history(
    start: int = Query(default=0, ge=0),
    end: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get market history for charting.
    
    Returns time series of price, volatility, liquidity, etc.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    history = simulation_state.step_history
    
    if end is not None:
        history = history[start:end]
    else:
        history = history[start:]
    
    timesteps = [h.timestep for h in history]
    
    return {
        "count": len(history),
        "timesteps": timesteps,
        
        "series": {
            "price": {
                "name": "Asset Price",
                "values": [h.market_price for h in history],
                "color": "#2196F3"
            },
            "volatility": {
                "name": "Volatility",
                "values": [h.volatility for h in history],
                "color": "#FF5722"
            },
            "liquidity": {
                "name": "Liquidity Index",
                "values": [h.liquidity_index for h in history],
                "color": "#4CAF50"
            },
            "interest_rate": {
                "name": "Interest Rate",
                "values": [h.interest_rate for h in history],
                "color": "#9C27B0"
            }
        }
    }


@router.get("/history/prices")
async def get_price_history() -> Dict[str, Any]:
    """Get detailed price history with technical indicators."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    history = simulation_state.step_history
    prices = [h.market_price for h in history]
    timesteps = [h.timestep for h in history]
    
    # Calculate returns
    returns = [0]
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / max(prices[i-1], 0.01)
        returns.append(ret)
    
    # Calculate moving averages
    ma_5 = _moving_average(prices, 5)
    ma_10 = _moving_average(prices, 10)
    
    # Calculate rolling volatility
    rolling_vol = _rolling_volatility(returns, 5)
    
    # Statistics
    stats = {
        "current": prices[-1] if prices else 0,
        "high": max(prices) if prices else 0,
        "low": min(prices) if prices else 0,
        "mean": np.mean(prices) if prices else 0,
        "std": np.std(prices) if prices else 0,
        "total_return": (prices[-1] / prices[0] - 1) if len(prices) > 1 else 0
    }
    
    return {
        "timesteps": timesteps,
        "prices": [round(p, 4) for p in prices],
        "returns": [round(r, 6) for r in returns],
        
        "indicators": {
            "ma_5": [round(m, 4) if m else None for m in ma_5],
            "ma_10": [round(m, 4) if m else None for m in ma_10],
            "rolling_volatility": [round(v, 4) if v else None for v in rolling_vol]
        },
        
        "statistics": {k: round(v, 4) for k, v in stats.items()}
    }


def _moving_average(values: List[float], window: int) -> List[Optional[float]]:
    """Calculate simple moving average."""
    result = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        result.append(np.mean(values[i-window+1:i+1]))
    return result


def _rolling_volatility(returns: List[float], window: int) -> List[Optional[float]]:
    """Calculate rolling volatility."""
    result = [None] * (window - 1)
    for i in range(window - 1, len(returns)):
        result.append(np.std(returns[i-window+1:i+1]))
    return result


# ============================================================================
# FIRE SALE DYNAMICS
# ============================================================================

@router.get("/fire-sale-risk")
async def get_fire_sale_risk() -> Dict[str, Any]:
    """
    Assess current fire sale risk.
    
    Estimates potential price impact if stressed banks liquidate assets.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    env = simulation_state.env
    network = env.network
    market = env.market
    
    # Find stressed banks
    stressed_banks = [b for b in network.banks.values() if b.capital_ratio < 0.08]
    
    # Estimate potential sell volume
    potential_sells = sum(b.balance_sheet.illiquid_assets * 0.3 for b in stressed_banks)
    
    # Estimate price impact using fire sale model
    current_price = market.get_state().asset_price
    
    # Price impact: P_new = P * exp(-kappa * volume)
    kappa = 0.001  # Price impact coefficient
    if potential_sells > 0:
        estimated_impact = 1 - np.exp(-kappa * potential_sells / 1000)
        projected_price = current_price * (1 - estimated_impact)
    else:
        estimated_impact = 0
        projected_price = current_price
    
    # Risk assessment
    if estimated_impact > 0.15:
        risk_level = "high"
    elif estimated_impact > 0.05:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "current_price": round(current_price, 4),
        
        "stressed_banks": {
            "count": len(stressed_banks),
            "ids": [b.bank_id for b in stressed_banks]
        },
        
        "potential_liquidation": {
            "volume": round(potential_sells, 2),
            "as_pct_of_market": round(potential_sells / max(sum(b.balance_sheet.illiquid_assets for b in network.banks.values()), 1), 4)
        },
        
        "price_impact": {
            "estimated_impact_pct": round(estimated_impact * 100, 2),
            "projected_price": round(projected_price, 4),
            "price_drop": round(current_price - projected_price, 4)
        },
        
        "risk_assessment": {
            "level": risk_level,
            "description": {
                "low": "Minimal fire sale risk",
                "medium": "Moderate fire sale risk - monitor closely",
                "high": "High fire sale risk - potential cascade"
            }[risk_level]
        }
    }


# ============================================================================
# INTEREST RATES
# ============================================================================

@router.get("/rates")
async def get_interest_rates() -> Dict[str, Any]:
    """Get interest rate information."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    history = simulation_state.step_history
    market_state = simulation_state.env.market.get_state()
    
    current_rate = market_state.interest_rate
    
    # Historical rates
    rates = [h.interest_rate for h in history]
    
    # Spread calculation
    base_rate = 0.02  # Assumed base rate
    spread = current_rate - base_rate
    
    return {
        "current_rate": round(current_rate, 4),
        "base_rate": base_rate,
        "spread": round(spread, 4),
        
        "history": {
            "timesteps": [h.timestep for h in history],
            "values": [round(r, 4) for r in rates]
        },
        
        "statistics": {
            "min": round(min(rates), 4) if rates else 0,
            "max": round(max(rates), 4) if rates else 0,
            "mean": round(np.mean(rates), 4) if rates else 0,
            "current_vs_mean": round(current_rate - np.mean(rates), 4) if rates else 0
        },
        
        "regime": "normal" if spread < 0.02 else ("elevated" if spread < 0.05 else "stressed")
    }


# ============================================================================
# MARKET SUMMARY
# ============================================================================

@router.get("/summary")
async def get_market_summary() -> Dict[str, Any]:
    """Get complete market summary for dashboard."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    market_state = simulation_state.env.market.get_state()
    history = simulation_state.step_history
    
    # Current values
    current = {
        "price": round(market_state.asset_price, 4),
        "interest_rate": round(market_state.interest_rate, 4),
        "volatility": round(market_state.volatility, 4),
        "liquidity": round(market_state.liquidity_index, 4)
    }
    
    # Trends (compare to 5 steps ago)
    trends = {}
    if len(history) > 5:
        prev = history[-6]
        trends = {
            "price": "up" if current["price"] > prev.market_price else "down",
            "volatility": "up" if current["volatility"] > prev.volatility else "down",
            "liquidity": "up" if current["liquidity"] > prev.liquidity_index else "down"
        }
    
    # Alerts
    alerts = []
    if market_state.volatility > 0.08:
        alerts.append({"level": "warning", "message": "High volatility detected"})
    if market_state.liquidity_index < 0.5:
        alerts.append({"level": "warning", "message": "Low liquidity"})
    if _estimate_sell_pressure(simulation_state.env) > 0.3:
        alerts.append({"level": "critical", "message": "High sell pressure"})
    
    return {
        "current": current,
        "trends": trends,
        "condition": market_state.condition.value if hasattr(market_state.condition, 'value') else "normal",
        "alerts": alerts,
        "last_updated": simulation_state.env.current_step
    }
