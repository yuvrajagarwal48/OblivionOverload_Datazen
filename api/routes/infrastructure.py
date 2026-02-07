"""
Infrastructure API Routes.
Endpoints for CCP status, exchange metrics, and infrastructure monitoring.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np

from .state import simulation_state

router = APIRouter(prefix="/infrastructure", tags=["Infrastructure"])


# ============================================================================
# INFRASTRUCTURE OVERVIEW
# ============================================================================

@router.get("/")
async def get_infrastructure_overview() -> Dict[str, Any]:
    """
    Get complete infrastructure status overview.
    
    Shows exchanges, CCPs, and system-wide infrastructure health.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    exchanges = simulation_state.exchanges
    ccps = simulation_state.ccps
    
    # Exchange summary
    exchange_data = []
    total_volume = 0
    total_fees = 0
    
    for ex in exchanges:
        state = ex.get_state()
        exchange_data.append({
            "exchange_id": ex.exchange_id,
            "congestion": round(state.congestion_level, 4),
            "volume": round(state.transaction_volume, 2),
            "fees": round(state.total_fees_collected, 2),
            "pending": state.pending_settlements,
            "status": "normal" if state.congestion_level < 0.7 else "congested"
        })
        total_volume += state.transaction_volume
        total_fees += state.total_fees_collected
    
    # CCP summary
    ccp_data = []
    total_margin = 0
    total_default_fund = 0
    
    for ccp in ccps:
        state = ccp.get_state()
        ccp_data.append({
            "ccp_id": ccp.ccp_id,
            "status": state.status.value,
            "total_margin": round(state.total_initial_margin + state.total_variation_margin, 2),
            "default_fund": round(state.default_fund_size, 2),
            "stress_level": round(state.stress_level, 4),
            "members": state.num_members,
            "distressed_members": state.num_defaulted_members
        })
        total_margin += state.total_initial_margin + state.total_variation_margin
        total_default_fund += state.default_fund_size
    
    # Overall system health
    avg_congestion = np.mean([e["congestion"] for e in exchange_data]) if exchange_data else 0
    avg_stress = np.mean([c["stress_level"] for c in ccp_data]) if ccp_data else 0
    
    return {
        "exchanges": exchange_data,
        "ccps": ccp_data,
        
        "aggregates": {
            "total_transaction_volume": round(total_volume, 2),
            "total_fees_collected": round(total_fees, 2),
            "total_margin_pool": round(total_margin, 2),
            "total_default_resources": round(total_default_fund, 2)
        },
        
        "health": {
            "avg_exchange_congestion": round(avg_congestion, 4),
            "avg_ccp_stress": round(avg_stress, 4),
            "infrastructure_status": _get_infra_status(avg_congestion, avg_stress)
        }
    }


def _get_infra_status(congestion: float, stress: float) -> str:
    """Determine overall infrastructure status."""
    if congestion < 0.5 and stress < 0.3:
        return "healthy"
    elif congestion < 0.8 and stress < 0.6:
        return "moderate"
    else:
        return "stressed"


# ============================================================================
# CCP STATUS
# ============================================================================

@router.get("/ccp")
async def list_ccps() -> Dict[str, Any]:
    """List all CCPs and their status."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    ccps = []
    for ccp in simulation_state.ccps:
        state = ccp.get_state()
        ccps.append({
            "ccp_id": ccp.ccp_id,
            "name": getattr(ccp, 'name', f"CCP-{ccp.ccp_id}"),
            "status": state.status.value,
            "stress_level": round(state.stress_level, 4)
        })
    
    return {"count": len(ccps), "ccps": ccps}


@router.get("/ccp/{ccp_id}")
async def get_ccp_status(ccp_id: int) -> Dict[str, Any]:
    """
    Get detailed CCP status.
    
    Includes margin pools, default resources, coverage ratios, and member status.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    ccp = None
    for c in simulation_state.ccps:
        if c.ccp_id == ccp_id:
            ccp = c
            break
    
    if not ccp:
        raise HTTPException(status_code=404, detail=f"CCP {ccp_id} not found")
    
    state = ccp.get_state()
    
    # Get member details
    members = []
    for member_id in ccp.members:
        member_state = ccp.get_member_state(member_id)
        if member_state:
            bank = simulation_state.env.network.banks.get(member_id)
            members.append({
                "member_id": member_id,
                "initial_margin": round(member_state.initial_margin, 2),
                "variation_margin": round(member_state.variation_margin, 2),
                "total_margin": round(member_state.total_margin, 2),
                "margin_calls_pending": round(member_state.margin_calls_pending, 2),
                "bank_status": bank.status.value if bank else "unknown"
            })
    
    # Sort by total margin (largest first)
    members.sort(key=lambda x: x["total_margin"], reverse=True)
    
    # Calculate coverage ratios
    total_margin = state.total_initial_margin + state.total_variation_margin
    total_resources = total_margin + state.default_fund_size + state.ccp_capital
    
    # Largest member exposure (for Cover-1)
    if members:
        largest_exposure = members[0]["total_margin"] * 10  # Approximate
        cover_1 = total_resources / max(largest_exposure, 1)
    else:
        cover_1 = 1.0
    
    return {
        "ccp_id": ccp_id,
        "name": getattr(ccp, 'name', f"CCP-{ccp_id}"),
        "status": state.status.value,
        
        # Margin Pool
        "margins": {
            "initial_margin_total": round(state.total_initial_margin, 2),
            "variation_margin_total": round(state.total_variation_margin, 2),
            "total_margin_pool": round(total_margin, 2)
        },
        
        # Default Resources
        "default_resources": {
            "default_fund": round(state.default_fund_size, 2),
            "ccp_capital": round(state.ccp_capital, 2),
            "total_prefunded": round(total_resources, 2)
        },
        
        # Coverage
        "coverage": {
            "stress_coverage_ratio": round(total_resources / max(total_margin * 0.2, 1), 2),
            "cover_1_ratio": round(cover_1, 2),
            "cover_2_ratio": round(cover_1 * 0.7, 2)  # Simplified
        },
        
        # Member Status
        "members": {
            "total": state.num_members,
            "distressed": state.num_defaulted_members,
            "on_margin_call": sum(1 for m in members if m["margin_calls_pending"] > 0),
            "details": members[:20]  # Top 20
        },
        
        # Activity
        "activity": {
            "pending_settlements": state.pending_settlements,
            "margin_calls_issued": state.margin_call_volume,
            "stress_level": round(state.stress_level, 4)
        },
        
        # Waterfall
        "waterfall": {
            "current_layer": 0 if state.status.value == "normal" else 1,
            "layers": [
                {"name": "Member Margin", "available": round(total_margin, 2)},
                {"name": "Default Fund", "available": round(state.default_fund_size, 2)},
                {"name": "CCP Capital", "available": round(state.ccp_capital, 2)},
                {"name": "Mutualization", "available": "As needed"}
            ]
        }
    }


@router.get("/ccp/{ccp_id}/members")
async def get_ccp_members(ccp_id: int) -> Dict[str, Any]:
    """Get all members of a CCP with their margin status."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    ccp = None
    for c in simulation_state.ccps:
        if c.ccp_id == ccp_id:
            ccp = c
            break
    
    if not ccp:
        raise HTTPException(status_code=404, detail=f"CCP {ccp_id} not found")
    
    members = []
    for member_id in ccp.members:
        member_state = ccp.get_member_state(member_id)
        bank = simulation_state.env.network.banks.get(member_id)
        
        if member_state and bank:
            # Calculate margin adequacy
            required_margin = bank.balance_sheet.total_liabilities * 0.1
            adequacy = member_state.total_margin / max(required_margin, 1)
            
            members.append({
                "member_id": member_id,
                "bank_tier": bank.tier,
                "bank_status": bank.status.value,
                "initial_margin": round(member_state.initial_margin, 2),
                "variation_margin": round(member_state.variation_margin, 2),
                "total_margin": round(member_state.total_margin, 2),
                "margin_calls_pending": round(member_state.margin_calls_pending, 2),
                "default_fund_contribution": round(member_state.default_fund_contribution, 2),
                "margin_adequacy": round(adequacy, 2),
                "is_adequate": adequacy >= 1.0
            })
    
    # Summary
    total_margin = sum(m["total_margin"] for m in members)
    inadequate = [m for m in members if not m["is_adequate"]]
    on_margin_call = [m for m in members if m["margin_calls_pending"] > 0]
    
    return {
        "ccp_id": ccp_id,
        "member_count": len(members),
        "members": members,
        
        "summary": {
            "total_margin_posted": round(total_margin, 2),
            "members_with_adequate_margin": len(members) - len(inadequate),
            "members_on_margin_call": len(on_margin_call),
            "total_margin_calls_pending": sum(m["margin_calls_pending"] for m in members)
        }
    }


# ============================================================================
# EXCHANGE STATUS
# ============================================================================

@router.get("/exchange")
async def list_exchanges() -> Dict[str, Any]:
    """List all exchanges and their status."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    exchanges = []
    for ex in simulation_state.exchanges:
        state = ex.get_state()
        exchanges.append({
            "exchange_id": ex.exchange_id,
            "name": getattr(ex, 'name', f"Exchange-{ex.exchange_id}"),
            "congestion": round(state.congestion_level, 4),
            "volume": round(state.transaction_volume, 2),
            "status": "normal" if state.congestion_level < 0.7 else "congested"
        })
    
    return {"count": len(exchanges), "exchanges": exchanges}


@router.get("/exchange/{exchange_id}")
async def get_exchange_status(exchange_id: int) -> Dict[str, Any]:
    """
    Get detailed exchange status.
    
    Includes volume, congestion, fees, and settlement timing.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    exchange = None
    for ex in simulation_state.exchanges:
        if ex.exchange_id == exchange_id:
            exchange = ex
            break
    
    if not exchange:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_id} not found")
    
    state = exchange.get_state()
    config = exchange.config if hasattr(exchange, 'config') else None
    
    # Calculate effective fee rate
    base_fee = config.base_fee if config else 0.001
    congestion_multiplier = 1 + state.congestion_level
    effective_fee = base_fee * congestion_multiplier
    
    return {
        "exchange_id": exchange_id,
        "name": getattr(exchange, 'name', f"Exchange-{exchange_id}"),
        
        # Volume & Activity
        "activity": {
            "transaction_volume": round(state.transaction_volume, 2),
            "transaction_count": state.transaction_count,
            "avg_transaction_size": round(state.transaction_volume / max(state.transaction_count, 1), 2)
        },
        
        # Capacity
        "capacity": {
            "max_throughput": config.max_throughput if config else 1000,
            "current_utilization": round(state.congestion_level, 4),
            "remaining_capacity": round(1 - state.congestion_level, 4)
        },
        
        # Congestion
        "congestion": {
            "level": round(state.congestion_level, 4),
            "order_backlog": state.order_backlog if hasattr(state, 'order_backlog') else 0,
            "queue_depth": state.queue_depth if hasattr(state, 'queue_depth') else 0,
            "status": "normal" if state.congestion_level < 0.5 else ("busy" if state.congestion_level < 0.8 else "congested")
        },
        
        # Timing
        "timing": {
            "avg_settlement_delay": round(state.avg_settlement_delay if hasattr(state, 'avg_settlement_delay') else 1, 2),
            "max_settlement_delay": state.max_settlement_delay if hasattr(state, 'max_settlement_delay') else 3,
            "pending_settlements": state.pending_settlements
        },
        
        # Fees
        "fees": {
            "base_fee_rate": round(base_fee, 4),
            "congestion_multiplier": round(congestion_multiplier, 2),
            "effective_fee_rate": round(effective_fee, 4),
            "total_collected": round(state.total_fees_collected, 2)
        },
        
        # Status
        "status": {
            "is_stressed": state.congestion_level > 0.8,
            "circuit_breaker_active": getattr(state, 'circuit_breaker_active', False)
        }
    }


# ============================================================================
# INFRASTRUCTURE ROUTER
# ============================================================================

@router.get("/routing")
async def get_routing_info() -> Dict[str, Any]:
    """Get infrastructure routing configuration and statistics."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    router = simulation_state.router
    
    if not router:
        return {"message": "Router not configured"}
    
    # Exchange utilization
    exchange_util = {}
    for ex in simulation_state.exchanges:
        state = ex.get_state()
        exchange_util[ex.exchange_id] = round(state.congestion_level, 4)
    
    # CCP membership
    ccp_membership = {}
    for ccp in simulation_state.ccps:
        ccp_membership[ccp.ccp_id] = len(ccp.members)
    
    return {
        "registered_banks": len(router.bank_registry) if hasattr(router, 'bank_registry') else 0,
        
        "exchange_utilization": exchange_util,
        "preferred_exchange": min(exchange_util.items(), key=lambda x: x[1])[0] if exchange_util else None,
        
        "ccp_membership": ccp_membership,
        
        "routing_strategy": "least_congested",
        "fallback_enabled": True
    }


# ============================================================================
# MARGIN CALL HISTORY
# ============================================================================

@router.get("/margin-calls")
async def get_margin_calls(
    ccp_id: Optional[int] = None,
    status: Optional[str] = Query(default=None, description="Filter: issued, met, failed")
) -> Dict[str, Any]:
    """Get margin call events."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # In a full implementation, this would query the history
    # For now, return current margin call status
    
    margin_calls = []
    
    for ccp in simulation_state.ccps:
        if ccp_id is not None and ccp.ccp_id != ccp_id:
            continue
        
        for member_id in ccp.members:
            member_state = ccp.get_member_state(member_id)
            if member_state and member_state.margin_calls_pending > 0:
                margin_calls.append({
                    "ccp_id": ccp.ccp_id,
                    "member_id": member_id,
                    "amount": round(member_state.margin_calls_pending, 2),
                    "status": "issued",
                    "timestep": simulation_state.env.current_step
                })
    
    return {
        "count": len(margin_calls),
        "margin_calls": margin_calls,
        "total_pending": sum(mc["amount"] for mc in margin_calls)
    }


# ============================================================================
# WATERFALL STATUS
# ============================================================================

@router.get("/waterfall")
async def get_waterfall_status() -> Dict[str, Any]:
    """Get default waterfall status across all CCPs."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    waterfalls = []
    
    for ccp in simulation_state.ccps:
        state = ccp.get_state()
        
        total_margin = state.total_initial_margin + state.total_variation_margin
        
        waterfalls.append({
            "ccp_id": ccp.ccp_id,
            "current_layer": 0,  # 0 = normal, 1+ = waterfall activated
            "layers": [
                {
                    "layer": 1,
                    "name": "Defaulter's Margin",
                    "available": round(total_margin, 2),
                    "used": 0,
                    "status": "ready"
                },
                {
                    "layer": 2,
                    "name": "Default Fund",
                    "available": round(state.default_fund_size, 2),
                    "used": 0,
                    "status": "ready"
                },
                {
                    "layer": 3,
                    "name": "CCP Capital",
                    "available": round(state.ccp_capital, 2),
                    "used": 0,
                    "status": "ready"
                },
                {
                    "layer": 4,
                    "name": "Mutualization",
                    "available": "Surviving members",
                    "used": 0,
                    "status": "standby"
                }
            ],
            "total_resources": round(total_margin + state.default_fund_size + state.ccp_capital, 2)
        })
    
    return {
        "ccps": waterfalls,
        "any_activated": any(w["current_layer"] > 0 for w in waterfalls)
    }
