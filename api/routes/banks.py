"""
Bank API Routes.
Endpoints for bank details, history, and transaction records.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np

from .state import simulation_state
from .models import (
    BankSummary, BankDetails, BalanceSheet, NetworkPosition,
    MarginStatus, BankHistory, BankHistoryPoint,
    BankTransactionHistory, TransactionRecord, TransactionSummary
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.bank_registry import bank_registry

router = APIRouter(prefix="/bank", tags=["Banks"])


def convert_numpy(obj):
    """Recursively convert numpy types to native Python types."""
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


# ============================================================================
# REAL BANK REGISTRY (must be before /{bank_id} to avoid route conflicts)
# ============================================================================

@router.get("/registry")
async def get_bank_registry() -> Dict[str, Any]:
    """Get all available real banks from the RBI data registry."""
    all_banks = bank_registry.get_available_list()
    return {
        "count": len(all_banks),
        "source": "RBI A Profile of Banks 2012-13",
        "banks": all_banks
    }


@router.get("/registry/search")
async def search_bank_registry(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Search real banks by name."""
    results = bank_registry.search_banks(q)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {"id": b.bank_id, "name": b.name, "tier": b.tier}
            for b in results
        ]
    }


@router.get("/registry/{bank_id}")
async def get_registry_bank(bank_id: int) -> Dict[str, Any]:
    """Get details of a specific bank from the registry."""
    bank = bank_registry.get_bank_by_id(bank_id)
    if not bank:
        raise HTTPException(status_code=404, detail=f"Bank ID {bank_id} not found in registry")
    return {"bank": bank.to_dict()}


# ============================================================================
# BANK LIST & SUMMARY
# ============================================================================

@router.get("/")
async def list_banks() -> Dict[str, Any]:
    """Get summary of all banks."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    env = simulation_state.env
    network = env.network
    
    banks = []
    for bank_id, bank in network.banks.items():
        # Calculate risk score
        risk_score = float(simulation_state._heuristic_pd(bank)) * 100
        
        bank_entry = {
            "bank_id": int(bank_id),
            "tier": int(bank.tier),
            "status": bank.status.value,
            "equity": round(float(bank.balance_sheet.equity), 2),
            "capital_ratio": round(float(bank.capital_ratio), 4),
            "cash": round(float(bank.balance_sheet.cash), 2),
            "risk_score": round(float(risk_score), 1)
        }
        # Include real bank name/metadata if available
        if hasattr(bank, 'name') and bank.name:
            bank_entry["name"] = bank.name
        if hasattr(bank, 'metadata') and bank.metadata:
            bank_entry["crar"] = bank.metadata.get("crar", 0)
            bank_entry["net_npa_ratio"] = bank.metadata.get("net_npa_ratio", 0)
        
        banks.append(bank_entry)
    
    # Sort by risk score (highest first)
    banks.sort(key=lambda x: x["risk_score"], reverse=True)
    
    return {
        "count": len(banks),
        "banks": banks,
        "summary": {
            "total_equity": round(float(sum(b["equity"] for b in banks)), 2),
            "avg_capital_ratio": round(float(np.mean([b["capital_ratio"] for b in banks])), 4),
            "at_risk": len([b for b in banks if b["risk_score"] > 50])
        }
    }


# ============================================================================
# BANK DETAILS
# ============================================================================

@router.get("/{bank_id}")
async def get_bank_details(bank_id: int) -> Dict[str, Any]:
    """
    Get complete details for a specific bank.
    
    Includes:
    - Full balance sheet
    - All financial ratios
    - Credit risk metrics (PD, LGD, EAD, EL)
    - Network position and centrality
    - Margin status at CCPs
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    if bank_id not in network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    bank = network.banks[bank_id]
    bs = bank.balance_sheet
    
    # Get centrality metrics
    centrality = network.get_centrality_metrics()
    bank_centrality = centrality.get(bank_id, {})
    
    # Get neighbors
    neighbors = network.get_neighbors(bank_id)
    creditors = network.get_creditors(bank_id)
    debtors = network.get_debtors(bank_id)
    
    # Calculate credit risk
    credit_risk = simulation_state.calculate_credit_risk(bank_id)
    
    # Get margin status
    margin_status = None
    if simulation_state.ccps:
        ccp = simulation_state.ccps[0]
        if bank_id in ccp.members:
            member_state = ccp.get_member_margin_status(bank_id)
            if member_state:
                margin_status = {
                    "ccp_id": ccp.ccp_id,
                    "initial_margin": float(member_state.get("initial_margin", 0)),
                    "variation_margin": float(member_state.get("variation_margin", 0)),
                    "total_margin": float(member_state.get("total_margin", 0)),
                    "margin_calls_pending": float(member_state.get("margin_calls_pending", 0)),
                    "margin_adequacy": float(member_state.get("margin_adequacy", 1.0))
                }
    
    # Calculate concentration
    exposures = list(bs.interbank_assets.values())
    largest_exposure = max(exposures) if exposures else 0
    concentration_index = 0
    if exposures and sum(exposures) > 0:
        concentration_index = sum((e / sum(exposures)) ** 2 for e in exposures)
    
    return convert_numpy({
        "bank_id": bank_id,
        "tier": bank.tier,
        "status": bank.status.value,
        
        # Balance Sheet
        "balance_sheet": {
            "cash": round(bs.cash, 2),
            "illiquid_assets": round(bs.illiquid_assets, 2),
            "interbank_assets": {int(k): round(v, 2) for k, v in bs.interbank_assets.items()},
            "interbank_liabilities": {int(k): round(v, 2) for k, v in bs.interbank_liabilities.items()},
            "external_liabilities": round(bs.external_liabilities, 2),
            "total_assets": round(bs.total_assets, 2),
            "total_liabilities": round(bs.total_liabilities, 2),
            "equity": round(bs.equity, 2)
        },
        
        # Ratios
        "ratios": {
            "capital_ratio": round(bank.capital_ratio, 4),
            "liquidity_ratio": round(bs.cash / max(bs.total_liabilities, 1), 4),
            "leverage_ratio": round(bs.total_assets / max(bs.equity, 1), 2),
            "interbank_ratio": round(bs.total_interbank_liabilities / max(bs.total_liabilities, 1), 4)
        },
        
        # Credit Risk
        "credit_risk": {
            "probability_of_default": round(credit_risk.get("probability_of_default", 0), 4),
            "loss_given_default": credit_risk.get("loss_given_default", 0.45),
            "exposure_at_default": round(credit_risk.get("exposure_at_default", bs.total_liabilities), 2),
            "expected_loss": round(
                credit_risk.get("probability_of_default", 0) * 
                credit_risk.get("loss_given_default", 0.45) * 
                credit_risk.get("exposure_at_default", bs.total_liabilities), 
                2
            ),
            "rating": credit_risk.get("rating", "BBB"),
            "systemic_importance": round(bank_centrality.get("eigenvector", 0), 4)
        },
        
        # Network Position
        "network_position": {
            "degree_centrality": round(bank_centrality.get("degree", 0), 4),
            "betweenness_centrality": round(bank_centrality.get("betweenness", 0), 4),
            "eigenvector_centrality": round(bank_centrality.get("eigenvector", 0), 4),
            "neighbors": neighbors,
            "creditors": creditors,
            "debtors": debtors,
            "num_connections": len(neighbors),
            "largest_exposure": round(largest_exposure, 2),
            "concentration_index": round(concentration_index, 4)
        },
        
        # Margin Status
        "margin_status": margin_status,
        
        # Flags
        "flags": {
            "is_solvent": bank.is_solvent,
            "is_liquid": bank.is_liquid,
            "is_core": bank.tier == 1,
            "below_capital_minimum": bank.capital_ratio < 0.08,
            "high_leverage": (bs.total_assets / max(bs.equity, 1)) > 15
        }
    })


# ============================================================================
# BANK HISTORY
# ============================================================================

@router.get("/{bank_id}/history")
async def get_bank_history(
    bank_id: int,
    start: int = Query(default=0, ge=0),
    end: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get historical state of a bank over time.
    
    Returns time series of key metrics: cash, equity, capital ratio, status.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if bank_id not in simulation_state.env.network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    history = simulation_state.get_bank_history(bank_id)
    
    # Apply range
    if end is not None:
        history = history[start:end]
    else:
        history = history[start:]
    
    # Format for charts
    timesteps = [h["timestep"] for h in history]
    
    return convert_numpy({
        "bank_id": bank_id,
        "count": len(history),
        "timesteps": timesteps,
        "data": history,
        
        # Pre-formatted series for charts
        "series": {
            "equity": [h.get("equity", 0) for h in history],
            "capital_ratio": [h.get("capital_ratio", 0) for h in history],
            "cash": [h.get("cash", 0) for h in history],
            "status": [h.get("status", "active") for h in history]
        }
    })


# ============================================================================
# TRANSACTION HISTORY
# ============================================================================

@router.get("/{bank_id}/transactions")
async def get_bank_transactions(
    bank_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    tx_type: Optional[str] = Query(default=None, description="Filter by type")
) -> Dict[str, Any]:
    """
    Get transaction history for a bank.
    
    Note: Transaction history is built from recorded actions during simulation.
    For complete history, ensure capture_state=True during steps.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if bank_id not in simulation_state.env.network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    transactions = simulation_state.get_bank_transactions(bank_id)
    
    # Filter by type if specified
    if tx_type:
        transactions = [t for t in transactions if t.get("type") == tx_type]
    
    # Apply limit
    transactions = transactions[-limit:]
    
    # Calculate summary
    total_inflows = sum(t["amount"] for t in transactions if t.get("direction") == "inflow")
    total_outflows = sum(t["amount"] for t in transactions if t.get("direction") == "outflow")
    counterparties = set(t.get("counterparty") for t in transactions if t.get("counterparty", -1) >= 0)
    
    return convert_numpy({
        "bank_id": bank_id,
        "count": len(transactions),
        "transactions": transactions,
        
        "summary": {
            "total_transactions": len(transactions),
            "total_inflows": round(total_inflows, 2),
            "total_outflows": round(total_outflows, 2),
            "net_flow": round(total_inflows - total_outflows, 2),
            "avg_transaction_size": round((total_inflows + total_outflows) / max(len(transactions), 1), 2),
            "counterparty_count": len(counterparties)
        }
    })


# ============================================================================
# BANK EXPOSURES
# ============================================================================

@router.get("/{bank_id}/exposures")
async def get_bank_exposures(bank_id: int) -> Dict[str, Any]:
    """
    Get detailed exposure breakdown for a bank.
    
    Shows who this bank owes money to and who owes money to this bank.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    if bank_id not in network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    bank = network.banks[bank_id]
    bs = bank.balance_sheet
    
    # Assets (money owed TO this bank)
    assets = []
    for debtor_id, amount in bs.interbank_assets.items():
        debtor = network.banks.get(debtor_id)
        assets.append({
            "counterparty_id": debtor_id,
            "amount": round(amount, 2),
            "counterparty_status": debtor.status.value if debtor else "unknown",
            "counterparty_capital_ratio": round(debtor.capital_ratio, 4) if debtor else 0,
            "at_risk": debtor.status.value == "defaulted" if debtor else False
        })
    
    # Liabilities (money owed BY this bank)
    liabilities = []
    for creditor_id, amount in bs.interbank_liabilities.items():
        creditor = network.banks.get(creditor_id)
        liabilities.append({
            "counterparty_id": creditor_id,
            "amount": round(amount, 2),
            "counterparty_status": creditor.status.value if creditor else "unknown",
            "counterparty_tier": creditor.tier if creditor else 2
        })
    
    # Sort by amount
    assets.sort(key=lambda x: x["amount"], reverse=True)
    liabilities.sort(key=lambda x: x["amount"], reverse=True)
    
    # Calculate metrics
    total_assets = sum(a["amount"] for a in assets)
    total_liabilities = sum(l["amount"] for l in liabilities)
    exposure_to_defaults = sum(a["amount"] for a in assets if a["at_risk"])
    
    return convert_numpy({
        "bank_id": bank_id,
        
        "assets": {
            "count": len(assets),
            "total": round(total_assets, 2),
            "details": assets
        },
        
        "liabilities": {
            "count": len(liabilities),
            "total": round(total_liabilities, 2),
            "details": liabilities
        },
        
        "summary": {
            "net_position": round(total_assets - total_liabilities, 2),
            "is_net_lender": total_assets > total_liabilities,
            "exposure_to_defaults": round(exposure_to_defaults, 2),
            "largest_asset_exposure": assets[0]["amount"] if assets else 0,
            "largest_liability_exposure": liabilities[0]["amount"] if liabilities else 0
        }
    })


# ============================================================================
# BANK COMPARISONS
# ============================================================================

@router.get("/compare")
async def compare_banks(
    bank_ids: str = Query(..., description="Comma-separated bank IDs")
) -> Dict[str, Any]:
    """Compare multiple banks side by side."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    ids = [int(x.strip()) for x in bank_ids.split(",")]
    network = simulation_state.env.network
    
    comparison = []
    for bank_id in ids:
        if bank_id in network.banks:
            bank = network.banks[bank_id]
            bs = bank.balance_sheet
            
            comparison.append({
                "bank_id": bank_id,
                "tier": bank.tier,
                "status": bank.status.value,
                "equity": round(bs.equity, 2),
                "capital_ratio": round(bank.capital_ratio, 4),
                "cash": round(bs.cash, 2),
                "total_assets": round(bs.total_assets, 2),
                "leverage": round(bs.total_assets / max(bs.equity, 1), 2),
                "pd": round(simulation_state._heuristic_pd(bank), 4)
            })
    
    return convert_numpy({
        "banks": comparison,
        "metrics": ["equity", "capital_ratio", "cash", "total_assets", "leverage", "pd"]
    })


# ============================================================================
# STRESSED / AT-RISK BANKS
# ============================================================================

@router.get("/stressed")
async def get_stressed_banks() -> Dict[str, Any]:
    """Get list of banks currently in stressed or critical condition."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    stressed = []
    critical = []
    defaulted = []
    
    for bank_id, bank in network.banks.items():
        info = {
            "bank_id": bank_id,
            "capital_ratio": round(bank.capital_ratio, 4),
            "cash": round(bank.balance_sheet.cash, 2),
            "equity": round(bank.balance_sheet.equity, 2),
            "pd": round(simulation_state._heuristic_pd(bank), 4)
        }
        
        if bank.status.value == "defaulted":
            defaulted.append(info)
        elif bank.capital_ratio < 0.04:
            critical.append(info)
        elif bank.capital_ratio < 0.08:
            stressed.append(info)
    
    return convert_numpy({
        "stressed": {
            "count": len(stressed),
            "banks": stressed
        },
        "critical": {
            "count": len(critical),
            "banks": critical
        },
        "defaulted": {
            "count": len(defaulted),
            "banks": defaulted
        },
        "summary": {
            "total_at_risk": len(stressed) + len(critical),
            "total_defaulted": len(defaulted),
            "system_health": "healthy" if len(critical) == 0 else ("stressed" if len(defaulted) == 0 else "crisis")
        }
    })
