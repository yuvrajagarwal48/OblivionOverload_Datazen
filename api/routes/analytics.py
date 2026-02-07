"""
Analytics API Routes.
Endpoints for systemic risk, network analysis, credit risk, and DebtRank.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np

from .state import simulation_state

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ============================================================================
# SYSTEMIC RISK METRICS
# ============================================================================

@router.get("/systemic-risk")
async def get_systemic_risk_metrics() -> Dict[str, Any]:
    """
    Get comprehensive systemic risk analysis.
    
    Includes:
    - DebtRank (systemic importance measure)
    - Contagion metrics (cascade depth, potential)
    - Network metrics (density, clustering)
    - System health indices
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    env = simulation_state.env
    network = env.network
    
    # Build inputs for risk analysis
    equity_vector = np.array([b.balance_sheet.equity for b in network.banks.values()])
    cash_vector = np.array([b.balance_sheet.cash for b in network.banks.values()])
    liability_vector = np.array([b.balance_sheet.total_liabilities for b in network.banks.values()])
    capital_ratios = np.array([b.capital_ratio for b in network.banks.values()])
    
    # Run risk analyzer
    report = simulation_state.risk_analyzer.analyze(
        exposure_matrix=network.liability_matrix.T,
        equity_vector=equity_vector,
        cash_vector=cash_vector,
        liability_vector=liability_vector,
        capital_ratios=capital_ratios,
        graph=network.graph
    )
    
    # Get network stats
    network_stats = network.get_network_stats()
    
    # Calculate additional metrics
    num_banks = len(network.banks)
    num_stressed = sum(1 for b in network.banks.values() if b.capital_ratio < 0.08)
    num_defaulted = sum(1 for b in network.banks.values() if b.status.value == "defaulted")
    
    # Liquidity index
    total_cash = sum(b.balance_sheet.cash for b in network.banks.values())
    total_liabilities = sum(b.balance_sheet.total_liabilities for b in network.banks.values())
    liquidity_index = total_cash / max(total_liabilities, 1)
    
    # Stress index
    stress_index = (num_stressed + num_defaulted * 2) / max(num_banks, 1)
    
    return {
        # DebtRank
        "debt_rank": {
            "aggregate": round(report.debt_rank, 4),
            "individual": {int(k): round(v, 4) for k, v in report.bank_risk_scores.items()},
            "systemically_important": report.systemically_important_banks,
            "vulnerable": report.vulnerable_banks
        },
        
        # Contagion
        "contagion": {
            "cascade_depth": round(report.contagion_depth, 2),
            "cascade_potential": round(report.cascade_potential, 4),
            "critical_banks": report.systemically_important_banks[:5]
        },
        
        # Network
        "network": {
            "density": round(report.network_density, 4),
            "clustering_coefficient": round(report.avg_clustering, 4),
            "largest_component_size": int(report.largest_component_size),
            "concentration_index": round(network_stats.concentration_index, 4) if hasattr(network_stats, 'concentration_index') else 0
        },
        
        # System Health
        "health": {
            "liquidity_index": round(liquidity_index, 4),
            "stress_index": round(stress_index, 4),
            "systemic_risk_index": round(report.systemic_risk_index, 4),
            "overall_status": "healthy" if stress_index < 0.1 else ("stressed" if stress_index < 0.3 else "crisis")
        },
        
        # Summary counts
        "counts": {
            "total_banks": num_banks,
            "stressed_banks": num_stressed,
            "defaulted_banks": num_defaulted,
            "healthy_banks": num_banks - num_stressed - num_defaulted
        }
    }


# ============================================================================
# DEBTRANK
# ============================================================================

@router.get("/debtrank")
async def get_debtrank() -> Dict[str, Any]:
    """
    Calculate DebtRank for all banks.
    
    DebtRank measures systemic importance - the fraction of total economic
    value that would be affected if a bank defaults.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    # Build exposure matrix
    n = len(network.banks)
    bank_ids = list(network.banks.keys())
    id_to_idx = {bid: i for i, bid in enumerate(bank_ids)}
    
    exposure_matrix = np.zeros((n, n))
    equity_vector = np.zeros(n)
    
    for bank_id, bank in network.banks.items():
        i = id_to_idx[bank_id]
        equity_vector[i] = max(bank.balance_sheet.equity, 0.01)
        
        for creditor_id, amount in bank.balance_sheet.interbank_liabilities.items():
            if creditor_id in id_to_idx:
                j = id_to_idx[creditor_id]
                exposure_matrix[j, i] = amount  # j is exposed to i's default
    
    # Calculate DebtRank
    from src.analytics.risk_metrics import DebtRankCalculator
    calculator = DebtRankCalculator(recovery_rate=0.0)
    aggregate_dr, individual_dr = calculator.calculate(exposure_matrix, equity_vector)
    
    # Map back to bank IDs
    individual_ranks = {bank_ids[i]: float(individual_dr[i]) for i in range(n)}
    
    # Sort by importance
    sorted_ranks = sorted(individual_ranks.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "aggregate_debt_rank": round(float(aggregate_dr), 4),
        "individual_ranks": {int(k): round(v, 4) for k, v in individual_ranks.items()},
        "ranking": [{"bank_id": int(bid), "debt_rank": round(dr, 4)} for bid, dr in sorted_ranks],
        "systemically_important": [int(bid) for bid, dr in sorted_ranks if dr > 0.05],
        "interpretation": {
            "0-0.05": "Low systemic importance",
            "0.05-0.15": "Moderate systemic importance",
            "0.15+": "High systemic importance (SIFI)"
        }
    }


# ============================================================================
# NETWORK ANALYTICS
# ============================================================================

@router.get("/network")
async def get_network_analytics() -> Dict[str, Any]:
    """
    Get comprehensive network structure analysis.
    
    Includes topology metrics, centrality distributions, and core-periphery structure.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    graph = network.graph
    
    # Basic topology
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = num_edges / max(num_nodes * (num_nodes - 1), 1)
    
    # Degree statistics
    degrees = [d for _, d in graph.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    
    # Centrality metrics
    centrality = network.get_centrality_metrics()
    
    # Extract statistics for each centrality type
    centrality_stats = {}
    for metric_type in ['degree', 'betweenness', 'eigenvector']:
        values = [c.get(metric_type, 0) for c in centrality.values()]
        if values:
            centrality_stats[metric_type] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4)
            }
    
    # Core-periphery structure
    core_banks = [bid for bid, bank in network.banks.items() if bank.tier == 1]
    periphery_banks = [bid for bid, bank in network.banks.items() if bank.tier == 2]
    
    # Component analysis
    import networkx as nx
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    
    num_components = len(components)
    largest_component = max(len(c) for c in components) if components else 0
    
    # Clustering
    try:
        global_clustering = nx.transitivity(graph)
        local_clustering = nx.clustering(graph.to_undirected())
        avg_local_clustering = np.mean(list(local_clustering.values())) if local_clustering else 0
    except:
        global_clustering = 0
        avg_local_clustering = 0
    
    return {
        # Topology
        "topology": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": round(density, 4),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree
        },
        
        # Components
        "components": {
            "num_components": num_components,
            "largest_component_size": largest_component,
            "fragmentation": round(1 - largest_component / max(num_nodes, 1), 4)
        },
        
        # Centrality
        "centrality": {
            "statistics": centrality_stats,
            "per_bank": {int(k): {m: round(v, 4) for m, v in metrics.items()} 
                        for k, metrics in centrality.items()}
        },
        
        # Core-Periphery
        "structure": {
            "core_banks": core_banks,
            "periphery_banks": periphery_banks,
            "core_count": len(core_banks),
            "periphery_count": len(periphery_banks)
        },
        
        # Clustering
        "clustering": {
            "global": round(global_clustering, 4),
            "average_local": round(avg_local_clustering, 4)
        }
    }


# ============================================================================
# NETWORK VISUALIZATION DATA
# ============================================================================

@router.get("/network/graph")
async def get_network_graph() -> Dict[str, Any]:
    """
    Get network data formatted for graph visualization (D3.js, vis.js, etc.)
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    centrality = network.get_centrality_metrics()
    
    # Build nodes
    nodes = []
    for bank_id, bank in network.banks.items():
        c = centrality.get(bank_id, {})
        pd = simulation_state._heuristic_pd(bank)
        
        nodes.append({
            "id": bank_id,
            "label": f"Bank {bank_id}",
            "tier": bank.tier,
            "status": bank.status.value,
            "equity": round(bank.balance_sheet.equity, 2),
            "capital_ratio": round(bank.capital_ratio, 4),
            
            # Visualization properties
            "size": 10 + c.get("eigenvector", 0) * 40,  # Size by systemic importance
            "color": _status_to_color(bank.status.value),
            "risk_intensity": min(pd * 2, 1.0)  # For heatmap coloring
        })
    
    # Build edges
    edges = []
    for u, v in network.graph.edges():
        weight = network.get_exposure(u, v)
        edges.append({
            "source": u,
            "target": v,
            "weight": round(weight, 2),
            "width": max(1, min(weight / 100, 10))  # Width for visualization
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "force"
    }


def _status_to_color(status: str) -> str:
    """Map bank status to color."""
    colors = {
        "active": "#4CAF50",
        "stressed": "#FF9800",
        "defaulted": "#F44336"
    }
    return colors.get(status, "#9E9E9E")


# ============================================================================
# CREDIT RISK
# ============================================================================

@router.get("/credit-risk")
async def get_system_credit_risk() -> Dict[str, Any]:
    """
    Get credit risk metrics for the entire system.
    
    Includes PD, LGD, EAD, EL for each bank and portfolio-level metrics.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    # Calculate per-bank credit risk
    bank_risks = {}
    total_el = 0
    pds = []
    
    for bank_id, bank in network.banks.items():
        risk = simulation_state.calculate_credit_risk(bank_id)
        
        pd = risk.get("probability_of_default", 0)
        lgd = risk.get("loss_given_default", 0.45)
        ead = risk.get("exposure_at_default", bank.balance_sheet.total_liabilities)
        el = pd * lgd * ead
        
        bank_risks[bank_id] = {
            "pd": round(pd, 4),
            "lgd": round(lgd, 4),
            "ead": round(ead, 2),
            "el": round(el, 2),
            "rating": risk.get("rating", "BBB")
        }
        
        total_el += el
        pds.append(pd)
    
    # Portfolio metrics
    avg_pd = np.mean(pds) if pds else 0
    pd_std = np.std(pds) if pds else 0
    
    # Simple VaR approximation (99th percentile)
    total_exposure = sum(b.balance_sheet.total_liabilities for b in network.banks.values())
    var_99 = total_exposure * (avg_pd + 2.33 * pd_std) * 0.45
    
    return {
        "per_bank": {int(k): v for k, v in bank_risks.items()},
        
        "portfolio": {
            "total_expected_loss": round(total_el, 2),
            "average_pd": round(avg_pd, 4),
            "pd_volatility": round(pd_std, 4),
            "total_exposure": round(total_exposure, 2),
            "value_at_risk_99": round(var_99, 2),
            "expected_shortfall": round(var_99 * 1.2, 2)  # Simplified
        },
        
        "rating_distribution": _get_rating_distribution(bank_risks),
        
        "high_risk_banks": [
            {"bank_id": int(k), **v} 
            for k, v in bank_risks.items() 
            if v["pd"] > 0.1
        ]
    }


def _get_rating_distribution(bank_risks: Dict) -> Dict[str, int]:
    """Count banks by rating."""
    distribution = {}
    for risk in bank_risks.values():
        rating = risk.get("rating", "BBB")
        distribution[rating] = distribution.get(rating, 0) + 1
    return distribution


@router.get("/credit-risk/{bank_id}")
async def get_bank_credit_risk(bank_id: int) -> Dict[str, Any]:
    """Get detailed credit risk for a specific bank."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if bank_id not in simulation_state.env.network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    risk = simulation_state.calculate_credit_risk(bank_id)
    bank = simulation_state.env.network.banks[bank_id]
    
    pd = risk.get("probability_of_default", 0)
    lgd = risk.get("loss_given_default", 0.45)
    ead = risk.get("exposure_at_default", bank.balance_sheet.total_liabilities)
    
    return {
        "bank_id": bank_id,
        "probability_of_default": round(pd, 4),
        "loss_given_default": round(lgd, 4),
        "exposure_at_default": round(ead, 2),
        "expected_loss": round(pd * lgd * ead, 2),
        "rating": risk.get("rating", "BBB"),
        "rating_outlook": "negative" if pd > 0.1 else ("stable" if pd > 0.02 else "positive"),
        
        "risk_factors": {
            "capital_ratio": round(bank.capital_ratio, 4),
            "leverage": round(bank.balance_sheet.total_assets / max(bank.balance_sheet.equity, 1), 2),
            "liquidity": round(bank.balance_sheet.cash / max(bank.balance_sheet.total_liabilities, 1), 4)
        }
    }


# ============================================================================
# CONTAGION SIMULATION
# ============================================================================

@router.post("/contagion/simulate")
async def simulate_contagion(
    shocked_bank: int = Query(..., description="Bank ID to shock"),
    shock_magnitude: float = Query(default=1.0, ge=0, le=1, description="Shock intensity (0-1)")
) -> Dict[str, Any]:
    """
    Simulate contagion from a single bank shock.
    
    Shows how defaults would cascade through the network.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    if shocked_bank not in network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {shocked_bank} not found")
    
    # Simple cascade simulation
    bank = network.banks[shocked_bank]
    initial_loss = bank.balance_sheet.equity * shock_magnitude
    
    # Track cascade
    defaulted = {shocked_bank}
    cascade_rounds = [[shocked_bank]]
    losses_per_round = [initial_loss]
    total_losses = initial_loss
    
    # Propagate for up to 10 rounds
    for round_num in range(10):
        new_defaults = []
        round_losses = 0
        
        for victim_id, victim in network.banks.items():
            if victim_id in defaulted:
                continue
            
            # Check exposure to defaulted banks
            exposure_loss = 0
            for def_id in defaulted:
                if def_id in victim.balance_sheet.interbank_assets:
                    exposure_loss += victim.balance_sheet.interbank_assets[def_id] * 0.55  # 45% recovery
            
            # Check if victim defaults
            if exposure_loss > victim.balance_sheet.equity * 0.5:
                new_defaults.append(victim_id)
                round_losses += victim.balance_sheet.equity
        
        if not new_defaults:
            break
        
        defaulted.update(new_defaults)
        cascade_rounds.append(new_defaults)
        losses_per_round.append(round_losses)
        total_losses += round_losses
    
    # Calculate system-wide impact
    total_equity = sum(b.balance_sheet.equity for b in network.banks.values())
    
    return {
        "shocked_bank": shocked_bank,
        "shock_magnitude": shock_magnitude,
        
        "cascade": {
            "depth": len(cascade_rounds),
            "total_defaults": len(defaulted),
            "rounds": cascade_rounds,
            "losses_per_round": [round(l, 2) for l in losses_per_round]
        },
        
        "impact": {
            "total_losses": round(total_losses, 2),
            "system_loss_fraction": round(total_losses / max(total_equity, 1), 4),
            "surviving_banks": len(network.banks) - len(defaulted)
        },
        
        "affected_banks": list(defaulted)
    }


# ============================================================================
# EXPOSURE MATRIX
# ============================================================================

@router.get("/exposure-matrix")
async def get_exposure_matrix() -> Dict[str, Any]:
    """
    Get the exposure matrix for heatmap visualization.
    
    Shows how much each bank owes to every other bank.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    bank_ids = sorted(network.banks.keys())
    n = len(bank_ids)
    
    # Build matrix
    matrix = [[0.0] * n for _ in range(n)]
    
    for i, row_id in enumerate(bank_ids):
        for j, col_id in enumerate(bank_ids):
            if i != j:
                exposure = network.get_exposure(row_id, col_id)
                matrix[i][j] = round(exposure, 2)
    
    # Find min/max for color scaling
    flat = [v for row in matrix for v in row]
    min_val = min(flat) if flat else 0
    max_val = max(flat) if flat else 1
    
    return {
        "name": "Interbank Exposure Matrix",
        "row_labels": [f"Bank {i}" for i in bank_ids],
        "col_labels": [f"Bank {i}" for i in bank_ids],
        "values": matrix,
        "min_value": min_val,
        "max_value": max_val,
        "color_scale": "YlOrRd"
    }
