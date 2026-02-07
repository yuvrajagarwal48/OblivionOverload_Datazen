"""
What-If / Counterfactual Analysis API Routes.
Endpoints for hypothetical transaction analysis and decision support.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import numpy as np
import copy
from datetime import datetime
import uuid

from .state import simulation_state
from .models import WhatIfTransactionRequest

router = APIRouter(prefix="/whatif", tags=["What-If Analysis"])


# ============================================================================
# COUNTERFACTUAL ANALYSIS
# ============================================================================

@router.post("/analyze")
async def analyze_transaction(request: WhatIfTransactionRequest) -> Dict[str, Any]:
    """
    Perform what-if analysis for a hypothetical transaction.
    
    Simulates:
    1. Baseline scenario (without transaction)
    2. Counterfactual scenario (with transaction)
    3. Compares outcomes across multiple simulations
    
    Returns risk assessment and recommendation.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    env = simulation_state.env
    network = env.network
    
    # Validate banks exist
    if request.initiator_id not in network.banks:
        raise HTTPException(status_code=404, detail=f"Initiator bank {request.initiator_id} not found")
    
    if request.counterparty_id is not None and request.counterparty_id not in network.banks:
        raise HTTPException(status_code=404, detail=f"Counterparty bank {request.counterparty_id} not found")
    
    analysis_id = f"WIF_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Get current state
    initiator = network.banks[request.initiator_id]
    counterparty = network.banks[request.counterparty_id] if request.counterparty_id else None
    
    # Run baseline simulation (without transaction)
    baseline_results = _run_monte_carlo(
        env=env,
        agents=simulation_state.agents,
        horizon=request.horizon,
        num_simulations=request.num_simulations,
        transaction=None
    )
    
    # Run counterfactual simulation (with transaction)
    transaction = {
        "type": request.transaction_type,
        "initiator_id": request.initiator_id,
        "counterparty_id": request.counterparty_id,
        "amount": request.amount,
        "interest_rate": request.interest_rate,
        "duration": request.duration,
        "collateral": request.collateral
    }
    
    cf_results = _run_monte_carlo(
        env=env,
        agents=simulation_state.agents,
        horizon=request.horizon,
        num_simulations=request.num_simulations,
        transaction=transaction
    )
    
    # Compute deltas
    deltas = {
        "initiator_equity_change": cf_results["initiator_equity"] - baseline_results["initiator_equity"],
        "initiator_capital_change": cf_results["initiator_capital"] - baseline_results["initiator_capital"],
        "system_defaults_change": cf_results["system_defaults"] - baseline_results["system_defaults"],
        "system_equity_change": cf_results["total_equity"] - baseline_results["total_equity"]
    }
    
    # Risk assessment
    risk_assessment = _compute_risk_assessment(
        baseline=baseline_results,
        counterfactual=cf_results,
        transaction=transaction,
        initiator=initiator,
        counterparty=counterparty
    )
    
    # Generate recommendation
    recommendation = _generate_recommendation(risk_assessment, deltas)
    
    return {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        
        # Transaction details
        "transaction": {
            "type": request.transaction_type,
            "initiator_id": request.initiator_id,
            "counterparty_id": request.counterparty_id,
            "amount": request.amount,
            "interest_rate": request.interest_rate,
            "duration": request.duration
        },
        
        # Baseline outcome (without transaction)
        "baseline": {
            "initiator_equity": round(baseline_results["initiator_equity"], 2),
            "initiator_capital_ratio": round(baseline_results["initiator_capital"], 4),
            "initiator_survives": baseline_results["initiator_survives"],
            "counterparty_equity": round(baseline_results.get("counterparty_equity", 0), 2),
            "counterparty_survives": baseline_results.get("counterparty_survives", True),
            "system_defaults": round(baseline_results["system_defaults"], 1),
            "total_equity": round(baseline_results["total_equity"], 2)
        },
        
        # Counterfactual outcome (with transaction)
        "counterfactual": {
            "initiator_equity": round(cf_results["initiator_equity"], 2),
            "initiator_capital_ratio": round(cf_results["initiator_capital"], 4),
            "initiator_survives": cf_results["initiator_survives"],
            "counterparty_equity": round(cf_results.get("counterparty_equity", 0), 2),
            "counterparty_survives": cf_results.get("counterparty_survives", True),
            "system_defaults": round(cf_results["system_defaults"], 1),
            "total_equity": round(cf_results["total_equity"], 2)
        },
        
        # Deltas
        "deltas": {k: round(v, 4) for k, v in deltas.items()},
        
        # Risk Assessment
        "risk_assessment": risk_assessment,
        
        # Recommendation
        "recommendation": recommendation,
        
        # Simulation info
        "simulation_info": {
            "horizon": request.horizon,
            "num_simulations": request.num_simulations,
            "confidence_level": 0.95
        }
    }


def _run_monte_carlo(
    env,
    agents: Dict,
    horizon: int,
    num_simulations: int,
    transaction: Optional[Dict]
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for baseline or counterfactual."""
    
    # Aggregators
    initiator_equities = []
    initiator_capitals = []
    counterparty_equities = []
    system_defaults_list = []
    total_equities = []
    initiator_survives_count = 0
    counterparty_survives_count = 0
    
    initiator_id = transaction["initiator_id"] if transaction else None
    counterparty_id = transaction.get("counterparty_id") if transaction else None
    
    for sim in range(num_simulations):
        # For simplicity, we'll run a single forward simulation
        # In production, you'd deep-copy the environment state
        
        # Simulate forward
        sim_result = _simulate_forward(env, agents, horizon, transaction, sim)
        
        initiator_equities.append(sim_result["initiator_equity"])
        initiator_capitals.append(sim_result["initiator_capital"])
        
        if counterparty_id:
            counterparty_equities.append(sim_result.get("counterparty_equity", 0))
            if sim_result.get("counterparty_survives", True):
                counterparty_survives_count += 1
        
        system_defaults_list.append(sim_result["system_defaults"])
        total_equities.append(sim_result["total_equity"])
        
        if sim_result["initiator_survives"]:
            initiator_survives_count += 1
    
    # Aggregate results
    return {
        "initiator_equity": np.mean(initiator_equities),
        "initiator_capital": np.mean(initiator_capitals),
        "initiator_survives": initiator_survives_count / num_simulations > 0.5,
        "initiator_survival_rate": initiator_survives_count / num_simulations,
        
        "counterparty_equity": np.mean(counterparty_equities) if counterparty_equities else 0,
        "counterparty_survives": counterparty_survives_count / max(num_simulations, 1) > 0.5,
        
        "system_defaults": np.mean(system_defaults_list),
        "total_equity": np.mean(total_equities),
        
        "std_defaults": np.std(system_defaults_list),
        "std_equity": np.std(total_equities)
    }


def _simulate_forward(
    env,
    agents: Dict,
    horizon: int,
    transaction: Optional[Dict],
    seed: int
) -> Dict[str, Any]:
    """Simulate forward from current state."""
    
    # Get current state as starting point
    network = env.network
    
    # Get current values
    initiator_id = transaction["initiator_id"] if transaction else list(network.banks.keys())[0]
    initiator = network.banks[initiator_id]
    
    counterparty_id = transaction.get("counterparty_id") if transaction else None
    counterparty = network.banks[counterparty_id] if counterparty_id else None
    
    # Simulate transaction effect
    if transaction:
        amount = transaction["amount"]
        tx_type = transaction["type"]
        
        # Simple simulation of transaction impact
        if tx_type == "loan_approval":
            # Initiator loses cash (lending)
            initiator_equity_delta = -amount * 0.01 * horizon  # Interest income
            if counterparty:
                counterparty_equity_delta = amount * 0.005 * horizon  # Benefit from loan
        elif tx_type == "margin_increase":
            initiator_equity_delta = -amount * 0.001 * horizon
            counterparty_equity_delta = 0
        else:
            initiator_equity_delta = 0
            counterparty_equity_delta = 0
    else:
        initiator_equity_delta = 0
        counterparty_equity_delta = 0
    
    # Add random noise
    np.random.seed(seed)
    noise = np.random.normal(0, 0.02)
    
    # Calculate final states
    final_initiator_equity = initiator.balance_sheet.equity * (1 + noise) + initiator_equity_delta
    final_initiator_capital = max(0.001, initiator.capital_ratio + noise * 0.1)
    
    final_counterparty_equity = 0
    if counterparty:
        final_counterparty_equity = counterparty.balance_sheet.equity * (1 + noise * 0.5) + counterparty_equity_delta
    
    # System-wide effects
    current_defaults = sum(1 for b in network.banks.values() if b.status.value == "defaulted")
    
    # Estimate additional defaults
    stressed = sum(1 for b in network.banks.values() if b.capital_ratio < 0.08)
    additional_defaults = int(stressed * 0.2 * np.random.random())
    
    if transaction and transaction["type"] == "loan_approval":
        # Loan might prevent counterparty default
        if counterparty and counterparty.capital_ratio < 0.08:
            if transaction["amount"] > counterparty.balance_sheet.total_liabilities * 0.1:
                additional_defaults = max(0, additional_defaults - 1)
    
    total_equity = sum(b.balance_sheet.equity for b in network.banks.values())
    
    return {
        "initiator_equity": final_initiator_equity,
        "initiator_capital": final_initiator_capital,
        "initiator_survives": final_initiator_equity > 0 and final_initiator_capital > 0.04,
        
        "counterparty_equity": final_counterparty_equity,
        "counterparty_survives": final_counterparty_equity > 0 if counterparty else True,
        
        "system_defaults": current_defaults + additional_defaults,
        "total_equity": total_equity * (1 + noise)
    }


def _compute_risk_assessment(
    baseline: Dict,
    counterfactual: Dict,
    transaction: Dict,
    initiator,
    counterparty
) -> Dict[str, Any]:
    """Compute comprehensive risk assessment."""
    
    # Default probabilities
    initiator_pd = 1 - counterfactual.get("initiator_survival_rate", 0.9)
    counterparty_pd = 1 - counterfactual.get("counterparty_survival_rate", 0.9) if counterparty else 0
    
    # Expected credit loss
    amount = transaction["amount"]
    lgd = 0.45  # Standard assumption
    ecl = initiator_pd * lgd * amount
    
    # Systemic impact
    defaults_increase = counterfactual["system_defaults"] - baseline["system_defaults"]
    
    # Cascade probability
    cascade_prob = min(0.5, max(0, defaults_increase * 0.1))
    
    # Liquidity impact
    liquidity_drain = amount * 0.1  # 10% of transaction ties up liquidity
    
    # Overall score (0-100, higher = riskier)
    score = (
        initiator_pd * 30 +
        counterparty_pd * 20 +
        cascade_prob * 25 +
        min(liquidity_drain / 1e6, 1) * 15 +
        min(ecl / 1e6, 1) * 10
    ) * 100
    
    score = min(100, max(0, score))
    
    return {
        "initiator_pd": round(initiator_pd, 4),
        "counterparty_pd": round(counterparty_pd, 4),
        "expected_credit_loss": round(ecl, 2),
        "loss_given_default": lgd,
        
        "system_impact": {
            "defaults_change": round(defaults_increase, 2),
            "cascade_probability": round(cascade_prob, 4),
            "contagion_depth": max(0, int(defaults_increase))
        },
        
        "liquidity_impact": {
            "liquidity_drain": round(liquidity_drain, 2),
            "margin_call_probability": round(min(initiator_pd * 2, 0.5), 4)
        },
        
        "overall_risk_score": round(score, 1),
        "risk_category": _score_to_category(score)
    }


def _score_to_category(score: float) -> str:
    """Convert risk score to category."""
    if score < 20:
        return "low"
    elif score < 40:
        return "moderate"
    elif score < 60:
        return "elevated"
    elif score < 80:
        return "high"
    else:
        return "very_high"


def _generate_recommendation(
    risk_assessment: Dict,
    deltas: Dict
) -> Dict[str, Any]:
    """Generate recommendation based on analysis."""
    
    score = risk_assessment["overall_risk_score"]
    category = risk_assessment["risk_category"]
    
    # Decision
    if score < 30:
        decision = "approve"
        confidence = 0.85
    elif score < 50:
        decision = "approve_with_conditions"
        confidence = 0.70
    elif score < 70:
        decision = "review_required"
        confidence = 0.55
    else:
        decision = "reject"
        confidence = 0.80
    
    # Reasoning
    reasons = []
    
    if risk_assessment["initiator_pd"] > 0.1:
        reasons.append("High initiator default probability")
    elif risk_assessment["initiator_pd"] < 0.02:
        reasons.append("Low initiator default probability")
    
    if risk_assessment["counterparty_pd"] > 0.1:
        reasons.append("High counterparty default probability")
    
    if deltas["system_defaults_change"] > 0:
        reasons.append(f"May trigger {abs(deltas['system_defaults_change']):.1f} additional defaults")
    elif deltas["system_defaults_change"] < -0.5:
        reasons.append("May prevent potential defaults")
    
    if deltas["system_equity_change"] > 0:
        reasons.append("Positive system equity impact")
    
    if risk_assessment["liquidity_impact"]["margin_call_probability"] > 0.2:
        reasons.append("Elevated margin call risk")
    
    # Conditions
    conditions = []
    if decision == "approve_with_conditions":
        if risk_assessment["initiator_pd"] > 0.05:
            conditions.append("Require additional collateral")
        if risk_assessment["liquidity_impact"]["liquidity_drain"] > 100000:
            conditions.append("Stagger transaction over multiple periods")
        conditions.append("Enhanced monitoring for 30 days")
    
    return {
        "decision": decision,
        "confidence": round(confidence, 2),
        "risk_category": category,
        "reasons": reasons,
        "conditions": conditions
    }


# ============================================================================
# QUICK ANALYSIS ENDPOINTS
# ============================================================================

@router.get("/quick-check/{bank_id}")
async def quick_risk_check(
    bank_id: int,
    amount: float = Query(..., description="Transaction amount"),
    transaction_type: str = Query(default="loan_approval")
) -> Dict[str, Any]:
    """Quick risk check for a proposed transaction."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = simulation_state.env.network
    
    if bank_id not in network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    bank = network.banks[bank_id]
    
    # Quick heuristic assessment
    pd = simulation_state._heuristic_pd(bank)
    
    # Impact estimates
    capital_impact = -amount / max(bank.balance_sheet.equity, 1) * 0.1
    new_capital_ratio = max(0, bank.capital_ratio + capital_impact)
    
    # Risk check
    if new_capital_ratio < 0.04:
        risk = "high"
        recommendation = "reject"
    elif new_capital_ratio < 0.08:
        risk = "elevated"
        recommendation = "review"
    elif pd > 0.1:
        risk = "moderate"
        recommendation = "approve_with_conditions"
    else:
        risk = "low"
        recommendation = "approve"
    
    return {
        "bank_id": bank_id,
        "transaction_type": transaction_type,
        "amount": amount,
        
        "current_state": {
            "equity": round(bank.balance_sheet.equity, 2),
            "capital_ratio": round(bank.capital_ratio, 4),
            "pd": round(pd, 4)
        },
        
        "post_transaction": {
            "estimated_capital_ratio": round(new_capital_ratio, 4),
            "capital_impact": round(capital_impact, 4)
        },
        
        "assessment": {
            "risk_level": risk,
            "recommendation": recommendation
        }
    }


@router.post("/compare-scenarios")
async def compare_scenarios(
    scenarios: List[WhatIfTransactionRequest]
) -> Dict[str, Any]:
    """Compare multiple hypothetical scenarios."""
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if len(scenarios) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 scenarios allowed")
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        # Simplified analysis for comparison
        network = simulation_state.env.network
        initiator = network.banks.get(scenario.initiator_id)
        counterparty = network.banks.get(scenario.counterparty_id) if scenario.counterparty_id else None
        
        if not initiator:
            results.append({"scenario_id": i, "error": "Invalid initiator"})
            continue
        
        pd = simulation_state._heuristic_pd(initiator)
        ecl = pd * 0.45 * scenario.amount
        
        capital_impact = -scenario.amount / max(initiator.balance_sheet.equity, 1) * 0.1
        new_capital = max(0, initiator.capital_ratio + capital_impact)
        
        results.append({
            "scenario_id": i,
            "transaction": {
                "type": scenario.transaction_type,
                "initiator": scenario.initiator_id,
                "counterparty": scenario.counterparty_id,
                "amount": scenario.amount
            },
            "risk_score": round(pd * 100 + (1 - new_capital / 0.12) * 50, 1),
            "expected_loss": round(ecl, 2),
            "post_capital_ratio": round(new_capital, 4)
        })
    
    # Rank scenarios
    results.sort(key=lambda x: x.get("risk_score", 100))
    
    return {
        "scenarios_analyzed": len(results),
        "results": results,
        "best_scenario": results[0]["scenario_id"] if results else None,
        "recommendation": f"Scenario {results[0]['scenario_id']} has lowest risk" if results else "No valid scenarios"
    }
