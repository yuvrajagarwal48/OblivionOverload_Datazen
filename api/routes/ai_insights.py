"""
AI Insights API Routes.
Endpoints for Gen AI analysis, report generation, and insights after simulation runs.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from .state import simulation_state

# Add project root for ai_insights imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_insights.src.data_aggregator_live import DataAggregatorLive, AggregatedMetrics
from ai_insights.src.gemini_analyzer import GeminiAnalyzer
from ai_insights.src.report_generator_genai import ReportGenerator

router = APIRouter(prefix="/ai-insights", tags=["AI Insights"])

# Module-level state
_last_analysis: Optional[Dict[str, Any]] = None
_last_metrics: Optional[Dict[str, Any]] = None


class AnalysisConfig(BaseModel):
    """Configuration for analysis generation."""
    focus: Optional[str] = None  # contagion, liquidity, leverage, regulation
    save_reports: bool = True
    output_dir: str = "outputs"


# ============================================================================
# GENERATE ANALYSIS
# ============================================================================

@router.post("/generate")
async def generate_ai_analysis(config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
    """
    Generate comprehensive AI analysis of the current simulation.
    
    This is the main endpoint — call after running simulation steps.
    Returns structured analysis with executive summary, risk scores,
    policy recommendations, and per-bank insights.
    
    The analysis uses Google Gemini API when available, with a robust
    structured fallback that still produces detailed professional output.
    """
    global _last_analysis, _last_metrics

    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not simulation_state.step_history:
        raise HTTPException(
            status_code=400,
            detail="No simulation steps recorded. Run /simulation/step or /simulation/run first."
        )

    config = config or AnalysisConfig()

    # Step 1: Aggregate metrics from live simulation
    aggregator = DataAggregatorLive(simulation_state)
    metrics = aggregator.aggregate()
    metrics_dict = _dataclass_to_dict(metrics)

    # Step 2: Generate AI analysis
    analyzer = GeminiAnalyzer()
    if config.focus:
        analysis = analyzer.analyze_focused(metrics_dict, config.focus)
    else:
        analysis = analyzer.analyze(metrics_dict)

    # Step 3: Optionally save reports
    saved_files = {}
    if config.save_reports:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate Markdown
        report_gen = ReportGenerator(metrics_dict, analysis.get("raw_text", ""), metrics.scenario_name)
        md_path = str(output_dir / "analysis_report.md")
        report_gen.generate_markdown_report(md_path)
        saved_files["markdown"] = md_path

        # Generate HTML
        html_path = str(output_dir / "analysis_report.html")
        report_gen.generate_html_report(html_path)
        saved_files["html"] = html_path

        # Export metrics JSON
        json_path = str(output_dir / "metrics.json")
        aggregator.export_to_json(json_path)
        saved_files["json"] = json_path

    # Store for later retrieval
    _last_analysis = analysis
    _last_metrics = metrics_dict

    return {
        "status": "success",
        "source": analysis.get("source", "unknown"),
        "analysis": analysis,
        "metrics_summary": {
            "num_banks": metrics.num_banks,
            "num_steps": metrics.num_steps,
            "scenario": metrics.scenario_name,
            "total_defaults": metrics.total_defaults,
            "total_stressed_max": metrics.total_stressed_max,
            "systemic_risk_index": round(metrics.systemic_risk_index, 4),
            "max_drawdown": round(metrics.max_drawdown, 4),
            "final_asset_price": round(metrics.final_asset_price, 4),
            "cascade_potential": metrics.cascade_potential,
        },
        "saved_files": saved_files,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# QUICK REPORT (text summary)
# ============================================================================

@router.get("/report/text", response_class=PlainTextResponse)
async def get_text_report() -> str:
    """
    Get the latest analysis as plain text.
    Quick endpoint for terminal/CLI consumption.
    """
    if _last_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis generated yet. POST /ai-insights/generate first.")

    return _last_analysis.get("raw_text", "No analysis text available.")


@router.get("/report/json")
async def get_json_report() -> Dict[str, Any]:
    """
    Get the complete latest analysis as structured JSON.
    Ideal for frontend consumption.
    """
    if _last_analysis is None or _last_metrics is None:
        raise HTTPException(status_code=404, detail="No analysis generated yet. POST /ai-insights/generate first.")

    return {
        "analysis": _last_analysis,
        "metrics": _last_metrics,
        "timestamp": _last_analysis.get("timestamp"),
    }


@router.get("/report/html", response_class=HTMLResponse)
async def get_html_report() -> str:
    """
    Get the latest analysis as a styled HTML report.
    Can be embedded in an iframe or opened directly.
    """
    if _last_analysis is None or _last_metrics is None:
        raise HTTPException(status_code=404, detail="No analysis generated yet. POST /ai-insights/generate first.")

    report_gen = ReportGenerator(
        _last_metrics,
        _last_analysis.get("raw_text", ""),
        _last_metrics.get("scenario_name", "unknown"),
    )
    return report_gen.generate_html_report()


# ============================================================================
# METRICS (raw data)
# ============================================================================

@router.get("/metrics")
async def get_aggregated_metrics() -> Dict[str, Any]:
    """
    Get raw aggregated metrics from the current simulation.
    Does NOT run AI analysis — just computes and returns all metrics.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not simulation_state.step_history:
        raise HTTPException(status_code=400, detail="No simulation steps recorded.")

    aggregator = DataAggregatorLive(simulation_state)
    metrics = aggregator.aggregate()

    return _dataclass_to_dict(metrics)


# ============================================================================
# RISK ASSESSMENT
# ============================================================================

@router.get("/risk-scores")
async def get_risk_scores() -> Dict[str, Any]:
    """
    Get risk assessment scores (0-100) for the current simulation.
    Quick endpoint for dashboard risk gauges.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not simulation_state.step_history:
        raise HTTPException(status_code=400, detail="No simulation steps recorded.")

    aggregator = DataAggregatorLive(simulation_state)
    metrics = aggregator.aggregate()
    metrics_dict = _dataclass_to_dict(metrics)

    analyzer = GeminiAnalyzer()
    risk = analyzer._build_risk_assessment(metrics_dict)

    return {
        "risk_scores": risk,
        "systemic_risk_index": round(metrics.systemic_risk_index, 4),
        "interpretation": {
            "0-25": "LOW — System healthy, normal operations",
            "25-50": "ELEVATED — Some stress, enhanced monitoring needed",
            "50-75": "SEVERE — Multiple failures, intervention recommended",
            "75-100": "CRITICAL — Systemic crisis, emergency measures required",
        },
        "current_status": risk["overall_rating"],
    }


# ============================================================================
# POLICY RECOMMENDATIONS
# ============================================================================

@router.get("/recommendations")
async def get_policy_recommendations() -> Dict[str, Any]:
    """
    Get policy recommendations based on latest analysis.
    Returns actionable items with priority levels and expected impact.
    """
    if _last_analysis is None:
        # Generate fresh if none cached
        if not simulation_state.is_initialized() or not simulation_state.step_history:
            raise HTTPException(status_code=400, detail="No analysis or simulation available.")

        aggregator = DataAggregatorLive(simulation_state)
        metrics_dict = _dataclass_to_dict(aggregator.aggregate())
        analyzer = GeminiAnalyzer()
        analysis = analyzer.analyze(metrics_dict)
        recs = analysis.get("policy_recommendations", [])
    else:
        recs = _last_analysis.get("policy_recommendations", [])

    return {
        "recommendations": recs,
        "count": len(recs),
        "high_priority": [r for r in recs if r.get("priority") == "HIGH"],
        "medium_priority": [r for r in recs if r.get("priority") == "MEDIUM"],
        "low_priority": [r for r in recs if r.get("priority") == "LOW"],
    }


# ============================================================================
# BANK INSIGHTS
# ============================================================================

@router.get("/bank-insights")
async def get_bank_insights() -> Dict[str, Any]:
    """
    Get per-bank insights from latest analysis.
    Returns status, risk notes, and DebtRank for each bank.
    """
    if _last_analysis is None:
        if not simulation_state.is_initialized() or not simulation_state.step_history:
            raise HTTPException(status_code=400, detail="No analysis or simulation available.")

        aggregator = DataAggregatorLive(simulation_state)
        metrics_dict = _dataclass_to_dict(aggregator.aggregate())
        analyzer = GeminiAnalyzer()
        analysis = analyzer.analyze(metrics_dict)
        insights = analysis.get("bank_level_insights", [])
    else:
        insights = _last_analysis.get("bank_level_insights", [])

    return {
        "insights": insights,
        "total_banks": len(insights),
        "defaulted": [b for b in insights if b.get("status") == "defaulted"],
        "at_risk": [b for b in insights if b.get("capital_ratio", 1) < 0.08 and b.get("status") != "defaulted"],
        "healthy": [b for b in insights if b.get("capital_ratio", 0) >= 0.10],
    }


# ============================================================================
# COMPARE SCENARIOS
# ============================================================================

@router.get("/compare")
async def get_comparison_data() -> Dict[str, Any]:
    """
    Returns current simulation metrics in a format suitable for
    multi-scenario comparison. Save results from multiple runs and
    compare them on the frontend.
    """
    if not simulation_state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not simulation_state.step_history:
        raise HTTPException(status_code=400, detail="No simulation steps recorded.")

    aggregator = DataAggregatorLive(simulation_state)
    metrics = aggregator.aggregate()

    return {
        "scenario": metrics.scenario_name,
        "num_banks": metrics.num_banks,
        "num_steps": metrics.num_steps,
        "total_defaults": metrics.total_defaults,
        "total_stressed_max": metrics.total_stressed_max,
        "systemic_risk_index": round(metrics.systemic_risk_index, 4),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "final_asset_price": round(metrics.final_asset_price, 4),
        "cascade_potential": metrics.cascade_potential,
        "price_volatility": round(metrics.price_volatility, 4),
        "avg_capital_ratio": round(metrics.avg_capital_ratio, 4),
        "liquidity_stress_events": metrics.liquidity_stress_events,
        "timestamp": metrics.timestamp,
    }


# ============================================================================
# HELPERS
# ============================================================================

def _dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dict, handling numpy types."""
    from dataclasses import asdict
    import numpy as np

    data = asdict(obj)

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): _convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert(v) for v in o]
        return o

    return _convert(data)
