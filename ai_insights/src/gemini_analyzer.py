"""
Improved Gen AI Analyzer with robust fallback analysis.
Supports Google Gemini API with structured fallback when API is unavailable.
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime


try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiAnalyzer:
    """
    AI-powered analysis engine for financial simulation results.
    Uses Google Gemini API when available, with comprehensive structured fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        self.model = None
        self.api_available = False

        if HAS_GEMINI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=4000,
                    ),
                )
                self.api_available = True
            except Exception:
                self.api_available = False

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis.

        Returns a structured dict with:
        - executive_summary: str
        - root_cause_analysis: str
        - contagion_analysis: str
        - risk_assessment: dict
        - policy_recommendations: list
        - bank_level_insights: list
        - raw_text: str (full narrative)
        - source: str ("gemini_api" or "structured_fallback")
        """
        if self.api_available:
            try:
                return self._analyze_with_gemini(metrics)
            except Exception as e:
                # Fall through to structured analysis
                pass

        return self._generate_structured_analysis(metrics)

    def analyze_focused(self, metrics: Dict[str, Any], focus: str) -> Dict[str, Any]:
        """Focused analysis on a specific area."""
        result = self.analyze(metrics)
        result["focus_area"] = focus
        return result

    # -------------------------------------------------------------------------
    # Gemini API path
    # -------------------------------------------------------------------------
    def _analyze_with_gemini(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(metrics)
        response = self.model.generate_content(prompt)
        text = response.text

        return {
            "executive_summary": self._extract_section(text, "Executive Summary", 500),
            "root_cause_analysis": self._extract_section(text, "Root Cause", 800),
            "contagion_analysis": self._extract_section(text, "Contagion", 600),
            "risk_assessment": self._build_risk_assessment(metrics),
            "policy_recommendations": self._extract_recommendations(text),
            "bank_level_insights": self._build_bank_insights(metrics),
            "raw_text": text,
            "source": "gemini_api",
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Structured fallback (no API needed)
    # -------------------------------------------------------------------------
    def _generate_structured_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis purely from metrics."""

        num_banks = metrics.get("num_banks", 0)
        num_defaults = metrics.get("total_defaults", 0)
        num_stressed = metrics.get("total_stressed_max", 0)
        systemic_risk = metrics.get("systemic_risk_index", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        final_price = metrics.get("final_asset_price", 1.0)
        cascade = metrics.get("cascade_potential", 0)
        scenario = metrics.get("scenario_name", "unknown")
        num_steps = metrics.get("num_steps", 0)
        price_vol = metrics.get("price_volatility", 0)
        avg_debtrank = metrics.get("average_debtrank", 0)
        liq_events = metrics.get("liquidity_stress_events", 0)

        default_rate = num_defaults / max(num_banks, 1)
        stress_rate = num_stressed / max(num_banks, 1)
        price_change = (final_price - 1.0) / 1.0 if final_price else 0

        # Determine severity
        if systemic_risk > 0.50:
            severity = "CRITICAL"
            severity_desc = "The financial system experienced a severe crisis with widespread failures."
        elif systemic_risk > 0.30:
            severity = "SEVERE"
            severity_desc = "The system was under significant stress with multiple institutions affected."
        elif systemic_risk > 0.15:
            severity = "ELEVATED"
            severity_desc = "Moderate stress was observed with isolated failures contained."
        else:
            severity = "LOW"
            severity_desc = "The system remained largely stable throughout the simulation."

        # Build executive summary
        exec_summary = (
            f"The {scenario} scenario simulation ran for {num_steps} timesteps across "
            f"{num_banks} banks. {severity_desc} "
            f"{num_defaults} bank(s) defaulted ({default_rate:.0%} default rate) and "
            f"up to {num_stressed} were simultaneously stressed ({stress_rate:.0%}). "
            f"The systemic risk index reached {systemic_risk:.1%}, with markets "
            f"experiencing a {max_drawdown:.1%} maximum drawdown. "
            f"Asset prices {'declined' if price_change < 0 else 'increased'} "
            f"by {abs(price_change):.1%} to ${final_price:.4f}."
        )

        # Root cause analysis
        root_causes = []
        if default_rate > 0.1:
            root_causes.append(
                "Insufficient capital buffers: Banks entered the stress period "
                "with inadequate equity relative to their exposures. When losses "
                "materialized, capital ratios quickly breached regulatory minimums."
            )
        if stress_rate > 0.3:
            root_causes.append(
                "Network interconnectedness: High bilateral exposures created "
                "contagion channels. The failure of systemically important institutions "
                "transmitted losses across the network."
            )
        if max_drawdown > 0.1:
            root_causes.append(
                "Market-driven losses: Asset price declines eroded bank balance sheets. "
                f"The {max_drawdown:.1%} drawdown reduced collateral values and triggered "
                "procyclical deleveraging."
            )
        if liq_events > num_steps * 0.3:
            root_causes.append(
                "Liquidity fragility: Persistent low liquidity conditions reduced banks' "
                "ability to meet obligations, forcing fire sales that amplified losses."
            )
        if not root_causes:
            root_causes.append(
                "The system demonstrated resilience. No critical failure mechanisms "
                "were activated during this scenario."
            )

        root_cause_text = "\n\n".join(f"**{i+1}. {rc}**" if i == 0 else f"**{i+1}.** {rc}"
                                       for i, rc in enumerate(root_causes))

        # Contagion analysis
        if cascade > 0:
            contagion_text = (
                f"Default contagion affected {cascade} banks through bilateral exposure "
                f"channels. The cascade propagated via interbank lending relationships "
                f"when creditor banks absorbed losses exceeding their equity buffers. "
                f"The average DebtRank of {avg_debtrank:.4f} indicates "
                f"{'high' if avg_debtrank > 0.1 else 'moderate' if avg_debtrank > 0.05 else 'low'} "
                f"systemic interconnectedness."
            )
        else:
            contagion_text = (
                "No significant contagion was observed. Default events, if any, "
                "were isolated and did not propagate through the network."
            )

        # Policy recommendations
        recommendations = []
        if default_rate > 0.05:
            recommendations.append({
                "priority": "HIGH",
                "area": "Capital Requirements",
                "action": "Increase minimum capital ratio from 8% to 10-12%",
                "rationale": f"With {default_rate:.0%} default rate, current buffers are insufficient",
                "expected_impact": "Estimated 30-50% reduction in default probability"
            })
        if stress_rate > 0.2:
            recommendations.append({
                "priority": "HIGH",
                "area": "Concentration Limits",
                "action": "Limit single-counterparty exposure to 15% of equity",
                "rationale": f"High stress rate ({stress_rate:.0%}) driven by concentrated exposures",
                "expected_impact": "Estimated 40-60% reduction in cascade severity"
            })
        if max_drawdown > 0.1:
            recommendations.append({
                "priority": "MEDIUM",
                "area": "Liquidity Coverage",
                "action": "Require 30-day high-quality liquid asset buffer",
                "rationale": f"Market drawdown of {max_drawdown:.1%} caused fire-sale spirals",
                "expected_impact": "20-30% reduction in fire-sale losses"
            })
        if avg_debtrank > 0.1:
            recommendations.append({
                "priority": "MEDIUM",
                "area": "SIFI Surcharges",
                "action": "Additional 2-3% capital surcharge for systemically important banks",
                "rationale": f"Average DebtRank of {avg_debtrank:.4f} indicates high interconnectedness",
                "expected_impact": "Reduced contagion from hub-bank failures"
            })
        recommendations.append({
            "priority": "LOW",
            "area": "Stress Testing",
            "action": "Run quarterly multi-scenario stress tests",
            "rationale": "Ongoing monitoring essential for early warning",
            "expected_impact": "Better preparedness and faster regulatory response"
        })

        # Bank-level insights
        bank_insights = self._build_bank_insights(metrics)

        # Risk assessment
        risk_assessment = self._build_risk_assessment(metrics)

        # Build full narrative
        raw_text = (
            f"## Executive Summary\n\n{exec_summary}\n\n"
            f"## Severity: {severity}\n\n"
            f"## Root Cause Analysis\n\n{root_cause_text}\n\n"
            f"## Contagion Analysis\n\n{contagion_text}\n\n"
            f"## Risk Assessment\n\n"
            f"- Overall Score: {risk_assessment['overall_score']}/100\n"
            f"- Default Risk: {risk_assessment['default_risk_score']}/100\n"
            f"- Market Risk: {risk_assessment['market_risk_score']}/100\n"
            f"- Contagion Risk: {risk_assessment['contagion_risk_score']}/100\n"
            f"- Liquidity Risk: {risk_assessment['liquidity_risk_score']}/100\n\n"
            f"## Policy Recommendations\n\n"
        )
        for rec in recommendations:
            raw_text += (
                f"### [{rec['priority']}] {rec['area']}\n"
                f"- **Action:** {rec['action']}\n"
                f"- **Rationale:** {rec['rationale']}\n"
                f"- **Expected Impact:** {rec['expected_impact']}\n\n"
            )

        return {
            "executive_summary": exec_summary,
            "severity": severity,
            "root_cause_analysis": root_cause_text,
            "contagion_analysis": contagion_text,
            "risk_assessment": risk_assessment,
            "policy_recommendations": recommendations,
            "bank_level_insights": bank_insights,
            "raw_text": raw_text,
            "source": "structured_fallback",
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------
    def _build_risk_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute numeric risk scores (0-100)."""
        num_banks = max(metrics.get("num_banks", 1), 1)
        default_rate = metrics.get("total_defaults", 0) / num_banks
        stress_rate = metrics.get("total_stressed_max", 0) / num_banks
        drawdown = metrics.get("max_drawdown", 0)
        systemic = metrics.get("systemic_risk_index", 0)
        liq_events = metrics.get("liquidity_stress_events", 0)
        num_steps = max(metrics.get("num_steps", 1), 1)

        default_score = min(int(default_rate * 200), 100)
        market_score = min(int(drawdown * 300), 100)
        contagion_score = min(int(stress_rate * 150), 100)
        liquidity_score = min(int((liq_events / num_steps) * 200), 100)

        overall = int(0.3 * default_score + 0.25 * market_score + 0.25 * contagion_score + 0.2 * liquidity_score)

        return {
            "overall_score": min(overall, 100),
            "overall_rating": "CRITICAL" if overall >= 75 else "SEVERE" if overall >= 50 else "ELEVATED" if overall >= 25 else "LOW",
            "default_risk_score": default_score,
            "market_risk_score": market_score,
            "contagion_risk_score": contagion_score,
            "liquidity_risk_score": liquidity_score,
        }

    def _build_bank_insights(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate per-bank insights."""
        outcomes = metrics.get("bank_outcomes", {})
        debtrank = metrics.get("top_10_debtrank", {})
        insights = []

        for bank_id, outcome in sorted(outcomes.items(), key=lambda x: x[1].get("debtrank", 0), reverse=True)[:10]:
            status = outcome.get("final_status", "unknown")
            cap_ratio = outcome.get("final_capital_ratio", 0)

            if status == "defaulted":
                note = "Defaulted during simulation — contributed to systemic contagion."
            elif cap_ratio < 0.08:
                note = "Near-default: capital ratio below regulatory minimum. Immediate intervention needed."
            elif cap_ratio < 0.10:
                note = "Under stress: capital ratio approaching danger zone. Enhanced monitoring recommended."
            else:
                note = "Healthy throughout simulation."

            insights.append({
                "bank_id": int(bank_id) if not isinstance(bank_id, int) else bank_id,
                "tier": outcome.get("tier", 0),
                "status": status,
                "capital_ratio": round(cap_ratio, 4),
                "equity": outcome.get("final_equity", 0),
                "debtrank": outcome.get("debtrank", debtrank.get(bank_id, 0)),
                "insight": note,
            })

        return insights

    def _build_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build Gemini prompt."""
        return f"""You are an expert financial systems analyst. Analyze this simulation data:

=== SIMULATION ===
Scenario: {metrics.get('scenario_name', 'unknown')}
Banks: {metrics.get('num_banks', 0)} | Steps: {metrics.get('num_steps', 0)}

=== OUTCOMES ===
Defaults: {metrics.get('total_defaults', 0)} | Stressed: {metrics.get('total_stressed_max', 0)}
Systemic Risk: {metrics.get('systemic_risk_index', 0):.1%}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
Final Price: ${metrics.get('final_asset_price', 0):.4f}
Cascade Potential: {metrics.get('cascade_potential', 0)} banks

=== TOP SYSTEMIC BANKS (DebtRank) ===
{json.dumps(metrics.get('top_10_debtrank', {}), indent=2)}

=== DEFAULT EVENTS ===
{json.dumps(metrics.get('default_events', [])[:10], indent=2)}

Provide:
1. **Executive Summary** — What happened (2–3 sentences)
2. **Root Cause Analysis** — Why it happened
3. **Contagion Mechanism** — How defaults spread
4. **Policy Recommendations** — Specific, actionable (with expected impact)
5. **Risk Score** — 0-100 overall rating

Be precise, use numbers, avoid filler.
"""

    @staticmethod
    def _extract_section(text: str, heading: str, max_len: int = 500) -> str:
        """Pull a section from Gemini output by heading."""
        lower = text.lower()
        idx = lower.find(heading.lower())
        if idx == -1:
            return ""
        start = text.find("\n", idx)
        if start == -1:
            start = idx
        chunk = text[start:start + max_len].strip()
        return chunk

    @staticmethod
    def _extract_recommendations(text: str) -> List[Dict[str, str]]:
        """Pull recommendations from Gemini output."""
        recs = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("-", "•", "*")) and len(line) > 20:
                recs.append({"recommendation": line.lstrip("-•* ").strip()})
        return recs[:8]
