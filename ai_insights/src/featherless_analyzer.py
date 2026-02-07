"""
Gen AI Analysis using Google Gemini API
Provides comprehensive financial system analysis with deep insights
"""

import os
import json
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class FeatherlessAnalyzer:
    """
    Analyzer using Google Gemini API
    for comprehensive financial system analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini analyzer
        
        Args:
            api_key: Google Gemini API key (defaults to GOOGLE_GEMINI_API_KEY env var)
        """
        if not HAS_GEMINI:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GOOGLE_GEMINI_API_KEY environment variable or pass api_key parameter"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4000
            )
        )
    
    def analyze(self, metrics: Dict[str, Any]) -> str:
        """
        Generate comprehensive analysis using Gemini API
        
        Args:
            metrics: Aggregated metrics dictionary from simulation
        
        Returns:
            Analysis report as string
        """
        
        prompt = self._build_prompt(metrics)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e
    
    def analyze_with_focus(
        self,
        metrics: Dict[str, Any],
        focus: str
    ) -> str:
        """
        Generate analysis with specific focus area
        
        Args:
            metrics: Aggregated metrics dictionary
            focus: Focus area (contagion, liquidity, leverage, regulation, counterparty)
        
        Returns:
            Focused analysis report
        """
        
        focus_descriptions = {
            'contagion': (
                "Focus heavily on how defaults propagated through the network, "
                "identifying specific contagion chains and vulnerability points"
            ),
            'liquidity': (
                "Focus on liquidity stress, fire sales, and how banks struggled "
                "to meet payment obligations"
            ),
            'leverage': (
                "Focus on leverage ratios, over-leveraging behavior, and how "
                "excessive leverage amplified losses"
            ),
            'regulation': (
                "Focus on regulatory compliance, capital adequacy, and how "
                "regulatory frameworks performed"
            ),
            'counterparty': (
                "Focus on counterparty risk, bilateral exposures, and how "
                "concentrated exposures created systemic vulnerabilities"
            )
        }
        
        focus_detail = focus_descriptions.get(focus, focus)
        
        prompt = self._build_prompt(metrics)
        enhanced_prompt = f"{prompt}\n\n## ANALYSIS FOCUS\n{focus_detail}"
        
        try:
            response = self.model.generate_content(enhanced_prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e
    
    def comparative_analysis(
        self,
        metrics_list: list,
        scenario_names: list
    ) -> str:
        """
        Compare results across multiple scenarios
        
        Args:
            metrics_list: List of metrics dictionaries from different scenarios
            scenario_names: Names of scenarios being compared
        
        Returns:
            Comparative analysis
        """
        
        comparison_prompt = self._build_comparison_prompt(metrics_list, scenario_names)
        
        try:
            response = self.model.generate_content(comparison_prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e
    
    def _build_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build analysis prompt from metrics"""
        
        return f"""You are an expert financial systems analyst specializing in banking networks,
systemic risk, and financial regulation. Analyze this financial simulation with depth and rigor:

=== SIMULATION DATA ===
Scenario: {metrics.get('scenario_name', 'Unknown')}
Banks: {metrics.get('num_banks', 0)}
Duration: {metrics.get('num_steps', 0)} timesteps

=== OUTCOMES ===
Total Defaults: {metrics.get('total_defaults', 0)}
Peak Stressed: {metrics.get('total_stressed_max', 0)} banks ({100*metrics.get('total_stressed_max', 0)/max(1, metrics.get('num_banks', 1)):.1f}%)
Systemic Risk: {metrics.get('systemic_risk_index', 0):.1%}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}  
Cascade Potential: {metrics.get('cascade_potential', 0)}

=== MARKET ===
Initial Price: $1.0000
Final Price: ${metrics.get('final_asset_price', 0):.4f}
Volatility: {metrics.get('price_volatility', 0):.4f}

=== DEFAULTS ===
{json.dumps(metrics.get('default_events', [])[:10], indent=2)}

=== TOP SYSTEMIC BANKS ===
{json.dumps(metrics.get('top_10_debtrank', {}), indent=2)}

Provide detailed analysis covering:
1. Executive summary of what happened
2. Root causes of failures
3. How defaults spread (contagion mechanism)
4. Systemic vulnerabilities exposed
5. Policy recommendations with specific numbers
6. Risk assessment (0-100 scale)
"""
    
    def _build_comparison_prompt(
        self,
        metrics_list: list,
        scenario_names: list
    ) -> str:
        """Build comparison prompt"""
        
        scenarios_json = {}
        for name, metrics in zip(scenario_names, metrics_list):
            scenarios_json[name] = {
                'defaults': metrics.get('total_defaults', 0),
                'stressed': metrics.get('total_stressed_max', 0),
                'risk': metrics.get('systemic_risk_index', 0),
                'drawdown': metrics.get('max_drawdown', 0)
            }
        
        return f"""Compare these {len(scenario_names)} financial network scenarios:

{json.dumps(scenarios_json, indent=2)}

Rank by severity, explain differences, and recommend policy priorities.
"""


class FeatherlessAnalyzerFactory:
    """Factory for creating analyzer instances with different configurations"""
    
    @staticmethod
    def create_standard_analyzer(api_key: Optional[str] = None) -> FeatherlessAnalyzer:
        """Create standard analyzer"""
        return FeatherlessAnalyzer(api_key)
    
    @staticmethod
    def create_batch_analyzer(num_instances: int = 1) -> list:
        """Create multiple analyzer instances for batch processing"""
        return [FeatherlessAnalyzer() for _ in range(num_instances)]
    
    @staticmethod
    def create_focused_analyzers() -> Dict[str, FeatherlessAnalyzer]:
        """Create analyzers configured for different focus areas"""
        return {
            'contagion': FeatherlessAnalyzer(),
            'liquidity': FeatherlessAnalyzer(),
            'leverage': FeatherlessAnalyzer(),
            'regulation': FeatherlessAnalyzer(),
            'counterparty': FeatherlessAnalyzer()
        }
