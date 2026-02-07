"""
AI Insights Module - Gen AI Analysis for Financial Simulations
"""

from .featherless_analyzer import FeatherlessAnalyzer, FeatherlessAnalyzerFactory
from .data_aggregator_genai import DataAggregator, AggregatedMetrics
from .report_generator_genai import ReportGenerator

__all__ = [
    'FeatherlessAnalyzer',
    'FeatherlessAnalyzerFactory',
    'DataAggregator',
    'AggregatedMetrics',
    'ReportGenerator',
]
