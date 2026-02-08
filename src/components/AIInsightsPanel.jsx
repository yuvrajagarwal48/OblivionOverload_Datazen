import React, { useState, useCallback } from 'react';
import { Brain, Loader2, AlertTriangle, ChevronDown, ChevronRight, Sparkles, ShieldAlert, Lightbulb, Building2 } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import * as api from '../services/api';
import './AIInsightsPanel.css';

/**
 * AIInsightsPanel — GenAI-powered analysis of the running simulation.
 *
 * Endpoints (on ai branch):
 *   POST /api/ai-insights/generate   → generate comprehensive analysis
 *   GET  /api/ai-insights/report/json → structured JSON report
 *   GET  /api/ai-insights/risk-scores → risk gauges (0-100)
 *   GET  /api/ai-insights/recommendations → policy recommendations
 *   GET  /api/ai-insights/bank-insights   → per-bank AI notes
 */
export default function AIInsightsPanel() {
  const simStatus = useSimulationStore((s) => s.simStatus);
  const timestep = useSimulationStore((s) => s.timestep);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [metricsSummary, setMetricsSummary] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    summary: true,
    risk: true,
    recommendations: false,
    banks: false,
  });

  const toggleSection = useCallback((section) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  }, []);

  const generateAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.generateAIAnalysis({});
      setAnalysis(result.analysis || {});
      setMetricsSummary(result.metrics_summary || {});
    } catch (err) {
      console.error('[AI Insights] Generation failed:', err);
      setError(err.message || 'Failed to generate AI analysis');
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getAIReport();
      setAnalysis(result.analysis || {});
      if (result.metrics) setMetricsSummary(result.metrics);
    } catch (err) {
      // If no report cached, generate fresh
      if (err.message?.includes('404')) {
        await generateAnalysis();
      } else {
        setError(err.message || 'Failed to fetch AI report');
      }
    } finally {
      setLoading(false);
    }
  }, [generateAnalysis]);

  const canGenerate = simStatus !== 'idle' && timestep > 0;

  // ─── Risk score bar ───
  const RiskBar = ({ label, score, maxScore = 100 }) => {
    const pct = Math.min(100, Math.max(0, (score / maxScore) * 100));
    const color = pct < 25 ? '#10b981' : pct < 50 ? '#f59e0b' : pct < 75 ? '#f97316' : '#ef4444';
    return (
      <div className="ai-risk-bar">
        <div className="ai-risk-bar-header">
          <span className="ai-risk-bar-label">{label}</span>
          <span className="ai-risk-bar-value" style={{ color }}>{score?.toFixed?.(1) ?? score}</span>
        </div>
        <div className="ai-risk-bar-track">
          <div className="ai-risk-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
        </div>
      </div>
    );
  };

  // ─── Section header ───
  const SectionHeader = ({ id, icon: Icon, title, count }) => (
    <button className="ai-section-header" onClick={() => toggleSection(id)}>
      {expandedSections[id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      <Icon size={14} />
      <span>{title}</span>
      {count != null && <span className="ai-section-count">{count}</span>}
    </button>
  );

  // ─── Render ───
  return (
    <div className="ai-insights-panel">
      {/* Header */}
      <div className="ai-header">
        <div className="ai-header-title">
          <Brain size={16} />
          <span>AI Insights</span>
          {analysis?.source && (
            <span className="ai-source-badge">{analysis.source}</span>
          )}
        </div>
        <div className="ai-header-actions">
          {analysis && (
            <button className="ai-btn-secondary" onClick={refreshReport} disabled={loading}>
              Refresh
            </button>
          )}
          <button
            className="ai-btn-primary"
            onClick={generateAnalysis}
            disabled={loading || !canGenerate}
            title={!canGenerate ? 'Run at least 1 simulation step first' : 'Generate AI analysis'}
          >
            {loading ? <Loader2 size={14} className="ai-spinner" /> : <Sparkles size={14} />}
            <span>{loading ? 'Analyzing…' : analysis ? 'Re-analyze' : 'Generate Analysis'}</span>
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="ai-error">
          <AlertTriangle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Empty state */}
      {!analysis && !loading && !error && (
        <div className="ai-empty">
          <Brain size={32} className="ai-empty-icon" />
          <p>Run the simulation, then click <strong>Generate Analysis</strong> for AI-powered insights.</p>
          {!canGenerate && <p className="ai-empty-hint">Start the simulation and run at least one step first.</p>}
        </div>
      )}

      {/* Results */}
      {analysis && (
        <div className="ai-results">
          {/* Metrics summary row */}
          {metricsSummary && (
            <div className="ai-metrics-row">
              <div className="ai-metric-chip">
                <span className="ai-metric-label">Banks</span>
                <span className="ai-metric-value">{metricsSummary.num_banks}</span>
              </div>
              <div className="ai-metric-chip">
                <span className="ai-metric-label">Steps</span>
                <span className="ai-metric-value">{metricsSummary.num_steps}</span>
              </div>
              <div className="ai-metric-chip">
                <span className="ai-metric-label">Defaults</span>
                <span className="ai-metric-value ai-metric-danger">{metricsSummary.total_defaults}</span>
              </div>
              <div className="ai-metric-chip">
                <span className="ai-metric-label">SRI</span>
                <span className="ai-metric-value">{metricsSummary.systemic_risk_index?.toFixed(3)}</span>
              </div>
              <div className="ai-metric-chip">
                <span className="ai-metric-label">Scenario</span>
                <span className="ai-metric-value">{metricsSummary.scenario}</span>
              </div>
            </div>
          )}

          {/* Executive Summary */}
          <SectionHeader id="summary" icon={Sparkles} title="Executive Summary" />
          {expandedSections.summary && (
            <div className="ai-section-body">
              {analysis.executive_summary ? (
                <p className="ai-summary-text">{analysis.executive_summary}</p>
              ) : analysis.raw_text ? (
                <pre className="ai-raw-text">{analysis.raw_text.slice(0, 1500)}</pre>
              ) : (
                <p className="ai-muted">No summary available.</p>
              )}
            </div>
          )}

          {/* Risk Scores */}
          <SectionHeader id="risk" icon={ShieldAlert} title="Risk Assessment" />
          {expandedSections.risk && (
            <div className="ai-section-body">
              {analysis.risk_assessment ? (
                <>
                  {analysis.risk_assessment.overall_score != null && (
                    <RiskBar label="Overall Risk" score={analysis.risk_assessment.overall_score} />
                  )}
                  {analysis.risk_assessment.contagion_risk != null && (
                    <RiskBar label="Contagion Risk" score={analysis.risk_assessment.contagion_risk} />
                  )}
                  {analysis.risk_assessment.liquidity_risk != null && (
                    <RiskBar label="Liquidity Risk" score={analysis.risk_assessment.liquidity_risk} />
                  )}
                  {analysis.risk_assessment.leverage_risk != null && (
                    <RiskBar label="Leverage Risk" score={analysis.risk_assessment.leverage_risk} />
                  )}
                  {analysis.risk_assessment.market_risk != null && (
                    <RiskBar label="Market Risk" score={analysis.risk_assessment.market_risk} />
                  )}
                  {analysis.risk_assessment.overall_rating && (
                    <div className={`ai-rating-badge rating-${analysis.risk_assessment.overall_rating.toLowerCase()}`}>
                      {analysis.risk_assessment.overall_rating}
                    </div>
                  )}
                </>
              ) : (
                <p className="ai-muted">No risk scores. Generate analysis first.</p>
              )}
            </div>
          )}

          {/* Policy Recommendations */}
          <SectionHeader
            id="recommendations"
            icon={Lightbulb}
            title="Policy Recommendations"
            count={analysis.policy_recommendations?.length}
          />
          {expandedSections.recommendations && (
            <div className="ai-section-body">
              {analysis.policy_recommendations?.length > 0 ? (
                <ul className="ai-rec-list">
                  {analysis.policy_recommendations.map((rec, i) => (
                    <li key={i} className={`ai-rec-item priority-${(rec.priority || 'medium').toLowerCase()}`}>
                      <div className="ai-rec-header">
                        <span className={`ai-priority-dot priority-${(rec.priority || 'medium').toLowerCase()}`} />
                        <span className="ai-rec-title">{rec.recommendation || rec.title || rec}</span>
                      </div>
                      {rec.rationale && <p className="ai-rec-rationale">{rec.rationale}</p>}
                      {rec.expected_impact && <p className="ai-rec-impact">Impact: {rec.expected_impact}</p>}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="ai-muted">No recommendations available.</p>
              )}
            </div>
          )}

          {/* Bank-Level Insights */}
          <SectionHeader
            id="banks"
            icon={Building2}
            title="Bank-Level Insights"
            count={analysis.bank_level_insights?.length}
          />
          {expandedSections.banks && (
            <div className="ai-section-body">
              {analysis.bank_level_insights?.length > 0 ? (
                <div className="ai-bank-grid">
                  {analysis.bank_level_insights.map((b, i) => (
                    <div key={i} className={`ai-bank-card status-${(b.status || 'active').toLowerCase()}`}>
                      <div className="ai-bank-card-header">
                        <span className="ai-bank-name">Bank {b.bank_id ?? i}</span>
                        <span className={`ai-bank-status status-${(b.status || 'active').toLowerCase()}`}>
                          {b.status || 'active'}
                        </span>
                      </div>
                      {b.capital_ratio != null && (
                        <div className="ai-bank-metric">
                          CAR: <strong>{(b.capital_ratio * 100).toFixed(1)}%</strong>
                        </div>
                      )}
                      {b.debtrank != null && (
                        <div className="ai-bank-metric">
                          DebtRank: <strong>{b.debtrank.toFixed(3)}</strong>
                        </div>
                      )}
                      {b.risk_notes && (
                        <p className="ai-bank-notes">{b.risk_notes}</p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="ai-muted">No bank-level insights available.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
