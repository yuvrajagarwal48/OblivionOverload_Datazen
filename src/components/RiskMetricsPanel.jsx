import React from 'react';
import { AlertTriangle, TrendingUp, Network, Target } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import './RiskMetricsPanel.css';

/**
 * RiskMetricsPanel — Advanced systemic risk metrics
 * Shows DebtRank, cascade analysis, critical banks, network concentration
 */
export default function RiskMetricsPanel() {
  const advancedMetrics = useSimulationStore((s) => s.advancedMetrics);
  const bankDebtRanks = useSimulationStore((s) => s.bankDebtRanks);
  const nodes = useSimulationStore((s) => s.nodes);

  // Get top 5 banks by DebtRank
  const topDebtRanks = Object.entries(bankDebtRanks)
    .map(([id, rank]) => ({
      id,
      rank,
      node: nodes.find(n => String(n.id) === String(id)),
    }))
    .filter(item => item.node)
    .sort((a, b) => b.rank - a.rank)
    .slice(0, 5);

  return (
    <div className="risk-metrics-panel">
      <div className="risk-header">
        <h2 className="risk-title">Systemic Risk Metrics</h2>
      </div>

      {/* Main Risk Indicators */}
      <div className="risk-cards-grid">
        <RiskCard
          icon={AlertTriangle}
          label="Systemic Risk Index"
          value={advancedMetrics.systemic_risk_index}
          format="percent"
          severity={getRiskSeverity(advancedMetrics.systemic_risk_index)}
        />
        <RiskCard
          icon={TrendingUp}
          label="Aggregate DebtRank"
          value={advancedMetrics.aggregate_debtrank}
          format="decimal"
          severity={getRiskSeverity(advancedMetrics.aggregate_debtrank / (nodes.length || 1))}
        />
        <RiskCard
          icon={Network}
          label="Cascade Depth"
          value={advancedMetrics.cascade_depth}
          format="integer"
          severity={getCascadeSeverity(advancedMetrics.cascade_depth)}
        />
        <RiskCard
          icon={Target}
          label="Cascade Potential"
          value={advancedMetrics.cascade_potential}
          format="percent"
          severity={getRiskSeverity(advancedMetrics.cascade_potential)}
        />
      </div>

      {/* Network Concentration */}
      <div className="risk-section">
        <h3 className="risk-section-title">Network Metrics</h3>
        <div className="network-metrics-grid">
          <MetricRow label="Density" value={advancedMetrics.network_density} format="percent" />
          <MetricRow label="Clustering" value={advancedMetrics.clustering_coefficient} format="decimal" />
          <MetricRow label="Avg Path Length" value={advancedMetrics.avg_path_length} format="decimal" />
          <MetricRow label="Concentration Index" value={advancedMetrics.concentration_index} format="percent" />
        </div>
      </div>

      {/* Top DebtRank Banks */}
      <div className="risk-section">
        <h3 className="risk-section-title">
          Top 5 Systemically Important Banks (DebtRank)
        </h3>
        <div className="debtrank-list">
          {topDebtRanks.length > 0 ? (
            topDebtRanks.map((item, idx) => (
              <div key={item.id} className="debtrank-item">
                <div className="debtrank-rank">#{idx + 1}</div>
                <div className="debtrank-info">
                  <span className="debtrank-label">
                    {item.node.label || `Bank ${item.id}`}
                  </span>
                  <span className={`debtrank-tier tier-${item.node.tier}`}>
                    T{item.node.tier}
                  </span>
                </div>
                <div className="debtrank-bar-container">
                  <div
                    className="debtrank-bar"
                    style={{ width: `${item.rank * 100}%` }}
                  />
                </div>
                <div className="debtrank-value">{(item.rank * 100).toFixed(1)}%</div>
              </div>
            ))
          ) : (
            <div className="debtrank-empty">No data yet</div>
          )}
        </div>
      </div>

      {/* Critical Banks Alert */}
      {advancedMetrics.critical_banks?.length > 0 && (
        <div className="risk-section">
          <div className="critical-alert">
            <AlertTriangle size={16} />
            <span className="critical-label">
              {advancedMetrics.critical_banks.length} Critical Bank(s) Under Stress
            </span>
            <div className="critical-banks">
              {advancedMetrics.critical_banks.map(id => (
                <span key={id} className="critical-bank-badge">
                  Bank {id}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function RiskCard({ icon: Icon, label, value, format, severity }) {
  let display = '—';
  if (value != null) {
    if (format === 'percent') {
      display = `${(value * 100).toFixed(1)}%`;
    } else if (format === 'integer') {
      display = Math.round(value);
    } else {
      display = value.toFixed(2);
    }
  }

  return (
    <div className={`risk-card severity-${severity}`}>
      <div className="risk-card-header">
        <Icon size={18} />
        <span className="risk-card-label">{label}</span>
      </div>
      <div className="risk-card-value">{display}</div>
    </div>
  );
}

function MetricRow({ label, value, format }) {
  let display = '—';
  if (value != null) {
    if (format === 'percent') {
      display = `${(value * 100).toFixed(1)}%`;
    } else {
      display = value.toFixed(2);
    }
  }

  return (
    <div className="metric-row">
      <span className="metric-row-label">{label}</span>
      <span className="metric-row-value">{display}</span>
    </div>
  );
}

function getRiskSeverity(value) {
  if (value == null) return 'neutral';
  if (value >= 0.6) return 'critical';
  if (value >= 0.3) return 'warning';
  return 'good';
}

function getCascadeSeverity(depth) {
  if (depth == null || depth === 0) return 'good';
  if (depth >= 5) return 'critical';
  if (depth >= 3) return 'warning';
  return 'good';
}
