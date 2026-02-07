import React from 'react';
import { Building2, Droplets, AlertTriangle, TrendingDown } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import { HEALTH_THRESHOLDS } from '../config';
import './SystemHealthPanel.css';

/**
 * System Health Panel — high-level metrics answering
 * "How bad is the system overall?"
 *
 * Metrics:
 *   - % Banks Defaulted
 *   - Total System Liquidity
 *   - System Risk Score (from equilibrium_score)
 *   - Volatility Index
 */
export default function SystemHealthPanel() {
  const metrics = useSimulationStore((s) => s.metrics);
  const nodes = useSimulationStore((s) => s.nodes);
  const timestep = useSimulationStore((s) => s.timestep);

  // Derive default percentage from nodes
  const totalBanks = nodes.length;
  const defaultedBanks = nodes.filter((n) => n.status === 'defaulted').length;
  const defaultPct = totalBanks > 0 ? defaultedBanks / totalBanks : 0;

  const cards = [
    {
      label: 'Banks Defaulted',
      value: totalBanks > 0 ? `${defaultedBanks} / ${totalBanks}` : '—',
      pct: defaultPct,
      severity: getDefaultSeverity(defaultPct),
      icon: Building2,
    },
    {
      label: 'System Liquidity',
      value: metrics.liquidity != null ? `${(metrics.liquidity * 100).toFixed(1)}%` : '—',
      pct: metrics.liquidity,
      severity: getSeverity(metrics.liquidity),
      icon: Droplets,
    },
    {
      label: 'Risk Score',
      value: metrics.equilibrium_score != null
        ? `${(metrics.equilibrium_score * 100).toFixed(1)}%`
        : '—',
      pct: metrics.equilibrium_score,
      severity: getSeverity(metrics.equilibrium_score),
      icon: AlertTriangle,
    },
    {
      label: 'Default Rate',
      value: metrics.default_rate != null
        ? `${(metrics.default_rate * 100).toFixed(1)}%`
        : '—',
      pct: metrics.default_rate != null ? 1 - metrics.default_rate : null,
      severity: getDefaultSeverity(metrics.default_rate),
      icon: TrendingDown,
    },
  ];

  return (
    <div className="health-panel">
      <div className="health-header">
        <h3 className="health-title">System Health</h3>
        {timestep > 0 && (
          <span className="health-step">t = {timestep}</span>
        )}
      </div>

      <div className="health-cards">
        {cards.map((card) => (
          <div
            className={`health-card severity-${card.severity}`}
            key={card.label}
          >
            <div className="health-card-top">
              <span className="health-icon"><card.icon size={18} /></span>
              <span className="health-label">{card.label}</span>
            </div>
            <div className="health-value">{card.value}</div>
            {card.pct != null && (
              <div className="health-bar-track">
                <div
                  className="health-bar-fill"
                  style={{ width: `${Math.max(0, Math.min(1, card.pct)) * 100}%` }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function getSeverity(value) {
  if (value == null) return 'neutral';
  if (value >= HEALTH_THRESHOLDS.good) return 'good';
  if (value >= HEALTH_THRESHOLDS.warning) return 'warning';
  return 'critical';
}

function getDefaultSeverity(value) {
  if (value == null) return 'neutral';
  if (value <= 0.05) return 'good';
  if (value <= 0.2) return 'warning';
  return 'critical';
}
