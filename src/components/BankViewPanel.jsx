import React from 'react';
import useSimulationStore from '../store/simulationStore';
import { capitalRatioToColor } from '../systems/cytoscapeStyles';
import './BankViewPanel.css';

/**
 * Bank View Panel — secondary read-only analysis.
 *
 * Shows after simulation starts, when user clicks nodes.
 *   - Multi-select bank list
 *   - Per-bank insights (capital ratio, stress, status, tier)
 *   - Sparkline history of capital_ratio / stress
 *   - Comparative view when 2+ banks selected
 *
 * Users CANNOT intervene — observation only.
 */
export default function BankViewPanel() {
  const selectedBanks = useSimulationStore((s) => s.selectedBanks);
  const nodes = useSimulationStore((s) => s.nodes);
  const bankHistory = useSimulationStore((s) => s.bankHistory);
  const deselectBank = useSimulationStore((s) => s.deselectBank);
  const clearBankSelection = useSimulationStore((s) => s.clearBankSelection);
  const bankViewOpen = useSimulationStore((s) => s.bankViewOpen);
  const toggleBankView = useSimulationStore((s) => s.toggleBankView);
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const loginAsBank = useSimulationStore((s) => s.loginAsBank);

  // Get full node data for selected banks
  const selectedNodes = selectedBanks
    .map((id) => nodes.find((n) => String(n.id) === id))
    .filter(Boolean);

  const hasSelection = selectedNodes.length > 0;

  // All bank nodes (not CCPs) for login list
  const allBankNodes = nodes.filter(n => (n.node_type || 'bank') !== 'ccp' && !String(n.id).startsWith('CCP'));

  return (
    <>
      {/* Toggle tab */}
      <button
        className={`bank-view-tab ${bankViewOpen ? 'open' : ''}`}
        onClick={toggleBankView}
      >
        🏦 Banks {hasSelection && `(${selectedNodes.length})`}
      </button>

      {/* Panel */}
      <div className={`bank-view-panel ${bankViewOpen ? 'open' : ''}`}>
        <div className="bank-view-header">
          <h3 className="bank-view-title">Bank Inspector</h3>
          {hasSelection && (
            <button className="bank-clear-btn" onClick={clearBankSelection}>
              Clear All
            </button>
          )}
        </div>

        {/* ── Login as Bank Section ── */}
        {allBankNodes.length > 0 && (
          <div className="bank-login-section">
            <h4 className="bank-login-section-title">🔑 Login as Bank</h4>
            <div className="bank-login-grid">
              {allBankNodes.map((node) => (
                <button
                  key={node.id}
                  className={`bank-login-grid-btn ${String(node.id) === String(currentBankId) ? 'active' : ''}`}
                  onClick={() => loginAsBank(String(node.id))}
                  title={`Login as Bank ${node.id}`}
                >
                  <span className="login-grid-dot" style={{ 
                    background: node.status === 'defaulted' ? '#ef4444' 
                      : node.status === 'active' ? '#10b981' : '#f59e0b'
                  }} />
                  <span className="login-grid-name">{node.label || `B${node.id}`}</span>
                </button>
              ))}
            </div>
            {currentBankId && (
              <div className="current-bank-badge">
                Viewing as: <strong>Bank {currentBankId}</strong>
              </div>
            )}
          </div>
        )}

        {/* ── Selected Bank Details ── */}
        {!hasSelection ? (
          <div className="bank-view-empty">
            <p>Click on nodes in the graph to inspect banks</p>
            <p className="bank-view-hint">Multi-select to compare</p>
          </div>
        ) : (
          <div className="bank-cards-list">
            {selectedNodes.map((node) => (
              <BankCard
                key={node.id}
                node={node}
                history={bankHistory[String(node.id)] || []}
                onRemove={() => deselectBank(node.id)}
              />
            ))}

            {/* Comparative table when 2+ banks selected */}
            {selectedNodes.length >= 2 && (
              <ComparisonTable nodes={selectedNodes} />
            )}
          </div>
        )}
      </div>
    </>
  );
}

/**
 * Individual bank card with insights
 */
function BankCard({ node, history, onRemove }) {
  const loginAsBank = useSimulationStore((s) => s.loginAsBank);
  
  const statusColor = node.status === 'defaulted'
    ? '#f85149'
    : node.status === 'active'
    ? '#3fb950'
    : '#e3b341';

  const handleLogin = () => {
    loginAsBank(String(node.id));
  };

  return (
    <div className="bank-card">
      <div className="bank-card-header">
        <div className="bank-card-id">
          <span
            className="bank-status-dot"
            style={{ background: statusColor }}
          />
          Bank {node.id}
        </div>
        <div className="bank-card-actions">
          <span className={`bank-tier tier-${node.tier}`}>
            Tier {node.tier}
          </span>
          <button 
            className="bank-login-btn" 
            onClick={handleLogin}
            title="Login as this bank"
          >
            🔑 Login
          </button>
          <button className="bank-remove-btn" onClick={onRemove}>
            ×
          </button>
        </div>
      </div>

      <div className="bank-metrics">
        {/* Capital Ratio */}
        <div className="bank-metric">
          <span className="bank-metric-label">Capital Ratio</span>
          <div className="bank-metric-row">
            <span
              className="bank-metric-value"
              style={{ color: capitalRatioToColor(node.capital_ratio) }}
            >
              {(node.capital_ratio * 100).toFixed(1)}%
            </span>
            <div className="mini-bar-track">
              <div
                className="mini-bar-fill"
                style={{
                  width: `${Math.min(node.capital_ratio / 0.2 * 100, 100)}%`,
                  background: capitalRatioToColor(node.capital_ratio),
                }}
              />
            </div>
          </div>
        </div>

        {/* Stress */}
        <div className="bank-metric">
          <span className="bank-metric-label">Stress Level</span>
          <div className="bank-metric-row">
            <span
              className="bank-metric-value"
              style={{
                color: node.stress > 0.6 ? '#f85149'
                     : node.stress > 0.3 ? '#e3b341'
                     : '#3fb950',
              }}
            >
              {(node.stress * 100).toFixed(1)}%
            </span>
            <div className="mini-bar-track">
              <div
                className="mini-bar-fill"
                style={{
                  width: `${node.stress * 100}%`,
                  background: node.stress > 0.6 ? '#f85149'
                            : node.stress > 0.3 ? '#e3b341'
                            : '#3fb950',
                }}
              />
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="bank-metric">
          <span className="bank-metric-label">Status</span>
          <span
            className="bank-status-badge"
            style={{
              color: statusColor,
              borderColor: statusColor,
              background: `${statusColor}15`,
            }}
          >
            {node.status}
          </span>
        </div>
      </div>

      {/* Mini sparkline for capital_ratio history */}
      {history.length > 1 && (
        <div className="bank-sparkline-container">
          <span className="bank-metric-label">Capital Ratio History</span>
          <Sparkline
            data={history.map((h) => h.capital_ratio)}
            color={capitalRatioToColor(node.capital_ratio)}
          />
        </div>
      )}

      {/* Mini sparkline for stress history */}
      {history.length > 1 && (
        <div className="bank-sparkline-container">
          <span className="bank-metric-label">Stress History</span>
          <Sparkline
            data={history.map((h) => h.stress)}
            color={node.stress > 0.6 ? '#f85149' : node.stress > 0.3 ? '#e3b341' : '#3fb950'}
          />
        </div>
      )}
    </div>
  );
}

/**
 * SVG Sparkline — lightweight, no charting library
 */
function Sparkline({ data, color = '#58a6ff', width = 200, height = 32 }) {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((val - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="sparkline-svg"
      preserveAspectRatio="none"
    >
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Last point dot */}
      {data.length > 0 && (
        <circle
          cx={(data.length - 1) / (data.length - 1) * width}
          cy={height - ((data[data.length - 1] - min) / range) * (height - 4) - 2}
          r="2.5"
          fill={color}
        />
      )}
    </svg>
  );
}

/**
 * Comparison table for 2+ selected banks
 */
function ComparisonTable({ nodes }) {
  return (
    <div className="comparison-section">
      <h4 className="comparison-title">⚖️ Comparison</h4>
      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              {nodes.map((n) => (
                <th key={n.id}>Bank {n.id}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Tier</td>
              {nodes.map((n) => (
                <td key={n.id}>
                  <span className={`bank-tier tier-${n.tier}`}>T{n.tier}</span>
                </td>
              ))}
            </tr>
            <tr>
              <td>Capital</td>
              {nodes.map((n) => (
                <td key={n.id} style={{ color: capitalRatioToColor(n.capital_ratio) }}>
                  {(n.capital_ratio * 100).toFixed(1)}%
                </td>
              ))}
            </tr>
            <tr>
              <td>Stress</td>
              {nodes.map((n) => (
                <td
                  key={n.id}
                  style={{
                    color: n.stress > 0.6 ? '#f85149'
                         : n.stress > 0.3 ? '#e3b341'
                         : '#3fb950',
                  }}
                >
                  {(n.stress * 100).toFixed(1)}%
                </td>
              ))}
            </tr>
            <tr>
              <td>Status</td>
              {nodes.map((n) => (
                <td
                  key={n.id}
                  style={{
                    color: n.status === 'defaulted' ? '#f85149'
                         : n.status === 'active' ? '#3fb950'
                         : '#e3b341',
                  }}
                >
                  {n.status}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
