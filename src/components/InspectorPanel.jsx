import React from 'react';
import useSimulationStore from '../store/simulationStore';
import { AlertCircle, LogIn } from 'lucide-react';
import './InspectorPanel.css';

function Sparkline({ data }) {
  if (!data || data.length === 0) return null;
  
  const width = 120;
  const height = 32;
  const padding = 2;
  
  const maxVal = Math.max(...data.map((d) => d.capital_ratio || 0));
  const minVal = Math.min(...data.map((d) => d.capital_ratio || 0));
  const range = maxVal - minVal || 1;
  
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1 || 1)) * (width - 2 * padding) + padding;
    const y = height - padding - ((d.capital_ratio - minVal) / range) * (height - 2 * padding);
    return `${x},${y}`;
  }).join(' ');
  
  return (
    <svg width={width} height={height} className="sparkline-svg">
      <polyline
        points={points}
        fill="none"
        stroke="#3b82f6"
        strokeWidth="2"
      />
    </svg>
  );
}

function InspectorPanel() {
  const nodes = useSimulationStore((state) => state?.nodes || []);
  const selectedBanks = useSimulationStore((state) => state?.selectedBanks || []);
  const bankHistory = useSimulationStore((state) => state?.bankHistory || {});
  const backendInitialized = useSimulationStore((state) => state?.backendInitialized ?? false);
  const toggleBankSelection = useSimulationStore((state) => state?.selectBank);
  const deselectBank = useSimulationStore((state) => state?.deselectBank);
  const clearBankSelection = useSimulationStore((state) => state?.clearBankSelection);
  const loginAsBank = useSimulationStore((state) => state?.loginAsBank);
  const currentBankId = useSimulationStore((state) => state?.currentBankId);

  // Show placeholder if simulation not initialized
  if (!backendInitialized || nodes.length === 0) {
    return (
      <div className="panel-uninitialized">
        <AlertCircle size={64} className="panel-uninitialized-icon" />
        <div className="panel-uninitialized-title">No Banks Available</div>
        <div className="panel-uninitialized-text">
          Initialize the simulation first to inspect banks. Banks will appear here once the simulation starts.
        </div>
      </div>
    );
  }

  const handleBankClick = (bankId) => {
    if (selectedBanks.includes(String(bankId))) {
      deselectBank && deselectBank(bankId);
    } else {
      toggleBankSelection && toggleBankSelection(bankId);
    }
  };

  // Group banks by tier
  const banksByTier = nodes.reduce((acc, node) => {
    const tier = node.tier || 3;
    if (!acc[tier]) acc[tier] = [];
    acc[tier].push(node);
    return acc;
  }, {});

  return (
    <div className="inspector-panel">
      <div className="inspector-header">
        <h2 className="inspector-title">Bank Inspector</h2>
        <button 
          onClick={clearBankSelection}
          className="inspector-clear-btn"
          disabled={selectedBanks.length === 0}
        >
          Clear Selection ({selectedBanks.length})
        </button>
      </div>

      {/* ── Login as Bank Section ── */}
      {nodes.length > 0 && (
        <div className="inspector-login-section">
          <h3 className="inspector-section-title">🔑 Login as Bank</h3>
          <div className="inspector-login-grid">
            {nodes
              .filter(n => (n.node_type || 'bank') !== 'ccp' && !String(n.id).startsWith('CCP'))
              .map((node) => (
                <button
                  key={node.id}
                  className={`inspector-login-btn ${String(node.id) === String(currentBankId) ? 'active' : ''}`}
                  onClick={() => loginAsBank(String(node.id))}
                  title={`Login as ${node.label || `Bank ${node.id}`}`}
                >
                  <LogIn size={12} />
                  <span>{node.label || `B${node.id}`}</span>
                </button>
              ))}
          </div>
          {currentBankId && (
            <div className="inspector-current-bank">
              ✅ Viewing as: <strong>{nodes.find(n => String(n.id) === String(currentBankId))?.label || `Bank ${currentBankId}`}</strong>
            </div>
          )}
        </div>
      )}

      {/* Selected Banks Detail View */}
      {selectedBanks.length > 0 && (
        <div className="inspector-section">
          <h3 className="inspector-section-title">Selected Banks ({selectedBanks.length})</h3>
          <div className="selected-banks-grid">
            {selectedBanks.map((bankId) => {
              const bank = nodes.find((n) => String(n.id) === String(bankId));
              if (!bank) return null;
              
              const history = bankHistory[bankId] || [];
              
              return (
                <div key={bankId} className="inspector-bank-card selected">
                  <div className="bank-card-header">
                    <div className="bank-card-id">
                      <span className={`bank-status-dot status-${bank.status || 'active'}`}></span>
                      {bank.label || `Bank ${bank.id}`}
                    </div>
                    <button
                      onClick={() => deselectBank(bankId)}
                      className="bank-deselect-btn"
                      title="Remove from selection"
                    >
                      ✕
                    </button>
                  </div>
                  
                  <div className="bank-tier-badge">
                    <span className={`tier-badge tier-${bank.tier}`}>
                      Tier {bank.tier}
                    </span>
                  </div>

                  <div className="bank-metrics">
                    <div className="bank-metric">
                      <div className="bank-metric-label">Capital Ratio</div>
                      <div className="bank-metric-row">
                        <div className="bank-metric-value" style={{ color: getColorForRatio(bank.capital_ratio) }}>
                          {((bank.capital_ratio || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="mini-bar-track">
                          <div 
                            className="mini-bar-fill"
                            style={{ 
                              width: `${(bank.capital_ratio || 0) * 100}%`,
                              background: getColorForRatio(bank.capital_ratio)
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    <div className="bank-metric">
                      <div className="bank-metric-label">Stress Level</div>
                      <div className="bank-metric-row">
                        <div className="bank-metric-value">
                          {((bank.stress || 0) * 100).toFixed(0)}%
                        </div>
                        <div className="mini-bar-track">
                          <div 
                            className="mini-bar-fill"
                            style={{ 
                              width: `${(bank.stress || 0) * 100}%`,
                              background: '#f59e0b'
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    <div className="bank-metric">
                      <div className="bank-metric-label">Status</div>
                      <span className={`bank-status-badge status-${bank.status || 'active'}`}>
                        {bank.status || 'active'}
                      </span>
                    </div>
                  </div>

                  {history.length > 0 && (
                    <div className="bank-sparkline-container">
                      <div className="bank-metric-label">Capital Ratio Trend</div>
                      <Sparkline data={history} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* All Banks by Tier */}
      {Object.entries(banksByTier)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([tier, banks]) => (
          <div key={tier} className="inspector-section">
            <h3 className="inspector-section-title">
              Tier {tier} Banks ({banks.length})
            </h3>
            <div className="banks-list">
              {banks.map((bank) => (
                <div
                  key={bank.id}
                  className={`inspector-bank-item ${selectedBanks.includes(String(bank.id)) ? 'selected' : ''}`}
                  onClick={() => handleBankClick(bank.id)}
                >
                  <div className="bank-item-left">
                    <span className={`bank-status-dot status-${bank.status || 'active'}`}></span>
                    <span className="bank-item-label">{bank.label || `Bank ${bank.id}`}</span>
                  </div>
                  <div className="bank-item-right">
                    <span 
                      className="bank-item-ratio"
                      style={{ color: getColorForRatio(bank.capital_ratio) }}
                    >
                      {((bank.capital_ratio || 0) * 100).toFixed(1)}%
                    </span>
                    <span className="bank-item-stress">
                      Stress: {((bank.stress || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

      {nodes.length === 0 && (
        <div className="inspector-empty">
          <p>No banks to inspect. Start a simulation to see bank data.</p>
        </div>
      )}
    </div>
  );
}

function getColorForRatio(ratio) {
  const r = ratio ?? 0.1;
  if (r >= 0.15) return '#10b981';
  if (r >= 0.08) return '#f59e0b';
  return '#ef4444';
}

export default InspectorPanel;
