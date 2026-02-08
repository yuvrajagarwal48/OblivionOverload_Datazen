import React, { useMemo, useEffect, useState, useRef } from 'react';
import useSimulationStore from '../store/simulationStore';
import { USE_MOCK } from '../config';
import * as api from '../services/api';
import WhatIfPanel from './WhatIfPanel';
import TransactionHistory from './TransactionHistory';
import { ArrowLeft } from 'lucide-react';
import './BankDashboard.css';

/**
 * BankDashboard — Restricted view for a single logged-in bank.
 *
 * API mode: fetches from GET /api/bank/{id}
 *           returns { bank_id, tier, status, balance_sheet, capital_ratios,
 *                     credit_risk, network_position, margin_status, flags }
 * Mock mode: reads from store nodes
 */
export default function BankDashboard() {
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const currentBankData = useSimulationStore((s) => s.currentBankData);
  const nodes = useSimulationStore((s) => s.nodes || []);
  const edges = useSimulationStore((s) => s.edges || []);
  const metrics = useSimulationStore((s) => s.metrics || {});
  const bankHistory = useSimulationStore((s) => s.bankHistory || {});
  const timestep = useSimulationStore((s) => s.timestep || 0);
  const simStatus = useSimulationStore((s) => s.simStatus);

  // API-fetched bank details (full balance sheet, credit risk, etc.)
  const [bankDetails, setBankDetails] = useState(null);
  const intervalRef = useRef(null);

  // Fetch bank details from API
  useEffect(() => {
    if (USE_MOCK || !currentBankId) return;
    if (simStatus !== 'running' && simStatus !== 'paused' && simStatus !== 'done') return;

    const fetchDetails = async () => {
      try {
        const data = await api.getBankDetails(Number(currentBankId));
        setBankDetails(data);
      } catch { /* silent */ }
    };

    fetchDetails();
    intervalRef.current = setInterval(fetchDetails, 2000);
    return () => clearInterval(intervalRef.current);
  }, [currentBankId, simStatus]);

  // Find our bank node from the simulation state (mock or ingested)
  const myNode = useMemo(() => {
    // In API mode, merge store node with API details
    const storeNode = nodes.find((n) => String(n.id) === String(currentBankId));
    if (!USE_MOCK && bankDetails) {
      const bs = bankDetails.balance_sheet || {};
      const cr = bankDetails.credit_risk || {};
      const np = bankDetails.network_position || {};
      const flags = bankDetails.flags || {};
      return {
        id: String(bankDetails.bank_id),
        name: storeNode?.name || storeNode?.label || '',
        tier: bankDetails.tier ?? storeNode?.tier,
        capital_ratio: bankDetails.capital_ratios?.current ?? bankDetails.capital_ratio ?? storeNode?.capital_ratio,
        stress: storeNode?.stress ?? 0,
        status: bankDetails.status ?? storeNode?.status,
        cash: bs.cash ?? storeNode?.cash,
        total_assets: bs.total_assets ?? storeNode?.total_assets,
        equity: bs.equity ?? storeNode?.equity,
        illiquid_assets: bs.illiquid_assets ?? storeNode?.illiquid_assets,
        external_liabilities: bs.external_liabilities ?? storeNode?.external_liabilities,
        interbank_assets: bs.interbank_assets ?? storeNode?.interbank_assets ?? {},
        interbank_liabilities: bs.interbank_liabilities ?? storeNode?.interbank_liabilities ?? {},
        debtrank: storeNode?.debtrank ?? 0,
        // Enriched fields from new API
        probability_of_default: cr.probability_of_default ?? null,
        expected_loss: cr.expected_loss ?? null,
        degree_centrality: np.degree_centrality ?? null,
        betweenness: np.betweenness_centrality ?? null,
        is_solvent: flags.is_solvent ?? bankDetails.is_solvent,
        is_liquid: flags.is_liquid ?? bankDetails.is_liquid,
        excess_cash: bs.excess_cash ?? bankDetails.excess_cash,
        centrality: bankDetails.centrality ?? np,
      };
    }
    return storeNode;
  }, [nodes, currentBankId, bankDetails]);

  // Find direct neighbors (connected via edges)
  const neighborIds = useMemo(() => {
    const ids = new Set();
    edges.forEach((e) => {
      const src = String(e.source ?? e.from);
      const tgt = String(e.target ?? e.to);
      const myId = String(currentBankId);
      if (src === myId) ids.add(tgt);
      if (tgt === myId) ids.add(src);
    });
    return [...ids];
  }, [edges, currentBankId]);

  const neighborNodes = useMemo(
    () => nodes.filter((n) => neighborIds.includes(String(n.id))),
    [nodes, neighborIds]
  );

  // Sparkline for own capital ratio
  const myHistory = bankHistory[String(currentBankId)] || [];

  const switchToObserver = useSimulationStore((s) => s.switchToObserver);

  const getHealthColor = (ratio) => {
    if (ratio == null) return '#64748b';
    if (ratio >= 0.08) return '#10b981';
    if (ratio >= 0.04) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="bank-dashboard">
      {/* ── Back to Observer ── */}
      <button className="bd-back-btn" onClick={switchToObserver}>
        <ArrowLeft size={14} />
        <span>Back to Inspector</span>
      </button>

      {/* ── Header ── */}
      <div className="bd-header">
        <div className="bd-bank-identity">
          <span className={`bd-tier-badge tier-${currentBankData?.tier || 2}`}>
            T{currentBankData?.tier || '?'}
          </span>
          <h2 className="bd-bank-name">{currentBankData?.name || `Bank ${currentBankId}`}</h2>
        </div>
        {timestep > 0 && (
          <div className="bd-timestep">Step {timestep}</div>
        )}
      </div>

      {/* ── Own Balance Sheet ── */}
      <section className="bd-section">
        <h3 className="bd-section-title">Your Balance Sheet</h3>
        {myNode ? (
          <>
            <div className="bd-metrics-grid">
              <MetricCard
                label="Capital Ratio"
                value={myNode.capital_ratio}
                format="percent"
                color={getHealthColor(myNode.capital_ratio)}
              />
              <MetricCard
                label="Cash"
                value={myNode.cash}
                format="currency"
              />
              <MetricCard
                label="Total Assets"
                value={myNode.total_assets}
                format="currency"
              />
              <MetricCard
                label="Equity"
                value={myNode.equity}
                format="currency"
              />
              <MetricCard
                label="Illiquid Assets"
                value={myNode.illiquid_assets}
                format="currency"
              />
              <MetricCard
                label="External Liabilities"
                value={myNode.external_liabilities}
                format="currency"
              />
              <MetricCard
                label="Stress Level"
                value={myNode.stress}
                format="percent"
                color={myNode.stress > 0.6 ? '#ef4444' : myNode.stress > 0.3 ? '#f59e0b' : '#10b981'}
              />
              <MetricCard
                label="Status"
                value={myNode.status || 'active'}
                format="status"
              />
              <MetricCard
                label="DebtRank"
                value={myNode.debtrank}
                format="percent"
                color={myNode.debtrank > 0.6 ? '#ef4444' : myNode.debtrank > 0.3 ? '#f59e0b' : '#10b981'}
              />
              {/* API-only fields from legacy endpoint */}
              {myNode.is_solvent != null && (
                <MetricCard
                  label="Solvent"
                  value={myNode.is_solvent ? 'Yes' : 'No'}
                  format="status"
                  color={myNode.is_solvent ? '#10b981' : '#ef4444'}
                />
              )}
              {myNode.is_liquid != null && (
                <MetricCard
                  label="Liquid"
                  value={myNode.is_liquid ? 'Yes' : 'No'}
                  format="status"
                  color={myNode.is_liquid ? '#10b981' : '#ef4444'}
                />
              )}
              {myNode.centrality != null && (
                <MetricCard
                  label="Centrality"
                  value={myNode.centrality}
                  format="decimal"
                />
              )}
            </div>

            {/* Interbank Exposures */}
            {(Object.keys(myNode.interbank_assets || {}).length > 0 || 
              Object.keys(myNode.interbank_liabilities || {}).length > 0) && (
              <div className="bd-interbank-section">
                <h4 className="bd-subsection-title">Interbank Exposures</h4>
                <div className="bd-interbank-grid">
                  {Object.keys(myNode.interbank_assets || {}).length > 0 && (
                    <div className="bd-interbank-col">
                      <span className="bd-interbank-label">Assets (Owed to me)</span>
                      {Object.entries(myNode.interbank_assets).map(([id, amount]) => (
                        <div key={id} className="bd-interbank-row">
                          <span>Bank {id}</span>
                          <span className="bd-interbank-amount">₹{amount.toFixed(0)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {Object.keys(myNode.interbank_liabilities || {}).length > 0 && (
                    <div className="bd-interbank-col">
                      <span className="bd-interbank-label">Liabilities (I owe)</span>
                      {Object.entries(myNode.interbank_liabilities).map(([id, amount]) => (
                        <div key={id} className="bd-interbank-row">
                          <span>Bank {id}</span>
                          <span className="bd-interbank-amount">₹{amount.toFixed(0)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="bd-empty">
            Waiting for simulation data…
          </div>
        )}

        {/* Mini Sparkline */}
        {myHistory.length > 1 && (
          <div className="bd-sparkline-container">
            <span className="bd-sparkline-label">Capital Ratio Trend</span>
            <MiniSparkline data={myHistory} />
          </div>
        )}
      </section>

      {/* ── Neighbors ── */}
      <section className="bd-section">
        <h3 className="bd-section-title">Direct Counterparties ({neighborNodes.length})</h3>
        {neighborNodes.length > 0 ? (
          <div className="bd-neighbor-list">
            {neighborNodes.map((n) => (
              <div key={n.id} className="bd-neighbor-chip">
                <span className={`bd-neighbor-dot status-${n.status || 'active'}`} />
                <span className="bd-neighbor-name">{n.label || `Bank ${n.id}`}</span>
                <span className={`bd-neighbor-tier tier-${n.tier}`}>T{n.tier}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="bd-empty">No direct connections yet</div>
        )}
      </section>

      {/* ── System Metrics (aggregated, no bank-level detail) ── */}
      <section className="bd-section">
        <h3 className="bd-section-title">System Metrics</h3>
        <div className="bd-system-metrics">
          <SystemMetric label="System Liquidity" value={metrics.liquidity} />
          <SystemMetric label="Default Rate" value={metrics.default_rate} />
          <SystemMetric label="Equilibrium" value={metrics.equilibrium_score} />
          <SystemMetric label="Volatility" value={metrics.volatility} />
        </div>
      </section>

      {/* ── What-If Panel ── */}
      <section className="bd-section">
        <WhatIfPanel neighborNodes={neighborNodes} />
      </section>

      {/* ── Transaction History ── */}
      <section className="bd-section">
        <TransactionHistory />
      </section>
    </div>
  );
}

/* ── Metric Card ── */
function MetricCard({ label, value, format, color }) {
  let display;
  if (value == null || value === undefined) {
    display = '—';
  } else if (format === 'percent') {
    display = `${(value * 100).toFixed(1)}%`;
  } else if (format === 'currency') {
    display = `₹${Number(value).toFixed(1)}`;
  } else if (format === 'decimal') {
    display = Number(value).toFixed(2);
  } else if (format === 'status') {
    display = value;
  } else {
    display = String(value);
  }

  return (
    <div className="bd-metric-card">
      <div className="bd-metric-label">{label}</div>
      <div className="bd-metric-value" style={color ? { color } : {}}>
        {display}
      </div>
    </div>
  );
}

/* ── System Metric Row ── */
function SystemMetric({ label, value }) {
  return (
    <div className="bd-sys-metric">
      <span className="bd-sys-label">{label}</span>
      <span className="bd-sys-value">
        {value != null ? (value * 100).toFixed(1) + '%' : '—'}
      </span>
    </div>
  );
}

/* ── Mini Sparkline ── */
function MiniSparkline({ data }) {
  const width = 200;
  const height = 36;
  const p = 2;

  const vals = data.map((d) => d.capital_ratio || 0);
  const max = Math.max(...vals);
  const min = Math.min(...vals);
  const range = max - min || 1;

  const points = vals
    .map((v, i) => {
      const x = (i / (vals.length - 1 || 1)) * (width - 2 * p) + p;
      const y = height - p - ((v - min) / range) * (height - 2 * p);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} className="bd-sparkline-svg">
      <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" />
    </svg>
  );
}
