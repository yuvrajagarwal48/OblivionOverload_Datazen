import React, { useState, useCallback } from 'react';
import { FlaskConical, Check, X, Loader2, TrendingUp, TrendingDown } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import { insertTransaction } from '../lib/supabase';
import { API_BASE_URL, USE_MOCK } from '../config';
import './WhatIfPanel.css';

const TX_TYPES = [
  { value: 'LEND', label: 'Lend', color: '#10b981' },
  { value: 'BORROW', label: 'Borrow', color: '#3b82f6' },
  { value: 'SELL_ASSETS', label: 'Sell Assets', color: '#f59e0b' },
];

/**
 * WhatIfPanel — Transaction proposal + mini-simulation.
 * Flow:
 *   1. Select type + counterparty + amount
 *   2. "Run Simulation" → calls backend /what_if or mock evaluator
 *   3. Shows Pass/Fail + risk delta
 *   4. Approve → stores in Supabase; Reject → clears
 */
export default function WhatIfPanel({ neighborNodes = [] }) {
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const nodes = useSimulationStore((s) => s.nodes || []);

  const [txType, setTxType] = useState('LEND');
  const [counterparty, setCounterparty] = useState('');
  const [amount, setAmount] = useState('');
  const [simulating, setSimulating] = useState(false);
  const [result, setResult] = useState(null); // { pass, riskBefore, riskAfter, message }
  const [saving, setSaving] = useState(false);

  const resetForm = useCallback(() => {
    setResult(null);
    setTxType('LEND');
    setCounterparty('');
    setAmount('');
  }, []);

  const runSimulation = useCallback(async () => {
    if (!counterparty || !amount || Number(amount) <= 0) return;
    setSimulating(true);
    setResult(null);

    try {
      if (USE_MOCK) {
        // Mock what-if evaluator
        await new Promise((r) => setTimeout(r, 800 + Math.random() * 700));

        const myNode = nodes.find((n) => String(n.id) === String(currentBankId));
        const riskBefore = myNode?.capital_ratio ?? 0.08;
        const amountVal = Number(amount);
        let riskAfter;

        if (txType === 'LEND') {
          riskAfter = riskBefore - amountVal * 0.002;
        } else if (txType === 'BORROW') {
          riskAfter = riskBefore + amountVal * 0.001;
        } else {
          // SELL_ASSETS
          riskAfter = riskBefore + amountVal * 0.003;
        }

        riskAfter = Math.max(0, Math.min(1, riskAfter));
        const pass = riskAfter >= 0.04; // regulatory minimum

        setResult({
          pass,
          riskBefore,
          riskAfter,
          message: pass
            ? 'Transaction passes risk thresholds.'
            : 'Capital ratio falls below 4% — regulatory violation.',
        });
      } else {
        const res = await fetch(`${API_BASE_URL}/what_if`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            bank_id: currentBankId,
            tx_type: txType,
            counterparty: Number(counterparty),
            amount: Number(amount),
          }),
        });
        const data = await res.json();
        setResult(data);
      }
    } catch (err) {
      setResult({ pass: false, message: `Simulation error: ${err.message}` });
    } finally {
      setSimulating(false);
    }
  }, [counterparty, amount, txType, currentBankId, nodes]);

  const handleApprove = useCallback(async () => {
    setSaving(true);
    try {
      await insertTransaction({
        initiator_bank_id: currentBankId,
        tx_type: txType,
        counterparty_id: Number(counterparty),
        amount: Number(amount),
        outcome: result?.pass ? 'PASS' : 'FAIL',
        risk_before: result?.riskBefore,
        risk_after: result?.riskAfter,
        approved: true,
      });
    } catch (e) {
      console.warn('Supabase insert failed (may not be configured):', e.message);
    }
    resetForm();
    setSaving(false);
  }, [currentBankId, txType, counterparty, amount, result, resetForm]);

  const handleReject = useCallback(() => {
    resetForm();
  }, [resetForm]);

  return (
    <div className="whatif-panel">
      <h3 className="whatif-title">
        <FlaskConical size={14} />
        What-If Simulation
      </h3>

      {/* Form */}
      <div className="whatif-form">
        <div className="whatif-row">
          <label className="whatif-label">Type</label>
          <div className="whatif-type-btns">
            {TX_TYPES.map((t) => (
              <button
                key={t.value}
                className={`whatif-type-btn ${txType === t.value ? 'active' : ''}`}
                style={txType === t.value ? { borderColor: t.color, color: t.color } : {}}
                onClick={() => setTxType(t.value)}
                disabled={!!result}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="whatif-row">
          <label className="whatif-label">Counterparty</label>
          <select
            className="whatif-select"
            value={counterparty}
            onChange={(e) => setCounterparty(e.target.value)}
            disabled={!!result}
          >
            <option value="">Select bank…</option>
            {neighborNodes.map((n) => (
              <option key={n.id} value={n.id}>
                {n.label || `Bank ${n.id}`}
              </option>
            ))}
          </select>
        </div>

        <div className="whatif-row">
          <label className="whatif-label">Amount (₹ Cr)</label>
          <input
            type="number"
            className="whatif-input"
            placeholder="e.g. 500"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            min="1"
            disabled={!!result}
          />
        </div>

        {!result && (
          <button
            className="whatif-run-btn"
            onClick={runSimulation}
            disabled={!counterparty || !amount || simulating}
          >
            {simulating ? (
              <>
                <Loader2 size={14} className="whatif-spin" />
                Simulating…
              </>
            ) : (
              <>
                <FlaskConical size={14} />
                Run Simulation
              </>
            )}
          </button>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className={`whatif-result ${result.pass ? 'pass' : 'fail'}`}>
          <div className="whatif-result-header">
            {result.pass ? (
              <span className="whatif-result-badge pass"><Check size={14} /> PASS</span>
            ) : (
              <span className="whatif-result-badge fail"><X size={14} /> FAIL</span>
            )}
          </div>

          {result.riskBefore != null && (
            <div className="whatif-risk-delta">
              <div className="whatif-risk-item">
                <span className="whatif-risk-label">Before</span>
                <span className="whatif-risk-value">
                  {(result.riskBefore * 100).toFixed(1)}%
                </span>
              </div>
              <div className="whatif-risk-arrow">
                {result.riskAfter >= result.riskBefore ? (
                  <TrendingUp size={16} style={{ color: '#10b981' }} />
                ) : (
                  <TrendingDown size={16} style={{ color: '#ef4444' }} />
                )}
              </div>
              <div className="whatif-risk-item">
                <span className="whatif-risk-label">After</span>
                <span
                  className="whatif-risk-value"
                  style={{ color: result.riskAfter >= 0.04 ? '#10b981' : '#ef4444' }}
                >
                  {(result.riskAfter * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}

          <p className="whatif-result-msg">{result.message}</p>

          <div className="whatif-actions">
            <button
              className="whatif-approve-btn"
              onClick={handleApprove}
              disabled={saving}
            >
              <Check size={14} />
              {saving ? 'Saving…' : 'Approve & Record'}
            </button>
            <button className="whatif-reject-btn" onClick={handleReject}>
              <X size={14} />
              Reject
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
