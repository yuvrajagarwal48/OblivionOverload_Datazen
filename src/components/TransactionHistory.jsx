import React, { useState, useEffect, useCallback } from 'react';
import { History, RefreshCw, Check, X } from 'lucide-react';
import { fetchTransactions } from '../lib/supabase';
import useSimulationStore from '../store/simulationStore';
import './TransactionHistory.css';

/**
 * TransactionHistory — Shows Supabase-backed log of
 * all proposed transactions for the logged-in bank.
 */
export default function TransactionHistory() {
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (currentBankId == null) return;
    setLoading(true);
    setError(null);
    try {
      const rows = await fetchTransactions(currentBankId);
      setTransactions(rows || []);
    } catch (err) {
      // Supabase may not be configured; show empty
      console.warn('TransactionHistory: fetch failed', err.message);
      setError('Could not load history');
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  }, [currentBankId]);

  // Load on mount and when bank changes
  useEffect(() => {
    load();
  }, [load]);

  const formatDate = (iso) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) +
      ' ' + d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="txhist-panel">
      <div className="txhist-header">
        <h3 className="txhist-title">
          <History size={14} />
          Transaction History
        </h3>
        <button className="txhist-refresh-btn" onClick={load} disabled={loading} title="Refresh">
          <RefreshCw size={13} className={loading ? 'txhist-spin' : ''} />
        </button>
      </div>

      {error && <div className="txhist-error">{error}</div>}

      {transactions.length === 0 && !loading && !error ? (
        <div className="txhist-empty">No transactions recorded yet.</div>
      ) : (
        <div className="txhist-table-wrap">
          <table className="txhist-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Type</th>
                <th>Counterparty</th>
                <th>Amount</th>
                <th>Result</th>
                <th>Approved</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((tx, i) => (
                <tr key={tx.id || i}>
                  <td className="txhist-date">{formatDate(tx.created_at)}</td>
                  <td>
                    <span className={`txhist-type txhist-type-${(tx.tx_type || '').toLowerCase()}`}>
                      {tx.tx_type}
                    </span>
                  </td>
                  <td className="txhist-cp">Bank {tx.counterparty_id}</td>
                  <td className="txhist-amount">₹{Number(tx.amount || 0).toFixed(0)} Cr</td>
                  <td>
                    <span className={`txhist-outcome ${tx.outcome === 'PASS' ? 'pass' : 'fail'}`}>
                      {tx.outcome === 'PASS' ? <Check size={12} /> : <X size={12} />}
                      {tx.outcome}
                    </span>
                  </td>
                  <td>
                    {tx.approved ? (
                      <span className="txhist-approved yes">Yes</span>
                    ) : (
                      <span className="txhist-approved no">No</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
