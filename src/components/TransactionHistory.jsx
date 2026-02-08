import React, { useMemo, useState, useEffect } from 'react';
import { History, ArrowRightLeft, TrendingUp, TrendingDown, Link, RefreshCw } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import * as api from '../services/api';
import { USE_MOCK } from '../config';
import './TransactionHistory.css';

/**
 * TransactionHistory — Shows transaction history from backend API.
 * Uses /api/bank/{id}/transactions endpoint when available,
 * falls back to activity log for real-time edge activity.
 */
export default function TransactionHistory() {
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const activityLog = useSimulationStore((s) => s.activityLog || []);
  const [apiTransactions, setApiTransactions] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch transaction history from backend API
  const fetchTransactions = async () => {
    if (!currentBankId || USE_MOCK) return;
    
    setLoading(true);
    try {
      const data = await api.getBankTransactions(Number(currentBankId), 100);
      setApiTransactions(data);
    } catch (err) {
      console.warn('Failed to fetch transaction history:', err);
      setApiTransactions(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTransactions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentBankId]);

  // Filter activity log for edge transactions involving this bank (fallback)
  const activityTransactions = useMemo(() => {
    if (!currentBankId) return [];
    
    return activityLog
      .filter(entry => 
        entry.type === 'LENDING' || 
        entry.type === 'REPAYMENT' || 
        entry.type === 'NEW_LINK'
      )
      .filter(entry => {
        // Extract bank IDs from detail string (e.g., "Bank 2 → Bank 5")
        const match = entry.detail?.match(/Bank (\d+) → Bank (\d+)/);
        if (!match) return false;
        const [, source, target] = match;
        return source === String(currentBankId) || target === String(currentBankId);
      })
      .sort((a, b) => b.timestep - a.timestep) // Most recent first
      .slice(0, 50); // Show last 50 transactions
  }, [activityLog, currentBankId]);

  // Use API transactions if available, otherwise use activity log
  const transactions = apiTransactions?.transactions || activityTransactions;

  // Determine icon and direction for each transaction
  const getTransactionDetails = (entry) => {
    // Handle API transaction format
    if (entry.type && entry.counterparty !== undefined) {
      const isOutgoing = entry.direction === 'outflow';
      let icon = ArrowRightLeft;
      
      if (entry.type === 'loan' || entry.type === 'lending') icon = isOutgoing ? TrendingDown : TrendingUp;
      if (entry.type === 'repayment') icon = TrendingUp;
      if (entry.type === 'margin') icon = TrendingDown;
      
      return {
        icon,
        direction: entry.direction === 'outflow' ? 'Outgoing' : 'Incoming',
        counterparty: entry.counterparty >= 0 ? `Bank ${entry.counterparty}` : 'System',
        amount: entry.amount?.toFixed(2) || '0',
        timestep: entry.timestep || 0,
      };
    }

    // Handle activity log format (fallback)
    const match = entry.detail?.match(/Bank (\d+) → Bank (\d+)/);
    if (!match) return { icon: ArrowRightLeft, direction: '—', counterparty: '—', timestep: entry.timestep };
    
    const [, source, target] = match;
    const isOutgoing = source === String(currentBankId);
    
    let icon = ArrowRightLeft;
    if (entry.type === 'LENDING') icon = TrendingDown;
    if (entry.type === 'REPAYMENT') icon = TrendingUp;
    if (entry.type === 'NEW_LINK') icon = Link;
    
    return {
      icon,
      direction: isOutgoing ? 'Outgoing' : 'Incoming',
      counterparty: isOutgoing ? `Bank ${target}` : `Bank ${source}`,
      amount: entry.message?.match(/₹([\d.]+)/)?.[1] || '0',
      timestep: entry.timestep,
    };
  };

  return (
    <div className="txhist-panel">
      <div className="txhist-header">
        <h3 className="txhist-title">
          <History size={14} />
          Transaction History
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {!USE_MOCK && (
            <button 
              className="txhist-refresh-btn"
              onClick={fetchTransactions}
              disabled={loading}
              title="Refresh transactions"
            >
              <RefreshCw size={12} className={loading ? 'txhist-spin' : ''} />
            </button>
          )}
          <span className="txhist-count">
            {apiTransactions?.summary?.total_transactions || transactions.length} transactions
          </span>
        </div>
      </div>

      {apiTransactions?.summary && (
        <div className="txhist-summary">
          <div className="txhist-summary-item">
            <span className="txhist-summary-label">Total Inflows</span>
            <span className="txhist-summary-value" style={{ color: '#10b981' }}>
              ₹{apiTransactions.summary.total_inflows.toFixed(2)} Cr
            </span>
          </div>
          <div className="txhist-summary-item">
            <span className="txhist-summary-label">Total Outflows</span>
            <span className="txhist-summary-value" style={{ color: '#ef4444' }}>
              ₹{apiTransactions.summary.total_outflows.toFixed(2)} Cr
            </span>
          </div>
          <div className="txhist-summary-item">
            <span className="txhist-summary-label">Net Flow</span>
            <span 
              className="txhist-summary-value" 
              style={{ color: apiTransactions.summary.net_flow >= 0 ? '#10b981' : '#ef4444' }}
            >
              ₹{apiTransactions.summary.net_flow.toFixed(2)} Cr
            </span>
          </div>
        </div>
      )}

      {transactions.length === 0 ? (
        <div className="txhist-empty">
          {loading 
            ? 'Loading transaction history...'
            : 'No transactions yet. Transactions will appear here as the simulation runs.'}
        </div>
      ) : (
        <div className="txhist-table-wrap">
          <table className="txhist-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Type</th>
                <th>Direction</th>
                <th>Counterparty</th>
                <th>Amount</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((tx, i) => {
                const details = getTransactionDetails(tx);
                const Icon = details.icon;
                const displayType = tx.type?.toUpperCase() || 'TRANSACTION';
                return (
                  <tr key={tx.id || i}>
                    <td className="txhist-step">#{details.timestep}</td>
                    <td>
                      <span className={`txhist-type txhist-type-${displayType.toLowerCase()}`}>
                        <Icon size={12} />
                        {displayType === 'NEW_LINK' ? 'New Link' : displayType}
                      </span>
                    </td>
                    <td>
                      <span className={`txhist-direction ${details.direction.toLowerCase()}`}>
                        {details.direction}
                      </span>
                    </td>
                    <td className="txhist-cp">{details.counterparty}</td>
                    <td className="txhist-amount">₹{details.amount} Cr</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
