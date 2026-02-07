import React, { useMemo } from 'react';
import { History, ArrowRightLeft, TrendingUp, TrendingDown, Link } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import './TransactionHistory.css';

/**
 * TransactionHistory — Shows real-time edge activity from the graph.
 * Displays lending, repayments, and new links involving the logged-in bank.
 */
export default function TransactionHistory() {
  const currentBankId = useSimulationStore((s) => s.currentBankId);
  const activityLog = useSimulationStore((s) => s.activityLog || []);

  // Filter activity log for edge transactions involving this bank
  const transactions = useMemo(() => {
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

  // Determine icon and direction for each transaction
  const getTransactionDetails = (entry) => {
    const match = entry.detail?.match(/Bank (\d+) → Bank (\d+)/);
    if (!match) return { icon: ArrowRightLeft, direction: '—', counterparty: '—' };
    
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
    };
  };

  return (
    <div className="txhist-panel">
      <div className="txhist-header">
        <h3 className="txhist-title">
          <History size={14} />
          Transaction History
        </h3>
        <span className="txhist-count">{transactions.length} transactions</span>
      </div>

      {transactions.length === 0 ? (
        <div className="txhist-empty">
          No transactions yet. Transactions will appear here as the simulation runs.
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
                return (
                  <tr key={tx.id || i}>
                    <td className="txhist-step">#{tx.timestep}</td>
                    <td>
                      <span className={`txhist-type txhist-type-${tx.type.toLowerCase()}`}>
                        <Icon size={12} />
                        {tx.type === 'NEW_LINK' ? 'New Link' : tx.type}
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
