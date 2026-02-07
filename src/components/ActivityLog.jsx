import React, { useEffect, useRef } from 'react';
import useSimulationStore from '../store/simulationStore';
import './ActivityLog.css';

/**
 * ActivityLog — Real-time scrollable feed of what every node and edge is doing.
 *
 * Reads from store.activityLog which is populated by:
 *   - ingestMetrics() — diffs previous vs current nodes → LEND, BORROW, HOARD, FIRE_SALE, DEFAULT, STRESSED
 *   - ingestEdges()   — diffs previous vs current edges → LENDING, REPAYMENT, NEW_LINK
 *   - mock events     — pushed directly in mock mode
 */
export default function ActivityLog() {
  const activityLog = useSimulationStore((s) => s.activityLog || []);
  const timestep = useSimulationStore((s) => s.timestep);
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activityLog.length]);

  // Group entries by timestep for visual clarity
  const grouped = {};
  activityLog.forEach((entry) => {
    const t = entry.timestep ?? 0;
    if (!grouped[t]) grouped[t] = [];
    grouped[t].push(entry);
  });

  const timesteps = Object.keys(grouped).map(Number).sort((a, b) => a - b);

  return (
    <div className="activity-log">
      <div className="activity-log-header">
        <h2 className="activity-log-title">📋 Live Activity Log</h2>
        <span className="activity-log-count">{activityLog.length} events</span>
        {timestep > 0 && <span className="activity-log-step">t = {timestep}</span>}
      </div>

      <div className="activity-log-feed" ref={scrollRef}>
        {activityLog.length === 0 ? (
          <div className="activity-log-empty">
            <p>No activity yet. Run the simulation to see node and edge actions.</p>
          </div>
        ) : (
          timesteps.map((t) => (
            <div key={t} className="activity-log-group">
              <div className="activity-log-step-divider">
                <span className="step-divider-line" />
                <span className="step-divider-label">Step {t}</span>
                <span className="step-divider-line" />
              </div>
              {grouped[t].map((entry) => (
                <div
                  key={entry.id}
                  className={`activity-log-entry type-${entry.type?.toLowerCase()}`}
                >
                  <span className="entry-icon">{entry.icon}</span>
                  <div className="entry-body">
                    <span className="entry-message">{entry.message}</span>
                    {entry.detail && (
                      <span className="entry-detail">{entry.detail}</span>
                    )}
                  </div>
                  <span
                    className="entry-badge"
                    style={{ background: entry.color || '#475569' }}
                  >
                    {entry.type?.replace(/_/g, ' ')}
                  </span>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Summary bar at bottom */}
      {activityLog.length > 0 && (
        <div className="activity-log-summary">
          <SummaryChip type="LEND" log={activityLog} color="#10b981" />
          <SummaryChip type="BORROW" log={activityLog} color="#3b82f6" />
          <SummaryChip type="HOARD" log={activityLog} color="#f59e0b" />
          <SummaryChip type="FIRE_SALE" log={activityLog} color="#f97316" />
          <SummaryChip type="DEFAULT" log={activityLog} color="#ef4444" />
          <SummaryChip type="LENDING" log={activityLog} color="#3b82f6" />
          <SummaryChip type="REPAYMENT" log={activityLog} color="#10b981" />
          <SummaryChip type="NEW_LINK" log={activityLog} color="#8b5cf6" />
        </div>
      )}
    </div>
  );
}

function SummaryChip({ type, log, color }) {
  const count = log.filter((e) => e.type === type).length;
  if (count === 0) return null;
  return (
    <span className="summary-chip" style={{ borderColor: color, color }}>
      {type.replace(/_/g, ' ')} × {count}
    </span>
  );
}
