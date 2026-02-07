import React from 'react';
import { Building2, Coins, AlertTriangle, BarChart3 } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import './AnalyticsPanel.css';

function AnalyticsPanel() {
  const events = useSimulationStore((state) => state?.events || []);
  const metrics = useSimulationStore((state) => state?.metrics || {});
  const nodes = useSimulationStore((state) => state?.nodes || []);

  // Calculate statistics
  const defaultedCount = nodes.filter((n) => n.status === 'defaulted').length;
  const activeCount = nodes.length - defaultedCount;
  const avgCapitalRatio = nodes.length > 0
    ? nodes.reduce((sum, n) => sum + (n.capital_ratio || 0), 0) / nodes.length
    : 0;
  const avgStress = nodes.length > 0
    ? nodes.reduce((sum, n) => sum + (n.stress || 0), 0) / nodes.length
    : 0;

  // Event breakdown
  const eventBreakdown = events.reduce((acc, event) => {
    acc[event.event_type] = (acc[event.event_type] || 0) + 1;
    return acc;
  }, {});

  // Recent events (last 10)
  const recentEvents = [...events].slice(-10).reverse();

  return (
    <div className="analytics-panel">
      {/* Stats Cards */}
      <div className="analytics-section">
        <h2 className="section-title">System Statistics</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon"><Building2 size={20} /></div>
            <div className="stat-content">
              <div className="stat-label">Active Banks</div>
              <div className="stat-value">{activeCount}</div>
              <div className="stat-sub">{defaultedCount} defaulted</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><Coins size={20} /></div>
            <div className="stat-content">
              <div className="stat-label">Avg Capital Ratio</div>
              <div className="stat-value">{(avgCapitalRatio * 100).toFixed(1)}%</div>
              <div className="stat-sub">across all banks</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><AlertTriangle size={20} /></div>
            <div className="stat-content">
              <div className="stat-label">Avg Stress Level</div>
              <div className="stat-value">{(avgStress * 100).toFixed(1)}%</div>
              <div className="stat-sub">system pressure</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><BarChart3 size={20} /></div>
            <div className="stat-content">
              <div className="stat-label">System Liquidity</div>
              <div className="stat-value">{metrics.liquidity?.toFixed(2) ?? 'N/A'}</div>
              <div className="stat-sub">normalized metric</div>
            </div>
          </div>
        </div>
      </div>

      {/* Event Breakdown */}
      <div className="analytics-section">
        <h2 className="section-title">Event Analysis</h2>
        <div className="event-breakdown">
          <div className="breakdown-grid">
            {Object.entries(eventBreakdown).map(([type, count]) => (
              <div key={type} className="breakdown-item">
                <div className="breakdown-label">{type.replace(/_/g, ' ')}</div>
                <div className="breakdown-bar-container">
                  <div 
                    className={`breakdown-bar breakdown-${type.toLowerCase()}`}
                    style={{ width: `${(count / events.length) * 100}%` }}
                  ></div>
                </div>
                <div className="breakdown-count">{count}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Event Timeline */}
      <div className="analytics-section">
        <h2 className="section-title">Recent Events</h2>
        <div className="event-timeline">
          {recentEvents.length === 0 ? (
            <div className="timeline-empty">
              <p>No events yet. Start the simulation to see events.</p>
            </div>
          ) : (
            recentEvents.map((event, idx) => (
              <div key={idx} className="timeline-event">
                <div className="timeline-dot"></div>
                <div className="timeline-content">
                  <div className="timeline-header">
                    <span className={`timeline-badge badge-${event.event_type.toLowerCase()}`}>
                      {event.event_type.replace(/_/g, ' ')}
                    </span>
                    <span className="timeline-step">t={event.timestep}</span>
                  </div>
                  <div className="timeline-details">
                    {event.from && <span>From: <strong>{event.from}</strong></span>}
                    {event.to && <span>To: <strong>{event.to}</strong></span>}
                    {event.bank && <span>Bank: <strong>{event.bank}</strong></span>}
                    {event.amount && <span>Amount: {event.amount.toFixed(2)}</span>}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Metrics Summary */}
      <div className="analytics-section">
        <h2 className="section-title">Advanced Metrics</h2>
        <div className="metrics-table">
          <div className="metrics-row">
            <div className="metrics-label">Default Rate</div>
            <div className="metrics-value">{metrics.default_rate?.toFixed(3) ?? 'N/A'}</div>
          </div>
          <div className="metrics-row">
            <div className="metrics-label">Equilibrium Score</div>
            <div className="metrics-value">{metrics.equilibrium_score?.toFixed(3) ?? 'N/A'}</div>
          </div>
          <div className="metrics-row">
            <div className="metrics-label">Volatility</div>
            <div className="metrics-value">{metrics.volatility?.toFixed(3) ?? 'N/A'}</div>
          </div>
          <div className="metrics-row">
            <div className="metrics-label">Total Events</div>
            <div className="metrics-value">{events.length}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticsPanel;
