import React from 'react';
import ScenarioSelector from './ScenarioSelector';
import SystemHealthPanel from './SystemHealthPanel';
import ControlBar from './ControlBar';
import './OverviewPanel.css';

function OverviewPanel() {
  return (
    <div className="overview-panel">
      <div className="overview-grid">
        {/* Left Column */}
        <div className="overview-left">
          <div className="overview-section">
            <h2 className="overview-section-title">Scenario Configuration</h2>
            <ScenarioSelector />
          </div>
        </div>

        {/* Right Column */}
        <div className="overview-right">
          <div className="overview-section">
            <h2 className="overview-section-title">System Health</h2>
            <SystemHealthPanel />
          </div>
        </div>
      </div>

      {/* Bottom Control Bar */}
      <div className="overview-controls">
        <ControlBar />
      </div>
    </div>
  );
}

export default OverviewPanel;
