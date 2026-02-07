import React from 'react';
import NetworkGraph from './NetworkGraph';
import ControlBar from './ControlBar';
import './NetworkPanel.css';

function NetworkPanel() {
  return (
    <div className="network-panel">
      <div className="network-view">
        <NetworkGraph />
      </div>
      <div className="network-controls">
        <ControlBar />
      </div>
    </div>
  );
}

export default NetworkPanel;
