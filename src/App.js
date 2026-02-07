import React, { useState } from 'react';
import useSimulationStore from './store/simulationStore';
import useWebSocket from './hooks/useWebSocket';
import LandingPage from './components/LandingPage';
import ActivityBar from './components/ActivityBar';
import ScenarioSelector from './components/ScenarioSelector';
import SystemHealthPanel from './components/SystemHealthPanel';
import ControlBar from './components/ControlBar';
import NetworkGraph from './components/NetworkGraph';
import AnalyticsPanel from './components/AnalyticsPanel';
import InspectorPanel from './components/InspectorPanel';
import { Zap } from 'lucide-react';
import './App.css';

/**
 * FinSim-MAPPO — IDE-Style Layout
 *
 * ┌──────────────────────────────────────────────────────┐
 * │  Title Bar (logo + controls + status)                │
 * ├────┬──────────┬──────────────────────┬───────────────┤
 * │    │          │                      │               │
 * │ A  │ Sidebar  │   Network Graph      │  Right Panel  │
 * │ c  │ (config/ │   (main editor)      │  (inspector)  │
 * │ t  │  health) │                      │               │
 * │    │          ├──────────────────────┤               │
 * │    │          │  Bottom Panel        │               │
 * │    │          │  (events/analytics)  │               │
 * └────┴──────────┴──────────────────────┴───────────────┘
 */
function App() {
  const showLanding = useSimulationStore((s) => s?.showLanding ?? true);
  const simStatus = useSimulationStore((s) => s?.simStatus || 'idle');
  const timestep = useSimulationStore((s) => s?.timestep || 0);

  const [activeSidebar, setActiveSidebar] = useState('config');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(true);

  // Activate WebSocket connection management
  useWebSocket();

  if (showLanding) {
    return <LandingPage />;
  }

  const handleActivityClick = (id) => {
    if (activeSidebar === id && sidebarOpen) {
      setSidebarOpen(false);
    } else {
      setActiveSidebar(id);
      setSidebarOpen(true);
    }
  };

  return (
    <div className="ide-app">
      {/* ── Title Bar ── */}
      <div className="ide-titlebar">
        <div className="titlebar-left">
          <div className="titlebar-logo">
            <Zap size={16} />
            <span>FinSim<span className="titlebar-accent">-MAPPO</span></span>
          </div>
        </div>
        <div className="titlebar-center">
          <ControlBar />
        </div>
        <div className="titlebar-right">
          <div className={`titlebar-status status-${simStatus}`}>
            <span className="status-dot" />
            <span>{simStatus}</span>
          </div>
          {timestep > 0 && (
            <div className="titlebar-step">
              <span className="step-label">STEP</span>
              <span className="step-value">{timestep}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Main Body ── */}
      <div className="ide-body">
        {/* Activity Bar */}
        <ActivityBar
          active={activeSidebar}
          sidebarOpen={sidebarOpen}
          onItemClick={handleActivityClick}
          rightPanelOpen={rightPanelOpen}
          onToggleRight={() => setRightPanelOpen(!rightPanelOpen)}
          bottomPanelOpen={bottomPanelOpen}
          onToggleBottom={() => setBottomPanelOpen(!bottomPanelOpen)}
        />

        {/* Left Sidebar */}
        {sidebarOpen && (
          <div className="ide-sidebar">
            <div className="sidebar-header">
              <span className="sidebar-title">
                {activeSidebar === 'config' ? 'CONFIGURATION' : 'SYSTEM HEALTH'}
              </span>
            </div>
            <div className="sidebar-content">
              {activeSidebar === 'config' ? (
                <ScenarioSelector />
              ) : (
                <SystemHealthPanel />
              )}
            </div>
          </div>
        )}

        {/* Center + Bottom */}
        <div className="ide-center">
          {/* Main Editor (Network Graph) */}
          <div className="ide-editor">
            <div className="editor-tab-bar">
              <div className="editor-tab active">
                <span className="editor-tab-dot" />
                <span>Network Graph</span>
              </div>
            </div>
            <div className="editor-content">
              <NetworkGraph />
            </div>
          </div>

          {/* Bottom Panel */}
          {bottomPanelOpen && (
            <div className="ide-bottom-panel">
              <div className="bottom-panel-tabs">
                <span className="bottom-tab active">Events & Analytics</span>
              </div>
              <div className="bottom-panel-content">
                <AnalyticsPanel />
              </div>
            </div>
          )}
        </div>

        {/* Right Panel (Inspector) */}
        {rightPanelOpen && (
          <div className="ide-right-panel">
            <div className="sidebar-header">
              <span className="sidebar-title">BANK INSPECTOR</span>
            </div>
            <div className="sidebar-content">
              <InspectorPanel />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
