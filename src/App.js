import React, { useState } from 'react';
import useSimulationStore from './store/simulationStore';
import useWebSocket from './hooks/useWebSocket';
import LoginPage from './components/LoginPage';
import LandingPage from './components/LandingPage';
import ActivityBar from './components/ActivityBar';
import ScenarioSelector from './components/ScenarioSelector';
import SystemHealthPanel from './components/SystemHealthPanel';
import ControlBar from './components/ControlBar';
import NetworkGraph from './components/NetworkGraph';
import AnalyticsPanel from './components/AnalyticsPanel';
import InspectorPanel from './components/InspectorPanel';
import BankDashboard from './components/BankDashboard';
import RiskMetricsPanel from './components/RiskMetricsPanel';
import MarketDataPanel from './components/MarketDataPanel';
import { Zap, LogOut } from 'lucide-react';
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
  const isAuthenticated = useSimulationStore((s) => s?.isAuthenticated ?? false);
  const restrictedMode = useSimulationStore((s) => s?.restrictedMode ?? false);
  const showLanding = useSimulationStore((s) => s?.showLanding ?? true);
  const simStatus = useSimulationStore((s) => s?.simStatus || 'idle');
  const timestep = useSimulationStore((s) => s?.timestep || 0);
  const currentBankData = useSimulationStore((s) => s?.currentBankData);
  const logout = useSimulationStore((s) => s?.logout);

  const [activeSidebar, setActiveSidebar] = useState('config');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(true);
  const [bottomTab, setBottomTab] = useState('events'); // 'events' | 'risk' | 'market'

  // Activate WebSocket connection management
  useWebSocket();

  // Entry flow: Landing → Login → Main App
  if (showLanding) {
    return <LandingPage />;
  }

  // Auth gate — show login after landing
  if (!isAuthenticated) {
    return <LoginPage />;
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
          {currentBankData && (
            <div className="titlebar-bank-name">
              <span className="bank-name-label">{currentBankData.name}</span>
            </div>
          )}
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
          <button className="titlebar-logout-btn" onClick={logout} title="Logout">
            <LogOut size={14} />
          </button>
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
                <button 
                  className={`bottom-tab ${bottomTab === 'events' ? 'active' : ''}`}
                  onClick={() => setBottomTab('events')}
                >
                  Events & Analytics
                </button>
                <button 
                  className={`bottom-tab ${bottomTab === 'risk' ? 'active' : ''}`}
                  onClick={() => setBottomTab('risk')}
                >
                  Risk Metrics
                </button>
                <button 
                  className={`bottom-tab ${bottomTab === 'market' ? 'active' : ''}`}
                  onClick={() => setBottomTab('market')}
                >
                  Market Data
                </button>
              </div>
              <div className="bottom-panel-content">
                {bottomTab === 'events' && <AnalyticsPanel />}
                {bottomTab === 'risk' && <RiskMetricsPanel />}
                {bottomTab === 'market' && <MarketDataPanel />}
              </div>
            </div>
          )}
        </div>

        {/* Right Panel (Inspector or BankDashboard) */}
        {rightPanelOpen && (
          <div className="ide-right-panel">
            <div className="sidebar-header">
              <span className="sidebar-title">
                {restrictedMode ? 'MY BANK' : 'BANK INSPECTOR'}
              </span>
            </div>
            <div className="sidebar-content">
              {restrictedMode ? <BankDashboard /> : <InspectorPanel />}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
