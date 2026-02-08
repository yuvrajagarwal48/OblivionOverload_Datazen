import React, { useState, useCallback, useRef } from 'react';
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
import BankDashboard from './components/BankDashboard';
import RiskMetricsPanel from './components/RiskMetricsPanel';
import MarketDataPanel from './components/MarketDataPanel';
import ActivityLog from './components/ActivityLog';
import AIInsightsPanel from './components/AIInsightsPanel';
import { Zap, LogOut, AlertCircle } from 'lucide-react';
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
  const restrictedMode = useSimulationStore((s) => s?.restrictedMode ?? false);
  const showLanding = useSimulationStore((s) => s?.showLanding ?? true);
  const simStatus = useSimulationStore((s) => s?.simStatus || 'idle');
  const timestep = useSimulationStore((s) => s?.timestep || 0);
  const currentBankData = useSimulationStore((s) => s?.currentBankData);
  const backendInitialized = useSimulationStore((s) => s?.backendInitialized ?? false);
  const logout = useSimulationStore((s) => s?.logout);
  const apiLoading = useSimulationStore((s) => s?.apiLoading ?? false);
  const apiError = useSimulationStore((s) => s?.apiError);
  const panelSizes = useSimulationStore((s) => s?.panelSizes || { leftSidebar: 280, rightPanel: 300, bottomPanel: 220 });
  const setPanelSize = useSimulationStore((s) => s?.setPanelSize);

  const [activeSidebar, setActiveSidebar] = useState('config');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(true);
  const [bottomTab, setBottomTab] = useState('activity'); // 'activity' | 'events' | 'risk' | 'market'
  
  // Resize state
  const [resizing, setResizing] = useState(null);
  const resizeStartRef = useRef({ x: 0, y: 0, size: 0 });

  const handleResizeStart = useCallback((panel, e) => {
    e.preventDefault();
    setResizing(panel);
    resizeStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      size: panelSizes[panel],
    };
  }, [panelSizes]);

  const handleResizeMove = useCallback((e) => {
    if (!resizing) return;
    
    const { x, y, size } = resizeStartRef.current;
    let newSize;
    
    if (resizing === 'leftSidebar') {
      newSize = Math.max(220, Math.min(500, size + (e.clientX - x)));
      setPanelSize('leftSidebar', newSize);
    } else if (resizing === 'rightPanel') {
      newSize = Math.max(240, Math.min(600, size - (e.clientX - x)));
      setPanelSize('rightPanel', newSize);
    } else if (resizing === 'bottomPanel') {
      newSize = Math.max(120, Math.min(window.innerHeight * 0.6, size - (e.clientY - y)));
      setPanelSize('bottomPanel', newSize);
    }
  }, [resizing, setPanelSize]);

  const handleResizeEnd = useCallback(() => {
    setResizing(null);
  }, []);

  // Attach global mouse handlers during resize
  React.useEffect(() => {
    if (resizing) {
      window.addEventListener('mousemove', handleResizeMove);
      window.addEventListener('mouseup', handleResizeEnd);
      return () => {
        window.removeEventListener('mousemove', handleResizeMove);
        window.removeEventListener('mouseup', handleResizeEnd);
      };
    }
  }, [resizing, handleResizeMove, handleResizeEnd]);

  // Activate WebSocket connection management
  useWebSocket();

  // Entry flow: Landing → Main App (bank login is now in the sidebar)
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
          {currentBankData && (
            <div className="titlebar-bank-name">
              <span className="bank-name-label">{currentBankData.name}</span>
            </div>
          )}
          {apiLoading && (
            <div className="titlebar-api-loading">
              <span className="api-spinner" />
              <span>API</span>
            </div>
          )}
          {apiError && (
            <div className="titlebar-api-error" title={apiError}>
              <span className="api-error-dot" />
              <span>API Error</span>
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
          <>
            <div className="ide-sidebar" style={{ width: `${panelSizes.leftSidebar}px` }}>
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
            <div 
              className="resize-handle-vertical"
              onMouseDown={(e) => handleResizeStart('leftSidebar', e)}
            />
          </>
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
            <>
              <div 
                className="resize-handle-horizontal"
                onMouseDown={(e) => handleResizeStart('bottomPanel', e)}
              />
              <div className="ide-bottom-panel" style={{ height: `${panelSizes.bottomPanel}px` }}>
                <div className="bottom-panel-tabs">
                  <button 
                    className={`bottom-tab ${bottomTab === 'activity' ? 'active' : ''}`}
                    onClick={() => setBottomTab('activity')}
                  >
                    Activity Log
                  </button>
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
                  <button 
                    className={`bottom-tab ${bottomTab === 'ai' ? 'active' : ''}`}
                    onClick={() => setBottomTab('ai')}
                  >
                    AI Insights
                  </button>
                </div>
                <div className="bottom-panel-content">
                  {bottomTab === 'activity' && <ActivityLog />}
                  {bottomTab === 'events' && <AnalyticsPanel />}
                  {bottomTab === 'risk' && <RiskMetricsPanel />}
                  {bottomTab === 'market' && <MarketDataPanel />}
                  {bottomTab === 'ai' && <AIInsightsPanel />}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Right Panel (Inspector or BankDashboard) */}
        {rightPanelOpen && (
          <>
            <div 
              className="resize-handle-vertical"
              onMouseDown={(e) => handleResizeStart('rightPanel', e)}
            />
            <div className="ide-right-panel" style={{ width: `${panelSizes.rightPanel}px` }}>
              <div className="sidebar-header">
                <span className="sidebar-title">
                  {restrictedMode ? 'MY BANK' : 'BANK INSPECTOR'}
                </span>
              </div>
              <div className="sidebar-content">
                {restrictedMode ? (
                  backendInitialized ? (
                    <BankDashboard />
                  ) : (
                    <div className="panel-uninitialized">
                      <AlertCircle size={64} className="panel-uninitialized-icon" />
                      <div className="panel-uninitialized-title">Simulation Not Initialized</div>
                      <div className="panel-uninitialized-text">
                        Please initialize the simulation from the Configuration panel to view your bank dashboard.
                      </div>
                    </div>
                  )
                ) : (
                  <InspectorPanel />
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
