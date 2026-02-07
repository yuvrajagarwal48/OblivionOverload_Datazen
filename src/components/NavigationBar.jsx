import React from 'react';
import useSimulationStore from '../store/simulationStore';
import { LayoutDashboard, Network, BarChart3, Building2, Zap } from 'lucide-react';
import './NavigationBar.css';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', Icon: LayoutDashboard },
  { id: 'network', label: 'Network', Icon: Network },
  { id: 'analytics', label: 'Analytics', Icon: BarChart3 },
  { id: 'inspector', label: 'Bank Inspector', Icon: Building2 },
];

function NavigationBar() {
  const activeView = useSimulationStore((state) => state?.activeView || 'overview');
  const setActiveView = useSimulationStore((state) => state?.setActiveView);
  const timestep = useSimulationStore((state) => state?.timestep || 0);
  const simStatus = useSimulationStore((state) => state?.simStatus || 'idle');

  return (
    <nav className="navigation-bar">
      <div className="nav-left">
        <div className="nav-logo">
          <Zap size={22} className="logo-icon" />
          <span className="logo-text">FinSim<span className="logo-accent">-MAPPO</span></span>
        </div>
        <div className="nav-divider" />
        <div className="nav-tabs">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-tab ${activeView === item.id ? 'active' : ''}`}
              onClick={() => setActiveView && setActiveView(item.id)}
            >
              <item.Icon size={16} className="nav-tab-icon" />
              <span className="nav-tab-label">{item.label}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="nav-right">
        <div className="nav-status">
          <span className={`status-indicator status-${simStatus}`}>
            <span className="status-dot"></span>
            {simStatus}
          </span>
          <div className="nav-timestep-container">
            <span className="nav-timestep-label">STEP</span>
            <span className="nav-timestep">{timestep}</span>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default NavigationBar;
