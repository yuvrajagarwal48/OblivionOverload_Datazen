import React from 'react';
import {
  Settings,
  Activity,
  Building2,
  PanelBottom,
  PanelRight,
} from 'lucide-react';
import './ActivityBar.css';

const TOP_ITEMS = [
  { id: 'config', label: 'Configuration', Icon: Settings },
  { id: 'health', label: 'System Health', Icon: Activity },
];

function ActivityBar({
  active,
  sidebarOpen,
  onItemClick,
  rightPanelOpen,
  onToggleRight,
  bottomPanelOpen,
  onToggleBottom,
}) {
  return (
    <div className="activity-bar">
      <div className="activity-top">
        {TOP_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`activity-btn ${active === item.id && sidebarOpen ? 'active' : ''}`}
            onClick={() => onItemClick(item.id)}
            title={item.label}
          >
            <item.Icon size={20} />
          </button>
        ))}
      </div>
      <div className="activity-bottom">
        <button
          className={`activity-btn ${bottomPanelOpen ? 'active' : ''}`}
          onClick={onToggleBottom}
          title="Toggle Bottom Panel"
        >
          <PanelBottom size={20} />
        </button>
        <button
          className={`activity-btn ${rightPanelOpen ? 'active' : ''}`}
          onClick={onToggleRight}
          title="Toggle Inspector Panel"
        >
          <PanelRight size={20} />
        </button>
        <button
          className="activity-btn"
          onClick={() => onItemClick('config')}
          title="Bank Inspector"
        >
          <Building2 size={20} />
        </button>
      </div>
    </div>
  );
}

export default ActivityBar;
