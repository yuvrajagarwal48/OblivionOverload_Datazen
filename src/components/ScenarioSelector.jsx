import React, { useState } from 'react';
import useSimulationStore from '../store/simulationStore';
import { SCENARIOS, CUSTOM_OPTIONS } from '../config';
import { Scale, Snowflake, Flame, Swords, FlaskConical, Check, ChevronUp, ChevronDown } from 'lucide-react';
import './ScenarioSelector.css';

const ICON_MAP = {
  scale: Scale,
  snowflake: Snowflake,
  flame: Flame,
  swords: Swords,
};

/**
 * Scenario Selector \u2014 primary interaction on the main UI.
 */
export default function ScenarioSelector() {
  const selectedScenario = useSimulationStore((s) => s.selectedScenario);
  const setScenario = useSimulationStore((s) => s.setScenario);
  const customConfig = useSimulationStore((s) => s.customConfig);
  const setCustomConfig = useSimulationStore((s) => s.setCustomConfig);
  const simStatus = useSimulationStore((s) => s.simStatus);

  const [customExpanded, setCustomExpanded] = useState(false);

  const isDisabled = simStatus === 'running';

  return (
    <div className="scenario-selector">
      <h2 className="scenario-title">Select Scenario</h2>
      <p className="scenario-subtitle">Choose a stress test to simulate</p>

      <div className="scenario-cards">
        {SCENARIOS.map((scenario) => {
          const IconComp = ICON_MAP[scenario.iconName] || Scale;
          return (
            <button
              key={scenario.id}
              className={`scenario-card ${selectedScenario === scenario.id ? 'selected' : ''}`}
              style={{ '--accent': scenario.color }}
              onClick={() => !isDisabled && setScenario(scenario.id)}
              disabled={isDisabled}
            >
              <span className="scenario-icon">
                <IconComp size={20} />
              </span>
              <div className="scenario-card-text">
                <span className="scenario-name">{scenario.name}</span>
                <span className="scenario-sub">{scenario.subtitle}</span>
              </div>
              {selectedScenario === scenario.id && (
                <span className="scenario-check"><Check size={16} /></span>
              )}
            </button>
          );
        })}
      </div>

      {/* Custom Scenario (Advanced) */}
      <div className="custom-scenario">
        <button
          className={`custom-toggle ${customExpanded ? 'expanded' : ''}`}
          onClick={() => {
            setCustomExpanded(!customExpanded);
            if (!customExpanded) setScenario('custom');
          }}
          disabled={isDisabled}
        >
          <FlaskConical size={16} />
          <span>Custom Scenario</span>
          <span className="custom-arrow">{customExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
        </button>

        {customExpanded && (
          <div className="custom-panel">
            {Object.entries(CUSTOM_OPTIONS).map(([key, options]) => (
              <div className="custom-field" key={key}>
                <label className="custom-label">
                  {key === 'shockType' ? 'Shock Type' :
                   key === 'severity' ? 'Severity' :
                   key === 'target' ? 'Target' : 'Policy'}
                </label>
                <div className="custom-options">
                  {options.map((opt) => (
                    <button
                      key={opt.value}
                      className={`custom-option ${customConfig[key] === opt.value ? 'active' : ''}`}
                      onClick={() => setCustomConfig(key, opt.value)}
                      disabled={isDisabled}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
