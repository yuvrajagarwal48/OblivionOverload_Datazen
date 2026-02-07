import React, { useState } from 'react';
import useSimulationStore from '../store/simulationStore';
import { Plus, Trash2, Building2, Shield, TrendingUp } from 'lucide-react';
import './CustomSetup.css';

/**
 * CustomSetup - Manual configuration for banks, CCPs, and market parameters
 */
export default function CustomSetup() {
  const manualConfig = useSimulationStore((s) => s.manualConfig);
  const setManualConfig = useSimulationStore((s) => s.setManualConfig);
  const simStatus = useSimulationStore((s) => s.simStatus);

  const [activeTab, setActiveTab] = useState('banks');
  const isDisabled = simStatus === 'running';

  const addBank = () => {
    const newBank = {
      id: `bank_${Date.now()}`,
      name: `Bank ${manualConfig.banks.length + 1}`,
      capital: 100,
      assets: 150,
      tier: 1,
    };
    setManualConfig('banks', [...manualConfig.banks, newBank]);
  };

  const removeBank = (id) => {
    setManualConfig('banks', manualConfig.banks.filter(b => b.id !== id));
  };

  const updateBank = (id, field, value) => {
    setManualConfig('banks', manualConfig.banks.map(b =>
      b.id === id ? { ...b, [field]: value } : b
    ));
  };

  const addCCP = () => {
    const newCCP = {
      id: `ccp_${Date.now()}`,
      name: `CCP ${manualConfig.ccps.length + 1}`,
      capital: 50,
      clearing_capacity: 200,
    };
    setManualConfig('ccps', [...manualConfig.ccps, newCCP]);
  };

  const removeCCP = (id) => {
    setManualConfig('ccps', manualConfig.ccps.filter(c => c.id !== id));
  };

  const updateCCP = (id, field, value) => {
    setManualConfig('ccps', manualConfig.ccps.map(c =>
      c.id === id ? { ...c, [field]: value } : c
    ));
  };

  const updateMarket = (field, value) => {
    setManualConfig('market', { ...manualConfig.market, [field]: value });
  };

  return (
    <div className="custom-setup">
      <div className="setup-header">
        <h2 className="setup-title">Custom Setup</h2>
        <p className="setup-subtitle">Configure banks, CCPs, and market parameters</p>
      </div>

      <div className="setup-tabs">
        <button
          className={`setup-tab ${activeTab === 'banks' ? 'active' : ''}`}
          onClick={() => setActiveTab('banks')}
          disabled={isDisabled}
        >
          <Building2 size={14} />
          <span>Banks ({manualConfig.banks.length})</span>
        </button>
        <button
          className={`setup-tab ${activeTab === 'ccps' ? 'active' : ''}`}
          onClick={() => setActiveTab('ccps')}
          disabled={isDisabled}
        >
          <Shield size={14} />
          <span>CCPs ({manualConfig.ccps.length})</span>
        </button>
        <button
          className={`setup-tab ${activeTab === 'market' ? 'active' : ''}`}
          onClick={() => setActiveTab('market')}
          disabled={isDisabled}
        >
          <TrendingUp size={14} />
          <span>Market</span>
        </button>
      </div>

      <div className="setup-content">
        {/* Banks Tab */}
        {activeTab === 'banks' && (
          <div className="setup-section">
            <div className="section-header">
              <span className="section-label">Banks Configuration</span>
              <button
                className="add-button"
                onClick={addBank}
                disabled={isDisabled}
              >
                <Plus size={14} />
                <span>Add Bank</span>
              </button>
            </div>

            {manualConfig.banks.length === 0 ? (
              <div className="empty-state">
                <Building2 size={48} className="empty-icon" />
                <p>No banks configured</p>
                <p className="empty-hint">Click "Add Bank" to start</p>
              </div>
            ) : (
              <div className="entity-list">
                {manualConfig.banks.map((bank) => (
                  <div key={bank.id} className="entity-card">
                    <div className="entity-header">
                      <input
                        type="text"
                        className="entity-name-input"
                        value={bank.name}
                        onChange={(e) => updateBank(bank.id, 'name', e.target.value)}
                        disabled={isDisabled}
                        placeholder="Bank name"
                      />
                      <button
                        className="delete-button"
                        onClick={() => removeBank(bank.id)}
                        disabled={isDisabled}
                        title="Remove bank"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <div className="entity-fields">
                      <div className="field-group">
                        <label className="field-label">Capital</label>
                        <input
                          type="number"
                          className="field-input"
                          value={bank.capital}
                          onChange={(e) => updateBank(bank.id, 'capital', parseFloat(e.target.value) || 0)}
                          disabled={isDisabled}
                          min="0"
                          step="10"
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label">Assets</label>
                        <input
                          type="number"
                          className="field-input"
                          value={bank.assets}
                          onChange={(e) => updateBank(bank.id, 'assets', parseFloat(e.target.value) || 0)}
                          disabled={isDisabled}
                          min="0"
                          step="10"
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label">Tier</label>
                        <select
                          className="field-select"
                          value={bank.tier}
                          onChange={(e) => updateBank(bank.id, 'tier', parseInt(e.target.value))}
                          disabled={isDisabled}
                        >
                          <option value={1}>Tier 1</option>
                          <option value={2}>Tier 2</option>
                          <option value={3}>Tier 3</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* CCPs Tab */}
        {activeTab === 'ccps' && (
          <div className="setup-section">
            <div className="section-header">
              <span className="section-label">Central Counterparties</span>
              <button
                className="add-button"
                onClick={addCCP}
                disabled={isDisabled}
              >
                <Plus size={14} />
                <span>Add CCP</span>
              </button>
            </div>

            {manualConfig.ccps.length === 0 ? (
              <div className="empty-state">
                <Shield size={48} className="empty-icon" />
                <p>No CCPs configured</p>
                <p className="empty-hint">Click "Add CCP" to start</p>
              </div>
            ) : (
              <div className="entity-list">
                {manualConfig.ccps.map((ccp) => (
                  <div key={ccp.id} className="entity-card">
                    <div className="entity-header">
                      <input
                        type="text"
                        className="entity-name-input"
                        value={ccp.name}
                        onChange={(e) => updateCCP(ccp.id, 'name', e.target.value)}
                        disabled={isDisabled}
                        placeholder="CCP name"
                      />
                      <button
                        className="delete-button"
                        onClick={() => removeCCP(ccp.id)}
                        disabled={isDisabled}
                        title="Remove CCP"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <div className="entity-fields">
                      <div className="field-group">
                        <label className="field-label">Capital</label>
                        <input
                          type="number"
                          className="field-input"
                          value={ccp.capital}
                          onChange={(e) => updateCCP(ccp.id, 'capital', parseFloat(e.target.value) || 0)}
                          disabled={isDisabled}
                          min="0"
                          step="10"
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label">Clearing Capacity</label>
                        <input
                          type="number"
                          className="field-input"
                          value={ccp.clearing_capacity}
                          onChange={(e) => updateCCP(ccp.id, 'clearing_capacity', parseFloat(e.target.value) || 0)}
                          disabled={isDisabled}
                          min="0"
                          step="10"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Market Tab */}
        {activeTab === 'market' && (
          <div className="setup-section">
            <div className="section-header">
              <span className="section-label">Market Parameters</span>
            </div>

            <div className="market-params">
              <div className="param-card">
                <label className="param-label">Initial Asset Price</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.initial_price}
                  onChange={(e) => updateMarket('initial_price', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  step="0.1"
                />
                <span className="param-hint">Starting price of tradeable assets</span>
              </div>

              <div className="param-card">
                <label className="param-label">Interest Rate (%)</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.interest_rate}
                  onChange={(e) => updateMarket('interest_rate', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="100"
                  step="0.1"
                />
                <span className="param-hint">Base lending rate for the system</span>
              </div>

              <div className="param-card">
                <label className="param-label">Liquidity Index</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.liquidity_index}
                  onChange={(e) => updateMarket('liquidity_index', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="1"
                  step="0.01"
                />
                <span className="param-hint">Market liquidity measure (0-1)</span>
              </div>

              <div className="param-card">
                <label className="param-label">Volatility</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.volatility}
                  onChange={(e) => updateMarket('volatility', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="1"
                  step="0.01"
                />
                <span className="param-hint">Price volatility factor (0-1)</span>
              </div>

              <div className="param-card">
                <label className="param-label">Episode Length</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.episode_length}
                  onChange={(e) => updateMarket('episode_length', parseInt(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="1"
                  step="1"
                />
                <span className="param-hint">Number of timesteps to simulate</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
